#include <cstdint>
#include <iostream>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <gpu_microbenchmarks.h>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Cuda fuction error checking macros
#define CUDA_CHECK(x)\
{ \
	cudaError_t e = x; \
	if(e != cudaSuccess) { \
		std::cerr << "CUDA error: " << cudaGetErrorString(e) << "\n"; \
		cudaDeviceReset(); \
		return -1.0; \
	} \
}

#define CUBLAS_CHECK(x)\
{ \
	cublasStatus_t status = (x); \
	if(status != CUBLAS_STATUS_SUCCESS) { \
		std::cerr << "cuBLAS error: " << status << "\n"; \
		return -1.0; \
	} \
}

//datatype declarations
struct gemmDims {
	int m;
	int n;
	int k;
};

// Function declarations
std::vector<gemmDims> setGemmSides(size_t globalMem, cudaDataType type, int numSides);
bool isSupportedAlgo(cublasGemmAlgo_t algo);
bool isSupportedDataType(cudaDataType type);
bool isSupportedComputeType(cudaDataType computeType);
bool isSupportedMathMode(cublasMath_t mathMode);
bool checkValidGemmConfig(
	int sm,
	cublasGemmAlgo_t algo, 
	cudaDataType type, 
	cudaDataType computeType,
	cublasMath_t mathMode
);
cudaError_t setMatrices(
		void** A, 
		void** B, 
		void** C, 
		int m,
		int n,
		int k,
		cudaDataType type
);
float computeTOPSSingleGPU(
		void* A, 
		void* B, 
		void* C, 
		int m, 
		int n, 
		int k,
		cudaDataType type,
		cudaDataType computeType,
		cublasGemmAlgo_t algo,
		cublasHandle_t handle
);
float computeTOPSSingleGPUTensor(
		void* A, 
		void* B, 
		void* C, 
		int m, 
		int n, 
		int k,
		cudaDataType type,
		cudaDataType computeType
);

// global triad for mem bandwidth
__global__
void global_triad(float* A, float* B, float* C, int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < n) {
		C[idx] = A[idx] + 3.0 * B[idx];
	}
} 

double global_GPU_mem_bandwidth(int device) {
	//set GPU to run on
	int numDevices = 0;
	CUDA_CHECK(cudaGetDeviceCount(&numDevices));
	if(device >= numDevices) {
		return -1;
	}
	CUDA_CHECK(cudaSetDevice(device));

	// get/set device data
	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
	size_t global_mem = prop.totalGlobalMem;
	uint64_t N = (global_mem/sizeof(float)/4);
	const int threads_per_block = 256;
	const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

	// initialise events
	float time = 0.0;
	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));

	// initialise triad data
	float* h_A = new float[N];
	float* h_B = new float[N];
	float* h_C = new float[N];

	for(int i = 0; i < N; ++i) {
		h_A[i] = 1.0;
		h_B[i] = 2.0;
		h_C[i] = 0.0;
	}

	cudaError_t err;

	//copy vectors to device using cudaMalloc
	float* d_A;
	float* d_B;
	float* d_C;
	CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));

	CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_C, h_C, N * sizeof(float), cudaMemcpyHostToDevice));

	// warmup runs
	for(int i = 0; i < 10; ++i) {
		global_triad<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, N);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	/* -- run triad kernel -- */
	int runs = 10;
	for(int i = 0; i < runs; ++i) {

		cudaEventRecord(start);
		global_triad<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, N);
		err = cudaGetLastError();
		if(err != cudaSuccess) {
			std::cerr << "Launch error: " << cudaGetErrorString(err) << "\n";
			return -1;
		}
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float ms = 0.0f;
		cudaEventElapsedTime(&ms, start, stop);
		time += ms;
	}

	//average results
	time /= runs;

	CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));
	float checksum = 0.0;
	for(int i = 0; i < N; ++i) {
		checksum += h_C[i];
	}

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));

	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_B));
	CUDA_CHECK(cudaFree(d_C));

	CUDA_CHECK(cudaDeviceReset());

	delete[] h_A;
	delete[] h_B;
	delete[] h_C;
	h_A = NULL;
	h_B = NULL;
	h_C = NULL;
	
	// return GB/s
	double bytes = 3.0 * N * sizeof(float);
	double GBs = (bytes/1e9)/(static_cast<double>(time)/1000.0);
	return GBs;
}

double single_GPU_TOPS (
		int device,
		cublasGemmAlgo_t algo,
		cudaDataType type,
		cudaDataType computeType,
		cublasMath_t mathMode
	) 
{
	// Check input variables
	cudaDeviceProp prop;
	CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
	int sm = prop.major * 10 + prop.minor;
	if(!checkValidGemmConfig(sm, algo, type, computeType, mathMode)) {
		std::cerr << "wrong input variables on single_GPU_TOPS\n";
		return -1.0;
	}

	// Check and set device
	int numDevices = 0;
	CUDA_CHECK(cudaGetDeviceCount(&numDevices));
	if(device > numDevices) {
		return -1.0;
	}
	CUDA_CHECK(cudaSetDevice(device));

	// Set cuBlas handle
	cublasHandle_t handle;
	CUBLAS_CHECK(cublasCreate(&handle));

	// Check if tensor
	bool isTensor;
	if(algo == CUBLAS_GEMM_DEFAULT_TENSOR_OP) {
		isTensor = true;
	}
	else {
		isTensor = false;
	}

	// Set math mode
	CUBLAS_CHECK(cublasSetMathMode(handle, mathMode));

	// Get device memory and define sizes for GEMM calculation
	size_t globalMem = prop.totalGlobalMem;

	// Set struct with sets of matrix side lengths 
	std::vector<gemmDims> matSides = setGemmSides(globalMem, type, 5);

	// set vector of TOPS results to average at the end
	float TOPS = 0;

	// create timers
	cudaEvent_t start, end;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&end));

	// Declare pointers to matrices on device
	void* A;
	void* B;
	void* C;
	
	// runs through every possible set of sides defined in matSides
	for(int i = 0; i < matSides.size(); ++i) {
		// set matrix values
		CUDA_CHECK(setMatrices(&A, &B, &C,
			matSides[i].m, matSides[i].n, matSides[i].k, 
			type
		));	

		// Run and record GEMM benchmark
		float result;
		if(!isTensor) {
			result = computeTOPSSingleGPU(
				A, 
				B, 
				C,
				matSides[i].m,
				matSides[i].n,
				matSides[i].k,
				type,
				computeType,
				algo,
				handle
			);
		}
		else {
			result = computeTOPSSingleGPUTensor(
				A, 
				B, 
				C, 
				matSides[i].m, 
				matSides[i].n, 
				matSides[i].k, 
				type, 
				computeType
			);
		}

		if(result == -1) {
			return -1;
		}
		TOPS += result;

		// free matrix memory
		cudaFree(A);
		cudaFree(B);
		cudaFree(C);
		cudaDeviceReset();
	}
	// calculate average TOPS
	TOPS /= static_cast<float>(matSides.size());

	return TOPS;
}

// Sets a set of sides for the matrices
// each side is stored as a vector
std::vector<gemmDims> setGemmSides(size_t globalMem, cudaDataType type, int numSides) {
	size_t elementSize;		// size of each element in bytes
	bool tensor;						// tensor core calculations must be multiples of 8
	switch(type) {
		case CUDA_R_32F:
			tensor = false;
			elementSize = 4;
			break;
		case CUDA_R_16F:
			tensor = true;
			elementSize = 2;
			break;
		case CUDA_R_8I:
			tensor = true;
			elementSize = 1;
			break;
		case CUDA_R_64F:
			tensor = false;
			elementSize = 8;
			break;
		default:
			throw std::runtime_error("unsupported type\n");
	}

	// Produce amount of elements to work around
	size_t calcMem;
	calcMem = globalMem * 0.7 / elementSize;
	
	size_t maxSide = std::sqrt(calcMem/3);	// divide by 3 for three matrices
	std::vector<gemmDims> dims;
	int vals = numSides;
	for(int i = 0; i < vals; ++i) {
		for(int j = 0; j < vals; ++j) {
			for(int k = 0; k < vals; ++k) {
				int x = maxSide / (i + 1);
				int y = maxSide / (j + 1);
				int z = maxSide / (k + 1);
				// Tensor cores muxt be in multiples of 32 on Turing
				if(tensor) {
					x = (x / 32) * 32;
					y = (y / 32) * 32;
					z = (z / 32) * 32;
				}
				gemmDims temp = {x, y, z};
				dims.push_back(temp);
			}
		}
	}
	return dims;
}

bool isSupportedAlgo(cublasGemmAlgo_t algo) {
	switch(algo){
		case CUBLAS_GEMM_DEFAULT:
			return true;
		case CUBLAS_GEMM_DEFAULT_TENSOR_OP:
			return true;
	}
	return false;
}

bool isSupportedDataType(cudaDataType type) {
	switch(type) {
		case CUDA_R_32F:
			return true;
		case CUDA_R_16F:
			return true;
		case CUDA_R_8I:
			return true;
		case CUDA_R_32I:
			return true;
		case CUDA_R_64F:
			return true;
	}
	return false;
}

bool isSupportedComputeType(cudaDataType computeType) {
	switch(computeType) {
		case CUDA_R_32F:
			return true;
		case CUDA_R_16F:
			return true;
		case CUDA_R_8I:
			return true;
		case CUDA_R_32I:
			return true;
		case CUDA_R_64F:
			return true;
	}
	return false;
}

bool isSupportedMathMode(cublasMath_t mathMode) {
	switch(mathMode) {
		case CUBLAS_DEFAULT_MATH:
			return true;
		case CUBLAS_TENSOR_OP_MATH:
			return true;
		case CUBLAS_TF32_TENSOR_OP_MATH:
			return true;
	}
	return false;
}

bool checkValidGemmConfig(
		int sm,
		cublasGemmAlgo_t algo, 
		cudaDataType type, 
		cudaDataType computeType,
		cublasMath_t mathMode
) {
	if(!isSupportedAlgo(algo)) {
		return false;
	}
	if(!isSupportedDataType(type)) {
		return false;
	}
	if(!isSupportedComputeType(computeType)) {
		return false;
	}
	if(!isSupportedMathMode(mathMode)) {
		return false;
	}

	// Check if Tensor core requested
	bool isTensor = 
		(mathMode == CUBLAS_TENSOR_OP_MATH) ||
		(mathMode == CUBLAS_TF32_TENSOR_OP_MATH) ||
		(algo == CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	
	// FP32 always valid
	if(type == CUDA_R_32F && computeType == CUDA_R_32F) {
		return true;
	}

	// FP16
	if(type == CUDA_R_16F) {
		if(!isTensor) {
			return false;
		}

		// Volta / Turing
		if(sm < 80) {
			return computeType == CUDA_R_32F;
		}

		// Ampere+
		if(sm >= 80) {
			return (computeType == CUDA_R_32F || computeType == CUDA_R_16F);
		}
	}

	// TF32
	if(type == CUDA_R_32F && mathMode == CUBLAS_TF32_TENSOR_OP_MATH) {
		return sm >= 80;
	}

	// INT8
	if(type == CUDA_R_8I) {
		if(!isTensor) {
			return false;
		}
		return computeType == CUDA_R_32I;
	}

	// FP64
	if(type == CUDA_R_64F) {
		return (computeType == CUDA_R_64F && mathMode == CUBLAS_DEFAULT_MATH);
	}

	return false;
}

// Matrix value setter kernels
// x, y are dimensions of the matrix
__global__
void set_matrix_32F(float* A, int x, int y) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < x * y) {
		A[idx] = 1.0f;
	}
}

__global__
void set_matrix_16F(__half* A, int x, int y) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < x * y) {
		A[idx] = static_cast<__half>(1.0);
	}
}
__global__
void set_matrix_8I(int8_t* A, int x, int y) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < x * y) {
		A[idx] = static_cast<int8_t>(1);
	}
}

__global__
void set_matrix_32I(int32_t* A, int x, int y) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < x * y) {
		A[idx] = static_cast<int32_t>(1);
	}
}

__global__
void set_matrix_64F(double* A, int x, int y) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < x * y) {
		A[idx] = static_cast<double>(1.0);
	}
}


// allocate memory and set values for matrices
cudaError_t setMatrices(void** A, void** B, void** C, int m, int n, int k, cudaDataType type) {
	int element_size;
	int threads = 256;
	int blocksA = 0;
	int blocksB = 0;
	int blocksC = 0;
	cudaError_t err;
	switch(type) 
		case CUDA_R_32F: {
			element_size = 4;
			err = cudaMalloc(A, m * k * element_size);
			if(err != cudaSuccess) {
				return err;
			}
			err = cudaMalloc(B, n * k * element_size);
			if(err != cudaSuccess) {
				cudaFree(A);
				return err;
			}
			err = cudaMalloc(C, m * n * element_size);
			if(err != cudaSuccess) {
				cudaFree(A);
				cudaFree(B);
				return err;
			}

			// Set thread and block sizes
			blocksA = ((m * k) + threads - 1) / threads;
			blocksB = ((k * n) + threads - 1) / threads;
			blocksC = ((m * n) + threads - 1) / threads;

			// -- investigate cudaMalloc, might be allocating to the wrong place by using &A -- //
			set_matrix_32F<<<blocksA, threads>>>(static_cast<float*>(*A), m, k);
			err = cudaGetLastError();
			if(err != cudaSuccess) {
				cudaFree(A);
				cudaFree(B);
				cudaFree(C);
				return err;
			}
			
			set_matrix_32F<<<blocksB, threads>>>(static_cast<float*>(*B), k, n);
			err = cudaGetLastError();
			if(err != cudaSuccess) {
				cudaFree(A);
				cudaFree(B);
				cudaFree(C);
				return err;
			}
			
			set_matrix_32F<<<blocksC, threads>>>(static_cast<float*>(*C), m, n);
			err = cudaGetLastError();
			if(err != cudaSuccess) {
				cudaFree(A);
				cudaFree(B);
				cudaFree(C);
				return err;
			}			
			cudaDeviceSynchronize();
			return cudaSuccess;
				

		case CUDA_R_16F: {
			element_size = 2;
			err = cudaMalloc(A, m * k * element_size);
			if(err != cudaSuccess) {
				return err;
			}
			err = cudaMalloc(B, n * k * element_size);
			if(err != cudaSuccess) {
				cudaFree(A);
				return err;
			}
			err = cudaMalloc(C, m * n * element_size);
			if(err != cudaSuccess) {
				cudaFree(A);
				cudaFree(B);
				return err;
			}

			// Set thread and block sizes
			blocksA = ((m * k) + threads - 1) / threads;
			blocksB = ((k * n) + threads - 1) / threads;
			blocksC = ((m * n) + threads - 1) / threads;

			set_matrix_16F<<<blocksA, threads>>>(static_cast<__half*>(*A), m, k);
			err = cudaGetLastError();
			if(err != cudaSuccess) {
				cudaFree(A);
				cudaFree(B);
				cudaFree(C);
				return err;
			}
			set_matrix_16F<<<blocksB, threads>>>(static_cast<__half*>(*B), k, n);
			err = cudaGetLastError();
			if(err != cudaSuccess) {
				cudaFree(A);
				cudaFree(B);
				cudaFree(C);
				return err;
			}
			set_matrix_16F<<<blocksC, threads>>>(static_cast<__half*>(*C), m, n);
			err = cudaGetLastError();
			if(err != cudaSuccess) {
				cudaFree(A);
				cudaFree(B);
				cudaFree(C);
				return err;
			}
			cudaDeviceSynchronize();
			return cudaSuccess;
		}

		case CUDA_R_8I: {
			element_size = 1;
			err = cudaMalloc(A, m * k * element_size);
			if(err != cudaSuccess) {
				return err;
			}
			err = cudaMalloc(B, n * k * element_size);
			if(err != cudaSuccess) {
				cudaFree(A);
				return err;
			}
			err = cudaMalloc(C, m * n * element_size);
			if(err != cudaSuccess) {
				cudaFree(A);
				cudaFree(B);
				return err;
			}

			// Set thread and block sizes
			blocksA = ((m * k) + threads - 1) / threads;
			blocksB = ((k * n) + threads - 1) / threads;
			blocksC = ((m * n) + threads - 1) / threads;

			set_matrix_8I<<<blocksA, threads>>>(static_cast<int8_t*>(*A), m, k);
			err = cudaGetLastError();
			if(err != cudaSuccess) {
				cudaFree(A);
				cudaFree(B); 
				cudaFree(C);
				return err;
			}
			set_matrix_8I<<<blocksB, threads>>>(static_cast<int8_t*>(*B), k, n);
			err = cudaGetLastError();
			if(err != cudaSuccess) {
				cudaFree(A);
				cudaFree(B);
				cudaFree(C);
				return err;
			}
			set_matrix_8I<<<blocksC, threads>>>(static_cast<int8_t*>(*C), m, n);
			err = cudaGetLastError();
			if(err != cudaSuccess) {
				cudaFree(A);
				cudaFree(B);
				cudaFree(C);
				return err;
			}
			cudaDeviceSynchronize();
			return cudaSuccess;
		}

		case CUDA_R_64F: {
			element_size = 8;
			err = cudaMalloc(A, m * k * element_size);
			if(err != cudaSuccess) {
				return err;
			}
			err = cudaMalloc(B, n * k * element_size);
			if(err != cudaSuccess) {
				cudaFree(A);
				return err;
			}
			err = cudaMalloc(C, m * n * element_size);
			if(err != cudaSuccess) {
				cudaFree(A);
				cudaFree(B);
				return err;
			}

			// Set thread and block sizes
			blocksA = ((m * k) + threads - 1) / threads;
			blocksB = ((k * n) + threads - 1) / threads;
			blocksC = ((m * n) + threads - 1) / threads;

			set_matrix_64F<<<blocksA, threads>>>(static_cast<double*>(*A), m, k);
			err = cudaGetLastError();
			if(err != cudaSuccess) {
				cudaFree(A);
				cudaFree(B);
				cudaFree(C);
				return err;
			}
			set_matrix_64F<<<blocksB, threads>>>(static_cast<double*>(*B), k, n);
			err = cudaGetLastError();
			if(err != cudaSuccess) {
				cudaFree(A);
				cudaFree(B);
				cudaFree(C);
				return err;
			}
			set_matrix_64F<<<blocksC, threads>>>(static_cast<double*>(*C), m, n);
			err = cudaGetLastError();
			if(err != cudaSuccess) {
				cudaFree(A);
				cudaFree(B);
				cudaFree(C);
				return err;
			}
			cudaDeviceSynchronize();
			return cudaSuccess;
		}

		default:
			throw std::runtime_error("unsupported type\n");
	}
	return cudaSuccess;
}

// runs GEMM benchmark and times performance
float computeTOPSSingleGPU(
		void* A, 
		void* B, 
		void* C, 
		int m, 
		int n, 
		int k,
		cudaDataType type,
		cudaDataType computeType,
		cublasGemmAlgo_t algo,
		cublasHandle_t handle
) {
	float TOPS = 0;

	cudaEvent_t start, end;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&end));
	float time = 0;
	double FLOPs = 0;

	const int32_t alpha = 2;
	const int32_t beta = 1;

	// Define C datatype and set up INT8 GEMM
	cudaDataType typeC = type;

	// run warmup function
	CUBLAS_CHECK(cublasGemmEx(
				handle,
				CUBLAS_OP_N,
				CUBLAS_OP_T,
				m, n, k,
				&alpha,
				A, type, m,
				B, type, n,
				&beta,
				C, typeC, m,
				computeType,
				algo
				)
			);
	cudaDeviceSynchronize();

	// loop x times and measure operations and speed
	int iterations = 10;
	for(int i = 0; i < iterations; ++i) {
		float ms = 0.0f;

		cudaEventRecord(start);
		CUBLAS_CHECK(cublasGemmEx(
					handle,
					CUBLAS_OP_N,
					CUBLAS_OP_T,
					m, n, k,
					&alpha,
					A, type, m,
					B, type, n,
					&beta,
					C, typeC, m,
					computeType,
					algo
					)
				);
		cudaEventRecord(end);
		cudaEventSynchronize(end);

		FLOPs = 2.0 * m * n * k;
		cudaEventElapsedTime(&ms, start, end);
		time = ms * 1e-3;
		double tflops = FLOPs / time * 1e-12;
		TOPS += tflops;
	}
	TOPS /= static_cast<float>(iterations);

	// Cleanup environment
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	return TOPS;
}

float computeTOPSSingleGPUTensor(
		void* A, 
		void* B, 
		void* C, 
		int m, 
		int n, 
		int k,
		cudaDataType type,
		cudaDataType computeType
) {
	// Set scalars
	float alpha = 1.0f;
	float beta = 0.5f;

	// Create workspace
	void * workspace;
	size_t workspaceSize = 64 * 1024 * 1024;
	CUDA_CHECK(cudaMalloc(&workspace, workspaceSize));

	// Initialise heuristic results
	cublasLtMatmulDesc_t desc;
	cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
	cublasLtMatmulPreference_t preference;
	int returnedHeuristics = 0;
	cublasLtMatmulHeuristicResult_t heuristicsResult[10];

	// create handle
	cublasLtHandle_t handle;
	cublasLtCreate(&handle);

	// Define op descriptors
	cublasOperation_t opA, opB;

	switch(type) {
		case(CUDA_R_8I):
			// Define matrix operation descriptors
			opA = CUBLAS_OP_T;
			opB = CUBLAS_OP_N;

			// Define desc
			cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32I, CUDA_R_32F);

			// Define matrix attributes
			cublasLtMatmulDescSetAttribute(
				desc,
				CUBLASLT_MATMUL_DESC_TRANSA,
				&opA,
				sizeof(opA)
			);
			cublasLtMatmulDescSetAttribute(
				desc,
				CUBLASLT_MATMUL_DESC_TRANSB,
				&opB,
				sizeof(opB)
			);

			// Define matrix layouts
			cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8I, k, m, k);
			cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8I, k, n, k);
			cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_8I, m, n, m);
			break;

		case(CUDA_R_16F):
			opA = CUBLAS_OP_N;
			opB = CUBLAS_OP_N;

			cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

			cublasLtMatmulDescSetAttribute(
				desc,
				CUBLASLT_MATMUL_DESC_TRANSA,
				&opA,
				sizeof(opA)
			);
			cublasLtMatmulDescSetAttribute(
				desc,
				CUBLASLT_MATMUL_DESC_TRANSB,
				&opB,
				sizeof(opB)
			);

			cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_16F, m, k, m);
			cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_16F, k, n, k);
			cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_16F, m, n, m);
	}


	// Define preference
	cublasLtMatmulPreferenceCreate(&preference);
	cublasLtMatmulPreferenceSetAttribute(
		preference,
		CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
		&workspaceSize,
		sizeof(workspaceSize)
	);

	// Get heuristic Gemm kernel
	cublasLtMatmulAlgoGetHeuristic(
		handle, 
		desc, 
		layoutA, 
		layoutB, 
		layoutC, 
		layoutC, 
		preference, 
		1, 
		heuristicsResult, 
		&returnedHeuristics
	);
	if(returnedHeuristics == 0) {
		std::cout << "0 heuristics found\n";
		return -1.0;
	}

	// set stream
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));


	CUBLAS_CHECK(
		cublasLtMatmul(
			handle,
			desc,
			&alpha,
			A, layoutA,
			B, layoutB,
			&beta,
			C, layoutC,
			C, layoutC,
			&heuristicsResult[0].algo,
			workspace,
			workspaceSize,
			stream
		)
	);
	cudaDeviceSynchronize();

	// setup timers
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	// Timed loop
	const int iterations = 10;
	double TOPS = 0.0;

	for(int i = 0; i < iterations; ++i) {
		cudaEventRecord(start);

		CUBLAS_CHECK(
			cublasLtMatmul(
				handle,
				desc,
				&alpha,
				A, layoutA,
				B, layoutB,
				&beta,
				C, layoutC,
				C, layoutC,
				&heuristicsResult[0].algo,
				workspace,
				workspaceSize,
				stream
			)
		);
		cudaEventRecord(end);
		cudaEventSynchronize(end);

		// calculate TOPS
		float ms;
		cudaEventElapsedTime(&ms, start, end);

		double flops = 2.0 * m * n * k;
		double tflops = flops / (ms * 1e-3) * 1e-12;
		TOPS += tflops;
	}

	TOPS /= iterations;

	// Cleanup environment
	cublasLtMatmulDescDestroy(desc);
	cublasLtMatrixLayoutDestroy(layoutA);
	cublasLtMatrixLayoutDestroy(layoutB);
	cublasLtMatrixLayoutDestroy(layoutC);
	cublasLtDestroy(handle);
	cudaEventDestroy(start);
	cudaEventDestroy(end);

	return static_cast<float>(TOPS);
}
