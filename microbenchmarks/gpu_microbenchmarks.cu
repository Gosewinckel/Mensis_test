#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <gpu_microbenchmarks.h>
#include <vector>
#include <cmath>
#include <ratio>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


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
	cudaGetDeviceCount(&numDevices);
	if(device >= numDevices) {
		return -1;
	}
	cudaSetDevice(device);

	// get/set device data
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device);
	int sm_count = prop.multiProcessorCount;
	const int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
	size_t global_mem = prop.totalGlobalMem;
	size_t shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
	int N = global_mem/sizeof(float)/4;
	const int threads_per_block = 256;
	const int num_blocks = (N + threads_per_block - 1) / threads_per_block;

	// initialise events
	float time = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// initialise triad data
	float* h_A = new float[N];
	float* h_B = new float[N];
	float* h_C = new float[N];

	for(int i = 0; i < N; ++i) {
		h_A[i] = 1.0;
		h_B[i] = 2.0;
		h_C[i] = 0.0;
	}

	//copy vectors to device using cudaMalloc
	float* d_A;
	float* d_B;
	float* d_C;
	cudaMalloc(&d_A, N * sizeof(float));
	cudaMalloc(&d_B, N * sizeof(float));
	cudaMalloc(&d_C, N * sizeof(float));
	cudaMemcpy(d_A, h_A, N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, N, cudaMemcpyHostToDevice);

	// warmup runs
	for(int i = 0; i < 10; ++i) {
		global_triad<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, N);
		cudaDeviceSynchronize();
	}

	/* -- run triad kernel -- */
	int runs = 10;
	for(int i = 0; i < runs; ++i) {

		cudaEventRecord(start);
		global_triad<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, N);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float ms = 0.0f;
		cudaEventElapsedTime(&ms, start, stop);
		time += ms;
	}

	//average results
	time /= runs;

	cudaMemcpy(h_C, d_C, N, cudaMemcpyDeviceToHost);
	float checksum = 0.0;
	for(int i = 0; i < N; ++i) {
		checksum += h_C[i];
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

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


__global__
void init_fp32(float* A, size_t size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < size) {
		A[idx] = fmodf(idx * 0.01f, 1.0f);
	}
}

__global__
void init_fp16(float* A, size_t size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < size) {
		float val = fmodf(idx * 0.001f, 1.0f);
		A[idx] = __float2half(val);
	}
}

__global__
void init_bf16(__nv_bfloat16* A, size_t size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < size) {
		float val = fmodf(idx * 0.001f, 1.0f);
		A[idx] = __float2bfloat16(val);
	}
}

__global__
void init_int8(int8_t* A, size_t size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < size) {
		int val = (idx % 255) - 127;
		A[idx] = static_cast<int8_t>(val);
	}
}

__global__
void init_fp64(double* A, size_t size) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < size) {
		A[idx] = fmodf(idx * 0.001, 1.0);
	}
}

double single_GPU_TOPS(
		int device,
		cublasGemmAlgo_t algo,
		cudaDataType_t type,
		cublasComputeType_t computeType,
		cublasMath_t mathMode
) {
	//check andf set device
	int numDevices = 0;
	cudaGetDeviceCount(&numDevices);
	if(device >= numDevices) {
		return -1;
	}
	cudaSetDevice(device);

	// Set handle
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSetMathMode(handle, mathMode);

	// find GPU onboard memory
	size_t free_bytes;
	size_t total_bytes;
	cudaMemGetInfo(&free_bytes, &total_bytes);

	// Each matrix = 20% of GPU memory
	uint64_t matrix_size = 0;
	matrix_size = (total_bytes * (1/5)) / 4;
	uint64_t sqrt = std::sqrt(matrix_size);

	float TOPS = 0.0; //trillion operations per second, avg results of each GEMM and return

	// Set of matrix side sizes in bits
	std::vector<uint64_t> edges = {sqrt/4, sqrt/5, sqrt/3, sqrt/2, (sqrt * 3)/4,
		(sqrt * 4)/5, (sqrt * 5)/4, (sqrt * 4)/3, sqrt * 2, sqrt * 3,
		sqrt * 4, sqrt};

	int threads = 256;
	int blocks = (matrix_size + threads - 1) / threads;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Create a loop that to generate and test GEMM computations
	// for different sizes of matrix
	for(int x = 0; x < edges.size(); ++x) {
		// generate matrices
		void* A;
		void* B;
		void* C;

		// declare matrix sizes
		uint64_t M;
		uint64_t N;
		uint64_t K;
		uint64_t matrix_volume;

		uint64_t tops = 0; // tops for each GEMM dimension, will average to fine TOPS

		// setup for cuda timers
		cudaEvent_t start, end;
		cudaEventCreate(&start);
		cudaEventCreate(&end);


		switch(type) {
			cudaError_t err;
			case CUDA_R_32F:
				// set matrix info
				matrix_volume = matrix_size/32;
				M = edges[x]/32;
				N = matrix_volume/M;
				K = M;

				//allocate memory on GPU
				err = cudaMalloc(&A, matrix_size);
				if(err != cudaSuccess) {
					std::cerr << "cudaMalloc failed\n";
					return -1.0;
				}
				err = cudaMalloc(&B, matrix_size);
				if(err != cudaSuccess) {
					std::cerr << "cudaMalloc failed\n";
					return -1.0;
				}
				err = cudaMalloc(&C, matrix_size);
				if(err != cudaSuccess) {
					std::cerr << "cudaMalloc failed\n";
					return -1.0;
				}
				
				// initialise matrix values
				init_fp32<<<blocks, threads>>>(static_cast<float*>(A), matrix_volume/32);
				init_fp32<<<blocks, threads>>>(static_cast<float*>(B), matrix_volume/32);
				init_fp32<<<blocks, threads>>>(static_cast<float*>(C), matrix_volume/32);
				break;
			
			case CUDA_R_16F:
				matrix_volume = matrix_size/16;
				M = edges[x]/16;
				N = matrix_volume/M;
				K = M;

				err = cudaMalloc(&A, matrix_size);
				if(err != cudaSuccess) {
					std::cerr << "cudaMalloc failed\n";
					return -1.0;
				}
				err = cudaMalloc(&B, matrix_size);
				if(err != cudaSuccess) {
					std::cerr << "cudaMalloc failed\n";
					return -1.0;
				}
				err = cudaMalloc(&C, matrix_size);
				if(err != cudaSuccess) {
					std::cerr << "cudaMalloc failed\n";
					return -1.0;
				}

				init_fp16<<<blocks, threads>>>(static_cast<float*>(A), matrix_volume/16);
				init_fp16<<<blocks, threads>>>(static_cast<float*>(B), matrix_volume/16);
				init_fp16<<<blocks, threads>>>(static_cast<float*>(C), matrix_volume/16);
				break;

			case CUDA_R_16BF:
				matrix_volume = matrix_size/16;
				M = edges[x]/16;
				N = matrix_volume/M;
				K = M;

				err = cudaMalloc(&A, matrix_size);
				if(err != cudaSuccess) {
					std::cerr << "cudaMalloc failed\n";
					return -1.0;
				}
				err = cudaMalloc(&B, matrix_size);
				if(err != cudaSuccess) {
					std::cerr << "cudaMalloc failed\n";
					return -1.0;
				}
				err = cudaMalloc(&C, matrix_size);
				if(err != cudaSuccess) {
					std::cerr << "cudaMalloc failed\n";
					return -1.0;
				}

				init_bf16<<<blocks, threads>>>(static_cast<__nv_bfloat16*>(A), matrix_volume/16);	
				init_bf16<<<blocks, threads>>>(static_cast<__nv_bfloat16*>(B), matrix_volume/16);
				init_bf16<<<blocks, threads>>>(static_cast<__nv_bfloat16*>(C), matrix_volume/16);

			case CUDA_R_8I:
				matrix_volume = matrix_size/8;
				M = edges[x]/8;
				N = matrix_volume/M;
				K = M;

				err = cudaMalloc(&A, matrix_size);
				if(err != cudaSuccess) {
					std::cerr << "cudaMalloc failed\n";
					return -1.0;
				}
				err = cudaMalloc(&B, matrix_size);
				if(err != cudaSuccess) {
					std::cerr << "cudaMalloc failed\n";
					return -1.0;
				}
				err = cudaMalloc(&C, matrix_size);
				if(err != cudaSuccess) {
					std::cerr << "cudaMalloc failed\n";
					return -1.0;
				}

				init_int8<<<blocks, threads>>>(static_cast<int8_t*>(A), matrix_volume/8);
				init_int8<<<blocks, threads>>>(static_cast<int8_t*>(B), matrix_volume/8);
				init_int8<<<blocks, threads>>>(static_cast<int8_t*>(C), matrix_volume/8);

			case CUDA_R_64F:
				matrix_volume = matrix_size/64;
				M = edges[x]/64;
				N = matrix_volume/M;
				K = M;

				err = cudaMalloc(&A, matrix_size);
				if(err != cudaSuccess) {
					std::cerr << "cudaMalloc failed\n";
					return -1.0;
				}
				err = cudaMalloc(&B, matrix_size);
				if(err != cudaSuccess) {
					std::cerr << "cudaMalloc failed\n";
					return -1.0;
				}
				err = cudaMalloc(&C, matrix_size);
				if(err != cudaSuccess) {
					std::cerr << "cudaMalloc failed\n";
					return -1.0;
				}

				init_fp64<<<blocks, threads>>>(static_cast<double*>(A), matrix_volume/64);
				init_fp64<<<blocks, threads>>>(static_cast<double*>(B), matrix_volume/64);
				init_fp64<<<blocks, threads>>>(static_cast<double*>(C), matrix_volume/64);

			default:
				std::cerr << "unsupported CUDA datatype\n";
				return -1.0;
		}

		// set remaining GEMM inputs
		float alpha = 1.0;
		float beta = 0.5;

		// Run warmup GEMM
		cublasGemmEx(
				handle,
				CUBLAS_OP_N, CUBLAS_OP_N,
				M, N, K,
				&alpha,
				A, type, M,
				B, type, N,
				&beta,
				C, type, K,
				computeType,
				algo
		);
		
		// Run GEMM multiple times and take average TOPS
		int runs = 100;
		for(int i = 0; i < runs; ++i) {
			cudaEventRecord(start);
			cublasGemmEx(
					handle,
					CUBLAS_OP_N, CUBLAS_OP_N,
					M, N, K,
					&alpha,
					A, type, M,
					B, type, N,
					&beta,
					C, type, K,
					computeType,
					algo
			);
			cudaEventRecord(end);
			float ms = 0.0f;
			cudaEventElapsedTime(&ms, start, end);
			double ops = M * N * K * 2;
			tops += ops/(ms/1000);
		}

		// Add results to output
		tops /= runs;
		TOPS += tops;

		// clean up memory
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		cudaFree(A);
		cudaFree(B);
		cudaFree(C);
	}

	cublasDestroy(handle);

	TOPS /= edges.size();
	return TOPS;
}

double tensor_GEMM(int device) {
	//TODO
	return -1.0;
}
