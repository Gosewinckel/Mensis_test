#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <gpu_microbenchmarks.h>
#include <vector>
#include <cmath>
#include <ratio>

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
		cudaEvent_t start, end;
		cudaEventCreate(&start);
		cudaEventCreate(&end);

		cudaEventRecord(start);
		global_triad<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, N);
		cudaEventRecord(end);
		cudaEventSynchronize(end);
		float ms = 0.0f;
		cudaEventElapsedTime(&ms, start, end);
		time += ms;
	}

	//average results
	time /= runs;

	cudaMemcpy(h_C, d_C, N, cudaMemcpyDeviceToHost);
	float checksum = 0.0;
	for(int i = 0; i < N; ++i) {
		checksum += h_C[i];
	}

	cudaEventElapsedTime(&time, start, stop);

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
	double GBs = (bytes/1e9)/(time/1000.0);
	return GBs;
}

double cuda_GEMM(int device) {
	int numDevices = 0;
	cudaGetDeviceCount(&numDevices);
	if(device >= numDevices) {
		return -1;
	}
	cudaSetDevice(device);

	// find GPU onboard memory
	size_t free_bytes;
	size_t total_bytes;
	cudaMemGetInfo(&free_bytes, &total_bytes);

	// Each matrix = 20% of GPU memory
	uint64_t matrix_size = 0;
	matrix_size = (total_bytes * (1/5)) / 4;
	uint64_t sqrt = std::sqrt(matrix_size);

	float TOPs = 0.0; //trillion operations per second, avg results of each GEMM and return

	// Set of matrix side sizes
	std::vector<uint64_t> edges = {sqrt/4, sqrt/5, sqrt/3, sqrt/2, (sqrt * 3)/4,
		(sqrt * 4)/5, (sqrt * 5)/4, (sqrt * 4)/3, sqrt * 2, sqrt * 3,
		sqrt * 4, sqrt};

	// Calculate TOPS for each GEMM dimensions
	for(int i = 0; i < edges.size(); ++i) {
		//dynamically allocated matrices
		float* d_A = new float[edges[i]];
		float* d_B = new float[matrix_size/edges[i]];
		float* d_C = new float[edges[i]];
		
		// Fill matrices with mock data
		for(int j = 0; j < edges[i]; ++j) {
			d_A[j] = 1.0;
			d_B[j] = 2.0;
		}
		for(int j = 0; j < matrix_size/edges[i]; ++j) {
			d_C[j] = 0;
		}
		// -- continue on with actual GEMM -- //
	}	

	// Set 

	return TOPs;

}

double tensor_GEMM(int device) {
	//TODO
	return -1.0;
}
