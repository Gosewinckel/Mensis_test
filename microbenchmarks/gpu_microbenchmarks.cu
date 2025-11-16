#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <gpu_microbenchmarks.h>

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
	
	// return GB/s
	double bytes = 3.0 * N * sizeof(float);
	double GBs = (bytes/1e9)/(time/1000.0);
	return GBs;
}
