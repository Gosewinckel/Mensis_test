#include<gtest/gtest.h>
#include <iostream>
#include "gpu_microbenchmarks.h"

TEST(GPUTest, global_bandwidth_test) {
	double speed = global_GPU_mem_bandwidth(0);
	std::cout << "global bandwidth: " << speed << "GB/s\n";
}

TEST(GPUTest, TOPS_test) {
	double TOPS;
	TOPS = single_GPU_TOPS(0, CUBLAS_GEMM_DEFAULT, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUBLAS_DEFAULT_MATH);
	std::cout << "TOPS on 32F: " << TOPS << "\n";
}
