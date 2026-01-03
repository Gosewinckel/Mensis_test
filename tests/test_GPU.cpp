#include<gtest/gtest.h>
#include <iostream>
#include "cublas_api.h"
#include "gpu_microbenchmarks.h"
#include "library_types.h"


TEST(GPUTest, global_bandwidth_test) {
	double speed = global_GPU_mem_bandwidth(0);
	std::cout << "global bandwidth: " << speed << "GB/s\n";
}


TEST(GPUTest, TOPS_test) {
	double TOPS;
	TOPS = single_GPU_TOPS(0, CUBLAS_GEMM_DEFAULT, CUDA_R_32F, CUDA_R_32F, CUBLAS_DEFAULT_MATH);
	std::cout << "TOPS on 32F: " << TOPS << "\n";
}


TEST(GPUTest, TOPS_test_16F_CUDA) {
	double TOPS;
	TOPS = single_GPU_TOPS(0, CUBLAS_GEMM_DEFAULT_TENSOR_OP, CUDA_R_16F, CUDA_R_32F, CUBLAS_DEFAULT_MATH);
	std::cout << "TOPS on 16F CUDA: " << TOPS << "\n";
}


TEST(GPUTest, TOPS_test_8I_CUDA) {
	double TOPS;
	TOPS = single_GPU_TOPS(0, CUBLAS_GEMM_DEFAULT_TENSOR_OP, CUDA_R_8I, CUDA_R_32I, CUBLAS_TENSOR_OP_MATH);
	std::cout << "TOPS on 8I CUDA: " << TOPS << "\n";
}


TEST(GPUTest, TOPS_test_64F_CUDA) {
	double TOPS;
	TOPS = single_GPU_TOPS(0, CUBLAS_GEMM_DEFAULT, CUDA_R_64F, CUDA_R_64F, CUBLAS_DEFAULT_MATH);
	std::cout << "TOPS on 64F CUDA: " << TOPS << "\n";
}
