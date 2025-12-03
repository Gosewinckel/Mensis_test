#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>

/**************************************************** 
 * This file defines a set of GPU benchmarks that 
 * will test the Global and block memory bandwidth,
 * host to device memcopy speed and CUDA code and
 * Tensor core compute throughput
 ****************************************************/ 

/**************************************************** 
 * global_GPU_mem_bandwidth -- tests the max GPU
 *		memory bandwidth from the global memory
 *
 * Params: - device: the number identifying the GPU
 *			to run on
 *
 * Returns: bandwidth in GB/s
 ****************************************************/ 
double global_GPU_mem_bandwidth(int device);

/**************************************************** 
 * single_GPU_TOPS -- tests peak computation speed on
 *		a single GPU 
 *		
 *
 * Params: device
 *			algo: CUBLAS_GEMM_DEFAULT for cuda cores
 *				  CUBLAS_GEMM_DEFAULT_TENSOR_OP for tensor cores
 *			type: CUDA_R_32F
 *				  CUDA_R_16F
 *				  CUDA_R_16BF
 *				  CUDA_R_8I
 *				  CUDA_R_64F
 *			computeType: CUBLAS_COMPUTE_32F
 *						 CUBLAS_COMPUTE_32F_FAST_TF32
 *						 CUBLAS_COMPUTE_16F
 *						 CUBLAS_COMPUTE_16BF
 *						 CUDA_R_64F
 *			mathMode: CUBLAS_DEFAULT_MATH
 *					  CUBLAS_TENSOR_OP_MATH
 *					  CUBLAS_TF32_TENSOR_OP_MATH
 *
 *
 * Returns: GFLOPS/s
 ****************************************************/ 
double single_GPU_TOPS(
		int device,
		cublasGemmAlgo_t algo,
		cudaDataType_t type,
		cublasComputeType_t computeType,
		cublasMath_t mathMode
);

/**************************************************** 
 * tensor_GEMM -- tests peak computation speed on
 *		tensor cores
 *
 * Params: device
 *
 * Returns: GFLOP/s
 ****************************************************/ 
double tensor_GEMM(int device);
