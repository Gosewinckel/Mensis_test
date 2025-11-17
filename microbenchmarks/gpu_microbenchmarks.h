#pragma once

/**************************************************** 
 * This file defines a set of GPU benchmarks that 
 * will test the Global and block memory bandwidth,
 * host to device memcopy speed and CUDA code and
 * Tensor core compute throughput
 ****************************************************/ 

/**************************************************** 
 * Global_GPU_mem_bandwidth -- tests the max GPU
 *		memory bandwidth from the global memory
 *
 * Params: - device: the number identifying the GPU
 *			to run on
 *
 * Returns: bandwidth in GB/s
 ****************************************************/ 
double global_GPU_mem_bandwidth(int device);

/**************************************************** 
 * cuda_GEMM -- tests peak computation speed on CUDA 
 *		cores
 *
 * Params: device
 *
 * Returns: GFLOPS/s
 ****************************************************/ 
double cuda_GEMM(int device);
