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
 * Params: None
 *
 * Returns: bandwidth in GB/s
 ****************************************************/ 
double global_GPU_mem_bandwidth(int device);
