#include<gtest/gtest.h>
#include <iostream>
#include "gpu_microbenchmarks.h"

TEST(GPUTest, global_bandwidth_test) {
	double speed = global_GPU_mem_bandwidth(0);
	std::cout << "global bandwidth: " << speed << "GB/s\n";
}
