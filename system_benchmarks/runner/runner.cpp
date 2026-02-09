#include "runner.h"
#include "cpu_microbenchmarks.h"
#include "cublas_api.h"
#include "gpu_microbenchmarks.h"
#include <string>

void runCollectors(json& output) {
	// Generate machine
	machine& machine = machine::getMachine();

	// create json obj to put collector functions.
	json collectors;

	// fill json output with machine information
	// CPU info
	collectors["CPUs"] = json::array();
	int numCPUs = machine.get_cpu_count();
	for(int i = 0; i < numCPUs; ++i) {
		json cpuOut;
		machine::CPU cpu = machine.get_cpu()[i];
		cpuOut["model"] = cpu.model;
		cpuOut["core_count"] = cpu.core_count;
		cpuOut["clocks"] = cpu.clocks;
		cpuOut["AVX_support"] = cpu.AVX_support;
		
		// define caches
		cpuOut["caches"] =  json::array();
		for(int j = 0; j < cpu.caches.size(); ++j) {
			json cache;
			cache["level"] = cpu.caches[j].level;
			cache["memory"] = cpu.caches[j].memory;
			cpuOut["caches"].push_back(cache);
		}

		// append to cpu collectors
		collectors["CPUs"].push_back(cpuOut);
	}

	// GPU info
	collectors["GPUs"] = json::array();
	int numGPUs = machine.get_gpu_count();
	for(int i = 0; i < numGPUs; ++i) {
		json gpuOut;
		machine::GPU gpu = machine.get_gpu()[i];
		gpuOut["device"] = gpu.device;
		gpuOut["model"] = gpu.model;
		gpuOut["streaming_multiprocessors"] = gpu.streaming_multiprocessors;
		gpuOut["memory_capacity"] = gpu.memory_capacity;
		gpuOut["computeMajor"] = gpu.computeMajor;
		gpuOut["computeMinor"] = gpu.computeMinor;
		gpuOut["hasTensorCores"] = gpu.hasTensorCores;

		collectors["GPUs"].push_back(gpuOut);
	}

	// Memory info
	int numMem = machine.get_memory().size();
	collectors["RAM"] = json::array();
	for(int i = 0; i < numMem; ++i) {
		json memOut;
		machine::Memory mem = machine.get_memory()[i];
		memOut["size"] = mem.size;
		memOut["speed"] = mem.speed;

		collectors["RAM"].push_back(memOut);
	}

	// Storage type
	int numStorage = machine.get_storage().size();
	for(int i = 0; i < numStorage; ++i) {
		json storageOut;
		machine::Storage storage = machine.get_storage()[i];
		storageOut["type"] = storage.type;

		collectors["storage"].push_back(storageOut);
	}

	// Push to external json object
	output["collectors"] = collectors;
}

void runMicrobenchmarks(json& outFile) {
	// Set up JSON for microbenchmarks
	json microbenchmarks;

	// CPU microbenchmarks
	json cpuOut;
	std::vector<gemm_data> gemms;
	set_gemms(&gemms);
	cpuOut["single_thread_GFLOPs"] = gflop_single(&gemms);
	cpuOut["multi_thread_GFLOPs"] = gflop_multi(&gemms);
	long triad = triad_size();
	cpuOut["bandwidth(GB/s)"] = bandwidth_single(triad);
	cpuOut["thread_wake_latency(threads/microsecond)"] = thread_wake_latency();
	cpuOut["task_dispatch_throughput(tasks/microsecond)"] = task_dispatch_throughput();
	cpuOut["synchronisation_overhead(microseconds)"] = synchronisation_overhead();
	microbenchmarks["CPU"] = cpuOut;

	// GPU microbenchmarks
	machine& machine = machine::getMachine();
	int numGPUs = machine.get_gpu_count();
	microbenchmarks["GPUs"] = json::array();
	for(int i = 0; i < numGPUs; ++i) {
		json gpuOut;
		gpuOut["bandwidth"] = global_GPU_mem_bandwidth(i);
		json tops;
		float f32 = single_GPU_TOPS(i, CUBLAS_GEMM_DEFAULT, CUDA_R_32F, CUDA_R_32F, CUBLAS_DEFAULT_MATH);
		float f16 = single_GPU_TOPS(i, CUBLAS_GEMM_DEFAULT_TENSOR_OP, CUDA_R_16F, CUDA_R_32F, CUBLAS_DEFAULT_MATH);
		float i8 = single_GPU_TOPS(i, CUBLAS_GEMM_DEFAULT_TENSOR_OP, CUDA_R_8I, CUDA_R_32I, CUBLAS_TENSOR_OP_MATH);
		tops["32F"] = f32;
		tops["16f"] = f16;
		tops["8i"] = i8;
		gpuOut["TOPS"] = tops;
		gpuOut["device"] = i;

		microbenchmarks["GPUs"].push_back(gpuOut);
	}

	outFile["microbenchmarks"] = microbenchmarks;
}
