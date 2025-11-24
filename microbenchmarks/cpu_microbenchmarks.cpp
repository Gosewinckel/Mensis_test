#include "machine.h"
#include <openblas/cblas.h>
#include <openblas/openblas_config.h>
#include <iostream>
#include <ratio>
#include <vector>
#include <chrono>
#include "cpu_microbenchmarks.h"
#include <omp.h>
#include <cmath>

void merge(std::vector<double> &data, int left, int mid, int right);
void mergesort(std::vector<double> &data, int start, int end);

void square_gemm(std::vector<gemm_data> *gemms) {
	machine& m = machine::getMachine();
	gemms->clear();
	
	// Max throuput will be matrix of max size that can fit on an L3 cache
	// Compute is memory bound so multiple CPU's would hinder computation until compute becomes massive
	std::vector<machine::CPU> cpu = m.get_cpu();
	int max_L3 = 0;
	for(int i = 0; i < cpu.size(); ++i) {
		for(int j = 0; j < cpu[i].caches.size(); ++j) {
			if(cpu[i].caches[j].level == 3) {
				if(cpu[i].caches[j].memory > max_L3) {
					max_L3 = cpu[i].caches[j].memory;
				}
				else {
					break;
				}
			}
		}
	}

	uint64_t num_bytes = (((max_L3 * 1000) / 3) / (3 * 8));	
	uint64_t sqrt = std::sqrt(num_bytes);


	uint64_t M = sqrt;
	uint64_t N = sqrt;
	uint64_t K = sqrt;
	double alpha = 1.0;
	double beta = 0.5;
	std::vector<double> A, B, C;
	for(int i = 0; i < M*K; ++i) {
		A.push_back(static_cast<double>(i % 10));
		B.push_back(static_cast<double>(i % 10));
		C.push_back(static_cast<double>(i % 10));
	}
	gemm_data buffer0{M, N, K, alpha, A, B, beta, C};
	gemms->push_back(buffer0);
}

void set_gemms(std::vector<gemm_data> *gemms) {
	// Clear vector in case of existing data
	gemms->clear();
	
	// set machine to gather machine data
	machine& m = machine::getMachine();

	// Max throuput will be matrix of max size that can fit on an L3 cache
	// Compute is memory bound so multiple CPU's would hinder computation until compute becomes massive
	std::vector<machine::CPU> cpu = m.get_cpu();
	int max_L3 = 0;
	for(int i = 0; i < cpu.size(); ++i) {
		for(int j = 0; j < cpu[i].caches.size(); ++j) {
			if(cpu[i].caches[j].level == 3) {
				if(cpu[i].caches[j].memory > max_L3) {
					max_L3 = cpu[i].caches[j].memory;
				}
				else {
					break;
				}
			}
		}
	}

	uint64_t num_bytes = (((max_L3 * 1000) / 3) / (3 * 8));
	uint64_t sqrt = std::sqrt(num_bytes);

	//We want the total computation to take up between 40-60% of the
	//L3 cache
	std::vector<uint64_t> edges = {sqrt/4, sqrt/5, sqrt/3,sqrt/2, (sqrt * 3)/4, (sqrt * 4) / 5,
									(sqrt * 5)/4, (sqrt * 4) /3, sqrt * 2, sqrt * 3, sqrt * 4, sqrt}; 

	//  Create gemm structs and append to gemms
	for(int i = 0; i < edges.size(); ++i) {
		uint64_t M = edges[i];
		uint64_t N = num_bytes/edges[i];
		uint64_t K = edges[i];
		double alpha = 1.0;
		double beta = 0.5;
		std::vector<double> A, B, C;
		for(int k = 0; k < M*K; ++k) {
			A.push_back(static_cast<double>(i % 10));
		}
		for(int k = 0; k < N*K; ++k) {
			B.push_back(static_cast<double>(i % 10));
		}
		for(int k = 0; k < M*N; ++k) {
			C.push_back(static_cast<double>(0));
		}
		gemms->push_back({M, N, K, alpha, A, B, beta, C});
	}
}

float bench_gemms(int thread_count, std::vector<gemm_data>* gemms) {

	long flops = 0;	//total No of flops computed
	double time = 0.0;	// total number of nanoseconds for computation
	
	// Set thread count (must be done for every call of GEMM)
	omp_set_num_threads(thread_count);


	// Run blas_gemm for every point in gemms
	for(int i = 0; i < gemms->size(); ++i) {
		//warmup run
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			(*gemms)[i].M, (*gemms)[i].N, (*gemms)[i].K, (*gemms)[i].alpha,
			(*gemms)[i].A.data(), (*gemms)[i].K, (*gemms)[i].B.data(),
			(*gemms)[i].N, (*gemms)[i].beta, (*gemms)[i].C.data(), (*gemms)[i].N);

		// Repeat computation 
		for(int j = 0; j < 100; ++j) {
			auto start = std::chrono::high_resolution_clock::now();

			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				(*gemms)[i].M, (*gemms)[i].N, (*gemms)[i].K, (*gemms)[i].alpha,
				(*gemms)[i].A.data(), (*gemms)[i].K, (*gemms)[i].B.data(),
				(*gemms)[i].N, (*gemms)[i].beta, (*gemms)[i].C.data(), (*gemms)[i].N);

			auto end = std::chrono::high_resolution_clock::now();
			time += std::chrono::duration<double>(end - start).count();
			flops += 2 * (*gemms)[i].M * (*gemms)[i].N * (*gemms)[i].K;
			++(*gemms)[i].C[0];
		}
	}

	// Calculate and return GFLOPS/s
	float GFLOPs = (flops / 1e9) / time; 
	return GFLOPs;
}

float gflop_single(std::vector<gemm_data>* gemms) {
	return bench_gemms(1, gemms);
}

float gflop_multi(std::vector<gemm_data>* gemms) {
	return bench_gemms(openblas_get_num_procs(), gemms);
}

long triad_size() {
	// Determine size of arrays
	machine& m = machine::getMachine();
	int cache_capacity = 0;
	std::vector<machine::CPU> cpu = m.get_cpu();
	for(int i = 0; i < cpu.size(); i++) {
		for(int j = 0; j < cpu[i].caches.size(); j++) {
			cache_capacity += cpu[i].caches[j].memory;
		}
	}
	// Convert to bytes
	cache_capacity *= 1000;
	int array_size = 0;
	if(cache_capacity/8 >= 1000000) {
		array_size = cache_capacity/8;
	}
	else {
		array_size = 1000000;
	}
	return array_size;
}

double bandwidth_single(long triad_size) {
	double* a = new double[triad_size] ;
	double* b = new double[triad_size];
	double* c = new double[triad_size];
	double scalar = 3.0;

	// Run triad 10 times and reset values in between
	double best_time = 0;
	// set array values
	for(int i = 0; i < triad_size; i++) {
		a[i] = 1.0;
		b[i] = 2.0;
		c[i] = 0.0;
	}

	double check;

	for(int i = 0; i < 10; i++) {
		//record time
		auto start = std::chrono::high_resolution_clock::now();
		for(int x = 0; x < triad_size; x++) {
			c[x] = a[x] + scalar * b[x];
		}
		auto end = std::chrono::high_resolution_clock::now();
		double runtime = std::chrono::duration<double>(end - start).count();
		if(runtime < best_time || best_time == 0) {
			best_time = runtime;
		}
		// Complete checksum
		check = 0.0;
		for(int j = 0; j < triad_size; ++j) {
			check += c[j];
		}
	}
	c[0] += 1;

	delete[] a;
	delete[] b;
	delete[] c;
	a = b = c = NULL;
	// Calculate top GB/s
	double bytes = 3.0 * triad_size * sizeof(double);
	double bandwidth_GBs = bytes/best_time/1e9;
	return bandwidth_GBs;
}

double bandwidth_multi(long triad_size) {
	double* a = new double[triad_size];
	double* b = new double[triad_size];
	double* c = new double[triad_size];
	double scalar = 3.0;
	double best_time = 0.0;
	for(int i = 0; i < 10; i++) {
		//set/reset values
		for(int j = 0; j < triad_size; j++) {
			a[i] = 1.0;
			b[i] = 2.0;
			c[i] = 0.0;
		}

		double check;

		// record time
		auto start = std::chrono::high_resolution_clock::now();
		#pragma omp parallel for
		for(int x = 0; x < triad_size; x++) {
			c[x] = a[x] + scalar * b[x];
		}
		auto end = std::chrono::high_resolution_clock::now();
		double runtime = std::chrono::duration<double>(end - start).count();
		if(runtime < best_time || best_time == 0.0) {
			best_time  = runtime;
		}
		//checksum
		check = 0.0;
		for(int j = 0; j < triad_size; ++j) {
			check += c[j];
		}
	}
	c[0] += 1;
	delete[] a;
	delete[] b;
	delete[] c;
	a = b = c = NULL;
	// Calculate GB/s
	double bytes = 3.0 * triad_size * sizeof(double);
	double bandwidth_GBs = bytes/best_time/1e9;
	return bandwidth_GBs;
}

double thread_wake_latency() {
	std::vector<double> times;
	for(int i = 0; i < 100; ++i) {
		#pragma omp parallel
		{
			auto start = std::chrono::high_resolution_clock::now();
			#pragma omp barrier
			auto end = std::chrono::high_resolution_clock::now();
			if(omp_get_thread_num() == 0) {
				times.push_back(std::chrono::duration<double, std::micro>(end - start).count());
			}
		}
	}
	// Use mergesort to sort vector then find median value
	mergesort(times, 0, times.size() - 1);	
	return times[times.size()/2];
}

void mergesort(std::vector<double> &data, int left, int right) {
	if(left >= right) {
		return;
	}
	int mid = left + (right - left) / 2;
	mergesort(data, left, mid);
	mergesort(data, mid + 1, right);
	merge(data, left, mid, right);
}

void merge(std::vector<double> &data, int left, int mid, int right) {
	int s1 = mid - left + 1;
	int s2 = right - mid;

	std::vector<double> L(s1), R(s2);
	for(int i = 0; i < s1; ++i) {
		L[i] = data[left + i];
	}
	for(int i = 0; i < s2; ++i) {
		R[i] = data[mid + i + 1];
	}

	int i = 0;
	int j = 0;
	int k = left;

	while(i < s1 && j < s2) {
		if(L[i] >= R[j]) {
			data[k] = L[i];
			++k;
			++i;
		}
		else if(L[i] < R[j]) {
			data[k] = R[j];
			++j;
			++k;
		}
	}
	while(i < s1) {
		data[k] = L[i];
		++k;
		++i;
	}
	while(j < s2) {
		data[k] = R[j];
		++k;
		++j;
	}
}

double task_dispatch_throughput() {
	int tasks = 1e6;
	auto start = std::chrono::high_resolution_clock::now();

	#pragma omp parallel for schedule(dynamic)
	for(int i = 0; i < tasks; i++) {
		// Lightweght task
		double x = std::sin(i * 0.001);
	}
	
	auto end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double>(end - start).count();
	return tasks/seconds;
}

double synchronisation_overhead() {
	int iterations = 1e6;
	auto start = std::chrono::high_resolution_clock::now();
	#pragma omp parallel 
	{
		for(int i = 0; i < iterations; ++i) {
			#pragma omp barrier
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	double seconds = std::chrono::duration<double, std::micro>(end - start).count();
	return seconds/iterations;
}
