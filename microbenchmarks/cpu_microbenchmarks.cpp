#include <openblas/cblas.h>
#include <vector>
#include <chrono>
#include "cpu_microbenchmarks.h"


void set_gemms(std::vector<gemm_data> *gemms) {
	// Clear vector in case of existing data
	gemms->clear();

	// Test 3 different sizes of approx equal compute
	// 1st multiple
	int M = 64;
	int N = 64;
	int K = 64;
	double alpha = 1.0;
	double beta = 0.5;
	std::vector<double> A, B, C;
	for(int i = 0; i < M*K; i++) {
		A.push_back(static_cast<double>(i % 10));
		B.push_back(static_cast<double>(i % 10));
		C.push_back(static_cast<double>(i % 10));
	}
	gemm_data buffer0{M, N, K, alpha, A, B, beta, C};
	gemms->push_back(buffer0);

	// 2nd multiple
	M = 128;
	N = 128;
	K = 32;
	A.clear();
	B.clear();
	C.clear();
	for(int i = 0; i < M*N; i++) {
		if(i < M*K) {
			A.push_back(static_cast<double>(i % 10));
			B.push_back(static_cast<double>(i % 10));
		}
		C.push_back(static_cast<double>(i % 10));
	}
	gemm_data buffer1{M, N, K, alpha, A, B, beta, C};
	gemms->push_back(buffer1);

	// 3rd multiple
	M = 32;
	N = 32;
	K = 128;
	A.clear();
	B.clear();
	C.clear();
	for(int i = 0; i < M*K; i++) {
		A.push_back(static_cast<double>(i % 10));
		B.push_back(static_cast<double>(i % 10));
		if(i < M*N) {
			C.push_back(static_cast<double>(i % 10));
		}
	}
	gemm_data buffer2{M, N, K, alpha, A, B, beta, C};
	gemms->push_back(buffer2);
}

float bench_gemms() {
	// Initialise GEMM matrices
	std::vector<gemm_data> gemms;
	set_gemms(&gemms);
	long flops = 0;	//total No of flops computed
	double time = 0.0;	// total number of nanoseconds for computation
	
	// warmup run
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			gemms[0].M, gemms[0].N, gemms[0].K, gemms[0].alpha,
			gemms[0].A.data(), gemms[0].K, gemms[0].B.data(),
			gemms[0].N, gemms[0].beta, gemms[0].C.data(), gemms[0].N);

	// repeat computation 100 times and track time taken
	for(int i = 0; i < 100; i++) {
		auto start = std::chrono::high_resolution_clock::now();		
		
		// run GEMM
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			gemms[0].M, gemms[0].N, gemms[0].K, gemms[0].alpha,
			gemms[0].A.data(), gemms[0].K, gemms[0].B.data(),
			gemms[0].N, gemms[0].beta, gemms[0].C.data(), gemms[0].N);

		auto end = std::chrono::high_resolution_clock::now();
		time += std::chrono::duration<double>(end - start).count();
		flops += 2 * gemms[0].M * gemms[0].N * gemms[0].K;
	}

	// Checksum so compiler doesn't remove logic
	++gemms[0].C[0]; 	

	//warmup for GEMM 2
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			gemms[1].M, gemms[1].N, gemms[1].K, gemms[1].alpha,
			gemms[1].A.data(), gemms[1].K, gemms[1].B.data(),
			gemms[1].N, gemms[1].beta, gemms[1].C.data(), gemms[1].N);
	
	for(int i = 0; i < 100; i++) {
		auto start = std::chrono::high_resolution_clock::now();

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			gemms[1].M, gemms[1].N, gemms[1].K, gemms[1].alpha,
			gemms[1].A.data(), gemms[1].K, gemms[1].B.data(),
			gemms[1].N, gemms[1].beta, gemms[1].C.data(), gemms[1].N);

		auto end = std::chrono::high_resolution_clock::now();
		time += std::chrono::duration<double>(end - start).count();
		flops += 2 * gemms[1].M * gemms[1].N * gemms[1].K;
	}

	++gemms[1].C[0];

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			gemms[2].M, gemms[2].N, gemms[2].K, gemms[2].alpha,
			gemms[2].A.data(), gemms[2].K, gemms[2].B.data(),
			gemms[2].N, gemms[2].beta, gemms[2].C.data(), gemms[2].N);
	
	for(int i = 0; i < 100; i++) {
		auto start = std::chrono::high_resolution_clock::now();

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			gemms[2].M, gemms[2].N, gemms[2].K, gemms[2].alpha,
			gemms[2].A.data(), gemms[2].K, gemms[2].B.data(),
			gemms[2].N, gemms[2].beta, gemms[2].C.data(), gemms[2].N);

		auto end = std::chrono::high_resolution_clock::now();
		time += std::chrono::duration<double>(end - start).count();
		flops += 2 * gemms[2].M * gemms[2].N * gemms[2].K;
	}

	++gemms[2].C[0];

	// Calculate and return GFLOPS/s
	float GFLOPs = (flops / 1e9) / time; 
	return GFLOPs;
}

float gflop_single() {
	openblas_set_num_threads(1);
	return bench_gemms();
}

float gflop_multi() {
	openblas_set_num_threads(openblas_get_num_procs());
	return bench_gemms();
}
