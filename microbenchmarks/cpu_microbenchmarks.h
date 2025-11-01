#pragma once
#include <cstdint>
#include <vector>

/**************************************************** 
 * gemm_data -- data structure constaining all required 
 *		info for a BLAS_GEMM calculation
 *
 ****************************************************/ 
struct gemm_data {
	const uint64_t M;	//rows of op(A) and C
	const uint64_t N;	//cold of op(B) and C
	const uint64_t K;	//cols of op(A) / rows of op(B)
	const double alpha;		//scalar multiple of A*B
	std::vector<double> A;	// matrix A in vector form
	std::vector<double> B;	//matrix b in vector form
	const double beta;	//scalar multiple of C
	std::vector<double> C;	//matrix C in vector form
};

/**************************************************** 
 * square_gemm -- creates square A,B matrices
 *		to be used to find peak GFLOP/s
 *
 * Params: None
 *
 * Usage:
 *		Returns a gemm_data struct to be used in 
 *		bench_gemms(see below)
 ****************************************************/ 
void square_gemm(std::vector<gemm_data> *gemms);

/**************************************************** 
 * set_gemms -- creates a set of gemm_data structs
 *		to be used for computation in benchmarking 
 *		function
 *
 * Params:
 *		- gemms: an empty vector of gemm_data structs
 *			to be filled with GEMM multiplications of
 *			various shapes
 * Usage: 
 *		Input an empty gemm_data vector then use
 *		vector in benchmark function
 ****************************************************/ 
void set_gemms(std::vector<gemm_data> *gemms);

/**************************************************** 
 * bench_gemms_single -- calculates blas_gemms and 
 *		calculates the GFLOP/s avg from all 
 *		calculations on a single CPU thread
 *
 * Params:
 *		- gemms: a vector of GEMM data to be
 *		calculated
 * Returns:
 *		- float representing GFLOP/s
****************************************************/ 
float bench_gemms(int thread_count, std::vector<gemm_data> *gemms);

/**************************************************** 
 * gflop_single -- returns the GFLOP/s processed
 *		by a single CPU thread
 ****************************************************/ 
float gflop_single(std::vector<gemm_data>* gemms);

/**************************************************** 
 * gflop_multi -- return GFLOP/s processed 
 *		by all available CPU compute on a machine
 ****************************************************/ 
float gflop_multi(std::vector<gemm_data>* gemms);

/**************************************************** 
 * triad_size -- define the size of the arrays in a STREAM triad
 ****************************************************/ 
long triad_size();

/**************************************************** 
 * bandwidth_ -- returns memory transfer speed in 
 *		MB/s by calculating a STREAM triad 
 *
 * Params: None
 ****************************************************/ 
double bandwidth_single(long triad_size);

double bandwidth_multi(long triad_size);

/**************************************************** 
 * thread_wake_latency -- measures how long it takes 
 *		to spawn and wake threads on CPU
 *
 * Params: None
 *
 * returns: barrier synchronisation latency in 
 *		microseconds
 ****************************************************/ 
double thread_wake_latency();

/**************************************************** 
 * taks_dispatch_throughput -- measures how many 
 *		tasks CPU can manage per second
 *
 * Params: None
 *
 * returns: Tasks per second
 ****************************************************/ 
double task_dispatch_throughput();
