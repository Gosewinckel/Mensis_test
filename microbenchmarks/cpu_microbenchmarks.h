#pragma once
#include <vector>

/**************************************************** 
 * gemm_data -- data structure constaining all required 
 *		info for a BLAS_GEMM calculation
 *
 ****************************************************/ 
struct gemm_data {
	const int M;	//rows of op(A) and C
	const int N;	//cold of op(B) and C
	const int K;	//cols of op(A) / rows of op(B)
	const double alpha;		//scalar multiple of A*B
	std::vector<double> A;	// matrix A in vector form
	std::vector<double> B;	//matrix b in vector form
	const double beta;	//scalar multiple of C
	std::vector<double> C;	//matrix C in vector form
};

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
float bench_gemms();

/**************************************************** 
 * gflop_single -- returns the GFLOP/s processed
 *		by a single CPU thread
 ****************************************************/ 
float gflop_single();

/**************************************************** 
 * gflop_multi -- return GFLOP/s processed 
 *		by all available CPU compute on a machine
 ****************************************************/ 
float gflop_multi();

