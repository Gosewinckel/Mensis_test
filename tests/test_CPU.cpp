#include <gtest/gtest.h>
#include <iostream>
#include <omp.h>
#include <openblas/cblas.h>
#include "cpu_microbenchmarks.h"

TEST(CPUTest, sizeTest) {
	std::vector<gemm_data> gemms;
	set_gemms(&gemms);
	int m1_sizeA = gemms[0].M * gemms[0].K;
	int m1_sizeB = gemms[0].K * gemms[0].N;
	int m1_sizeC = gemms[0].M * gemms[0].N;
	// check sizes of arrays
	EXPECT_EQ(m1_sizeA, gemms[0].A.size());
	EXPECT_EQ(m1_sizeB, gemms[0].B.size());
	EXPECT_EQ(m1_sizeC, gemms[0].C.size());

	int m2_sizeA = gemms[1].M * gemms[1].K;
	int m2_sizeB = gemms[1].K * gemms[1].N;
	int m2_sizeC = gemms[1].M * gemms[1].N;	
	EXPECT_EQ(m2_sizeA, gemms[1].A.size());
	EXPECT_EQ(m2_sizeB, gemms[1].B.size());
	EXPECT_EQ(m2_sizeC, gemms[1].C.size());

	int m3_sizeA = gemms[2].M * gemms[2].K;
	int m3_sizeB = gemms[2].K * gemms[2].N;
	int m3_sizeC = gemms[2].M * gemms[2].N;
	EXPECT_EQ(m3_sizeA, gemms[2].A.size());
	EXPECT_EQ(m3_sizeB, gemms[2].B.size());
	EXPECT_EQ(m3_sizeC, gemms[2].C.size());
}

TEST(CPUTest, valTest) {
	std::vector<gemm_data> gemms;
	set_gemms(&gemms);
	EXPECT_EQ(0, gemms[0].A[0]);
	EXPECT_EQ(gemms[0].A.size() - 1, gemms[0].A[gemms[0].A.size() - 1]);
}

TEST(CPUTest, GEMMtest) {

	openblas_set_num_threads(openblas_get_num_procs());
	omp_set_num_threads(omp_get_max_threads());
	std::cout << "max threads: " << omp_get_max_threads() << "\n";
	std::cout << "\n" << "thread count: " << openblas_get_num_threads() << "\n";
	
	std::vector<gemm_data>* gemms = new std::vector<gemm_data>;
	square_gemm(gemms);

	float multiGFLOPSs = gflop_multi(gemms);
	std::cout << "finished multithread\n"; 
	float singleGFLOPs = gflop_single(gemms);
	std::cout << "\n";
	std::cout << "Max GFLOP/s: \n";
	std::cout << "Single thread GFLOP/s: " << singleGFLOPs << "\n";
	std::cout << "Multi thread GFLOP/s: " << multiGFLOPSs << "\n";
	std::cout <<"\n";

	set_gemms(gemms);
	multiGFLOPSs = gflop_multi(gemms);
	std::cout << "finished multithread\n"; 
	singleGFLOPs = gflop_single(gemms);
	std::cout << "\n";
	std::cout << "Reqalistic GFLOP/s: \n";
	std::cout << "Single thread GFLOP/s: " << singleGFLOPs << "\n";
	std::cout << "Multi thread GFLOP/s: " << multiGFLOPSs << "\n";
	std::cout <<"\n";
	//if(singleGFLOPs >= multiGFLOPSs) {
	//	EXPECT_TRUE(false);
	//}
}
