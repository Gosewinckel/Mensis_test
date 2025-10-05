#include <gtest/gtest.h>
#include <iostream>
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


