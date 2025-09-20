#include <gtest/gtest.h>
#include "machine.h"

TEST(machineTest, defaultTest) {
	machine m;
	EXPECT_EQ(m.get_os().system, "Linux");
}

TEST(machineTest, cpu_count_test) {
	machine m;
	EXPECT_EQ(m.get_cpu_count(), 1);
}
