#include <gtest/gtest.h>
#include "machine.h"

TEST(machineTest, defaultTest) {
	machine m;
	EXPECT_EQ(m.get_os().system, "linux");
}
