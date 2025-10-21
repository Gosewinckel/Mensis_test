#include <gtest/gtest.h>
#include <iostream>
#include "machine.h"


TEST(OSTest, defaultTest) {
	machine& m = machine::getMachine();
	EXPECT_EQ(m.get_os().system, "Linux");
}

TEST(CPUtest, cpu_count_test) {
	machine& m = machine::getMachine();
	EXPECT_EQ(m.get_cpu_count(), 1);
}

TEST(CPUTest, cpuModelTest ) {
	machine& m = machine::getMachine();
	std::cout << "Model: " << m.get_cpu()[0].model << "\n";
}

TEST(CPUTest, cpuCoreCountTest) {
	machine& m = machine::getMachine();
	std::cout  << "Core count: " << m.get_cpu()[0].core_count << "\n";
}

TEST(CPUTest, cpuAVXSupport) {
	machine& m = machine::getMachine();
	std::cout << "AVX support: " << m.get_cpu()[0].AVX_support << "\n";
}

TEST(CPUTest, cpuCacheSizes) {
	machine& m = machine::getMachine();
	for(int i = 0; i < m.get_cpu()[0].caches.size(); i++) {
		std::cout << "Cache " << i << " Level: " << m.get_cpu()[0].caches[i].level << ", Size: " << m.get_cpu()[0].caches[i].memory << "\n";
	}
}

TEST(CPUTest, cpuClockSpeed) {
	machine& m = machine::getMachine();
	std::cout << "Clock speed: " << m.get_cpu()[0].clocks << "\n";
}
