#include <gtest/gtest.h>
#include <string>
#include <iostream>
#include "disk_microbenchmarks.h"

const std::string file = "testFile.txt";

TEST(diskTEST, write_speed_test) {
	double speed = disk_write_speed(file);
	std::cout << "Disk write speed: " << speed << "GB/s\n";
}

TEST(diskTEST, read_speed_test) {
	double speed = disk_read_speed(file);
	std::cout <<  "Disk read speed: " << speed << " GB/s\n";
}
