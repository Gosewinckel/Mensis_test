#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

double disk_write_speed(std::string filename) {
	std::ofstream test_file(filename);
	const long long file_size_bytes = 1024 * 1024 * 1024;
	std::vector<char> data(file_size_bytes, 'A');
	auto start = std::chrono::high_resolution_clock::now();
	if(test_file.is_open()) {
		test_file.write(data.data(), data.size());
		test_file.close();
	}
	else {
		return -1.0;
	}
	auto end = std::chrono::high_resolution_clock::now();
	double time = std::chrono::duration<double>(end - start).count();
	test_file.close();
	return 1/time;
}
