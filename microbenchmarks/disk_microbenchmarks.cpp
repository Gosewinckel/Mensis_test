#include <vector>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

double disk_write_speed(std::string filename) {
	std::ofstream test_file(filename);
	const long long file_size_bytes = 1024 * 1024 * 1024;
	std::vector<char> data(file_size_bytes, 'A');
	return 0.0;
}
