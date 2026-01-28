#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include "runner.h"

using json = nlohmann::json;

/**************************************************** 
 * Using main as a way to run C++ benchmarks and 
 * output to JSON. Main is run by python and the 
 * data read out of json file
 ****************************************************/ 
int main() {
	// setup file to be written to
	// File will be reset every run so no need to worry about checking
	std::ofstream f("raw_data/microbenchmarks_data.json");

	// json object that will go into data
	json benchmarks;

	// Run benchmarks and collectors
	runCollectors(benchmarks);
	runMicrobenchmarks(benchmarks);

	f << benchmarks.dump(2);
	return 0;
}
