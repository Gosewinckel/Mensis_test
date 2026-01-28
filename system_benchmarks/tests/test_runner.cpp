#include <gtest/gtest.h>
#include "runner.h"
#include "nlohmann/json.hpp"
#include <fstream>

TEST(runnerTest, collectors) {
	json output;
	std::ofstream file("test1.json");
	runCollectors(output);
	runMicrobenchmarks(output);
	file << output.dump(2);

}
