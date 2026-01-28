#pragma once
#include "machine.h"
#include "cpu_microbenchmarks.h"
#include "gpu_microbenchmarks.h"
#include "disk_microbenchmarks.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

/**************************************************** 
 * runCollectors -- runs machine collector functions 
 *		and writes results to a .json file
 *
 * Params: - outFile: .json file address where
 *		results will be written
 *
 * Returns: Boolean indicating success or failure
 *
 * Usage: Used by main runfile to put results in
 *		same file as microbenchmarks and AI_kernel
 *		results
 ****************************************************/ 
void runCollectors(json& output);

/**************************************************** 
 * runMicrobenchmarks -- runs microbenchmarks on 
 *		machices hardware and writes results to
 *		a .json file
 *
 * Params: - outFile: .json file address where results
 *		will be stored
 *
 * Returns: Boolean indicating success or failure
 *
 * Usage: Used by main runfile to put results into
 *		same file as collectors, to be used later for
 *		system analytics
 ****************************************************/ 
void runMicrobenchmarks(json& output);
