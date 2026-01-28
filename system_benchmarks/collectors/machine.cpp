#include "machine.h"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <filesystem>
#include <set>
#include <map>
#include <array>
#include <stdexcept>
#include <sstream>
#include <cuda_runtime.h>
#include <algorithm>

#if defined(_WIN32) || defined(_WIN64)
	#include <windows.h>
	#include <VersionHelpers.h>
#elif defined(__linux__) || defined(__APPLE__) || defined(__unix__)
	#include <sys/utsname.h>
	#include <cstdio>
#endif


machine::machine() {
	// run setters
	set_os();
	set_cpu();
	set_gpu();
	set_memory();
	set_storage();
	//set_network();
}

machine& machine::getMachine() {
	static machine machine;
	return machine;
}

//Define member functions for Windows
#if defined(_WIN32) || defined(_WIN64)

void machine::set_os() {
	OSVERSIONINFOEXA version;
	ZeroMemory(&version, sizeof(OSVERSIONINFOEXA));
	version.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEXA);

	if(GetVersionEx((OSVERSIONINFO*)&VERSION)) {
		os.system = "Windows";
		os.version = version.dwMajorVersion + "." + version.dwMinorVersion;
		os.kernel = "NT Kernel " + std::to_string(version.dwBuildNumber);
	}
	else {
		os.system = "Windows";
		os.version = "unknown";
		os.kernel = "unknown";
	}
}

// Define member functions for Unix
#elif defined(__linux__) || defined(__unix__)



// helper function to run system commands and get output as a string
std::string exec(const char* cmd) {
	std::array<char, 128> buff;
	std::string result;
	std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
	if(!pipe) {
		throw std::runtime_error("popen() failed");
	}
	while(!feof(pipe.get())) {
		if(fgets(buff.data(), 128, pipe.get()) != nullptr) {
			result += buff.data();
		}
	}
	return result;
}

// OS setter
void machine::set_os() {
	struct utsname buffer;
	if(uname(&buffer) == 0) {
		os.system = buffer.sysname;
		os.version = buffer.version;
		os.kernel = std::string(buffer.sysname) + " " + buffer.release;
	}
	else {
		os.system = "Unknown unix";
		os.version = "unknown";
		os.kernel = "unknown";
	}
}


// CPU setter
void machine::set_cpu() {
	CPU cpuInfo;
	// Find number of CPU's
	std::set<int> physicalPackages;			// set of physical CPU's
	std::map<int, std::string> packageMap;		// Maps CPU to a cpu file
	for(const auto& entry : std::filesystem::directory_iterator("/sys/devices/system/cpu")) {
		if(entry.path().filename().string().rfind("cpu", 0) == 0 && isdigit(entry.path().filename().string()[3])) {
			std::ifstream f(entry.path()/"topology/physical_package_id");
			int id;
			if(f >> id && physicalPackages.find(id) == physicalPackages.end()) {
				physicalPackages.insert(id);
				packageMap[id] = entry.path().string();
			}
		}
	}
	// set number of CPU's
	cpu_count = physicalPackages.size();
	
	// loop through CPU's
	for(const auto& [pkgId, cpuDir] : packageMap) {
		std::ifstream cpuinfo("/proc/cpuinfo");
		std::string line;
		while(std::getline(cpuinfo, line)) {
			//Skip other CPU's
			if(line.find("physical id") != std::string::npos) {
				int physId = std::stoi(line.substr(line.find(":") + 1));
				if(physId != pkgId) {
					continue;
				}
			}
			// find CPU model
			if(line.find("model name") != std::string::npos) {
				cpuInfo.model = line.substr(line.find(":") + 2);
			}
			// find clock speed
			if(line.find("cpu MHz") != std::string::npos) {
				cpuInfo.clocks = std::stod(line.substr(line.find(":") + 2));
			}
			// AVX support
			if(line.find("flags") != std::string::npos) {
				cpuInfo.AVX_support = (line.find("avx") != std::string::npos);
			}
		}

		// Core count for current CPU
		int coreCount = 0;
		for(const auto& entry: std::filesystem::directory_iterator("/sys/devices/system/cpu")) {
			if(entry.path().filename().string().rfind("cpu", 0) == 0 && isdigit(entry.path().filename().string()[3])) {
				std::ifstream f(entry.path()/"topology/physical_package_id");
				int physId;
				if(f >> physId && physId == pkgId) {
					++coreCount;
				}
			}
		}
		cpuInfo.core_count = coreCount;

		// Cache sizes
		for(const auto& cache_i: std::filesystem::directory_iterator(cpuDir + "/cache")) {
			if(cache_i.path().filename().string().rfind("index", 0) == 0 && isdigit(cache_i.path().filename().string()[5])) {
				// Find cache level
				std::ifstream f(cache_i.path()/"level");
				int level = 0;
				f >> level;

				// find cache size
				std::ifstream s(cache_i.path()/"size");
				size_t size;
				s >> size;
				
				// add to caches
				cpuInfo.caches.push_back({level, size});
			}
		}
	}	
	cpu.push_back(cpuInfo);
}


/**************************************************** 
 * set GPU function
 ****************************************************/ 
void machine::set_gpu() {
	std::vector<GPU> gpus;
	
	// Get device count
	int count = 0;
	cudaError_t err = cudaGetDeviceCount(&count);
	if(err != cudaSuccess) {
		return;
	}
	gpu_count = count;

	// Get information for each GPU
	for(int i = 0; i < count; ++i) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		GPU gpu;
		gpu.device = i;
		gpu.model = prop.name;
		gpu.streaming_multiprocessors = prop.multiProcessorCount;
		gpu.memory_capacity = prop.totalGlobalMem;
		gpu.computeMajor= prop.major;
		gpu.computeMinor = prop.minor;
		gpu.hasTensorCores = (prop.major >= 7);

		gpus.push_back(gpu);
	}
	gpu = gpus;
}
/**************************************************** 
 * set memory fuction
 *
 * used in machine constructor
 *
 ****************************************************/ 
void machine::set_memory() {
	const char *cmd = "sudo dmidecode --type memory";
	std::string dmi;
	dmi = exec(cmd);
	std::istringstream iss(dmi);
	std::string line;
	int sizebuff;
	int speedbuff;

	// loop through dmidecode output
	while(std::getline(iss, line)) {
		if(line.find("Size") != std::string::npos && line.find("No Module Installed") == std::string::npos) {
			sizebuff = std::stoi(line.substr(line.find(":") + 2));
		}
		else if(line.find("Speed") != std::string::npos) {
			speedbuff = std::stoi(line.substr(line.find(":") + 2));
			memory.push_back({sizebuff, speedbuff});

			// move to next memory device
			while(std::getline(iss, line)) {
				if(line.find("Memory Device") != std::string::npos) {
					break;
				}
			}
		}
	}
}

/**************************************************** 
 * set Storage
 *
 * used in machine constructor to set storage type
 * and read/write throughput
 ****************************************************/ 
void machine::set_storage() {
	for(const auto& entry : std::filesystem::directory_iterator("/sys/block")) {
		std::ifstream m(entry.path()/"queue/rotational");
		std::string line;

		while(std::getline(m, line)) {
			if(std::stoi(line) == 0) {
				storage.push_back({"SSD/NVMe"});
			}
			else if(std::stoi(line) == 1) {
				storage.push_back({"HDD"});
			}
		}
	}
}

/**************************************************** 
 * set Network
 *
 * used in machine constructor to find NIC model
 * and link speed
 ****************************************************/ 
void machine::set_network() {
	const char *cmd = "ls /sys/class/net";
	std::string hard_nets;
	hard_nets = exec(cmd);
	std::istringstream iss(hard_nets);
	std::string line;
	std::vector<std::string> wired_nets;
	
	// loop through network links to check for wired interfaces
	while(std::getline(iss, line)) {
		if(line == "lo") continue;
		if(line.rfind("docker", 0) == 0) continue;
		if(line.rfind("veth", 0) == 0) continue;
		if(line.rfind("br", 0) == 0) continue;
		if(line.rfind("virbr", 0) == 0) continue;
		if(line.rfind("tun", 0) == 0) continue;
		else {
			wired_nets.push_back(line);
		}
	}
	 
	//loop through again to check for active IP link
	for(int i = 0; i < wired_nets.size(); ++i) {
		std::string ip_linked = "ip -4 addr show " + wired_nets[i] + " | grep 'inet ' | awk '{print $2}'";
		std::string ip;
		ip = exec(ip_linked.data());
		if(ip.empty()) {
			wired_nets.erase(wired_nets.begin() + i);
		}
		// check if end of vector is found
		if(wired_nets.size() == i - 1) {
			break;
		}
	}

	int high_speed = 0;
	if(wired_nets.size() == 1) {
		std::ifstream m("/sys/class/net/" + wired_nets[0] + "/speed");
		std::string speed;
		std::getline(m, speed);
		network.link_speed = std::stoi(speed);
	}
	else {
		for(int i = 0; i < wired_nets.size(); i++) {
			std::ifstream m("/sys/class/net/" + wired_nets[0] + "/speed");
			std::string speed;
			std::getline(m, speed);
			if(std::stoi(speed) > high_speed) {
				high_speed = std::stoi(speed);
			}
		}
		network.link_speed = high_speed;
	}
}

// Define member functions for macOS
#elif defined(__APPLE__)

std::cerr << "Mensis test not currently supported on macOS\n";

void machine::set_os() {
	struct utsname buffer;
	if(uname(&buffer) == 0) {
		os.system = buffer.sysname;
		os.version = buffer.version;
		os.kernel = std::string(buffer.sysname) + " " + buffer.release;
	}
	else {
		os.system = "Unknown unix";
		os.version = "unknown";
		os.kernel = "unknown";
	}
}
// Define functions for unknown OS
#else

void machine::set_os() {
	os.system = "Unknown OS";
	os.version = "unknown";
	os.kernel = "unknown";
}

#endif

