#include "machine.h"
#include <string>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <filesystem>
#include <set>
#include <map>

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
	set_cpu_count();
}

//Define member functions for Windows
#if defined(_WIN32) || defined(_WIN64)

std::cerr << "Mensis test not currently supported on Windows\n";

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
		int coreCount;
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

