#include "machine.h"
#include <string>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <filesystem>
#include <set>

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

// CPU count setter
void machine::set_cpu_count() {
	std::set<int> packages;
	for(const auto& entry : std::filesystem::directory_iterator("/sys/devices/system/cpu")) {
		if(entry.path().filename().string().rfind("cpu", 0) == 0 && isdigit(entry.path().filename().string()[3])) {
			std::ifstream f(entry.path()/"topology/physical_package_id");
			int id;
			if(f >> id) {
				packages.insert(id);
			}
		}
	}
	cpu_count = packages.size();
}

// CPU setter
void machine::set_cpu() {
	// Loop through CPU's
	for(int i = 0; i < cpu_count; i++) {
		// set CPU model
		std::ifstream cpuinfo("/proc/cpuinfo");
		std::string line;
		while(std::getline(cpuinfo, line)) {
			if(line.find("model name") != std::string::npos) {
				cpu[i].model = line.substr(line.find(":") + 2);
			}
		}

		// find core count
		cpu[i].core_count = sysconf(_SC_NPROCESSORS_ONLN);

		// find clock speed
		cpuinfo.seekg(0, std::ios::beg);
		line.clear();
		while(std::getline(cpuinfo, line)) {
			if(line.find("cpu MHz") != std::string::npos) {
				cpu[i].clocks = std::stod(line.substr(line.find(":") + 2));
			}
		}
		// if clock speed not found set to -1
		if(cpu[i].clocks <=0.0) {
			cpu[i].clocks = -1.0;
		}

		// AVX support
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

