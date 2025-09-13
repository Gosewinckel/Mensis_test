#include "machine.h"
#include <string>

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
	os.system = "linux";
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
#elif defined(__linux__) || defined(__APPLE__) || defined(__unix__)

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

