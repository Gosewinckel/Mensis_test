#include <iostream>
#include "collectors/machine.h"

int main() {
	std::cout << "running\n";
	machine m;
	std::cout << m.get_os().system << "\n";
	std::cout << m.get_cpu()[0].core_count << "\n";
	std::cout << m.get_cpu()[0].AVX_support << "\n";
	std::cout << m.get_cpu()[0].model << "\n";
	std::cout << m.get_cpu()[0].clocks << "\n";
	for(int i = 0; i < m.get_cpu()[0].caches.size(); i++) {
		std::cout << "cache level: " << m.get_cpu()[0].caches[i].level << ", cache size: " << m.get_cpu()[0].caches[i].memory << "\n";
	}

	// memory
	std::cout << "memory capacity: " << m.get_memory().size << "\n";
	return 0;
}
