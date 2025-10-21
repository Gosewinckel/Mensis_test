#include <iostream>
#include "collectors/machine.h"

int main() {
	std::cout << "running\n";
	machine& m = machine::getMachine();
	std::cout << m.get_os().system << "\n";
	std::cout << m.get_cpu()[0].core_count << "\n";
	std::cout << m.get_cpu()[0].AVX_support << "\n";
	std::cout << m.get_cpu()[0].model << "\n";
	std::cout << m.get_cpu()[0].clocks << "\n";
	for(int i = 0; i < m.get_cpu()[0].caches.size(); i++) {
		std::cout << "cache level: " << m.get_cpu()[0].caches[i].level << ", cache size: " << m.get_cpu()[0].caches[i].memory << "\n";
	}
	std::cout << "memory info:" << m.get_memory()[1].size << m.get_memory()[1].speed << "\n";
	std::cout << "Storage type:" << m.get_storage()[0].type << "\n";

	// memory
	return 0;
}
