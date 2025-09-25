#include <iostream>
#include "collectors/machine.h"

int main() {
	std::cout << "running\n";
	machine m;
	std::cout << m.get_os().system << "\n";
	std::cout << m.get_cpu()[0].core_count << "\n";
	std::cout << m.get_cpu()[0].AVX_support << "\n";
	std::cout << m.get_cpu()[0].model << "\n";
	return 0;
}
