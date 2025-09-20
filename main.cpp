#include <iostream>
#include "collectors/machine.h"

int main() {
	std::cout << "running\n";
	machine m;
	std::cout << m.get_os().system << "\n";
	return 0;
}
