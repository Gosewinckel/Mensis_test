/**************************************************** 
 * machine -- class that defines the software and 
 *		hardware for a single computing node
 *
 * Purpose -- gather and colate information 
 *		about the software and hardware on this machine
 *
 * Use -- To be run automatically by the coordinator 
 *
 ****************************************************/ 
#pragma once
#include <string>

class machine {
	// Struct definitions for hardware components
	struct CPU {
		std::string model;
		int core_count;
		int clocks;
		bool AVX_support;
		int cache_sizes;
	};
	
	struct GPU {
		std::string model;
		int streaming_multiprocessors;
		int memory_capacity;
		int memory_bandwidth;
		std::string driver;
	};

	struct Memory {
		int size;
		int speed;
	};

	struct Storage {
		std::string type;
		int read_throughput;
		int write_throughput;
	};

	struct Network {
		std::string NIC_model;
		int link_speed;
	};

	//Struct definitions for software stack
	struct OS {
		std::string system;
		std::string version;
		std::string kernel;
	};

	struct tech_stack {
		std::string python_version;
		std::string cpp_version;
		std::string pytorch_version;
		std::string CUDA_version;
	};

	public:
		machine();
		~machine();

		// getters to access computer information
		CPU *get_cpu() {return cpu;}
		GPU *get_gpu() {return gpu;}
		Memory get_memory() {return memory;}
		Storage get_storage() {return storage;}
		Network get_network() {return network;}
		OS get_os() {return os;}
		tech_stack get_software() {return software;}
		int get_power_limit() {return power_limit;}

	private:
		CPU *cpu = nullptr;		// array of CPU's in computer
		GPU *gpu = nullptr;		// array of GPU's in computer
		Memory memory;			//RAM
		Storage storage;		// Long term storage
		Network network;
		OS os;
		tech_stack software;
		int power_limit;

		void set_cpu();
		void set_gpu();
		void set_memory();
		void set_storage();
		void set_network();
		void set_os();
		void set_tech();
		void set_power_limit();
};
