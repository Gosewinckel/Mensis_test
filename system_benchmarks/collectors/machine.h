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
#include <vector>

class machine {
	public:
		// Struct definitions for hardware components
		struct CPU {
			std::string model;		// Model of CPU (brand, number)
			int core_count;			// Number of cores
			float clocks;			// Clock speed
			bool AVX_support;		
			struct cache {
				int level;			// 1,2,3
				int memory;		// KB
			};
			std::vector<cache> caches;
		};
	
		struct GPU {
			std::string model;
			int streaming_multiprocessors;
			int memory_capacity;
			int memory_bandwidth;
			std::string driver;
		};

		struct Memory {
			int size;		// GB	
			int speed;		// MT/s
		};

		struct Storage {
			std::string type;
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

		// set as static (only one machine per program)
		static machine& getMachine();

		// getters to access computer information
		int get_cpu_count() {return cpu_count;}
		const std::vector<CPU>& get_cpu() const {return cpu;}
		const std::vector<GPU>& get_gpu() const {return gpu;}
		const std::vector<Memory>& get_memory() const {return memory;}
		const std::vector<Storage>& get_storage() const {return storage;}
		Network get_network() {return network;}
		OS get_os() {return os;}
		tech_stack get_software() {return software;}
		int get_power_limit() {return power_limit;}

	private:
		machine();

		int cpu_count;				//number of CPU's in machine
		std::vector<CPU> cpu;		// array of CPU's in computer
		int gpu_count;				//number of GPU's in machine
		std::vector<GPU> gpu;		// array of GPU's in computer
		std::vector<Memory> memory;			//RAM
		std::vector<Storage> storage;	//list of storage devices
		Network network;
		OS os;
		tech_stack software;
		int power_limit;

		void set_cpu();
		void set_gpu_count();
		void set_gpu();
		void set_memory();
		void set_storage();
		void set_network();
		void set_os();
		void set_tech();
		void set_power_limit();
};
