# Mensis_test
A Linux-based benchmarking tool for analysing a computer's suitability for AI inference/training workloads. It evaluates system performance for machine learning workloads by collecting hardware details, running microbenchmarks (e.g., memory bandwidth via STREAM, compute via GEMM with OpenMP/CUDA parallelism), and testing AI-specific kernels (e.g., Transformer layers in PyTorch across precisions like FP16/FP32). This provides a realistic view of how custom Transformer-based models might perform on the given hardware, helping engineers and researchers identify bottlenecks like compute vs. bandwidth constraints.

Originating as a research project into AI hardware limitations, this V1 has evolved into a functional prototype. It serves as a foundation for exploring workload optimizations, with insights drawn from real benchmarks (see Analysis section below). Future versions will expand capabilities, including multi-GPU support and cross-platform compatibility.

## Features
Mensis_test is a comprehensive set of tests and benchmarks that produces hardware performance metrics as structured JSON output. Benchmarks and tests  include:
- Data collection: Automatically gathers system information (e.g. CPU threads, GPU streaming multiprocessors, RAM size) using tools such as lscpu, dmidecode as well as the linux filesystem to provide a detailed view of the machine’s hardware.
- Performance benchmarks:
  - System microbenchmarks: Runs a set of benchmarks (e.g. STREAM triad from memory bandwidth, GEMM for compute throughput) to determine peak performance metrics for each hardware component.
  - AI-kernels: Records performance on a set of AI kernels (e.g. GEMM, attention, softmax) implemented in PyTorch that are designed to model the user's transformer model based on user provided input parameters.
- Output and analysis:
  - Generates a single parsable JSON file split into four sections (system, microbenchmarks, kernel_performance and transformer_estimate) allowing for simple human or programmed analysis.
  - Allows the user to input their own model parameters that will determine the dimensions of the AI-kernel benchmarks, this allows the program to estimate the performance of the users specific model.

These features allow the user to diagnose AI hardware limitations such as bandwidth-bound vs compute-bound on their specific model and hardware setup.

## Requirements
Mensis_test requires the user to be on a linux OS and have an NVIDIA GPU, preferably Turing architecture or newer for Tensor core support on FP16 computation. AI-kernels are also not designed for multi-GPU systems, this will be added in future versions.

Dependencies that will be installed in installer:
- Python 3
- PyTorch
- C++
- CUDA
- OpenMP
- lscpu
- dmidecode

## Instalation/setup
Clone the repo

```
Git clone https://github.com/Gosewinckel/Mensis_test
cd  Mensis_test
```

Run installer script with root privileges

```
./install_Mensis_test.sh
```

Create build directory

```
mkdir build
```

Compile C++ code

```
cmake -S . -B build
cmake –build build --j
```

## Usage
Run Mensis_test

```
python main.py
```

Once the program is run the user will be prompted for the parameters that determine the shape of their model. The parameters are:
1. Batch size(int): The number of independent sequences processed in parallel in one forward pass
2. Sequence length(int): The number of tokens in a sequence
3. Model width(int): The dimensionality of each token’s embedding vector
4. Number of attention heads(int): how many independent attention subspaces the model uses per layer
5. Feed forward (MLP) size(int): Multi-layered perceptron dimension
6. Data type used for computation(dtype): the datatype That the model and sequence are in
7. Number of transformer blocks(int): the number of layers in a model

Two models are provided below for those who don’t have a particular model but would like to benchmark their hardware on realistic model sizes. The batch sizes vary in the real world but I have tried to give typical single GPU cases. Sequence length is based on typical maximum attention.

**GPT-3 (175B parameters):**
Batch size inference = 1\
Batch size training = 8\
Sequence length = 2048\
Model width = 12288\
Number of attention heads = 96\
Feed forward (MLP) size = 49152\
Dtype = float16\
Number of transformer blocks = 96\

**LLaMA-7B:**
Batch size inference = 1\
Batch size training = 8\
Sequence length = 2048\
Model width = 4096\
Number of attention heads = 32\
Feed forward (MLP) size = 11008\
Dtype = float16\
Number of transformer blocks = 32\

## Results
All benchmark results are recorded in a JSON file that will be located in the results folder. The JSON output is separated into four sections, System for the machines hardware information, microbenchmarks for the hardware performance results, AI-kernels for the PyTorch transformer kernel results and transformer_estimate for an estimate of model time and the proportion of total computation each component of the transformer took. An example of the output file can be found in the example.json file.

## Analysing LLaMA-7B Model Performance
The following is an analysis of the performance of the LLaMA-7B model (parameters defined above) on my personal computer. The machine has an intel i5 CPU and an NVIDIA 2060 and is in no way designed to perform AI workloads effectively, therefore it was appropriate to measure performance on a smaller model.  The following analysis offers a look into raw hardware performance, AI-kernel performance and potential bottlenecks and optimisation targets within the model.
Hardware Overview
My PC is a single CPU, single GPU machine with 16GB of RAM.

CPU: Intel(R) Core(TM) i5-9400F CPU @ 2.90GHz
- 6 cores
- 1MB of L2 cache memory, 36MB of L3 cache memory

GPU: NVIDIA GeForce RTX 2060
- Has tensor cores
- 30 streaming multiprocessors
- 6GB VRAM

With an i5 with 6 cores this machine is clearly limited by its CPU. With only 6 cores to run parallel processes orchestration and setup will have limited performance. Fortunately most computation will not be occurring on the CPU so the model’s compute time once set up on the GPU should not be severely limited by the dated CPU. The 2060 GPU is not an ideal device for running AI workloads. Given it is originally designed for gaming this is no surprise but there are some positives to this device. The GPU is on Turing architecture and has dedicated tensor cores, speeding up computation for 16 bit floating point calculations, it is however limited by only having 6GB of global VRAM meaning fitting an entire model on the device is unlikely. There is 16GB of RAM available on the machine so if the model is smaller than 16GB it can be run without accessing SSD storage however only certain parts of the model will be present on the device at any one time.

### Microbenchmark Results
CPU:
|  Bandwidth (GB/s) | Single thread throughput (GFLOP/s)  |  Multi thread throughput (GFLOP/s) | Synchronisation overhead (microseconds)  | Task dispatch throughput (tasks / microsecond)  | Thread wake latency (threads/ microsecond)  |
|---|---|---|---|---|---|
| 9.534868528934567  | 50.40638732910156  | 213.20870971679688  | 0.572838318  | 42.122264590668046  | 0.608  |

Compute throughput is computed by calculating GEMMs of varying sizes. Multithreading achieves a 4.3x speedup over single thread performance however is limited by having only 6 cores to work on. Memory bandwidth is calculated by running a STREAM triad. This processor is limited to 9.5GB/s which for ;larger models will become a significant performance bottleneck as it will lead to the GPU being starved of work. Particularly on this machine which has only 6GB of VRAM and data will need to be moved around per training run having only 9.5GB/s bandwidth will become more of a problem the larger the model gets.

GPU:
| Bandwidth (GB/s) | float_32 TOPS  | float_16 TOPS   | int_8  TOPS  |
|---|---|---|---|
| 297.1730003797946  | 6.624858379364014  | 26.971389770507812  | 60.296382904052734  |

VRAM bandwidth is calculated using a triad, similarly to CPU memory bandwidth. Trillion operations per second (TOPS) is calculated by running optimised GEMM calculations on multiple matrix sizes for each datatype. Given that VRAM is only 6GB, 300GB/s bandwidth is sufficient, and when compared with the CPU bandwidth this will not be the point of slowdown when getting computation tasks to the device. Importantly for AI workloads, 16 bit floating point computation for GEMM workloads gives a 4x speedup over 32 bit floating point. This is largely a result of the computation happening on tensor cores instead of CUDA cores. Often it is worthwhile to accept the lower definition datatype in order to achieve significantly faster compute speeds.

### AI-kernels
The AI-kernels are implemented in PyTorch and are designed to simulate a model’s components so that they can be observed and measured individually for a better understanding of model performance. There are multiple GEMM workloads throughout a transformer model but they will often have varying performance on different hardware as a result of the differing shapes of the matrices. There are also some components to a model which cannot be broken down further in a realistic manner, for example the attention block could be broken down into components but there are a lot of small optimisations that can be achieved when running the entire layer, meaning the broken down components would not offer a realistic measurement for the performance of attention. The following table provides each benchmark and its associated compute and performance metrics as performed on a LLaMA-7B size model using float_16 parameters.

| Kernel  | FLOP  | bytes  | time(s)  |  Bandwidth (GB/s) | TOPS  | Compute efficiency  | Memory efficiency  |
|---|---|---|---|---|---|---|---|
| QKV_projection |  206158430208 | 167772160  | 0.008168783221626653  | 20.538206908934406  | 25.237348649698603  | 0.9357081286665726  | 0.06911195459441491  |
| attention_output_projection  | 68719476736  | 67108864 | 0.0027935559302568434  | 24.02273864401559  | 24.599284371471963  | 0.9120510504197431  | 0.08083755460056574  |
| fnn_gemm  | 184683593728  | 152043520  | 0.007176992382155732  | 21.184851801992735  | 25.732728125388803  | 0.9540749788698898  | 0.07128794262910144  |
| attention_gemm  | 34359738368  | 603979776  | 0.0024328876205254348  | 248.25633987547582  | 14.123027335138179  | 0.5236299447417119  | 0.8353933215944853  |
| softmax  | 1073741824  | 536870912  | 0.002071438640123233 | 259.1778011672413  | 0.5183556023344825  | 0.01921872053108975  | 0.8721445112308505 |
| attention  | 68719476736  | 67108864  | 0.004598042749566957  |  14.59509353329094 | 14.945375778089923 | 0.5541196024845603  | 0.049113121025927804  |
| layernorm  | 58720256  | 33554432  | 0.00018155350931920111  | 184.8184159360189  | 0.32343222788803316  | 0.011991678242761298  | 0.6219219636367245  |
| activation(GELU)  | 125829120  | 33554432  | 0.0004575649998150766  | 73.33260195504664  | 0.27499725733142494  | 0.010195887556084483  | 0.24676737745799832  |

Compute and memory efficiency are measurements of how close the kernel is to reaching the peak for each of these metrics, this helps determine if the kernel is compute or memory bound.

Below is a table of the components of a transformer and what percentage of compute time is associated with each kernel.
| Kernel  | QKV_projection  | Attention_output_projection  |  Attention | FFN  | Layernorm  | Elementwise  |
|---|---|---|---|---|---|---|
| Percentage  | 25.809604060012653  | 8.826354001969957  | 14.527703771493098  | 45.352000830371026  | 1.147251448374537  | 4.337085887778735  |

From these results there are a few significant conclusions that can be drawn about the systems performance on this model. The most important is that the multi-layered perceptron (labeled FFN) makes up 45% of the total computation in the model. This makes sense given that it is the biggest computational component but easily singles it out as being an ideal target for optimisation. Also because the FFN computation is completely compute bound having a 94% compute efficiency, it is being limited by the computational power of the machine’s GPU. This combined with the fact that the next 2 most intensive computations are also GEMM’s (attention is not entirely a GEMM computation but is still compute bound) the speed of running this model would improve significantly with the improved computation speed of a more advanced GPU. The attention GEMM is the only GEMM computation that does not approach max practical TOPS, this is because attention has tall-skinny and wide-short matrices resulting in fewer arithmetic operations per byte loaded. This limits it from reaching maximum computational performance. Despite this attention is still predominantly compute bound and would overall benefit mostly from faster computational speeds. While operations such as layernorm and GELU would benefit from improved memory bandwidth, they make up a negligible proportion of compute time compared to matrix multiplication.

## Future versions/TODO
This is version 1 of Mensis_test, designed to help people designing AI programs on personal computers like mine. Future implementations will improve upon this foundation so that it is able to assess more complex machines and assist a broader array of researchers and engineers. Features that will be included in future versions include:
- Multi-GPU support
- Windows and Mac support
- AMD, intel and Apple silicon GPU support
- Distributed systems (cluster and clouds)
- Improved personalisation for transformer models, e.g. custom layer configs, size check to see if the model will fit on VRAM

## License
This project is released under the MIT License.
