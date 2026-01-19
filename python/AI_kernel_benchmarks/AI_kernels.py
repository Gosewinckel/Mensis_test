import torch
from torch import nn
import time

class Gemm_diagnostics:
    def __init__(self):
        self.m = 0          # dimensions of the computation
        self.n = 0
        self.k = 0
        self.batch = 0
        self.size = 0       # size of the computation
        self.time = 0.0     # time taken 
        self.flops = 0.0    # total floating point operations
        self.tops = 0.0     # total throughput per second
        self.peak_allocated_mem = 0.0  
        self.times = []     # set of times for Gemm 

class Gemm_perf:
    def __init__(self):
        self.time = 0.0
        self.tops = 0.0
        self.efficiency = 0.0
        self.oom = False
        self.paging = False
        self.fallback_kernel = False

class Softmax_perf:
    def __init__(self):
        self.time = 0.0
        self.bandwidth = 0.0

class Attention_perf:
    def __init__(self):
        self.time = 0.0
        self.flops = 0
        self.tops = 0.0
        self.bytes_moved = 0
        self.bandwidth = 0.0

class Layernorm_perf:
    def __init__(self):
        self.time = 0.0
        self.bytes_moved = 0
        self.data_throughput = 0.0  # different from bandwidth as data is not necessarily being moved from dram
        self.flops = 0.0
        self.tops = 0.0

class Activation_perf:
    def __init__(self):
        self.time = 0.0
        self.flops = 0
        self. tops = 0.0
        self.bytes_moved = 0
        self.data_throughput = 0.0

# Construct compute environment
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)

device = "cuda"

# Going to benchmark 2 different sizes for each benchmark
# the first will be a mid sized benchmark to measure computational efficiency
# compared to peak compute
# The second is to stress test the machine running larger models


# linear_layer_GEMM_TOPS -- Runs and benchmarks a Gemm to test a mid sized test case 
#           That is representative of a linear layer with the intention of finding 
#           highest performance for this shape of Gemm
#
# Params: None
#
# Returns: Gemm_diagnostics object
#
# Usage: Run a midsized computation to be compared with  the users custom model.
def linear_layer_GEMM_TOPS(dtype):

    # define output
    output = Gemm_diagnostics()

    # Define layer variables
    B = 8           # batch size
    S = 2048        # Sequence length(tokens) 
    d_model = 4096  # model width
    d_out = 11008   # feed foreward network width (MLP expansion)

    output.m = B*S
    output.n = d_out
    output.k = d_model
    output.batch = 1
    output.size = B*S * d_model + d_model*d_out + B*S * d_out

    # define matrices
    X = torch.randn(B*S, d_model, device = device, dtype=dtype)
    W = torch.randn(d_out, d_model, device=device, dtype=dtype)

    # start timing benchmark
    iterations = 100
    peak_mem = 0.0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_host_memory_stats()
    times = []
    avgTime = 0.0

    for _ in range(iterations):
        start = time.perf_counter()
        Y = torch.matmul(X, W.T)
        torch.cuda.synchronize()
        end = time.perf_counter()
        avgTime += end - start
        times.append(end - start)

    temp_mem = torch.cuda.max_memory_allocated()
    if temp_mem > peak_mem:
        peak_mem = temp_mem

    avgTime /=  iterations

    # calulate TOPS
    FLOPs = 2 * (B*S) * d_out * d_model
    TOPS = FLOPs / avgTime / 1e12

    # Generate output
    output.peak_allocated_mem = peak_mem
    output.tops = TOPS
    output.time = avgTime
    output.times = times
    output.flops = FLOPs
    return output

# attention_score_Gemm: runs a Gemm modelling a midsized attention Gemm
#
def attention_score_Gemm(dtype):
    
    output = Gemm_diagnostics()

    B = 8 #8
    S = 128 #128
    H = 12 #12
    d_head = 64 #64

    # will compute multiple Gemms
    Q = torch.randn(B, H, S, d_head, device=device, dtype=dtype)
    K = torch.randn(B, H, S, d_head, device=device, dtype=dtype)

    Q = Q.reshape(B * H, S, d_head)
    K = K.reshape(B * H, S, d_head)

    output.m = S
    output.n = S
    output.k = d_head
    output.batch = B*H

    iterations = 100
    peak_mem = 0.0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_host_memory_stats()
    times = []
    avgTime = 0.0

    start = time.perf_counter()
    for _ in range(iterations):
        Y = torch.matmul(Q, K.transpose(-1, -2))

    torch.cuda.synchronize()
    end = time.perf_counter()

    temp_mem = torch.cuda.max_memory_allocated()
    if temp_mem > peak_mem:
        peak_mem = temp_mem

    avgTime = (end - start) / iterations
    flops = 2 * B * H * S * S * d_head
    tops = flops / avgTime / 1e12

    output.peak_allocated_mem = peak_mem
    output.tops = tops
    output.flops = flops
    output.time = avgTime
    output.times = []

    return output


# user_model_custom_GEMM -- runs a GEMM based on the size of the 
#       users model to stress test the machine for their specific workload
# 
# Params: - M, N, K: matrix side dimensions
#
# Usage: Takes matrix dimensions defined by user to match their model for linear layers.
#       Model parameters get run and compared to base performance
#
# Returns: Gemm_diagnostics object
def user_model_custom_Gemm(M, N, K, dtype):

    #define output
    output = Gemm_diagnostics()

    output.m = M
    output.n = N
    output.k = K
    output.size = M*K + N*K + M*N

    # define matrices and check if memory allocation is succesful
    try:
        X = torch.randn(M, N, device=device, dtype=dtype)
    except RuntimeError:
        return output
    try:
        W = torch.randn(K, N, device=device, dtype=dtype)
    except RuntimeError:
        return output

    # set number of iterations and timing env
    iterations = 100
    peak_mem = 0    # peak memory allocated
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    avgTime = 0.0
    times = []

    # Gemm execution loop
    for _ in range(iterations):
        start = time.perf_counter()
        Y = torch.matmul(X, W.T)
        torch.cuda.synchronize()
        end = time.perf_counter()
        avgTime += end - start
        times.append(end - start)

    #complete timing and datataking
    temp_peak = torch.cuda.max_memory_allocated()
    if temp_peak > peak_mem:
        peak_mem = temp_peak

    # calculate avg time and TOPS
    avgTime /= iterations
    flops = 2 * M * N * K
    tops = flops / avgTime / 1e12

    output.time = avgTime
    output.times = times
    output.tops = tops
    output.peak_allocated_mem = peak_mem
    output.flops = flops
    
    return output


# custom_attention_Gemm -- Runs custom attention Gemm based on users model params
def custom_attention_Gemm(B, S, H, head, dtype):
    
    output = Gemm_diagnostics()

    output.m = S
    output.n = S
    output.k = head
    output.batch = B*H

    try:
        X = torch.randn(B, H, S, head, device=device, dtype=dtype)
    except:
        return output
    try:
        W = torch.randn(B, H, S, head, device=device, dtype=dtype)
    except:
        return output

    X = X.reshape(B*H, S, head)
    W = W.reshape(B*H, S, head)

    iterations = 100
    peak_mem = 0.0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_host_memory_stats()
    times = []
    avgTime = 0.0

    start = time.perf_counter()
    for _ in range(iterations):
        Y = torch.matmul(X, W.transpose(-1, -2))

    torch.cuda.synchronize()
    end = time.perf_counter()

    temp_mem = torch.cuda.max_memory_allocated()
    if temp_mem > peak_mem:
        peak_mem = temp_mem

    avgTime = (end - start) / iterations
    flops = 2 * B * H * S * S * head
    tops = flops / avgTime / 1e12

    output.peak_allocated_mem = peak_mem
    output.tops = tops
    output.flops = flops
    output.time = avgTime
    output.times = []

    return output



# check_kernel_behavior -- compares the diagnostics from the mid sized Gemm
#       to the custom Gemm run based on the users requirements
#
# Params: - perf1: Gemm_diagnostics taken from custom Gemm
#         - perf2: Gemm_diagnostics from standard mid sized comparison Gemm
#
# Usage: Used to analyse performance on different parts of the model based on 
#           users model parameter inputs. Measures performance of different 
#           parts of the model by comparing to a mid sized standard test case
# 
# Returns: Gemm_perf object
#
def check_kernel_behaviour(perf1, perf2):
    
    # create output
    results = Gemm_perf()

    # Check for out of memory and return if true
    if perf1.tops == 0:
        results.oom = True
        return results

    # check workspace pressure 
    # if there is a peak in memory allocation compared to computation size
    # this implies workspace failure
    mem_ratio1 = perf1.size / perf1.peak_allocated_mem
    mem_ratio2 = perf2.size / perf2.peak_allocated_mem
    if mem_ratio1 > 1.5 * mem_ratio2:
        results.oom = True

    # Check for paging
    # If computation time is highly varied it is likely that paging is occuring
    # as computation could be getting done on the CPU
    if max(perf1.times) > 8*min(perf1.times):
        results.paging = True

    # Run float32 Gemm top test if tensor cores were used
    X = torch.randn(perf1.m, perf1.k, device=device, dtype=torch.float32)
    W = torch.randn(perf1.n, perf1.k, device=device, dtype=torch.float32)

    avgTime = 0.0
    torch.cuda.synchronize()
    start = time.perf_counter()

    for _ in range(100):
        torch.matmul(X, W.T)


    torch.cuda.synchronize()
    end = time.perf_counter()
    avgTime = (end - start) / 10

    # Compare FP16 time to FP32 time, FP16 time should be significantly lower
    if perf1.time > avgTime/2:
        results.fallback_kernel = True

    # Calculate overall efficiency of computation compared to practical limit
    efficiency = (perf1.flops/perf2.flops) / (perf1.time/perf2.time)
    results.efficiency = efficiency

    results.time = perf1.time
    results.tops = perf1.tops

    return results

    
# custom_softmax -- runs a softmax benchmark based on users model parameters
#
# Params: - B: batch size
#         - H: number of heads
#         - S: number of tokens
#
# Returns: Softmax_perf object, used to compare against bandwidth
#
# Usage: Run with users model parameters to compare against ideal GPU bandwidth
def custom_softmax(B, H, S, dtype):
    
    output = Softmax_perf()

    # Create randomised embedding matrix
    embeddings = torch.randn(B*H, S, S, device=device, dtype=dtype)

    # warm-up
    for _ in range(5):
        Y = torch.softmax(embeddings, dim=-1)
    torch.cuda.synchronize()

    iterations = 100

    # start benchmark runs
    start = time.perf_counter()
    for _ in range(iterations):
        Y = torch.softmax(embeddings, dim=-1)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avgTime = (end - start) / iterations

    bytes_moved = 2 * embeddings.numel() * embeddings.element_size()
    bandwidth = bytes_moved / avgTime / 1e9     #GB/s

    output.time = avgTime
    output.bandwidth = bandwidth

    return output


# custom_attention -- runs a through a full iteration of attention and returns 
#       an attention_perf object
#
# Params: - B: batch size
#         - H: number of heads
#         - S: number of tokens
#         - d_head  
#
# Returns: - attention_perf object
def custom_attention(B, H, S, d_head, dtype):
    
    output = Attention_perf()

    # Create tensors to be used in attention
    Q = torch.randn(B, H, S, d_head, device=device, dtype=dtype)
    K = torch.randn(B, H, S, d_head, device=device, dtype=dtype)
    V = torch.randn(B, H, S, d_head, device=device, dtype=dtype)

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    # Warmup runs
    for _ in range(10):
        torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    # Setup timing
    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()
    
    end = time.perf_counter()
    avgTime = (end - start) / iterations

    # TOPS is calculated differently as there are multiple matmuls being
    # computed. Softmax operations are ignored as it is negligable
    flops = 4 * B * H * S * S * d_head
    tops = flops / avgTime / 1e12

    # Bandwidth must also be calculated to gain a fuller understanding 
    # of computation time
    bytes_moved = 4 * B * H * S * d_head * 2
    bandwidth = bytes_moved / avgTime / 1e9

    output.time = avgTime
    output.flops = flops
    output.tops = tops
    output.bytes_moved = bytes_moved
    output.bandwidth = bandwidth

    return output

# Layernorm -- benchmark for layernorm function acting on the MLP
#
# Params: - B: batch size
#         - S: tokens
#         - d_model: enbedding size
#
# Returns: Layernorm_perf object
# 
# Usage: Use with users model parameters to measure performance of layernorm
#       given their model size on the current machine.
def custom_layernorm(B, S, d_model, dtype):
    
    output = Layernorm_perf()

    X = torch.randn(B, S, d_model, device=device, dtype=dtype)
    gamma = torch.ones(d_model, device=device, dtype=dtype)
    beta = torch.zeros(d_model, device=device, dtype=dtype)

    # Warmup runs
    for _ in range(10):
        torch.nn.functional.layer_norm(X, (d_model,), gamma, beta)
    torch.cuda.synchronize()

    # Setup timed benchmark
    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        torch.nn.functional.layer_norm(X, (d_model,), gamma, beta)
        torch.cuda.synchronize()

    end = time.perf_counter()
    
    avgTime = (end - start) / iterations

    # Construct output values
    flops = 7 * B * S * d_model
    bytes_moved = 10 * B * S * d_model * X.element_size()

    tops = flops / avgTime / 1e12
    bandwidth = bytes_moved / avgTime / 1e9

    output.time = avgTime
    output.bytes_moved = bytes_moved
    output.data_throughput = bandwidth
    output.flops = flops
    output.tops = tops

    return output

# custom_gelu_activation -- benchmarks elementwise residual + activation based on 
#       users model parameters
#
# Params: - B: batch size
#         - S: num tokens
#         - d_model: embeddingt size
#
# Returns: Activation_perf object
#
# Usage: Used to benchmark performance of GELU on users machine
def custom_gelu_activation(B, S, d_model, dtype):

    output = Activation_perf()

    # Define tensors
    X = torch.randn(B, S, d_model, device=device, dtype=dtype)
    residual = torch.randn(B, S, d_model, device=device, dtype=dtype)
    bias = torch.randn(d_model, device=device, dtype=dtype)

    # Warmup runs
    for _ in range(10):
        Y = torch.nn.functional.gelu(X + residual + bias)
    torch.cuda.synchronize()

    # Setup timing loop
    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        Y = torch.nn.functional.gelu(X + residual + bias)
        torch.cuda.synchronize()
    end = time.perf_counter()

    avgTime = (end - start) / iterations

    flops = 15 * B * S * d_model
    tops = flops / avgTime / 1e12

    bytes_moved = 2 * B * S * d_model * X.element_size()
    data_throughput = bytes_moved / avgTime / 1e9

    output.time = avgTime
    output.flops = flops
    output.tops = tops
    output.bytes_moved = bytes_moved
    output.data_throughput = data_throughput

    return output

    
