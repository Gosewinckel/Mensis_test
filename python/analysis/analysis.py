from python.python_runner import runner as  run
from pathlib import Path
import subprocess
import torch
import json


def get_model_parameters():
    # Define model object
    model = run.Model_params()

    # Get user input on model parameters
    print("Input model parameters one at a time. These values will be used to benchmark\nyour model on this computer.")
    
    # batch size
    while True:
        try:
            model.B = int(input("batch size: "))
            break
        except ValueError:
            print("invalid entry, must be of type int")


    # Sequence length
    while True:
        try:
            model.S = int(input("Sequence length: "))
            break
        except ValueError:
            print("invalid entry, must be of type int")

    # d_model
    while True:
        try:
            model.d_model = int(input("Model width: "))
            break
        except ValueError:
            print("invalid entry, must be of type int")

    # H - number of attention heads
    while True:
        try:
            model.H = int(input("Number of attention heads: "))
            break
        except ValueError:
            print("invalid entry, must be of type int")
            
    # d_head - attention head dimension
    model.d_head = int(model.d_model/model.H)

    # d_ff - Feed forward (MLP) hidden size
    while True:
        try:
            model.d_ff = int(input("feed forward (MLP) size: "))
            break
        except ValueError:
            print("invalid entry, must be of type int")
    
    # compute dtype
    while True:
        dtypes = ["float32", "float16", "int8", "uint8"]
        print(f"dtype must be one of the following: {dtypes}")
        model.dtype = input("data type used for computation (int the form float16, bfloat16...): ")
        if not isinstance(model.dtype, str):
            print("must be a str")
        elif model.dtype not in dtypes:
            print(f"must be one of available dtypes listed above.")
        else:
            break

    if model.dtype == "float32":
        model.dtype = torch.float32
    elif model.dtype == "float16":
        model.dtype = torch.float16
    elif model.dtype == "int8":
        model.dtype = torch.int8
    elif model.dtype == "uint8":
        model.dtype = torch.uint8

    while True:
        try:
            model.n_layers = int(input("Number of transformer blocks: "))
            break
        except TypeError:
            print("invalid entry, must be of type int")

    return model


def run_runners():
    # Get model parameters
    model = get_model_parameters()

    # run C++ microbenchmarks/collectors
    subprocess.run(
        ["./build/main_exe"],
        capture_output = False,
        text = True,
        check =  True,
    )

    # Run pytorch kernel benchmarks
    current_dir = Path(__file__).resolve().parent
    kernel_file = current_dir.parent.parent / "raw_data" / "kernel_data.json"
    run.run_kernel_benchmarks(model, kernel_file) 
    return model

# Create a final output file with all relevant results 
def analyse_results(model, sys_file, kernel_file, out_file):
    # output hardware information
    fsys = open(sys_file)
    sys = json.load(fsys)

    # kernel benchmarks
    fkernel = open(kernel_file)
    kernel = json.load(fkernel)
    kernel = kernel["AI_kernels"]

    results = {
        "system": {},
        "microbenchmarks": {},
        "kernel_performance": {},
        "transformer_estimate": {}
    }
    # add system information and microbenchmarks results to json file
    results["system"] = sys["collectors"]
    results["microbenchmarks"] = sys["microbenchmarks"]

    # kernel benchmark results
    # find peak gpu throughput for eff
    peak_16f = 0
    peak_32f = 0
    peak_8i = 0
    peak_bw = 0
    for i in sys["microbenchmarks"]["GPUs"]:
        if i["TOPS"]["16f"] > peak_16f:
            peak_16f = i["TOPS"]["16f"]
        if i["TOPS"]["32F"] > peak_32f:
            peak_32f = i["TOPS"]["32F"]
        if i["TOPS"]["8i"] > peak_8i:
            peak_8i = i["TOPS"]["8i"]
        if i["bandwidth"] > peak_bw:
            peak_bw = i["bandwidth"]

    peak_comp = 0
    if model.dtype == torch.float16:
        peak_comp = peak_32f
    if model.dtype == torch.float16:
        peak_comp = peak_16f
    if model.dtype == torch.int8:
        peak_comp = peak_8i


    results["kernel_performance"] = {
        "QKV_projection": {
            "flops": kernel["layer_gemm"]["flops"],
            "bytes": kernel["layer_gemm"]["bytes_moved"],
            "time(s)": kernel["layer_gemm"]["time"],
            "tops": kernel["layer_gemm"]["tops"],
            "bandwidth(GB/s)": kernel["layer_gemm"]["bytes_moved"]/kernel["layer_gemm"]["time"]/1e9,
            "compute_efficiency": kernel["layer_gemm"]["tops"]/peak_comp,
            "memory_efficiency": (kernel["layer_gemm"]["bytes_moved"]/kernel["layer_gemm"]["time"])/peak_bw/1e9
        },
        "attention_output_projection": {
            "flops": kernel["attention_output_projection"]["flops"],
            "bytes": kernel["attention_output_projection"]["bytes_moved"],
            "time(s)": kernel["attention_output_projection"]["time"],
            "tops": kernel["attention_output_projection"]["tops"],
            "bandwidth(GB/s)": kernel["attention_output_projection"]["bytes_moved"]/kernel["attention_output_projection"]["time"]/1e9,
            "compute_efficiency": kernel["attention_output_projection"]["tops"]/peak_comp,
            "memory_efficiency": (kernel["attention_output_projection"]["bytes_moved"]/kernel["attention_output_projection"]["time"])/peak_bw/1e9
        },
        "fnn_gemm": {
            "flops": kernel["ffn_gemm"]["flops"],
            "bytes": kernel["ffn_gemm"]["bytes_moved"],
            "time(s)": kernel["ffn_gemm"]["time"],
            "tops": kernel["ffn_gemm"]["tops"],
            "bandwidth(GB/s)": kernel["ffn_gemm"]["bytes_moved"]/kernel["ffn_gemm"]["time"]/1e9,
            "compute_efficiency": kernel["ffn_gemm"]["tops"]/peak_comp,
            "memory_efficiency": (kernel["ffn_gemm"]["bytes_moved"]/kernel["ffn_gemm"]["time"])/peak_bw/1e9
        },
        "attention_gemm": {
            "flops": kernel["attention_gemm"]["flops"],
            "bytes": kernel["attention_gemm"]["bytes_moved"],
            "time(s)": kernel["attention_gemm"]["time"],
            "tops": kernel["attention_gemm"]["tops"],
            "bandwidth(GB/s)": kernel["attention_gemm"]["bytes_moved"]/kernel["attention_gemm"]["time"]/1e9,
            "compute_efficiency": kernel["attention_gemm"]["tops"]/peak_comp,
            "memory_efficiency": (kernel["attention_gemm"]["bytes_moved"]/kernel["attention_gemm"]["time"])/peak_bw/1e9
        },
        "softmax": {
            "flops": kernel["softmax"]["flops"],
            "bytes": kernel["softmax"]["bytes_moved"],
            "time(s)": kernel["softmax"]["time"],
            "tops": kernel["softmax"]["tops"],
            "bandwidth(GB/s)": kernel["softmax"]["bytes_moved"]/kernel["softmax"]["time"]/1e9,
            "compute_efficiency": kernel["softmax"]["tops"]/peak_comp,
            "memory_efficiency": (kernel["softmax"]["bytes_moved"]/kernel["softmax"]["time"])/peak_bw/1e9
        },
        "attention": {
            "flops": kernel["attention"]["flops"],
            "bytes": kernel["attention"]["bytes_moved"],
            "time(s)": kernel["attention"]["time"],
            "tops": kernel["attention"]["tops"],
            "bandwidth(GB/s)": kernel["attention"]["bytes_moved"]/kernel["attention"]["time"]/1e9,
            "compute_efficiency": kernel["attention"]["tops"]/peak_comp,
            "memory_efficiency": (kernel["attention"]["bytes_moved"]/kernel["attention"]["time"])/peak_bw/1e9
        },
        "layernorm": {
            "flops": kernel["layernorm"]["flops"],
            "bytes": kernel["layernorm"]["bytes_moved"],
            "time(s)": kernel["layernorm"]["time"],
            "tops": kernel["layernorm"]["tops"],
            "bandwidth(GB/s)": kernel["layernorm"]["bytes_moved"]/kernel["layernorm"]["time"]/1e9,
            "compute_efficiency": kernel["layernorm"]["tops"]/peak_comp,
            "memory_efficiency": (kernel["layernorm"]["bytes_moved"]/kernel["layernorm"]["time"])/peak_bw/1e9
        },
        "activation(GELU)": {
            "flops": kernel["activation"]["flops"],
            "bytes": kernel["activation"]["bytes_moved"],
            "time(s)": kernel["activation"]["time"],
            "tops": kernel["activation"]["tops"],
            "bandwidth(GB/s)": kernel["activation"]["bytes_moved"]/kernel["activation"]["time"]/1e9,
            "compute_efficiency": kernel["activation"]["tops"]/peak_comp,
            "memory_efficiency": (kernel["activation"]["bytes_moved"]/kernel["activation"]["time"])/peak_bw/1e9
        }
    }

    # transformer estimate
    QKV_projection_time = kernel["layer_gemm"]["time"]
    attention_output_projection_time = kernel["attention_output_projection"]["time"]
    attention_time = kernel["attention"]["time"]
    ffn_time = 2 * kernel["ffn_gemm"]["time"]   # some models have a multiple of 3
    layernorm_time = 2 * kernel["layernorm"]["time"]
    activation_time = 3 * kernel["activation"]["time"]

    transformer_layer_time = QKV_projection_time + attention_output_projection_time + attention_time + ffn_time + layernorm_time + activation_time
    
    # output transformer estimates
    results["transformer_estimate"] = {
            "model_time(s)": transformer_layer_time * model.n_layers,
            "component_percentages": {
                "QKV_projection": (QKV_projection_time/transformer_layer_time)*100,
                "Attention_output_projection": (attention_output_projection_time/transformer_layer_time)*100,
                "Attention": (attention_time/transformer_layer_time)*100,
                "FFN": (ffn_time/transformer_layer_time)*100,
                "Layernorm": (layernorm_time/transformer_layer_time)*100,
                "Elementwise": (activation_time/transformer_layer_time)*100
            }
    }

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
