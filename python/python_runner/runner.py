from AI_kernel_benchmarks import AI_kernels as ai
import torch
import json

# Defines a function that will run the set of AI kernel benchmarks
# created in pytorch and writes the results to a provided file
# Inputs will be the user defined model parameters and the output
# file the results will be sent to


# Class that defines the users model parameters that are to be benchmarked
class Model_params:
    def __init__(self):
        self.B = 0              # Batch size
        self.S = 0              # Sequence length
        self.d_model = 0        # model width
        self.H = 0              # number of attention heads
        self.d_head = 0         # attention head dimension
        self.d_ff = 0           # Feed-forward (MLP)
        self.dtype = torch.float16

    # Define setter functions
    def set_B(self, b):
        self.B = b
    def set_S(self, s):
        self.S = s
    def set_d_model(self, d_model):
        self.d_model = d_model
    def set_H(self, h):
        self.H = h
    def set_d_head(self, d_head):
        self.d_head = d_head
    def set_d_ff(self, d_ff):
        self.d_ff = d_ff
    def set_dtype(self, dtype):
        self.dtype = dtype

    # Define getters
    def get_B(self):
        return self.B
    def get_S(self):
        return self.S
    def get_d_model(self):
        return self.d_model
    def get_H(self):
        return self.H
    def get_d_head(self):
        return self.d_head
    def get_d_ff(self):
        return self.d_ff
    def get_dtype(self):
        return self.dtype

# run_kernel_benchmarks -- runs the AI kernel benchmarks and sends the results to a JSON file
#
# Params : - model: Model_params object
#          - output_file: .json file where results will be written to
#
# Returns: None
#
# Usage: used to get set of benchmark results to be used for analysis
def run_kernel_benchmarks(model, output_file):

    # Run all benchmarks and put them in the ai_kernel_benchmarks section
    # of the json file
    
    # Run std midsized Gemm to compare against
    vanilla_Gemm = ai.linear_layer_GEMM_TOPS(model.get_dtype())

    # Custom layer Gemm
    custom_layer_Gemm = ai.user_model_custom_Gemm(
            model.get_B() * model.get_S(), 
            model.get_d_model(), 
            model.get_d_ff(),
            model.get_dtype()

    )
    
    # std attention Gemm
    vanilla_attention_gemm = ai.attention_score_Gemm(model.get_dtype())

    # Custom attention Gemm
    custom_attention_Gemm = ai.custom_attention_Gemm(
            model.get_B(),
            model.get_S(),
            model.get_H(),
            model.get_d_head(),
            model.get_dtype()
    )

    # Gemm kernel behaviour
    gemm_behaviour = ai.check_kernel_behaviour(custom_layer_Gemm, vanilla_Gemm)

    # Softmax
    softmax = ai.custom_softmax(
            model.get_B(),
            model.get_H(),
            model.get_S(),
            model.get_dtype()
    )

    # Full attention run
    attention = ai.custom_attention(
            model.get_B(),
            model.get_H(),
            model.get_S(),
            model.get_d_head(),
            model.get_dtype()
    )

    # Layernorm
    layernorm = ai.custom_layernorm(
            model.get_B(),
            model.get_S(),
            model.get_d_model(),
            model.get_dtype()
    )

    # Gelu activation
    activation = ai.custom_gelu_activation(
            model.get_B(), 
            model.get_S(),
            model.get_d_model(),
            model.get_dtype()
    )

    # output kernel benchmark results to json format
    output = {
        "AI_kernels": {
            "layer_gemm": {
                "time": gemm_behaviour.time,
                "tops": gemm_behaviour.tops,
                "efficiency": gemm_behaviour.efficiency,
                "oom": gemm_behaviour.oom,
                "paging": gemm_behaviour.paging,
                "fallback_kernel": gemm_behaviour.fallback_kernel
            },
            "attention_gemm": {
                "time": custom_attention_Gemm.time,
                "tops": custom_attention_Gemm.tops
            },
            "softmax": {
                "time": softmax.time,
                "bandwidth": softmax.bandwidth
            },
            "attention": {
                "time": attention.time,
                "flops": attention.flops,
                "tops": attention.tops,
                "bytes_moved": attention.bytes_moved,
                "bandwidth": attention.bandwidth
            },
            "layernorm": {
                "time": layernorm.time,
                "bytes_moved": layernorm.bytes_moved,
                "data_throughput": layernorm.data_throughput,
                "flops": layernorm.flops,
                "tops": layernorm.tops
            },
            "activation": {
                "time": activation.time,
                "flops": activation.flops,
                "tops": activation.tops,
                "bytes_moved": activation.bytes_moved,
                "data_throughput": activation.data_throughput
            }
        }
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
