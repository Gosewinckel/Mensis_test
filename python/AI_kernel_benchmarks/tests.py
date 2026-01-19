import unittest
import torch
from AI_kernel_benchmarks import AI_kernels as ai

class TestKernelBenchmarks(unittest.TestCase):

    def testLinearGemm(self):
        TOPS = ai.linear_layer_GEMM_TOPS(dtype=torch.float16)
        print(f"small TOPS: {TOPS.tops}")
        self.assertTrue(TOPS.tops > 0)

    def testLinStress(self):
        TOPS = ai.user_model_custom_Gemm(2048, 12288, 49152, dtype=torch.float16)
        print(f"big TOPS: {TOPS.tops}")
        self.assertTrue(TOPS.tops > 0)

    def testKernelBehaviour(self):
        perf2 = ai.linear_layer_GEMM_TOPS(dtype=torch.float16)
        perf1 = ai.user_model_custom_Gemm(2048, 12288, 49152, dtype=torch.float16)
        diagnostics = ai.check_kernel_behaviour(perf1, perf2)

        print(f"time: {diagnostics.time} \n")
        print(f"tops: {diagnostics.tops} \n")
        print(f"efficiency: {diagnostics.efficiency} \n")
        print(f"oom: {diagnostics.oom} \n")
        print(f"paging: {diagnostics.paging} \n")
        print(f"fallback_kernel: {diagnostics.fallback_kernel} \n")

        self.assertTrue(diagnostics.tops > 0)

    def testAttentionScore(self):
        diagnostics = ai.attention_score_Gemm(dtype=torch.float16)
        print(f"time: {diagnostics.time} \n")
        print(f"tops: {diagnostics.tops} \n")

        self.assertTrue(diagnostics.tops > 0)

    def testCustomAttention(self):
        diags = ai.custom_attention_Gemm(1, 2048, 96, 128, dtype=torch.float16)

        print("CUSTOM ATTENTION QK")
        print(f"time: {diags.time}")
        print(f"tops: {diags.tops}")

    def testCustomSoftmax(self):
        output = ai.custom_softmax(1, 96, 2048, dtype=torch.float16)
        print(f"softmax time: {output.time}")
        print(f"softmax bandwidth: {output.bandwidth}")
        self.assertTrue(output.bandwidth > 0)

    def testCustomAttention(self):
        output = ai.custom_attention(1, 2048, 96, 128, dtype=torch.float16)
        print(f"attention tops: {output.tops}")
        print(f"attention bandwidth: {output.bandwidth}")

    def testCustomLayernorm(self):
        output = ai.custom_layernorm(8, 2048, 4096, dtype=torch.float16)
        print(f"layernorm bandwidth: {output.data_throughput}")
        print(f"layernorm tops: {output.tops}")

    def testCustomGelu(self):
        output = ai.custom_gelu_activation(8, 2048, 4096, dtype=torch.float16)
        print(f"gelu bandwidth: {output.data_throughput}")
        print(f"gelu tops: {output.tops}")


if __name__ == "__main__":
    unittest.main()
