import unittest
from python_runner import runner

class TestRunner(unittest.TestCase):
    
    def testRunner(self):
        input = runner.Model_params()
        runner.run_kernel_benchmarks(input, "test.json")
