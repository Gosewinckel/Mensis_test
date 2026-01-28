add_test([=[runnerTest.collectors]=]  /home/Mensis/Source-code/github.com/Gosewinckel/Mensis_test/build/system_benchmarks/tests/test_runner [==[--gtest_filter=runnerTest.collectors]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[runnerTest.collectors]=]  PROPERTIES DEF_SOURCE_LINE /home/Mensis/Source-code/github.com/Gosewinckel/Mensis_test/system_benchmarks/tests/test_runner.cpp:6 WORKING_DIRECTORY /home/Mensis/Source-code/github.com/Gosewinckel/Mensis_test/build/system_benchmarks/tests SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_runner_TESTS runnerTest.collectors)
