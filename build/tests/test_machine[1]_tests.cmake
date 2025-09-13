add_test([=[machineTest.defaultTest]=]  /home/Mensis/Source-code/github.com/Gosewinckel/Mensis_test/build/tests/test_machine [==[--gtest_filter=machineTest.defaultTest]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[machineTest.defaultTest]=]  PROPERTIES WORKING_DIRECTORY /home/Mensis/Source-code/github.com/Gosewinckel/Mensis_test/build/tests SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_machine_TESTS machineTest.defaultTest)
