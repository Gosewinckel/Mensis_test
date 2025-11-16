include(FetchContent)

FetchContent_Declare(
	googletest
	URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
FetchContent_MakeAvailable(googletest)
enable_testing()

add_executable(test_machine test_machine.cpp)
add_executable(test_CPU test_CPU.cpp)
add_executable(test_disk test_disk.cpp)
add_executable(test_GPU test_GPU.cpp)

target_link_libraries(test_machine PRIVATE gtest_main collectors)
target_link_libraries(test_CPU PRIVATE gtest_main microbenchmarks)
target_link_libraries(test_disk PRIVATE gtest_main microbenchmarks)
target_link_libraries(test_GPU PRIVATE gtest_main microbenchmarks)

include(GoogleTest)
gtest_discover_tests(test_machine)
gtest_discover_tests(test_CPU)
gtest_discover_tests(test_disk)
gtest_discover_tests(test_GPU)
