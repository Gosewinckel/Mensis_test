include(FetchContent)

FetchContent_Declare(
	googletest
	URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
FetchContent_MakeAvailable(googletest)
enable_testing()

add_executable(test_machine test_machine.cpp)

target_link_libraries(test_machine PRIVATE gtest_main collectors)

include(GoogleTest)
gtest_discover_tests(test_machine)
