
cmake_minimum_required(VERSION 3.10)
project(Mensis_test LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(system_benchmarks)

# Executables
add_executable(main_exe main.cpp)

add_library(third_party INTERFACE)
target_include_directories(third_party INTERFACE ${CMAKE_SOURCE_DIR}/third_party)
target_include_directories(main_exe PUBLIC ${CMAKE_SOURCE_DIR}/third_party)

target_link_libraries(main_exe PRIVATE system_benchmarks)
