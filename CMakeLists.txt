
cmake_minimum_required(VERSION 3.10)
project(Mensis_test LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(system_benchmarks)

# Executables
add_executable(main_exe main.cpp)

