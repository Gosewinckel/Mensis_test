enable_testing()

cmake_minimum_required(VERSION 3.10)
project(Mensis_test LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(Mensis_test main.cpp)

add_subdirectory(collectors)
add_subdirectory(tests)
