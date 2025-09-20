enable_testing()

cmake_minimum_required(VERSION 3.10)
project(Mensis_test LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)



add_subdirectory(tests)
add_subdirectory(collectors)

add_executable(main_exe main.cpp)

target_link_libraries(main_exe PRIVATE collectors)

