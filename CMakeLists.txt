enable_testing()

cmake_minimum_required(VERSION 3.10)
project(Mensis_test LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_path(OpenBLAS_INCLUDE_DIR cblas.h)
find_library(OpenBLAS_LIB openblas)

if (NOT OpenBLAS_LIB)
	message(FATAL_ERROR "OpenBLAS not found")
endif()

find_package(OpenMP REQUIRED)

#if (NOT TARGET BLAS::BLAS)
#	add_library(BLAS::BLAS INTERFACE IMPORTED)
#	set_target_properties(BLAS::BLAS PROPERTIES INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARIES}")
#endif()

add_subdirectory(tests)
add_subdirectory(collectors)
add_subdirectory(microbenchmarks)

# Executables
add_executable(main_exe main.cpp)


# Link libraries
target_link_libraries(main_exe PRIVATE collectors)
target_link_libraries(main_exe PRIVATE microbenchmarks)
