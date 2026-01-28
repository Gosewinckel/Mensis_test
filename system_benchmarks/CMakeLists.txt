enable_testing()

find_path(OpenBLAS_INCLUDE_DIR cblas.h)
find_library(OpenBLAS_LIB openblas)

if (NOT OpenBLAS_LIB)
	message(FATAL_ERROR "OpenBLAS not found")
endif()

find_package(OpenMP REQUIRED)

find_package(CUDAToolkit REQUIRED)

#if (NOT TARGET BLAS::BLAS)
#	add_library(BLAS::BLAS INTERFACE IMPORTED)
#	set_target_properties(BLAS::BLAS PROPERTIES INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARIES}")
#endif()

add_subdirectory(tests)
add_subdirectory(collectors)
add_subdirectory(microbenchmarks)
add_subdirectory(runner)

add_library(system_benchmarks INTERFACE)

target_link_libraries(system_benchmarks 
	INTERFACE 
		collectors
		microbenchmarks
		runner
)
