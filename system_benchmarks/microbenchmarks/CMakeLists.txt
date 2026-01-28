add_library(microbenchmarks cpu_microbenchmarks.cpp cpu_microbenchmarks.h disk_microbenchmarks.cpp disk_microbenchmarks.h gpu_microbenchmarks.h gpu_microbenchmarks.cu)

enable_language(CUDA)

target_link_libraries(microbenchmarks PRIVATE ${OpenBLAS_LIB})

target_link_libraries(microbenchmarks PRIVATE OpenMP::OpenMP_CXX)

target_link_libraries(microbenchmarks PRIVATE collectors)

set(CUDAToolkit_ROOT /opt/cuda)
find_package(CUDAToolkit REQUIRED)
target_include_directories(microbenchmarks PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(microbenchmarks PRIVATE CUDA::cudart CUDA::cublas)

target_include_directories(microbenchmarks PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

