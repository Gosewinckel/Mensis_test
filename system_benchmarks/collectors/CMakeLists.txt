add_library(collectors machine.cpp machine.h)

set(CUDAToolkit_ROOT /opt/cuda)
find_package(CUDAToolkit REQUIRED)
target_include_directories(collectors PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(collectors PRIVATE CUDA::cudart CUDA::cublas)

target_include_directories(collectors PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
