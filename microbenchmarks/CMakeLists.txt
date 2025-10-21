add_library(microbenchmarks cpu_microbenchmarks.cpp cpu_microbenchmarks.h)

target_include_directories(microbenchmarks PRIVATE ${OpenBLAS_INCLUDE_DIR})
target_link_libraries(microbenchmarks PRIVATE ${OpenBLAS_LIB})

target_link_libraries(microbenchmarks PRIVATE OpenMP::OpenMP_CXX)

target_link_libraries(microbenchmarks PRIVATE collectors)

target_include_directories(microbenchmarks PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

