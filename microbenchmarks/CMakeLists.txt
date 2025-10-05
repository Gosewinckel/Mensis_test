add_library(microbenchmarks cpu_microbenchmarks.cpp cpu_microbenchmarks.h)

target_link_libraries(microbenchmarks PRIVATE BLAS::BLAS)

target_include_directories(microbenchmarks PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

