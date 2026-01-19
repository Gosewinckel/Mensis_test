add_library(collectors machine.cpp machine.h)

target_include_directories(collectors PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
