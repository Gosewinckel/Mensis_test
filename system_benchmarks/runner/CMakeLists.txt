add_library(runner runner.cpp runner.h)

target_link_libraries(runner
	PUBLIC
		collectors
		microbenchmarks
		third_party
)

target_include_directories(runner PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

