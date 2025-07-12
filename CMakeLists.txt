cmake_minimum_required(VERSION 3.14)
project(MyCProject C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(include)

file(GLOB SRC_FILES src/*.c)
add_library(nnlib STATIC ${SRC_FILES})

enable_testing()

add_subdirectory(test)
