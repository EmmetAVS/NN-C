cmake_minimum_required(VERSION 3.14)
project(MyCProject C)

option(DEBUG "Enable debug sanitizer mode" OFF)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(include)

if(MSVC)
  target_compile_options(nnlib PRIVATE /wd2099)
endif()

file(GLOB SRC_FILES src/*.c)
add_library(nnlib STATIC ${SRC_FILES})

target_compile_options(nnlib PRIVATE -Wall)

enable_testing()

add_subdirectory(test)

file(GLOB EXAMPLE_DIRS RELATIVE "${CMAKE_SOURCE_DIR}/examples" "${CMAKE_SOURCE_DIR}/examples/*")

if(UNIX)
    set(EXTRA_LIBS m)
endif()

foreach(example_dir ${EXAMPLE_DIRS})
    if(IS_DIRECTORY "${CMAKE_SOURCE_DIR}/examples/${example_dir}")
        file(GLOB EXAMPLE_SOURCES "${CMAKE_SOURCE_DIR}/examples/${example_dir}/*.c")

        if(EXAMPLE_SOURCES)
            set(target_name "example_${example_dir}")

            add_executable(${target_name} ${EXAMPLE_SOURCES})
            target_include_directories(${target_name} PRIVATE include)
            target_link_libraries(${target_name} PRIVATE nnlib ${EXTRA_LIBS})

            if(DEBUG)
                target_compile_options(${target_name} PRIVATE -g -O0 -fsanitize=address)
                target_link_options(${target_name} PRIVATE -fsanitize=address)
            endif()

            set_target_properties(${target_name} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/examples/${example_dir}"
            )
        endif()
    endif()
endforeach()