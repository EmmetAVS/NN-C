file(GLOB TEST_SOURCES "*.c")

foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)

    add_executable(${test_name} ${test_source})

    target_link_libraries(${test_name} PRIVATE nnlib)

    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()