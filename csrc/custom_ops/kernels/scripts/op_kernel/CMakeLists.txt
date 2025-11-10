# set custom compile options
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0)
endif()

add_ops_compile_options(ALL OPTIONS -DASCENDC_DUMP=0 --cce-auto-sync=off)

add_kernels_compile()