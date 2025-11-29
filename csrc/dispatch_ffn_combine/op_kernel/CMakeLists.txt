# set custom compile options
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0 )
endif()

add_ops_compile_options(ALL OPTIONS -DASCENDC_DUMP=0 --cce-auto-sync=on -DL2_CACHE_HINT -DHCCL_COMM)
add_kernels_compile()