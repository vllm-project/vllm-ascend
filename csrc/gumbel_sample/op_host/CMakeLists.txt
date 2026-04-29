add_ops_compile_options(
    OP_NAME GumbelSample
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
)

target_sources(op_host_aclnn PRIVATE
    gumbel_sample_def.cpp
)

target_sources(optiling PRIVATE
    gumbel_sample_tiling.cpp
)

target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
    gumbel_sample_infershape.cpp
)
