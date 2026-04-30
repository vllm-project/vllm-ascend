add_ops_compile_options(
        OP_NAME CopyAndExpandDflashInputs
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)

target_sources(op_host_aclnn PRIVATE
        copy_and_expand_dflash_inputs_def.cpp
)

target_sources(optiling PRIVATE
        copy_and_expand_dflash_inputs_tiling.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        copy_and_expand_dflash_inputs_infershape.cpp
)