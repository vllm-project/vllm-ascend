add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        dcut_recurrent_gated_delta_rule_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME DcutRecurrentGatedDeltaRule
    OPTIONS
        --cce-auto-sync=on
        -Wno-deprecated-declarations
        -Werror
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE dcut_recurrent_gated_delta_rule ACLNNTYPE aclnn_exclude)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../vendor/op_host
    )
    # add_modules_sources does not recurse into the legacy arch35 directory.
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        arch35/dcut_recurrent_gated_delta_rule_tiling_arch35.cpp
    )
endif()
