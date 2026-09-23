add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        moe_gating_top_k_with_map_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME MoeGatingTopKWithMap
    OPTIONS
        --cce-auto-sync=on
        -Wno-deprecated-declarations
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE moe_gating_top_k_with_map ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()

# Static adapters resolve ../ascendc/common relative to binary/<soc>/src/<op>.
# Mirror the packaged common header there without embedding source paths in options.
if(ASCEND_BINARY_OUT_DIR)
    set(_topk_common_dir "${ASCEND_BINARY_OUT_DIR}/ascend950/src/ascendc/common")
    file(MAKE_DIRECTORY "${_topk_common_dir}")
    configure_file("${PROJECT_SOURCE_DIR}/common/include/cann_compat.h"
                   "${_topk_common_dir}/cann_compat.h" COPYONLY)
endif()
