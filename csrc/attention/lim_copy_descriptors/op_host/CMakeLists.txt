# -----------------------------------------------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        lim_copy_descriptors_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME LimCopyDescriptors
    OPTIONS -Wno-deprecated-declarations
)
if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE lim_copy_descriptors ACLNNTYPE aclnn_exclude)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
