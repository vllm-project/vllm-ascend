# SPDX-License-Identifier: Apache-2.0
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE engram_hash_ids_def.cpp)
endif()

add_ops_compile_options(
    OP_NAME EngramHashIds
    OPTIONS --cce-auto-sync=off
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE engram_hash_ids ACLNNTYPE aclnn)
endif()
