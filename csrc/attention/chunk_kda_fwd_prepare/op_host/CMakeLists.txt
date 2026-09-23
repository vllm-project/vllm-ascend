# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the BSD 3-Clause License (the "License").
# -----------------------------------------------------------------------------------------------------------

# Both the operator scan and dependency expansion can visit this directory.
include_guard(GLOBAL)

set(CATLASS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/catlass/include")
get_filename_component(CATLASS_INCLUDE_DIR_ABS ${CATLASS_INCLUDE_DIR} ABSOLUTE)
set(COMMON_KERNEL_UTILS_DIR "${CMAKE_SOURCE_DIR}/moe/common")
get_filename_component(COMMON_KERNEL_UTILS_DIR_ABS ${COMMON_KERNEL_UTILS_DIR} ABSOLUTE)

add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_kda_fwd_prepare_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME ChunkKdaFwdPrepare
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -I${CATLASS_INCLUDE_DIR_ABS}
            -I${COMMON_KERNEL_UTILS_DIR_ABS}
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE chunk_kda_fwd_prepare ACLNNTYPE aclnn_exclude)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CATLASS_INCLUDE_DIR_ABS}
    )
endif()

if (ENABLE_TEST)
    add_subdirectory(tests)
endif()
