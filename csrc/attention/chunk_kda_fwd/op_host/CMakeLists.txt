# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------
include_guard(GLOBAL)

set(CATLASS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/catlass/include")
get_filename_component(CATLASS_INCLUDE_DIR_ABS ${CATLASS_INCLUDE_DIR} ABSOLUTE)
set(COMMON_KERNEL_UTILS_DIR "${CMAKE_SOURCE_DIR}/moe/common")
get_filename_component(COMMON_KERNEL_UTILS_DIR_ABS ${COMMON_KERNEL_UTILS_DIR} ABSOLUTE)

add_op_to_compiled_list()
set(chunk_kda_fwd_depends
    "attention/kda_gate_cumsum;moe/chunk_gated_delta_rule_fwd_h;attention/chunk_kda_fwd_prepare;moe/chunk_fwd_h;attention/chunk_kda_fwd_finalize"
    CACHE STRING "Kernel source dependencies for chunk_kda_fwd" FORCE)
# Dependencies must contribute opdefs, tiling and op_api, not just kernel
# sources. Otherwise a filtered build lacks the V2 operators' runtime config.
set(_kda_fwd_build_all_ops FALSE)
if (NOT DEFINED ASCEND_OP_NAME OR "${ASCEND_OP_NAME}" STREQUAL ""
        OR "${ASCEND_OP_NAME}" STREQUAL "all" OR "${ASCEND_OP_NAME}" STREQUAL "ALL")
    set(_kda_fwd_build_all_ops TRUE)
endif()
foreach(_kda_dep IN LISTS chunk_kda_fwd_depends)
    get_filename_component(_kda_dep_name "${_kda_dep}" NAME)
    if (_kda_fwd_build_all_ops OR _kda_dep_name IN_LIST ASCEND_OP_NAME)
        continue()
    endif()
    set(_kda_dep_dir "${CMAKE_SOURCE_DIR}/${_kda_dep}")
    if (NOT BUILD_OPEN_PROJECT AND
            EXISTS "${TOP_DIR}/asl/ops/cann/ops/built-in/tbe/impl/ascendc/${_kda_dep_name}")
        continue()
    endif()
    # Each dependency guards its registration globally: the moe scan may have
    # already visited it. Do not rely on COMPILED_OPS, which survives reconfigure.
    message(STATUS "[kda-deps] add dependency op ${_kda_dep_name} to this build")
    add_subdirectory(
        ${_kda_dep_dir}/op_host
        ${CMAKE_BINARY_DIR}/chunk_kda_fwd_deps/${_kda_dep_name}
    )
endforeach()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_kda_fwd_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME ChunkKdaFwd
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -I${CATLASS_INCLUDE_DIR_ABS}
            -I${COMMON_KERNEL_UTILS_DIR_ABS}
)
if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME ChunkKdaFwd
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE chunk_kda_fwd ACLNNTYPE aclnn_exclude)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CATLASS_INCLUDE_DIR_ABS}
    )
endif()
