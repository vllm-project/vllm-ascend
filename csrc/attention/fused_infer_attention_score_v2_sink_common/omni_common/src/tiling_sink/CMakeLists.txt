# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16)

project(tiling_sink_project)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/intf.cmake)

set(ATTENTION_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../../attention)
set(src_files
  ${ATTENTION_SRC}/common/op_host/fia_tiling_info.cpp
  ${ATTENTION_SRC}/common/op_host/arch32/fia_tiling_nonquant_mla.cpp
  ${ATTENTION_SRC}/common/op_host/arch32/fia_tiling_nonquant.cpp
  ${ATTENTION_SRC}/common/op_host/arch32/fia_tiling_empty_tensor.cpp
  ${ATTENTION_SRC}/common/op_host/fia_tiling_shape.cpp
  ${ATTENTION_SRC}/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.cpp
  ${ATTENTION_SRC}/fused_infer_attention_score/op_host/arch32/fused_infer_attention_score_tiling_check.cpp
  ${ATTENTION_SRC}/fused_infer_attention_score/op_host/arch32/fused_infer_attention_score_tiling_check_single_para.cpp
  ${ATTENTION_SRC}/fused_infer_attention_score/op_host/arch32/fused_infer_attention_score_tiling_check_existence.cpp
  ${ATTENTION_SRC}/fused_infer_attention_score/op_host/arch32/fused_infer_attention_score_tiling_check_feature.cpp
  ${ATTENTION_SRC}/fused_infer_attention_score/op_host/arch32/fused_infer_attention_score_tiling_check_consistency.cpp
  ${ATTENTION_SRC}/fused_infer_attention_score/op_host/arch32/fused_infer_attention_score_tiling_info_parser.cpp
  ${ATTENTION_SRC}/fused_infer_attention_score/op_host/arch32/fused_infer_attention_score_tiling_v3.cpp
  ${ATTENTION_SRC}/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_v2.cpp
  ${ATTENTION_SRC}/prompt_flash_attention/op_host/prompt_flash_attention_tiling.cpp
  ${ATTENTION_SRC}/prompt_flash_attention/op_host/prompt_flash_attention_tiling_v2.cpp
  ${ATTENTION_SRC}/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp
  ${ATTENTION_SRC}/incre_flash_attention/op_host/incre_flash_attention_tiling_v2.cpp
  ${ATTENTION_SRC}/incre_flash_attention/op_host/incre_flash_attention_tiling_check.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/device_op_impl_registry.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tiling_sink_kernel.cpp
)

foreach(f ${src_files})
  if(NOT EXISTS ${f})
    message(WARNING "File not found: ${f}")
    return()
  endif()
endforeach()

add_library(opmaster SHARED
  ${src_files}
)

if("${PRODUCT}" STREQUAL "ascend310p")
  set_target_properties(opmaster PROPERTIES PREFIX "Ascend310P-v${SYS_VERSION}-lib")
else()
  set_target_properties(opmaster PROPERTIES PREFIX "Ascend-v${SYS_VERSION}-lib")
endif()

if (BUILD_OPEN_PROJECT)
  add_custom_command(
      TARGET opmaster
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
              $<TARGET_FILE:opmaster>
              ${CMAKE_BINARY_DIR}/Ascend310P-v${SYS_VERSION}-libopmaster.so
  )
endif ()

target_include_directories(opmaster PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${ATTENTION_SRC}/common/op_host
  ${ATTENTION_SRC}/common/op_kernel
  ${ATTENTION_SRC}/fused_infer_attention_score/op_host
  ${ATTENTION_SRC}/prompt_flash_attention/op_host
  ${ATTENTION_SRC}/incre_flash_attention/op_host
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/tiling_sink
)

if (BUILD_OPEN_PROJECT)
  target_include_directories(opmaster PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/base
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/op_common
    ${ASCEND_CANN_PACKAGE_PATH}/include/aicpu
    ${ASCEND_CANN_PACKAGE_PATH}/include/aicpu_common/context/common
    ${ASCEND_CANN_PACKAGE_PATH}/include/aicpu_common/context/utils
    ${ASCEND_CANN_PACKAGE_PATH}/include/aicpu_common/context/cpu_proto
    ${ASCEND_CANN_PACKAGE_PATH}/include/toolchain
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
  )
else ()
  target_include_directories(opmaster PRIVATE
    ${TOP_DIR}/ops-base/include
    ${TOP_DIR}/ops-base/pkg_inc
    ${TOP_DIR}/ops-base/pkg_inc/op_common
    ${TOP_DIR}/ops-base/include/base
    ${TOP_DIR}/ops-base/include/aicpu_common/context/common
    ${TOP_DIR}/ops-base/include/aicpu_common/context/utils
    ${TOP_DIR}/ops-base/include/aicpu_common/context/cpu_proto
    ${TOP_DIR}/metadef/inc/external
    ${TOP_DIR}/metadef/inc
    ${TOP_DIR}/abl/slog/inc/toolchain  # codespell:ignore abl
    ${TOP_DIR}/metadef/inc/external/base
    ${TOP_DIR}/abl/slog/inc  # codespell:ignore abl
  )
endif ()

if (BUILD_OPEN_PROJECT)
  target_link_directories(opmaster PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64/device/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/devlib/device
  )
endif()

target_link_libraries(opmaster PRIVATE
  $<BUILD_INTERFACE:$<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,intf_pub_aicpu,intf_pub_cxx17>>
  -Wl,--whole-archive
  -Wl,--no-whole-archive
  mmpa
  tiling_api
  exe_meta_device
  platform_static
  -Wl,--no-as-needed
  c_sec
  -ldl
  -Wl,--as-needed
)

if (NOT BUILD_OPEN_PROJECT)
  target_link_libraries(opmaster PRIVATE
    $<BUILD_INTERFACE:slog_headers>
    -Wl,--no-as-needed
    slog
    -Wl,--as-needed
  )
endif ()

target_compile_definitions(opmaster PRIVATE
  _FORTIFY_SOURCE=2
  google=ascend_private
  DEVICE_OP_TILING_LIB
  BUILT_IN_TILING_SINK
)

target_compile_definitions(opmaster PRIVATE
  OPS_UTILS_LOG_SUB_MOD_NAME="OP_TILING"
  OPS_UTILS_LOG_PACKAGE_TYPE=$<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,"[Custom]","">
)

target_compile_options(opmaster PRIVATE
  -O2
  -fstack-protector-all
  -fvisibility-inlines-hidden
  -fvisibility=hidden
  -frename-registers
  -fpeel-loops
  -DEIGEN_NO_DEBUG
  -DEIGEN_MPL2_ONLY
  -DNDEBUG
  -DEIGEN_HAS_CXX11_MATH
  -DEIGEN_OS_GNULINUX
  -DEigen=ascend_Eigen
)

target_link_options(opmaster PRIVATE
  -Wl,-z,relro,-z,now
  $<$<NOT:$<STREQUAL:${RESERVED_SYMBOL},true>>:-s>
  -Wl,-Bsymbolic
)

if (NOT BUILD_OPEN_PROJECT)
  install (
    TARGETS opmaster OPTIONAL
    LIBRARY DESTINATION lib
  )
endif ()
