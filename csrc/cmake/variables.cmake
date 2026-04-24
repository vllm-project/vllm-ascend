# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(COMMON_NAME common_${PKG_NAME})
set(OPHOST_NAME ophost_${PKG_NAME})
set(OPSTATIC_NAME cann_${PKG_NAME}_static)
set(OPAPI_NAME opapi_${PKG_NAME})
set(OPGRAPH_NAME opgraph_${PKG_NAME})
set(GRAPH_PLUGIN_NAME graph_plugin_${PKG_NAME})
set(ONNX_PLUGIN_NAME op_${PKG_NAME}_onnx_plugin)
if(NOT CANN_3RD_LIB_PATH)
  set(CANN_3RD_LIB_PATH ${PROJECT_SOURCE_DIR}/third_party)
endif()
if(NOT CANN_3RD_PKG_PATH)
  set(CANN_3RD_PKG_PATH ${PROJECT_SOURCE_DIR}/third_party/pkg)
endif()
# interface, 用于收集aclnn/aclnn_inner/aclnn_exclude的def文件
add_library(${OPHOST_NAME}_opdef_aclnn_obj INTERFACE)
add_library(${OPHOST_NAME}_opdef_aclnn_inner_obj INTERFACE)
add_library(${OPHOST_NAME}_opdef_aclnn_exclude_obj INTERFACE)
add_library(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE)
# interface, 用于收集ops proto头文件
add_library(${GRAPH_PLUGIN_NAME}_proto_headers INTERFACE)

# global variables
set(COMPILED_OPS CACHE STRING "Compiled Ops" FORCE)
set(COMPILED_OP_DIRS CACHE STRING "Compiled Ops Dirs" FORCE)

# src path
get_filename_component(OPS_TRANSFORMER_CMAKE_DIR           "${OPS_TRANSFORMER_DIR}/cmake"                               REALPATH)
get_filename_component(OPS_TRANSFORMER_COMMON_INC          "${OPS_TRANSFORMER_DIR}/common/include"                      REALPATH)
get_filename_component(OPS_TRANSFORMER_COMMON_INC_COMMON   "${OPS_TRANSFORMER_COMMON_INC}/common"                       REALPATH)
get_filename_component(OPS_TRANSFORMER_COMMON_INC_EXTERNAL "${OPS_TRANSFORMER_COMMON_INC}/external"                     REALPATH)
get_filename_component(OPS_TRANSFORMER_COMMON_INC_HEADERS  "${OPS_TRANSFORMER_COMMON_INC_EXTERNAL}/aclnn_kernels"       REALPATH)
get_filename_component(OPS_KERNEL_BINARY_SCRIPT     "${OPS_TRANSFORMER_DIR}/scripts/kernel/binary_script"       REALPATH)
get_filename_component(OPS_KERNEL_BINARY_CONFIG     "${OPS_TRANSFORMER_DIR}/scripts/kernel/binary_config"       REALPATH)

# python
if(NOT DEFINED ASCEND_PYTHON_EXECUTABLE)
  set(ASCEND_PYTHON_EXECUTABLE python3 CACHE STRING "")
endif()

set(BIN_STATIC_INSTALL_DIR          packages/vendors/${VENDOR_NAME}_transformer/op_impl/ai_core/tbe/static)
if (ENABLE_BUILT_IN)
  set(ACLNN_INC_INSTALL_DIR           ops_transformer/built-in/op_impl/ai_core/tbe/op_api/include/aclnnop)
  set(ACLNN_INC_LEVEL2_INSTALL_DIR    ops_transformer/built-in/op_impl/ai_core/tbe/op_api/include/aclnnop/level2)
  set(ACLNN_LIB_INSTALL_DIR           ops_transformer/built-in/op_impl/ai_core/tbe/op_api/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
  set(OPS_INFO_INSTALL_DIR            ops_transformer/built-in/op_impl/ai_core/tbe/config)
  set(IMPL_INSTALL_DIR                ops_transformer/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc)
  set(IMPL_DYNAMIC_INSTALL_DIR        ops_transformer/built-in/op_impl/ai_core/tbe/impl/ops_transformer/dynamic)
  set(BIN_KERNEL_INSTALL_DIR          ops_transformer/built-in/op_impl/ai_core/tbe/kernel)
  set(BIN_KERNEL_CONFIG_INSTALL_DIR   ops_transformer/built-in/op_impl/ai_core/tbe/kernel/config)
  set(BIN_STATIC_INSTALL_DIR          ops_transformer/built-in/op_impl/ai_core/tbe/static)
  set(OPHOST_INC_INSTALL_PATH         ops_transformer/built-in/op_impl/ai_core/tbe/op_host/include)
  set(OPHOST_LIB_INSTALL_PATH         ops_transformer/built-in/op_impl/ai_core/tbe/op_host/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
  set(OPTILING_LIB_INSTALL_DIR        ${OPHOST_LIB_INSTALL_PATH})
  set(OPGRAPH_INC_INSTALL_DIR         ops_transformer/built-in/op_graph/inc)
  set(OPGRAPH_LIB_INSTALL_DIR         ops_transformer/built-in/op_graph/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
  set(ONNX_PLUGIN_LIB_INSTALL_DIR     ops_transformer/built-in/framework/onnx)
  set(COMMON_INC_INSTALL_DIR          ops_transformer/include)
  set(COMMON_LIB_INSTALL_DIR          ops_transformer/lib)
  set(VERSION_INFO_INSTALL_DIR        ops_transformer)
  set(IMPL_INSTALL_DIR                ops_transformer/built-in/op_impl/ai_core/tbe/impl/ops_transformer)
endif()

if (ENABLE_TEST)
  set(UTEST_FRAMEWORK_OLD FALSE CACHE BOOL "UTEST_FRAMEWORK_OLD")
  set(UTEST_FRAMEWORK_NEW FALSE CACHE BOOL "UTEST_FRAMEWORK_NEW")
endif()

# util path
set(ASCEND_TENSOR_COMPILER_PATH ${ASCEND_DIR}/compiler)
set(ASCEND_CCEC_COMPILER_PATH ${ASCEND_TENSOR_COMPILER_PATH}/ccec_compiler/bin)
set(OP_BUILD_TOOL ${ASCEND_DIR}/tools/opbuild/op_build)
set(UT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/tests/ut/framework_normal)

# output path
set(ASCEND_AUTOGEN_PATH     ${CMAKE_BINARY_DIR}/autogen)
set(ASCEND_KERNEL_SRC_DST   ${CMAKE_BINARY_DIR}/tbe/ascendc)
set(ASCEND_KERNEL_CONF_DST  ${CMAKE_BINARY_DIR}/tbe/config)
set(ASCEND_GRAPH_CONF_DST   ${CMAKE_BINARY_DIR}/tbe/graph)
file(MAKE_DIRECTORY ${ASCEND_AUTOGEN_PATH})
file(MAKE_DIRECTORY ${ASCEND_KERNEL_SRC_DST})
file(MAKE_DIRECTORY ${ASCEND_KERNEL_CONF_DST})
file(MAKE_DIRECTORY ${ASCEND_GRAPH_CONF_DST})
set(CUSTOM_COMPILE_OPTIONS "custom_compile_options.ini")
set(CUSTOM_OPC_OPTIONS "custom_opc_options.ini")
execute_process(
  COMMAND rm -rf ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS}
  COMMAND rm -rf ${ASCEND_AUTOGEN_PATH}/${CUSTOM_OPC_OPTIONS}
  COMMAND touch ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS}
  COMMAND touch ${ASCEND_AUTOGEN_PATH}/${CUSTOM_OPC_OPTIONS}
)

# pack path
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/build_out)

set(OPAPI_INCLUDE
  ${C_SEC_INCLUDE}
  ${PLATFORM_INC_DIRS}
  ${METADEF_INCLUDE_DIRS}
  ${NNOPBASE_INCLUDE_DIRS}
  ${NPURUNTIME_INCLUDE_DIRS}
  ${AICPU_INC_DIRS}
  ${OPS_TRANSFORMER_DIR}/
  ${OPS_TRANSFORMER_DIR}/common/include
  ${OPS_TRANSFORMER_DIR}/common/include/external
  ${OPS_TRANSFORMER_DIR}/common/include/common
  ${OPS_TRANSFORMER_DIR}/common/include/static
  ${OPS_TRANSFORMER_DIR}/common/stub/op_api
  $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:${TOP_DIR}/output/${PRODUCT}/aclnnop_resource>

  ${OPS_TRANSFORMER_DIR}/mc2/common/inc
  ${OPS_TRANSFORMER_DIR}/mc2/common/new_mc2_mm
  ${OPS_TRANSFORMER_DIR}/mc2/3rd
  ${OPS_TRANSFORMER_DIR}/mc2
)

if (NOT BUILD_OPEN_PROJECT)
  list(APPEND OPAPI_INCLUDE
    ${TOP_DIR}/asl/ops/cann/ops/built-in/op_proto/runtime
    ${TOP_DIR}/ace/comop/inc/external
    ${TOP_DIR}/ops/ops-nn/matmul/common/op_host/op_api
    ${TOP_DIR}/asl/ops/cann/ops/utils/inc/log/inner
    ${TOP_DIR}/asl/ops/cann/ops/utils/inc/error
    ${TOP_DIR}/ace/comop/inc/external
    ${TOP_DIR}/ace/npuruntime/inc/external
    ${TOP_DIR}/ace/npuruntime/inc/nnopbase
    ${TOP_DIR}/asl/ops/cann/ops/mc2/communication_and_computation
    ${TOP_DIR}/ace/npuruntime/acl/inc/external/acl/error_codes
    ${TOP_DIR}/asl/ops/cann/ops/built-in/op_tiling/runtime
    ${TOP_DIR}/asl/ops/cann/ops/built-in
    ${TOP_DIR}/ops-base/pkg_inc/op_common/op_host
    ${TOP_DIR}/ops-base/pkg_inc
    ${TOP_DIR}/ops-base/include
    ${TOP_DIR}/asl/ops/cann/ops/built-in/op_fallback
  )
else()
  list(APPEND OPAPI_INCLUDE ${OPBASE_INC_DIRS})
endif()

set(OP_TILING_INCLUDE
  ${C_SEC_INCLUDE}
  ${PLATFORM_INC_DIRS}
  ${METADEF_INCLUDE_DIRS}
  ${TILINGAPI_INC_DIRS}
  ${NPURUNTIME_INCLUDE_DIRS}
  ${OPBASE_INC_DIRS}
  ${NNOPBASE_INCLUDE_DIRS}
  ${AICPU_INC_DIRS}
  ${OPS_TRANSFORMER_DIR}
  ${JSON_INCLUDE_DIR}
  ${OPS_TRANSFORMER_DIR}/common/include
  ${OPS_TRANSFORMER_DIR}/common/include/
  ${OPS_TRANSFORMER_DIR}/common/stub/op_tiling
  
  ${OPS_TRANSFORMER_DIR}/mc2/common
  ${OPS_TRANSFORMER_DIR}/mc2/common/inc
  ${OPS_TRANSFORMER_DIR}/mc2/common/new_mc2_mm
  ${OPS_TRANSFORMER_DIR}/mc2/3rd
  ${OPS_TRANSFORMER_DIR}/mc2
  ${NNOPBASE_INCLUDE_DIRS}
  ${AICPU_INC_DIRS}
)

if (NOT BUILD_OPEN_PROJECT)
  list(APPEND OP_TILING_INCLUDE
    ${TOP_DIR}/abl/msprof/inc
    ${METADEF_INC_DIR}/../common/util
    ${TOP_DIR}/asl/ops/cann/ops/utils/inc
    ${TOP_DIR}/ace/comop/inc
    ${TOP_DIR}/ace/comop/hccl/open_source/inc
    ${TOP_DIR}/asl/ops/cann/ops/built-in/op_tiling/cube
    ${TOP_DIR}/ace/npuruntime/inc/external
    ${TOP_DIR}/asl/ops/cann/ops/built-in/op_tiling
    ${TOP_DIR}/asl/ops/cann/ops/built-in/op_api/inc
    ${TOP_DIR}/asl/ops/cann/ops/built-in/op_proto/runtime
    ${TOP_DIR}/asl/ops/cann/ops/common/inc
    ${TOP_DIR}/asl/ops/cann/ops/ops-nn/inner
    ${TOP_DIR}/asl/ops/cann/ops/matmul
    ${TOP_DIR}/ace/npuruntime/acl/inc/external/acl/error_codes
    ${TOP_DIR}/asl/ops/cann/ops/mc2/common/inc
    ${TOP_DIR}/asl/ops/cann/ops/built-in/op_tiling/runtime
    ${TOP_DIR}/asl/ops/cann/ops/built-in

    ${TOP_DIR}/asl/ops/cann/ops/mc2/communication_and_computation
    ${TOP_DIR}/ops-base/pkg_inc/op_common/op_host
    ${TOP_DIR}/ops-base/pkg_inc
    ${TOP_DIR}/ops-base/include
    ${TOP_DIR}/asl/ops/cann/ops/built-in/op_fallback

    ${TOP_DIR}/asl/ops/cann/ops/built-in/op_proto/runtime
    ${TOP_DIR}/asl/ops/cann/ops/built-in/op_tiling/device/error
    ${METADEF_DIR}
    ${RUNTIME_INC_DIR}/runtime/platform/inc
    ${METADEF_DIR}/inc/external/ge
  )
endif()

set(OP_PROTO_INCLUDE
  ${C_SEC_INCLUDE}
  ${PLATFORM_INC_DIRS}
  ${METADEF_INCLUDE_DIRS}
  ${OPBASE_INC_DIRS}
  ${NPURUNTIME_INCLUDE_DIRS}
  ${OPS_TRANSFORMER_DIR}/common/include/
  ${OPS_TRANSFORMER_DIR}
  ${OPS_TRANSFORMER_DIR}/mc2/common

  ${OPS_TRANSFORMER_DIR}/common/include

  ${OPS_TRANSFORMER_DIR}/mc2/common/inc
  ${OPS_TRANSFORMER_DIR}/mc2/common/new_mc2_mm
  ${OPS_TRANSFORMER_DIR}/mc2/3rd
  ${OPS_TRANSFORMER_DIR}/mc2
)

if (NOT BUILD_OPEN_PROJECT)
  list(APPEND OP_PROTO_INCLUDE
    ${TOP_DIR}/abl/msprof/inc
    ${METADEF_INC_DIR}/../common/util
    ${TOP_DIR}/ace/comop/inc
    ${TOP_DIR}/ace/comop/hccl/open_source/inc
    ${TOP_DIR}/asl/ops/cann/ops/utils/inc
    ${TOP_DIR}/asl/ops/cann/ops/built-in/op_tiling/cube
    ${TOP_DIR}/asl/ops/cann/ops/utils/inc/log/inner
    ${TOP_DIR}/asl/ops/cann/ops/built-in/op_tiling
    ${TOP_DIR}/ace/npuruntime/inc/external
    ${TOP_DIR}/asl/ops/cann/ops/common/inc
    ${TOP_DIR}/asl/ops/cann/ops/built-in/op_proto/runtime
    ${TOP_DIR}/tmp/host-prefix/src/host-build/atc/opcompiler/ascendc_compiler/api/kernel_tiling
    ${TOP_DIR}/asl/ops/cann/ops/ops-nn/inner
    ${TOP_DIR}/asl/ops/cann/ops/matmul
    ${TOP_DIR}/ace/npuruntime/acl/inc/external/acl/error_codes
    ${TOP_DIR}/asl/ops/cann/ops/built-in/op_tiling/runtime
    ${TOP_DIR}/asl/ops/cann/ops/built-in
    ${TOP_DIR}/ops-base/pkg_inc/op_common/op_host
    ${TOP_DIR}/ops-base/pkg_inc
    ${TOP_DIR}/ops-base/include
    ${TOP_DIR}/asl/ops/cann/ops/built-in/op_fallback
  )
endif()

if (NOT BUILD_OPEN_PROJECT)
  list(APPEND AICPU_INCLUDE
    ${TOP_DIR}/asl/ops/cann/ops/built-in/aicpu/impl
    ${TOP_DIR}/asl/ops/cann/ops/built-in/aicpu/impl/utils
    ${TOP_DIR}/asl/ops/cann/ops/built-in/aicpu/impl/kernels/host/runtime/utils
    ${TOP_DIR}/asl/ops/cann/ops/built-in/aicpu/impl/kernels/normalized/random
    ${TOP_DIR}/asl/ops/cann/ops/built-in/aicpu/context/inc
    ${TOP_DIR}/asl/ops/cann/ops/built-in/aicpu/context/common/*.h
    ${TOP_DIR}/ace/comop/inc
    ${TOP_DIR}/ops-base/include
    ${TOP_DIR}/ops-base/pkg_inc
    ${TOP_DIR}/asl/ops/cann/ops/built-in/op_fallback
  )
endif()

set(AICPU_INCLUDE
  ${OPBASE_INC_DIRS}
  ${AICPU_INC_DIRS}
  ${C_SEC_INCLUDE}
  ${NNOPBASE_INCLUDE_DIRS}
  ${HCCL_EXTERNAL_INCLUDE}
  ${OPS_TRANSFORMER_DIR}/common/inc/common
  ${METADEF_INCLUDE_DIRS}
)

set(ONNX_PLUGIN_COMMON_INCLUDE
  ${OPS_TRANSFORMER_DIR}
  ${OPS_TRANSFORMER_DIR}/common/include/framework
  ${OPS_TRANSFORMER_DIR}/common/include/op_graph
  ${OPS_TRANSFORMER_DIR}/common/stub/inc/framework
)

set(AICPU_DEFINITIONS
  -O2
  -std=c++17
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
  -fno-common
  -fPIC
)

if(EXISTS ${TOP_DIR}/build/product/onetrack/sys_version/sys_version.conf)
    execute_process(COMMAND grep -Po "^\\d+\\.\\d+" ${TOP_DIR}/build/product/onetrack/sys_version/sys_version.conf
        OUTPUT_VARIABLE SYS_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else()
    execute_process(COMMAND grep -Po "(?<=Version=)[0-9]+\.[0-9]+" ${OPS_TRANSFORMER_DIR}/version.info
        OUTPUT_VARIABLE SYS_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

include_directories(${PROJECT_SOURCE_DIR}/utils/inc)

set(OPS_CATEGORY_LIST
  "attention"
  "moe"
  "mc2"
)