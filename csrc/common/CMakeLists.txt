# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
    foreach(SUB_DIR ${CURRENT_DIRS})
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
            add_subdirectory(${SUB_DIR})
        endif()
    endforeach()
endif()

if (BUILD_OPEN_PROJECT)
    add_library(ops_transformer_utils_tiling_headers INTERFACE)

    target_include_directories(ops_transformer_utils_tiling_headers INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/slog>>
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef>>
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime>>
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof>>
            $<$<BOOL:OFF${BUILD_OPS_RTY_KERNEL}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/op_common>>
            $<INSTALL_INTERFACE:include/ops_adv/utils>
    )

    target_compile_definitions(ops_transformer_utils_tiling_headers INTERFACE
            OPS_UTILS_LOG_SUB_MOD_NAME="OP_TILING"
            OPS_UTILS_LOG_PACKAGE_TYPE=$<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,"[Custom]","">
    )

    add_library(ops_transformer_utils_proto_headers INTERFACE)

    target_include_directories(ops_transformer_utils_proto_headers INTERFACE
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/slog>>
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef>>
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/aclnn/opdev>>
            $<$<BOOL:OFF${BUILD_OPS_RTY_KERNEL}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/op_common>>
            $<INSTALL_INTERFACE:include/ops_adv/utils>
    )

    target_compile_definitions(ops_transformer_utils_proto_headers INTERFACE
            OPS_UTILS_LOG_SUB_MOD_NAME="OP_PROTO"
            OPS_UTILS_LOG_PACKAGE_TYPE=$<IF:$<BOOL:${BUILD_OPEN_PROJECT}>,"[Custom]","">
    )

    add_library(${COMMON_NAME}_obj OBJECT)

    file(GLOB CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/tiling_base/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

    target_sources(${COMMON_NAME}_obj PRIVATE ${CPP_SOURCES})

    target_include_directories(${COMMON_NAME}_obj
      PRIVATE
      ${C_SEC_INCLUDE}
      ${TILINGAPI_INC_DIRS}
      ${METADEF_INCLUDE_DIRS}
      ${OPS_TRANSFORMER_DIR}/common/include
      ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
      ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
      ${ASCEND_CANN_PACKAGE_PATH}/include/op_common
      ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
      ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/op_common
      ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/runtime
      ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc
      ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/op_common
      ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling
    )

    target_link_libraries(${COMMON_NAME}_obj
      PRIVATE
      $<BUILD_INTERFACE:intf_pub_cxx17>
      tiling_api
      dlog
    )
else()
    add_library(${COMMON_NAME}_obj OBJECT)

    file(GLOB CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/tiling_base/*.cpp")

    target_sources(${COMMON_NAME}_obj PRIVATE ${CPP_SOURCES})

    target_include_directories(${COMMON_NAME}_obj
      PRIVATE
      ${C_SEC_INCLUDE}
      ${TILINGAPI_INC_DIRS}
      ${METADEF_INCLUDE_DIRS}
      ${OPS_TRANSFORMER_DIR}/common/include
      ${OPS_TRANSFORMER_DIR}/common/inc
    )

    target_link_libraries(${COMMON_NAME}_obj
      PRIVATE
      $<BUILD_INTERFACE:intf_pub_cxx17>
      tiling_api
    )
endif()

if (NOT BUILD_OPS_RTY_KERNEL)
    if (BUILD_OPEN_PROJECT)
        ExternalProject_Add(tiling_sink_task
            SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/tiling_sink
            CONFIGURE_COMMAND ${CMAKE_COMMAND}
                -G ${CMAKE_GENERATOR}
                -DBUILD_OPEN_PROJECT=${BUILD_OPEN_PROJECT}
                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                -DASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}
                -DCMAKE_C_COMPILER=${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-gcc
                -DCMAKE_CXX_COMPILER=${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++
                -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                -DSYS_VERSION=${SYS_VERSION}
                -DOPS_ADV_CMAKE_DIR=${OPS_ADV_CMAKE_DIR}
                -DPRODUCT_SIDE=device
                -DENABLE_BUILT_IN=${ENABLE_BUILT_IN}
                <SOURCE_DIR>
            INSTALL_COMMAND ""
            BUILD_ALWAYS TRUE
            BUILD_BYPRODUCTS
        )
        add_subdirectory(src/framework)
    else()
        add_subdirectory(src/tiling_sink)
    endif()
endif()
