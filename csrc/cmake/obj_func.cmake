# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# useage: add_modules_sources(DIR OPTYPE ACLNNTYPE)
# ACLNNTYPE 支持类型aclnn/aclnn_inner/aclnn_exclude
# OPTYPE 和 ACLNNTYPE 需一一对应

# 用于custom自定算子包host侧obj生成
macro(add_modules_sources)
  set(oneValueArgs OP_API_INDEPENDENT OP_API_DIR)
  set(multiValueArgs OPTYPE ACLNNTYPE)

  cmake_parse_arguments(MODULE "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  # 该段代码作用为兼容op_api新旧目录结构(旧： 嵌套于op_host下； 新： 与op_host同级)
  if (NOT DEFINED MODULE_OP_API_INDEPENDENT)
    set(MODULE_OP_API_INDEPENDENT OFF)
  endif()
    if(MODULE_OP_API_INDEPENDENT)
      # 新结构：op_api与op_host同级，需要指定有效路径
      if (NOT DEFINED MODULE_OP_API_DIR OR NOT EXISTS "${MODULE_OP_API_DIR}")
        message(FATAL_ERROR "OP_API_INDEPENDENT=ON时，必须传递有效的OP_API_DIR路径")
      endif()
      set(OP_API_SRC_DIR "${MODULE_OP_API_DIR}")
    else()
      # 旧结构：op_api嵌套在op_host目录下
      set(OP_API_SRC_DIR "${SOURCE_DIR}/op_api")
  endif()

  # opapi 默认全部编译
  file(GLOB OPAPI_SRCS ${OP_API_SRC_DIR}/*.cpp)
  if (OPAPI_SRCS)
    # aclnn
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_SRCS})
  else()
    if (NOT TARGET ${OPHOST_NAME}_opapi_obj)
      add_opapi_modules()
      add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/opapi_stub.cpp
          COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/opapi_stub.cpp
      )
      target_sources(${OPHOST_NAME}_opapi_obj PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/opapi_stub.cpp
      )
    endif()
  endif()
  file(GLOB OPAPI_HEADERS ${OP_API_SRC_DIR}/aclnn_*.h)
  if (OPAPI_HEADERS)
    target_sources(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE ${OPAPI_HEADERS})
  endif()

  # 是否编译该算子已经由op_add_subdirectory和每个二级目录判断完毕，默认走到这里全编
  file(GLOB OPINFER_SRCS ${SOURCE_DIR}/*_infershape*.cpp)
  add_infer_modules()
  set(PROTO_STUB_FILE ${CMAKE_CURRENT_BINARY_DIR}/proto_stub.cpp)

  if(NOT EXISTS ${PROTO_STUB_FILE})
      file(WRITE ${PROTO_STUB_FILE} "// Auto-generated stub file\n")
  endif()

  # 标记为生成的文件
  set_source_files_properties(
      ${PROTO_STUB_FILE}
      PROPERTIES GENERATED TRUE
  )

  if (OPINFER_SRCS)
      target_sources(${OPHOST_NAME}_infer_obj PRIVATE ${OPINFER_SRCS})
  else()
      target_sources(${OPHOST_NAME}_infer_obj PRIVATE ${PROTO_STUB_FILE})
  endif()

  file(GLOB_RECURSE SUB_OPTILING_SRC ${SOURCE_DIR}/op_tiling/*.cpp)
  file(GLOB OPTILING_SRCS 
      ${SOURCE_DIR}/*fallback*.cpp
      ${SOURCE_DIR}/*_tiling*.cpp
      ${SOURCE_DIR}/op_tiling/arch35/*.cpp
      ${SOURCE_DIR}/../op_graph/fallback_*.cpp
      ${SOURCE_DIR}/../graph_plugin/fallback_*.cpp)
  if (OPTILING_SRCS OR SUB_OPTILING_SRC)
    # tiling
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${OPTILING_SRCS} ${SUB_OPTILING_SRC})
    # target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${SOURCE_DIR}/../../ ${SOURCE_DIR})
  endif()

  file(GLOB AICPU_SRCS ${SOURCE_DIR}/*_aicpu*.cpp)
  if(AICPU_SRCS)
    add_aicpu_kernel_modules()
    target_sources(${OPHOST_NAME}_aicpu_obj PRIVATE ${AICPU_SRCS})
  endif()

  if (MODULE_OPTYPE)
    list(LENGTH MODULE_OPTYPE OpTypeLen)
    list(LENGTH MODULE_ACLNNTYPE AclnnTypeLen)
    if(NOT ${OpTypeLen} EQUAL ${AclnnTypeLen})
      message(FATAL_ERROR "OPTYPE AND ACLNNTYPE Should be One-to-One")
    endif()
    math(EXPR index "${OpTypeLen} - 1")
    foreach(i RANGE ${index})
      list(GET MODULE_OPTYPE ${i} OpType)
      list(GET MODULE_ACLNNTYPE ${i} AclnnType)
      if (${AclnnType} STREQUAL "aclnn" OR ${AclnnType} STREQUAL "aclnn_inner" OR ${AclnnType} STREQUAL "aclnn_exclude")
        file(GLOB OPDEF_SRCS ${SOURCE_DIR}/${OpType}_def*.cpp)

        if (OPDEF_SRCS)
          target_sources(${OPHOST_NAME}_opdef_${AclnnType}_obj INTERFACE ${OPDEF_SRCS})
        endif()
      elseif(${AclnnType} STREQUAL "no_need_aclnn")
        message(STATUS "aicpu or host aicpu no need aclnn.")
      else()
        message(FATAL_ERROR "ACLNN TYPE UNSPPORTED, ONLY SUPPORT aclnn/aclnn_inner/aclnn_exclude")
      endif()
    endforeach()
  else()
    file(GLOB OPDEF_SRCS ${SOURCE_DIR}/*_def*.cpp)
    if(OPDEF_SRCS)
      message(FATAL_ERROR
      "Should Manually specify aclnn/aclnn_inner/aclnn_exclude\n"
      "usage: add_modules_sources(OPTYPE optypes ACLNNTYPE aclnntypes)\n"
      "example: add_modules_sources(OPTYPE add ACLNNTYPE aclnn_exclude)"
      )
    endif()
  endif()
endmacro()

macro(add_modules_sources_with_soc)
  set(oneValueArgs OP_API_INDEPENDENT OP_API_DIR)
  set(multiValueArgs OPTYPE ACLNNTYPE)

  cmake_parse_arguments(MODULE "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  # 该段代码作用为兼容op_api新旧目录结构(旧： 嵌套于op_host下； 新： 与op_host同级)
  if (NOT DEFINED MODULE_OP_API_INDEPENDENT)
    set(MODULE_OP_API_INDEPENDENT OFF)
  endif()
    if(MODULE_OP_API_INDEPENDENT)
      # 新结构：op_api与op_host同级，需要指定有效路径
      if (NOT DEFINED MODULE_OP_API_DIR OR NOT EXISTS "${MODULE_OP_API_DIR}")
        message(FATAL_ERROR "OP_API_INDEPENDENT=ON时，必须传递有效的OP_API_DIR路径")
      endif()
      set(OP_API_SRC_DIR "${MODULE_OP_API_DIR}")
    else()
      # 旧结构：op_api嵌套在op_host目录下
      set(OP_API_SRC_DIR "${SOURCE_DIR}/op_api")
  endif()

  # opapi 默认全部编译
  file(GLOB OPAPI_SRCS ${OP_API_SRC_DIR}/*.cpp)
  if (OPAPI_SRCS)
    # aclnn
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_SRCS})
  else()
    if (NOT TARGET ${OPHOST_NAME}_opapi_obj)
      add_opapi_modules()
      add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/opapi_stub.cpp
          COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/opapi_stub.cpp
      )
      target_sources(${OPHOST_NAME}_opapi_obj PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/opapi_stub.cpp
      )
    endif()
  endif()
  file(GLOB OPAPI_HEADERS ${OP_API_SRC_DIR}/aclnn_*.h)
  if (OPAPI_HEADERS)
    target_sources(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE ${OPAPI_HEADERS})
  endif()

  # 是否编译该算子已经由op_add_subdirectory和每个二级目录判断完毕，默认走到这里全编

  file(GLOB OPINFER_SRCS ${SOURCE_DIR}/*_infershape*.cpp)
  foreach(ARCH ${ARCH_DIRECTORY})
    file(GLOB_RECURSE files ${SOURCE_DIR}/${ARCH}/*_infershape*.cpp)
    list(APPEND OPINFER_SRCS ${files})
  endforeach()

  if (OPINFER_SRCS)
    # proto
    add_infer_modules()
    target_sources(${OPHOST_NAME}_infer_obj PRIVATE ${OPINFER_SRCS})
  else()
    if (NOT TARGET ${OPHOST_NAME}_infer_obj)
      add_library(${OPHOST_NAME}_infer_obj OBJECT)
      add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/proto_stub.cpp
          COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/proto_stub.cpp
      )
      target_sources(${OPHOST_NAME}_infer_obj PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/proto_stub.cpp
      )
    endif()
  endif()

  foreach(ARCH ${ARCH_DIRECTORY})
    file(GLOB_RECURSE files ${SOURCE_DIR}/${ARCH}/*_tiling*.cpp)
    list(APPEND SUB_OPTILING_SRC ${files})
  endforeach()
  file(GLOB OPTILING_SRCS
      ${SOURCE_DIR}/*fallback*.cpp
      ${SOURCE_DIR}/*_tiling*.cpp
      ${SOURCE_DIR}/../op_graph/fallback_*.cpp
      ${SOURCE_DIR}/../graph_plugin/fallback_*.cpp)
  if (OPTILING_SRCS OR SUB_OPTILING_SRC)
    # tiling
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${OPTILING_SRCS} ${SUB_OPTILING_SRC})
    # target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${SOURCE_DIR}/../../ ${SOURCE_DIR})
  endif()

  file(GLOB AICPU_SRCS ${SOURCE_DIR}/*_aicpu*.cpp)
  if(AICPU_SRCS)
    add_aicpu_kernel_modules()
    target_sources(${OPHOST_NAME}_aicpu_obj PRIVATE ${AICPU_SRCS})
  endif()

  if (MODULE_OPTYPE)
    list(LENGTH MODULE_OPTYPE OpTypeLen)
    list(LENGTH MODULE_ACLNNTYPE AclnnTypeLen)
    if(NOT ${OpTypeLen} EQUAL ${AclnnTypeLen})
      message(FATAL_ERROR "OPTYPE AND ACLNNTYPE Should be One-to-One")
    endif()
    math(EXPR index "${OpTypeLen} - 1")
    foreach(i RANGE ${index})
      list(GET MODULE_OPTYPE ${i} OpType)
      list(GET MODULE_ACLNNTYPE ${i} AclnnType)
      if (${AclnnType} STREQUAL "aclnn" OR ${AclnnType} STREQUAL "aclnn_inner" OR ${AclnnType} STREQUAL "aclnn_exclude")
        file(GLOB OPDEF_SRCS ${SOURCE_DIR}/${OpType}_def*.cpp)

        if (OPDEF_SRCS)
          target_sources(${OPHOST_NAME}_opdef_${AclnnType}_obj INTERFACE ${OPDEF_SRCS})
        endif()
      elseif(${AclnnType} STREQUAL "no_need_aclnn")
        message(STATUS "aicpu or host aicpu no need aclnn.")
      else()
        message(FATAL_ERROR "ACLNN TYPE UNSPPORTED, ONLY SUPPORT aclnn/aclnn_inner/aclnn_exclude")
      endif()
    endforeach()
  else()
    file(GLOB OPDEF_SRCS ${SOURCE_DIR}/*_def*.cpp)
    if(OPDEF_SRCS)
      message(FATAL_ERROR
      "Should Manually specify aclnn/aclnn_inner/aclnn_exclude\n"
      "usage: add_modules_sources(OPTYPE optypes ACLNNTYPE aclnntypes)\n"
      "example: add_modules_sources(OPTYPE add ACLNNTYPE aclnn_exclude)"
      )
    endif()
  endif()
endmacro()

macro(add_mc2_modules_sources)
  set(multiValueArgs OPTYPE ACLNNTYPE)

  cmake_parse_arguments(MODULE "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  #opapi 默认全部编译
  file(GLOB OPAPI_SRCS ${SOURCE_DIR}/../op_api/*.cpp)
  if (OPAPI_SRCS)
    # aclnn
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_SRCS})
  endif()

  # 获取算子层级目录名称，判断是否编译该算子
  get_filename_component(PARENT_DIR ${SOURCE_DIR} DIRECTORY)
  get_filename_component(OP_NAME ${PARENT_DIR} NAME)
  list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
  # 记录全局的COMPILED_OPS和COMPILED_OP_DIRS，其中COMPILED_OP_DIRS只记录到算子名，例如moe/moe_token_permute_with_routing_map_grad
  set(COMPILED_OPS ${COMPILED_OPS} ${OP_NAME} CACHE STRING "Compiled Ops" FORCE)
  set(COMPILED_OP_DIRS ${COMPILED_OP_DIRS} ${PARENT_DIR} CACHE STRING "Compiled Ops Dirs" FORCE)
  
  file(GLOB OPINFER_SRCS ${SOURCE_DIR}/*_infershape*.cpp)
  if (OPINFER_SRCS)
    # proto
    add_infer_modules()
    target_sources(${OPHOST_NAME}_infer_obj PRIVATE ${OPINFER_SRCS})
  endif()

  file(GLOB_RECURSE OPTILING_SRCS
      ${SOURCE_DIR}/op_tiling/*.cpp
      ${SOURCE_DIR}/op_tiling/arch35/*.cpp
      ${SOURCE_DIR}/op_tiling/common/*.cpp
      ${SOURCE_DIR}/../op_graph/fallback*.cpp
  )
  if (OPTILING_SRCS)
    # tiling
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE 
      ${OPTILING_SRCS}
      ${OPS_TRANSFORMER_DIR}/mc2/common/src/matmul_formulaic_tiling.cpp
      ${OPS_TRANSFORMER_DIR}/mc2/common/src/mc2_tiling_utils.cpp
      ${OPS_TRANSFORMER_DIR}/mc2/common/src/mc2_matmul_tiling_cfg.cpp
      ${OPS_TRANSFORMER_DIR}/mc2/common/src/mc2_log.cpp
      ${OPS_TRANSFORMER_DIR}/mc2/3rd/ops_legacy/op_tiling/op_cache_tiling.cpp
      ${OPS_TRANSFORMER_DIR}/mc2/3rd/ops_legacy/op_tiling/runtime_kb_api.cpp
    )
  endif()

  file(GLOB GENTASK_SRCS
      ${SOURCE_DIR}/../op_graph/*_gen_task*.cpp
  )
  if(GENTASK_SRCS)
    add_opmaster_ct_gentask_modules()
    target_sources(${OPHOST_NAME}_opmaster_ct_gentask_obj PRIVATE ${GENTASK_SRCS})
  endif()

  file(GLOB AICPU_SRCS ${SOURCE_DIR}/*_aicpu*.cpp)
  if(AICPU_SRCS)
    add_aicpu_kernel_modules()
    target_sources(${OPHOST_NAME}_aicpu_obj PRIVATE ${AICPU_SRCS})
  endif()

  if (MODULE_OPTYPE)
    list(LENGTH MODULE_OPTYPE OpTypeLen)
    list(LENGTH MODULE_ACLNNTYPE AclnnTypeLen)
    if(NOT ${OpTypeLen} EQUAL ${AclnnTypeLen})
      message(FATAL_ERROR "OPTYPE AND ACLNNTYPE Should be One-to-One")
    endif()
    math(EXPR index "${OpTypeLen} - 1")
    foreach(i RANGE ${index})
      list(GET MODULE_OPTYPE ${i} OpType)
      list(GET MODULE_ACLNNTYPE ${i} AclnnType)
      if (${AclnnType} STREQUAL "aclnn" OR ${AclnnType} STREQUAL "aclnn_inner" OR ${AclnnType} STREQUAL "aclnn_exclude")
        file(GLOB OPDEF_SRCS ${SOURCE_DIR}/${OpType}_def*.cpp)

        if (OPDEF_SRCS)
          target_sources(${OPHOST_NAME}_opdef_${AclnnType}_obj INTERFACE ${OPDEF_SRCS})
        endif()
      elseif(${AclnnType} STREQUAL "no_need_aclnn")
        message(STATUS "aicpu or host aicpu no need aclnn.")
      else()
        message(FATAL_ERROR "ACLNN TYPE UNSPPORTED, ONLY SUPPORT aclnn/aclnn_inner/aclnn_exclude")
      endif()
    endforeach()
  else()
    file(GLOB OPDEF_SRCS ${SOURCE_DIR}/*_def*.cpp)
    if(OPDEF_SRCS)
      message(FATAL_ERROR
      "Should Manually specify aclnn/aclnn_inner/aclnn_exclude\n"
      "usage: add_modules_sources(OPTYPE optypes ACLNNTYPE aclnntypes)\n"
      "example: add_modules_sources(OPTYPE add ACLNNTYPE aclnn_exclude)"
      )
    endif()
  endif()
endmacro()


# usage: add_modules_sources_aicpu(OPTYPE ACLNNTYPE DEPENDENCIES COMPUTE_UNIT TILING_DIR DISABLE_IN_OPP)
# ACLNNTYPE 支持类型aclnn/aclnn_inner/aclnn_exclude
# OPTYPE 和 ACLNNTYPE 需一一对应
# DEPENDENCIES 指定依赖的算子名称列表，如果开启 experimental，则会优先加载 experimental 下的算子
# COMPUTE_UNIT 设置支持芯片版本号，必须与TILING_DIR一一对应，示例：ascend910b ascend910_95
# TILING_DIR 设置所支持芯片类型对应的tiling文件目录，必须与COMPUTE_UNIT一一对应，示例：arch32 arch35
# DISABLE_IN_OPP 设置是否在opp包中编译tiling文件，布尔类型：TRUE，FALSE
macro(add_modules_sources_aicpu)
  set(oneValueArgs DISABLE_IN_OPP)
  set(multiValueArgs OPTYPE ACLNNTYPE DEPENDENCIES COMPUTE_UNIT TILING_DIR)

  cmake_parse_arguments(MODULE "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  get_filename_component(OP_NAME ${SOURCE_DIR} NAME)

  add_dependent_ops("${MODULE_DEPENDENCIES}")


  # opapi l0 默认全部编译
  file(GLOB OPAPI_L0_SRCS ${SOURCE_DIR}/op_api/*.cpp)
  list(FILTER OPAPI_L0_SRCS EXCLUDE REGEX "aclnn_")
  if(OPAPI_L0_SRCS)
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_L0_SRCS})
  endif()

  file(GLOB OPAPI_HEADERS ${SOURCE_DIR}/op_api/aclnn_*.h)
  if(OPAPI_HEADERS)
    target_sources(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE ${OPAPI_HEADERS})
  endif()

  file(GLOB OPAPI_L2_SRCS ${SOURCE_DIR}/op_api/aclnn_*.cpp)
  if(OPAPI_L2_SRCS)
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_L2_SRCS})
  endif()

  file(GLOB OPINFER_SRCS ${SOURCE_DIR}/op_host/*_infershape*.cpp)
  if(OPINFER_SRCS)
    add_infer_modules()
    target_sources(${OPHOST_NAME}_infer_obj PRIVATE ${OPINFER_SRCS})
  endif()

  file(GLOB OPTILING_SRCS ${SOURCE_DIR}/*_tiling*.cpp)
  if(OPTILING_SRCS)
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${OPTILING_SRCS})
  else()
    if (NOT TARGET ${OPHOST_NAME}_tiling_obj)
      add_tiling_modules()
      add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/optiling_stub.cpp
          COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/optiling_stub.cpp
      )
      target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
          ${CMAKE_CURRENT_BINARY_DIR}/optiling_stub.cpp
      )
    endif()
  endif()


  file(GLOB AICPU_SRCS ${SOURCE_DIR}/op_kernel_aicpu/*_aicpu*.cpp)

  if(AICPU_SRCS)
    if(NOT BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
      add_aicpu_kernel_modules()
      target_sources(${OPHOST_NAME}_aicpu_obj PRIVATE ${AICPU_SRCS})
    else()
      file(GLOB AICPU_JSON_FILE ${SOURCE_DIR}/op_kernel_aicpu/*.json)
      add_aicpu_cust_kernel_modules(${OP_NAME} ${AICPU_SRCS} ${AICPU_JSON_FILE})
    endif()
  endif()

  if(MODULE_OPTYPE)
    list(LENGTH MODULE_OPTYPE OpTypeLen)
    list(LENGTH MODULE_ACLNNTYPE AclnnTypeLen)
    if(NOT ${OpTypeLen} EQUAL ${AclnnTypeLen})
      message(FATAL_ERROR "OPTYPE AND ACLNNTYPE Should be One-to-One")
    endif()
    math(EXPR index "${OpTypeLen} - 1")
    foreach(i RANGE ${index})
      list(GET MODULE_OPTYPE ${i} OpType)
      list(GET MODULE_ACLNNTYPE ${i} AclnnType)
      if(${AclnnType} STREQUAL "aclnn"
         OR ${AclnnType} STREQUAL "aclnn_inner"
         OR ${AclnnType} STREQUAL "aclnn_exclude"
        )
        file(GLOB OPDEF_SRCS ${SOURCE_DIR}/op_host/${OpType}_def*.cpp)
        if(OPDEF_SRCS)
          target_sources(${OPHOST_NAME}_opdef_${AclnnType}_obj INTERFACE ${OPDEF_SRCS})
        endif()
      else()
        message(FATAL_ERROR "ACLNN TYPE UNSPPORTED, ONLY SUPPORT aclnn/aclnn_inner/aclnn_exclude")
      endif()
    endforeach()
  else()
    file(GLOB OPDEF_SRCS ${SOURCE_DIR}/op_host/*_def*.cpp)
    if(OPDEF_SRCS)
      message(
        FATAL_ERROR
          "Should Manually specify aclnn/aclnn_inner/aclnn_exclude\n"
          "usage: add_modules_sources_aicpu(OPTYPE optypes ACLNNTYPE aclnntypes)\n"
          "example: add_modules_sources_aicpu(OPTYPE add ACLNNTYPE aclnn_exclude)"
        )
    endif()
  endif()

  file(GLOB OP_GRAPH_PROTO_HEADERS ${SOURCE_DIR}/op_graph/*_proto*.h)
  if(OP_GRAPH_PROTO_HEADERS)
    target_sources(${GRAPH_PLUGIN_NAME}_proto_headers INTERFACE ${OP_GRAPH_PROTO_HEADERS})
  endif()

  set(ENABLE_AICPU ON CACHE BOOL "enable aicpu kernel" FORCE)

endmacro()

# 添加opapi object
function(add_opapi_modules)
  if (NOT TARGET ${OPHOST_NAME}_opapi_obj)
    add_library(${OPHOST_NAME}_opapi_obj OBJECT)
    unset(OPAPI_UT_DEPEND_INC)
    if(UT_TEST_ALL OR OP_API_UT)
      set(OPAPI_UT_DEPEND_INC ${UT_PATH}/op_api/stub)
    endif()
    target_include_directories(${OPHOST_NAME}_opapi_obj
      PRIVATE
      ${OPAPI_INCLUDE}
      ${OPAPI_UT_DEPEND_INC}
      $<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include>
      $<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/aclnn>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef/common/util>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/op_common>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/op_common/op_host>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include/op_common>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include/op_common/op_host>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/hccl/external>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc/profiling>>
      ${OPS_TRANSFORMER_DIR}/mc2/common/inc
      ${OPS_TRANSFORMER_DIR}/mc2/3rd
    )
    target_compile_definitions(${OPHOST_NAME}_opapi_obj PRIVATE
      _GLIBCXX_USE_CXX11_ABI=0
      BUILD_OPEN_PROJECT_API=1
    )
    target_compile_options(${OPHOST_NAME}_opapi_obj
      PRIVATE
      -Dgoogle=ascend_private
      -DACLNN_LOG_FMT_CHECK
    )
    target_link_libraries(${OPHOST_NAME}_opapi_obj
      PUBLIC
      $<BUILD_INTERFACE:intf_pub>
      $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
      -Wl,--whole-archive
      ops_aclnn
      -Wl,--no-whole-archive
      $<$<BOOL:${dlog_FOUND}>:$<BUILD_INTERFACE:dlog_headers>>
      nnopbase
      profapi
      ge_common_base
      ascend_dump
      ascendalog
      dl
      )
  endif()
endfunction()

set(INFER_OBJ_INCLUDE
  ${OP_PROTO_INCLUDE}
  $<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include>
  $<BUILD_INTERFACE:${OPS_TRANSFORMER_DIR}/common/include>
  $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>>
  $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include/op_common>>
  $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/hccl/external>>
  $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef/common/util>>
  $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/external>>
  $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base>>
  ${OPS_TRANSFORMER_DIR}/mc2/common/inc
  ${OPS_TRANSFORMER_DIR}/mc2/3rd
)

# 添加infer object
function(add_infer_modules)
  if (NOT TARGET ${OPHOST_NAME}_infer_obj)
    add_library(${OPHOST_NAME}_infer_obj OBJECT)
    target_include_directories(${OPHOST_NAME}_infer_obj
      PRIVATE ${INFER_OBJ_INCLUDE}
    )
    target_compile_definitions(${OPHOST_NAME}_infer_obj
      PRIVATE
      LOG_CPP
      OPS_UTILS_LOG_SUB_MOD_NAME="OP_PROTO"
      $<$<BOOL:${ENABLE_TEST}>:ASCEND_OPSPROTO_UT>
    )
    target_compile_options(${OPHOST_NAME}_infer_obj
      PRIVATE
      $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1>
      -Dgoogle=ascend_private
      -fvisibility=hidden
    )
    target_link_libraries(${OPHOST_NAME}_infer_obj
      PRIVATE
      $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
      $<BUILD_INTERFACE:intf_pub>
      $<BUILD_INTERFACE:ops_transformer_utils_proto_headers>
      $<$<BOOL:${alog_FOUND}>:$<BUILD_INTERFACE:alog_headers>>
      $<$<BOOL:${dlog_FOUND}>:$<BUILD_INTERFACE:dlog_headers>>
      -Wl,--whole-archive
      rt2_registry_static
      -Wl,--no-whole-archive
      -Wl,--no-as-needed
      exe_graph
      graph
      graph_base
      register
      ascendalog
      error_manager
      platform
      -Wl,--as-needed
      c_sec
    )
  endif()
endfunction()

# 添加tiling object
function(add_tiling_modules)
  if (NOT TARGET ${OPHOST_NAME}_tiling_obj)
    add_library(${OPHOST_NAME}_tiling_obj OBJECT)
    add_dependencies(${OPHOST_NAME}_tiling_obj json)
    target_include_directories(${OPHOST_NAME}_tiling_obj
      PRIVATE ${OP_TILING_INCLUDE}
      $<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include>
      $<BUILD_INTERFACE:${OPS_TRANSFORMER_DIR}/common/include>
      
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include/op_common/op_host>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef/common/util>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling>>
      ${OPS_TRANSFORMER_DIR}/mc2/common/inc
      ${OPS_TRANSFORMER_DIR}/mc2/3rd
    )
    target_compile_definitions(${OPHOST_NAME}_tiling_obj
      PRIVATE
      LOG_CPP
      OPS_UTILS_LOG_SUB_MOD_NAME="OP_TILING"
      $<$<BOOL:${ENABLE_TEST}>:ASCEND_OPTILING_UT>
    )
    target_compile_options(${OPHOST_NAME}_tiling_obj
      PRIVATE
      $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1>
      -Dgoogle=ascend_private
      -fvisibility=hidden
      -fno-strict-aliasing
    )
    target_link_libraries(${OPHOST_NAME}_tiling_obj
      PRIVATE
      $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
      $<BUILD_INTERFACE:intf_pub>
      $<BUILD_INTERFACE:ops_transformer_utils_tiling_headers>
      $<$<BOOL:${alog_FOUND}>:$<BUILD_INTERFACE:alog_headers>>
      $<$<BOOL:${dlog_FOUND}>:$<BUILD_INTERFACE:dlog_headers>>
      -Wl,--whole-archive
      rt2_registry_static
      -Wl,--no-whole-archive
      -Wl,--no-as-needed
      graph
      graph_base
      exe_graph
      platform
      register
      # ascendalog
      error_manager
      -Wl,--as-needed
      -Wl,--whole-archive
      tiling_api
      -Wl,--no-whole-archive
      # mmpa
      c_sec
    )
  endif()
endfunction()

function(add_graph_plugin_modules)
  if(NOT TARGET ${GRAPH_PLUGIN_NAME}_obj)
    add_library(${GRAPH_PLUGIN_NAME}_obj OBJECT)
    target_include_directories(${GRAPH_PLUGIN_NAME}_obj PRIVATE 
      ${OP_PROTO_INCLUDE}
    )
    target_compile_definitions(${GRAPH_PLUGIN_NAME}_obj PRIVATE OPS_UTILS_LOG_SUB_MOD_NAME="GRAPH_PLUGIN" LOG_CPP)
    target_compile_options(
      ${GRAPH_PLUGIN_NAME}_obj PRIVATE $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                                       -fvisibility=hidden
      )
    target_link_libraries(
      ${GRAPH_PLUGIN_NAME}_obj
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
              $<$<TARGET_EXISTS:ops_base_util_objs>:$<TARGET_OBJECTS:ops_base_util_objs>>
              $<$<TARGET_EXISTS:ops_base_infer_objs>:$<TARGET_OBJECTS:ops_base_infer_objs>>
      )
  endif()
endfunction()

# 添加gentask object
function(add_opmaster_ct_gentask_modules)
  message(STATUS "add_opmaster_ct_gentask_modules start")
  if (NOT TARGET ${OPHOST_NAME}_opmaster_ct_gentask_obj)
    add_library(${OPHOST_NAME}_opmaster_ct_gentask_obj OBJECT)
    add_dependencies(${OPHOST_NAME}_opmaster_ct_gentask_obj json)

    target_include_directories(${OPHOST_NAME}_opmaster_ct_gentask_obj
      PRIVATE ${OP_TILING_INCLUDE}
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include/experiment/metadef/common/util>>
    )
    target_compile_definitions(${OPHOST_NAME}_opmaster_ct_gentask_obj
      PRIVATE
      LOG_CPP
    )
    target_compile_options(${OPHOST_NAME}_opmaster_ct_gentask_obj
      PRIVATE
      $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1>
      -Dgoogle=ascend_private
      -fvisibility=hidden
      -fno-strict-aliasing
    )
    message(STATUS "xxxx compile add_opmaster_ct_gentask_modules")
    target_link_libraries(${OPHOST_NAME}_opmaster_ct_gentask_obj
      PRIVATE
      $<BUILD_INTERFACE:intf_pub_cxx17>
      $<$<BOOL:${alog_FOUND}>:$<BUILD_INTERFACE:alog_headers>>
      $<$<BOOL:${dlog_FOUND}>:$<BUILD_INTERFACE:dlog_headers>>
      # $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:alog_headers>>
      # $<$<NOT:$<BOOL:${BUILD_OPEN_PROJECT}>>:$<BUILD_INTERFACE:slog_headers>>
    )
  endif()
endfunction()


# useage: add_graph_plugin_sources()
macro(add_graph_plugin_sources)
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  # 获取算子层级目录名称，判断是否编译该算子
  get_filename_component(PARENT_DIR ${SOURCE_DIR} DIRECTORY)
  get_filename_component(OP_NAME ${PARENT_DIR} NAME)
  if(DEFINED ASCEND_OP_NAME
     AND NOT "${ASCEND_OP_NAME}" STREQUAL ""
     AND NOT "${ASCEND_OP_NAME}" STREQUAL "all"
     AND NOT "${ASCEND_OP_NAME}" STREQUAL "ALL"
    )
    if(NOT ${OP_NAME} IN_LIST ASCEND_OP_NAME)
      return()
    endif()
  endif()

  file(GLOB GRAPH_PLUGIN_SRCS 
      ${SOURCE_DIR}/*_graph_plugin*.cpp
  )
  if(GRAPH_PLUGIN_SRCS)
    add_graph_plugin_modules()
    target_sources(${GRAPH_PLUGIN_NAME}_obj PRIVATE ${GRAPH_PLUGIN_SRCS})
  endif()

  file(GLOB GRAPH_PLUGIN_PROTO_HEADERS ${SOURCE_DIR}/*_proto*.h)
  if(GRAPH_PLUGIN_PROTO_HEADERS)
    target_sources(${GRAPH_PLUGIN_NAME}_proto_headers INTERFACE ${GRAPH_PLUGIN_PROTO_HEADERS})
  endif()
endmacro()

function(protobuf_generate_external comp c_var h_var)
  if (NOT ARGN)
    message(SEND_ERROR "Error: protobuf_generate_external() called without any proto files")
    return()
  endif()

  set(${c_var})
  set(${h_var})
  set(_add_target FALSE)

  set(extra_option "")
  foreach(arg ${ARGN})
    if ("${arg}" MATCHES "--proto_path")
      set(extra_option ${arg})
    endif()
  endforeach()

  foreach(file ${ARGN})
    if ("${file}" STREQUAL "TARGET")
      set(_add_target TRUE)
      continue()
    endif()

    if ("${file}" MATCHES "--proto_path")
      continue()
    endif()

    get_filename_component(abs_file ${file} ABSOLUTE)
    get_filename_component(file_name ${file} NAME_WE)
    get_filename_component(file_dir ${abs_file} PATH)
    get_filename_component(parent_subdir ${file_dir} NAME)

    if ("${parent_subdir}" STREQUAL "proto")
      set(proto_output_path ${CMAKE_BINARY_DIR}/proto/${comp}/proto)
    else()
      set(proto_output_path ${CMAKE_BINARY_DIR}/proto/${comp}/proto/${parent_subdir})
    endif()
    list(APPEND ${c_var} "${proto_output_path}/${file_name}.pb.cc")
    list(APPEND ${h_var} "${proto_output_path}/${file_name}.pb.h")

    add_custom_command(
      OUTPUT "${proto_output_path}/${file_name}.pb.cc" "${proto_output_path}/${file_name}.pb.h"
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      COMMAND ${CMAKE_COMMAND} -E make_directory "${proto_output_path}"
      COMMAND ${CMAKE_COMMAND} -E echo "generate proto cpp_out ${comp} by ${abs_file}"
      COMMAND ${Protobuf_PROTOC_EXECUTABLE} -I${file_dir} ${extra_option} --cpp_out=${proto_output_path} ${abs_file}
      DEPENDS ${abs_file} ascend_protobuf_build_transformer
      COMMENT "Running C++ protocol buffer compiler on ${file}" VERBATIM)

  endforeach()

    if (_add_target)
      add_custom_target(
        ${comp} DEPENDS ${${c_var}} ${${h_var}}) 
    endif()

    set_source_files_properties(${${c_var}} ${${h_var}} PROPERTIES GENERATED TRUE)
    set(${c_var} ${${c_var}} PARENT_SCOPE)
    set(${h_var} ${${h_var}} PARENT_SCOPE)

endfunction()

function(add_onnx_plugin_modules)
  if (NOT TARGET ${ONNX_PLUGIN_NAME}_obj)
    set(ge_onnx_proto_srcs
      ${ASCEND_DIR}/include/proto/ge_onnx.proto)
    
    protobuf_generate_external(onnx ge_onnx_proto_cc ge_onnx_proto_h ${ge_onnx_proto_srcs})

    add_library(${ONNX_PLUGIN_NAME}_obj OBJECT ${ge_onnx_proto_h})
    # 为特定目标设置C++14标准
    set_target_properties(${ONNX_PLUGIN_NAME}_obj PROPERTIES
      CXX_STANDARD 14
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
    )
    target_include_directories(${ONNX_PLUGIN_NAME}_obj PRIVATE ${OP_PROTO_INCLUDE} ${Protobuf_INCLUDE} ${Protobuf_PATH} ${CMAKE_BINARY_DIR}/proto ${ONNX_PLUGIN_COMMON_INCLUDE} ${JSON_INCLUDE_DIR} ${ABSL_SOURCE_DIR})
    target_compile_definitions(${ONNX_PLUGIN_NAME}_obj PRIVATE OPS_UTILS_LOG_SUB_MOD_NAME="ONNX_PLUGIN" LOG_CPP)

    if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
      target_compile_options(
        ${ONNX_PLUGIN_NAME}_obj PRIVATE -Dgoogle=ascend_private -fvisibility=hidden -Wno-shadow -Wno-unused-parameter
      )
    else()
      target_compile_options(
        ${ONNX_PLUGIN_NAME}_obj PRIVATE $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                                       -fvisibility=hidden -Wno-shadow -Wno-unused-parameter
      )
    endif()

    target_link_libraries(
      ${ONNX_PLUGIN_NAME}_obj
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx14,intf_pub_cxx14>>
              $<BUILD_INTERFACE:dlog_headers>
              $<$<TARGET_EXISTS:ops_base_util_objs>:$<TARGET_OBJECTS:ops_base_util_objs>>
              $<$<TARGET_EXISTS:ops_base_infer_objs>:$<TARGET_OBJECTS:ops_base_infer_objs>>
              json
      )
  endif()

endfunction()

macro(add_onnx_plugin_sources)
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  file(GLOB ONNX_PLUGIN_SRCS ${SOURCE_DIR}/*_onnx_plugin.cpp)
  if(ONNX_PLUGIN_SRCS)
    add_onnx_plugin_modules()
    target_sources(${ONNX_PLUGIN_NAME}_obj PRIVATE ${ONNX_PLUGIN_SRCS})
  else()
    message(WARNING "No onnx plugin source files found in ${SOURCE_DIR}")
  endif()
endmacro()