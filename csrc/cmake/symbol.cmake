# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

function(gen_common_symbol)
  install(DIRECTORY ${OPS_TRANSFORMER_COMMON_INC_HEADERS}
    DESTINATION ${COMMON_INC_INSTALL_DIR}
  )
endfunction()

# ophost shared
function(gen_ophost_symbol)
  add_library(${OPHOST_NAME} SHARED
    $<$<TARGET_EXISTS:${OPHOST_NAME}_infer_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_infer_obj>>
    $<$<TARGET_EXISTS:${OPHOST_NAME}_tiling_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>>
    $<$<TARGET_EXISTS:${OPHOST_NAME}_aicpu_objs>:$<TARGET_OBJECTS:${OPHOST_NAME}_aicpu_objs>>
    $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
    $<$<TARGET_EXISTS:${OPHOST_NAME}_opmaster_ct_gentask_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_opmaster_ct_gentask_obj>>
  )

  target_link_libraries(
    ${OPHOST_NAME}
    PRIVATE $<BUILD_INTERFACE:intf_pub_cxx17>
            c_sec
            -Wl,--no-as-needed
            register
            $<$<TARGET_EXISTS:opsbase>:opsbase>
            -Wl,--as-needed
            -Wl,--whole-archive
            rt2_registry_static
            -Wl,--no-whole-archive
            tiling_api
            -Wl,-Bsymbolic
    )

  target_link_directories(${OPHOST_NAME}
    PRIVATE
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
  )

  install(TARGETS ${OPHOST_NAME}
    LIBRARY DESTINATION ${OPHOST_LIB_INSTALL_PATH}
  )
endfunction()

# graph_plugin shared
function(gen_opgraph_symbol)
  add_library(${OPGRAPH_NAME} SHARED
    $<$<TARGET_EXISTS:${GRAPH_PLUGIN_NAME}_obj>:$<TARGET_OBJECTS:${GRAPH_PLUGIN_NAME}_obj>>
  )

  target_link_libraries(
    ${OPGRAPH_NAME}
    PRIVATE $<BUILD_INTERFACE:intf_pub_cxx17>
            c_sec
            -Wl,--no-as-needed
            register
            $<$<TARGET_EXISTS:opsbase>:opsbase>
            -Wl,--as-needed
            -Wl,--whole-archive
            rt2_registry_static
            -Wl,--no-whole-archive
            -Wl,-Bsymbolic
    )

  target_link_directories(${OPGRAPH_NAME}
    PRIVATE
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
  )

  set_target_properties(${OPGRAPH_NAME} PROPERTIES OUTPUT_NAME "opgraph_transformer")
  install(TARGETS ${OPGRAPH_NAME}
    LIBRARY DESTINATION ${OPGRAPH_LIB_INSTALL_DIR}
  )
  install(FILES ${ASCEND_GRAPH_CONF_DST}/ops_proto_transformer.h
    DESTINATION ${OPGRAPH_INC_INSTALL_DIR} OPTIONAL
  )
endfunction()

function(gen_opapi_symbol)
  # opapi shared	
  add_library(${OPAPI_NAME} SHARED	
    $<$<TARGET_EXISTS:${OPHOST_NAME}_opapi_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_opapi_obj>>
    $<$<TARGET_EXISTS:opbuild_gen_aclnn_all>:$<TARGET_OBJECTS:opbuild_gen_aclnn_all>>
  )

  target_link_libraries(${OPAPI_NAME}
    PUBLIC
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--whole-archive
    ops_aclnn
    -Wl,--no-whole-archive
    nnopbase
    profapi
    ge_common_base
    ascend_dump
    ascendalog
    opapi_math
    dl
  )
endfunction()

function(gen_built_in_opapi_symbol)
  install(TARGETS ${OPAPI_NAME}
    LIBRARY DESTINATION ${ACLNN_LIB_INSTALL_DIR}
  )
endfunction()

function(gen_cust_opapi_symbol)
  #op_api
  set_target_properties(${OPAPI_NAME} PROPERTIES OUTPUT_NAME "cust_opapi")

  install(TARGETS ${OPAPI_NAME}
    LIBRARY DESTINATION ${ACLNN_LIB_INSTALL_DIR}
  )
endfunction()

function(gen_cust_optiling_symbol)
  # op_tiling
  if(NOT TARGET ${OPHOST_NAME}_tiling_obj)
    return()
  endif()
  add_library(cust_opmaster SHARED
    $<$<TARGET_EXISTS:${OPHOST_NAME}_tiling_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>
    $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>>
  )
  target_link_libraries(cust_opmaster
    PRIVATE
    c_sec
    tiling_api
    -Wl,--no-as-needed
    register
    $<$<TARGET_EXISTS:opsbase>:opsbase>
    -Wl,--as-needed
    -Wl,--whole-archive
    rt2_registry_static
    -Wl,--no-whole-archive
  )
  target_link_directories(cust_opmaster
    PRIVATE
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
  )
  set_target_properties(cust_opmaster PROPERTIES OUTPUT_NAME "cust_opmaster_rt2.0")

  install(TARGETS cust_opmaster
    LIBRARY DESTINATION ${OPTILING_LIB_INSTALL_DIR}
  )
  add_custom_target(optiling_compat ALL
    COMMAND ln -sf lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:cust_opmaster>
            ${CMAKE_BINARY_DIR}/liboptiling.so
  )
  install(FILES ${CMAKE_BINARY_DIR}/liboptiling.so
    DESTINATION ${OPTILING_INSTALL_DIR}
  )
endfunction()

function(gen_cust_proto_symbol)
  # op_proto
  if(NOT TARGET ${OPHOST_NAME}_infer_obj)
    return()
  endif()
  add_library(cust_proto SHARED
    $<$<TARGET_EXISTS:${OPHOST_NAME}_infer_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_infer_obj>>
    $<$<TARGET_EXISTS:${GRAPH_PLUGIN_NAME}_obj>:$<TARGET_OBJECTS:${GRAPH_PLUGIN_NAME}_obj>>
  )
  target_link_libraries(cust_proto
    PRIVATE
    c_sec
    -Wl,--no-as-needed
    register
    $<$<TARGET_EXISTS:opsbase>:opsbase>
    -Wl,--as-needed
    -Wl,--whole-archive
    rt2_registry_static
    -Wl,--no-whole-archive
  )
  target_link_directories(cust_proto
    PRIVATE
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
  )
  set_target_properties(cust_proto PROPERTIES OUTPUT_NAME "cust_opsproto_rt2.0")

  install(TARGETS cust_proto
    LIBRARY DESTINATION ${OPPROTO_LIB_INSTALL_DIR}
  )
  file(GLOB_RECURSE proto_headers
    ${ASCEND_AUTOGEN_PATH}/*_proto.h
  )
  install(FILES ${proto_headers}
    DESTINATION ${OPPROTO_INC_INSTALL_DIR} OPTIONAL
  )
endfunction()

function(gen_cust_aicpu_json_symbol)
  get_property(ALL_AICPU_JSON_FILES GLOBAL PROPERTY AICPU_JSON_FILES)
  if(NOT ALL_AICPU_JSON_FILES)
    message(STATUS "No aicpu json files to merge, skipping.")
    return()
  endif()

  set(MERGED_JSON ${CMAKE_BINARY_DIR}/cust_aicpu_kernel.json)
  add_custom_command(
    OUTPUT ${MERGED_JSON}
    COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/util/merge_aicpu_info_json.sh ${CMAKE_SOURCE_DIR} ${MERGED_JSON} ${ALL_AICPU_JSON_FILES}
    DEPENDS ${ALL_AICPU_JSON_FILES}
    COMMENT "Merging Json files into ${MERGED_JSON}"
    VERBATIM
  )
  add_custom_target(merge_aicpu_json ALL DEPENDS ${MERGED_JSON})
  install(
    FILES ${MERGED_JSON}
    DESTINATION packages/vendors/${VENDOR_NAME}_transformer/op_impl/cpu/config
    OPTIONAL
  )
endfunction()

function(gen_cust_aicpu_kernel_symbol)
  if(NOT AICPU_CUST_OBJ_TARGETS)
    message(STATUS "No aicpu cust obj targets found, skipping.")
    return()
  endif()

  set(ARM_CXX_COMPILER ${ASCEND_DIR}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++)
  set(ARM_SO_OUTPUT ${CMAKE_BINARY_DIR}/libtransformer_aicpu_kernels.so)

  set(ALL_OBJECTS "")
  foreach(tgt IN LISTS AICPU_CUST_OBJ_TARGETS)
    list(APPEND ALL_OBJECTS $<TARGET_OBJECTS:${tgt}>)
  endforeach()

  message(STATUS "Linking cust_aicpu_kernels with ARM toolchain: ${ARM_CXX_COMPILER}")
  message(STATUS "Objects: ${ALL_OBJECTS}")
  message(STATUS "Output: ${ARM_SO_OUTPUT}")

  if(EXISTS ${ASCEND_DIR}/ops_base/lib64/libaicpu_context.a)
    set(LIBAICPU_CONTEXT_PATH ${ASCEND_DIR}/ops_base/lib64/libaicpu_context.a)
  else()
    set(LIBAICPU_CONTEXT_PATH ${ASCEND_DIR}/lib64/libaicpu_context.a)
  endif()

  if(EXISTS ${ASCEND_DIR}/ops_base/lib64/libbase_ascend_protobuf.a)
    set(LIBBASE_ASCEND_PROTOBUF_PATH ${ASCEND_DIR}/ops_base/lib64/libbase_ascend_protobuf.a)
  else()
    set(LIBBASE_ASCEND_PROTOBUF_PATH ${ASCEND_DIR}/lib64/libbase_ascend_protobuf.a)
  endif()

  add_custom_command(
    OUTPUT ${ARM_SO_OUTPUT}
    COMMAND ${ARM_CXX_COMPILER} -shared ${ALL_OBJECTS}
      -Wl,--whole-archive
      ${LIBAICPU_CONTEXT_PATH}
      ${LIBBASE_ASCEND_PROTOBUF_PATH}
      -Wl,--no-whole-archive
      -Wl,-Bsymbolic
      -Wl,--exclude-libs=libbase_ascend_protobuf.a
      -s
      -o ${ARM_SO_OUTPUT}
    DEPENDS ${AICPU_CUST_OBJ_TARGETS}
    COMMENT "Linking cust_aicpu_kernels.so using ARM toolchain"
  )
  add_custom_target(cust_aicpu_kernels ALL DEPENDS ${ARM_SO_OUTPUT})

  install(
    FILES ${ARM_SO_OUTPUT}
    DESTINATION packages/vendors/${VENDOR_NAME}_transformer/op_impl/cpu/aicpu_kernel/impl
    OPTIONAL
  )
endfunction()

function(gen_onnx_plugin_symbol)
  add_library(
    ${ONNX_PLUGIN_NAME} SHARED
    $<$<TARGET_EXISTS:${ONNX_PLUGIN_NAME}_obj>:$<TARGET_OBJECTS:${ONNX_PLUGIN_NAME}_obj>>
  )

  target_link_libraries(
    ${ONNX_PLUGIN_NAME}
    PRIVATE $<BUILD_INTERFACE:intf_pub_cxx14>
            c_sec
            -Wl,--no-as-needed
            register
            $<$<TARGET_EXISTS:opsbase>:opsbase>
            -Wl,--as-needed
            -Wl,--whole-archive
            rt2_registry_static
            -Wl,--no-whole-archive
    )

  target_link_directories(${ONNX_PLUGIN_NAME} PRIVATE ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64)

  install(
    TARGETS ${ONNX_PLUGIN_NAME}
    LIBRARY DESTINATION ${ONNX_PLUGIN_LIB_INSTALL_DIR}
    )

endfunction()

function(gen_norm_symbol)
  gen_common_symbol()

  if (ENABLE_OPS_HOST)
    gen_ophost_symbol()
    gen_opapi_symbol()
    gen_built_in_opapi_symbol()
  endif()

  gen_opgraph_symbol()
  gen_onnx_plugin_symbol()

endfunction()

function(gen_cust_symbol)
  gen_opapi_symbol()

  gen_cust_opapi_symbol()

  gen_cust_optiling_symbol()

  gen_cust_proto_symbol()

  gen_cust_aicpu_json_symbol()

  gen_cust_aicpu_kernel_symbol()
endfunction()