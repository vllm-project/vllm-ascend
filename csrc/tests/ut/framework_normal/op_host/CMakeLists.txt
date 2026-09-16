# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.18.4)

if(UT_TEST_ALL OR OP_HOST_UT)
  ## set variables
  set(OP_HOST_UT_EXE ${PKG_NAME}_op_host_ut)

  if(NOT TARGET softmax_tiling_ut_stub)
    set(_SOFTMAX_STUB_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../common)
    add_library(softmax_tiling_ut_stub STATIC
        ${_SOFTMAX_STUB_COMMON_DIR}/softmax_tiling_mocker.cpp
    )
    target_include_directories(softmax_tiling_ut_stub PRIVATE
        ${_SOFTMAX_STUB_COMMON_DIR}
        ${OP_TILING_INCLUDE}
    )
    target_link_libraries(softmax_tiling_ut_stub PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
    )
  endif()

  ## add optiling ut cases static lib: libtransformer_op_tiling_ut_cases.a
  add_optiling_ut_modules(${OP_TILING_MODULE_NAME})

  ## add opinfershape ut cases static lib: libtransformer_op_infershape_ut_cases.a
  add_infershape_ut_modules(${OP_INFERSHAPE_MODULE_NAME})

  ## add ophost source so: libophost_transformer_ut.so
  add_library(${OPHOST_NAME}_ut SHARED
      $<$<TARGET_EXISTS:${OPHOST_NAME}_infer_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_infer_obj>>
      $<$<TARGET_EXISTS:${OPHOST_NAME}_tiling_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>>
      $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
      $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
      $<$<TARGET_EXISTS:opbase_infer_objs>:$<TARGET_OBJECTS:opbase_infer_objs>>
      $<$<TARGET_EXISTS:opbase_tiling_objs>:$<TARGET_OBJECTS:opbase_tiling_objs>>
  )
  target_link_directories(${OPHOST_NAME}_ut PRIVATE
      ${ASCEND_DIR}/lib64
  )
  target_link_libraries(${OPHOST_NAME}_ut PRIVATE
      $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
      -Wl,--no-as-needed
      -Wl,--as-needed
      # The softmax UT stub intentionally redefines three symbols from
      # `tiling_api` so the mocked implementations win during UT linking.
      # GNU ld supports `--allow-multiple-definition`, but does not provide a
      # matching `--no-allow-multiple-definition`, so we cannot narrow the
      # relaxation further on this toolchain.
      -Wl,--allow-multiple-definition
      $<$<TARGET_EXISTS:softmax_tiling_ut_stub>:softmax_tiling_ut_stub>
      $<$<TARGET_EXISTS:mc2_tiling_ut_stub>:mc2_tiling_ut_stub>
      -Wl,--whole-archive
          rt2_registry_static
      -Wl,--no-whole-archive
      tiling_api
      runtime
      acl_rt
      ascendalog
      dl
  )

  ## add ophost ut exe:  transformer_op_host_ut
  add_executable(${OP_HOST_UT_EXE}
      test_op_host_main.cpp
  )
  target_compile_definitions(${OP_HOST_UT_EXE} PRIVATE LOG_CPP)
  target_compile_options(${OP_HOST_UT_EXE} PUBLIC -fPIE -fno-access-control)
  add_dependencies(${OP_HOST_UT_EXE} ${OPHOST_NAME}_ut)
  target_include_directories(${OP_HOST_UT_EXE} PRIVATE
      ${ASCEND_DIR}/pkg_inc
      )
  target_link_directories(${OP_HOST_UT_EXE} PRIVATE
      ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
      )
  target_link_libraries(${OP_HOST_UT_EXE} PRIVATE
      $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
      -Wl,--no-as-needed
      ${OPHOST_NAME}_ut
      ascendalog
      dl
      $<$<TARGET_EXISTS:mc2_tiling_ut_stub>:mc2_tiling_ut_stub>
      $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
      $<$<TARGET_EXISTS:opbase_infer_objs>:$<TARGET_OBJECTS:opbase_infer_objs>>
      -Wl,--as-needed
      -Wl,--whole-archive
          ${OP_TILING_MODULE_NAME}_cases
          ${OP_INFERSHAPE_MODULE_NAME}_cases
      -Wl,--no-whole-archive
      tiling_api
      -Wl,--no-as-needed
          metadef
      -Wl,--as-needed
      register
      opp_registry
      graph
      platform
      gtest
      unified_dlog
  )

  ## exec ophost ut exe
  if(ENABLE_UT_EXEC)
    if(ENABLE_ASAN)
      execute_process(
        COMMAND ${CMAKE_C_COMPILER} -print-file-name=libasan.so
        OUTPUT_VARIABLE LIBASAN_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE result
      )
      if(NOT result EQUAL 0)
        message(FATAL_ERROR "compiler not support asan, please disable asan")
      endif()

      execute_process(
        COMMAND ${CMAKE_C_COMPILER} -print-file-name=libstdc++.so
        OUTPUT_VARIABLE LIBSTDC_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE stdc_result
      )
      if(NOT stdc_result EQUAL 0)
        message(FATAL_ERROR "not find libstdc++.so, please confirm.")
      endif()

      add_custom_command(
        TARGET ${OP_HOST_UT_EXE}
        POST_BUILD
        COMMAND
          ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}
          LD_PRELOAD=${LIBASAN_PATH}:${LIBSTDC_PATH}
          ASAN_OPTIONS=detect_leaks=0 ./${OP_HOST_UT_EXE}
        COMMENT "Run ops op_host utest with asan"
        )
    else()
      add_custom_command(
        TARGET ${OP_HOST_UT_EXE}
        POST_BUILD
        COMMAND
          ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH} ./${OP_HOST_UT_EXE}
        COMMENT "Run ops op_host utest"
        )
    endif()
  endif()

  if(ENABLE_VALGRIND)
    add_custom_command(
      TARGET ${OP_HOST_UT_EXE}
      POST_BUILD
      COMMAND
        ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}
        valgrind
          --undef-value-errors=no
          --leak-check=full
          --log-file=valgrind_${OP_HOST_UT_EXE}.log
          ./${OP_HOST_UT_EXE}
      COMMENT "Run ops op_host utest by valgrind, output saved to valgrind_${OP_HOST_UT_EXE}.log"
    )
  endif()
endif()
