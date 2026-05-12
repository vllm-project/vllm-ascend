# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

function(register_op_name op_name)
  get_property(op_list GLOBAL PROPERTY ALL_OP_LIST)
  if(NOT op_list)
    set(op_list "")
  endif()

  list(FIND op_list "${op_name}" idx)
  if(idx EQUAL -1)
    list(APPEND op_list "${op_name}")
    set_property(GLOBAL PROPERTY ALL_OP_LIST "${op_list}")
  endif()
endfunction()

if(UT_TEST_ALL OR OP_HOST_UT)
  set(OP_TILING_MODULE_NAME
      ${PKG_NAME}_op_tiling_ut
      CACHE STRING "op_tiling ut module name" FORCE
    )
  set(OP_INFERSHAPE_MODULE_NAME
      ${PKG_NAME}_op_infershape_ut
      CACHE STRING "op_infershape ut module name" FORCE
    )
  function(add_optiling_ut_modules OP_TILING_MODULE_NAME)
    # add optiling ut common object: transformer_op_tiling_ut_common_obj
    add_library(${OP_TILING_MODULE_NAME}_common_obj OBJECT)
    file(GLOB OP_TILING_UT_COMMON_SRC ${UT_COMMON_INC}/tiling_context_faker.cpp
         ${UT_COMMON_INC}/tiling_case_executor.cpp
      )
    target_sources(${OP_TILING_MODULE_NAME}_common_obj PRIVATE ${OP_TILING_UT_COMMON_SRC})
    target_include_directories(
      ${OP_TILING_MODULE_NAME}_common_obj PRIVATE ${JSON_INCLUDE_DIR} ${GTEST_INCLUDE}
                                                  ${ASCEND_DIR}/include/base/context_builder ${ASCEND_DIR}/pkg_inc
                                                  ${ASCEND_DIR}/include/ascendc/basic_api
      )
    target_link_libraries(
      ${OP_TILING_MODULE_NAME}_common_obj PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> json gtest c_sec
      )

    # add optiling ut cases object: transformer_op_tiling_ut_cases_obj
    if(NOT TARGET ${OP_TILING_MODULE_NAME}_cases_obj)
      add_library(${OP_TILING_MODULE_NAME}_cases_obj OBJECT ${UT_PATH}/empty.cpp)
    endif()
    target_include_directories(
      ${OP_TILING_MODULE_NAME}_cases_obj PRIVATE ${UT_COMMON_INC} ${GTEST_INCLUDE} ${ASCEND_DIR}/include
                                                 ${ASCEND_DIR}/include/base/context_builder ${OP_TILING_INCLUDE}
                                                 $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef/common/util>>
                                                 ${ASCEND_DIR}/include/ascendc/basic_api
      )
    target_compile_definitions(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE
            LOG_CPP
      )
    target_link_libraries(${OP_TILING_MODULE_NAME}_cases_obj 
      PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
      $<$<BOOL:${dlog_FOUND}>:$<BUILD_INTERFACE:dlog_headers>>
      gtest)

    # add op tiling ut cases static lib: libtransformer_op_tiling_ut_cases.a
    add_library(${OP_TILING_MODULE_NAME}_cases STATIC)
    target_link_libraries(
      ${OP_TILING_MODULE_NAME}_cases PRIVATE ${OP_TILING_MODULE_NAME}_common_obj ${OP_TILING_MODULE_NAME}_cases_obj
      )
  endfunction()

  function(add_infershape_ut_modules OP_INFERSHAPE_MODULE_NAME)
    # add opinfershape ut common object: transformer_op_infershape_ut_common_obj
    add_library(${OP_INFERSHAPE_MODULE_NAME}_common_obj OBJECT)
    file(GLOB OP_INFERSHAPE_UT_COMMON_SRC 
      ${UT_COMMON_INC}/infer_shape_context_faker.cpp
      ${UT_COMMON_INC}/infer_shape_case_executor.cpp
      ${UT_COMMON_INC}/infer_datatype_context_faker.cpp
      ${UT_COMMON_INC}/infer_shaperange_context_faker.cpp
    )
    target_sources(${OP_INFERSHAPE_MODULE_NAME}_common_obj PRIVATE ${OP_INFERSHAPE_UT_COMMON_SRC})
    target_include_directories(
      ${OP_INFERSHAPE_MODULE_NAME}_common_obj PRIVATE ${ASCEND_DIR}/include/base/context_builder
      ${ASCEND_DIR}/pkg_inc
      ${ASCEND_DIR}/include/ascendc/basic_api
      )
    target_link_libraries(
      ${OP_INFERSHAPE_MODULE_NAME}_common_obj PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> json gtest c_sec
      )

    # add opinfershape ut cases object: transformer_op_infershape_ut_cases_obj
    if(NOT TARGET ${OP_INFERSHAPE_MODULE_NAME}_cases_obj)
      add_library(${OP_INFERSHAPE_MODULE_NAME}_cases_obj OBJECT ${UT_PATH}/empty.cpp)
    endif()
    target_include_directories(
      ${OP_INFERSHAPE_MODULE_NAME}_cases_obj PRIVATE ${UT_COMMON_INC} ${GTEST_INCLUDE} ${ASCEND_DIR}/include
                                                     ${ASCEND_DIR}/pkg_inc ${ASCEND_DIR}/include/base/context_builder
                                                     ${OPBASE_INC_DIRS}
                                                     ${ASCEND_DIR}/include/ascendc/basic_api
      )
    target_link_libraries(
      ${OP_INFERSHAPE_MODULE_NAME}_cases_obj 
        PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        $<$<BOOL:${dlog_FOUND}>:$<BUILD_INTERFACE:dlog_headers>>
        gtest
      )

    # add op infershape ut cases static lib: libtransformer_op_infershape_ut_cases.a
    add_library(${OP_INFERSHAPE_MODULE_NAME}_cases STATIC)
    target_link_libraries(
      ${OP_INFERSHAPE_MODULE_NAME}_cases PRIVATE ${OP_INFERSHAPE_MODULE_NAME}_common_obj
                                                 ${OP_INFERSHAPE_MODULE_NAME}_cases_obj
      )
  endfunction()
endif()

if(UT_TEST_ALL OR OP_API_UT)
  set(OP_API_MODULE_NAME
      ${PKG_NAME}_op_api_ut
      CACHE STRING "op_api ut module name" FORCE
    )
  function(add_opapi_ut_modules OP_API_MODULE_NAME)
    # add opapi ut L2 obj
    if(NOT TARGET ${OP_API_MODULE_NAME}_cases_obj)
      add_library(${OP_API_MODULE_NAME}_cases_obj OBJECT)
    endif()
    target_sources(${OP_API_MODULE_NAME}_cases_obj PRIVATE
                      ${UT_PATH}/op_api/stub/opdev/platform.cpp
                      ${UT_PATH}/op_api/stub/opdev/nnopbase.cpp)
    target_include_directories(
      ${OP_API_MODULE_NAME}_cases_obj
      PRIVATE ${JSON_INCLUDE_DIR} ${HI_PYTHON_INC_TEMP} ${UT_PATH}/op_api/stub ${OP_API_UT_COMMON_INC}
              ${ASCEND_DIR}/include ${ASCEND_DIR}/include/aclnn ${ASCEND_DIR}/include/aclnnop
              ${OPAPI_INCLUDE}
              ${ASCEND_DIR}/pkg_inc 
              ${ASCEND_DIR}/include/ascendc/basic_api 
              ${ASCEND_CANN_PACKAGE_PATH}/runtime/pkg_inc
      )
    target_link_libraries(${OP_API_MODULE_NAME}_cases_obj 
      PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
      $<$<BOOL:${dlog_FOUND}>:$<BUILD_INTERFACE:dlog_headers>>
      gtest)
  endfunction()
endif()

if(UT_TEST_ALL OR OP_KERNEL_UT)
  set(OP_KERNEL_MODULE_NAME
      ${PKG_NAME}_op_kernel_ut
      CACHE STRING "op_kernel ut module name" FORCE
    )
  function(add_opkernel_ut_modules OP_KERNEL_MODULE_NAME)
    # add opkernel ut common object: transformer_op_kernel_ut_common_obj
    add_library(${OP_KERNEL_MODULE_NAME}_common_obj OBJECT)
    file(GLOB OP_KERNEL_UT_COMMON_SRC ${UT_COMMON_INC}/tiling_context_faker.cpp
         ${UT_COMMON_INC}/tiling_case_executor.cpp ${PROJECT_SOURCE_DIR}/tests/ut/framework_normal/op_kernel/data_utils.cpp
      )
    target_sources(${OP_KERNEL_MODULE_NAME}_common_obj PRIVATE ${OP_KERNEL_UT_COMMON_SRC})
    target_include_directories(
      ${OP_KERNEL_MODULE_NAME}_common_obj PRIVATE ${JSON_INCLUDE_DIR} ${GTEST_INCLUDE}
                                                  ${OPBASE_INC_DIRS} ${ASCEND_DIR}/include/base
                                                  ${ASCEND_DIR}/include/base/context_builder
                                                  ${ASCEND_DIR}/include/experiment
                                                  ${ASCEND_DIR}/include/experiment/metadef/common/util
      )
    target_link_libraries(
      ${OP_KERNEL_MODULE_NAME}_common_obj PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> json gtest c_sec
      )

    foreach(socVersion ${fastOpTestSocVersions})
      # add op kernel ut cases obj: transformer_op_tiling_ut_${socVersion}_cases
      if(NOT TARGET ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj)
        add_library(${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj OBJECT)
      endif()
      target_link_libraries(${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj PRIVATE gcov)

      # add op kernel ut cases dynamic lib: libtransformer_op_tiling_ut_${socVersion}_cases.so
      add_library(
        ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases SHARED
        $<TARGET_OBJECTS:${OP_KERNEL_MODULE_NAME}_common_obj>
        $<TARGET_OBJECTS:${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj>
        )
      target_link_libraries(
        ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases
        PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> ${OP_KERNEL_MODULE_NAME}_common_obj
                ${OP_KERNEL_MODULE_NAME}_${socVersion}_cases_obj
        )
    endforeach()
  endfunction()
endif()

if(UT_TEST_ALL
   OR OP_HOST_UT
   OR OP_API_UT
  )
  function(add_modules_ut_sources)
    set(options OPTION_RESERVED)
    set(oneValueArgs UT_NAME MODE DIR)
    set(multiValueArgs MULIT_RESERVED)
    cmake_parse_arguments(MODULE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    get_filename_component(ARCH_NAME ${MODULE_DIR} NAME)
    if(${ARCH_NAME} STREQUAL "op_host")
      # tiling.cpp, infershape.cpp under op_host/
      if("${MODULE_UT_NAME}" STREQUAL "${OP_TILING_MODULE_NAME}")
        get_filename_component(UT_DIR ${MODULE_DIR} DIRECTORY)
        get_filename_component(TESTS_DIR ${UT_DIR} DIRECTORY)
        get_filename_component(OP_NAME_DIR ${TESTS_DIR} DIRECTORY)
        get_filename_component(OP_NAME ${OP_NAME_DIR} NAME)
        list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
        # if "--ops" is not NULL, opName not include, jump over. if "--ops" is NULL, include all.
        if(NOT "${ASCEND_OP_NAME}" STREQUAL "ALL" AND INDEX EQUAL -1)
          return()
        endif()

        # add op_tiling ut common object: transformer_op_tiling_ut_cases_obj
        if(NOT TARGET ${MODULE_UT_NAME}_cases_obj)
          add_library(${MODULE_UT_NAME}_cases_obj OBJECT)
        endif()
        file(GLOB OPHOST_TILING_CASES_SRC ${MODULE_DIR}/test_*_tiling.cpp)
        target_sources(${MODULE_UT_NAME}_cases_obj ${MODULE_MODE} ${OPHOST_TILING_CASES_SRC})
      endif()

      if("${MODULE_UT_NAME}" STREQUAL "${OP_INFERSHAPE_MODULE_NAME}")
        get_filename_component(UT_DIR ${MODULE_DIR} DIRECTORY)
        get_filename_component(TESTS_DIR ${UT_DIR} DIRECTORY)
        get_filename_component(OP_NAME_DIR ${TESTS_DIR} DIRECTORY)
        get_filename_component(OP_NAME ${OP_NAME_DIR} NAME)
        list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
        # if "--ops" is not NULL, opName not include, jump over. if "--ops" is NULL, include all.
        if(NOT "${ASCEND_OP_NAME}" STREQUAL "ALL" AND INDEX EQUAL -1)
          return()
        endif()

        # add op_infershape ut common object: transformer_op_infershape_ut_cases_obj
        if(NOT TARGET ${MODULE_UT_NAME}_cases_obj)
          add_library(${MODULE_UT_NAME}_cases_obj OBJECT)
        endif()
        file(GLOB OPHOST_INFERSHAPE_CASES_SRC ${MODULE_DIR}/test_*_infershape.cpp)
        target_sources(${MODULE_UT_NAME}_cases_obj ${MODULE_MODE} ${OPHOST_INFERSHAPE_CASES_SRC})
      endif()
    endif()

    # tiling.cpp, infershape.cpp under arch*/
    if(${ARCH_NAME} MATCHES "^arch")
      # if "--soc" not include current soc, jump over
      list(FIND ARCH_DIRECTORY ${ARCH_NAME} INDEX)
      if(INDEX EQUAL -1)
        return()
      endif()

      if("${MODULE_UT_NAME}" STREQUAL "${OP_TILING_MODULE_NAME}")
        get_filename_component(UT_TYPE_DIR ${MODULE_DIR} DIRECTORY)
        get_filename_component(UT_DIR ${UT_TYPE_DIR} DIRECTORY)
        get_filename_component(TESTS_DIR ${UT_DIR} DIRECTORY)
        get_filename_component(OP_NAME_DIR ${TESTS_DIR} DIRECTORY)
        get_filename_component(OP_NAME ${OP_NAME_DIR} NAME)
        list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
        # if "--ops" is not NULL, opName not include, jump over. if "--ops" is NULL, include all.
        if(NOT "${ASCEND_OP_NAME}" STREQUAL "ALL" AND INDEX EQUAL -1)
          return()
        endif()

        # add op_tiling ut common object: transformer_op_tiling_ut_cases_obj
        if(NOT TARGET ${MODULE_UT_NAME}_cases_obj)
          add_library(${MODULE_UT_NAME}_cases_obj OBJECT)
        endif()
        file(GLOB OPHOST_TILING_CASES_SRC ${MODULE_DIR}/test_*_tiling.cpp)
        target_sources(${MODULE_UT_NAME}_cases_obj ${MODULE_MODE} ${OPHOST_TILING_CASES_SRC})
      endif()

      if("${MODULE_UT_NAME}" STREQUAL "${OP_INFERSHAPE_MODULE_NAME}")
        get_filename_component(UT_TYPE_DIR ${MODULE_DIR} DIRECTORY)
        get_filename_component(UT_DIR ${UT_TYPE_DIR} DIRECTORY)
        get_filename_component(TESTS_DIR ${UT_DIR} DIRECTORY)
        get_filename_component(OP_NAME_DIR ${TESTS_DIR} DIRECTORY)
        get_filename_component(OP_NAME ${OP_NAME_DIR} NAME)

        list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
        # if "--ops" is not NULL, opName not include, jump over. if "--ops" is NULL, include all.
        if(NOT "${ASCEND_OP_NAME}" STREQUAL "ALL" AND INDEX EQUAL -1)
          return()
        endif()

        # add op_infershape ut common object: transformer_op_infershape_ut_cases_obj
        if(NOT TARGET ${MODULE_UT_NAME}_cases_obj)
          add_library(${MODULE_UT_NAME}_cases_obj OBJECT)
        endif()
        file(GLOB OPHOST_INFERSHAPE_CASES_SRC ${MODULE_DIR}/test_*_infershape.cpp)
        target_sources(${MODULE_UT_NAME}_cases_obj ${MODULE_MODE} ${OPHOST_INFERSHAPE_CASES_SRC})
      endif()
    endif()

    # op_api ut
    if("${MODULE_UT_NAME}" STREQUAL "${OP_API_MODULE_NAME}")
      get_filename_component(OP_HOST_DIR ${MODULE_DIR} DIRECTORY)
      get_filename_component(OP_HOST_NAME ${OP_HOST_DIR} NAME)
      if("${OP_HOST_NAME}" STREQUAL "op_host")
        get_filename_component(UT_DIR ${OP_HOST_DIR} DIRECTORY)
      else()
        get_filename_component(UT_DIR ${MODULE_DIR} DIRECTORY)
      endif()
      get_filename_component(TESTS_DIR ${UT_DIR} DIRECTORY)
      get_filename_component(OP_NAME_DIR ${TESTS_DIR} DIRECTORY)
      get_filename_component(OP_NAME ${OP_NAME_DIR} NAME)
      list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
      # if "--ops" is not NULL, opName not include, jump over. if "--ops" is NULL, include all.
      if(NOT "${ASCEND_OP_NAME}" STREQUAL "ALL" AND INDEX EQUAL -1)
        return()
      endif()

      # add op_api ut common object: transformer_op_api_ut_cases_obj
      register_op_name(${OP_NAME})
      if(NOT TARGET ${MODULE_UT_NAME}_cases_obj)
        add_library(${MODULE_UT_NAME}_cases_obj OBJECT)
      endif()
      file(GLOB OPAPI_CASES_SRC ${MODULE_DIR}/test_aclnn_*.cpp)
      target_sources(${MODULE_UT_NAME}_cases_obj ${MODULE_MODE} ${OPAPI_CASES_SRC})
    endif()
  endfunction()
endif()

if(UT_TEST_ALL OR OP_KERNEL_UT)
  set(fastOpTestSocVersions
      ""
      CACHE STRING "fastOp Test SocVersions"
    )
  function(AddOpTestCase opName supportedSocVersion otherCompileOptions tilingSrcFiles)
    get_filename_component(ARCH_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    if(${ARCH_NAME} STREQUAL "op_kernel")
      get_filename_component(UT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    elseif(${ARCH_NAME} MATCHES "^arch")
      get_filename_component(UT_TYPE_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
      get_filename_component(UT_DIR ${UT_TYPE_DIR} DIRECTORY)
    endif()
    get_filename_component(TESTS_DIR ${UT_DIR} DIRECTORY)
    get_filename_component(OP_NAME_DIR ${TESTS_DIR} DIRECTORY)
    get_filename_component(OP_NAME ${OP_NAME_DIR} NAME)
    list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
    # if "--ops" is not NULL, opName not include, jump over. if "--ops" is NULL, include all.
    if(NOT "${ASCEND_OP_NAME}" STREQUAL "ALL" AND INDEX EQUAL -1)
      return()
    endif()

    # find kernel file
    if(ARGN)
      list(GET ARGN 0 KernelFile)
    else()
      file(GLOB KernelFile "${PROJECT_SOURCE_DIR}/*/${opName}/op_kernel/${opName}.cpp")
    endif()
    
    # find case file
    get_filename_component(ARCH_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    # arch35
    if(${ARCH_NAME} STREQUAL "arch35")
      list(FIND ARCH_DIRECTORY ${ARCH_NAME} INDEX)
      if(NOT INDEX EQUAL -1)
        file(GLOB OPKERNEL_CASES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_${opName}*.cpp)
      else()
        return()
      endif()    
    # op_kernel/arch20/arch22/arch32/arch38
    else()
      list(FIND ARCH_DIRECTORY "arch35" INDEX)
      if(INDEX EQUAL -1)
        file(GLOB OPKERNEL_CASES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_${opName}*.cpp)
      else()
        return()
      endif()
    endif()

    # standardize opType
    set(opType "")
    string(REPLACE "_" ";" opTypeTemp "${opName}")
    foreach(word IN LISTS opTypeTemp)
      string(SUBSTRING "${word}" 0 1 firstLetter)
      string(SUBSTRING "${word}" 1 -1 restOfWord)
      string(TOUPPER "${firstLetter}" firstLetter)
      string(TOLOWER "${restOfWord}" restOfWord)
      set(opType "${opType}${firstLetter}${restOfWord}")
    endforeach()

    # standardize tiling files
    string(REPLACE "," ";" tilingSrc "${tilingSrcFiles}")

    foreach(oriSocVersion ${supportedSocVersion})
      # standardize socVersion
      string(REPLACE "ascend" "Ascend" socVersion "${oriSocVersion}")

      # add tiling tmp so: ${opName}_${socVersion}_tiling_tmp.so
      add_library(${opName}_${socVersion}_tiling_tmp SHARED ${tilingSrc} $<TARGET_OBJECTS:${COMMON_NAME}_obj>)
      target_include_directories(
        ${opName}_${socVersion}_tiling_tmp
        PRIVATE ${OPBASE_INC_DIRS} ${ASCEND_DIR}/include/base 
                ${PROJECT_SOURCE_DIR}/common/include
                ${ASCEND_DIR}/include/experiment 
                ${ASCEND_DIR}/include/experiment/metadef/common/util
                ${ASCEND_DIR}/include
        )
      target_compile_definitions(${opName}_${socVersion}_tiling_tmp PRIVATE LOG_CPP _GLIBCXX_USE_CXX11_ABI=0)
      target_link_libraries(
        ${opName}_${socVersion}_tiling_tmp
        PRIVATE -Wl,--no-as-needed $<$<TARGET_EXISTS:opsbase>:opsbase> -Wl,--as-needed -Wl,--whole-archive tiling_api
                -Wl,--no-whole-archive gcov
                $<$<BOOL:${dlog_FOUND}>:$<BUILD_INTERFACE:dlog_headers>>
        )

      # gen ascendc tiling head files
      set(tilingFile ${CMAKE_CURRENT_BINARY_DIR}/${opName}_tiling_data.h)
      if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${opName}_tiling.h")
        set(compileOptions -include "${CMAKE_CURRENT_SOURCE_DIR}/${opName}_tiling.h")
      else()
        set(compileOptions -include ${tilingFile})
      endif()
      set(CUSTOM_TILING_DATA_KEYS "")
      string(REGEX MATCH "-DUT_CUSTOM_TILING_DATA_KEYS=([^ ]+)" matchedPart "${otherCompileOptions}")
      if(CMAKE_MATCH_1)
        set(CUSTOM_TILING_DATA_KEYS ${CMAKE_MATCH_1})
        string(REGEX REPLACE "-DUT_CUSTOM_TILING_DATA_KEYS=[^ ]+" "" modifiedString ${otherCompileOptions})
        set(otherCompileOptions ${modifiedString})
      endif()
      string(REPLACE " " ";" options "${otherCompileOptions}")
      foreach(option IN LISTS options)
        set(compileOptions ${compileOptions} ${option})
      endforeach()
      message("compileOptions: ${compileOptions}")
      set(gen_tiling_head_file ${OPS_TRANSFORMER_DIR}/tests/ut/framework_normal/op_kernel/scripts/gen_tiling_head_file.sh)
      set(gen_tiling_so_path ${CMAKE_CURRENT_BINARY_DIR}/lib${opName}_${socVersion}_tiling_tmp.so)
      set(gen_tiling_head_tag ${opName}_${socVersion}_gen_head)
      set(gen_cmd "bash ${gen_tiling_head_file} ${opType} ${opName} ${gen_tiling_so_path} ${CUSTOM_TILING_DATA_KEYS}")
      message("gen tiling head file to ${tilingFile}, command:")
      message("${gen_cmd}")
      add_custom_command(
        OUTPUT ${tilingFile}
        COMMAND rm -f ${tilingFile}
        COMMAND bash -c ${gen_cmd}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${opName}_${socVersion}_tiling_tmp
        )
      add_custom_target(${gen_tiling_head_tag} ALL DEPENDS ${tilingFile})

      # add object: ${opName}_${socVersion}_cases_obj
      add_library(${opName}_${socVersion}_cases_obj OBJECT ${KernelFile} ${OPKERNEL_CASES_SRC})
      add_dependencies(${opName}_${socVersion}_cases_obj ${gen_tiling_head_tag})
      target_compile_options(
        ${opName}_${socVersion}_cases_obj PRIVATE -g ${compileOptions} -DUT_SOC_VERSION="${socVersion}"
        )
      target_include_directories(
        ${opName}_${socVersion}_cases_obj
        PRIVATE ${ASCEND_DIR}/include/base/context_builder ${PROJECT_SOURCE_DIR}/tests/ut/framework_normal/op_kernel
                ${PROJECT_SOURCE_DIR}/tests/ut/framework_normal/common
                ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/impl/basic_api
                ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc
                ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include
                ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/basic_api
                ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/adv_api
                ${ASCEND_DIR}/${SYSTEM_PREFIX}/include/ascendc/highlevel_api
        )
      target_link_libraries(
        ${opName}_${socVersion}_cases_obj PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17> tikicpulib::${socVersion}
                                                  gtest
        )

      # add object: transformer_op_kernel_ut_${oriSocVersion}_cases_obj
      if(NOT TARGET ${OP_KERNEL_MODULE_NAME}_${oriSocVersion}_cases_obj)
        add_library(
          ${OP_KERNEL_MODULE_NAME}_${oriSocVersion}_cases_obj OBJECT
          $<TARGET_OBJECTS:${opName}_${socVersion}_cases_obj>
          )
      endif()
      target_link_libraries(
        ${OP_KERNEL_MODULE_NAME}_${oriSocVersion}_cases_obj PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
                                                                    $<TARGET_OBJECTS:${opName}_${socVersion}_cases_obj>
        )

      list(FIND fastOpTestSocVersions "${oriSocVersion}" index)
      if(index EQUAL -1)
        set(fastOpTestSocVersions
            ${fastOpTestSocVersions} ${oriSocVersion}
            CACHE STRING "fastOp Test SocVersions" FORCE
          )
      endif()
    endforeach()
  endfunction()
endif()

if(UT_TEST_ALL OR OP_KERNEL_AICPU_UT)
  include(${PROJECT_SOURCE_DIR}/cmake/third_party/gtest.cmake)
  function(AddAicpuOpTestCase opName)
    get_filename_component(UT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    get_filename_component(OP_NAME ${UT_DIR} NAME)
    list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
    # if "--ops" is not NULL, opName not include, jump over. if "--ops" is NULL, include all.
    if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND INDEX EQUAL -1)
      return()
    endif()

    ## find kernel file
    file(GLOB KernelFile "${PROJECT_SOURCE_DIR}/*/${opName}/op_kernel_aicpu/${opName}_aicpu.cpp")

    ## add object: ${opName}_cases_obj
    file(GLOB OPKERNEL_CASES_SRC ${UT_DIR}/tests/ut/op_kernel_aicpu/test_${opName}*.cpp)
    
    message(STATUS "aicpu kernel info: ${opName}, ${KernelFile}, ${OPKERNEL_CASES_SRC}")

    add_library(${opName}_cases_obj OBJECT
            ${KernelFile}
            ${OPKERNEL_CASES_SRC}
            )
    target_compile_options(${opName}_cases_obj PRIVATE 
            -g
            )

    ## add op_kernel_aicpu test header file search path, so that header files can be referenced based on relative path
    target_include_directories(${opName}_cases_obj PRIVATE
            ${AICPU_INCLUDE}
            ${OPBASE_INC_DIRS}
            ${AICPU_INC_DIRS}
            ${PROJECT_SOURCE_DIR}/tests/ut/op_kernel_aicpu
            )
    target_link_libraries(${opName}_cases_obj PRIVATE
            $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
            -ldl
            gtest
            c_sec
            Eigen3::EigenMath
            $<$<TARGET_EXISTS:opsbase>:opsbase>
            )

    ## add object: math_op_kernel_ut_cases_obj
    if(NOT TARGET ${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj)
      add_library(
        ${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj OBJECT
        $<TARGET_OBJECTS:${opName}_cases_obj>
        )
    else()
      target_sources(${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj PRIVATE $<TARGET_OBJECTS:${opName}_cases_obj>)
    endif()

    target_link_libraries(${AICPU_OP_KERNEL_MODULE_NAME}_cases_obj PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan>
            $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
            -ldl
            $<TARGET_OBJECTS:${opName}_cases_obj>
            gtest
            c_sec
            Eigen3::EigenMath
      $<$<TARGET_EXISTS:opsbase>:opsbase>
            )
  endfunction()
endif()