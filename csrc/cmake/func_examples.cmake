# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

function(add_execute_example)
    cmake_parse_arguments(EXAMPLE "" "TARGET_NAME;SCRIPT;TEST_CASE;ACLNN_FUNC" "" ${ARGN})

    # 获取需使用的 opapi 动态库绝对路径
    if (NOT EXAMPLE_ACLNN_FUNC)
        message(FATAL_ERROR "Example(${EXAMPLE_TEST_CASE}) not give aclnn func name")
    endif ()
    set(_get_py ${OPS_ADV_DIR}/cmake/scripts/examples/get_opapi_abs_path.py)
    execute_process(
            COMMAND ${HI_PYTHON} ${_get_py} "-f=${EXAMPLE_ACLNN_FUNC}"
            OUTPUT_VARIABLE OPAPI_SHARED_REL_PATH
    )
    message(STATUS "Example(${EXAMPLE_TEST_CASE}) Func(${EXAMPLE_ACLNN_FUNC}) use ${OPAPI_SHARED_REL_PATH}")
    if (NOT OPAPI_SHARED_REL_PATH)
        message(FATAL_ERROR "Example(${EXAMPLE_TEST_CASE}) can't get opapi path")
    endif ()
    get_filename_component(OPAPI_SHARED_REL_DIR ${OPAPI_SHARED_REL_PATH} DIRECTORY)

    target_link_libraries(${EXAMPLE_TARGET_NAME}
            PRIVATE
                $<BUILD_INTERFACE:intf_pub>
                ${OPAPI_SHARED_REL_PATH}
                -lascendcl
                -lnnopbase
                -lc_sec
    )
    target_compile_options(${EXAMPLE_TARGET_NAME}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>
    )
    set_target_properties(${EXAMPLE_TARGET_NAME}
            PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${EXAMPLE_TEST_CASE}
                BUILD_RPATH "${OPAPI_SHARED_REL_DIR}"
    )

    set(_execute_flag OFF)
    if ("${TESTS_EXAMPLE_OPS_TEST}" STREQUAL "")
        return()
    elseif ("ALL" IN_LIST TESTS_EXAMPLE_OPS_TEST OR "all" IN_LIST TESTS_EXAMPLE_OPS_TEST)
        set(_execute_flag ON)
    elseif ("${EXAMPLE_TARGET_NAME}" IN_LIST TESTS_EXAMPLE_OPS_TEST)
        set(_execute_flag ON)
    endif ()

    if (_execute_flag)
        add_custom_command(
                TARGET ${EXAMPLE_TARGET_NAME} POST_BUILD
                COMMAND bash ${EXAMPLE_SCRIPT} ${EXAMPLE_TARGET_NAME} $<TARGET_FILE:${EXAMPLE_TARGET_NAME}> ${EXAMPLE_TEST_CASE}
                WORKING_DIRECTORY $<TARGET_FILE_DIR:${EXAMPLE_TARGET_NAME}>
                COMMENT "Run ${EXAMPLE_TARGET_NAME}"
        )
    endif ()

    if (NOT TARGET ops_test_example)
        add_custom_target(ops_test_example)
    endif ()
    add_dependencies(ops_test_example ${EXAMPLE_TARGET_NAME})
endfunction()
