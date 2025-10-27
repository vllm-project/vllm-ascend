# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

########################################################################################################################
# 预定义变量
########################################################################################################################

# 所使用的产品类型
set(OPS_ADV_UTEST_OPS_TEST_ASCEND_PRODUCT_TYPE ascend910B1)

########################################################################################################################
# 环境检查
########################################################################################################################

find_package(GTest CONFIG)
if (NOT ${GTest_FOUND})
    get_filename_component(ASCEND_3RD_LIB_PATH "$ENV{ASCEND_3RD_LIB_PATH}" REALPATH)
    if (EXISTS "${ASCEND_3RD_LIB_PATH}/cmake/modules")
        list(APPEND CMAKE_MODULE_PATH ${ASCEND_3RD_LIB_PATH}/cmake/modules)
    endif ()
    if (EXISTS "${ASCEND_3RD_LIB_PATH}/gtest/lib/cmake/GTest")
        list(APPEND CMAKE_PREFIX_PATH ${ASCEND_3RD_LIB_PATH}/gtest/lib/cmake/GTest)
        find_package(GTest CONFIG REQUIRED)
    endif ()
endif ()
if (NOT ${GTest_FOUND})
    message(FATAL_ERROR "Can't find any googletest.")
endif ()
message(STATUS "Use googletest from ${GTest_DIR}")
get_target_property(GTEST_GTEST_INC GTest::gtest INTERFACE_INCLUDE_DIRECTORIES)

list(APPEND CMAKE_PREFIX_PATH ${ASCEND_CANN_PACKAGE_PATH}/toolkit/tools/tikicpulib/lib/cmake)
find_package(tikicpulib REQUIRED)

# ASAN / UBSAN 场景随编译执行用例场景下, 将相关检查在编译前执行, 避免出现编译完成后又无法执行的情况, 影响使用体验.
if (ENABLE_ASAN OR ENABLE_UBSAN)
    set(SAN_LD_PRELOAD "LD_PRELOAD=")
    if (ENABLE_ASAN)
        # libasan.so
        execute_process(COMMAND ${CMAKE_C_COMPILER} --print-file-name=libasan.so
                        RESULT_VARIABLE _RST
                        OUTPUT_VARIABLE ASAN_SHARED_PATH)
        if (_RST)
            message(FATAL_ERROR "Can't get libasan.so path with ${CMAKE_C_COMPILER}")
        endif ()
        get_filename_component(ASAN_SHARED_PATH "${ASAN_SHARED_PATH}" DIRECTORY)
        get_filename_component(ASAN_SHARED_PATH "${ASAN_SHARED_PATH}/libasan.so" REALPATH)
        if (NOT EXISTS ${ASAN_SHARED_PATH})
            message(FATAL_ERROR "ASAN_SHARED_PATH=${ASAN_SHARED_PATH} not exist.")
        endif ()
        set(SAN_LD_PRELOAD "${SAN_LD_PRELOAD}:${ASAN_SHARED_PATH}")
    endif ()
    if (ENABLE_UBSAN)
        # libubsan.so
        execute_process(COMMAND ${CMAKE_C_COMPILER} --print-file-name=libubsan.so
                        RESULT_VARIABLE _RST
                        OUTPUT_VARIABLE UBSAN_SHARED_PATH)
        if (_RST)
            message(FATAL_ERROR "Can't get libubsan.so path with ${CMAKE_C_COMPILER}")
        endif ()
        get_filename_component(UBSAN_SHARED_PATH "${UBSAN_SHARED_PATH}" DIRECTORY)
        get_filename_component(UBSAN_SHARED_PATH "${UBSAN_SHARED_PATH}/libubsan.so" REALPATH)
        if (NOT EXISTS ${UBSAN_SHARED_PATH})
            message(FATAL_ERROR "UBSAN_SHARED_PATH=${UBSAN_SHARED_PATH} not exist.")
        endif ()
        set(SAN_LD_PRELOAD "${SAN_LD_PRELOAD}:${UBSAN_SHARED_PATH}")
    endif ()
    # libstdc++.so
    execute_process(COMMAND ${CMAKE_C_COMPILER} --print-file-name=libstdc++.so
                    RESULT_VARIABLE _RST
                    OUTPUT_VARIABLE STDC_SHARED_PATH)
    if (_RST)
        message(FATAL_ERROR "Can't get libstdc++.so path with ${CMAKE_C_COMPILER}")
    endif ()
    get_filename_component(STDC_SHARED_PATH "${STDC_SHARED_PATH}" DIRECTORY)
    get_filename_component(STDC_SHARED_PATH "${STDC_SHARED_PATH}/libstdc++.so" REALPATH)
    if (NOT EXISTS ${STDC_SHARED_PATH})
        message(FATAL_ERROR "STDC_SHARED_PATH=${STDC_SHARED_PATH} not exist.")
    endif ()
    set(SAN_LD_PRELOAD "${SAN_LD_PRELOAD}:${STDC_SHARED_PATH}")
endif ()


########################################################################################################################
# 公共配置
########################################################################################################################

# 开关类
option(TESTS_UT_OPS_TEST_CI_PR         "Build UTest in push request scene."  OFF)
