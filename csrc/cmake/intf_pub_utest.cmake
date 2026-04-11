# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# UTest 场景, 编译 Target 名称公共前缀
set(UTest_NamePrefix UTest)

# UTest 场景, 公共配置
add_library(intf_pub_utest INTERFACE)
target_compile_definitions(intf_pub_utest
        INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:_GLIBCXX_USE_CXX11_ABI=0>    # 必须设置, 以保证与 CANN 包内其他依赖库兼容
            ASCENDC_OP_TEST
            ASCENDC_OP_TEST_UT
            $<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
            $<$<CONFIG:Debug>:CFG_BUILD_DEBUG>
)
target_compile_options(intf_pub_utest
        INTERFACE
            -fPIC
            $<$<CXX_COMPILER_ID:GNU>:$<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>>
            $<$<CXX_COMPILER_ID:Clang>:$<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},10.0.0>,-fstack-protector-strong,-fstack-protector-all>>
            -g
            $<$<BOOL:${ENABLE_GCOV}>:$<$<CXX_COMPILER_ID:GNU>:--coverage -fprofile-arcs -ftest-coverage>>
            $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -fsanitize-address-use-after-scope -fsanitize=leak>
            # 在 Clang 编译器场景下 使能 -fsanitize=undefined 会默认开启基本所有的 UBSAN 检查项, 只有以下检查项不会开启
            #   float-divide-by-zero, unsigned-integer-overflow, implicit-conversion, local-bounds 及 nullability-* 类检查.
            # 故在 Clang 编译器使能 UBSAN 场景下, 需开启 -fsanitize=undefined 使能时仍未开启的对应检查项
            # 在 GNU 编译器场景下, 官方文档并未对使能 -fsanitize=undefined 时开启的默认检查项范围进行说明, 故手工开启常用基本检查项, 避免能力遗漏
            $<$<BOOL:${ENABLE_UBSAN}>:-fsanitize=undefined -fsanitize=float-divide-by-zero -fno-sanitize=alignment>
            # $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:Clang>:-fsanitize=unsigned-integer-overflow>>    # GNU 不支持这些检查项
            # $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:Clang>:$<$<VERSION_GREATER_EQUAL:${CMAKE_C_COMPILER_VERSION},10.0.0>:-fsanitize=implicit-conversion>>>    # GNU 不支持这些检查项, Clang高版本才支持这些检查项
            $<$<BOOL:${ENABLE_UBSAN}>:$<$<CXX_COMPILER_ID:GNU>:-fsanitize=shift -fsanitize=integer-divide-by-zero -fsanitize=signed-integer-overflow -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow -fsanitize=bool -fsanitize=enum -fsanitize=vptr>>
            $<$<BOOL:${ENABLE_ASAN} OR ${ENABLE_UBSAN}>:-fno-omit-frame-pointer -fsanitize-recover=all>
            -Wall -fno-common -fno-strict-aliasing
            -Wundef -Wcast-qual -Wpointer-arith -Wdate-time
            -Wfloat-equal -Wformat=2 -Wshadow
            -Wsign-compare -Wunused-macros -Wvla -Wdisabled-optimization -Wempty-body -Wignored-qualifiers
            $<$<CXX_COMPILER_ID:GNU>:-Wimplicit-fallthrough=3> -Wtype-limits -Wshift-negative-value -Wswitch-default
            -Wframe-larger-than=67108864    # 67108864=65536 * 1024, 兼容 ASAN场景对栈的额外消耗
            -Woverloaded-virtual
            -Wnon-virtual-dtor $<$<CXX_COMPILER_ID:GNU>:-Wshift-overflow=2> -Wshift-count-overflow
            -Wwrite-strings -Wmissing-format-attribute -Wformat-nonliteral
            -Wdelete-non-virtual-dtor $<$<CXX_COMPILER_ID:GNU>:-Wduplicated-cond>
            $<$<CXX_COMPILER_ID:GNU>:-Wtrampolines>
            $<$<CXX_COMPILER_ID:GNU>:-Wsized-deallocation>
            $<$<CXX_COMPILER_ID:GNU>:-Wlogical-op>
            $<$<CXX_COMPILER_ID:GNU>:-Wsuggest-attribute=format>
            $<$<COMPILE_LANGUAGE:C>:-Wnested-externs>
            $<$<CXX_COMPILER_ID:GNU>:-Wduplicated-branches>
            $<$<CXX_COMPILER_ID:GNU>:-Wformat-signedness>
            $<$<CXX_COMPILER_ID:GNU>:-Wreturn-local-addr> -Wextra
            -Wredundant-decls -Wfloat-conversion
            $<$<CXX_COMPILER_ID:Clang>:-Wno-tautological-unsigned-enum-zero-compare>
            $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs -ftest-coverage>
)
target_include_directories(intf_pub_utest
        INTERFACE
            ${ASCEND_CANN_PACKAGE_PATH}/include
            ${ASCEND_CANN_PACKAGE_PATH}/include/external
            ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
            ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
            ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
            ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)
if (TESTS_UT_OPS_TEST_CI_PR)
    target_compile_definitions(intf_pub_utest
            INTERFACE
                TESTS_UT_OPS_TEST_CI_PR
    )
endif ()
target_link_directories(intf_pub_utest
        INTERFACE
            ${ASCEND_CANN_PACKAGE_PATH}/lib64
            ${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64
            ${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64/stub
)
target_link_libraries(intf_pub_utest
        INTERFACE
            $<$<BOOL:${ENABLE_GCOV}>:$<$<CXX_COMPILER_ID:GNU>:gcov>>
            pthread
)
target_link_options(intf_pub_utest
        INTERFACE
            $<$<BOOL:${ENABLE_GCOV}>:$<$<CXX_COMPILER_ID:GNU>:-fprofile-arcs -ftest-coverage>>
            $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address>
            $<$<BOOL:${ENABLE_UBSAN}>:-fsanitize=undefined>
)
