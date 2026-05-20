# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Custom 包场景, Host 侧各 Target 公共编译配置
# 注意: 为保证与 built-in 包编译流程兼容, intf_pub 名称不可变更
add_library(intf_pub INTERFACE)
target_include_directories(intf_pub
        INTERFACE
            ${ASCEND_CANN_PACKAGE_PATH}/include
            ${ASCEND_CANN_PACKAGE_PATH}/include/external
            ${ASCEND_CANN_PACKAGE_PATH}/include/platform
            ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
            ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)
target_link_directories(intf_pub
        INTERFACE
            ${ASCEND_CANN_PACKAGE_PATH}/lib64
)
target_compile_options(intf_pub
        INTERFACE
            -fPIC
            -O2
            -Wall -Wundef -Wcast-qual -Wpointer-arith -Wdate-time
            -Wfloat-equal -Wformat=2 -Wshadow
            -Wsign-compare -Wunused-macros -Wvla -Wdisabled-optimization -Wempty-body -Wignored-qualifiers
            -Wimplicit-fallthrough=3 -Wtype-limits -Wshift-negative-value -Wswitch-default
            -Wframe-larger-than=98304 -Woverloaded-virtual
            -Wnon-virtual-dtor -Wshift-overflow=2 -Wshift-count-overflow
            -Wwrite-strings -Wmissing-format-attribute -Wformat-nonliteral
            -Wdelete-non-virtual-dtor -Wduplicated-cond
            -Wtrampolines -Wsized-deallocation -Wlogical-op -Wsuggest-attribute=format
            -Wduplicated-branches
            -Wformat-signedness
            -Wreturn-local-addr -Wextra
            -Wredundant-decls -Wfloat-conversion
            -Wno-write-strings -Wall -Wno-dangling-else -Wno-comment -Wno-conversion-null -Wno-return-type
            -Wno-unknown-pragmas -Wno-sign-compare
            -Wno-error=undef
            -Wno-error=comment
            -Wno-error=conversion-null
            -Wno-error=dangling-else
            -Wno-error=return-type
            -Wno-error=shadow
            -Wno-error=sign-compare
            -Wno-error=unknown-pragmas
            -Wno-error=unused-parameter
            -Wno-error=cast-qual
            -Wno-error=format=
            -Wno-error=maybe-uninitialized
            -Wno-error=missing-field-initializers
            -Wno-error=redundant-decls
            -Wno-error=unused-variable
            $<$<COMPILE_LANGUAGE:C>:-Wnested-externs>
            $<$<CONFIG:Debug>:-g>
            $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
            $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs -ftest-coverage>
)
target_compile_definitions(intf_pub
        INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:_GLIBCXX_USE_CXX11_ABI=0>     # 必须设置, 以保证与 CANN 包内其他依赖库兼容
            $<$<CONFIG:Release>:_FORTIFY_SOURCE=2>
)
target_link_options(intf_pub
        INTERFACE
            $<$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>:-pie>
            $<$<CONFIG:Release>:-s>
            -Wl,-z,relro
            -Wl,-z,now
            -Wl,-z,noexecstack
            $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs -ftest-coverage>

)

# intf_pub_cxx14 for c++14
add_library(intf_pub_cxx14 INTERFACE)
target_compile_options(intf_pub_cxx14 INTERFACE
  -Wall
  -fPIC
  $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
  $<$<CONFIG:Debug>:-g>
  $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
  $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs -ftest-coverage>
)
target_compile_definitions(intf_pub_cxx14 INTERFACE
  _GLIBCXX_USE_CXX11_ABI=0
  $<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
  $<$<CONFIG:Debug>:CFG_BUILD_DEBUG>
  WIN64=1
  LINUX=0
)
target_link_options(intf_pub_cxx14 INTERFACE
  -Wl,-z,relro
  -Wl,-z,now
  -Wl,-z,noexecstack
  $<$<CONFIG:Release>:-s>
  $<$<CONFIG:Release>:-Wl,--build-id=none>
  $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs -ftest-coverage> 
)
target_link_directories(intf_pub_cxx14 INTERFACE)
target_link_libraries(intf_pub_cxx14 INTERFACE
  -lpthread
)

# intf_pub_cxx17 for c++17
add_library(intf_pub_cxx17 INTERFACE)
target_compile_options(intf_pub_cxx17 INTERFACE
    -Wall
    -fPIC
    $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
    $<$<CONFIG:Debug>:-g>
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
    $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs -ftest-coverage>  
  )
target_compile_definitions(intf_pub_cxx17 INTERFACE
    _GLIBCXX_USE_CXX11_ABI=0
    $<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
    $<$<CONFIG:Debug>:CFG_BUILD_DEBUG>
    WIN64=1
    LINUX=0)
target_link_options(intf_pub_cxx17 INTERFACE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    $<$<CONFIG:Release>:-s>
    $<$<CONFIG:Release>:-Wl,--build-id=none>
    $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs -ftest-coverage>   
  )
target_link_directories(intf_pub_cxx17 INTERFACE)
target_link_libraries(intf_pub_cxx17 INTERFACE
  -lpthread)

#########intf_pub_aicpu#########
add_library(intf_pub_aicpu INTERFACE)
target_compile_options(intf_pub_aicpu INTERFACE
  -Wall
  -fPIC
  $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
  $<$<CONFIG:Debug>:-g>
  $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
  $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs -ftest-coverage> 
)
target_compile_definitions(intf_pub_aicpu INTERFACE
  $<$<NOT:$<STREQUAL:${PRODUCT_SIDE},device>>:_GLIBCXX_USE_CXX11_ABI=0>
  $<$<STREQUAL:${PRODUCT_SIDE},device>:_GLIBCXX_USE_CXX11_ABI=1>
  $<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
  $<$<CONFIG:Debug>:CFG_BUILD_DEBUG>
  WIN64=1
  LINUX=0
)
target_link_options(intf_pub_aicpu INTERFACE
  -Wl,-z,relro
  -Wl,-z,now
  -Wl,-z,noexecstack
  $<$<CONFIG:Release>:-Wl,--build-id=none>
  $<$<BOOL:${ENABLE_GCOV}>:-fprofile-arcs -ftest-coverage>
)
target_link_directories(intf_pub_aicpu INTERFACE)
target_link_libraries(intf_pub_aicpu INTERFACE
  -lpthread
)
