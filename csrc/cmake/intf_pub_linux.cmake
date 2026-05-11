# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
if(TARGET intf_pub)
  message(STATUS "intf_pub has been found, no need add library")
  return()
endif()

# intf_pub for c++11
add_library(intf_pub INTERFACE)
target_compile_options(intf_pub INTERFACE
  -Wall
  -fPIC
  $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
  $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
)
target_compile_definitions(intf_pub INTERFACE
  _GLIBCXX_USE_CXX11_ABI=0
  $<$<CONFIG:Release>:CFG_BUILD_NDEBUG>
  $<$<CONFIG:Debug>:CFG_BUILD_DEBUG>
  WIN64=1
  LINUX=0
)
target_link_options(intf_pub INTERFACE
  -Wl,-z,relro
  -Wl,-z,now
  -Wl,-z,noexecstack
  $<$<CONFIG:Release>:-Wl,--build-id=none>
)
target_link_directories(intf_pub INTERFACE)
target_link_libraries(intf_pub INTERFACE
  -lpthread
)

# intf_pub_cxx14 for c++14
add_library(intf_pub_cxx14 INTERFACE)
target_compile_options(intf_pub_cxx14 INTERFACE
  -Wall
  -fPIC
  $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
  $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
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
  $<$<CONFIG:Release>:-Wl,--build-id=none>
)
target_link_directories(intf_pub_cxx14 INTERFACE)
target_link_libraries(intf_pub_cxx14 INTERFACE
  -lpthread
)

# intf_pub_cxx14 for c++17
add_library(intf_pub_cxx17 INTERFACE)
target_compile_options(intf_pub_cxx17 INTERFACE
    -Wall
    -fPIC
    $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)
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
    $<$<CONFIG:Release>:-Wl,--build-id=none>)
target_link_directories(intf_pub_cxx17 INTERFACE)
target_link_libraries(intf_pub_cxx17 INTERFACE
  -lpthread)

#########intf_pub_aicpu#########
add_library(intf_pub_aicpu INTERFACE)
target_compile_options(intf_pub_aicpu INTERFACE
  -Wall
  -fPIC
  $<IF:$<VERSION_GREATER:${CMAKE_C_COMPILER_VERSION},4.8.5>,-fstack-protector-strong,-fstack-protector-all>
  $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
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
)
target_link_directories(intf_pub_aicpu INTERFACE)
target_link_libraries(intf_pub_aicpu INTERFACE
  -lpthread
)