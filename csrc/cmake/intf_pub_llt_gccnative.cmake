# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
if(TARGET intf_llt_pub)
  message(STATUS "intf_llt_pub has been found, no need add library")
  return()
endif()
add_library(intf_llt_pub INTERFACE)
target_include_directories(intf_llt_pub INTERFACE
  ${GTEST_INCLUDE}
  ${GMOCK_INCLUDE}
)
target_compile_definitions(intf_llt_pub INTERFACE
  _GLIBCXX_USE_CXX11_ABI=0
  CFG_BUILD_DEBUG
)
target_compile_options(intf_llt_pub INTERFACE
  -g
  --coverage
  -fprofile-arcs
  -ftest-coverage
  -w
  $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
  -fPIC
)
target_link_options(intf_llt_pub INTERFACE
  -fprofile-arcs -ftest-coverage
)
target_link_libraries(intf_llt_pub INTERFACE
  gcov
  pthread
)

if(TARGET intf_llt_pub_asan)
  message(STATUS "intf_llt_pub_asan has been found, no need add library")
  return()
endif()
add_library(intf_llt_pub_asan INTERFACE)
target_include_directories(intf_llt_pub_asan INTERFACE
  ${GTEST_INCLUDE}
  ${GMOCK_INCLUDE}
)
target_compile_definitions(intf_llt_pub_asan INTERFACE
  _GLIBCXX_USE_CXX11_ABI=0
  CFG_BUILD_DEBUG
)
target_compile_options(intf_llt_pub_asan INTERFACE
  -g
  --coverage
  -fprofile-arcs
  -ftest-coverage
  -w
  $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
  $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer -g>
  -fPIC
)
target_link_options(intf_llt_pub_asan INTERFACE
  -fprofile-arcs -ftest-coverage
  $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address>
)
target_link_libraries(intf_llt_pub_asan INTERFACE
  gcov
  pthread
)

if(TARGET intf_llt_pub_asan_cxx14)
  message(STATUS "intf_llt_pub_asan_cxx14 has been found, no need add library")
  return()
endif()
add_library(intf_llt_pub_asan_cxx14 INTERFACE)
target_include_directories(intf_llt_pub_asan_cxx14 INTERFACE
  ${GTEST_INCLUDE}
  ${GMOCK_INCLUDE}
)
target_compile_definitions(intf_llt_pub_asan_cxx14 INTERFACE
  _GLIBCXX_USE_CXX11_ABI=0
  CFG_BUILD_DEBUG
)
target_compile_options(intf_llt_pub_asan_cxx14 INTERFACE
  -g
  --coverage
  -fprofile-arcs
  -ftest-coverage
  -w
  $<$<COMPILE_LANGUAGE:CXX>:-std=c++14>
  $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer -g>
  -fPIC
)
target_link_options(intf_llt_pub_asan_cxx14 INTERFACE
  -fprofile-arcs -ftest-coverage
  $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address>
)
target_link_libraries(intf_llt_pub_asan_cxx14 INTERFACE
  gcov
  pthread
)

if(TARGET intf_llt_pub_asan_cxx17)
  message(STATUS "intf_llt_pub_asan_cxx17 has been found, no need add library")
  return()
endif()
add_library(intf_llt_pub_asan_cxx17 INTERFACE)
target_include_directories(intf_llt_pub_asan_cxx17 INTERFACE
  ${GTEST_INCLUDE}
  ${GMOCK_INCLUDE}
)
target_compile_definitions(intf_llt_pub_asan_cxx17 INTERFACE
  _GLIBCXX_USE_CXX11_ABI=0
  CFG_BUILD_DEBUG
)
target_compile_options(intf_llt_pub_asan_cxx17 INTERFACE
  -g
  --coverage
  -fprofile-arcs
  -ftest-coverage
  -w
  $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
  $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address -fsanitize-recover=address,all -fno-omit-frame-pointer -g>
  -fPIC
)
target_link_options(intf_llt_pub_asan_cxx17 INTERFACE
  -fprofile-arcs -ftest-coverage
  $<$<BOOL:${ENABLE_ASAN}>:-fsanitize=address>
)
target_link_libraries(intf_llt_pub_asan_cxx17 INTERFACE
  gcov
  pthread
)
