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

unset(gtest_FOUND CACHE)
unset(GTEST_INCLUDE CACHE)
unset(GTEST_STATIC_LIBRARY CACHE)
unset(GTEST_MAIN_STATIC_LIBRARY CACHE)
unset(GMOCK_STATIC_LIBRARY CACHE)
unset(GMOCK_MAIN_STATIC_LIBRARY CACHE)

set(GTEST_INSTALL_PATH ${CANN_3RD_LIB_PATH}/gtest)
message("GTEST_INSTALL_PATH=${GTEST_INSTALL_PATH}")
find_path(GTEST_INCLUDE
        NAMES gtest/gtest.h
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${GTEST_INSTALL_PATH}/include)
find_library(GTEST_STATIC_LIBRARY
        NAMES libgtest.a
        PATH_SUFFIXES lib lib64
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${GTEST_INSTALL_PATH})
find_library(GTEST_MAIN_STATIC_LIBRARY
        NAMES libgtest_main.a
        PATH_SUFFIXES lib lib64
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${GTEST_INSTALL_PATH})
find_library(GMOCK_STATIC_LIBRARY
        NAMES libgmock.a
        PATH_SUFFIXES lib lib64
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${GTEST_INSTALL_PATH})
find_library(GMOCK_MAIN_STATIC_LIBRARY
        NAMES libgmock_main.a
        PATH_SUFFIXES lib lib64
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH
        PATHS ${GTEST_INSTALL_PATH})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(gtest
        FOUND_VAR
        gtest_FOUND
        REQUIRED_VARS
        GTEST_INCLUDE
        GTEST_STATIC_LIBRARY
        GTEST_MAIN_STATIC_LIBRARY
        GMOCK_STATIC_LIBRARY
        GMOCK_MAIN_STATIC_LIBRARY
        )
message("gtest found:${gtest_FOUND}")

if(gtest_FOUND AND NOT FORCE_REBUILD_CANN_3RD)
    message("gtest found in ${GTEST_INSTALL_PATH}, and not force rebuild cann third_party")
else()
    set(REQ_URL "https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz")
    set (gtest_CXXFLAGS "-D_GLIBCXX_USE_CXX11_ABI=0 -O2 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")
    set (gtest_CFLAGS   "-D_GLIBCXX_USE_CXX11_ABI=0 -O2 -D_FORTIFY_SOURCE=2 -fPIC -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")

    include(ExternalProject)
    ExternalProject_Add(third_party_gtest
            URL ${REQ_URL}
            TLS_VERIFY OFF
            DOWNLOAD_DIR ${CANN_3RD_PKG_PATH}
            CONFIGURE_COMMAND ${CMAKE_COMMAND}
            -DCMAKE_CXX_FLAGS=${gtest_CXXFLAGS}
            -DCMAKE_C_FLAGS=${gtest_CFLAGS}
            -DCMAKE_INSTALL_PREFIX=${GTEST_INSTALL_PATH}
            -DCMAKE_INSTALL_LIBDIR=lib
            -DBUILD_SHARED_LIBS=OFF
            <SOURCE_DIR>
            BUILD_COMMAND $(MAKE)
            INSTALL_COMMAND $(MAKE) install
            EXCLUDE_FROM_ALL TRUE
            )
endif()

set(GTEST_INCLUDE ${GTEST_INSTALL_PATH}/include)

add_library(gtest STATIC IMPORTED)
add_dependencies(gtest third_party_gtest)

add_library(gmock STATIC IMPORTED)
add_dependencies(gmock third_party_gtest)

add_library(gtest_main STATIC IMPORTED)
add_dependencies(gtest_main third_party_gtest)

if (NOT EXISTS ${GTEST_INSTALL_PATH}/include)
  file(MAKE_DIRECTORY "${GTEST_INSTALL_PATH}/include")
endif ()

set_target_properties(gtest PROPERTIES
        IMPORTED_LOCATION ${GTEST_INSTALL_PATH}/lib/libgtest.a
        INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INSTALL_PATH}/include)

set_target_properties(gmock PROPERTIES
        IMPORTED_LOCATION ${GTEST_INSTALL_PATH}/lib/libgmock.a
        INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INSTALL_PATH}/include)

set_target_properties(gtest_main PROPERTIES
        IMPORTED_LOCATION ${GTEST_INSTALL_PATH}/lib/libgtest_main.a
        INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INSTALL_PATH}/include)
