# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
include(ExternalProject)
set(PROTOBUF_VERSION_PKG protobuf-25.1.tar.gz)
set(ASCEND_PROTOBUF_DIR ${CANN_3RD_LIB_PATH}/ascend_protobuf)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ascend_protobuf_build_transformer
    FOUND_VAR
    ascend_protobuf_build_transformer_FOUND
    REQUIRED_VARS
    ASCEND_PROTOBUF_SHARED_INCLUDE
)

set(ASCEND_PROTOBUF_SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/ascend_protobuf)
if(ascend_protobuf_build_transformer_FOUND AND NOT FORCE_REBUILD_CANN_3RD)
    message(STATUS "[ThirdPartyLib][ascend protobuf] ascend_protobuf_shared found, skip compile.")
    cmake_print_variables(ASCEND_PROTOBUF_SHARED_INCLUDE)
    cmake_print_variables(ASCEND_PROTOC)
    set(Protobuf_INCLUDE ${ASCEND_PROTOBUF_SHARED_INCLUDE})
    set(Protobuf_PATH ${ASCEND_PROTOC})
    set(Protobuf_PROTOC_EXECUTABLE ${Protobuf_PATH}/protoc)
    add_library(ascend_protobuf_build_transformer INTERFACE)
else()
    message(STATUS "[ThirdPartyLib][ascend protobuf] ascend protobuf shared not found, finding binary file.")
    if(EXISTS "${CANN_3RD_LIB_PATH}/protobuf/protobuf-all-25.1.tar.gz")
        set(REQ_URL "file://${CANN_3RD_LIB_PATH}/protobuf/protobuf-all-25.1.tar.gz")
        message(STATUS "[ThirdPartyLib][ascend protobuf] found in ${REQ_URL}.")
    elseif(EXISTS "${CANN_3RD_LIB_PATH}/pkg/${PROTOBUF_VERSION_PKG}")
        set(REQ_URL "file://${CANN_3RD_LIB_PATH}/pkg/${PROTOBUF_VERSION_PKG}")
        message(STATUS "[ThirdPartyLib][ascend protobuf] found in ${REQ_URL}.")
    else()
        set(REQ_URL "https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz")
        message(STATUS "[ThirdPartyLib][ascend protobuf] ${REQ_URL} not found, need download.")
    endif()
    
    set(protobuf_CXXFLAGS "-Wno-maybe-uninitialized -Wno-unused-parameter -fPIC -fstack-protector-all -D_FORTIFY_SOURCE=2 -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -Dgoogle=ascend_private")
    set(protobuf_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")

    ExternalProject_Add(ascend_protobuf_build_transformer
                        URL ${REQ_URL}
                        DOWNLOAD_DIR ${CANN_3RD_LIB_PATH}/pkg
                        PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/build/modules/patch/protobuf_25.1_change_version.patch
                        CONFIGURE_COMMAND ${CMAKE_COMMAND}
                            -DCMAKE_MESSAGE_LOG_LEVEL=ERROR
                            -DCMAKE_INSTALL_LIBDIR=lib
                            -Dprotobuf_WITH_ZLIB=OFF
                            -DLIB_PREFIX=ascend_
                            -DCMAKE_SKIP_RPATH=TRUE
                            -Dprotobuf_BUILD_TESTS=OFF
                            -DBUILD_SHARED_LIBS=OFF
                            -DCMAKE_CXX_STANDARD=14
                            -DCMAKE_CXX_FLAGS=${protobuf_CXXFLAGS}
                            -DCMAKE_CXX_LDFLAGS=${protobuf_LDFLAGS}
                            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
                            -DCMAKE_INSTALL_PREFIX=${ASCEND_PROTOBUF_DIR}
                            -Dprotobuf_BUILD_PROTOC_BINARIES=ON
                            -Dprotobuf_ABSL_PROVIDER=module
                            -DABSL_ROOT_DIR=${ABSL_SOURCE_DIR}
                            <SOURCE_DIR>
                        SOURCE_DIR ${ASCEND_PROTOBUF_SOURCE_DIR}
                        BUILD_COMMAND $(MAKE)
                        INSTALL_COMMAND ""
                        EXCLUDE_FROM_ALL TRUE
    )
    if(TARGET abseil_build_transformer)
        add_dependencies(ascend_protobuf_build_transformer abseil_build_transformer)
    endif()

    ExternalProject_Get_Property(ascend_protobuf_build_transformer SOURCE_DIR)
    ExternalProject_Get_Property(ascend_protobuf_build_transformer BINARY_DIR)

    set(Protobuf_INCLUDE ${SOURCE_DIR}/src)
    set(Protobuf_PATH ${BINARY_DIR})
    set(Protobuf_PROTOC_EXECUTABLE ${Protobuf_PATH}/protoc)

    add_custom_command(
        OUTPUT ${Protobuf_PROTOC_EXECUTABLE}
        DEPENDS ascend_protobuf_build_transformer
    )
endif()