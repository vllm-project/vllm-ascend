# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (unified_dlog_FOUND)
    message(STATUS "Package unified_dlog has been found.")
    return()
endif()

set(_cmake_targets_defined "")
set(_cmake_targets_not_defined "")
set(_cmake_expected_targets "")
foreach(_cmake_expected_target IN ITEMS unified_dlog unified_dlog_headers)
    list(APPEND _cmake_expected_targets "${_cmake_expected_target}")
    if(TARGET "${_cmake_expected_target}")
        list(APPEND _cmake_targets_defined "${_cmake_expected_target}")
    else()
        list(APPEND _cmake_targets_not_defined "${_cmake_expected_target}")
    endif()
endforeach()
unset(_cmake_expected_target)

if(_cmake_targets_defined STREQUAL _cmake_expected_targets)
    unset(_cmake_targets_defined)
    unset(_cmake_targets_not_defined)
    unset(_cmake_expected_targets)
    unset(CMAKE_IMPORT_FILE_VERSION)
    cmake_policy(POP)
    return()
endif()

if(NOT _cmake_targets_defined STREQUAL "")
    string(REPLACE ";" ", " _cmake_targets_defined_text "${_cmake_targets_defined}")
    string(REPLACE ";" ", " _cmake_targets_not_defined_text "${_cmake_targets_not_defined}")
    message(FATAL_ERROR "Some (but not all) targets in this export set were already defined.\nTargets Defined: ${_cmake_targets_defined_text}\nTargets not yet defined: ${_cmake_targets_not_defined_text}\n")
endif()
unset(_cmake_targets_defined)
unset(_cmake_targets_not_defined)
unset(_cmake_expected_targets)

find_path(_INCLUDE_ROOT_DIR
    NAMES toolchain/dlog_pub.h
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

find_library(unified_dlog_SHARED_LIBRARY
    NAMES libunified_dlog.so
    PATH_SUFFIXES lib64
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(unified_dlog
    FOUND_VAR
        unified_dlog_FOUND
    REQUIRED_VARS
        _INCLUDE_ROOT_DIR
        unified_dlog_SHARED_LIBRARY
)

if(unified_dlog_FOUND)
    set(unified_dlog_INCLUDE_DIR "${_INCLUDE_ROOT_DIR}")
    include(CMakePrintHelpers)
    message(STATUS "Variables in unified_dlog module:")
    cmake_print_variables(unified_dlog_INCLUDE_DIR)
    cmake_print_variables(unified_dlog_SHARED_LIBRARY)

    add_library(unified_dlog SHARED IMPORTED)
    set_target_properties(unified_dlog PROPERTIES
        INTERFACE_COMPILE_DEFINITIONS "LOG_CPP;PROCESS_LOG"
        INTERFACE_LINK_LIBRARIES "unified_dlog_headers"
        IMPORTED_LOCATION "${unified_dlog_SHARED_LIBRARY}"
    )

    add_library(unified_dlog_headers INTERFACE IMPORTED)
    set_target_properties(unified_dlog_headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${unified_dlog_INCLUDE_DIR};${unified_dlog_INCLUDE_DIR}/toolchain"
    )

    include(CMakePrintHelpers)
    cmake_print_properties(TARGETS unified_dlog
        PROPERTIES INTERFACE_COMPILE_DEFINITIONS INTERFACE_LINK_LIBRARIES IMPORTED_LOCATION
    )
    cmake_print_properties(TARGETS unified_dlog_headers
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
    )
endif()

# Cleanup temporary variables.
set(_INCLUDE_ROOT_DIR)
