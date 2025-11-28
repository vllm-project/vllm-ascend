# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

########################################################################################################################
# Environment Check
########################################################################################################################

# Python3
find_package(Python3)
if ((NOT Python3_FOUND) OR (${Python3_EXECUTABLE} STREQUAL ""))
    message(FATAL_ERROR "Can't find python3.")
endif ()
set(HI_PYTHON   "${Python3_EXECUTABLE}" CACHE   STRING   "python executor")

# Get the base CANN path
if (CUSTOM_ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH  ${CUSTOM_ASCEND_CANN_PACKAGE_PATH})
elseif (DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH  $ENV{ASCEND_HOME_PATH})
elseif (DEFINED ENV{ASCEND_OPP_PATH})
    get_filename_component(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_OPP_PATH}/.." ABSOLUTE)
else()
    set(ASCEND_CANN_PACKAGE_PATH  "/usr/local/Ascend/latest")
endif ()
message(STATUS "ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}")

########################################################################################################################
# Common Configuration
########################################################################################################################

# Switches
option(PREPARE_BUILD              "Prepare build."                  OFF)
option(ENABLE_OPS_HOST            "Build ops host."                 ON)
option(ENABLE_OPS_KERNEL          "Build ops kernel."               ON)
if (TESTS_EXAMPLE_OPS_TEST OR TESTS_UT_OPS_TEST)
    set(ENABLE_OPS_KERNEL         OFF)
endif ()
set(OP_DEBUG_CONFIG               "false"                         CACHE   STRING   "op debug config")

# Path configuration
#   Source tree related paths
get_filename_component(OPS_ADV_DIR                  "${CMAKE_CURRENT_SOURCE_DIR}"           REALPATH)
get_filename_component(OPS_ADV_CMAKE_DIR            "${OPS_ADV_DIR}/cmake"                  REALPATH)
get_filename_component(OPS_ADV_UTILS_KERNEL_INC     "${OPS_ADV_DIR}/utils/inc/kernel"   REALPATH)


#   Build tree related paths
set(ASCEND_IMPL_OUT_DIR           ${CMAKE_CURRENT_BINARY_DIR}/impl                     CACHE   STRING "ascend impl output directories")
set(ASCEND_BINARY_OUT_DIR         ${CMAKE_CURRENT_BINARY_DIR}/binary                   CACHE   STRING "ascend binary output directories")
set(ASCEND_AUTOGEN_DIR            ${CMAKE_CURRENT_BINARY_DIR}/autogen                  CACHE   STRING "Auto generate file directories")
set(ASCEND_CUSTOM_OPTIONS         ${ASCEND_AUTOGEN_DIR}/custom_compile_options.ini)
set(ASCEND_CUSTOM_TILING_KEYS     ${ASCEND_AUTOGEN_DIR}/custom_tiling_keys.ini)
set(ASCEND_CUSTOM_OPC_OPTIONS     ${ASCEND_AUTOGEN_DIR}/custom_opc_options.ini)
set(OP_BUILD_TOOL                 ${ASCEND_CANN_PACKAGE_PATH}/tools/opbuild/op_build   CACHE   STRING   "op_build tool")
file(MAKE_DIRECTORY ${ASCEND_AUTOGEN_DIR})
file(REMOVE ${ASCEND_CUSTOM_OPTIONS})
file(TOUCH ${ASCEND_CUSTOM_OPTIONS})
file(REMOVE ${ASCEND_CUSTOM_TILING_KEYS})
file(TOUCH ${ASCEND_CUSTOM_TILING_KEYS})
file(REMOVE ${ASCEND_CUSTOM_OPC_OPTIONS})
file(TOUCH ${ASCEND_CUSTOM_OPC_OPTIONS})
if (BUILD_OPEN_PROJECT)
    if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/ascend_project/cmake)
        set(ASCEND_PROJECT_DIR       ${ASCEND_CANN_PACKAGE_PATH}/tools/ascend_project)
    else()
        set(ASCEND_PROJECT_DIR       ${ASCEND_CANN_PACKAGE_PATH}/tools/op_project_templates/ascendc/customize)
    endif()
    set(ASCEND_CMAKE_DIR         ${ASCEND_PROJECT_DIR}/cmake   CACHE   STRING   "ascend project cmake")
    set(IMPL_INSTALL_DIR         packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl)
    set(IMPL_DYNAMIC_INSTALL_DIR packages/vendors/${VENDOR_NAME}/op_impl/ai_core/tbe/${VENDOR_NAME}_impl/dynamic)
    set(ACLNN_INC_INSTALL_DIR    packages/vendors/${VENDOR_NAME}/op_api/include)
else()
    set(ASCEND_CMAKE_DIR         ${TOP_DIR}/asl/ops/cann/ops/built-in/ascendc/samples/customize/cmake   CACHE   STRING   "ascend project cmake")
    set(IMPL_INSTALL_DIR         lib/ascendc/impl)
    set(IMPL_DYNAMIC_INSTALL_DIR lib/ascendc/impl/dynamic)
    set(ACLNN_INC_INSTALL_DIR    lib/include)
    set(OPS_STATIC_TYPES         infer train)
    set(OPS_STATIC_SCRIPT        ${TOP_DIR}/asl/ops/cann/ops/built-in/kernel/binary_script/build_opp_kernel_static.py)
endif ()
set(ASCENDC_CMAKE_UTIL_DIR       ${ASCEND_CMAKE_DIR}/util)
set(CUSTOM_DIR         ${CMAKE_BINARY_DIR}/custom)
set(TILING_CUSTOM_DIR  ${CUSTOM_DIR}/op_impl/ai_core/tbe/op_tiling)
set(TILING_CUSTOM_FILE ${TILING_CUSTOM_DIR}/liboptiling.so)

# Temporary adaptation for ascendc changes, to be removed after switching to the new version of ascendc
if(EXISTS ${ASCENDC_CMAKE_UTIL_DIR}/ascendc_gen_options.py)
    set(ADD_OPS_COMPILE_OPTION_V2 ON)
else()
    set(ADD_OPS_COMPILE_OPTION_V2 OFF)
endif()

########################################################################################################################
# CMake Options, Default Parameters Setting
#   Configure CMake options and default parameters according to the CMake build process
#   CMake build process: 1) Configuration phase; 2) Build phase; 3) Installation phase;
########################################################################################################################
if (BUILD_OPEN_PROJECT)
    # Build phase
    #   Build type
    #       The Generator in CMake is a tool used to generate native build systems. Generally divided into two types:
    #       1. Single-configuration generator:
    #          In the configuration phase, only one build type is allowed to be specified through the variable CMAKE_BUILD_TYPE;
    #          In the build phase, the build type cannot be changed, and only the build type specified through the variable CMAKE_BUILD_TYPE in the configuration phase can be used;
    #          Common generators of this type include: Ninja, Unix Makefiles
    #       2. Multi-configuration generator:
    #          In the configuration phase, only the list of build types available in the build phase is specified through the variable CMAKE_CONFIGURATION_TYPES;
    #          In the build phase, the specific build type of the build phase is specified through the "--config" parameter;
    #          Common generators of this type include: Xcode, Visual Studio
    #       Therefore:
    #           1. In the single-configuration generator scenario, if the build type (CMAKE_BUILD_TYPE) is not specified, the default is Debug;
    #           2. In the multi-configuration generator scenario, if the build types available in the build phase (CMAKE_CONFIGURATION_TYPES) are not specified,
    #              it is defaulted to the full set of build types allowed by CMake [Debug;Release;MinSizeRel;RelWithDebInfo]
    get_property(GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if (GENERATOR_IS_MULTI_CONFIG)
        if (NOT CMAKE_CONFIGURATION_TYPES)
            set(CMAKE_CONFIGURATION_TYPES "Debug;Release;MinSizeRel;RelWithDebInfo" CACHE STRING "Configuration Build type" FORCE)
        endif ()
    else ()
        if (NOT CMAKE_BUILD_TYPE)
            set(CMAKE_BUILD_TYPE          "Debug"                                   CACHE STRING "Build type(default Debug)" FORCE)
        endif ()
    endif ()

    # Build phase
    #   Executable runtime library file search path RPATH
    #       Do not skip RPATH in UTest and Example scenarios
    if (TESTS_UT_OPS_TEST OR TESTS_EXAMPLE_OPS_TEST)
        set(CMAKE_SKIP_RPATH FALSE)
    else ()
        set(CMAKE_SKIP_RPATH TRUE)
    endif ()

    # Build phase
    #   CCACHE configuration
    if (ENABLE_CCACHE)
        if (CUSTOM_CCACHE)
            set(CCACHE_PROGRAM ${CUSTOM_CCACHE})
        else()
            find_program(CCACHE_PROGRAM ccache)
        endif ()
        if (CCACHE_PROGRAM)
            set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM} CACHE PATH "C cache Compiler")
            set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM} CACHE PATH "CXX cache Compiler")
        endif ()
    endif ()

    # Installation phase
    #   Installation path
    #       When CMAKE_INSTALL_PREFIX is not explicitly set (i.e., CMAKE_INSTALL_PREFIX takes the default value),
    #       correct its value to be level with the build tree root directory CMAKE_CURRENT_BINARY_DIR
    if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        get_filename_component(_Install_Path_Prefix "${CMAKE_CURRENT_BINARY_DIR}/../output" REALPATH)
        set(CMAKE_INSTALL_PREFIX    "${_Install_Path_Prefix}"  CACHE STRING "Install path" FORCE)
    endif ()
endif ()

########################################################################################################################
# Public Compilation Parameters
########################################################################################################################
list(TRANSFORM ASCEND_COMPUTE_UNIT TOLOWER)
if (BUILD_OPEN_PROJECT)
    message(STATUS "ENABLE_CCACHE=${ENABLE_CCACHE}, CUSTOM_CCACHE=${CUSTOM_CCACHE}")
    message(STATUS "CCACHE_PROGRAM=${CCACHE_PROGRAM}")
    message(STATUS "ASCEND_COMPUTE_UNIT=${ASCEND_COMPUTE_UNIT}")
    message(STATUS "ASCEND_OP_NAME=${ASCEND_OP_NAME}")
    message(STATUS "TILING_KEY=${TILING_KEY}")
    message(STATUS "TESTS_UT_OPS_TEST=${TESTS_UT_OPS_TEST}")
    message(STATUS "TESTS_EXAMPLE_OPS_TEST=${TESTS_EXAMPLE_OPS_TEST}")
endif ()

########################################################################################################################
# Preprocessing
########################################################################################################################
if (BUILD_OPEN_PROJECT)
    if (NOT PREPARE_BUILD AND ENABLE_OPS_KERNEL)
        if (TILING_KEY)
            string(REPLACE ";" "::" EP_TILING_KEY "${TILING_KEY}")
        else()
            set(EP_TILING_KEY FALSE)
        endif ()

        if (OPS_COMPILE_OPTIONS)
            string(REPLACE ";" "::" EP_OPS_COMPILE_OPTIONS "${OPS_COMPILE_OPTIONS}")
        else()
            set(EP_OPS_COMPILE_OPTIONS FALSE)
        endif ()

        string(REPLACE ";" "::" EP_ASCEND_COMPUTE_UNIT "${ASCEND_COMPUTE_UNIT}")

        execute_process(COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scripts/prepare.sh
                -s ${CMAKE_CURRENT_SOURCE_DIR}
                -b ${CMAKE_CURRENT_BINARY_DIR}/prepare_build
                -p ${ASCEND_CANN_PACKAGE_PATH}
                --autogen-dir ${ASCEND_AUTOGEN_DIR}
                --build-open-project ${BUILD_OPEN_PROJECT}
                --binary-out-dir ${ASCEND_BINARY_OUT_DIR}
                --impl-out-dir ${ASCEND_IMPL_OUT_DIR}
                --op-build-tool ${OP_BUILD_TOOL}
                --ascend-cmake-dir ${ASCEND_CMAKE_DIR}
                --tiling-key ${EP_TILING_KEY}
                --ops-compile-options ${EP_OPS_COMPILE_OPTIONS}
                --check-compatible ${CHECK_COMPATIBLE}
                --ascend-compute_unit ${EP_ASCEND_COMPUTE_UNIT}
                --op_debug_config ${OP_DEBUG_CONFIG}
                --ascend-op-name "${ASCEND_OP_NAME}"
                RESULT_VARIABLE result
                OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE PREPARE_BUILD_OUTPUT_VARIABLE)
        if (result)
            message(FATAL_ERROR "Error: ops prepare build failed.")
        endif ()

        file(REMOVE ${ASCEND_CUSTOM_OPTIONS})
        file(TOUCH ${ASCEND_CUSTOM_OPTIONS})
        file(REMOVE ${ASCEND_CUSTOM_TILING_KEYS})
        file(TOUCH ${ASCEND_CUSTOM_TILING_KEYS})
        file(REMOVE ${ASCEND_CUSTOM_OPC_OPTIONS})
        file(TOUCH ${ASCEND_CUSTOM_OPC_OPTIONS})
    endif ()
endif ()

########################################################################################################################
# Other Configuration
########################################################################################################################
if (BUILD_OPEN_PROJECT)
    if (TESTS_UT_OPS_TEST)
        include(${OPS_ADV_CMAKE_DIR}/config_utest.cmake)
    endif ()
endif ()
