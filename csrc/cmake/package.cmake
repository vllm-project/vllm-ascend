# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#### CPACK to package run #####

# download makeself package
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/third_party/makeself-fetch.cmake)

# mc2_matmul KB install
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/runtimeKB.cmake)

function(pack_custom)
  message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
      message(STATUS "Detected architecture: x86_64")
      set(ARCH x86_64)
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
      message(STATUS "Detected architecture: ARM64")
      set(ARCH aarch64)
  else ()
      message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif ()
  set(PACK_CUSTOM_NAME "cann-ops-transformer-${VENDOR_NAME}-linux-${ARCH}")
  npu_op_package(${PACK_CUSTOM_NAME}
    TYPE RUN
    CONFIG
      ENABLE_SOURCE_PACKAGE True
      ENABLE_BINARY_PACKAGE True
      INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/
      ENABLE_DEFAULT_PACKAGE_NAME_RULE False
  )

  npu_op_package_add(${PACK_CUSTOM_NAME}
    LIBRARY
      cust_opapi
  )
  if (TARGET cust_proto)
    npu_op_package_add(${PACK_CUSTOM_NAME}
        LIBRARY
        cust_proto
    )
  endif()
  if (TARGET cust_opmaster)
    npu_op_package_add(${PACK_CUSTOM_NAME}
        LIBRARY
        cust_opmaster
    )
  endif()
endfunction()

function(pack_tiling_sink)
  ExternalProject_Get_Property(tiling_sink_task BINARY_DIR)

  if(ENABLE_BUILT_IN)
    set(TRANSFORMER_OPMASTER_SO ${BINARY_DIR}/libtiling_device_transformer.so)
    set(INSTALL_DIR "ops_transformer/built-in/op_impl/ai_core/tbe/op_tiling_device/lib")
  else()
    set(TRANSFORMER_OPMASTER_SO ${BINARY_DIR}/libcust_opmaster.so)
    set(INSTALL_DIR "packages/vendors/${VENDOR_NAME}_transformer/op_impl/ai_core/tbe/op_master_device/lib")
  endif()
  install(CODE "
    if(EXISTS \"${TRANSFORMER_OPMASTER_SO}\")
      file(
        INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}/${INSTALL_DIR}\"
        TYPE FILE FILES \"${TRANSFORMER_OPMASTER_SO}\")
    endif()
  ")
endfunction()

function(pack_built_in)
  #### built-in package ####
  message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
      message(STATUS "Detected architecture: x86_64")
      set(ARCH x86_64)
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
      message(STATUS "Detected architecture: ARM64")
      set(ARCH aarch64)
  else ()
      message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif ()

  set(script_prefix ${CMAKE_SOURCE_DIR}/scripts/package/ops_transformer/scripts)
  install(DIRECTORY ${script_prefix}/
      DESTINATION share/info/ops_transformer/script
      FILE_PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 文件权限
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
      DIRECTORY_PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 目录权限
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
      REGEX "(setenv|prereq_check)\\.(bash|fish|csh)" EXCLUDE
  )

  set(SCRIPTS_FILES
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/check_version_required.awk
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_func.inc
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_interface.sh
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_interface.csh
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_interface.fish
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/version_compatiable.inc
      ${CMAKE_SOURCE_DIR}/scripts/package/common/py/merge_binary_info_config.py
  )

  install(FILES ${SCRIPTS_FILES}
      DESTINATION share/info/ops_transformer/script
  )
  set(COMMON_FILES
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/install_common_parser.sh
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_func_v2.inc
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_installer.inc
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/script_operator.inc
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/version_cfg.inc
  )

  set(PACKAGE_FILES
      ${COMMON_FILES}
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/multi_version.inc
  )
  set(LATEST_MANGER_FILES
      ${COMMON_FILES}
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/common_func.inc
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/version_compatiable.inc
      ${CMAKE_SOURCE_DIR}/scripts/package/common/sh/check_version_required.awk
  )
  set(CONF_FILES
      ${CMAKE_SOURCE_DIR}/scripts/package/common/cfg/path.cfg
  )
  install(FILES ${CMAKE_SOURCE_DIR}/version.info
      DESTINATION share/info/ops_transformer
  )
  install(FILES ${CONF_FILES}
      DESTINATION ops_transformer/conf
  )
  install(FILES ${PACKAGE_FILES}
      DESTINATION share/info/ops_transformer/script
  )
  install(FILES ${LATEST_MANGER_FILES}
      DESTINATION latest_manager
  )
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/scripts/package/latest_manager/scripts/
      DESTINATION latest_manager
  )

  string(FIND "${ASCEND_COMPUTE_UNIT}" ";" SEMICOLON_INDEX)
  if (SEMICOLON_INDEX GREATER -1)
      # 截取分号前的字串
      math(EXPR SUBSTRING_LENGTH "${SEMICOLON_INDEX}")
      string(SUBSTRING "${ASCEND_COMPUTE_UNIT}" 0 "${SUBSTRING_LENGTH}" compute_unit)
  else()
      # 没有分号取全部内容
      set(compute_unit "${ASCEND_COMPUTE_UNIT}")
  endif()

  message(STATUS "current compute_unit is: ${compute_unit}")
  pack_tiling_sink()

  # ============= CPack =============
  set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
  set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
  string(REGEX REPLACE "^.*[Aa]scend" "" soc_version_temp "${ASCEND_COMPUTE_UNIT}")
  # 检查是否成功提取
  if("${soc_version_temp}" STREQUAL "${ASCEND_COMPUTE_UNIT}")
    set(soc_version "unknown")
  else()
    set(soc_version "${soc_version_temp}")
  endif()
  
  if(NOT ENABLE_OPS_KERNEL)
    set(CPACK_PACKAGE_FILE_NAME "CANN--${CPACK_PACKAGE_NAME}.run")
  else()
    if("${VERSION}" STREQUAL "")
      if("${soc_version}" STREQUAL "910_93")
        set(CPACK_PACKAGE_FILE_NAME "cann-A3-ops-transformer_linux-${ARCH}.run")
      else()
        set(CPACK_PACKAGE_FILE_NAME "cann-${soc_version}-ops-transformer_linux-${ARCH}.run")
      endif()
    else()
      if("${soc_version}" STREQUAL "910_93")
        set(CPACK_PACKAGE_FILE_NAME "cann-A3-ops-transformer_${VERSION}_linux-${ARCH}.run")
      else()
        set(CPACK_PACKAGE_FILE_NAME "cann-${soc_version}-ops-transformer_${VERSION}_linux-${ARCH}.run")
      endif()
    endif()
    
  endif()

  set(CPACK_INSTALL_PREFIX "/")

  set(CPACK_CMAKE_SOURCE_DIR "${CMAKE_SOURCE_DIR}")
  set(CPACK_CMAKE_BINARY_DIR "${CMAKE_BINARY_DIR}")
  set(CPACK_CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
  set(CPACK_CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  set(CPACK_MAKESELF_PATH "${MAKESELF_PATH}")
  set(CPACK_SOC "${compute_unit}")
  set(CPACK_ARCH "${ARCH}")
  set(CPACK_SET_DESTDIR ON)
  set(CPACK_VERSION "${VERSION}")
  set(CPACK_GENERATOR External)
  if (ENABLE_BUILT_IN)
    set(CPACK_EXTERNAL_PACKAGE_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/makeself_built_in.cmake")
  endif()
  set(CPACK_EXTERNAL_ENABLE_STAGING true)
  set(CPACK_PACKAGE_DIRECTORY "${CMAKE_INSTALL_PREFIX}")

  message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
  include(CPack)
endfunction()