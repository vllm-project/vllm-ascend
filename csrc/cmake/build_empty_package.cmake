# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

function(cpack_empty_package)
  include(cmake/third_party/makeself-fetch.cmake)
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
      message(STATUS "Detected architecture: x86_64")
      set(ARCH x86_64)
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
      message(STATUS "Detected architecture: ARM64")
      set(ARCH aarch64)
  else ()
      message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif ()

  # CPack config
  install(FILES ${CMAKE_SOURCE_DIR}/version.info
      DESTINATION share/info/ops_transformer
  )
  install(FILES ${CMAKE_SOURCE_DIR}/scripts/package/ops_transformer/scripts/help.info
      DESTINATION share/info/ops_transformer/script
  )
  install(FILES ${CMAKE_SOURCE_DIR}/scripts/package/ops_transformer/scripts/empty_package_scripts/install.sh
      DESTINATION share/info/ops_transformer/script
      PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )
  install(FILES ${CMAKE_SOURCE_DIR}/scripts/package/ops_transformer/scripts/empty_package_scripts/cleanup.sh
      DESTINATION share/info/ops_transformer/script
      PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
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
  set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/build_out)
  string(REGEX REPLACE "^.*[Aa]scend" "" soc_version_temp "${ASCEND_COMPUTE_UNIT}")
  # 检查是否成功提取
  if("${soc_version_temp}" STREQUAL "${ASCEND_COMPUTE_UNIT}")
    set(soc_version "unknown")
  else()
    set(soc_version "${soc_version_temp}")
  endif()
  
  if("${VERSION}" STREQUAL "")
      set(CPACK_PACKAGE_FILE_NAME "cann-${soc_version}-ops-transformer_linux-${ARCH}.run")
  else()
      set(CPACK_PACKAGE_FILE_NAME "cann-${soc_version}-ops-transformer_${VERSION}_linux-${ARCH}.run")
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
  set(CPACK_GENERATOR External)
  if (ENABLE_BUILT_IN)
    set(CPACK_EXTERNAL_PACKAGE_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/makeself_built_in.cmake")
  endif()
  set(CPACK_EXTERNAL_ENABLE_STAGING true)
  set(CPACK_PACKAGE_DIRECTORY "${CMAKE_INSTALL_PREFIX}")

  message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
  include(CPack)
endfunction()