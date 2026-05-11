# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(aicpu_FOUND)
  message(STATUS "aicpu has been found")
  return()
endif()

include(FindPackageHandleStandardArgs)

if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
  set(AICPU_INC_DIRS
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include/aicpu
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include/experiment
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include/experiment/msprof
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/pkg_inc/aicpu_common/context
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/pkg_inc/aicpu_common/context/common
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/pkg_inc/aicpu_common/context/cpu_proto
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/pkg_inc/aicpu_common/context/utils
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include/experiment/datagw/aicpu/common
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/include/aicpu_common/context/utils
  )
else()
  set(AICPU_INC_DIRS
    ${TOP_DIR}/abl/msprof/inc  # codespell:ignore abl
    ${TOP_DIR}/ace/comop/inc
    ${TOP_DIR}/inc/aicpu/cpu_kernels
    ${TOP_DIR}/inc/external/aicpu
    ${TOP_DIR}/asl/ops/cann/ops/built-in/aicpu/context/inc
    ${TOP_DIR}/asl/ops/cann/ops/built-in/aicpu/impl/utils
    ${TOP_DIR}/asl/ops/cann/ops/built-in/aicpu/impl
    ${TOP_DIR}/ops-base/pkg_inc/aicpu_common/context/common
    ${TOP_DIR}/open_source/eigen
  )
endif()

message(STATUS "Using AICPU include dirs: ${AICPU_INC_DIRS}")
