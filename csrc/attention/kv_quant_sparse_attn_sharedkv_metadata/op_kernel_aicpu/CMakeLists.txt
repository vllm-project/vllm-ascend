# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------

if (BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
  if (NOT (UT_TEST_ALL OR OP_KERNEL_AICPU_UT))
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
    set(CMAKE_CXX_COMPILER ${ASCEND_DIR}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++)
  endif()

  file(GLOB_RECURSE JSON_FILE ${CMAKE_CURRENT_SOURCE_DIR}/*.json)
  file(GLOB AICPU_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*_aicpu*.cpp)
  message(STATUS "[kv_quant_sparse_attn_sharedkv_metadata] Found aicpu sources: ${AICPU_SRC}, ascend dir: ${ASCEND_DIR}, ophost name: ${OPHOST_NAME}")

  add_aicpu_cust_kernel_modules(kv_quant_sparse_attn_sharedkv_metadata ${AICPU_SRC} ${JSON_FILE})
endif()

if(UT_TEST_ALL OR OP_KERNEL_AICPU_UT)
    AddAicpuOpTestCase(kv_quant_sparse_attn_sharedkv_metadata)
endif()
