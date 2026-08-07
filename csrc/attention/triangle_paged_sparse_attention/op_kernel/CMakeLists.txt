# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    npu_op_kernel_options(ascendc_kernels ALL OPTIONS -g -O0)
else()
    npu_op_kernel_options(ascendc_kernels ALL OPTIONS -O3)
endif()

npu_op_kernel_sources(ascendc_kernels
    KERNEL_DIR ./
)

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}/
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name}
    LIBRARY ascendc_kernels
)
