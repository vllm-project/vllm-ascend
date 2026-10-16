# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
    msa_index_score_def.cpp
    )
endif()

# A2/A3 and Ascend 950 share the operator-private Catlass snapshot.
# Keep the msa_ header names isolated from the repository Catlass submodule.
set(MSA_CATLASS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel")
if(NOT EXISTS "${MSA_CATLASS_INCLUDE_DIR}/catlass/msa_catlass.hpp")
    message(FATAL_ERROR
        "MsaIndexScore requires vendored Catlass headers under op_kernel/catlass.")
endif()

add_ops_compile_options(
    OP_NAME MsaIndexScore
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -fpermissive
    OPTIONS -I${MSA_CATLASS_INCLUDE_DIR}
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE msa_index_score ACLNNTYPE aclnn)
endif()
