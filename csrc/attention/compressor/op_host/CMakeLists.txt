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

get_filename_component(_COMPRESSOR_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
set(compressor_dsa_by_cache_dir
    ${_COMPRESSOR_DIR}/compressor_dsa_by_cache
    CACHE INTERNAL "Source directory for CompressorDsaByCache opFile"
)
set(COMPILED_OPS
    ${COMPILED_OPS}
    compressor_dsa_by_cache
    CACHE STRING "Compiled Ops" FORCE
)
set(COMPILED_OP_DIRS
    ${COMPILED_OP_DIRS}
    ${compressor_dsa_by_cache_dir}
    CACHE STRING "Compiled Ops Dirs" FORCE
)

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        compressor_def.cpp
        compressor_dsa_by_cache_def.cpp
    )
endif()

add_ops_compile_options(
        OP_NAME Compressor
                OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

add_ops_compile_options(
        OP_NAME CompressorDsaByCache
                OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(
        OPTYPE compressor compressor_dsa_by_cache
        ACLNNTYPE aclnn aclnn
    )
endif()
