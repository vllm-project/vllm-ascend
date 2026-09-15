# This program is free software; you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_op_to_compiled_list()

add_ops_compile_options(
        OP_NAME VllmBlasstAttentionScore
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

# def.cpp 编入 op_host_aclnnExc：autogen 只产出空壳 stub，
# 正式 aclnn wrapper 使用 op_api/ 下手写版本（追加 2 个 ListInt host attr，
# 见 op_api/aclnn_vllm_blasst_attention_score.cpp），模式对齐 sparse_attention_score。
# 注意：def 文件名必须保持 vllm_blasst_attention_score_def.cpp ——
# 顶层 CMakeLists 从 def 文件名推导 autogen/exc/<name>_proto.cpp 的期望名，
# 而 opbuild 生成器按算子类名 VllmBlasstAttentionScore 实际产出，
# 两者一致（同为 vllm_blasst_attention_score）才能通过编译。
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        vllm_blasst_attention_score_def.cpp
    )
endif()

if (NOT BUILD_OPS_RTY_KERNEL)
    # OPTYPE 必须与 def 文件名前缀一致（vllm_ 开头），否则 *_def*.cpp 的
    # glob 匹配不到 vllm_blasst_attention_score_def.cpp。
    add_modules_sources(OPTYPE vllm_blasst_attention_score ACLNNTYPE aclnn_exclude)
endif()

# 手写 proto（InferShape/InferDataType 注册）：autogen exc proto 是空壳 stub，
# 正式注册用本目录手写版本。opsproto 覆盖常规库流程，cust_proto 覆盖
# build.sh --pkg 打包流程（custom_build.cmake 先于本目录 include，target 已存在）。
if (TARGET opsproto)
    target_sources(opsproto PRIVATE
        blasst_attention_score_proto.cpp
    )
endif()
if (TARGET cust_proto)
    target_sources(cust_proto PRIVATE
        blasst_attention_score_proto.cpp
    )
endif()

# tiling 源码由 add_modules_sources 的 *_tiling*.cpp glob 自动收编进
# ${OPHOST_NAME}_tiling_obj，这里仅追加本目录头文件包含路径。
# （host-list-only 设计：seq 经 host attrs 进 TilingData，tiling 无 D2H、
# 不依赖 acl_rt，无需链接 libacl_rt.so。）
if (TARGET ${OPHOST_NAME}_tiling_obj)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
