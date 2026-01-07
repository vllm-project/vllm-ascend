/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "split_mrope_tiling.h"

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)

static constexpr uint32_t FP32_DTYPE_SIZE = 4;
static constexpr uint32_t TILING_BF16 = 20;
static constexpr uint32_t TILING_FP16 = 21;
static constexpr uint64_t BUFFER_NUM = 1;

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // Get attribute value
    uint32_t index = 0U;
    const uint64_t num_q_heads = *(context->GetAttrs()->GetAttrPointer<uint64_t>(index++));
    const uint64_t num_kv_heads = *(context->GetAttrs()->GetAttrPointer<uint64_t>(index++));
    const uint64_t head_size = *(context->GetAttrs()->GetAttrPointer<uint64_t>(index++));
    const auto attrMRopeSection = context->GetAttrs()->GetAttrPointer<gert::ContinuousVector>(index++);
    const uint64_t* mrope_section = reinterpret_cast<const uint64_t*>(attrMRopeSection->GetData());
    // is_neox_style
    const bool is_neox_style = *(context->GetAttrs()->GetAttrPointer<uint64_t>(index++));

    uint64_t mrope_section_0 = mrope_section[0];
    uint64_t mrope_section_1 = mrope_section[1];
    uint64_t mrope_section_2 = mrope_section[2];

    // input variable
    const gert::StorageShape* in_query_shape = context->GetInputShape(1);
    const gert::StorageShape* in_cos_sin_cache_shape = context->GetInputShape(2);

    // platform
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t core_num = static_cast<uint64_t>(platform.GetCoreNum());

    uint64_t num_tokens = static_cast<uint64_t>(context->GetOutputShape(0)->GetStorageShape().GetDim(0));

    uint64_t max_ub_size;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, max_ub_size);

    // Maximum number of lines that ub can load
    uint64_t rotary_dim = static_cast<uint64_t>(in_cos_sin_cache_shape->GetStorageShape().GetDim(1));
    uint64_t num_heads_max = num_q_heads > num_kv_heads ? num_q_heads : num_kv_heads;

    uint64_t all_size = is_neox_style == 1UL ?
        static_cast<uint64_t>(num_heads_max * (rotary_dim * 8UL + head_size) * FP32_DTYPE_SIZE) :
        static_cast<uint64_t>(num_heads_max * (rotary_dim * 10UL + head_size) * FP32_DTYPE_SIZE);

    uint64_t max_n_per_loop_for_ub = max_ub_size / all_size;

    // total_data_num
    uint64_t total_data_num = num_tokens;
    // front_core
    uint64_t front_core = total_data_num % core_num != 0 ? static_cast<uint64_t>(total_data_num % core_num) : core_num;
    // tail_core
    uint64_t tail_core = total_data_num <= core_num ? 0 : core_num - front_core;

    // num_tokens_each_tail_core
    uint64_t num_tokens_each_tail_core = total_data_num / core_num;
    // loop_time_each_tail_core
    uint64_t loop_time_each_tail_core = (num_tokens_each_tail_core + max_n_per_loop_for_ub - 1) / max_n_per_loop_for_ub;
    uint64_t num_tokens_tail_core_each_loop =
        loop_time_each_tail_core <= 1UL ? num_tokens_each_tail_core : max_n_per_loop_for_ub;
    // num_tokens_each_front_core
    uint64_t num_tokens_each_front_core = (total_data_num + core_num - 1) / core_num;
    // loop_time_each_front_core
    uint64_t loop_time_each_front_core =
        (num_tokens_each_front_core + max_n_per_loop_for_ub - 1UL) / static_cast<uint64_t>(max_n_per_loop_for_ub);
    // num_tokens_front_core_each_loop
    uint64_t num_tokens_front_core_each_loop =
        loop_time_each_front_core == 1UL ? num_tokens_each_front_core : max_n_per_loop_for_ub;

    // num_tokens_front_core_last_loop
    uint64_t num_tokens_front_core_last_loop =
        loop_time_each_front_core == 1UL ?
        0 :
        num_tokens_each_front_core - num_tokens_front_core_each_loop * (loop_time_each_front_core - 1UL);
    // num_tokens_tail_core_last_loop
    uint64_t num_tokens_tail_core_last_loop =
        static_cast<uint64_t>(loop_time_each_front_core) == 1UL ?
        0 :
        num_tokens_each_tail_core - num_tokens_tail_core_each_loop * (loop_time_each_tail_core - 1UL);

    SplitMropeTilingData tiling;
    tiling.set_num_q_heads(num_q_heads);
    tiling.set_num_kv_heads(num_kv_heads);
    tiling.set_head_size(head_size);
    tiling.set_rotary_dim(rotary_dim);
    tiling.set_mrope_section_0(mrope_section_0);
    tiling.set_mrope_section_1(mrope_section_1);
    tiling.set_mrope_section_2(mrope_section_2);
    tiling.set_is_neox_style(is_neox_style);
    tiling.set_num_tokens(num_tokens);

    tiling.set_front_core(front_core);
    tiling.set_tail_core(tail_core);
    tiling.set_num_tokens_each_front_core(num_tokens_each_front_core);
    tiling.set_num_tokens_each_tail_core(num_tokens_each_tail_core);

    tiling.set_loop_time_each_front_core(loop_time_each_front_core);
    tiling.set_num_tokens_front_core_each_loop(num_tokens_front_core_each_loop);
    tiling.set_num_tokens_front_core_last_loop(num_tokens_front_core_last_loop);

    tiling.set_loop_time_each_tail_core(loop_time_each_tail_core);
    tiling.set_num_tokens_tail_core_each_loop(num_tokens_tail_core_each_loop);
    tiling.set_num_tokens_tail_core_last_loop(num_tokens_tail_core_last_loop);

    // 获取数据类型
    auto qDtype = context->GetInputDesc(1)->GetDataType();
    uint64_t tiling_key = 0;
    if (qDtype == ge::DT_BF16) {
        tiling_key = TILING_BF16;
    }
    if (qDtype == ge::DT_FLOAT16) {
        tiling_key = TILING_FP16;
    }
    context->SetTilingKey(tiling_key);

    uint64_t block_dim = front_core + tail_core;
    context->SetBlockDim(block_dim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t sysWorkspaceSize = platform.GetLibApiWorkSpaceSize();
    size_t* workspaces = context->GetWorkspaceSizes(1);
    size_t UserWorkspaceSize = 0;
    workspaces[0] = sysWorkspaceSize + UserWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SplitMrope)
    .Tiling(TilingFunc);
} // optiling