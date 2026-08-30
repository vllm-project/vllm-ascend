/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_v2_sink_tiling_index.h
 * \brief
 */

#ifndef FUSED_INFER_ATTENTION_SCORE_V2_SINK_TILING_INDEX_H
#define FUSED_INFER_ATTENTION_SCORE_V2_SINK_TILING_INDEX_H

#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
// Inputs Index
static constexpr uint32_t QUERY_INDEX = 0;
static constexpr uint32_t KEY_INDEX = 1;
static constexpr uint32_t VALUE_INDEX = 2;
static constexpr uint32_t PSE_SHIFT_INDEX = 3;
static constexpr uint32_t ATTEN_MASK_INDEX = 4;
static constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 5;
static constexpr uint32_t ACTUAL_SEQ_KV_INDEX = 6;
static constexpr uint32_t DEQUANT_SCALE1_INDEX = 7;
static constexpr uint32_t QUANT_SCALE1_INDEX = 8;
static constexpr uint32_t DEQUANT_SCALE2_INDEX = 9;
static constexpr uint32_t QUANT_SCALE2_INDEX = 10;
static constexpr uint32_t QUANT_OFFSET2_INDEX = 11;
static constexpr uint32_t ANTIQUANT_SCALE_INDEX = 12;
static constexpr uint32_t ANTIQUANT_OFFSET_INDEX = 13;
static constexpr uint32_t BLOCK_TABLE_INDEX = 14;
static constexpr uint32_t QUERY_PADDING_SIZE_INDEX = 15;
static constexpr uint32_t KV_PADDING_SIZE_INDEX = 16;
static constexpr uint32_t KEY_ANTIQUANT_SCALE_INDEX = 17;
static constexpr uint32_t KEY_ANTIQUANT_OFFSET_INDEX = 18;
static constexpr uint32_t VALUE_ANTIQUANT_SCALE_INDEX = 19;
static constexpr uint32_t VALUE_ANTIQUANT_OFFSET_INDEX = 20;
static constexpr uint32_t QUERY_ROPE_INDEX = 21;
static constexpr uint32_t KEY_ROPE_INDEX = 22;
static constexpr uint32_t KEY_ROPE_ANTIQUANT_SCALE_INDEX = 23;
static constexpr uint32_t DEQUANT_SCALE_QUERY_INDEX = 24;
static constexpr uint32_t METADATA_INDEX = 25;
static constexpr uint32_t LEARNABLE_SINK_INDEX = 26;
static constexpr uint32_t KEY_SINK_INDEX = 27;
static constexpr uint32_t KEY_ROPE_SINK_INDEX = 28;
static constexpr uint32_t VALUE_SINK_INDEX = 29;

// Attributes Index
static constexpr uint32_t ATTR_N_INDEX = 0;
static constexpr uint32_t ATTR_SCALE_INDEX = 1;
static constexpr uint32_t ATTR_PRE_TOKEN_INDEX = 2;
static constexpr uint32_t ATTR_NEXT_TOKEN_INDEX = 3;
static constexpr uint32_t ATTR_INPUT_LAYOUT_INDEX = 4;
static constexpr uint32_t ATTR_NUM_KV_HEADS_INDEX = 5;
static constexpr uint32_t ATTR_SPARSE_MODE_INDEX = 6;
static constexpr uint32_t ATTR_INNER_PRECISE_INDEX = 7;
static constexpr uint32_t ATTR_BLOCK_SIZE_INDEX = 8;
static constexpr uint32_t ANTIQUANT_MODE_INDEX = 9;
static constexpr uint32_t SOFTMAX_LSE_FLAG_INDEX = 10;
static constexpr uint32_t KEY_ANTIQUANT_MODE_INDEX = 11;
static constexpr uint32_t VALUE_ANTIQUANT_MODE_INDEX = 12;
static constexpr uint32_t QUERY_QUANT_MODE_INDEX = 13;
static constexpr uint32_t PSE_TYPE_INDEX = 14;
static constexpr uint32_t ATTR_SINK_NUMBER_INDEX = 15;
static constexpr uint32_t ATTR_BATCH_INVARIANT_INDEX = 16;
static constexpr uint32_t SOFTMAX_MAX_SUM_FLAG_INDEX = 17;

// Output Index
static constexpr uint32_t ATTENTION_OUT_INDEX = 0;
static constexpr uint32_t SOFTMAX_LSE_INDEX = 1;
static constexpr uint32_t SOFTMAX_MAX_INDEX = 2;
static constexpr uint32_t SOFTMAX_SUM_INDEX = 3;
} // namespace optiling

#endif // FUSED_INFER_ATTENTION_SCORE_V2_SINK_TILING_INDEX_H