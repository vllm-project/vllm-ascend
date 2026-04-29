/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file paged_select_attention_tiling_index.h
 * \brief Input / attr / output slot indices for the specialized paged select attention op.
 */

#ifndef PAGED_SELECT_ATTENTION_TILING_INDEX_H
#define PAGED_SELECT_ATTENTION_TILING_INDEX_H

#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t VALUE_INDEX = 2;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 3;
constexpr uint32_t ACTUAL_SEQ_KV_INDEX = 4;
constexpr uint32_t BLOCK_TABLE_INDEX = 5;
constexpr uint32_t SELECTED_KV_INDICES_INDEX = 6;

constexpr uint32_t ATTR_NUM_HEADS_INDEX = 0;
constexpr uint32_t ATTR_SCALE_VALUE_INDEX = 1;
constexpr uint32_t ATTR_NUM_KV_HEADS_INDEX = 2;
constexpr uint32_t ATTR_BLOCK_SIZE_INDEX = 3;

constexpr uint32_t ATTENTION_OUT_INDEX = 0;
constexpr uint32_t SOFTMAX_LSE_INDEX = 1;
} // namespace optiling

#endif // PAGED_SELECT_ATTENTION_TILING_INDEX_H
