/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file custom_fused_infer_attention_v310_tiling_base.h
 * \brief
 */
#ifndef IFA_TILING_BASE_DEFINE_H
#define IFA_TILING_BASE_DEFINE_H

#include <cstdint>
using namespace ge;
namespace optiling {

constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t KEY_INPUT_INDEX = 1;
constexpr uint32_t VALUE_INPUT_INDEX = 2;
constexpr uint32_t ATTN_MASK_INPUT_INDEX = 3;
constexpr uint32_t ACT_SEQ_LEN_Q_INPUT_INDEX = 4;
constexpr uint32_t ACT_SEQ_LEN_INPUT_INDEX = 5;
constexpr uint32_t BLOCK_TABLE_INPUT_INDEX = 6;
constexpr uint32_t OUTPUT_INDEX = 0;
constexpr uint32_t NUM_HEADS_ATTR_INDEX = 0;
constexpr uint32_t SCALE_VALUE_ATTR_INDEX = 1;
constexpr uint32_t LAYOUT_ATTR_INDEX = 2;
constexpr uint32_t KV_NUM_HEADS_ATTR_INDEX = 3;
constexpr uint32_t BLOCK_SIZE_ATTR_INDEX = 4;
constexpr uint32_t INNER_PRECISE_ATTR_INDEX = 5;
constexpr uint32_t DEFAULT_QUERY_SEQ_STEP_HEAD_DIM_128_LESS = 32;
constexpr uint32_t DEFAULT_QUERY_SEQ_STEP_HEAD_DIM_256 = 16;

const uint32_t MAX_CORE_NUM = 50;

}// namespace optiling


#endif // IFA_TILING_BASE_DEFINE_H