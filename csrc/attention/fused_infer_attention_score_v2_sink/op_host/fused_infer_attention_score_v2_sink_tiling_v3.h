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
 * \file fused_infer_attention_score_v2_sink_tiling_v3.h
 * \brief
 */
#ifndef FUSED_INFER_ATTENTION_SCORE_V2_SINK_TILING_V3
#define FUSED_INFER_ATTENTION_SCORE_V2_SINK_TILING_V3
#include <exe_graph/runtime/tiling_context.h>

#ifdef ASCENDC_OP_TEST
#define FIA_EXTERN_C extern "C"
#else
#define FIA_EXTERN_C
#endif
namespace optiling {

FIA_EXTERN_C ge::graphStatus TilingFusedInferAttentionScoreV2SinkV3(gert::TilingContext *context);
bool RouteToFia(gert::TilingContext *context);

static constexpr uint32_t QUERY_INPUT_INDEX = 0;
static constexpr uint32_t KEY_INPUT_INDEX = 1;
static constexpr uint32_t VALUE_INPUT_INDEX = 2;
static constexpr uint32_t PSE_SHIFT_INPUT_INDEX = 3;
static constexpr uint32_t ATTEN_MASK_INPUT_INDEX = 4;
static constexpr uint32_t ACT_SEQ_LEN_INPUT_INDEX = 5;
static constexpr uint32_t DEQUANT_SCALE_1_INPUT_INDEX = 6;
static constexpr uint32_t QUANT_SCALE_1_INPUT_INDEX = 7;
static constexpr uint32_t DEQUANT_SCALE_2_INPUT_INDEX = 8;
static constexpr uint32_t QUANT_SCALE_2_INPUT_INDEX = 9;
static constexpr uint32_t QUANT_OFFSET_2_INPUT_INDEX = 10;
static constexpr uint32_t ANTIQUANT_SCALE_INPUT_INDEX = 11;
static constexpr uint32_t ANTIQUANT_OFFSET_INPUT_INDEX = 12;
static constexpr uint32_t BLOCK_TABLE_INPUT_INDEX = 13;
static constexpr uint32_t KV_PADDING_SIZE_INPUT_INDEX = 14;
static constexpr uint32_t PER_TOKEN_Split_B = 0;
static constexpr uint32_t PER_TOKEN_Split_S = 1;
static constexpr uint32_t BNSD_B_IDX = 0;
static constexpr uint32_t BNSD_N_IDX = 1;
static constexpr uint32_t BNSD_S_IDX = 2;
static constexpr uint32_t BNSD_D_IDX = 3;
static constexpr uint32_t BSND_B_IDX = 0;
static constexpr uint32_t BSND_S_IDX = 1;
static constexpr uint32_t BSND_N_IDX = 2;
static constexpr uint32_t BSND_D_IDX = 3;
static constexpr uint32_t BSH_B_IDX = 0;
static constexpr uint32_t BSH_S_IDX = 1;
static constexpr uint32_t BSH_H_IDX = 2;
static constexpr uint32_t BNSD_NZ_ANTIQUANT_N_IDX = 0;
static constexpr uint32_t BNSD_NZ_ANTIQUANT_S_IDX = 1;
static constexpr uint32_t BNSD_NZ_ANTIQUANT_D_IDX = 2;
static constexpr uint32_t BSND_NZ_ANTIQUANT_N_IDX = 0;
static constexpr uint32_t BSND_NZ_ANTIQUANT_D_IDX = 1;
static constexpr uint32_t BSH_NZ_ANTIQUANT_H_IDX = 0;

static constexpr uint32_t DIM_BH = 2;
static constexpr uint32_t BH_B_IDX = 0;
static constexpr uint32_t BH_H_IDX = 1;
static constexpr uint32_t BND_B_IDX = 0;
static constexpr uint32_t BND_N_IDX = 1;
static constexpr uint32_t BND_D_IDX = 2;
static constexpr uint32_t OUTPUT_INDEX = 0;
static constexpr uint32_t NUM_HEADS_ATTR_INDEX = 0;
static constexpr uint32_t SCALE_VALUE_ATTR_INDEX = 1;
static constexpr uint32_t LAYOUT_ATTR_INDEX = 2;
static constexpr uint32_t KV_NUM_HEADS_ATTR_INDEX = 3;
static constexpr uint32_t BLOCK_SIZE_ATTR_INDEX = 4;
static constexpr uint32_t INNER_PRECISE_ATTR_INDEX = 5;
static constexpr uint32_t FP32_BYTES = 4;
static constexpr uint32_t PSE_SHIFT_B = 0;
static constexpr uint32_t PSE_SHIFT_N = 1;
static constexpr uint32_t PSE_SHIFT_S0 = 2;
static constexpr uint32_t PSE_SHIFT_S1 = 3;
static constexpr uint32_t ITER_NUM = 2;
static constexpr uint32_t HIGH_PRECISION_ITER_NUM = 3; // 高精度场景的迭代次数
static constexpr uint32_t KVINT4_ITER_NUM = 4; // kv int4量化的迭代次数
static constexpr uint32_t IFA_HIGH_PRECISION = 0;
static constexpr uint32_t IFA_HIGH_PERFORMANCE = 1;
static constexpr int64_t MSD_VEC_LOAD = 1024;
static constexpr uint32_t MAX_BLOCK_SIZE = 512;
static constexpr uint32_t COPYND2NZ_SRC_STRIDE_LIMITATION = 65535;

static constexpr uint32_t DEAL_BN2_NUM = 2;
static constexpr uint32_t MAX_CORE_NUM = 50;
static constexpr uint32_t MAX_CORE_NUM_REGBASE = 66;
static constexpr uint32_t MAX_SIZE_BATCH = 256U;
static constexpr uint32_t BYTE_BLOCK = 32;
static constexpr uint32_t KVINT4_BYTE_BLOCK = 64;
static constexpr uint32_t NUM_BYTES_FLOAT = 4;
static constexpr uint32_t NUM_BYTES_FLOAT16 = 2;
static constexpr uint32_t NUM_BYTES_BF16 = 2;
static constexpr uint32_t NUM_BYTES_BOOL = 1;
static constexpr uint32_t NUM_BYTES_INT8 = 1;
static constexpr uint32_t NUM_BYTES_UNDEF = 0;
static constexpr uint32_t MAX_MATMUL_BASE = 512;
static constexpr uint32_t MATMUL_BASE_N = 256;
static constexpr uint32_t MAX_MATMUL_BASE_M = 128;
static constexpr uint32_t MAX_SPLIT_SIZE = 8192;
static constexpr uint32_t L0B_SIZE = 64U * 1024U;
static constexpr uint32_t L0C_SIZE = 128U * 1024U;
static constexpr uint32_t DIM_BNSD = 4;
static constexpr uint32_t DIM_BNSD_OR_BSND = 4;
static constexpr uint32_t DIM_BSH = 3;
static constexpr uint32_t DIM_TND = 3;
static constexpr uint32_t DIM_PER_TOKEN_KvSplit = 2;
static constexpr uint32_t DIM_PER_TOKEN = 3;
static constexpr uint32_t PER_CHANNEL_MODE = 0;
static constexpr uint32_t PER_TOKEN_MODE = 1;
static constexpr uint32_t PER_CHANNEL_TOKEN_MODE = 2;
static constexpr uint32_t DEQUANT_PER_CHANNEL_MODE = 0;
static constexpr uint32_t DEQUANT_PER_TOKEN_MODE = 1;
static constexpr uint32_t DEQUANT_PER_TENSOR_HEAD_MODE = 2;
static constexpr uint32_t DEQUANT_PER_TOKEN_HEAD_MODE = 3;
static constexpr uint32_t DEQUANT_PER_TOKEN_PA_MODE = 4;
static constexpr uint32_t DEQUANT_PER_TOKEN_HEAD_PA_MODE = 5;
static constexpr uint32_t DIM_PER_CHANNEL_BNSD = 4;
static constexpr uint32_t DIM_PER_CHANNEL_BSND = 3;
static constexpr uint32_t DIM_PER_CHANNEL_BSH = 2;
static constexpr uint32_t DIM_PER_CHANNEL_KVNZ_BNSD = 3;
static constexpr uint32_t DIM_PER_CHANNEL_KVNZ_BSND = 2;
static constexpr uint32_t DIM_PER_CHANNEL_KVNZ_BSH = 1;
static constexpr uint32_t DIM_PER_TENSOR = 1;
static constexpr uint32_t BLOCK_SIZE = 16;
static constexpr uint32_t PER_TOKEN_N = 0;
static constexpr uint32_t PER_TOKEN_B = 1;
static constexpr uint32_t PER_TOKEN_S = 2;
static constexpr uint32_t FIA_BALANCE_SG_BASIC_SIZE = 128; // 均衡分核格式，cube M轴的tiling大小
static constexpr uint32_t DIM_B = 0;
static constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
static constexpr uint32_t BLOCKSIZE_CALC_256 = 256;
static constexpr uint32_t WS_TMP_SIZE_PER_CORE = 65536;
static constexpr uint32_t DIM_IDX_1 = 1;
static constexpr uint32_t DIM_IDX_2 = 2;
static constexpr uint32_t DIM_IDX_3 = 3;
static constexpr uint32_t HALF_REDUCE_RATE = 2;
static constexpr uint32_t WS_REPEAT_NUM = 4;
static constexpr uint32_t NUM0 = 0;
static constexpr uint32_t NUM1 = 1;
static constexpr uint32_t NUM2 = 2;
static constexpr uint32_t NUM3 = 3;
static constexpr uint32_t NUM4 = 4;
static constexpr uint32_t NUM5 = 5;
static constexpr uint32_t NUM6 = 6;
static constexpr uint32_t NUM7 = 7;
static constexpr uint32_t NUM8 = 8;
static constexpr uint32_t NUM9 = 9;
static constexpr uint32_t NUM15 = 15;
static constexpr uint32_t NUM16 = 16;
static constexpr uint32_t NUM24 = 24;
static constexpr uint32_t NUM32 = 32;
static constexpr uint32_t NUM64 = 64;
static constexpr uint32_t NUM100 = 100;
static constexpr uint32_t NUM128 = 128;
static constexpr uint32_t NUM256 = 256;
static constexpr uint32_t NUM512 = 512;
static constexpr uint32_t NUM1024 = 1024;
} // namespace optiling
#endif // FUSED_INFER_ATTENTION_SCORE_V2_SINK_TILING_V3