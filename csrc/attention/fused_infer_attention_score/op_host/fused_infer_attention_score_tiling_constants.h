/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_tiling_constants.h
 * \brief
 */

#ifndef FUSED_INFER_ATTENTION_SCORE_TILING_CONSTANTS_H
#define FUSED_INFER_ATTENTION_SCORE_TILING_CONSTANTS_H

#include "../../common/op_host/fia_tiling_info.h"

namespace optiling {
namespace arch35FIA {
// shape limit
constexpr uint32_t B_LIMIT = 65536;
constexpr uint32_t N1_LIMIT = 256;
constexpr uint32_t N2_LIMIT = 256;
constexpr uint32_t G_LIMIT = 64;
constexpr uint32_t D_LIMIT = 512;
constexpr uint32_t T_LIMIT = 1048576;
constexpr uint32_t H_LIMIT = 65535;
constexpr uint32_t S_LIMIT = 20971520;

constexpr uint32_t INPUT_Q_SHAPE_MIN_DIMS = 3;
constexpr uint32_t INPUT_Q_SHAPE_MAX_DIMS = 4;
constexpr uint32_t INPUT_KV_SHAPE_MIN_DIMS = 3;
constexpr uint32_t INPUT_KV_SHAPE_MAX_DIMS = 5;

constexpr uint32_t DIM_NUM_0 = 0;
constexpr uint32_t DIM_NUM_1 = 1;
constexpr uint32_t DIM_NUM_2 = 2;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;
constexpr uint32_t DIM_NUM_5 = 5;

// PA
constexpr uint32_t BLOCK_SIZE_MAX = 512;
constexpr uint32_t BLOCK_SIZE_MAX_FOR_NO_QUANT = 1024;
constexpr uint32_t BLOCK_SIZE_FOR_MXFP8 = 512;
constexpr uint32_t BLOCK_SIZE_1024_FOR_MXFP8 = 1024;
constexpr uint32_t BLOCK_SIZE_ALIGN_SIZE_16 = 16;
constexpr uint32_t BLOCK_SIZE_ALIGN_SIZE_128 = 128;

// sparse mode
// constexpr uint32_t SPARSE_MODE_NO_MASK = 0;
// constexpr uint32_t SPARSE_MODE_ALL_MASK = 1;
// constexpr uint32_t SPARSE_MODE_LEFT_UP = 2;
// constexpr uint32_t SPARSE_MODE_RIGHT_DOWN = 3;
// constexpr uint32_t SPARSE_MODE_BAND = 4;
constexpr uint32_t SPARSE_OPTIMIZE_ATTENTION_SIZE = 2048;

// mask
constexpr uint32_t MASK_DIM_S = 1;
constexpr uint32_t MASK_DIM_SS = 2;
constexpr uint32_t MASK_DIM_BSS = 3;
constexpr uint32_t MASK_DIM_B1SS = 4;

// dequant mode
constexpr uint32_t PER_CHANNEL_MODE = 0;
constexpr uint32_t PER_TOKEN_MODE = 1;
constexpr uint32_t PER_TENSOR_HEAD_MODE = 2;
constexpr uint32_t PER_TOKEN_HEAD_MODE = 3;
constexpr uint32_t PER_TOKEN_PA_MODE = 4;
constexpr uint32_t PER_TOKEN_HEAD_PA_MODE = 5;
constexpr uint32_t PER_TOKEN_GROUP_MODE = 6;
constexpr uint32_t PER_BLOCK_MODE = 7;
constexpr uint32_t PER_CHANNEL_GROUP_MODE = 8;

// ROPE MLA
constexpr uint32_t MLA_QKD_SIZE = 192;
constexpr uint32_t MLA_VD_SIZE = 128;
constexpr uint32_t MLA_D_DIM_512 = 512;
constexpr uint32_t MLA_ROPE_D_DIM_64 = 64;

// inner precise
constexpr uint32_t INNER_PRECISE_LIMIT = 4;
constexpr uint32_t HIGH_PRECISION = 0;
constexpr uint32_t HIGH_PERFORMANCE = 1;

constexpr uint32_t BYTE_BLOCK = 32;
constexpr int64_t SHAPE_PARAMS_CONST = 1;
constexpr int64_t SHAPE_NUM_ONE = 1;

// datatype size
constexpr uint32_t FLOAT16SIZE = 2;
constexpr uint32_t BFLOAT16SIZE = 2;
constexpr uint32_t INT8SIZE = 1;
constexpr uint32_t FLOAT8SIZE = 1;
constexpr float INT4SIZE = 0.5f;
constexpr float FLOAT4SIZE = 0.5f;

constexpr uint32_t SOUTER_32 = 32;
constexpr uint32_t SOUTER_64 = 64;
constexpr uint32_t SOUTER_128 = 128;
constexpr uint32_t SINNER_64 = 64;
constexpr uint32_t SINNER_128 = 128;
constexpr uint32_t SINNER_256 = 256;
constexpr uint32_t SINNER_512 = 512;
constexpr uint32_t DSIZE_64 = 64;
constexpr uint32_t DSIZE_128 = 128;
constexpr uint32_t DSIZE_192 = 192;
constexpr uint32_t DSIZE_256 = 256;
constexpr uint32_t DSIZE_512 = 512;
constexpr uint32_t DSIZE_576 = 576;
constexpr uint32_t CV_RATIO = 2;

constexpr uint32_t DOUBLE_BUFFER_NUM=2;

// num
constexpr uint32_t NUM1 = 1;
constexpr uint32_t NUM8 = 8;
constexpr uint32_t NUM_16 = 16;
constexpr uint32_t NUM_32 = 32;
constexpr uint32_t NUM_64 = 64;
constexpr uint32_t NUM_128 = 128;
constexpr uint32_t NUM_192 = 192;
constexpr uint32_t NUM_256 = 256;
constexpr uint32_t NUM_512 = 512;
constexpr uint32_t NUM_1024 = 1024;
constexpr uint32_t NUM_2048 = 2048;

const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},           // Used to indicate a DataType field has not been set.
    {ge::DT_FLOAT, "DT_FLOAT"},                   // float type
    {ge::DT_FLOAT16, "DT_FLOAT16"},               // fp16 type
    {ge::DT_INT8, "DT_INT8"},                     // int8 type
    {ge::DT_INT16, "DT_INT16"},                   // int16 type
    {ge::DT_UINT16, "DT_UINT16"},                 // uint16 type
    {ge::DT_UINT8, "DT_UINT8"},                   // uint8 type
    {ge::DT_INT32, "DT_INT32"},                   // int32 type
    {ge::DT_INT64, "DT_INT64"},                   // int64 type
    {ge::DT_UINT32, "DT_UINT32"},                 // unsigned int32
    {ge::DT_UINT64, "DT_UINT64"},                 // unsigned int64
    {ge::DT_BOOL, "DT_BOOL"},                     // bool type
    {ge::DT_DOUBLE, "DT_DOUBLE"},                 // double type
    {ge::DT_DUAL, "DT_DUAL"},                     // dual output type
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},   // dual output int8 type
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"}, // dual output uint8 type
    {ge::DT_COMPLEX32, "DT_COMPLEX32"},           // complex32 type
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},           // complex64 type
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},         // complex128 type
    {ge::DT_QINT8, "DT_QINT8"},                   // qint8 type
    {ge::DT_QINT16, "DT_QINT16"},                 // qint16 type
    {ge::DT_QINT32, "DT_QINT32"},                 // qint32 type
    {ge::DT_QUINT8, "DT_QUINT8"},                 // quint8 type
    {ge::DT_QUINT16, "DT_QUINT16"},               // quint16 type
    {ge::DT_RESOURCE, "DT_RESOURCE"},             // resource type
    {ge::DT_STRING_REF, "DT_STRING_REF"},         // string ref type
    {ge::DT_STRING, "DT_STRING"},                 // string type
    {ge::DT_VARIANT, "DT_VARIANT"},               // dt_variant type
    {ge::DT_BF16, "DT_BFLOAT16"},                 // dt_bfloat16 type
    {ge::DT_INT4, "DT_INT4"},
    {ge::DT_UINT1, "DT_UINT1"},
    {ge::DT_INT2, "DT_INT2"},
    {ge::DT_UINT2, "DT_UINT2"},
    {ge::DT_HIFLOAT8, "DT_HIFLOAT8"},
    {ge::DT_FLOAT8_E4M3FN, "DT_FLOAT8_E4M3FN"},
    {ge::DT_FLOAT4_E2M1, "DT_FLOAT4_E2M1"},
    {ge::DT_FLOAT8_E8M0, "DT_FLOAT8_E8M0"}
};

const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME,                  {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8}},
    {KEY_NAME,                    {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT4, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1}},
    {VALUE_NAME,                  {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_INT4, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT4_E2M1}},
    {PSE_SHIFT_NAME,              {ge::DT_FLOAT16, ge::DT_BF16}},
    {ATTEN_MASK_NAME,             {ge::DT_BOOL, ge::DT_INT8, ge::DT_UINT8}},
    {DEQUANT_SCALE1_NAME,         {ge::DT_UINT64, ge::DT_FLOAT}},
    {QUANT_SCALE1_NAME,           {ge::DT_FLOAT}},
    {DEQUANT_SCALE2_NAME,         {ge::DT_UINT64, ge::DT_FLOAT}},
    {QUANT_SCALE2_NAME,           {ge::DT_FLOAT, ge::DT_BF16}},
    {QUANT_OFFSET2_NAME,          {ge::DT_FLOAT, ge::DT_BF16}},
    {ANTIQUANT_SCALE_NAME,        {ge::DT_FLOAT16, ge::DT_BF16}},
    {ANTIQUANT_OFFSET_NAME,       {ge::DT_FLOAT16, ge::DT_BF16}},
    {BLOCK_TABLE_NAME,            {ge::DT_INT32}},
    {KEY_ANTIQUANT_SCALE_NAME,    {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}},
    {KEY_ANTIQUANT_OFFSET_NAME,   {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}},
    {VALUE_ANTIQUANT_SCALE_NAME,  {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}},
    {VALUE_ANTIQUANT_OFFSET_NAME, {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}},
    {KEY_SHARED_PREFIX_NAME,      {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8}},
    {VALUE_SHARED_PREFIX_NAME,    {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8}},
    {QUERY_ROPE_NAME,             {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8}},
    {KEY_ROPE_NAME,               {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8}},
    {DEQUANT_SCALE_QUERY_NAME,    {ge::DT_FLOAT}},
    {ATTEN_OUT_NAME,              {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8}},
    {SOFTMAX_LSE_NAME,            {ge::DT_FLOAT}},
};

const std::set<ge::Format> FORMAT_SUPPORT_SET = {ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NCDHW};
}
} // namespace optiling

#endif // FUSED_INFER_ATTENTION_SCORE_TILING_CONSTANTS_H