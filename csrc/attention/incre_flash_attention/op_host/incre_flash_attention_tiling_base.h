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
 * \file incre_flash_attention_tiling_base.h
 * \brief
 */
#ifndef INCRE_FLASH_ATTENTION_TILING_BASE_H
#define INCRE_FLASH_ATTENTION_TILING_BASE_H

#include <map>


namespace optiling {
using namespace ge;

const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},           // Used to indicate a DataType field has not been set.
    {ge::DT_FLOAT, "DT_FLOAT"},                   // float type
    {ge::DT_FLOAT16, "DT_FLOAT16"},               // fp16 type
    {ge::DT_INT8, "DT_INT8"},                     // int8 type
    {ge::DT_INT16, "DT_INT16"},                   // int16 type
    {ge::DT_UINT16, "DT_UINT16"},                 // uint16 type
    {ge::DT_UINT8, "DT_UINT8"},                   // uint8 type
    {ge::DT_INT32, "DT_INT32"},                   // uint32 type
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
    {ge::DT_FLOAT4_E2M1, "DT_FLOAT4_E2M1"}
};

constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t KEY_INPUT_INDEX = 1;
constexpr uint32_t VALUE_INPUT_INDEX = 2;
constexpr uint32_t PSE_SHIFT_INPUT_INDEX = 3;
constexpr uint32_t ATTEN_MASK_INPUT_INDEX = 4;
constexpr uint32_t ACT_SEQ_LEN_INPUT_INDEX = 5;
constexpr uint32_t DEQUANT_SCALE_1_INPUT_INDEX = 6;
constexpr uint32_t QUANT_SCALE_1_INPUT_INDEX = 7;
constexpr uint32_t DEQUANT_SCALE_2_INPUT_INDEX = 8;
constexpr uint32_t QUANT_SCALE_2_INPUT_INDEX = 9;
constexpr uint32_t QUANT_OFFSET_2_INPUT_INDEX = 10;
constexpr uint32_t ANTIQUANT_SCALE_INPUT_INDEX = 11;
constexpr uint32_t ANTIQUANT_OFFSET_INPUT_INDEX = 12;
constexpr uint32_t BLOCK_TABLE_INPUT_INDEX = 13;
constexpr uint32_t KV_PADDING_SIZE_INPUT_INDEX = 14;
constexpr uint32_t PER_TOKEN_Split_B = 0;
constexpr uint32_t PER_TOKEN_Split_S = 1;
constexpr uint32_t BNSD_B_IDX = 0;
constexpr uint32_t BNSD_N_IDX = 1;
constexpr uint32_t BNSD_S_IDX = 2;
constexpr uint32_t BNSD_D_IDX = 3;
constexpr uint32_t BSND_B_IDX = 0;
constexpr uint32_t BSND_S_IDX = 1;
constexpr uint32_t BSND_N_IDX = 2;
constexpr uint32_t BSND_D_IDX = 3;
constexpr uint32_t BSH_B_IDX = 0;
constexpr uint32_t BSH_S_IDX = 1;
constexpr uint32_t BSH_H_IDX = 2;
constexpr uint32_t BNSD_NZ_ANTIQUANT_N_IDX = 0;
constexpr uint32_t BNSD_NZ_ANTIQUANT_S_IDX = 1;
constexpr uint32_t BNSD_NZ_ANTIQUANT_D_IDX = 2;
constexpr uint32_t BSND_NZ_ANTIQUANT_N_IDX = 0;
constexpr uint32_t BSND_NZ_ANTIQUANT_D_IDX = 1;
constexpr uint32_t BSH_NZ_ANTIQUANT_H_IDX = 0;

constexpr uint32_t DIM_BH = 2;
constexpr uint32_t BH_B_IDX = 0;
constexpr uint32_t BH_H_IDX = 1;
constexpr uint32_t BND_B_IDX = 0;
constexpr uint32_t BND_N_IDX = 1;
constexpr uint32_t BND_D_IDX = 2;
constexpr uint32_t OUTPUT_INDEX = 0;
constexpr uint32_t NUM_HEADS_ATTR_INDEX = 0;
constexpr uint32_t SCALE_VALUE_ATTR_INDEX = 1;
constexpr uint32_t LAYOUT_ATTR_INDEX = 2;
constexpr uint32_t KV_NUM_HEADS_ATTR_INDEX = 3;
constexpr uint32_t BLOCK_SIZE_ATTR_INDEX = 4;
constexpr uint32_t INNER_PRECISE_ATTR_INDEX = 5;
constexpr uint32_t FP32_BYTES = 4;
constexpr uint32_t PSE_SHIFT_B = 0;
constexpr uint32_t PSE_SHIFT_N = 1;
constexpr uint32_t PSE_SHIFT_S0 = 2;
constexpr uint32_t PSE_SHIFT_S1 = 3;
constexpr uint32_t ITER_NUM = 2;
constexpr uint32_t HIGH_PRECISION_ITER_NUM = 3;  // 高精度场景的迭代次数
constexpr uint32_t KVINT4_ITER_NUM = 4;  // kv int4量化的迭代次数
constexpr uint32_t IFA_HIGH_PRECISION = 0;
constexpr uint32_t IFA_HIGH_PERFORMANCE = 1;
constexpr int64_t MSD_VEC_LOAD = 1024;
constexpr uint32_t MAX_BLOCK_SIZE = 512;
constexpr uint32_t COPYND2NZ_SRC_STRIDE_LIMITATION = 65535;

constexpr uint32_t DEAL_BN2_NUM = 2;
constexpr uint32_t MAX_CORE_NUM = 50;
constexpr uint32_t MAX_CORE_NUM_REGBASE = 66;
constexpr uint32_t MAX_SIZE_BATCH = 256U;
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t KVINT4_BYTE_BLOCK = 64;
constexpr uint32_t NUM_BYTES_FLOAT = 4;
constexpr uint32_t NUM_BYTES_FLOAT16 = 2;
constexpr uint32_t NUM_BYTES_BF16 = 2;
constexpr uint32_t NUM_BYTES_BOOL = 1;
constexpr uint32_t NUM_BYTES_INT8 = 1;
constexpr uint32_t NUM_BYTES_UNDEF = 0;
constexpr uint32_t MAX_MATMUL_BASE = 512;
constexpr uint32_t MATMUL_BASE_N = 256;
constexpr uint32_t MAX_MATMUL_BASE_M = 128;
constexpr uint32_t MAX_SPLIT_SIZE = 8192;
constexpr uint32_t L0B_SIZE = 64U * 1024U;
constexpr uint32_t L0C_SIZE = 128U * 1024U;
constexpr uint32_t DIM_BNSD = 4;
constexpr uint32_t DIM_BNSD_OR_BSND = 4;
constexpr uint32_t DIM_BSH = 3;
constexpr uint32_t DIM_TND = 3;
constexpr uint32_t DIM_PER_TOKEN_KvSplit = 2;
constexpr uint32_t DIM_PER_TOKEN = 3;
constexpr uint32_t PER_CHANNEL_MODE = 0;
constexpr uint32_t PER_TOKEN_MODE = 1;
constexpr uint32_t PER_CHANNEL_TOKEN_MODE = 2;
constexpr uint32_t DEQUANT_PER_CHANNEL_MODE = 0;
constexpr uint32_t DEQUANT_PER_TOKEN_MODE = 1;
constexpr uint32_t DEQUANT_PER_TENSOR_HEAD_MODE = 2;
constexpr uint32_t DEQUANT_PER_TOKEN_HEAD_MODE = 3;
constexpr uint32_t DEQUANT_PER_TOKEN_PA_MODE = 4;
constexpr uint32_t DEQUANT_PER_TOKEN_HEAD_PA_MODE = 5;
constexpr uint32_t DIM_PER_CHANNEL_BNSD = 4;
constexpr uint32_t DIM_PER_CHANNEL_BSND = 3;
constexpr uint32_t DIM_PER_CHANNEL_BSH = 2;
constexpr uint32_t DIM_PER_CHANNEL_KVNZ_BNSD = 3;
constexpr uint32_t DIM_PER_CHANNEL_KVNZ_BSND = 2;
constexpr uint32_t DIM_PER_CHANNEL_KVNZ_BSH = 1;
constexpr uint32_t DIM_PER_TENSOR = 1;
constexpr uint32_t BLOCK_SIZE = 16;
constexpr uint32_t PER_TOKEN_N = 0;
constexpr uint32_t PER_TOKEN_B = 1;
constexpr uint32_t PER_TOKEN_S = 2;
constexpr uint32_t FIA_BALANCE_SG_BASIC_SIZE = 128;   // 均衡分核格式，cube M轴的tiling大小
constexpr uint32_t DIM_B = 0;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t BLOCKSIZE_CALC_256 = 256;
constexpr uint32_t WS_TMP_SIZE_PER_CORE = 65536;
constexpr uint32_t DIM_IDX_1 = 1;
constexpr uint32_t DIM_IDX_2 = 2;
constexpr uint32_t DIM_IDX_3 = 3;
constexpr uint32_t HALF_REDUCE_RATE = 2;
constexpr uint32_t WS_REPEAT_NUM = 4;
constexpr uint32_t NUM0 = 0;
constexpr uint32_t NUM1 = 1;
constexpr uint32_t NUM2 = 2;
constexpr uint32_t NUM3 = 3;
constexpr uint32_t NUM4 = 4;
constexpr uint32_t NUM5 = 5;
constexpr uint32_t NUM6 = 6;
constexpr uint32_t NUM7 = 7;
constexpr uint32_t NUM8 = 8;
constexpr uint32_t NUM9 = 9;
constexpr uint32_t NUM15 = 15;
constexpr uint32_t NUM16 = 16;
constexpr uint32_t NUM24 = 24;
constexpr uint32_t NUM32 = 32;
constexpr uint32_t NUM64 = 64;
constexpr uint32_t NUM100 = 100;
constexpr uint32_t NUM128 = 128;
constexpr uint32_t NUM256 = 256;
constexpr uint32_t NUM512 = 512;
constexpr uint32_t NUM1024 = 1024;
}// namespace optiling


#endif // INCRE_FLASH_ATTENTION_TILING_BASE_H