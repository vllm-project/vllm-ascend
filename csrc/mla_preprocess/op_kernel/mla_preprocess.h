// Adapted from
//   https://gitee.com/ascend/ascend-transformer-boost
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//

#ifndef __MLA_PREPROCESS_H__
#define __MLA_PREPROCESS_H__

// sync
constexpr int32_t QUANT1 = 1;
constexpr int32_t MM1 = 2;
constexpr int32_t MM1QUANT = 3;
constexpr int32_t RMSNORMQUANT2 = 4;
constexpr int32_t MM2 = 5;
constexpr int32_t MM2QUANT = 6;
constexpr int32_t BMM3 = 7;
constexpr int32_t BMM3SPLIT = 8;
constexpr int32_t MM2OUT = 9;
constexpr int32_t EINSUMOUT = 11;
constexpr int32_t EINSUMQUANT = 12;

// ropeConcat
constexpr uint32_t ELE_NUM_FP16 = 16;         // nums of fp16 elements in one block
constexpr uint32_t ELE_NUM_FP32 = 8;          // nums of fp32 elements in one block
constexpr uint8_t DEFAULT_REPEAT_STRIDE = 8;  // stride, 8 * 32 = 256

// rmsNormQuant
constexpr int32_t NUM_PER_REP_FP32 = 64;  // ONE_REPEAT_BYTE_SIZE / sizeof(float);
constexpr float ZERO = 0;
constexpr uint32_t BUF_FACTOR = 3;         // 1(g) + 1(sqx) + 1(sum) = 3
constexpr uint32_t OFFSET_GAMMA = 0;       // the offset of gamma is 0
constexpr uint32_t OFFSET_SQX = 1;         // the offset of sqx is 1
constexpr uint32_t OFFSET_SUM = 2;         // the offset of sum is 2
constexpr uint32_t OFFSET_WORKSPACE = 3;   // the offset of workspace is 3
constexpr uint32_t REPEAT_TIME_256 = 256;  // 128 default stride
constexpr uint32_t REPEAT_TIME_128 = 128;  // 128 default stride
constexpr uint32_t REPEAT_TIME_64 = 64;    // 64 default stride

constexpr uint8_t CACHE_MODE_KVCACHE = 0;       // single input single output
constexpr uint8_t CACHE_MODE_KROPE_CTKV = 1;    // double in and double out
constexpr uint8_t CACHE_MODE_INT8_NZCACHE = 2;  // high performance KV NZ format/quant int8
constexpr uint8_t CACHE_MODE_NZCACHE = 3;

// pp matmul
constexpr uint32_t HIDDTEN_STATE = 7168;
constexpr uint32_t FLOAT_BLOCK_SIZE = 64;
constexpr uint32_t HALF_BLOCK_SIZE = 64;
constexpr uint32_t HALF_VECTOR_SIZE = 64;
constexpr uint32_t MM1_OUT_SIZE = 2112;
constexpr uint32_t SPLIT_SIZE_ONE = 576;
constexpr uint32_t SPLIT_SIZE_TWO = 1536;
constexpr uint32_t SPLIT_RMSNRORM_SIZE_ONE = 512;
constexpr uint32_t SPLIT_RMSNRORM_SIZE_TWO = 64;
constexpr uint32_t ROPE_SPLIT_SIZE_ONE = 64;
constexpr uint32_t ROPE_SPLIT_SIZE_TWO = 128;

constexpr uint32_t MMSIZE1 = 128 * 192;  // 24576
constexpr uint32_t MMSIZE2 = 64 * 128;   // 8192

constexpr uint64_t L0_PINGPONG_BUFFER_LEN = 32768;   // 32 KB
constexpr uint64_t L1_PINGPONG_BUFFER_LEN = 262144;  // 256 KB
constexpr uint64_t BLOCK_SIZE_16 = 16;
constexpr uint64_t BLOCK_SIZE_32 = 32;
constexpr uint64_t CUBE_MATRIX_SIZE_512 = 16 * 32;  // 16 * 23
constexpr uint64_t FB_BUFF_SIZE = 1024 * 7;
constexpr uint64_t SCALE_L1_LEN = 4096;
constexpr uint64_t BIAS_L1_LEN = 2048;

constexpr uint64_t CONST_0 = 0;
constexpr uint64_t CONST_4 = 4;
constexpr uint64_t CONST_8 = 8;
constexpr uint64_t CONST_32 = 32;
constexpr uint64_t CONST_64 = 64;
constexpr uint64_t CONST_128 = 128;

// ropeConcat
constexpr uint32_t ROPE_CONCAT_NUM_BUFFER = 2;

// rmsNormQuant
constexpr uint32_t OFFSET_ABS = 3;             // the offset of abs is 3
constexpr uint32_t OFFSET_WORKSPACE_BF16 = 4;  // the offset of workspace is 4

// sync bf16
constexpr int32_t AIC_MM1_START = 2;
constexpr int32_t AIC_MM3_START = 3;
constexpr int32_t AIC_MM2_START = 6;
constexpr int32_t MMAIC = 7;
constexpr int32_t MMAIV = 8;

constexpr uint32_t MAX_HW_SYNC_COUNTER = 5;
constexpr uint32_t SYNC_MODE = 2;

// TilingKey
constexpr uint32_t KEY_FP16_CACHEMODE_0_QUANTMODE_0 = 0;
constexpr uint32_t KEY_FP16_CACHEMODE_1_QUANTMODE_0 = 1;
constexpr uint32_t KEY_BF16_CACHEMODE_0_QUANTMODE_0 = 256;
constexpr uint32_t KEY_BF16_CACHEMODE_1_QUANTMODE_0 = 257;
constexpr uint32_t KEY_BF16_CACHEMODE_3_QUANTMODE_0 = 259;
constexpr uint32_t KEY_BF16_CACHEMODE_0_QUANTMODE_0_INNER = 256 + 512;
constexpr uint32_t KEY_BF16_CACHEMODE_1_QUANTMODE_0_INNER = 257 + 512;
constexpr uint32_t KEY_BF16_CACHEMODE_3_QUANTMODE_0_INNER = 259 + 512;

enum class QuantMode : int32_t {
    PER_TENSOR_ASYMM_QUANT = 0,
    PER_TOKEN_SYMM_QUANT,
    PER_TOKEN_ASYMM_QUANT,
    NO_QUANT,
};

#endif  // __MLA_PREPROCESS_H__
