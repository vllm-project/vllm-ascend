/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_TILINGKEY_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_TILINGKEY_H

#include "kernel_tiling/kernel_tiling.h"

// Arch35 (ascend910_93 / ascend950) tiling keys
#define GSA_BASE_TILING 30000

#define GSA_FP16_D128_TILING 30001
#define GSA_BF16_D128_TILING 30002
#define GSA_FP8_D128_TILING 30003
#define GSA_FP8_D128_BF16_TILING 30004

// Arch22 (ascend910b) tiling keys
#define GSA_BASE_ARCH22_TILING 40000

// softmaxPrecision=0: float online-softmax + rescale
#define GSA_FP16_D128_ARCH22_TILING 40001
#define GSA_BF16_D128_ARCH22_TILING 40002
// softmaxPrecision=1: half Softmax + float Rescale (fp16 only; bf16 rejected by host)
#define GSA_FP16_D128_ARCH22_HALFSM_TILING 40005

// returnSoftmaxlse=1 (LseMode::OUT_ONLY). TILING_KEY_IS needs integer literals
// (expression macros like base+offset are dropped from fatbin).
#define GSA_LSE_OUT_OFFSET 100000000
#define GSA_FP16_D128_TILING_LSE_OUT 100030001
#define GSA_BF16_D128_TILING_LSE_OUT 100030002
#define GSA_FP16_D128_ARCH22_TILING_LSE_OUT 100040001
#define GSA_BF16_D128_ARCH22_TILING_LSE_OUT 100040002
#define GSA_FP16_D128_ARCH22_HALFSM_TILING_LSE_OUT 100040005

#endif  // GENERIC_BLOCK_SPARSE_ATTENTION_TILINGKEY_H
