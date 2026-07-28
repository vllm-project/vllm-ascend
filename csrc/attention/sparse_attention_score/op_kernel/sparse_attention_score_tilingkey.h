/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPARSE_ATTENTION_SCORE_TILINGKEY_H
#define SPARSE_ATTENTION_SCORE_TILINGKEY_H

#include "kernel_tiling/kernel_tiling.h"

#define SASA_BASE_TILING 10000

#define SASA_FP16_D128_TILING 10001
#define SASA_BF16_D128_TILING 10002
#define SASA_FP8_D128_TILING 10003
#define SASA_FP16_D64_TILING 10004
#define SASA_BF16_D64_TILING 10005


#define SASA_FP16_D128_ARCH22_TILING  20001
#define SASA_BF16_D128_ARCH22_TILING  20002

#endif  // SPARSE_ATTENTION_SCORE_TILINGKEY_H
