/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BF16_TO_FP32_ROUTER_CAST_TILING_H_
#define BF16_TO_FP32_ROUTER_CAST_TILING_H_

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Bf16ToFp32StaticCastTilingData)
TILING_DATA_FIELD_DEF(int64_t, elementNum);
TILING_DATA_FIELD_DEF(int64_t, blockFormer);
TILING_DATA_FIELD_DEF(int64_t, ubFormer);
TILING_DATA_FIELD_DEF(int64_t, useAlignedCopy);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Bf16ToFp32StaticCast, Bf16ToFp32StaticCastTilingData)

struct Bf16ToFp32StaticCastCompileInfo {};
} // namespace optiling

#endif // BF16_TO_FP32_ROUTER_CAST_TILING_H_
