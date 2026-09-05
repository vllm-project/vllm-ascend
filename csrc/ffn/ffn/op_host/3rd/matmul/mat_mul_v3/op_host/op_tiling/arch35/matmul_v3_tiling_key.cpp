/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file matmul_v3_tiling_key.cc
 * \brief
 */

#include "../../../op_kernel/arch35/mat_mul_tiling_key.h"
#include "matmul_v3_tiling_key.h"

using namespace optiling::matmul_v3_advanced;

uint64_t MatMulV3TilingKey::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(apiLevel_), static_cast<uint64_t>(atrans_),
                              static_cast<uint64_t>(btrans_), static_cast<uint64_t>(batchModel_),
                              static_cast<uint64_t>(model_), static_cast<uint64_t>(fullLoad_),
                              static_cast<uint64_t>(out_));
}
