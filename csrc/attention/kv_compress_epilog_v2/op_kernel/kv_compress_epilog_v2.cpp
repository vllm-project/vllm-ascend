/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kv_compress_epilog_v2_kernel.h"
#include "kv_compress_epilog_v2_layout_2d.h"
#include "kv_compress_epilog_v2_layout_4d.h"

using namespace AscendC;

// TilingKey 分发矩阵：百位 0/1 → 2d/4d 布局；个位 0/1/2 → FP8-g32/FP4-g32/FP4-g16。
extern "C" __global__ __aicore__ void kv_compress_epilog_v2(
    GM_ADDR cache, GM_ADDR x, GM_ADDR slot_mapping, GM_ADDR cache_out, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr) {
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(KvCompressEpilogV2TilingData, tilingDataValue, tiling);
    const KvCompressEpilogV2TilingData *tilingData = &tilingDataValue;
    TPipe pipe;
    const int64_t overflowMode =
        AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    if (TILING_KEY_IS(2000)) {
        KvCompressEpilogV2Ops::KvCompressEpilogV2Layout2dFp8Kernel<
            DTYPE_X, DTYPE_SLOT_MAPPING, DTYPE_CACHE> op(&pipe);
        op.Init(cache, x, slot_mapping, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2001)) {
        KvCompressEpilogV2Ops::KvCompressEpilogV2Layout2dFp4G32Kernel<
            DTYPE_X, DTYPE_SLOT_MAPPING> op(&pipe);
        op.Init(cache, x, slot_mapping, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2002)) {
        KvCompressEpilogV2Ops::KvCompressEpilogV2Layout2dFp4G16Kernel<
            DTYPE_X, DTYPE_SLOT_MAPPING> op(&pipe);
        op.Init(cache, x, slot_mapping, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2100)) {
        KvCompressEpilogV2Ops::KvCompressEpilogV2Layout4dFp8Kernel<
            DTYPE_X, DTYPE_SLOT_MAPPING, DTYPE_CACHE> op(&pipe);
        op.Init(cache, x, slot_mapping, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2101)) {
        KvCompressEpilogV2Ops::KvCompressEpilogV2Layout4dFp4G32Kernel<
            DTYPE_X, DTYPE_SLOT_MAPPING> op(&pipe);
        op.Init(cache, x, slot_mapping, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2102)) {
        KvCompressEpilogV2Ops::KvCompressEpilogV2Layout4dFp4G16Kernel<
            DTYPE_X, DTYPE_SLOT_MAPPING> op(&pipe);
        op.Init(cache, x, slot_mapping, tilingData);
        op.Process();
    }
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(overflowMode);
}
