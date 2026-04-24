/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_compress_epilog.cpp
 * \brief KV compress epilog kernel implementation
 */

#include "kv_compress_epilog.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void kv_compress_epilog(
    GM_ADDR kv_compress_cache,
    GM_ADDR x,
    GM_ADDR slot_mapping,
    GM_ADDR kv_compress_cache_out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userspace = GetUserWorkspace(workspace);
    if (userspace == nullptr) {
        return;
    }

    TPipe pipe;

    // Get tiling data
    GET_TILING_DATA_WITH_STRUCT(KvCompressEpilogTilingData, tilingDataIn, tiling);
    const KvCompressEpilogTilingData* __restrict__ tilingData = &tilingDataIn;

    // Dispatch based on tiling key
    if (TILING_KEY_IS(0)) {
        KvCompressEpilogOps::KvCompressEpilogRegBase<DTYPE_X, DTYPE_SLOT_MAPPING, DTYPE_KV_COMPRESS_CACHE> op(&pipe);
        op.Init(x, slot_mapping, kv_compress_cache, tilingData);
        op.Process();
    }
}
