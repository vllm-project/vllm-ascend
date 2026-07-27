/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file indexer_compress_epilog_v2.cpp
 * \brief
 */

#include "indexer_compress_epilog_v2_multi_row.h"
#include "indexer_compress_epilog_v2_single_row.h"

#define SINGLE_ROW_TILING_KEY 20020
#define MULTI_ROW_TILING_KEY 20021

using namespace AscendC;

extern "C" __global__ __aicore__ void indexer_compress_epilog_v2(
    GM_ADDR indexer_compress_cache,
    GM_ADDR x,
    GM_ADDR slot_mapping,
    GM_ADDR indexer_compress_cache_out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    if (TILING_KEY_IS(MULTI_ROW_TILING_KEY)) {
        IndexerCompressEpilogV2::IndexerCompressEpilogV2MultiRow<DTYPE_X, DTYPE_INDEXER_COMPRESS_CACHE> op;
        op.Init(x, slot_mapping, indexer_compress_cache, userWs, &tilingData, &pipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(SINGLE_ROW_TILING_KEY)) {
        IndexerCompressEpilogV2::IndexerCompressEpilogV2SingleRow<DTYPE_X, DTYPE_INDEXER_COMPRESS_CACHE> op;
        op.Init(x, slot_mapping, indexer_compress_cache, userWs, &tilingData, &pipe);
        op.Process();
        return;
    }
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}