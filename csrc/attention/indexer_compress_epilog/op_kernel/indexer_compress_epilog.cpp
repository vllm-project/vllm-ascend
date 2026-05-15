/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_clip_quant.cpp
 * \brief
 */

#include "indexer_compress_epilog_multi_row.h"
#include "indexer_compress_epilog_single_row.h"
#include "indexer_compress_epilog_multi_row_mx_fp8.h"
#include "indexer_compress_epilog_single_row_mx_fp8.h"

#define SINGLE_ROW_NORMAL_QUANT 10001
#define SINGLE_ROW_MXFP8_QUANT 10000
#define MULTI_ROW_NORMAL_QUANT 10011
#define MULTI_ROW_MXFP8_QUANT 10010

using namespace AscendC;

extern "C" __global__ __aicore__ void indexer_compress_epilog(GM_ADDR x, GM_ADDR slot_mapping, GM_ADDR indexer_compress_cache, 
                                                              GM_ADDR indexer_compress_cache_scale, GM_ADDR workspace, GM_ADDR tiling)
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
    if (TILING_KEY_IS(MULTI_ROW_NORMAL_QUANT)) {
        IndexerCompressEpilog::IndexerCompressEpilogMultiRow<DTYPE_X, DTYPE_INDEXER_COMPRESS_CACHE, DTYPE_INDEXER_COMPRESS_CACHE_SCALE> op;
        op.Init(x, slot_mapping, indexer_compress_cache, indexer_compress_cache_scale, userWs, &tilingData, &pipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(SINGLE_ROW_NORMAL_QUANT)) {
        IndexerCompressEpilog::IndexerCompressEpilogSingleRow<DTYPE_X, DTYPE_INDEXER_COMPRESS_CACHE, DTYPE_INDEXER_COMPRESS_CACHE_SCALE> op;
        op.Init(x, slot_mapping, indexer_compress_cache, indexer_compress_cache_scale, userWs, &tilingData, &pipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(MULTI_ROW_MXFP8_QUANT)){
        IndexerCompressEpilog::IndexerCompressEpilogMultiRowMxFp8<DTYPE_X, DTYPE_INDEXER_COMPRESS_CACHE, DTYPE_INDEXER_COMPRESS_CACHE_SCALE> op;
        op.Init(x, slot_mapping, indexer_compress_cache, indexer_compress_cache_scale, userWs, &tilingData, &pipe);
        op.Process();
        return;
    } else if (TILING_KEY_IS(SINGLE_ROW_MXFP8_QUANT)){
        IndexerCompressEpilog::IndexerCompressEpilogSingleRowMxFp8<DTYPE_X, DTYPE_INDEXER_COMPRESS_CACHE, DTYPE_INDEXER_COMPRESS_CACHE_SCALE> op;
        op.Init(x, slot_mapping, indexer_compress_cache, indexer_compress_cache_scale, userWs, &tilingData, &pipe);
        op.Process();
        return;
    } 
}