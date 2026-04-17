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
 * \file scatter_pa_kv_cache_vllm.cpp
 * \brief scatter_pa_kv_cache_vllm
 */

#include "scatter_pa_kv_cache_vllm_normal_nz_fully_load.h"
#include "scatter_pa_kv_cache_vllm_normal_nz_not_fully_load.h"

#include "scatter_pa_kv_cache_vllm_normal.h"
#include "scatter_pa_kv_cache_vllm_compress_rope.h"
#include "scatter_pa_kv_cache_vllm_compress_alibi.h"
#include "scatter_pa_kv_cache_vllm_normal_siso.h"
#include "scatter_pa_kv_cache_vllm_compress_omni.h"
#include "scatter_pa_kv_cache_vllm_nhsd.h"

using namespace AscendC;
// norm
#define NORMAL_FULLY_LOAD 1001
#define NORMAL_NOT_FULLY_LOAD 1000
// nz
#define NZ_FULLY_LOAD 2001
#define NZ_NOT_FULLY_LOAD 2000
// alibi
#define ALIBI 3000
// rope
#define ROPE 4000
// siso
#define SISO_FULLY_LOAD 5001
#define SISO_NOT_FULLY_LOAD 5000
// omni
#define OMNI 6000
// norm_nct
#define NORM_NCT_FULLY_LOAD 7001
#define NORM_NCT_NOT_FULLY_LOAD 7000
// siso_nct
#define SISO_NCT_FULLY_LOAD 8001
#define SISO_NCT_NOT_FULLY_LOAD 8000
// NHSD
#define NHSD 9000

extern "C" __global__ __aicore__ void scatter_pa_kv_cache_vllm(GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping,
                                                          GM_ADDR value, GM_ADDR value_cache_in, GM_ADDR compress_lens,
                                                          GM_ADDR compress_seq_offset, GM_ADDR seq_lens,
                                                          GM_ADDR key_cache_out, GM_ADDR value_cache_out,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    // norm
    if (TILING_KEY_IS(NORMAL_FULLY_LOAD)) {
        ScatterPaKvCacheVllm::ScatterPaKvCacheVllmNormal<DTYPE_KEY, DTYPE_VALUE, DTYPE_SLOT_MAPPING> op(&pipe, &tilingData);
        op.Init(key, value, slot_mapping, key_cache_out, value_cache_out);
        op.ProcessFullyLoad();
    } else if (TILING_KEY_IS(NORMAL_NOT_FULLY_LOAD)) {
        ScatterPaKvCacheVllm::ScatterPaKvCacheVllmNormal<DTYPE_KEY, DTYPE_VALUE, DTYPE_SLOT_MAPPING> op(&pipe, &tilingData);
        op.Init(key, value, slot_mapping, key_cache_out, value_cache_out);
        op.Process();
    }
    // nz
    else if (TILING_KEY_IS(NZ_FULLY_LOAD)) {
        ScatterPaKvCacheVllm::ScatterPaKvCacheVllmNormalNzFullyLoad<DTYPE_KEY, DTYPE_VALUE, DTYPE_SLOT_MAPPING> op(&pipe, &tilingData);
        op.Init(key, slot_mapping, value, key_cache_out, value_cache_out);
        op.Process();
    } else if (TILING_KEY_IS(NZ_NOT_FULLY_LOAD)) {
        ScatterPaKvCacheVllm::ScatterPaKvCacheVllmNormalNzNotFullyLoad<DTYPE_KEY, DTYPE_VALUE, DTYPE_SLOT_MAPPING> op(&pipe, &tilingData);
        op.Init(key, slot_mapping, value, key_cache_out, value_cache_out);
        op.Process();
    }
    // alibi
    else if (TILING_KEY_IS(ALIBI)) {
        ScatterPaKvCacheVllm::ScatterPaKvCacheVllmCompressAlibi<DTYPE_KEY> op;
        op.Init(&pipe, &tilingData);
        op.Process(key, value, key_cache_in, value_cache_in, slot_mapping, compress_lens, seq_lens, key_cache_out,
                   value_cache_out);
    }
#if (ORIG_DTYPE_KEY == DT_FLOAT16 || ORIG_DTYPE_KEY == DT_BF16)
    // rope
    else if (TILING_KEY_IS(ROPE)) {
        ScatterPaKvCacheVllm::ScatterPaKvCacheVllmCompressRope<DTYPE_KEY> op;
        op.Init(&pipe, &tilingData);
        op.Method(key, value, key_cache_in, value_cache_in, slot_mapping, compress_lens, seq_lens, compress_seq_offset,
                  key_cache_out, value_cache_out);
    }
    // omni
    else if (TILING_KEY_IS(OMNI)) {
        ScatterPaKvCacheVllm::ScatterPaKvCacheVllmCompressOmni<DTYPE_KEY> op;
        op.Init(&pipe, &tilingData);
        op.Method(key, value, key_cache_in, value_cache_in, slot_mapping, compress_lens, seq_lens, compress_seq_offset,
                  key_cache_out, value_cache_out);
    } else if (TILING_KEY_IS(NHSD)) {
        ScatterPaKvCacheVllm::ScatterPaKvCacheVllmNHSD<DTYPE_KEY, DTYPE_SLOT_MAPPING> op(&pipe);
        op.Init(key, value, slot_mapping, key_cache_out, value_cache_out, &tilingData);
        op.Process();
    }
#endif
    // siso
    else if (TILING_KEY_IS(SISO_FULLY_LOAD)) {
        ScatterPaKvCacheVllm::ScatterPaKvCacheVllmNormalSiso<DTYPE_KEY, DTYPE_SLOT_MAPPING> op(&pipe, &tilingData);
        op.Init(key, slot_mapping, key_cache_out);
        op.ProcessFullyLoad();
    } else if (TILING_KEY_IS(SISO_NOT_FULLY_LOAD)) {
        ScatterPaKvCacheVllm::ScatterPaKvCacheVllmNormalSiso<DTYPE_KEY, DTYPE_SLOT_MAPPING> op(&pipe, &tilingData);
        op.Init(key, slot_mapping, key_cache_out);
        op.Process();
    }
    // norm_nct
    else if TILING_KEY_IS (NORM_NCT_FULLY_LOAD) {
        ScatterPaKvCacheVllm::ScatterPaKvCacheVllmNormal<DTYPE_KEY, DTYPE_VALUE, DTYPE_SLOT_MAPPING, true> op(&pipe, &tilingData);
        op.Init(key, value, slot_mapping, key_cache_out, value_cache_out);
        op.ProcessFullyLoad();
    } else if TILING_KEY_IS (NORM_NCT_NOT_FULLY_LOAD) {
        ScatterPaKvCacheVllm::ScatterPaKvCacheVllmNormal<DTYPE_KEY, DTYPE_VALUE, DTYPE_SLOT_MAPPING, true> op(&pipe, &tilingData);
        op.Init(key, value, slot_mapping, key_cache_out, value_cache_out);
        op.Process();
    }
    // siso_nct
    else if (TILING_KEY_IS(SISO_NCT_FULLY_LOAD)) {
        ScatterPaKvCacheVllm::ScatterPaKvCacheVllmNormalSiso<DTYPE_KEY, DTYPE_SLOT_MAPPING, true> op(&pipe, &tilingData);
        op.Init(key, slot_mapping, key_cache_out);
        op.ProcessFullyLoad();
    } else if (TILING_KEY_IS(SISO_NCT_NOT_FULLY_LOAD)) {
        ScatterPaKvCacheVllm::ScatterPaKvCacheVllmNormalSiso<DTYPE_KEY, DTYPE_SLOT_MAPPING, true> op(&pipe, &tilingData);
        op.Init(key, slot_mapping, key_cache_out);
        op.Process();
    }
}
