/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include "kernel_operator.h"
#include "lightning_attention_decode.h"

using namespace LightningAttention;

#define COPY_TILING_DATA(tiling) \
    GET_TILING_DATA(tilingDataIn, tiling); \
    const LightningAttentionDecodeTilingData *__restrict tilingData = &tilingDataIn

extern "C" __global__ __aicore__ void lightning_attention_decode(
        GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR slope_rate, GM_ADDR kv_caches_ref_in, GM_ADDR slot_ids,
        GM_ADDR attention_out, GM_ADDR kv_caches_ref_out, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    COPY_TILING_DATA(tiling);
#if (ORIG_DTYPE_QUERY == DT_FLOAT16)
    LightningAttentionDecode<half> op;
    op.Init(query, key, value, slope_rate, kv_caches_ref_in, slot_ids, attention_out, kv_caches_ref_out,
            workspace, tilingData, &pipe);
    op.Process();
#elif (ORIG_DTYPE_QUERY == DT_BF16)
    LightningAttentionDecode<bfloat16_t> op;
    op.Init(query, key, value, slope_rate, kv_caches_ref_in, slot_ids, attention_out, kv_caches_ref_out,
            workspace, tilingData, &pipe);
    op.Process();
#elif (ORIG_DTYPE_QUERY == DT_FLOAT)
    LightningAttentionDecode<float> op;
    const TCubeTiling *__restrict mm1tiling = &(tilingData->mm1TilingData);
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm1, mm1tiling);
    op.Init(query, key, value, slope_rate, kv_caches_ref_in, slot_ids, attention_out, kv_caches_ref_out,
            workspace, tilingData, &pipe);
    op.Process();
#endif
}
