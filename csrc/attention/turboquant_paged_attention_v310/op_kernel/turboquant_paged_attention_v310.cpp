/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "turboquant_paged_attention_v310.h"

using namespace AscendC;
using namespace TurboQuantRead;

/*
 * Tiling key = 210 + bits. As on the write side, only the BIT-WIDTH is
 * compile-time (unpack shifts must be constants to unroll); variant and
 * codebook are runtime tiling-data fields, so three instantiations cover all
 * twelve A/B scenarios.
 */
extern "C" __global__ __aicore__ void turboquant_paged_attention_v310(
    GM_ADDR query, GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR key_norms, GM_ADDR value_norms,
    GM_ADDR block_table, GM_ADDR seq_lens, GM_ADDR signs, GM_ADDR centroids, GM_ADDR attn_out,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(TurboquantPagedAttentionV310TilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    TQReadParams params{query, key_cache, value_cache, key_norms, value_norms,
                        block_table, seq_lens, signs, centroids, attn_out};

    if (TILING_KEY_IS(212)) {
        TPipe pipe;
        TQPagedAttention<2> op(&tilingData);
        op.Init(params, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(213)) {
        TPipe pipe;
        TQPagedAttention<3> op(&tilingData);
        op.Init(params, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(214)) {
        TPipe pipe;
        TQPagedAttention<4> op(&tilingData);
        op.Init(params, &pipe);
        op.Process();
    }
}
