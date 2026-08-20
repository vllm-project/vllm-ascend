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
#include "turboquant_reshape_and_cache_v310.h"

using namespace AscendC;
using namespace TurboQuantWrite;

/*
 * Tiling key = 200 + bits. Only the BIT-WIDTH is compile-time: the pack shift
 * amounts must be constants to unroll and LEVELS sizes the codebook buffer.
 * The variant (MSE / MSE+QJL) and codebook (uniform / Lloyd-Max) selectors are
 * runtime tiling-data fields, so all twelve A/B scenarios are reachable from
 * these three instantiations instead of twelve.
 */
extern "C" __global__ __aicore__ void turboquant_reshape_and_cache_v310(
    GM_ADDR key, GM_ADDR value, GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR slot_mapping,
    GM_ADDR signs, GM_ADDR centroids, GM_ADDR key_norms, GM_ADDR value_norms, GM_ADDR workspaceGM,
    GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(TurboquantReshapeAndCacheV310TilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    TQWriteParams params{key, value, key_cache, value_cache, slot_mapping,
                         key_norms, value_norms, signs, centroids};

    if (TILING_KEY_IS(202)) {
        TPipe pipe;
        TQReshapeAndCache<2> op(&tilingData);
        op.Init(params, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(203)) {
        TPipe pipe;
        TQReshapeAndCache<3> op(&tilingData);
        op.Init(params, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(204)) {
        TPipe pipe;
        TQReshapeAndCache<4> op(&tilingData);
        op.Init(params, &pipe);
        op.Process();
    }
}
