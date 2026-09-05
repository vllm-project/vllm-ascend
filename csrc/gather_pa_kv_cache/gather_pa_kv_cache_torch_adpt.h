/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef VLLM_ASCEND_GATHER_PA_KV_CACHE_TORCH_ADPT_H
#define VLLM_ASCEND_GATHER_PA_KV_CACHE_TORCH_ADPT_H

#include "op_host/gather_pa_kv_cache.h"

namespace vllm_ascend {

void gather_pa_kv_cache(const at::Tensor &keyCache, const at::Tensor &valueCache,
                        const at::Tensor &blockTables, const at::Tensor &seqLens,
                        at::Tensor &key, at::Tensor &value,
                        const c10::optional<at::Tensor> &seqOffset, c10::string_view cacheMode,
                        bool isSeqLensCumsum)
{
    auto [tilingPtr, blockDim] = ::gather_pa_kv_cache::gather_pa_kv_cache_tiling(
        keyCache, valueCache, blockTables, seqLens, key, value, seqOffset, cacheMode, isSeqLensCumsum);

    void *keyCachePtr = keyCache.data_ptr();
    void *valueCachePtr = valueCache.data_ptr();
    void *blockTablesPtr = blockTables.data_ptr();
    void *seqLensPtr = seqLens.data_ptr();
    void *seqOffsetPtr = seqOffset.has_value() ? seqOffset.value().data_ptr() : seqLensPtr;
    void *keyPtr = key.data_ptr();
    void *valuePtr = value.data_ptr();

    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
    at_npu::native::OpCommand cmd;
    cmd.Name("gather_pa_kv_cache");
    cmd.SetCustomHandler([stream, keyCachePtr, valueCachePtr, blockTablesPtr, seqLensPtr, seqOffsetPtr, keyPtr,
                          valuePtr, tilingPtr, blockDim]() -> int {
        gather_pa_kv_cache_impl(stream, keyCachePtr, valueCachePtr, blockTablesPtr, seqLensPtr, seqOffsetPtr,
                                keyPtr, valuePtr, tilingPtr, blockDim);
        return 0;
    });
    cmd.Run();
}

}  // namespace vllm_ascend

#endif  // VLLM_ASCEND_GATHER_PA_KV_CACHE_TORCH_ADPT_H
