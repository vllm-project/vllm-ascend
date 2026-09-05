/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef VLLM_ASCEND_GATHER_PA_KV_CACHE_HOST_H
#define VLLM_ASCEND_GATHER_PA_KV_CACHE_HOST_H

#include <algorithm>
#include <climits>
#include <string>
#include <tuple>

#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "gather_pa_kv_cache_tiling.h"

namespace gather_pa_kv_cache {

constexpr int32_t DATA_BLOCK_SIZE = 32;
constexpr int32_t MAX_CAPTURE_BATCHES = 1024;
constexpr int32_t NZ_TMP_BUF_SIZE = 148 * 1024;
constexpr int32_t NZ_LOCATE_INFO_SIZE = 40 * 1024;
constexpr int32_t NZ_LOCATE_INFO_PADDING = 2 * DATA_BLOCK_SIZE;
constexpr int32_t NZ_MAX_BLOCK_TABLE_WIDTH =
    (NZ_LOCATE_INFO_SIZE - NZ_LOCATE_INFO_PADDING) / static_cast<int32_t>(sizeof(int32_t)) - 1;
constexpr int32_t NZ_MAX_BLOCK_SIZE = 65536;

inline bool IsSupportedNdDtype(at::ScalarType dtype)
{
    return dtype == at::kHalf || dtype == at::kBFloat16 || dtype == at::kChar;
}

inline void CheckCacheLayout(const at::Tensor &cache, const char *name)
{
    TORCH_CHECK(cache.dim() == 4, name, " must be a 4D tensor");
    TORCH_CHECK(cache.size(0) > 0 && cache.size(1) > 0 && cache.size(2) > 0 && cache.size(3) > 0,
                name, " dimensions must be positive");
    int64_t expectedStride = 1;
    for (int64_t dim = cache.dim() - 1; dim >= 1; --dim) {
        TORCH_CHECK(cache.stride(dim) == expectedStride,
                    name, " must be contiguous in every dimension except the first");
        expectedStride *= cache.size(dim);
    }
    TORCH_CHECK(cache.stride(0) >= expectedStride, name, " has an invalid first-dimension stride");
}

inline void CheckMetadataTensor(const at::Tensor &tensor, const char *name, int64_t dim)
{
    TORCH_CHECK(tensor.dim() == dim, name, " must be a ", dim, "D tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kInt, name, " must have dtype int32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline void CheckSameDevice(const at::Tensor &reference, const at::Tensor &tensor, const char *name)
{
    TORCH_CHECK(tensor.device() == reference.device(), name, " must be on the same device as key_cache");
}

inline std::tuple<void *, uint32_t> gather_pa_kv_cache_tiling(
    const at::Tensor &keyCache, const at::Tensor &valueCache, const at::Tensor &blockTables,
    const at::Tensor &seqLens, at::Tensor &key, at::Tensor &value,
    const c10::optional<at::Tensor> &seqOffset, c10::string_view cacheMode, bool isSeqLensCumsum)
{
    const std::string cacheModeString(cacheMode.data(), cacheMode.size());
    const bool isNz = cacheModeString == "PA_NZ";
    TORCH_CHECK(cacheModeString == "Norm" || isNz,
                "cache_mode must be either \"Norm\" or \"PA_NZ\", but got ", cacheModeString);

    CheckCacheLayout(keyCache, "key_cache");
    CheckCacheLayout(valueCache, "value_cache");
    CheckMetadataTensor(blockTables, "block_tables", 2);
    CheckMetadataTensor(seqLens, "seq_lens", 1);
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");

    CheckSameDevice(keyCache, valueCache, "value_cache");
    CheckSameDevice(keyCache, blockTables, "block_tables");
    CheckSameDevice(keyCache, seqLens, "seq_lens");
    CheckSameDevice(keyCache, key, "key");
    CheckSameDevice(keyCache, value, "value");

    TORCH_CHECK(keyCache.scalar_type() == valueCache.scalar_type(),
                "key_cache and value_cache must have the same dtype");
    TORCH_CHECK(keyCache.scalar_type() == key.scalar_type(), "key and key_cache must have the same dtype");
    TORCH_CHECK(valueCache.scalar_type() == value.scalar_type(), "value and value_cache must have the same dtype");
    TORCH_CHECK(keyCache.size(0) == valueCache.size(0), "key_cache and value_cache must have the same block count");
    TORCH_CHECK(key.size(0) == value.size(0), "key and value must have the same token count");
    TORCH_CHECK(key.size(0) <= INT_MAX, "key and value token count must not exceed INT_MAX");

    int64_t blockSize;
    int64_t tokenSizeK;
    int64_t tokenSizeV;
    const int64_t typeByte = keyCache.element_size();
    if (isNz) {
        TORCH_CHECK(keyCache.scalar_type() == at::kChar,
                    "PA_NZ mode only supports int8 on non-Ascend-950 hardware");
        TORCH_CHECK(key.dim() == 2, "key must be a 2D tensor in PA_NZ mode");
        TORCH_CHECK(value.dim() == 2, "value must be a 2D tensor in PA_NZ mode");
        TORCH_CHECK(keyCache.size(3) == DATA_BLOCK_SIZE && valueCache.size(3) == DATA_BLOCK_SIZE,
                    "PA_NZ cache innermost dimension must contain 32 int8 elements");

        blockSize = keyCache.size(2);
        TORCH_CHECK(valueCache.size(2) == blockSize, "key_cache and value_cache must have the same block size");
        TORCH_CHECK(keyCache.size(1) * keyCache.size(3) == key.size(1),
                    "key shape must match the PA_NZ key_cache token size");
        TORCH_CHECK(valueCache.size(1) * valueCache.size(3) == value.size(1),
                    "value shape must match the PA_NZ value_cache token size");
        tokenSizeK = key.size(1);
        tokenSizeV = value.size(1);
        TORCH_CHECK(tokenSizeK * typeByte <= NZ_TMP_BUF_SIZE && tokenSizeV * typeByte <= NZ_TMP_BUF_SIZE,
                    "PA_NZ token size must not exceed ", NZ_TMP_BUF_SIZE, " bytes");
        TORCH_CHECK(blockSize <= NZ_MAX_BLOCK_SIZE,
                    "PA_NZ block size must not exceed ", NZ_MAX_BLOCK_SIZE);
        TORCH_CHECK(blockTables.size(1) <= NZ_MAX_BLOCK_TABLE_WIDTH,
                    "PA_NZ block_tables width must not exceed ", NZ_MAX_BLOCK_TABLE_WIDTH);
    } else {
        TORCH_CHECK(IsSupportedNdDtype(keyCache.scalar_type()),
                    "Norm mode only supports float16, bfloat16, and int8 on non-Ascend-950 hardware");
        TORCH_CHECK(key.dim() == 3, "key must be a 3D tensor in Norm mode");
        TORCH_CHECK(value.dim() == 3, "value must be a 3D tensor in Norm mode");

        blockSize = keyCache.size(1);
        TORCH_CHECK(valueCache.size(1) == blockSize, "key_cache and value_cache must have the same block size");
        TORCH_CHECK(keyCache.size(2) == key.size(1) && keyCache.size(3) == key.size(2),
                    "key shape must match the trailing dimensions of key_cache");
        TORCH_CHECK(valueCache.size(2) == value.size(1) && valueCache.size(3) == value.size(2),
                    "value shape must match the trailing dimensions of value_cache");
        tokenSizeK = key.size(1) * key.size(2);
        tokenSizeV = value.size(1) * value.size(2);
        TORCH_CHECK(blockSize * tokenSizeK * typeByte <= INT_MAX,
                    "one key cache block must not exceed INT_MAX bytes");
        TORCH_CHECK(blockSize * tokenSizeV * typeByte <= INT_MAX,
                    "one value cache block must not exceed INT_MAX bytes");
    }

    const int64_t numBatches = blockTables.size(0);
    TORCH_CHECK(numBatches > 0 && numBatches <= MAX_CAPTURE_BATCHES,
                "block_tables batch size must be in [1, ", MAX_CAPTURE_BATCHES, "]");
    TORCH_CHECK(blockTables.size(1) > 0 && blockTables.size(1) <= INT_MAX,
                "block_tables width must be in [1, INT_MAX]");
    const int64_t expectedSeqLens = numBatches + (isSeqLensCumsum ? 1 : 0);
    TORCH_CHECK(seqLens.numel() == expectedSeqLens, "seq_lens must contain ", expectedSeqLens,
                " elements for the selected is_seq_lens_cumsum mode");

    if (seqOffset.has_value()) {
        CheckMetadataTensor(seqOffset.value(), "seq_offset", 1);
        CheckSameDevice(keyCache, seqOffset.value(), "seq_offset");
        TORCH_CHECK(seqOffset.value().numel() == numBatches,
                    "seq_offset must contain one element per block_tables row");
    }

    TORCH_CHECK(blockSize > 0 && blockSize <= INT_MAX, "cache block size must be in [1, INT_MAX]");
    TORCH_CHECK(tokenSizeK > 0 && tokenSizeK <= INT_MAX, "key token size must be in [1, INT_MAX] elements");
    TORCH_CHECK(tokenSizeV > 0 && tokenSizeV <= INT_MAX, "value token size must be in [1, INT_MAX] elements");
    TORCH_CHECK(tokenSizeK * typeByte % DATA_BLOCK_SIZE == 0,
                "key token size in bytes must be aligned to ", DATA_BLOCK_SIZE);
    TORCH_CHECK(tokenSizeV * typeByte % DATA_BLOCK_SIZE == 0,
                "value token size in bytes must be aligned to ", DATA_BLOCK_SIZE);

    GatherPaKvCacheTilingData tilingData = {
        .blockSize = static_cast<int32_t>(blockSize),
        .numTokens = static_cast<int32_t>(numBatches),
        .numblkTabCol = static_cast<int32_t>(blockTables.size(1)),
        .tokenSizeK = static_cast<int32_t>(tokenSizeK),
        .tokenSizeV = static_cast<int32_t>(tokenSizeV),
        .typeByte = static_cast<int32_t>(typeByte),
        .hasSeqStarts = seqOffset.has_value() ? 1 : 0,
        .isSeqLensCumsum = isSeqLensCumsum ? 1 : 0,
        .kCacheBlockStride = keyCache.stride(0),
        .vCacheBlockStride = valueCache.stride(0),
        .tilingKey = isNz ? TILING_KEY_NZ
                          : (keyCache.scalar_type() == at::kChar ? TILING_KEY_ND_INT8 : TILING_KEY_ND_B16),
    };

    const int64_t tilingSize = sizeof(GatherPaKvCacheTilingData);
    static auto globalTilingData = at::empty(
        {tilingSize * MAX_CAPTURE_BATCHES}, at::TensorOptions().dtype(at::kByte).device(keyCache.device()));
    const int64_t tilingOffset = (numBatches - 1) * tilingSize;
    void *tilingPtr = globalTilingData.data_ptr<uint8_t>() + tilingOffset;
    const aclError copyResult = aclrtMemcpy(tilingPtr, tilingSize, &tilingData, tilingSize,
                                            ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(copyResult == ACL_SUCCESS, "failed to copy gather_pa_kv_cache tiling data, error code: ", copyResult);

    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform != nullptr, "failed to get AscendC platform information");
    uint32_t blockDim = platform->GetCoreNumAiv();
    TORCH_CHECK(blockDim > 0, "gather_pa_kv_cache requires at least one vector core");
    if (isNz && key.size(0) > 0) {
        blockDim = std::min(blockDim, static_cast<uint32_t>(key.size(0)));
    }
    return std::make_tuple(tilingPtr, blockDim);
}

}  // namespace gather_pa_kv_cache

#endif  // VLLM_ASCEND_GATHER_PA_KV_CACHE_HOST_H
