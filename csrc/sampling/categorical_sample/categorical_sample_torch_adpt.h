/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#ifndef CATEGORICAL_SAMPLE_TORCH_ADPT_H
#define CATEGORICAL_SAMPLE_TORCH_ADPT_H

namespace vllm_ascend {
namespace {
constexpr int64_t CATEGORICAL_SAMPLE_MAX_VOCAB_SIZE = 1'048'576;

void check_categorical_sample_metadata(
    const at::Tensor& tensor,
    const at::Tensor& processed_logits,
    c10::ScalarType dtype,
    c10::string_view name)
{
    TORCH_CHECK(tensor.is_privateuseone(), name, " must be an NPU tensor");
    TORCH_CHECK(tensor.device() == processed_logits.device(), name, " must be on the same device as processed_logits");
    TORCH_CHECK(tensor.dim() == 1, name, " must be 1D");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype");
}
}  // namespace

std::tuple<at::Tensor, at::Tensor> npu_categorical_sample(
    const at::Tensor& processed_logits,
    const at::Tensor& expanded_idx_mapping,
    const at::Tensor& temperature,
    const at::Tensor& seed,
    const at::Tensor& pos,
    bool return_lse,
    bool apply_temperature,
    const c10::optional<at::Tensor>& logits_cache,
    const c10::optional<at::Tensor>& logits_cache_col,
    bool use_fp64)
{
    TORCH_CHECK(processed_logits.is_privateuseone(), "processed_logits must be an NPU tensor");
    TORCH_CHECK(processed_logits.dim() == 2, "processed_logits must be 2D");
    TORCH_CHECK(
        processed_logits.scalar_type() == at::kHalf || processed_logits.scalar_type() == at::kBFloat16 ||
            processed_logits.scalar_type() == at::kFloat,
        "processed_logits must have dtype float16, bfloat16, or float32");
    TORCH_CHECK(processed_logits.size(0) > 0, "processed_logits must contain at least one row");
    TORCH_CHECK(
        processed_logits.size(1) > 0 && processed_logits.size(1) <= CATEGORICAL_SAMPLE_MAX_VOCAB_SIZE,
        "processed_logits vocabulary size must be in [1, ", CATEGORICAL_SAMPLE_MAX_VOCAB_SIZE, "]");
    TORCH_CHECK(processed_logits.stride(1) == 1, "processed_logits vocabulary dimension must be contiguous");
    TORCH_CHECK(
        processed_logits.stride(0) == 0 || processed_logits.stride(0) >= processed_logits.size(1),
        "processed_logits row stride is invalid");

    check_categorical_sample_metadata(
        expanded_idx_mapping, processed_logits,
        expanded_idx_mapping.scalar_type() == at::kLong ? at::kLong : at::kInt,
        "expanded_idx_mapping");
    check_categorical_sample_metadata(temperature, processed_logits, at::kFloat, "temperature");
    check_categorical_sample_metadata(seed, processed_logits, at::kLong, "seed");
    check_categorical_sample_metadata(pos, processed_logits, at::kLong, "pos");
    TORCH_CHECK(
        expanded_idx_mapping.size(0) == processed_logits.size(0),
        "expanded_idx_mapping length must equal the number of rows");
    TORCH_CHECK(pos.size(0) == processed_logits.size(0), "pos length must equal the number of rows");
    TORCH_CHECK(temperature.size(0) > 0, "temperature must contain at least one request");
    TORCH_CHECK(seed.size(0) == temperature.size(0), "seed and temperature must have the same length");

    int64_t logitsCacheStride = 0;
    int64_t logitsCacheColStride = 0;
    if (logits_cache.has_value()) {
        const at::Tensor& logitsCache = logits_cache.value();
        TORCH_CHECK(logitsCache.is_privateuseone(), "logits_cache must be an NPU tensor");
        TORCH_CHECK(
            logitsCache.device() == processed_logits.device(),
            "logits_cache must be on the same device as processed_logits");
        TORCH_CHECK(
            logitsCache.scalar_type() == at::kFloat || logitsCache.scalar_type() == at::kHalf ||
                logitsCache.scalar_type() == at::kBFloat16,
            "logits_cache must have dtype float16, bfloat16, or float32");
        TORCH_CHECK(
            logitsCache.dim() == 2 || logitsCache.dim() == 3,
            "logits_cache must be 2D or 3D");
        TORCH_CHECK(
            logitsCache.size(0) == temperature.size(0),
            "logits_cache request dimension must equal the temperature length");
        TORCH_CHECK(
            logitsCache.size(-1) >= processed_logits.size(1),
            "logits_cache vocabulary dimension is narrower than the input vocabulary size");
        TORCH_CHECK(
            logitsCache.stride(-1) == 1,
            "logits_cache vocabulary dimension must be contiguous");
        if (logitsCache.dim() == 3) {
            TORCH_CHECK(
                logitsCache.size(1) > 0,
                "logits_cache must contain at least one cache column");
            TORCH_CHECK(
                logitsCache.stride(1) >= processed_logits.size(1) && logitsCache.stride(1) <= UINT32_MAX,
                "logits_cache column stride is invalid");
        }
        logitsCacheColStride = logitsCache.dim() == 3 ? logitsCache.stride(1) : logitsCache.size(-1);
        const int64_t cacheRowElements =
            logitsCache.dim() == 3 ? (logitsCache.size(1) - 1) * logitsCacheColStride + processed_logits.size(1)
                                             : processed_logits.size(1);
        TORCH_CHECK(
            logitsCache.stride(0) >= cacheRowElements && logitsCache.stride(0) <= UINT32_MAX,
            "logits_cache request stride is invalid");
        logitsCacheStride = logitsCache.stride(0);

        if (logits_cache_col.has_value()) {
            const at::Tensor& logitsCacheCol = logits_cache_col.value();
            TORCH_CHECK(
                logitsCacheCol.is_privateuseone(),
                "logits_cache_col must be an NPU tensor");
            TORCH_CHECK(
                logitsCacheCol.device() == processed_logits.device(),
                "logits_cache_col must be on the same device as processed_logits");
            TORCH_CHECK(
                logitsCacheCol.scalar_type() == at::kInt,
                "logits_cache_col must have dtype int32");
            TORCH_CHECK(
                logitsCacheCol.dim() == 0 || logitsCacheCol.dim() == 1,
                "logits_cache_col must be a scalar or 1D");
            TORCH_CHECK(logitsCacheCol.is_contiguous(), "logits_cache_col must be contiguous");
            TORCH_CHECK(
                logitsCacheCol.dim() == 0 ||
                    logitsCacheCol.size(0) == processed_logits.size(0),
                "a per-token logits_cache_col must have one entry per logits row");
        }
    } else {
        TORCH_CHECK(
            !logits_cache_col.has_value(),
            "logits_cache_col requires logits_cache");
    }

    const auto rowOptions = processed_logits.options();
    at::Tensor sampledTokenIds = at::empty({processed_logits.size(0)}, rowOptions.dtype(at::kLong));
    // The ACLNN executor rejects zero-storage outputs. Always pass a real LSE
    // workspace, then preserve the public return_lse=False contract below.
    at::Tensor lseWorkspace = at::empty({processed_logits.size(0)}, rowOptions.dtype(at::kFloat));
    int64_t rowStride = processed_logits.stride(0);

    EXEC_NPU_CMD(
        aclnnCategoricalSample,
        processed_logits,
        expanded_idx_mapping,
        temperature,
        seed,
        pos,
        logits_cache,
        logits_cache_col,
        rowStride,
        logitsCacheStride,
        apply_temperature,
        return_lse,
        use_fp64,
        logitsCacheColStride,
        sampledTokenIds,
        lseWorkspace);
    at::Tensor lse = return_lse ? lseWorkspace : at::empty({0}, rowOptions.dtype(at::kFloat));
    return std::make_tuple(sampledTokenIds, lse);
}
}  // namespace vllm_ascend

#endif
