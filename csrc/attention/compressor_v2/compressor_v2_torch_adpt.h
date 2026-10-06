/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */
#ifndef VLLM_ASCEND_COMPRESSOR_V2_TORCH_ADPT_H
#define VLLM_ASCEND_COMPRESSOR_V2_TORCH_ADPT_H

namespace vllm_ascend {
at::Tensor compressor_v2(
    const at::Tensor& x, const at::Tensor& wkv, const at::Tensor& wgate,
    at::Tensor& state_cache, const c10::optional<at::Tensor>& state_block_table,
    const c10::optional<at::Tensor>& cu_seqlens,
    const c10::optional<at::Tensor>& seqused,
    const c10::optional<at::Tensor>& start_pos, int64_t cmp_ratio)
{
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, "x must have shape [T,H] or [B,S,H]");
    TORCH_CHECK(wkv.dim() == 2 && wgate.sizes() == wkv.sizes(),
                "wkv and wgate must have the same [D,H] shape");
    TORCH_CHECK(x.size(-1) == wkv.size(1), "x and weight hidden dimensions must match");
    TORCH_CHECK(cmp_ratio >= 2 && cmp_ratio <= 128, "cmp_ratio must be in [2,128]");
    TORCH_CHECK(state_cache.dim() == 3 && state_cache.size(2) == 2 * wkv.size(0),
                "state_cache must have shape [blocks,block_size,2*D]");
    TORCH_CHECK(state_cache.stride(2) == 1 && state_cache.stride(1) == state_cache.size(2),
                "state_cache must be contiguous within each block");

    at::SmallVector<int64_t, 3> output_shape;
    if (x.dim() == 3) {
        TORCH_CHECK(!cu_seqlens.has_value(), "cu_seqlens must be None for [B,S,H] inputs");
        output_shape = {x.size(0), (x.size(1) + cmp_ratio - 1) / cmp_ratio, wkv.size(0)};
    } else {
        TORCH_CHECK(cu_seqlens.has_value() && cu_seqlens->dim() == 1 && cu_seqlens->numel() >= 1,
                    "cu_seqlens must have shape [B+1] for [T,H] inputs");
        output_shape = {std::min(x.size(0), x.size(0) / cmp_ratio + cu_seqlens->numel() - 1),
                        wkv.size(0)};
    }
    auto output = at::empty(output_shape, x.options());
    const int64_t state_block_stride = state_cache.stride(0);
    EXEC_NPU_CMD(aclnnCompressorV2, x, wkv, wgate, state_cache, state_block_table,
                 cu_seqlens, seqused, start_pos, cmp_ratio, state_block_stride, output);
    return output;
}
}  // namespace vllm_ascend
#endif
