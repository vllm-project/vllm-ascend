/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef KV_CACHE_FULL_BLOCK_DUMP_TORCH_ADPT_H
#define KV_CACHE_FULL_BLOCK_DUMP_TORCH_ADPT_H

namespace vllm_ascend {

inline void npu_kv_cache_full_block_dump(
    const at::Tensor& src_cache_0,
    const at::Tensor& src_cache_1,
    const at::Tensor& dst_cache_0,
    const at::Tensor& dst_cache_1,
    const at::Tensor& src_block_ids,
    const at::Tensor& dst_block_ids)
{
    TORCH_CHECK(src_cache_0.device().is_privateuseone(),
                "src_cache_0 must be on NPU.");
    TORCH_CHECK(src_cache_1.device().is_privateuseone(),
                "src_cache_1 must be on NPU.");
    TORCH_CHECK(dst_cache_0.device().is_privateuseone(),
                "dst_cache_0 must be NPU-addressable memory.");
    TORCH_CHECK(dst_cache_1.device().is_privateuseone(),
                "dst_cache_1 must be NPU-addressable memory.");
    TORCH_CHECK(src_block_ids.device().is_privateuseone(),
                "src_block_ids must be on NPU.");
    TORCH_CHECK(dst_block_ids.device().is_privateuseone(),
                "dst_block_ids must be on NPU.");
    const auto source_device = src_cache_0.device();
    TORCH_CHECK(src_cache_1.device() == source_device &&
                    dst_cache_0.device() == source_device &&
                    dst_cache_1.device() == source_device &&
                    src_block_ids.device() == source_device &&
                    dst_block_ids.device() == source_device,
                "all KV cache dump tensors must be on the same NPU.");
    TORCH_CHECK(src_cache_0.dim() == 3,
                "src_cache_0 must be [blocks, block_size, dim].");
    TORCH_CHECK(src_cache_1.dim() == 3,
                "src_cache_1 must be [blocks, block_size, dim].");
    TORCH_CHECK(dst_cache_0.dim() == 3,
                "dst_cache_0 must be [blocks, block_size, dim].");
    TORCH_CHECK(dst_cache_1.dim() == 3,
                "dst_cache_1 must be [blocks, block_size, dim].");
    TORCH_CHECK(src_cache_0.is_contiguous() && src_cache_1.is_contiguous() &&
                    dst_cache_0.is_contiguous() && dst_cache_1.is_contiguous(),
                "KV cache dump payload tensors must be contiguous.");
    TORCH_CHECK(src_block_ids.dim() == 1,
                "src_block_ids must be [rows].");
    TORCH_CHECK(dst_block_ids.dim() == 1,
                "dst_block_ids must be [rows].");
    TORCH_CHECK(src_block_ids.is_contiguous() &&
                    dst_block_ids.is_contiguous(),
                "KV cache dump block-id tensors must be contiguous.");
    TORCH_CHECK(src_block_ids.scalar_type() == at::ScalarType::Int,
                "src_block_ids must be int32.");
    TORCH_CHECK(dst_block_ids.scalar_type() == at::ScalarType::Int,
                "dst_block_ids must be int32.");
    TORCH_CHECK(src_block_ids.numel() == dst_block_ids.numel(),
                "source and destination block-id rows must match.");
    TORCH_CHECK(src_cache_0.scalar_type() == dst_cache_0.scalar_type(),
                "source and destination cache plane 0 must have the same dtype.");
    TORCH_CHECK(src_cache_1.scalar_type() == dst_cache_1.scalar_type(),
                "source and destination cache plane 1 must have the same dtype.");
    TORCH_CHECK(src_cache_0.scalar_type() == src_cache_1.scalar_type(),
                "all KV cache dump payload tensors must have the same dtype; "
                "the registered CANN dtype combinations are correlated.");
    TORCH_CHECK(src_cache_0.data_ptr() != dst_cache_0.data_ptr(),
                "KV cache dump plane 0 source and destination arenas must not alias.");
    TORCH_CHECK(src_cache_1.data_ptr() != dst_cache_1.data_ptr(),
                "KV cache dump plane 1 source and destination arenas must not alias.");

    EXEC_NPU_CMD(aclnnKvCacheFullBlockDump,
                 src_cache_0,
                 src_cache_1,
                 dst_cache_0,
                 dst_cache_1,
                 src_block_ids,
                 dst_block_ids);
}

}  // namespace vllm_ascend

#endif  // KV_CACHE_FULL_BLOCK_DUMP_TORCH_ADPT_H
