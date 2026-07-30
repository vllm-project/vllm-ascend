/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */
#ifndef SPARSE_KV_PATCH_TORCH_ADPT_H
#define SPARSE_KV_PATCH_TORCH_ADPT_H

namespace vllm_ascend {

inline void npu_sparse_kv_patch_out(
    const at::Tensor &paged_ctkv,
    const at::Tensor &paged_kpe,
    const at::Tensor &slot_mapping,
    const at::Tensor &current_topk_slots,
    at::Tensor &prefetched_ctkv,
    at::Tensor &prefetched_kpe)
{
    TORCH_CHECK(
        paged_ctkv.scalar_type() == at::kHalf ||
            paged_ctkv.scalar_type() == at::kBFloat16,
        "paged_ctkv must be float16 or bfloat16");
    TORCH_CHECK(paged_kpe.scalar_type() == paged_ctkv.scalar_type(),
                "paged_kpe dtype must match paged_ctkv");
    TORCH_CHECK(slot_mapping.scalar_type() == at::kInt ||
                    slot_mapping.scalar_type() == at::kLong,
                "slot_mapping must be int32 or int64");
    TORCH_CHECK(current_topk_slots.scalar_type() == at::kInt,
                "current_topk_slots must be int32");

    TORCH_CHECK(paged_ctkv.dim() == 4 &&
                    paged_ctkv.size(1) == 128 &&
                    paged_ctkv.size(2) == 1 &&
                    paged_ctkv.size(3) == 512,
                "paged_ctkv must have shape [num_blocks, 128, 1, 512]");
    TORCH_CHECK(paged_kpe.dim() == 4 &&
                    paged_kpe.size(0) == paged_ctkv.size(0) &&
                    paged_kpe.size(1) == 128 &&
                    paged_kpe.size(2) == 1 &&
                    paged_kpe.size(3) == 64,
                "paged_kpe must have shape [num_blocks, 128, 1, 64]");
    TORCH_CHECK(slot_mapping.dim() == 1,
                "slot_mapping must be one-dimensional");
    TORCH_CHECK(current_topk_slots.dim() == 2 &&
                    current_topk_slots.size(0) == slot_mapping.size(0) &&
                    current_topk_slots.size(1) == 8,
                "current_topk_slots must have shape [num_actual, 8]");

    const int64_t num_actual = slot_mapping.size(0);
    TORCH_CHECK(prefetched_ctkv.dim() == 3 &&
                    prefetched_ctkv.size(0) == num_actual &&
                    prefetched_ctkv.size(2) == 512,
                "prefetched_ctkv must have shape [num_actual, topk, 512]");
    TORCH_CHECK(prefetched_kpe.dim() == 3 &&
                    prefetched_kpe.size(0) == num_actual &&
                    prefetched_kpe.size(1) == prefetched_ctkv.size(1) &&
                    prefetched_kpe.size(2) == 64,
                "prefetched_kpe must have shape [num_actual, topk, 64]");
    TORCH_CHECK(prefetched_ctkv.scalar_type() ==
                    paged_ctkv.scalar_type() &&
                    prefetched_kpe.scalar_type() ==
                    paged_kpe.scalar_type(),
                "prefetched outputs must match cache dtype");

    const at::Tensor *tensors[] = {
        &paged_kpe, &slot_mapping, &current_topk_slots,
        &prefetched_ctkv, &prefetched_kpe};
    const char *names[] = {
        "paged_kpe", "slot_mapping", "current_topk_slots",
        "prefetched_ctkv", "prefetched_kpe"};
    for (size_t i = 0; i < 5; ++i) {
        TORCH_CHECK(tensors[i]->device() == paged_ctkv.device(),
                    names[i], " must be on device ",
                    paged_ctkv.device());
        TORCH_CHECK(tensors[i]->is_contiguous(),
                    names[i], " must be contiguous");
    }
    TORCH_CHECK(paged_ctkv.is_contiguous(),
                "paged_ctkv must be contiguous");

    EXEC_NPU_CMD(aclnnSparseKvPatch, paged_ctkv, paged_kpe,
                 slot_mapping, current_topk_slots,
                 prefetched_ctkv, prefetched_kpe);
}

}  // namespace vllm_ascend

#endif  // SPARSE_KV_PATCH_TORCH_ADPT_H
