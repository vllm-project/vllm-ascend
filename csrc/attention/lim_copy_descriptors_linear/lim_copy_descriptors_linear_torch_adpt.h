/**
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef LIM_COPY_DESCRIPTORS_LINEAR_TORCH_ADPT_H_
#define LIM_COPY_DESCRIPTORS_LINEAR_TORCH_ADPT_H_

#include <ATen/ATen.h>

namespace vllm_ascend {

// Moves missed KV rows from the shared host DRAM pool into the fused-overlap
// selection buffer on NPU (GM -> UB -> GM DataCopyPad), on the caller's
// stream. No output tensors: the operator's effect is the copy itself.
//
// ``miss_src_ids`` / ``miss_dst_slots`` (int32 [rows, miss_stride]) carry the
// external LRU plan of the group's plan-owner layer: the source logical
// tokens and the LINEAR destination slots. The Python side folds the
// logical->physical row map into the slots before launching
// (dst = physical_row * capacity + slot), so the kernel needs no pool-entry
// indirection.
// ``miss_counts`` (int32 [rows]) bounds each row's live misses.
// ``source_block_table`` (int32 [rows, width]) maps logical ->
// physical blocks in the host pool.
// ``active_geometry`` (int64 [2]) carries the per-step geometry:
//   [0] active_rows        current batch's token-row count, which bounds the
//                          kernel loop so stale miss counts from a larger
//                          previous step are never read;
//   [1] block_table_width  live row stride of ``source_block_table``.
// ``host_ptrs`` / ``dev_ptrs`` (int64 [2]) are the host pool (K latent, rope)
// and NPU selection buffer base pointers.
// ``params`` (int64 [6]) carries the deployment constants:
//   [block_size, capacity(topk_buffer_size), miss_stride, kv_dim, rope_dim,
//    row_capacity]. A destination slot is valid only when it is below
//   row_capacity * capacity (the whole selection buffer), which keeps
//   replayed graphs safe with stale plan contents.
inline void npu_lim_copy_descriptors_linear(
    const at::Tensor& missSrcIds, const at::Tensor& missDstSlots,
    const at::Tensor& missCounts, const at::Tensor& sourceBlockTable,
    const at::Tensor& hostPtrs, const at::Tensor& devPtrs,
    const at::Tensor& params, const at::Tensor& activeGeometry)
{
    TORCH_CHECK(missSrcIds.dim() == 2 && missSrcIds.scalar_type() == at::kInt,
                "miss_src_ids must be a rank-2 int32 tensor.");
    // The kernel reads every plan tensor as int32 (miss/dst share the same
    // [rows, miss_stride] addressing ``b * miss_stride + w``), so the dst
    // slots and the per-row counts must be pinned to int32 as well; an
    // int64 input would silently misaddress GM.
    TORCH_CHECK(missDstSlots.dim() == 2 && missDstSlots.scalar_type() == at::kInt &&
                missDstSlots.sizes() == missSrcIds.sizes(),
                "miss_dst_slots must be an int32 tensor with the same [rows, miss_stride] shape as miss_src_ids.");
    TORCH_CHECK(missCounts.dim() == 1 && missCounts.size(0) <= missSrcIds.size(0) &&
                missCounts.scalar_type() == at::kInt,
                "miss_counts must be an int32 [rows] tensor with rows <= miss_src_ids.rows.");
    TORCH_CHECK(sourceBlockTable.dim() == 2 && sourceBlockTable.scalar_type() == at::kInt,
                "source_block_table must be a rank-2 int32 tensor.");
    TORCH_CHECK(hostPtrs.numel() == 2 && devPtrs.numel() == 2 &&
                hostPtrs.scalar_type() == at::kLong && devPtrs.scalar_type() == at::kLong,
                "pointer tensors must be int64 [2].");
    TORCH_CHECK(params.numel() >= 6 && params.scalar_type() == at::kLong,
                "params must be int64 [block_size, capacity, miss_stride, "
                "kv_dim, rope_dim, row_capacity].");
    TORCH_CHECK(activeGeometry.numel() == 2 && activeGeometry.scalar_type() == at::kLong,
                "active_geometry must be int64 [2] = [active_rows, block_table_width].");
    const at::Tensor tensors[] = {
        missSrcIds, missDstSlots, missCounts, sourceBlockTable,
        hostPtrs, devPtrs, params, activeGeometry};
    const auto device = missSrcIds.device();
    for (const auto& tensor : tensors) {
        TORCH_CHECK(tensor.device() == device, "all LimCopyDescriptorsLinear tensors must share one NPU.");
        TORCH_CHECK(tensor.is_contiguous(), "all LimCopyDescriptorsLinear tensors must be contiguous.");
    }

    EXEC_NPU_CMD(aclnnLimCopyDescriptorsLinear,
                 missSrcIds, missDstSlots, missCounts, sourceBlockTable,
                 hostPtrs, devPtrs, params, activeGeometry);
}

}  // namespace vllm_ascend
#endif
