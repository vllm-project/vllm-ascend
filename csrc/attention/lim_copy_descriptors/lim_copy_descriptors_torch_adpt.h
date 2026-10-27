/**
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef LIM_COPY_DESCRIPTORS_TORCH_ADPT_H_
#define LIM_COPY_DESCRIPTORS_TORCH_ADPT_H_

#include <ATen/ATen.h>

namespace vllm_ascend {

// Moves missed KV rows from the host DRAM pool into per-request hot rows.
// No output tensors: the operator's effect is the copy itself.
// ``active_geometry`` (int64 [2]) carries the per-step geometry:
//   [0] active_rows        current batch's request count, which bounds the
//                          kernel loop so stale miss counts from a larger
//                          previous step are never read;
//   [1] block_table_width  live row stride of ``source_block_table``, used
//                          for the block-table row lookup (it can exceed
//                          max_model_len / block_size when the engine
//                          reserves encoder tokens).
// ``params`` (int64 [5]) carries the deployment constants:
//   [block_size, hot_tokens, miss_stride, kv_dim, rope_dim].
inline void npu_lim_copy_descriptors(
    const at::Tensor& missSrcIds, const at::Tensor& missDstSlots,
    const at::Tensor& missCounts, const at::Tensor& sourceBlockTable,
    const at::Tensor& poolEntries, const at::Tensor& hostPtrs,
    const at::Tensor& devPtrs, const at::Tensor& params,
    const at::Tensor& activeGeometry)
{
    TORCH_CHECK(missSrcIds.dim() == 2 && missSrcIds.scalar_type() == at::kInt,
                "miss_src_ids must be a rank-2 int32 tensor.");
    // The kernel reads every plan tensor as int32 (miss/dst share the same
    // [B, miss_stride] addressing ``b * miss_stride + w``), so the dst slots
    // and the two per-request vectors must be pinned to int32 as well; an
    // int64 input would silently misaddress GM.
    TORCH_CHECK(missDstSlots.dim() == 2 && missDstSlots.scalar_type() == at::kInt &&
                missDstSlots.sizes() == missSrcIds.sizes(),
                "miss_dst_slots must be an int32 tensor with the same [B, miss_stride] shape as miss_src_ids.");
    TORCH_CHECK(missCounts.dim() == 1 && missCounts.size(0) <= missSrcIds.size(0) &&
                missCounts.scalar_type() == at::kInt,
                "miss_counts must be an int32 [B] tensor with B <= miss_src_ids.rows.");
    TORCH_CHECK(poolEntries.dim() == 1 && poolEntries.size(0) > 0 && poolEntries.size(0) <= missSrcIds.size(0) &&
                poolEntries.scalar_type() == at::kInt,
                "pool_entries must be an int32 [activeB] tensor with 0 < activeB <= B.");
    TORCH_CHECK(sourceBlockTable.dim() == 2 && sourceBlockTable.scalar_type() == at::kInt,
                "source_block_table must be a rank-2 int32 tensor.");
    TORCH_CHECK(hostPtrs.numel() == 2 && devPtrs.numel() == 2 &&
                hostPtrs.scalar_type() == at::kLong && devPtrs.scalar_type() == at::kLong,
                "pointer tensors must be int64 [2].");
    TORCH_CHECK(params.numel() >= 5 && params.scalar_type() == at::kLong,
                "params must be int64 [block_size, hot_tokens, miss_stride, "
                "kv_dim, rope_dim].");
    TORCH_CHECK(activeGeometry.numel() == 2 && activeGeometry.scalar_type() == at::kLong,
                "active_geometry must be int64 [2] = [active_rows, block_table_width].");
    const at::Tensor tensors[] = {
        missSrcIds, missDstSlots, missCounts, sourceBlockTable, poolEntries,
        hostPtrs, devPtrs, params, activeGeometry};
    const auto device = missSrcIds.device();
    for (const auto& tensor : tensors) {
        TORCH_CHECK(tensor.device() == device, "all LimCopyDescriptors tensors must share one NPU.");
        TORCH_CHECK(tensor.is_contiguous(), "all LimCopyDescriptors tensors must be contiguous.");
    }

    EXEC_NPU_CMD(aclnnLimCopyDescriptors,
                 missSrcIds, missDstSlots, missCounts, sourceBlockTable, poolEntries,
                 hostPtrs, devPtrs, params, activeGeometry);
}

}  // namespace vllm_ascend
#endif
