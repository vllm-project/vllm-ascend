/**
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file lim_copy_descriptors_linear.cpp
 * \brief AscendC kernel for LimCopyDescriptorsLinear
 *
 * Resolves the external LRU plan-owner layer's miss plan through the source
 * block table and moves missed KV rows from the registered host DRAM pool
 * into the fused-overlap selection buffer on NPU (GM -> UB -> GM
 * DataCopyPad).
 *
 * The traversal and the copy pipeline are shared with the hot-row variant
 * (see csrc/common/include/kernel/lim_copy_descriptors_core.h); this file
 * only supplies the linear-slot destination policy.
 *
 * The destination slots are LINEAR: the Python side folds the
 * logical->physical row map into the plan before launching
 * (dst = physical_row * capacity + slot), so the kernel addresses the flat
 * selection buffer directly with no pool-entry indirection.
 *
 * The per-step ``active_geometry`` input carries ``[active_rows,
 * block_table_width]``: ``active_rows`` is the current batch's real token-row
 * count (not the descriptor capacity), so stale miss counts from a larger
 * previous step are never read, and ``block_table_width`` is the live row
 * stride of ``source_block_table`` (which may exceed
 * ``max_model_len / block_size`` when the engine reserves encoder tokens),
 * so the block-table row lookup can never be shifted by a derived stride.
 *
 * Invalid entries are skipped on both sides — negative source/slot, a
 * destination slot beyond the whole selection buffer (row_capacity *
 * capacity), a block index beyond the table, or an unmapped physical block
 * — which keeps replayed graphs safe with stale plan contents.
 */
#include "kernel_operator.h"
#include "lim_copy_descriptors_core.h"

using namespace AscendC;

namespace {
// params layout: [block_size, capacity, miss_stride, kv_dim, rope_dim,
// row_capacity].
constexpr uint32_t LCD_PARAM_CAPACITY = 1;
constexpr uint32_t LCD_PARAM_ROW_CAPACITY = 5;

class LinearDstPolicy {
public:
    static constexpr bool kHasPoolEntries = false;
    struct RowCtx {
    };

    __aicore__ inline void InitRowGeometry(const AscendC::GlobalTensor<int64_t> &params, uint64_t blockSize)
    {
        (void)blockSize;
        capacity_ = static_cast<uint64_t>(params.GetValue(LCD_PARAM_CAPACITY));
        rowCapacity_ = static_cast<uint64_t>(params.GetValue(LCD_PARAM_ROW_CAPACITY));
        // Linear destination slots address the flat selection buffer:
        // [row_capacity, capacity, 1, dim] laid out row-major.
        dstSlotBound_ = rowCapacity_ * capacity_;
    }

    __aicore__ inline bool GeometryValid() const
    {
        return capacity_ != 0 && rowCapacity_ != 0 && dstSlotBound_ != 0;
    }

    __aicore__ inline uint64_t DstSlotBound() const
    {
        return dstSlotBound_;
    }

    __aicore__ inline RowCtx RefreshRow(uint64_t b, int64_t cnt)
    {
        (void)b;
        (void)cnt;
        return RowCtx{};
    }

    __aicore__ inline bool RowCopyable(const RowCtx &rowCtx) const
    {
        (void)rowCtx;
        return true;
    }

    __aicore__ inline uint64_t DstBase(const RowCtx &rowCtx, uint64_t dstSlot) const
    {
        (void)rowCtx;
        return dstSlot;
    }

private:
    uint64_t capacity_ = 0;
    uint64_t rowCapacity_ = 0;
    uint64_t dstSlotBound_ = 0;
};

using LimCopyDescriptorsLinearKernel =
    vllm_ascend_lcd::LimCopyDescriptorsCore<LinearDstPolicy, LimCopyDescriptorsLinearTilingData>;
} // namespace

extern "C" __global__ __aicore__ void lim_copy_descriptors_linear(
    GM_ADDR missSrcIds, GM_ADDR missDstSlots, GM_ADDR missCounts,
    GM_ADDR sourceBlockTable, GM_ADDR hostPtrs,
    GM_ADDR devPtrs, GM_ADDR params, GM_ADDR activeGeometry,
    GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    if (g_coreType == AIC) {
        return;
    }
    if (TILING_KEY_IS(0)) {
        GET_TILING_DATA(tilingData, tiling);
        TPipe pipe;
        LimCopyDescriptorsLinearKernel op(&pipe, &tilingData);
        op.Init(missSrcIds, missDstSlots, missCounts, sourceBlockTable,
                nullptr, hostPtrs, devPtrs, params, activeGeometry);
        op.Process();
    }
}
