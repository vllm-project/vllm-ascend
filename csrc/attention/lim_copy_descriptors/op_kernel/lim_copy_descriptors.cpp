/**
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file lim_copy_descriptors.cpp
 * \brief AscendC kernel for LimCopyDescriptors
 *
 * Resolves the owner LIM's miss plan through the source block table and
 * moves missed KV rows from the registered host DRAM pool into
 * per-request hot rows (GM -> UB -> GM DataCopyPad).
 *
 * The per-step ``active_geometry`` input carries ``[active_rows,
 * block_table_width]``: ``active_rows`` is the current batch's real request
 * count (not the descriptor capacity), so stale miss counts from a larger
 * previous step are never read, and ``block_table_width`` is the live row
 * stride of ``source_block_table`` (which may exceed
 * ``max_model_len / block_size`` when the engine reserves encoder tokens),
 * so the block-table row lookup can never be shifted by a derived stride.
 *
 * The traversal and the copy pipeline are shared with the linear-slot
 * variant (see csrc/common/include/kernel/lim_copy_descriptors_core.h);
 * this file only supplies the hot-row destination policy: the destination
 * row comes from the per-request ``pool_entries`` table and a destination
 * slot is bounded by the request's row width (hot + tail tokens). Invalid
 * entries are skipped on both sides — negative source/slot, a destination
 * slot beyond the row width, a block index beyond the table, an unmapped
 * physical block, or a negative pool entry — which keeps replayed graphs
 * safe with stale plan contents.
 */
#include "kernel_operator.h"
#include "lim_copy_descriptors_core.h"

using namespace AscendC;

namespace {
// params layout: [block_size, hot_tokens, miss_stride, kv_dim, rope_dim].
constexpr uint32_t LCD_PARAM_HOT = 1;

class HotRowDstPolicy {
public:
    static constexpr bool kHasPoolEntries = true;
    struct RowCtx {
        int64_t poolRow;
    };

    __aicore__ inline void BindPoolEntries(GM_ADDR poolEntries)
    {
        poolEntries_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(poolEntries));
    }

    __aicore__ inline void InitRowGeometry(const AscendC::GlobalTensor<int64_t> &params, uint64_t blockSize)
    {
        const uint64_t hotTokens = static_cast<uint64_t>(params.GetValue(LCD_PARAM_HOT));
        rowTokens_ = hotTokens + 2 * blockSize;
    }

    __aicore__ inline bool GeometryValid() const
    {
        return rowTokens_ != 0;
    }

    __aicore__ inline uint64_t DstSlotBound() const
    {
        return rowTokens_;
    }

    __aicore__ inline RowCtx RefreshRow(uint64_t b, int64_t cnt)
    {
        return RowCtx{cnt > 0 ? static_cast<int64_t>(poolEntries_.GetValue(b)) : -1};
    }

    __aicore__ inline bool RowCopyable(const RowCtx &rowCtx) const
    {
        return rowCtx.poolRow >= 0;
    }

    __aicore__ inline uint64_t DstBase(const RowCtx &rowCtx, uint64_t dstSlot) const
    {
        return static_cast<uint64_t>(rowCtx.poolRow) * rowTokens_ + dstSlot;
    }

private:
    AscendC::GlobalTensor<int32_t> poolEntries_;
    uint64_t rowTokens_ = 0;
};

using LimCopyDescriptorsKernel =
    vllm_ascend_lcd::LimCopyDescriptorsCore<HotRowDstPolicy, LimCopyDescriptorsTilingData>;
} // namespace

extern "C" __global__ __aicore__ void lim_copy_descriptors(
    GM_ADDR missSrcIds, GM_ADDR missDstSlots, GM_ADDR missCounts,
    GM_ADDR sourceBlockTable, GM_ADDR poolEntries, GM_ADDR hostPtrs,
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
        LimCopyDescriptorsKernel op(&pipe, &tilingData);
        op.Init(missSrcIds, missDstSlots, missCounts, sourceBlockTable, poolEntries,
                hostPtrs, devPtrs, params, activeGeometry);
        op.Process();
    }
}
