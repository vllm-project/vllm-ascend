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
 * The flattened (request, miss-slot) space is traversed with a grid-stride
 * loop: every AIV block strides through the whole space, spreading one
 * request's misses across blocks to avoid a one-block serial tail, and the
 * traversal is bounded by ``active_rows``. Slots at or beyond a request's
 * real ``miss_counts[b]`` are skipped by a cheap count test, and invalid
 * entries are skipped on both sides — negative source/slot, a destination
 * slot beyond the row width (hot + tail), a block index beyond the table,
 * an unmapped physical block, or a negative pool entry — which keeps
 * replayed graphs safe with stale plan contents.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace {
// Upper bound of one KV row staged through UB as two separate copies
// (kv_lora_rank latent, qk_rope_head_dim rope), both bf16. The per-model dims
// travel in params and must fit these capacities; the Python side validates
// the same bound before launching.
constexpr int64_t LCD_KV_DIM_MAX = 512;    // kv_lora_rank
constexpr int64_t LCD_ROPE_DIM_MAX = 64;   // qk_rope_head_dim
constexpr int64_t LCD_KV_UB_BYTES = LCD_KV_DIM_MAX * sizeof(uint16_t);
constexpr int64_t LCD_ROPE_UB_BYTES = LCD_ROPE_DIM_MAX * sizeof(uint16_t);
// params layout: [block_size, hot_tokens, miss_stride, kv_dim, rope_dim].
// Per-step values (active request count, live block-table width) travel in
// the dedicated ``active_geometry`` input, not in params.
constexpr uint32_t LCD_PARAM_BLOCK = 0;
constexpr uint32_t LCD_PARAM_HOT = 1;
constexpr uint32_t LCD_PARAM_MISS_STRIDE = 2;
constexpr uint32_t LCD_PARAM_KV_DIM = 3;
constexpr uint32_t LCD_PARAM_ROPE_DIM = 4;

class LimCopyDescriptorsKernel {
public:
    __aicore__ inline LimCopyDescriptorsKernel(TPipe *pipe, const LimCopyDescriptorsTilingData *tiling)
        : pipe_(pipe), tiling_(tiling) {}

    __aicore__ inline void Init(GM_ADDR missSrcIds, GM_ADDR missDstSlots, GM_ADDR missCounts,
                                GM_ADDR sourceBlockTable, GM_ADDR poolEntries, GM_ADDR hostPtrs,
                                GM_ADDR devPtrs, GM_ADDR params, GM_ADDR activeGeometry)
    {
        missSrc_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(missSrcIds));
        missDst_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(missDstSlots));
        missCounts_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(missCounts));
        srcBlockTable_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sourceBlockTable));
        poolEntries_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(poolEntries));
        hostPtrs_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(hostPtrs));
        devPtrs_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(devPtrs));
        params_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(params));
        activeGeometry_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(activeGeometry));

        blockSize_ = static_cast<uint64_t>(params_.GetValue(LCD_PARAM_BLOCK));
        hotTokens_ = static_cast<uint64_t>(params_.GetValue(LCD_PARAM_HOT));
        missStride_ = static_cast<uint64_t>(params_.GetValue(LCD_PARAM_MISS_STRIDE));
        kvDim_ = static_cast<uint64_t>(params_.GetValue(LCD_PARAM_KV_DIM));
        ropeDim_ = static_cast<uint64_t>(params_.GetValue(LCD_PARAM_ROPE_DIM));
        activeRows_ = static_cast<uint64_t>(activeGeometry_.GetValue(0));
        maxBlocks_ = static_cast<uint64_t>(activeGeometry_.GetValue(1));
        usedCore_ = static_cast<uint64_t>(tiling_->usedCoreNum);
        rowTokens_ = hotTokens_ + 2 * blockSize_;
        kvSrcGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(hostPtrs_.GetValue(0)));
        ropeSrcGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(hostPtrs_.GetValue(1)));
        kvDstGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(devPtrs_.GetValue(0)));
        ropeDstGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(devPtrs_.GetValue(1)));

        // The Python side rejects unsupported head dims before launching; the
        // guard keeps a mismatched build a no-op instead of an OOB UB write.
        // A zero block size would fault the division below, so it degrades to
        // a no-op as well (the manager always passes cache_config.block_size,
        // this only guards a corrupted params tensor).
        if (kvDim_ == 0 || ropeDim_ == 0 || kvDim_ > LCD_KV_DIM_MAX || ropeDim_ > LCD_ROPE_DIM_MAX ||
            blockSize_ == 0 || rowTokens_ == 0) {
            activeRows_ = 0;
        }
        // Rope is staged right after the KV latent copy of the same row.
        kvUbOffset_ = kvDim_;
        // Per-buffer size is one row's staging bytes; ``num`` is the depth.
        pipe_->InitBuffer(copyQueue_, 2, LCD_KV_UB_BYTES + LCD_ROPE_UB_BYTES);
    }

    __aicore__ inline void Process()
    {
        const uint64_t used = usedCore_ > 0 ? usedCore_ : 1;
        const uint64_t blockIdx = static_cast<uint64_t>(GetBlockIdx());
        if (blockIdx >= used) {
            return;
        }
        // Grid-stride over the flattened (b, w) space bounded by activeRows_:
        // each block strides through the miss space, spreading one request's
        // misses across blocks to avoid a one-block serial tail, while the
        // total stays bounded by the live request count so stale miss counts
        // from a larger previous step are never read. The traversal covers
        // activeRows_ * missStride_ slots, but only positions below a
        // request's real miss count are copied; the rest are skipped by the
        // cheap count test. Eager custom-op launches on this A3 stack may
        // schedule a random block subset, but graph-mode replays schedule
        // the full grid.
        //
        // ``b`` only changes every missStride_/used iterations, so the two
        // per-request GM scalars (miss count, pool entry) are cached and
        // refreshed only when ``b`` changes; without the cache the skip path
        // would re-read them from global memory on every slot.
        const uint64_t total = activeRows_ * missStride_;
        uint64_t flat = blockIdx;
        int64_t lastB = -1;
        int64_t cnt = 0;
        int64_t poolRowRaw = -1;
        while (flat < total) {
            const uint64_t b = flat / missStride_;
            const uint64_t w = flat - b * missStride_;
            if (static_cast<int64_t>(b) != lastB) {
                lastB = static_cast<int64_t>(b);
                cnt = static_cast<int64_t>(missCounts_.GetValue(b));
                poolRowRaw = cnt > 0 ? static_cast<int64_t>(poolEntries_.GetValue(b)) : -1;
            }
            if (cnt > 0 && w < static_cast<uint64_t>(cnt) && poolRowRaw >= 0) {
                CopyMiss(b, w, static_cast<uint64_t>(poolRowRaw));
            }
            flat += used;
        }
    }

private:
    __aicore__ inline void CopyMiss(uint64_t b, uint64_t w, uint64_t poolRow)
    {
        const int64_t srcLogical = static_cast<int64_t>(missSrc_.GetValue(b * missStride_ + w));
        const int64_t dstSlot = static_cast<int64_t>(missDst_.GetValue(b * missStride_ + w));
        // Source side and destination side are both bounded: a stale plan
        // from a replayed graph must never read past the block table nor
        // write past this request's hot row (rowTokens_ = hot + tail).
        if (srcLogical < 0 || dstSlot < 0 || static_cast<uint64_t>(dstSlot) >= rowTokens_) {
            return;
        }
        const uint64_t logicalBlock = static_cast<uint64_t>(srcLogical) / blockSize_;
        if (logicalBlock >= maxBlocks_) {
            return;
        }
        const int64_t physBlock = static_cast<int64_t>(srcBlockTable_.GetValue(b * maxBlocks_ + logicalBlock));
        if (physBlock < 0) {
            return;
        }
        const uint64_t physToken =
            static_cast<uint64_t>(physBlock) * blockSize_ + static_cast<uint64_t>(srcLogical) % blockSize_;
        const uint64_t dstBase = poolRow * rowTokens_ + static_cast<uint64_t>(dstSlot);
        const uint64_t srcKv = physToken * kvDim_;
        const uint64_t srcRope = physToken * ropeDim_;
        const uint64_t dstKv = dstBase * kvDim_;
        const uint64_t dstRope = dstBase * ropeDim_;

        LocalTensor<uint16_t> local = copyQueue_.AllocTensor<uint16_t>();
        DataCopyPadExtParams<uint16_t> padParams{false, 0, 0, 0};
        DataCopyExtParams kvParams{1, static_cast<uint32_t>(kvDim_ * sizeof(uint16_t)), 0, 0, 0};
        DataCopyExtParams ropeParams{1, static_cast<uint32_t>(ropeDim_ * sizeof(uint16_t)), 0, 0, 0};
        DataCopyPad(local[0], kvSrcGm_[srcKv], kvParams, padParams);
        DataCopyPad(local[kvUbOffset_], ropeSrcGm_[srcRope], ropeParams, padParams);
        copyQueue_.EnQue(local);
        local = copyQueue_.DeQue<uint16_t>();
        DataCopyPad(kvDstGm_[dstKv], local[0], kvParams);
        DataCopyPad(ropeDstGm_[dstRope], local[kvUbOffset_], ropeParams);
        copyQueue_.FreeTensor(local);
    }

    GlobalTensor<int32_t> missSrc_;
    GlobalTensor<int32_t> missDst_;
    GlobalTensor<int32_t> missCounts_;
    GlobalTensor<int32_t> srcBlockTable_;
    GlobalTensor<int32_t> poolEntries_;
    GlobalTensor<int64_t> hostPtrs_;
    GlobalTensor<int64_t> devPtrs_;
    GlobalTensor<int64_t> params_;
    GlobalTensor<int64_t> activeGeometry_;

    GlobalTensor<uint16_t> kvSrcGm_;
    GlobalTensor<uint16_t> ropeSrcGm_;
    GlobalTensor<uint16_t> kvDstGm_;
    GlobalTensor<uint16_t> ropeDstGm_;

    TPipe *pipe_ = nullptr;
    const LimCopyDescriptorsTilingData *tiling_ = nullptr;
    // Depth-2 queue overlaps the next GM->UB load with the previous UB->GM store.
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQueue_;
    uint64_t kvUbOffset_ = 0;
    uint64_t blockSize_ = 128;
    uint64_t hotTokens_ = 0;
    uint64_t rowTokens_ = 0;
    uint64_t missStride_ = 0;
    uint64_t kvDim_ = LCD_KV_DIM_MAX;
    uint64_t ropeDim_ = LCD_ROPE_DIM_MAX;
    uint64_t maxBlocks_ = 0;
    uint64_t activeRows_ = 0;
    uint64_t usedCore_ = 1;
};
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
