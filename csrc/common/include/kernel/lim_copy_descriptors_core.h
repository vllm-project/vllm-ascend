/**
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file lim_copy_descriptors_core.h
 * \brief Shared AscendC core for the LimCopyDescriptors copy kernels.
 *
 * Both sparse-KV-offload side-stream pre-copy kernels share the same
 * GM -> UB -> GM DataCopyPad pipeline and the same graph-replay safety
 * rules; only the destination addressing differs:
 *
 * - LimCopyDescriptors (fused Copy-SFA pipeline): misses land in
 *   per-request hot rows; the destination row comes from a per-request
 *   pool-entry table and slots are bounded by the row width
 *   (hot + tail tokens).
 * - LimCopyDescriptorsLinear (fused_overlap pipeline): misses land in the
 *   external-LRU-plan selection buffer; the Python side folds the
 *   logical->physical row map into absolute linear slots
 *   (dst = physical_row * capacity + slot) and slots are bounded by the
 *   whole selection buffer (row_capacity * capacity).
 *
 * The per-variant behavior is supplied by a DstPolicy passed as a template
 * argument, so each operator keeps its own fixed ABI (input list and
 * params layout stay public per-op contracts) while the traversal, the
 * source-side resolution and the copy tail exist exactly once.
 *
 * This header lives in the shared kernel include directory, which the
 * custom-op build stages into ``src/ascendc/common`` on every operator's
 * kernel include path; include it as
 * ``#include "lim_copy_descriptors_core.h"``.
 */

#ifndef LIM_COPY_DESCRIPTORS_CORE_H_
#define LIM_COPY_DESCRIPTORS_CORE_H_

#include "kernel_operator.h"

namespace vllm_ascend_lcd {

using namespace AscendC;

// Upper bound of one KV row staged through UB as two separate copies
// (kv_lora_rank latent, qk_rope_head_dim rope), both bf16. The per-model dims
// travel in params and must fit these capacities; the Python side validates
// the same bound before launching.
constexpr int64_t LCD_KV_DIM_MAX = 512;   // kv_lora_rank
constexpr int64_t LCD_ROPE_DIM_MAX = 64;  // qk_rope_head_dim
constexpr int64_t LCD_KV_UB_BYTES = LCD_KV_DIM_MAX * sizeof(uint16_t);
constexpr int64_t LCD_ROPE_UB_BYTES = LCD_ROPE_DIM_MAX * sizeof(uint16_t);

// Common params layout positions shared by both variants:
// [block_size, <variant row geometry...>, miss_stride, kv_dim, rope_dim].
// Per-step values (active row count, live block-table width) travel in the
// dedicated ``active_geometry`` input, not in params. The variant-specific
// positions (hot_tokens at 1; capacity at 1 and row_capacity at 5) are read
// by the corresponding DstPolicy.
constexpr uint32_t LCD_PARAM_BLOCK = 0;
constexpr uint32_t LCD_PARAM_MISS_STRIDE = 2;
constexpr uint32_t LCD_PARAM_KV_DIM = 3;
constexpr uint32_t LCD_PARAM_ROPE_DIM = 4;

/*!
 * \brief Grid-stride miss-copy core shared by both LimCopyDescriptors ops.
 *
 * \tparam DstPolicy  destination addressing policy (see below)
 * \tparam TilingData per-op generated tiling struct (``usedCoreNum``)
 *
 * The DstPolicy contract:
 * - ``static constexpr bool kHasPoolEntries`` - whether the operator ABI
 *   carries a pool-entries input (hot-row variant);
 * - ``InitRowGeometry(params, blockSize)`` - read the variant params and
 *   derive the row geometry (bounds / scales);
 * - ``GeometryValid()`` - false when the derived geometry is unusable (the
 *   kernel then degrades to a no-op, mirroring the head-dim guard);
 * - ``DstSlotBound()`` - exclusive upper bound of a destination slot;
 * - ``RowCtx`` / ``RefreshRow(b, cnt)`` - per-row cached state refreshed
 *   only when the row index changes (the hot-row policy caches the pool
 *   entry here; the linear policy has nothing to cache);
 * - ``RowCopyable(rowCtx)`` - row-level skip test;
 * - ``DstBase(rowCtx, dstSlot)`` - absolute destination token index;
 * - ``BindPoolEntries(gm)`` - only instantiated when kHasPoolEntries.
 */
template <typename DstPolicy, typename TilingData>
class LimCopyDescriptorsCore {
public:
    __aicore__ inline LimCopyDescriptorsCore(TPipe *pipe, const TilingData *tiling)
        : pipe_(pipe), tiling_(tiling) {}

    __aicore__ inline void Init(GM_ADDR missSrcIds, GM_ADDR missDstSlots, GM_ADDR missCounts,
                                GM_ADDR sourceBlockTable, GM_ADDR poolEntries, GM_ADDR hostPtrs,
                                GM_ADDR devPtrs, GM_ADDR params, GM_ADDR activeGeometry)
    {
        missSrc_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(missSrcIds));
        missDst_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(missDstSlots));
        missCounts_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(missCounts));
        srcBlockTable_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sourceBlockTable));
        if constexpr (DstPolicy::kHasPoolEntries) {
            policy_.BindPoolEntries(poolEntries);
        } else {
            (void)poolEntries;
        }
        hostPtrs_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(hostPtrs));
        devPtrs_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(devPtrs));
        params_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(params));
        activeGeometry_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(activeGeometry));

        blockSize_ = static_cast<uint64_t>(params_.GetValue(LCD_PARAM_BLOCK));
        missStride_ = static_cast<uint64_t>(params_.GetValue(LCD_PARAM_MISS_STRIDE));
        kvDim_ = static_cast<uint64_t>(params_.GetValue(LCD_PARAM_KV_DIM));
        ropeDim_ = static_cast<uint64_t>(params_.GetValue(LCD_PARAM_ROPE_DIM));
        activeRows_ = static_cast<uint64_t>(activeGeometry_.GetValue(0));
        maxBlocks_ = static_cast<uint64_t>(activeGeometry_.GetValue(1));
        usedCore_ = static_cast<uint64_t>(tiling_->usedCoreNum);
        policy_.InitRowGeometry(params_, blockSize_);
        kvSrcGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(hostPtrs_.GetValue(0)));
        ropeSrcGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(hostPtrs_.GetValue(1)));
        kvDstGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(devPtrs_.GetValue(0)));
        ropeDstGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(devPtrs_.GetValue(1)));

        // The Python side rejects unsupported head dims before launching; the
        // guard keeps a mismatched build a no-op instead of an OOB UB write.
        // A zero block size would fault the division below, so it degrades to
        // a no-op as well (the manager always passes cache_config.block_size,
        // this only guards a corrupted params tensor).
        if (kvDim_ == 0 || ropeDim_ == 0 || kvDim_ > static_cast<uint64_t>(LCD_KV_DIM_MAX) ||
            ropeDim_ > static_cast<uint64_t>(LCD_ROPE_DIM_MAX) || blockSize_ == 0 || !policy_.GeometryValid()) {
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
        // each block strides through the miss space, spreading one row's
        // misses across blocks to avoid a one-block serial tail, while the
        // total stays bounded by the live row count so stale miss counts
        // from a larger previous step are never read. The traversal covers
        // activeRows_ * missStride_ slots, but only positions below a
        // row's real miss count are copied; the rest are skipped by the
        // cheap count test. Eager custom-op launches on this A3 stack may
        // schedule a random block subset, but graph-mode replays schedule
        // the full grid.
        //
        // ``b`` only changes every missStride_/used iterations, so the
        // per-row GM scalars (miss count and any policy row state, e.g.
        // the hot-row pool entry) are cached and refreshed only when ``b``
        // changes; without the cache the skip path would re-read them from
        // global memory on every slot.
        const uint64_t total = activeRows_ * missStride_;
        uint64_t flat = blockIdx;
        int64_t lastB = -1;
        int64_t cnt = 0;
        typename DstPolicy::RowCtx rowCtx{};
        while (flat < total) {
            const uint64_t b = flat / missStride_;
            const uint64_t w = flat - b * missStride_;
            if (static_cast<int64_t>(b) != lastB) {
                lastB = static_cast<int64_t>(b);
                cnt = static_cast<int64_t>(missCounts_.GetValue(b));
                rowCtx = policy_.RefreshRow(b, cnt);
            }
            if (cnt > 0 && w < static_cast<uint64_t>(cnt) && policy_.RowCopyable(rowCtx)) {
                CopyMiss(b, w, rowCtx);
            }
            flat += used;
        }
    }

private:
    __aicore__ inline void CopyMiss(uint64_t b, uint64_t w, const typename DstPolicy::RowCtx &rowCtx)
    {
        const int64_t srcLogical = static_cast<int64_t>(missSrc_.GetValue(b * missStride_ + w));
        const int64_t dstSlot = static_cast<int64_t>(missDst_.GetValue(b * missStride_ + w));
        // Source side and destination side are both bounded: a stale plan
        // from a replayed graph must never read past the block table nor
        // write past the policy's destination bound (row width for the
        // hot-row variant, the whole selection buffer for the linear one).
        if (srcLogical < 0 || dstSlot < 0 || static_cast<uint64_t>(dstSlot) >= policy_.DstSlotBound()) {
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
        const uint64_t dstBase = policy_.DstBase(rowCtx, static_cast<uint64_t>(dstSlot));
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

    DstPolicy policy_;

    GlobalTensor<int32_t> missSrc_;
    GlobalTensor<int32_t> missDst_;
    GlobalTensor<int32_t> missCounts_;
    GlobalTensor<int32_t> srcBlockTable_;
    GlobalTensor<int64_t> hostPtrs_;
    GlobalTensor<int64_t> devPtrs_;
    GlobalTensor<int64_t> params_;
    GlobalTensor<int64_t> activeGeometry_;

    GlobalTensor<uint16_t> kvSrcGm_;
    GlobalTensor<uint16_t> ropeSrcGm_;
    GlobalTensor<uint16_t> kvDstGm_;
    GlobalTensor<uint16_t> ropeDstGm_;

    TPipe *pipe_ = nullptr;
    const TilingData *tiling_ = nullptr;
    // Depth-2 queue overlaps the next GM->UB load with the previous UB->GM store.
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQueue_;
    uint64_t kvUbOffset_ = 0;
    uint64_t blockSize_ = 128;
    uint64_t missStride_ = 0;
    uint64_t kvDim_ = LCD_KV_DIM_MAX;
    uint64_t ropeDim_ = LCD_ROPE_DIM_MAX;
    uint64_t maxBlocks_ = 0;
    uint64_t activeRows_ = 0;
    uint64_t usedCore_ = 1;
};

}  // namespace vllm_ascend_lcd

#endif  // LIM_COPY_DESCRIPTORS_CORE_H_
