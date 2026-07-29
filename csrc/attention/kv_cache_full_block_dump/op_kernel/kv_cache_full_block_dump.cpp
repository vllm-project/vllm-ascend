/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "kernel_operator.h"

namespace KvCacheFullBlockDumpNs {
using namespace AscendC;

constexpr int32_t NOOP_DST_BLOCK_ID = -1;

template <typename Plane0T, typename Plane1T>
class KvCacheFullBlockDumpKernel {
public:
    __aicore__ inline KvCacheFullBlockDumpKernel(
        TPipe* pipe, const KvCacheFullBlockDumpTilingData* tiling)
        : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR srcCache0,
        GM_ADDR srcCache1,
        GM_ADDR dstCache0,
        GM_ADDR dstCache1,
        GM_ADDR srcBlockIds,
        GM_ADDR dstBlockIds)
    {
        coreIdx_ = GetBlockIdx();
        if (coreIdx_ >= tiling_->usedCoreNum) {
            return;
        }

        dstBlockIdsGm_.SetGlobalBuffer((__gm__ int32_t*)dstBlockIds);

        // Graph replay keeps this operator in every layer. Probe only the
        // destination sentinel first so an idle core does not initialize its
        // 32-KiB UB queue or even bind the payload GM tensors. This preserves
        // the standard Init/Process resource lifetime for real copies while
        // making the all-no-op graph path read destination metadata only.
        if (!HasAssignedDump()) {
            return;
        }

        srcCache0Gm_.SetGlobalBuffer((__gm__ Plane0T*)srcCache0);
        srcCache1Gm_.SetGlobalBuffer((__gm__ Plane1T*)srcCache1);
        dstCache0Gm_.SetGlobalBuffer((__gm__ Plane0T*)dstCache0);
        dstCache1Gm_.SetGlobalBuffer((__gm__ Plane1T*)dstCache1);
        srcBlockIdsGm_.SetGlobalBuffer((__gm__ int32_t*)srcBlockIds);
        pipe_->InitBuffer(copyQueue_, 1, tiling_->bufferBytes);
        active_ = true;
    }

    __aicore__ inline void Process()
    {
        if (!active_) {
            return;
        }

        // Assign linear [row, plane, contiguous_chunk] tasks in a core-strided
        // order.
        // Contiguous per-core ranges would place adjacent chunks of a row on
        // the same core once taskCount exceeds AIV count. The strided mapping
        // preserves block-internal parallelism when only a few decode rows
        // carry a real dump, while remaining balanced for many prefill rows.
        // The all-no-op graph path reads only destination ids; it never reads
        // source ids or touches either payload plane.
        const int64_t tasksPerRow = tiling_->tasksPerRow;
        for (int64_t task = coreIdx_; task < tiling_->taskCount;
             task += tiling_->usedCoreNum) {
            const int64_t row = task / tasksPerRow;
            const int64_t rowTask = task - row * tasksPerRow;
            const int32_t dstBlockId = dstBlockIdsGm_.GetValue(row);
            if (dstBlockId == NOOP_DST_BLOCK_ID) {
                continue;
            }
            ASSERT_MSG(dstBlockId >= 0,
                       "KV cache dump destination block id must be >= -1");
            const int32_t srcBlockId = srcBlockIdsGm_.GetValue(row);
            // Both source and destination block 0 are valid. Only -1 is the
            // generic no-op sentinel in the destination ids.
            ASSERT_MSG(
                srcBlockId >= 0,
                "KV cache dump source block id must be non-negative");
            ASSERT_MSG(
                srcBlockId < tiling_->srcBlockNum,
                "KV cache dump source block id exceeds capacity");
            ASSERT_MSG(
                dstBlockId < tiling_->dstBlockNum,
                "KV cache dump destination block id exceeds capacity");

            if (rowTask < tiling_->plane0TasksPerRow) {
                CopyChunk(srcCache0Gm_, dstCache0Gm_, srcBlockId,
                          dstBlockId, rowTask,
                          tiling_->plane0ElementsPerBlock,
                          tiling_->plane0ChunkElements);
            } else {
                const int64_t plane1Task =
                    rowTask - tiling_->plane0TasksPerRow;
                CopyChunk(srcCache1Gm_, dstCache1Gm_, srcBlockId,
                          dstBlockId, plane1Task,
                          tiling_->plane1ElementsPerBlock,
                          tiling_->plane1ChunkElements);
            }
        }
    }

private:
    __aicore__ inline bool HasAssignedDump()
    {
        const int64_t tasksPerRow = tiling_->tasksPerRow;
        for (int64_t task = coreIdx_; task < tiling_->taskCount;
             task += tiling_->usedCoreNum) {
            const int64_t row = task / tasksPerRow;
            const int32_t dstBlockId = dstBlockIdsGm_.GetValue(row);
            ASSERT_MSG(dstBlockId >= NOOP_DST_BLOCK_ID,
                       "KV cache dump destination block id must be >= -1");
            if (dstBlockId != NOOP_DST_BLOCK_ID) {
                return true;
            }
        }
        return false;
    }

    template <typename T>
    __aicore__ inline void CopyChunk(
        GlobalTensor<T>& source,
        GlobalTensor<T>& destination,
        int32_t srcBlockId,
        int32_t dstBlockId,
        int64_t chunkIndex,
        int64_t elementsPerBlock,
        int64_t chunkElements)
    {
        const int64_t srcBase = static_cast<int64_t>(srcBlockId) *
            elementsPerBlock;
        const int64_t dstBase = static_cast<int64_t>(dstBlockId) *
            elementsPerBlock;
        const int64_t elementOffset = chunkIndex * chunkElements;
        const int64_t remainingElements = elementsPerBlock - elementOffset;
        const int64_t elementCount = remainingElements < chunkElements
            ? remainingElements
            : chunkElements;
        const uint32_t chunkBytes = static_cast<uint32_t>(
            elementCount * sizeof(T));
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1), chunkBytes, 0, 0, 0};

        LocalTensor<T> local = copyQueue_.AllocTensor<T>();
        DataCopyPad(local, source[srcBase + elementOffset], copyParams,
                    padParams);
        copyQueue_.EnQue(local);
        local = copyQueue_.DeQue<T>();
        DataCopyPad(destination[dstBase + elementOffset], local, copyParams);
        copyQueue_.FreeTensor(local);
    }

private:
    TPipe* pipe_;
    const KvCacheFullBlockDumpTilingData* tiling_;
    // The same local tensor is first filled by MTE2 and then consumed by
    // MTE3. Bind both queue positions so EnQue/DeQue carries that dependency
    // explicitly, matching the proven GM -> UB -> GM pattern used elsewhere
    // in this repository.
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> copyQueue_;
    GlobalTensor<Plane0T> srcCache0Gm_;
    GlobalTensor<Plane1T> srcCache1Gm_;
    GlobalTensor<Plane0T> dstCache0Gm_;
    GlobalTensor<Plane1T> dstCache1Gm_;
    GlobalTensor<int32_t> srcBlockIdsGm_;
    GlobalTensor<int32_t> dstBlockIdsGm_;
    int64_t coreIdx_ = 0;
    bool active_ = false;
};
}  // namespace KvCacheFullBlockDumpNs

extern "C" __global__ __aicore__ void kv_cache_full_block_dump(
    GM_ADDR srcCache0,
    GM_ADDR srcCache1,
    GM_ADDR dstCache0,
    GM_ADDR dstCache1,
    GM_ADDR srcBlockIds,
    GM_ADDR dstBlockIds,
    GM_ADDR dstCache0Out,
    GM_ADDR dstCache1Out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    KvCacheFullBlockDumpNs::KvCacheFullBlockDumpKernel<
        DTYPE_SRC_CACHE_0, DTYPE_SRC_CACHE_1>
        kernel(&pipe, &tilingData);
    kernel.Init(srcCache0, srcCache1, dstCache0, dstCache1,
                srcBlockIds, dstBlockIds);
    kernel.Process();
}
