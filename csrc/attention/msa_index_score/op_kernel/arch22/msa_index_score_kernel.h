/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file msa_index_score_kernel.h
 * \brief MsaIndexScore Atlas A2/A3 核函数：AIC 做 paged QKᵀ，AIV 做（可选反量化）+ mask + 分段 RowMax。
 *        不支持 Ascend 950 / FP8。
 *
 * 非量化：query/key 同为 bf16|fp16，score 侧不乘 scale。
 * int8 量化：key 为 int8；AIV 将 page DataCopy+Cast 到 fp 暂存后通知 AIC 做 Mmad；
 * AIV 在 mask 前按 scale[NP,N_kv,P] 对 S 做列乘完成 per-token 反量化。
 */

#ifndef MSA_INDEX_SCORE_KERNEL_H
#define MSA_INDEX_SCORE_KERNEL_H

#include "kernel_operator.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/gemm/gemm_type.hpp"

#include "../msa_index_score_common.h"
#include "../msa_index_score_debug.h"
#include "msa_block_mmad.h"
#include "msa_index_score_task.h"
#include "msa_index_score_epilogue.h"

namespace MsaIndexScoreNs {

template <class ElementQ_, bool IS_QUANT>
class MsaIndexScoreKernel {
public:
    using ElementQ = ElementQ_;
    using ArchTag = Catlass::Arch::AtlasA2;

    using LayoutQ = Catlass::layout::RowMajor;
    using LayoutK = Catlass::layout::ColumnMajor;
    using LayoutS = Catlass::layout::RowMajor;

    using QType = Catlass::Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Catlass::Gemm::GemmType<ElementQ, LayoutK>;
    // S workspace 元素：非量化 fp16（fixpipe F322F16 直接写，写读流量减半），int8 fp32
    // （S 幅值大、1e-3 容差下 fp16 精度余量不足 15%，保持 fp32 反量化）。
    using ElementS = std::conditional_t<IS_QUANT, float, half>;
    using SType = Catlass::Gemm::GemmType<ElementS, LayoutS>;

    // int8 的 K 源是每 stile 复用的 per-core scratch，保持 2 级 L1B 流水（3 级实测竞态崩溃）；
    // 同样出于该 scratch 的时序保守性，unit flag 只在非量化路径开启。
    using BlockMmad = MsaBlockMmad<QType, KType, SType, IS_QUANT ? 2U : 3U, !IS_QUANT>;

    static constexpr uint32_t REVERSE_DEPTH = MSA_WORKSPACE_STAGES - 1;
    static constexpr uint32_t STILE_WIDTH = MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE;
    static constexpr bool IS_QUANT_V = IS_QUANT;
    // 单 page cast 暂存：int8 + half 共用 epilogue 前半段 UB（S 尚未载入）。
    static constexpr uint32_t UB_OFF_CAST_I8 = 0;
    static constexpr uint32_t UB_SIZE_CAST_I8 = MSA_BLOCK_SIZE * MSA_K_TILE * sizeof(int8_t);
    static constexpr uint32_t UB_OFF_CAST_FP = UB_OFF_CAST_I8 + UB_SIZE_CAST_I8;
    static constexpr uint32_t UB_SIZE_CAST_FP = MSA_BLOCK_SIZE * MSA_K_TILE * sizeof(half);
    static_assert(UB_OFF_CAST_FP + UB_SIZE_CAST_FP <= ArchTag::UB_SIZE, "cast UB out of bounds");

    __aicore__ inline MsaIndexScoreKernel() {}

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR blockTable, GM_ADDR scale, GM_ADDR actualSeqQlen,
                                GM_ADDR actualSeqKlen, GM_ADDR startLoc, GM_ADDR score, GM_ADDR workspace,
                                const MsaIndexScoreTilingData *__restrict tiling)
    {
        tiling_ = tiling;
        gQuery_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementQ *>(query));
        if constexpr (IS_QUANT_V) {
            gKeyInt8_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(key));
        } else {
            gKeyFp_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementQ *>(key));
        }
        if (IS_QUANT_V || tiling_->keyLayout == MSA_KEY_LAYOUT_TND) {
            gKeyScratch_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementQ *>(
                reinterpret_cast<__gm__ float *>(workspace) + tiling_->kScratchOffsetElems));
        }
        if (tiling_->keyLayout != MSA_KEY_LAYOUT_TND) {
            gBlockTable_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(blockTable));
        }
        gScale_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scale));
        gActualSeqQlen_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(actualSeqQlen));
        gActualSeqKlen_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(actualSeqKlen));
        gStartLoc_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(startLoc));
        gScore_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(score));
        gWorkspace_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementS *>(workspace));

        scheduler_.Init(tiling_->batch, tiling_->numQHeads, tiling_->maxBlocksPerBatch, tiling_->scoreBlockStride,
                        tiling_->sparseMode, tiling_->initBlocks, tiling_->localBlocks, tiling_->keyLayout,
                        gActualSeqQlen_, gActualSeqKlen_, gStartLoc_);
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIC {
            ProcessCube();
        }
        if ASCEND_IS_AIV {
            ProcessVector();
        }
    }

private:
    __aicore__ inline void ProcessCube()
    {
        Catlass::Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);
        Catlass::Arch::CrossCoreFlagWithReverse<REVERSE_DEPTH> flagSReady{MSA_FLAG_S_READY, MSA_FLAG_S_READY_REVERSE};
        Catlass::Arch::CrossCoreFlag flagKReady{MSA_FLAG_K_READY};

        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint64_t coreWsBase = static_cast<uint64_t>(coreIdx) * MSA_WORKSPACE_STAGES * MSA_STILE_ELEM_NUM;
        const uint64_t coreKScratch = static_cast<uint64_t>(coreIdx) * MSA_K_SCRATCH_ELEM_NUM;
        const uint32_t totalTasks = scheduler_.TotalTasks();

#if MSA_INDEX_SCORE_DEBUG
        if (MsaDebugPrimaryCore()) {
            AscendC::printf("[MSA_DBG][AIC] tiling batch=%u totalQ=%u Hq=%u D=%u isQuant=%u numPages=%u "
                            "totalTasks=%u\n",
                            tiling_->batch, tiling_->totalQ, tiling_->numQHeads, tiling_->headDim, tiling_->isQuant,
                            tiling_->numPages, totalTasks);
        }
#endif

        uint32_t tileSeq = 0;
        MsaTask task;
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTasks; taskIdx += coreNum) {
            scheduler_.Decode(taskIdx, task);
#if MSA_INDEX_SCORE_DEBUG
            const bool dbgTask = MsaDebugPrimaryCore() && (taskIdx == coreIdx);
#else
            constexpr bool dbgTask = false;
#endif
            // 只对可见 S-tile 做 QKᵀ + 握手；因果不可见尾由 AIV 直接写 -inf，避免空转同步。
            bool needLoadQ = true;
            for (uint32_t st = 0; st < task.numComputeSTiles; ++st) {
                const bool needKScratch = StileNeedsKScratch(task, st * MSA_BLOCKS_PER_STILE);
                if (needKScratch) {
                    // MIX 1AIC:2AIV：AIV→AIC 的 0x2 flag 需两个 AIV 都 Set 后才放行。
                    Catlass::Arch::CrossCoreWaitFlag(flagKReady);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                const uint64_t sBase =
                    coreWsBase + static_cast<uint64_t>(tileSeq % MSA_WORKSPACE_STAGES) * MSA_STILE_ELEM_NUM;
                ComputeSTile(blockMmad, task, st * MSA_BLOCKS_PER_STILE, sBase, coreKScratch, needLoadQ,
                             dbgTask && (st == 0));
                needLoadQ = false;
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagSReady);
                ++tileSeq;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline bool KeyBlockNeedsScratch(const MsaTask &task, uint32_t blk) const
    {
        if constexpr (IS_QUANT_V) {
            return true;
        }
        if (tiling_->keyLayout != MSA_KEY_LAYOUT_TND) {
            return false;
        }
        // TND fp16：完整 128-token 页与 BBND 一样直接从 packed GM mmad。
        // 仅最后一条请求的尾页可能越过 T2，需要 AIV 填零后再走 scratch。
        const uint32_t nValidTok = KeyBlockValidTokens(task, blk);
        return (nValidTok > 0) && (nValidTok < MSA_BLOCK_SIZE) && (task.batchIdx + 1U == tiling_->batch);
    }

    __aicore__ inline bool StileNeedsKScratch(const MsaTask &task, uint32_t blkBase) const
    {
        for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
            const uint32_t blk = blkBase + j;
            if (blk >= task.visibleEndBlk) {
                break;
            }
            if (KeyBlockNeedsScratch(task, blk)) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline uint32_t KeyBlockValidTokens(const MsaTask &task, uint32_t blk) const
    {
        if (tiling_->keyLayout != MSA_KEY_LAYOUT_TND) {
            return MSA_BLOCK_SIZE;
        }
        const uint32_t seqTok = blk * tiling_->blockSize;
        if (task.kvLen <= 0 || seqTok >= static_cast<uint32_t>(task.kvLen)) {
            return 0;
        }
        const uint32_t seqRemain = static_cast<uint32_t>(task.kvLen) - seqTok;
        return seqRemain < MSA_BLOCK_SIZE ? seqRemain : MSA_BLOCK_SIZE;
    }

    __aicore__ inline uint64_t KeyBlockGmOffset(const MsaTask &task, uint32_t blk) const
    {
        if (tiling_->keyLayout == MSA_KEY_LAYOUT_TND) {
            const uint32_t tokenStart = task.cuKStart + blk * tiling_->blockSize;
            const uint32_t tokStride = tiling_->numKvHeads * tiling_->headDim;
            return static_cast<uint64_t>(tokenStart) * tokStride;
        }
        const int32_t page = gBlockTable_.GetValue(task.batchIdx * tiling_->maxBlocksPerBatch + blk);
        return static_cast<uint64_t>(page) * tiling_->strideKvBlock;
    }

    template <typename T>
    __aicore__ inline void CopyGmToUbPartial(const AscendC::LocalTensor<T> &ub, const AscendC::GlobalTensor<T> &gm,
                                             uint32_t nElem)
    {
        if (nElem == 0) {
            return;
        }
        const uint32_t bytes = nElem * static_cast<uint32_t>(sizeof(T));
        if ((bytes % 32U) == 0U) {
            AscendC::DataCopy(ub, gm, nElem);
            return;
        }
        AscendC::DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = bytes;
        params.srcStride = 0;
        params.dstStride = 0;
        AscendC::DataCopyPadExtParams<T> pad;
        pad.isPad = false;
        pad.leftPadding = 0;
        pad.rightPadding = 0;
        pad.paddingValue = 0;
        AscendC::DataCopyPad(ub, gm, params, pad);
    }

    /// AIV：把本 S-tile 可见 block 的 K 收进 per-core scratch（int8 cast 或 TND fp copy+尾填充）。
    __aicore__ inline void GatherKeySTileToScratch(Catlass::Arch::Resource<ArchTag> &resource, const MsaTask &task,
                                                   uint32_t blkBase, uint64_t kScratchBase)
    {
        AscendC::LocalTensor<int8_t> ubI8 = resource.ubBuf.template GetBufferByByte<int8_t>(UB_OFF_CAST_I8);
        AscendC::LocalTensor<half> ubHalf = resource.ubBuf.template GetBufferByByte<half>(UB_OFF_CAST_FP);
        const uint32_t headDim = tiling_->headDim;
        const uint32_t nElem = MSA_BLOCK_SIZE * headDim;
        const uint32_t pageStride = MSA_BLOCK_SIZE * MSA_K_TILE;

        for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
            const uint32_t blk = blkBase + j;
            if (blk >= task.visibleEndBlk) {
                break;
            }
            if (!KeyBlockNeedsScratch(task, blk)) {
                continue;
            }
            const uint64_t kOffset = KeyBlockGmOffset(task, blk);
            const uint64_t scratchOff = kScratchBase + static_cast<uint64_t>(j) * pageStride;
            const uint32_t nValidTok = KeyBlockValidTokens(task, blk);
            const uint32_t nValid = nValidTok * headDim;

            if constexpr (IS_QUANT_V) {
                // 与已通过的 PA int8 路径对齐：整页 DataCopy+Cast（nElem），EVENT_ID0。
                // TND 仅在尾页会越过 T2 时改为部分拷贝；中间整页禁止走 CopyGmToUbPartial。
                uint32_t copyN = nElem;
                if (tiling_->keyLayout == MSA_KEY_LAYOUT_TND) {
                    const uint32_t tokenStart = task.cuKStart + blk * tiling_->blockSize;
                    const uint32_t packedEnd = tokenStart + MSA_BLOCK_SIZE;
                    if (tiling_->totalK > 0 && packedEnd > tiling_->totalK) {
                        copyN = nValid;
                    }
                }
                if (copyN < nElem) {
                    AscendC::Duplicate(ubHalf, static_cast<half>(0), nElem);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                }
                if (copyN > 0) {
                    if (copyN == nElem) {
                        AscendC::DataCopy(ubI8, gKeyInt8_[kOffset], nElem);
                    } else {
                        CopyGmToUbPartial(ubI8, gKeyInt8_[kOffset], copyN);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::Cast(ubHalf, ubI8, AscendC::RoundMode::CAST_NONE, copyN);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopy(gKeyScratch_[scratchOff], ubHalf, nElem);
            } else {
                AscendC::LocalTensor<ElementQ> ubK = resource.ubBuf.template GetBufferByByte<ElementQ>(UB_OFF_CAST_FP);
                if (nValid < nElem) {
                    AscendC::Duplicate(ubK, static_cast<ElementQ>(0), nElem);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                }
                if (nValid > 0) {
                    CopyGmToUbPartial(ubK, gKeyFp_[kOffset], nValid);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
                AscendC::DataCopy(gKeyScratch_[scratchOff], ubK, nElem);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ComputeSTile(BlockMmad &blockMmad, const MsaTask &task, uint32_t blkBase, uint64_t sBase,
                                        uint64_t kScratchBase, bool needLoadL1, bool dumpDebug)
    {
        const uint32_t mActual = task.mActual;
        const uint32_t headDim = tiling_->headDim;
        const LayoutQ layoutQ(mActual, headDim, tiling_->strideQn);
        const LayoutK layoutKScratch(headDim, MSA_BLOCK_SIZE, headDim);
        const LayoutK layoutKGm(headDim, MSA_BLOCK_SIZE, tiling_->strideKvToken);
        const LayoutS layoutS(mActual, MSA_BLOCK_SIZE, STILE_WIDTH);
        const Catlass::GemmCoord shape{mActual, MSA_BLOCK_SIZE, headDim};
        const uint64_t qOffset = static_cast<uint64_t>(task.globalRowBase) * tiling_->strideQn;
        const uint32_t pageStride = MSA_BLOCK_SIZE * MSA_K_TILE;

#if MSA_INDEX_SCORE_DEBUG
        if (dumpDebug) {
            const uint32_t qDumpN = headDim < MSA_DEBUG_DUMP_ELEMS ? headDim : MSA_DEBUG_DUMP_ELEMS;
            AscendC::printf("[MSA_DBG][AIC] ComputeSTile blkBase=%u mActual=%u headDim=%u isQuant=%u\n", blkBase,
                            mActual, headDim, static_cast<uint32_t>(IS_QUANT_V));
            MsaDumpGmSample(gQuery_[qOffset], MSA_DUMP_DESC_Q, qDumpN);
        }
#endif

        for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
            const uint32_t blk = blkBase + j;
            if (blk >= task.visibleEndBlk) {
                break;
            }
            if (KeyBlockNeedsScratch(task, blk)) {
                const uint64_t scratchOff = kScratchBase + static_cast<uint64_t>(j) * pageStride;
                blockMmad(gQuery_[qOffset], layoutQ, gKeyScratch_[scratchOff], layoutKScratch,
                          gWorkspace_[sBase + static_cast<uint64_t>(j) * MSA_BLOCK_SIZE], layoutS, shape, needLoadL1);
            } else if constexpr (!IS_QUANT_V) {
                const uint64_t kOffset = KeyBlockGmOffset(task, blk);
                blockMmad(gQuery_[qOffset], layoutQ, gKeyFp_[kOffset], layoutKGm,
                          gWorkspace_[sBase + static_cast<uint64_t>(j) * MSA_BLOCK_SIZE], layoutS, shape, needLoadL1);
            }
            needLoadL1 = false;
        }

#if MSA_INDEX_SCORE_DEBUG
        if (dumpDebug) {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::printf("[MSA_DBG][AIC] after DOT S row0 blk0:");
            MsaPrintGmFloats(gWorkspace_[sBase], MSA_DEBUG_DUMP_ELEMS);
        }
#endif
    }

    __aicore__ inline void ProcessVector()
    {
        Catlass::Arch::Resource<ArchTag> resource;
        MsaSegRowMaxEpilogue<IS_QUANT_V> epilogue;
        epilogue.Init(resource, tiling_->numQHeads, tiling_->strideOutHead, tiling_->strideOutToken,
                      tiling_->maxBlocksPerBatch, tiling_->strideScalePage, tiling_->strideScaleHead, IS_QUANT_V,
                      tiling_->keyLayout, tiling_->totalK, gScore_, gScale_, gBlockTable_);
        Catlass::Arch::CrossCoreFlagWithReverse<REVERSE_DEPTH> flagSReady{MSA_FLAG_S_READY, MSA_FLAG_S_READY_REVERSE};
        Catlass::Arch::CrossCoreFlag flagKReady{MSA_FLAG_K_READY};

        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        const uint32_t subIdx = AscendC::GetSubBlockIdx();
        const uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint64_t coreWsBase = static_cast<uint64_t>(coreIdx) * MSA_WORKSPACE_STAGES * MSA_STILE_ELEM_NUM;
        const uint64_t coreKScratch = static_cast<uint64_t>(coreIdx) * MSA_K_SCRATCH_ELEM_NUM;
        const uint32_t totalTasks = scheduler_.TotalTasks();

        uint32_t tileSeq = 0;
        MsaTask task;
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTasks; taskIdx += coreNum) {
            scheduler_.Decode(taskIdx, task);
#if MSA_INDEX_SCORE_DEBUG
            const bool dbgTask = MsaDebugPrimaryCore() && (taskIdx == coreIdx) && (subIdx == 0);
#else
            constexpr bool dbgTask = false;
#endif
            epilogue.BeginTask(task, subIdx, subBlockNum);
            for (uint32_t st = 0; st < task.numComputeSTiles; ++st) {
                if (StileNeedsKScratch(task, st * MSA_BLOCKS_PER_STILE)) {
                    // AIV0 写 scratch；AIV0/AIV1 barrier 后双方都 Set，满足 1:2 MIX 握手。
                    if (subIdx == 0) {
                        GatherKeySTileToScratch(resource, task, st * MSA_BLOCKS_PER_STILE, coreKScratch);
                    }
                    Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagKReady);
                }
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagSReady);
                const uint64_t sBase =
                    coreWsBase + static_cast<uint64_t>(tileSeq % MSA_WORKSPACE_STAGES) * MSA_STILE_ELEM_NUM;
                epilogue.ProcessSTile(gWorkspace_[sBase], task, st * MSA_BLOCKS_PER_STILE, subIdx, subBlockNum,
                                      dbgTask && (st == 0));
                ++tileSeq;
            }
            // 因果不可见尾：不握手，直接写 -inf（与 AIC 跳过这些 tile 对齐）。
            for (uint32_t st = task.numComputeSTiles; st < task.numSTiles; ++st) {
                epilogue.ProcessSTile(gWorkspace_[0], task, st * MSA_BLOCKS_PER_STILE, subIdx, subBlockNum, false);
            }
            epilogue.EndTask();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    const MsaIndexScoreTilingData *__restrict tiling_ = nullptr;
    MsaTaskScheduler scheduler_;

    AscendC::GlobalTensor<ElementQ> gQuery_;
    AscendC::GlobalTensor<ElementQ> gKeyFp_;
    AscendC::GlobalTensor<int8_t> gKeyInt8_;
    AscendC::GlobalTensor<ElementQ> gKeyScratch_;
    AscendC::GlobalTensor<int32_t> gBlockTable_;
    AscendC::GlobalTensor<float> gScale_;
    AscendC::GlobalTensor<int32_t> gActualSeqQlen_;
    AscendC::GlobalTensor<int32_t> gActualSeqKlen_;
    AscendC::GlobalTensor<int32_t> gStartLoc_;
    AscendC::GlobalTensor<float> gScore_;
    AscendC::GlobalTensor<ElementS> gWorkspace_;
};

} // namespace MsaIndexScoreNs

#endif // MSA_INDEX_SCORE_KERNEL_H
