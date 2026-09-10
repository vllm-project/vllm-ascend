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
 *
 * 非量化：query/key 同为 bf16|fp16，score 侧不乘 scale。
 * int8 量化：key 为 int8；AIV 将 page DataCopy+Cast 到四槽 fp 暂存后通知 AIC 做 Mmad；
 * gather(st+4) 叠在 cube(st+1..st+3) 上；两 AIV 对分 8 page（每页仍串行 MTE2→V→MTE3）。
 * 同 AIV 上 MTE2∥MTE3 / 多 page 并发 MTE2 会写坏多 page int8。S 侧仍为 fp32 反量化。
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

    // int8：三槽 K scratch 后与非量化同样 3 级 L1B + unit flag（单槽时 3 级会与下一 stile
    // 的 cast 重写竞态，950 arch35 仍保持 2 级 / 关 unit flag）。
    using BlockMmad = MsaBlockMmad<QType, KType, SType, MSA_L1B_STAGES_FP16, true>;

    static constexpr uint32_t REVERSE_DEPTH = MSA_WORKSPACE_STAGES - 1;
    static constexpr uint32_t STILE_WIDTH = MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE;
    static constexpr bool IS_QUANT_V = IS_QUANT;
    static_assert(MSA_K_SCRATCH_STAGES_A2 == 4, "A2 int8 K scratch handshake assumes 4 slots");
    // int8：cast 放 epilogue 尾部。EVENT_ID2 每页成对 MTE2_V / V_MTE3 / MTE3_MTE2。
    // 非量化 TND 仍占用 epilogue S 区（gather 在 WaitS 之前，S 尚未载入）。
    static constexpr uint32_t UB_OFF_CAST_I8 = IS_QUANT ? MsaSegRowMaxEpilogue<true>::UB_OFF_TAIL : 0;
    static constexpr uint32_t UB_SIZE_CAST_I8 = MSA_BLOCK_SIZE * MSA_K_TILE * sizeof(int8_t);
    static constexpr uint32_t UB_OFF_CAST_FP = UB_OFF_CAST_I8 + UB_SIZE_CAST_I8;
    static constexpr uint32_t UB_SIZE_CAST_FP = MSA_BLOCK_SIZE * MSA_K_TILE * sizeof(half);
    static_assert(UB_OFF_CAST_FP + UB_SIZE_CAST_FP <= ArchTag::UB_SIZE, "cast UB out of bounds");
    static_assert(!IS_QUANT || UB_OFF_CAST_I8 >= MsaSegRowMaxEpilogue<true>::UB_TOTAL,
                  "int8 cast UB overlaps epilogue");

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
        Catlass::Arch::CrossCoreFlag flagK0{MSA_FLAG_K_READY};
        Catlass::Arch::CrossCoreFlag flagK1{MSA_FLAG_K_READY_REVERSE};
        Catlass::Arch::CrossCoreFlag flagK2{MSA_FLAG_K_READY_2};
        Catlass::Arch::CrossCoreFlag flagK3{MSA_FLAG_K_READY_3};

        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint64_t coreWsBase = static_cast<uint64_t>(coreIdx) * MSA_WORKSPACE_STAGES * MSA_STILE_ELEM_NUM;
        const uint32_t kStages = IS_QUANT_V ? MSA_K_SCRATCH_STAGES_A2 : 1U;
        const uint64_t coreKScratch = static_cast<uint64_t>(coreIdx) * kStages * MSA_K_SCRATCH_ELEM_NUM;
        const uint32_t totalTasks = scheduler_.TotalTasks();

        uint32_t tileSeq = 0;
        MsaTask task;
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTasks; taskIdx += coreNum) {
            scheduler_.Decode(taskIdx, task);
            // 只对可见 S-tile 做 QKᵀ + 握手；因果不可见尾由 AIV 直接写 -inf，避免空转同步。
            bool needLoadQ = true;
            for (uint32_t st = 0; st < task.numComputeSTiles; ++st) {
                const bool needKScratch = StileNeedsKScratch(task, st * MSA_BLOCKS_PER_STILE);
                if (needKScratch) {
                    // MIX 1AIC:2AIV：AIV→AIC 的 0x2 flag 需两个 AIV 都 Set 后才放行。
                    // 绑 PIPE_MTE2，避免 PIPE_ALL 把上一 stile 的 cube/fixpipe 冲掉。
                    WaitKScratchReady(flagK0, flagK1, flagK2, flagK3, st);
                }
                const uint64_t sBase =
                    coreWsBase + static_cast<uint64_t>(tileSeq % MSA_WORKSPACE_STAGES) * MSA_STILE_ELEM_NUM;
                ComputeSTile(blockMmad, task, st * MSA_BLOCKS_PER_STILE, sBase, KScratchSlot(coreKScratch, st),
                             needLoadQ);
                needLoadQ = false;
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagSReady);
                ++tileSeq;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline uint64_t KScratchSlot(uint64_t coreKScratch, uint32_t st) const
    {
        if constexpr (IS_QUANT_V) {
            return coreKScratch + static_cast<uint64_t>(st % MSA_K_SCRATCH_STAGES_A2) * MSA_K_SCRATCH_ELEM_NUM;
        }
        return coreKScratch;
    }

    __aicore__ inline void WaitKScratchReady(Catlass::Arch::CrossCoreFlag &flagK0, Catlass::Arch::CrossCoreFlag &flagK1,
                                             Catlass::Arch::CrossCoreFlag &flagK2, Catlass::Arch::CrossCoreFlag &flagK3,
                                             uint32_t st)
    {
        if constexpr (IS_QUANT_V) {
            const uint32_t slot = st % MSA_K_SCRATCH_STAGES_A2;
            if (slot == 0U) {
                Catlass::Arch::CrossCoreWaitFlag<0x2, PIPE_MTE2>(flagK0);
            } else if (slot == 1U) {
                Catlass::Arch::CrossCoreWaitFlag<0x2, PIPE_MTE2>(flagK1);
            } else if (slot == 2U) {
                Catlass::Arch::CrossCoreWaitFlag<0x2, PIPE_MTE2>(flagK2);
            } else {
                Catlass::Arch::CrossCoreWaitFlag<0x2, PIPE_MTE2>(flagK3);
            }
        } else {
            (void)flagK1;
            (void)flagK2;
            (void)flagK3;
            Catlass::Arch::CrossCoreWaitFlag<0x2, PIPE_MTE2>(flagK0);
        }
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
        if ((bytes % MSA_DATABLOCK_BYTES) == 0U) {
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

    /// int8 gather：单 i8 + 单 fp16，每页 MTE2→V→MTE3 串行。subIdx/subBlockNum 对分 page。
    __aicore__ inline void GatherKeySTileQuantPiped(Catlass::Arch::Resource<ArchTag> &resource, const MsaTask &task,
                                                    uint32_t blkBase, uint64_t kScratchBase, uint32_t subIdx,
                                                    uint32_t subBlockNum)
    {
        AscendC::LocalTensor<int8_t> ubI8 = resource.ubBuf.template GetBufferByByte<int8_t>(UB_OFF_CAST_I8);
        AscendC::LocalTensor<half> ubHalf = resource.ubBuf.template GetBufferByByte<half>(UB_OFF_CAST_FP);
        const uint32_t headDim = tiling_->headDim;
        const uint32_t nElem = MSA_BLOCK_SIZE * headDim;
        const uint32_t pageStride = MSA_BLOCK_SIZE * MSA_K_TILE;
        const uint32_t split = (subBlockNum == 0U) ? 1U : subBlockNum;

        for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
            if ((j % split) != subIdx) {
                continue;
            }
            const uint32_t blk = blkBase + j;
            if (blk >= task.visibleEndBlk) {
                continue;
            }
            if (!KeyBlockNeedsScratch(task, blk)) {
                continue;
            }
            const uint64_t kOffset = KeyBlockGmOffset(task, blk);
            const uint64_t scratchOff = kScratchBase + static_cast<uint64_t>(j) * pageStride;
            const uint32_t nValidTok = KeyBlockValidTokens(task, blk);
            const uint32_t nValid = nValidTok * headDim;

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
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            }
            if (copyN > 0) {
                if (copyN == nElem) {
                    AscendC::DataCopy(ubI8, gKeyInt8_[kOffset], nElem);
                } else {
                    CopyGmToUbPartial(ubI8, gKeyInt8_[kOffset], copyN);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID2);
                AscendC::Cast(ubHalf, ubI8, AscendC::RoundMode::CAST_NONE, copyN);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
            AscendC::DataCopy(gKeyScratch_[scratchOff], ubHalf, nElem);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        }
    }

    /// AIV：把本 S-tile 可见 block 的 K 收进 per-core scratch（int8 cast 或 TND fp copy+尾填充）。
    /// int8：两 AIV 按 j%subBlockNum 对分 page；非量化仍由调用方只让 AIV0 进来。
    __aicore__ inline void GatherKeySTileToScratch(Catlass::Arch::Resource<ArchTag> &resource, const MsaTask &task,
                                                   uint32_t blkBase, uint64_t kScratchBase, uint32_t subIdx,
                                                   uint32_t subBlockNum)
    {
        if constexpr (IS_QUANT_V) {
            GatherKeySTileQuantPiped(resource, task, blkBase, kScratchBase, subIdx, subBlockNum);
            return;
        }
        AscendC::LocalTensor<ElementQ> ubK = resource.ubBuf.template GetBufferByByte<ElementQ>(UB_OFF_CAST_FP);
        const uint32_t headDim = tiling_->headDim;
        const uint32_t nElem = MSA_BLOCK_SIZE * headDim;
        const uint32_t pageStride = MSA_BLOCK_SIZE * MSA_K_TILE;
        const uint32_t split = (subBlockNum == 0U) ? 1U : subBlockNum;

        for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
            if ((j % split) != subIdx) {
                continue;
            }
            const uint32_t blk = blkBase + j;
            if (blk >= task.visibleEndBlk) {
                continue;
            }
            if (!KeyBlockNeedsScratch(task, blk)) {
                continue;
            }
            const uint64_t kOffset = KeyBlockGmOffset(task, blk);
            const uint64_t scratchOff = kScratchBase + static_cast<uint64_t>(j) * pageStride;
            const uint32_t nValidTok = KeyBlockValidTokens(task, blk);
            const uint32_t nValid = nValidTok * headDim;

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
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void NotifyKScratchReady(Catlass::Arch::CrossCoreFlag &flagK0,
                                               Catlass::Arch::CrossCoreFlag &flagK1,
                                               Catlass::Arch::CrossCoreFlag &flagK2,
                                               Catlass::Arch::CrossCoreFlag &flagK3, uint32_t st)
    {
        if constexpr (IS_QUANT_V) {
            const uint32_t slot = st % MSA_K_SCRATCH_STAGES_A2;
            if (slot == 0U) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagK0);
            } else if (slot == 1U) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagK1);
            } else if (slot == 2U) {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagK2);
            } else {
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagK3);
            }
        } else {
            (void)flagK1;
            (void)flagK2;
            (void)flagK3;
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagK0);
        }
    }

    __aicore__ inline void IssueKGatherAndNotify(Catlass::Arch::Resource<ArchTag> &resource, const MsaTask &task,
                                                 uint32_t st, uint64_t coreKScratch, uint32_t subIdx,
                                                 uint32_t subBlockNum, Catlass::Arch::CrossCoreFlag &flagK0,
                                                 Catlass::Arch::CrossCoreFlag &flagK1,
                                                 Catlass::Arch::CrossCoreFlag &flagK2,
                                                 Catlass::Arch::CrossCoreFlag &flagK3)
    {
        if constexpr (IS_QUANT_V) {
            GatherKeySTileToScratch(resource, task, st * MSA_BLOCKS_PER_STILE, KScratchSlot(coreKScratch, st), subIdx,
                                    subBlockNum);
        } else if (subIdx == 0U) {
            GatherKeySTileToScratch(resource, task, st * MSA_BLOCKS_PER_STILE, KScratchSlot(coreKScratch, st), 0U, 1U);
        }
        NotifyKScratchReady(flagK0, flagK1, flagK2, flagK3, st);
    }

    __aicore__ inline void ComputeSTile(BlockMmad &blockMmad, const MsaTask &task, uint32_t blkBase, uint64_t sBase,
                                        uint64_t kScratchBase, bool needLoadL1)
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
    }

    __aicore__ inline void ProcessVector()
    {
        Catlass::Arch::Resource<ArchTag> resource;
        MsaSegRowMaxEpilogue<IS_QUANT_V> epilogue;
        epilogue.Init(resource, tiling_->numQHeads, tiling_->strideOutHead, tiling_->strideOutToken,
                      tiling_->maxBlocksPerBatch, tiling_->strideScalePage, tiling_->strideScaleHead, IS_QUANT_V,
                      tiling_->keyLayout, tiling_->totalK, gScore_, gScale_, gBlockTable_);
        Catlass::Arch::CrossCoreFlagWithReverse<REVERSE_DEPTH> flagSReady{MSA_FLAG_S_READY, MSA_FLAG_S_READY_REVERSE};
        Catlass::Arch::CrossCoreFlag flagK0{MSA_FLAG_K_READY};
        Catlass::Arch::CrossCoreFlag flagK1{MSA_FLAG_K_READY_REVERSE};
        Catlass::Arch::CrossCoreFlag flagK2{MSA_FLAG_K_READY_2};
        Catlass::Arch::CrossCoreFlag flagK3{MSA_FLAG_K_READY_3};

        const uint32_t subBlockNum = AscendC::GetSubBlockNum();
        const uint32_t subIdx = AscendC::GetSubBlockIdx();
        const uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint64_t coreWsBase = static_cast<uint64_t>(coreIdx) * MSA_WORKSPACE_STAGES * MSA_STILE_ELEM_NUM;
        const uint32_t kStages = IS_QUANT_V ? MSA_K_SCRATCH_STAGES_A2 : 1U;
        const uint64_t coreKScratch = static_cast<uint64_t>(coreIdx) * kStages * MSA_K_SCRATCH_ELEM_NUM;
        const uint32_t totalTasks = scheduler_.TotalTasks();

        uint32_t tileSeq = 0;
        MsaTask task;
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTasks; taskIdx += coreNum) {
            scheduler_.Decode(taskIdx, task);
            epilogue.BeginTask(task, subIdx, subBlockNum);
            if constexpr (IS_QUANT_V) {
                // 四槽：先填 0/1/2/3。WaitS(st) 后 slot[st%4] 空闲，gather(st+4) 叠在
                // cube(st+1..st+3) 上。两 AIV 对分 page，每页串行 MTE2→V→MTE3。
                const uint32_t nSt = task.numComputeSTiles;
                const uint32_t nPref = (nSt < MSA_K_SCRATCH_STAGES_A2) ? nSt : MSA_K_SCRATCH_STAGES_A2;
                for (uint32_t p = 0; p < nPref; ++p) {
                    if (StileNeedsKScratch(task, p * MSA_BLOCKS_PER_STILE)) {
                        IssueKGatherAndNotify(resource, task, p, coreKScratch, subIdx, subBlockNum, flagK0, flagK1,
                                              flagK2, flagK3);
                    }
                }
                for (uint32_t st = 0; st < nSt; ++st) {
                    Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagSReady);
                    const uint32_t stPref = st + MSA_K_SCRATCH_STAGES_A2;
                    const uint64_t sBase =
                        coreWsBase + static_cast<uint64_t>(tileSeq % MSA_WORKSPACE_STAGES) * MSA_STILE_ELEM_NUM;
                    if (stPref < nSt && StileNeedsKScratch(task, stPref * MSA_BLOCKS_PER_STILE)) {
                        IssueKGatherAndNotify(resource, task, stPref, coreKScratch, subIdx, subBlockNum, flagK0, flagK1,
                                              flagK2, flagK3);
                    }
                    epilogue.ProcessSTile(gWorkspace_[sBase], task, st * MSA_BLOCKS_PER_STILE);
                    ++tileSeq;
                }
            } else {
                for (uint32_t st = 0; st < task.numComputeSTiles; ++st) {
                    if (StileNeedsKScratch(task, st * MSA_BLOCKS_PER_STILE)) {
                        IssueKGatherAndNotify(resource, task, st, coreKScratch, subIdx, subBlockNum, flagK0, flagK1,
                                              flagK2, flagK3);
                    }
                    Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagSReady);
                    const uint64_t sBase =
                        coreWsBase + static_cast<uint64_t>(tileSeq % MSA_WORKSPACE_STAGES) * MSA_STILE_ELEM_NUM;
                    epilogue.ProcessSTile(gWorkspace_[sBase], task, st * MSA_BLOCKS_PER_STILE);
                    ++tileSeq;
                }
            }
            // 因果不可见尾：不握手，直接写 -inf（与 AIC 跳过这些 tile 对齐）。
            for (uint32_t st = task.numComputeSTiles; st < task.numSTiles; ++st) {
                epilogue.ProcessSTile(gWorkspace_[0], task, st * MSA_BLOCKS_PER_STILE);
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
