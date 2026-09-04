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
 * \brief MsaIndexScore Ascend 950：计算路径照搬 A2 核函数：AIC 做 paged QKᵀ，AIV 做（可选反量化）+ mask + 分段 RowMax。
 *        第一阶段：计算/握手照搬 A2，ArchTag=AtlasA5。
 *
 * 非量化：query/key 同为 bf16|fp16|fp8，score 侧不乘 scale。
 * int8 量化：key 为 int8；AIV 将 page DataCopy+Cast 到 fp 暂存后通知 AIC 做 Mmad；
 * AIV 在 mask 前按 scale[NP,N_kv,P] 对 S 做列乘完成 per-token 反量化。
 *
 * MIX 编译：`if ASCEND_IS_AIC` 拦不住模板实例化。AIV 再实例化 FP8 Cube
 *（Mmad / LoadData2DParamsV2）会让 bisheng 栈砸（exit 134）。Cube 路径仅在
 * `__DAV_C310_CUBE__` / `__DAV_310R6_CUBE__` 下编译。
 */

#ifndef MSA_INDEX_SCORE_KERNEL_ARCH35_H
#define MSA_INDEX_SCORE_KERNEL_H

#include "kernel_operator.h"

#include "catlass/msa_catlass.hpp"
#include "catlass/arch/msa_arch.hpp"
#include "catlass/arch/msa_resource.hpp"
#include "catlass/arch/msa_cross_core_sync.hpp"
#include "catlass/msa_coord.hpp"
#include "catlass/msa_gemm_coord.hpp"
#include "catlass/msa_matrix_coord.hpp"
#include "catlass/layout/msa_layout.hpp"
#include "catlass/gemm/msa_gemm_type.hpp"

#include "../msa_index_score_common.h"
#if defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__)
#include "msa_block_mmad.h"
#endif
#include "../arch22/msa_index_score_task.h"
#include "msa_seg_row_max_epilogue.h"

namespace MsaIndexScoreNs {

template <class ElementQ_, bool IS_QUANT>
class MsaIndexScoreKernel {
public:
    using ElementQ = ElementQ_;
    using ArchTag = Catlass::Arch::AtlasA5;

    using LayoutQ = Catlass::layout::RowMajor;
    using LayoutK = Catlass::layout::ColumnMajor;
    using LayoutS = Catlass::layout::RowMajor;

    using QType = Catlass::Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Catlass::Gemm::GemmType<ElementQ, LayoutK>;
    // S workspace 元素：非量化 fp16（fixpipe F322F16 直接写，写读流量减半），int8 fp32
    // （S 幅值大、1e-3 容差下 fp16 精度余量不足 15%，保持 fp32 反量化）。
    using ElementS = std::conditional_t<IS_QUANT, float, half>;
    using SType = Catlass::Gemm::GemmType<ElementS, LayoutS>;

#if defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__)
    // int8 的 K 源是每 stile 复用的 per-core scratch，保持 2 级 L1B 流水（3 级实测竞态崩溃）；
    // 同样出于该 scratch 的时序保守性，unit flag 只在非量化路径开启。
    using BlockMmad = MsaBlockMmad<QType, KType, SType, IS_QUANT ? MSA_L1B_STAGES_INT8 : MSA_L1B_STAGES_FP16, false>;
#endif

    static constexpr uint32_t REVERSE_DEPTH = MSA_WORKSPACE_STAGES - 1;
    static constexpr uint32_t STILE_WIDTH = MSA_BLOCKS_PER_STILE * MSA_BLOCK_SIZE;
    static constexpr bool IS_QUANT_V = IS_QUANT;
    static constexpr bool USE_C2UB = (MSA_A5_USE_C2UB != 0);
    static constexpr uint32_t K_SCRATCH_ELEMS = USE_C2UB ? MSA_A5_K_SCRATCH_ELEM_NUM : MSA_K_SCRATCH_ELEM_NUM;
    // C_to_UB：cast 必须在 epilogue stage 之后。放在 ping 64KB 处会盖住 T64/deq/stage，
    // 下一页 gather 会把尚未 Flush 的 score 清掉（TND/int8 多 page：1e29→0、fill 变 0）。
    static constexpr uint32_t UB_OFF_CAST_I8 =
        USE_C2UB ? (((MsaSegRowMaxEpilogue<IS_QUANT_V>::UB_TOTAL + MSA_UB_ALIGN_BYTES - 1U) / MSA_UB_ALIGN_BYTES) *
                    MSA_UB_ALIGN_BYTES) :
                   0;
    static constexpr uint32_t UB_SIZE_CAST_I8 = MSA_BLOCK_SIZE * MSA_K_TILE * sizeof(int8_t);
    static constexpr uint32_t UB_OFF_CAST_FP = UB_OFF_CAST_I8 + UB_SIZE_CAST_I8;
    static constexpr uint32_t UB_SIZE_CAST_FP = MSA_BLOCK_SIZE * MSA_K_TILE * sizeof(half);
    static_assert(UB_OFF_CAST_FP + UB_SIZE_CAST_FP <= ArchTag::UB_SIZE, "int8 cast UB out of bounds");

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
#if defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__)
        ProcessCube();
#else
        ProcessVector();
#endif
    }

private:
#if defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__)
    __aicore__ inline void ProcessCube()
    {
        Catlass::Arch::Resource<ArchTag> resource;
        BlockMmad blockMmad(resource);
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint64_t coreKScratch = static_cast<uint64_t>(coreIdx) * K_SCRATCH_ELEMS;
        const uint32_t totalTasks = scheduler_.TotalTasks();
        if constexpr (USE_C2UB) {
            ProcessCubeC2Ub(resource, blockMmad, coreIdx, coreNum, coreKScratch, totalTasks);
        } else {
            ProcessCubeGm(blockMmad, coreIdx, coreNum, coreKScratch, totalTasks);
        }
    }

    __aicore__ inline void ProcessCubeGm(BlockMmad &blockMmad, uint32_t coreIdx, uint32_t coreNum,
                                         uint64_t coreKScratch, uint32_t totalTasks)
    {
        Catlass::Arch::CrossCoreFlagWithReverse<REVERSE_DEPTH> flagSReady{MSA_FLAG_S_READY, MSA_FLAG_S_READY_REVERSE};
        Catlass::Arch::CrossCoreFlag flagKReady{MSA_FLAG_K_READY};
        const uint64_t coreWsBase = static_cast<uint64_t>(coreIdx) * MSA_WORKSPACE_STAGES * MSA_STILE_ELEM_NUM;
        uint32_t tileSeq = 0;
        MsaTask task;
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTasks; taskIdx += coreNum) {
            scheduler_.Decode(taskIdx, task);
            bool needLoadQ = true;
            for (uint32_t st = 0; st < task.numComputeSTiles; ++st) {
                const bool needKScratch = StileNeedsKScratch(task, st * MSA_BLOCKS_PER_STILE);
                if (needKScratch) {
                    Catlass::Arch::CrossCoreWaitFlag(flagKReady);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                const uint64_t sBase =
                    coreWsBase + static_cast<uint64_t>(tileSeq % MSA_WORKSPACE_STAGES) * MSA_STILE_ELEM_NUM;
                ComputeSTile(blockMmad, task, st * MSA_BLOCKS_PER_STILE, sBase, coreKScratch, needLoadQ);
                needLoadQ = false;
                Catlass::Arch::CrossCoreSetFlagWithReverse<0x2, PIPE_FIX>(flagSReady);
                ++tileSeq;
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessCubeC2Ub(Catlass::Arch::Resource<ArchTag> &resource, BlockMmad &blockMmad,
                                           uint32_t coreIdx, uint32_t coreNum, uint64_t coreKScratch,
                                           uint32_t totalTasks)
    {
        MsaTask task;
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTasks; taskIdx += coreNum) {
            scheduler_.Decode(taskIdx, task);
            bool needLoadQ = true;
            uint32_t pageSeq = 0;
            for (uint32_t blk = 0; blk < task.visibleEndBlk; ++blk) {
                const uint32_t ping = pageSeq % MSA_A5_S_STAGES;
                if (KeyBlockNeedsScratch(task, blk)) {
                    AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K);
                    AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_K + MSA_A5_AIV_FLAG_OFFSET);
                    // 与 GM 路径一致：等 AIV MTE3 写完 scratch 再 MTE2 进 L1，否则 int8/TND 读到 0。
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + ping);
                AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + ping + MSA_A5_AIV_FLAG_OFFSET);
                auto ubS = resource.ubBuf.template GetBufferByByte<float>(ping * MSA_A5_S_PING_BYTES);
                ComputePageToUb(blockMmad, task, blk, ubS, coreKScratch, needLoadQ);
                needLoadQ = false;
                AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_CV + ping);
                AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_CV + ping + MSA_A5_AIV_FLAG_OFFSET);
                ++pageSeq;
            }
        }
        AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + 0);
        AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_FIX>(MSA_A5_FLAG_VC + 1);
    }
#endif

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
        if constexpr (sizeof(T) == 1U && !std::is_same_v<T, uint8_t>) {
            CopyGmToUbPartial(ub.template ReinterpretCast<uint8_t>(), gm.template ReinterpretCast<uint8_t>(), nElem);
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

    /// AIV：把一个可见 page 的 K 收进 per-core scratch（int8 cast 或 TND fp copy+尾填充）。
    __aicore__ inline void GatherKeyPageToScratch(Catlass::Arch::Resource<ArchTag> &resource, const MsaTask &task,
                                                  uint32_t blk, uint64_t scratchOff)
    {
        AscendC::LocalTensor<int8_t> ubI8 = resource.ubBuf.template GetBufferByByte<int8_t>(UB_OFF_CAST_I8);
        AscendC::LocalTensor<ElementQ> ubFp = resource.ubBuf.template GetBufferByByte<ElementQ>(UB_OFF_CAST_FP);
        const uint32_t headDim = tiling_->headDim;
        const uint32_t nElem = MSA_BLOCK_SIZE * headDim;
        const uint64_t kOffset = KeyBlockGmOffset(task, blk);
        const uint32_t nValidTok = KeyBlockValidTokens(task, blk);
        const uint32_t nValid = nValidTok * headDim;

        if constexpr (IS_QUANT_V) {
            uint32_t copyN = nElem;
            if (tiling_->keyLayout == MSA_KEY_LAYOUT_TND) {
                const uint32_t tokenStart = task.cuKStart + blk * tiling_->blockSize;
                const uint32_t packedEnd = tokenStart + MSA_BLOCK_SIZE;
                if (tiling_->totalK > 0 && packedEnd > tiling_->totalK) {
                    copyN = nValid;
                }
            }
            if (copyN < nElem) {
                AscendC::Duplicate(ubFp, static_cast<ElementQ>(0), nElem);
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
                AscendC::Cast(ubFp, ubI8, AscendC::RoundMode::CAST_NONE, copyN);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(gKeyScratch_[scratchOff], ubFp, nElem);
        } else {
            AscendC::LocalTensor<ElementQ> ubK = resource.ubBuf.template GetBufferByByte<ElementQ>(UB_OFF_CAST_FP);
            if (nValid < nElem) {
                if constexpr (sizeof(ElementQ) == 1U) {
                    AscendC::Duplicate(ubK.template ReinterpretCast<uint8_t>(), static_cast<uint8_t>(0), nElem);
                } else {
                    AscendC::Duplicate(ubK, static_cast<ElementQ>(0), nElem);
                }
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

    /// AIV：把本 S-tile 可见 block 的 K 收进 per-core scratch（int8 cast 或 TND fp copy+尾填充）。
    __aicore__ inline void GatherKeySTileToScratch(Catlass::Arch::Resource<ArchTag> &resource, const MsaTask &task,
                                                   uint32_t blkBase, uint64_t kScratchBase)
    {
        const uint32_t pageStride = MSA_BLOCK_SIZE * MSA_K_TILE;
        for (uint32_t j = 0; j < MSA_BLOCKS_PER_STILE; ++j) {
            const uint32_t blk = blkBase + j;
            if (blk >= task.visibleEndBlk) {
                break;
            }
            if (!KeyBlockNeedsScratch(task, blk)) {
                continue;
            }
            const uint64_t scratchOff = kScratchBase + static_cast<uint64_t>(j) * pageStride;
            GatherKeyPageToScratch(resource, task, blk, scratchOff);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

#if defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__)
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

    __aicore__ inline void ComputePageToUb(BlockMmad &blockMmad, const MsaTask &task, uint32_t blk,
                                           const AscendC::LocalTensor<float> &ubS, uint64_t kScratchBase,
                                           bool needLoadL1)
    {
        const uint32_t mActual = task.mActual;
        const uint32_t headDim = tiling_->headDim;
        const LayoutQ layoutQ(mActual, headDim, tiling_->strideQn);
        const LayoutK layoutKScratch(headDim, MSA_BLOCK_SIZE, headDim);
        const LayoutK layoutKGm(headDim, MSA_BLOCK_SIZE, tiling_->strideKvToken);
        const Catlass::GemmCoord shape{mActual, MSA_BLOCK_SIZE, headDim};
        const uint64_t qOffset = static_cast<uint64_t>(task.globalRowBase) * tiling_->strideQn;
        if (KeyBlockNeedsScratch(task, blk)) {
            blockMmad.ComputeToUb(gQuery_[qOffset], layoutQ, gKeyScratch_[kScratchBase], layoutKScratch, ubS, shape,
                                  needLoadL1);
        } else if constexpr (!IS_QUANT_V) {
            const uint64_t kOffset = KeyBlockGmOffset(task, blk);
            blockMmad.ComputeToUb(gQuery_[qOffset], layoutQ, gKeyFp_[kOffset], layoutKGm, ubS, shape, needLoadL1);
        }
    }
#endif

    __aicore__ inline void ProcessVector()
    {
        Catlass::Arch::Resource<ArchTag> resource;
        MsaSegRowMaxEpilogue<IS_QUANT_V> epilogue;
        epilogue.Init(resource, tiling_->numQHeads, tiling_->strideOutHead, tiling_->strideOutToken,
                      tiling_->maxBlocksPerBatch, tiling_->strideScalePage, tiling_->strideScaleHead, IS_QUANT_V,
                      tiling_->keyLayout, tiling_->totalK, gScore_, gScale_, gBlockTable_);
        const uint32_t subBlockNum = MSA_AIV_PER_AIC;
        const uint32_t subIdx = AscendC::GetBlockIdx() % subBlockNum;
        const uint32_t coreIdx = AscendC::GetBlockIdx() / subBlockNum;
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint64_t coreKScratch = static_cast<uint64_t>(coreIdx) * K_SCRATCH_ELEMS;
        const uint32_t totalTasks = scheduler_.TotalTasks();
        if constexpr (USE_C2UB) {
            ProcessVectorC2Ub(resource, epilogue, subIdx, subBlockNum, coreIdx, coreNum, coreKScratch, totalTasks);
        } else {
            ProcessVectorGm(resource, epilogue, subIdx, subBlockNum, coreIdx, coreNum, coreKScratch, totalTasks);
        }
    }

    __aicore__ inline void ProcessVectorGm(Catlass::Arch::Resource<ArchTag> &resource,
                                           MsaSegRowMaxEpilogue<IS_QUANT_V> &epilogue, uint32_t subIdx,
                                           uint32_t subBlockNum, uint32_t coreIdx, uint32_t coreNum,
                                           uint64_t coreKScratch, uint32_t totalTasks)
    {
        Catlass::Arch::CrossCoreFlagWithReverse<REVERSE_DEPTH> flagSReady{MSA_FLAG_S_READY, MSA_FLAG_S_READY_REVERSE};
        Catlass::Arch::CrossCoreFlag flagKReady{MSA_FLAG_K_READY};
        const uint64_t coreWsBase = static_cast<uint64_t>(coreIdx) * MSA_WORKSPACE_STAGES * MSA_STILE_ELEM_NUM;
        uint32_t tileSeq = 0;
        MsaTask task;
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTasks; taskIdx += coreNum) {
            scheduler_.Decode(taskIdx, task);
            epilogue.BeginTask(task, subIdx, subBlockNum);
            for (uint32_t st = 0; st < task.numComputeSTiles; ++st) {
                if (StileNeedsKScratch(task, st * MSA_BLOCKS_PER_STILE)) {
                    if (subIdx == 0) {
                        GatherKeySTileToScratch(resource, task, st * MSA_BLOCKS_PER_STILE, coreKScratch);
                    }
                    Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                    Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagKReady);
                }
                Catlass::Arch::CrossCoreWaitFlagWithReverse<0x2, PIPE_MTE3>(flagSReady);
                const uint64_t sBase =
                    coreWsBase + static_cast<uint64_t>(tileSeq % MSA_WORKSPACE_STAGES) * MSA_STILE_ELEM_NUM;
                epilogue.ProcessSTile(gWorkspace_[sBase], task, st * MSA_BLOCKS_PER_STILE);
                ++tileSeq;
            }
            for (uint32_t st = task.numComputeSTiles; st < task.numSTiles; ++st) {
                epilogue.ProcessSTile(gWorkspace_[0], task, st * MSA_BLOCKS_PER_STILE);
            }
            epilogue.EndTask();
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessVectorC2Ub(Catlass::Arch::Resource<ArchTag> &resource,
                                             MsaSegRowMaxEpilogue<IS_QUANT_V> &epilogue, uint32_t subIdx,
                                             uint32_t subBlockNum, uint32_t coreIdx, uint32_t coreNum,
                                             uint64_t coreKScratch, uint32_t totalTasks)
    {
        AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_VC + 0);
        AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_VC + 1);
        MsaTask task;
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTasks; taskIdx += coreNum) {
            scheduler_.Decode(taskIdx, task);
            epilogue.BeginTask(task, subIdx, subBlockNum);
            uint32_t pageSeq = 0;
            for (uint32_t blk = 0; blk < task.visibleEndBlk; ++blk) {
                const uint32_t ping = pageSeq % MSA_A5_S_STAGES;
                if (KeyBlockNeedsScratch(task, blk)) {
                    if (subIdx == 0) {
                        GatherKeyPageToScratch(resource, task, blk, coreKScratch);
                    }
                    Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();
                    // MODE4：Cube Wait K 与 K+16，两个 AIV 都要 Set。
                    AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_MTE3>(MSA_A5_FLAG_K);
                }
                AscendC::CrossCoreWaitFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_CV + ping);
                epilogue.ProcessPageUb(ping, task, blk);
                AscendC::CrossCoreSetFlag<MSA_A5_SYNC_MODE4, PIPE_V>(MSA_A5_FLAG_VC + ping);
                ++pageSeq;
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

#endif // MSA_INDEX_SCORE_KERNEL_ARCH35_H
