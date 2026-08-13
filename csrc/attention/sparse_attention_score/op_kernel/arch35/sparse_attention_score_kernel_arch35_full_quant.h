/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPARSE_ATTENTION_SCORE_KERNEL_ARCH35_FULL_QUANT_H
#define SPARSE_ATTENTION_SCORE_KERNEL_ARCH35_FULL_QUANT_H

#include "kernel_utils.hpp"

using namespace NpuArch;
using namespace tla;

namespace SasaKernelArch35 {

template <
    class BlockMmadQK,
    class EpilogueOnlineSoftmax,
    class BlockMmadPV,
    class EpilogueRescaleO,
    Format qFormat,
    Format kvFormat>
class SasaFullQuantKernelArch35 {
public:
    using ArchTag = typename BlockMmadPV::ArchTag;

    using ElementQ = typename BlockMmadQK::ElementA;
    using ElementK = typename BlockMmadQK::ElementB;
    using ElementS = typename EpilogueOnlineSoftmax::ElementInput;
    using ElementP = typename BlockMmadPV::ElementA;
    using ElementV = typename BlockMmadPV::ElementB;
    using ElementOTmp = typename BlockMmadPV::ElementC;
    using ElementO = typename EpilogueOnlineSoftmax::ElementInput;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;

    using LayoutTagL1P = typename BlockMmadPV::LayoutTagL1A;

    static constexpr uint32_t PRE_LAUNCH = 2;
    static constexpr uint32_t MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1;
    static constexpr uint32_t UB_S_OTMP_BUF_STAGES = 2;

    __aicore__ inline
    SasaFullQuantKernelArch35() {}

    __aicore__ inline
    void operator()(SasaFullQuantKernelParamsArch35 const &params)
    {
        __gm__ SparseAttentionScoreTilingData *sasaTilingData =
            reinterpret_cast<__gm__ SparseAttentionScoreTilingData *>(params.tiling);
        FetchBaseShapeInfo(sasaTilingData);
        CalcOnChipBufTileInfo(sasaTilingData);

        // global buffers
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        AscendC::GlobalTensor<ElementV> gV;
        gV.SetGlobalBuffer((__gm__ ElementV *)params.v);
        AscendC::GlobalTensor<int32_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int32_t *)params.actualQseqlen);
        AscendC::GlobalTensor<int32_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int32_t *)params.actualKvseqlen);
        AscendC::GlobalTensor<int32_t> gSelectIdx;
        gSelectIdx.SetGlobalBuffer((__gm__ int32_t *)params.selectIdx);
        AscendC::GlobalTensor<int32_t> gSelectNumIdx;
        if (params.selectNumIdx != nullptr) {
            gSelectNumIdx.SetGlobalBuffer((__gm__ int32_t *)params.selectNumIdx);
        }
        AscendC::GlobalTensor<int32_t> gBlockTable;
        gBlockTable.SetGlobalBuffer((__gm__ int32_t *)params.blockTable);
        AscendC::GlobalTensor<float> gQDequantScale;
        gQDequantScale.SetGlobalBuffer((__gm__ float *)params.qDequantScale);
        AscendC::GlobalTensor<float> gKDequantScale;
        gKDequantScale.SetGlobalBuffer((__gm__ float *)params.kDequantScale);
        AscendC::GlobalTensor<float> gVDequantScale;
        gVDequantScale.SetGlobalBuffer((__gm__ float *)params.vDequantScale);
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);

        // Identity sparse index: workspace[0]=0, used to make SparseKL1TileNLoad
        // do a contiguous read from the block start
        AscendC::GlobalTensor<int32_t> gIdentityIdx;
        gIdentityIdx.SetGlobalBuffer((__gm__ int32_t *)params.workSpace);

        // cross core data move dst buffers
        AscendC::LocalTensor<ElementP> l1PTensor[MAX_CROSS_CORE_BUF_STAGES];
        AscendC::LocalTensor<ElementS> ubSTensor[UB_S_OTMP_BUF_STAGES];
        AscendC::LocalTensor<ElementOTmp> ubOTmpTensor[UB_S_OTMP_BUF_STAGES];
        InitCrossCoreDstBuf(l1PTensor, ubSTensor, ubOTmpTensor);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        InitSyncFlags<4, 4, 4>();

        // Write identity index [0] to workspace so SparseKL1TileNLoad does contiguous read
#ifdef __DAV_CUBE__
        gIdentityIdx.SetValue(0, 0);
#endif
        AscendC::SyncAll<false>();
#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
        BlockMmadQK blockMmadQK(resource, mm1L1TileHelper_);
        BlockMmadPV blockMmadPV(resource, mm2L1AddrStart_, mm2L1TileHelper_);
#endif
#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue_);
        EpilogueRescaleO epilogueRescaleO(resource);
#endif

        uint32_t groupSize = groupSize_;
        // FP8 full-quant: per-head dequant scale prevents multi-head grouping,
        // fall back to processing one head at a time
        uint32_t effectiveGroupSize = 1;
        // Q/O stride: TND → numHeads * D
        int64_t strideQO = qHeads_ * embed_;
        // KV stride: [numPhysicalBlocks, blockSize, kvHeads, D]
        // Per-row stride within a block = kvHeads * D
        int64_t strideKVRow = kvHeads_ * embed_;
        // Per-block stride = blockSize * kvHeads * D
        int64_t strideKVBlock = static_cast<int64_t>(blockSize_) * strideKVRow;

        uint32_t embedRound = RoundUp(embed_, 16);
        uint32_t qBaseTile = 128;
        uint32_t kvBaseTile = blockSize_;

        // Task loop: each task = 1 Q token x 1 KV head group
        // For FP8, totalTaskNum was set to totalQTokens * kvHeads by host,
        // but we process groupSize heads sequentially within each task
        uint32_t maxQBlocksPerBatch = CeilDiv(maxQSeqlen_, blockSize_);

        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum_; taskIdx += coreNum) {
            uint32_t qToken = taskIdx / kvHeads_;
            uint32_t kvHeadIdx = taskIdx % kvHeads_;
            uint32_t qHeadStart = kvHeadIdx * groupSize;
            // Determine batch index and token-in-batch
            uint32_t batchIdx = 0;
            uint32_t qTokenInBatch = qToken;
            if (actSeqAval_) {
                uint32_t accum = 0;
                for (uint32_t b = 0; b < batch_; ++b) {
                    uint32_t batchLen = static_cast<uint32_t>(gActualQseqlen.GetValue(b));
                    if (qToken < accum + batchLen) {
                        batchIdx = b;
                        qTokenInBatch = qToken - accum;
                        break;
                    }
                    accum += batchLen;
                }
            }

            // selectIdx base: [kvHeadIdx, qToken, :] → kvHeadIdx * maxQSeqlen * topK + qToken * topK
            int64_t selectIdxBase = static_cast<int64_t>(kvHeadIdx) * maxQSeqlen_ * topK_ +
                                    static_cast<int64_t>(qToken) * topK_;

            // Number of valid KV blocks for this token
            int32_t selectNum = gSelectNumIdx.GetValue(static_cast<int64_t>(kvHeadIdx) * maxQSeqlen_ + qToken);
            uint32_t validTopK = (selectNum <= 0) ? 0U :
                (static_cast<uint32_t>(selectNum) < topK_) ? static_cast<uint32_t>(selectNum) : topK_;

            if (validTopK == 0) {
                continue;
            }

            // Causal: last block is always the diagonal block, with partial tile
            uint32_t kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(batchIdx));
            uint32_t qSeqlen = static_cast<uint32_t>(gActualQseqlen.GetValue(batchIdx));
            uint32_t historyLen = kvSeqlen - qSeqlen;
            uint32_t lastBlockTileSize = (historyLen + qTokenInBatch) % blockSize_ + 1;

            // Build valid block list
            uint32_t kvSLoopNum = validTopK;
            int32_t validLogicalIds[16];
            int32_t validPhysicalIds[16];
            uint32_t validTileSize[16];

            for (uint32_t i = 0; i < kvSLoopNum; i++) {
                int32_t logicalId = gSelectIdx.GetValue(selectIdxBase + i);
                int32_t physicalId = gBlockTable.GetValue(
                    static_cast<int64_t>(batchIdx) * maxBlocksPerBatch_ + logicalId);
                validLogicalIds[i] = logicalId;
                validPhysicalIds[i] = physicalId;
                uint32_t lastLogicalBlockId = (historyLen + qTokenInBatch) / blockSize_;
                validTileSize[i] = (static_cast<uint32_t>(logicalId) == lastLogicalBlockId) ?
                    lastBlockTileSize : blockSize_;
            }

            // FP8: process each head in the group sequentially (per-head dequant scale)
            for (uint32_t headInGroup = 0; headInGroup < groupSize; headInGroup++) {
            uint32_t qHeadIdx = qHeadStart + headInGroup;
            int64_t gmOffsetQHead = static_cast<int64_t>(qToken) * strideQO +
                                    static_cast<int64_t>(qHeadIdx) * embed_;
            int64_t gmOffsetOHead = gmOffsetQHead;

#ifdef __DAV_CUBE__
            uint32_t logicalQBlock = (historyLen + qTokenInBatch) / blockSize_;
            uint32_t qDequantScaleOffset = batchIdx * qHeads_ * maxQBlocksPerBatch +
                qHeadIdx * maxQBlocksPerBatch + logicalQBlock;
            float qDequantScaleValue = gQDequantScale.GetValue(qDequantScaleOffset);
#endif
            uint32_t rowNum = effectiveGroupSize;
            uint32_t rowNumRound = RoundUp(rowNum, 16);
#ifdef __DAV_CUBE__
            auto gmQLayoutTla = tla::MakeLayout<ElementQ, LayoutQ>(qBaseTile, strideQO);
            auto gmQTensorTla = tla::MakeTensor(gQ[gmOffsetQHead], gmQLayoutTla, Arch::PositionGM{});
            GemmCoord actualBlockShapeQ{rowNum, embed_, 0};
            blockMmadQK.loadQGM(gmQTensorTla, actualBlockShapeQ);
#endif
            for (uint32_t kvBlockIdx = 0; kvBlockIdx < kvSLoopNum + PRE_LAUNCH; kvBlockIdx++) {
                
                if (kvBlockIdx < kvSLoopNum) {
                    uint32_t kvSTileSizeAct = validTileSize[kvBlockIdx];
                    int32_t physicalBlockId = validPhysicalIds[kvBlockIdx];

                    // K GM offset: physicalBlockId * strideKVBlock + kvHeadIdx * D
                    int64_t gmOffsetK = static_cast<int64_t>(physicalBlockId) * strideKVBlock +
                                        static_cast<int64_t>(kvHeadIdx) * embed_;

                    GemmCoord actualBlockShapeQK{rowNum, kvSTileSizeAct, embed_};
                    uint32_t ubSBufId = kvBlockIdx % UB_S_OTMP_BUF_STAGES;
                    auto ubSLayoutTla = tla::MakeLayout<ElementS, LayoutS>(rowNumRound, RoundUp(kvSTileSizeAct, 16));
                    auto ubSTensorTla = tla::MakeTensor(ubSTensor[ubSBufId], ubSLayoutTla, Arch::PositionUB{});
                    uint32_t Mm1ToSmFlagId = ubSBufId;
                    Arch::CrossCoreFlag mm1ToSmFlag(Mm1ToSmFlagId);

#ifdef __DAV_CUBE__
                    // K tensor: ColumnMajor layout, stride = strideKVRow (kvHeads*D)
                    // gK[gmOffsetK] points to start of current physical block for this kv head
                    auto gmKLayoutTla = tla::MakeLayout<ElementK, LayoutK>(strideKVRow, blockSize_);
                    auto gmKTensorTla = tla::MakeTensor(gK[gmOffsetK], gmKLayoutTla, Arch::PositionGM{});

                    uint64_t prefixSumL0AStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        kvBlockIdx, mm1L0ATotalStages_, mm2L0ATotalStages_, kvSLoopNum, true);
                    uint64_t prefixSumL0BStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        kvBlockIdx, mm1L0BTotalStages_, mm2L0BTotalStages_, kvSLoopNum, true);

                    uint32_t kvDequantScaleOffset = batchIdx * kvHeads_ * maxBlocksPerBatch_ +
                        kvHeadIdx * maxBlocksPerBatch_ +
                        static_cast<uint32_t>(validLogicalIds[kvBlockIdx]);
                    float kDequantScaleValue = gKDequantScale.GetValue(kvDequantScaleOffset);
                    float combinedDeqScalar = qDequantScaleValue * kDequantScaleValue;
                    uint64_t deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&combinedDeqScalar));
                    blockMmadQK(
                        gmKTensorTla, ubSTensorTla, gIdentityIdx,
                        actualBlockShapeQK,
                        0, blockSize_,
                        blockSize_, blockSize_, 1, 1,
                        prefixSumL0AStages, prefixSumL0BStages,
                        mm1ToSmFlag, deqScalar);
                    if (kvBlockIdx == kvSLoopNum - 1)
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
#endif
                    // Online Softmax
                    uint32_t l1PBufId = kvBlockIdx % pL1BufNum_;
                    uint32_t smToMm2FlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                    Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);
                    auto l1PLayoutTla = tla::MakeLayout<ElementP, NpuArch::layout::zN>(rowNum, kvSTileSizeAct);
                    auto l1PTensorTla = tla::MakeTensor(l1PTensor[l1PBufId], l1PLayoutTla, Arch::PositionL1{});

#ifdef __DAV_VEC__
                    epilogueOnlineSoftmax(
                        l1PTensorTla,
                        actualBlockShapeQK,
                        (kvBlockIdx == 0),
                        ubSBufId,
                        l1PBufId,
                        mm1ToSmFlag,
                        smToMm2Flag);
#endif
                }
                if (kvBlockIdx >= PRE_LAUNCH) {
                    uint32_t kvBlockIdxDe = kvBlockIdx - PRE_LAUNCH;
                    uint32_t kvSTileSizeAct = validTileSize[kvBlockIdxDe];
                    int32_t physicalBlockIdV = validPhysicalIds[kvBlockIdxDe];

                    int64_t gmOffsetV = static_cast<int64_t>(physicalBlockIdV) * strideKVBlock +
                                        static_cast<int64_t>(kvHeadIdx) * embed_;

                    // PV matmul
                    GemmCoord actualBlockShapePV{rowNum, embed_, kvSTileSizeAct};
                    uint32_t ubOTmpBufId = kvBlockIdxDe % UB_S_OTMP_BUF_STAGES;
                    uint32_t Mm2ToReFlagId = ubOTmpBufId + UB_S_OTMP_BUF_STAGES + pL1BufNum_;

#ifdef __DAV_CUBE__
                    uint32_t l1PBufId = kvBlockIdxDe % pL1BufNum_;
                    auto ubOTmpLayoutTla = tla::MakeLayout<ElementOTmp, LayoutOTmp>(rowNumRound, embedRound);
                    auto ubOTmpTensorTla = tla::MakeTensor(ubOTmpTensor[ubOTmpBufId],
                        ubOTmpLayoutTla, Arch::PositionUB{});
                    uint32_t smToMm2FlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                    Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);
                    Arch::CrossCoreFlag mm2ToReFlag(Mm2ToReFlagId);

                    auto gmVLayoutTla = tla::MakeLayout<ElementV, LayoutV>(blockSize_, strideKVRow);
                    auto gmVTensorTla = tla::MakeTensor(gV[gmOffsetV], gmVLayoutTla, Arch::PositionGM{});

                    uint64_t prefixSumL0AStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        kvBlockIdxDe, mm1L0ATotalStages_, mm2L0ATotalStages_, kvSLoopNum, false);
                    uint64_t prefixSumL0BStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        kvBlockIdxDe, mm1L0BTotalStages_, mm2L0BTotalStages_, kvSLoopNum, false);

                    uint32_t kvDequantScaleOffsetDe = batchIdx * kvHeads_ * maxBlocksPerBatch_ +
                        kvHeadIdx * maxBlocksPerBatch_ +
                        static_cast<uint32_t>(validLogicalIds[kvBlockIdxDe]);
                    float vDequantScaleValue = gVDequantScale.GetValue(kvDequantScaleOffsetDe);
                    uint64_t deqScalarPv = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&vDequantScaleValue));
                    blockMmadPV(
                        gmVTensorTla, ubOTmpTensorTla, gIdentityIdx,
                        actualBlockShapePV,
                        0, blockSize_,
                        blockSize_, blockSize_, 1, 1,
                        prefixSumL0AStages, prefixSumL0BStages,
                        smToMm2Flag, mm2ToReFlag, deqScalarPv);
#endif
#ifdef __DAV_VEC__
                    Arch::CrossCoreFlag mm2ToReFlag(Mm2ToReFlagId);
                    uint32_t curTileMod = kvBlockIdxDe % (PRE_LAUNCH + 1);
                    auto gmOLayoutTla = tla::MakeLayout<ElementO, LayoutO>(qBaseTile, strideQO);
                    auto gmOTensorTla = tla::MakeTensor(gO[gmOffsetOHead], gmOLayoutTla, Arch::PositionGM{});
                    epilogueRescaleO(
                        gmOTensorTla, actualBlockShapePV,
                        curTileMod, kvBlockIdxDe,
                        (kvBlockIdxDe == 0),
                        (kvBlockIdxDe == kvSLoopNum - 1),
                        mm2ToReFlag,
                        true);
#endif
                }
            }
            } // end headInGroup loop
        }
        ReleaseSyncFlags<4, 4, 4>();
    }

private:
    __aicore__ inline
    void FetchBaseShapeInfo(__gm__ SparseAttentionScoreTilingData *tilingData)
    {
        batch_ = tilingData->batch;
        qHeads_ = tilingData->numHeads;
        kvHeads_ = tilingData->kvHeads;
        embed_ = tilingData->embeddingSize;
        blockSize_ = tilingData->blockSize;
        topK_ = tilingData->topK;
        maxBlocksPerBatch_ = tilingData->maxBlocksPerBatch;
        totalTaskNum_ = tilingData->totalTaskNum;
        scaleValue_ = tilingData->scaleValue;
        maxQSeqlen_ = tilingData->maxQSeqlen;
        groupSize_ = tilingData->groupSize;
        actSeqAval_ = true;
    }

    __aicore__ inline
    void CalcOnChipBufTileInfo(__gm__ SparseAttentionScoreTilingData *tilingData)
    {
        // V1: use fixed tile sizes matching blockSize
        // Use L0 tile M size for L1 tile M to match BlockMmad expectations
        mm1L1TileM_ = 128;
        mm1L1TileN_ = blockSize_;
        mm1L1TileKLeft_ = embed_;
        mm1L1TileKRight_ = embed_;
        mm2L1TileM_ = 128;
        mm2L1TileN_ = embed_;
        mm2L1TileKLeft_ = blockSize_;
        mm2L1TileKRight_ = blockSize_;
        qL1BufNum_ = 1;
        kL1BufNum_ = 1;
        vL1BufNum_ = 1;
        pL1BufNum_ = MAX_CROSS_CORE_BUF_STAGES;

        Gemm::Block::Mm1L1TileHelper mm1L1TileHelper(mm1L1TileM_, mm1L1TileN_, mm1L1TileKLeft_, mm1L1TileKRight_,
            qL1BufNum_, kL1BufNum_);
        mm1L1TileHelper_ = mm1L1TileHelper;
        Gemm::Block::Mm2L1TileHelper mm2L1TileHelper(mm2L1TileM_, mm2L1TileN_, mm2L1TileKLeft_, mm2L1TileKRight_,
            pL1BufNum_, vL1BufNum_);
        mm2L1TileHelper_ = mm2L1TileHelper;
        mm2L1AddrStart_ = mm1L1TileM_ * mm1L1TileKLeft_ * qL1BufNum_ * sizeof(ElementQ) +
            mm1L1TileKRight_ * mm1L1TileN_ * kL1BufNum_ * sizeof(ElementK);
        mm1L0ATotalStages_ = CeilDiv(mm1L1TileM_, BlockMmadQK::L0_TILE_M) *
            CeilDiv(mm1L1TileKLeft_, BlockMmadQK::L0_TILE_K);
        mm1L0BTotalStages_ = CeilDiv(mm1L1TileN_, BlockMmadQK::L0_TILE_N) *
            CeilDiv(mm1L1TileKRight_, BlockMmadQK::L0_TILE_K);
        mm2L0ATotalStages_ = CeilDiv(mm2L1TileM_, BlockMmadPV::L0_TILE_M) *
            CeilDiv(mm2L1TileKLeft_, BlockMmadPV::L0_TILE_K);
        mm2L0BTotalStages_ = CeilDiv(mm2L1TileKRight_, BlockMmadPV::L0_TILE_K) *
            CeilDiv(mm2L1TileN_, BlockMmadPV::L0_TILE_N);
    }

    __aicore__ inline
    uint64_t CalcCrossMm1Mm2PrefixSumL0ABStages(
        uint32_t kvBlockIdx, uint32_t singleMm1L0Stages,
        uint32_t singleMm2L0Stages, uint32_t kvSLoopNum,
        bool isCurPhaseMm1)
    {
        uint64_t prefixSumStages;
        if (isCurPhaseMm1) {
            if (kvBlockIdx <= PRE_LAUNCH) {
                prefixSumStages = kvBlockIdx * singleMm1L0Stages;
            } else {
                prefixSumStages = kvBlockIdx * singleMm1L0Stages +
                    (kvBlockIdx - PRE_LAUNCH) * singleMm2L0Stages;
            }
        } else {
            uint32_t mm1Done = (kvBlockIdx + 1 + PRE_LAUNCH < kvSLoopNum) ?
                (kvBlockIdx + 1 + PRE_LAUNCH) : kvSLoopNum;
            prefixSumStages = mm1Done * singleMm1L0Stages +
                kvBlockIdx * singleMm2L0Stages;
        }
        return prefixSumStages;
    }

    __aicore__ inline
    void InitCrossCoreDstBuf(
        AscendC::LocalTensor<ElementP> (&l1PTensor)[MAX_CROSS_CORE_BUF_STAGES],
        AscendC::LocalTensor<ElementS> (&ubSTensor)[UB_S_OTMP_BUF_STAGES],
        AscendC::LocalTensor<ElementOTmp> (&ubOTmpTensor)[UB_S_OTMP_BUF_STAGES])
    {
        for (uint32_t i = 0; i < pL1BufNum_; i++) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<ElementP>(
                mm2L1AddrStart_ + mm2L1TileM_ * mm2L1TileKLeft_ * sizeof(ElementP) * i);
        }
        uint32_t rowNumPerSubCore = EpilogueOnlineSoftmax::SM_ROW_MAX_ELEM_NUM;
        uint32_t colNumPerSubCore = EpilogueOnlineSoftmax::SM_COL_MAX_ELEM_NUM;
        uint32_t rescaleCol = EpilogueRescaleO::RESCALE_COL_MAX_ELEM_NUM;
        for (uint32_t i = 0; i < UB_S_OTMP_BUF_STAGES; i++) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(
                rowNumPerSubCore * colNumPerSubCore * sizeof(ElementS) * i);
            ubOTmpTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                rowNumPerSubCore * colNumPerSubCore * sizeof(ElementS) * UB_S_OTMP_BUF_STAGES +
                rowNumPerSubCore * colNumPerSubCore * sizeof(ElementP) * UB_S_OTMP_BUF_STAGES +
                rowNumPerSubCore * rescaleCol * sizeof(ElementOTmp) * i);
        }
    }

    template <uint32_t MM1_SM_MODE, uint32_t MM2_RE_MODE, uint32_t SM_MM2_MODE>
    __aicore__ inline
    void InitSyncFlags()
    {
#ifdef __DAV_CUBE__
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        if constexpr (SM_MM2_MODE == 4U) {
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(2);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(18);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(3);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(19);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(4);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(20);
        }
#endif
#ifdef __DAV_VEC__
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (MM1_SM_MODE == 4U) {
            AscendC::CrossCoreSetFlag<MM1_SM_MODE, PIPE_V>(0);
            AscendC::CrossCoreSetFlag<MM1_SM_MODE, PIPE_V>(1);
        }
        if constexpr (MM2_RE_MODE == 4U) {
            AscendC::CrossCoreSetFlag<MM2_RE_MODE, PIPE_V>(5);
            AscendC::CrossCoreSetFlag<MM2_RE_MODE, PIPE_V>(6);
        }
#endif
    }

    template <uint32_t MM1_SM_MODE, uint32_t MM2_RE_MODE, uint32_t SM_MM2_MODE>
    __aicore__ inline
    void ReleaseSyncFlags()
    {
#ifdef __DAV_CUBE__
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        if constexpr (MM1_SM_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(0);
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(1);
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(16);
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(17);
        }
        if constexpr (MM2_RE_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(5);
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(21);
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(6);
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(22);
        }
#endif
#ifdef __DAV_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (SM_MM2_MODE == 4U) {
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(2);
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(3);
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(4);
        }
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    Arch::Resource<ArchTag> resource;
    uint32_t batch_;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    uint32_t embed_;
    uint32_t blockSize_;
    uint32_t topK_;
    uint32_t maxBlocksPerBatch_;
    uint32_t totalTaskNum_;
    float scaleValue_;
    uint32_t maxQSeqlen_;
    uint32_t groupSize_;
    uint32_t actSeqAval_;
    // L1 tile info
    uint32_t mm1L1TileM_;
    uint32_t mm1L1TileN_;
    uint32_t mm1L1TileKLeft_;
    uint32_t mm1L1TileKRight_;
    uint32_t mm2L1TileM_;
    uint32_t mm2L1TileN_;
    uint32_t mm2L1TileKLeft_;
    uint32_t mm2L1TileKRight_;
    uint32_t qL1BufNum_;
    uint32_t kL1BufNum_;
    uint32_t vL1BufNum_;
    uint32_t pL1BufNum_;
    uint32_t mm1L0ATotalStages_;
    uint32_t mm1L0BTotalStages_;
    uint32_t mm2L0ATotalStages_;
    uint32_t mm2L0BTotalStages_;
    uint32_t mm2L1AddrStart_ = 0;
    Gemm::Block::Mm1L1TileHelper mm1L1TileHelper_;
    Gemm::Block::Mm2L1TileHelper mm2L1TileHelper_;
};

}  // namespace SasaKernelArch35

#endif  // SPARSE_ATTENTION_SCORE_KERNEL_ARCH35_FULL_QUANT_H
