/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../arch35/kernel_utils.hpp"
#include "../kernel_common.hpp"

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
class SasaRegularKernelArch35 {
public:
    using ArchTag = typename BlockMmadPV::ArchTag;

    using ElementQ = typename BlockMmadQK::ElementA;
    using ElementK = typename BlockMmadQK::ElementB;
    using ElementS = typename EpilogueOnlineSoftmax::ElementInput;
    using ElementP = typename BlockMmadPV::ElementA;
    using ElementV = typename BlockMmadPV::ElementB;
    using ElementOTmp = typename BlockMmadPV::ElementC;
    using ElementO = typename BlockMmadQK::ElementA;

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
    SasaRegularKernelArch35() {}

    __aicore__ inline
    void operator()(SasaKernelParamsArch35 const &params)
    {
        __gm__ SparseAttn::SparseAttentionScoreTilingData *tilingData =
            reinterpret_cast<__gm__ SparseAttn::SparseAttentionScoreTilingData *>(params.tiling);
        FetchBaseShapeInfo(tilingData);
        CalcOnChipBufTileInfo(tilingData);

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK *)params.v);
        AscendC::GlobalTensor<int32_t> gSelectIdx;
        gSelectIdx.SetGlobalBuffer((__gm__ int32_t *)params.selectIdx);
        AscendC::GlobalTensor<int32_t> gBlockTable;
        gBlockTable.SetGlobalBuffer((__gm__ int32_t *)params.blockTable);
        AscendC::GlobalTensor<int32_t> gSelectNumIdx;
        gSelectNumIdx.SetGlobalBuffer((__gm__ int32_t *)params.selectNumIdx);
        AscendC::GlobalTensor<int32_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int32_t *)params.actualQseqlen);
        AscendC::GlobalTensor<int32_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int32_t *)params.actualKvseqlen);
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
        AscendC::GlobalTensor<int32_t> gIdentityIdx;
        gIdentityIdx.SetGlobalBuffer((__gm__ int32_t *)params.workSpace);

        AscendC::LocalTensor<ElementP> l1PTensor[MAX_CROSS_CORE_BUF_STAGES];
        AscendC::LocalTensor<ElementS> ubSTensor[UB_S_OTMP_BUF_STAGES];
        AscendC::LocalTensor<ElementOTmp> ubOTmpTensor[UB_S_OTMP_BUF_STAGES];
        InitCrossCoreDstBuf(l1PTensor, ubSTensor, ubOTmpTensor);

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

        InitSyncFlags<4, 4, 4>();

#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
        gIdentityIdx.SetValue(0, 0);
        for (uint32_t i = 1; i < topK_; i++) {
            gIdentityIdx.SetValue(i, 0);
        }
#endif
        AscendC::SyncAll<false>();
#ifdef __DAV_CUBE__
        BlockMmadQK blockMmadQK(resource, mm1L1TileHelper_);
        BlockMmadPV blockMmadPV(resource, mm2L1AddrStart_, mm2L1TileHelper_);
#endif
#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue_);
        EpilogueRescaleO epilogueRescaleO(resource);
#endif

        uint32_t groupSize = groupSize_;
        int64_t strideQO = qHeads_ * embed_;
        int64_t strideKVBlock = static_cast<int64_t>(blockSize_) * kvHeads_ * embed_;
        int64_t strideKVRow = kvHeads_ * embed_;
        uint32_t embedRound = RoundUp(embed_, 16);

        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum_; taskIdx += coreNum) {
            uint32_t qToken = taskIdx / kvHeads_;
            uint32_t kvHeadIdx = taskIdx % kvHeads_;
            uint32_t qHeadStart = kvHeadIdx * groupSize;
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

            int64_t gmOffsetQ = static_cast<int64_t>(qToken) * strideQO +
                                static_cast<int64_t>(qHeadStart) * embed_;
            int64_t gmOffsetO = gmOffsetQ;

            int64_t selectIdxBase = static_cast<int64_t>(kvHeadIdx) * maxQSeqlen_ * topK_ +
                                    static_cast<int64_t>(qToken) * topK_;
            uint32_t validTopK = topK_;
            if (params.selectNumIdx != nullptr) {
                int64_t selectNumOffset = static_cast<int64_t>(kvHeadIdx) * maxQSeqlen_ + qToken;
                validTopK = static_cast<uint32_t>(gSelectNumIdx.GetValue(selectNumOffset));
            }
            if (validTopK == 0) continue;

            uint32_t kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(batchIdx));
            uint32_t qSeqlen = static_cast<uint32_t>(gActualQseqlen.GetValue(batchIdx));
            uint32_t historyLen = kvSeqlen - qSeqlen;
            uint32_t lastBlockTileSize = (historyLen + qTokenInBatch) % blockSize_ + 1;

            uint32_t kvSLoopNum = validTopK;
            int32_t validPhysicalIds[16];
            uint32_t validTileSize[16];
            uint32_t lastLogicalBlockId = (historyLen + qTokenInBatch) / blockSize_;
            uint32_t actualLoopNum = 0;
            for (uint32_t i = 0; i < kvSLoopNum && i < topK_; i++) {
                int32_t logicalId = gSelectIdx.GetValue(selectIdxBase + i);
                if (logicalId < 0) continue;
                int64_t btOffset = static_cast<int64_t>(batchIdx) * maxBlocksPerBatch_ + logicalId;
                int32_t physicalId = gBlockTable.GetValue(btOffset);
                validPhysicalIds[actualLoopNum] = physicalId;
                validTileSize[actualLoopNum] = (static_cast<uint32_t>(logicalId) == lastLogicalBlockId) ?
                    lastBlockTileSize : blockSize_;
                actualLoopNum++;
            }
            kvSLoopNum = actualLoopNum;

            uint32_t rowNum = groupSize;
            uint32_t rowNumRound = RoundUp(rowNum, 16);

#ifdef __DAV_CUBE__
            auto gmQLayoutTla = tla::MakeLayout<ElementQ, LayoutQ>(qBaseTile_, embed_);
            auto gmQTensorTla = tla::MakeTensor(gQ[gmOffsetQ], gmQLayoutTla, Arch::PositionGM{});
            GemmCoord actualBlockShapeQ{rowNum, embed_, 0};
            blockMmadQK.loadQGM(gmQTensorTla, actualBlockShapeQ);
#endif
#ifdef __DAV_VEC__
            auto gmOLayoutTla = tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, embed_);
            auto gmOTensorTla = tla::MakeTensor(gO[gmOffsetO], gmOLayoutTla, Arch::PositionGM{});
#endif

            for (uint32_t kvBlockIdx = 0; kvBlockIdx < kvSLoopNum + PRE_LAUNCH; kvBlockIdx++) {
                if (kvBlockIdx < kvSLoopNum) {
                    uint32_t kvSTileSizeAct = validTileSize[kvBlockIdx];
                    int32_t physicalBlockId = validPhysicalIds[kvBlockIdx];

                    int64_t gmOffsetK = static_cast<int64_t>(physicalBlockId) * strideKVBlock +
                                        static_cast<int64_t>(kvHeadIdx) * embed_;

                    GemmCoord actualBlockShapeQK{rowNum, kvSTileSizeAct, embed_};
                    uint32_t ubSBufId = kvBlockIdx % UB_S_OTMP_BUF_STAGES;
                    auto ubSLayoutTla = tla::MakeLayout<ElementS, LayoutS>(
                        rowNumRound, RoundUp(kvSTileSizeAct, 16));
                    auto ubSTensorTla = tla::MakeTensor(
                        ubSTensor[ubSBufId], ubSLayoutTla, Arch::PositionUB{});
                    uint32_t Mm1ToSmFlagId = ubSBufId;
                    Arch::CrossCoreFlag mm1ToSmFlag(Mm1ToSmFlagId);

#ifdef __DAV_CUBE__
                    auto gmKLayoutTla = tla::MakeLayout<ElementK, LayoutK>(strideKVRow, blockSize_);
                    auto gmKTensorTla = tla::MakeTensor(gK[gmOffsetK], gmKLayoutTla, Arch::PositionGM{});

                    uint64_t prefixSumL0AStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        kvBlockIdx, mm1L0ATotalStages_, mm2L0ATotalStages_, kvSLoopNum, true);
                    uint64_t prefixSumL0BStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        kvBlockIdx, mm1L0BTotalStages_, mm2L0BTotalStages_, kvSLoopNum, true);
                    blockMmadQK(
                        gmKTensorTla, ubSTensorTla, gIdentityIdx,
                        actualBlockShapeQK,
                        0, blockSize_,
                        blockSize_, blockSize_, 1, 1,
                        prefixSumL0AStages, prefixSumL0BStages,
                        mm1ToSmFlag);
                    if (kvBlockIdx == kvSLoopNum - 1)
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
#endif
                    uint32_t l1PBufId = kvBlockIdx % pL1BufNum_;
                    uint32_t smToMm2FlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                    Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);
                    auto l1PLayoutTla = tla::MakeLayout<ElementP, NpuArch::layout::zN>(rowNum, kvSTileSizeAct);
                    auto l1PTensorTla = tla::MakeTensor(
                        l1PTensor[l1PBufId], l1PLayoutTla, Arch::PositionL1{});

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

                    GemmCoord actualBlockShapePV{rowNum, embed_, kvSTileSizeAct};
                    uint32_t ubOTmpBufId = kvBlockIdxDe % UB_S_OTMP_BUF_STAGES;
                    uint32_t Mm2ToReFlagId = ubOTmpBufId + UB_S_OTMP_BUF_STAGES + pL1BufNum_;

#ifdef __DAV_CUBE__
                    uint32_t l1PBufId = kvBlockIdxDe % pL1BufNum_;
                    auto ubOTmpLayoutTla = tla::MakeLayout<ElementOTmp, LayoutOTmp>(rowNumRound, embedRound);
                    auto ubOTmpTensorTla = tla::MakeTensor(
                        ubOTmpTensor[ubOTmpBufId], ubOTmpLayoutTla, Arch::PositionUB{});
                    uint32_t smToMm2FlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                    Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);
                    Arch::CrossCoreFlag mm2ToReFlag(Mm2ToReFlagId);

                    auto gmVLayoutTla = tla::MakeLayout<ElementV, LayoutV>(blockSize_, strideKVRow);
                    auto gmVTensorTla = tla::MakeTensor(gV[gmOffsetV], gmVLayoutTla, Arch::PositionGM{});

                    uint64_t prefixSumL0AStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        kvBlockIdxDe, mm1L0ATotalStages_, mm2L0ATotalStages_, kvSLoopNum, false);
                    uint64_t prefixSumL0BStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        kvBlockIdxDe, mm1L0BTotalStages_, mm2L0BTotalStages_, kvSLoopNum, false);
                    blockMmadPV(
                        gmVTensorTla, ubOTmpTensorTla, gIdentityIdx,
                        actualBlockShapePV,
                        kvBlockIdxDe, blockSize_,
                        blockSize_, blockSize_, 1, kvSLoopNum,
                        prefixSumL0AStages, prefixSumL0BStages,
                        smToMm2Flag, mm2ToReFlag);
#endif
#ifdef __DAV_VEC__
                    Arch::CrossCoreFlag mm2ToReFlag(Mm2ToReFlagId);
                    uint32_t curTileMod = kvBlockIdxDe % (PRE_LAUNCH + 1);
                    epilogueRescaleO(
                        gmOTensorTla, actualBlockShapePV,
                        curTileMod, kvBlockIdxDe,
                        (kvBlockIdxDe == 0),
                        (kvBlockIdxDe == kvSLoopNum - 1),
                        mm2ToReFlag);
#endif
                }
            }
        }
        ReleaseSyncFlags<4, 4, 4>();
    }

private:
    __aicore__ inline
    void FetchBaseShapeInfo(__gm__ SparseAttn::SparseAttentionScoreTilingData *tilingData)
    {
        batch_ = tilingData->batch;
        qHeads_ = tilingData->numHeads;
        kvHeads_ = tilingData->kvHeads;
        embed_ = tilingData->embeddingSize;
        blockSize_ = tilingData->blockSize;
        topK_ = tilingData->topK;
        maxBlocksPerBatch_ = tilingData->maxBlocksPerBatch;
        totalTaskNum_ = tilingData->totalTaskNum;
        firstBatchTaskNum_ = tilingData->firstBatchTaskNum;
        scaleValue_ = tilingData->scaleValue;
        maxQSeqlen_ = tilingData->maxQSeqlen;
        groupSize_ = tilingData->groupSize;
        qBaseTile_ = tilingData->qBaseTile;
        kvBaseTile_ = tilingData->kvBaseTile;
        actSeqAval_ = true;
    }

    __aicore__ inline
    void CalcOnChipBufTileInfo(__gm__ SparseAttn::SparseAttentionScoreTilingData *tilingData)
    {
        mm1L1TileM_ = tilingData->mm1L1TileM;
        mm1L1TileN_ = tilingData->mm1L1TileN;
        mm1L1TileKLeft_ = tilingData->mm1L1TileKLeft;
        mm1L1TileKRight_ = tilingData->mm1L1TileKRight;
        mm2L1TileM_ = tilingData->mm2L1TileM;
        mm2L1TileN_ = tilingData->mm2L1TileN;
        mm2L1TileKLeft_ = tilingData->mm2L1TileKLeft;
        mm2L1TileKRight_ = tilingData->mm2L1TileKRight;
        qL1BufNum_ = tilingData->qL1BufNum;
        kL1BufNum_ = tilingData->kL1BufNum;
        vL1BufNum_ = tilingData->vL1BufNum;
        pL1BufNum_ = tilingData->pL1BufNum;
        Gemm::Block::Mm1L1TileHelper mm1L1TileHelper(
            mm1L1TileM_, mm1L1TileN_, mm1L1TileKLeft_, mm1L1TileKRight_, qL1BufNum_, kL1BufNum_);
        mm1L1TileHelper_ = mm1L1TileHelper;
        Gemm::Block::Mm2L1TileHelper mm2L1TileHelper(
            mm2L1TileM_, mm2L1TileN_, mm2L1TileKLeft_, mm2L1TileKRight_, pL1BufNum_, vL1BufNum_);
        mm2L1TileHelper_ = mm2L1TileHelper;
        mm2L1AddrStart_ = mm1L1TileM_ * mm1L1TileKLeft_ * qL1BufNum_ * sizeof(ElementQ) +
            mm1L1TileKRight_ * mm1L1TileN_ * kL1BufNum_ * sizeof(ElementK);
        uint32_t mL0LoopQK = CeilDiv(groupSize_, static_cast<uint32_t>(BlockMmadQK::L0_TILE_M));
        uint32_t mL0LoopPV = CeilDiv(groupSize_, static_cast<uint32_t>(BlockMmadPV::L0_TILE_M));
        mm1L0ATotalStages_ = mL0LoopQK * (embed_ / BlockMmadQK::L0_TILE_K);
        mm1L0BTotalStages_ = (kvBaseTile_ / BlockMmadQK::L0_TILE_N) * (embed_ / BlockMmadQK::L0_TILE_K);
        mm2L0ATotalStages_ = mL0LoopPV * (kvBaseTile_ / BlockMmadPV::L0_TILE_K);
        mm2L0BTotalStages_ = (kvBaseTile_ / BlockMmadPV::L0_TILE_K) * (embed_ / BlockMmadPV::L0_TILE_N);
    }

    __aicore__ inline
    uint64_t CalcCrossMm1Mm2PrefixSumL0ABStages(
        uint32_t kvBlockIdx, uint32_t singleMm1L0Stages,
        uint32_t singleMm2L0Stages, uint32_t kvSLoopNum,
        bool isCurPhaseMm1)
    {
        uint64_t prefixSumStages;
        if (isCurPhaseMm1) {
            prefixSumStages = (kvBlockIdx <= PRE_LAUNCH) ?
                kvBlockIdx * singleMm1L0Stages :
                kvBlockIdx * singleMm1L0Stages + (kvBlockIdx - PRE_LAUNCH) * singleMm2L0Stages;
        } else {
            prefixSumStages = (kvBlockIdx < kvSLoopNum - PRE_LAUNCH) ?
                (kvBlockIdx + 1 + PRE_LAUNCH) * singleMm1L0Stages + kvBlockIdx * singleMm2L0Stages :
                kvSLoopNum * singleMm1L0Stages + kvBlockIdx * singleMm2L0Stages;
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
    // basic shape info
    uint32_t batch_;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    uint32_t embed_;
    uint32_t blockSize_;
    uint32_t topK_;
    uint32_t maxBlocksPerBatch_;
    uint32_t totalTaskNum_;
    uint32_t firstBatchTaskNum_;
    float scaleValue_;
    uint32_t maxQSeqlen_;
    uint32_t groupSize_;
    uint32_t actSeqAval_;
    // base tile info
    uint32_t qBaseTile_;
    uint32_t kvBaseTile_;
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

} // namespace SasaKernelArch35
