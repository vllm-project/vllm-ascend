/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPARSE_ATTENTION_SCORE_KERNEL_ARCH22_H
#define SPARSE_ATTENTION_SCORE_KERNEL_ARCH22_H

#include "../kernel_common.hpp"
#include "kernel_utils.hpp"

using namespace NpuArch;

namespace SasaKernelArch22 {

template <
    class BlockMmadQK,
    class EpilogueOnlineSoftmax,
    class BlockMmadPV,
    class EpilogueRescaleO>
class SasaRegularKernelArch22 {
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
    using LayoutLse = layout::RowMajor;
    using LayoutUpdate = layout::RowMajor;

    static constexpr uint32_t PRE_LAUNCH = 2;
    static constexpr uint32_t MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1;
    static constexpr uint64_t WORKSPACE_BLOCK_SIZE_DB = 131072;
    static constexpr uint32_t QK_READY_ID = 1;
    static constexpr uint32_t SOFTMAX_READY_ID = 2;
    static constexpr uint32_t PV_READY_ID = 3;

    __aicore__ inline
    SasaRegularKernelArch22() {}

    __aicore__ inline
    void operator()(SasaKernelParamsArch22 const &params)
    {
        __gm__ SparseAttn::SparseAttentionScoreTilingData *tilingData =
            reinterpret_cast<__gm__ SparseAttn::SparseAttentionScoreTilingData *>(params.tiling);
        FetchTilingData(tilingData);

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        AscendC::GlobalTensor<ElementV> gV;
        gV.SetGlobalBuffer((__gm__ ElementV *)params.v);
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
        AscendC::GlobalTensor<float> gLse;
        gLse.SetGlobalBuffer((__gm__ float *)params.softmaxLse);

        // Init identity tensor for paged attention (same as arch35 approach)
        AscendC::GlobalTensor<int32_t> gIdentityIdx;
        gIdentityIdx.SetGlobalBuffer((__gm__ int32_t *)params.workSpace);

        // Init workspace global tensors for 4 pipeline stages
        uint64_t identityIdxSize = static_cast<uint64_t>(topK_) * sizeof(int32_t);
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)(params.workSpace + identityIdxSize));
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)(params.workSpace + identityIdxSize + mm1OutSize_));
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace + identityIdxSize + mm1OutSize_ + smOnlineOutSize_));
        AscendC::GlobalTensor<ElementOTmp> gOUpdate;
        gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace + identityIdxSize + mm1OutSize_ +
            smOnlineOutSize_ + mm2OutSize_));

        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();

#ifdef __DAV_C220_CUBE__
        // Initialize Cube core hardware events
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);

        static constexpr uint32_t L1_QK_SIZE =
            BlockMmadQK::L1TileShape::M * BlockMmadQK::L1TileShape::K * sizeof(ElementQ) +
            BlockMmadQK::L1TileShape::N * BlockMmadQK::L1TileShape::K * sizeof(ElementK) * 2;
        BlockMmadQK blockMmadQK(resource);
        BlockMmadPV blockMmadPV(resource, L1_QK_SIZE);
#endif

#ifdef __DAV_C220_VEC__
        // Initialize hardware events for vector core
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue_);
        EpilogueRescaleO epilogueRescaleO(resource);

        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif

#ifdef __DAV_C220_CUBE__
        coreIdx = AscendC::GetBlockIdx();
        gIdentityIdx.SetValue(0, 0);
        for (uint32_t i = 1; i < topK_; i++) {
            gIdentityIdx.SetValue(i, 0);
        }
#endif
        AscendC::SyncAll<false>();

        uint32_t groupSize = groupSize_;
        int64_t strideQO = qHeads_ * embed_;
        int64_t strideKVBlock = static_cast<int64_t>(blockSize_) * kvHeads_ * embed_;
        int64_t strideKVRow = kvHeads_ * embed_;
        uint32_t embedRound = RoundUp(embed_, 16);
        uint32_t rowNumRound = RoundUp(groupSize, 16);

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
            uint32_t kvYBlockNum = (kvSeqlen + blockSize_ - 1) / blockSize_;
            int64_t blockTOffset = static_cast<int64_t>(batchIdx) * maxBlocksPerBatch_;

#ifdef __DAV_C220_CUBE__
            // Load Q into L1 once per task
            LayoutQ gmQLayout(rowNum, embed_);
            blockMmadQK.loadQGM(gQ[gmOffsetQ], gmQLayout, rowNum, groupSize, embed_);
#endif

#ifdef __DAV_C220_VEC__
            // Setup output layout for rescaleO
            LayoutO gmOLayout(qSeqlen, strideQO);
            LayoutLse gmLseLayout(qSeqlen, qHeads_);
#endif
            // kvSLoopNum = (kvSLoopNum == 1) ? 2 : kvSLoopNum;
            for (uint32_t kvBlockIdx = 0; kvBlockIdx < kvSLoopNum + PRE_LAUNCH; kvBlockIdx++) {
                // === Stage 1+2: QK Matmul & Online Softmax ===
                if (kvBlockIdx < kvSLoopNum) {
                    uint32_t kvSTileSizeAct = validTileSize[kvBlockIdx];
                    int32_t physicalBlockId = validPhysicalIds[kvBlockIdx];
                    int64_t gmOffsetK = static_cast<int64_t>(physicalBlockId) * strideKVBlock +
                                        static_cast<int64_t>(kvHeadIdx) * embed_;

                    uint32_t stageId = kvBlockIdx % MAX_CROSS_CORE_BUF_STAGES;
                    uint64_t gmOffsetS = static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * MAX_CROSS_CORE_BUF_STAGES +
                        static_cast<uint64_t>(stageId) * WORKSPACE_BLOCK_SIZE_DB;

#ifdef __DAV_C220_CUBE__
                    // Stage 1: QK Matmul
                    LayoutK gmKLayout(strideKVRow, blockSize_);
                    LayoutS ubSLayout(rowNumRound, RoundUp(kvSTileSizeAct, 16));
                    GemmCoord actualBlockShapeQK{rowNum, kvSTileSizeAct, embed_};

                    blockMmadQK(gQ[gmOffsetQ], gK[gmOffsetK], gS[gmOffsetS],
                        gBlockTable[blockTOffset], gIdentityIdx,
                        gmQLayout, gmKLayout, ubSLayout,
                        actualBlockShapeQK,
                        0, 0, blockSize_, strideKVRow,
                        blockSize_, 1, 1, kvSTileSizeAct);

                    NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady_);
#endif

#ifdef __DAV_C220_VEC__
                    // Stage 2: Online Softmax
                    uint64_t gmOffsetP = gmOffsetS;
                    LayoutS ubSLayout(rowNumRound, RoundUp(kvSTileSizeAct, 16));
                    LayoutP ubPLayout(rowNumRound, RoundUp(kvSTileSizeAct, 16));
                    GemmCoord actualBlockShapeQK{rowNum, kvSTileSizeAct, embed_};

                    NpuArch::Arch::CrossCoreWaitFlag(qkReady_);

                    epilogueOnlineSoftmax(gP[gmOffsetP], gS[gmOffsetS],
                        ubPLayout, ubSLayout,
                        actualBlockShapeQK,
                        (kvBlockIdx == 0), 0, 1, groupSize,
                        stageId, softmaxReady_);
#endif
                }

                // === Stage 3+4: PV Matmul & RescaleO ===
                if (kvBlockIdx >= PRE_LAUNCH) {
                    uint32_t kvBlockIdxDe = kvBlockIdx - PRE_LAUNCH;
                    uint32_t kvSTileSizeAct = validTileSize[kvBlockIdxDe];
                    int32_t physicalBlockIdV = validPhysicalIds[kvBlockIdxDe];
                    int64_t gmOffsetV = static_cast<int64_t>(physicalBlockIdV) * strideKVBlock +
                                        static_cast<int64_t>(kvHeadIdx) * embed_;

                    uint32_t stageId = kvBlockIdxDe % MAX_CROSS_CORE_BUF_STAGES;
                    uint64_t gmOffsetP = static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * MAX_CROSS_CORE_BUF_STAGES +
                        static_cast<uint64_t>(stageId) * WORKSPACE_BLOCK_SIZE_DB;
                    uint64_t gmOffsetOTmp = static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * MAX_CROSS_CORE_BUF_STAGES +
                        static_cast<uint64_t>(stageId) * WORKSPACE_BLOCK_SIZE_DB;

#ifdef __DAV_C220_CUBE__
                    // Stage 3: PV Matmul
                    LayoutP ubPLayout(rowNumRound, RoundUp(kvSTileSizeAct, 16));
                    LayoutV gmVLayout(blockSize_, strideKVRow);
                    LayoutOTmp ubOTmpLayout(rowNumRound, embedRound);
                    GemmCoord actualBlockShapePV{rowNum, embed_, kvSTileSizeAct};

                    blockMmadPV(gP[gmOffsetP], gV[gmOffsetV], gOTmp[gmOffsetOTmp],
                        gBlockTable[blockTOffset], gIdentityIdx,
                        ubPLayout, gmVLayout, ubOTmpLayout,
                        actualBlockShapePV,
                        0, 0, blockSize_,
                        kvSTileSizeAct, strideKVRow, 1,
                        softmaxReady_, blockSize_, 1, 1);
                    NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady_);
#endif

#ifdef __DAV_C220_VEC__
                    // Stage 4: RescaleO
                    uint64_t gmOffsetUpdate = static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB;
                    LayoutOTmp ubOTmpLayout(rowNumRound, embedRound);
                    LayoutUpdate ubUpdateLayout(rowNumRound, embedRound);
                    GemmCoord actualBlockShapePV{rowNum, embed_, validTileSize[kvBlockIdxDe]};

                    NpuArch::Arch::CrossCoreWaitFlag(pvReady_);

                    epilogueRescaleO(gO[gmOffsetO], gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate],
                        gLse[static_cast<int64_t>(qToken) * qHeads_ + qHeadStart],
                        gmOLayout, ubOTmpLayout, ubUpdateLayout, gmLseLayout,
                        actualBlockShapePV,
                        1, groupSize,
                        (kvBlockIdxDe == 0), (kvBlockIdxDe == kvSLoopNum - 1),
                        stageId);
#endif
                }
            }
        }

#ifdef __DAV_C220_CUBE__
        // Wait for all Cube core events
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
#endif

#ifdef __DAV_C220_VEC__
        // Wait for all VECTOR core events
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
#endif
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline
    void FetchTilingData(__gm__ SparseAttn::SparseAttentionScoreTilingData *tilingData)
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
        mm1OutSize_ = tilingData->mm1OutSize;
        smOnlineOutSize_ = tilingData->smOnlineOutSize;
        mm2OutSize_ = tilingData->mm2OutSize;
        updateSize_ = tilingData->updateSize;
        actSeqAval_ = true;
    }

    __aicore__ inline
    void InitSyncFlags()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(2);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(1);
    }

    __aicore__ inline
    void ReleaseSyncFlags()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(2);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(1);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag qkReady_{QK_READY_ID};
    Arch::CrossCoreFlag softmaxReady_{SOFTMAX_READY_ID};
    Arch::CrossCoreFlag pvReady_{PV_READY_ID};
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
    // workspace partition sizes
    uint64_t mm1OutSize_;
    uint64_t smOnlineOutSize_;
    uint64_t mm2OutSize_;
    uint64_t updateSize_;
};

}  // namespace SasaKernelArch22

#endif  // SPARSE_ATTENTION_SCORE_KERNEL_ARCH22_H