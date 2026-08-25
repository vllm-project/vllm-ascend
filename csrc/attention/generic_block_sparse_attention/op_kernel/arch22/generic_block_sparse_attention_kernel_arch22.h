/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_KERNEL_ARCH22_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_KERNEL_ARCH22_H

#include "../kernel_common.hpp"
#include "../generic_block_sparse_attention_metadata_kernel.h"
#include "../generic_block_sparse_attention_fd_utils.h"
#include "kernel_utils.hpp"
#include "generic_block_sparse_attention_fd_combine_arch22.h"

using namespace NpuArch;

namespace GsaKernelArch22 {

template <
    class BlockMmadQK,
    class EpilogueOnlineSoftmax,
    class BlockMmadPV,
    class EpilogueRescaleO>
class GsaRegularKernelArch22 {
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
    GsaRegularKernelArch22() {}

    __aicore__ inline
    void operator()(GsaKernelParamsArch22 const &params)
    {
        __gm__ GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData *tilingData =
            reinterpret_cast<__gm__ GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData *>(params.tiling);
        FetchTilingData(tilingData, params.metaData);
        __gm__ const GsaMetadata::Metadata *meta =
            reinterpret_cast<__gm__ const GsaMetadata::Metadata *>(params.metaData);
        const uint32_t physicalAicNum = AscendC::GetBlockNum();
        if (!GsaFd::ValidateMetadata(meta, tilingData, physicalAicNum)) {
            return;
        }
        const bool fdEnabled = tilingData->fdStaticEnabled != 0U &&
            (static_cast<uint32_t>(meta->fdScheduleFlags) & GsaMetadata::FD_SCHEDULE_ENABLED) != 0U;

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        AscendC::GlobalTensor<ElementV> gV;
        gV.SetGlobalBuffer((__gm__ ElementV *)params.v);
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx;
        gSparseBlockIdx.SetGlobalBuffer((__gm__ int32_t *)params.sparseBlockIdx);
        AscendC::GlobalTensor<int32_t> gBlockTable;
        gBlockTable.SetGlobalBuffer((__gm__ int32_t *)params.blockTable);
        AscendC::GlobalTensor<int32_t> gSparseBlockCount;
        gSparseBlockCount.SetGlobalBuffer((__gm__ int32_t *)params.sparseBlockCount);
        AscendC::GlobalTensor<int64_t> gCuSeqLengths;
        if (params.cuSeqLengths != nullptr) {
            gCuSeqLengths.SetGlobalBuffer((__gm__ int64_t *)params.cuSeqLengths);
        }
        AscendC::GlobalTensor<int64_t> gCuSeqLengthsKv;
        if (params.cuSeqLengthsKv != nullptr) {
            gCuSeqLengthsKv.SetGlobalBuffer((__gm__ int64_t *)params.cuSeqLengthsKv);
        }
        AscendC::GlobalTensor<int32_t> gSequsedQ;
        const bool hasSequsedQ = (params.sequsedQ != nullptr);
        if (hasSequsedQ) {
            gSequsedQ.SetGlobalBuffer((__gm__ int32_t *)params.sequsedQ);
        }
        AscendC::GlobalTensor<int32_t> gSequsedKv;
        const bool hasSequsedKv = (params.sequsedKv != nullptr);
        if (hasSequsedKv) {
            gSequsedKv.SetGlobalBuffer((__gm__ int32_t *)params.sequsedKv);
        }
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
        AscendC::GlobalTensor<float> gLse;
        gLse.SetGlobalBuffer((__gm__ float *)params.softmaxLse);
        AscendC::GlobalTensor<float> gPartialLse;
        gPartialLse.SetGlobalBuffer((__gm__ float *)(params.workSpace + tilingData->fdPartialLseOffset));
        AscendC::GlobalTensor<float> gPartialO;
        gPartialO.SetGlobalBuffer((__gm__ float *)(params.workSpace + tilingData->fdPartialOOffset));

        // Workspace: [S][P][OTmp][OUpdate][identity]. Identity must not precede gS —
        // Fixpipe into GM immediately after identity was flaky on arch22.
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)params.workSpace);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)(params.workSpace + mm1OutSize_));
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace + mm1OutSize_ + smOnlineOutSize_));
        AscendC::GlobalTensor<ElementOTmp> gOUpdate;
        gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace + mm1OutSize_ +
            smOnlineOutSize_ + mm2OutSize_));
        AscendC::GlobalTensor<int32_t> gIdentityIdx;
        gIdentityIdx.SetGlobalBuffer((__gm__ int32_t *)(params.workSpace + mm1OutSize_ +
            smOnlineOutSize_ + mm2OutSize_ + updateSize_));

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
        int64_t strideKVRow = kvHeads_ * embed_;
        uint32_t embedRound = RoundUp(embed_, 16);
        uint32_t rowNumRound = RoundUp(groupSize, 16);

#ifdef __DAV_C220_VEC__
        // The scheduler omits storage padding from its packed task space.
        // Initialize those rows explicitly to avoid propagating stale GM data.
        if (hasSequsedQ) {
            uint32_t paddingTask = 0U;
            for (uint32_t batchIdx = 0U; batchIdx < batch_; ++batchIdx) {
                const uint32_t storageStart = static_cast<uint32_t>(gCuSeqLengths.GetValue(batchIdx));
                const uint32_t storageEnd = static_cast<uint32_t>(gCuSeqLengths.GetValue(batchIdx + 1U));
                const uint32_t storageLen = storageEnd - storageStart;
                uint32_t actualLen = static_cast<uint32_t>(gSequsedQ.GetValue(batchIdx));
                actualLen = actualLen < storageLen ? actualLen : storageLen;
                for (uint32_t token = actualLen; token < storageLen; ++token) {
                    for (uint32_t kvHeadIdx = 0U; kvHeadIdx < kvHeads_; ++kvHeadIdx, ++paddingTask) {
                        if (paddingTask % coreNum != coreIdx) {
                            continue;
                        }
                        const uint32_t qStorageToken = storageStart + token;
                        const uint32_t qHeadStart = kvHeadIdx * groupSize;
                        const uint64_t gmOffsetO =
                            (static_cast<uint64_t>(qStorageToken) * qHeads_ + qHeadStart) * embed_;
                        const uint64_t gmOffsetLse =
                            static_cast<uint64_t>(qStorageToken) * qHeads_ + qHeadStart;
                        epilogueRescaleO.WriteEmptyOutput(
                            gO[gmOffsetO], gLse[gmOffsetLse], groupSize, embed_);
                    }
                }
            }
        }
#endif

        uint32_t taskLoopStart = coreIdx;
        uint32_t taskLoopEnd = totalTaskNum_;
        uint32_t taskLoopStep = coreNum;
        uint32_t scheduleFirstBlock = 0U;
        uint32_t scheduleLastBlock = 0U;
        if (fdEnabled) {
            taskLoopStart = totalTaskNum_;
            taskLoopEnd = totalTaskNum_;
            taskLoopStep = 1U;
            if (coreIdx < static_cast<uint32_t>(meta->fdActiveCoreNum)) {
                const __gm__ GsaMetadata::DecodeSchedule &schedule = meta->decodeSchedules[coreIdx];
                taskLoopStart = static_cast<uint32_t>(schedule.baseTaskStart);
                taskLoopEnd = static_cast<uint32_t>(schedule.baseTaskEnd);
                scheduleFirstBlock = static_cast<uint32_t>(schedule.firstBlockStart);
                scheduleLastBlock = static_cast<uint32_t>(schedule.lastBlockEnd);
            }
        }

        for (uint32_t taskIdx = taskLoopStart; taskIdx < taskLoopEnd; taskIdx += taskLoopStep) {
            uint32_t rawBegin = 0U;
            uint32_t rawEnd = topK_;
            uint32_t fdPartialTaskId = 0U;
            uint32_t fdPartialCount = 0U;
            const bool isFdPartial = fdEnabled &&
                GsaFd::FindPartialTask(meta, taskIdx, coreIdx, fdPartialTaskId, fdPartialCount);
            if (fdEnabled) {
                rawBegin = taskIdx == taskLoopStart ? scheduleFirstBlock : 0U;
                rawEnd = taskIdx + 1U == taskLoopEnd ? scheduleLastBlock : topK_;
            }

            uint32_t qStorageToken = 0U;
            uint32_t qTokenInBatch = 0U;
            uint32_t batchIdx = 0U;
            uint32_t kvHeadIdx = 0U;
            if (!GsaFd::DecodeTaskStorage(taskIdx, kvHeads_, batch_, gCuSeqLengths, gSequsedQ,
                hasSequsedQ, qStorageToken, qTokenInBatch, batchIdx, kvHeadIdx)) {
                continue;
            }
            uint32_t qHeadStart = kvHeadIdx * groupSize;

            uint32_t kvStorageLen = static_cast<uint32_t>(
                gCuSeqLengthsKv.GetValue(static_cast<int64_t>(batchIdx + 1)) -
                gCuSeqLengthsKv.GetValue(static_cast<int64_t>(batchIdx)));
            uint32_t qStorageLen = static_cast<uint32_t>(
                gCuSeqLengths.GetValue(static_cast<int64_t>(batchIdx + 1)) -
                gCuSeqLengths.GetValue(static_cast<int64_t>(batchIdx)));
            uint32_t kvSeqlen = hasSequsedKv ?
                static_cast<uint32_t>(gSequsedKv.GetValue(static_cast<int64_t>(batchIdx))) : kvStorageLen;
            uint32_t qSeqlen = hasSequsedQ ?
                static_cast<uint32_t>(gSequsedQ.GetValue(static_cast<int64_t>(batchIdx))) : qStorageLen;
            int64_t gmOffsetQ = static_cast<int64_t>(qStorageToken) * strideQO +
                                static_cast<int64_t>(qHeadStart) * embed_;
            int64_t gmOffsetO = gmOffsetQ;
            const int64_t gmOffsetLse = static_cast<int64_t>(qStorageToken) * qHeads_ + qHeadStart;

            if (qSeqlen == 0U || kvSeqlen == 0U) {
#ifdef __DAV_C220_VEC__
                if (isFdPartial) {
                    epilogueRescaleO.WriteNeutralPartial(gPartialO, gPartialLse, fdPartialTaskId,
                        groupSize, embed_, tilingData->fdLseSubStride);
                } else {
                    epilogueRescaleO.WriteEmptyOutput(gO[gmOffsetO], gLse[gmOffsetLse], groupSize, embed_);
                }
#endif
                continue;
            }

            // TND + isPackedGQA=1: sparseBlockIdx 3D [N_kv, totalQBlocks, topK]
            // totalQBlocks spans storage (cu) blocks; align with metadata qStorageBlockStarts.
            uint32_t globalQBlock = 0;
            for (uint32_t b = 0; b < batchIdx; ++b) {
                uint32_t qLen = static_cast<uint32_t>(
                    gCuSeqLengths.GetValue(static_cast<int64_t>(b + 1)) -
                    gCuSeqLengths.GetValue(static_cast<int64_t>(b)));
                globalQBlock += (qLen + blockShapeX_ - 1) / blockShapeX_;
            }
            globalQBlock += qTokenInBatch / blockShapeX_;
            int64_t sparseIdxBase = static_cast<int64_t>(kvHeadIdx) * qBlockNum_ * topK_ +
                                  static_cast<int64_t>(globalQBlock) * topK_;
            uint32_t validTopK = topK_;
            if (params.sparseBlockCount != nullptr) {
                // sparseBlockCount 2D: [N_kv, totalQBlocks]
                int64_t countOffset = static_cast<int64_t>(kvHeadIdx) * qBlockNum_ +
                                      static_cast<int64_t>(globalQBlock);
                validTopK = static_cast<uint32_t>(gSparseBlockCount.GetValue(countOffset));
            }
            if (fdEnabled) {
                rawBegin = rawBegin < validTopK ? rawBegin : validTopK;
                rawEnd = rawEnd < validTopK ? rawEnd : validTopK;
            } else {
                rawEnd = validTopK;
            }

            uint32_t historyLen = kvSeqlen - qSeqlen;
            uint32_t lastBlockTileSize = (historyLen + qTokenInBatch) % blockShapeY_ + 1;

            uint32_t kvSLoopNum = rawEnd - rawBegin;
            int32_t validPhysicalIds[GsaMetadata::MAX_SPARSE_BLOCK_CAPACITY];
            uint32_t validTileSize[GsaMetadata::MAX_SPARSE_BLOCK_CAPACITY];
            uint32_t lastLogicalBlockId = (historyLen + qTokenInBatch) / blockShapeY_;
            uint32_t actualLoopNum = 0;
            for (uint32_t i = rawBegin; i < rawEnd && i < topK_; i++) {
                int32_t logicalId = gSparseBlockIdx.GetValue(sparseIdxBase + i);
                if (logicalId < 0) continue;
                int64_t btOffset = static_cast<int64_t>(batchIdx) * maxBlocksPerBatch_ + logicalId;
                int32_t physicalId = gBlockTable.GetValue(btOffset);
                validPhysicalIds[actualLoopNum] = physicalId;
                validTileSize[actualLoopNum] = (static_cast<uint32_t>(logicalId) == lastLogicalBlockId) ?
                    lastBlockTileSize : blockShapeY_;
                actualLoopNum++;
            }
            kvSLoopNum = actualLoopNum;
            if (kvSLoopNum == 0U) {
#ifdef __DAV_C220_VEC__
                if (isFdPartial) {
                    epilogueRescaleO.WriteNeutralPartial(gPartialO, gPartialLse, fdPartialTaskId,
                        groupSize, embed_, tilingData->fdLseSubStride);
                } else {
                    epilogueRescaleO.WriteEmptyOutput(gO[gmOffsetO], gLse[gmOffsetLse], groupSize, embed_);
                }
#endif
                continue;
            }
            // Disable prefetch when kv blocks <= PRE_LAUNCH (avoids empty CrossCore rounds).
            uint32_t preLaunch = (kvSLoopNum > PRE_LAUNCH) ? PRE_LAUNCH : 0;

            uint32_t rowNum = groupSize;
            int64_t blockTOffset = static_cast<int64_t>(batchIdx) * maxBlocksPerBatch_;
            // qS tile along TND S-axis; equals tokens per Q-block (tiling currently requires ==1).
            uint32_t qSBlockSize = blockShapeX_;

#ifdef __DAV_C220_CUBE__
            // Load Q into L1 once per task
            LayoutQ gmQLayout(rowNum, embed_);
            blockMmadQK.loadQGM(gQ[gmOffsetQ], gmQLayout, rowNum, groupSize, embed_);
#endif

#ifdef __DAV_C220_VEC__
            // Rescale LSE treats layout*.shape(0) as qS (must match qSBlockSize arg).
            LayoutO gmOLayout(qSBlockSize, strideQO);
            LayoutLse gmLseLayout(qSBlockSize, qHeads_);
#endif
            for (uint32_t kvBlockIdx = 0; kvBlockIdx < kvSLoopNum + preLaunch; kvBlockIdx++) {
                // === Stage 1+2: QK Matmul & Online Softmax ===
                if (kvBlockIdx < kvSLoopNum) {
                    uint32_t kvSTileSizeAct = validTileSize[kvBlockIdx];
                    int32_t physicalBlockId = validPhysicalIds[kvBlockIdx];
                    int64_t gmOffsetK = static_cast<int64_t>(physicalBlockId) * static_cast<int64_t>(kStride0_) +
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
                        (kvBlockIdx == 0), 0, qSBlockSize, groupSize,
                        stageId, softmaxReady_);
#endif
                }

                // === Stage 3+4: PV Matmul & RescaleO ===
                if (kvBlockIdx >= preLaunch) {
                    uint32_t kvBlockIdxDe = kvBlockIdx - preLaunch;
                    uint32_t kvSTileSizeAct = validTileSize[kvBlockIdxDe];
                    int32_t physicalBlockIdV = validPhysicalIds[kvBlockIdxDe];
                    int64_t gmOffsetV = static_cast<int64_t>(physicalBlockIdV) * static_cast<int64_t>(vStride0_) +
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

                    // LSE GM must use storage token index (same as O), not packed task index.
                    if (isFdPartial) {
                        epilogueRescaleO.ProcessPartial(
                            gO[gmOffsetO], gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate],
                            gLse[gmOffsetLse], gmOLayout, ubOTmpLayout, ubUpdateLayout, gmLseLayout,
                            actualBlockShapePV, qSBlockSize, groupSize,
                            (kvBlockIdxDe == 0), (kvBlockIdxDe == kvSLoopNum - 1), stageId,
                            gPartialO, gPartialLse, fdPartialTaskId, tilingData->fdLseSubStride);
                    } else {
                        epilogueRescaleO(
                            gO[gmOffsetO], gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate],
                            gLse[gmOffsetLse], gmOLayout, ubOTmpLayout, ubUpdateLayout, gmLseLayout,
                            actualBlockShapePV, qSBlockSize, groupSize,
                            (kvBlockIdxDe == 0), (kvBlockIdxDe == kvSLoopNum - 1), stageId);
                    }
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
        if (fdEnabled) {
            AscendC::SyncAll<false>();
#ifdef __DAV_C220_VEC__
            GenericBlockSparseAttentionFdCombineArch22<ElementO, Arch::Resource<ArchTag>> combine(resource);
            combine(meta, tilingData, gPartialLse, gPartialO, gO,
                gCuSeqLengths, gSequsedQ, hasSequsedQ);
#endif
        }
    }

private:
    __aicore__ inline
    void FetchTilingData(__gm__ GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData *tilingData,
                         GM_ADDR metaData)
    {
        batch_ = tilingData->batch;
        qHeads_ = tilingData->numHeads;
        kvHeads_ = tilingData->kvHeads;
        embed_ = tilingData->embeddingSize;
        blockShapeY_ = tilingData->blockShapeY;
        blockShapeX_ = tilingData->blockShapeX;
        blockSize_ = tilingData->blockSize;
        qBlockNum_ = tilingData->qBlockNum;
        topK_ = tilingData->topK;
        maxBlocksPerBatch_ = tilingData->maxBlocksPerBatch;
        // Full AICPU metadata protocol overlay (no tiling fallback for task schedule).
        __gm__ GsaMetadata::Metadata *meta = reinterpret_cast<__gm__ GsaMetadata::Metadata *>(metaData);
        totalTaskNum_ = static_cast<uint32_t>(meta->saTotalTaskNum);
        scaleValue_ = tilingData->scaleValue;
        maxQSeqlen_ = tilingData->maxQSeqlen;
        groupSize_ = tilingData->groupSize;
        qBaseTile_ = tilingData->qBaseTile;
        kvBaseTile_ = tilingData->kvBaseTile;
        mm1OutSize_ = tilingData->mm1OutSize;
        smOnlineOutSize_ = tilingData->smOnlineOutSize;
        mm2OutSize_ = tilingData->mm2OutSize;
        updateSize_ = tilingData->updateSize;
        kStride0_ = tilingData->kStride0;
        vStride0_ = tilingData->vStride0;
    }

    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag qkReady_{QK_READY_ID};
    Arch::CrossCoreFlag softmaxReady_{SOFTMAX_READY_ID};
    Arch::CrossCoreFlag pvReady_{PV_READY_ID};
    // basic shape info
    uint32_t batch_;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    uint32_t embed_;
    uint32_t blockShapeY_;
    uint32_t blockShapeX_;
    uint32_t blockSize_;
    uint32_t qBlockNum_;
    uint32_t topK_;
    uint32_t maxBlocksPerBatch_;
    uint32_t totalTaskNum_;
    float scaleValue_;
    uint32_t maxQSeqlen_;
    uint32_t groupSize_;
    // PAGED_BBND page base strides (elements); may exceed blockSize*Nkv*D when dim0 is strided.
    uint64_t kStride0_;
    uint64_t vStride0_;
    // base tile info
    uint32_t qBaseTile_;
    uint32_t kvBaseTile_;
    // workspace partition sizes
    uint64_t mm1OutSize_;
    uint64_t smOnlineOutSize_;
    uint64_t mm2OutSize_;
    uint64_t updateSize_;
};

}  // namespace GsaKernelArch22

#endif  // GENERIC_BLOCK_SPARSE_ATTENTION_KERNEL_ARCH22_H
