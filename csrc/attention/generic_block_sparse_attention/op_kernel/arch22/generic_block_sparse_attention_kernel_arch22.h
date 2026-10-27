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
#include "../gbsa_sparse_kv_geometry.hpp"
#include "kernel_utils.hpp"
#include "generic_block_sparse_attention_fd_combine_arch22.h"

using namespace NpuArch;

namespace GbsaKernelArch22 {

template <class BlockMmadQK, class EpilogueOnlineSoftmax, class BlockMmadPV, class EpilogueRescaleO,
          bool isPaged = true, bool hasResidualBlock = false>
class GbsaRegularKernelArch22 {
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

    __aicore__ inline GbsaRegularKernelArch22() {}

    __aicore__ inline void operator()(GbsaKernelParamsArch22 const &params)
    {
        if (params.tiling == nullptr || params.metaData == nullptr || params.cuSeqLengths == nullptr ||
            (isPaged && (params.blockTable == nullptr || params.sequsedKv == nullptr)) ||
            (!isPaged && (params.blockTable != nullptr || params.cuSeqLengthsKv == nullptr))) {
            return;
        }
        __gm__ GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData *tilingData =
            reinterpret_cast<__gm__ GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData *>(params.tiling);
        FetchTilingData(tilingData, params.metaData);
        __gm__ const GbsaMetadata::Metadata *meta =
            reinterpret_cast<__gm__ const GbsaMetadata::Metadata *>(params.metaData);
        const uint32_t physicalAicNum = AscendC::GetBlockNum();
        if (!GsaFd::ValidateMetadata(meta, tilingData, physicalAicNum)) {
            return;
        }
        const bool fdEnabled = tilingData->fdStaticEnabled != 0U &&
                               (static_cast<uint32_t>(meta->fdScheduleFlags) & GbsaMetadata::FD_SCHEDULE_ENABLED) != 0U;

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        AscendC::GlobalTensor<ElementV> gV;
        gV.SetGlobalBuffer((__gm__ ElementV *)params.v);
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx;
        gSparseBlockIdx.SetGlobalBuffer((__gm__ int32_t *)params.sparseBlockIdx);
        AscendC::GlobalTensor<int32_t> gBlockTable;
        if constexpr (isPaged) {
            gBlockTable.SetGlobalBuffer((__gm__ int32_t *)params.blockTable);
        }
        AscendC::GlobalTensor<int64_t> gCuSeqLengthsKv;
        if constexpr (!isPaged) {
            gCuSeqLengthsKv.SetGlobalBuffer((__gm__ int64_t *)params.cuSeqLengthsKv);
        }
        AscendC::GlobalTensor<int32_t> gSparseBlockCount;
        gSparseBlockCount.SetGlobalBuffer((__gm__ int32_t *)params.sparseBlockCount);
        AscendC::GlobalTensor<int64_t> gCuSeqLengths;
        if (params.cuSeqLengths != nullptr) {
            gCuSeqLengths.SetGlobalBuffer((__gm__ int64_t *)params.cuSeqLengths);
        }
        AscendC::GlobalTensor<int32_t> gSequsedQ;
        const bool hasSequsedQ = (params.sequsedQ != nullptr);
        if (hasSequsedQ) {
            gSequsedQ.SetGlobalBuffer((__gm__ int32_t *)params.sequsedQ);
        }
        AscendC::GlobalTensor<int32_t> gSequsedKv;
        if (params.sequsedKv != nullptr) {
            gSequsedKv.SetGlobalBuffer((__gm__ int32_t *)params.sequsedKv);
        }
        // Validate dense storage prefixes on all cores before the FD synchronization.
        if constexpr (!isPaged) {
            if (gCuSeqLengthsKv.GetValue(0) != 0 ||
                gCuSeqLengthsKv.GetValue(batch_) != static_cast<int64_t>(tilingData->totalKvTokens)) {
                return;
            }
            for (uint32_t b = 0; b < batch_; ++b) {
                const int64_t start = gCuSeqLengthsKv.GetValue(b);
                const int64_t end = gCuSeqLengthsKv.GetValue(b + 1U);
                if (start < 0 || end < start || end > static_cast<int64_t>(tilingData->totalKvTokens) ||
                    end - start > 0x7fffffffLL) {
                    return;
                }
                if (params.sequsedKv != nullptr) {
                    const int32_t used = gSequsedKv.GetValue(b);
                    if (used < 0 || used > end - start) {
                        return;
                    }
                }
            }
        }
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
        AscendC::GlobalTensor<float> gLse;
        gLse.SetGlobalBuffer((__gm__ float *)params.softmaxLse);
        AscendC::GlobalTensor<float> gPartialLse;
        gPartialLse.SetGlobalBuffer((__gm__ float *)(params.workSpace + tilingData->fdPartialLseOffset));
        AscendC::GlobalTensor<float> gPartialO;
        gPartialO.SetGlobalBuffer((__gm__ float *)(params.workSpace + tilingData->fdPartialOOffset));

        // Workspace: [S][P][OTmp][OUpdate].
        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS *)params.workSpace);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP *)(params.workSpace + mm1OutSize_));
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace + mm1OutSize_ + smOnlineOutSize_));
        AscendC::GlobalTensor<ElementOTmp> gOUpdate;
        gOUpdate.SetGlobalBuffer(
            (__gm__ ElementOTmp *)(params.workSpace + mm1OutSize_ + smOnlineOutSize_ + mm2OutSize_));

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

        uint32_t embedRoundQk = RoundUp(embed_, 16);
        uint32_t l1QkSize = BlockMmadQK::L1TileShape::M * embedRoundQk * sizeof(ElementQ) +
                            BlockMmadQK::L1TileShape::N * embedRoundQk * sizeof(ElementK) * 2;
        BlockMmadQK blockMmadQK(resource, 0, embedRoundQk);
        BlockMmadPV blockMmadPV(resource, l1QkSize);
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
#endif

        uint32_t groupSize = groupSize_;
        int64_t strideQO = qHeads_ * embed_;
        int64_t strideKVRow = kvHeads_ * embed_;
        uint32_t embedRound = RoundUp(embed_, 16);
        uint32_t rowNumRound = RoundUp(groupSize, 16);

#ifdef __DAV_C220_VEC__
        if (hasSequsedQ) {
            uint64_t paddingTask = 0U;
            for (uint32_t batchIdx = 0U; batchIdx < batch_; ++batchIdx) {
                const int64_t storageStart = gCuSeqLengths.GetValue(batchIdx);
                const int64_t storageEnd = gCuSeqLengths.GetValue(batchIdx + 1U);
                const int64_t actualLenValue = static_cast<int64_t>(gSequsedQ.GetValue(batchIdx));
                if (storageStart < 0 || storageEnd < storageStart || actualLenValue < 0) {
                    continue;
                }
                const int64_t storageLen = storageEnd - storageStart;
                const int64_t actualLen = actualLenValue < storageLen ? actualLenValue : storageLen;
                for (int64_t token = actualLen; token < storageLen; ++token) {
                    for (uint32_t kvHeadIdx = 0U; kvHeadIdx < kvHeads_; ++kvHeadIdx, ++paddingTask) {
                        if (paddingTask % coreNum != coreIdx) {
                            continue;
                        }
                        const uint64_t qStorageToken = static_cast<uint64_t>(storageStart + token);
                        const uint32_t qHeadStart = kvHeadIdx * groupSize;
                        const uint64_t gmOffsetO = (qStorageToken * qHeads_ + qHeadStart) * embed_;
                        const uint64_t gmOffsetLse = qStorageToken * qHeads_ + qHeadStart;
                        epilogueRescaleO.WriteEmptyOutput(gO[gmOffsetO], gLse[gmOffsetLse], groupSize, embed_);
                    }
                }
            }
        }
        if (fdEnabled) {
            // FD partitions virtual KV blocks and can omit tasks with no blocks.
            // Such tasks still need an explicit empty output.
            for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum_; taskIdx += coreNum) {
                bool hasDecodeTask = false;
                for (uint32_t scheduleCoreIdx = 0U;
                     scheduleCoreIdx < static_cast<uint32_t>(meta->fdActiveCoreNum); ++scheduleCoreIdx) {
                    const __gm__ GbsaMetadata::DecodeSchedule &schedule = meta->decodeSchedules[scheduleCoreIdx];
                    if (taskIdx >= static_cast<uint32_t>(schedule.baseTaskStart) &&
                        taskIdx < static_cast<uint32_t>(schedule.baseTaskEnd)) {
                        hasDecodeTask = true;
                        break;
                    }
                }
                if (hasDecodeTask) {
                    continue;
                }
                uint32_t qStorageToken = 0U;
                uint32_t qTokenInBatch = 0U;
                uint32_t batchIdx = 0U;
                uint32_t kvHeadIdx = 0U;
                if (!GsaFd::DecodeTaskStorage(taskIdx, kvHeads_, batch_, gCuSeqLengths, gSequsedQ, hasSequsedQ,
                                              qStorageToken, qTokenInBatch, batchIdx, kvHeadIdx)) {
                    continue;
                }
                const uint64_t gmOffsetO =
                    (static_cast<uint64_t>(qStorageToken) * qHeads_ + kvHeadIdx * groupSize) * embed_;
                const uint64_t gmOffsetLse = static_cast<uint64_t>(qStorageToken) * qHeads_ + kvHeadIdx * groupSize;
                epilogueRescaleO.WriteEmptyOutput(gO[gmOffsetO], gLse[gmOffsetLse], groupSize, embed_);
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
                const __gm__ GbsaMetadata::DecodeSchedule &schedule = meta->decodeSchedules[coreIdx];
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
            const bool isFdPartial =
                fdEnabled && GsaFd::FindPartialTask(meta, taskIdx, coreIdx, fdPartialTaskId, fdPartialCount);
            if (fdEnabled) {
                rawBegin = taskIdx == taskLoopStart ? scheduleFirstBlock : 0U;
            }

            uint32_t qStorageToken = 0U;
            uint32_t qTokenInBatch = 0U;
            uint32_t batchIdx = 0U;
            uint32_t kvHeadIdx = 0U;
            if (!GsaFd::DecodeTaskStorage(taskIdx, kvHeads_, batch_, gCuSeqLengths, gSequsedQ, hasSequsedQ,
                                          qStorageToken, qTokenInBatch, batchIdx, kvHeadIdx)) {
                continue;
            }
            uint32_t qHeadStart = kvHeadIdx * groupSize;

            uint32_t qStorageLen = static_cast<uint32_t>(gCuSeqLengths.GetValue(static_cast<int64_t>(batchIdx + 1)) -
                                                         gCuSeqLengths.GetValue(static_cast<int64_t>(batchIdx)));
            int64_t kvStorageStart = 0;
            uint32_t kvStorageLen = 0;
            uint32_t kvSeqlen = 0;
            if constexpr (isPaged) {
                kvSeqlen = static_cast<uint32_t>(gSequsedKv.GetValue(static_cast<int64_t>(batchIdx)));
            } else {
                kvStorageStart = gCuSeqLengthsKv.GetValue(static_cast<int64_t>(batchIdx));
                kvStorageLen = static_cast<uint32_t>(gCuSeqLengthsKv.GetValue(static_cast<int64_t>(batchIdx + 1U)) -
                                                     kvStorageStart);
                kvSeqlen = params.sequsedKv == nullptr ?
                               kvStorageLen :
                               static_cast<uint32_t>(gSequsedKv.GetValue(static_cast<int64_t>(batchIdx)));
            }
            uint32_t qSeqlen =
                hasSequsedQ ? static_cast<uint32_t>(gSequsedQ.GetValue(static_cast<int64_t>(batchIdx))) : qStorageLen;
            int64_t gmOffsetQ =
                static_cast<int64_t>(qStorageToken) * strideQO + static_cast<int64_t>(qHeadStart) * embed_;
            int64_t gmOffsetO = gmOffsetQ;
            const int64_t gmOffsetLse = static_cast<int64_t>(qStorageToken) * qHeads_ + qHeadStart;

            // Causal decode requires kvSeqlen >= qSeqlen (historyLen would underflow); no-mask allows any lengths.
            const bool isCausalMask = (tilingData->maskType == 1U);
            if (qSeqlen == 0U || kvSeqlen == 0U || (isCausalMask && kvSeqlen < qSeqlen)) {
#ifdef __DAV_C220_VEC__
                if (isFdPartial) {
                    epilogueRescaleO.WriteNeutralPartial(gPartialO, gPartialLse, fdPartialTaskId, groupSize, embed_,
                                                         tilingData->fdLseSubStride);
                } else {
                    epilogueRescaleO.WriteEmptyOutput(gO[gmOffsetO], gLse[gmOffsetLse], groupSize, embed_);
                }
#endif
                continue;
            }

            // layout_sparse_pattern=KVN_TotalQB_KB: sparseBlockIdx 3D [N_kv, totalQBlocks, topK]
            // totalQBlocks spans storage (cu) blocks; align with metadata qStorageBlockStarts.
            uint32_t globalQBlock = 0;
            for (uint32_t b = 0; b < batchIdx; ++b) {
                uint32_t qLen = static_cast<uint32_t>(gCuSeqLengths.GetValue(static_cast<int64_t>(b + 1)) -
                                                      gCuSeqLengths.GetValue(static_cast<int64_t>(b)));
                globalQBlock += (qLen + blockShapeX_ - 1) / blockShapeX_;
            }
            globalQBlock += qTokenInBatch / blockShapeX_;
            int64_t sparseIdxBase =
                static_cast<int64_t>(kvHeadIdx) * qBlockNum_ * topK_ + static_cast<int64_t>(globalQBlock) * topK_;
            uint32_t explicitBlockCount = topK_;
            if (params.sparseBlockCount != nullptr) {
                // sparseBlockCount 2D: [N_kv, totalQBlocks]
                int64_t countOffset = static_cast<int64_t>(kvHeadIdx) * qBlockNum_ + static_cast<int64_t>(globalQBlock);
                const int32_t count = gSparseBlockCount.GetValue(countOffset);
                explicitBlockCount = count > 0 ? static_cast<uint32_t>(count) : 0U;
                explicitBlockCount = explicitBlockCount < topK_ ? explicitBlockCount : topK_;
            }
            uint32_t visibleKvEnd = kvSeqlen;
            if (isCausalMask) {
                visibleKvEnd = kvSeqlen - qSeqlen + qTokenInBatch + 1U;
            }
            uint32_t virtualBlockCount = explicitBlockCount;
            uint32_t residualBlockId = 0U;
            if constexpr (hasResidualBlock) {
                const uint32_t residualSize = visibleKvEnd % blockShapeY_;
                residualBlockId = visibleKvEnd / blockShapeY_;
                virtualBlockCount += static_cast<uint32_t>(residualSize != 0U);
            }
            rawEnd = virtualBlockCount;
            if (fdEnabled) {
                rawBegin = rawBegin < virtualBlockCount ? rawBegin : virtualBlockCount;
                if (taskIdx + 1U == taskLoopEnd) {
                    rawEnd = scheduleLastBlock < virtualBlockCount ? scheduleLastBlock : virtualBlockCount;
                }
            }

            // Gather the selected logical blocks into a compressed token band.
            uint32_t validKvTokens = 0;
            bool sparseBandHasClipped = false;
            uint32_t sparseBandFullPrefixSlots = 0U;
            uint32_t selectNum = rawEnd > rawBegin ? rawEnd - rawBegin : 0U;
            auto gSparseBlockIdxBase = gSparseBlockIdx[sparseIdxBase];
            GbsaSparseGeometry::CalcSparseKvSize<hasResidualBlock>(
                gSparseBlockIdxBase, explicitBlockCount, rawBegin, rawEnd, blockShapeY_, visibleKvEnd, validKvTokens,
                sparseBandHasClipped, sparseBandFullPrefixSlots);
            const uint32_t explicitBegin = rawBegin < explicitBlockCount ? rawBegin : explicitBlockCount;
            const uint32_t explicitEnd = rawEnd < explicitBlockCount ? rawEnd : explicitBlockCount;
            const uint32_t localExplicitBlockCount = explicitEnd - explicitBegin;
            // A residual-only FD slice must resolve the implicit id before reading sparse index GM.
            auto gTaskSparseBlockIdx = gSparseBlockIdxBase[explicitBegin];
            uint32_t kvSLoopNum = (validKvTokens + kvBaseTile_ - 1U) / kvBaseTile_;
            uint32_t yBlockNum = (visibleKvEnd + blockShapeY_ - 1U) / blockShapeY_;
            int64_t gmOffsetKvBase = kvStorageStart * strideKVRow + static_cast<int64_t>(kvHeadIdx) * embed_;
            if (kvSLoopNum == 0U) {
#ifdef __DAV_C220_VEC__
                if (isFdPartial) {
                    epilogueRescaleO.WriteNeutralPartial(gPartialO, gPartialLse, fdPartialTaskId, groupSize, embed_,
                                                         tilingData->fdLseSubStride);
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
            const uint32_t kvTileWidth = kvBaseTile_;
            AscendC::GlobalTensor<int32_t> gBatchBlockTable;
            if constexpr (isPaged) {
                gBatchBlockTable = gBlockTable[blockTOffset];
            }

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
                    const uint32_t tileStart = kvBlockIdx * kvBaseTile_;
                    uint32_t kvSTileSizeAct = min(kvBaseTile_, validKvTokens - tileStart);
                    int64_t gmOffsetK = gmOffsetKvBase;

                    uint32_t stageId = kvBlockIdx % MAX_CROSS_CORE_BUF_STAGES;
                    uint64_t gmOffsetS =
                        static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * MAX_CROSS_CORE_BUF_STAGES +
                        static_cast<uint64_t>(stageId) * WORKSPACE_BLOCK_SIZE_DB;

#ifdef __DAV_C220_CUBE__
                    // Stage 1: QK Matmul
                    LayoutK gmKLayout(strideKVRow, kvTileWidth);
                    LayoutS ubSLayout(rowNumRound, RoundUp(kvSTileSizeAct, 16));
                    GemmCoord actualBlockShapeQK{rowNum, kvSTileSizeAct, embed_};

                    blockMmadQK(gQ[gmOffsetQ], gK[gmOffsetK], gS[gmOffsetS], gBatchBlockTable, gTaskSparseBlockIdx,
                                gmQLayout, gmKLayout, ubSLayout, actualBlockShapeQK, kvBlockIdx, kvSLoopNum,
                                kvBaseTile_, strideKVRow, blockShapeY_, selectNum, yBlockNum, visibleKvEnd,
                                blockSize_, kStride0_, localExplicitBlockCount, residualBlockId,
                                sparseBandHasClipped, sparseBandFullPrefixSlots);

                    NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady_);
#endif

#ifdef __DAV_C220_VEC__
                    // Stage 2: Online Softmax
                    uint64_t gmOffsetP = gmOffsetS;
                    LayoutS ubSLayout(rowNumRound, RoundUp(kvSTileSizeAct, 16));
                    LayoutP ubPLayout(rowNumRound, RoundUp(kvSTileSizeAct, 16));
                    GemmCoord actualBlockShapeQK{rowNum, kvSTileSizeAct, embed_};

                    NpuArch::Arch::CrossCoreWaitFlag(qkReady_);

                    epilogueOnlineSoftmax(gP[gmOffsetP], gS[gmOffsetS], ubPLayout, ubSLayout, actualBlockShapeQK,
                                          (kvBlockIdx == 0), 0, qSBlockSize, groupSize, stageId, softmaxReady_);
#endif
                }

                // === Stage 3+4: PV Matmul & RescaleO ===
                if (kvBlockIdx >= preLaunch) {
                    uint32_t kvBlockIdxDe = kvBlockIdx - preLaunch;
                    const uint32_t tileStart = kvBlockIdxDe * kvBaseTile_;
                    uint32_t kvSTileSizeAct = min(kvBaseTile_, validKvTokens - tileStart);
                    int64_t gmOffsetV = gmOffsetKvBase;

                    uint32_t stageId = kvBlockIdxDe % MAX_CROSS_CORE_BUF_STAGES;
                    uint64_t gmOffsetP =
                        static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * MAX_CROSS_CORE_BUF_STAGES +
                        static_cast<uint64_t>(stageId) * WORKSPACE_BLOCK_SIZE_DB;
                    uint64_t gmOffsetOTmp =
                        static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB * MAX_CROSS_CORE_BUF_STAGES +
                        static_cast<uint64_t>(stageId) * WORKSPACE_BLOCK_SIZE_DB;

#ifdef __DAV_C220_CUBE__
                    // Stage 3: PV Matmul
                    LayoutP ubPLayout(rowNumRound, RoundUp(kvSTileSizeAct, 16));
                    LayoutV gmVLayout(kvTileWidth, strideKVRow);
                    LayoutOTmp ubOTmpLayout(rowNumRound, embedRound);
                    GemmCoord actualBlockShapePV{rowNum, embed_, kvSTileSizeAct};

                    blockMmadPV(gP[gmOffsetP], gV[gmOffsetV], gOTmp[gmOffsetOTmp], gBatchBlockTable,
                                gTaskSparseBlockIdx, ubPLayout, gmVLayout, ubOTmpLayout, actualBlockShapePV,
                                kvBlockIdxDe, kvSLoopNum, kvBaseTile_, visibleKvEnd, strideKVRow, 1, softmaxReady_,
                                blockShapeY_, selectNum, yBlockNum, blockSize_, vStride0_, localExplicitBlockCount,
                                residualBlockId, sparseBandHasClipped, sparseBandFullPrefixSlots);
                    NpuArch::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady_);
#endif

#ifdef __DAV_C220_VEC__
                    // Stage 4: RescaleO
                    uint64_t gmOffsetUpdate = static_cast<uint64_t>(coreIdx) * WORKSPACE_BLOCK_SIZE_DB;
                    LayoutOTmp ubOTmpLayout(rowNumRound, embedRound);
                    LayoutUpdate ubUpdateLayout(rowNumRound, embedRound);
                    GemmCoord actualBlockShapePV{rowNum, embed_, kvSTileSizeAct};

                    NpuArch::Arch::CrossCoreWaitFlag(pvReady_);

                    if (isFdPartial) {
                        epilogueRescaleO.ProcessPartial(
                            gO[gmOffsetO], gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate], gLse[gmOffsetLse], gmOLayout,
                            ubOTmpLayout, ubUpdateLayout, gmLseLayout, actualBlockShapePV, qSBlockSize, groupSize,
                            (kvBlockIdxDe == 0), (kvBlockIdxDe == kvSLoopNum - 1), stageId, gPartialO, gPartialLse,
                            fdPartialTaskId, tilingData->fdLseSubStride);
                    } else {
                        epilogueRescaleO(gO[gmOffsetO], gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate],
                                         gLse[gmOffsetLse], gmOLayout, ubOTmpLayout, ubUpdateLayout, gmLseLayout,
                                         actualBlockShapePV, qSBlockSize, groupSize, (kvBlockIdxDe == 0),
                                         (kvBlockIdxDe == kvSLoopNum - 1), stageId);
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
            using RescaleDispatchPolicy = typename EpilogueRescaleO::DispatchPolicy;
            constexpr bool OUTPUT_LSE = RescaleDispatchPolicy::LSE_MODE == Epilogue::LseMode::OUT_ONLY;
            GenericBlockSparseAttentionFdCombineArch22<ElementO, Arch::Resource<ArchTag>, OUTPUT_LSE> combine(resource);
            combine(meta, tilingData, gPartialLse, gPartialO, gO, gLse, gCuSeqLengths, gSequsedQ, hasSequsedQ);
#endif
        }
    }

private:
    __aicore__ inline void FetchTilingData(
        __gm__ GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData *tilingData, GM_ADDR metaData)
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
        __gm__ GbsaMetadata::Metadata *meta = reinterpret_cast<__gm__ GbsaMetadata::Metadata *>(metaData);
        totalTaskNum_ = static_cast<uint32_t>(meta->saTotalTaskNum);
        scaleValue_ = tilingData->scaleValue;
        groupSize_ = tilingData->groupSize;
        qBaseTile_ = tilingData->baseTileInfo.qBaseTile;
        kvBaseTile_ = tilingData->baseTileInfo.kvBaseTile;
        mm1OutSize_ = tilingData->mm1OutSize;
        smOnlineOutSize_ = tilingData->smOnlineOutSize;
        mm2OutSize_ = tilingData->mm2OutSize;
        updateSize_ = tilingData->updateSize;
        kStride0_ = tilingData->kStride0;
        vStride0_ = tilingData->vStride0;
        residualBlockMode_ = tilingData->residualBlockMode;
        isConsistentTopk_ = tilingData->isConsistentTopk;
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
    uint32_t groupSize_;
    // PA_BBND page base strides (elements); may exceed blockSize*Nkv*D when dim0 is strided.
    uint64_t kStride0_;
    uint64_t vStride0_;
    uint32_t residualBlockMode_;
    uint32_t isConsistentTopk_;
    // base tile info
    uint32_t qBaseTile_;
    uint32_t kvBaseTile_;
    // workspace partition sizes
    uint64_t mm1OutSize_;
    uint64_t smOnlineOutSize_;
    uint64_t mm2OutSize_;
    uint64_t updateSize_;
};

} // namespace GbsaKernelArch22

#endif // GENERIC_BLOCK_SPARSE_ATTENTION_KERNEL_ARCH22_H
