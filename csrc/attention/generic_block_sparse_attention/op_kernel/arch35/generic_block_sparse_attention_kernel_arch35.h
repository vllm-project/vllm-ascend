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
#include "../generic_block_sparse_attention_metadata_kernel.h"
#include "../generic_block_sparse_attention_fd_utils.h"
#include "generic_block_sparse_attention_fd_combine_arch35.h"

using namespace NpuArch;
using namespace tla;

namespace GsaKernelArch35 {

template <
    class BlockMmadQK,
    class EpilogueOnlineSoftmax,
    class BlockMmadPV,
    class EpilogueRescaleO,
    Format qFormat,
    Format kvFormat>
class GsaRegularKernelArch35 {
public:
    using ArchTag = typename BlockMmadPV::ArchTag;

    using ElementQ = typename BlockMmadQK::ElementA;
    using ElementK = typename BlockMmadQK::ElementB;
    using ElementS = typename EpilogueOnlineSoftmax::ElementInput;
    using ElementP = typename BlockMmadPV::ElementA;
    using ElementV = typename BlockMmadPV::ElementB;
    using ElementOTmp = typename BlockMmadPV::ElementC;
    using ElementO = typename BlockMmadQK::ElementA;
    using ElementLse = typename EpilogueRescaleO::ElementLse;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using LayoutLse = layout::RowMajor;

    using LayoutTagL1P = typename BlockMmadPV::LayoutTagL1A;

    static constexpr uint32_t PRE_LAUNCH = 2;
    static constexpr uint32_t MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1;
    static constexpr uint32_t UB_S_OTMP_BUF_STAGES = 2;

    __aicore__ inline
    GsaRegularKernelArch35() {}

    __aicore__ inline
    void operator()(GsaKernelParamsArch35 const &params)
    {
        __gm__ GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData *tilingData =
            reinterpret_cast<__gm__ GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData *>(params.tiling);
        FetchBaseShapeInfo(tilingData, params.metaData);
        CalcOnChipBufTileInfo(tilingData);
        __gm__ const GsaMetadata::Metadata *meta =
            reinterpret_cast<__gm__ const GsaMetadata::Metadata *>(params.metaData);
        uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = AscendC::GetBlockNum();
        if (!GsaFd::ValidateMetadata(meta, tilingData, coreNum)) {
            return;
        }
        const bool fdEnabled = tilingData->fdStaticEnabled != 0U &&
            (static_cast<uint32_t>(meta->fdScheduleFlags) & GsaMetadata::FD_SCHEDULE_ENABLED) != 0U;

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK *)params.v);
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
        AscendC::GlobalTensor<ElementLse> gLse;
        gLse.SetGlobalBuffer((__gm__ ElementLse *)params.softmaxLse);
        AscendC::GlobalTensor<int32_t> gIdentityIdx;
        gIdentityIdx.SetGlobalBuffer((__gm__ int32_t *)params.workSpace);
        AscendC::GlobalTensor<float> gPartialLse;
        gPartialLse.SetGlobalBuffer((__gm__ float *)(params.workSpace + tilingData->fdPartialLseOffset));
        AscendC::GlobalTensor<float> gPartialO;
        gPartialO.SetGlobalBuffer((__gm__ float *)(params.workSpace + tilingData->fdPartialOOffset));

        AscendC::LocalTensor<ElementP> l1PTensor[MAX_CROSS_CORE_BUF_STAGES];
        AscendC::LocalTensor<ElementS> ubSTensor[UB_S_OTMP_BUF_STAGES];
        AscendC::LocalTensor<ElementOTmp> ubOTmpTensor[UB_S_OTMP_BUF_STAGES];
        InitCrossCoreDstBuf(l1PTensor, ubSTensor, ubOTmpTensor);

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
        int64_t strideKVRow = kvHeads_ * embed_;
        uint32_t embedRound = RoundUp(embed_, 16);

#ifdef __DAV_VEC__
        // Metadata schedules packed actual Q tokens. Explicitly initialize the
        // storage-only tail of every request so graph padding can never expose
        // allocator residue through the attention output (or public LSE).
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
                        auto gmOLayout = tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, embed_);
                        auto gmOTensor = tla::MakeTensor(gO[gmOffsetO], gmOLayout, Arch::PositionGM{});
                        auto gmLseLayout = tla::MakeLayout<ElementLse, LayoutLse>(qBaseTile_, 1);
                        auto gmLseTensor = tla::MakeTensor(gLse[gmOffsetLse], gmLseLayout, Arch::PositionGM{});
                        epilogueRescaleO.WriteEmptyOutput(
                            gmOTensor, gmLseTensor, GemmCoord{groupSize, embed_, 0});
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
            uint32_t qToken = taskIdx / kvHeads_;
            uint32_t kvHeadIdx = taskIdx % kvHeads_;
            uint32_t qHeadStart = kvHeadIdx * groupSize;
            uint32_t batchIdx = 0;
            uint32_t qTokenInBatch = qToken;
            // Task space = packed actual Q tokens (seqused if present, else cu storage).
            // GM / sparse index use cu storage offsets (pad at end of each batch segment).
            uint32_t accum = 0;
            for (uint32_t b = 0; b < batch_; ++b) {
                uint32_t storageLen = static_cast<uint32_t>(
                    gCuSeqLengths.GetValue(static_cast<int64_t>(b + 1)) -
                    gCuSeqLengths.GetValue(static_cast<int64_t>(b)));
                uint32_t batchLen = hasSequsedQ ?
                    static_cast<uint32_t>(gSequsedQ.GetValue(static_cast<int64_t>(b))) : storageLen;
                if (qToken < accum + batchLen) {
                    batchIdx = b;
                    qTokenInBatch = qToken - accum;
                    break;
                }
                accum += batchLen;
            }

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
            int64_t qStorageToken = gCuSeqLengths.GetValue(static_cast<int64_t>(batchIdx)) +
                                    static_cast<int64_t>(qTokenInBatch);
            int64_t gmOffsetQ = qStorageToken * strideQO +
                                static_cast<int64_t>(qHeadStart) * embed_;
            int64_t gmOffsetO = gmOffsetQ;
            // LSE [T, N, 1]: packed GQA writes groupSize contiguous heads for one token.
            int64_t gmOffsetLse = qStorageToken * qHeads_ + qHeadStart;

#ifdef __DAV_VEC__
            auto gmOLayoutTla = tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, embed_);
            auto gmOTensorTla = tla::MakeTensor(gO[gmOffsetO], gmOLayoutTla, Arch::PositionGM{});
            auto gmLseLayoutTla = tla::MakeLayout<ElementLse, LayoutLse>(qBaseTile_, 1);
            auto gmLseTensorTla = tla::MakeTensor(gLse[gmOffsetLse], gmLseLayoutTla, Arch::PositionGM{});
#endif
            if (qSeqlen == 0U || kvSeqlen == 0U) {
#ifdef __DAV_VEC__
                epilogueRescaleO.WriteEmptyOutput(
                    gmOTensorTla, gmLseTensorTla, GemmCoord{groupSize, embed_, 0});
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

            uint32_t rowNum = groupSize;
            uint32_t rowNumRound = RoundUp(rowNum, 16);

            if (kvSLoopNum == 0U) {
#ifdef __DAV_VEC__
                if (isFdPartial) {
                    epilogueRescaleO.WriteNeutralPartial(gPartialO, gPartialLse, fdPartialTaskId,
                        groupSize, embed_, tilingData->fdLseSubStride);
                } else {
                    epilogueRescaleO.WriteEmptyOutput(
                        gmOTensorTla, gmLseTensorTla, GemmCoord{groupSize, embed_, 0});
                }
#endif
                continue;
            }

#ifdef __DAV_CUBE__
            auto gmQLayoutTla = tla::MakeLayout<ElementQ, LayoutQ>(qBaseTile_, embed_);
            auto gmQTensorTla = tla::MakeTensor(gQ[gmOffsetQ], gmQLayoutTla, Arch::PositionGM{});
            GemmCoord actualBlockShapeQ{rowNum, embed_, 0};
            blockMmadQK.loadQGM(gmQTensorTla, actualBlockShapeQ);
#endif
            for (uint32_t kvBlockIdx = 0; kvBlockIdx < kvSLoopNum + PRE_LAUNCH; kvBlockIdx++) {
                if (kvBlockIdx < kvSLoopNum) {
                    uint32_t kvSTileSizeAct = validTileSize[kvBlockIdx];
                    int32_t physicalBlockId = validPhysicalIds[kvBlockIdx];

                    int64_t gmOffsetK = static_cast<int64_t>(physicalBlockId) * static_cast<int64_t>(kStride0_) +
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

                    int64_t gmOffsetV = static_cast<int64_t>(physicalBlockIdV) * static_cast<int64_t>(vStride0_) +
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
                    if (isFdPartial) {
                        epilogueRescaleO.ProcessPartial(
                            gmOTensorTla, gmLseTensorTla, actualBlockShapePV,
                            curTileMod, kvBlockIdxDe, (kvBlockIdxDe == 0),
                            (kvBlockIdxDe == kvSLoopNum - 1), mm2ToReFlag,
                            gPartialO, gPartialLse, fdPartialTaskId, tilingData->fdLseSubStride);
                    } else {
                        epilogueRescaleO(
                            gmOTensorTla, gmLseTensorTla, actualBlockShapePV,
                            curTileMod, kvBlockIdxDe, (kvBlockIdxDe == 0),
                            (kvBlockIdxDe == kvSLoopNum - 1), mm2ToReFlag);
                    }
#endif
                }
            }
        }
        ReleaseSyncFlags<4, 4, 4>();
        if (fdEnabled) {
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<false>();
#ifdef __DAV_VEC__
            GenericBlockSparseAttentionFdCombineArch35<ElementO, Arch::Resource<ArchTag>> combine(resource);
            combine(meta, tilingData, gPartialLse, gPartialO, gO, gCuSeqLengths, gSequsedQ, hasSequsedQ);
#endif
        }
    }

private:
    __aicore__ inline
    void FetchBaseShapeInfo(__gm__ GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData *tilingData,
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
        kStride0_ = tilingData->kStride0;
        vStride0_ = tilingData->vStride0;
    }

    __aicore__ inline
    void CalcOnChipBufTileInfo(__gm__ GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData *tilingData)
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

} // namespace GsaKernelArch35
