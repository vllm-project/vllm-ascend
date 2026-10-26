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

namespace GbsaKernelArch35 {

template <class EpiloguePrepareKvOffsets, class BlockMmadQK, class EpilogueOnlineSoftmax, class BlockMmadPV,
          class EpilogueRescaleO, Format qFormat, Format kvFormat, bool isPaged, bool hasResidualBlock>
class GbsaRegularKernelArch35 {
    // TODO 此处断言应放置于tiling中
    static_assert(qFormat == Format::TND, "Only TND Q is enabled currently");
    static_assert(!isPaged || kvFormat == Format::BSND, "Only PA_BBND is enabled currently");
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

    __aicore__ inline GbsaRegularKernelArch35() {}

    __aicore__ inline void operator()(GbsaKernelParamsArch35 const &params)
    {
        if (params.tiling == nullptr || params.metaData == nullptr || params.cuSeqLengths == nullptr ||
            ((params.blockTable != nullptr) != isPaged) ||
            (isPaged && params.sequsedKv == nullptr) ||
            (!isPaged && params.cuSeqLengthsKv == nullptr)) {
            return;
        }
        __gm__ GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData *tilingData =
            reinterpret_cast<__gm__ GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData *>(params.tiling);
        FetchBaseShapeInfo(tilingData, params.metaData);
        CalcOnChipBufTileInfo(tilingData);
        CalcUBufTileInfo();
        __gm__ const GbsaMetadata::Metadata *meta =
            reinterpret_cast<__gm__ const GbsaMetadata::Metadata *>(params.metaData);
        uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t coreNum = AscendC::GetBlockNum();
        if (!GsaFd::ValidateMetadata(meta, tilingData, coreNum)) {
            return;
        }
        const bool fdEnabled = tilingData->fdStaticEnabled != 0U &&
                               (static_cast<uint32_t>(meta->fdScheduleFlags) & GbsaMetadata::FD_SCHEDULE_ENABLED) != 0U;

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK *)params.v);
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
        // Validate storage prefixes on every core before entering cross-core synchronization.
        // Metadata owns task scheduling; it cannot check the actual Q/K/V tensor capacities.
        // TODO : 此处对seqlen进行校验，但是无法拦截，直接选择了返回，应删除
        uint64_t actualQTokens = 0;
        if (!ValidateSeqLengths<qFormat>(gCuSeqLengths, gSequsedQ, hasSequsedQ, batch_,
                                  tilingData->totalQTokens, actualQTokens) ||
            actualQTokens * kvHeads_ != totalTaskNum_) {
            return;
        }
        const bool hasSequsedKv = params.sequsedKv != nullptr;
        if constexpr (!isPaged) {
            uint64_t actualKvTokens = 0;
            if (!ValidateSeqLengths<kvFormat>(gCuSeqLengthsKv, gSequsedKv, hasSequsedKv, batch_,
                                      tilingData->totalKvTokens, actualKvTokens)) {
                return;
            }
        } else {
            for (uint32_t b = 0; b < batch_; ++b) {
                const int32_t len = gSequsedKv.GetValue(b);
                if (len < 0 || static_cast<uint64_t>(len) >
                    static_cast<uint64_t>(maxBlocksPerBatch_) * blockSize_) {
                    return;
                }
            }
        }
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
        AscendC::GlobalTensor<ElementLse> gLse;
        gLse.SetGlobalBuffer((__gm__ ElementLse *)params.softmaxLse);
        AscendC::GlobalTensor<float> gPartialLse;
        gPartialLse.SetGlobalBuffer((__gm__ float *)(params.workSpace + tilingData->fdPartialLseOffset));
        AscendC::GlobalTensor<float> gPartialO;
        gPartialO.SetGlobalBuffer((__gm__ float *)(params.workSpace + tilingData->fdPartialOOffset));
        AscendC::GlobalTensor<ElementOTmp> gLo;
        // Both AIVs share their physical AIC's two full LO tiles.
        uint32_t loCoreIdx = AscendC::GetBlockIdx();
#ifdef __DAV_VEC__
        loCoreIdx /= AscendC::GetSubBlockNum();
#endif
        const uint64_t loTileElems = static_cast<uint64_t>(qBaseTile_) * RoundUp(embed_, 16U);
        gLo.SetGlobalBuffer(reinterpret_cast<__gm__ ElementOTmp *>(params.workSpace) +
                           loCoreIdx * UB_S_OTMP_BUF_STAGES * loTileElems);

        AscendC::GlobalTensor<uint64_t> gKvOffsetInfo;
        const uint32_t kvOffsetTileWords = tilingData->kvOffsetTileWords;
        gKvOffsetInfo.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(
            params.workSpace + tilingData->kvOffsetInfoOffset) +
            static_cast<uint64_t>(loCoreIdx) * KV_OFFSET_STAGES * kvOffsetTileWords);

        AscendC::LocalTensor<ElementP> l1PTensor[MAX_CROSS_CORE_BUF_STAGES];
        AscendC::LocalTensor<ElementS> ubSTensor[UB_S_OTMP_BUF_STAGES];
        AscendC::LocalTensor<ElementOTmp> ubOTmpTensor[UB_S_OTMP_BUF_STAGES];
        InitCrossCoreDstBuf(l1PTensor, ubSTensor, ubOTmpTensor);

        const uint32_t kvOffsetGroupWords = CalcKvOffsetGroupWords<isPaged>(mm1L1TileN_, blockShapeY_);
        const uint64_t strideKVRow = GbsaLayoutHelper<kvFormat>::TokenStride(kvHeads_, embed_);

        InitSyncFlags<4, 4, 4>();

#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
#endif
        AscendC::SyncAll<false>();
#ifdef __DAV_CUBE__
        BlockMmadQK blockMmadQK(resource, mm1L1TileHelper_);
        BlockMmadPV blockMmadPV(resource, mm2L1AddrStart_, mm2L1TileHelper_);
#endif
#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        EpiloguePrepareKvOffsets epiloguePrepareKvOffsets(resource, uBufTileHelper_, kvBaseTile_, mm1L1TileN_,
                                                         blockShapeY_, blockSize_, strideKVRow, kStride0_, vStride0_);
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue_, uBufTileHelper_);
        EpilogueRescaleO epilogueRescaleO(resource, uBufTileHelper_);
        epilogueRescaleO.SetLoWorkspace(gLo, loTileElems);
#endif

        uint32_t groupSize = groupSize_;
        uint64_t strideQHead = GbsaLayoutHelper<qFormat>::HeadStride(0, embed_);
        uint32_t embedRound = RoundUp(embed_, 16);

#ifdef __DAV_VEC__
        if (hasSequsedQ) {
            uint32_t paddingTask = 0U;
            for (uint32_t batchIdx = 0U; batchIdx < batch_; ++batchIdx) {
                const uint32_t storageStart = static_cast<uint32_t>(gCuSeqLengths.GetValue(batchIdx));
                const uint32_t storageEnd = static_cast<uint32_t>(gCuSeqLengths.GetValue(batchIdx + 1U));
                const uint32_t storageLen = storageEnd - storageStart;
                uint32_t actualLen = static_cast<uint32_t>(gSequsedQ.GetValue(batchIdx));
                actualLen = actualLen < storageLen ? actualLen : storageLen;
                const BatchSeqInfo qSeq{storageStart, storageLen, actualLen};
                for (uint32_t token = actualLen; token < storageLen; ++token) {
                    for (uint32_t kvHeadIdx = 0U; kvHeadIdx < kvHeads_; ++kvHeadIdx, ++paddingTask) {
                        if (paddingTask % coreNum != coreIdx) {
                            continue;
                        }
                        const uint32_t qHeadStart = kvHeadIdx * groupSize;
                        const uint64_t gmOffsetO = GetTensorOffset<qFormat>(qSeq, token, qHeadStart, qHeads_, embed_);
                        const uint64_t gmOffsetLse = GetLseOffset<qFormat>(qSeq, token, qHeadStart, qHeads_);
                        auto gmOLayout = tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, strideQHead);
                        auto gmOTensor = tla::MakeTensor(gO[gmOffsetO], gmOLayout, Arch::PositionGM{});
                        auto gmLseLayout = tla::MakeLayout<ElementLse, LayoutLse>(qBaseTile_, 1);
                        auto gmLseTensor = tla::MakeTensor(gLse[gmOffsetLse], gmLseLayout, Arch::PositionGM{});
                        epilogueRescaleO.WriteEmptyOutput(gmOTensor, gmLseTensor, GemmCoord{groupSize, embed_, 0});
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
                const __gm__ GbsaMetadata::DecodeSchedule &schedule = meta->decodeSchedules[coreIdx];
                taskLoopStart = static_cast<uint32_t>(schedule.baseTaskStart);
                taskLoopEnd = static_cast<uint32_t>(schedule.baseTaskEnd);
                scheduleFirstBlock = static_cast<uint32_t>(schedule.firstBlockStart);
                scheduleLastBlock = static_cast<uint32_t>(schedule.lastBlockEnd);
            }
        }

        for (uint32_t taskIdx = taskLoopStart; taskIdx < taskLoopEnd; taskIdx += taskLoopStep) {
            uint32_t rawBegin = 0U;
            const bool isLastScheduledTask = taskIdx + 1U == taskLoopEnd;
            uint32_t fdPartialTaskId = 0U;
            uint32_t fdPartialCount = 0U;
            const bool isFdPartial =
                fdEnabled && GsaFd::FindPartialTask(meta, taskIdx, coreIdx, fdPartialTaskId, fdPartialCount);
            if (fdEnabled) {
                rawBegin = taskIdx == taskLoopStart ? scheduleFirstBlock : 0U;
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
                uint32_t storageLen = static_cast<uint32_t>(gCuSeqLengths.GetValue(static_cast<int64_t>(b + 1)) -
                                                            gCuSeqLengths.GetValue(static_cast<int64_t>(b)));
                uint32_t batchLen =
                    hasSequsedQ ? static_cast<uint32_t>(gSequsedQ.GetValue(static_cast<int64_t>(b))) : storageLen;
                if (qToken < accum + batchLen) {
                    batchIdx = b;
                    qTokenInBatch = qToken - accum;
                    break;
                }
                accum += batchLen;
            }

            uint32_t qStorageLen = static_cast<uint32_t>(gCuSeqLengths.GetValue(static_cast<int64_t>(batchIdx + 1)) -
                                                         gCuSeqLengths.GetValue(static_cast<int64_t>(batchIdx)));
            BatchSeqInfo kvSeq{};
            if constexpr (isPaged) {
                kvSeq.actualLen = static_cast<uint32_t>(gSequsedKv.GetValue(batchIdx));
            } else {
                // Values were validated before synchronization; use cu for storage and seqused for visibility.
                ReadBatchSeqInfo<kvFormat>(gCuSeqLengthsKv, gSequsedKv, hasSequsedKv, batchIdx,
                                tilingData->totalKvTokens, kvSeq);
            }
            const uint32_t kvSeqlen = kvSeq.actualLen;
            uint32_t qSeqlen =
                hasSequsedQ ? static_cast<uint32_t>(gSequsedQ.GetValue(static_cast<int64_t>(batchIdx))) : qStorageLen;
            const BatchSeqInfo qSeq{
                static_cast<uint64_t>(gCuSeqLengths.GetValue(batchIdx)), qStorageLen, qSeqlen};
            uint64_t gmOffsetQ = GetTensorOffset<qFormat>(qSeq, qTokenInBatch, qHeadStart, qHeads_, embed_);
            uint64_t gmOffsetO = gmOffsetQ;
            // LSE [T, N, 1]: packed GQA writes groupSize contiguous heads for one token.
            uint64_t gmOffsetLse = GetLseOffset<qFormat>(qSeq, qTokenInBatch, qHeadStart, qHeads_);

#ifdef __DAV_VEC__
            auto gmOLayoutTla = tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, strideQHead);
            auto gmOTensorTla = tla::MakeTensor(gO[gmOffsetO], gmOLayoutTla, Arch::PositionGM{});
            auto gmLseLayoutTla = tla::MakeLayout<ElementLse, LayoutLse>(qBaseTile_, 1);
            auto gmLseTensorTla = tla::MakeTensor(gLse[gmOffsetLse], gmLseLayoutTla, Arch::PositionGM{});
#endif
            if (qSeqlen == 0U || kvSeqlen == 0U || kvSeqlen < qSeqlen) {
#ifdef __DAV_VEC__
                if (isFdPartial) {
                    epilogueRescaleO.WriteNeutralPartial(gPartialO, gPartialLse, fdPartialTaskId, groupSize, embed_,
                                                         tilingData->fdLseSubStride);
                } else {
                    epilogueRescaleO.WriteEmptyOutput(gmOTensorTla, gmLseTensorTla, GemmCoord{groupSize, embed_, 0});
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
            const uint32_t visibleKvEnd = kvSeqlen - qSeqlen + qTokenInBatch + 1;
            uint32_t virtualBlockCount = explicitBlockCount;
            uint32_t residualBlockId = 0U;
            if constexpr (hasResidualBlock) {
                const uint32_t residualSize = visibleKvEnd % blockShapeY_;
                residualBlockId = visibleKvEnd / blockShapeY_;
                virtualBlockCount += static_cast<uint32_t>(residualSize != 0U);
            }
            uint32_t rawEnd = virtualBlockCount;
            if (fdEnabled) {
                rawBegin = rawBegin < virtualBlockCount ? rawBegin : virtualBlockCount;
                if (isLastScheduledTask) {
                    rawEnd = scheduleLastBlock < virtualBlockCount ? scheduleLastBlock : virtualBlockCount;
                }
            }

            uint32_t validKvTokens = 0;
            // FD changes only the selected list range. All tile coordinates below
            // are local to this range, including ordinary tasks starting at zero.
            auto gSparseBlockIdxBase = gSparseBlockIdx[sparseIdxBase];
            AscendC::GlobalTensor<int32_t> gBatchBlockTable;
            if constexpr (isPaged) {
                gBatchBlockTable = gBlockTable[static_cast<uint64_t>(batchIdx) * maxBlocksPerBatch_];
            }

            // Valid, sorted indices leave a visible prefix with at most one partial block.
            CalcSparseKvSize<hasResidualBlock>(gSparseBlockIdxBase, explicitBlockCount, rawBegin, rawEnd,
                                               blockShapeY_, visibleKvEnd, validKvTokens);
            const uint32_t explicitBegin = rawBegin < explicitBlockCount ? rawBegin : explicitBlockCount;
            const uint32_t explicitEnd = rawEnd < explicitBlockCount ? rawEnd : explicitBlockCount;
            const uint32_t localExplicitBlockCount = explicitEnd - explicitBegin;
            // Rebase the explicit list to this FD slice. For a residual-only slice this points to the
            // end of the explicit list, but GetSparseBlockId resolves the implicit block before any GM read.
            auto gTaskSparseBlockIdx = gSparseBlockIdxBase[explicitBegin];
            const uint32_t kvSLoopNum = CeilDiv(validKvTokens, kvBaseTile_);
            uint32_t rowNum = groupSize;
            uint32_t rowNumRound = RoundUp(rowNum, 16);

            if (kvSLoopNum == 0U) {
#ifdef __DAV_VEC__
                if (isFdPartial) {
                    epilogueRescaleO.WriteNeutralPartial(gPartialO, gPartialLse, fdPartialTaskId, groupSize, embed_,
                                                         tilingData->fdLseSubStride);
                } else {
                    epilogueRescaleO.WriteEmptyOutput(gmOTensorTla, gmLseTensorTla, GemmCoord{groupSize, embed_, 0});
                }
#endif
                continue;
            }

            const uint32_t lastKvTileSize = validKvTokens - (kvSLoopNum - 1U) * kvBaseTile_;
            const uint32_t lastMm1L0AStages = mm1L0ATotalStages_;
            const uint32_t lastMm1L0BStages =
                CeilDiv(lastKvTileSize, static_cast<uint32_t>(BlockMmadQK::L0_TILE_N)) *
                CeilDiv(embed_, static_cast<uint32_t>(BlockMmadQK::L0_TILE_K));

#ifdef __DAV_CUBE__
            // GM stays at batch/head (Dense) or cache/head (PA); the common loader
            // resolves each local tile through the selected sparse list range.
            const uint64_t gmOffsetKv = isPaged ? static_cast<uint64_t>(kvHeadIdx) * embed_ :
                GetTensorOffset<kvFormat>(kvSeq, 0, kvHeadIdx, kvHeads_, embed_);
            auto gmQLayoutTla = tla::MakeLayout<ElementQ, LayoutQ>(qBaseTile_, strideQHead);
            auto gmQTensorTla = tla::MakeTensor(gQ[gmOffsetQ], gmQLayoutTla, Arch::PositionGM{});
            GemmCoord actualBlockShapeQ{rowNum, embed_, 0};
            blockMmadQK.loadQGM(gmQTensorTla, actualBlockShapeQ);
#endif
#ifdef __DAV_VEC__
            // Warm up tile 0. Subsequently prepare i+1 before softmax i while Cube computes QK i.
            {
                const uint32_t prepareIdx = 0U;
                AscendC::CrossCoreWaitFlag<4, PIPE_S>(KV_OFFSET_FLAG_BASE);
                epiloguePrepareKvOffsets(
                    gKvOffsetInfo[static_cast<uint64_t>(prepareIdx % KV_OFFSET_STAGES) * kvOffsetTileWords],
                    gTaskSparseBlockIdx, gBatchBlockTable,
                    prepareIdx, min(kvBaseTile_, validKvTokens - prepareIdx * kvBaseTile_),
                    localExplicitBlockCount, residualBlockId);
                AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(KV_OFFSET_FLAG_BASE);
            }
#endif
            for (uint32_t kvBlockIdx = 0; kvBlockIdx < kvSLoopNum + PRE_LAUNCH; kvBlockIdx++) {
#ifdef __DAV_VEC__
                if (kvBlockIdx + 1U < kvSLoopNum) {
                    const uint32_t prepareIdx = kvBlockIdx + 1U;
                    const uint32_t flag = KV_OFFSET_FLAG_BASE + prepareIdx % KV_OFFSET_STAGES;
                    AscendC::CrossCoreWaitFlag<4, PIPE_S>(flag);
                    epiloguePrepareKvOffsets(
                        gKvOffsetInfo[static_cast<uint64_t>(prepareIdx % KV_OFFSET_STAGES) * kvOffsetTileWords],
                        gTaskSparseBlockIdx, gBatchBlockTable,
                        prepareIdx, min(kvBaseTile_, validKvTokens - prepareIdx * kvBaseTile_),
                        localExplicitBlockCount, residualBlockId);
                    AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(flag);
                }
#endif
                if (kvBlockIdx < kvSLoopNum) {
                    uint32_t kvSTileSizeAct = min(kvBaseTile_, validKvTokens - kvBlockIdx * kvBaseTile_);

                    GemmCoord actualBlockShapeQK{rowNum, kvSTileSizeAct, embed_};
                    uint32_t ubSBufId = kvBlockIdx % UB_S_OTMP_BUF_STAGES;
                    auto ubSLayoutTla = tla::MakeLayout<ElementS, LayoutS>(rowNumRound, RoundUp(kvSTileSizeAct, 16));
                    auto ubSTensorTla = tla::MakeTensor(ubSTensor[ubSBufId], ubSLayoutTla, Arch::PositionUB{});
                    uint32_t Mm1ToSmFlagId = ubSBufId;
                    Arch::CrossCoreFlag mm1ToSmFlag(Mm1ToSmFlagId);

#ifdef __DAV_CUBE__
                    // Ready must gate scalar GM descriptor reads, not only MTE2.
                    const uint32_t sparseFlag = KV_OFFSET_FLAG_BASE + kvBlockIdx % KV_OFFSET_STAGES;
                    AscendC::CrossCoreWaitFlag<4, PIPE_S>(sparseFlag);
                    AscendC::CrossCoreWaitFlag<4, PIPE_S>(sparseFlag + 16U);
                    auto gmKLayoutTla = tla::MakeLayout<ElementK, LayoutK>(strideKVRow, kvBaseTile_);
                    auto gmKTensorTla = tla::MakeTensor(gK[gmOffsetKv], gmKLayoutTla, Arch::PositionGM{});

                    uint64_t prefixSumL0AStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        kvBlockIdx, mm1L0ATotalStages_, lastMm1L0AStages, mm2L0ATotalStages_, kvSLoopNum, true);
                    uint64_t prefixSumL0BStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        kvBlockIdx, mm1L0BTotalStages_, lastMm1L0BStages, mm2L0BTotalStages_, kvSLoopNum, true);
                    blockMmadQK(
                        gmKTensorTla, ubSTensorTla, gTaskSparseBlockIdx, actualBlockShapeQK, kvBlockIdx,
                        kvBaseTile_, blockShapeY_, prefixSumL0AStages, prefixSumL0BStages, mm1ToSmFlag, strideKVRow,
                        gBatchBlockTable, blockSize_, kStride0_, localExplicitBlockCount, residualBlockId,
                        gKvOffsetInfo[static_cast<uint64_t>(kvBlockIdx % KV_OFFSET_STAGES) * kvOffsetTileWords],
                        kvOffsetGroupWords);
                    if (kvBlockIdx == kvSLoopNum - 1)
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
#endif
                    uint32_t l1PBufId = kvBlockIdx % pL1BufNum_;
                    uint32_t smToMm2FlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                    Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);
                    auto l1PLayoutTla = tla::MakeLayout<ElementP, NpuArch::layout::zN>(rowNum, kvSTileSizeAct);
                    auto l1PTensorTla = tla::MakeTensor(l1PTensor[l1PBufId], l1PLayoutTla, Arch::PositionL1{});

#ifdef __DAV_VEC__
                    epilogueOnlineSoftmax(l1PTensorTla, actualBlockShapeQK, (kvBlockIdx == 0), ubSBufId, l1PBufId,
                                          mm1ToSmFlag, smToMm2Flag);
#endif
                }
                if (kvBlockIdx >= PRE_LAUNCH) {
                    uint32_t kvBlockIdxDe = kvBlockIdx - PRE_LAUNCH;
                    uint32_t kvSTileSizeAct = min(kvBaseTile_, validKvTokens - kvBlockIdxDe * kvBaseTile_);

                    GemmCoord actualBlockShapePV{rowNum, embed_, kvSTileSizeAct};
                    uint32_t ubOTmpBufId = kvBlockIdxDe % uBufTileHelper_.loUbBufNum;
                    uint32_t Mm2ToReFlagId = ubOTmpBufId + UB_S_OTMP_BUF_STAGES + pL1BufNum_;

#ifdef __DAV_CUBE__
                    uint32_t l1PBufId = kvBlockIdxDe % pL1BufNum_;
                    auto ubOTmpLayoutTla = tla::MakeLayout<ElementOTmp, LayoutOTmp>(rowNumRound, embedRound);
                    auto ubOTmpTensorTla =
                        tla::MakeTensor(ubOTmpTensor[ubOTmpBufId], ubOTmpLayoutTla, Arch::PositionUB{});
                    uint32_t smToMm2FlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                    Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);
                    Arch::CrossCoreFlag mm2ToReFlag(Mm2ToReFlagId);

                    auto gmVLayoutTla = tla::MakeLayout<ElementV, LayoutV>(kvBaseTile_, strideKVRow);
                    auto gmVTensorTla = tla::MakeTensor(gV[gmOffsetKv], gmVLayoutTla, Arch::PositionGM{});

                    uint64_t prefixSumL0AStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        kvBlockIdxDe, mm1L0ATotalStages_, lastMm1L0AStages, mm2L0ATotalStages_, kvSLoopNum, false);
                    uint64_t prefixSumL0BStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        kvBlockIdxDe, mm1L0BTotalStages_, lastMm1L0BStages, mm2L0BTotalStages_, kvSLoopNum, false);
                    // The same local tile index controls sparse addressing and P/L0 rotation.
                    if (uBufTileHelper_.loGmTransit) {
                        auto gmLoLayout = tla::MakeLayout<ElementOTmp, LayoutOTmp>(rowNumRound, embedRound);
                        auto gmLoTensor = tla::MakeTensor(gLo[ubOTmpBufId * loTileElems], gmLoLayout,
                                                         Arch::PositionGM{});
                        blockMmadPV(
                            gmVTensorTla, gmLoTensor, gTaskSparseBlockIdx, actualBlockShapePV, kvBlockIdxDe,
                            kvBaseTile_, blockShapeY_, prefixSumL0AStages, prefixSumL0BStages, smToMm2Flag,
                            mm2ToReFlag, strideKVRow, gBatchBlockTable, blockSize_, vStride0_,
                            localExplicitBlockCount, residualBlockId,
                            gKvOffsetInfo[static_cast<uint64_t>(kvBlockIdxDe % KV_OFFSET_STAGES) * kvOffsetTileWords],
                            kvOffsetGroupWords, mm1L1TileN_);
                    } else {
                        blockMmadPV(
                            gmVTensorTla, ubOTmpTensorTla, gTaskSparseBlockIdx, actualBlockShapePV, kvBlockIdxDe,
                            kvBaseTile_, blockShapeY_, prefixSumL0AStages, prefixSumL0BStages, smToMm2Flag,
                            mm2ToReFlag, strideKVRow, gBatchBlockTable, blockSize_, vStride0_,
                            localExplicitBlockCount, residualBlockId,
                            gKvOffsetInfo[static_cast<uint64_t>(kvBlockIdxDe % KV_OFFSET_STAGES) * kvOffsetTileWords],
                            kvOffsetGroupWords, mm1L1TileN_);
                    }
                    // All QK/PV scalar descriptor reads are complete. GM holds only
                    // addresses, so no MTE2 data-read completion is needed for its reuse.
                    const uint32_t freeFlag = KV_OFFSET_FLAG_BASE + kvBlockIdxDe % KV_OFFSET_STAGES;
                    AscendC::CrossCoreSetFlag<4, PIPE_S>(freeFlag);
                    AscendC::CrossCoreSetFlag<4, PIPE_S>(freeFlag + 16U);
#endif
#ifdef __DAV_VEC__
                    Arch::CrossCoreFlag mm2ToReFlag(Mm2ToReFlagId);
                    uint32_t curTileMod = kvBlockIdxDe % (PRE_LAUNCH + 1);
                    if (isFdPartial) {
                        epilogueRescaleO.ProcessPartial(gmOTensorTla, gmLseTensorTla, actualBlockShapePV, curTileMod,
                                                        kvBlockIdxDe, (kvBlockIdxDe == 0),
                                                        (kvBlockIdxDe == kvSLoopNum - 1), mm2ToReFlag, gPartialO,
                                                        gPartialLse, fdPartialTaskId, tilingData->fdLseSubStride);
                    } else {
                        epilogueRescaleO(gmOTensorTla, gmLseTensorTla, actualBlockShapePV, curTileMod, kvBlockIdxDe,
                                         (kvBlockIdxDe == 0), (kvBlockIdxDe == kvSLoopNum - 1), mm2ToReFlag);
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
            using RescaleDispatchPolicy = typename EpilogueRescaleO::DispatchPolicy;
            constexpr bool OUTPUT_LSE = RescaleDispatchPolicy::LSE_MODE == Epilogue::LseMode::OUT_ONLY;
            GenericBlockSparseAttentionFdCombineArch35<ElementO, Arch::Resource<ArchTag>, OUTPUT_LSE> combine(resource);
            combine(meta, tilingData, gPartialLse, gPartialO, gO, gLse, gCuSeqLengths, gSequsedQ, hasSequsedQ);
#endif
        }
    }

private:
    __aicore__ inline void FetchBaseShapeInfo(
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
        kStride0_ = tilingData->kStride0;
        vStride0_ = tilingData->vStride0;
        isConsistentTopk_ = tilingData->isConsistentTopk;
    }

    __aicore__ inline void CalcOnChipBufTileInfo(
        __gm__ GenericBlockSparseAttn::GenericBlockSparseAttentionTilingData *tilingData)
    {
        mm1L1TileM_ = tilingData->mmPhaseL1TileInfo.mm1L1TileM;
        mm1L1TileN_ = tilingData->mmPhaseL1TileInfo.mm1L1TileN;
        mm1L1TileKLeft_ = tilingData->mmPhaseL1TileInfo.mm1L1TileKLeft;
        mm1L1TileKRight_ = tilingData->mmPhaseL1TileInfo.mm1L1TileKRight;
        mm2L1TileM_ = tilingData->mmPhaseL1TileInfo.mm2L1TileM;
        mm2L1TileN_ = tilingData->mmPhaseL1TileInfo.mm2L1TileN;
        mm2L1TileKLeft_ = tilingData->mmPhaseL1TileInfo.mm2L1TileKLeft;
        mm2L1TileKRight_ = tilingData->mmPhaseL1TileInfo.mm2L1TileKRight;
        qL1BufNum_ = tilingData->mmPhaseL1TileInfo.qL1BufNum;
        kL1BufNum_ = tilingData->mmPhaseL1TileInfo.kL1BufNum;
        vL1BufNum_ = tilingData->mmPhaseL1TileInfo.vL1BufNum;
        pL1BufNum_ = tilingData->mmPhaseL1TileInfo.pL1BufNum;
        Gemm::Block::Mm1L1TileHelper mm1L1TileHelper(mm1L1TileM_, mm1L1TileN_, mm1L1TileKLeft_, mm1L1TileKRight_,
                                                     qL1BufNum_, kL1BufNum_);
        mm1L1TileHelper_ = mm1L1TileHelper;
        Gemm::Block::Mm2L1TileHelper mm2L1TileHelper(mm2L1TileM_, mm2L1TileN_, mm2L1TileKLeft_, mm2L1TileKRight_,
                                                     pL1BufNum_, vL1BufNum_);
        mm2L1TileHelper_ = mm2L1TileHelper;
        mm2L1AddrStart_ = mm1L1TileM_ * mm1L1TileKLeft_ * qL1BufNum_ * sizeof(ElementQ) +
                          mm1L1TileKRight_ * mm1L1TileN_ * kL1BufNum_ * sizeof(ElementK);
        uint32_t mL0LoopQK = CeilDiv(groupSize_, static_cast<uint32_t>(BlockMmadQK::L0_TILE_M));
        uint32_t mL0LoopPV = CeilDiv(groupSize_, static_cast<uint32_t>(BlockMmadPV::L0_TILE_M));
        const uint32_t qkKLoop = CeilDiv(embed_, static_cast<uint32_t>(BlockMmadQK::L0_TILE_K));
        const uint32_t pvKLoop = CeilDiv(kvBaseTile_, static_cast<uint32_t>(BlockMmadPV::L0_TILE_K));
        mm1L0ATotalStages_ = mL0LoopQK * qkKLoop;
        mm1L0BTotalStages_ =
            CeilDiv(kvBaseTile_, static_cast<uint32_t>(BlockMmadQK::L0_TILE_N)) * qkKLoop;
        mm2L0ATotalStages_ = mL0LoopPV * pvKLoop;
        mm2L0BTotalStages_ = pvKLoop * CeilDiv(embed_, static_cast<uint32_t>(BlockMmadPV::L0_TILE_N));
    }

    __aicore__ inline uint64_t CalcCrossMm1Mm2PrefixSumL0ABStages(
        uint32_t kvBlockIdx, uint32_t singleMm1L0Stages, uint32_t lastMm1L0Stages,
        uint32_t singleMm2L0Stages, uint32_t kvSLoopNum, bool isCurPhaseMm1)
    {
        uint64_t prefixSumStages;
        if (isCurPhaseMm1) {
            const uint32_t launchedMm2Num = kvBlockIdx > PRE_LAUNCH ? kvBlockIdx - PRE_LAUNCH : 0U;
            prefixSumStages = static_cast<uint64_t>(kvBlockIdx) * singleMm1L0Stages +
                              static_cast<uint64_t>(launchedMm2Num) * singleMm2L0Stages;
        } else {
            const uint32_t desiredMm1Num = kvBlockIdx + PRE_LAUNCH + 1U;
            const uint32_t launchedMm1Num = desiredMm1Num < kvSLoopNum ? desiredMm1Num : kvSLoopNum;
            uint64_t launchedMm1Stages = static_cast<uint64_t>(launchedMm1Num) * singleMm1L0Stages;
            if (launchedMm1Num == kvSLoopNum) {
                launchedMm1Stages = static_cast<uint64_t>(kvSLoopNum - 1U) * singleMm1L0Stages +
                                    lastMm1L0Stages;
            }
            prefixSumStages = launchedMm1Stages + static_cast<uint64_t>(kvBlockIdx) * singleMm2L0Stages;
        }
        return prefixSumStages;
    }

    __aicore__ inline void CalcUBufTileInfo()
    {
        constexpr uint32_t VECTOR_SUB_CORE_NUM = 2U;
        // P is stored in NZ format: even an 8-row sub-core tile occupies 16 rows.
        const uint32_t qBaseTilePerSubCore = RoundUp(qBaseTile_ / VECTOR_SUB_CORE_NUM, 16U);
        constexpr uint32_t statsRows = Epilogue::Block::UBufTileHelper::STATS_ROW_NUM;
        uint32_t kvBaseTilePerSubCore = RoundUp(kvBaseTile_, 32);
        uint32_t embedPerSubCore = RoundUp(embed_, 32);
        // D in (128, 256] with qBaseTile=128 cannot hold two complete LO
        // buffers in the 248-KB usable UB budget. In that case LO is single
        // buffered; the MM2/rescale cross-core flag serializes its reuse.
        const bool loGmTransit = embed_ > 256U && qBaseTile_ > 64U;
        const uint32_t loUbBufNum = !loGmTransit && embed_ > 128U && qBaseTile_ > 64U ?
                                       1U : UB_S_OTMP_BUF_STAGES;
        uint32_t sStartOffset = 0;
        uint32_t pStartOffset =
            sStartOffset + qBaseTilePerSubCore * kvBaseTilePerSubCore * sizeof(ElementS) * UB_S_OTMP_BUF_STAGES;
        uint32_t loStartOffset =
            pStartOffset + qBaseTilePerSubCore * kvBaseTilePerSubCore * sizeof(ElementP) * UB_S_OTMP_BUF_STAGES;
        uint32_t goStartOffset =
            loStartOffset + qBaseTilePerSubCore * embedPerSubCore * sizeof(ElementOTmp) * loUbBufNum;
        uint32_t lmStartOffset = goStartOffset + qBaseTilePerSubCore * embedPerSubCore * sizeof(ElementOTmp);
        uint32_t gmStartOffset = lmStartOffset + statsRows * sizeof(float);
        uint32_t dmStartOffset = gmStartOffset + statsRows * sizeof(float);
        uint32_t llStartOffset = dmStartOffset + statsRows * (PRE_LAUNCH + 1) * sizeof(float);
        uint32_t glStartOffset = llStartOffset + statsRows * sizeof(float);
        uint32_t lseStartOffset = glStartOffset + statsRows * sizeof(float);
        uint32_t maskStartOffset = lseStartOffset + statsRows * 32U;
        constexpr uint32_t usableUbBytes = 248U * 1024U;
        const uint32_t kvOffsetGroupBytes =
            CalcKvOffsetGroupWords<isPaged>(mm1L1TileN_, blockShapeY_) * sizeof(uint64_t);
        // Preserve the resident LO layout when a second descriptor group does not fit.
        const uint32_t kvOffsetUbBufNum = loGmTransit ||
            maskStartOffset + 2U * kvOffsetGroupBytes <= usableUbBytes ? 2U : 1U;

        uint32_t loUbRowNum = qBaseTilePerSubCore;
        if (loGmTransit) {
            // Keep GO resident. Allocate all remaining usable UB to two row chunks of LO.
            const uint32_t statsBytes = maskStartOffset - lmStartOffset;
            const uint32_t maskBytes = max(qBaseTilePerSubCore * kvBaseTilePerSubCore,
                kvOffsetGroupBytes * kvOffsetUbBufNum);
            goStartOffset = loStartOffset;
            lmStartOffset = goStartOffset + qBaseTilePerSubCore * embedPerSubCore * sizeof(ElementOTmp);
            gmStartOffset = lmStartOffset + statsRows * sizeof(float);
            dmStartOffset = gmStartOffset + statsRows * sizeof(float);
            llStartOffset = dmStartOffset + statsRows * (PRE_LAUNCH + 1) * sizeof(float);
            glStartOffset = llStartOffset + statsRows * sizeof(float);
            lseStartOffset = glStartOffset + statsRows * sizeof(float);
            maskStartOffset = lmStartOffset + statsBytes;
            loStartOffset = maskStartOffset + maskBytes;
            loUbRowNum = min(qBaseTilePerSubCore,
                            (usableUbBytes - loStartOffset) / (embedPerSubCore * sizeof(ElementOTmp) * loUbBufNum));
        }

        uBufTileHelper_ = Epilogue::Block::UBufTileHelper(qBaseTilePerSubCore, kvBaseTilePerSubCore, embedPerSubCore,
                                                          sStartOffset, pStartOffset, loStartOffset, goStartOffset,
                                                          lmStartOffset, gmStartOffset, dmStartOffset, llStartOffset,
                                                          glStartOffset, lseStartOffset, maskStartOffset, loUbBufNum);
        uBufTileHelper_.loUbRowNum = loUbRowNum;
        uBufTileHelper_.loGmTransit = loGmTransit;
        uBufTileHelper_.kvOffsetUbBufNum = kvOffsetUbBufNum;
    }

    __aicore__ inline void InitCrossCoreDstBuf(AscendC::LocalTensor<ElementP> (&l1PTensor)[MAX_CROSS_CORE_BUF_STAGES],
                                               AscendC::LocalTensor<ElementS> (&ubSTensor)[UB_S_OTMP_BUF_STAGES],
                                               AscendC::LocalTensor<ElementOTmp> (&ubOTmpTensor)[UB_S_OTMP_BUF_STAGES])
    {
        for (uint32_t i = 0; i < pL1BufNum_; i++) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<ElementP>(
                mm2L1AddrStart_ + mm2L1TileM_ * mm2L1TileKLeft_ * sizeof(ElementP) * i);
        }
        for (uint32_t i = 0; i < UB_S_OTMP_BUF_STAGES; i++) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(
                uBufTileHelper_.sStartOffset +
                uBufTileHelper_.qBaseTilePerSubCore * uBufTileHelper_.kvBaseTilePerSubCore * sizeof(ElementS) * i);
            ubOTmpTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                uBufTileHelper_.loStartOffset +
                uBufTileHelper_.loUbRowNum * uBufTileHelper_.embedPerSubCore * sizeof(ElementOTmp) *
                    (i % uBufTileHelper_.loUbBufNum));
        }
    }

    template <uint32_t MM1_SM_MODE, uint32_t MM2_RE_MODE, uint32_t SM_MM2_MODE>
    __aicore__ inline void InitSyncFlags()
    {
#ifdef __DAV_CUBE__
        for (uint32_t i = 0; i < KV_OFFSET_STAGES; ++i) {
            AscendC::CrossCoreSetFlag<4, PIPE_S>(KV_OFFSET_FLAG_BASE + i);
            AscendC::CrossCoreSetFlag<4, PIPE_S>(KV_OFFSET_FLAG_BASE + i + 16U);
        }
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
        // Seed the descriptor scratch buffer's first-use completion event.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID5);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID6);
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
    __aicore__ inline void ReleaseSyncFlags()
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
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID6);
        for (uint32_t i = 0; i < KV_OFFSET_STAGES; ++i) {
            AscendC::CrossCoreWaitFlag<4, PIPE_S>(KV_OFFSET_FLAG_BASE + i);
        }
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
    uint32_t groupSize_;
    // PA_BBND page base strides (elements); may exceed blockSize*Nkv*D when dim0 is strided.
    uint64_t kStride0_;
    uint64_t vStride0_;
    uint32_t isConsistentTopk_;
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
    Epilogue::Block::UBufTileHelper uBufTileHelper_;
};

} // namespace GbsaKernelArch35
