/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_TOKEN_DISPATCH_H
#define MEGA_MOE_TOKEN_DISPATCH_H

#include "../mega_moe_job_context.h"
#include "../mega_moe_impl.h"

namespace MegaMoeImpl {

// Token Dispatch uses a logical block job for work partitioning and a physical
// workspace slot for count/flag publication. Neither index is derived here.
struct TokenDispatchContext {
    BlockJobContext blockJob;
    BlockWorkspaceContext countWorkspace;
    uint32_t worldSize;
    uint32_t expertCapacityPerRank; // Weight/workspace capacity, including reserved expert slots.
    uint32_t tokenHiddenDim;
    uint32_t topK;
    uint32_t blockNumPerRank;
    uint64_t maxOutputSize;
    uint64_t sendTotalNum;
    MegaMoeDispatchBufferConfig bufferConfig;
    uint32_t maskAlignSize;
    uint32_t maskSlotSize;
    uint64_t quantWinOffset;
    uint32_t quantTokenAlignBytes;
    uint32_t quantScaleAlignBytes;
    uint32_t quantTokenScaleAlignBytes;
    int32_t dispatchFlagSlotsPerExpert;
    uint32_t activationElementsPerByte;
};

struct TokenDispatchArgs {
    GM_ADDR *winRankAddr;
    GM_ADDR maskRecvPtr;
    GM_ADDR expertRevTokenNumsPtr;
    GM_ADDR metaInfoPtr;
    GM_ADDR cumsumInfoPtr;
    GM_ADDR dispatchRevDataPtr;
    GM_ADDR dispatchRevScalePtr;
    GM_ADDR dispatchToGmm1FlagPtr;
    GM_ADDR sendCntFlagPtr;
};

template <typename ActivationType>
struct TokenDispatchScratch {
    GlobalTensor<int32_t> expertRevNumsGlobalTensor;
    GlobalTensor<int32_t> cumsumInfoGlobalTensor;
    LocalTensor<int32_t> topkIndexTensor;
    LocalTensor<int32_t> sendCntTensor;
    LocalTensor<uint8_t> maskBatchTensor;
    LocalTensor<uint32_t> maskBatchU32Tensor;
    LocalTensor<int32_t> expertTokenCntTensor;
    LocalTensor<int32_t> validTopkIndexTensor;
    LocalTensor<int32_t> cumsumInfoTensor;
    // Contiguous runtime-sized ring. A slot view is built from this UB base on demand.
    uint32_t copyTmpBaseAddr;
    LocalTensor<int32_t> metaInfoTensor;
    LocalTensor<int32_t> expertTokenNumsOutTensor;
    int64_t revTokenElemCnt;
    int64_t revScaleElemCnt;
    uint64_t cumsumRevCntInRank;
};

// Prototype: MegaMoe::DispatchBuffInit. Builds the bounded-UB scratch layout used by Token Dispatch.
template <typename ActivationType, bool EnableA8W4>
__aicore__ inline void InitTokenDispatchBuffers(const TokenDispatchContext &context, const TokenDispatchArgs &args,
                                                TokenDispatchScratch<ActivationType> &scratch)
{
    scratch.expertRevNumsGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(args.expertRevTokenNumsPtr));
    if constexpr (g_coreType == AIC) {
        return;
    }
    // Only AIV1 performs SendCnt/Dispatch and later exports the cumsum result.
    if (GetSubBlockIdx() != 1) {
        return;
    }
    const MegaMoeDispatchBufferConfig &bufferConfig = context.bufferConfig;
    scratch.revTokenElemCnt = context.tokenHiddenDim / context.activationElementsPerByte;
    scratch.revScaleElemCnt =
        Ops::Base::CeilDiv(static_cast<int64_t>(context.tokenHiddenDim), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;
    uint32_t cumsumInfoTensorSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(context.worldSize * context.expertCapacityPerRank * sizeof(int32_t)),
                             static_cast<int64_t>(ALIGN_32));
    if constexpr (EnableA8W4) {
        scratch.cumsumInfoGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(
            args.cumsumInfoPtr + static_cast<uint64_t>(cumsumInfoTensorSize) * context.countWorkspace.blockIdx));
    }
    scratch.cumsumRevCntInRank = 0;

    uint32_t expertTokenCntTensorAddr = 0;
    uint32_t expertTokenCntTensorSize = ALIGN_32;
    scratch.expertTokenCntTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, expertTokenCntTensorAddr, expertTokenCntTensorSize / sizeof(int32_t));

    uint32_t cumsumInfoTensorAddr = expertTokenCntTensorAddr + expertTokenCntTensorSize;
    scratch.cumsumInfoTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, cumsumInfoTensorAddr, cumsumInfoTensorSize / sizeof(int32_t));

    uint32_t sendCntTensorAddr = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    uint32_t sendCntTensorSize = context.worldSize * static_cast<uint32_t>(ALIGN_32);
    scratch.sendCntTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, sendCntTensorAddr, sendCntTensorSize / sizeof(int32_t));

    uint32_t maskBatchTensorAddr = sendCntTensorAddr + sendCntTensorSize;
    uint32_t maskBatchTensorSize = static_cast<uint32_t>(bufferConfig.routeItemsPerBatch / 8);
    scratch.maskBatchTensor = LocalTensor<uint8_t>(TPosition::VECCALC, maskBatchTensorAddr, maskBatchTensorSize);
    scratch.maskBatchU32Tensor =
        LocalTensor<uint32_t>(TPosition::VECCALC, maskBatchTensorAddr, maskBatchTensorSize / sizeof(uint32_t));

    uint32_t validTopkIndexTensorAddr = maskBatchTensorAddr + maskBatchTensorSize;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(bufferConfig.routeItemsPerBatch * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    scratch.validTopkIndexTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, validTopkIndexTensorAddr, validTopkIndexTensorSize / sizeof(int32_t));

    uint32_t topkIndexTensorAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t topkIndexTensorSize = validTopkIndexTensorSize;
    scratch.topkIndexTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, topkIndexTensorAddr, topkIndexTensorSize / sizeof(int32_t));

    scratch.copyTmpBaseAddr = topkIndexTensorAddr + topkIndexTensorSize;
    uint32_t copyTmpTotalSize = static_cast<uint32_t>(bufferConfig.bufferCount) * context.quantTokenScaleAlignBytes;

    uint32_t expertTokenNumsOutTensorAddr = scratch.copyTmpBaseAddr + copyTmpTotalSize;
    uint32_t expertTokenNumsOutTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(context.expertCapacityPerRank * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    scratch.expertTokenNumsOutTensor = LocalTensor<int32_t>(TPosition::VECCALC, expertTokenNumsOutTensorAddr,
                                                            expertTokenNumsOutTensorSize / sizeof(int32_t));
    uint32_t metaInfoTensorAddr = expertTokenNumsOutTensorAddr + expertTokenNumsOutTensorSize;
    uint32_t metaInfoTensorSize = static_cast<uint32_t>(bufferConfig.bufferCount) * INT32_PER_256B * sizeof(int32_t);
    scratch.metaInfoTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, metaInfoTensorAddr, metaInfoTensorSize / sizeof(int32_t));

    // ComputeExpertTokenCountAndNotify overwrites every cumsum element before it is read. Clearing it here
    // would also introduce an unsynchronized V-to-S WAW with the first scalar updates.
}

template <typename ActivationType>
__aicore__ inline LocalTensor<ActivationType> GetDispatchCopyBuffer(
    const TokenDispatchContext &context, const TokenDispatchScratch<ActivationType> &scratch, int32_t bufferIdx)
{
    return LocalTensor<ActivationType>(
        TPosition::VECCALC,
        scratch.copyTmpBaseAddr + static_cast<uint32_t>(bufferIdx) * context.quantTokenScaleAlignBytes,
        context.quantTokenScaleAlignBytes / sizeof(ActivationType));
}

template <bool IsBufferReuse, bool TopkWeightsPrefetch, typename ActivationType>
__aicore__ inline void FetchDispatchTokenAndMetaInfo(const TokenDispatchContext &context,
                                                     TokenDispatchScratch<ActivationType> &scratch,
                                                     int32_t bufferIdx, int32_t topkIndex, int32_t remoteRankIdx,
                                                     GlobalTensor<ActivationType> &remoteRankGlobalTensor)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx);
    LocalTensor<ActivationType> copyTmpTensor = GetDispatchCopyBuffer(context, scratch, bufferIdx);
    int32_t tokenIndex = topkIndex / static_cast<int32_t>(context.topK);
    uint64_t remoteCopyOffset =
        static_cast<uint64_t>(tokenIndex) * static_cast<uint64_t>(context.quantTokenScaleAlignBytes);
    if constexpr (IsBufferReuse) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    }
    DataCopy(copyTmpTensor, remoteRankGlobalTensor[remoteCopyOffset], context.quantTokenScaleAlignBytes);
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);

    if constexpr (IsBufferReuse) {
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
    }
    scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(RANK_ID, remoteRankIdx);
    scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(TOKEN_ID, tokenIndex);
    scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(TOPK_INDEX,
                                                                topkIndex % static_cast<int32_t>(context.topK));
    if constexpr (TopkWeightsPrefetch) {
        SetFlag<AscendC::HardEvent::MTE2_S>(eventId);
    } else {
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }
}

template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void StoreDispatchTokenAndMetaInfo(
    const TokenDispatchContext &context, TokenDispatchScratch<ActivationType> &scratch, int32_t bufferIdx,
    int32_t dstIdx, GlobalTensor<ActivationType> &tokenRevGlobalTensor,
    GlobalTensor<QuantScaleType> &scaleRevGlobalTensor, GlobalTensor<int32_t> &metaInfoGlobalTensor,
    int32_t copyStartIdx, int32_t copyIdx)
{
    TEventID eventId = static_cast<TEventID>(bufferIdx);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
    LocalTensor<ActivationType> tokenScaleBuffer = GetDispatchCopyBuffer(context, scratch, bufferIdx);
    LocalTensor<QuantScaleType> scaleBuffer =
        tokenScaleBuffer[context.quantTokenAlignBytes].template ReinterpretCast<QuantScaleType>();
    if constexpr (TopkWeightsPrefetch) {
        WaitFlag<AscendC::HardEvent::MTE2_S>(eventId);
        uint32_t weightOffsetInUb = context.quantTokenAlignBytes + context.quantScaleAlignBytes;
        LocalTensor<int32_t> weightBitsTensor = tokenScaleBuffer[weightOffsetInUb].template ReinterpretCast<int32_t>();
        int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx + copyIdx);
        int32_t weightBits =
            weightBitsTensor.GetValue(static_cast<uint32_t>(topkIndex % static_cast<int32_t>(context.topK)));
        scratch.metaInfoTensor[bufferIdx * INT32_PER_256B].SetValue(WEIGHT_INDEX, weightBits);
        SetFlag<AscendC::HardEvent::S_MTE3>(eventId);
    }
    DataCopyPad(tokenRevGlobalTensor[dstIdx * scratch.revTokenElemCnt], tokenScaleBuffer,
                {1, static_cast<uint16_t>(scratch.revTokenElemCnt * sizeof(ActivationType)), 0U, 0U, 0U});
    DataCopyPad(scaleRevGlobalTensor[dstIdx * scratch.revScaleElemCnt], scaleBuffer,
                {1, static_cast<uint16_t>(scratch.revScaleElemCnt * sizeof(QuantScaleType)), 0U, 0U, 0U});
    WaitFlag<AscendC::HardEvent::S_MTE3>(eventId);
    DataCopy(metaInfoGlobalTensor[dstIdx * INT32_PER_256B], scratch.metaInfoTensor[bufferIdx * INT32_PER_256B],
             INT32_PER_256B);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
    SetFlag<AscendC::HardEvent::MTE3_S>(eventId);
}

// Prototype: MegaMoe::CopyGMToGMPerToken. Copies one dispatch shard and emits its metadata.
template <typename ActivationType, typename QuantScaleType, bool TopkWeightsPrefetch>
__aicore__ inline void CopyTokensAndMetaForDispatch(
    const TokenDispatchContext &context, const TokenDispatchArgs &args,
    TokenDispatchScratch<ActivationType> &scratch, int32_t rowDstOffset, int32_t remoteRankIdx,
    int32_t copyStartIdx, int32_t copyNum)
{
    const MegaMoeDispatchBufferConfig &bufferConfig = context.bufferConfig;
    int32_t bufferCount = bufferConfig.bufferCount;

    GlobalTensor<ActivationType> remoteRankGlobalTensor;
    GlobalTensor<ActivationType> tokenRevGlobalTensor;
    GlobalTensor<QuantScaleType> scaleRevGlobalTensor;
    GlobalTensor<int32_t> metaInfoGlobalTensor;
    tokenRevGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(args.dispatchRevDataPtr + rowDstOffset * scratch.revTokenElemCnt));
    scaleRevGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ QuantScaleType *>(args.dispatchRevScalePtr + rowDstOffset * scratch.revScaleElemCnt));
    remoteRankGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(args.winRankAddr[remoteRankIdx] + context.quantWinOffset));
    metaInfoGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(args.metaInfoPtr + rowDstOffset * INT32_PER_256B * sizeof(int32_t)));

    // The only caller enters this function for a non-empty match interval, so copyNum is at least one.
    // GatherMask's V-to-S synchronization covers the first scalar read, and the drain below covers
    // cross-call reuse of the copy and metadata rings.
    int32_t firstTopkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx);
    FetchDispatchTokenAndMetaInfo<false, TopkWeightsPrefetch>(context, scratch, 0, firstTopkIndex, remoteRankIdx,
                                                              remoteRankGlobalTensor);

    int32_t firstUseEnd = copyNum < bufferCount ? copyNum : bufferCount;
    for (int32_t issueIdx = 1; issueIdx < firstUseEnd; ++issueIdx) {
        int32_t copyIdx = issueIdx - 1;
        int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx + issueIdx);
        FetchDispatchTokenAndMetaInfo<false, TopkWeightsPrefetch>(context, scratch, issueIdx, topkIndex, remoteRankIdx,
                                                                  remoteRankGlobalTensor);
        StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, scratch, copyIdx, copyIdx, tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
            copyStartIdx, copyIdx);
    }

    for (int32_t issueIdx = bufferCount; issueIdx < copyNum; ++issueIdx) {
        int32_t copyIdx = issueIdx - 1;
        int32_t issueBufferIdx = issueIdx % bufferCount;
        int32_t copyBufferIdx = copyIdx % bufferCount;
        int32_t topkIndex = scratch.validTopkIndexTensor.GetValue(copyStartIdx + issueIdx);
        FetchDispatchTokenAndMetaInfo<true, TopkWeightsPrefetch>(context, scratch, issueBufferIdx, topkIndex,
                                                                 remoteRankIdx, remoteRankGlobalTensor);
        StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
            context, scratch, copyBufferIdx, copyIdx, tokenRevGlobalTensor, scaleRevGlobalTensor, metaInfoGlobalTensor,
            copyStartIdx, copyIdx);
    }

    StoreDispatchTokenAndMetaInfo<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
        context, scratch, (copyNum - 1) % bufferCount, copyNum - 1, tokenRevGlobalTensor, scaleRevGlobalTensor,
        metaInfoGlobalTensor, copyStartIdx, copyNum - 1);

    for (int32_t bufferIdx = 0; bufferIdx < firstUseEnd; ++bufferIdx) {
        TEventID eventId = static_cast<TEventID>(bufferIdx);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<AscendC::HardEvent::MTE3_S>(eventId);
    }
}

struct DispatchShardRange {
    uint32_t remoteRankIdx;
    uint32_t rankSegmentDstRowBegin;
    int32_t matchOrdinalBegin;
    int32_t matchOrdinalEnd;
    int32_t dstRowBegin;
};

template <typename ActivationType>
__aicore__ inline DispatchShardRange CalculateDispatchShardRange(
    const TokenDispatchContext &context, const TokenDispatchScratch<ActivationType> &scratch,
    uint32_t localExpertId, uint32_t dispatchShardIdx)
{
    uint32_t remoteRankIdx = dispatchShardIdx / context.blockNumPerRank;
    uint32_t rankShardIdx = dispatchShardIdx % context.blockNumPerRank;
    uint32_t rankSegmentDstRowBegin =
        (remoteRankIdx == 0U && localExpertId == 0U) ?
            0U :
            static_cast<uint32_t>(
                scratch.cumsumInfoTensor.GetValue(localExpertId * context.worldSize + remoteRankIdx - 1U));
    DispatchShardRange range{remoteRankIdx, rankSegmentDstRowBegin, 0, 0, 0};
    if (rankSegmentDstRowBegin >= context.maxOutputSize) {
        return range;
    }

    int32_t rankTokenCount =
        scratch.cumsumInfoTensor.GetValue(localExpertId * context.worldSize + remoteRankIdx) -
        static_cast<int32_t>(rankSegmentDstRowBegin);
    int32_t rankDispatchRowCount =
        rankSegmentDstRowBegin + rankTokenCount > context.maxOutputSize ?
            static_cast<int32_t>(context.maxOutputSize - rankSegmentDstRowBegin) :
            rankTokenCount;
    int32_t rowsPerRankShard =
        Ops::Base::CeilDiv(rankDispatchRowCount, static_cast<int32_t>(context.blockNumPerRank));
    int32_t rankShardRowBegin = static_cast<int32_t>(rankShardIdx) * rowsPerRankShard;
    range.dstRowBegin = static_cast<int32_t>(rankSegmentDstRowBegin) + rankShardRowBegin;
    int32_t dispatchRowCount =
        range.dstRowBegin + rowsPerRankShard > rankSegmentDstRowBegin + rankDispatchRowCount ?
            static_cast<int32_t>(rankSegmentDstRowBegin + rankDispatchRowCount - range.dstRowBegin) :
            rowsPerRankShard;
    if (dispatchRowCount > 0) {
        range.matchOrdinalBegin = rankShardRowBegin;
        range.matchOrdinalEnd = rankShardRowBegin + dispatchRowCount;
    }
    return range;
}

// Prototype: MegaMoe::MetaInfoCalAndDispatch. Scans route batches and dispatches one block's token shards.
template <typename ActivationType, typename QuantScaleType, bool EnableA8W4, uint32_t PipelineTileM,
          bool TopkWeightsPrefetch>
__aicore__ inline void DispatchExpertTokens(const TokenDispatchContext &context, const TokenDispatchArgs &args,
                                            TokenDispatchScratch<ActivationType> &scratch, uint32_t localExpertId)
{
    constexpr int32_t dispatchWaveRowCount = static_cast<int32_t>(PipelineTileM);
    const MegaMoeDispatchBufferConfig &bufferConfig = context.bufferConfig;
    int32_t expertGlobalRowBegin =
        localExpertId == 0 ? 0 : scratch.cumsumInfoTensor.GetValue(localExpertId * context.worldSize - 1);
    __gm__ int32_t *dispatchWaveReadyCount = reinterpret_cast<__gm__ int32_t *>(args.dispatchToGmm1FlagPtr) +
                                             static_cast<uint64_t>(localExpertId) *
                                                 context.dispatchFlagSlotsPerExpert;

    for (uint32_t dispatchShardIdx = context.blockJob.jobIndex;
         dispatchShardIdx < context.worldSize * context.blockNumPerRank;
         dispatchShardIdx += context.blockJob.totalJobs) {
        DispatchShardRange shardRange =
            CalculateDispatchShardRange(context, scratch, localExpertId, dispatchShardIdx);

        GlobalTensor<uint8_t> remoteRankMaskGlobal;
        int32_t matchedRouteCount = 0;
        int32_t dispatchedRowCount = 0;
        for (int32_t batchIdx = 0;
             batchIdx < bufferConfig.routeBatchCount && matchedRouteCount < shardRange.matchOrdinalEnd;
             ++batchIdx) {
            int32_t batchRouteBegin = batchIdx * bufferConfig.routeItemsPerBatch;
            bool isLastBatch = batchIdx == bufferConfig.routeBatchCount - 1;
            int32_t validRouteCount = bufferConfig.routeItemsPerBatch;
            int32_t maskSliceBytes = bufferConfig.routeItemsPerBatch / 8;
            if (isLastBatch) {
                validRouteCount = static_cast<int32_t>(context.sendTotalNum - static_cast<uint64_t>(batchRouteBegin));
                if (batchRouteBegin / 8 + maskSliceBytes > static_cast<int32_t>(context.maskAlignSize)) {
                    maskSliceBytes = static_cast<int32_t>(context.maskAlignSize) - batchRouteBegin / 8;
                }
            }
            remoteRankMaskGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(
                args.maskRecvPtr +
                (static_cast<uint64_t>(localExpertId) * context.worldSize + shardRange.remoteRankIdx) *
                    context.maskSlotSize +
                static_cast<uint64_t>(batchRouteBegin / 8)));
            DataCopy(scratch.maskBatchTensor, remoteRankMaskGlobal, static_cast<uint32_t>(maskSliceBytes));
            SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_EVENT_ID1>();
            CreateVecIndex(scratch.topkIndexTensor, batchRouteBegin, bufferConfig.routeItemsPerBatch);
            uint64_t batchMatchedRouteCount = 0;
            GatherMask(scratch.validTopkIndexTensor, scratch.topkIndexTensor, scratch.maskBatchU32Tensor, true,
                       static_cast<uint32_t>(validRouteCount), {1, 1, 0, 0}, batchMatchedRouteCount);
            SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID4>();

            int32_t batchMatchOrdinalBegin = matchedRouteCount;
            int32_t batchMatchOrdinalEnd = matchedRouteCount + static_cast<int32_t>(batchMatchedRouteCount);
            int32_t dispatchMatchOrdinalBegin =
                batchMatchOrdinalBegin > shardRange.matchOrdinalBegin ? batchMatchOrdinalBegin :
                                                                        shardRange.matchOrdinalBegin;
            int32_t dispatchMatchOrdinalEnd =
                batchMatchOrdinalEnd < shardRange.matchOrdinalEnd ? batchMatchOrdinalEnd : shardRange.matchOrdinalEnd;
            if (dispatchMatchOrdinalEnd > dispatchMatchOrdinalBegin) {
                int32_t batchLocalMatchBegin = dispatchMatchOrdinalBegin - batchMatchOrdinalBegin;
                int32_t batchDispatchRowCount = dispatchMatchOrdinalEnd - dispatchMatchOrdinalBegin;
                int32_t dispatchDstRowBegin =
                    static_cast<int32_t>(shardRange.rankSegmentDstRowBegin) + dispatchMatchOrdinalBegin;
                CopyTokensAndMetaForDispatch<ActivationType, QuantScaleType, TopkWeightsPrefetch>(
                    context, args, scratch, dispatchDstRowBegin, shardRange.remoteRankIdx, batchLocalMatchBegin,
                    batchDispatchRowCount);
                dispatchedRowCount += batchDispatchRowCount;
            }
            matchedRouteCount = batchMatchOrdinalEnd;
        }

        if (dispatchedRowCount > 0) {
            SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID5>();
            int32_t coreExpertRowBegin = shardRange.dstRowBegin - expertGlobalRowBegin;
            int32_t coreExpertRowEnd = coreExpertRowBegin + dispatchedRowCount;
            int32_t firstWaveIdx = coreExpertRowBegin / dispatchWaveRowCount;
            int32_t lastWaveIdx = (coreExpertRowEnd - 1) / dispatchWaveRowCount;
            for (int32_t waveIdx = firstWaveIdx; waveIdx <= lastWaveIdx; ++waveIdx) {
                int32_t waveExpertRowBegin = waveIdx * dispatchWaveRowCount;
                int32_t waveExpertRowEnd = waveExpertRowBegin + dispatchWaveRowCount;
                int32_t overlapRowBegin =
                    coreExpertRowBegin > waveExpertRowBegin ? coreExpertRowBegin : waveExpertRowBegin;
                int32_t overlapRowEnd = coreExpertRowEnd < waveExpertRowEnd ? coreExpertRowEnd : waveExpertRowEnd;
                AtomicAdd(dispatchWaveReadyCount + static_cast<int64_t>(waveIdx) * INT_CACHELINE,
                          overlapRowEnd - overlapRowBegin);
            }
        }
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TOKEN_DISPATCH_H
