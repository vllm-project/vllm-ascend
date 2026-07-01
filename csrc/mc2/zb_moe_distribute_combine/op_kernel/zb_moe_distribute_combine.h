/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ZbMoeDistributeCombine operator kernel function header file, for a3
 * Author: Hu Tianlun
 * Create: 2026-01-22
 * Note:
 * History: 2026-01-22 create ZbMoeDistributeCombine operator kernel function header file, for a3
 */
#ifndef ZB_MOE_DISTRIBUTE_COMBINE_H
#define ZB_MOE_DISTRIBUTE_COMBINE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "zb_moe_distribute_v2_base.h"
#include "zb_moe_distribute_combine_tiling.h"
#include "shmem_sync_flag.h"
#include "shmem.h"
#include <type_traits>

using namespace AscendC;
#define SHMEM_PUT_BY_DTYPE(dtype, ...)                            \
    do {                                                          \
        if constexpr (std::is_same_v<dtype, half>) {              \
            aclshmem_half_put_nbi(__VA_ARGS__);                  \
        } else if constexpr (std::is_same_v<dtype, bfloat16_t>) { \
            aclshmem_bfloat16_put_nbi(__VA_ARGS__);              \
        } else if constexpr (std::is_same_v<dtype, float>) {      \
            aclshmem_float_put_nbi(__VA_ARGS__);                 \
        }                                                         \
    } while (0)

#define SHMEM_GET_BY_DTYPE(dtype, ...)                            \
    do {                                                          \
        if constexpr (std::is_same_v<dtype, half>) {              \
            aclshmem_half_get_nbi(__VA_ARGS__);                  \
        } else if constexpr (std::is_same_v<dtype, bfloat16_t>) { \
            aclshmem_bfloat16_get_nbi(__VA_ARGS__);              \
        }                                                         \
    } while (0)

namespace MoeDistributeCombineV2Impl {
using namespace MoeDistributeV2Base;
constexpr uint8_t BUFFER_NUM = 2;
constexpr uint32_t ALIGNED_LEN = 256U;
constexpr uint32_t GM2GM_COPY_BLOCK_SIZE = 64U * 1024U;
constexpr uint64_t COMBINE_SYNC_FLAG_BASE_OFFSET = ShmemSyncFlagImpl::FLAG_AREA_BASE;
constexpr uint64_t COMBINE_SYNC_STATE_OFFSET = 32U;

#define TemplateMC2TypeClass \
    typename ExpandXType, typename XType, typename ExpandIdxType, bool IsNeedReduceScatter, bool IsInt8Quant
#define TemplateMC2TypeFunc ExpandXType, XType, ExpandIdxType, IsNeedReduceScatter, IsInt8Quant

template <TemplateMC2TypeClass>
class ZbMoeDistributeCombine
{
public:
    __aicore__ inline ZbMoeDistributeCombine(){};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount,
                                GM_ADDR tpSendCount, GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX,
                                GM_ADDR elasticInfo, GM_ADDR oriX, GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2,
                                GM_ADDR constExpertV, GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const ZbMoeDistributeCombineTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyValidExpandXToShmem();
    __aicore__ inline void InputToDstOutput();
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum);
    __aicore__ inline GM_ADDR GetSyncFlagAddrByRankId(uint32_t rankId);
    __aicore__ inline void SetSyncFlag();
    __aicore__ inline void WaitSyncFlag();
    __aicore__ inline void InitInputAndOutput(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx,
                                              GM_ADDR epSendCount, GM_ADDR expertScales, GM_ADDR XOut);
    __aicore__ inline void InitAttrs(const ZbMoeDistributeCombineTilingData *tilingData);
    __aicore__ inline void InitTilingAttrs(const ZbMoeDistributeCombineTilingData *tilingData);

    TPipe *tpipe_{nullptr};
    GlobalTensor<int32_t> expertIdsGM_;
    GlobalTensor<int32_t> expandIdxGM_;
    GlobalTensor<ExpandIdxType> epSendCountGM_;
    GlobalTensor<float> expertScalesGT_;
    GlobalTensor<XType> srcWinGMTensor;
    GlobalTensor<XType> XOutGT_;
    GM_ADDR expandXGM_;
    GM_ADDR oriXGM_{0};

    LocalTensor<float> tokenReduceTensor_;
    LocalTensor<float> weightedSumTensor_;
    LocalTensor<float> sumTokenTensor_;

    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epRankId_{0};
    uint32_t aivId_{0};
    uint32_t moeExpertPerRankNum_{0};
    uint32_t moeExpertNum_{0};
    uint32_t hExpandXTypeSize_{0};
    uint32_t hFloatAlign32Size_{0};
    uint32_t hFloatAlign256Size_{0};
    uint32_t hExpandXAlign32Size_{0};
    uint32_t validTokenNum_{0};
    bool needCopyExpandX_{false};

    TQue<QuePosition::VECIN, 1> moeSumQueue_;

    TBuf<> expertScalesBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> tokenSumBuf_;
    TBuf<> weightedSumBuf_;
    TBuf<> xOutBuf_;
    TBuf<> copyInBuf_;
    TBuf<> indexCountsBuf_;
    TBuf<> statusBuf_;
    TBuf<> waitStatusBuf_;
    TBuf<> gatherMaskOutBuf_;
    TBuf<> statusSumBuf_;

    LocalTensor<float> expertScalesLocal_;
    GM_ADDR metaDataGvaGM_{0};
    uint32_t rankNumPerBlock_{0};
    uint32_t curBlockStartRankId_{0};
    uint32_t curBlockEndRankId_{0};
};

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeCombine<TemplateMC2TypeFunc>::InitInputAndOutput(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR expertScales, GM_ADDR XOut)
{
    expandXGM_ = expandX;
    expertIdsGM_.SetGlobalBuffer((__gm__ ExpandIdxType *)expertIds);
    expandIdxGM_.SetGlobalBuffer((__gm__ int32_t *)expandIdx);
    epSendCountGM_.SetGlobalBuffer((__gm__ int32_t *)epSendCount);
    expertScalesGT_.SetGlobalBuffer((__gm__ float *)expertScales);
    XOutGT_.SetGlobalBuffer((__gm__ XType *)XOut);
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeCombine<TemplateMC2TypeFunc>::InitTilingAttrs(
    const ZbMoeDistributeCombineTilingData *tilingData)
{
    axisBS_ = tilingData->moeDistributeCombineV2Info.bs;
    axisH_ = tilingData->moeDistributeCombineV2Info.h;
    axisK_ = tilingData->moeDistributeCombineV2Info.k;
    aivNum_ = tilingData->moeDistributeCombineV2Info.aivNum;
    epRankId_ = tilingData->moeDistributeCombineV2Info.epRankId;
    epWorldSize_ = tilingData->moeDistributeCombineV2Info.epWorldSize;
    moeExpertPerRankNum_ = tilingData->moeDistributeCombineV2Info.moeExpertPerRankNum;
    moeExpertNum_ = tilingData->moeDistributeCombineV2Info.moeExpertNum;
}

template <TemplateMC2TypeClass>
__aicore__ inline void
ZbMoeDistributeCombine<TemplateMC2TypeFunc>::InitAttrs(
    const ZbMoeDistributeCombineTilingData *tilingData)
{
    InitTilingAttrs(tilingData);
    uint32_t hFloatSize = axisH_ * static_cast<uint32_t>(sizeof(float));
    hFloatAlign32Size_ = Ceil(hFloatSize, UB_ALIGN) * UB_ALIGN;
    hFloatAlign256Size_ = Ceil(hFloatSize, ALIGNED_LEN) * ALIGNED_LEN;
    hExpandXTypeSize_ = axisH_ * sizeof(ExpandXType);
    hExpandXAlign32Size_ = Ceil(hExpandXTypeSize_, UB_ALIGN) * UB_ALIGN;
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeCombine<TemplateMC2TypeFunc>::Init(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR tpSendCount,
    GM_ADDR expertScales, GM_ADDR xActiveMask, GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX,
    GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, GM_ADDR XOut, GM_ADDR workspaceGM,
    TPipe *pipe, const ZbMoeDistributeCombineTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();

    InitInputAndOutput(expandX, expertIds, expandIdx, epSendCount, expertScales, XOut);
    InitAttrs(tilingData);
    if (oriX != 0) {
        oriXGM_ = oriX;
        uint32_t lastLocalExpertId = epRankId_ * moeExpertPerRankNum_ + moeExpertPerRankNum_ - 1U;
        uint32_t lastPrefixIndex = lastLocalExpertId * epWorldSize_ + epWorldSize_ - 1U;
        int32_t count = epSendCountGM_.GetValue(lastPrefixIndex);
        validTokenNum_ = (count > 0) ? static_cast<uint32_t>(count) : 0U;
    }
    needCopyExpandX_ = (oriXGM_ != 0) && (validTokenNum_ > 0U);
    metaDataGvaGM_ = (GM_ADDR)tilingData->moeDistributeCombineV2Info.shmemptr;
    SplitToCore(epWorldSize_, aivNum_, curBlockStartRankId_, curBlockEndRankId_, rankNumPerBlock_);

    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeCombine<TemplateMC2TypeFunc>::SplitToCore(uint32_t curSendCnt,
                                                                                 uint32_t curUseAivNum,
                                                                                 uint32_t &startTokenId,
                                                                                 uint32_t &endTokenId,
                                                                                 uint32_t &sendTokenNum)
{
    // Split the local batch tokens evenly across AIVs.
    sendTokenNum = curSendCnt / curUseAivNum;
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum;
    startTokenId = sendTokenNum * aivId_;
    if (aivId_ < remainderTokenNum) {
        sendTokenNum += 1;
        startTokenId += aivId_;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateMC2TypeClass>
__aicore__ inline GM_ADDR
ZbMoeDistributeCombine<TemplateMC2TypeFunc>::GetSyncFlagAddrByRankId(uint32_t rankId)
{
    auto metaPtr = reinterpret_cast<__gm__ uint8_t *>(aclshmem_ptr(metaDataGvaGM_, rankId));
    return (GM_ADDR)(metaPtr) + COMBINE_SYNC_FLAG_BASE_OFFSET;
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeCombine<TemplateMC2TypeFunc>::SetSyncFlag()
{
    if (rankNumPerBlock_ == 0U) {
        SyncAll<true>();
        return;
    }

    uint32_t statusCntAlign = Ceil(rankNumPerBlock_, 8) * 8;
    tpipe_->InitBuffer(statusBuf_, statusCntAlign * UB_ALIGN);
    LocalTensor<int32_t> statusTensor = statusBuf_.Get<int32_t>();
    Duplicate<int32_t>(statusTensor, 0, rankNumPerBlock_ * 8);
    uint64_t mask[2] = {0x101010101010101, 0};
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor, 0x3F800000, mask, statusCntAlign / 8, 1, 8);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();

    GlobalTensor<int32_t> gmRemoteStatusGT;
    for (uint32_t targetRankId = curBlockStartRankId_; targetRankId < curBlockEndRankId_; ++targetRankId) {
        auto ptr = GetSyncFlagAddrByRankId(targetRankId) + epRankId_ * COMBINE_SYNC_STATE_OFFSET;
        gmRemoteStatusGT.SetGlobalBuffer((__gm__ int32_t *)(ptr));
        DataCopy<int32_t>(gmRemoteStatusGT,
                          statusTensor[(targetRankId - curBlockStartRankId_) * 8],
                          8UL);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeCombine<TemplateMC2TypeFunc>::WaitSyncFlag()
{
    if (rankNumPerBlock_ == 0U) {
        SyncAll<true>();
        return;
    }

    uint32_t waitStatusBufSize =
        (((rankNumPerBlock_ * UB_ALIGN) > 256U) ? (rankNumPerBlock_ * UB_ALIGN) : 256U);
    tpipe_->InitBuffer(waitStatusBuf_, waitStatusBufSize);
    uint32_t maskAlign = Ceil(epWorldSize_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(gatherMaskOutBuf_, maskAlign);
    tpipe_->InitBuffer(statusSumBuf_, UB_ALIGN);

    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
    LocalTensor<float> statusSumOutTensor = statusSumBuf_.Get<float>(UB_ALIGN);
    LocalTensor<float> statusFp32Tensor = waitStatusBuf_.Get<float>();
    GlobalTensor<float> statusFp32TensorGT;
    auto ptr = GetSyncFlagAddrByRankId(epRankId_);
    statusFp32TensorGT.SetGlobalBuffer((__gm__ float *)(ptr));

    uint32_t mask = 1U;
    float compareTarget = static_cast<float>(1.0) * rankNumPerBlock_;
    float sumOfFlag = static_cast<float>(-1.0);
    float minTarget = compareTarget - static_cast<float>(0.5);
    float maxTarget = compareTarget + static_cast<float>(0.5);
    DataCopyParams intriParams{static_cast<uint16_t>(rankNumPerBlock_), 1, 0, 0};

    SyncFunc<AscendC::HardEvent::S_V>();
    while ((sumOfFlag < minTarget) || (sumOfFlag > maxTarget)) {
        DataCopy(statusFp32Tensor,
                 statusFp32TensorGT[curBlockStartRankId_ * COMBINE_SYNC_STATE_OFFSET / sizeof(float)],
                 intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(statusSumOutTensor, statusFp32Tensor, gatherMaskOutTensor, mask, rankNumPerBlock_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }

    SyncFunc<AscendC::HardEvent::MTE3_S>();
    DataCopyParams intriOutParams{static_cast<uint16_t>(rankNumPerBlock_), 1, 0, 0};
    uint64_t duplicateMask[2] = {0x101010101010101, 0};
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(rankNumPerBlock_, 8), 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(statusFp32TensorGT[curBlockStartRankId_ * COMBINE_SYNC_STATE_OFFSET / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(),
             intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    SyncAll<true>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeCombine<TemplateMC2TypeFunc>::CopyValidExpandXToShmem()
{
    if (!needCopyExpandX_) {
        return;
    }

    uint32_t startTokenId, endTokenId, sendTokenNum;
    SplitToCore(validTokenNum_, aivNum_, startTokenId, endTokenId, sendTokenNum);
    if (sendTokenNum == 0U) {
        return;
    }

    constexpr uint32_t COPY_BUFFER_NUM = 2U;
    tpipe_->InitBuffer(copyInBuf_, COPY_BUFFER_NUM * GM2GM_COPY_BLOCK_SIZE);
    LocalTensor<uint8_t> copyLocal = copyInBuf_.Get<uint8_t>();
    GlobalTensor<uint8_t> oriXByteGT;
    GlobalTensor<uint8_t> expandXByteGT;
    oriXByteGT.SetGlobalBuffer((__gm__ uint8_t *)oriXGM_);
    expandXByteGT.SetGlobalBuffer((__gm__ uint8_t *)expandXGM_);

    uint64_t srcOffset = static_cast<uint64_t>(startTokenId) * hExpandXTypeSize_;
    uint64_t dstOffset = srcOffset;
    uint64_t dataSizeRemain = static_cast<uint64_t>(sendTokenNum) * hExpandXTypeSize_;
    DataCopyPadExtParams<uint8_t> copyPadParams{false, 0U, 0U, 0U};

    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint64_t i = 0; dataSizeRemain > 0U; ++i) {
        uint32_t copySize = dataSizeRemain > GM2GM_COPY_BLOCK_SIZE ?
            GM2GM_COPY_BLOCK_SIZE : static_cast<uint32_t>(dataSizeRemain);
        uint32_t bufferOffset = (i & 1U) * GM2GM_COPY_BLOCK_SIZE;
        event_t eventId = (i & 1U) ? EVENT_ID0 : EVENT_ID1;
        DataCopyExtParams copyParams{1U, copySize, 0U, 0U, 0U};

        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        DataCopyPad(copyLocal[bufferOffset], oriXByteGT[srcOffset], copyParams, copyPadParams);
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(eventId);
        AscendC::WaitFlag<HardEvent::MTE2_MTE3>(eventId);
        DataCopyPad(expandXByteGT[dstOffset], copyLocal[bufferOffset], copyParams);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(eventId);

        srcOffset += copySize;
        dstOffset += copySize;
        dataSizeRemain -= copySize;
    }
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeCombine<TemplateMC2TypeFunc>::InputToDstOutput()
{
    if (axisBS_ == 0U) {
        return;
    }
    uint32_t startTokenId, endTokenId, sendTokenNum;
    SplitToCore(axisBS_, aivNum_, startTokenId, endTokenId, sendTokenNum);
    if (sendTokenNum == 0U) {
        return;
    }

    // tpipe_->Reset();
    uint32_t scaleAlignLen = Ceil(sendTokenNum * axisK_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(expertScalesBuf_, scaleAlignLen);
    tpipe_->InitBuffer(indexCountsBuf_, scaleAlignLen);

    DataCopyExtParams bskParams{1U, static_cast<uint32_t>(sendTokenNum * axisK_ * sizeof(uint32_t)), 0U, 0U, 0U};
    const DataCopyPadExtParams<float> copyPadFloatParams{false, 0U, 0U, 0U};
    expertScalesLocal_ = expertScalesBuf_.Get<float>();
    DataCopyPad(expertScalesLocal_, expertScalesGT_[startTokenId * axisK_], bskParams, copyPadFloatParams);

    LocalTensor<int32_t> expandIdxLocal = indexCountsBuf_.Get<int32_t>();
    DataCopyPadExtParams<int32_t> copyPadint32Params{false, 0U, 0U, 0U};
    DataCopyPad(expandIdxLocal, expandIdxGM_[startTokenId * axisK_], bskParams, copyPadint32Params);

    uint32_t expertIdsAlignLen = Ceil(axisK_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsAlignLen);
    DataCopyExtParams kParams{1U, static_cast<uint32_t>(axisK_ * sizeof(ExpandIdxType)), 0U, 0U, 0U};
    DataCopyPadExtParams<ExpandIdxType> copyPadParams{false, 0U, 0U, 0U};
    LocalTensor<int32_t> expertIdsTensor = expertIdsBuf_.Get<int32_t>();

    DataCopyExtParams xOutCopyParams{1U, static_cast<uint32_t>(hExpandXTypeSize_), 0U, 0U, 0U};
    DataCopyPadExtParams<ExpandXType> copyPadExtParams{false, 0U, 0U, 0U};

    tpipe_->InitBuffer(xOutBuf_, hExpandXAlign32Size_);
    tpipe_->InitBuffer(moeSumQueue_, BUFFER_NUM, hExpandXAlign32Size_);
    tpipe_->InitBuffer(tokenSumBuf_, hFloatAlign32Size_);
    tpipe_->InitBuffer(weightedSumBuf_, hFloatAlign256Size_);
    tpipe_->InitBuffer(sumFloatBuf_, hFloatAlign32Size_);
    tokenReduceTensor_ = tokenSumBuf_.Get<float>();
    weightedSumTensor_ = weightedSumBuf_.Get<float>();
    sumTokenTensor_ = sumFloatBuf_.Get<float>();

    for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; tokenIndex++) {
        uint32_t localTokenId = tokenIndex - startTokenId;
        DataCopyPad(expertIdsTensor, expertIdsGM_[tokenIndex * axisK_], kParams, copyPadParams);
        PipeBarrier<PIPE_ALL>();

        // Accumulate the weighted top-k expert outputs for one original token.
        Duplicate(sumTokenTensor_, static_cast<float>(0), axisH_);

        for (uint32_t topkId = 0U; topkId < axisK_; topkId++) {
            int32_t dstExpertId = expertIdsTensor.GetValue(topkId);
            int32_t srcRankId = dstExpertId / moeExpertPerRankNum_;
            uint32_t offsetIdx = dstExpertId * epWorldSize_ + epRankId_;
            uint32_t col = offsetIdx % moeExpertNum_;
            int32_t remoteBase = (col == 0) ? 0 : epSendCountGM_(offsetIdx - 1);
            int32_t remoteOffset = expandIdxLocal(localTokenId * axisK_ + topkId);
            float scale = expertScalesLocal_.GetValue(localTokenId * axisK_ + topkId);

            auto srcPtr = reinterpret_cast<__gm__ uint8_t *>(aclshmem_ptr(expandXGM_, srcRankId));
            srcWinGMTensor.SetGlobalBuffer(
                (__gm__ ExpandXType *)(srcPtr + hExpandXTypeSize_ * (remoteBase + remoteOffset)));

            LocalTensor<ExpandXType> tmpToken = moeSumQueue_.AllocTensor<ExpandXType>();
            DataCopyPad(tmpToken, srcWinGMTensor, xOutCopyParams, copyPadExtParams);
            moeSumQueue_.EnQue(tmpToken);
            tmpToken = moeSumQueue_.DeQue<ExpandXType>();
            Cast(tokenReduceTensor_, tmpToken, AscendC::RoundMode::CAST_NONE, axisH_);
            PipeBarrier<PIPE_V>();
            AscendC::Muls(weightedSumTensor_, tokenReduceTensor_, scale, axisH_);
            PipeBarrier<PIPE_V>();
            AscendC::Add(sumTokenTensor_, sumTokenTensor_, weightedSumTensor_, axisH_);
            moeSumQueue_.FreeTensor<ExpandXType>(tmpToken);
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<ExpandXType> xOutLocal = xOutBuf_.Get<ExpandXType>();
        Cast(xOutLocal, sumTokenTensor_, AscendC::RoundMode::CAST_RINT, axisH_);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopyPad(XOutGT_[tokenIndex * axisH_], xOutLocal, xOutCopyParams);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeCombine<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        if (needCopyExpandX_) {
            // Legacy path: copy local ori_x into SHMEM expand_x, then cross-rank
            // barrier so remote readers see the staged data.
            CopyValidExpandXToShmem();
            SyncAll<true>();
            tpipe_->Reset();
        }
        SetSyncFlag();
        WaitSyncFlag();
        SyncAll<true>();
        InputToDstOutput();
    }
}

}  // namespace ZbMoeDistributeCombineImpl
#endif  // ZB_MOE_DISTRIBUTE_COMBINE_H
