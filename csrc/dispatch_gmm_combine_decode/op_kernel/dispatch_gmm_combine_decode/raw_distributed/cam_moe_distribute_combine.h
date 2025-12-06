/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef CAM_MOE_DISTRIBUTE_COMBINE_H
#define CAM_MOE_DISTRIBUTE_COMBINE_H
#define OPT_RANK_OFFSET 512

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../dispatch_gmm_combine_decode_base.h"
#include "../../dispatch_gmm_combine_decode_tiling.h"

namespace MoeDistributeCombineImpl {
constexpr uint8_t BUFFER_NUM = 2;  // multi-buf
constexpr uint32_t STATE_OFFSET = 512;
constexpr uint32_t STATE_SIZE = 1024 * 1024;  // 1M
constexpr uint32_t RANK_SIZE_ON_WIN_512 = 512 * 1024;
constexpr uint32_t RANK_SIZE_ON_WIN_256 = 256 * 1024;
constexpr uint32_t TP_RANK_SIZE_ON_WIN = 0;
constexpr uint32_t UB_ALIGN = 32;
constexpr uint32_t SELF_STATE_OFFSET = 256 * 1024;
constexpr uint8_t EP_DOMAIN = 0;
constexpr uint8_t TP_DOMAIN = 1;
constexpr uint64_t WIN_STATE_OFFSET = 512 * 1024;
constexpr uint64_t STATE_WIN_OFFSET = 900 * 1024;
constexpr uint16_t SEND_SYNC_EVENT_ID = 9;
constexpr uint16_t RECV_SYNC_EVENT_ID = 10;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

using namespace AscendC;

struct CombineCalcInfo {
    uint64_t expertPerSizeOnWin_;
    uint32_t epRankId_;
    uint32_t epWorldSize_;
    uint32_t moeExpertPerRankNum_;
    uint32_t sharedExpertRankNum_;
    uint32_t axisH_;
    uint32_t moeSendNum_;
    bool isShardExpert_;
    GM_ADDR epSendCount_;
    __gm__ HcclOpResParam *epWinContext_;
    uint64_t winDataSizeOffset_;
};

template <TemplateMC2TypeClass>
class CamMoeDistributeCombine
{
public:
    __aicore__ inline CamMoeDistributeCombine(){};
    __aicore__ inline void Init(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount,
                                GM_ADDR tpSendCount, GM_ADDR scales, GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const DispatchGmmCombineDecodeTilingData *tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void AllToAllSend();
    __aicore__ inline void ReducePermute();

    __aicore__ inline CombineCalcInfo &GetCalcInfo()
    {
        return calcInfo_;
    }

    __aicore__ inline void TPipeSet(AscendC::TPipe *pipe)
    {
        tpipe_ = pipe;
    }

private:
    __aicore__ inline void InitStatusTargetSum();
    __aicore__ inline void AlltoAllBuffInit();
    __aicore__ inline void ReduceScatterTrans();
    __aicore__ inline void SetWaitTpStatusAndDisPatch();
    __aicore__ inline void CustomAdd(LocalTensor<ExpandXType> &dst, LocalTensor<ExpandXType> &src0,
                                     LocalTensor<ExpandXType> &src1, uint32_t dataCnt);
    __aicore__ inline void ExpertAlltoAllDispatchInnerCopyAdd(uint32_t tokenNumLoop, uint32_t srcStartTokenIdx,
                                                              uint32_t ep, uint32_t expertIdx);
    __aicore__ inline void ExpertAlltoAllDispatchCopyAdd();
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void BuffInit();
    __aicore__ inline void SplitCoreCal();
    __aicore__ inline void SetStatus();
    __aicore__ inline void WaitDispatch();
    __aicore__ GM_ADDR GetWinAddrByRankId(const int32_t rankId, const uint8_t domain, const uint8_t expertLocalId = 0U)
    {
        if (domain == EP_DOMAIN) {
            return (GM_ADDR)((epRankId_ == rankId)
                                 ? epWinContext_->localWindowsIn
                                 : ((HcclRankRelationResV2 *)(epWinContext_->remoteRes[rankId].nextDevicePtr))
                                       ->windowsIn) +
                   winDataSizeOffset_ + expertLocalId * expertPerSizeOnWin_ + rankId * OPT_RANK_OFFSET;
        } else {
            return (GM_ADDR)((tpRankId_ == rankId)
                                 ? tpWinContext_->localWindowsIn
                                 : ((HcclRankRelationResV2 *)(tpWinContext_->remoteRes[rankId].nextDevicePtr))
                                       ->windowsIn) +
                   winDataSizeOffset_ + rankId * OPT_RANK_OFFSET;
        }
    }

    __aicore__ GM_ADDR GetWinStateAddrByRankId(const int32_t rankId, const uint8_t domain)
    {
        if (domain == EP_DOMAIN) {
            return (GM_ADDR)((epRankId_ == rankId)
                                 ? epWinContext_->localWindowsExp
                                 : ((HcclRankRelationResV2 *)(epWinContext_->remoteRes[rankId].nextDevicePtr))
                                       ->windowsExp) +
                   dataState_ * WIN_STATE_OFFSET;
        } else {
            return (GM_ADDR)((tpRankId_ == rankId)
                                 ? tpWinContext_->localWindowsExp
                                 : ((HcclRankRelationResV2 *)(tpWinContext_->remoteRes[rankId].nextDevicePtr))
                                       ->windowsExp) +
                   dataState_ * WIN_STATE_OFFSET;
        }
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return (x < y) ? x : y;
    }

    __aicore__ static void DoCombineRecv(void *ptr)
    {
        auto *combiner = (CamMoeDistributeCombine<TemplateMC2TypeFunc> *)ptr;
        combiner->ReducePermute();
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<ExpandXType> expandXGM_;
    GlobalTensor<ExpandIdxType> expertIdsGM_;
    GlobalTensor<ExpandIdxType> expandIdxGM_;
    GlobalTensor<ExpandIdxType> epSendCountGM_;
    GlobalTensor<ExpandIdxType> tpSendCountGM_;
    GlobalTensor<float> expandScalesGM_;
    GlobalTensor<ExpandXType> expandOutGlobal_;
    GlobalTensor<ExpandXType> rankWindow_;
    GlobalTensor<int32_t> rankStates_;
    GlobalTensor<float> epStatusSpaceGlobalTensor_;
    GlobalTensor<float> tpStatusSpaceGlobalTensor_;
    GlobalTensor<ExpandXType> tpRankWindow_;
    GlobalTensor<ExpandXType> rowTmpGlobal_;
    GM_ADDR workspaceGM_;
    GM_ADDR epWindowGM_;
    GM_ADDR epStatusSpaceGm_;
    GM_ADDR tpWindowGM_;
    GM_ADDR tpStatusSpaceGm_;
    GM_ADDR stateGM_;

    LocalTensor<ExpandXType> winTpSendCountTensor_;
    LocalTensor<ExpandXType> gmTpSendCountTensor_;
    LocalTensor<ExpandXType> outTensor_;
    LocalTensor<float> winTpSendCountFloatTensor_;
    LocalTensor<float> gmTpSendCountFloatTensor_;
    LocalTensor<ExpandIdxType> epSendCountLocal_;

    CombineCalcInfo calcInfo_;
    uint32_t axisBS_{0};
    uint32_t axisMaxBs_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t tpWorldSize_{0};
    uint32_t epRankId_{0};
    uint32_t tpRankId_{0};
    uint32_t coreIdx_{0};  // aiv id
    uint32_t sharedExpertRankNum_{0};
    uint32_t moeExpertNum_{0};
    uint32_t moeExpertPerRankNum_{0};
    uint32_t moeSendNum_{0};  // moeExpertPerRankNum_ * epWorldSize_
    uint32_t tpScatterNum_{0};
    uint32_t firstTpTokenEndIdx_{0};
    uint32_t firstTpTokenEndOffset_{0};
    uint32_t endTok_{0};
    __gm__ HcclOpResParam *epWinContext_{nullptr};
    __gm__ HcclOpResParam *tpWinContext_{nullptr};
    uint32_t epDataOffsetOnWin_{0};
    uint32_t tpDataOffsetOnWin_{0};
    uint32_t epStateOffsetOnWin_{0};
    uint32_t tpStateOffsetOnWin_{0};
    uint32_t axisHFloatSize_{0};
    uint32_t axisHExpandXTypeSize_{0};
    uint32_t bsKNum_{0};
    uint32_t startRankId_{0};
    uint32_t endRankId_{0};
    uint32_t sendRankNum_{0};
    uint32_t ubSize_{0};
    uint32_t dataState_{0};
    uint32_t stateOffset_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t totalWinSize_{0};
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> moeQueue_;
    TQue<QuePosition::VECIN, 1> moeSumQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> gmTpSendCountQueue_;
    TQue<QuePosition::VECIN, 1> gmTpSendCountInQueue_;
    TQue<QuePosition::VECIN, 1> winTpSendCountInQueue_;
    TQue<QuePosition::VECOUT, 1> xOutQueue_;
    TBuf<> readStateBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> expandScalesBuf_;
    TBuf<> rowTmpFloatBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> mulBuf_;
    TBuf<> sendCountBuf_;
    TBuf<> indexCountsBuf_;
    TBuf<> winTpSendCountFloatBuf_;
    TBuf<> gmTpSendCountFloatBuf_;
    TBuf<> tokenBuf_;
    TBuf<> statusBuf_;
    TBuf<> gatherMaskOutBuf_;  // gather mask output buf
    TBuf<> gatherTmpBuf_;
    TBuf<> statusSumOutBuf_;
    float sumTarget_{0.0};
    int32_t epStateValue_;
    bool isShardExpert_{false};
};

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeDistributeCombine<TemplateMC2TypeFunc>::Init(
    GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx, GM_ADDR epSendCount, GM_ADDR tpSendCount, GM_ADDR scales,
    GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe, const DispatchGmmCombineDecodeTilingData *tilingData)
{
    tpipe_ = pipe;
    coreIdx_ = GetBlockIdx();
    epRankId_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.epRankId;
    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    epWinContext_ = (__gm__ HcclOpResParam *)contextGM0;
    GlobalTensor<int32_t> selfDataStatusTensor;
    GM_ADDR statusDataSpaceGm = (GM_ADDR)epWinContext_->localWindowsExp;
    selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm + STATE_WIN_OFFSET));
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        selfDataStatusTensor[coreIdx_ * UB_ALIGN]);
    __asm__ __volatile__("");
    dataState_ = selfDataStatusTensor(coreIdx_ * UB_ALIGN);
    if (dataState_ == 0) {
        selfDataStatusTensor(coreIdx_ * UB_ALIGN) = 1;
    } else {
        selfDataStatusTensor(coreIdx_ * UB_ALIGN) = 0;
    }
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        selfDataStatusTensor[coreIdx_ * UB_ALIGN]);
    __asm__ __volatile__("");
    pipe_barrier(PIPE_ALL);

    workspaceGM_ = workspaceGM;
    expandXGM_.SetGlobalBuffer((__gm__ ExpandXType *)expandX);
    expertIdsGM_.SetGlobalBuffer((__gm__ ExpandIdxType *)expertIds);
    expandIdxGM_.SetGlobalBuffer((__gm__ ExpandIdxType *)expandIdx);
    epSendCountGM_.SetGlobalBuffer((__gm__ int32_t *)epSendCount);
    expandScalesGM_.SetGlobalBuffer((__gm__ float *)scales);
    expandOutGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)XOut);
    axisBS_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.bs;
    axisH_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.h;
    axisK_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.k;
    if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
        aivNum_ = get_block_num();
    } else {
        aivNum_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.aivNum;
    }
    ubSize_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.totalUbSize;
    sharedExpertRankNum_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.sharedExpertRankNum;
    moeExpertNum_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNum;
    moeExpertPerRankNum_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNumPerRank;
    epWorldSize_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.epRankSize;
    axisMaxBs_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.globalBs / epWorldSize_;
    moeSendNum_ = epWorldSize_ * moeExpertPerRankNum_;
    tpWorldSize_ = 1;
    tpRankId_ = 0;
    totalWinSize_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.totalWinSize;
    stateOffset_ = (moeSendNum_ > 512) ? (STATE_OFFSET / 2) : STATE_OFFSET;
    expertPerSizeOnWin_ =
        static_cast<uint64_t>(axisMaxBs_) * static_cast<uint64_t>(axisH_) * static_cast<uint64_t>(sizeof(ExpandXType));
    winDataSizeOffset_ = static_cast<uint64_t>(dataState_) * static_cast<uint64_t>(moeSendNum_) * expertPerSizeOnWin_;
    epWindowGM_ = GetWinAddrByRankId(epRankId_, EP_DOMAIN);
    epStatusSpaceGm_ = GetWinStateAddrByRankId(epRankId_, EP_DOMAIN);
    epStatusSpaceGlobalTensor_.SetGlobalBuffer((__gm__ float *)epStatusSpaceGm_);
    epDataOffsetOnWin_ = epRankId_ * moeExpertPerRankNum_ * static_cast<uint32_t>(expertPerSizeOnWin_);
    epStateOffsetOnWin_ = epRankId_ * stateOffset_;
    isShardExpert_ = (epRankId_ < sharedExpertRankNum_);
    axisHFloatSize_ = axisH_ * sizeof(float);
    axisHExpandXTypeSize_ = axisH_ * sizeof(ExpandXType);
    bsKNum_ = axisBS_ * axisK_;

    if constexpr (IsNeedReduceScatter) {
        tpSendCountGM_.SetGlobalBuffer((__gm__ int32_t *)tpSendCount);
        tpWindowGM_ = GetWinAddrByRankId(tpRankId_, TP_DOMAIN);
        tpStatusSpaceGm_ = GetWinStateAddrByRankId(tpRankId_, TP_DOMAIN);
        tpStatusSpaceGlobalTensor_.SetGlobalBuffer((__gm__ float *)tpStatusSpaceGm_);
        tpDataOffsetOnWin_ = tpRankId_ * TP_RANK_SIZE_ON_WIN;
        tpStateOffsetOnWin_ = tpRankId_ * stateOffset_;
        uint32_t tpScatterRankWinOffset = (tpRankId_ == 0) ? TP_RANK_SIZE_ON_WIN : 0;
        GM_ADDR rankGM = tpWindowGM_ + tpScatterRankWinOffset;
        tpRankWindow_.SetGlobalBuffer((__gm__ ExpandXType *)rankGM);
    }

    InitStatusTargetSum();
    if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
        coreIdx_ = get_block_idx();
    }
    SplitCoreCal();

    calcInfo_.epRankId_ = epRankId_;
    calcInfo_.epWorldSize_ = epWorldSize_;
    calcInfo_.expertPerSizeOnWin_ = expertPerSizeOnWin_;
    calcInfo_.moeExpertPerRankNum_ = moeExpertPerRankNum_;
    calcInfo_.sharedExpertRankNum_ = sharedExpertRankNum_;
    calcInfo_.axisH_ = axisH_;
    calcInfo_.moeSendNum_ = moeSendNum_;
    calcInfo_.isShardExpert_ = isShardExpert_;
    calcInfo_.epSendCount_ = epSendCount;
    calcInfo_.epWinContext_ = epWinContext_;
    calcInfo_.winDataSizeOffset_ = winDataSizeOffset_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeDistributeCombine<TemplateMC2TypeFunc>::InitStatusTargetSum()
{
    // ep state
    GlobalTensor<int32_t> selfStatusTensor;
    selfStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(epStatusSpaceGm_ + SELF_STATE_OFFSET));
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        selfStatusTensor[coreIdx_ * UB_ALIGN]);
    __asm__ __volatile__("");
    int32_t state = selfStatusTensor(coreIdx_ * UB_ALIGN);
    if (state == 0) {
        sumTarget_ = static_cast<float>(1.0);
        selfStatusTensor(coreIdx_ * UB_ALIGN) = 0x3F800000;  // 1.0f
        epStateValue_ = 0x3F800000;                          // 1.0f
    } else {
        sumTarget_ = static_cast<float>(0.0);
        selfStatusTensor(coreIdx_ * UB_ALIGN) = 0;
        epStateValue_ = 0;
    }
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        selfStatusTensor[coreIdx_ * UB_ALIGN]);
    __asm__ __volatile__("");
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeDistributeCombine<TemplateMC2TypeFunc>::BuffInit()
{
    tpipe_->Reset();
    tpipe_->InitBuffer(readStateBuf_, UB_ALIGN);
    uint32_t sendNumAlign = Ceil(moeSendNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(sendCountBuf_, sendNumAlign);
    if constexpr (IsNeedReduceScatter) {
        tpipe_->InitBuffer(winTpSendCountInQueue_, BUFFER_NUM, axisHExpandXTypeSize_);
        tpipe_->InitBuffer(gmTpSendCountInQueue_, BUFFER_NUM, axisHExpandXTypeSize_);
        tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, axisHExpandXTypeSize_);
        if constexpr (AscendC::IsSameType<ExpandXType, bfloat16_t>::value) {
            tpipe_->InitBuffer(winTpSendCountFloatBuf_, axisHFloatSize_);
            tpipe_->InitBuffer(gmTpSendCountFloatBuf_, axisHFloatSize_);
            winTpSendCountFloatTensor_ = winTpSendCountFloatBuf_.Get<float>();
            gmTpSendCountFloatTensor_ = gmTpSendCountFloatBuf_.Get<float>();
        }
    } else {
        tpipe_->InitBuffer(gmTpSendCountQueue_, BUFFER_NUM, axisHExpandXTypeSize_);
    }
    epSendCountLocal_ = sendCountBuf_.Get<int32_t>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeDistributeCombine<TemplateMC2TypeFunc>::AlltoAllBuffInit()
{
    tpipe_->Reset();
    uint32_t bsMulTopkSizeAligned = Ceil(axisBS_ * axisK_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(readStateBuf_, UB_ALIGN);
    tpipe_->InitBuffer(statusBuf_, sendRankNum_ * UB_ALIGN);
    tpipe_->InitBuffer(expertIdsBuf_, bsMulTopkSizeAligned);
    tpipe_->InitBuffer(expandScalesBuf_, bsMulTopkSizeAligned);
    tpipe_->InitBuffer(tokenBuf_, axisH_ * sizeof(ExpandXType));
    tpipe_->InitBuffer(rowTmpFloatBuf_, axisHFloatSize_);
    tpipe_->InitBuffer(mulBuf_, axisHFloatSize_);
    tpipe_->InitBuffer(sumFloatBuf_, axisHFloatSize_);
    tpipe_->InitBuffer(indexCountsBuf_, bsMulTopkSizeAligned);
    tpipe_->InitBuffer(moeSumQueue_, BUFFER_NUM, axisHExpandXTypeSize_);
    tpipe_->InitBuffer(gatherMaskOutBuf_, epWorldSize_ * sizeof(float));
    tpipe_->InitBuffer(gatherTmpBuf_, sizeof(uint32_t));
    tpipe_->InitBuffer(statusSumOutBuf_, sizeof(float));
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeDistributeCombine<TemplateMC2TypeFunc>::SplitCoreCal()
{
    sendRankNum_ = epWorldSize_ / aivNum_;
    uint32_t remainderRankNum = epWorldSize_ % aivNum_;
    startRankId_ = sendRankNum_ * coreIdx_;
    if (coreIdx_ < remainderRankNum) {
        sendRankNum_++;
        startRankId_ += coreIdx_;
    } else {
        startRankId_ += remainderRankNum;
    }
    endRankId_ = startRankId_ + sendRankNum_;
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeDistributeCombine<TemplateMC2TypeFunc>::ReduceScatterTrans()
{
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(tpSendCountGM_[tpRankId_]);
    __asm__ __volatile__("");
    uint32_t offset = tpSendCountGM_.GetValue(tpRankId_) * axisH_;
    GlobalTensor<ExpandXType> dataCopyInGM = expandXGM_[offset];
    GM_ADDR rankGM = GetWinAddrByRankId(1 - tpRankId_, TP_DOMAIN) + tpDataOffsetOnWin_;
    rankWindow_.SetGlobalBuffer((__gm__ ExpandXType *)rankGM);
    uint32_t copyStartIdx = 0;
    if (startRankId_ > 0) {
        __asm__ __volatile__("");
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            epSendCountGM_[epWorldSize_ + startRankId_ - 1]);
        __asm__ __volatile__("");
        copyStartIdx = epSendCountGM_.GetValue(epWorldSize_ + startRankId_ - 1);
    }
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        epSendCountGM_[epWorldSize_ + endRankId_ - 1]);
    __asm__ __volatile__("");
    uint32_t copyEndIdx = epSendCountGM_.GetValue(epWorldSize_ + endRankId_ - 1);
    LocalTensor<ExpandXType> tmpUb;
    for (uint32_t tokenNumIdx = copyStartIdx; tokenNumIdx < copyEndIdx; tokenNumIdx++) {
        tmpUb = moeQueue_.AllocTensor<ExpandXType>();
        DataCopy(tmpUb, dataCopyInGM[tokenNumIdx * axisH_], axisH_);
        moeQueue_.EnQue(tmpUb);
        tmpUb = moeQueue_.DeQue<ExpandXType>();
        DataCopy(rankWindow_[tokenNumIdx * axisH_], tmpUb, axisH_);
        moeQueue_.FreeTensor<ExpandXType>(tmpUb);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeDistributeCombine<TemplateMC2TypeFunc>::SetWaitTpStatusAndDisPatch()
{
    pipe_barrier(PIPE_ALL);
    if (startRankId_ >= epWorldSize_) {
        return;
    }
    if constexpr (IsNeedReduceScatter) {
        uint32_t tpToRankId = 1 - tpRankId_;
        pipe_barrier(PIPE_ALL);
        LocalTensor<float> statusFlagUb = readStateBuf_.Get<float>();
        statusFlagUb(0) = sumTarget_;
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        GlobalTensor<float> tpWindowInstatusFp32Tensor_;
        stateGM_ = GetWinStateAddrByRankId(tpToRankId, TP_DOMAIN) + coreIdx_ * stateOffset_;
        tpWindowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)stateGM_);
        DataCopy<float>(tpWindowInstatusFp32Tensor_, statusFlagUb, 8UL);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        LocalTensor<float> statusFp32Tensor_ = readStateBuf_.Get<float>();
        float sumOfFlag = static_cast<float>(-1.0);
        uint32_t statusRankOffset = coreIdx_ * stateOffset_ / sizeof(float);
        while (sumOfFlag != sumTarget_) {
            DataCopy<float>(statusFp32Tensor_, tpStatusSpaceGlobalTensor_[statusRankOffset], 8);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            sumOfFlag = statusFp32Tensor_.GetValue(0);
            SyncFunc<AscendC::HardEvent::S_MTE2>();
        }
    }
    ExpertAlltoAllDispatchCopyAdd();
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeDistributeCombine<TemplateMC2TypeFunc>::ExpertAlltoAllDispatchCopyAdd()
{
    if (startRankId_ >= epWorldSize_) {
        return;
    }
    uint32_t curRankExpertNum = 0;
    DataCopyExtParams epSendCntParams;
    if (isShardExpert_) {
        curRankExpertNum = 1;
        epSendCntParams = {1U, static_cast<uint32_t>(epWorldSize_ * sizeof(uint32_t)), 0U, 0U, 0U};
    } else {
        curRankExpertNum = moeExpertPerRankNum_;
        epSendCntParams = {1U, static_cast<uint32_t>(moeSendNum_ * sizeof(uint32_t)), 0U, 0U, 0U};
    }
    DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(epSendCountLocal_, epSendCountGM_, epSendCntParams, copyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint32_t preCount = 0;
    uint32_t startTokenIdx = 0;
    uint32_t curTokenNum = 0;

    for (uint32_t expertIdx = 0U; expertIdx < curRankExpertNum; expertIdx++) {
        uint32_t sendEpCount = endRankId_ - startRankId_;
        for (uint32_t i = 0; i < sendEpCount; ++i) {
            uint32_t ep = startRankId_ + (i + epRankId_) % sendEpCount;
            if ((ep > 0) || (expertIdx > 0U)) {
                preCount = epSendCountLocal_.GetValue(expertIdx * epWorldSize_ + ep - 1);
            } else {
                preCount = 0;
            }
            curTokenNum = epSendCountLocal_.GetValue(expertIdx * epWorldSize_ + ep) - preCount;
            if (curTokenNum == 0) {
                continue;
            }
            startTokenIdx = preCount * axisH_;
            ExpertAlltoAllDispatchInnerCopyAdd(curTokenNum, startTokenIdx, ep, expertIdx);
        }
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeDistributeCombine<TemplateMC2TypeFunc>::ExpertAlltoAllDispatchInnerCopyAdd(
    uint32_t tokenNumLoop, uint32_t srcStartTokenIdx, uint32_t ep, uint32_t expertIdx)
{
    GM_ADDR rankGM = GetWinAddrByRankId(ep, EP_DOMAIN, expertIdx) + epDataOffsetOnWin_;
    if ((isShardExpert_) && (ep < sharedExpertRankNum_)) {
        rankGM = GetWinAddrByRankId(epRankId_, EP_DOMAIN, expertIdx) + ep * moeExpertPerRankNum_ * expertPerSizeOnWin_;
    }
    rankWindow_.SetGlobalBuffer((__gm__ ExpandXType *)rankGM);
    uint32_t dataCnt = axisH_;
    for (uint32_t loopIdx = 0; loopIdx < tokenNumLoop; loopIdx++) {
        if constexpr (IsNeedReduceScatter) {
            gmTpSendCountTensor_ = gmTpSendCountInQueue_.AllocTensor<ExpandXType>();
            DataCopy(gmTpSendCountTensor_, expandXGM_[srcStartTokenIdx], dataCnt);
            gmTpSendCountInQueue_.EnQue(gmTpSendCountTensor_);

            winTpSendCountTensor_ = winTpSendCountInQueue_.AllocTensor<ExpandXType>();
            DataCopy(winTpSendCountTensor_, tpRankWindow_[srcStartTokenIdx], dataCnt);
            winTpSendCountInQueue_.EnQue(winTpSendCountTensor_);

            gmTpSendCountTensor_ = gmTpSendCountInQueue_.DeQue<ExpandXType>();
            winTpSendCountTensor_ = winTpSendCountInQueue_.DeQue<ExpandXType>();
            outTensor_ = xOutQueue_.AllocTensor<ExpandXType>();

            CustomAdd(outTensor_, winTpSendCountTensor_, gmTpSendCountTensor_, dataCnt);
            gmTpSendCountInQueue_.FreeTensor<ExpandXType>(gmTpSendCountTensor_);
            winTpSendCountInQueue_.FreeTensor<ExpandXType>(winTpSendCountTensor_);
            xOutQueue_.EnQue(outTensor_);

            outTensor_ = xOutQueue_.DeQue<ExpandXType>();
            DataCopy(rankWindow_[loopIdx * dataCnt], outTensor_, dataCnt);
            xOutQueue_.FreeTensor<ExpandXType>(outTensor_);
        } else {
            gmTpSendCountTensor_ = gmTpSendCountQueue_.AllocTensor<ExpandXType>();
            DataCopy(gmTpSendCountTensor_, expandXGM_[srcStartTokenIdx], dataCnt);
            ExpandXType val = expandXGM_[srcStartTokenIdx].GetValue(0);
            gmTpSendCountQueue_.EnQue(gmTpSendCountTensor_);
            gmTpSendCountTensor_ = gmTpSendCountQueue_.DeQue<ExpandXType>();
            DataCopy(rankWindow_[loopIdx * dataCnt], gmTpSendCountTensor_, dataCnt);
            gmTpSendCountQueue_.FreeTensor<ExpandXType>(gmTpSendCountTensor_);
        }
        srcStartTokenIdx += dataCnt;
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeDistributeCombine<TemplateMC2TypeFunc>::CustomAdd(LocalTensor<ExpandXType> &dst,
                                                                               LocalTensor<ExpandXType> &src0,
                                                                               LocalTensor<ExpandXType> &src1,
                                                                               uint32_t dataCnt)
{
    if constexpr (AscendC::IsSameType<ExpandXType, bfloat16_t>::value) {
        Cast(winTpSendCountFloatTensor_, src0, RoundMode::CAST_NONE, dataCnt);
        Cast(gmTpSendCountFloatTensor_, src1, RoundMode::CAST_NONE, dataCnt);
        pipe_barrier(PIPE_V);
        Add(winTpSendCountFloatTensor_, winTpSendCountFloatTensor_, gmTpSendCountFloatTensor_, dataCnt);
        pipe_barrier(PIPE_V);
        Cast(dst, winTpSendCountFloatTensor_, RoundMode::CAST_ROUND, dataCnt);
    } else {
        Add(dst, src0, src1, dataCnt);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeDistributeCombine<TemplateMC2TypeFunc>::SetStatus()
{
    pipe_barrier(PIPE_ALL);
    if (startRankId_ >= epWorldSize_) {
        return;
    }

    LocalTensor<int32_t> statusFlagUb = readStateBuf_.Get<int32_t>();
    statusFlagUb.SetValue(0, epStateValue_);
    SyncFunc<AscendC::HardEvent::S_MTE3>();

    for (uint32_t epIdx = startRankId_; epIdx < endRankId_; epIdx++) {
        stateGM_ = GetWinStateAddrByRankId(epIdx, EP_DOMAIN) + epStateOffsetOnWin_;
        rankStates_.SetGlobalBuffer((__gm__ int32_t *)stateGM_);
        DataCopy(rankStates_, statusFlagUb, 8);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeDistributeCombine<TemplateMC2TypeFunc>::WaitDispatch()
{
    if (startRankId_ < epWorldSize_) {
        LocalTensor<float> statusTensor = statusBuf_.Get<float>();
        LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
        LocalTensor<uint32_t> gatherTmpTensor = gatherTmpBuf_.Get<uint32_t>();
        LocalTensor<float> statusSumOutTensor = statusSumOutBuf_.Get<float>();
        PipeBarrier<PIPE_ALL>();

        gatherTmpTensor.SetValue(0, 1);
        uint32_t mask = 1;  // gatherMask + sum
        uint64_t rsvdCnt = 0;
        DataCopyParams intriParams{static_cast<uint16_t>(sendRankNum_), 1,
                                   static_cast<uint16_t>((moeSendNum_ > 512) ? 7 : 15), 0};  // srcStride is 15 blocks
        float sumOfFlag = static_cast<float>(-1.0);
        float minTarget = (sumTarget_ * sendRankNum_) - (float)0.5;
        float maxTarget = (sumTarget_ * sendRankNum_) + (float)0.5;
        SumParams sumParams{1, sendRankNum_, sendRankNum_};
        SyncFunc<AscendC::HardEvent::S_V>();
        while ((sumOfFlag < minTarget) || (sumOfFlag > maxTarget)) {
            DataCopy<float>(statusTensor, epStatusSpaceGlobalTensor_[startRankId_ * stateOffset_ / sizeof(float)],
                            intriParams);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            GatherMask(gatherMaskOutTensor, statusTensor, gatherTmpTensor, true, mask,
                       {1, (uint16_t)sendRankNum_, 1, 0}, rsvdCnt);
            PipeBarrier<PIPE_V>();
            Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
            SyncFunc<AscendC::HardEvent::V_S>();
            sumOfFlag = statusSumOutTensor.GetValue(0);
        }
    }

    if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(RECV_SYNC_EVENT_ID);
        AscendC::CrossCoreWaitFlag(RECV_SYNC_EVENT_ID);
    } else {
        SyncAll<true>();
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeDistributeCombine<TemplateMC2TypeFunc>::LocalWindowCopy()
{
    uint32_t beginIndex = 0;
    uint32_t endIndex = 0;
    uint32_t processLen = 0;
    uint32_t tokenOffset = 0;
    if (axisBS_ < aivNum_) {
        uint32_t aivNumPerToken = aivNum_ / axisBS_;  // axisBS_ < aivNum_
        if (coreIdx_ >= (axisBS_ * aivNumPerToken)) {
            return;
        }
        uint32_t tokenIndex = coreIdx_ / aivNumPerToken;
        processLen = ((axisH_ / UB_ALIGN) / aivNumPerToken) * UB_ALIGN;
        tokenOffset = processLen * (coreIdx_ % aivNumPerToken);
        if ((coreIdx_ % aivNumPerToken) == (aivNumPerToken - 1)) {
            processLen = axisH_ - ((aivNumPerToken - 1) * processLen);
        }
        beginIndex = tokenIndex;
        endIndex = beginIndex + 1U;
    } else {
        uint32_t tokenPerAivNum = axisBS_ / aivNum_;
        uint32_t remainderToken = axisBS_ % aivNum_;
        beginIndex = tokenPerAivNum * coreIdx_;
        if (coreIdx_ < remainderToken) {
            tokenPerAivNum++;
            beginIndex = tokenPerAivNum * coreIdx_;
        } else {
            beginIndex += remainderToken;
        }
        endIndex = beginIndex + tokenPerAivNum;
        processLen = axisH_;
    }
    LocalTensor<ExpandIdxType> expertIdsLocal = expertIdsBuf_.Get<ExpandIdxType>();
    LocalTensor<float> expandScalesLocal = expandScalesBuf_.Get<float>();

    LocalTensor<float> rowTmpFloatLocal = rowTmpFloatBuf_.Get<float>();
    LocalTensor<float> mulBufLocal = mulBuf_.Get<float>();
    LocalTensor<float> sumFloatBufLocal = sumFloatBuf_.Get<float>();

    LocalTensor<ExpandIdxType> indexCountsLocal = indexCountsBuf_.Get<ExpandIdxType>();
    const DataCopyExtParams bskParams = {1U, static_cast<uint32_t>(bsKNum_ * sizeof(uint32_t)), 0U, 0U, 0U};
    const DataCopyPadExtParams<ExpandIdxType> copyPadParams{false, 0U, 0U, 0U};
    const DataCopyPadExtParams<float> copyPadFloatParams{false, 0U, 0U, 0U};

    DataCopyPad(indexCountsLocal, expandIdxGM_, bskParams, copyPadParams);
    DataCopyPad(expertIdsLocal, expertIdsGM_, bskParams, copyPadParams);
    DataCopyPad(expandScalesLocal, expandScalesGM_, bskParams, copyPadFloatParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t tokenIndex = beginIndex; tokenIndex < endIndex; tokenIndex++) {
        uint32_t index = tokenIndex * axisK_;
        SyncFunc<AscendC::HardEvent::MTE3_V>();
        Duplicate(sumFloatBufLocal, (float)0, axisH_);
        for (uint32_t i = 0; i < axisK_; i++) {
            int32_t moeExpert = expertIdsLocal.GetValue(index);
            if (moeExpert < 0) {
                index++;
                continue;
            }
            float scaleVal = expandScalesLocal.GetValue(index);
            GM_ADDR wAddr = (__gm__ uint8_t *)(epWindowGM_) +
                            expertPerSizeOnWin_ * moeExpertPerRankNum_ * sharedExpertRankNum_ +
                            expertPerSizeOnWin_ * moeExpert + indexCountsLocal.GetValue(index) * axisHExpandXTypeSize_ +
                            tokenOffset * sizeof(ExpandXType);
            rowTmpGlobal_.SetGlobalBuffer((__gm__ ExpandXType *)wAddr);
            ExpandXType val = rowTmpGlobal_.GetValue(0);
            LocalTensor<ExpandXType> tmpUb = moeSumQueue_.AllocTensor<ExpandXType>();
            DataCopy(tmpUb, rowTmpGlobal_, processLen);
            moeSumQueue_.EnQue(tmpUb);
            tmpUb = moeSumQueue_.DeQue<ExpandXType>();
            Cast(rowTmpFloatLocal, tmpUb, AscendC::RoundMode::CAST_NONE, processLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(mulBufLocal, rowTmpFloatLocal, scaleVal, processLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(sumFloatBufLocal, sumFloatBufLocal, mulBufLocal, processLen);
            index++;
            moeSumQueue_.FreeTensor<ExpandXType>(tmpUb);
        }
        LocalTensor<ExpandXType> rowTmpLocal = tokenBuf_.Get<ExpandXType>();
        if (sharedExpertRankNum_ > 0U) {
            uint32_t temp = (epRankId_ * axisBS_) / sharedExpertRankNum_;
            uint32_t moeOnShareRank = Ceil((tokenIndex + 1 + temp) * sharedExpertRankNum_, axisBS_) - 1 - epRankId_;
            uint32_t preCnt = (moeOnShareRank + epRankId_) * axisBS_ / sharedExpertRankNum_ -
                              epRankId_ * axisBS_ / sharedExpertRankNum_;
            __gm__ ExpandXType *shareAddr =
                (__gm__ ExpandXType *)(epWindowGM_ + moeOnShareRank * expertPerSizeOnWin_ * moeExpertPerRankNum_) +
                (tokenIndex - preCnt) * axisH_ + tokenOffset;
            GlobalTensor<ExpandXType> shareTokGlobal;
            shareTokGlobal.SetGlobalBuffer((__gm__ ExpandXType *)(shareAddr));
            SyncFunc<AscendC::HardEvent::V_MTE2>();
            DataCopy(rowTmpLocal, shareTokGlobal, processLen);
            SyncFunc<AscendC::HardEvent::MTE2_V>();
            Cast(rowTmpFloatLocal, rowTmpLocal, AscendC::RoundMode::CAST_NONE, processLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(sumFloatBufLocal, sumFloatBufLocal, rowTmpFloatLocal, processLen);
        }

        AscendC::PipeBarrier<PIPE_V>();
        LocalTensor<ExpandXType> sumBufLocal = tokenBuf_.Get<ExpandXType>();
        Cast(sumBufLocal, sumFloatBufLocal, AscendC::RoundMode::CAST_RINT, processLen);
        SyncFunc<AscendC::HardEvent::V_MTE3>();
        DataCopy(expandOutGlobal_[tokenIndex * axisH_ + tokenOffset], sumBufLocal, processLen);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeDistributeCombine<TemplateMC2TypeFunc>::Process()
{
    SyncAll<true>();
    if constexpr (IsNeedReduceScatter) {
        tpipe_->InitBuffer(moeQueue_, BUFFER_NUM, axisHExpandXTypeSize_);
        ReduceScatterTrans();
    }
    if constexpr ((EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) == 0) {
        BuffInit();
        SetWaitTpStatusAndDisPatch();
    }
    AlltoAllBuffInit();
    SetStatus();
    WaitDispatch();
    LocalWindowCopy();
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeDistributeCombine<TemplateMC2TypeFunc>::AllToAllSend()
{
    if constexpr (IsNeedReduceScatter) {
        tpipe_->InitBuffer(moeQueue_, BUFFER_NUM, axisHExpandXTypeSize_);
        ReduceScatterTrans();
    }
    BuffInit();
    if constexpr ((EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) == 0) {
        SetWaitTpStatusAndDisPatch();
        AlltoAllBuffInit();
    }
    if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SEND_SYNC_EVENT_ID);
        AscendC::CrossCoreWaitFlag(SEND_SYNC_EVENT_ID);
    } else {
        SyncAll<true>();
    }
    SetStatus();
    if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
        AscendC::CrossCoreWaitFlag(RECV_SYNC_EVENT_ID);
    } else {
        SyncAll<true>();
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void CamMoeDistributeCombine<TemplateMC2TypeFunc>::ReducePermute()
{
    AlltoAllBuffInit();
    if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
        AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(SEND_SYNC_EVENT_ID);
    } else {
        SyncAll<true>();
    }

    WaitDispatch();
    LocalWindowCopy();

    if constexpr (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) {
        AscendC::CrossCoreWaitFlag(SEND_SYNC_EVENT_ID);
    }
}
}  // namespace MoeDistributeCombineImpl

#endif  // CAM_MOE_DISTRIBUTE_COMBINE_IMPL_H
