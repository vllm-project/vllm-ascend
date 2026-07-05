/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description: ZbMoeDistributeDispatch operator kernel function header file, for A3 ZB (zero-buffer)
 * Author: Hu Tianlun
 * Create: 2026-01-03
 * Note:
 * History: 2026-01-03 create ZbMoeDistributeDispatch operator kernel function header file
 */

#ifndef ZB_MOE_DISTRIBUTE_DISPATCH_H
#define ZB_MOE_DISTRIBUTE_DISPATCH_H

#include "shmem.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "zb_moe_distribute_dispatch_tiling.h"
#include "shmem_sync_flag.h"
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

namespace ZbMoeDistributeDispatchImpl {
constexpr uint8_t BUFFER_NUM = 2;
constexpr uint8_t BUFFER_SINGLE = 1;
constexpr int UB_ALIGN_SIZE = 32;
constexpr uint64_t ALIGNED_LEN_256 = 256UL;
constexpr uint64_t ALIGNED_LEN_64 = 64UL;
constexpr uint32_t UB_32B_ALIGN = 32;
constexpr uint32_t FLAG_CNT_ALIGN = UB_32B_ALIGN / sizeof(int32_t);
constexpr uint32_t MAX_UB_SIZE = 170U * 1024U;
constexpr uint32_t UB_ALIGN = 32U;
// Keep dispatch and combine sync flags in separate windows: some ranks may enter combine while others are still
// finishing dispatch, so sharing the same flag slots can race with cleanup.
constexpr uint64_t DISPATCH_SYNC_FLAG_BASE_OFFSET = ShmemSyncFlagImpl::FLAG_AREA_BASE + 64UL * 1024UL;
constexpr uint64_t DISPATCH_SYNC_STATE_OFFSET = 32U;

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define TemplateMC2TypeClass                                                                               \
    typename XType, typename ExpandXOutType, bool StaticQuant, bool DynamicQuant, bool IsSmoothScaleExist, \
        bool IsNeedAllgather
#define TemplateMC2TypeFunc XType, ExpandXOutType, StaticQuant, DynamicQuant, IsSmoothScaleExist, IsNeedAllgather

template <TemplateMC2TypeClass>
class ZbMoeDistributeDispatch
{
public:
    __aicore__ inline ZbMoeDistributeDispatch(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR elasticInfo,
                                GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR tpSendCountsOut,
                                GM_ADDR workspaceGM, TPipe *pipe,
                                const ZbMoeDistributeDispatchTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SendToSharedExpert();
    __aicore__ inline void SendToMoeExpert();
    __aicore__ inline void SendCountNotify();
    __aicore__ inline void WaitNotify();
    __aicore__ inline void InputToDstOutput();
    __aicore__ inline void LocalLayout();
    __aicore__ inline void SetLayoutStatus();
    __aicore__ inline GM_ADDR GetSyncFlagAddrByRankId(uint32_t rankId);
    __aicore__ inline void SetSyncFlag();
    __aicore__ inline void WaitSyncFlag();
    __aicore__ inline void CleanUp();
    __aicore__ inline void ReduceMaxInplace(const LocalTensor<float> &srcLocal, uint32_t count);
    __aicore__ inline void QuantProcess();
    __aicore__ inline void QuantInit(GM_ADDR scales);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum, bool isFront = true);
    __aicore__ inline void CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt);
    __aicore__ inline void ReorderRecvDataOutput(int32_t rankId, LocalTensor<int32_t> &transLt, bool isCumSum = false);

    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<int32_t> allExpertTokenNumsGMTensor_;
    GlobalTensor<int32_t> expandIdxGMTensor_;
    GlobalTensor<int32_t> LocalNotifyDataTensor_;
    GlobalTensor<float> LocalNotifyStatusTensor_;
    GlobalTensor<int32_t> magicValTensor_;
    GlobalTensor<float> layoutStatusTensor_;
    GlobalTensor<ExpandXOutType> dstWinGMTensor;
    GlobalTensor<float> dstScaleGMTensor;
    GlobalTensor<int64_t> expertTokenNumsGlobal;
    GlobalTensor<int32_t> sendCountsGlobal;

    LocalTensor<ExpandXOutType> xTmpTensor_;
    LocalTensor<XType> xInTensor_;
    LocalTensor<ExpandXOutType> xOutTensor_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<float> layoutFlag_;
    LocalTensor<float> statusTensor_;
    LocalTensor<float> layoutStatusFp32Tensor_;
    LocalTensor<int32_t> dstExpIdTensor_;
    LocalTensor<int32_t> subExpIdTensor_;
    LocalTensor<float> workLocalTensor_;
    LocalTensor<int32_t> recvDataTensor_;
    LocalTensor<int32_t> expandIdsTensor_;

    TBuf<> expertIdsBuf_;
    TBuf<> expandIdsBuf_;
    TBuf<> statusBuf_;
    TBuf<> notifyBuf_;
    TBuf<> rowMaxBuf_;
    TBuf<> receiveDataCastFloatBuf_;
    TBuf<> smoothScalesBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> subExpBuf_;
    TBuf<> layoutWaitStatusBuf_;
    TBuf<> gatherMaskTBuf_;
    TBuf<> sendCountLocalBuf_;
    TBuf<> recvDataBuf_;
    TBuf<> sendCountBuf_;
    TBuf<> syncStatusBuf_;
    TBuf<> syncWaitStatusBuf_;
    TBuf<> syncGatherMaskBuf_;
    TBuf<> syncStatusSumBuf_;
    TBuf<QuePosition::VECCALC> clearBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_;
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECOUT, 1> xOutQueue_;

    GM_ADDR expandXOutGM_;
    GM_ADDR dynamicScalesOutGM_;

    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t sharedUsedAivNum_{0};
    uint32_t moeUsedAivNum_{0};
    uint32_t epWorldSize_{0};
    int32_t epRankId_{0};
    uint32_t aivId_{0};
    uint32_t coreNum_{0};
    uint32_t sharedExpertRankNum_{0};
    uint32_t moeExpertNum_{0};
    uint32_t moeExpertNumPerRank_{0};
    uint32_t hOutSize_{0};
    uint32_t hOutAlignUbSize_{0};
    uint32_t hOutSizeAlign_{0};
    uint32_t totalUsedUB_{0};
    uint64_t sendToMoeExpTokenCnt_{0};
    bool isShareExpertRankFlag_ = false;
    float exp_flag;
    uint32_t expertTokenNumsType_{1};
    int32_t expertIdsCnt_{0};
    uint32_t maxSize_{0};
    uint32_t bufferNum_{0};
    int32_t magicVal_{0};

    DataCopyExtParams expandXCopyParams_;
    DataCopyExtParams scaleCopyParams_;
    DataCopyExtParams xCopyParams_;
    GM_ADDR metaDataGvaGM_{0};
    uint32_t rankNumPerBlock_{0};
    uint32_t curBlockStartRankId_{0};
    uint32_t curBlockEndRankId_{0};
};

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeDispatch<TemplateMC2TypeFunc>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR elasticInfo, GM_ADDR expandXOut,
    GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut,
    GM_ADDR tpSendCountsOut, GM_ADDR workspaceGM, TPipe *pipe,
    const ZbMoeDistributeDispatchTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    coreNum_ = GetBlockNum();
    epRankId_ = tilingData->moeDistributeDispatchInfo.epRankId;

    axisBS_ = tilingData->moeDistributeDispatchInfo.bs;
    axisH_ = tilingData->moeDistributeDispatchInfo.h;
    epWorldSize_ = tilingData->moeDistributeDispatchInfo.epWorldSize;
    sharedExpertRankNum_ = tilingData->moeDistributeDispatchInfo.sharedExpertRankNum;
    moeExpertNum_ = tilingData->moeDistributeDispatchInfo.moeExpertNum;
    axisK_ = tilingData->moeDistributeDispatchInfo.k;
    aivNum_ = tilingData->moeDistributeDispatchInfo.aivNum;
    expertTokenNumsType_ = tilingData->moeDistributeDispatchInfo.expertTokenNumsType;

    // The shared metadata buffer is laid out as:
    // per-core magic value, per-rank layout flags, per-core send-count flags, then send-count data.
    metaDataGvaGM_ = (GM_ADDR)tilingData->moeDistributeDispatchInfo.shmemptr;
    GM_ADDR statusDataSpaceGm = (GM_ADDR)(metaDataGvaGM_);
    magicValTensor_.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm) + aivId_ * FLAG_CNT_ALIGN);

    GM_ADDR layoutStatusGm = (GM_ADDR)(statusDataSpaceGm + coreNum_ * UB_32B_ALIGN);
    layoutStatusTensor_.SetGlobalBuffer((__gm__ float *)(layoutStatusGm));
    GM_ADDR localNotifyStatusGm = (GM_ADDR)(layoutStatusGm + coreNum_ * UB_32B_ALIGN);
    LocalNotifyStatusTensor_.SetGlobalBuffer((__gm__ float *)(localNotifyStatusGm) + aivId_ * FLAG_CNT_ALIGN);
    GM_ADDR localNotifyDataSpaceGm = (GM_ADDR)(localNotifyStatusGm + coreNum_ * ALIGNED_LEN_64);
    LocalNotifyDataTensor_.SetGlobalBuffer((__gm__ int32_t *)(localNotifyDataSpaceGm));

    TBuf<> tBuf;
    tpipe_->InitBuffer(tBuf, UB_32B_ALIGN);
    LocalTensor<int32_t> tempLocal = tBuf.Get<int32_t>();
    tempLocal(0) = 1;

    // Increase the per-core magic value once per launch so stale flags from previous launches are ignored.
    AscendC::SetAtomicAdd<int32_t>();
    AscendC::SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    DataCopy(magicValTensor_, tempLocal, FLAG_CNT_ALIGN);
    AscendC::SetAtomicNone();
    AscendC::SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    magicVal_ = magicValTensor_.GetValue(0);
    SplitToCore(epWorldSize_, aivNum_, curBlockStartRankId_, curBlockEndRankId_, rankNumPerBlock_);
    PipeBarrier<PIPE_ALL>();

    exp_flag = (float)magicVal_;

    tpipe_->InitBuffer(notifyBuf_, FLAG_CNT_ALIGN);
    layoutFlag_ = notifyBuf_.Get<float>();
    layoutFlag_.SetValue(0, exp_flag);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(LocalNotifyStatusTensor_, layoutFlag_, FLAG_CNT_ALIGN);

    if (epRankId_ < sharedExpertRankNum_) {
        isShareExpertRankFlag_ = true;
    }

    uint32_t sharedExpertNum = tilingData->moeDistributeDispatchInfo.sharedExpertNum;
    uint32_t moeExpertRankNum = epWorldSize_ - sharedExpertRankNum_;
    moeExpertNumPerRank_ = moeExpertNum_ / moeExpertRankNum;

    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    expandIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(expandIdxOut));
    expandXOutGM_ = expandXOut;
    dynamicScalesOutGM_ = dynamicScalesOut;
    expertTokenNumsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(expertTokenNumsOut));
    sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCountsOut));
    allExpertTokenNumsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(workspaceGM));

    hOutSize_ = axisH_ * sizeof(ExpandXOutType);
    hOutSizeAlign_ = Ceil(hOutSize_, UB_ALIGN) * UB_ALIGN;
    // Quantized output stores one fp32 scale after the aligned token payload.
    uint32_t hScaleSizeAlign = hOutSizeAlign_ + UB_ALIGN;
    uint32_t hScaleIdxSize = hScaleSizeAlign;
    if (sharedExpertRankNum_ != 0U) {
        sharedUsedAivNum_ = (aivNum_ * sharedExpertNum) / (axisK_ + sharedExpertNum);
        if (sharedUsedAivNum_ == 0) {
            sharedUsedAivNum_ = 1;
        }
    }
    expertIdsCnt_ = axisBS_ * axisK_;
    moeUsedAivNum_ = aivNum_ - sharedUsedAivNum_;
    PipeBarrier<PIPE_ALL>();

    uint32_t statusBufSize = UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    totalUsedUB_ += statusBufSize;

    statusTensor_ = statusBuf_.Get<float>();
    statusTensor_.SetValue(0, exp_flag);
    hOutAlignUbSize_ = Ceil(hScaleIdxSize, UB_ALIGN) * UB_ALIGN;
    uint32_t hFp32Size = axisH_ * sizeof(float);
    uint32_t expertIdsSize = expertIdsCnt_ * sizeof(int32_t);
    uint32_t xActivateMaskSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN) * sizeof(half);
    uint32_t bsAlign256 = Ceil(axisBS_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    uint32_t bsKAlign256 = Ceil(expertIdsCnt_ * sizeof(half), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(half);
    uint32_t expertIdsBufSize = expertIdsSize > bsAlign256 ? expertIdsSize : bsAlign256;
    expertIdsSize = Ceil(expertIdsSize, UB_ALIGN) * UB_ALIGN;
    maxSize_ = hFp32Size > expertIdsSize ? hFp32Size : expertIdsSize;
    maxSize_ = maxSize_ > xActivateMaskSize ? maxSize_ : xActivateMaskSize;
    maxSize_ = maxSize_ > bsKAlign256 ? maxSize_ : bsKAlign256;
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize);
    totalUsedUB_ += expertIdsSize;
    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();

    uint32_t sendCountLocalBufSize = moeExpertNum_ * sizeof(int32_t);
    tpipe_->InitBuffer(sendCountLocalBuf_, sendCountLocalBufSize);

    tpipe_->InitBuffer(gatherMaskTBuf_, maxSize_);
    totalUsedUB_ += maxSize_;
    workLocalTensor_ = gatherMaskTBuf_.Get<float>();

    if constexpr (DynamicQuant || StaticQuant) {
        QuantInit(scales);
        // Reuse quantization buffers as temporary expert-id buffers after initialization.
        dstExpBuf_ = receiveDataCastFloatBuf_;
        subExpBuf_ = smoothScalesBuf_;
    } else {
        tpipe_->InitBuffer(dstExpBuf_, maxSize_);
        totalUsedUB_ += maxSize_;
        tpipe_->InitBuffer(subExpBuf_, maxSize_);
        totalUsedUB_ += maxSize_;
        uint32_t tmpTotalUB = totalUsedUB_ + hOutAlignUbSize_ * BUFFER_NUM;
        bufferNum_ = tmpTotalUB > MAX_UB_SIZE ? BUFFER_SINGLE : BUFFER_NUM;
        tpipe_->InitBuffer(xQueue_, bufferNum_, hOutAlignUbSize_);
    }

    dstExpIdTensor_ = dstExpBuf_.Get<int32_t>();
    subExpIdTensor_ = subExpBuf_.Get<int32_t>();

    xCopyParams_ = {1U, static_cast<uint32_t>(axisH_ * sizeof(XType)), 0U, 0U, 0U};
    expandXCopyParams_ = {1U, static_cast<uint32_t>(hOutSizeAlign_), 0U, 0U, 0U};
    scaleCopyParams_ = {1U, sizeof(float), 0U, 0U, 0U};
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeDispatch<TemplateMC2TypeFunc>::QuantInit(GM_ADDR scales)
{
    uint32_t hAlignSize = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(receiveDataCastFloatBuf_, maxSize_);
    totalUsedUB_ += maxSize_;
    tpipe_->InitBuffer(smoothScalesBuf_, maxSize_);
    totalUsedUB_ += maxSize_;
    if constexpr (DynamicQuant) {
        tpipe_->InitBuffer(rowMaxBuf_, UB_ALIGN);
    }
    uint32_t tmpTotalUB = totalUsedUB_ + BUFFER_NUM * hAlignSize + hOutAlignUbSize_ * BUFFER_NUM;
    bufferNum_ = tmpTotalUB > MAX_UB_SIZE ? BUFFER_SINGLE : BUFFER_NUM;
    tpipe_->InitBuffer(xInQueue_, bufferNum_, hAlignSize);
    tpipe_->InitBuffer(xOutQueue_, bufferNum_, hOutAlignUbSize_);
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeDispatch<TemplateMC2TypeFunc>::SplitToCore(uint32_t curSendCnt,
                                                                                 uint32_t curUseAivNum,
                                                                                 uint32_t &startTokenId,
                                                                                 uint32_t &endTokenId,
                                                                                 uint32_t &sendTokenNum, bool isFront)
{
    sendTokenNum = curSendCnt / curUseAivNum;
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum;
    uint32_t newAivId = isFront ? aivId_ : aivId_ - moeUsedAivNum_;
    startTokenId = sendTokenNum * newAivId;
    if (newAivId < remainderTokenNum) {
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeDispatch<TemplateMC2TypeFunc>::CalTokenSendExpertCnt(
    uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt)
{
    Duplicate<int32_t>(dstExpIdTensor_, dstExpertId, calCnt);
    PipeBarrier<PIPE_V>();
    Sub(subExpIdTensor_, expertIdsTensor_, dstExpIdTensor_, calCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> tmpFp32 = subExpIdTensor_.ReinterpretCast<float>();
    LocalTensor<float> tmpoutFp32 = dstExpIdTensor_.ReinterpretCast<float>();
    Abs(tmpoutFp32, tmpFp32, calCnt);
    PipeBarrier<PIPE_V>();
    Mins(subExpIdTensor_, dstExpIdTensor_, 1, calCnt);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(tmpoutFp32, tmpFp32, workLocalTensor_, calCnt);
    SyncFunc<AscendC::HardEvent::V_S>();
    int32_t curOtherExpertCnt = dstExpIdTensor_(0);
    if (calCnt >= curOtherExpertCnt) {
        curExpertCnt = calCnt - curOtherExpertCnt;
    } else {
        curExpertCnt = 0;
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline GM_ADDR
ZbMoeDistributeDispatch<TemplateMC2TypeFunc>::GetSyncFlagAddrByRankId(uint32_t rankId)
{
    auto metaPtr = reinterpret_cast<__gm__ uint8_t *>(aclshmem_ptr(metaDataGvaGM_, rankId));
    return (GM_ADDR)(metaPtr) + DISPATCH_SYNC_FLAG_BASE_OFFSET;
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeDispatch<TemplateMC2TypeFunc>::SetSyncFlag()
{
    if (rankNumPerBlock_ == 0U) {
        return;
    }

    uint32_t statusCntAlign = Ceil(rankNumPerBlock_, 8) * 8;
    tpipe_->InitBuffer(syncStatusBuf_, statusCntAlign * UB_ALIGN);
    LocalTensor<int32_t> statusTensor = syncStatusBuf_.Get<int32_t>();
    Duplicate<int32_t>(statusTensor, 0, rankNumPerBlock_ * 8);
    uint64_t mask[2] = {0x101010101010101, 0};
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor, 0x3F800000, mask, statusCntAlign / 8, 1, 8);
    PipeBarrier<PIPE_ALL>();

    GlobalTensor<int32_t> gmRemoteStatusGT;
    for (uint32_t targetRankId = curBlockStartRankId_; targetRankId < curBlockEndRankId_; ++targetRankId) {
        auto ptr = GetSyncFlagAddrByRankId(targetRankId) + epRankId_ * DISPATCH_SYNC_STATE_OFFSET;
        gmRemoteStatusGT.SetGlobalBuffer((__gm__ int32_t *)(ptr));
        DataCopy<int32_t>(gmRemoteStatusGT,
                          statusTensor[(targetRankId - curBlockStartRankId_) * 8],
                          8UL);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeDispatch<TemplateMC2TypeFunc>::WaitSyncFlag()
{
    if (rankNumPerBlock_ == 0U) {
        return;
    }

    uint32_t waitStatusBufSize =
        (((rankNumPerBlock_ * UB_ALIGN) > 256U) ? (rankNumPerBlock_ * UB_ALIGN) : 256U);
    tpipe_->InitBuffer(syncWaitStatusBuf_, waitStatusBufSize);
    uint32_t maskAlign = Ceil(epWorldSize_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(syncGatherMaskBuf_, maskAlign);
    tpipe_->InitBuffer(syncStatusSumBuf_, UB_ALIGN);

    LocalTensor<float> gatherMaskOutTensor = syncGatherMaskBuf_.Get<float>();
    LocalTensor<float> statusSumOutTensor = syncStatusSumBuf_.Get<float>(UB_ALIGN);
    LocalTensor<float> statusFp32Tensor = syncWaitStatusBuf_.Get<float>();
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
                 statusFp32TensorGT[curBlockStartRankId_ * DISPATCH_SYNC_STATE_OFFSET / sizeof(float)],
                 intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(statusSumOutTensor, statusFp32Tensor, gatherMaskOutTensor, mask, rankNumPerBlock_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }

    SyncFunc<AscendC::HardEvent::MTE3_S>();
    DataCopyParams intriOutParams{static_cast<uint16_t>(rankNumPerBlock_), 1, 0, 0};
    uint64_t duplicateMask[2] = {0x101010101010101, 0};
    LocalTensor<int32_t> cleanStateTensor = syncWaitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(rankNumPerBlock_, 8), 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(statusFp32TensorGT[curBlockStartRankId_ * DISPATCH_SYNC_STATE_OFFSET / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(),
             intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeDispatch<TemplateMC2TypeFunc>::LocalLayout()
{
    LocalTensor<int32_t> sendCountLt = sendCountLocalBuf_.Get<int32_t>();

    Duplicate<int32_t>(sendCountLt, 0, moeExpertNum_);
    PipeBarrier<PIPE_V>();

    // Count this rank's tokens per target expert and atomically publish the counts to shared metadata.
    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    sendToMoeExpTokenCnt_ = axisBS_ * axisK_;
    uint32_t startTokenId, endTokenId, sendTokenNum;
    SplitToCore(sendToMoeExpTokenCnt_, moeUsedAivNum_, startTokenId, endTokenId, sendTokenNum);
    if (startTokenId >= sendToMoeExpTokenCnt_) {
        return;
    }

    for (int32_t i = startTokenId; i < endTokenId; ++i) {
        int32_t expertIdx = static_cast<int32_t>(expertIdsTensor_(i));
        int32_t curCnt = sendCountLt.GetValue(expertIdx) + 1;
        sendCountLt.SetValue(expertIdx, curCnt);
    }

    PipeBarrier<PIPE_V>();
    AscendC::SetAtomicAdd<int32_t>();
    uint32_t sendSize = moeExpertNum_ * sizeof(int32_t);
    const DataCopyExtParams sendCountDataCopyParams{1U, sendSize, 0U, 0U, 0U};

    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(LocalNotifyDataTensor_, sendCountLt, sendCountDataCopyParams);
    AscendC::SetAtomicNone();
    PipeBarrier<PIPE_MTE3>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeDispatch<TemplateMC2TypeFunc>::SetLayoutStatus()
{
    // Notify every rank that this rank has finished writing its local expert counts.
    uint32_t rankNumPerBlock = 0U, startRankId = 0U, endRankId = 0U;
    SplitToCore(epWorldSize_, moeUsedAivNum_, startRankId, endRankId, rankNumPerBlock);
    if (rankNumPerBlock == 0U) {
        return;
    }

    for (uint32_t targetRankId = startRankId; targetRankId < endRankId; targetRankId++) {
        aclshmem_float_put_nbi(layoutStatusTensor_[epRankId_], statusTensor_, 1, targetRankId);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeDispatch<TemplateMC2TypeFunc>::SendCountNotify()
{
    // Split the peer-rank polling work across MoE AIVs. Each AIV waits for a subset of ranks and then fetches
    // their local expert counts into this rank's workspace.
    uint32_t rankNumPerBlock = 0U, startRankId = 0U, endRankId = 0U;
    SplitToCore(epWorldSize_, moeUsedAivNum_, startRankId, endRankId, rankNumPerBlock);
    if (rankNumPerBlock == 0U) {
        return;
    }

    tpipe_->InitBuffer(layoutWaitStatusBuf_, UB_ALIGN);
    for (uint32_t targetRankId = startRankId; targetRankId < endRankId; targetRankId++) {
        float curVal = static_cast<float>(-1.0);
        layoutStatusFp32Tensor_ = layoutWaitStatusBuf_.Get<float>();
        DataCopyParams intriParams{1, static_cast<uint16_t>(1 * sizeof(uint32_t)), 0, 0};
        float minFlagVal = exp_flag - static_cast<float>(0.5);
        float maxFlagVal = exp_flag + static_cast<float>(0.5);

        // Wait until the target rank's layout flag matches this launch's magic value.
        while ((curVal < minFlagVal) || (curVal > maxFlagVal)) {
            DataCopy(layoutStatusFp32Tensor_, layoutStatusTensor_[targetRankId], intriParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            curVal = layoutStatusFp32Tensor_.GetValue(0);
        }

        // The fetched vector is targetRank's per-expert send-count row.
        shmem_get_int32_mem_nbi(allExpertTokenNumsGMTensor_[targetRankId * moeExpertNum_],
                                LocalNotifyDataTensor_, moeExpertNum_, targetRankId);
        PipeBarrier<PIPE_ALL>();
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeDispatch<TemplateMC2TypeFunc>::WaitNotify()
{
    uint32_t rankNumPerBlock = 0U, startRankId = 0U, endRankId = 0U;
    SplitToCore(epWorldSize_, moeUsedAivNum_, startRankId, endRankId, rankNumPerBlock);
    if (rankNumPerBlock == 0U) {
        return;
    }
    uint32_t recvDataAlignLen = Ceil(moeExpertNum_ * epWorldSize_ * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE;
    tpipe_->InitBuffer(recvDataBuf_, recvDataAlignLen);

    recvDataTensor_ = recvDataBuf_.Get<int32_t>();
    DataCopyExtParams recvDataParams = {1U, static_cast<uint32_t>(recvDataAlignLen), 0, 0, 0};
    DataCopyPadExtParams<int32_t> copyPadInt32Params{false, 0U, 0U, 0U};
    DataCopyPad(recvDataTensor_, allExpertTokenNumsGMTensor_, recvDataParams, copyPadInt32Params);
    PipeBarrier<PIPE_ALL>();

    tpipe_->InitBuffer(sendCountBuf_, Ceil(moeExpertNum_ * sizeof(int32_t), UB_ALIGN_SIZE) * UB_ALIGN_SIZE);
    LocalTensor<int32_t> recvTokenLt = sendCountBuf_.Get<int32_t>();

    // Convert the global rank-by-expert count matrix into per-rank prefix sums for output offsets.
    for (uint32_t rank = startRankId; rank < endRankId; ++rank) {
        ReorderRecvDataOutput(rank, recvTokenLt, true);
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(moeExpertNum_ * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(sendCountsGlobal[rank * moeExpertNum_], recvTokenLt, copyParams);

        if (rank == static_cast<uint32_t>(epRankId_)) {
            SyncFunc<AscendC::HardEvent::MTE3_S>();
            int64_t tokenSums = 0;
            for (uint32_t localMoeIndex = 0; localMoeIndex < moeExpertNumPerRank_; ++localMoeIndex) {
                uint32_t curIndex = epWorldSize_ * (localMoeIndex + 1) - 1;
                int32_t prevCount = (localMoeIndex == 0) ? 0 : recvTokenLt.GetValue(curIndex - epWorldSize_);
                int32_t currCount = recvTokenLt.GetValue(curIndex);
                tokenSums = ((expertTokenNumsType_ == 0U) ? tokenSums : 0) + (currCount - prevCount);
                expertTokenNumsGlobal.SetValue(localMoeIndex, tokenSums);
            }
        }
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeDispatch<TemplateMC2TypeFunc>::ReorderRecvDataOutput(
    int32_t rankId, LocalTensor<int32_t> &transLt, bool isCumSum)
{
    uint32_t moeExpertPerRankNum = moeExpertNum_ / epWorldSize_;
    uint32_t startExpId = rankId * moeExpertPerRankNum;
    uint32_t endExpId = rankId * moeExpertPerRankNum + moeExpertPerRankNum;

    SyncFunc<AscendC::HardEvent::V_S>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    // Traverse source ranks inside each expert so the cumulative value remains a count, not an offset.
    int32_t prefixSum = 0;
    for (uint32_t expId = startExpId; expId < endExpId; ++expId) {
        for (uint32_t srcRank = 0; srcRank < epWorldSize_; ++srcRank) {
            uint32_t index = (expId - startExpId) * epWorldSize_ + srcRank;
            uint32_t pairIdx = srcRank * moeExpertNum_ + expId;

            int32_t curRecvCount = recvDataTensor_(pairIdx);
            prefixSum += curRecvCount;
            transLt(index) = isCumSum ? prefixSum : curRecvCount;
        }
    }
    PipeBarrier<PIPE_ALL>();
    SyncFunc<AscendC::HardEvent::S_MTE2>();
}

/*
 * Shared-expert ranks use all AIVs for MoE expert sends.
 * MoE-expert ranks split AIVs between shared-expert sends and MoE expert sends.
 */
template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeDispatch<TemplateMC2TypeFunc>::InputToDstOutput()
{
    bool isSendShared = (aivId_ >= moeUsedAivNum_) && (sharedExpertRankNum_ != 0);
    if (isSendShared) {
        SendToSharedExpert();
        return;
    }
    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    SendToMoeExpert();
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeDispatch<TemplateMC2TypeFunc>::SendToMoeExpert()
{
    uint32_t startTokenId, endTokenId, sendTokenNum;
    SplitToCore(sendToMoeExpTokenCnt_, moeUsedAivNum_, startTokenId, endTokenId, sendTokenNum);
    if (startTokenId >= sendToMoeExpTokenCnt_) {
        return;
    }

    DataCopyPadExtParams<XType> copyPadExtParams{false, 0U, 0U, 0U};
    tpipe_->InitBuffer(expandIdsBuf_, sendTokenNum * sizeof(uint32_t));
    expandIdsTensor_ = expandIdsBuf_.Get<int32_t>();

    // Flattened token-topk indices are distributed across AIVs. For each selected expert, compute the remote rank,
    // the expert's global base offset, and this token's local offset inside that expert.
    for (int32_t index = startTokenId; index < endTokenId; ++index) {
        int32_t tokenIndex = index / axisK_;
        uint32_t dstExpertId = expertIdsTensor_(index);
        int32_t dstRankId = dstExpertId / moeExpertNumPerRank_ + sharedExpertRankNum_;
        int32_t curExpertCnt = 0;

        if ((tokenIndex > 0) && (index > 0)) {
            // Count previous occurrences of dstExpertId in the already-scanned prefix.
            CalTokenSendExpertCnt(dstExpertId, index, curExpertCnt);
        }

        // expandIdx records the per-expert local offset used by combine to read the packed token back.
        expandIdsTensor_.SetValue(index - startTokenId, curExpertCnt);
        uint32_t offsetIdx = dstExpertId * epWorldSize_ + epRankId_;
        uint32_t col = offsetIdx % moeExpertNum_;
        // sendCountsGlobal is already a prefix-sum table, so the previous slot is the base offset.
        int32_t dstExpertOffset = (col == 0) ? 0 : sendCountsGlobal(offsetIdx - 1);
        auto dstPtr = reinterpret_cast<__gm__ uint8_t *>(aclshmem_ptr(expandXOutGM_, dstRankId));
        dstWinGMTensor.SetGlobalBuffer(
            (__gm__ ExpandXOutType *)(dstPtr + hOutSizeAlign_ * (dstExpertOffset + curExpertCnt)));

        if constexpr (DynamicQuant || StaticQuant) {
            // Quantized dispatch writes the packed token and one fp32 scale to separate remote windows.
            auto dstScalePtr = reinterpret_cast<__gm__ uint8_t *>(aclshmem_ptr(dynamicScalesOutGM_, dstRankId));
            dstScaleGMTensor.SetGlobalBuffer(
                (__gm__ float *)(dstScalePtr + sizeof(float) * (dstExpertOffset + curExpertCnt)));
            xInTensor_ = xInQueue_.AllocTensor<XType>();
            DataCopyPad(xInTensor_, xGMTensor_[tokenIndex * axisH_], xCopyParams_, copyPadExtParams);
            xInQueue_.EnQue(xInTensor_);
            xInTensor_ = xInQueue_.DeQue<XType>();
            xOutTensor_ = xOutQueue_.AllocTensor<ExpandXOutType>();
            QuantProcess();
            xOutQueue_.EnQue(xOutTensor_);
            xOutTensor_ = xOutQueue_.DeQue<ExpandXOutType>();

            DataCopyPad(dstWinGMTensor, xOutTensor_, expandXCopyParams_);
            LocalTensor<float> xOutFp32Tensor = xOutTensor_.template ReinterpretCast<float>();
            DataCopyPad(dstScaleGMTensor, xOutFp32Tensor[hOutSizeAlign_ / sizeof(float)], scaleCopyParams_);
            xOutQueue_.FreeTensor<ExpandXOutType>(xOutTensor_);
        } else {
            xTmpTensor_ = xQueue_.AllocTensor<ExpandXOutType>();
            DataCopyPad(xTmpTensor_, xGMTensor_[tokenIndex * axisH_], xCopyParams_, copyPadExtParams);
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<ExpandXOutType>();
            DataCopyPad(dstWinGMTensor, xTmpTensor_, expandXCopyParams_);
            xQueue_.FreeTensor<ExpandXOutType>(xTmpTensor_);
        }
    }

    DataCopyExtParams expandIdxParams = {1U, static_cast<uint32_t>(sendTokenNum * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPad(expandIdxGMTensor_[startTokenId], expandIdsTensor_, expandIdxParams);
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeDispatch<TemplateMC2TypeFunc>::SendToSharedExpert()
{
    uint32_t startTokenId, endTokenId, sendTokenNum;
    SplitToCore(axisBS_, sharedUsedAivNum_, startTokenId, endTokenId, sendTokenNum, false);
    if (startTokenId >= axisBS_) {
        return;
    }

    for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; ++tokenIndex) {
        uint32_t temp = (epRankId_ * axisBS_) / sharedExpertRankNum_;
        // Distribute shared-expert tokens evenly across shared-expert ranks.
        uint32_t moeOnShareRank = Ceil((tokenIndex + 1 + temp) * sharedExpertRankNum_, axisBS_) - 1 - epRankId_;

        auto dstPtr = reinterpret_cast<__gm__ uint8_t *>(aclshmem_ptr(expandXOutGM_, moeOnShareRank));
        dstWinGMTensor.SetGlobalBuffer(
            (__gm__ ExpandXOutType *)(dstPtr + hOutSizeAlign_ * epRankId_));

        // Shared expert data copy path is intentionally reserved for follow-up implementation.
        // DataCopyPadExtParams<XType> copyPadExtParams{false, 0U, 0U, 0U};
        // uint32_t preCnt = (moeOnShareRank + epRankId_) * axisBS_ / sharedExpertRankNum_ -
        //                   epRankId_ * axisBS_ / sharedExpertRankNum_;
        // if constexpr (DynamicQuant || StaticQuant) {
        //     auto dstScalePtr = reinterpret_cast<__gm__ uint8_t *>(aclshmem_ptr(dynamicScalesOutGM_, moeOnShareRank));
        //     dstScaleGMTensor.SetGlobalBuffer((__gm__ float *)(dstScalePtr));
        //     xInTensor_ = xInQueue_.AllocTensor<XType>();
        //     DataCopyPad(xInTensor_, xGMTensor_[tokenIndex * axisH_], xCopyParams_, copyPadExtParams);
        //     xInQueue_.EnQue(xInTensor_);
        //     xInTensor_ = xInQueue_.DeQue<XType>();
        //     xOutTensor_ = xOutQueue_.AllocTensor<ExpandXOutType>();
        //     QuantProcess();
        //     xOutQueue_.EnQue(xOutTensor_);
        //     xOutTensor_ = xOutQueue_.DeQue<ExpandXOutType>();
        //
        //     if (isShareExpertRankFlag_) {
        //         LocalTensor<float> xOutFp32Tensor = xOutTensor_.template ReinterpretCast<float>();
        //         DataCopyExtParams dataCopyParamsFloat = {1U, sizeof(float), 0U, 0U, 0U};
        //         DataCopyPad(dstScaleGMTensor[tokenIndex], xOutFp32Tensor[axisH_ / sizeof(float)],
        //                     dataCopyParamsFloat);
        //         DataCopy(dstWinGMTensor[tokenIndex * axisH_], xOutTensor_, axisH_);
        //     } else {
        //         DataCopy(dstWinGMTensor[(tokenIndex - preCnt) * axisH_], xOutTensor_, axisH_);
        //     }
        //     xOutQueue_.FreeTensor(xOutTensor_);
        // } else {
        //     xTmpTensor_ = xQueue_.AllocTensor<ExpandXOutType>();
        //     DataCopyPad(xTmpTensor_, xGMTensor_[tokenIndex * axisH_], expandXCopyParams_, copyPadExtParams);
        //     xQueue_.EnQue(xTmpTensor_);
        //     xTmpTensor_ = xQueue_.DeQue<ExpandXOutType>();
        //     if (isShareExpertRankFlag_) {
        //         DataCopy(dstWinGMTensor[tokenIndex * axisH_], xTmpTensor_, axisH_);
        //     } else {
        //         DataCopy(dstWinGMTensor[(tokenIndex - preCnt) * axisH_], xTmpTensor_, axisH_);
        //     }
        //     xQueue_.FreeTensor<ExpandXOutType>(xTmpTensor_);
        // }
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeDispatch<TemplateMC2TypeFunc>::ReduceMaxInplace(
    const LocalTensor<float> &srcLocal, uint32_t count)
{
    constexpr uint64_t elemPerRepeatFp32 = 64UL;
    uint64_t repsFp32 = count / elemPerRepeatFp32;
    uint64_t offsetsFp32 = repsFp32 * elemPerRepeatFp32;
    uint64_t remsFp32 = count % elemPerRepeatFp32;
    if (likely(repsFp32 > 1)) {
        Max(srcLocal, srcLocal[elemPerRepeatFp32], srcLocal, elemPerRepeatFp32, repsFp32 - 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(remsFp32 > 0) && unlikely(offsetsFp32 > 0)) {
        Max(srcLocal, srcLocal[offsetsFp32], srcLocal, remsFp32, 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    uint32_t mask = (repsFp32 > 0) ? elemPerRepeatFp32 : count;
    WholeReduceMax(srcLocal, srcLocal, mask, 1, 8, 1, 8);
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeDispatch<TemplateMC2TypeFunc>::QuantProcess()
{
    float dynamicScale = 0.0;
    LocalTensor<float> floatLocalTemp;
    floatLocalTemp = receiveDataCastFloatBuf_.Get<float>();

    Cast(floatLocalTemp, xInTensor_, RoundMode::CAST_NONE, axisH_);
    xInQueue_.FreeTensor<XType>(xInTensor_);
    PipeBarrier<PIPE_V>();

    if constexpr (DynamicQuant) {
        LocalTensor<float> floatLocalAbsTemp = smoothScalesBuf_.Get<float>();

        Abs(floatLocalAbsTemp, floatLocalTemp, axisH_);
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(floatLocalAbsTemp, axisH_);

        SyncFunc<AscendC::HardEvent::V_S>();
        dynamicScale = float(127.0) / floatLocalAbsTemp.GetValue(0);
        SyncFunc<AscendC::HardEvent::S_V>();
        Muls(floatLocalTemp, floatLocalTemp, dynamicScale, axisH_);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<half> halfLocalTemp = floatLocalTemp.ReinterpretCast<half>();
    LocalTensor<int32_t> int32LocalTemp = floatLocalTemp.ReinterpretCast<int32_t>();

    Cast(int32LocalTemp, floatLocalTemp, RoundMode::CAST_RINT, axisH_);
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, axisH_);
    PipeBarrier<PIPE_V>();
    Cast(xOutTensor_, halfLocalTemp, RoundMode::CAST_TRUNC, axisH_);

    floatLocalTemp = xOutTensor_.template ReinterpretCast<float>();
    floatLocalTemp.SetValue(hOutSizeAlign_ / sizeof(float), float(1.0) / dynamicScale);
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeDispatch<TemplateMC2TypeFunc>::CleanUp()
{
    uint32_t clearAlign = Ceil(moeExpertNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(clearBuf_, clearAlign);

    if (aivId_ == 0) {
        LocalTensor<int32_t> cleanTempLt = clearBuf_.GetWithOffset<int32_t>(moeExpertNum_, 0);
        Duplicate<int32_t>(cleanTempLt, 0, moeExpertNum_);
        PipeBarrier<PIPE_ALL>();
        DataCopy(LocalNotifyDataTensor_, cleanTempLt, moeExpertNum_);
        PipeBarrier<PIPE_ALL>();
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void ZbMoeDistributeDispatch<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        LocalLayout();
        SyncAll<true>();
        SetLayoutStatus();
        SendCountNotify();
        SyncAll<true>();
        WaitNotify();
        SyncAll<true>();
        InputToDstOutput();
        // SetSyncFlag();
        // WaitSyncFlag();
        // SyncAll<true>();
        CleanUp();
    }
}

}  // namespace ZbMoeDistributeDispatchImpl
#endif  // ZB_MOE_DISTRIBUTE_DISPATCH_H
