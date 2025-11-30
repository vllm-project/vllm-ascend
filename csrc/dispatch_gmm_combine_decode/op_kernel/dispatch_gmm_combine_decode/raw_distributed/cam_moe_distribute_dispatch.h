/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CAM_MOE_DISTRIBUTE_DISPATCH_H
#define CAM_MOE_DISTRIBUTE_DISPATCH_H
#define OPT_RANK_OFFSET 512

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../dispatch_gmm_combine_decode_base.h"
#include "../../dispatch_gmm_combine_decode_tiling.h"

namespace MoeDistributeDispatchImpl {
constexpr uint8_t BUFFER_NUM = 2;             // 多buf
constexpr uint32_t STATE_OFFSET = 512;        // 状态空间偏移地址
constexpr uint32_t STATE_SIZE = 1024 * 1024;  // 1M
constexpr uint32_t UB_ALIGN = 32;             // UB按32字节对齐
constexpr uint32_t SELF_STATE_OFFSET = 256 * 1024;
constexpr uint8_t COMM_NUM = 2;  // 通信域大小
constexpr uint8_t COMM_EP_IDX = 0;
constexpr uint8_t COMM_TP_IDX = 1;
constexpr uint32_t GATHER_NUM_PER_TIME = 6;
// 先写死这个偏移，如果TP固定为2，可直接往起始数据偏移开始读写
constexpr uint64_t WIN_STATE_OFFSET = 512 * 1024;
constexpr uint64_t STATE_WIN_OFFSET = 900 * 1024;
constexpr uint32_t TP_STATE_SIZE = 100 * 1024;
constexpr int CAM_MAX_RANK_SIZE = 384;                // Cam通信库最大支持的npu卡数
constexpr int64_t IPC_DATA_OFFSET = 2 * 1024 * 1024;  // 前2MB作为flag标志位，之后100MB作为数据存储

// 循环优化相关变量
using countType = uint8_t;  // 循环优化使用的数据类型
constexpr uint32_t LOOP_OPT_MAX_BS = 64;
constexpr uint32_t LOOP_OPT_MAX_MOE_RANK = 256;
constexpr uint32_t TOPK_ELEM_COUNT_PER_BLOCK = UB_ALIGN / sizeof(int32_t);
constexpr uint32_t TABLE_ELEM_COUNT_PER_BLOCK = UB_ALIGN / sizeof(countType);
constexpr uint32_t INT32_NUM_PER_BLOCK = UB_ALIGN / sizeof(int32_t);

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define TemplateDispatchTypeClass                                                                          \
    typename XType, typename ExpandXOutType, bool StaticQuant, bool DynamicQuant, bool IsSmoothScaleExist, \
        bool IsNeedAllgater
#define TemplateDispatchTypeFunc XType, ExpandXOutType, StaticQuant, DynamicQuant, IsSmoothScaleExist, IsNeedAllgater

using namespace AscendC;
template <TemplateDispatchTypeClass>
class CamMoeDistributeDispatch
{
public:
    __aicore__ inline CamMoeDistributeDispatch(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expandXOut,
                                GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expertTokenNumsOut,
                                GM_ADDR sendCountsOut, GM_ADDR outputRecvCount, GM_ADDR tpSendCountsOut,
                                GM_ADDR workspaceGM, TPipe *pipe, const DispatchGmmCombineDecodeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SendToSharedExpert();
    __aicore__ inline void SendToMoeExpert();
    __aicore__ inline void AlltoAllDispatch();
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void QuantProcess(uint32_t expertIndex);
    __aicore__ inline void LocalSharedExpertCopyWindow(uint32_t rankIndex, uint32_t tokenOffset,
                                                       uint32_t currendTokenIndex, uint32_t &dynamicScalesLocalIdx);
    __aicore__ inline void SetStatus();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void GetCumSum(LocalTensor<int32_t> &inLocal, LocalTensor<int32_t> &outLocal, int32_t totalCount,
                                     GM_ADDR gmOutputRecvCount);
    __aicore__ inline void CreateZeroTensor(LocalTensor<uint32_t> &outTensor);
    __aicore__ inline void AllGatherSetStatusAndWait();
    __aicore__ inline void ResetStatus();
    __aicore__ inline void QuantInit(GM_ADDR scales);
    __aicore__ inline void AllgatherProcessOut();
    __aicore__ inline void UpdataMultiMoeTokenNumsOut();
    __aicore__ inline void UpdataTokenNumsOut();
    __aicore__ inline GM_ADDR GetWindAddrByRankId(uint8_t ctxIdx, const int32_t rankId)
    {
        uint32_t curRankId = ctxIdx == COMM_EP_IDX ? epRankId_ : tpRankId_;
        if (curRankId == rankId) {
            return (GM_ADDR)(winContext_[ctxIdx]->localWindowsIn) + winDataSizeOffset_ + rankId * OPT_RANK_OFFSET;
        }
        return (GM_ADDR)(((HcclRankRelationResV2 *)(winContext_[ctxIdx]->remoteRes[rankId].nextDevicePtr))->windowsIn) +
               winDataSizeOffset_ + rankId * OPT_RANK_OFFSET;
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(uint8_t ctxIdx, const int32_t rankId)
    {
        uint32_t curRankId = ctxIdx == COMM_EP_IDX ? epRankId_ : tpRankId_;
        if (curRankId == rankId) {
            return (GM_ADDR)(winContext_[ctxIdx]->localWindowsExp) + dataState_ * WIN_STATE_OFFSET;
        }
        return (GM_ADDR)(((HcclRankRelationResV2 *)(winContext_[ctxIdx]->remoteRes[rankId].nextDevicePtr))
                             ->windowsExp) +
               dataState_ * WIN_STATE_OFFSET;
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return (x < y) ? x : y;
    }
    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<float> scalesGMTensor_;
    GlobalTensor<ExpandXOutType> expandXOutGMTensor_;
    GlobalTensor<float> dynamicScalesOutGMTensor_;
    GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
    GlobalTensor<ExpandXOutType> windowInQuantTensor_;
    GlobalTensor<int32_t> windowInstatusTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_;
    GlobalTensor<ExpandXOutType> winTpGatherOutGMTensor_;
    GlobalTensor<float> fpWinTpGatherOutGMTensor_;
    GlobalTensor<int32_t> winTpEpCntGMTensor_;
    LocalTensor<ExpandXOutType> xTmpTensor_;
    LocalTensor<int32_t> tpTmpTensor_;
    LocalTensor<XType> xInTensor_;
    LocalTensor<ExpandXOutType> xOutTensor_;
    LocalTensor<float> xOutFp32Tensor_;
    LocalTensor<int32_t> expertCountTensor_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<int32_t> receivestatusTensor_;
    LocalTensor<float> rowMaxTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<float> dynamicScalesTensor_;
    TBuf<> dynamicScalesBuf_;
    TBuf<> expertCountBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> statusBuf_;
    TBuf<> gatherMaskOutBuf_;  // gather mask输出buf
    TBuf<> getTotalBuf_;       // 计算totalCnt
    TBuf<> scalarBuf_;         // 辅助gather tensor定义
    TBuf<> rowMaxBuf_;
    TBuf<> receiveDataCastFloatBuf_;
    TBuf<> smoothScalesBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_;  // 非量化使用，量化场景接收也可使用
    TQue<QuePosition::VECIN, 1> xInQueue_;                         // 量化使用，量化前的输入
    TQue<QuePosition::VECOUT, 1> xOutQueue_;                       // 量化使用，量化后的输出
    GM_ADDR expandXOutGM_;
    GM_ADDR expandIdxOutGM_;
    GM_ADDR expertTokenNumsOutGM_;  // 这个输出没有使用
    GM_ADDR sendCountsOutGM_;
    GM_ADDR outputRecvCountGM_;
    GM_ADDR sendTpCountOutGM_;
    GM_ADDR statusSpaceGm_;
    GM_ADDR windowGM_;
    GM_ADDR tpWindowGM_;
    GM_ADDR tpStatusWindowGM_;
    GM_ADDR tpLocalWindowGM_;
    GM_ADDR tpLocalStatusWindowGM_;
    GlobalTensor<GM_ADDR> peerMemsAddrGm_;
    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t axisBS_{0};
    uint32_t axisMaxBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t sharedUsedAivNum_{0};
    uint32_t moeUsedAivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t tpWorldSize_{0};
    uint32_t epRankId_{0};
    uint32_t tpGatherRankId_{0};       // gather 对端ID
    uint32_t tpRankId_{0};             // 本卡 ID
    uint32_t aivId_{0};                // aiv id
    uint32_t sharedExpertRankNum_{0};  // 共享专家卡数
    uint32_t moeExpertRankNum_{0};     // moe专家卡数，等于worldSize_ - 共享专家卡数
    uint32_t moeExpertNumPerRank_{0};
    uint32_t moeExpertNum_{0};
    uint32_t totalExpertNum_{0};
    uint32_t bufferSizePerRank_{0};
    uint32_t recvWinBlockNum_{0};
    uint32_t hSize_{0};
    uint32_t hOutSize_{0};
    uint32_t hCommuSize_{0};
    uint32_t scaleParamPad_{0};
    uint32_t axisHCommu_{0};
    uint32_t startExpertId_;
    uint32_t endExpertId_;
    uint32_t sendExpertNum_;
    uint32_t localCopyCoreNum_;
    uint32_t totalCnt_;
    uint32_t lastCore_{0};
    uint32_t dataState_{0};
    uint32_t stateOffset_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t windyquantOffset_;
    bool isShareExpertRank_ = false;
    bool isQuant_ = false;
    float sumTarget_;
    uint64_t totalWinSize_{0};
    uint32_t gatherCount_{0};
    uint32_t expertTokenNumsType_{1};
    uint32_t preCnt_{0};
    __gm__ HcclOpResParam *winContext_[COMM_NUM]{nullptr, nullptr};
    // 循环优化使用的变量
    TBuf<> sendTableIdsBuf_;
    LocalTensor<countType> tableLocalTensor_;
    LocalTensor<countType> sendCountLocalTensor_;
    uint32_t moeExpertRankNumAligned_;
    uint32_t moeExpertRankNumInt16Aligned_;
    uint32_t tableElemCount_;
    bool enableAivOpt_{false};
};

template <TemplateDispatchTypeClass>
__aicore__ inline void CamMoeDistributeDispatch<TemplateDispatchTypeFunc>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR outputRecvCount, GM_ADDR tpSendCountsOut,
    GM_ADDR workspaceGM, TPipe *pipe, const DispatchGmmCombineDecodeTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    epRankId_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.epRankId;
    GlobalTensor<int32_t> selfDataStatusTensor;
    GM_ADDR statusDataSpaceGm;

    winContext_[COMM_EP_IDX] = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext_[COMM_TP_IDX] = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<1>();  // 没有相关公共宏

    statusDataSpaceGm = (GM_ADDR)(winContext_[COMM_EP_IDX]->localWindowsExp);
    selfDataStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusDataSpaceGm + STATE_WIN_OFFSET));

    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        selfDataStatusTensor[aivId_ * UB_ALIGN]);
    __asm__ __volatile__("");
    dataState_ = selfDataStatusTensor(aivId_ * UB_ALIGN);
    if (dataState_ == 0) {
        selfDataStatusTensor(aivId_ * UB_ALIGN) = 1;
    } else {
        selfDataStatusTensor(aivId_ * UB_ALIGN) = 0;
    }
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        selfDataStatusTensor[aivId_ * UB_ALIGN]);
    __asm__ __volatile__("");
    pipe_barrier(PIPE_ALL);
    axisBS_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.bs;
    axisH_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.h;
    epWorldSize_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.epRankSize;
    // axisMaxBS_ = axisBS_;
    axisMaxBS_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.globalBs / epWorldSize_;
    moeExpertNum_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNum;
    sharedExpertRankNum_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.sharedExpertRankNum;
    expertTokenNumsType_ = 0;
    totalWinSize_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.totalWinSize;
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    moeExpertNumPerRank_ = moeExpertNum_ / moeExpertRankNum_;
    expertPerSizeOnWin_ = axisMaxBS_ * axisH_ * sizeof(XType);
    winDataSizeOffset_ = dataState_ * epWorldSize_ * expertPerSizeOnWin_ * moeExpertNumPerRank_;
    tpRankId_ = 0;
    windowGM_ = GetWindAddrByRankId(COMM_EP_IDX, epRankId_);
    statusSpaceGm_ = GetWindStateAddrByRankId(COMM_EP_IDX, epRankId_);
    tpGatherRankId_ = tpRankId_ == 0 ? 1 : 0;
    axisK_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.k;
    aivNum_ = 48;
    tpWorldSize_ = 1;
    xGMTensor_.SetGlobalBuffer((__gm__ XType *)x);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    expandXOutGMTensor_.SetGlobalBuffer((__gm__ ExpandXOutType *)expandXOut);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ float *)dynamicScalesOut);
    expertTokenNumsOutGMTensor_.SetGlobalBuffer((__gm__ int64_t *)expertTokenNumsOut);
    windowInQuantTensor_.SetGlobalBuffer((__gm__ ExpandXOutType *)windowGM_);
    windowInstatusTensor_.SetGlobalBuffer((__gm__ int32_t *)(statusSpaceGm_));
    windowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)(statusSpaceGm_));
    if constexpr (IsNeedAllgater) {
        tpLocalWindowGM_ = GetWindAddrByRankId(COMM_TP_IDX, tpRankId_);
        tpLocalStatusWindowGM_ = GetWindStateAddrByRankId(COMM_TP_IDX, tpRankId_);
        tpWindowGM_ = GetWindAddrByRankId(COMM_TP_IDX, tpGatherRankId_);
        tpStatusWindowGM_ = GetWindStateAddrByRankId(COMM_TP_IDX, tpGatherRankId_);
        winTpGatherOutGMTensor_.SetGlobalBuffer((__gm__ ExpandXOutType *)tpWindowGM_);
        fpWinTpGatherOutGMTensor_.SetGlobalBuffer((__gm__ float *)tpWindowGM_);
        winTpEpCntGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(tpStatusWindowGM_ + TP_STATE_SIZE));
    }
    expandXOutGM_ = expandXOut;
    expandIdxOutGM_ = expandIdxOut;    // 无GlobalTensor
    sendCountsOutGM_ = sendCountsOut;  // 无GlobalTensor
    outputRecvCountGM_ = outputRecvCount;
    sendTpCountOutGM_ = tpSendCountsOut;
    isQuant_ = StaticQuant | DynamicQuant;
    hSize_ = axisH_ * sizeof(XType);
    hOutSize_ = axisH_ * sizeof(ExpandXOutType);  // 如有量化，需要量化后通信
    scaleParamPad_ = (isQuant_ ? 128 : 0);        // 预留128B给量化参数，实际只使用了4B(fp32)
    hCommuSize_ = hOutSize_ + scaleParamPad_;
    axisHCommu_ = hCommuSize_ / sizeof(ExpandXOutType);
    if (sharedExpertRankNum_ != 0) {                 // 后面的卡才需要发给共享专家发数据
        sharedUsedAivNum_ = aivNum_ / (axisK_ + 1);  // 均等分，取整
        if (sharedUsedAivNum_ == 0) {
            sharedUsedAivNum_ = 1;
        }
    }
    moeUsedAivNum_ = aivNum_ - sharedUsedAivNum_;
    bufferSizePerRank_ = 32 * hSize_;
    recvWinBlockNum_ = epWorldSize_ * moeExpertNumPerRank_;
    isShareExpertRank_ = (epRankId_ < sharedExpertRankNum_) ? true : false;
    windyquantOffset_ = epWorldSize_ * axisMaxBS_ * hOutSize_;
    GlobalTensor<int32_t> selfStatusTensor;
    selfStatusTensor.SetGlobalBuffer((__gm__ int32_t *)(statusSpaceGm_ + SELF_STATE_OFFSET));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        selfStatusTensor[aivId_ * UB_ALIGN]);
    int32_t state = selfStatusTensor(aivId_ * UB_ALIGN);
    stateOffset_ = (recvWinBlockNum_ > 512) ? (STATE_OFFSET / 2) : STATE_OFFSET;
    tpipe_->InitBuffer(statusBuf_, recvWinBlockNum_ * UB_ALIGN);  // expertNum * 32B
    statusTensor_ = statusBuf_.Get<int32_t>();  // 保存发送数据量及flag，同时用于计算windows中的偏移
    Duplicate<int32_t>(statusTensor_, 0, recvWinBlockNum_ * 8);  // 8 = UB_ALIGN / sizeof(int32_t)
    if (state == 0) {
        sumTarget_ = (float)1.0;
        selfStatusTensor(aivId_ * UB_ALIGN) = 0x3F800000;
        uint64_t mask[2] = {0x101010101010101, 0};  // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0x3F800000
        Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, recvWinBlockNum_ / 8, 1, 8);  // 0x3F800000是float的1
    } else {
        sumTarget_ = 0.0;
        selfStatusTensor(aivId_ * UB_ALIGN) = 0;
    }
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        selfStatusTensor[aivId_ * UB_ALIGN]);
    tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hCommuSize_);  // 14k *2
    if (isQuant_) {
        QuantInit(scales);
    }
    uint32_t expertIdsSize = Ceil(axisBS_ * axisK_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;  // 约束32对齐
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsSize);                                        // BS * K * 4
    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();
    tpipe_->InitBuffer(expertCountBuf_, expertIdsSize);  // BS * K * 4
    expertCountTensor_ = expertCountBuf_.Get<int32_t>();

    tpipe_->InitBuffer(gatherMaskOutBuf_, recvWinBlockNum_ * sizeof(float));  // worldsize * 4B
    tpipe_->InitBuffer(getTotalBuf_,
                       epWorldSize_ * moeExpertNumPerRank_ * sizeof(int32_t));  // worldsize * 单卡专家数 * 4B
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 2);                               // 72B

    moeExpertRankNumAligned_ = Ceil(moeExpertNum_, TABLE_ELEM_COUNT_PER_BLOCK) * TABLE_ELEM_COUNT_PER_BLOCK;
    if (axisBS_ <= LOOP_OPT_MAX_BS && moeExpertRankNumAligned_ <= LOOP_OPT_MAX_MOE_RANK &&
        axisK_ % TOPK_ELEM_COUNT_PER_BLOCK == 0) {
        // UB空间限制BS不大于64、路由专家数量不大于256；对齐要求限制axisK_是8的倍数
        enableAivOpt_ = true;
        moeExpertRankNumInt16Aligned_ = moeExpertRankNumAligned_ / 2;  // 每个int16_t装2个uint8_t
        tableElemCount_ = (axisBS_ + 1) * moeExpertRankNumAligned_;    // 额外加一行（首行全0）

        tpipe_->InitBuffer(sendTableIdsBuf_, tableElemCount_ * sizeof(countType));
        tableLocalTensor_ = sendTableIdsBuf_.Get<countType>();
        sendCountLocalTensor_ = tableLocalTensor_[axisBS_ * moeExpertRankNumAligned_];  // 计算完成后，最后一行为count
    }
}

template <TemplateDispatchTypeClass>
__aicore__ inline void CamMoeDistributeDispatch<TemplateDispatchTypeFunc>::QuantInit(GM_ADDR scales)
{
    tpipe_->InitBuffer(xInQueue_, BUFFER_NUM, hSize_);        // 14K *2
    tpipe_->InitBuffer(xOutQueue_, BUFFER_NUM, hCommuSize_);  // 7K *2
    scalesGMTensor_.SetGlobalBuffer((__gm__ float *)scales);
    uint32_t hFp32Size = axisH_ * sizeof(float);
    if constexpr (DynamicQuant) {
        tpipe_->InitBuffer(rowMaxBuf_, UB_ALIGN);  // 32B
    }
    tpipe_->InitBuffer(receiveDataCastFloatBuf_, 1 * hFp32Size);   // 28KB
    tpipe_->InitBuffer(smoothScalesBuf_, axisH_ * sizeof(float));  // 28KB
    smoothScalesTensor_ = smoothScalesBuf_.Get<float>();
    tpipe_->InitBuffer(dynamicScalesBuf_, axisBS_ * sizeof(float));  // 32 * 4
    dynamicScalesTensor_ = dynamicScalesBuf_.Get<float>();
}

template <TemplateDispatchTypeClass>
__aicore__ inline void CamMoeDistributeDispatch<TemplateDispatchTypeFunc>::SendToSharedExpert()
{
    uint32_t sendTokenNum = axisBS_ / sharedUsedAivNum_;       // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = axisBS_ % sharedUsedAivNum_;  // 余数
    uint32_t newAivId = aivId_ - moeUsedAivNum_;  // 由于是后面的核作为发送的共享专家，因此需要换算
    uint32_t startTokenId = sendTokenNum * newAivId;  // 每个aiv发送时的起始rankid
    if (newAivId < remainderTokenNum) {               // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    if (startTokenId >= axisBS_) {
        return;
    }
    uint32_t endTokenId = startTokenId + sendTokenNum;
    for (uint32_t tokenShuffleIndex = 0; tokenShuffleIndex < sendTokenNum; ++tokenShuffleIndex) {
        uint32_t tokenIndex = startTokenId + ((tokenShuffleIndex + epRankId_) % sendTokenNum);
        uint32_t temp = (epRankId_ * axisBS_) / sharedExpertRankNum_;
        uint32_t moeOnShareRank = Ceil((tokenIndex + 1 + temp) * sharedExpertRankNum_, axisBS_) - 1 - epRankId_;  // dst
        uint32_t preCnt = (moeOnShareRank + epRankId_) * axisBS_ / sharedExpertRankNum_ -
                          epRankId_ * axisBS_ / sharedExpertRankNum_;  // 发给该共享专家已经有多少token数据
        GlobalTensor<ExpandXOutType> dstWinGMTensor;
        dstWinGMTensor.SetGlobalBuffer((__gm__ ExpandXOutType *)(GetWindAddrByRankId(COMM_EP_IDX, moeOnShareRank) +
                                                                 expertPerSizeOnWin_ * epRankId_));
        if constexpr (DynamicQuant || StaticQuant) {
            xInTensor_ = xInQueue_.AllocTensor<XType>();
            DataCopy(xInTensor_, xGMTensor_[tokenIndex * axisH_], axisH_);  // 约束对齐
            xInQueue_.EnQue(xInTensor_);
            xInTensor_ = xInQueue_.DeQue<XType>();
            xOutTensor_ = xOutQueue_.AllocTensor<ExpandXOutType>();
            QuantProcess(0);
            xOutQueue_.EnQue(xOutTensor_);

            xOutTensor_ = xOutQueue_.DeQue<ExpandXOutType>();
            if (isShareExpertRank_) {
                xOutFp32Tensor_ = xOutTensor_.template ReinterpretCast<float>();
                DataCopyExtParams dataCopyParamsFloat = {1U, sizeof(float), 0U, 0U, 0U};
                DataCopyPad(dynamicScalesOutGMTensor_[tokenIndex], xOutFp32Tensor_[axisH_ / sizeof(float)],
                            dataCopyParamsFloat);
                if constexpr (IsNeedAllgater) {
                    DataCopy(winTpGatherOutGMTensor_[tokenIndex * axisHCommu_], xOutTensor_, axisHCommu_);  // 约束对齐
                }
                DataCopy(expandXOutGMTensor_[tokenIndex * axisH_], xOutTensor_, axisH_);  // 约束对齐
            } else {
                DataCopy(dstWinGMTensor[(tokenIndex - preCnt) * axisHCommu_], xOutTensor_, axisHCommu_);  // 约束对齐
            }
            xOutQueue_.FreeTensor(xOutTensor_);
        } else {
            xTmpTensor_ = xQueue_.AllocTensor<ExpandXOutType>();
            DataCopy(xTmpTensor_, xGMTensor_[tokenIndex * axisH_], axisH_);  // 约束对齐
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<ExpandXOutType>();
            if (isShareExpertRank_) {
                if constexpr (IsNeedAllgater) {
                    DataCopy(winTpGatherOutGMTensor_[tokenIndex * axisHCommu_], xTmpTensor_, axisHCommu_);
                }
                DataCopy(expandXOutGMTensor_[tokenIndex * axisHCommu_], xTmpTensor_, axisHCommu_);
            } else {
                DataCopy(dstWinGMTensor[(tokenIndex - preCnt) * axisHCommu_], xTmpTensor_, axisHCommu_);  // 约束对齐
            }
            xQueue_.FreeTensor<ExpandXOutType>(xTmpTensor_);
        }
    }
}

template <TemplateDispatchTypeClass>
__aicore__ inline void CamMoeDistributeDispatch<TemplateDispatchTypeFunc>::SendToMoeExpert()
{
    uint32_t expertIdsCnt = axisBS_ * axisK_;
    uint32_t sendTokenNum = expertIdsCnt / moeUsedAivNum_;       // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = expertIdsCnt % moeUsedAivNum_;  // 余数
    uint32_t startTokenId = sendTokenNum * aivId_;               // 每个aiv发送时的起始rankid
    if (aivId_ < remainderTokenNum) {  // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += aivId_;
    } else {
        startTokenId += remainderTokenNum;
    }
    uint32_t endTokenId = startTokenId + sendTokenNum;
    GlobalTensor<ExpandXOutType> dstWinGMTensor;
    for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; ++tokenIndex) {
        int32_t dstExpertId = expertIdsTensor_(tokenIndex);
        if (dstExpertId < 0) {
            continue;
        }
        uint32_t tempRankId = dstExpertId / moeExpertNumPerRank_ + sharedExpertRankNum_;
        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindAddrByRankId(COMM_EP_IDX, tempRankId) +
                                            (expertPerSizeOnWin_ *
                                             (epRankId_ * moeExpertNumPerRank_ + dstExpertId % moeExpertNumPerRank_)) +
                                            hCommuSize_ * expertCountTensor_(tokenIndex));  // 计算地址偏移
        dstWinGMTensor.SetGlobalBuffer((__gm__ ExpandXOutType *)rankGM);
        if constexpr (DynamicQuant || StaticQuant) {
            xInTensor_ = xInQueue_.AllocTensor<XType>();
            DataCopy(xInTensor_, xGMTensor_[tokenIndex / axisK_ * axisH_], axisH_);  // 约束对齐
            xInQueue_.EnQue(xInTensor_);
            xInTensor_ = xInQueue_.DeQue<XType>();
            xOutTensor_ = xOutQueue_.AllocTensor<ExpandXOutType>();
            uint32_t expertIndex = sharedExpertRankNum_ != 0 ? (dstExpertId + 1) : dstExpertId;
            QuantProcess(expertIndex);
            xOutQueue_.EnQue(xOutTensor_);

            xOutTensor_ = xOutQueue_.DeQue<ExpandXOutType>();
            DataCopy(dstWinGMTensor, xOutTensor_, axisHCommu_);  // 约束对齐
            xOutQueue_.FreeTensor(xOutTensor_);
        } else {
            xTmpTensor_ = xQueue_.AllocTensor<ExpandXOutType>();
            DataCopy(xTmpTensor_, xGMTensor_[tokenIndex / axisK_ * axisH_], axisH_);  // 约束对齐
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<ExpandXOutType>();
            DataCopy(dstWinGMTensor, xTmpTensor_, axisHCommu_);  // 约束对齐
            xQueue_.FreeTensor<ExpandXOutType>(xTmpTensor_);
        }
    }
    if (aivId_ == (moeUsedAivNum_ - 1) && (!enableAivOpt_)) {
        // 不启用循环优化时，这里才需要写出结果
        GlobalTensor<int32_t> expandIdxGMTensor;
        expandIdxGMTensor.SetGlobalBuffer((__gm__ int32_t *)expandIdxOutGM_);
        DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt * sizeof(uint32_t)), 0U, 0U, 0U};
        DataCopyPad(expandIdxGMTensor, expertCountTensor_, expertIdsCntParams);
    }
}

template <TemplateDispatchTypeClass>
__aicore__ inline void CamMoeDistributeDispatch<TemplateDispatchTypeFunc>::AlltoAllDispatch()
{
    uint32_t expertIdsCnt = axisBS_ * axisK_;
    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, copyPadParams);
    AscendC::TQueSync<PIPE_MTE2, PIPE_S> expertCntLocalSync;
    expertCntLocalSync.SetFlag(0);
    expertCntLocalSync.WaitFlag(0);
    if (enableAivOpt_) {
        LocalTensor<int16_t> tableInt16LocalTensor_ = tableLocalTensor_.template ReinterpretCast<int16_t>();
        Duplicate(tableInt16LocalTensor_, (int16_t)0, tableElemCount_ / 2);  // 清零
        SyncFunc<AscendC::HardEvent::V_S>();
        for (int tokenIndex = 0; tokenIndex < expertIdsCnt; ++tokenIndex) {  // 填表。默认为0，发送置1
            int expertId = expertIdsTensor_(tokenIndex);
            if (expertId < 0) {
                continue;
            }
            tableLocalTensor_((tokenIndex / axisK_ + 1) * moeExpertRankNumAligned_ + expertId) = 1;
        }
        pipe_barrier(PIPE_ALL);

        // 分核，确定每个核要处理的token
        uint32_t sendTokenNum = expertIdsCnt / moeUsedAivNum_;
        uint32_t remainderTokenNum = expertIdsCnt % moeUsedAivNum_;
        uint32_t startTokenId = sendTokenNum * aivId_;
        if (aivId_ < remainderTokenNum) {
            sendTokenNum += 1;
            startTokenId += aivId_;
        } else {
            startTokenId += remainderTokenNum;
        }
        uint32_t endTokenId = startTokenId + sendTokenNum;
        uint32_t startTokenRow = startTokenId / axisK_;
        uint32_t endTokenRow = (endTokenId + axisK_ - 1) / axisK_;

        for (int row = 1; row <= axisBS_; ++row) {
            Add(tableInt16LocalTensor_[row * moeExpertRankNumInt16Aligned_],
                tableInt16LocalTensor_[row * moeExpertRankNumInt16Aligned_],
                tableInt16LocalTensor_[(row - 1) * moeExpertRankNumInt16Aligned_], moeExpertRankNumInt16Aligned_);
            pipe_barrier(PIPE_V);
        }

        // 计算完成后，下标为的i的行为下标为i+1的token在远端的偏移，最后一行为总count
        GlobalTensor<int32_t> expandIdxGMTensor;
        if (aivId_ < moeUsedAivNum_) {
            SyncFunc<AscendC::HardEvent::V_S>();
            for (int row = startTokenRow; row < endTokenRow; ++row) {
                for (int expertIndex = 0; expertIndex < axisK_; ++expertIndex) {
                    int32_t expertId = expertIdsTensor_(row * axisK_ + expertIndex);
                    if (expertId < 0) {
                        continue;
                    }
                    expertCountTensor_(row * axisK_ + expertIndex) =
                        (int32_t)tableLocalTensor_(row * moeExpertRankNumAligned_ + expertId);
                }
                SyncFunc<AscendC::HardEvent::S_MTE3>();
                expandIdxGMTensor.SetGlobalBuffer(
                    (__gm__ int32_t *)(expandIdxOutGM_ + row * axisK_ * sizeof(uint32_t)));
                DataCopy(expandIdxGMTensor, expertCountTensor_[row * axisK_], axisK_);
            }
        }

        // 分核，确定每个核要set status的rank
        uint32_t preTotalExpertNum = sharedExpertRankNum_ + moeExpertNum_;
        uint32_t preSendExpertNum = preTotalExpertNum / aivNum_;
        uint32_t preRemainderRankNum = preTotalExpertNum % aivNum_;
        uint32_t preStartExpertId = preSendExpertNum * aivId_;
        if (aivId_ < preRemainderRankNum) {
            preSendExpertNum += 1;
            preStartExpertId += aivId_;
        } else {
            preStartExpertId += preRemainderRankNum;
        }
        uint32_t preEndExpertId = preStartExpertId + preSendExpertNum;
        preStartExpertId = preStartExpertId >= sharedExpertRankNum_ ? preStartExpertId : sharedExpertRankNum_;

        SyncFunc<AscendC::HardEvent::V_S>();
        for (int32_t tmpExpertId = preStartExpertId; tmpExpertId < preEndExpertId; ++tmpExpertId) {
            statusTensor_(tmpExpertId * INT32_NUM_PER_BLOCK + 1) =
                (int32_t)sendCountLocalTensor_(tmpExpertId - sharedExpertRankNum_);
        }
    } else {
        for (uint32_t tokenIndex = 0; tokenIndex < expertIdsCnt; ++tokenIndex) {
            // 防止越界，越界判断(expertId >= epWorldSize_) || (expertId < sharedExpertRankNum_)
            int32_t expertId = expertIdsTensor_(tokenIndex) + sharedExpertRankNum_;
            if (expertId < 0) {
                continue;
            }
            expertCountTensor_(tokenIndex) = statusTensor_(expertId * INT32_NUM_PER_BLOCK + 1);
            statusTensor_(expertId * INT32_NUM_PER_BLOCK + 1)++;
        }
    }
    if (!isShareExpertRank_) {
        for (uint32_t curSatatusExpId = 0; curSatatusExpId < sharedExpertRankNum_; ++curSatatusExpId) {
            int32_t curExpertCnt = (curSatatusExpId + 1 + epRankId_) * axisBS_ / sharedExpertRankNum_ -
                                   (curSatatusExpId + epRankId_) * axisBS_ / sharedExpertRankNum_;
            statusTensor_((curSatatusExpId)*INT32_NUM_PER_BLOCK + 1) = curExpertCnt;
        }
    }
    if ((sharedExpertRankNum_ != 0) && (aivId_ >= moeUsedAivNum_)) {  // 后面的核进行发给共享专家
        SendToSharedExpert();
        return;
    }
    SendToMoeExpert();
}

template <TemplateDispatchTypeClass>
__aicore__ inline void CamMoeDistributeDispatch<TemplateDispatchTypeFunc>::SetStatus()
{
    pipe_barrier(PIPE_ALL);
    SyncAll<true>();
    totalExpertNum_ = sharedExpertRankNum_ + moeExpertNum_;
    sendExpertNum_ = totalExpertNum_ / aivNum_;  // 每个aiv需要处理的专家数
    uint32_t remainderRankNum = totalExpertNum_ % aivNum_;
    startExpertId_ = sendExpertNum_ * aivId_;  // + sharedExpertRankNum_, 每个aiv发送的起始rankid
    if (aivId_ < remainderRankNum) {           // 前remainderRankNum个aiv需要多发1个卡的数据
        sendExpertNum_ += 1;
        startExpertId_ += aivId_;
    } else {
        startExpertId_ += remainderRankNum;
    }
    endExpertId_ = startExpertId_ + sendExpertNum_;
    if (startExpertId_ >= totalExpertNum_) {  // 多余的核return
        return;
    }
    GlobalTensor<int32_t> rankGMTensor;
    uint32_t offset = stateOffset_ * epRankId_;
    for (uint32_t rankIndex = startExpertId_; rankIndex < endExpertId_; ++rankIndex) {
        uint32_t dstRankId = rankIndex;
        if (moeExpertNumPerRank_ > 1 && (rankIndex >= sharedExpertRankNum_)) {
            dstRankId = ((rankIndex - sharedExpertRankNum_) / moeExpertNumPerRank_ + sharedExpertRankNum_);
            offset =
                (epRankId_ + (rankIndex - sharedExpertRankNum_) % moeExpertNumPerRank_ * epWorldSize_) * stateOffset_;
        }
        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindStateAddrByRankId(COMM_EP_IDX, dstRankId) + offset);  // 计算地址偏移
        rankGMTensor.SetGlobalBuffer((__gm__ int32_t *)rankGM);
        DataCopy<int32_t>(rankGMTensor, statusTensor_[rankIndex * 8], 8UL);  // 8时数据大小，按32对齐拷贝
    }
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>();
}

template <TemplateDispatchTypeClass>
__aicore__ inline void CamMoeDistributeDispatch<TemplateDispatchTypeFunc>::QuantProcess(uint32_t expertIndex)
{
    float dynamicScale = 0.0;
    LocalTensor<float> floatLocalTemp;
    floatLocalTemp = receiveDataCastFloatBuf_.Get<float>();
    Cast(floatLocalTemp, xInTensor_, RoundMode::CAST_NONE, axisH_);
    xInQueue_.FreeTensor<XType>(xInTensor_);
    pipe_barrier(PIPE_V);
    if constexpr (IsSmoothScaleExist) {
        if constexpr (DynamicQuant) {
            SyncFunc<AscendC::HardEvent::V_MTE2>();  // ub复用，循环同步
        }
        DataCopy(smoothScalesTensor_, scalesGMTensor_[expertIndex * axisH_], axisH_);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Mul(floatLocalTemp, floatLocalTemp, smoothScalesTensor_, axisH_);
        pipe_barrier(PIPE_V);
    }
    if constexpr (DynamicQuant) {
        LocalTensor<float> floatLocalAbsTemp = smoothScalesBuf_.Get<float>();
        rowMaxTensor_ = rowMaxBuf_.Get<float>();
        Abs(floatLocalAbsTemp, floatLocalTemp, axisH_);
        pipe_barrier(PIPE_V);
        ReduceMax(rowMaxTensor_, floatLocalAbsTemp, floatLocalAbsTemp, axisH_, false);
        SyncFunc<AscendC::HardEvent::V_S>();
        dynamicScale = float(127.0) / rowMaxTensor_.GetValue(0);
        SyncFunc<AscendC::HardEvent::S_V>();
        Muls(floatLocalTemp, floatLocalTemp, dynamicScale, axisH_);
        pipe_barrier(PIPE_V);
    }
    LocalTensor<half> halfLocalTemp = floatLocalTemp.ReinterpretCast<half>();
    LocalTensor<int32_t> int32LocalTemp = floatLocalTemp.ReinterpretCast<int32_t>();
    Cast(int32LocalTemp, floatLocalTemp, RoundMode::CAST_RINT, axisH_);
    pipe_barrier(PIPE_V);
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    Cast(halfLocalTemp, int32LocalTemp, RoundMode::CAST_ROUND, axisH_);
    pipe_barrier(PIPE_V);
    Cast(xOutTensor_, halfLocalTemp, RoundMode::CAST_TRUNC, axisH_);
    floatLocalTemp = xOutTensor_.template ReinterpretCast<float>();
    floatLocalTemp.SetValue(axisH_ / sizeof(float), float(1.0) / dynamicScale);  // int8->float32
}

template <TemplateDispatchTypeClass>
__aicore__ inline void CamMoeDistributeDispatch<TemplateDispatchTypeFunc>::LocalSharedExpertCopyWindow(
    uint32_t rankIndex, uint32_t tokenOffset, uint32_t currendTokenIndex, uint32_t &dynamicScalesLocalIdx)
{
    xTmpTensor_ = xQueue_.AllocTensor<ExpandXOutType>();
    DataCopy(xTmpTensor_,
             windowInQuantTensor_[rankIndex * (expertPerSizeOnWin_ / sizeof(ExpandXOutType)) +
                                  currendTokenIndex * axisHCommu_],
             axisHCommu_);
    xQueue_.EnQue(xTmpTensor_);
    xTmpTensor_ = xQueue_.DeQue<ExpandXOutType>();
    if constexpr (DynamicQuant || StaticQuant) {
        pipe_barrier(PIPE_ALL);
        xOutFp32Tensor_ = xTmpTensor_.template ReinterpretCast<float>();
        dynamicScalesTensor_.SetValue(dynamicScalesLocalIdx++, xOutFp32Tensor_.GetValue(axisH_ / sizeof(float)));
        pipe_barrier(PIPE_ALL);
    }
    if constexpr (IsNeedAllgater) {
        DataCopy(winTpGatherOutGMTensor_[tokenOffset * axisH_], xTmpTensor_, axisH_);
    }
    DataCopy(expandXOutGMTensor_[tokenOffset * axisH_], xTmpTensor_, axisH_);
    xQueue_.FreeTensor(xTmpTensor_);
}

template <TemplateDispatchTypeClass>
__aicore__ inline void CamMoeDistributeDispatch<TemplateDispatchTypeFunc>::WaitDispatch()
{
    uint32_t rscvStatusNum = isShareExpertRank_ ? epWorldSize_ : recvWinBlockNum_;
    uint32_t recStatusNumPerCore = rscvStatusNum / aivNum_;  // 每个aiv需要处理的专家数
    uint32_t remainderRankNum = rscvStatusNum % aivNum_;
    uint32_t startStatusIndex = recStatusNumPerCore * aivId_;  // + sharedExpertRankNum_, 每个aiv发送的起始rankid
    if (aivId_ < remainderRankNum) {                           // 前remainderRankNum个aiv需要多发1个卡的数据
        recStatusNumPerCore += 1;
        startStatusIndex += aivId_;
    } else {
        startStatusIndex += remainderRankNum;
    }
    if (startStatusIndex >= rscvStatusNum) {
        SyncAll<true>();
        return;
    }
    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
    LocalTensor<uint32_t> gatherTmpTensor = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    gatherTmpTensor.SetValue(0, 1);
    LocalTensor<float> statusSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    statusFp32Tensor_ = statusTensor_.ReinterpretCast<float>();
    uint32_t mask = 1;  // gatherMask + sum 相关参数
    uint64_t rsvdCnt = 0;
    SumParams sumParams{1, recStatusNumPerCore, recStatusNumPerCore};
    float sumOfFlag = static_cast<float>(-1.0);
    float minTarget = (sumTarget_ * recStatusNumPerCore) - (float)0.5;
    float maxTarget = (sumTarget_ * recStatusNumPerCore) + (float)0.5;
    DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore), 1,
                               static_cast<uint16_t>((recvWinBlockNum_ > 512) ? 7 : 15), 0};  // srcStride为15个block
    SyncFunc<AscendC::HardEvent::S_V>();
    while ((sumOfFlag < minTarget) || (sumOfFlag > maxTarget)) {
        DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[startStatusIndex * stateOffset_ / sizeof(float)],
                 intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        GatherMask(gatherMaskOutTensor, statusFp32Tensor_, gatherTmpTensor, true, mask,
                   {1, (uint16_t)recStatusNumPerCore, 1, 0}, rsvdCnt);
        pipe_barrier(PIPE_V);
        Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }
    SyncAll<true>();
}

template <TemplateDispatchTypeClass>
__aicore__ inline void CamMoeDistributeDispatch<TemplateDispatchTypeFunc>::GetCumSum(LocalTensor<int32_t> &inLocal,
                                                                                     LocalTensor<int32_t> &outLocal,
                                                                                     int32_t totalCount,
                                                                                     GM_ADDR gmOutputRecvCount)
{
    statusFp32Tensor_ = statusTensor_.ReinterpretCast<float>();
    DataCopyParams intriParams{static_cast<uint16_t>(recvWinBlockNum_), 1,
                               static_cast<uint16_t>((recvWinBlockNum_ > 512) ? 7 : 15), 0};  // srcStride为15个block
    DataCopy(statusTensor_, windowInstatusTensor_, intriParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    if (isShareExpertRank_) {
        for (uint32_t curSatatusExpId = 0; curSatatusExpId < sharedExpertRankNum_; ++curSatatusExpId) {
            int32_t curExpertCnt = (curSatatusExpId + 1 + epRankId_) * axisBS_ / sharedExpertRankNum_ -
                                   (curSatatusExpId + epRankId_) * axisBS_ / sharedExpertRankNum_;
            statusTensor_((curSatatusExpId)*INT32_NUM_PER_BLOCK + 1) = curExpertCnt;
        }
    }
    outLocal = gatherMaskOutBuf_.Get<int32_t>();  // 内存复用
    LocalTensor<float> getTotalLocal = getTotalBuf_.Get<float>();
    // gather mask在一起
    TBuf<> gatherTmpBuf;
    TBuf<> workLocalBuf;
    tpipe_->InitBuffer(gatherTmpBuf, sizeof(uint32_t) * recvWinBlockNum_ / 4);
    LocalTensor<uint32_t> gatherTmpTensor = gatherTmpBuf.Get<uint32_t>();
    Duplicate(gatherTmpTensor, (uint32_t)33686018, recvWinBlockNum_ / 4);  // 0000 0010 0000 0010 0000 0010 0000 0010
    PipeBarrier<PIPE_V>();
    uint32_t mask = recvWinBlockNum_ * 8;  // 512 / 32
    uint64_t rsvdCnt = 0;
    GatherMask(outLocal, inLocal, gatherTmpTensor, true, mask, {1, 1, 0, 0}, rsvdCnt);
    AscendC::GlobalTensor<int32_t> recvCountTensor;
    recvCountTensor.SetGlobalBuffer((__gm__ int32_t *)gmOutputRecvCount);
    uint32_t localExpertNum = isShareExpertRank_ ? 1 : moeExpertNumPerRank_;
    AscendC::DataCopyExtParams dataCopyParams = {
        1U, static_cast<uint32_t>(localExpertNum * epWorldSize_ * sizeof(int32_t)), 0U, 0U, 0U};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    AscendC::DataCopyPad(recvCountTensor, outLocal.ReinterpretCast<int32_t>(), dataCopyParams);
    SyncFunc<AscendC::HardEvent::MTE3_V>();
    // 再用cumsum累加，按照列相加
    int typeSize = sizeof(int32_t);
    int32_t elementsPerBlock = 32 / typeSize;
    int32_t elementsPerRepeat = 256 / typeSize;
    int32_t firstMaxRepeat = epWorldSize_;
    int32_t iter1OutputCount = firstMaxRepeat;
    int32_t iter1AlignEnd = ((iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
    int32_t finalWorkLocalNeedSize = iter1AlignEnd;
    tpipe_->InitBuffer(workLocalBuf, finalWorkLocalNeedSize * sizeof(int32_t));
    LocalTensor<float> workLocalTensor = workLocalBuf.Get<float>();
    LocalTensor<float> tmpFp32 = outLocal.ReinterpretCast<float>();
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(getTotalLocal, tmpFp32, workLocalTensor, epWorldSize_);
    totalCnt_ = getTotalLocal.ReinterpretCast<int32_t>().GetValue(0);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(tmpFp32, tmpFp32, workLocalTensor, totalCount);
    PipeBarrier<PIPE_V>();
}

template <TemplateDispatchTypeClass>
__aicore__ inline void
CamMoeDistributeDispatch<TemplateDispatchTypeFunc>::CreateZeroTensor(LocalTensor<uint32_t> &outLocal)
{
    TBuf<> outBuf;
    tpipe_->InitBuffer(outBuf, UB_ALIGN);
    outLocal = outBuf.Get<uint32_t>();
    for (uint32_t i = 0; i < 2; i++) {
        outLocal.SetValue(i, 0);
    }
}

template <TemplateDispatchTypeClass>
__aicore__ inline void CamMoeDistributeDispatch<TemplateDispatchTypeFunc>::LocalWindowCopy()
{
    uint32_t totalMoeExpert = 0;
    LocalTensor<int32_t> outCountLocal;
    if (isShareExpertRank_) {
        totalMoeExpert = epWorldSize_;
    } else {
        totalMoeExpert = epWorldSize_ * moeExpertNumPerRank_;
    }
    sendExpertNum_ = totalMoeExpert / aivNum_;  // 每个aiv需要处理的专家数
    uint32_t remainderRankNum = totalMoeExpert % aivNum_;
    startExpertId_ = sendExpertNum_ * aivId_;  // + sharedExpertRankNum_, 每个aiv发送的起始rankid
    if (aivId_ < remainderRankNum) {           // 前remainderRankNum个aiv需要多发1个卡的数据
        sendExpertNum_ += 1;
        startExpertId_ += aivId_;
    } else {
        startExpertId_ += remainderRankNum;
    }
    endExpertId_ = startExpertId_ + sendExpertNum_;
    if (startExpertId_ >= totalMoeExpert) {  // 多余的核return
        return;
    }
    GetCumSum(statusTensor_, outCountLocal, startExpertId_ + 1, outputRecvCountGM_);
    uint32_t index = 0;
    uint32_t beginIdx = 0;
    DataCopyExtParams dataCopyParamsFloat = {1U, sizeof(float), 0U, 0U, 0U};
    for (uint32_t index = startExpertId_; index < endExpertId_; index++) {
        uint32_t i = index - startExpertId_;
        if (i > 0) {
            outCountLocal.SetValue(i, outCountLocal.GetValue(i - 1) + outCountLocal.GetValue(index));
        }
        uint32_t count = statusTensor_.GetValue(index * INT32_NUM_PER_BLOCK + 1);
        beginIdx = outCountLocal.GetValue(i) - count;
        if constexpr (IsNeedAllgater) {
            gatherCount_ += count;
        }
        if (i == 0) {
            preCnt_ = beginIdx;
        }
        if (isShareExpertRank_) {
            if (index < sharedExpertRankNum_) {  // 共享专家前面排布的是本卡数据，只需要统计epRecvCnt，不需要去搬出
                beginIdx += count;
                continue;
            }
        }
        uint32_t winOffset = index;
        if (!isShareExpertRank_) {
            if (moeExpertNumPerRank_ > 1) {
                winOffset =
                    index % epWorldSize_ * moeExpertNumPerRank_ + index / epWorldSize_;  // 转换成数据区的排布偏移
            }
        }
        GM_ADDR wAddr = (__gm__ uint8_t *)(windowGM_) + winOffset * expertPerSizeOnWin_;
        GlobalTensor<ExpandXOutType> tokGlobal;
        GlobalTensor<ExpandXOutType> expandXOutGlobal;
        for (uint32_t j = 0; j < count; j++) {
            tokGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(wAddr + j * hCommuSize_));
            xTmpTensor_ = xQueue_.AllocTensor<ExpandXOutType>();
            DataCopy(xTmpTensor_, tokGlobal, axisHCommu_);
            xQueue_.EnQue(xTmpTensor_);
            xTmpTensor_ = xQueue_.DeQue<ExpandXOutType>();
            if constexpr (DynamicQuant || StaticQuant) {
                pipe_barrier(PIPE_ALL);
                xOutFp32Tensor_ = xTmpTensor_.template ReinterpretCast<float>();
                DataCopyPad(dynamicScalesOutGMTensor_[beginIdx + j], xOutFp32Tensor_[axisH_ / sizeof(float)],
                            dataCopyParamsFloat);
                pipe_barrier(PIPE_ALL);
            }
            if constexpr (IsNeedAllgater) {
                DataCopy(winTpGatherOutGMTensor_[(beginIdx + j) * axisHCommu_], xTmpTensor_, axisHCommu_);
            }
            expandXOutGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(expandXOutGM_) + (beginIdx + j) * axisH_,
                                             axisH_);
            DataCopy(expandXOutGlobal, xTmpTensor_, axisH_);
            xQueue_.FreeTensor(xTmpTensor_);
        }
        beginIdx += count;
    }
    if constexpr (!IsNeedAllgater) {
        totalCnt_ = beginIdx;
    }
    lastCore_ = MIN(totalMoeExpert, aivNum_) - 1;
    if constexpr (IsNeedAllgater) {
        DataCopyExtParams dataCopyOutParams = {1U, static_cast<uint32_t>(sendExpertNum_ * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPad(winTpEpCntGMTensor_[startExpertId_], outCountLocal, dataCopyOutParams);
    }
    DataCopyExtParams dataCopyOutParams = {1U, static_cast<uint32_t>(sendExpertNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    GlobalTensor<int32_t> sendCountsGlobal;
    sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCountsOutGM_));
    DataCopyPad(sendCountsGlobal[startExpertId_], outCountLocal, dataCopyOutParams);
    PipeBarrier<PIPE_MTE3>();
}

template <TemplateDispatchTypeClass>
__aicore__ inline void CamMoeDistributeDispatch<TemplateDispatchTypeFunc>::AllGatherSetStatusAndWait()
{
    pipe_barrier(PIPE_ALL);
    if (startExpertId_ >= totalExpertNum_) {
        return;
    }
    GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindStateAddrByRankId(COMM_TP_IDX, tpGatherRankId_) + stateOffset_ * aivId_);
    GlobalTensor<float> tpwindowInstatusFp32Tensor_;
    tpwindowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)(rankGM));
    statusTensor_(aivId_ * INT32_NUM_PER_BLOCK + 1) = gatherCount_;
    statusTensor_(aivId_ * INT32_NUM_PER_BLOCK + 2) = preCnt_;
    LocalTensor<float> statusFp32Tensor_ = statusTensor_.ReinterpretCast<float>();
    statusFp32Tensor_(aivId_ * 8) = sumTarget_;
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy<float>(tpwindowInstatusFp32Tensor_, statusFp32Tensor_[aivId_ * 8],
                    UB_ALIGN);  // 12是数据大小，按32对齐拷贝
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    float sumOfFlag = static_cast<float>(-1.0);
    rankGM =
        (__gm__ uint8_t *)(GetWindStateAddrByRankId(COMM_TP_IDX, tpRankId_) + stateOffset_ * aivId_);  // 计算地址偏移
    tpwindowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)(rankGM));
    while (sumOfFlag != sumTarget_) {
        DataCopy(statusFp32Tensor_, tpwindowInstatusFp32Tensor_, UB_ALIGN);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        sumOfFlag = statusFp32Tensor_.GetValue(0);
        SyncFunc<AscendC::HardEvent::S_MTE2>();
    }
}

template <TemplateDispatchTypeClass>
__aicore__ inline void CamMoeDistributeDispatch<TemplateDispatchTypeFunc>::AllgatherProcessOut()
{
    if (startExpertId_ >= totalExpertNum_) {
        return;
    }
    // 获取需要allgather的tokens数量
    GlobalTensor<float> tpwindowInstatusFp32Tensor_;
    GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindStateAddrByRankId(COMM_TP_IDX, tpRankId_) + stateOffset_ * aivId_);
    tpwindowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)rankGM);
    LocalTensor<float> statusFp32Tensor_ = statusTensor_.ReinterpretCast<float>();
    DataCopy(statusFp32Tensor_, tpwindowInstatusFp32Tensor_, UB_ALIGN);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint32_t coreGatherCount = statusFp32Tensor_.ReinterpretCast<int32_t>().GetValue(1);
    uint32_t preCount = statusFp32Tensor_.ReinterpretCast<int32_t>().GetValue(2);
    gatherCount_ = coreGatherCount;
    preCnt_ = preCount;
    GlobalTensor<int32_t> sendCountsGlobal;
    GlobalTensor<int32_t> tpGlobal;
    // 搬运另一个tp域卡传来的epRcvCnt
    sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCountsOutGM_));
    tpGlobal.SetGlobalBuffer((__gm__ int32_t *)(tpLocalStatusWindowGM_ + TP_STATE_SIZE));
    DataCopyExtParams dataCopyParams = {1U, static_cast<uint32_t>(sendExpertNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> copyPadParams{false, 0U, 0U, 0U};
    tpTmpTensor_ = xQueue_.AllocTensor<int32_t>();
    DataCopyPad(tpTmpTensor_, tpGlobal[startExpertId_], dataCopyParams, copyPadParams);
    xQueue_.EnQue(tpTmpTensor_);
    tpTmpTensor_ = xQueue_.DeQue<int32_t>();
    DataCopyPad(sendCountsGlobal[epWorldSize_ + startExpertId_], tpTmpTensor_, dataCopyParams);
    xQueue_.FreeTensor(tpTmpTensor_);
    if (coreGatherCount == 0) {
        return;
    }
    // 输出起始偏移本卡数据
    GlobalTensor<ExpandXOutType> tokGlobal;
    GlobalTensor<ExpandXOutType> expandXOutGlobal;
    DataCopyExtParams dataCopyParamsFloat = {1U, sizeof(float), 0U, 0U, 0U};
    for (uint32_t i = 0; i < coreGatherCount; i++) {
        tokGlobal.SetGlobalBuffer((__gm__ ExpandXOutType *)(tpLocalWindowGM_ + (preCount + i) * hCommuSize_));
        xTmpTensor_ = xQueue_.AllocTensor<ExpandXOutType>();
        DataCopy(xTmpTensor_, tokGlobal, axisHCommu_);
        xQueue_.EnQue(xTmpTensor_);
        xTmpTensor_ = xQueue_.DeQue<ExpandXOutType>();
        expandXOutGlobal.SetGlobalBuffer(
            (__gm__ ExpandXOutType *)(expandXOutGM_ + (preCount + totalCnt_ + i) * hOutSize_));
        DataCopy(expandXOutGlobal, xTmpTensor_, axisH_);
        if constexpr (StaticQuant || DynamicQuant) {
            xOutFp32Tensor_ = xTmpTensor_.template ReinterpretCast<float>();
            DataCopyPad(dynamicScalesOutGMTensor_[preCount + totalCnt_ + i], xOutFp32Tensor_[axisH_ / sizeof(float)],
                        dataCopyParamsFloat);
        }
        xQueue_.FreeTensor(xTmpTensor_);
    }
}

// 更新多专家卡上的tokenNumsOut tensor
template <TemplateDispatchTypeClass>
__aicore__ inline void CamMoeDistributeDispatch<TemplateDispatchTypeFunc>::UpdataMultiMoeTokenNumsOut()
{
    uint32_t tokenSums = 0;
    GlobalTensor<int32_t> sendCountsGlobal;
    sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCountsOutGM_));
    for (uint32_t localMoeIndex = 0; localMoeIndex < moeExpertNumPerRank_; ++localMoeIndex) {
        if (localMoeIndex == 0) {
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                sendCountsGlobal[epWorldSize_ - 1]);
            uint32_t firstMoeCnt = sendCountsGlobal.GetValue(epWorldSize_ - 1);
            tokenSums = firstMoeCnt + gatherCount_;
            expertTokenNumsOutGMTensor_.SetValue(localMoeIndex, tokenSums);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                expertTokenNumsOutGMTensor_[localMoeIndex]);
        } else {
            uint32_t preIndex = epWorldSize_ * (localMoeIndex - 1) + epWorldSize_ - 1;
            uint32_t curIndex = epWorldSize_ * localMoeIndex + epWorldSize_ - 1;
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                sendCountsGlobal[preIndex]);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                sendCountsGlobal[curIndex]);
            uint32_t preMoeIndexCnt = sendCountsGlobal.GetValue(preIndex);
            uint32_t curMoeIndexCnt = sendCountsGlobal.GetValue(curIndex);
            tokenSums =
                ((expertTokenNumsType_ == 0) ? tokenSums : 0) + (curMoeIndexCnt - preMoeIndexCnt) + gatherCount_;
            expertTokenNumsOutGMTensor_.SetValue(localMoeIndex, tokenSums);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                expertTokenNumsOutGMTensor_[localMoeIndex]);
        }
    }
}

// 更新tokenNumsOut tensor
template <TemplateDispatchTypeClass>
__aicore__ inline void CamMoeDistributeDispatch<TemplateDispatchTypeFunc>::UpdataTokenNumsOut()
{
    // 最后一个核做更新，Moe专家只有最后一个核有计算出所有 sendCountsGlobal
    if (!isShareExpertRank_ && moeExpertNumPerRank_ > 1) {
        SyncAll<true>();
        if (aivId_ != lastCore_) return;
        SyncFunc<AscendC::HardEvent::MTE3_S>();
        UpdataMultiMoeTokenNumsOut();
    } else {
        if (aivId_ != lastCore_) return;
        uint32_t tokenNum = 0;
        // Moe专家token总数在Cumsum内计算得出
        tokenNum = totalCnt_;
        if constexpr (IsNeedAllgater) {
            tokenNum += preCnt_;
            tokenNum += gatherCount_;
        }
        expertTokenNumsOutGMTensor_.SetValue(0, tokenNum);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            expertTokenNumsOutGMTensor_);
    }
    // token总数 = 其他专家搬进来的token数 + allgather拿到的另一张卡token数
    if constexpr (IsNeedAllgater) {
        GlobalTensor<int32_t> sendTpCountsGlobal;
        sendTpCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendTpCountOutGM_));
        sendTpCountsGlobal.SetValue(tpRankId_, totalCnt_);
        sendTpCountsGlobal.SetValue(tpGatherRankId_, gatherCount_ + preCnt_);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            sendTpCountsGlobal);  // 当前tpId只会为0或1，只需要刷一次Cache
    }
}

template <TemplateDispatchTypeClass>
__aicore__ inline void CamMoeDistributeDispatch<TemplateDispatchTypeFunc>::Process()
{
    if ASCEND_IS_AIV {  // 全aiv处理
        AlltoAllDispatch();
        SetStatus();
        WaitDispatch();
        LocalWindowCopy();
        if constexpr (IsNeedAllgater) {
            AllGatherSetStatusAndWait();
            AllgatherProcessOut();
        }
        UpdataTokenNumsOut();
    }
}

}  // namespace MoeDistributeDispatchImpl
#endif  // CAM_MOE_DISTRIBUTE_DISPATCH_H
