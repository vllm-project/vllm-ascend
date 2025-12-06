#ifndef MOE_COMBINE_NORMAL_H
#define MOE_COMBINE_NORMAL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../common/moe_distribute_base.h"
#include "moe_combine_normal_tiling.h"

namespace MoeCombineNormalImpl {
constexpr uint32_t RANK_ID_OFFSET_IN_SRC_INFO = 0U;
constexpr uint32_t TOKEN_IDX_OFFSET_IN_SRC_INFO = 1U;
constexpr uint32_t TOPK_IDX_OFFSET_IN_SRC_INFO = 2U;
constexpr uint64_t COMBINE_STATE_WIN_OFFSET = 3UL * 1024UL * 1024UL;
constexpr uint64_t MAGIC_WIN_OFFSET = 975UL * 1024UL;
constexpr uint32_t TOKEN_SRC_INFO_LEN = 3U;
constexpr uint32_t UB_32_ALIGN = 32U;
constexpr uint32_t MUL_256_ALIGN = 256U;
constexpr uint64_t WIN_512_ALIGN = 512UL;
constexpr uint32_t FLOAT_NUM_PER_ALIGN = 8U;
constexpr uint8_t  DOUBLE_BUFFER = 2;

template<AscendC::HardEvent event>
__aicore__ inline void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#define TemplateMC2TypeClass typename RecvXType, typename XType, typename SrcInfoType
#define TemplateMC2TypeFunc RecvXType, XType, SrcInfoType

using namespace AscendC;
template <TemplateMC2TypeClass>
class MoeCombineNormal {
public:
    __aicore__ inline MoeCombineNormal() {};
    __aicore__ inline void Init(GM_ADDR recvX, GM_ADDR tokenSrcInfo, GM_ADDR epRecvCount, GM_ADDR topkWeights,
                                GM_ADDR tpRecvCount,GM_ADDR XOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeCombineNormalTilingData *tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void InitMagic();
    __aicore__ inline void InitGlobalBuffer(GM_ADDR recvX, GM_ADDR tokenSrcInfo, GM_ADDR epRecvCount,
                                            GM_ADDR topkWeights, GM_ADDR XOut);
    __aicore__ inline void InitTilingData(const MoeCombineNormalTilingData *tilingData);
    __aicore__ inline void InitBuffLen();
    __aicore__ inline void CopyBufferToShareAndSetStatus();
    __aicore__ inline void CopyBufferToShare(uint32_t srcRankId, uint32_t srcTokenId, uint32_t srcTopkId, uint32_t tkIndex);
    __aicore__ inline void ReadBufferFromRemote();
    __aicore__ inline void WaitBuffCopy(uint32_t tokenIndex);
    __aicore__ inline void SetStatusBySrcInfo(uint32_t srcRankId, uint32_t srcTokenId, uint32_t srcTopkId);
    __aicore__ inline void ReadBufferAndWeightedSum(uint32_t tokenIndex, uint32_t startTokenIndex);

    __aicore__ GM_ADDR GetStateAddrByRankId(const int32_t rankId)
    {
        GM_ADDR bufferAddr;
        if (epRankId_ == rankId) {
            bufferAddr = (GM_ADDR)epWinContext_->localWindowsIn;
        } else {
            bufferAddr = (GM_ADDR)((HcclRankRelationResV2 *)epWinContext_->remoteRes[rankId].nextDevicePtr)->windowsIn;
        }
        return (GM_ADDR)(bufferAddr + winDataSizeOffset_);
    }

    __aicore__ GM_ADDR GetBufferAddrByRankId(const int32_t rankId)
    {
        return GetStateAddrByRankId(rankId) + COMBINE_STATE_WIN_OFFSET;
    }

    __aicore__ inline void SplitCoreCal(uint32_t totalNum, uint32_t &perCoreNum, uint32_t &startIdx, uint32_t &endIdx)
    {
        perCoreNum = totalNum / aivNum_;
        uint32_t remainderRankNum = totalNum % aivNum_;

        startIdx = perCoreNum * coreIdx_;
        if (coreIdx_ < remainderRankNum) {
            perCoreNum++;
            startIdx += coreIdx_;
        } else {
            startIdx += remainderRankNum;
        }
        endIdx = startIdx + perCoreNum;
    }

    __gm__ HcclOpResParam *epWinContext_{nullptr};
    __gm__ HcclOpResParam *tpWinContext_{nullptr};
    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epRankId_{0};
    uint32_t coreIdx_{0};
    uint32_t moeExpertNum_{0};
    uint32_t moeExpertPerRankNum_{0};
    uint32_t magic_{0};
    uint64_t winDataSizeOffset_{0};
    uint32_t selfSendCnt_{0};
    uint32_t hRecvXTypeLen_{0};
    uint32_t h32AlignFloatLen_{0};
    uint32_t h256AlignFloatLen_{0};
    uint32_t h32AlignRecvXLen_{0};
    uint32_t h512AlignRecvXLen_{0};

    TPipe *tpipe_{nullptr};
    TQue<QuePosition::VECIN, 1> weightedSumQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> localCopyQueue_;
    TBuf<> stateBuf_;
    TBuf<> topkWeightsBuf_;
    TBuf<> tokenFloatBuf_;
    TBuf<> sumFloatBuf_;
    TBuf<> weightedMulBuf_;
    TBuf<> srcInfoBuf_;
    TBuf<> xOutBuf_;
    TBuf<> tempStateBuf_;

    GlobalTensor<RecvXType> recvXGM_;
    GlobalTensor<SrcInfoType> tokenSrcInfoGM_;
    GlobalTensor<SrcInfoType> epRecvCountGM_;
    GlobalTensor<float> topkWeightsGM_;
    GlobalTensor<XType> xOutGlobal_;
    GM_ADDR localRankGM_;
    GM_ADDR workspaceGM_;
};

template <TemplateMC2TypeClass>
__aicore__ inline void MoeCombineNormal<TemplateMC2TypeFunc>::InitMagic()
{
    auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    epWinContext_ = (__gm__ HcclOpResParam*)contextGM0;

    GlobalTensor<int32_t> selfMagicTensor;
    selfMagicTensor.SetGlobalBuffer((__gm__ int32_t*)((GM_ADDR)epWinContext_->localWindowsExp + MAGIC_WIN_OFFSET +
                                                       coreIdx_ * WIN_512_ALIGN));
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfMagicTensor);
    magic_ = selfMagicTensor(0);
    selfMagicTensor(0) = ((magic_ == 0) ? 1 : 0);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(selfMagicTensor);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeCombineNormal<TemplateMC2TypeFunc>::InitGlobalBuffer(
    GM_ADDR recvX, GM_ADDR tokenSrcInfo, GM_ADDR epRecvCount, GM_ADDR topkWeights, GM_ADDR XOut)
{
    recvXGM_.SetGlobalBuffer((__gm__ RecvXType*)recvX);
    tokenSrcInfoGM_.SetGlobalBuffer((__gm__ SrcInfoType*)tokenSrcInfo);
    epRecvCountGM_.SetGlobalBuffer((__gm__ int32_t*)epRecvCount);
    topkWeightsGM_.SetGlobalBuffer((__gm__ float*)topkWeights);
    xOutGlobal_.SetGlobalBuffer((__gm__ XType*)XOut);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeCombineNormal<TemplateMC2TypeFunc>::InitTilingData(const MoeCombineNormalTilingData *tilingData)
{
    axisBS_ = tilingData->moeCombineNormalInfo.bs;
    axisH_ = tilingData->moeCombineNormalInfo.h;
    axisK_ = tilingData->moeCombineNormalInfo.k;
    aivNum_ = tilingData->moeCombineNormalInfo.aivNum;
    moeExpertNum_ = tilingData->moeCombineNormalInfo.moeExpertNum;
    moeExpertPerRankNum_ = tilingData->moeCombineNormalInfo.moeExpertPerRankNum;
    epWorldSize_ = tilingData->moeCombineNormalInfo.epWorldSize;
    epRankId_ = tilingData->moeCombineNormalInfo.epRankId;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeCombineNormal<TemplateMC2TypeFunc>::InitBuffLen()
{
    uint32_t hFloatSize = axisH_ * static_cast<uint32_t>(sizeof(float));
    h32AlignFloatLen_ = Ceil(hFloatSize, UB_32_ALIGN) * UB_32_ALIGN;
    h256AlignFloatLen_ = Ceil(hFloatSize, MUL_256_ALIGN) * MUL_256_ALIGN;
    hRecvXTypeLen_ = axisH_ * sizeof(RecvXType);
    h32AlignRecvXLen_ = Ceil(hRecvXTypeLen_, UB_32_ALIGN) * UB_32_ALIGN;
    h512AlignRecvXLen_ = Ceil(hRecvXTypeLen_, WIN_512_ALIGN) * WIN_512_ALIGN;
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeCombineNormal<TemplateMC2TypeFunc>::Init(GM_ADDR recvX, GM_ADDR tokenSrcInfo,
                                                                      GM_ADDR epRecvCount, GM_ADDR topkWeights,
                                                                      GM_ADDR tpRecvCount, GM_ADDR XOut,
                                                                      GM_ADDR workspaceGM, TPipe *pipe,
                                                                      const MoeCombineNormalTilingData *tilingData)
{
    workspaceGM_ = workspaceGM;
    tpipe_ = pipe;
    coreIdx_ = GetBlockIdx();

    InitMagic();
    InitGlobalBuffer(recvX, tokenSrcInfo, epRecvCount, topkWeights, XOut);
    InitTilingData(tilingData);
    InitBuffLen();

    PipeBarrier<PIPE_ALL>();
    winDataSizeOffset_ = static_cast<uint64_t>(magic_) * (tilingData->moeCombineNormalInfo.totalWinSize / 2UL);
    localRankGM_ = GetBufferAddrByRankId(epRankId_);
    DataCacheCleanAndInvalid<SrcInfoType, CacheLine::SINGLE_CACHE_LINE,
                             DcciDst::CACHELINE_OUT>(epRecvCountGM_[moeExpertNum_ - 1]);
    selfSendCnt_ = epRecvCountGM_(moeExpertNum_ - 1);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeCombineNormal<TemplateMC2TypeFunc>::CopyBufferToShareAndSetStatus()
{
    PipeBarrier<PIPE_ALL>();
    uint32_t perBlockSendNum = 0, startTokenId = 0, endTokenId = 0;
    SplitCoreCal(selfSendCnt_, perBlockSendNum, startTokenId, endTokenId);
    if (perBlockSendNum == 0U) {
        return;
    }

    uint32_t blockLen = static_cast<uint32_t>(perBlockSendNum * TOKEN_SRC_INFO_LEN * sizeof(uint32_t));
    tpipe_->Reset();
    tpipe_->InitBuffer(stateBuf_, UB_32_ALIGN);
    tpipe_->InitBuffer(localCopyQueue_, DOUBLE_BUFFER, h32AlignRecvXLen_);
    tpipe_->InitBuffer(srcInfoBuf_, blockLen);
    LocalTensor<uint32_t> statusTensor = stateBuf_.AllocTensor<uint32_t>();
    Duplicate<uint32_t>(statusTensor, 0x3F800000, FLOAT_NUM_PER_ALIGN);

    LocalTensor<SrcInfoType> srcInfoLocal = srcInfoBuf_.Get<SrcInfoType>();
    const DataCopyExtParams dataCopyParams{1U, blockLen, 0U, 0U, 0U};
    const DataCopyPadExtParams<SrcInfoType> padParams{false, 0U, 0U, 0U};
    DataCopyPad(srcInfoLocal, tokenSrcInfoGM_[startTokenId * TOKEN_SRC_INFO_LEN], dataCopyParams, padParams);

    SyncFunc<AscendC::HardEvent::MTE2_S>();
    for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; tokenIndex++) {
        uint32_t index = (tokenIndex - startTokenId) * TOKEN_SRC_INFO_LEN;
        uint32_t srcRankId  = static_cast<uint32_t>(srcInfoLocal(index + RANK_ID_OFFSET_IN_SRC_INFO));
        uint32_t srcTokenId = static_cast<uint32_t>(srcInfoLocal(index + TOKEN_IDX_OFFSET_IN_SRC_INFO));
        uint32_t srcTopkId  = static_cast<uint32_t>(srcInfoLocal(index + TOPK_IDX_OFFSET_IN_SRC_INFO));
        CopyBufferToShare(srcRankId, srcTokenId, srcTopkId, tokenIndex);
        PipeBarrier<PIPE_ALL>();
        SetStatusBySrcInfo(srcRankId, srcTokenId, srcTopkId);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeCombineNormal<TemplateMC2TypeFunc>::CopyBufferToShare(uint32_t srcRankId, uint32_t srcTokenId,
                                                                                   uint32_t srcTopkId, uint32_t tkIndex)
{
    uint32_t tokenOffset = tkIndex * axisH_;
    GM_ADDR dstGM = GetBufferAddrByRankId(srcRankId) + (srcTokenId * axisK_ + srcTopkId) * h512AlignRecvXLen_;
    GlobalTensor<XType> dstWindow;
    dstWindow.SetGlobalBuffer((__gm__ XType*)dstGM);
    DataCopyExtParams xOutCopyParams{1U, static_cast<uint32_t>(hRecvXTypeLen_), 0U, 0U, 0U};
    DataCopyPadExtParams<RecvXType> copyPadExtParams{false, 0U, 0U, 0U};

    LocalTensor<RecvXType> localCopyTensor;
    localCopyTensor = localCopyQueue_.AllocTensor<RecvXType>();
    DataCopyPad(localCopyTensor, recvXGM_[tokenOffset], xOutCopyParams, copyPadExtParams);
    localCopyQueue_.EnQue(localCopyTensor);
    localCopyTensor = localCopyQueue_.DeQue<RecvXType>();
    DataCopyPad(dstWindow, localCopyTensor, xOutCopyParams);
    localCopyQueue_.FreeTensor<RecvXType>(localCopyTensor);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeCombineNormal<TemplateMC2TypeFunc>::SetStatusBySrcInfo(uint32_t srcRankId, uint32_t srcTokenId,
                                                                                    uint32_t srcTopkId)
{
    LocalTensor<uint32_t> statusTensor = stateBuf_.AllocTensor<uint32_t>();
    GM_ADDR stateGM = GetStateAddrByRankId(srcRankId) + (srcTokenId * axisK_ + srcTopkId) * UB_32_ALIGN;
    GlobalTensor<uint32_t> stateGMTensor;
    stateGMTensor.SetGlobalBuffer((__gm__ uint32_t*)stateGM);
    DataCopy<uint32_t>(stateGMTensor, statusTensor, FLOAT_NUM_PER_ALIGN);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeCombineNormal<TemplateMC2TypeFunc>::WaitBuffCopy(uint32_t tokenIndex)
{
    uint32_t calCount = axisK_ * FLOAT_NUM_PER_ALIGN;
    GM_ADDR stateGM = GetStateAddrByRankId(epRankId_) + tokenIndex * axisK_ * UB_32_ALIGN; // Calculate address offset
    GlobalTensor<float> stateGMTensor;
    stateGMTensor.SetGlobalBuffer((__gm__ float*)stateGM);
    float current = (float)0.0;
    float target  = (float)1.0 * axisK_ * FLOAT_NUM_PER_ALIGN;
    SumParams sumPerKParams{1, calCount, calCount};
    LocalTensor<float> stateTensorLocal = stateBuf_.Get<float>();
    LocalTensor<float> tempStateTensorLocal = tempStateBuf_.Get<float>();
    while (current != target) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy<float>(stateTensorLocal, stateGMTensor, calCount);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(tempStateTensorLocal, stateTensorLocal, sumPerKParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        current = tempStateTensorLocal(0);
    }
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<float>(tempStateTensorLocal, (float)0.0, calCount);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy<float>(stateGMTensor, tempStateTensorLocal, calCount);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeCombineNormal<TemplateMC2TypeFunc>::ReadBufferAndWeightedSum(uint32_t tokenIndex,
                                                                                          uint32_t startTokenIndex)
{
    LocalTensor<float> tokenFloatLocal     = tokenFloatBuf_.Get<float>();
    LocalTensor<float> weightedMulBufLocal = weightedMulBuf_.Get<float>();
    LocalTensor<float> sumFloatBufLocal    = sumFloatBuf_.Get<float>();
    LocalTensor<float> topkWeightsLocal    = topkWeightsBuf_.Get<float>();
    LocalTensor<uint32_t> stateTensorLocal = stateBuf_.Get<uint32_t>();
    Duplicate(sumFloatBufLocal, static_cast<float>(0), axisH_);
    const DataCopyExtParams xOutCopyParams{1U, static_cast<uint32_t>(hRecvXTypeLen_), 0U, 0U, 0U};

    for (uint32_t topkId = 0U; topkId < axisK_; topkId++) {
        float scale = topkWeightsLocal.GetValue((tokenIndex - startTokenIndex) * axisK_ + topkId);
        GM_ADDR localTokenAddr = localRankGM_ + (tokenIndex * axisK_ + topkId) * h512AlignRecvXLen_;
        GlobalTensor<XType> localTokenTensor;
        localTokenTensor.SetGlobalBuffer((__gm__ XType*)localTokenAddr);

        LocalTensor<XType> tmpToken = weightedSumQueue_.AllocTensor<XType>();
        const DataCopyPadExtParams<RecvXType> copyPadExtParams{false, 0U, 0U, 0U};
        DataCopyPad(tmpToken, localTokenTensor, xOutCopyParams, copyPadExtParams);
        weightedSumQueue_.EnQue(tmpToken);
        tmpToken = weightedSumQueue_.DeQue<XType>();
        Cast(tokenFloatLocal, tmpToken, AscendC::RoundMode::CAST_NONE, axisH_);
        PipeBarrier<PIPE_V>();
        AscendC::Muls(weightedMulBufLocal, tokenFloatLocal, scale, axisH_);
        PipeBarrier<PIPE_V>();
        AscendC::Add(sumFloatBufLocal, sumFloatBufLocal, weightedMulBufLocal, axisH_);
        weightedSumQueue_.FreeTensor<XType>(tmpToken);
    }
    PipeBarrier<PIPE_V>();
    LocalTensor<XType> xOutLocal = xOutBuf_.Get<XType>();
    Cast(xOutLocal, sumFloatBufLocal, AscendC::RoundMode::CAST_RINT, axisH_);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(xOutGlobal_[tokenIndex * axisH_], xOutLocal, xOutCopyParams);
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeCombineNormal<TemplateMC2TypeFunc>::ReadBufferFromRemote()
{
    if (axisBS_ == 0U) {
        return;
    }
    uint32_t tokenPerBlock = 0U, startTokenIndex = 0U, endTokenIndex = 0U;
    SplitCoreCal(axisBS_, tokenPerBlock, startTokenIndex, endTokenIndex);

    if (tokenPerBlock == 0U) {
        return;
    }

    tpipe_->Reset();
    tpipe_->InitBuffer(xOutBuf_, h32AlignRecvXLen_);
    tpipe_->InitBuffer(tokenFloatBuf_, h32AlignFloatLen_);
    tpipe_->InitBuffer(weightedMulBuf_, h256AlignFloatLen_);
    tpipe_->InitBuffer(sumFloatBuf_, h32AlignFloatLen_);
    tpipe_->InitBuffer(weightedSumQueue_, DOUBLE_BUFFER, h32AlignRecvXLen_);
    tpipe_->InitBuffer(stateBuf_, (axisK_) * UB_32_ALIGN);
    tpipe_->InitBuffer(tempStateBuf_, (axisK_) * UB_32_ALIGN);
    tpipe_->InitBuffer(topkWeightsBuf_, tokenPerBlock * axisK_ * sizeof(float));

    LocalTensor<float> topkWeightsLocal = topkWeightsBuf_.Get<float>();
    const DataCopyExtParams bskParams{1U, static_cast<uint32_t>(tokenPerBlock * axisK_ * sizeof(float)), 0U, 0U, 0U};
    const DataCopyPadExtParams<float> copyPadFloatParams{false, 0U, 0U, 0U};
    DataCopyPad(topkWeightsLocal, topkWeightsGM_[startTokenIndex * axisK_], bskParams, copyPadFloatParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t tokenIndex = startTokenIndex; tokenIndex < endTokenIndex; tokenIndex++) {
        WaitBuffCopy(tokenIndex);
        SyncFunc<AscendC::HardEvent::MTE3_V>(); // Sync with result datacopy on same tensor
        ReadBufferAndWeightedSum(tokenIndex, startTokenIndex);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void MoeCombineNormal<TemplateMC2TypeFunc>::Process()
{
    if ASCEND_IS_AIV { // All AIV processing
        CopyBufferToShareAndSetStatus();
        ReadBufferFromRemote();
    }
}

} // MoeCombineNormalImpl
#endif // MOE_COMBINE_IMPL_H
