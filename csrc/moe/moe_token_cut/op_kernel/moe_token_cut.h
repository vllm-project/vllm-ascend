/**
• Copyright (c) 2026 Huawei Technologies Co., Ltd.

• Licensed under the Apache License, Version 2.0.

*/
#ifndef MOE_TOKEN_CUT_H
#define MOE_TOKEN_CUT_H

#include "kernel_operator.h"
#include "moe_token_cut_tiling.h"
#include "utils/moe_distribute_base.h"

namespace MoeTokenCutImpl {

using namespace AscendC;

static constexpr uint32_t UB_ALIGN_SIZE = 32;
constexpr uint32_t MAX_BATCH_SIZE = 96;
// 通信 slot 按 64B cache line 对齐，避免相邻 source 的 flag/counts
// 共享 cache line，降低接收核轮询 DCCI 带来的互相干扰。
constexpr uint32_t ALIGN_BYTES = 64;
constexpr uint32_t LEVEL_TOKEN_GRANULARITY = 128;
constexpr uint32_t MAGIC_SHIFT = 32;
constexpr uint64_t STATUS_MASK = 0xFFFFFFFFULL;
constexpr uint32_t CONTROL_INT8_BYTES_ALIGNED = MAX_BATCH_SIZE;
constexpr uint32_t CONTROL_LOCAL_BYTES =
    MAX_BATCH_SIZE * (sizeof(int8_t) + sizeof(int8_t) + sizeof(int32_t));
// Restore 阶段复用 UB 固定区域做双缓冲。该阶段结束并完成全核同步后，
// 同一片 UB 才会被 moe_token_cut 的统计/决策阶段使用。
constexpr int64_t RESTORE_PING_PONG_BYTES = 57 * 1024;
constexpr int32_t RESTORE_UB_HEAD_OFFSET = 76 * 1024;
constexpr int32_t RESTORE_UB_MID_OFFSET =
    RESTORE_UB_HEAD_OFFSET + RESTORE_PING_PONG_BYTES + UB_ALIGN_SIZE;
// 置为 false 可关掉分段计时，避免 printf 和 PIPE_ALL 影响性能。
constexpr bool ENABLE_CYCLE_PROFILE = false;
constexpr int64_t CYCLE_TO_TIME_BASE = 50;
#define DATA_FLUSH(_gm_tensor, _type) \
    Barrier(); \
    DataCacheCleanAndInvalid<_type, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(_gm_tensor);\
    __asm__("NOP"); \
    dsb(DSB_ALL);

/**
• GM -> UB 的统一搬运接口，size 的单位是字节。

• 调用方负责在消费 LocalTensor 前插入 MTE2 -> S/V 同步。

*/
template <typename TT>
__aicore__ inline void CpGMtoUB(
    LocalTensor<TT> outputLt, GlobalTensor<TT> inputGt, uint32_t size)
{
    DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    DataCopyPadExtParams<TT> padParams{false, 0U, 0U, 0U};
    DataCopyPad(outputLt, inputGt, dataCopyParams, padParams);
}

/**
• UB -> GM 的统一搬运接口，size、srcStride、dstStride 的单位均为字节。

• 调用方负责保证 LocalTensor 已生产完成，并在复用 LocalTensor 前等待 MTE3。

*/
template <typename TT>
__aicore__ inline void CpUBtoGM(GlobalTensor<TT> outputGt, LocalTensor<TT> inputLt,
    uint16_t count, uint32_t size, uint32_t srcStride = 0, uint32_t dstStride = 0)
{
    DataCopyExtParams dataCopyParams(count, size, srcStride / UB_ALIGN_SIZE, dstStride, 0);
    DataCopyPad(outputGt, inputLt, dataCopyParams);
}

template <typename TT>
__aicore__ inline void CpGMtoUBRaw(
    __ubuf__ TT *ubAddr, __gm__ TT *gmAddr, uint32_t size)
{
    LocalTensor<uint8_t> ubTensor;
    GlobalTensor<uint8_t> gmTensor;
    DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    ubTensor.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ubAddr);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(gmAddr));
    DataCopyPadExtParams<uint8_t> padParams;
    DataCopyPad(ubTensor, gmTensor, dataCopyParams, padParams);
}

template <typename TT>
__aicore__ inline void CpUBtoGMRaw(
    __gm__ TT *gmAddr, __ubuf__ TT *ubAddr, uint32_t size)
{
    LocalTensor<uint8_t> ubTensor;
    GlobalTensor<uint8_t> gmTensor;
    DataCopyExtParams dataCopyParams(1, size, 0, 0, 0);
    ubTensor.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(ubAddr);
    gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(gmAddr));
    DataCopyPad(gmTensor, ubTensor, dataCopyParams);
}

template <HardEvent event>
__aicore__ inline void SyncFunc()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}

/**
• Fused pre-dispatch token pruning kernel.

*
• Execution is split into four globally ordered phases:

•   1. Every AIV builds the same local expert histogram and route-index lists.

•   2. Dedicated send/receive AIVs exchange histograms through HCCL windows.

•   3. Every AIV loads the complete receive region once and independently derives

•      the same deleted-expert mask and row prefix sum in UB.

•   4. AIVs 1..N-1 compact the retained top-k rows without overlapping writes;

•      AIV 0 only publishes the fixed-shape metadata tensors.

*
• Process 中只保留一次 SyncAll，用于等待全部 status/counts 接收完成。

• hidden 使用模板类型 T；weights 固定 bfloat16，压缩映射相同。

*/
template <typename T>
class MoeTokenCut {
public:
    __aicore__ inline void Init(GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR hidden,
        GM_ADDR cutIdx, GM_ADDR newTopk, GM_ADDR newTopkWeight, GM_ADDR newHidden,
        GM_ADDR newResiduals,
        GM_ADDR actualBs, GM_ADDR prefixIndex, GM_ADDR noCutRankNum, GM_ADDR activeMask,
        GM_ADDR localCutCount, GM_ADDR batchSize, GM_ADDR enableCut,
        GM_ADDR residuals, GM_ADDR discardLabel, GM_ADDR discardTokens,
        GM_ADDR discardMoeReqId, GM_ADDR reqIdList, GM_ADDR discardResiduals,
        GM_ADDR discardTopk, GM_ADDR discardTopkWeight, GM_ADDR moeCnt,
        GM_ADDR historyBs, GM_ADDR originalBs, GM_ADDR workspace, TPipe *pipe,
        const MoeTokenCutTilingData *tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t align) const
    {
        return (value + align - 1U) / align * align;
    }

    __aicore__ inline uint32_t ExpertCountOffset(uint32_t expert) const
    {
        return (expert / expertPerRank_) * expertCountStride_ +
            expert % expertPerRank_;
    }

    __aicore__ inline GM_ADDR GetWindowAddr(uint32_t rank) const;
    __aicore__ inline GM_ADDR GetSlotAddr(uint32_t targetRank, uint32_t sourceRank) const;
    __aicore__ inline void InitTilingData(const MoeTokenCutTilingData *tiling);
    __aicore__ inline void InitGlobalBuffer(GM_ADDR topkIds, GM_ADDR topkWeights,
        GM_ADDR hidden, GM_ADDR cutIdx, GM_ADDR newTopk, GM_ADDR newTopkWeight,
        GM_ADDR newHidden, GM_ADDR newResiduals, GM_ADDR actualBs,
        GM_ADDR prefixIndex, GM_ADDR noCutRankNum,
        GM_ADDR activeMask, GM_ADDR localCutCount, GM_ADDR batchSize,
        GM_ADDR enableCut, GM_ADDR residuals, GM_ADDR discardLabel,
        GM_ADDR discardTokens, GM_ADDR discardMoeReqId, GM_ADDR reqIdList,
        GM_ADDR discardResiduals, GM_ADDR discardTopk, GM_ADDR discardTopkWeight,
        GM_ADDR moeCnt, GM_ADDR historyBs, GM_ADDR originalBs);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void LoadRuntimeControls();
    __aicore__ inline void RestoreDiscarded();
    __aicore__ inline void BuildLocalStatistics();
    __aicore__ inline void SendCounts();
    __aicore__ inline void WaitCounts();
    __aicore__ inline void CalculateRankLevels(LocalTensor<int32_t> globalCounts,
        LocalTensor<int32_t> expertLevels, LocalTensor<float> levelWork,
        LocalTensor<float> rankLevels, LocalTensor<float> reduceWork);
    __aicore__ inline void DecideAndBuildPrefix(
        LocalTensor<int8_t> cutIdxLocal, LocalTensor<int32_t> prefixLocal,
        LocalTensor<int8_t> maskLocal);
    __aicore__ inline void BuildIdentityOutputs(
        LocalTensor<int8_t> cutIdxLocal, LocalTensor<int32_t> prefixLocal,
        LocalTensor<int8_t> maskLocal);
    __aicore__ inline void CompressRows(
        LocalTensor<int8_t> cutIdxLocal, LocalTensor<int32_t> prefixLocal);
    __aicore__ inline uint64_t ReadFlag(uint32_t sourceRank);
    __aicore__ inline void FlushInt32(GlobalTensor<int32_t> tensor, uint32_t count) const;
    template <typename TT>
    __aicore__ inline void CopyGmToGm(GlobalTensor<TT> dst, GlobalTensor<TT> src,
        int32_t elementNum, int32_t usedBlockNum, int32_t firstBlock);

    GlobalTensor<int32_t> topkIdsGm_;
    GlobalTensor<T> topkWeightsGm_;
    GlobalTensor<T> hiddenGm_;
    GlobalTensor<int8_t> cutIdxGm_;
    GlobalTensor<int32_t> newTopkGm_;
    GlobalTensor<T> newTopkWeightGm_;
    GlobalTensor<T> newHiddenGm_;
    GlobalTensor<T> newResidualsGm_;
    GlobalTensor<T> residualsGm_;
    GlobalTensor<int32_t> actualBsGm_;
    GlobalTensor<int32_t> prefixIndexGm_;
    GlobalTensor<int32_t> noCutRankNumGm_;
    GlobalTensor<int8_t> activeMaskGm_;
    GlobalTensor<int32_t> localCutCountGm_;
    GlobalTensor<int32_t> batchSizeGm_;
    GlobalTensor<int32_t> enableCutGm_;
    GlobalTensor<int32_t> discardLabelGm_;
    GlobalTensor<T> discardTokensGm_;
    GlobalTensor<int32_t> discardMoeReqIdGm_;
    GlobalTensor<int32_t> reqIdListGm_;
    GlobalTensor<T> discardResidualsGm_;
    GlobalTensor<int32_t> discardTopkGm_;
    GlobalTensor<T> discardTopkWeightGm_;
    GlobalTensor<int32_t> moeCntGm_;
    GlobalTensor<int32_t> historyBsGm_;
    GlobalTensor<int32_t> originalBsGm_;

    __gm__ HcclOpResParamCustom *WinContext_{nullptr};


    TPipe *pipe_{nullptr};
    TBuf<> localCountsBuf_;
    TBuf<> localTokenIndicesBuf_;
    TBuf<> tempBuf_;
    TBuf<> restoreCtrlBuf_;

    uint32_t blockIdx_{0};
    uint32_t blockNum_{0};
    uint32_t batchSize_{0};
    bool enableCut_{false};
    bool enablePad_{true};
    uint32_t step_{0};
    uint32_t topK_{0};
    uint32_t hiddenDim_{0};
    uint32_t rankId_{0};
    uint32_t layerIdx_{0};
    uint32_t rankNum_{0};
    uint32_t expertNum_{0};
    uint32_t expertPerRank_{0};
    uint32_t levelLimit_{0};
    uint32_t cutCountThreshold_{0};
    uint32_t smallExpertTokenThreshold_{0};
    uint32_t aivNum_{0};
    uint32_t sendAivNum_{0};
    uint32_t recvAivNum_{0};

    uint32_t slotStride_{0};
    uint32_t countsBytesAligned_{0};
    uint32_t expertCountStride_{0};
    uint32_t paddedExpertCount_{0};
    uint32_t rankBytesAligned_{0};
    uint32_t localIndicesBytesAligned_{0};
    uint32_t topkElementsAligned_{0};
    uint32_t topkBytesAligned_{0};
    uint32_t hiddenBytesAligned_{0};
    uint32_t tokenIndicesBytesAligned_{0};
    uint32_t compareMaskBytesAligned_{0};
    uint32_t compressBufferOffset_{0};
    uint32_t tempBufferBytes_{0};
    uint64_t windowBytes_{0};
    uint64_t receiveRegionBytes_{0};
    uint64_t windowBaseOffset_{0};
};

template <typename T>
__aicore__ inline void MoeTokenCut<T>::Init(GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR hidden,
    GM_ADDR cutIdx, GM_ADDR newTopk, GM_ADDR newTopkWeight, GM_ADDR newHidden,
    GM_ADDR newResiduals,
    GM_ADDR actualBs, GM_ADDR prefixIndex, GM_ADDR noCutRankNum, GM_ADDR activeMask,
    GM_ADDR localCutCount, GM_ADDR batchSize, GM_ADDR enableCut,
    GM_ADDR residuals, GM_ADDR discardLabel, GM_ADDR discardTokens,
    GM_ADDR discardMoeReqId, GM_ADDR reqIdList, GM_ADDR discardResiduals,
    GM_ADDR discardTopk, GM_ADDR discardTopkWeight, GM_ADDR moeCnt,
    GM_ADDR historyBs, GM_ADDR originalBs, GM_ADDR workspace, TPipe *pipe,
    const MoeTokenCutTilingData *tiling)
{
    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();
    InitTilingData(tiling);
    InitGlobalBuffer(topkIds, topkWeights, hidden, cutIdx, newTopk, newTopkWeight,
        newHidden, newResiduals, actualBs, prefixIndex, noCutRankNum,
        activeMask, localCutCount,
        batchSize,
        enableCut, residuals, discardLabel, discardTokens, discardMoeReqId,
        reqIdList, discardResiduals, discardTopk, discardTopkWeight, moeCnt,
        historyBs, originalBs);
    LoadRuntimeControls();
    InitBuffers(pipe);

    // counts/index/status 均在 UB 或 HCCL window 中处理，不使用用户 workspace。
    (void)workspace;
    if (enableCut_) {
        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        WinContext_ = reinterpret_cast<__gm__ HcclOpResParamCustom *>(contextGM0);
    }

}

template <typename T>
__aicore__ inline void MoeTokenCut<T>::InitTilingData(
    const MoeTokenCutTilingData *tiling)
{
    const MoeTokenCutInfo &info = tiling->info;
    topK_ = info.topK;
    hiddenDim_ = info.hiddenDim;
    rankId_ = info.rankId;
    layerIdx_ = info.layerIdx;
    rankNum_ = info.rankNum;
    expertNum_ = info.expertNum;
    expertPerRank_ = info.expertPerRank;
    levelLimit_ = info.levelLimit;
    cutCountThreshold_ = info.cutCountThreshold;
    smallExpertTokenThreshold_ = info.smallExpertTokenThreshold;
    aivNum_ = info.aivNum;
    sendAivNum_ = info.sendAivNum;
    recvAivNum_ = info.recvAivNum;

    windowBytes_ = info.windowBytes;

    // Each rank's vector row must start on a 32B boundary, independently of
    // the number of physical experts per rank. Padding is internal only.
    expertCountStride_ =
        AlignUp(expertPerRank_ * sizeof(int32_t), UB_ALIGN_SIZE) / sizeof(int32_t);
    paddedExpertCount_ = rankNum_ * expertCountStride_;
    countsBytesAligned_ = AlignUp(paddedExpertCount_ * sizeof(int32_t), ALIGN_BYTES);
    rankBytesAligned_ = AlignUp(rankNum_ * sizeof(int32_t), ALIGN_BYTES);
    localIndicesBytesAligned_ =
        AlignUp(expertNum_ * MAX_BATCH_SIZE * sizeof(int32_t), ALIGN_BYTES);
    topkElementsAligned_ = AlignUp(MAX_BATCH_SIZE * topK_, 64U);
    topkBytesAligned_ = topkElementsAligned_ * sizeof(int32_t);
    hiddenBytesAligned_ = AlignUp(hiddenDim_ * sizeof(T), ALIGN_BYTES);
    tokenIndicesBytesAligned_ = topkBytesAligned_;
    compareMaskBytesAligned_ =
        AlignUp((topkElementsAligned_ + 7U) / 8U, ALIGN_BYTES);
    slotStride_ = countsBytesAligned_ + ALIGN_BYTES;
    // 每张卡在自身 HCCL input window 尾部为所有 source rank 预留的接收区长度。
    receiveRegionBytes_ = static_cast<uint64_t>(rankNum_) * slotStride_;
    // CompressRows 仍需读取 decision 阶段生成的 cutIdx/prefix，因此压缩
    // scratch 必须放在完整 decision 区之后，不能从 tempBuf_ 起始处复用。
    compressBufferOffset_ = AlignUp(
        static_cast<uint32_t>(receiveRegionBytes_) +
            3U * countsBytesAligned_ + CONTROL_LOCAL_BYTES + 2U * rankBytesAligned_,
        ALIGN_BYTES);
    windowBaseOffset_ = windowBytes_ - receiveRegionBytes_;
}

template <typename T>
__aicore__ inline void MoeTokenCut<T>::InitGlobalBuffer(
    GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR hidden, GM_ADDR cutIdx,
    GM_ADDR newTopk, GM_ADDR newTopkWeight, GM_ADDR newHidden,
    GM_ADDR newResiduals, GM_ADDR actualBs,
    GM_ADDR prefixIndex, GM_ADDR noCutRankNum, GM_ADDR activeMask,
    GM_ADDR localCutCount, GM_ADDR batchSize, GM_ADDR enableCut, GM_ADDR residuals,
    GM_ADDR discardLabel, GM_ADDR discardTokens, GM_ADDR discardMoeReqId,
    GM_ADDR reqIdList, GM_ADDR discardResiduals, GM_ADDR discardTopk,
    GM_ADDR discardTopkWeight, GM_ADDR moeCnt, GM_ADDR historyBs, GM_ADDR originalBs)
{
    topkIdsGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(topkIds), MAX_BATCH_SIZE * topK_);

    topkWeightsGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(topkWeights), MAX_BATCH_SIZE * topK_);
    newTopkWeightGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(newTopkWeight), MAX_BATCH_SIZE * topK_);
    discardTopkWeightGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(discardTopkWeight), MAX_BATCH_SIZE * topK_);
    
    hiddenGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T *>(hidden), MAX_BATCH_SIZE * hiddenDim_);
    newHiddenGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T *>(newHidden), MAX_BATCH_SIZE * hiddenDim_);
    newResidualsGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T *>(newResiduals), MAX_BATCH_SIZE * hiddenDim_);
    residualsGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T *>(residuals), MAX_BATCH_SIZE * hiddenDim_);
    discardTokensGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T *>(discardTokens), MAX_BATCH_SIZE * hiddenDim_);
    discardResidualsGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T *>(discardResiduals), MAX_BATCH_SIZE * hiddenDim_);
    cutIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(cutIdx), MAX_BATCH_SIZE);
    newTopkGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(newTopk), MAX_BATCH_SIZE * topK_);
    actualBsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(actualBs), 1);
    prefixIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(prefixIndex), MAX_BATCH_SIZE);
    noCutRankNumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(noCutRankNum), 1);
    activeMaskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(activeMask), MAX_BATCH_SIZE);
    localCutCountGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(localCutCount), 1);
    batchSizeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(batchSize), 1);
    enableCutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(enableCut), 1);
    discardLabelGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(discardLabel), MAX_BATCH_SIZE);
    discardMoeReqIdGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(discardMoeReqId), MAX_BATCH_SIZE);
    reqIdListGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(reqIdList), MAX_BATCH_SIZE);
    discardTopkGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(discardTopk), MAX_BATCH_SIZE * topK_);
    moeCntGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(moeCnt), 1);
    historyBsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(historyBs), 1);
    originalBsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(originalBs), 1);
}

template <typename T>
__aicore__ inline void MoeTokenCut<T>::LoadRuntimeControls()
{
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        batchSizeGm_[0]);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        enableCutGm_[0]);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        localCutCountGm_[0]);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
        noCutRankNumGm_[0]);
    int32_t runtimeStep = localCutCountGm_.GetValue(0);
    int32_t runtimeBs = batchSizeGm_.GetValue(0);

    batchSize_ = static_cast<uint32_t>(runtimeBs);
    enableCut_ = enableCutGm_.GetValue(0) != 0;
    enablePad_ = noCutRankNumGm_.GetValue(0) != 0;
    step_ = static_cast<uint32_t>(runtimeStep);
}

template <typename T>
__aicore__ inline void MoeTokenCut<T>::InitBuffers(TPipe *pipe)
{
    pipe_ = pipe;
    // localCounts/localTokenIndices 跨阶段保留；tempBuf 由统计、决策和压缩阶段复用。
    const uint32_t buildBytes = topkBytesAligned_ +
        tokenIndicesBytesAligned_ + compareMaskBytesAligned_;
    const uint32_t decisionBytes = static_cast<uint32_t>(receiveRegionBytes_) +
        3U * countsBytesAligned_ + CONTROL_LOCAL_BYTES + 2U * rankBytesAligned_;
    const uint32_t compressBytes =
        AlignUp(topK_ * sizeof(int32_t), ALIGN_BYTES) +
        AlignUp(topK_ * sizeof(T), ALIGN_BYTES) +
        AlignUp(hiddenDim_ * sizeof(T), ALIGN_BYTES) +
        AlignUp(hiddenDim_ * sizeof(T), ALIGN_BYTES);
    const uint32_t commLocalBytes = countsBytesAligned_ + ALIGN_BYTES;
    const uint32_t decisionAndCompressBytes =
        compressBufferOffset_ + compressBytes;
    tempBufferBytes_ =
        buildBytes > decisionAndCompressBytes ? buildBytes : decisionAndCompressBytes;
    tempBufferBytes_ =
        tempBufferBytes_ > commLocalBytes ? tempBufferBytes_ : commLocalBytes;

    pipe_->InitBuffer(localCountsBuf_, countsBytesAligned_);
    pipe_->InitBuffer(localTokenIndicesBuf_, localIndicesBytesAligned_);
    pipe_->InitBuffer(tempBuf_, tempBufferBytes_);
    pipe_->InitBuffer(restoreCtrlBuf_,
        AlignUp(MAX_BATCH_SIZE * sizeof(int32_t), ALIGN_BYTES));
}

template <typename T>
__aicore__ inline void MoeTokenCut<T>::FlushInt32(
    GlobalTensor<int32_t> tensor, uint32_t count) const
{
    constexpr uint32_t elementsPerCacheLine = 64U / sizeof(int32_t);
    for (uint32_t offset = 0; offset < count; offset += elementsPerCacheLine) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
            tensor[offset]);
    }
    dsb(DSB_ALL);
}
template <typename T>
template <typename TT>
__aicore__ inline void MoeTokenCut<T>::CopyGmToGm(
    GlobalTensor<TT> dst, GlobalTensor<TT> src, int32_t elementNum,
    int32_t usedBlockNum, int32_t firstBlock)
{
    if (usedBlockNum <= 0 || elementNum <= 0) {
        return;
    }
    const int32_t perCore = (elementNum + usedBlockNum - 1) / usedBlockNum;
    pipe_barrier(PIPE_ALL);
    const int32_t offset =
        perCore * (static_cast<int32_t>(blockIdx_) - firstBlock);
    if (offset < 0) {
        return;
    }
    int32_t len = elementNum - offset;
    if (len > perCore) {
        len = perCore;
    }
    pipe_barrier(PIPE_ALL);
    if (len <= 0) {
        return;
    }

    __gm__ TT *input =
        const_cast<__gm__ TT *>(src[offset].GetPhyAddr());
    __gm__ TT *output =
        const_cast<__gm__ TT *>(dst[offset].GetPhyAddr());
    __ubuf__ TT *ping = reinterpret_cast<__ubuf__ TT *>(
        static_cast<uint64_t>(RESTORE_UB_HEAD_OFFSET));
    __ubuf__ TT *pong = reinterpret_cast<__ubuf__ TT *>(
        static_cast<uint64_t>(RESTORE_UB_MID_OFFSET));
    int64_t bytesRemain =
        static_cast<int64_t>(len) * static_cast<int64_t>(sizeof(TT));
    int32_t elementOffset = 0;

    PipeBarrier<PIPE_ALL>();
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (int64_t chunk = 0; bytesRemain > 0; ++chunk) {
        const uint32_t copyBytes = bytesRemain > RESTORE_PING_PONG_BYTES
            ? static_cast<uint32_t>(RESTORE_PING_PONG_BYTES)
            : static_cast<uint32_t>(bytesRemain);
        const event_t eventId = (chunk & 1) != 0 ? EVENT_ID0 : EVENT_ID1;
        __ubuf__ TT *ub = (chunk & 1) != 0 ? ping : pong;
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        CpGMtoUBRaw(ub, input + elementOffset, copyBytes);
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);
        CpUBtoGMRaw(output + elementOffset, ub, copyBytes);
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        bytesRemain -= copyBytes;
        elementOffset += static_cast<int32_t>(copyBytes / sizeof(TT));
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
    PipeBarrier<PIPE_ALL>();
}
template <typename T>
__aicore__ inline void MoeTokenCut<T>::RestoreDiscarded()
{
    

    icache_preload(8);
    DATA_FLUSH(moeCntGm_, int32_t);
    const int32_t addCnt = moeCntGm_.GetValue(0);
    SyncAll();
    if (!enablePad_ || addCnt <= 0) {
        return;
    }

    DATA_FLUSH(batchSizeGm_, int32_t);
    DATA_FLUSH(historyBsGm_, int32_t);
    DATA_FLUSH(originalBsGm_, int32_t);
    const int32_t runningBs = batchSizeGm_.GetValue(0);
    const int32_t historyBs = historyBsGm_.GetValue(0);
    const int32_t originalBs = originalBsGm_.GetValue(0);
    pipe_barrier(PIPE_ALL);
    const uint32_t n = static_cast<uint32_t>(addCnt);
    const uint32_t runningLabelBs =
        static_cast<uint32_t>(originalBs + historyBs);
    const int32_t group = static_cast<int32_t>(blockNum_ / 4U);
    pipe_barrier(PIPE_ALL);

    LocalTensor<int32_t> ctrlLocal = restoreCtrlBuf_.Get<int32_t>();
    Duplicate(ctrlLocal, static_cast<int32_t>(1), n);
    SyncFunc<HardEvent::S_MTE2>();
    CopyGmToGm(reqIdListGm_[runningLabelBs], discardMoeReqIdGm_,
        static_cast<int32_t>(n), static_cast<int32_t>(blockNum_), 0);
    CpUBtoGM(discardLabelGm_[runningLabelBs], ctrlLocal, 1, n * sizeof(int32_t));
    CopyGmToGm(hiddenGm_[static_cast<uint32_t>(runningBs) * hiddenDim_],
        discardTokensGm_, static_cast<int32_t>(n * hiddenDim_), group, 0);
    CopyGmToGm(residualsGm_[static_cast<uint32_t>(runningBs) * hiddenDim_],
        discardResidualsGm_, static_cast<int32_t>(n * hiddenDim_),
        group, group);
    CopyGmToGm(topkIdsGm_[static_cast<uint32_t>(runningBs) * topK_],
        discardTopkGm_, static_cast<int32_t>(n * topK_), group, group * 2);

    CopyGmToGm(topkWeightsGm_[static_cast<uint32_t>(runningBs) * topK_],
        discardTopkWeightGm_, static_cast<int32_t>(n * topK_), group,
        group * 3);
    
    SyncFunc<HardEvent::MTE3_S>();

    if (blockIdx_ == 0U) {
        ctrlLocal.SetValue(0, runningBs + addCnt);
        SyncFunc<HardEvent::S_MTE3>();
        CpUBtoGM(batchSizeGm_, ctrlLocal, 1, sizeof(int32_t));
    } else if (blockIdx_ == 1U) {
        ctrlLocal.SetValue(0, historyBs + addCnt);
        SyncFunc<HardEvent::S_MTE3>();
        CpUBtoGM(historyBsGm_, ctrlLocal, 1, sizeof(int32_t));
    } else {
        ctrlLocal.SetValue(0, 0);
        SyncFunc<HardEvent::S_MTE3>();
        CpUBtoGM(moeCntGm_, ctrlLocal, 1, sizeof(int32_t));
    }

    // combine_and_dispatch_rebatch 在这里结束 kernel；融合后补齐完成边界。
    SyncAll<true>();
}

template <typename T>
__aicore__ inline GM_ADDR MoeTokenCut<T>::GetWindowAddr(uint32_t rank) const
{
    if (rank == rankId_) {
        return reinterpret_cast<GM_ADDR>(WinContext_->localWindowsIn);
    }
    auto relation = reinterpret_cast<__gm__ HcclRankRelationResV2Custom *>(
        WinContext_->remoteRes[rank].nextDevicePtr);
    return reinterpret_cast<GM_ADDR>(relation->windowsIn);
}

template <typename T>
__aicore__ inline GM_ADDR MoeTokenCut<T>::GetSlotAddr(uint32_t targetRank, uint32_t sourceRank) const
{
    return GetWindowAddr(targetRank) + windowBaseOffset_ +
        static_cast<uint64_t>(sourceRank) * slotStride_;
}


template <typename T>
__aicore__ inline void MoeTokenCut<T>::BuildLocalStatistics()
{
    // 参考 dispatch setup 的 IndexSort：单遍扫描 route，累计各 expert
    // 的 token 数，并将 BS token ID 写入对应 expert 的固定分区。
    // localCounts：每个 expert 的 token 数。
    // localTokenIndices：每个 expert 的 token ID。
    LocalTensor<int32_t> localCounts = localCountsBuf_.Get<int32_t>();
    LocalTensor<int32_t> localTokenIndices = localTokenIndicesBuf_.Get<int32_t>();
    LocalTensor<int32_t> topkLocal = tempBuf_.Get<int32_t>();
    const uint32_t topkElements = batchSize_ * topK_;

    Duplicate(localCounts, static_cast<int32_t>(0), paddedExpertCount_);
    SyncFunc<HardEvent::V_S>();
    if (topkElements == 0U) {
        return;
    }
    CpGMtoUB(topkLocal, topkIdsGm_, topkElements * sizeof(int32_t));
    SyncFunc<HardEvent::MTE2_S>();

    for (uint32_t route = 0; route < topkElements; ++route) {
        int32_t expert = topkLocal.GetValue(route);
        const uint32_t countOffset = ExpertCountOffset(static_cast<uint32_t>(expert));
        int32_t expertTokenIndex =
            localCounts.GetValue(countOffset);
        int32_t token = static_cast<int32_t>(route / topK_);
        uint32_t index = static_cast<uint32_t>(expert) * MAX_BATCH_SIZE +
            static_cast<uint32_t>(expertTokenIndex);
        localTokenIndices.SetValue(index, token);
        localCounts.SetValue(
            countOffset, expertTokenIndex + 1);
    }

    SyncFunc<HardEvent::S_MTE3>();
}

template <typename T>
__aicore__ inline void MoeTokenCut<T>::SendCounts()
{
    if (blockIdx_ >= sendAivNum_) {
        return;
    }
    // BuildLocalStatistics 已在每个 AIV 的 UB 中生成完整 counts。
    LocalTensor<int32_t> localCounts = localCountsBuf_.Get<int32_t>();
    LocalTensor<int64_t> flagLocal = tempBuf_.Get<int64_t>();

    // 高 32 位保存作为 magic 的层号，低 32 位保存截断状态。

    const uint32_t magic = layerIdx_ + 1U;
    const uint32_t sendStatus =
        (!enableCut_ || batchSize_ <= cutCountThreshold_) ? 2U : 1U;
    const uint64_t sendFlag = 
        (static_cast<uint64_t>(magic) << MAGIC_SHIFT) | sendStatus;
    for (uint32_t peer = blockIdx_; peer < rankNum_; peer += sendAivNum_) {
        GM_ADDR receiveRegion = GetSlotAddr(peer, rankId_);
        GlobalTensor<int32_t> remoteCounts;
        remoteCounts.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t *>(receiveRegion), paddedExpertCount_);
        CpUBtoGM(remoteCounts, localCounts, 1, paddedExpertCount_ * sizeof(int32_t));
        SyncFunc<HardEvent::MTE3_S>();

        // 必须先确认 counts 的远端写完成，再发布 status：
        // 1 表示可截断，2 表示已达到阈值、不再截断。
        flagLocal.SetValue(0, static_cast<int64_t>(sendFlag));
        SyncFunc<HardEvent::S_MTE3>();
        GlobalTensor<int64_t> remoteFlag;
        remoteFlag.SetGlobalBuffer(
            reinterpret_cast<__gm__ int64_t *>(
                receiveRegion + countsBytesAligned_), 1);
        CpUBtoGM(remoteFlag, flagLocal, 1, sizeof(int64_t));
        SyncFunc<HardEvent::MTE3_S>();
    }
}

template <typename T>
__aicore__ inline uint64_t MoeTokenCut<T>::ReadFlag(uint32_t sourceRank)
{
    LocalTensor<int64_t> flagLocal = tempBuf_.Get<int64_t>();
    GlobalTensor<int64_t> flagGm;
    flagGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(
        GetSlotAddr(rankId_, sourceRank) + countsBytesAligned_), 4);
    SyncFunc<HardEvent::S_MTE2>();
    CpGMtoUB(flagLocal, flagGm, UB_ALIGN_SIZE);
    SyncFunc<HardEvent::MTE2_S>();
    return static_cast<uint64_t>(flagLocal.GetValue(0));
}

template <typename T>
__aicore__ inline void MoeTokenCut<T>::WaitCounts()
{
    if (blockIdx_ < sendAivNum_ || blockIdx_ >= sendAivNum_ + recvAivNum_) {
        return;
    }
    uint32_t recvIndex = blockIdx_ - sendAivNum_;
    const uint64_t expectedMagic = static_cast<uint64_t>(layerIdx_ + 1U);
    for (uint32_t source = recvIndex; source < rankNum_; source += recvAivNum_) {
        // 层号可以为 0，因此还要校验低位状态，避免初始全零 flag 被误判。
        while (true) {
            uint64_t flag = ReadFlag(source);
            uint32_t status = static_cast<uint32_t>(flag & STATUS_MASK);
            if ((flag >> MAGIC_SHIFT) == expectedMagic &&
                (status == 1U || status == 2U)) {
                break;
            }
        }
    }
}


template <typename T>
__aicore__ inline void MoeTokenCut<T>::CalculateRankLevels(
    LocalTensor<int32_t> globalCounts, LocalTensor<int32_t> expertLevels,
    LocalTensor<float> levelWork, LocalTensor<float> rankLevels,
    LocalTensor<float> reduceWork)
{
    // 128 是 2 的幂，直接用整数右移计算 ceil(count / 128)，
    // 避免 float Cast + Muls + trunc。
    Adds(expertLevels, globalCounts,
        static_cast<int32_t>(LEVEL_TOKEN_GRANULARITY - 1U), paddedExpertCount_);
    PipeBarrier<PIPE_V>();
    ShiftRight(expertLevels, expertLevels, static_cast<int32_t>(7), paddedExpertCount_);
    PipeBarrier<PIPE_V>();
    Cast(levelWork, expertLevels, RoundMode::CAST_ROUND, paddedExpertCount_);
    PipeBarrier<PIPE_V>();

    // The padded stride aligns every source row. Reduce only real experts;
    // keep results separate from the source and temporary reduction buffer.
    for (uint32_t rank = 0; rank < rankNum_; ++rank) {
        ReduceSum(rankLevels[rank], levelWork[rank * expertCountStride_],
            reduceWork, expertPerRank_);
        PipeBarrier<PIPE_V>();
    }
    SyncFunc<HardEvent::V_S>();
}

template <typename T>
__aicore__ inline void MoeTokenCut<T>::DecideAndBuildPrefix(
    LocalTensor<int8_t> cutIdxLocal, LocalTensor<int32_t> prefixLocal,
    LocalTensor<int8_t> maskLocal)
{
    // 每个 AIV 将本卡包含所有 source slot 的连续接收区一次搬入 UB。
    // 后续 status、全局计数、删除决策和 prefix 均只访问本地副本。
    LocalTensor<int32_t> localCounts = localCountsBuf_.Get<int32_t>();
    LocalTensor<int32_t> localTokenIndices = localTokenIndicesBuf_.Get<int32_t>();
    LocalTensor<int32_t> base = tempBuf_.Get<int32_t>();
    LocalTensor<int32_t> receiveRegionLocal = base;
    const uint32_t receiveRegionWords =
        static_cast<uint32_t>(receiveRegionBytes_ / sizeof(int32_t));
    const uint32_t countsWords = countsBytesAligned_ / sizeof(int32_t);
    LocalTensor<int32_t> globalCountsLocal = base[receiveRegionWords];
    LocalTensor<int32_t> deletedMaskLocal =
        base[receiveRegionWords + countsWords];
    LocalTensor<int32_t> levelWorkLocal =
        base[receiveRegionWords + 2U * countsWords];
    const uint32_t rankLevelsOffset = receiveRegionWords + 3U * countsWords;
    LocalTensor<float> rankLevelsLocal =
        base[rankLevelsOffset].template ReinterpretCast<float>();
    const uint32_t controlOffset =
        rankLevelsOffset + rankBytesAligned_ / sizeof(int32_t);
    LocalTensor<int32_t> controlLocal = base[controlOffset];
    const uint32_t rankOffset = controlOffset + CONTROL_LOCAL_BYTES / sizeof(int32_t);
    LocalTensor<int32_t> rankFlagsLocal = base[rankOffset];

    GM_ADDR receiveRegionAddr = GetSlotAddr(rankId_, 0);
    GlobalTensor<int32_t> receiveRegionGm;
    receiveRegionGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(receiveRegionAddr), receiveRegionWords);
    FlushInt32(receiveRegionGm, receiveRegionWords);
    CpGMtoUB(receiveRegionLocal, receiveRegionGm,
        static_cast<uint32_t>(receiveRegionBytes_));
    SyncFunc<HardEvent::MTE2_V>();

    Duplicate(globalCountsLocal, static_cast<int32_t>(0), paddedExpertCount_);
    PipeBarrier<PIPE_V>();

    for (uint32_t source = 0; source < rankNum_; ++source) {
        const uint32_t slotOffset = source * (slotStride_ / sizeof(int32_t));
        LocalTensor<int32_t> sourceCounts = receiveRegionLocal[slotOffset];
        Add(globalCountsLocal, globalCountsLocal, sourceCounts, paddedExpertCount_);
        PipeBarrier<PIPE_V>();
    }
    SyncFunc<HardEvent::V_S>();

    for (uint32_t source = 0; source < rankNum_; ++source) {
        const uint32_t slotOffset = source * (slotStride_ / sizeof(int32_t));
        int32_t flag = receiveRegionLocal.GetValue(slotOffset + countsWords);
        rankFlagsLocal.SetValue(source, flag);
    }

    // level work 和 deleted mask 分区均常驻本 AIV 的 UB。
    LocalTensor<float> levelWork =
        levelWorkLocal.template ReinterpretCast<float>();
    LocalTensor<int32_t> expertLevelsInt = deletedMaskLocal;
    LocalTensor<float> reduceWork = controlLocal.template ReinterpretCast<float>();
    CalculateRankLevels(
        globalCountsLocal, expertLevelsInt, levelWork, rankLevelsLocal, reduceWork);

    // rank flags 已完成统计，复用其 UB 空间保存按 level 降序排列的 rank。
    // rank 数量较少，使用稳定插入排序避免额外申请 UB。
    LocalTensor<int32_t> rankOrderLocal = rankFlagsLocal;
    for (uint32_t rank = 0; rank < rankNum_; ++rank) {
        rankOrderLocal.SetValue(rank, static_cast<int32_t>(rank));
    }
    for (uint32_t pos = 1; pos < rankNum_; ++pos) {
        const int32_t keyRank = rankOrderLocal.GetValue(pos);
        const float keyLevel =
            rankLevelsLocal.GetValue(static_cast<uint32_t>(keyRank));
        uint32_t insertPos = pos;
        while (insertPos > 0U) {
            const int32_t prevRank =
                rankOrderLocal.GetValue(insertPos - 1U);
            const float prevLevel =
                rankLevelsLocal.GetValue(static_cast<uint32_t>(prevRank));
            if (prevLevel >= keyLevel) {
                break;
            }
            rankOrderLocal.SetValue(insertPos, prevRank);
            --insertPos;
        }
        rankOrderLocal.SetValue(insertPos, keyRank);
    }

    // 每张卡只根据自身 batch 状态决定是否截断；其他 source rank 的
    // status 仅用于 noCutRankNum 统计，不参与目标 expert rank 的判断。
    const uint32_t targetLevel = levelLimit_;
    const bool localCanCut = enableCut_ && batchSize_ > cutCountThreshold_;
    const uint32_t oddStepExpertOffset = expertPerRank_ / 2U;
    const uint32_t startOffset =
        (step_ & 1U) != 0U ? oddStepExpertOffset : 0U;

    // Expert 档位已完成 rank reduce，复用该区域初始化 deleted mask。
    Duplicate(deletedMaskLocal, static_cast<int32_t>(0), expertNum_);
    SyncFunc<HardEvent::V_S>();

    if (localCanCut) {
        for (uint32_t rankPos = 0; rankPos < rankNum_; ++rankPos) {
            const uint32_t rank = static_cast<uint32_t>(
                rankOrderLocal.GetValue(rankPos));
            const uint32_t begin = rank * expertPerRank_;
            int32_t level = static_cast<int32_t>(rankLevelsLocal.GetValue(rank));
            // 不排序，只按 expert id 单遍扫描 token 数不超过阈值的小专家。
            // 候选不足时允许该 rank 仍高于目标档位，这是用精度换取低开销。
            for (uint32_t localExpert = 0; localExpert < expertPerRank_ && level > static_cast<int32_t>(targetLevel); ++localExpert) {
                const uint32_t expert =  begin + (startOffset + localExpert) % expertPerRank_;
                int32_t count = globalCountsLocal.GetValue(ExpertCountOffset(expert));
                if (count <= 0 ||
                    static_cast<uint32_t>(count) > smallExpertTokenThreshold_) {
                    continue;
                }
                deletedMaskLocal.SetValue(expert, 1);
                level -= (count + static_cast<int32_t>(LEVEL_TOKEN_GRANULARITY) - 1) /
                    static_cast<int32_t>(LEVEL_TOKEN_GRANULARITY);
            }
        }
    }

    // Duplicate 不支持 int8。按 int16 解释后填充：先整体置 0，再从对齐
    // 首地址覆盖前 batchSize 个 1。0x0101 一次写入两个 int8 的 1；
    // batchSize 为奇数时会多写一个 1，随后用标量纠正。
    LocalTensor<int16_t> cutIdxInt16 = cutIdxLocal.template ReinterpretCast<int16_t>();
    Duplicate(cutIdxInt16, static_cast<int16_t>(0), MAX_BATCH_SIZE / 2U);
    PipeBarrier<PIPE_V>();
    if (batchSize_ > 0U) {
        Duplicate(cutIdxInt16, static_cast<int16_t>(0x0101), (batchSize_ + 1U) / 2U);
        SyncFunc<HardEvent::V_S>();
        if ((batchSize_ & 1U) != 0U) {
            cutIdxLocal.SetValue(batchSize_, static_cast<int8_t>(0));
        }
    } else {
        SyncFunc<HardEvent::V_S>();
    }

    int32_t remaining = static_cast<int32_t>(batchSize_);
    const int32_t minRemaining =  static_cast<int32_t>(cutCountThreshold_ < batchSize_ ? cutCountThreshold_ : batchSize_);
    if (localCanCut) {
        for (uint32_t rankPos = 0; rankPos < rankNum_; ++rankPos) {
            const uint32_t rank = static_cast<uint32_t>(
                rankOrderLocal.GetValue(rankPos));
            const uint32_t begin = rank * expertPerRank_;
            for (uint32_t localExpert = 0;
                localExpert < expertPerRank_;
                ++localExpert) {
                const uint32_t expert =
                    begin + (startOffset + localExpert) % expertPerRank_;
                if (deletedMaskLocal.GetValue(expert) == 0) {
                    continue;
                }
                int32_t count = localCounts.GetValue(ExpertCountOffset(expert));
                for (int32_t i = 0; i < count; ++i) {
                    int32_t token =
                        localTokenIndices.GetValue(
                            expert * MAX_BATCH_SIZE + i);
                    const uint32_t tokenIndex =
                        static_cast<uint32_t>(token);
                    if (cutIdxLocal.GetValue(tokenIndex) != 0 &&
                        remaining > minRemaining) {
                        cutIdxLocal.SetValue(tokenIndex, 0);
                        --remaining;
                    }
                }
            }
        }
    }

    // 包含式前缀和：保留的源行 i 最终写到 prefix[i] - 1。
    int32_t prefix = 0;
    for (uint32_t token = 0; token < MAX_BATCH_SIZE; ++token) {
        prefix += static_cast<int32_t>(cutIdxLocal.GetValue(token));
        prefixLocal.SetValue(token, prefix);
    }

    // Dispatch v2 要求 1D mask true 在前：压缩后前 actualBs 行为有效 token。
    const uint32_t kept = static_cast<uint32_t>(prefix);
    LocalTensor<int16_t> maskInt16 = maskLocal.template ReinterpretCast<int16_t>();
    Duplicate(maskInt16, static_cast<int16_t>(0), MAX_BATCH_SIZE / 2U);
    PipeBarrier<PIPE_V>();
    if (kept > 0U) {
        Duplicate(maskInt16, static_cast<int16_t>(0x0101), (kept + 1U) / 2U);
        SyncFunc<HardEvent::V_S>();
        if ((kept & 1U) != 0U) {
            maskLocal.SetValue(kept, static_cast<int8_t>(0));
        }
    } else {
        SyncFunc<HardEvent::V_S>();
    }

    if (blockIdx_ == 0U) {
        SyncFunc<HardEvent::S_MTE3>();
        CpUBtoGM(cutIdxGm_, cutIdxLocal, 1, MAX_BATCH_SIZE * sizeof(int8_t));
        CpUBtoGM(prefixIndexGm_, prefixLocal, 1, MAX_BATCH_SIZE * sizeof(int32_t));
        CpUBtoGM(activeMaskGm_, maskLocal, 1, MAX_BATCH_SIZE * sizeof(int8_t));
        SyncFunc<HardEvent::MTE3_S>();
        actualBsGm_.SetValue(0, prefix);
        // actualBs 会被后续算子消费，在 AIV 0 结束前显式刷出，
        // 避免系统 cache 中的旧值被观察到。
        FlushInt32(actualBsGm_, 1);
    }
}

template <typename T>
__aicore__ inline void MoeTokenCut<T>::BuildIdentityOutputs(
    LocalTensor<int8_t> cutIdxLocal, LocalTensor<int32_t> prefixLocal,
    LocalTensor<int8_t> maskLocal)
{
    // enable_cut=0 要求同一 EP 组所有 rank 同时走该路径。每个 AIV
    // 独立生成相同的 identity 控制信息，因此无需跨核或跨卡同步。
    for (uint32_t token = 0; token < batchSize_; ++token) {
        cutIdxLocal.SetValue(token, static_cast<int8_t>(1));
        maskLocal.SetValue(token, static_cast<int8_t>(1));
        prefixLocal.SetValue(token, static_cast<int32_t>(token + 1U));
    }

    if (blockIdx_ == 0U) {
        SyncFunc<HardEvent::S_MTE3>();
        CpUBtoGM(cutIdxGm_, cutIdxLocal, 1,
            batchSize_ * sizeof(int8_t));
        CpUBtoGM(prefixIndexGm_, prefixLocal, 1,
            batchSize_ * sizeof(int32_t));
        CpUBtoGM(activeMaskGm_, maskLocal, 1,
            batchSize_ * sizeof(int8_t));
        SyncFunc<HardEvent::MTE3_S>();
        actualBsGm_.SetValue(0, static_cast<int32_t>(batchSize_));
        FlushInt32(actualBsGm_, 1);
    }
}

template <typename T>
__aicore__ inline void MoeTokenCut<T>::CompressRows(
    LocalTensor<int8_t> cutIdxLocal, LocalTensor<int32_t> prefixLocal)
{
    // 每个保留行的 prefix 值唯一，各 AIV 可按 token 分片并发写入，无需原子操作。
    // cutIdx/prefix 直接复用本 AIV 决策阶段生成的 UB 副本；topK / hidden 行通过
    // MTE 整行搬运，不在 k / H 循环中逐元素访问 GM。
    LocalTensor<int32_t> base =
        tempBuf_.Get<int32_t>()[compressBufferOffset_ / sizeof(int32_t)];
    const uint32_t topkBytesAligned =
        AlignUp(topK_ * sizeof(int32_t), ALIGN_BYTES);
    LocalTensor<int32_t> topkLocal = base;
    LocalTensor<T> weightLocal = base[
        topkBytesAligned / sizeof(int32_t)]
                                    .template ReinterpretCast<T>();
    const uint32_t weightBytesAligned =
        AlignUp(topK_ * sizeof(T), ALIGN_BYTES);
    LocalTensor<T> hiddenLocal = base[
        (topkBytesAligned + weightBytesAligned) / sizeof(int32_t)]
                                    .template ReinterpretCast<T>();
    const uint32_t hiddenBytesAligned =
        AlignUp(hiddenDim_ * sizeof(T), ALIGN_BYTES);
    LocalTensor<T> residualLocal = base[
        (topkBytesAligned + weightBytesAligned + hiddenBytesAligned) /
        sizeof(int32_t)]
                                      .template ReinterpretCast<T>();

    // AIV 0 专门写回控制结果，其余 AIV 重新连续编号后分摊 topK 行。
    const uint32_t copyAivNum = aivNum_ - 1U;
    const uint32_t copyAivIdx = blockIdx_ - 1U;
    const uint32_t rowsPerCore = (batchSize_ + copyAivNum - 1U) / copyAivNum;
    const uint32_t begin = copyAivIdx * rowsPerCore;
    uint32_t end = begin + rowsPerCore;
    end = end < batchSize_ ? end : batchSize_;
    for (uint32_t token = begin; token < end; ++token) {
        if (cutIdxLocal.GetValue(token) == 0) {
            continue;
        }
        const uint32_t dst = static_cast<uint32_t>(prefixLocal.GetValue(token) - 1);
        CpGMtoUB(topkLocal, topkIdsGm_[token * topK_], topK_ * sizeof(int32_t));

        CpGMtoUB(
                weightLocal, topkWeightsGm_[token * topK_], topK_ * sizeof(T));
        CpGMtoUB(
                hiddenLocal, hiddenGm_[token * hiddenDim_], hiddenDim_ * sizeof(T));
        CpGMtoUB(
                residualLocal, residualsGm_[token * hiddenDim_],
                hiddenDim_ * sizeof(T));
        SyncFunc<HardEvent::MTE2_MTE3>();

        CpUBtoGM(newTopkGm_[dst * topK_], topkLocal, 1, topK_ * sizeof(int32_t));
    
        CpUBtoGM(
                newTopkWeightGm_[dst * topK_], weightLocal, 1, topK_ * sizeof(T));
        CpUBtoGM(
                newHiddenGm_[dst * hiddenDim_], hiddenLocal, 1, hiddenDim_ * sizeof(T));
        CpUBtoGM(
                newResidualsGm_[dst * hiddenDim_], residualLocal, 1,
                hiddenDim_ * sizeof(T));
        
        SyncFunc<HardEvent::MTE3_S>();
    }
}

template <typename T>
__aicore__ inline void MoeTokenCut<T>::Process()
{
    RestoreDiscarded();
    // RestoreDiscarded may increase batchSizeGm_; refresh all runtime values
    // before building statistics or identity outputs.
    LoadRuntimeControls();
    if (!enableCut_) {
        LocalTensor<int32_t> base = tempBuf_.Get<int32_t>();
        const uint32_t controlOffset =
            static_cast<uint32_t>(receiveRegionBytes_ / sizeof(int32_t)) +
            (3U * countsBytesAligned_ + rankBytesAligned_) / sizeof(int32_t);
        LocalTensor<int32_t> controlLocal = base[controlOffset];
        LocalTensor<int8_t> controlInt8 =
            controlLocal.template ReinterpretCast<int8_t>();
        LocalTensor<int8_t> cutIdxLocal = controlInt8;
        LocalTensor<int8_t> maskLocal =
            controlInt8[CONTROL_INT8_BYTES_ALIGNED];
        LocalTensor<int32_t> prefixLocal =
            controlLocal[
                (2U * CONTROL_INT8_BYTES_ALIGNED) / sizeof(int32_t)];
        BuildIdentityOutputs(cutIdxLocal, prefixLocal, maskLocal);
        if (blockIdx_ != 0U) {
            CompressRows(cutIdxLocal, prefixLocal);
        }
        return;
    }

    int64_t t0 = 0;
    int64_t t1 = 0;
    int64_t t2 = 0;
    int64_t t3 = 0;
    int64_t t4 = 0;
    int64_t t5 = 0;
    int64_t t6 = 0;
    if constexpr (ENABLE_CYCLE_PROFILE) {
        PipeBarrier<PIPE_ALL>();
        t0 = AscendC::GetSystemCycle();
    }

    // 阶段1：所有 AIV 独立生成相同的本地 counts/token index。
    BuildLocalStatistics();
    if constexpr (ENABLE_CYCLE_PROFILE) {
        PipeBarrier<PIPE_ALL>();
        t1 = AscendC::GetSystemCycle();
    }

    // 阶段2：发送核与接收核使用不相交的 AIV 分组并发通信。
    SendCounts();
    if constexpr (ENABLE_CYCLE_PROFILE) {
        PipeBarrier<PIPE_ALL>();
        t2 = AscendC::GetSystemCycle();
    }
    WaitCounts();
    if constexpr (ENABLE_CYCLE_PROFILE) {
        PipeBarrier<PIPE_ALL>();
        t3 = AscendC::GetSystemCycle();
    }
    SyncAll<true>();
    if constexpr (ENABLE_CYCLE_PROFILE) {
        PipeBarrier<PIPE_ALL>();
        t4 = AscendC::GetSystemCycle();
    }

    // 阶段3：每个 AIV 一次搬入完整接收区，在 UB 中独立生成相同结果。
    LocalTensor<int32_t> base = tempBuf_.Get<int32_t>();
    const uint32_t controlOffset =
        static_cast<uint32_t>(receiveRegionBytes_ / sizeof(int32_t)) +
        (3U * countsBytesAligned_ + rankBytesAligned_) / sizeof(int32_t);
    LocalTensor<int32_t> controlLocal = base[controlOffset];
    LocalTensor<int8_t> controlInt8 = controlLocal.template ReinterpretCast<int8_t>();
    LocalTensor<int8_t> cutIdxLocal = controlInt8;
    LocalTensor<int8_t> maskLocal =
        controlInt8[CONTROL_INT8_BYTES_ALIGNED];
    LocalTensor<int32_t> prefixLocal =
        controlLocal[
            (2U * CONTROL_INT8_BYTES_ALIGNED) / sizeof(int32_t)];
    DecideAndBuildPrefix(cutIdxLocal, prefixLocal, maskLocal);
    if constexpr (ENABLE_CYCLE_PROFILE) {
        PipeBarrier<PIPE_ALL>();
        t5 = AscendC::GetSystemCycle();
    }

    // 阶段4：AIV 0 专门写控制输出，其他 AIV 直接复用本地结果并行压缩。
    if (blockIdx_ != 0U) {
        CompressRows(cutIdxLocal, prefixLocal);
    }
    if constexpr (ENABLE_CYCLE_PROFILE) {
        PipeBarrier<PIPE_ALL>();
        t6 = AscendC::GetSystemCycle();
        const int64_t buildUs = (t1 - t0) / CYCLE_TO_TIME_BASE;
        const int64_t sendUs = (t2 - t1) / CYCLE_TO_TIME_BASE;
        const int64_t waitUs = (t3 - t2) / CYCLE_TO_TIME_BASE;
        const int64_t syncUs = (t4 - t3) / CYCLE_TO_TIME_BASE;
        const int64_t decideUs = (t5 - t4) / CYCLE_TO_TIME_BASE;
        const int64_t compressUs = (t6 - t5) / CYCLE_TO_TIME_BASE;
        const int64_t totalUs = (t6 - t0) / CYCLE_TO_TIME_BASE;
        //只打 rank0，避免 8 卡同时刷屏；发送核看 send，接收核看 wait。
        // if (rankId_ == 1U) {
        //     printf("[MoeTokenCut] rank=%u aiv=%u build=%ld us send=%ld us "
        //         "wait=%ld us sync=%ld us decide=%ld us compress=%ld us "
        //         "total=%ld us\n",
        //         rankId_, blockIdx_, buildUs, sendUs, waitUs, syncUs,
        //         decideUs, compressUs, totalUs);
        // }
    }
}

}  // namespace MoeTokenCutImpl

#endif  // MOE_TOKEN_CUT_H