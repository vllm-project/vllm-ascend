/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_tail_attention_service_vector_mla.h
 * \brief
 */
#ifndef SPARSE_TAIL_ATTENTION_SERVICE_VECTOR_MLA_H
#define SPARSE_TAIL_ATTENTION_SERVICE_VECTOR_MLA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_tail_attention_common.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename SFAT> class SFAVectorService {
public:
    using T = float;
    using KV_T = typename SFAT::kvType;
    using OUT_T = typename SFAT::outputType;
    using UPDATE_T = T;
    using MM1_OUT_T = float;
    using MM2_OUT_T = float;

    __aicore__ inline SFAVectorService(){};
    __aicore__ inline void ProcessVec1L(const RunInfo &info);
    __aicore__ inline void ProcessVec2L(const RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct ConstInfo &constInfo,
                                      const CopySfaMtpAttentionTilingDataMla *__restrict tilingData);
    __aicore__ inline void InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm);
    __aicore__ inline void InitVec0GlobalTensor(const GlobalTensor<int32_t> &kvValidSizeGm,
                                                const GlobalTensor<KV_T> &kvMergeGm,
                                                const GlobalTensor<KV_T> &keyRopeGm, const GlobalTensor<KV_T> &keyGm,
                                                const GlobalTensor<int32_t> &blkTableGm);
    __aicore__ inline void InitSourceAwareGatherGlobalTensor(
        const GlobalTensor<KV_T> &dramKeyRopeGm,
        const GlobalTensor<KV_T> &dramKeyGm,
        const GlobalTensor<int32_t> &dramBlockTableGm,
        const GlobalTensor<int32_t> &sourceTokenIdsGm,
        const GlobalTensor<int32_t> &copyCountsGm,
        uint32_t copyCap, uint32_t dramMaxBlockNum);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm,
                                                GlobalTensor<int32_t> actualSeqLengthsQGm,
                                                GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm,
                                                GlobalTensor<T> lseSumFdGm, GlobalTensor<int32_t> topKGm);
    __aicore__ inline void InitVec2GlobalTensor(GlobalTensor<T> accumOutGm, GlobalTensor<UPDATE_T> vec2ResGm,
                                                GlobalTensor<MM2_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm,
                                                GlobalTensor<T> stage1PGm, GlobalTensor<T> stage1MGm,
                                                GlobalTensor<T> stage1LGm, GlobalTensor<T> prevPGm,
                                                GlobalTensor<T> prevMGm, GlobalTensor<T> prevLGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitSoftmaxDefaultBuffer();
    // ================================Base Vector==========================================
    __aicore__ inline void RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void CalcAmlaScaleFromMax(LocalTensor<T> scaleUb, LocalTensor<KV_T> scaleKvUb,
                                                LocalTensor<T> tmpUb, LocalTensor<T> maxUb, uint32_t dealRowCount);
    // ================================Vector0==========================================
    __aicore__ inline void MergeKv(const RunInfo &runInfo);
    __aicore__ inline void MergeKvTailContiguous(const RunInfo &runInfo);
    // Compile lean ranges for hit-only, all-miss, and one prefix boundary.
    template <bool SOURCE_ORDER, bool HAS_MISS, bool ALIGNED_MISS = false,
              bool ALL_MISS = false>
    __aicore__ inline void MergeKvRange(const RunInfo &runInfo, int64_t s2GmStartOffset,
                                        int64_t s2GmLimit, uint32_t validSizePart,
                                        int64_t missRangeSize, int64_t sourceRangeStart);
    template <bool ALIGNED_MISS = false>
    __aicore__ inline void MergeSourceAwareSparseRange(
        const RunInfo &runInfo, int64_t s2GmStartOffset,
        int64_t s2GmLimit, uint32_t validSizePart,
        int32_t missCount, int64_t sourceTileStart);
    template <bool LOAD_SOURCE_IDS>
    __aicore__ inline void LoadSourceAwareMetadata(
        int64_t topkGmBaseOffset, int64_t sourceRangeStart,
        int64_t rangeSize, int64_t sourceRangeSize);
    __aicore__ inline void FinishMergeKvRange(
        const RunInfo &runInfo, int64_t s2GmStartOffset,
        int64_t s2GmLimit, uint32_t validSizePart,
        int64_t mergeMte3Idx, int64_t mte2Size);
    __aicore__ inline int64_t GetKeyGmOffset(int64_t realS2Idx, const RunInfo &runInfo, int64_t s2IdLimit);
    __aicore__ inline int64_t GetKeyRopeGmOffset(int64_t realS2Idx, const RunInfo &runInfo, int64_t s2IdLimit);
    __aicore__ inline void GetRealS2Idx(
        int64_t s2GmOffset, int64_t &realS2Idx,
        int64_t topkGmBaseOffset, const RunInfo &runInfo);
    __aicore__ inline int32_t GetSourceAwareMissCount(
        const RunInfo &runInfo);
    __aicore__ inline int64_t GetStaggeredSparseIndex(
        int64_t virtualSparseIndex, const RunInfo &runInfo);
    __aicore__ inline void CopyInKv(
        int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
        int64_t realS2Idx1, int64_t realS2Idx2,
        const RunInfo &runInfo);
    __aicore__ inline void CopyInHbmKvPair(
        int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
        int64_t realS2Idx1, int64_t realS2Idx2, int64_t s2IdLimit,
        const RunInfo &runInfo);
    __aicore__ inline void CopyInDramKv(
        int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
        int64_t sourceTokenIdx,
        const RunInfo &runInfo);
    template <bool ALL_MISS = false>
    __aicore__ inline void CopyInSourceAwareKvPair(
        int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
        int32_t sourceToken0, int32_t sourceToken1,
        int64_t destinationSlot0, int64_t destinationSlot1,
        int32_t &persistentDestination0,
        int32_t &persistentDestination1,
        int64_t s2IdLimit, const RunInfo &runInfo);
    __aicore__ inline int64_t GetDramKeyGmOffset(
        int64_t sourceTokenIdx, const RunInfo &runInfo);
    __aicore__ inline void CopyMissToPersistentCache(
        int64_t ubRow, int32_t destinationSlot,
        const RunInfo &runInfo);
    __aicore__ inline void CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size, int64_t s2StartGmOffset,
                                             int64_t mergeMte3Idx, const RunInfo &runInfo);
    template <bool ALIGNED_MISS = false>
    __aicore__ inline void CopyOutSourceAwareResult(
        int64_t mte2Size, int64_t mte3Size,
        int64_t s2StartGmOffset, int64_t mergeMte3Idx,
        const RunInfo &runInfo, int64_t missRangeSize,
        int64_t sourceRangeStart, bool hasActualMiss,
        const int32_t *persistentCopies,
        int32_t persistentCopyCount);
    __aicore__ inline void SetInfInBlk(const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount,
                                       uint64_t startId, uint64_t endId);
    __aicore__ inline void SetMidInf(const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount,
                                     uint64_t startId, uint64_t endId);
    __aicore__ inline void CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                          int64_t keyBNBOffset,int64_t s2IdLimit, const RunInfo &runInfo);
    // ================================Vector1==========================================
    __aicore__ inline void ProcessVec1SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm1ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount, uint32_t loopId);
    __aicore__ inline void SoftmaxFlashV2Compute(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                 LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);
    __aicore__ inline void AmlaVecCompute(const RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb,
                                          LocalTensor<uint8_t> &softmaxTmpUb, uint32_t startRow, uint32_t dealRowCount,
                                          uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void ElewiseCompute(const RunInfo &info, const LocalTensor<T> &mmResUb, uint32_t dealRowCount,
                                          uint32_t columnCount);
    __aicore__ inline void ProcessAmlaNupdate(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                       LocalTensor<T> &softmaxSumUb, LocalTensor<T> &softmaxMaxUb);
    // ================================Vecotr2==========================================
    __aicore__ inline void ProcessVec2SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm2ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void ProcessVec2Inner(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t mStartRow,
                                            uint32_t mDealSize);
    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb, uint32_t wsMStart,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void Bmm2ResCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                              uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Stage1StateCopyOut(const RunInfo &info, LocalTensor<T> &stage1PUb, uint32_t wsMStart,
                                              uint32_t dealRowCount, uint32_t columnCount,
                                              uint32_t actualColumnCount, uint32_t softmaxBaseOffset,
                                              uint32_t softmaxIdx);
    __aicore__ inline void Stage2MergeAndCopyOut(const RunInfo &info, LocalTensor<T> &curPUb, uint32_t wsMStart,
                                                 uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount, uint32_t softmaxBaseOffset,
                                                 uint32_t softmaxIdx);
    __aicore__ inline void Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                             uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline uint64_t CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx);
    __aicore__ inline void GetConfusionTransposeTiling(int64_t numR, int64_t numC, const uint32_t stackBufferSize,
                                                       const uint32_t typeSize, ConfusionTransposeTiling &tiling);

    static constexpr uint64_t BYTE_BLOCK = 32UL;
    static constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
    static constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;

private:
    static constexpr bool PAGE_ATTENTION = SFAT::pageAttention;
    static constexpr int TEMPLATE_MODE = SFAT::templateMode;
    static constexpr bool FLASH_DECODE = SFAT::flashDecode;
    static constexpr int STAGE_MODE = SFAT::stageMode;
    static constexpr SFA_LAYOUT LAYOUT_T = SFAT::layout;
    static constexpr SFA_LAYOUT KV_LAYOUT_T = SFAT::kvLayout;

    static constexpr uint64_t MERGE_CACHE_GM_BUF_NUM = 4;
    static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_INPUT_BUF1_PONG_FLAG = 3;
    static constexpr uint64_t SYNC_INPUT_BUF2_FLAG = 4;
    static constexpr uint64_t SYNC_INPUT_BUF2_PONG_FLAG = 5;
    static constexpr uint64_t SYNC_OUTPUT_BUF1_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF2_FLAG = 5;
    static constexpr uint32_t INPUT1_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_32K;
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_1K;
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = ConstInfo::BUFFER_SIZE_BYTE_32K / sizeof(T);  // 32768/4=8096
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);                                // 32/4=8
    static constexpr T FLOAT_E_SCALAR = 8388608;
    static constexpr T LN2 = 0.6931471805599453094172;
    static constexpr T RECIP_OF_LN2 = 1 / LN2;
    static constexpr T SOFTMAX_MIN_NUM = -2e38;
    static constexpr uint32_t SOURCE_METADATA_CAPACITY = 512;
    static constexpr uint32_t ROPE_MERGE_BUFFER_BYTES =
        2 * 32 * 64 * sizeof(KV_T);
    static constexpr uint32_t SOURCE_METADATA_UB_OFFSET =
        ROPE_MERGE_BUFFER_BYTES / sizeof(int32_t);
    static constexpr uint32_t DEST_METADATA_UB_OFFSET =
        SOURCE_METADATA_UB_OFFSET + SOURCE_METADATA_CAPACITY;
    static_assert(
        (DEST_METADATA_UB_OFFSET + SOURCE_METADATA_CAPACITY) *
                sizeof(int32_t) <=
            ConstInfo::BUFFER_SIZE_BYTE_8K * 2,
        "source-aware metadata must fit in inputBuff2.");

    const CopySfaMtpAttentionTilingDataMla *__restrict tilingData;

    uint32_t pingpongFlag = 0U;
    ConstInfo constInfo = {};

    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<UPDATE_T> vec2ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<T> stage1PGm;
    GlobalTensor<T> stage1MGm;
    GlobalTensor<T> stage1LGm;
    GlobalTensor<T> prevPGm;
    GlobalTensor<T> prevMGm;
    GlobalTensor<T> prevLGm;
    GlobalTensor<int32_t> blkTableGm_;

    GlobalTensor<KV_T> kvMergeGm_;
    GlobalTensor<KV_T> keyRopeGm_;
    GlobalTensor<KV_T> keyGm_;
    GlobalTensor<int32_t> topkGm_;
    GlobalTensor<int32_t> kvValidSizeGm_;
    GlobalTensor<KV_T> dramKeyRopeGm_;
    GlobalTensor<KV_T> dramKeyGm_;
    GlobalTensor<int32_t> dramBlockTableGm_;
    GlobalTensor<int32_t> sourceTokenIdsGm_;
    GlobalTensor<int32_t> copyCountsGm_;
    uint32_t copyCap_ = 0;
    uint32_t dramMaxBlockNum_ = 0;
    int32_t copyCountIndex_ = -1;
    int32_t cachedCopyCount_ = 0;

    // ================================Local Buffer====================================
    TBuf<> inputBuff1;            // 32K
    TBuf<> inputBuff2;            // 16K
    TBuf<> outputBuff1;           // 32K
    TBuf<> outputBuff2;           // 4K

    TBuf<> tmpBuff1;              // 32K
    TBuf<> v0ValidSizeBuff;       // 8K

    TBuf<> nValueBuff;
    TBuf<> cofValueBuff;
    TBuf<> aMlaSumBuff;
    TBuf<> softmaxMaxBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxExpBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxSumBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxMaxDefaultBuff; // 2K
    TBuf<> softmaxSumDefaultBuff; // 2K

    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    LocalTensor<T> nValueUb;
    LocalTensor<T> cofValueUb;
    LocalTensor<T> aMlaSumUb;
    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;
    LocalTensor<KV_T> kvMergUb_;
    LocalTensor<KV_T> ropeMergUb_;
    LocalTensor<int32_t> v0ValidSizeUb_;
    LocalTensor<int32_t> sourceTokenIdsUb_;
    LocalTensor<int32_t> topkDestSlotsUb_;
};

template <typename SFAT> __aicore__ inline void SFAVectorService<SFAT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K * 2);
    pipe->InitBuffer(inputBuff2, ConstInfo::BUFFER_SIZE_BYTE_8K * 2);
    pipe->InitBuffer(outputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputBuff2, ConstInfo::BUFFER_SIZE_BYTE_4K);

    pipe->InitBuffer(tmpBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(v0ValidSizeBuff, ConstInfo::BUFFER_SIZE_BYTE_8K);

    // M_MAX = 512/2vector = 256, 256 * sizeof(T) * N_Buffer
    pipe->InitBuffer(nValueBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(cofValueBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(aMlaSumBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxExpBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxSumBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_1K);
    pipe->InitBuffer(softmaxSumDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_1K);

    nValueUb = nValueBuff.Get<T>();
    cofValueUb = cofValueBuff.Get<T>();
    aMlaSumUb = aMlaSumBuff.Get<T>();

    softmaxMaxUb = softmaxMaxBuff.Get<T>();
    softmaxSumUb = softmaxSumBuff.Get<T>();
    softmaxExpUb = softmaxExpBuff.Get<T>();

    softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
    softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();

    kvMergUb_ = inputBuff1.Get<KV_T>();
    ropeMergUb_ = inputBuff2.Get<KV_T>();
    // Vec0 uses only the first 8 KiB of inputBuff2 for the ping-pong RoPE
    // gather (2 * 32 * 64 * sizeof(KV_T)).  Reuse part of the otherwise idle
    // upper half for one 512-token source/destination metadata tile.  Later
    // vector stages may reuse inputBuff2 after Vec0 has completed.
    sourceTokenIdsUb_ =
        inputBuff2.Get<int32_t>()[SOURCE_METADATA_UB_OFFSET];
    topkDestSlotsUb_ =
        inputBuff2.Get<int32_t>()[DEST_METADATA_UB_OFFSET];

    v0ValidSizeUb_ = v0ValidSizeBuff.Get<int32_t>();
}

template <typename SFAT>
template <bool LOAD_SOURCE_IDS>
__aicore__ inline void
SFAVectorService<SFAT>::LoadSourceAwareMetadata(
    int64_t topkGmBaseOffset, int64_t sourceRangeStart,
    int64_t rangeSize, int64_t sourceRangeSize)
{
    ASSERT_MSG(
        rangeSize > 0 &&
            rangeSize <= static_cast<int64_t>(SOURCE_METADATA_CAPACITY) &&
            sourceRangeSize >= 0 && sourceRangeSize <= rangeSize,
        "source-aware metadata range exceeds the UB tile capacity.");
    if (rangeSize <= 0 ||
        rangeSize > static_cast<int64_t>(SOURCE_METADATA_CAPACITY) ||
        sourceRangeSize < 0 || sourceRangeSize > rangeSize) {
        return;
    }

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<int32_t> padParams;
    const int64_t metadataGmOffset =
        topkGmBaseOffset + sourceRangeStart;
    if constexpr (LOAD_SOURCE_IDS) {
        copyParams.blockLen = sourceRangeSize * sizeof(int32_t);
        DataCopyPad(
            sourceTokenIdsUb_, sourceTokenIdsGm_[metadataGmOffset],
            copyParams, padParams);
    }
    copyParams.blockLen = rangeSize * sizeof(int32_t);
    DataCopyPad(
        topkDestSlotsUb_, topkGm_[metadataGmOffset],
        copyParams, padParams);
    SetFlag<AscendC::HardEvent::MTE2_S>(0);
    WaitFlag<AscendC::HardEvent::MTE2_S>(0);
}

template <typename SFAT>
__aicore__ inline void
SFAVectorService<SFAT>::InitParams(const struct ConstInfo &constInfo,
                                                 const CopySfaMtpAttentionTilingDataMla *__restrict tilingData)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;
}

template <typename SFAT>
__aicore__ inline void
SFAVectorService<SFAT>::InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm)
{
    this->mm2ResInt32Gm = mm2ResInt32Gm;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::InitVec0GlobalTensor(
    const GlobalTensor<int32_t> &kvValidSizeGm, const GlobalTensor<KV_T> &kvMergeGm,
    const GlobalTensor<KV_T> &keyRopeGm, const GlobalTensor<KV_T> &keyGm, const GlobalTensor<int32_t> &blkTableGm)
{
    this->kvMergeGm_ = kvMergeGm;
    this->keyRopeGm_ = keyRopeGm;
    this->keyGm_ = keyGm;
    this->blkTableGm_ = blkTableGm;
    this->kvValidSizeGm_ = kvValidSizeGm;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::InitSourceAwareGatherGlobalTensor(
    const GlobalTensor<KV_T> &dramKeyRopeGm,
    const GlobalTensor<KV_T> &dramKeyGm,
    const GlobalTensor<int32_t> &dramBlockTableGm,
    const GlobalTensor<int32_t> &sourceTokenIdsGm,
    const GlobalTensor<int32_t> &copyCountsGm,
    uint32_t copyCap, uint32_t dramMaxBlockNum)
{
    this->dramKeyRopeGm_ = dramKeyRopeGm;
    this->dramKeyGm_ = dramKeyGm;
    this->dramBlockTableGm_ = dramBlockTableGm;
    this->sourceTokenIdsGm_ = sourceTokenIdsGm;
    this->copyCountsGm_ = copyCountsGm;
    this->copyCap_ = copyCap;
    this->dramMaxBlockNum_ = dramMaxBlockNum;
    this->copyCountIndex_ = -1;
    this->cachedCopyCount_ = 0;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm,
    GlobalTensor<int32_t> actualSeqLengthsQGm, GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm,
    GlobalTensor<T> lseSumFdGm, GlobalTensor<int32_t> topKGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->actualSeqLengthsQGm = actualSeqLengthsQGm;
    this->actualSeqLengthsKVGm = actualSeqLengthsKVGm;
    this->lseMaxFdGm = lseMaxFdGm;
    this->lseSumFdGm = lseSumFdGm;
    this->topkGm_ = topKGm;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::InitVec2GlobalTensor(GlobalTensor<T> accumOutGm,
                                                                    GlobalTensor<UPDATE_T> vec2ResGm,
                                                                    GlobalTensor<MM2_OUT_T> mm2ResGm,
                                                                    GlobalTensor<OUT_T> attentionOutGm,
                                                                    GlobalTensor<T> stage1PGm,
                                                                    GlobalTensor<T> stage1MGm,
                                                                    GlobalTensor<T> stage1LGm,
                                                                    GlobalTensor<T> prevPGm,
                                                                    GlobalTensor<T> prevMGm,
                                                                    GlobalTensor<T> prevLGm)
{
    this->accumOutGm = accumOutGm;
    this->vec2ResGm = vec2ResGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
    this->stage1PGm = stage1PGm;
    this->stage1MGm = stage1MGm;
    this->stage1LGm = stage1LGm;
    this->prevPGm = prevPGm;
    this->prevMGm = prevMGm;
    this->prevLGm = prevLGm;
}

template <typename SFAT> __aicore__ inline void SFAVectorService<SFAT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SFAT> __aicore__ inline void SFAVectorService<SFAT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SFAT> __aicore__ inline void SFAVectorService<SFAT>::InitSoftmaxDefaultBuffer()
{
    Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
    Duplicate(softmaxSumDefaultUb, ConstInfo::FLOAT_ZERO, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::CalcAmlaScaleFromMax(LocalTensor<T> scaleUb,
                                                                    LocalTensor<KV_T> scaleKvUb,
                                                                    LocalTensor<T> tmpUb,
                                                                    LocalTensor<T> maxUb,
                                                                    uint32_t dealRowCount)
{
    Muls(tmpUb, maxUb, ((T)(-1.0)) * RECIP_OF_LN2, dealRowCount);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(tmpUb, tmpUb, RoundMode::CAST_ROUND, dealRowCount);
    AscendC::PipeBarrier<PIPE_V>();
    Muls(scaleUb, maxUb, RECIP_OF_LN2, dealRowCount);
    AscendC::PipeBarrier<PIPE_V>();
    Add(scaleUb, scaleUb, tmpUb, dealRowCount);
    AscendC::PipeBarrier<PIPE_V>();
    Muls(scaleUb, scaleUb, LN2, dealRowCount);
    AscendC::PipeBarrier<PIPE_V>();
    Exp(scaleUb, scaleUb, dealRowCount);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(scaleKvUb, scaleUb, RoundMode::CAST_ROUND, dealRowCount);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(scaleUb, scaleKvUb, RoundMode::CAST_NONE, dealRowCount);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ComputeLogSumExpAndCopyToGm(const RunInfo &info,
                                                                                         const MSplitInfo &mSplitInfo,
                                                                                         LocalTensor<T> &softmaxSumUb,
                                                                                         LocalTensor<T> &softmaxMaxUb)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint64_t baseOffset = mSplitInfo.nBufferStartM / 2;
    size_t size = mSplitInfo.vecDealM * FP32_BLOCK_ELEMENT_NUM;
    uint64_t accumTmpOutNum = CalcAccumOffset(info.bIdx, info.gS1Idx);
    uint64_t offset = (accumTmpOutNum * constInfo.kvHeadNum * constInfo.mBaseSize +
                       info.tndCoreStartKVSplitPos * constInfo.kvHeadNum * constInfo.mBaseSize +
                       mSplitInfo.nBufferStartM + mSplitInfo.vecStartM) *
                       FP32_BLOCK_ELEMENT_NUM;
    if (info.actualSingleProcessSInnerSize != 0) {
        LocalTensor<T> tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(tmp, softmaxSumUb[baseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseSumFdGm[offset], tmp, size);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);

        tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(tmp, softmaxMaxUb[baseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseMaxFdGm[offset], tmp, size);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    } else {
        matmul::InitOutput<T>(lseSumFdGm[offset], size, ConstInfo::FLOAT_ZERO);
        matmul::InitOutput<T>(lseMaxFdGm[offset], size, SOFTMAX_MIN_NUM);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ElewiseCompute(const RunInfo &info,
                                                                            const LocalTensor<T> &mmResUb,
                                                                            uint32_t dealRowCount, uint32_t columnCount)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), dealRowCount * columnCount);
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        uint64_t s2ValidSizeFirstPart = v0ValidSizeUb_.GetValue(128 + info.loop % MERGE_CACHE_GM_BUF_NUM);
        uint64_t s2ValidSizeSecondPart = v0ValidSizeUb_.GetValue(256 + info.loop % MERGE_CACHE_GM_BUF_NUM);
        int64_t s2ProcessSize = info.actualSingleProcessSInnerSize;
        int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * constInfo.sparseBlockSize);
        int64_t s2Mid = CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
        if (s2Mid > s2ProcessSize) {
            s2Mid = s2ProcessSize;
        }
        if (unlikely(s2ValidSizeFirstPart < s2Mid)) {
            int64_t s2StartCeilAlign = CeilAlign(s2ValidSizeFirstPart, 8);
            int64_t s2MidFloorAlign = s2Mid / 8 * 8;
            SetInfInBlk(mmResUb, dealRowCount, columnCount, s2ValidSizeFirstPart,
                        s2StartCeilAlign >= s2Mid ? s2Mid : s2StartCeilAlign);
            SetMidInf(mmResUb, dealRowCount, columnCount, s2StartCeilAlign, s2MidFloorAlign);
            SetInfInBlk(mmResUb, dealRowCount, columnCount,
                        s2StartCeilAlign <= s2MidFloorAlign ? s2MidFloorAlign : s2StartCeilAlign, s2Mid);
        }
        if (unlikely(s2ValidSizeSecondPart < s2ProcessSize - s2Mid)) {
            int64_t s2StartCeilAlign = CeilAlign(s2Mid + s2ValidSizeSecondPart, 8);
            int64_t s2EndFloorAlign = s2ProcessSize / 8 * 8;
            SetInfInBlk(mmResUb, dealRowCount, columnCount, s2Mid + s2ValidSizeSecondPart,
                        s2StartCeilAlign >= s2ProcessSize ? s2ProcessSize : s2StartCeilAlign);
            SetMidInf(mmResUb, dealRowCount, columnCount, s2StartCeilAlign, s2EndFloorAlign);
            SetInfInBlk(mmResUb, dealRowCount, columnCount,
                        s2StartCeilAlign <= s2EndFloorAlign ? s2EndFloorAlign : s2StartCeilAlign,
                        s2ProcessSize);
        }
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::SetInfInBlk(const LocalTensor<T> &mmResUb,
                                                                         uint32_t dealRowCount, uint32_t columnCount,
                                                                         uint64_t startId, uint64_t endId)
{
    if (startId >= endId) {
        return;
    }

    uint64_t startFloorAlignSize = startId / BLOCK_ELEMENT_NUM * BLOCK_ELEMENT_NUM;
    uint64_t notComputePreMaskOneBlk = (1 << (startId - startFloorAlignSize)) - 1;
    uint64_t notComputePostMaskOneBlk = ~((1 << (endId - startFloorAlignSize)) - 1);
    uint64_t notComputeMaskOneBlk = notComputePreMaskOneBlk ^ notComputePostMaskOneBlk;

    uint64_t maskOneBlk = ~notComputeMaskOneBlk;
    uint64_t mask[1] = {maskOneBlk};
    for (int i = 1; i < 8; i++) {
        mask[0] = mask[0] | (maskOneBlk << (i * 8));
    }
    for (uint64_t rowId = 0; rowId < dealRowCount; rowId += 8) {
        Duplicate(mmResUb[rowId * columnCount + startFloorAlignSize], SOFTMAX_MIN_NUM, mask,
                  1, CeilDiv(columnCount, 8), 0);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::SetMidInf(const LocalTensor<T> &mmResUb,
                                                                       uint32_t dealRowCount, uint32_t columnCount,
                                                                       uint64_t startId, uint64_t endId)
{
    if (startId >= endId) {
        return;
    }
    for (uint64_t rowId = 0; rowId < dealRowCount; rowId++) {
        Duplicate(mmResUb[rowId * columnCount + startId], SOFTMAX_MIN_NUM, endId - startId);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::SoftmaxFlashV2Compute(
    const RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop - 1) % (constInfo.preLoadNum);
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
    }
    if (actualColumnCount !=0) {
        SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
        SoftMaxTiling newTiling =
            SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);
        SoftmaxFlashV2<T, true, true, false, false, SFA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
        mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
        softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    } else {
        uint32_t dealRowCountAlign = SFAAlign(dealRowCount, FP32_BLOCK_ELEMENT_NUM);
        DataCopy(softmaxSumUb[softmaxOutOffset], inSumTensor, dealRowCountAlign);
        AscendC::PipeBarrier<PIPE_V>();
        DataCopy(softmaxMaxUb[softmaxOutOffset], inMaxTensor, dealRowCountAlign);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::AmlaVecCompute(
    const RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t calCount = dealRowCount;
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    // compute n(i)
    LocalTensor<T> nTmp = softmaxTmpUb.template ReinterpretCast<T>();
    LocalTensor<T> nUpdateTmp = nTmp[SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Muls(nTmp, softmaxMaxUb[softmaxOutOffset], ((T)(-1.0)) * RECIP_OF_LN2, calCount);

    AscendC::PipeBarrier<PIPE_V>();
    Cast(nTmp, nTmp, RoundMode::CAST_ROUND, calCount);
    AscendC::PipeBarrier<PIPE_V>();

    uint32_t prOutIdx = (info.loop - 1) % (constInfo.preLoadNum);
    uint32_t PreSoftmaxOutOffset = prOutIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    // n(i) - n(i-1)
    if (info.isFirstSInnerLoop) {
        Duplicate(nUpdateTmp, ConstInfo::FLOAT_ZERO, calCount); // n1=n0
    } else {
        Sub(nUpdateTmp, nTmp, nValueUb[PreSoftmaxOutOffset], calCount);
    }
    AscendC::PipeBarrier<PIPE_V>();
    // update n(i), DataCopy not support when calCount is not align 32B, so use Adds
    Adds(nValueUb[softmaxOutOffset], nTmp, ConstInfo::FLOAT_ZERO, calCount);
    AscendC::PipeBarrier<PIPE_V>();

    // update softmax res
    LocalTensor<T> nUpdateTmp2 = nTmp[2 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    LocalTensor<KV_T> nTmp_KvT = nTmp[3 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)].template ReinterpretCast<KV_T>();
    LocalTensor<T> tmpCofUb = nTmp[4 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    LocalTensor<T> epsUb = nTmp[5 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Muls(nUpdateTmp2, softmaxMaxUb[softmaxOutOffset], RECIP_OF_LN2, calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Add(nTmp, nUpdateTmp2, nTmp, calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Muls(nTmp, nTmp, LN2, calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Exp(nTmp, nTmp, calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Cast(nTmp_KvT, nTmp, RoundMode::CAST_ROUND, calCount);       // fp32->fp16/bf16
    AscendC::PipeBarrier<PIPE_V>();
    Cast(nUpdateTmp2, nTmp_KvT, RoundMode::CAST_NONE, calCount); // fp16/bf16->fp32
    AscendC::PipeBarrier<PIPE_V>();
    if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
        Mul(aMlaSumUb[softmaxOutOffset], softmaxSumUb[softmaxOutOffset], nUpdateTmp2, calCount);
    }
    if (actualColumnCount == 0) {
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        return;
    }
    LocalTensor<T> nTmp3 = nTmp[6 * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
    Brcb(nTmp3, nUpdateTmp2, (dealRowCount + 7) / 8, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    RowMuls(mmResUb, mmResUb, nTmp3, dealRowCount, columnCount, actualColumnCount);

    Div(tmpCofUb, nTmp, nUpdateTmp2, calCount); // cof(i)=tmpS32/tmpS16
    if (info.isFirstSInnerLoop) {
        Duplicate(cofValueUb[softmaxOutOffset], (T)1.0, calCount);       // cof_0=1
        AscendC::PipeBarrier<PIPE_V>();
        Div(epsUb, cofValueUb[softmaxOutOffset], tmpCofUb, calCount);    // 1 / cof(i)
    } else {
        AscendC::PipeBarrier<PIPE_V>();
        Div(epsUb, cofValueUb[PreSoftmaxOutOffset], tmpCofUb, calCount); // cof(i - 1) / cof(i)
    }
    AscendC::PipeBarrier<PIPE_V>();

    Adds(cofValueUb[softmaxOutOffset], tmpCofUb, ConstInfo::FLOAT_ZERO, calCount); // store cof(i)
    Adds(epsUb, epsUb, (T)(-1.0), calCount); // cof(i - 1) / cof(i) - 1
    AscendC::PipeBarrier<PIPE_V>();
    Muls(epsUb, epsUb, (T)1.5, calCount);    // (cof(i - 1) - cof(i)) / cof(i) * 1.5

    Maxs(nUpdateTmp, nUpdateTmp, (T)(-30.0), calCount); // N = max(n(i) - n(i-1), -30)
    AscendC::PipeBarrier<PIPE_V>();
    Adds(epsUb, epsUb, (T)(0.000001), calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Add(nUpdateTmp, nUpdateTmp, epsUb, calCount);
    AscendC::PipeBarrier<PIPE_V>();
    Muls(nUpdateTmp, nUpdateTmp, FLOAT_E_SCALAR, calCount); // N = N * pow(2, 23)
    AscendC::PipeBarrier<PIPE_V>();

    // nUpdate int32 out
    LocalTensor<int32_t> tmQue = outputBuff2.Get<int32_t>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    LocalTensor<int32_t> nInt32Out = tmQue[startRow];

    Cast(nInt32Out, nUpdateTmp, RoundMode::CAST_ROUND, dealRowCount);
    AscendC::PipeBarrier<PIPE_V>();

    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::DealBmm1ResBaseBlock(
    const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t loopId)
{
    uint32_t computeSize = dealRowCount * columnCount;
    uint64_t inOutGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                             (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow) * columnCount;
    LocalTensor<MM1_OUT_T> mmResUb = inputBuff1.Get<MM1_OUT_T>();
    mmResUb = mmResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM1_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(mmResUb, mm1ResGm[inOutGmOffset], computeSize);
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        if (loopId == 0) {
            WaitFlag<HardEvent::MTE2_S>(0);
        }
    }
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    ElewiseCompute(info, mmResUb, dealRowCount, columnCount);

    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<T> tmpAFloorUb = tmpBuff1.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();

    SoftmaxFlashV2Compute(info, mSplitInfo, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount,
                            info.actualSingleProcessSInnerSize);

    AscendC::PipeBarrier<PIPE_V>();
    AmlaVecCompute(info, mSplitInfo, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount,
                    info.actualSingleProcessSInnerSize);

    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<KV_T> tmpMMResCastTensor = outputBuff1.Get<KV_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);

    Cast(tmpMMResCastTensor, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(vec1ResGm[inOutGmOffset], tmpMMResCastTensor, computeSize);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ProcessAmlaNupdate(const RunInfo &info, const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    if (info.isFirstSInnerLoop) {
        return;
    }

    LocalTensor<int32_t> nUpdateTensor = outputBuff2.Get<int32_t>(); // shape:1/2*s1*g
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);

    constexpr uint32_t dGroupSize = 128U;
    constexpr uint32_t mSplitSize = 64U;
    constexpr uint32_t ONE_BLOCK_SIZE = 32U; // 32B

    uint32_t subMSize = SFAAlign(mSplitInfo.vecDealM, 16U);
    uint16_t elementPerBlock = ONE_BLOCK_SIZE / sizeof(int32_t);
    uint32_t loopCount = (subMSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = subMSize - (loopCount - 1) * mSplitSize;

    for (uint32_t loop = 0, processMSize = mSplitSize; loop < loopCount; loop++) {
        if (loop == (loopCount - 1)) {
            processMSize = tailSplitSize;
        }
        LocalTensor<int32_t> tmpQue = outputBuff1.Get<int32_t>();

        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        for (uint32_t i = 0; i < dGroupSize / elementPerBlock; i++) {
            Brcb(tmpQue[i * elementPerBlock],
                 nUpdateTensor[loop * mSplitSize],
                 static_cast<uint8_t>((processMSize + elementPerBlock - 1) / elementPerBlock),
                 {static_cast<uint16_t>(dGroupSize / elementPerBlock),
                  static_cast<uint16_t>(dGroupSize)});
        }

        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);

        uint64_t baseoffset = (info.bn2IdxInCurCore % constInfo.preLoadNum) * constInfo.bmm2ResUbSize +
                              (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + loop * mSplitSize) * constInfo.headDim;

        SetAtomicAdd<int32_t>();
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = static_cast<uint16_t>(processMSize);
        dataCopyParams.blockLen = dGroupSize * sizeof(int32_t) / ONE_BLOCK_SIZE;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = static_cast<uint16_t>((constInfo.headDim - dGroupSize) *
                                                         sizeof(int32_t) / ONE_BLOCK_SIZE);
        for (uint32_t i = 0; i < constInfo.headDim / dGroupSize; i++) {
            DataCopy(mm2ResInt32Gm[baseoffset + i * dGroupSize] ,tmpQue, dataCopyParams);
        }
        SetAtomicNone();
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ProcessVec1SingleBuf(const RunInfo &info,
                                                                                  const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t mSplitSize = info.actualSingleProcessSInnerSize == 0 ?
        16 : BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign;
    mSplitSize = mSplitSize / 8 * 8;

    if (mSplitSize > mSplitInfo.vecDealM) {
        mSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * mSplitSize;

    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = 256 * sizeof(int32_t);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPadExtParams<int32_t> padParams;
        DataCopyPad(v0ValidSizeUb_[128], kvValidSizeGm_[info.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2)],
                    dataCopyParams, padParams);
        SetFlag<HardEvent::MTE2_S>(0);
        if (unlikely(loopCount == 0)) {
            WaitFlag<HardEvent::MTE2_S>(0);
        }
    }
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm1ResBaseBlock(info, mSplitInfo, i * mSplitSize, dealSize, info.actualSingleProcessSInnerSizeAlign, i);
        pingpongFlag ^= 1;
    }
}

template <typename SFAT>
__aicore__ inline int32_t
SFAVectorService<SFAT>::GetSourceAwareMissCount(
    const RunInfo &runInfo)
{
    int32_t countIndex = static_cast<int32_t>(runInfo.bIdx);
    if constexpr (SFAT::mtpMode) {
        // MTP metadata is [T]. topKBaseOffset already identifies the active
        // query row; the fixed 2048 divisor compiles to a shift.
        countIndex = static_cast<int32_t>(
            runInfo.topKBaseOffset /
            static_cast<uint64_t>(SFA_OFFLOAD_SPARSE_INDICES_CAPACITY));
    }
    if (copyCountIndex_ != countIndex) {
        cachedCopyCount_ = copyCountsGm_.GetValue(countIndex);
        copyCountIndex_ = countIndex;
    }
    const int32_t missCount = cachedCopyCount_;
    const int32_t maxCopyCount = static_cast<int32_t>(copyCap_);
    ASSERT_MSG(
        missCount >= 0 &&
            missCount <= maxCopyCount,
        "copy_count exceeds the source-aware gather capacity.");
    if (missCount < 0 ||
        missCount > maxCopyCount) {
        return 0;
    }
    return missCount;
}

template <typename SFAT>
__aicore__ inline int64_t
SFAVectorService<SFAT>::GetStaggeredSparseIndex(
    int64_t virtualSparseIndex, const RunInfo &runInfo)
{
    constexpr int64_t chunkSize = 512;
    constexpr int64_t chunkCount = 4;
    constexpr int64_t sparseTokenCount = chunkSize * chunkCount;
    ASSERT_MSG(
        runInfo.sparseTokenCount == sparseTokenCount &&
            constInfo.s2BaseSize == chunkSize &&
            (virtualSparseIndex & (chunkSize - 1)) == 0,
        "source-aware stagger requires 2048 sparse tokens and 512-token tiles.");
    if (runInfo.sparseTokenCount != sparseTokenCount ||
        constInfo.s2BaseSize != chunkSize ||
        (virtualSparseIndex & (chunkSize - 1)) != 0) {
        return virtualSparseIndex;
    }

    const int64_t virtualChunk = virtualSparseIndex >> 9;
    const int64_t offsetInChunk = virtualSparseIndex & (chunkSize - 1);
    // Rotate the three proven stagger phases across requests and variable
    // MTP query rows so adjacent rows do not hit DRAM at the same phase.
    int64_t staggerPhase = static_cast<int64_t>(runInfo.bIdx % 3);
    if constexpr (SFAT::mtpMode) {
        const int64_t topkRow = static_cast<int64_t>(
            runInfo.topKBaseOffset /
            static_cast<uint64_t>(SFA_OFFLOAD_SPARSE_INDICES_CAPACITY));
        const int64_t queryRow = topkRow % 3;
        staggerPhase = (staggerPhase + queryRow) % 3;
    }
    // phase 0 -> C3,C2,C1,C0; phase 1 -> C2,C1,C0,C3;
    // phase 2 -> C1,C0,C3,C2.
    const int64_t firstChunk =
        3 - staggerPhase;
    const int64_t sourceChunk =
        (firstChunk + chunkCount - virtualChunk) & 3;
    return sourceChunk * chunkSize + offsetInChunk;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::GetRealS2Idx(
    int64_t s2GmOffset, int64_t &realS2Idx,
    int64_t topkGmBaseOffset, const RunInfo &runInfo)
{
    const int64_t topkGmIdx =
        (s2GmOffset + runInfo.s2Idx * constInfo.s2BaseSize) /
        constInfo.sparseBlockSize;
    if (unlikely(topkGmIdx >= runInfo.sparseValidBlockCount)) {
        realS2Idx = -1;
        return;
    }
    if (topkGmIdx < runInfo.sparseTokenCount) {
        const int32_t sparseIndex =
            topkGm_.GetValue(topkGmBaseOffset + topkGmIdx);
        realS2Idx =
            sparseIndex < 0 ? -1 : static_cast<int64_t>(sparseIndex);
        return;
    }

    const int64_t tailOffset =
        topkGmIdx - runInfo.sparseTokenCount;
    if (runInfo.tailSlotStart < 0 ||
        tailOffset >= runInfo.tailTokenCount) {
        realS2Idx = -1;
        return;
    }
    realS2Idx =
        static_cast<int64_t>(runInfo.tailSlotStart) + tailOffset;
}

template <typename SFAT>
__aicore__ inline int64_t SFAVectorService<SFAT>::GetKeyGmOffset(int64_t realS2Idx,
                                                                 const RunInfo &runInfo, int64_t s2IdLimit)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t realKeyGmOffset = 0;
    if constexpr (PAGE_ATTENTION) {
        int64_t blkTableIdx = realS2Idx / constInfo.kvCacheBlockSize;
        int64_t blkTableOffset = realS2Idx % constInfo.kvCacheBlockSize;
        realKeyGmOffset = blkTableGm_.GetValue(runInfo.bIdx * constInfo.maxBlockNumPerBatch + blkTableIdx) *
                                static_cast<int64_t>(constInfo.kvCacheBlockSize) *
                                static_cast<int64_t>(constInfo.kvHeadNum) +
                                blkTableOffset;
    } else {
        realKeyGmOffset = (runInfo.tensorBOffset +
                           realS2Idx * constInfo.kvHeadNum * constInfo.headDim) /
                           constInfo.headDim;
    }
    return realKeyGmOffset;
}

template <typename SFAT>
__aicore__ inline int64_t SFAVectorService<SFAT>::GetKeyRopeGmOffset(int64_t realS2Idx,
                                                                  const RunInfo &runInfo, int64_t s2IdLimit)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t realKeyRopeGmOffset = 0;
    realKeyRopeGmOffset = (runInfo.tensorBRopeOffset +
                           realS2Idx * constInfo.kvHeadNum * constInfo.headDimRope) /
                           constInfo.headDimRope;
    return realKeyRopeGmOffset;
}

template <typename SFAT>
__aicore__ inline int64_t SFAVectorService<SFAT>::GetDramKeyGmOffset(
    int64_t sourceTokenIdx, const RunInfo &runInfo)
{
    if (sourceTokenIdx < 0) {
        return -1;
    }
    const int64_t blockCol = sourceTokenIdx / constInfo.kvCacheBlockSize;
    const int64_t blockOffset = sourceTokenIdx % constInfo.kvCacheBlockSize;
    ASSERT_MSG(
        blockCol < static_cast<int64_t>(dramMaxBlockNum_),
        "active miss source token exceeds the DRAM block table.");
    if (blockCol >= static_cast<int64_t>(dramMaxBlockNum_)) {
        return -1;
    }
    const int32_t physicalBlock = dramBlockTableGm_.GetValue(
        static_cast<int64_t>(runInfo.bIdx) * dramMaxBlockNum_ + blockCol);
    ASSERT_MSG(
        physicalBlock >= 0,
        "active DRAM block-table entry must be non-negative.");
    if (physicalBlock < 0) {
        return -1;
    }
    return static_cast<int64_t>(physicalBlock) *
               constInfo.kvCacheBlockSize +
           blockOffset;
}

template <typename SFAT>
__aicore__ inline void
SFAVectorService<SFAT>::CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                       int64_t keyBNBOffset,int64_t s2IdLimit, const RunInfo &runInfo)
{
    if (keyBNBOffset < 0) {
        return;
    }
    int64_t validS2Count =
        (realS2Idx + constInfo.sparseBlockSize > s2IdLimit ? s2IdLimit - realS2Idx : constInfo.sparseBlockSize);
    DataCopyExtParams intriParams;
    intriParams.blockLen = validS2Count * constInfo.headDim * sizeof(KV_T);
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * 32 * 512 + (mte2Size - mte3Size) * constInfo.headDim],
                keyGm_[keyBNBOffset * constInfo.headDim], intriParams, padParams);
    intriParams.blockLen = validS2Count * constInfo.headDimRope * sizeof(KV_T);

    DataCopyPad(ropeMergUb_[mergeMte3Idx % 2 * 32 * 64 + (mte2Size - mte3Size) * constInfo.headDimRope],
                keyRopeGm_[keyBNBOffset * constInfo.headDimRope], intriParams, padParams);
    mte2Size += validS2Count;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::CopyInDramKv(
    int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
    int64_t sourceTokenIdx,
    const RunInfo &runInfo)
{
    if (sourceTokenIdx < 0) {
        return;
    }
    ASSERT_MSG(
        constInfo.sparseBlockSize == 1,
        "source-aware gather requires sparse_block_size=1.");
    const int64_t dramOffset =
        GetDramKeyGmOffset(sourceTokenIdx, runInfo);
    if (dramOffset < 0) {
        return;
    }

    const int64_t ubRow = mte2Size - mte3Size;
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = constInfo.headDim * sizeof(KV_T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    DataCopyPad(
        kvMergUb_[mergeMte3Idx % 2 * 32 * 512 +
                  ubRow * constInfo.headDim],
        dramKeyGm_[dramOffset * constInfo.headDim],
        copyParams, padParams);

    copyParams.blockLen = constInfo.headDimRope * sizeof(KV_T);
    DataCopyPad(
        ropeMergUb_[mergeMte3Idx % 2 * 32 * 64 +
                    ubRow * constInfo.headDimRope],
        dramKeyRopeGm_[dramOffset * constInfo.headDimRope],
        copyParams, padParams);
    mte2Size += 1;
}

template <typename SFAT>
template <bool ALL_MISS>
__aicore__ inline void
SFAVectorService<SFAT>::CopyInSourceAwareKvPair(
    int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
    int32_t sourceToken0, int32_t sourceToken1,
    int64_t destinationSlot0, int64_t destinationSlot1,
    int32_t &persistentDestination0,
    int32_t &persistentDestination1,
    int64_t s2IdLimit, const RunInfo &runInfo)
{
    const bool source0FromDram = ALL_MISS || sourceToken0 >= 0;
    const bool source1FromDram = ALL_MISS || sourceToken1 >= 0;
    persistentDestination0 = -1;
    persistentDestination1 = -1;
    if (!source0FromDram && !source1FromDram) {
        CopyInHbmKvPair(
            mte2Size, mte3Size, mergeMte3Idx,
            destinationSlot0, destinationSlot1,
            s2IdLimit, runInfo);
        return;
    }

    const int64_t keyOffset0 = GetKeyGmOffset(
        destinationSlot0, runInfo, s2IdLimit);
    const int64_t keyOffset1 = GetKeyGmOffset(
        destinationSlot1, runInfo, s2IdLimit);
    int64_t keySrcStride = 0;
    int64_t keyRopeSrcStride = 0;
    if (keyOffset0 >= 0 && keyOffset1 >= 0) {
        const int64_t keyOffsetDelta = keyOffset0 > keyOffset1
            ? keyOffset0 - keyOffset1
            : keyOffset1 - keyOffset0;
        keySrcStride =
            (keyOffsetDelta - constInfo.sparseBlockSize) *
            constInfo.headDim * sizeof(KV_T);
        if constexpr (!PAGE_ATTENTION) {
            const int64_t keyRopeOffset0 = GetKeyRopeGmOffset(
                destinationSlot0, runInfo, s2IdLimit);
            const int64_t keyRopeOffset1 = GetKeyRopeGmOffset(
                destinationSlot1, runInfo, s2IdLimit);
            const int64_t keyRopeOffsetDelta =
                keyRopeOffset0 > keyRopeOffset1
                    ? keyRopeOffset0 - keyRopeOffset1
                    : keyRopeOffset1 - keyRopeOffset0;
            keyRopeSrcStride =
                (keyRopeOffsetDelta - constInfo.sparseBlockSize) *
                constInfo.headDimRope * sizeof(KV_T);
        }
    }

    // Match CopyInHbmKvPair: its two-block DataCopy emits the lower physical
    // HBM address first.  Preserve that row order when one or both payloads
    // come directly from DRAM so Attention sees the split path's token order.
    const bool canUseHbmPairOrder =
        keyOffset0 >= 0 && keyOffset1 >= 0 &&
        keySrcStride >= 0 && keySrcStride < INT32_MAX &&
        (PAGE_ATTENTION ||
         (keyRopeSrcStride >= 0 && keyRopeSrcStride < INT32_MAX)) &&
        destinationSlot0 + constInfo.sparseBlockSize < s2IdLimit &&
        destinationSlot1 + constInfo.sparseBlockSize < s2IdLimit;
    const bool swapPair = canUseHbmPairOrder && keyOffset1 < keyOffset0;

    if (swapPair) {
        persistentDestination0 = source1FromDram
            ? static_cast<int32_t>(destinationSlot1) : -1;
        persistentDestination1 = source0FromDram
            ? static_cast<int32_t>(destinationSlot0) : -1;
        if (source1FromDram) {
            CopyInDramKv(
                mte2Size, mte3Size, mergeMte3Idx,
                sourceToken1, runInfo);
        } else {
            CopyInSingleKv(
                mte2Size, mte3Size, mergeMte3Idx,
                destinationSlot1, keyOffset1,
                s2IdLimit, runInfo);
        }
        if (source0FromDram) {
            CopyInDramKv(
                mte2Size, mte3Size, mergeMte3Idx,
                sourceToken0, runInfo);
        } else {
            CopyInSingleKv(
                mte2Size, mte3Size, mergeMte3Idx,
                destinationSlot0, keyOffset0,
                s2IdLimit, runInfo);
        }
        return;
    }

    persistentDestination0 = source0FromDram
        ? static_cast<int32_t>(destinationSlot0) : -1;
    persistentDestination1 = source1FromDram
        ? static_cast<int32_t>(destinationSlot1) : -1;
    if (source0FromDram) {
        CopyInDramKv(
            mte2Size, mte3Size, mergeMte3Idx,
            sourceToken0, runInfo);
    } else {
        CopyInSingleKv(
            mte2Size, mte3Size, mergeMte3Idx,
            destinationSlot0, keyOffset0,
            s2IdLimit, runInfo);
    }
    if (source1FromDram) {
        CopyInDramKv(
            mte2Size, mte3Size, mergeMte3Idx,
            sourceToken1, runInfo);
    } else {
        CopyInSingleKv(
            mte2Size, mte3Size, mergeMte3Idx,
            destinationSlot1, keyOffset1,
            s2IdLimit, runInfo);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::CopyInHbmKvPair(
    int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
    int64_t realS2Idx1, int64_t realS2Idx2, int64_t s2IdLimit,
    const RunInfo &runInfo)
{
    int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit);
    int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, runInfo, s2IdLimit);
    if (unlikely(keyOffset1 < 0 && keyOffset2 < 0)) {
        return;
    }

    int64_t keySrcStride = 0;
    int64_t keyRopeSrcStride = 0;
    if constexpr (PAGE_ATTENTION) {
        int64_t blkTableSrcStride =
        ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) :
        (keyOffset2 - keyOffset1)) - constInfo.sparseBlockSize);
        keySrcStride = blkTableSrcStride * constInfo.headDim * sizeof(KV_T);
        keyRopeSrcStride = blkTableSrcStride * constInfo.headDimRope * sizeof(KV_T);
    } else {
        int64_t keyRopeOffset1 = GetKeyRopeGmOffset(realS2Idx1, runInfo, s2IdLimit);
        int64_t keyRopeOffset2 = GetKeyRopeGmOffset(realS2Idx2, runInfo, s2IdLimit);
        keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) :
                        (keyOffset2 - keyOffset1)) - constInfo.sparseBlockSize) * constInfo.headDim * sizeof(KV_T);
        keyRopeSrcStride = ((keyRopeOffset1 > keyRopeOffset2 ? (keyRopeOffset1 - keyRopeOffset2) :
                            (keyRopeOffset2 - keyRopeOffset1)) - constInfo.sparseBlockSize) *
                             constInfo.headDimRope * sizeof(KV_T);
    }

    if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0 ||
        (!PAGE_ATTENTION && (keyRopeSrcStride >= INT32_MAX || keyRopeSrcStride < 0)) ||
        realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
        realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit)) {
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyOffset1, s2IdLimit, runInfo);
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyOffset2, s2IdLimit, runInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockLen = constInfo.sparseBlockSize * constInfo.headDim * sizeof(KV_T);
        intriParams.blockCount = (keyOffset1 >= 0) + (keyOffset2 >= 0);
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t startGmOffset = keyOffset1 > -1 ? keyOffset1 : keyOffset2;
        if (keyOffset2 > -1 && keyOffset2 < keyOffset1) {
            startGmOffset = keyOffset2;
        }
        DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * 32 * 512 + (mte2Size - mte3Size) * constInfo.headDim],
                    keyGm_[startGmOffset * constInfo.headDim], intriParams, padParams);

        intriParams.blockLen = constInfo.sparseBlockSize * constInfo.headDimRope * sizeof(KV_T);
        intriParams.dstStride = 0;
        intriParams.srcStride = keyRopeSrcStride;
        DataCopyPad(ropeMergUb_[mergeMte3Idx % 2 * 32 * 64 + (mte2Size - mte3Size) * constInfo.headDimRope],
                    keyRopeGm_[startGmOffset * constInfo.headDimRope], intriParams, padParams);
        mte2Size += ((keyOffset1 > -1) + (keyOffset2 > -1)) * constInfo.sparseBlockSize;
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::CopyInKv(
    int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
    int64_t realS2Idx1, int64_t realS2Idx2,
    const RunInfo &runInfo)
{
    int64_t s2IdLimit = runInfo.curActualSeqLenOri;
    if (constInfo.sparseMode == 3) {
        s2IdLimit =
            runInfo.curActualSeqLenOri - runInfo.actS1Size +
            runInfo.gS1Idx / constInfo.gSize + 1;
    }
    CopyInHbmKvPair(
        mte2Size, mte3Size, mergeMte3Idx,
        realS2Idx1, realS2Idx2, s2IdLimit, runInfo);
}

template <typename SFAT>
__aicore__ inline void
SFAVectorService<SFAT>::CopyMissToPersistentCache(
    int64_t ubRow, int32_t destinationSlot,
    const RunInfo &runInfo)
{
    if (destinationSlot < 0) {
        return;
    }
    const int64_t destinationOffset = GetKeyGmOffset(
        destinationSlot, runInfo, runInfo.curActualSeqLenOri);
    ASSERT_MSG(
        destinationOffset >= 0,
        "active miss destination exceeds the HBM block table.");
    if (destinationOffset < 0) {
        return;
    }

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = constInfo.headDim * sizeof(KV_T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(
        keyGm_[destinationOffset * constInfo.headDim],
        kvMergUb_[ubRow * constInfo.headDim], copyParams);

    copyParams.blockLen = constInfo.headDimRope * sizeof(KV_T);
    DataCopyPad(
        keyRopeGm_[destinationOffset * constInfo.headDimRope],
        ropeMergUb_[ubRow * constInfo.headDimRope], copyParams);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size,
                                                                 int64_t s2GmStartOffset, int64_t mergeMte3Idx,
                                                                 const RunInfo &runInfo)
{
    if (mte2Size <= mte3Size) {
        return;
    }
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = mte2Size - mte3Size;
    dataCopyParams.blockLen = constInfo.headDim * sizeof(KV_T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    DataCopyPad(kvMergeGm_[runInfo.loop % 4 * 512 * 576 + (s2GmStartOffset + mte3Size)*constInfo.headDim],
                kvMergUb_[mergeMte3Idx % 2 * 32 * 512], dataCopyParams);

    dataCopyParams.blockLen = constInfo.headDimRope * sizeof(KV_T);
    DataCopyPad(kvMergeGm_[runInfo.loop % 4 * 512 * 576 + 512 * 512 + (s2GmStartOffset + mte3Size) *
                constInfo.headDimRope], ropeMergUb_[mergeMte3Idx % 2 * 32 * 64], dataCopyParams);
}

template <typename SFAT>
template <bool ALIGNED_MISS>
__aicore__ inline void
SFAVectorService<SFAT>::CopyOutSourceAwareResult(
    int64_t mte2Size, int64_t mte3Size,
    int64_t s2GmStartOffset, int64_t mergeMte3Idx,
    const RunInfo &runInfo, int64_t missRangeSize,
    int64_t sourceRangeStart, bool hasActualMiss,
    const int32_t *persistentCopies,
    int32_t persistentCopyCount)
{
    CopyOutMrgeResult(
        mte2Size, mte3Size, s2GmStartOffset,
        mergeMte3Idx, runInfo);
    if constexpr (ALIGNED_MISS) {
        if (persistentCopyCount <= 0 || mte2Size <= mte3Size) {
            return;
        }
        for (int32_t copyIndex = 0;
             copyIndex < persistentCopyCount; ++copyIndex) {
            const uint32_t packedCopy = static_cast<uint32_t>(
                persistentCopies[copyIndex]);
            const int32_t destinationSlot = static_cast<int32_t>(
                packedCopy >> 5);
            const int64_t row = packedCopy & 31U;
            const int64_t ubRow =
                mergeMte3Idx % 2 * 32 + row;
            CopyMissToPersistentCache(
                ubRow, destinationSlot, runInfo);
        }
        return;
    }
    if (!hasActualMiss || mte2Size <= mte3Size ||
        mte3Size >= missRangeSize) {
        return;
    }

    const int64_t missEnd =
        mte2Size < missRangeSize ? mte2Size : missRangeSize;
    for (int64_t rangeOffset = mte3Size;
         rangeOffset < missEnd; ++rangeOffset) {
        const int64_t ubRow =
            mergeMte3Idx % 2 * 32 + rangeOffset - mte3Size;
        const int64_t sourceIndex =
            sourceRangeStart + rangeOffset;
        const int32_t destinationSlot =
            topkGm_.GetValue(
                runInfo.topKBaseOffset + sourceIndex);
        CopyMissToPersistentCache(
            ubRow, destinationSlot, runInfo);
    }
}

template <typename SFAT>
__aicore__ inline void
SFAVectorService<SFAT>::MergeKvTailContiguous(const RunInfo &runInfo)
{
    const int64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    const int64_t s2Pair =
        CeilDiv(s2ProcessSize, 2L * constInfo.sparseBlockSize);
    const int64_t s2GmStartOffset = GetSubBlockIdx() == 0
        ? 0
        : CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
    int64_t s2GmLimit = GetSubBlockIdx() == 0
        ? CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize
        : s2ProcessSize;
    if (s2GmLimit > s2ProcessSize) {
        s2GmLimit = s2ProcessSize;
    }

    const int64_t tailTokenOffset =
        runInfo.s2Idx * constInfo.s2BaseSize /
            constInfo.sparseBlockSize -
        runInfo.sparseTokenCount;
    int64_t logicalToken =
        static_cast<int64_t>(runInfo.tailSlotStart) +
        tailTokenOffset + s2GmStartOffset;
    int64_t remaining = s2GmLimit > s2GmStartOffset
        ? s2GmLimit - s2GmStartOffset
        : 0;
    int64_t s2IdLimit = runInfo.curActualSeqLenOri;
    if (constInfo.sparseMode == 3) {
        s2IdLimit =
            runInfo.curActualSeqLenOri - runInfo.actS1Size +
            runInfo.gS1Idx / constInfo.gSize + 1;
    }

    int64_t mergeMte3Idx = 0;
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);

    while (remaining > 0 && logicalToken < s2IdLimit) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);

        const int64_t blockOffset =
            logicalToken % constInfo.kvCacheBlockSize;
        int64_t copyCount = constInfo.kvCacheBlockSize - blockOffset;
        if (copyCount > 32) {
            copyCount = 32;
        }
        if (copyCount > remaining) {
            copyCount = remaining;
        }
        if (copyCount > s2IdLimit - logicalToken) {
            copyCount = s2IdLimit - logicalToken;
        }

        const int64_t keyOffset =
            GetKeyGmOffset(logicalToken, runInfo, s2IdLimit);
        if (unlikely(keyOffset < 0 || copyCount <= 0)) {
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            break;
        }

        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen =
            copyCount * constInfo.headDim * sizeof(KV_T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<KV_T> padParams;
        DataCopyPad(
            kvMergUb_[mergeMte3Idx % 2 * 32 * 512],
            keyGm_[keyOffset * constInfo.headDim],
            copyParams, padParams);

        copyParams.blockLen =
            copyCount * constInfo.headDimRope * sizeof(KV_T);
        DataCopyPad(
            ropeMergUb_[mergeMte3Idx % 2 * 32 * 64],
            keyRopeGm_[keyOffset * constInfo.headDimRope],
            copyParams, padParams);

        mte2Size += copyCount;
        CopyOutMrgeResult(
            mte2Size, mte3Size, s2GmStartOffset,
            mergeMte3Idx, runInfo);
        mte3Size = mte2Size;
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
        mergeMte3Idx++;
        logicalToken += copyCount;
        remaining -= copyCount;
    }

    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    v0ValidSizeUb_.SetValue(
        runInfo.loop % MERGE_CACHE_GM_BUF_NUM, mte2Size);
    SetFlag<AscendC::HardEvent::S_MTE3>(1);
    WaitFlag<AscendC::HardEvent::S_MTE3>(1);
    DataCopyExtParams validSizeCopyParams;
    validSizeCopyParams.blockCount = 1;
    validSizeCopyParams.blockLen = 128 * sizeof(int32_t);
    validSizeCopyParams.srcStride = 0;
    validSizeCopyParams.dstStride = 0;
    DataCopyPad(
        kvValidSizeGm_[
            runInfo.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2) +
            GetSubBlockIdx() * 128],
        v0ValidSizeUb_, validSizeCopyParams);
}

// b s1 k
template <typename SFAT>
template <bool SOURCE_ORDER, bool HAS_MISS, bool ALIGNED_MISS,
          bool ALL_MISS>
__aicore__ inline void SFAVectorService<SFAT>::MergeKvRange(
    const RunInfo &runInfo, int64_t s2GmStartOffset, int64_t s2GmLimit,
    uint32_t validSizePart, int64_t missRangeSize,
    int64_t sourceRangeStart)
{
    const int64_t topkGmBaseOffset = runInfo.topKBaseOffset;
    int64_t missBase = 0;
    int64_t s2IdLimit = 0;
    if constexpr (SOURCE_ORDER) {
        s2IdLimit = runInfo.curActualSeqLenOri;
        if (constInfo.sparseMode == 3) {
            s2IdLimit =
                runInfo.curActualSeqLenOri - runInfo.actS1Size +
                runInfo.gS1Idx / constInfo.gSize + 1;
        }
    }
    if constexpr (HAS_MISS && !ALIGNED_MISS) {
        missBase =
            static_cast<int64_t>(runInfo.bIdx) * copyCap_;
    }
    if constexpr (SOURCE_ORDER && SFAT::mtpMode) {
        const int64_t sourceMetadataSize =
            HAS_MISS && ALIGNED_MISS ? missRangeSize : 0;
        LoadSourceAwareMetadata<HAS_MISS && ALIGNED_MISS>(
            topkGmBaseOffset, sourceRangeStart,
            s2GmLimit - s2GmStartOffset, sourceMetadataSize);
    }

    int64_t mergeMte3Idx = 0;
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    int64_t realS2Idx0 = -1;
    int64_t realS2Idx1 = -1;
    bool needWaitMte3ToMte2 = true;
    bool flushHasActualMiss = false;
    // Pack (destination_slot, UB row) into one int32.  Hit-only pairs append
    // nothing, so persistent cache writeback scales with actual query miss
    // occurrences rather than scanning all 32 gathered rows per flush.
    int32_t persistentCopies[32];
    int32_t persistentCopyCount = 0;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    for (int64_t s2GmOffsetArray = s2GmStartOffset;
         s2GmOffsetArray < s2GmLimit;
         s2GmOffsetArray += 2 * constInfo.sparseBlockSize) {
        if (needWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            needWaitMte3ToMte2 = false;
        }
        if constexpr (SOURCE_ORDER) {
            const int64_t rangeOffset =
                s2GmOffsetArray - s2GmStartOffset;
            const int64_t sourceIndex0 =
                sourceRangeStart + rangeOffset;
            const int64_t sourceIndex1 = sourceIndex0 + 1;
            if constexpr (HAS_MISS) {
                if constexpr (ALIGNED_MISS) {
                    const bool source0FromDram = ALL_MISS ||
                        rangeOffset < missRangeSize;
                    const bool source1FromDram = ALL_MISS ||
                        rangeOffset + 1 < missRangeSize;
                    realS2Idx0 =
                        topkDestSlotsUb_.GetValue(rangeOffset);
                    realS2Idx1 =
                        topkDestSlotsUb_.GetValue(rangeOffset + 1);
                    const int64_t pairRowStart = mte2Size - mte3Size;
                    if (likely(!source0FromDram && !source1FromDram)) {
                        CopyInHbmKvPair(
                            mte2Size, mte3Size, mergeMte3Idx,
                            realS2Idx0, realS2Idx1,
                            s2IdLimit, runInfo);
                    } else {
                        // MTP miss metadata is a compact prefix.  Do not read
                        // source IDs for the hit-only suffix of a partial tile.
                        const int32_t sourceToken0 = source0FromDram
                            ? sourceTokenIdsUb_.GetValue(rangeOffset) : -1;
                        const int32_t sourceToken1 = source1FromDram
                            ? sourceTokenIdsUb_.GetValue(rangeOffset + 1) : -1;
                        flushHasActualMiss = true;
                        int32_t persistentDestination0 = -1;
                        int32_t persistentDestination1 = -1;
                        CopyInSourceAwareKvPair<ALL_MISS>(
                            mte2Size, mte3Size, mergeMte3Idx,
                            sourceToken0, sourceToken1,
                            realS2Idx0, realS2Idx1,
                            persistentDestination0,
                            persistentDestination1,
                            s2IdLimit, runInfo);
                        if (persistentDestination0 >= 0) {
                            ASSERT_MSG(
                                persistentCopyCount < 32 &&
                                    pairRowStart >= 0 && pairRowStart < 32,
                                "source-aware persistent copy metadata overflow.");
                            persistentCopies[persistentCopyCount++] =
                                static_cast<int32_t>(
                                    (static_cast<uint32_t>(
                                         persistentDestination0) << 5) |
                                    static_cast<uint32_t>(pairRowStart));
                        }
                        if (persistentDestination1 >= 0) {
                            ASSERT_MSG(
                                persistentCopyCount < 32 &&
                                    pairRowStart + 1 >= 0 &&
                                    pairRowStart + 1 < 32,
                                "source-aware persistent copy metadata overflow.");
                            persistentCopies[persistentCopyCount++] =
                                static_cast<int32_t>(
                                    (static_cast<uint32_t>(
                                         persistentDestination1) << 5) |
                                    static_cast<uint32_t>(pairRowStart + 1));
                        }
                    }
                } else {
                    const bool source0FromDram =
                        rangeOffset < missRangeSize;
                    const bool source1FromDram =
                        rangeOffset + 1 < missRangeSize;
                    realS2Idx0 = source0FromDram
                        ? sourceTokenIdsGm_.GetValue(missBase + sourceIndex0)
                        : topkGm_.GetValue(topkGmBaseOffset + sourceIndex0);
                    realS2Idx1 = source1FromDram
                        ? sourceTokenIdsGm_.GetValue(missBase + sourceIndex1)
                        : topkGm_.GetValue(topkGmBaseOffset + sourceIndex1);
                    flushHasActualMiss = flushHasActualMiss ||
                        source0FromDram || source1FromDram;
                    if (source0FromDram || source1FromDram) {
                        if (source0FromDram) {
                            CopyInDramKv(
                                mte2Size, mte3Size, mergeMte3Idx,
                                realS2Idx0, runInfo);
                        } else {
                            const int64_t keyOffset0 =
                                GetKeyGmOffset(
                                    realS2Idx0, runInfo, s2IdLimit);
                            CopyInSingleKv(
                                mte2Size, mte3Size, mergeMte3Idx,
                                realS2Idx0, keyOffset0,
                                s2IdLimit, runInfo);
                        }
                        if (source1FromDram) {
                            CopyInDramKv(
                                mte2Size, mte3Size, mergeMte3Idx,
                                realS2Idx1, runInfo);
                        } else {
                            const int64_t keyOffset1 =
                                GetKeyGmOffset(
                                    realS2Idx1, runInfo, s2IdLimit);
                            CopyInSingleKv(
                                mte2Size, mte3Size, mergeMte3Idx,
                                realS2Idx1, keyOffset1,
                                s2IdLimit, runInfo);
                        }
                    } else {
                        CopyInHbmKvPair(
                            mte2Size, mte3Size, mergeMte3Idx,
                            realS2Idx0, realS2Idx1,
                            s2IdLimit, runInfo);
                    }
                }
            } else {
                if constexpr (SFAT::mtpMode) {
                    realS2Idx0 =
                        topkDestSlotsUb_.GetValue(rangeOffset);
                    realS2Idx1 =
                        topkDestSlotsUb_.GetValue(rangeOffset + 1);
                } else {
                    realS2Idx0 =
                        topkGm_.GetValue(
                            topkGmBaseOffset + sourceIndex0);
                    realS2Idx1 =
                        topkGm_.GetValue(
                            topkGmBaseOffset + sourceIndex1);
                }
                CopyInHbmKvPair(
                    mte2Size, mte3Size, mergeMte3Idx,
                    realS2Idx0, realS2Idx1,
                    s2IdLimit, runInfo);
            }
        } else {
            GetRealS2Idx(
                s2GmOffsetArray, realS2Idx0,
                topkGmBaseOffset, runInfo);
            if (unlikely(realS2Idx0 < 0)) {
                CopyOutMrgeResult(
                    mte2Size, mte3Size, s2GmStartOffset,
                    mergeMte3Idx, runInfo);
                SetFlag<AscendC::HardEvent::MTE3_MTE2>(
                    mergeMte3Idx % 2);
                mergeMte3Idx++;
                break;
            }
            GetRealS2Idx(
                s2GmOffsetArray + constInfo.sparseBlockSize,
                realS2Idx1, topkGmBaseOffset, runInfo);
            CopyInKv(
                mte2Size, mte3Size, mergeMte3Idx,
                realS2Idx0, realS2Idx1, runInfo);
        }
        if ((mte2Size - mte3Size + 2 * constInfo.sparseBlockSize > 32) ||
            s2GmOffsetArray + 2 * constInfo.sparseBlockSize >=
                s2GmLimit) {
            if constexpr (SOURCE_ORDER && HAS_MISS) {
                CopyOutSourceAwareResult<ALIGNED_MISS>(
                    mte2Size, mte3Size, s2GmStartOffset,
                    mergeMte3Idx, runInfo, missRangeSize,
                    sourceRangeStart, flushHasActualMiss,
                    persistentCopies, persistentCopyCount);
                flushHasActualMiss = false;
                persistentCopyCount = 0;
            } else {
                CopyOutMrgeResult(
                    mte2Size, mte3Size, s2GmStartOffset,
                    mergeMte3Idx, runInfo);
            }
            mte3Size = mte2Size;
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2);
            mergeMte3Idx++;
            needWaitMte3ToMte2 = true;
        }
    }
    FinishMergeKvRange(
        runInfo, s2GmStartOffset, s2GmLimit, validSizePart,
        mergeMte3Idx, mte2Size);
}

template <typename SFAT>
template <bool ALIGNED_MISS>
__aicore__ inline void
SFAVectorService<SFAT>::MergeSourceAwareSparseRange(
    const RunInfo &runInfo, int64_t s2GmStartOffset,
    int64_t s2GmLimit, uint32_t validSizePart,
    int32_t missCount, int64_t sourceTileStart)
{
    ASSERT_MSG(
        constInfo.sparseBlockSize == 1 &&
            sourceTileStart >= 0 &&
            (s2GmStartOffset & 1) == 0 &&
            ((s2GmLimit - s2GmStartOffset) & 1) == 0 &&
            sourceTileStart + constInfo.s2BaseSize <=
                runInfo.sparseTokenCount,
        "source-aware range must be a complete sparse 512-token tile.");

    const int64_t sourceRangeStart =
        sourceTileStart + s2GmStartOffset;
    const int64_t rangeSize =
        s2GmLimit - s2GmStartOffset;
    const int64_t missAfterRangeStart =
        static_cast<int64_t>(missCount) - sourceRangeStart;
    const int64_t missRangeSize = missAfterRangeStart <= 0
        ? 0
        : (missAfterRangeStart < rangeSize ? missAfterRangeStart : rangeSize);
    if (missRangeSize == 0) {
        MergeKvRange<true, false>(
            runInfo, s2GmStartOffset, s2GmLimit,
            validSizePart, 0, sourceRangeStart);
    } else if constexpr (ALIGNED_MISS) {
        if (missRangeSize == rangeSize) {
            MergeKvRange<true, true, true, true>(
                runInfo, s2GmStartOffset, s2GmLimit,
                validSizePart, missRangeSize, sourceRangeStart);
        } else {
            MergeKvRange<true, true, true, false>(
                runInfo, s2GmStartOffset, s2GmLimit,
                validSizePart, missRangeSize, sourceRangeStart);
        }
    } else {
        MergeKvRange<true, true, false>(
            runInfo, s2GmStartOffset, s2GmLimit,
            validSizePart, missRangeSize, sourceRangeStart);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::FinishMergeKvRange(
    const RunInfo &runInfo, int64_t s2GmStartOffset,
    int64_t s2GmLimit, uint32_t validSizePart,
    int64_t mergeMte3Idx, int64_t mte2Size)
{
    if (unlikely(s2GmStartOffset + mte2Size < s2GmLimit)) {
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
        Duplicate(kvMergUb_, static_cast<KV_T>(0.0), constInfo.headDim);
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = constInfo.headDim * sizeof(KV_T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset++) {
            DataCopyPad(kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + s2GmOffset * constInfo.headDim],
                        kvMergUb_, dataCopyParams);
        }
        dataCopyParams.blockLen = constInfo.headDimRope * sizeof(KV_T);
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset++) {
            DataCopyPad(kvMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + 512 * constInfo.headDim +
                                   s2GmOffset * constInfo.headDimRope],
                        kvMergUb_, dataCopyParams);
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx & 1);
        mergeMte3Idx++;
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    v0ValidSizeUb_.SetValue(runInfo.loop % MERGE_CACHE_GM_BUF_NUM, mte2Size);
    SetFlag<AscendC::HardEvent::S_MTE3>(1);
    WaitFlag<AscendC::HardEvent::S_MTE3>(1);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = 128 * sizeof(int32_t);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(kvValidSizeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2) + validSizePart * 128],
                v0ValidSizeUb_, dataCopyParams);
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::MergeKv(const RunInfo &runInfo)
{
    int64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * constInfo.sparseBlockSize);
    int64_t s2Mid =
        CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
    if (s2Mid > s2ProcessSize) {
        s2Mid = s2ProcessSize;
    }

    uint32_t part = GetSubBlockIdx();
    const int64_t rangeStart = part == 0 ? 0 : s2Mid;
    const int64_t rangeEnd = part == 0 ? s2Mid : s2ProcessSize;
    if constexpr (SFAT::sourceAwareGather) {
        const int32_t missCount = GetSourceAwareMissCount(runInfo);
        const int64_t virtualTileStart =
            static_cast<int64_t>(runInfo.s2Idx) *
            constInfo.s2BaseSize;
        if (virtualTileStart < runInfo.sparseTokenCount) {
            int64_t sourceTileStart = virtualTileStart;
            // Stagger source tiles by default.  An including operator may
            // define SFA_CANONICAL_SOURCE_TILES when exact canonical
            // reduction order is required.
#if !defined(SFA_CANONICAL_SOURCE_TILES)
            if (missCount > 0) {
                sourceTileStart = GetStaggeredSparseIndex(
                    virtualTileStart, runInfo);
            }
#endif
            MergeSourceAwareSparseRange<SFAT::mtpMode>(
                runInfo, rangeStart, rangeEnd, part,
                missCount, sourceTileStart);
        } else if constexpr (SFAT::mtpMode) {
            MergeKvTailContiguous(runInfo);
        } else {
            // Tail keeps the original sparse-and-tail gather semantics.
            MergeKvRange<false, false>(
                runInfo, rangeStart, rangeEnd, part, 0, 0);
        }
    } else {
        MergeKvRange<false, false>(
            runInfo, rangeStart, rangeEnd, part, 0, 0);
    }
    return;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ProcessVec1L(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ?
            mSplitInfo.nBufferDealM :
            (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }

        CrossCoreWaitFlag(constInfo.syncC1V1);
        // vec1 compute
        ProcessVec1SingleBuf(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_MTE3>(
            constInfo.syncV1C2);
        CrossCoreWaitFlag(constInfo.syncC2V1);
        // add nUpdate to mm2ResGm
        if (info.actualSingleProcessSInnerSize != 0) {
            ProcessAmlaNupdate(info, mSplitInfo);
            CrossCoreSetFlag<ConstInfo::SFA_SYNC_MODE2, PIPE_MTE3>(
                constInfo.syncV1NupdateC2);
        }
        // move lse for flash decode
        if (info.s2Idx == info.curSInnerLoopTimes - 1) {
            if (info.tndIsS2SplitCore) {
                if constexpr (FLASH_DECODE) {
                    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
                    auto sumTensor = softmaxSumUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
                    auto maxTensor = softmaxMaxUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T)];
                    ComputeLogSumExpAndCopyToGm(info, mSplitInfo, sumTensor, maxTensor);
                }
            }
        }
    }
}

template <typename SFAT>
__aicore__ inline uint64_t SFAVectorService<SFAT>::CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx)
{
    return 0;
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ProcessVec2SingleBuf(const RunInfo &info,
                                                                                  const MSplitInfo &mSplitInfo)
{
    if (info.s2Idx + 1 != info.curSInnerLoopTimes) {
        return;
    }
    if (mSplitInfo.vecDealM == 0) {
        return;
    }

    ProcessVec2Inner(info, mSplitInfo, 0, mSplitInfo.vecDealM);
}

template <typename SFAT> __aicore__ inline void SFAVectorService<SFAT>::ProcessVec2L(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ?
            mSplitInfo.nBufferDealM :
            (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }
        CrossCoreWaitFlag(constInfo.syncC2V2);
        ProcessVec2SingleBuf(info, mSplitInfo);
    }
}

template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::ProcessVec2Inner(const RunInfo &info,
                                                                              const MSplitInfo &mSplitInfo,
                                                                              uint32_t mStartRow, uint32_t mDealSize)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDim;
    if constexpr (STAGE_MODE == SFA_STAGE_STAGE2) {
        mSplitSize = mSplitSize > 8U ? 8U : mSplitSize;
    }
    if (mSplitSize > mDealSize) {
        mSplitSize = mDealSize;
    }

    uint32_t loopCount = (mDealSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mDealSize - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(info, mSplitInfo, i * mSplitSize + mStartRow, dealSize,
                             constInfo.headDim, constInfo.headDim);
        pingpongFlag ^= 1;
    }
}


template <typename SFAT>
__aicore__ inline void SFAVectorService<SFAT>::GetConfusionTransposeTiling(
    int64_t numR, int64_t numC, const uint32_t stackBufferSize, const uint32_t typeSize,
    ConfusionTransposeTiling &tiling)
{
    (void)stackBufferSize;
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t height = numC;
    uint32_t width = numR;
    uint32_t highBlock = height / BLOCK_CUBE;
    uint32_t stride = height * blockSize * typeSize / ONE_BLK_SIZE;
    uint32_t repeat = width / blockSize;

    tiling.param0 = blockSize;
    tiling.param1 = height;
    tiling.param2 = width;
    tiling.param3 = highBlock;
    tiling.param4 = stride;
    tiling.param5 = repeat;
}

template <typename SFAT>
__aicore__ inline void
SFAVectorService<SFAT>::Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                                        uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
                                                        uint32_t actualColumnCount)
{
    LocalTensor<T> tmp = outputBuff1.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(tmp, bmm2ResUb, columnCount * dealRowCount);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    uint64_t accumTmpOutNum = CalcAccumOffset(info.bIdx, info.gS1Idx);
    uint64_t offset = accumTmpOutNum * constInfo.kvHeadNum * constInfo.mBaseSize * constInfo.headDim +
                      info.tndCoreStartKVSplitPos * constInfo.kvHeadNum * constInfo.mBaseSize * constInfo.headDim +
                      wsMStart * actualColumnCount;
    GlobalTensor<T> dst = accumOutGm[offset];
    if (info.actualSingleProcessSInnerSize== 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = actualColumnCount * sizeof(T);
        dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(dst, tmp, dataCopyParams);
    } else {
        matmul::InitOutput<T>(dst, dealRowCount * actualColumnCount, ConstInfo::FLOAT_ZERO);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SFAT>
__aicore__ inline void
SFAVectorService<SFAT>::Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb,
                                                           uint32_t wsMStart, uint32_t dealRowCount,
                                                           uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[info.attenOutOffset + wsMStart * actualColumnCount], attenOutUb, dataCopyParams);
    return;
}

template <typename SFAT>
__aicore__ inline void
SFAVectorService<SFAT>::Bmm2CastAndCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                                         uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
                                                         uint32_t actualColumnCount)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputBuff1.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) {
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    } else {
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    Bmm2DataCopyOutTrans(info, tmpBmm2ResCastTensor, wsMStart, dealRowCount, columnCount, actualColumnCount);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SFAT>
__aicore__ inline void
SFAVectorService<SFAT>::Stage1StateCopyOut(const RunInfo &info, LocalTensor<T> &stage1PUb,
                                                          uint32_t wsMStart, uint32_t dealRowCount,
                                                          uint32_t columnCount, uint32_t actualColumnCount,
                                                          uint32_t softmaxBaseOffset, uint32_t softmaxIdx)
{
    uint64_t statePBaseOffset = info.attenOutOffset + wsMStart * actualColumnCount;
    uint64_t stateScalarBaseOffset = info.attenOutOffset / constInfo.headDim + wsMStart;
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t rowAlign = SFAAlign(dealRowCount, FP32_BLOCK_ELEMENT_NUM);
    LocalTensor<T> pCopyUb = outputBuff1.Get<T>();
    LocalTensor<T> scratch = outputBuff2.Get<T>();
    LocalTensor<T> rawLUb = scratch;
    LocalTensor<T> scaleUb = scratch[rowAlign];
    LocalTensor<T> scaleTmpUb = scratch[2 * rowAlign];
    LocalTensor<KV_T> scaleKvUb = scratch[3 * rowAlign].template ReinterpretCast<KV_T>();
    LocalTensor<T> scaleRowUb = scratch[4 * rowAlign];
    LocalTensor<T> stage1MUb = softmaxMaxUb[softmaxIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + softmaxBaseOffset];
    LocalTensor<T> stage1LUb = aMlaSumUb[softmaxIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + softmaxBaseOffset];

    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    Adds(pCopyUb, stage1PUb, ConstInfo::FLOAT_ZERO, vec2ComputeSize);
    Adds(rawLUb, stage1LUb, ConstInfo::FLOAT_ZERO, dealRowCount);
    CalcAmlaScaleFromMax(scaleUb, scaleKvUb, scaleTmpUb, stage1MUb, dealRowCount);
    Div(rawLUb, rawLUb, scaleUb, dealRowCount);
    AscendC::PipeBarrier<PIPE_V>();
    Brcb(scaleRowUb, scaleUb, (dealRowCount + 7) / 8, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    RowDivs(pCopyUb, pCopyUb, scaleRowUb, dealRowCount, columnCount, actualColumnCount);
    AscendC::PipeBarrier<PIPE_V>();

    DataCopyExtParams pParams;
    pParams.blockCount = dealRowCount;
    pParams.blockLen = actualColumnCount * sizeof(T);
    pParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(T));
    pParams.dstStride = 0;

    DataCopyExtParams scalarParams;
    scalarParams.blockCount = 1;
    scalarParams.blockLen = dealRowCount * sizeof(T);
    scalarParams.srcStride = 0;
    scalarParams.dstStride = 0;
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    DataCopyPad(stage1PGm[statePBaseOffset], pCopyUb, pParams);
    DataCopyPad(stage1MGm[stateScalarBaseOffset], stage1MUb, scalarParams);
    DataCopyPad(stage1LGm[stateScalarBaseOffset], rawLUb, scalarParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SFAT>
__aicore__ inline void
SFAVectorService<SFAT>::Stage2MergeAndCopyOut(const RunInfo &info, LocalTensor<T> &curPUb,
                                                             uint32_t wsMStart, uint32_t dealRowCount,
                                                             uint32_t columnCount, uint32_t actualColumnCount,
                                                             uint32_t softmaxBaseOffset, uint32_t softmaxIdx)
{
    uint64_t statePBaseOffset = info.attenOutOffset + wsMStart * actualColumnCount;
    uint64_t stateScalarBaseOffset = info.attenOutOffset / constInfo.headDim + wsMStart;
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t rowAlign = SFAAlign(dealRowCount, FP32_BLOCK_ELEMENT_NUM);

    LocalTensor<T> prevPUb = inputBuff2.Get<T>();
    LocalTensor<T> scratch = outputBuff2.Get<T>();
    LocalTensor<T> prevMUb = scratch;
    LocalTensor<T> prevLUb = scratch[rowAlign];
    LocalTensor<T> curMUb = scratch[2 * rowAlign];
    LocalTensor<T> curLUb = scratch[3 * rowAlign];
    LocalTensor<T> mergedMUb = scratch[4 * rowAlign];
    LocalTensor<T> coefPrevUb = scratch[5 * rowAlign];
    LocalTensor<T> coefCurUb = scratch[6 * rowAlign];
    LocalTensor<T> mergedLUb = scratch[7 * rowAlign];

    DataCopyExtParams pParams;
    pParams.blockCount = dealRowCount;
    pParams.blockLen = actualColumnCount * sizeof(T);
    pParams.srcStride = 0;
    pParams.dstStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(T));

    DataCopyExtParams scalarParams;
    scalarParams.blockCount = 1;
    scalarParams.blockLen = dealRowCount * sizeof(T);
    scalarParams.srcStride = 0;
    scalarParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    DataCopyPad(prevPUb, prevPGm[statePBaseOffset], pParams, padParams);
    DataCopyPad(prevMUb, prevMGm[stateScalarBaseOffset], scalarParams, padParams);
    DataCopyPad(prevLUb, prevLGm[stateScalarBaseOffset], scalarParams, padParams);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);

    Adds(curMUb, softmaxMaxUb[softmaxIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + softmaxBaseOffset],
         ConstInfo::FLOAT_ZERO, dealRowCount);
    Adds(curLUb, aMlaSumUb[softmaxIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + softmaxBaseOffset],
         ConstInfo::FLOAT_ZERO, dealRowCount);
    AscendC::PipeBarrier<PIPE_V>();

    LocalTensor<T> scaleTmpUb = scratch[8 * rowAlign];
    LocalTensor<T> scaleUb = scratch[9 * rowAlign];
    LocalTensor<KV_T> scaleKvUb = scratch[10 * rowAlign].template ReinterpretCast<KV_T>();
    LocalTensor<T> scaleRowUb = scratch[11 * rowAlign];
    CalcAmlaScaleFromMax(scaleUb, scaleKvUb, scaleTmpUb, curMUb, dealRowCount);
    Div(curLUb, curLUb, scaleUb, dealRowCount);
    AscendC::PipeBarrier<PIPE_V>();
    Brcb(scaleRowUb, scaleUb, (dealRowCount + 7) / 8, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    RowDivs(curPUb, curPUb, scaleRowUb, dealRowCount, columnCount, actualColumnCount);
    AscendC::PipeBarrier<PIPE_V>();

    Max(mergedMUb, prevMUb, curMUb, dealRowCount);
    AscendC::PipeBarrier<PIPE_V>();
    Sub(coefPrevUb, prevMUb, mergedMUb, dealRowCount);
    Sub(coefCurUb, curMUb, mergedMUb, dealRowCount);
    AscendC::PipeBarrier<PIPE_V>();
    Exp(coefPrevUb, coefPrevUb, dealRowCount);
    Exp(coefCurUb, coefCurUb, dealRowCount);
    AscendC::PipeBarrier<PIPE_V>();
    Mul(prevLUb, prevLUb, coefPrevUb, dealRowCount);
    Mul(curLUb, curLUb, coefCurUb, dealRowCount);
    AscendC::PipeBarrier<PIPE_V>();
    Add(mergedLUb, prevLUb, curLUb, dealRowCount);
    AscendC::PipeBarrier<PIPE_V>();

    uint32_t brcbCount = rowAlign * FP32_BLOCK_ELEMENT_NUM;
    LocalTensor<T> coefPrevRowUb = scratch[8 * rowAlign];
    LocalTensor<T> coefCurRowUb = scratch[8 * rowAlign + brcbCount];
    LocalTensor<T> mergedLRowUb = scratch[8 * rowAlign + 2 * brcbCount];
    Brcb(coefPrevRowUb, coefPrevUb, (dealRowCount + 7) / 8, {1, 8});
    Brcb(coefCurRowUb, coefCurUb, (dealRowCount + 7) / 8, {1, 8});
    Brcb(mergedLRowUb, mergedLUb, (dealRowCount + 7) / 8, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();

    RowMuls(prevPUb, prevPUb, coefPrevRowUb, dealRowCount, columnCount, actualColumnCount);
    RowMuls(curPUb, curPUb, coefCurRowUb, dealRowCount, columnCount, actualColumnCount);
    AscendC::PipeBarrier<PIPE_V>();
    Add(curPUb, curPUb, prevPUb, vec2ComputeSize);
    AscendC::PipeBarrier<PIPE_V>();
    RowDivs(curPUb, curPUb, mergedLRowUb, dealRowCount, columnCount, actualColumnCount);
    AscendC::PipeBarrier<PIPE_V>();

    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    Bmm2CastAndCopyOut(info, curPUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
}

template <typename SFAT>
__aicore__ inline void
SFAVectorService<SFAT>::Bmm2ResCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                                     uint32_t dealRowCount, uint32_t columnCount,
                                                     uint32_t actualColumnCount)
{
    if constexpr (FLASH_DECODE) {
        if (info.tndIsS2SplitCore) {
            Bmm2FDDataCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
        } else {
            Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
        }
    } else {
        Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
    }
}

template <typename SFAT>
__aicore__ inline void
SFAVectorService<SFAT>::DealBmm2ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                           uint32_t startRow, uint32_t dealRowCount,
                                                           uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t mStart = mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
    uint64_t srcGmOffset = (info.bn2IdxInCurCore % constInfo.preLoadNum) * constInfo.bmm2ResUbSize +
                            mStart * columnCount;
    LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputBuff1.Get<MM2_OUT_T>();
    tmpBmm2ResUb = tmpBmm2ResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset], vec2ComputeSize);

    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    bmm2ResUb.SetSize(vec2ComputeSize);
    LocalTensor<T> absBmm2ResUb = bmm2ResUb.template ReinterpretCast<T>();
    Abs(absBmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t> cmpMaskUb = absBmm2ResUb.template ReinterpretCast<uint8_t>();
    CompareScalar(cmpMaskUb, absBmm2ResUb, (T)1e10, CMPMODE::LE, vec2ComputeSize);
    AscendC::PipeBarrier<PIPE_V>();
    Select(tmpBmm2ResUb, cmpMaskUb, tmpBmm2ResUb, ConstInfo::FLOAT_ZERO,
           SELMODE::VSEL_TENSOR_SCALAR_MODE, vec2ComputeSize);
    AscendC::PipeBarrier<PIPE_V>();
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t idx = info.loop % (constInfo.preLoadNum);
    LocalTensor<T> tmpSumUb = v0ValidSizeBuff.Get<T>()[384];
    Brcb(tmpSumUb, aMlaSumUb[idx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset], (dealRowCount + 7) / 8, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (STAGE_MODE == SFA_STAGE_STAGE1) {
        Adds(bmm2ResUb, tmpBmm2ResUb, ConstInfo::FLOAT_ZERO, vec2ComputeSize);
        AscendC::PipeBarrier<PIPE_V>();
        Stage1StateCopyOut(info, bmm2ResUb, mStart, dealRowCount, columnCount, actualColumnCount, baseOffset, idx);
    }
    if constexpr (STAGE_MODE == SFA_STAGE_STAGE2) {
        Adds(bmm2ResUb, tmpBmm2ResUb, ConstInfo::FLOAT_ZERO, vec2ComputeSize);
        AscendC::PipeBarrier<PIPE_V>();
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
        Stage2MergeAndCopyOut(info, bmm2ResUb, mStart, dealRowCount, columnCount, actualColumnCount, baseOffset, idx);
        return;
    }
    RowDivs(bmm2ResUb, tmpBmm2ResUb, tmpSumUb, dealRowCount, columnCount, actualColumnCount);
    AscendC::PipeBarrier<PIPE_V>();
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    Bmm2ResCopyOut(info, bmm2ResUb, mStart, dealRowCount, columnCount, actualColumnCount);
}

template <typename SFAT>
__aicore__ inline void
SFAVectorService<SFAT>::RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 0;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 1;
    repeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    if (columnRepeatCount <= dealRowCount) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub, dtypeMask, dealRowCount, repeatParamsDiv);
            offset += dtypeMask;
        }
    } else {
        BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0RepStride = 8;
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8;
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub[i * FP32_BLOCK_ELEMENT_NUM], dtypeMask, columnRepeatCount,
                columnRepeatParams);
            offset += columnCount;
        }
    }
    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub, dRemain, dealRowCount, repeatParamsDiv);
    }
}

template <typename SFAT>
__aicore__ inline void
SFAVectorService<SFAT>::RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;   // 32/4 * 2 = 16
    }

    uint32_t dLoop = actualColumnCount / repeatElementNum;
    uint32_t dRemain = actualColumnCount % repeatElementNum;
    if (columnCount < REPEATE_STRIDE_UP_BOUND * blockElementNum) {
        BinaryRepeatParams repeatParams;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0RepStride = columnCount / blockElementNum;
        repeatParams.src1RepStride = 1;
        repeatParams.dstRepStride = columnCount / blockElementNum;

        if (dLoop <= dealRowCount) {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, dealRowCount, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            BinaryRepeatParams columnRepeatParams;
            columnRepeatParams.src0BlkStride = 1;
            columnRepeatParams.src1BlkStride = 0;
            columnRepeatParams.dstBlkStride = 1;
            columnRepeatParams.src0RepStride = 8;
            columnRepeatParams.src1RepStride = 0;
            columnRepeatParams.dstRepStride = 8;
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum,
                    dLoop, columnRepeatParams);
            }
        }

        if (dRemain > 0) {
            Mul(dstUb[dLoop * repeatElementNum], src0Ub[dLoop * repeatElementNum], src1Ub, dRemain, dealRowCount,
                repeatParams);
        }
    } else {
        BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = 8;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = 8;
        repeatParams.dstBlkStride = 1;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum, dLoop,
                repeatParams);
            if (dRemain > 0) {
                Mul(dstUb[i * columnCount + dLoop * repeatElementNum],
                    src0Ub[i * columnCount + dLoop * repeatElementNum], src1Ub[i * blockElementNum], dRemain, 1,
                    repeatParams);
            }
        }
    }
}

#endif // SPARSE_TAIL_ATTENTION_SERVICE_VECTOR_MLA_H
