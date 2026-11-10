/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_quant_sparse_flash_attention_vllm_service_vector_mla.h
 * \brief
 */
#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_VLLM_SERVICE_VECTOR_MLA_H
#define KV_QUANT_SPARSE_FLASH_ATTENTION_VLLM_SERVICE_VECTOR_MLA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "kv_quant_sparse_flash_attention_vllm_common.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename QSFAT> class QSFAVectorService {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using KV_T = typename QSFAT::kvType;
    using K_ROPE_T = typename QSFAT::kRopeType;
    using OUT_T = typename QSFAT::outputType;
    using UPDATE_T = T;
    using MM1_OUT_T = float;
    using MM2_OUT_T = float;
    bool NO_AMLA = true;

    __aicore__ inline QSFAVectorService(){};
    __aicore__ inline void ProcessVec1L(const RunInfo &info);
    __aicore__ inline void ProcessVec2L(const RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct ConstInfo &constInfo,
                                      const KvQuantSparseFlashAttentionVllmTilingDataMla *__restrict tilingData);
    __aicore__ inline void InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm);
    __aicore__ inline void InitVec0GlobalTensor(const GlobalTensor<int32_t> &kvValidSizeGm,
                                                const GlobalTensor<K_ROPE_T> &kvMergeGm,
                                                const GlobalTensor<K_ROPE_T> &keyRopeGm,
                                                const GlobalTensor<KV_T> &keyGm,
                                                const GlobalTensor<int32_t> &blkTableGm);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<K_ROPE_T> vec1ResGm,
                                                GlobalTensor<int32_t> actualSeqLengthsQGm,
                                                GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm,
                                                GlobalTensor<T> lseSumFdGm, GlobalTensor<int32_t> topKGm,
                                                GlobalTensor<T> softmaxMaxGm, GlobalTensor<T> softmaxSumGm);
    __aicore__ inline void InitVec2GlobalTensor(GlobalTensor<T> accumOutGm, GlobalTensor<UPDATE_T> vec2ResGm,
                                                GlobalTensor<MM2_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitSoftmaxDefaultBuffer();
    // ================================Base Vector==========================================
    __aicore__ inline void RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    // ================================Vector0==========================================
    __aicore__ inline void DecodeTopkPairOptimized(int32_t idx0, int32_t idx1, int64_t posInBlock, int64_t& realS2Idx0,
        int64_t& realS2Idx1);
    __aicore__ inline void GetRealS2IdxOptimized(int64_t topkIdx, int64_t posInBlock, int64_t& realS2Idx0, int64_t&
        realS2Idx1, int64_t topkGmBaseOffset);
    __aicore__ inline int64_t GetKeyBNBOffsetOptimized(int64_t realS2Idx, int64_t blkTableBase, int64_t s2IdLimit, const
        RunInfo& kvVecRunInfo);
    __aicore__ inline void CopyInSingleKvOptimized(int64_t& mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t
        realS2Idx, int64_t keyBNBOffset, int64_t s2IdLimit, const RunInfo& kvVecRunInfo);
    __aicore__ inline void CopyInKvOptimized(int64_t& mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t
        realS2Idx1, int64_t realS2Idx2, int64_t s2IdLimit, int64_t blkTableBase, const RunInfo& kvVecRunInfo);
    __aicore__ inline bool PrepareAddressBatch( int64_t topkBase, int32_t count, int64_t s2IdLimit, const RunInfo& info,
        bool wholeQuery = false);
    __aicore__ inline bool PrepareQueryAddressCache( int64_t topkBase, int64_t s2IdLimit, const RunInfo& info);
    __aicore__ inline bool PrepareQueryAddresses(int64_t topkBase, int64_t s2IdLimit, const RunInfo& info);
    __aicore__ inline void MergeKvOptimized(const RunInfo& kvVecRunInfo);
    __aicore__ inline void CopyOutMrgeResultOptimized(int64_t mte2Size, int64_t mte3Size, int64_t s2GmStartOffset,
        int64_t mergeMte3Idx, const RunInfo& kvVecRunInfo);
    __aicore__ inline void MergeKv(const RunInfo &runInfo);
    __aicore__ inline void SetInfInBlk(const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount,
                                       uint64_t startId, uint64_t endId);
    __aicore__ inline void SetMidInf(const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount,
                                     uint64_t startId, uint64_t endId);
    // ================================Vector1==========================================
    __aicore__ inline void ProcessVec1SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm1ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount, uint32_t loopId);
    __aicore__ inline void SoftmaxFlashV2Compute(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                 LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);
    __aicore__ inline void ElewiseCompute(const RunInfo &info, const LocalTensor<T> &mmResUb, uint32_t dealRowCount,
                                          uint32_t columnCount);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                       LocalTensor<T> &softmaxSumUb, LocalTensor<T> &softmaxMaxUb);
    __aicore__ inline void CopyFALseToGm(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                         LocalTensor<T> &softmaxSumUb, LocalTensor<T> &softmaxMaxUb);
    // ================================Vecotr2==========================================
    __aicore__ inline void ProcessVec2SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm2ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void ProcessVec2Inner(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                            uint32_t mStartRow, uint32_t mDealSize);
    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb, uint32_t wsMStart,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void Bmm2ResCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                          uint32_t dealRowCount, uint32_t columnCount,
                                          uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                              uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                             uint32_t dealRowCount, uint32_t columnCount,
                                             uint32_t actualColumnCount);
    __aicore__ inline uint64_t CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx);
    __aicore__ inline void GetConfusionTransposeTiling(int64_t numR, int64_t numC, const uint32_t stackBufferSize,
                                                       const uint32_t typeSize, ConfusionTransposeTiling &tiling);

    // BLOCK和REPEAT的字节数
    static constexpr uint64_t BYTE_BLOCK = 32UL;
    static constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
    // BLOCK和REPEAT的FP32元素数
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
    // repeat stride不能超过256
    static constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;

private:
    static constexpr bool PAGE_ATTENTION = QSFAT::pageAttention;
    static constexpr int TEMPLATE_MODE = QSFAT::templateMode;
    static constexpr bool FLASH_DECODE = QSFAT::flashDecode;
    static constexpr QSFA_LAYOUT LAYOUT_T = QSFAT::layout;
    static constexpr QSFA_LAYOUT KV_LAYOUT_T = QSFAT::kvLayout;

    static constexpr uint64_t MERGE_CACHE_GM_BUF_NUM = 4;
    static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_INPUT_BUF1_PONG_FLAG = 3;
    static constexpr uint64_t SYNC_INPUT_BUF2_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF1_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF2_FLAG = 5;
    static constexpr uint32_t INPUT1_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_32K;
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_512B / sizeof(T);
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = ConstInfo::BUFFER_SIZE_BYTE_32K / sizeof(T);  // 32768/4=8096
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);                                // 32/4=8
    static constexpr uint32_t LIMIT_DEAL_ROW = 16U;
    static constexpr T FLOAT_E_SCALAR = 8388608;
    static constexpr T LN2 = 0.6931471805599453094172;
    static constexpr T RECIP_OF_LN2 = 1 / LN2;
    static constexpr T SOFTMAX_MIN_NUM = -2e38;

    const KvQuantSparseFlashAttentionVllmTilingDataMla *__restrict tilingData;

    uint32_t pingpongFlag = 0U;
    ConstInfo constInfo = {};

    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<K_ROPE_T> vec1ResGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<T> vec2ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<T> softmaxMaxGm;
    GlobalTensor<T> softmaxSumGm;
    GlobalTensor<int32_t> blkTableGm_;

    GlobalTensor<K_ROPE_T> kvMergeGm_;
    GlobalTensor<K_ROPE_T> keyRopeGm_;
    GlobalTensor<KV_T> keyGm_;
    GlobalTensor<int32_t> topkGm_;
    GlobalTensor<int32_t> kvValidSizeGm_;

    // ================================Local Buffer区====================================
    TBuf<> inputBuff1;  // 32K * 2
    TBuf<> inputBuff2;  // 32K
    TBuf<> outputBuff1; // 32K
    TBuf<> outputBuff2; // 4K

    TBuf<> tmpBuff1;         // 32K
    TBuf<> tmpBuff2;         // 8K
    TBuf<> v0ValidSizeBuff;  // 8K

    TBuf<> softmaxMaxBuff;        // PRE_LOAD_NUM * 1K
    TBuf<> softmaxExpBuff;        // PRE_LOAD_NUM * 1K
    TBuf<> softmaxSumBuff;        // PRE_LOAD_NUM * 1K
    TBuf<> softmaxMaxDefaultBuff; // 1K
    TBuf<> softmaxSumDefaultBuff; // 1K

    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;
    LocalTensor<KV_T> kvMergUb_;
    LocalTensor<int32_t> v0ValidSizeUb_;
    TBuf<> addressTableBuff_;
    static constexpr uint32_t MTE2_SLOT_NUM = 3;
    // INT8 arch22 layouts are fixed by the tiling checks. Preserve compile-time
    // row strides so gather address arithmetic does not gain runtime overhead.
    static constexpr uint32_t combineBytes_ = QSFAT::noRope ? 528U : 656U;
    static constexpr uint32_t combineDim_ = combineBytes_;
    static constexpr uint32_t combineDimAlign_ = QSFAT::noRope ? 544U : 672U;
    static constexpr uint32_t mte2SlotElems_ = 32 * combineDimAlign_;
    GlobalTensor<int64_t> topkI64Gm_;
    int64_t sparseBlockSize_ = 1;
    int64_t sparseBlockCount_ = 0;
    int64_t s2BaseSize_ = 0;
    // The address-table buffer is a 4-KiB allocation; one page-table entry is INT32.
    static constexpr int32_t ADDRESS_TABLE_CACHE_PAGES =
        ConstInfo::BUFFER_SIZE_BYTE_4K / sizeof(int32_t);
    static constexpr int32_t ADDRESS_TABLE_SEGMENT_PAGES = ADDRESS_TABLE_CACHE_PAGES;
    // Temporary table staging reuses the existing 32-KiB inputBuff2.
    static constexpr int32_t ADDRESS_TABLE_MAX_PAGES =
        ConstInfo::BUFFER_SIZE_BYTE_32K / sizeof(int32_t);
    static constexpr int32_t ADDRESS_RESULT_CAPACITY =
        ConstInfo::BUFFER_SIZE_BYTE_8K / sizeof(int32_t) - 512;
    int32_t largeTableCount_ = 0;
    int32_t largeTableBatch_ = -1;
    int32_t largeTableBegin_ = -1;
    int32_t addressTableBatch_ = -1;
    bool addressTableValid_ = false;
    bool queryAddressesValid_ = false;
    int32_t queryAddressCount_ = 0;
    int32_t kvCacheBlockShift_ = -1;
    int64_t kvCacheBlockMask_ = 0;
};

template <typename QSFAT> __aicore__ inline void QSFAVectorService<QSFAT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K * 2); // 2:pingpong
    pipe->InitBuffer(inputBuff2, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputBuff2, ConstInfo::BUFFER_SIZE_BYTE_4K);

    pipe->InitBuffer(tmpBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff2, ConstInfo::BUFFER_SIZE_BYTE_8K);
    pipe->InitBuffer(v0ValidSizeBuff, ConstInfo::BUFFER_SIZE_BYTE_8K);

    pipe->InitBuffer(softmaxMaxBuff, ConstInfo::BUFFER_SIZE_BYTE_512B * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxExpBuff, ConstInfo::BUFFER_SIZE_BYTE_512B * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxSumBuff, ConstInfo::BUFFER_SIZE_BYTE_512B * constInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_512B);
    pipe->InitBuffer(softmaxSumDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_512B);

    softmaxMaxUb = softmaxMaxBuff.Get<T>();
    softmaxSumUb = softmaxSumBuff.Get<T>();
    softmaxExpUb = softmaxExpBuff.Get<T>();

    softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
    softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();

    kvMergUb_ = inputBuff1.Get<KV_T>();

    v0ValidSizeUb_ = v0ValidSizeBuff.Get<int32_t>();
    pipe->InitBuffer(addressTableBuff_, ConstInfo::BUFFER_SIZE_BYTE_4K);
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::InitParams(const struct ConstInfo &constInfo,
                                     const KvQuantSparseFlashAttentionVllmTilingDataMla *__restrict tilingData)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;
    sparseBlockSize_ = constInfo.sparseBlockSize;
    sparseBlockCount_ = constInfo.sparseBlockCount;
    s2BaseSize_ = constInfo.s2BaseSize;
    int64_t block = constInfo.kvCacheBlockSize;
    if (block > 0 && (block & (block - 1)) == 0) {
        kvCacheBlockShift_ = 0;
        kvCacheBlockMask_ = block - 1;
        while (block > 1) {
            block >>= 1;
            ++kvCacheBlockShift_;
        }
    }

}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::InitMm2ResInt32GmGlobalTensor(GlobalTensor<int32_t> mm2ResInt32Gm)
{
    this->mm2ResInt32Gm = mm2ResInt32Gm;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitVec0GlobalTensor(
    const GlobalTensor<int32_t> &kvValidSizeGm, const GlobalTensor<K_ROPE_T> &kvMergeGm,
    const GlobalTensor<K_ROPE_T> &keyRopeGm, const GlobalTensor<KV_T> &keyGm, const GlobalTensor<int32_t> &blkTableGm)
{
    this->kvMergeGm_ = kvMergeGm;
    this->keyRopeGm_ = keyRopeGm;
    this->keyGm_ = keyGm;
    this->blkTableGm_ = blkTableGm;
    this->kvValidSizeGm_ = kvValidSizeGm;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<K_ROPE_T> vec1ResGm,
    GlobalTensor<int32_t> actualSeqLengthsQGm, GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm,
    GlobalTensor<T> lseSumFdGm, GlobalTensor<int32_t> topKGm,
    GlobalTensor<T> softmaxMaxGm, GlobalTensor<T> softmaxSumGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->actualSeqLengthsQGm = actualSeqLengthsQGm;
    this->actualSeqLengthsKVGm = actualSeqLengthsKVGm;
    this->lseMaxFdGm = lseMaxFdGm;
    this->lseSumFdGm = lseSumFdGm;
    this->topkGm_ = topKGm;
    topkI64Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(const_cast<__gm__ int32_t*>(topKGm.GetPhyAddr())));
    this->softmaxMaxGm = softmaxMaxGm;
    this->softmaxSumGm = softmaxSumGm;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::InitVec2GlobalTensor(GlobalTensor<T> accumOutGm,
                                                                      GlobalTensor<T> vec2ResGm,
                                                                      GlobalTensor<MM2_OUT_T> mm2ResGm,
                                                                      GlobalTensor<OUT_T> attentionOutGm)
{
    this->accumOutGm = accumOutGm;
    this->vec2ResGm = vec2ResGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename QSFAT> __aicore__ inline void QSFAVectorService<QSFAT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename QSFAT> __aicore__ inline void QSFAVectorService<QSFAT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename QSFAT> __aicore__ inline void QSFAVectorService<QSFAT>::InitSoftmaxDefaultBuffer()
{
    Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, SOFTMAX_TMP_BUFFER_OFFSET);
    Duplicate(softmaxSumDefaultUb, ConstInfo::FLOAT_ZERO, SOFTMAX_TMP_BUFFER_OFFSET);
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ComputeLogSumExpAndCopyToGm(const RunInfo &info,
                                                                                         const MSplitInfo &mSplitInfo,
                                                                                         LocalTensor<T> &softmaxSumUb,
                                                                                         LocalTensor<T> &softmaxMaxUb)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint64_t qsfaBaseOffset = mSplitInfo.nBufferStartM / 2;
    size_t qsfaSize = mSplitInfo.vecDealM * FP32_BLOCK_ELEMENT_NUM;
    uint64_t qsfaAccumTmpOutNum = CalcAccumOffset(info.bIdx, info.gS1Idx);
    uint64_t qsfaOffset = (qsfaAccumTmpOutNum * constInfo.kvHeadNum * constInfo.mBaseSize +              // taskoffset
                       info.tndCoreStartKVSplitPos * constInfo.kvHeadNum * constInfo.mBaseSize + // 份数offset
                       mSplitInfo.nBufferStartM + mSplitInfo.vecStartM) *
                       FP32_BLOCK_ELEMENT_NUM; // m轴offset
    if (info.actualSingleProcessSInnerSize != 0) {
        LocalTensor<T> qsfaTmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(qsfaTmp, softmaxSumUb[qsfaBaseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseSumFdGm[qsfaOffset], qsfaTmp, qsfaSize);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);

        qsfaTmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(qsfaTmp, softmaxMaxUb[qsfaBaseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseMaxFdGm[qsfaOffset], qsfaTmp, qsfaSize);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    } else {
        matmul::InitOutput<T>(lseSumFdGm[qsfaOffset], qsfaSize, ConstInfo::FLOAT_ZERO);
        matmul::InitOutput<T>(lseMaxFdGm[qsfaOffset], qsfaSize, SOFTMAX_MIN_NUM);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::CopyFALseToGm(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                        LocalTensor<T> &softmaxSumUb, LocalTensor<T> &softmaxMaxUb)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint64_t baseOffset = mSplitInfo.nBufferStartM / 2;
    size_t size = mSplitInfo.vecDealM;

    int64_t offset = 0;
    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        uint64_t actualSeqQTotal = (info.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(constInfo.batchSize - 1);
        uint64_t actualSeqQPrefixSum = (info.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(info.bIdx - 1);
        offset += info.n2Idx * actualSeqQTotal * constInfo.gSize +
                  (actualSeqQPrefixSum + info.gS1Idx / constInfo.gSize) * constInfo.gSize +
                  mSplitInfo.nBufferStartM + mSplitInfo.vecStartM;
    } else {
        offset += info.bIdx * constInfo.kvHeadNum * constInfo.qSeqSize * constInfo.gSize +
                  info.n2Idx * constInfo.qSeqSize * constInfo.gSize +
                  info.gS1Idx / constInfo.gSize * constInfo.gSize +
                  mSplitInfo.nBufferStartM + mSplitInfo.vecStartM;
    }

    if (info.actualSingleProcessSInnerSize != 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = sizeof(T) * size;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        size_t alignedSize = (sizeof(T) * size + 31) / 32 * 32 / sizeof(T);
        LocalTensor<T> tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(tmp, softmaxMaxUb[baseOffset], alignedSize);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopyPad(softmaxMaxGm[offset], tmp, dataCopyParams);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);

        tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(tmp, softmaxSumUb[baseOffset], alignedSize);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopyPad(softmaxSumGm[offset], tmp, dataCopyParams);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    } else {
        matmul::InitOutput<T>(softmaxSumGm[offset], size, ConstInfo::FLOAT_ZERO);
        matmul::InitOutput<T>(softmaxMaxGm[offset], size, SOFTMAX_MIN_NUM);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ElewiseCompute(const RunInfo &info,
                                                                const LocalTensor<T> &mmResUb,
                                                                uint32_t dealRowCount, uint32_t columnCount)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), dealRowCount * columnCount);
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        // v0的无效值判断
        uint64_t qsfaS2ValidSizeFirstPart = v0ValidSizeUb_.GetValue(128 + info.loop % MERGE_CACHE_GM_BUF_NUM);
        uint64_t qsfaS2ValidSizeSecondPart = v0ValidSizeUb_.GetValue(256 + info.loop % MERGE_CACHE_GM_BUF_NUM);

        int64_t qsfaS2ProcessSize = info.actualSingleProcessSInnerSize;
        int64_t qsfaS2Pair = CeilDiv(qsfaS2ProcessSize, 2L * constInfo.sparseBlockSize);
        int64_t qsfaS2Mid = CeilDiv(qsfaS2Pair, 2L) * 2 * constInfo.sparseBlockSize;
        if (qsfaS2Mid > qsfaS2ProcessSize) {
            qsfaS2Mid = qsfaS2ProcessSize;
        }
        if (unlikely(qsfaS2ValidSizeFirstPart < qsfaS2Mid)) {
            int64_t qsfaS2StartCeilAlign = CeilAlign(qsfaS2ValidSizeFirstPart, 8);
            int64_t qsfaS2MidFloorAlign = qsfaS2Mid / 8 * 8;
            // 场景一 s2Mid > s2ValidSizeFirstPart + oneBlk
            // 可以推导出s2StartCeilAlign < s2Mid   第一阶段取到s2StartCeilAlign
            // s2StartCeilAlign <= s2MidFloorAlign 第二阶段取到s2MidFloorAlign
            // 场景二 s2Mid <= s2ValidSizeFirstPart + oneBlk
            // 可以推导出 s2StartCeilAlign >= s2Mid 第一阶段取到mid
            // s2StartCeilAlign > s2MidFloorAlign 第二阶段取到s2StartCeilAlign
            SetInfInBlk(mmResUb, dealRowCount, columnCount, qsfaS2ValidSizeFirstPart,
                        qsfaS2StartCeilAlign >= qsfaS2Mid ? qsfaS2Mid : qsfaS2StartCeilAlign);
            SetMidInf(mmResUb, dealRowCount, columnCount, qsfaS2StartCeilAlign, qsfaS2MidFloorAlign);
            SetInfInBlk(mmResUb, dealRowCount, columnCount,
                        qsfaS2StartCeilAlign <= qsfaS2MidFloorAlign ? \
                        qsfaS2MidFloorAlign : qsfaS2StartCeilAlign, qsfaS2Mid);
        }
        if (unlikely(qsfaS2ValidSizeSecondPart < qsfaS2ProcessSize - qsfaS2Mid)) {
            // 场景一 s2Mid + s2ValidSizeSecondPart > s2ProcessSize + oneBlk
            // 可以推导出 s2StartCeilAlign < s2ProcessSize 第一阶段取到s2StartCeilAlign
            // s2StartCeilAlign <= s2EndFloorAlign 第二阶段取到s2EndFloorAlign
            // 场景二 s2Mid + s2ValidSizeSecondPart <= s2ProcessSize + oneBlk
            // 可以推导出 s2StartCeilAlign >= s2ProcessSize 第一阶段取到s2ProcessSize
            // s2StartCeilAlign > s2EndFloorAlign 第二阶段取到s2StartCeilAlign
            int64_t qsfaS2StartCeilAlign = CeilAlign(qsfaS2Mid + qsfaS2ValidSizeSecondPart, 8);
            int64_t qsfaS2EndFloorAlign = qsfaS2ProcessSize / 8 * 8;
            SetInfInBlk(mmResUb, dealRowCount, columnCount, qsfaS2Mid + qsfaS2ValidSizeSecondPart,
                        qsfaS2StartCeilAlign >= qsfaS2ProcessSize ? qsfaS2ProcessSize : qsfaS2StartCeilAlign);
            SetMidInf(mmResUb, dealRowCount, columnCount, qsfaS2StartCeilAlign, qsfaS2EndFloorAlign);
            SetInfInBlk(mmResUb, dealRowCount, columnCount,
                        qsfaS2StartCeilAlign <= qsfaS2EndFloorAlign ? qsfaS2EndFloorAlign : qsfaS2StartCeilAlign,
                        qsfaS2ProcessSize);
        }
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::SetInfInBlk(const LocalTensor<T> &mmResUb,
                                                             uint32_t dealRowCount, uint32_t columnCount,
                                                             uint64_t startId, uint64_t endId)
{
    //       startId     endId
    // x x x   0      0   0     x x x
    // 从startId到endId部分置-inf, endId、startId为endId一个blk内部的下标
    if (startId >= endId) {
        return;
    }

    uint64_t qsfaStartFloorAlignSize = startId / BLOCK_ELEMENT_NUM * BLOCK_ELEMENT_NUM;
    uint64_t qsfaNotComputePreMaskOneBlk = (1 << (startId - qsfaStartFloorAlignSize)) - 1;
    uint64_t qsfaNotComputePostMaskOneBlk = ~((1 << (endId - qsfaStartFloorAlignSize)) - 1);
    uint64_t qsfaNotComputeMaskOneBlk = qsfaNotComputePreMaskOneBlk ^ qsfaNotComputePostMaskOneBlk;

    uint64_t qsfaMaskOneBlk = ~qsfaNotComputeMaskOneBlk;
    uint64_t mask[1] = {qsfaMaskOneBlk};
    for (int i = 1; i < 8; i++) {
        mask[0] = mask[0] | (qsfaMaskOneBlk << (i * 8));
    }
    for (uint64_t qsfaRowId = 0; qsfaRowId < dealRowCount; qsfaRowId += 8) {
        Duplicate(mmResUb[qsfaRowId * columnCount + qsfaStartFloorAlignSize], SOFTMAX_MIN_NUM, mask,
                  1, CeilDiv(columnCount, 8), 0);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::SetMidInf(const LocalTensor<T> &mmResUb,
                                                           uint32_t dealRowCount, uint32_t columnCount,
                                                           uint64_t startId, uint64_t endId)
{
    if (startId >= endId) {
        return;
    }
    // startId        endId
    //    0      ...    0
    // 从startId到endId部分置-inf, startId、endId为32B对齐的下标
    for (uint64_t qsfaRowId = 0; qsfaRowId < dealRowCount; qsfaRowId++) {
        Duplicate(mmResUb[qsfaRowId * columnCount + startId], SOFTMAX_MIN_NUM, endId - startId);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::SoftmaxFlashV2Compute(
    const RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset;
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop - 1) % (constInfo.preLoadNum);
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset];
    }
    if (actualColumnCount !=0) {
        SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
        SoftMaxTiling newTiling =
            SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);
        SoftmaxFlashV2<T, true, true, false, false, QSFA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
        mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
        softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    } else {
        DataCopy(softmaxSumUb[softmaxOutOffset], inSumTensor, dealRowCount);
        PipeBarrier<PIPE_V>();
        DataCopy(softmaxMaxUb[softmaxOutOffset], inMaxTensor, dealRowCount);
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::DealBmm1ResBaseBlock(
    const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t loopId)
{
    uint32_t qsfaComputeSize = dealRowCount * columnCount;
    uint64_t qsfaInOutGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                             (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow) * columnCount;
    LocalTensor<MM1_OUT_T> qsfaMmResUb = inputBuff1.Get<MM1_OUT_T>();
    qsfaMmResUb = qsfaMmResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM1_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(qsfaMmResUb, mm1ResGm[qsfaInOutGmOffset], qsfaComputeSize);
    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        if (loopId == 0) {
            WaitFlag<HardEvent::MTE2_S>(0);
        }
    }
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    ElewiseCompute(info, qsfaMmResUb, dealRowCount, columnCount);

    PipeBarrier<PIPE_V>();
    LocalTensor<T> qsfaTmpAFloorUb = tmpBuff1.Get<T>();
    LocalTensor<uint8_t> qsfaSoftmaxTmpUb = qsfaTmpAFloorUb.template ReinterpretCast<uint8_t>();

    SoftmaxFlashV2Compute(info, mSplitInfo, qsfaMmResUb, qsfaSoftmaxTmpUb, startRow, dealRowCount, columnCount,
                            info.actualSingleProcessSInnerSize);

    PipeBarrier<PIPE_V>();
    LocalTensor<K_ROPE_T> tmpMMResCastTensor = outputBuff1.Get<K_ROPE_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);

    Cast(tmpMMResCastTensor, qsfaMmResUb, AscendC::RoundMode::CAST_ROUND, qsfaComputeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(vec1ResGm[qsfaInOutGmOffset], tmpMMResCastTensor, qsfaComputeSize);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec1SingleBuf(const RunInfo &info,
                                                                                  const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t qsfaMSplitSize = info.actualSingleProcessSInnerSize == 0 ?
        16 : (BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign);
    // 1. 向下8对齐是因为UB操作至少32B
    // 2. info.actualSingleProcessSInnerSizeAlign最大512, mSplitSize可以确保最小为16
    qsfaMSplitSize = qsfaMSplitSize >> 3U << 3U;

    if (qsfaMSplitSize > mSplitInfo.vecDealM) {
        qsfaMSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t qsfaLoopCount = (mSplitInfo.vecDealM + qsfaMSplitSize - 1) / qsfaMSplitSize;
    uint32_t qsfaTailSplitSize = mSplitInfo.vecDealM - (qsfaLoopCount - 1) * qsfaMSplitSize;

    if constexpr (TEMPLATE_MODE == V_TEMPLATE) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = 256 * sizeof(int32_t);
        dataCopyParams.dstStride = 0;
        dataCopyParams.srcStride = 0;
        DataCopyPadExtParams<int32_t> padParams;
        // 额外偏移128个元素，避免不同loop下v0和v1互相影响
        DataCopyPad(v0ValidSizeUb_[128], kvValidSizeGm_[info.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2)],
                    dataCopyParams, padParams);
        SetFlag<HardEvent::MTE2_S>(0);
        if (unlikely(qsfaLoopCount == 0)) {
            // scalar同步影响较大，挪到循环内部进行
            WaitFlag<HardEvent::MTE2_S>(0);
        }
    }
    for (uint32_t qsfaI = 0, dealSize = qsfaMSplitSize; qsfaI < qsfaLoopCount; qsfaI++) {
        if (qsfaI == (qsfaLoopCount - 1)) {
            dealSize = qsfaTailSplitSize;
        }
        DealBmm1ResBaseBlock(info, mSplitInfo, qsfaI * qsfaMSplitSize, dealSize,
            info.actualSingleProcessSInnerSizeAlign, qsfaI);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

// b s1 k
template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::MergeKv(const RunInfo &info)
{
    MergeKvOptimized(info);
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec1L(const RunInfo &info)
{
    uint32_t qsfaNBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t qsfaNBufferTail = info.actMBaseSize - (qsfaNBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t qsfaI = 0; qsfaI < qsfaNBufferLoopTimes; qsfaI++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = qsfaI;
        mSplitInfo.nBufferStartM = qsfaI * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (qsfaI + 1 != qsfaNBufferLoopTimes) ? constInfo.nBufferMBaseSize : qsfaNBufferTail;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ? mSplitInfo.nBufferDealM :
                                                                (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }

        CrossCoreWaitFlag(constInfo.syncC1V1);
        // vec1 compute
        ProcessVec1SingleBuf(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::QSFA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C2);
        // move lse for flash decode
        if (info.s2Idx == info.curSInnerLoopTimes - 1 && (constInfo.returnSoftmaxLse || info.tndIsS2SplitCore)) {
            uint32_t outIdx = info.loop % (constInfo.preLoadNum);
            auto sumTensor = softmaxSumUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET];
            auto maxTensor = softmaxMaxUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET];
            if (constInfo.returnSoftmaxLse) {
                CopyFALseToGm(info, mSplitInfo, sumTensor, maxTensor);
            }
            if (info.tndIsS2SplitCore) {
                if constexpr (FLASH_DECODE) {
                    ComputeLogSumExpAndCopyToGm(info, mSplitInfo, sumTensor, maxTensor);
                }
            }
        }
    }
}

template <typename QSFAT>
__aicore__ inline uint64_t QSFAVectorService<QSFAT>::CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx)
{
    return 0;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec2SingleBuf(const RunInfo &info,
                                                                      const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }

    uint32_t gPreSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDim;
    if (gPreSplitSize > mSplitInfo.vecDealM) {
        gPreSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mSplitInfo.vecDealM + gPreSplitSize - 1) / gPreSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * gPreSplitSize;

    for (uint32_t i = 0, dealSize = gPreSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(info, mSplitInfo, i * gPreSplitSize, dealSize, constInfo.headDim, constInfo.headDim);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::DealBmm2ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                                      uint32_t startRow, uint32_t dealRowCount,
                                                                      uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t baseOffset = startRow;
    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    bmm2ResUb.SetSize(vec2ComputeSize);

    size_t batchBase = 0;
    uint64_t inOutBaseOffset = (mSplitInfo.vecStartM + startRow) * columnCount;
    uint64_t srcGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;

    LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputBuff1.Get<MM2_OUT_T>();
    tmpBmm2ResUb = tmpBmm2ResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset + batchBase], vec2ComputeSize);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    DataCopy(bmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    // 除第一个循环外，均需要更新中间计算结果
    if (info.s2Idx > 0) {
        event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        LocalTensor<T> bmm2ResPreUb = inputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
        uint64_t vecPre2ResGmOffset = ((info.loop - 1) % constInfo.preLoadNum) * constInfo.bmm2ResUbSize +
            inOutBaseOffset;
        DataCopy(bmm2ResPreUb, vec2ResGm[vecPre2ResGmOffset + batchBase], vec2ComputeSize);
        SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
        LocalTensor<T> softmaxExpBrcb =  tmpBuff2.Get<T>();
        Brcb(softmaxExpBrcb, softmaxExpUb[(info.loop % constInfo.preLoadNum) * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset],
            (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        RowMuls(bmm2ResPreUb, bmm2ResPreUb, softmaxExpBrcb, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
        Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    }
    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
        LocalTensor<T> softmaxSumBrcb =  tmpBuff2.Get<T>();
        Brcb(softmaxSumBrcb, softmaxSumUb[(info.loop % constInfo.preLoadNum) * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset],
            (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        RowDivs(bmm2ResUb, bmm2ResUb, softmaxSumBrcb, dealRowCount, columnCount, actualColumnCount);

        PipeBarrier<PIPE_V>();
        Bmm2ResCopyOut(info, bmm2ResUb, mSplitInfo.vecStartM + startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        PipeBarrier<PIPE_V>();
        LocalTensor<T> tmpBmm2Res = outputBuff1.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        DataCopy(tmpBmm2Res, bmm2ResUb, dealRowCount * columnCount);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);

        uint64_t vecPre2ResGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
        DataCopy(vec2ResGm[vecPre2ResGmOffset + batchBase], tmpBmm2Res, vec2ComputeSize);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    }
}

template <typename QSFAT> __aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec2L(const RunInfo &info)
{
    uint32_t qsfaNBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t qsfaNBufferTail = info.actMBaseSize - (qsfaNBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t qsfaI = 0; qsfaI < qsfaNBufferLoopTimes; qsfaI++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = qsfaI;
        mSplitInfo.nBufferDealM = (qsfaI + 1 != qsfaNBufferLoopTimes) ? constInfo.nBufferMBaseSize : qsfaNBufferTail;
        mSplitInfo.nBufferStartM = qsfaI * constInfo.nBufferMBaseSize;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ? mSplitInfo.nBufferDealM :
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

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::ProcessVec2Inner(const RunInfo &info,
                                                                  const MSplitInfo &mSplitInfo,
                                                                  uint32_t mStartRow, uint32_t mDealSize)
{
    uint32_t qsfaMSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDim;
    if (qsfaMSplitSize > mDealSize) {
        qsfaMSplitSize = mDealSize;
    }

    uint32_t qsfaLoopCount = (mDealSize + qsfaMSplitSize - 1) / qsfaMSplitSize;
    uint32_t qsfaTailSplitSize = mDealSize - (qsfaLoopCount - 1) * qsfaMSplitSize;
    for (uint32_t qsfaI = 0, dealSize = qsfaMSplitSize; qsfaI < qsfaLoopCount; qsfaI++) {
        if (qsfaI == (qsfaLoopCount - 1)) {
            dealSize = qsfaTailSplitSize;
        }
        DealBmm2ResBaseBlock(info, mSplitInfo, qsfaI * qsfaMSplitSize + mStartRow, dealSize,
                             constInfo.headDim, constInfo.headDim);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::GetConfusionTransposeTiling(
    int64_t numR, int64_t numC, const uint32_t stackBufferSize, const uint32_t typeSize,
    ConfusionTransposeTiling &tiling)
{
    (void)stackBufferSize;
    uint32_t qsfaBlockSize = ONE_BLK_SIZE / typeSize;
    uint32_t qsfaHeight = numC;
    uint32_t qsfaWidth = numR;
    uint32_t qsfaHighBlock = qsfaHeight / BLOCK_CUBE;
    uint32_t qsfaStride = qsfaHeight * qsfaBlockSize * typeSize / ONE_BLK_SIZE;
    uint32_t qsfaRepeat = qsfaWidth / qsfaBlockSize;

    tiling.param0 = qsfaBlockSize;
    tiling.param1 = qsfaHeight;
    tiling.param2 = qsfaWidth;
    tiling.param3 = qsfaHighBlock;
    tiling.param4 = qsfaStride;
    tiling.param5 = qsfaRepeat;
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                            uint32_t wsMStart, uint32_t dealRowCount,
                                            uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<T> tmp = outputBuff1.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(tmp, bmm2ResUb, columnCount * dealRowCount);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    uint64_t accumTmpOutNum = CalcAccumOffset(info.bIdx, info.gS1Idx);
    uint64_t offset = accumTmpOutNum * constInfo.kvHeadNum * constInfo.mBaseSize * constInfo.headDim + // taskoffset
        info.tndCoreStartKVSplitPos * constInfo.kvHeadNum * constInfo.mBaseSize * constInfo.headDim + // 份数offset
        wsMStart * actualColumnCount; // m轴offset
    GlobalTensor<T> dst = accumOutGm[offset];
    if (info.actualSingleProcessSInnerSize == 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = actualColumnCount * sizeof(T);
        dataCopyParams.dstStride = 0;
        dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(T));
        DataCopyPad(dst, tmp, dataCopyParams);
    } else {
        matmul::InitOutput<T>(dst, dealRowCount * actualColumnCount, ConstInfo::FLOAT_ZERO);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb,
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

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::Bmm2CastAndCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                             uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
                                             uint32_t actualColumnCount)
{
    LocalTensor<OUT_T> qsfaTmpBmm2ResCastTensor = outputBuff1.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(qsfaTmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    } else {
        Cast(qsfaTmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    Bmm2DataCopyOutTrans(info, qsfaTmpBmm2ResCastTensor, wsMStart, dealRowCount, columnCount, actualColumnCount);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::Bmm2ResCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                         uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    if constexpr (!FLASH_DECODE) {
        Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
    } else {
        if (info.tndIsS2SplitCore) {
            Bmm2FDDataCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
        } else {
            Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
        }
    }
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                  uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // divs by row, 每行的元素除以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    uint32_t qsfaDtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t qsfaDLoop = actualColumnCount / qsfaDtypeMask;
    uint32_t qsfaDRemain = actualColumnCount % qsfaDtypeMask;

    BinaryRepeatParams qsfaRepeatParamsDiv;
    qsfaRepeatParamsDiv.src0BlkStride = 1;
    qsfaRepeatParamsDiv.src1BlkStride = 0;
    qsfaRepeatParamsDiv.dstBlkStride = 1;
    qsfaRepeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    qsfaRepeatParamsDiv.src1RepStride = 1;
    qsfaRepeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t qsfaColumnRepeatCount = qsfaDLoop;
    if (qsfaColumnRepeatCount <= dealRowCount) {
        uint32_t qsfaOffset = 0;
        for (uint32_t qsfaI = 0; qsfaI < qsfaDLoop; qsfaI++) {
            Div(dstUb[qsfaOffset], src0Ub[qsfaOffset], src1Ub, qsfaDtypeMask, dealRowCount, qsfaRepeatParamsDiv);
            qsfaOffset += qsfaDtypeMask;
        }
    } else {
        BinaryRepeatParams qsfaColumnRepeatParams;
        qsfaColumnRepeatParams.src0BlkStride = 1;
        qsfaColumnRepeatParams.src1BlkStride = 0;
        qsfaColumnRepeatParams.dstBlkStride = 1;
        qsfaColumnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        qsfaColumnRepeatParams.src1RepStride = 0;
        qsfaColumnRepeatParams.dstRepStride = 8;  // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        uint32_t qsfaOffset = 0;
        for (uint32_t qsfaI = 0; qsfaI < dealRowCount; qsfaI++) {
            Div(dstUb[qsfaOffset], src0Ub[qsfaOffset], src1Ub[qsfaI * FP32_BLOCK_ELEMENT_NUM], qsfaDtypeMask,
                qsfaColumnRepeatCount, qsfaColumnRepeatParams);
            qsfaOffset += columnCount;
        }
    }
    if (qsfaDRemain > 0) {
        Div(dstUb[qsfaDLoop * qsfaDtypeMask], src0Ub[qsfaDLoop * qsfaDtypeMask], src1Ub, qsfaDRemain,
            dealRowCount, qsfaRepeatParamsDiv);
    }
}

template <typename QSFAT>
__aicore__ inline void
QSFAVectorService<QSFAT>::RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                  uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // muls by row, 每行的元素乘以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount] src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    // dealRowCount is repeat times, must be less 256
    uint32_t qsfaRepeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t qsfaBlockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        qsfaRepeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        qsfaBlockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;   // 32/4 * 2 = 16
    }

    // 每次只能连续读取256B的数据进行计算，故每次只能处理256B/sizeof(dType)=
    // 列方向分dLoop次，每次处理8列数据
    uint32_t qsfaDLoop = actualColumnCount / qsfaRepeatElementNum;
    uint32_t qsfaDRemain = actualColumnCount % qsfaRepeatElementNum;
    // REPEATE_STRIDE_UP_BOUND=256， 此限制由于src0RepStride数据类型为uint8之多256个datablock间距
    if (columnCount < REPEATE_STRIDE_UP_BOUND * qsfaBlockElementNum) {
        BinaryRepeatParams qsfaRepeatParams;
        qsfaRepeatParams.src0BlkStride = 1;
        qsfaRepeatParams.src1BlkStride = 0;
        qsfaRepeatParams.dstBlkStride = 1;
        qsfaRepeatParams.src0RepStride = columnCount / qsfaBlockElementNum;
        qsfaRepeatParams.src1RepStride = 1;
        qsfaRepeatParams.dstRepStride = columnCount / qsfaBlockElementNum;

        // 如果以列为repeat所处理的次数小于行处理次数，则以列方式处理。反之则以行进行repeat处理
        if (qsfaDLoop <= dealRowCount) {
            uint32_t qsfaOffset = 0;
            for (uint32_t qsfaI = 0; qsfaI < qsfaDLoop; qsfaI++) {
                Mul(dstUb[qsfaOffset], src0Ub[qsfaOffset], src1Ub, qsfaRepeatElementNum, dealRowCount,
                    qsfaRepeatParams);
                qsfaOffset += qsfaRepeatElementNum;
            }
        } else {
            BinaryRepeatParams qsfaColumnRepeatParams;
            qsfaColumnRepeatParams.src0BlkStride = 1;
            qsfaColumnRepeatParams.src1BlkStride = 0;
            qsfaColumnRepeatParams.dstBlkStride = 1;
            qsfaColumnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            qsfaColumnRepeatParams.src1RepStride = 0;
            qsfaColumnRepeatParams.dstRepStride = 8;  // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            for (uint32_t qsfaI = 0; qsfaI < dealRowCount; qsfaI++) {
                Mul(dstUb[qsfaI * columnCount], src0Ub[qsfaI * columnCount], src1Ub[qsfaI * qsfaBlockElementNum],
                    qsfaRepeatElementNum, qsfaDLoop, qsfaColumnRepeatParams);
            }
        }

        // 最后一次完成[dealRowCount, dRemain] * [dealRowCount, blockElementNum] 只计算有效部分
        if (qsfaDRemain > 0) {
            Mul(dstUb[qsfaDLoop * qsfaRepeatElementNum], src0Ub[qsfaDLoop * qsfaRepeatElementNum], src1Ub,
                qsfaDRemain, dealRowCount, qsfaRepeatParams);
        }
    } else {
        BinaryRepeatParams qsfaRepeatParams;
        qsfaRepeatParams.src0RepStride = 8; // 每个repeat为256B数据，正好8个datablock
        qsfaRepeatParams.src0BlkStride = 1;
        qsfaRepeatParams.src1RepStride = 0;
        qsfaRepeatParams.src1BlkStride = 0;
        qsfaRepeatParams.dstRepStride = 8;
        qsfaRepeatParams.dstBlkStride = 1;
        // 每次计算一行，共计算dealRowCount行
        for (uint32_t qsfaI = 0; qsfaI < dealRowCount; qsfaI++) {
            // 计算一行中的dLoop个repeat, 每个repeat计算256/block_size 个data_block
            Mul(dstUb[qsfaI * columnCount], src0Ub[qsfaI * columnCount], src1Ub[qsfaI * qsfaBlockElementNum],
                qsfaRepeatElementNum, qsfaDLoop, qsfaRepeatParams);
            //  计算一行中的尾块
            if (qsfaDRemain > 0) {
                Mul(dstUb[qsfaI * columnCount + qsfaDLoop * qsfaRepeatElementNum],
                    src0Ub[qsfaI * columnCount + qsfaDLoop * qsfaRepeatElementNum],
                    src1Ub[qsfaI * qsfaBlockElementNum], qsfaDRemain, 1, qsfaRepeatParams);
            }
        }
    }
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::DecodeTopkPairOptimized(int32_t idx0, int32_t idx1, int64_t posInBlock,
    int64_t& realS2Idx0, int64_t& realS2Idx1)
{
    if (sparseBlockSize_ == 1 && posInBlock == 0) {
        realS2Idx0 = idx0;
        realS2Idx1 = idx1;
        return;
    }
    realS2Idx0 = static_cast<int64_t>(idx0) * sparseBlockSize_ + posInBlock;
    realS2Idx1 = static_cast<int64_t>(idx1) * sparseBlockSize_ + posInBlock;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::GetRealS2IdxOptimized(int64_t topkIdx, int64_t posInBlock, int64_t&
    realS2Idx0, int64_t& realS2Idx1, int64_t topkGmBaseOffset)
{
    if (unlikely(topkIdx >= sparseBlockCount_)) {
        realS2Idx0 = -1;
        realS2Idx1 = -1;
        return;
    }
    int64_t elemOffset = topkGmBaseOffset + topkIdx;
    // 相邻 sparse block 的 topk 下标连续，8 字节对齐且第二个下标不越界时一次取 int64
    if ((topkIdx + 1 < sparseBlockCount_) && ((elemOffset & 1) == 0)) {
        int64_t packed = topkI64Gm_.GetValue(static_cast<uint64_t>(elemOffset >> 1));
        int32_t idx0 = static_cast<int32_t>(packed);
        int32_t idx1 = static_cast<int32_t>(static_cast<uint64_t>(packed) >> 32);
        DecodeTopkPairOptimized(idx0, idx1, posInBlock, realS2Idx0, realS2Idx1);
        return;
    }
    int32_t idx0 = topkGm_.GetValue(static_cast<uint64_t>(elemOffset));
    if (unlikely(topkIdx + 1 >= sparseBlockCount_)) {
        realS2Idx0 = (sparseBlockSize_ == 1 && posInBlock == 0) ?
                         idx0 :
                         static_cast<int64_t>(idx0) * sparseBlockSize_ + posInBlock;
        realS2Idx1 = -1;
        return;
    }
    int32_t idx1 = topkGm_.GetValue(static_cast<uint64_t>(elemOffset + 1));
    DecodeTopkPairOptimized(idx0, idx1, posInBlock, realS2Idx0, realS2Idx1);
}

template <typename QSFAT>
__aicore__ inline int64_t QSFAVectorService<QSFAT>::GetKeyBNBOffsetOptimized(int64_t realS2Idx, int64_t blkTableBase,
    int64_t s2IdLimit, const RunInfo& kvVecRunInfo)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t realKeyBNBOffset = 0;
    if constexpr (PAGE_ATTENTION) {
        int64_t blkTableIdx;
        int64_t blkTableOffset;
        if (kvCacheBlockShift_ >= 0) {
            blkTableIdx = realS2Idx >> kvCacheBlockShift_;
            blkTableOffset = realS2Idx & kvCacheBlockMask_;
        } else {
            blkTableIdx = realS2Idx / constInfo.kvCacheBlockSize;
            blkTableOffset = realS2Idx % constInfo.kvCacheBlockSize;
        }
        // kvHeadNum 编译期为 1，页内偏移不再乘 head
        realKeyBNBOffset =
            blkTableGm_.GetValue(blkTableBase + blkTableIdx) * static_cast<int64_t>(constInfo.kvCacheBlockSize) +
            blkTableOffset;
    } else {
        realKeyBNBOffset =
            (kvVecRunInfo.tensorBOffset + realS2Idx * constInfo.kvHeadNum * constInfo.combineHeadDim) /
            constInfo.combineHeadDim;
    }
    return realKeyBNBOffset;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::CopyInSingleKvOptimized(int64_t& mte2Size, int64_t mte3Size, int64_t
    mergeMte3Idx, int64_t realS2Idx, int64_t keyBNBOffset, int64_t s2IdLimit, const RunInfo& kvVecRunInfo)
{
    if (keyBNBOffset < 0) {
        return;
    }
    int64_t validS2Count = ((realS2Idx + constInfo.sparseBlockSize > s2IdLimit) ? (s2IdLimit - realS2Idx) :
                                                                                       constInfo.sparseBlockSize);
    DataCopyExtParams intriParams;

    intriParams.blockCount = validS2Count;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    // 当前仅支持COMBINE模式
    if (constInfo.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::COMBINE) {
        intriParams.blockLen = combineBytes_;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = combineDimAlign_ - combineDim_;
        padParams.paddingValue = 0;
        DataCopyPad(
            kvMergUb_[(mergeMte3Idx % MTE2_SLOT_NUM) * mte2SlotElems_ + (mte2Size - mte3Size) * combineDimAlign_],
            keyGm_[keyBNBOffset * combineDim_], intriParams, padParams);
    }
    mte2Size += validS2Count;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::CopyInKvOptimized(int64_t& mte2Size, int64_t mte3Size, int64_t
    mergeMte3Idx, int64_t realS2Idx1, int64_t realS2Idx2, int64_t s2IdLimit, int64_t blkTableBase, const RunInfo&
    kvVecRunInfo)
{
    int64_t keyBNBOffset1 = GetKeyBNBOffsetOptimized(realS2Idx1, blkTableBase, s2IdLimit, kvVecRunInfo);
    int64_t keyBNBOffset2 = GetKeyBNBOffsetOptimized(realS2Idx2, blkTableBase, s2IdLimit, kvVecRunInfo);
    if (unlikely(keyBNBOffset1 < 0 && keyBNBOffset2 < 0)) {
        return;
    }

    int64_t sparseBlockSrcStride =
        ((keyBNBOffset1 > keyBNBOffset2 ? (keyBNBOffset1 - keyBNBOffset2) : (keyBNBOffset2 - keyBNBOffset1)) -
         sparseBlockSize_);
    int64_t keySrcStride = sparseBlockSrcStride * static_cast<int64_t>(combineBytes_);
    if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0 ||
                 realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
                 realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit) ||
        constInfo.sparseBlockSize > 1) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        CopyInSingleKvOptimized(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyBNBOffset1, s2IdLimit, kvVecRunInfo);
        CopyInSingleKvOptimized(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyBNBOffset2, s2IdLimit, kvVecRunInfo);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockCount = (keyBNBOffset1 >= 0) + (keyBNBOffset2 >= 0);
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t startGmOffset = keyBNBOffset1 > -1 ? keyBNBOffset1 : keyBNBOffset2;
        if (keyBNBOffset2 > -1 && keyBNBOffset2 < keyBNBOffset1) {
            startGmOffset = keyBNBOffset2;
        }

        // 当前仅支持COMBINE模式
        if (constInfo.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::COMBINE) {
            intriParams.blockLen = static_cast<uint32_t>(sparseBlockSize_) * combineBytes_;
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = combineDimAlign_ - combineDim_;
            padParams.paddingValue = 0;
            DataCopyPad(
                kvMergUb_[(mergeMte3Idx % MTE2_SLOT_NUM) * mte2SlotElems_ + (mte2Size - mte3Size) * combineDimAlign_],
                keyGm_[startGmOffset * combineDim_], intriParams, padParams);
        }
        mte2Size += ((keyBNBOffset1 > -1) + (keyBNBOffset2 > -1)) * constInfo.sparseBlockSize;
    }
}

template <typename QSFAT>
__aicore__ inline bool QSFAVectorService<QSFAT>::PrepareAddressBatch( int64_t topkBase, int32_t count, int64_t
    s2IdLimit, const RunInfo& info, bool wholeQuery)
{
    queryAddressesValid_ = false;
    queryAddressCount_ = 0;
    // Short tails keep the existing scalar path; all temporary storage below
    // is bounded by the current AIV's S2 segment, not by the table width or K.
    if (count < 128 || count > ADDRESS_RESULT_CAPACITY ||
        ((count + 127) / 128 * 128) * 7 * sizeof(int32_t) > ConstInfo::BUFFER_SIZE_BYTE_32K * 2) {
        return false;
    }
    PipeBarrier<PIPE_ALL>();
    const int32_t span = ((count + 127) / 128) * 128;
    int32_t full = count / 128 * 128;
    constexpr int32_t PAGE_ID_SLOT_ELEMS = 8; // One 32-byte DMA destination per page ID.
    LocalTensor<int32_t> work = inputBuff1.Get<int32_t>();
    LocalTensor<int32_t> indices = work;
    LocalTensor<int32_t> page = work[span];
    LocalTensor<int32_t> offset = work[span * 2];
    LocalTensor<float> values = work[span * 3].template ReinterpretCast<float>();
    LocalTensor<float> reduceWork = work[span * 4].template ReinterpretCast<float>();
    LocalTensor<float> bounds = work[span * 5].template ReinterpretCast<float>();
    LocalTensor<int32_t> dmaOffsets = work[span * 6];
    LocalTensor<int32_t> addresses = v0ValidSizeUb_[512];
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = count * sizeof(int32_t);
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<int32_t> padding{false, 0, 0, 0};
    if (wholeQuery) {
        const int32_t blocks = info.actS2Size / 512;
        const int32_t tail = info.actS2Size % 512;
        if (blocks > 0) {
            params.blockCount = blocks;
            params.blockLen = 256 * sizeof(int32_t);
            params.srcStride = 256 * sizeof(int32_t);
            DataCopyPad(indices, topkGm_[topkBase + GetSubBlockIdx() * 256], params, padding);
        }
        if (tail > 0) {
            params.blockCount = 1;
            params.blockLen = tail * sizeof(int32_t);
            params.srcStride = 0;
            DataCopyPad(indices[blocks * 256], topkGm_[topkBase + blocks * 512], params, padding);
        }
    } else {
        DataCopyPad(indices, topkGm_[topkBase], params, padding);
    }
    params.blockCount = 1;
    params.srcStride = 0;
    SetFlag<HardEvent::MTE2_S>(6);
    WaitFlag<HardEvent::MTE2_S>(6);
    const int64_t tableWidth = constInfo.maxBlockNumPerBatch;
    const int64_t indexLimit = s2IdLimit < tableWidth * 128L ? s2IdLimit : tableWidth * 128L;
    int32_t minIndex = INT32_MAX;
    int32_t maxIndex = 0;
    // FP32 exactly represents every index in this range. Use scalar bounds
    // beyond it rather than accepting rounded, potentially invalid indices.
    if (indexLimit <= (1L << 24)) {
        SetFlag<HardEvent::MTE2_V>(6);
        WaitFlag<HardEvent::MTE2_V>(6);
        Cast(values, indices, RoundMode::CAST_NONE, full);
        PipeBarrier<PIPE_V>();
        ReduceMin(bounds, values, reduceWork, full, false);
        PipeBarrier<PIPE_V>();
        ReduceMax(bounds[8], values, reduceWork, full, false);
        SetFlag<HardEvent::V_S>(6);
        WaitFlag<HardEvent::V_S>(6);
        minIndex = static_cast<int32_t>(bounds.GetValue(0));
        maxIndex = static_cast<int32_t>(bounds.GetValue(8));
        for (int32_t j = full; j < count; ++j) {
            const int32_t index = indices.GetValue(j);
            minIndex = index < minIndex ? index : minIndex;
            maxIndex = index > maxIndex ? index : maxIndex;
        }
    } else {
        for (int32_t j = 0; j < count; ++j) {
            const int32_t index = indices.GetValue(j);
            minIndex = index < minIndex ? index : minIndex;
            maxIndex = index > maxIndex ? index : maxIndex;
        }
    }
    if (minIndex < 0) {
        // The operator contract places invalid indices after the valid prefix.
        // Search the AIV's ordered subsequence; do not translate any sentinel.
        int32_t lo = 0;
        int32_t hi = count;
        while (lo < hi) {
            const int32_t mid = lo + (hi - lo) / 2;
            if (indices.GetValue(mid) >= 0) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        count = lo;
        if (count < 128) {
            return false;
        }
        full = count / 128 * 128;
        minIndex = INT32_MAX;
        maxIndex = 0;
        if (indexLimit <= (1L << 24)) {
            // values already contains the original full-vector conversion.
            ReduceMin(bounds, values, reduceWork, full, false);
            PipeBarrier<PIPE_V>();
            ReduceMax(bounds[8], values, reduceWork, full, false);
            SetFlag<HardEvent::V_S>(6);
            WaitFlag<HardEvent::V_S>(6);
            minIndex = static_cast<int32_t>(bounds.GetValue(0));
            maxIndex = static_cast<int32_t>(bounds.GetValue(8));
            for (int32_t j = full; j < count; ++j) {
                const int32_t index = indices.GetValue(j);
                minIndex = index < minIndex ? index : minIndex;
                maxIndex = index > maxIndex ? index : maxIndex;
            }
        } else {
            for (int32_t j = 0; j < count; ++j) {
                const int32_t index = indices.GetValue(j);
                minIndex = index < minIndex ? index : minIndex;
                maxIndex = index > maxIndex ? index : maxIndex;
            }
        }
    }
    if (minIndex < 0 || maxIndex >= indexLimit) {
        return false;
    }
    const int32_t minPage = minIndex >> 7;
    const int32_t maxPage = maxIndex >> 7;
    const int64_t effectivePages = (indexLimit + 127) / 128;
    const int64_t batchBase = info.bIdx * tableWidth;
    const bool cachedWindow = largeTableBatch_ == static_cast<int32_t>(info.bIdx) &&
        minPage >= largeTableBegin_ && maxPage < largeTableBegin_ + largeTableCount_;
    const bool useWindow = effectivePages <= ADDRESS_TABLE_CACHE_PAGES || cachedWindow ||
        maxPage - minPage + 1 <= ADDRESS_TABLE_CACHE_PAGES;
    const int32_t windowBegin = cachedWindow ? largeTableBegin_ :
        (effectivePages <= ADDRESS_TABLE_CACHE_PAGES ? 0 : minPage);
    const int32_t rangePages = maxPage - minPage + 1;
    // Prefer contiguous table reads while their size fits both the temporary
    // buffer and the aligned storage required by per-index page-ID reads.
    const bool gatherPageIds = !useWindow &&
        (rangePages > ADDRESS_TABLE_MAX_PAGES || rangePages > count * 8);
    const int32_t tableBegin = useWindow ? windowBegin : minPage;
    LocalTensor<int32_t> table = useWindow ? addressTableBuff_.Get<int32_t>() : inputBuff2.Get<int32_t>();
    if (gatherPageIds) {
        // Issue independent page-ID reads through MTE2, then wait once. This
        // avoids a table-size ceiling and serial GM GetValue page lookups.
        // Bound sparse staging independently of whole-query K. A chunk uses
        // at most 256 * 32 bytes, including destination alignment padding.
        for (int32_t base = 0; base < count; base += 256) {
            const int32_t chunk = count - base < 256 ? count - base : 256;
            params.blockLen = sizeof(int32_t);
            for (int32_t j = 0; j < chunk; ++j) {
                const int32_t logicalPage = indices.GetValue(base + j) >> 7;
                DataCopyPad(table[j * PAGE_ID_SLOT_ELEMS], blkTableGm_[batchBase + logicalPage], params, padding);
                dmaOffsets.SetValue(j, j * PAGE_ID_SLOT_ELEMS * sizeof(int32_t));
            }
            SetFlag<HardEvent::MTE2_V>(6);
            WaitFlag<HardEvent::MTE2_V>(6);
            SetFlag<HardEvent::S_V>(6);
            WaitFlag<HardEvent::S_V>(6);
            const int32_t vectorCount = chunk / 128 * 128;
            if (vectorCount > 0) {
                Gather(addresses[base], table, dmaOffsets.template ReinterpretCast<uint32_t>(), 0, vectorCount);
            }
            SetFlag<HardEvent::V_S>(6);
            WaitFlag<HardEvent::V_S>(6);
            SetFlag<HardEvent::MTE2_S>(6);
            WaitFlag<HardEvent::MTE2_S>(6);
            for (int32_t j = vectorCount; j < chunk; ++j) {
                addresses.SetValue(base + j, table.GetValue(j * PAGE_ID_SLOT_ELEMS));
            }
            SetFlag<HardEvent::V_MTE2>(6);
            WaitFlag<HardEvent::V_MTE2>(6);
        }
    } else if (!useWindow || !cachedWindow) {
        const int32_t tableCount = useWindow ?
            static_cast<int32_t>(effectivePages - tableBegin < ADDRESS_TABLE_SEGMENT_PAGES ?
                                 effectivePages - tableBegin : ADDRESS_TABLE_SEGMENT_PAGES) : maxPage - minPage + 1;
        for (int32_t start = 0; start < tableCount; start += ADDRESS_TABLE_SEGMENT_PAGES) {
            const int32_t size = tableCount - start < ADDRESS_TABLE_SEGMENT_PAGES ?
                                 tableCount - start : ADDRESS_TABLE_SEGMENT_PAGES;
            params.blockLen = size * sizeof(int32_t);
            DataCopyPad(table[start], blkTableGm_[batchBase + tableBegin + start], params, padding);
        }
        if (useWindow) {
            largeTableBatch_ = info.bIdx;
            largeTableBegin_ = tableBegin;
            largeTableCount_ = tableCount;
        }
    }
    SetFlag<HardEvent::MTE2_V>(6);
    WaitFlag<HardEvent::MTE2_V>(6);
    ShiftRight(page, indices, static_cast<int32_t>(7), full);
    PipeBarrier<PIPE_V>();
    ShiftLeft(offset, page, static_cast<int32_t>(7), full);
    PipeBarrier<PIPE_V>();
    Sub(offset, indices, offset, full);
    if (!gatherPageIds) {
        Adds(page, page, -tableBegin, full);
        PipeBarrier<PIPE_V>();
        ShiftLeft(page, page, static_cast<int32_t>(2), full);
        PipeBarrier<PIPE_V>();
        Gather(addresses, table, page.template ReinterpretCast<uint32_t>(), 0, full);
    }
    PipeBarrier<PIPE_V>();
    Cast(values, addresses, RoundMode::CAST_NONE, full);
    PipeBarrier<PIPE_V>();
    ReduceMin(bounds, values, reduceWork, full, false);
    PipeBarrier<PIPE_V>();
    ReduceMax(bounds[8], values, reduceWork, full, false);
    SetFlag<HardEvent::V_S>(6);
    WaitFlag<HardEvent::V_S>(6);
    if (bounds.GetValue(0) < 0 || bounds.GetValue(8) > (INT32_MAX - 127) / 128) {
        return false;
    }
    ShiftLeft(addresses, addresses, static_cast<int32_t>(7), full);
    PipeBarrier<PIPE_V>();
    Add(addresses, addresses, offset, full);
    SetFlag<HardEvent::V_S>(6);
    WaitFlag<HardEvent::V_S>(6);
    for (int32_t j = full; j < count; ++j) {
        const int32_t index = indices.GetValue(j);
        const int32_t physicalPage = gatherPageIds ? addresses.GetValue(j) : table.GetValue((index >> 7) - tableBegin);
        if (physicalPage < 0 || physicalPage > (INT32_MAX - 127) / 128) {
            return false;
        }
        addresses.SetValue(j, physicalPage * 128 + (index & 127));
    }
    queryAddressCount_ = count;
    queryAddressesValid_ = true;
    return true;
}

template <typename QSFAT>
__aicore__ inline bool QSFAVectorService<QSFAT>::PrepareQueryAddressCache( int64_t topkBase, int64_t s2IdLimit, const
    RunInfo& info)
{
    const int64_t valid = info.actS2Size;
    const int64_t count = valid / 512 * 256 + valid % 512;
    if (valid <= 0 || valid > sparseBlockCount_ || count > ADDRESS_RESULT_CAPACITY) {
        queryAddressesValid_ = false;
        return false;
    }
    return PrepareAddressBatch(topkBase, static_cast<int32_t>(count), s2IdLimit, info, true);
}

template <typename QSFAT>
__aicore__ inline bool QSFAVectorService<QSFAT>::PrepareQueryAddresses(int64_t topkBase, int64_t s2IdLimit, const
    RunInfo& info)
{
    if constexpr (QSFAT::adaptiveAddresses) {
        return PrepareQueryAddressCache(topkBase, s2IdLimit, info);
    }
    // V0-only scratch: previous V1/V2 calls have returned. Preserve the
    // address cache above v0ValidSizeUb_[512] across subsequent S2 blocks.
    PipeBarrier<PIPE_ALL>();
    queryAddressesValid_ = false;
    int32_t valid = info.actS2Size;
    if (valid <= 0 || valid > sparseBlockCount_) {
        return false;
    }
    int32_t blocks = valid / 512;
    int32_t tail = valid % 512;
    int32_t count = blocks * 256 + tail;
    int32_t span = ((count + 127) / 128) * 128;
    LocalTensor<int32_t> work = inputBuff1.Get<int32_t>();
    LocalTensor<int32_t> indices = work;
    LocalTensor<int32_t> page = work[span];
    LocalTensor<int32_t> offset = work[span * 2];
    LocalTensor<int32_t> table = addressTableBuff_.Get<int32_t>();
    LocalTensor<int32_t> addresses = v0ValidSizeUb_[512];
    DataCopyExtParams params;
    params.blockCount = 1;
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<int32_t> padding{false, 0, 0, 0};
    if (addressTableBatch_ != static_cast<int32_t>(info.bIdx)) {
        params.blockLen = constInfo.maxBlockNumPerBatch * sizeof(int32_t);
        DataCopyPad(table, blkTableGm_[info.bIdx * constInfo.maxBlockNumPerBatch], params, padding);
        SetFlag<HardEvent::MTE2_S>(6);
        WaitFlag<HardEvent::MTE2_S>(6);
        addressTableValid_ = true;
        for (int32_t i = 0; i < constInfo.maxBlockNumPerBatch; ++i) {
            int32_t physicalPage = table.GetValue(i);
            // Keep the cached physical token offset within signed int32.
            if (physicalPage < 0 || physicalPage > (INT32_MAX - 127) / 128) {
                addressTableValid_ = false;
            }
        }
        addressTableBatch_ = info.bIdx;
    }
    if (!addressTableValid_) {
        return false;
    }
    if (blocks > 0) {
        params.blockCount = blocks;
        params.blockLen = 256 * sizeof(int32_t);
        params.srcStride = 256 * sizeof(int32_t);
        DataCopyPad(indices, topkGm_[topkBase + GetSubBlockIdx() * 256], params, padding);
    }
    if (tail >= 128) {
        params.blockCount = 1;
        params.blockLen = tail * sizeof(int32_t);
        params.srcStride = 0;
        DataCopyPad(indices[blocks * 256], topkGm_[topkBase + blocks * 512], params, padding);
    }
    SetFlag<HardEvent::MTE2_V>(6);
    WaitFlag<HardEvent::MTE2_V>(6);
    int32_t full = (count / 128) * 128;
    if (full > 0) {
        // Validate before Gather: negative/sentinel and causal-filtered
        // indices use the existing compaction/termination path instead.
        LocalTensor<float> values = work[span * 3].template ReinterpretCast<float>();
        LocalTensor<float> reduceWork = work[span * 4].template ReinterpretCast<float>();
        LocalTensor<float> bounds = work[span * 5].template ReinterpretCast<float>();
        Cast(values, indices, RoundMode::CAST_NONE, full);
        PipeBarrier<PIPE_V>();
        ReduceMin(bounds, values, reduceWork, full, false);
        PipeBarrier<PIPE_V>();
        ReduceMax(bounds[8], values, reduceWork, full, false);
        SetFlag<HardEvent::V_S>(6);
        WaitFlag<HardEvent::V_S>(6);
        int64_t indexLimit = s2IdLimit < constInfo.maxBlockNumPerBatch * 128L ?
                                 s2IdLimit :
                                 constInfo.maxBlockNumPerBatch * 128L;
        if (bounds.GetValue(0) < 0 || bounds.GetValue(8) >= indexLimit) {
            return false;
        }
        ShiftRight(page, indices, static_cast<int32_t>(7), full);
        PipeBarrier<PIPE_V>();
        ShiftLeft(offset, page, static_cast<int32_t>(7), full);
        PipeBarrier<PIPE_V>();
        Sub(offset, indices, offset, full);
        ShiftLeft(page, page, static_cast<int32_t>(2), full);
        PipeBarrier<PIPE_V>();
        Gather(addresses, table, page.template ReinterpretCast<uint32_t>(), 0, full);
        PipeBarrier<PIPE_V>();
        ShiftLeft(addresses, addresses, static_cast<int32_t>(7), full);
        PipeBarrier<PIPE_V>();
        Add(addresses, addresses, offset, full);
    }
    SetFlag<HardEvent::V_S>(6);
    WaitFlag<HardEvent::V_S>(6);
    for (int32_t j = full; j < count; ++j) {
        int32_t idx = topkGm_.GetValue(topkBase + blocks * 512 + (j - blocks * 256));
        if (idx < 0 || idx >= s2IdLimit || (idx >> 7) >= constInfo.maxBlockNumPerBatch) {
            return false;
        }
        int32_t physicalPage = table.GetValue(idx >> 7);
        addresses.SetValue(j, physicalPage * 128 + (idx & 127));
    }
    queryAddressesValid_ = true;
    return true;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::MergeKvOptimized(const RunInfo& kvVecRunInfo)
{
    int64_t s2ProcessSize = kvVecRunInfo.actualSingleProcessSInnerSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2L * constInfo.sparseBlockSize);
    int64_t topkGmBaseOffset = 0;

    if constexpr (LAYOUT_T == QSFA_LAYOUT::TND) {
        uint64_t qsfaActualSeqQPrefixSum =
            (kvVecRunInfo.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(kvVecRunInfo.bIdx - 1);
        topkGmBaseOffset += (qsfaActualSeqQPrefixSum + kvVecRunInfo.gS1Idx / constInfo.gSize) *
                                constInfo.kvHeadNum * constInfo.sparseBlockCount +
                            kvVecRunInfo.n2Idx * constInfo.sparseBlockCount;
    } else {
        topkGmBaseOffset += kvVecRunInfo.bIdx * constInfo.qSeqSize * constInfo.sparseBlockCount +
                            kvVecRunInfo.gS1Idx / constInfo.gSize * constInfo.sparseBlockCount;
    }
    int64_t qsfaMergeMte3Idx = 0;
    int64_t qsfaMte2Size = 0;
    int64_t qsfaMte3Size = 0;
    int64_t qsfaS2IdxArray0 = -1;
    int64_t qsfaS2IdxArray1 = -1;
    bool qsfaNeedWaitMte3ToMte2 = true;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(2);
    int64_t qsfaS2GmStartOffset = GetSubBlockIdx() == 0 ? 0 : CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
    int64_t qsfaS2GmLimit =
        GetSubBlockIdx() == 0 ? CeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize : s2ProcessSize;
    if (qsfaS2GmLimit > s2ProcessSize) {
        qsfaS2GmLimit = s2ProcessSize;
    }
    int64_t s2IdLimit = kvVecRunInfo.curActualSeqLenOri;
    if (constInfo.sparseMode == 3) {
        s2IdLimit =
            kvVecRunInfo.curActualSeqLenOri - kvVecRunInfo.actS1Size + kvVecRunInfo.gS1Idx / constInfo.gSize + 1;
    }
    int64_t blkTableBase = kvVecRunInfo.bIdx * constInfo.maxBlockNumPerBatch;
    int64_t s2Step = sparseBlockSize_ << 1;
    int64_t globalPos = qsfaS2GmStartOffset + kvVecRunInfo.s2Idx * s2BaseSize_;
    int64_t topkIdx = globalPos / sparseBlockSize_;
    int64_t posInBlock = globalPos - topkIdx * sparseBlockSize_;
    int64_t lastTopkIdx = topkIdx;
    if (qsfaS2GmStartOffset < qsfaS2GmLimit) {
        lastTopkIdx = topkIdx + 2 * ((qsfaS2GmLimit - 1 - qsfaS2GmStartOffset) / s2Step);
    }
    bool useI64All = (qsfaS2GmStartOffset < qsfaS2GmLimit) && ((topkGmBaseOffset + topkIdx) & 1) == 0 &&
                     (lastTopkIdx + 1 < sparseBlockCount_);
    bool useQueryAddresses = false;
    bool useAddressSegments = false;
    if constexpr (QSFAT::noRope && PAGE_ATTENTION && TEMPLATE_MODE == V_TEMPLATE) {
        // The cache occupies the unused upper 6KB of v0ValidSizeBuff.
        // Capacity is ADDRESS_RESULT_CAPACITY INT32 row addresses per AIV.
        if (sparseBlockSize_ == 1 && kvCacheBlockShift_ == 7 && s2BaseSize_ == 512 && sparseBlockCount_ > 0 &&
            !kvVecRunInfo.tndIsS2SplitCore && constInfo.maxBlockNumPerBatch > 0 &&
            (QSFAT::adaptiveAddresses || (sparseBlockCount_ <= 2560 && constInfo.maxBlockNumPerBatch <= 256))) {
            if constexpr (QSFAT::adaptiveAddresses) {
                const int64_t validK = kvVecRunInfo.actS2Size;
                const int64_t addressCount = validK / 512 * 256 + validK % 512;
                const int64_t alignedCount = (addressCount + 127) / 128 * 128;
                useAddressSegments = addressCount > ADDRESS_RESULT_CAPACITY ||
                    alignedCount * 7 * sizeof(int32_t) > ConstInfo::BUFFER_SIZE_BYTE_32K * 2;
            }
            if (useAddressSegments) {
                PrepareAddressBatch(topkGmBaseOffset + topkIdx,
                    static_cast<int32_t>(qsfaS2GmLimit - qsfaS2GmStartOffset), s2IdLimit, kvVecRunInfo);
            } else if (kvVecRunInfo.s2Idx == 0) {
                PrepareQueryAddresses(topkGmBaseOffset, s2IdLimit, kvVecRunInfo);
            }
            useQueryAddresses = queryAddressesValid_;
        }
        if (useQueryAddresses) {
            int32_t totalRows = static_cast<int32_t>(qsfaS2GmLimit - qsfaS2GmStartOffset);
            int32_t queryCacheBase = useAddressSegments ? 0 : kvVecRunInfo.s2Idx * 256;
            if (!useAddressSegments && kvVecRunInfo.s2Idx == kvVecRunInfo.actS2Size / 512) {
                queryCacheBase += qsfaS2GmStartOffset;
            }
            if constexpr (QSFAT::adaptiveAddresses) {
                // Keep the original S2 limit for invalid-row zero fill below;
                // only gather/dequantize the valid prefix in this segment.
                const int32_t remaining = queryAddressCount_ - queryCacheBase;
                totalRows = remaining < totalRows ? (remaining > 0 ? remaining : 0) : totalRows;
            }
            DataCopyExtParams kvParams;
            kvParams.blockCount = 2;
            kvParams.blockLen = 528;
            kvParams.srcStride = 0;
            kvParams.dstStride = 0;
            DataCopyPadExtParams<KV_T> kvPadding{true, 0, 16, 0};
            for (int32_t row = 0; row < totalRows; row += 32) {
                int32_t rows = totalRows - row < 32 ? totalRows - row : 32;
                WaitFlag<HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx % MTE2_SLOT_NUM);
                int64_t ubBase = (qsfaMergeMte3Idx % MTE2_SLOT_NUM) * mte2SlotElems_;
                for (int32_t j = 0; j < rows; j += 2) {
                    int64_t p0, p1;
                    p0 = v0ValidSizeUb_.GetValue(512 + queryCacheBase + row + j);
                    p1 = j + 1 < rows ? v0ValidSizeUb_.GetValue(512 + queryCacheBase + row + j + 1) : -1;
                    int64_t lo = p0 < p1 ? p0 : p1;
                    int64_t gap = p0 > p1 ? p0 - p1 : p1 - p0;
                    int64_t stride = (gap - 1) * 528;
                    if (j + 1 < rows && p0 >= 0 && p1 >= 0 && stride >= 0 && stride < INT32_MAX) {
                        kvParams.blockCount = 2;
                        kvParams.srcStride = stride;
                        DataCopyPad(kvMergUb_[ubBase + j * 544], keyGm_[lo * 528], kvParams, kvPadding);
                    } else {
                        kvParams.blockCount = 1;
                        kvParams.srcStride = 0;
                        if (p0 >= 0) {
                            DataCopyPad(kvMergUb_[ubBase + j * 544], keyGm_[p0 * 528], kvParams, kvPadding);
                        }
                        if (j + 1 < rows && p1 >= 0) {
                            DataCopyPad(kvMergUb_[ubBase + (j + 1) * 544], keyGm_[p1 * 528], kvParams, kvPadding);
                        }
                    }
                }
                qsfaMte2Size += rows;
                CopyOutMrgeResultOptimized(qsfaMte2Size, qsfaMte3Size, qsfaS2GmStartOffset, qsfaMergeMte3Idx,
                                           kvVecRunInfo);
                qsfaMte3Size = qsfaMte2Size;
                SetFlag<HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx % MTE2_SLOT_NUM);
                ++qsfaMergeMte3Idx;
            }
        }
    }
    if (!useQueryAddresses) {
        for (int64_t s2GmOffsetArray = qsfaS2GmStartOffset; s2GmOffsetArray < qsfaS2GmLimit;
             s2GmOffsetArray += s2Step) {
            if (qsfaNeedWaitMte3ToMte2) {
                WaitFlag<AscendC::HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx % MTE2_SLOT_NUM);
                qsfaNeedWaitMte3ToMte2 = false;
            }
            if (useI64All) {
                int64_t packed = topkI64Gm_.GetValue(static_cast<uint64_t>((topkGmBaseOffset + topkIdx) >> 1));
                DecodeTopkPairOptimized(static_cast<int32_t>(packed),
                                        static_cast<int32_t>(static_cast<uint64_t>(packed) >> 32), posInBlock,
                                        qsfaS2IdxArray0, qsfaS2IdxArray1);
            } else {
                GetRealS2IdxOptimized(topkIdx, posInBlock, qsfaS2IdxArray0, qsfaS2IdxArray1, topkGmBaseOffset);
            }
            topkIdx += 2;
            if (unlikely(qsfaS2IdxArray0 < 0)) {
                CopyOutMrgeResultOptimized(qsfaMte2Size, qsfaMte3Size, qsfaS2GmStartOffset, qsfaMergeMte3Idx,
                                           kvVecRunInfo);
                SetFlag<AscendC::HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx % MTE2_SLOT_NUM);
                qsfaMergeMte3Idx++;
                break;
            }
            CopyInKvOptimized(qsfaMte2Size, qsfaMte3Size, qsfaMergeMte3Idx, qsfaS2IdxArray0, qsfaS2IdxArray1, s2IdLimit,
                              blkTableBase, kvVecRunInfo);
            if ((qsfaMte2Size - qsfaMte3Size + s2Step > 32) || s2GmOffsetArray + s2Step >= qsfaS2GmLimit) {
                CopyOutMrgeResultOptimized(qsfaMte2Size, qsfaMte3Size, qsfaS2GmStartOffset, qsfaMergeMte3Idx,
                                           kvVecRunInfo);
                qsfaMte3Size = qsfaMte2Size;
                SetFlag<AscendC::HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx % MTE2_SLOT_NUM);
                qsfaMergeMte3Idx++;
                qsfaNeedWaitMte3ToMte2 = true;
            }
        }
    }

    if (unlikely(qsfaS2GmStartOffset + qsfaMte2Size < qsfaS2GmLimit)) {
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx % MTE2_SLOT_NUM);
        LocalTensor<K_ROPE_T> mergeUb = kvMergUb_.template ReinterpretCast<K_ROPE_T>();
        Duplicate(mergeUb, static_cast<K_ROPE_T>(0.0), constInfo.headDim);
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        uint64_t blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = constInfo.headDim / blockElementNum;
        dataCopyParams.blockLen = blockElementNum * sizeof(K_ROPE_T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (constInfo.s2BaseSize - 1) * blockElementNum * sizeof(K_ROPE_T);
        for (int64_t s2GmOffset = qsfaS2GmStartOffset + qsfaMte2Size; s2GmOffset < qsfaS2GmLimit; s2GmOffset++) {
            if constexpr (QSFAT::noRope) {
                DataCopyExtParams ndParams{1, static_cast<uint32_t>(constInfo.headDim * sizeof(K_ROPE_T)), 0, 0,
                                           0};
                DataCopyPad(kvMergeGm_[kvVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                                       s2GmOffset * constInfo.headDim],
                            mergeUb, ndParams);
            } else {
                DataCopyPad(
                    kvMergeGm_[kvVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + s2GmOffset * blockElementNum],
                    mergeUb, dataCopyParams);
            }
        }
        if constexpr (!QSFAT::noRope) {
            dataCopyParams.blockCount = (QSFAT::noRope ? 0U : 64U) / blockElementNum;
            for (int64_t s2GmOffset = qsfaS2GmStartOffset + qsfaMte2Size; s2GmOffset < qsfaS2GmLimit; s2GmOffset++) {
                DataCopyPad(kvMergeGm_[kvVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                                       512 * constInfo.headDim + s2GmOffset * blockElementNum],
                            mergeUb, dataCopyParams);
            }
        }
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(qsfaMergeMte3Idx % MTE2_SLOT_NUM);
        qsfaMergeMte3Idx++;
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(2);
    v0ValidSizeUb_.SetValue(kvVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM, qsfaMte2Size);
    SetFlag<AscendC::HardEvent::S_MTE3>(1);
    WaitFlag<AscendC::HardEvent::S_MTE3>(1);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = 128 * sizeof(int32_t);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(kvValidSizeGm_[kvVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2) + GetSubBlockIdx() * 128],
                v0ValidSizeUb_, dataCopyParams);
    return;
}

template <typename QSFAT>
__aicore__ inline void QSFAVectorService<QSFAT>::CopyOutMrgeResultOptimized(int64_t mte2Size, int64_t mte3Size, int64_t
    s2GmStartOffset, int64_t mergeMte3Idx, const RunInfo& kvVecRunInfo)
{
    if (mte2Size <= mte3Size) {
        return;
    }
    int32_t dealRow = mte2Size - mte3Size;
    SetFlag<AscendC::HardEvent::MTE2_V>(0);
    WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    LocalTensor<half> kvTensorAsFp16 = tmpBuff1.template Get<half>();
    uint64_t mask = ConstInfo::BUFFER_SIZE_BYTE_256B / sizeof(half);
    LocalTensor<KV_T> srcTensor = kvMergUb_[(mergeMte3Idx % MTE2_SLOT_NUM) * mte2SlotElems_];
    // packed row is a compile-time constant: rope 64 -> 21 blocks, rope 0 -> 17 blocks
    static constexpr uint32_t ropeDim = (QSFAT::noRope ? 0U : 64U);
    static constexpr uint32_t packedBytes = 512 + ropeDim * sizeof(K_ROPE_T) + 16;
    static constexpr uint8_t rowStrideBlk =
        static_cast<uint8_t>((packedBytes + ConstInfo::BUFFER_SIZE_BYTE_32B - 1) / ConstInfo::BUFFER_SIZE_BYTE_32B);
    static constexpr uint32_t scaleOffset = 512 + ropeDim * sizeof(K_ROPE_T) / sizeof(KV_T);
    if (dealRow == 1) {
        Cast(kvTensorAsFp16, srcTensor, RoundMode::CAST_NONE, mask, 4, {1, 1, 8, 4});
    } else {
        uint8_t repeatTimes = static_cast<uint8_t>(dealRow);
        Cast(kvTensorAsFp16, srcTensor, RoundMode::CAST_NONE, mask, repeatTimes, {1, 1, 32, rowStrideBlk});
        Cast(kvTensorAsFp16[128], srcTensor[128], RoundMode::CAST_NONE, mask, repeatTimes, {1, 1, 32, rowStrideBlk});
        Cast(kvTensorAsFp16[256], srcTensor[256], RoundMode::CAST_NONE, mask, repeatTimes, {1, 1, 32, rowStrideBlk});
        Cast(kvTensorAsFp16[384], srcTensor[384], RoundMode::CAST_NONE, mask, repeatTimes, {1, 1, 32, rowStrideBlk});
    }
    PipeBarrier<PIPE_V>();
    LocalTensor<T> antiQuantScale = tmpBuff2.template Get<T>();
    LocalTensor<T> oriQuantScaleTensor = srcTensor[scaleOffset].template ReinterpretCast<T>();
    if (dealRow == 1) {
        Brcb(antiQuantScale, oriQuantScaleTensor, 1, {1, 4});
    } else {
        DataCopyParams params;
        params.blockCount = dealRow;
        params.blockLen = 1;
        params.dstStride = 0;
        params.srcStride = (512 + ropeDim * sizeof(K_ROPE_T)) / ConstInfo::BUFFER_SIZE_BYTE_32B;
        LocalTensor<T> tmpAntiQuantScale = antiQuantScale[ConstInfo::BUFFER_SIZE_BYTE_1K];
        DataCopy(tmpAntiQuantScale, oriQuantScaleTensor, params);
        PipeBarrier<PIPE_V>();
        Brcb(antiQuantScale, tmpAntiQuantScale, dealRow, {1, 4});
    }
    PipeBarrier<PIPE_V>();
    uint32_t dealLoop = CeilDiv(dealRow, LIMIT_DEAL_ROW);
    uint32_t dealRowFp32 = LIMIT_DEAL_ROW;
    uint32_t element = LIMIT_DEAL_ROW * constInfo.headDim;
    LocalTensor<T> kvTensorAsFp32 = inputBuff2.template Get<T>();
    LocalTensor<K_ROPE_T> antiKvTensorAsB16 = tmpBuff1.template Get<K_ROPE_T>();
    if constexpr (QSFAT::noRope) {
        antiKvTensorAsB16 = outputBuff1.template Get<K_ROPE_T>();
    }
    for (uint32_t i = 0; i < dealLoop; i++) {
        if (i == dealLoop - 1) {
            dealRowFp32 = dealRow - i * LIMIT_DEAL_ROW;
        }
        Cast(kvTensorAsFp32, kvTensorAsFp16[i * element], RoundMode::CAST_NONE,
             static_cast<uint32_t>(dealRowFp32 * constInfo.headDim));
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < constInfo.tileSize / FP32_REPEAT_ELEMENT_NUM; j++) {
            Mul(kvTensorAsFp32[j * FP32_REPEAT_ELEMENT_NUM], kvTensorAsFp32[j * FP32_REPEAT_ELEMENT_NUM],
                antiQuantScale[i * LIMIT_DEAL_ROW * 32], FP32_REPEAT_ELEMENT_NUM, 4 * dealRowFp32,
                {1, 1, 0, 16, 16, 1});
        }
        PipeBarrier<PIPE_V>();
        if constexpr (QSFAT::noRope) {
            if (i == 0) {
                // Preserve overlap until the first write into the ND output slot.
                WaitFlag<HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
            }
        }
        if constexpr (IsSameType<K_ROPE_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
            Cast(antiKvTensorAsB16[i * element], kvTensorAsFp32, RoundMode::CAST_RINT,
                 static_cast<uint32_t>(dealRowFp32 * constInfo.headDim));
        } else {
            Cast(antiKvTensorAsB16[i * element], kvTensorAsFp32, RoundMode::CAST_ROUND,
                 static_cast<uint32_t>(dealRowFp32 * constInfo.headDim));
        }
        PipeBarrier<PIPE_V>();
    }

    if constexpr (QSFAT::noRope) {
        SetFlag<HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        DataCopyExtParams ndParams;
        ndParams.blockCount = 1;
        ndParams.blockLen = dealRow * constInfo.headDim * sizeof(K_ROPE_T);
        ndParams.srcStride = 0;
        ndParams.dstStride = 0;
        DataCopyPad(kvMergeGm_[kvVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                               (s2GmStartOffset + mte3Size) * constInfo.headDim],
                    antiKvTensorAsB16, ndParams);
        SetFlag<HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        return;
    }

    LocalTensor<K_ROPE_T> antiKvTensorAsB16Nz = outputBuff1.template Get<K_ROPE_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    int dataBlocks = REPEAT_BLOCK_BYTE / BYTE_BLOCK;
    int loops = CeilDiv(dealRow, dataBlocks);
    uint64_t tail = dealRow - (loops - 1) * dataBlocks;
    uint64_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2;
    uint64_t blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;
    uint8_t repeatTimes = static_cast<uint8_t>(constInfo.headDim / blockElementNum);
    for (int i = 0; i < loops; i++) {
        mask = (i == loops - 1) ? tail * blockElementNum : repeatElementNum;
        Copy(antiKvTensorAsB16Nz[i * repeatElementNum], antiKvTensorAsB16[i * dataBlocks * constInfo.headDim],
             mask, repeatTimes, {1, 32, static_cast<uint16_t>(dealRow), 1});
    }
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = constInfo.headDim / blockElementNum;
    dataCopyParams.blockLen = dealRow * blockElementNum * sizeof(K_ROPE_T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = (constInfo.s2BaseSize - dealRow) * blockElementNum * sizeof(K_ROPE_T);
    DataCopyPad(kvMergeGm_[kvVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 +
                           (s2GmStartOffset + mte3Size) * blockElementNum],
                antiKvTensorAsB16Nz, dataCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);

    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    if constexpr (!QSFAT::noRope) {
        LocalTensor<K_ROPE_T> kRopeUb = srcTensor[512].template ReinterpretCast<K_ROPE_T>();
        LocalTensor<K_ROPE_T> kRopeUbNz = outputBuff2.template Get<K_ROPE_T>();
        Copy(kRopeUbNz, kRopeUb, ropeDim, static_cast<uint8_t>(dealRow),
             {static_cast<uint16_t>(dealRow), 1, 1, rowStrideBlk});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        dataCopyParams.blockCount = ropeDim / blockElementNum;
        DataCopyPad(kvMergeGm_[kvVecRunInfo.loop % MERGE_CACHE_GM_BUF_NUM * 512 * 576 + 512 * 512 +
                               (s2GmStartOffset + mte3Size) * blockElementNum],
                    kRopeUbNz, dataCopyParams);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

#endif // KV_QUANT_SPARSE_FLASH_ATTENTION_VLLM_SERVICE_VECTOR_MLA_H
