/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file turbo_quant_sparse_attn_sharedkv_scfa_block_vector.h
 * \brief
 */
#ifndef TURBO_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_VECTOR_H
#define TURBO_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../turbo_quant_sparse_attn_sharedkv_common.h"

namespace SASKernel {
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename SAST>
class TurboQuantSparseAttnSharedkvScfaBlockVector {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using KV_T = typename SAST::kvType;
    using OUT_T = typename SAST::outputType;
    using UPDATE_T = T;
    using SINKS_T = T;
    using MM1_OUT_T = float;
    using MM2_OUT_T = float;

    __aicore__ inline TurboQuantSparseAttnSharedkvScfaBlockVector(){};
    __aicore__ inline void ProcessVec0L(const RunInfo &runInfo);
    __aicore__ inline void ProcessVec1L(const RunInfo &info);
    __aicore__ inline void ProcessVec2L(const RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct ConstInfo &constInfo,
                                      const TurboQuantSparseAttnSharedkvTilingData *__restrict tilingData);
    __aicore__ inline void InitVec0GlobalTensor(const GlobalTensor<KV_T> &kvMergeGm, const GlobalTensor<KV_T> &oriKvGm,
                                                const GlobalTensor<KV_T> &cmpKvGm, const GlobalTensor<uint8_t> &cmpTqGm,
                                                const GlobalTensor<int32_t> &oriBlockTableGm,
                                                const GlobalTensor<int32_t> &cmpBlockTableGm);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm,
                                                GlobalTensor<int32_t> actualSeqLengthsQGm,
                                                GlobalTensor<int32_t> actualSeqLengthsKVGm,
                                                GlobalTensor<int32_t> topKGm, GlobalTensor<T> sinksGm,
                                                GlobalTensor<T> softmaxLseGm);
    __aicore__ inline void InitVec2GlobalTensor(GlobalTensor<T> accumOutGm, GlobalTensor<UPDATE_T> vec2ResGm,
                                                GlobalTensor<MM2_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void CopySinksIn();
    __aicore__ inline void SliceAndContactSinksValue(uint32_t nIdx, uint32_t dealRowCount);
    __aicore__ inline void InitSoftmaxDefaultBuffer();
    // ================================Base Vector==========================================
    __aicore__ inline void RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    // ================================Vector0==========================================
    __aicore__ inline int64_t GetKeyGmOffset(int64_t realS2Idx, const RunInfo &runInfo, int64_t s2IdLimit,
                                             const LocalTensor<int32_t> &blockTableCache, uint32_t blockTableCacheSize);
    __aicore__ inline void GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx, int64_t topkGmBaseOffset,
                                        const RunInfo &runInfo, const LocalTensor<int32_t> &topkCache,
                                        int64_t topkCacheStart, uint32_t topkCacheSize);
    __aicore__ inline void CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx1,
                                    int64_t realS2Idx2, const RunInfo &runInfo,
                                    const LocalTensor<int32_t> &blockTableCache, uint32_t blockTableCacheSize);
    __aicore__ inline void CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size, int64_t s2StartGmOffset,
                                             int64_t mergeMte3Idx, const RunInfo &runInfo);
    __aicore__ inline void DequantTq4Rows(LocalTensor<uint8_t> slots, LocalTensor<KV_T> output, int64_t rowCount);
    __aicore__ inline void CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                          int64_t keyBNBOffset, int64_t s2IdLimit, const RunInfo &runInfo);
    // ================================Vector1==========================================
    __aicore__ inline void ProcessVec1SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm1ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount, uint32_t loopId);
    __aicore__ inline void SoftmaxFlashV2Compute(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                 LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);

    __aicore__ inline void ElewiseCompute(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                          const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount);
    __aicore__ inline void SetInfInBlk(const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount,
                                       int64_t startId, int64_t endId);
    __aicore__ inline void ProcessLse(const RunInfo &info, const MSplitInfo &mSplitInfo);
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
    __aicore__ inline void Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                             uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline uint64_t CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx);

    // BLOCK和REPEAT的字节数
    static constexpr uint64_t BYTE_BLOCK = 32UL;
    static constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
    // BLOCK和REPEAT的FP32元素数
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
    // repeat stride不能超过256
    static constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;

private:
    static constexpr bool PAGE_ATTENTION = SAST::pageAttention;
    static constexpr int TEMPLATE_MODE = SAST::templateMode;
    static constexpr bool FLASH_DECODE = SAST::flashDecode;
    static constexpr SAS_LAYOUT LAYOUT_T = SAST::layout;
    static constexpr SAS_LAYOUT KV_LAYOUT_T = SAST::kvLayout;

    static constexpr uint64_t MERGE_CACHE_GM_BUF_NUM = 3;
    static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_INPUT_BUF1_PONG_FLAG = 3;
    static constexpr uint64_t SYNC_INPUT_BUF2_FLAG = 4;
    static constexpr uint64_t SYNC_INPUT_BUF2_PONG_FLAG = 5;
    static constexpr uint64_t SYNC_OUTPUT_BUF1_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF2_FLAG = 5;
    static constexpr uint64_t SYNC_SINKS_BUF_FLAG = 6;
    static constexpr uint64_t SYNC_INPUT_V0BUF_FLAG = 7;
    static constexpr uint32_t INPUT1_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_32K;
    static constexpr uint32_t INPUT2_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_16K;
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_1K;
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = ConstInfo::BUFFER_SIZE_BYTE_32K / sizeof(T); // 32768/4=8096
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);                               // 32/4=8
    static constexpr uint32_t MAX_N1_SIZE = 128U;
    // tmpBuff1: batched work/codes/indices followed by two aligned raw-slot regions.
    static constexpr uint32_t TQ4_MAX_STAGED_ROWS = INPUT2_BUFFER_OFFSET / (TQ4_HEAD_DIM * sizeof(KV_T));
    static constexpr uint32_t TQ4_RAW_REGION_BYTES = TQ4_MAX_STAGED_ROWS * TQ4_SLOT_ROW_BYTES;
    static constexpr uint32_t TQ4_RAW_BUFFER_BYTES = 2U * TQ4_RAW_REGION_BYTES;
    static constexpr uint32_t TQ4_DEQUANT_BYTES_PER_ROW =
        TQ4_HEAD_DIM * (sizeof(float) + sizeof(half) + sizeof(int32_t));
    static constexpr uint32_t TQ4_DEQUANT_BATCH_ROWS =
        (ConstInfo::BUFFER_SIZE_BYTE_32K - TQ4_RAW_BUFFER_BYTES) / TQ4_DEQUANT_BYTES_PER_ROW;
    static constexpr uint32_t TQ4_WORK_BYTES = TQ4_DEQUANT_BATCH_ROWS * TQ4_HEAD_DIM * sizeof(float);
    static constexpr uint32_t TQ4_SIGNED_BYTES = TQ4_DEQUANT_BATCH_ROWS * TQ4_HEAD_DIM * sizeof(half);
    static constexpr uint32_t TQ4_INDEX_BYTES = TQ4_DEQUANT_BATCH_ROWS * TQ4_HEAD_DIM * sizeof(int32_t);
    static constexpr uint32_t TQ4_SIGNED_OFFSET = TQ4_WORK_BYTES;
    static constexpr uint32_t TQ4_INDEX_OFFSET = TQ4_SIGNED_OFFSET + TQ4_SIGNED_BYTES;
    static constexpr uint32_t TQ4_RAW_OFFSET = TQ4_INDEX_OFFSET + TQ4_INDEX_BYTES;
    static constexpr uint32_t TQ4_CENTROID_COUNT = 16U;
    static constexpr uint32_t TQ4_SCALE_GATHER_INDEX_OFFSET = TQ4_CENTROID_COUNT * sizeof(float);
    static constexpr uint32_t TQ4_SCALE_GATHER_INDEX_BYTES = TQ4_DEQUANT_BATCH_ROWS * sizeof(uint32_t);
    static constexpr uint32_t TQ4_SCALE_CARRIER_BYTES = TQ4_DEQUANT_BATCH_ROWS * BYTE_BLOCK;
    static constexpr uint32_t TQ4_SCALE_BROADCAST_OFFSET = BYTE_BLOCK;
    static constexpr uint32_t TQ4_SCALE_BROADCAST_BYTES = (TQ4_DEQUANT_BATCH_ROWS + FP32_BLOCK_ELEMENT_NUM - 1U) /
                                                          FP32_BLOCK_ELEMENT_NUM * FP32_BLOCK_ELEMENT_NUM *
                                                          FP32_BLOCK_ELEMENT_NUM * sizeof(float);
    // Vec0 and Vec2 execute in different pipeline stages, but keep Vec2's existing 512B scratch region reserved.
    static constexpr uint32_t V0_VEC2_TEMP_OFFSET = 384U * sizeof(T);
    static constexpr uint32_t V0_VEC2_TEMP_BYTES = ConstInfo::BUFFER_SIZE_BYTE_512B;
    static constexpr uint32_t TQ4_TOPK_CACHE_BYTES = V0_VEC2_TEMP_OFFSET;
    static constexpr uint32_t TQ4_BLOCK_TABLE_CACHE_OFFSET = V0_VEC2_TEMP_OFFSET + V0_VEC2_TEMP_BYTES;
    static constexpr uint32_t TQ4_BLOCK_TABLE_CACHE_BYTES =
        ConstInfo::BUFFER_SIZE_BYTE_8K - TQ4_BLOCK_TABLE_CACHE_OFFSET;
    static constexpr uint32_t TQ4_TOPK_CACHE_CAPACITY = TQ4_TOPK_CACHE_BYTES / sizeof(int32_t);
    static constexpr uint32_t TQ4_BLOCK_TABLE_CACHE_CAPACITY = TQ4_BLOCK_TABLE_CACHE_BYTES / sizeof(int32_t);
    static_assert(TQ4_DEQUANT_BATCH_ROWS > 0U, "TQ4 dequant scratch cannot hold one row");
    static_assert(TQ4_RAW_OFFSET + TQ4_RAW_BUFFER_BYTES <= ConstInfo::BUFFER_SIZE_BYTE_32K,
                  "TQ4 batched dequant scratch exceeds tmpBuff1");
    static_assert(TQ4_SCALE_GATHER_INDEX_OFFSET + TQ4_SCALE_GATHER_INDEX_BYTES <= ConstInfo::BUFFER_SIZE_BYTE_256B,
                  "TQ4 scale gather indices overlap tqCentBuff");
    static_assert(TQ4_SCALE_CARRIER_BYTES + TQ4_DEQUANT_BATCH_ROWS * sizeof(half) <= TQ4_SIGNED_BYTES,
                  "TQ4 scale extraction exceeds signed-code scratch");
    static_assert(TQ4_SCALE_BROADCAST_OFFSET + TQ4_SCALE_BROADCAST_BYTES <= TQ4_INDEX_BYTES,
                  "TQ4 scale broadcast exceeds index scratch");
    static_assert(TQ4_TOPK_CACHE_BYTES <= TQ4_BLOCK_TABLE_CACHE_OFFSET, "TQ4 index caches overlap in v0ValidSizeBuff");
    static constexpr T SOFTMAX_MIN_NUM = -2e38;
    static constexpr SINKS_T R0 = 1.0f;

    const TurboQuantSparseAttnSharedkvTilingData *__restrict tilingData;

    uint32_t pingpongFlag = 0U;
    ConstInfo constInfo = {};

    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<T> softmaxMaxGm;
    GlobalTensor<T> softmaxSumGm;
    GlobalTensor<T> sinksGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<UPDATE_T> vec2ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<T> softmaxLseGm;

    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<KV_T> kvMergeGm_;
    GlobalTensor<KV_T> keyGm_;
    GlobalTensor<int32_t> topkGm_;
    GlobalTensor<KV_T> oriKvGm_;
    GlobalTensor<KV_T> cmpKvGm_;
    GlobalTensor<uint8_t> cmpTqGm_;
    GlobalTensor<int32_t> oriBlockTableGm_;
    GlobalTensor<int32_t> cmpBlockTableGm_;

    // ================================Local Buffer区====================================
    TBuf<> inputBuff1;  // 32K
    TBuf<> inputBuff2;  // 16K
    TBuf<> outputBuff1; // 32K
    TBuf<> outputBuff2; // 32K

    TBuf<> tmpBuff1;        // 32K
    TBuf<> v0ValidSizeBuff; // 8K
    TBuf<> tqCentBuff;      // 256B

    TBuf<> sinksBuff;     // 1K
    TBuf<> sinksBrcbBuff; // 12K

    TBuf<> softmaxMaxBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxExpBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxSumBuff;        // PRE_LOAD_NUM * 2K
    TBuf<> softmaxMaxDefaultBuff; // 2K
    TBuf<> softmaxSumDefaultBuff; // 2K

    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;
    LocalTensor<KV_T> kvMergUb_;
    LocalTensor<int32_t> v0ValidSizeUb_;
    LocalTensor<SINKS_T> sinksUb;
    LocalTensor<SINKS_T> sinksBrcbUb;

    uint32_t mergeMte3Idx = 0;
};

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K * 2); // 2:pingpong
    pipe->InitBuffer(inputBuff2, ConstInfo::BUFFER_SIZE_BYTE_16K * 2); // 2:pingpong
    pipe->InitBuffer(outputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    if (constInfo.returnSoftmaxLse) {
        pipe->InitBuffer(outputBuff2, ConstInfo::BUFFER_SIZE_BYTE_1K);
    }

    pipe->InitBuffer(tmpBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(v0ValidSizeBuff, ConstInfo::BUFFER_SIZE_BYTE_8K);
    if (constInfo.kvQuantMode == 3) {
        pipe->InitBuffer(tqCentBuff, ConstInfo::BUFFER_SIZE_BYTE_256B);
        LoadTq4Centroids(tqCentBuff.Get<float>());
        LocalTensor<uint32_t> scaleGatherIndices =
            tqCentBuff.Get<uint32_t>()[TQ4_SCALE_GATHER_INDEX_OFFSET / sizeof(uint32_t)];
        for (uint32_t row = 0; row < TQ4_DEQUANT_BATCH_ROWS; ++row) {
            scaleGatherIndices.SetValue(row, row * BYTE_BLOCK);
        }
        PipeBarrier<PIPE_ALL>();
    }

    // M_MAX = 512/2vector = 256, 256 * sizeof(T) * N_Buffer

    pipe->InitBuffer(softmaxMaxBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxExpBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxSumBuff, ConstInfo::BUFFER_SIZE_BYTE_1K * constInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_1K);
    pipe->InitBuffer(softmaxSumDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_1K);

    pipe->InitBuffer(sinksBuff, MAX_N1_SIZE * sizeof(SINKS_T));
    // 分配256+N1大小内存，其中256是m轴VEC最大切块
    pipe->InitBuffer(sinksBrcbBuff, MAX_N1_SIZE * sizeof(SINKS_T) * BLOCK_ELEMENT_NUM * 3U);

    softmaxMaxUb = softmaxMaxBuff.Get<T>();
    softmaxSumUb = softmaxSumBuff.Get<T>();
    softmaxExpUb = softmaxExpBuff.Get<T>();

    softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
    softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();

    kvMergUb_ = inputBuff2.Get<KV_T>();

    v0ValidSizeUb_ = v0ValidSizeBuff.Get<int32_t>();

    sinksUb = sinksBuff.Get<SINKS_T>();
    sinksBrcbUb = sinksBrcbBuff.Get<SINKS_T>();
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::InitParams(
    const struct ConstInfo &constInfo, const TurboQuantSparseAttnSharedkvTilingData *__restrict tilingData)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::InitVec0GlobalTensor(
    const GlobalTensor<KV_T> &kvMergeGm, const GlobalTensor<KV_T> &oriKvGm, const GlobalTensor<KV_T> &cmpKvGm,
    const GlobalTensor<uint8_t> &cmpTqGm, const GlobalTensor<int32_t> &oriBlockTableGm,
    const GlobalTensor<int32_t> &cmpBlockTableGm)
{
    this->kvMergeGm_ = kvMergeGm;
    this->oriKvGm_ = oriKvGm;
    this->cmpKvGm_ = cmpKvGm;
    this->cmpTqGm_ = cmpTqGm;
    this->oriBlockTableGm_ = oriBlockTableGm;
    this->cmpBlockTableGm_ = cmpBlockTableGm;
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm, GlobalTensor<int32_t> actualSeqLengthsQGm,
    GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<int32_t> topKGm, GlobalTensor<SINKS_T> sinksGm,
    GlobalTensor<T> softmaxLseGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->actualSeqLengthsQGm = actualSeqLengthsQGm;
    this->actualSeqLengthsKVGm = actualSeqLengthsKVGm;
    this->topkGm_ = topKGm;
    this->sinksGm = sinksGm;
    this->softmaxLseGm = softmaxLseGm;
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::InitVec2GlobalTensor(
    GlobalTensor<T> accumOutGm, GlobalTensor<UPDATE_T> vec2ResGm, GlobalTensor<MM2_OUT_T> mm2ResGm,
    GlobalTensor<OUT_T> attentionOutGm)
{
    this->accumOutGm = accumOutGm;
    this->vec2ResGm = vec2ResGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::CopySinksIn()
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = constInfo.qHeadNum * sizeof(T);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<T> padParams;
    DataCopyPad(sinksUb, sinksGm, dataCopyParams, padParams);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINKS_BUF_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_SINKS_BUF_FLAG);
    uint32_t repeatTimes = (constInfo.qHeadNum + BLOCK_ELEMENT_NUM - 1U) / BLOCK_ELEMENT_NUM; // 每次处理 8 datablocks
    Brcb(sinksBrcbUb, sinksUb, repeatTimes, {1, BLOCK_ELEMENT_NUM});
    PipeBarrier<PIPE_V>();

    DataCopyParams repeatParams;
    repeatParams.blockCount = 1; // 搬到有一个块超过单个vec核减分核M轴大小即可，核间切分每个vec256
    repeatParams.blockLen = constInfo.qHeadNum;
    repeatParams.srcStride = 0U;
    repeatParams.dstStride = 0U;
    for (uint32_t i = 1U; i <= 256U / constInfo.qHeadNum; i++) {
        DataCopy(sinksBrcbUb[constInfo.qHeadNum * BLOCK_ELEMENT_NUM * i], sinksBrcbUb, repeatParams);
    }
    PipeBarrier<PIPE_V>();
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::SliceAndContactSinksValue(
    uint32_t nIdx, uint32_t dealRowCount)
{
    // WholeReduceMax接口中repeatTimes支持范围（0,255），因此需要分多次调用WholeReduceMax，每次repeatTime=128
    uint32_t repeatTimesOnce = 128;
    uint32_t loopTimes = (dealRowCount + repeatTimesOnce - 1) / repeatTimesOnce;
    uint32_t repeatTimes = repeatTimesOnce;

    for (uint32_t loop = 0; loop < loopTimes; ++loop) {
        if (loop == loopTimes - 1) {
            repeatTimes = dealRowCount - loop * repeatTimesOnce;
        }
        WholeReduceMax(softmaxMaxDefaultUb[loop * repeatTimesOnce],
                       sinksBrcbUb[(nIdx + loop * repeatTimesOnce) * BLOCK_ELEMENT_NUM],
                       BLOCK_ELEMENT_NUM * BLOCK_ELEMENT_NUM, repeatTimes, 1, 0, 1, ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
    }
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::InitSoftmaxDefaultBuffer()
{
    CopySinksIn();
    Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
    Duplicate(softmaxSumDefaultUb, R0, SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T));
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::ElewiseCompute(
    const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow, const LocalTensor<T> &mmResUb,
    uint32_t dealRowCount, uint32_t columnCount)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.softmaxScale), dealRowCount * columnCount);

    // cmp_sparse_indices is capacity-sized and may be padded with -1.  The Cube
    // path still computes the capacity-sized tile, so mask the padded columns
    // before softmax instead of allowing stale merge-cache rows to contribute.
    if (!info.isOriOnly && constInfo.sparseBlockCount > 0 && info.cmpS2IdLimit > 0) {
        uint32_t rowBase = mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
        for (uint32_t row = 0; row < dealRowCount; ++row) {
            uint32_t mPos = rowBase + row;
            uint32_t qRel = mPos / constInfo.gSize;
            uint64_t topkRow = info.topKBaseOffset + static_cast<uint64_t>(qRel) * constInfo.sparseBlockCount;
            int32_t probeCount = static_cast<int32_t>(constInfo.sparseBlockCount);
            if (info.cmpS2IdLimit < probeCount) {
                probeCount = info.cmpS2IdLimit;
            }
            int32_t validCount = probeCount;
            int32_t probeTopk = probeCount > 0 ? topkGm_.GetValue(topkRow + probeCount - 1) : -1;
            if (probeTopk < 0) {
                validCount = 0;
                while (validCount < probeCount && topkGm_.GetValue(topkRow + static_cast<uint32_t>(validCount)) >= 0) {
                    ++validCount;
                }
            }
            int32_t tileStart = static_cast<int32_t>(info.relativeS2Idx * constInfo.s2BaseSize);
            int32_t maskStart = validCount - tileStart;
            if (maskStart < 0) {
                maskStart = 0;
            }
            if (maskStart < static_cast<int32_t>(info.actualSingleProcessSInnerSize)) {
                SetInfInBlk(mmResUb[row * columnCount], 1, columnCount, maskStart,
                            static_cast<int64_t>(columnCount) - 1);
            }
        }
    }
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::SetInfInBlk(const LocalTensor<T> &mmResUb,
                                                                                      uint32_t dealRowCount,
                                                                                      uint32_t columnCount,
                                                                                      int64_t startId, int64_t endId)
{
    if (startId > endId) {
        return;
    }
    int64_t firstBlock = startId / BLOCK_ELEMENT_NUM;
    int64_t lastBlock = endId / BLOCK_ELEMENT_NUM;
    for (int64_t block = firstBlock; block <= lastBlock; ++block) {
        int64_t blockStart = block * BLOCK_ELEMENT_NUM;
        int64_t begin = startId > blockStart ? startId - blockStart : 0;
        int64_t finish = endId < blockStart + BLOCK_ELEMENT_NUM - 1 ? endId - blockStart : BLOCK_ELEMENT_NUM - 1;
        uint64_t maskValue = 0;
        for (int64_t bit = begin; bit <= finish; ++bit) {
            maskValue |= (1ULL << static_cast<uint64_t>(bit));
        }
        uint64_t mask[1] = {maskValue};
        Duplicate(mmResUb[blockStart], SOFTMAX_MIN_NUM, mask, dealRowCount, 1, columnCount / BLOCK_ELEMENT_NUM);
    }
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::ProcessLse(const RunInfo &info,
                                                                                     const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint64_t lseOffset;
    if (constInfo.outputLayout == SAS_LAYOUT::TND) {
        uint32_t tBase = actualSeqLengthsQGm.GetValue(info.bIdx);
        lseOffset = (tBase + info.s1Idx) * constInfo.gSize +               // T轴、s1轴偏移
                    info.n2IdxReal * constInfo.qSeqSize * constInfo.gSize; // N2轴偏移
    } else if (constInfo.outputLayout == SAS_LAYOUT::BSND) {
        lseOffset = info.bIdx * constInfo.qSeqSize * constInfo.kvHeadNum * constInfo.gSize + // B轴偏移
                    info.n2IdxReal * constInfo.qSeqSize * constInfo.gSize +                  // N2轴偏移
                    info.s1Idx * constInfo.gSize;                                            // S1轴偏移
    }
    lseOffset = lseOffset + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2;
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    auto sumTensor = softmaxSumUb[softmaxOffset];
    auto maxTensor = softmaxMaxUb[softmaxOffset];
    auto outLSETensor = outputBuff2.Get<T>();
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = mSplitInfo.vecDealM * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    PipeBarrier<PIPE_V>();
    Log(outLSETensor, sumTensor, mSplitInfo.vecDealM);
    PipeBarrier<PIPE_V>();
    Add(outLSETensor, outLSETensor, maxTensor, mSplitInfo.vecDealM);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);

    DataCopyPad(softmaxLseGm[lseOffset], outLSETensor, dataCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::SoftmaxFlashV2Compute(
    const RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset;
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb[startRow];
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop - 1) % (constInfo.preLoadNum);
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset];
    }
    if (actualColumnCount != 0) {
        SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
        SoftMaxTiling newTiling =
            SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);
        SoftmaxFlashV2<T, true, true, false, false, SAS_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
            mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
            softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    } else {
        uint32_t dealRowCountAlign = SASAlign(dealRowCount, FP32_BLOCK_ELEMENT_NUM);
        DataCopy(softmaxSumUb[softmaxOutOffset], inSumTensor, dealRowCountAlign);
        PipeBarrier<PIPE_V>();
        DataCopy(softmaxMaxUb[softmaxOutOffset], inMaxTensor, dealRowCountAlign);
    }
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::DealBmm1ResBaseBlock(
    const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
    uint32_t loopId)
{
    uint32_t computeSize = dealRowCount * columnCount;
    uint64_t inOutGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                             (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow) * columnCount;
    LocalTensor<MM1_OUT_T> mmResUb = inputBuff1.Get<MM1_OUT_T>();
    mmResUb = mmResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM1_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(mmResUb, mm1ResGm[inOutGmOffset], computeSize);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    ElewiseCompute(info, mSplitInfo, startRow, mmResUb, dealRowCount, columnCount);

    PipeBarrier<PIPE_V>();
    LocalTensor<T> tmpAFloorUb = tmpBuff1.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();

    SoftmaxFlashV2Compute(info, mSplitInfo, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount,
                          info.actualSingleProcessSInnerSize);

    PipeBarrier<PIPE_V>();
    LocalTensor<KV_T> tmpMMResCastTensor = outputBuff1.Get<KV_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);

    Cast(tmpMMResCastTensor, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    pingpongFlag ^= 1; // pingpong 0 1 切换

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(vec1ResGm[inOutGmOffset], tmpMMResCastTensor, computeSize);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::ProcessVec1SingleBuf(
    const RunInfo &info, const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t mSplitSize = info.actualSingleProcessSInnerSize == 0 ?
                              16 :
                              BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign;
    // 1. 向下8对齐是因为UB操作至少32B
    // 2. info.actualSingleProcessSInnerSizeAlign最大512, mSplitSize可以确保最小为16
    mSplitSize = mSplitSize / 8 * 8;

    if (mSplitSize > mSplitInfo.vecDealM) {
        mSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * mSplitSize;

    SliceAndContactSinksValue((mSplitInfo.nBufferStartM + mSplitInfo.vecStartM) % constInfo.qHeadNum,
                              mSplitInfo.vecDealM);

    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm1ResBaseBlock(info, mSplitInfo, i * mSplitSize, dealSize, info.actualSingleProcessSInnerSizeAlign, i);
    }
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::GetRealS2Idx(
    int64_t s2GmOffset, int64_t &realS2Idx, int64_t topkGmBaseOffset, const RunInfo &runInfo,
    const LocalTensor<int32_t> &topkCache, int64_t topkCacheStart, uint32_t topkCacheSize)
{
    int64_t cmpS2Offset = s2GmOffset;
    int64_t topkGmIdx = cmpS2Offset / constInfo.sparseBlockSize;
    if (unlikely(topkGmIdx >= constInfo.sparseBlockCount || s2GmOffset >= runInfo.v0S2DealSize)) {
        realS2Idx = -1;
        return;
    }
    int32_t topkValue = 0;
    int64_t topkCacheIdx = topkGmIdx - topkCacheStart;
    if (topkCacheIdx >= 0 && topkCacheIdx < static_cast<int64_t>(topkCacheSize)) {
        topkValue = topkCache.GetValue(static_cast<uint32_t>(topkCacheIdx));
    } else {
        topkValue = topkGm_.GetValue(topkGmBaseOffset + topkGmIdx);
    }
    realS2Idx = topkValue * static_cast<int64_t>(constInfo.sparseBlockSize) +
                static_cast<int64_t>(cmpS2Offset % constInfo.sparseBlockSize);
}

template <typename SAST>
__aicore__ inline int64_t TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::GetKeyGmOffset(
    int64_t realS2Idx, const RunInfo &runInfo, int64_t s2IdLimit, const LocalTensor<int32_t> &blockTableCache,
    uint32_t blockTableCacheSize)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t realKeyGmOffset = 0;
    if constexpr (KV_LAYOUT_T == SAS_LAYOUT::PA_ND) {
        int64_t blkTableIdx = realS2Idx / constInfo.paCmpBlockSize;
        int64_t blkTableOffset = realS2Idx % constInfo.paCmpBlockSize;
        int64_t blockId = blkTableIdx < static_cast<int64_t>(blockTableCacheSize) ?
                              blockTableCache.GetValue(static_cast<uint32_t>(blkTableIdx)) :
                              cmpBlockTableGm_.GetValue(runInfo.bIdx * constInfo.cmpMaxBlockNumPerBatch + blkTableIdx);
        if (constInfo.kvQuantMode == 3) {
            realKeyGmOffset = blockId * static_cast<int64_t>(constInfo.cmpKvStride0) +
                              blkTableOffset * static_cast<int64_t>(TQ4_SLOT_BYTES);
        } else {
            realKeyGmOffset =
                blockId * static_cast<int64_t>(constInfo.cmpKvStride0) +
                blkTableOffset * static_cast<int64_t>(constInfo.kvHeadNum) * static_cast<int64_t>(constInfo.headDim);
        }

    } else if constexpr (KV_LAYOUT_T == SAS_LAYOUT::BSND) {
        realKeyGmOffset = runInfo.bIdx * constInfo.kvSeqSize / constInfo.cmpRatio * constInfo.kvHeadNum +
                          realS2Idx * constInfo.kvHeadNum;
    } else if constexpr (KV_LAYOUT_T == SAS_LAYOUT::TND) {
        realKeyGmOffset =
            (runInfo.tensorCmpBOffset + realS2Idx * constInfo.kvHeadNum * constInfo.headDim) / constInfo.headDim;
    }
    return realKeyGmOffset;
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::DequantTq4Rows(LocalTensor<uint8_t> slots,
                                                                                         LocalTensor<KV_T> output,
                                                                                         int64_t rowCount)
{
    LocalTensor<float> work = tmpBuff1.Get<float>();
    LocalTensor<half> signedCodes = tmpBuff1.Get<half>()[TQ4_SIGNED_OFFSET / sizeof(half)];
    LocalTensor<int32_t> indices = tmpBuff1.Get<int32_t>()[TQ4_INDEX_OFFSET / sizeof(int32_t)];
    LocalTensor<uint32_t> unsignedIndices = indices.template ReinterpretCast<uint32_t>();
    LocalTensor<float> centroids = tqCentBuff.Get<float>();
    LocalTensor<int4b_t> packedCodes = slots.template ReinterpretCast<int4b_t>();
    LocalTensor<half> slotHalf = slots.template ReinterpretCast<half>();
    LocalTensor<half> scaleCarriers = signedCodes;
    LocalTensor<half> scalesFp16 = signedCodes[TQ4_SCALE_CARRIER_BYTES / sizeof(half)];
    LocalTensor<float> scalesFp32 = indices.template ReinterpretCast<float>();
    LocalTensor<float> scalesBrcb = scalesFp32[TQ4_SCALE_BROADCAST_OFFSET / sizeof(float)];
    LocalTensor<uint32_t> scaleGatherIndices =
        tqCentBuff.Get<uint32_t>()[TQ4_SCALE_GATHER_INDEX_OFFSET / sizeof(uint32_t)];

    for (int64_t base = 0; base < rowCount; base += TQ4_DEQUANT_BATCH_ROWS) {
        uint32_t currentRows =
            static_cast<uint32_t>(base + TQ4_DEQUANT_BATCH_ROWS <= rowCount ? TQ4_DEQUANT_BATCH_ROWS : rowCount - base);
        uint32_t elementCount = currentRows * TQ4_HEAD_DIM;
        for (uint32_t row = 0; row < currentRows; ++row) {
            uint32_t sourceRow = static_cast<uint32_t>(base) + row;
            Cast(signedCodes[row * TQ4_HEAD_DIM], packedCodes[sourceRow * TQ4_SLOT_ROW_BYTES * 2U],
                 RoundMode::CAST_NONE, TQ4_HEAD_DIM);
        }
        PipeBarrier<PIPE_V>();
        Adds(signedCodes, signedCodes, static_cast<half>(8.0f), elementCount);
        PipeBarrier<PIPE_V>();
        Muls(signedCodes, signedCodes, static_cast<half>(TQ4_CENTROID_BYTE_STRIDE), elementCount);
        PipeBarrier<PIPE_V>();
        Cast(indices, signedCodes, RoundMode::CAST_ROUND, elementCount);
        PipeBarrier<PIPE_V>();
        Gather(work, centroids, unsignedIndices, 0, elementCount);
        PipeBarrier<PIPE_V>();

        // Extract one aligned carrier per staged row, then broadcast its leading FP16 scale.
        DataCopyParams scaleCopyParams;
        scaleCopyParams.blockCount = static_cast<uint16_t>(currentRows);
        scaleCopyParams.blockLen = 1;
        scaleCopyParams.srcStride = static_cast<uint16_t>(TQ4_SLOT_ROW_BYTES / BYTE_BLOCK - 1U);
        scaleCopyParams.dstStride = 0;
        uint32_t scaleSourceOffset =
            (static_cast<uint32_t>(base) * TQ4_SLOT_ROW_BYTES + TQ4_PACKED_BYTES) / sizeof(half);
        DataCopy(scaleCarriers, slotHalf[scaleSourceOffset], scaleCopyParams);
        PipeBarrier<PIPE_V>();
        Gather(scalesFp16, scaleCarriers, scaleGatherIndices, 0, currentRows);
        PipeBarrier<PIPE_V>();
        Cast(scalesFp32, scalesFp16, RoundMode::CAST_NONE, currentRows);
        PipeBarrier<PIPE_V>();
        uint8_t scaleBrcbRepeats =
            static_cast<uint8_t>((currentRows + FP32_BLOCK_ELEMENT_NUM - 1U) / FP32_BLOCK_ELEMENT_NUM);
        Brcb(scalesBrcb, scalesFp32, scaleBrcbRepeats, {1, FP32_BLOCK_ELEMENT_NUM});
        PipeBarrier<PIPE_V>();
        RowMuls(work, work, scalesBrcb, currentRows, TQ4_HEAD_DIM, TQ4_HEAD_DIM);
        PipeBarrier<PIPE_V>();
        if constexpr (IsSameType<KV_T, bfloat16_t>::value) {
            Cast(output[base * TQ4_HEAD_DIM], work, RoundMode::CAST_RINT, elementCount);
        } else {
            Cast(output[base * TQ4_HEAD_DIM], work, RoundMode::CAST_ROUND, elementCount);
        }
        // The next batch reuses work; the final batch is synchronized by CopyOutMrgeResult's V_MTE3 event.
        if (base + static_cast<int64_t>(currentRows) < rowCount) {
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::CopyInSingleKv(
    int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx, int64_t keyBNBOffset,
    int64_t s2IdLimit, const RunInfo &runInfo)
{
    if (keyBNBOffset < 0) {
        return;
    }
    if (constInfo.kvQuantMode == 3) {
        int64_t stagedRow = mte2Size - mte3Size;
        if (stagedRow == 0) {
            SetFlag<HardEvent::V_MTE2>(0);
            WaitFlag<HardEvent::V_MTE2>(0);
        }
        LocalTensor<uint8_t> slot = tmpBuff1.Get<uint8_t>()[TQ4_RAW_OFFSET + mergeMte3Idx % 2 * TQ4_RAW_REGION_BYTES +
                                                            stagedRow * TQ4_SLOT_ROW_BYTES];
        DataCopyExtParams copyParams{1, TQ4_SLOT_BYTES, 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{true, 0, static_cast<uint8_t>(TQ4_SLOT_ROW_BYTES - TQ4_SLOT_BYTES), 0};
        DataCopyPad(slot, cmpTqGm_[keyBNBOffset], copyParams, padParams);
        ++mte2Size;
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
    if constexpr (KV_LAYOUT_T == SAS_LAYOUT::PA_ND) {
        DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * INPUT2_BUFFER_OFFSET / sizeof(KV_T) +
                              (mte2Size - mte3Size) * constInfo.headDim],
                    cmpKvGm_[keyBNBOffset], intriParams, padParams);
    } else {
        DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * INPUT2_BUFFER_OFFSET / sizeof(KV_T) +
                              (mte2Size - mte3Size) * constInfo.headDim],
                    cmpKvGm_[keyBNBOffset * constInfo.headDim], intriParams, padParams);
    }
    mte2Size += validS2Count;
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::CopyInKv(
    int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx1, int64_t realS2Idx2,
    const RunInfo &runInfo, const LocalTensor<int32_t> &blockTableCache, uint32_t blockTableCacheSize)
{
    int64_t s2IdLimit = runInfo.cmpS2IdLimit;

    int64_t keyOffset1 = GetKeyGmOffset(realS2Idx1, runInfo, s2IdLimit, blockTableCache, blockTableCacheSize);
    int64_t keyOffset2 = GetKeyGmOffset(realS2Idx2, runInfo, s2IdLimit, blockTableCache, blockTableCacheSize);
    if (unlikely(keyOffset1 < 0 && keyOffset2 < 0)) {
        return;
    }
    if (constInfo.kvQuantMode == 3) {
        int64_t validHitCount = (keyOffset1 >= 0) + (keyOffset2 >= 0);
        int64_t keySrcStride = keyOffset2 - keyOffset1 - static_cast<int64_t>(TQ4_SLOT_BYTES);
        // A two-burst copy is valid only when increasing GM order matches logical sparse-index order.
        // Descending, duplicate, tail, and stride-limit cases retain the ordered single-slot path.
        if (validHitCount == 2 && keySrcStride >= 0 && keySrcStride < INT32_MAX) {
            int64_t stagedRow = mte2Size - mte3Size;
            if (stagedRow == 0) {
                SetFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(0);
            }
            LocalTensor<uint8_t> slot =
                tmpBuff1.Get<uint8_t>()[TQ4_RAW_OFFSET + mergeMte3Idx % 2 * TQ4_RAW_REGION_BYTES +
                                        stagedRow * TQ4_SLOT_ROW_BYTES];
            DataCopyExtParams copyParams{static_cast<uint16_t>(validHitCount), TQ4_SLOT_BYTES,
                                         static_cast<uint32_t>(keySrcStride), 0, 0};
            DataCopyPadExtParams<uint8_t> padParams{true, 0, static_cast<uint8_t>(TQ4_SLOT_ROW_BYTES - TQ4_SLOT_BYTES),
                                                    0};
            DataCopyPad(slot, cmpTqGm_[keyOffset1], copyParams, padParams);
            mte2Size += validHitCount;
        } else {
            CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyOffset1, s2IdLimit, runInfo);
            CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyOffset2, s2IdLimit, runInfo);
        }
        return;
    }

    int64_t keySrcStride = 0;
    if constexpr (KV_LAYOUT_T == SAS_LAYOUT::PA_ND) {
        int64_t blkTableSrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) : (keyOffset2 - keyOffset1)) -
                                     constInfo.sparseBlockSize * constInfo.headDim);
        keySrcStride = blkTableSrcStride * sizeof(KV_T);
    } else if constexpr (KV_LAYOUT_T == SAS_LAYOUT::BSND) {
        keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) : (keyOffset2 - keyOffset1)) -
                        constInfo.sparseBlockSize) *
                       constInfo.headDim * sizeof(KV_T);
    } else if constexpr (KV_LAYOUT_T == SAS_LAYOUT::TND) {
        keySrcStride = ((keyOffset1 > keyOffset2 ? (keyOffset1 - keyOffset2) : (keyOffset2 - keyOffset1)) -
                        constInfo.sparseBlockSize) *
                       constInfo.headDim * sizeof(KV_T);
    }
    if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0 || realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
                 realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit)) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        // 因为需要拷贝两块
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
        if constexpr (KV_LAYOUT_T == SAS_LAYOUT::PA_ND) {
            DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * INPUT2_BUFFER_OFFSET / sizeof(KV_T) +
                                  (mte2Size - mte3Size) * constInfo.headDim],
                        cmpKvGm_[startGmOffset], intriParams, padParams);
        } else {
            DataCopyPad(kvMergUb_[mergeMte3Idx % 2 * INPUT2_BUFFER_OFFSET / sizeof(KV_T) +
                                  (mte2Size - mte3Size) * constInfo.headDim],
                        cmpKvGm_[startGmOffset * constInfo.headDim], intriParams, padParams);
        }
        mte2Size += ((keyOffset1 > -1) + (keyOffset2 > -1)) * constInfo.sparseBlockSize;
    }
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::CopyOutMrgeResult(
    int64_t mte2Size, int64_t mte3Size, int64_t s2GmStartOffset, int64_t mergeMte3Idx, const RunInfo &runInfo)
{
    if (mte2Size <= mte3Size) {
        return;
    }
    if (constInfo.kvQuantMode == 3) {
        SetFlag<AscendC::HardEvent::MTE2_V>(0);
        WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        LocalTensor<uint8_t> slots = tmpBuff1.Get<uint8_t>()[TQ4_RAW_OFFSET + mergeMte3Idx % 2 * TQ4_RAW_REGION_BYTES];
        LocalTensor<KV_T> output = kvMergUb_[mergeMte3Idx % 2 * INPUT2_BUFFER_OFFSET / sizeof(KV_T)];
        DequantTq4Rows(slots, output, mte2Size - mte3Size);
        SetFlag<AscendC::HardEvent::V_MTE3>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
    } else {
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
    }

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = mte2Size - mte3Size;
    dataCopyParams.blockLen = constInfo.headDim * sizeof(KV_T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    DataCopyPad(kvMergeGm_[runInfo.cmpLoop % MERGE_CACHE_GM_BUF_NUM * 512 * 512 +
                           (s2GmStartOffset + mte3Size) * constInfo.headDim],
                kvMergUb_[mergeMte3Idx % 2 * INPUT2_BUFFER_OFFSET / sizeof(KV_T)], dataCopyParams);
}

// b s1 k
template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::ProcessVec0L(const RunInfo &runInfo)
{
    int64_t s2ProcessSize = runInfo.v0S2DealSize;
    int64_t s2Pair = CeilDiv(s2ProcessSize, 2 * constInfo.sparseBlockSize);
    int64_t topkGmBaseOffset = runInfo.topKBaseOffset + runInfo.v0S2Start;
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    int64_t s2IdxArray0 = -1;
    int64_t s2IdxArray1 = -1;
    bool needWaitMte3ToMte2 = true;
    int64_t s2SplitPoint = SASAlign(s2Pair, 2) * constInfo.sparseBlockSize;
    int64_t s2GmStartOffset = GetSubBlockIdx() == 0 ? 0 : s2SplitPoint;
    int64_t s2GmLimit = GetSubBlockIdx() == 0 ? s2SplitPoint : s2ProcessSize;
    if (s2GmLimit > s2ProcessSize) {
        s2GmLimit = s2ProcessSize;
    }
    LocalTensor<int32_t> topkCache = v0ValidSizeBuff.Get<int32_t>();
    LocalTensor<int32_t> blockTableCache =
        v0ValidSizeBuff.Get<int32_t>()[TQ4_BLOCK_TABLE_CACHE_OFFSET / sizeof(int32_t)];
    int64_t topkCacheStart = 0;
    uint32_t topkCacheSize = 0;
    uint32_t blockTableCacheSize = 0;
    bool hasIndexPrefetch = false;
    if (constInfo.kvQuantMode == 3) {
        // Prefetch only a complete required range. A zero cache size keeps the original GM scalar fallback.
        topkCacheStart = s2GmStartOffset / constInfo.sparseBlockSize;
        int64_t topkCacheEnd = CeilDiv(s2GmLimit, constInfo.sparseBlockSize);
        if (topkCacheEnd > static_cast<int64_t>(constInfo.sparseBlockCount)) {
            topkCacheEnd = constInfo.sparseBlockCount;
        }
        int64_t requiredTopkCount = topkCacheEnd - topkCacheStart;
        if (requiredTopkCount > 0 && requiredTopkCount <= TQ4_TOPK_CACHE_CAPACITY) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(requiredTopkCount * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> padParams;
            DataCopyPad(topkCache, topkGm_[topkGmBaseOffset + topkCacheStart], copyParams, padParams);
            topkCacheSize = static_cast<uint32_t>(requiredTopkCount);
            hasIndexPrefetch = true;
        }
        if constexpr (KV_LAYOUT_T == SAS_LAYOUT::PA_ND) {
            int64_t requiredBlockTableCount =
                runInfo.cmpS2IdLimit > 0 ?
                    CeilDiv(runInfo.cmpS2IdLimit, static_cast<int64_t>(constInfo.paCmpBlockSize)) :
                    0;
            if (requiredBlockTableCount > 0 &&
                requiredBlockTableCount <= static_cast<int64_t>(constInfo.cmpMaxBlockNumPerBatch) &&
                requiredBlockTableCount <= TQ4_BLOCK_TABLE_CACHE_CAPACITY) {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(requiredBlockTableCount * sizeof(int32_t)), 0, 0,
                                             0};
                DataCopyPadExtParams<int32_t> padParams;
                DataCopyPad(blockTableCache, cmpBlockTableGm_[runInfo.bIdx * constInfo.cmpMaxBlockNumPerBatch],
                            copyParams, padParams);
                blockTableCacheSize = static_cast<uint32_t>(requiredBlockTableCount);
                hasIndexPrefetch = true;
            }
        }
        if (hasIndexPrefetch) {
            SetFlag<HardEvent::MTE2_S>(0);
            WaitFlag<HardEvent::MTE2_S>(0);
        }
    }
    // 处理两个基本块
    for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit;
         s2GmOffsetArray += 2 * constInfo.sparseBlockSize) {
        if (needWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
            needWaitMte3ToMte2 = false;
        }
        GetRealS2Idx(s2GmOffsetArray, s2IdxArray0, topkGmBaseOffset, runInfo, topkCache, topkCacheStart, topkCacheSize);
        if (unlikely(s2IdxArray0 < 0)) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo);
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
            mergeMte3Idx++;
            break;
        }
        GetRealS2Idx(s2GmOffsetArray + constInfo.sparseBlockSize, s2IdxArray1, topkGmBaseOffset, runInfo, topkCache,
                     topkCacheStart, topkCacheSize);
        CopyInKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1, runInfo, blockTableCache,
                 blockTableCacheSize);
        if ((mte2Size - mte3Size + 2 * constInfo.sparseBlockSize > TQ4_MAX_STAGED_ROWS) ||
            s2GmOffsetArray + 2 * constInfo.sparseBlockSize >= s2GmLimit) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo);
            mte3Size = mte2Size;
            SetFlag<AscendC::HardEvent::MTE3_MTE2>(mergeMte3Idx % 2 + SYNC_INPUT_BUF2_FLAG);
            mergeMte3Idx++;
            needWaitMte3ToMte2 = true;
        }
    }
    return;
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::ProcessVec1L(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

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
        CrossCoreSetFlag<ConstInfo::SAS_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C2);

        // move lse for flash decode or FA
        if (constInfo.returnSoftmaxLse && info.s2Idx == info.curSInnerLoopTimes - 1) {
            ProcessLse(info, mSplitInfo);
        }
    }
}

template <typename SAST>
__aicore__ inline uint64_t TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::CalcAccumOffset(uint32_t bN2Idx,
                                                                                              uint32_t gS1Idx)
{
    return 0;
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::ProcessVec2SingleBuf(
    const RunInfo &info, const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }

    ProcessVec2Inner(info, mSplitInfo, 0, mSplitInfo.vecDealM);
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::ProcessVec2L(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

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

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::ProcessVec2Inner(const RunInfo &info,
                                                                                           const MSplitInfo &mSplitInfo,
                                                                                           uint32_t mStartRow,
                                                                                           uint32_t mDealSize)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDim;
    if (mSplitSize > mDealSize) {
        mSplitSize = mDealSize;
    }

    uint32_t loopCount = (mDealSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mDealSize - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(info, mSplitInfo, i * mSplitSize + mStartRow, dealSize, constInfo.headDim,
                             constInfo.headDim);
    }
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::Bmm2FDDataCopyOut(
    const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
    uint32_t actualColumnCount)
{
    LocalTensor<T> tmp = outputBuff1.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(tmp, bmm2ResUb, columnCount * dealRowCount);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    uint64_t accumTmpOutNum = CalcAccumOffset(info.bIdx, info.gS1Idx);
    uint64_t offset =
        accumTmpOutNum * constInfo.kvHeadNum * constInfo.mBaseSize * constInfo.headDim +              // taskoffset
        info.tndCoreStartKVSplitPos * constInfo.kvHeadNum * constInfo.mBaseSize * constInfo.headDim + // 份数offset
        wsMStart * actualColumnCount;                                                                 // m轴offset
    GlobalTensor<T> dst = accumOutGm[offset];
    if (info.actualSingleProcessSInnerSize == 0) {
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

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::Bmm2DataCopyOutTrans(
    const RunInfo &info, LocalTensor<OUT_T> &attenOutUb, uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
    uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[info.attenOutOffset + wsMStart * actualColumnCount], attenOutUb, dataCopyParams);
    return;
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::Bmm2CastAndCopyOut(
    const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
    uint32_t actualColumnCount)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputBuff1.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    } else {
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    Bmm2DataCopyOutTrans(info, tmpBmm2ResCastTensor, wsMStart, dealRowCount, columnCount, actualColumnCount);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::Bmm2ResCopyOut(
    const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
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

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::DealBmm2ResBaseBlock(
    const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
    uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t mStart = mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
    uint64_t srcGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + mStart * columnCount;
    LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputBuff1.Get<MM2_OUT_T>();
    tmpBmm2ResUb = tmpBmm2ResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset], vec2ComputeSize);

    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    bmm2ResUb.SetSize(vec2ComputeSize);
    DataCopy(bmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
    pingpongFlag ^= 1; // pingpong 0 1切换

    uint32_t inOutBaseOffset = mStart * columnCount;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;

    // 除第一个循环外，均需要更新中间计算结果
    if (!info.isFirstSInnerLoop) {
        event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);

        LocalTensor<MM2_OUT_T> bmm2ResPreUb = inputBuff1.Get<MM2_OUT_T>();
        bmm2ResPreUb = bmm2ResPreUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

        uint64_t vec2ResGmOffset = ((info.loop - 1) % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
        DataCopy(bmm2ResPreUb, vec2ResGm[vec2ResGmOffset], vec2ComputeSize);

        SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

        uint32_t idx = info.loop % (constInfo.preLoadNum);
        LocalTensor<T> expUb = v0ValidSizeBuff.Get<T>()[V0_VEC2_TEMP_OFFSET / sizeof(T)];
        Brcb(expUb, softmaxExpUb[idx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset], (dealRowCount + 7) / 8,
             {1, 8});
        PipeBarrier<PIPE_V>();

        RowMuls(bmm2ResPreUb, bmm2ResPreUb, expUb, dealRowCount, columnCount, actualColumnCount);
        AscendC::PipeBarrier<PIPE_V>();
        Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
        AscendC::PipeBarrier<PIPE_V>();

        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);
        pingpongFlag ^= 1; // pingpong 0 1 切换
    }

    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.isLastS2Loop) {
        uint32_t idx = info.loop % (constInfo.preLoadNum);
        LocalTensor<T> tmpSumUb = v0ValidSizeBuff.Get<T>()[V0_VEC2_TEMP_OFFSET / sizeof(T)];
        Brcb(tmpSumUb, softmaxSumUb[idx * SOFTMAX_TMP_BUFFER_OFFSET / sizeof(T) + baseOffset], (dealRowCount + 7) / 8,
             {1, 8});
        PipeBarrier<PIPE_V>();
        RowDivs(bmm2ResUb, bmm2ResUb, tmpSumUb, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
        Bmm2ResCopyOut(info, bmm2ResUb, mStart, dealRowCount, columnCount, actualColumnCount);
    } else {
        LocalTensor<T> outUb = outputBuff1.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        DataCopy(outUb, bmm2ResUb, dealRowCount * columnCount);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        uint64_t vec2ResGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
        DataCopy(vec2ResGm[vec2ResGmOffset], outUb, vec2ComputeSize);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    }
}

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::RowDivs(
    LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount)
{
    // divs by row, 每行的元素除以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
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
        columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
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

template <typename SAST>
__aicore__ inline void TurboQuantSparseAttnSharedkvScfaBlockVector<SAST>::RowMuls(
    LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub, uint32_t dealRowCount, uint32_t columnCount,
    uint32_t actualColumnCount)
{
    // muls by row, 每行的元素乘以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount] src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    // dealRowCount is repeat times, must be less 256
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;   // 32/4 * 2 = 16
    }

    // 每次只能连续读取256B的数据进行计算，故每次只能处理256B/sizeof(dType)=
    // 列方向分dLoop次，每次处理8列数据
    uint32_t dLoop = actualColumnCount / repeatElementNum;
    uint32_t dRemain = actualColumnCount % repeatElementNum;
    // REPEATE_STRIDE_UP_BOUND=256， 此限制由于src0RepStride数据类型为uint8之多256个datablock间距
    if (columnCount < REPEATE_STRIDE_UP_BOUND * blockElementNum) {
        BinaryRepeatParams repeatParams;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0RepStride = columnCount / blockElementNum;
        repeatParams.src1RepStride = 1;
        repeatParams.dstRepStride = columnCount / blockElementNum;

        // 如果以列为repeat所处理的次数小于行处理次数，则以列方式处理。反之则以行进行repeat处理
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
            columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            columnRepeatParams.src1RepStride = 0;
            columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum,
                    dLoop, columnRepeatParams);
            }
        }

        // 最后一次完成[dealRowCount, dRemain] * [dealRowCount, blockElementNum] 只计算有效部分
        if (dRemain > 0) {
            Mul(dstUb[dLoop * repeatElementNum], src0Ub[dLoop * repeatElementNum], src1Ub, dRemain, dealRowCount,
                repeatParams);
        }
    } else {
        BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = 8; // 每个repeat为256B数据，正好8个datablock
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = 8;
        repeatParams.dstBlkStride = 1;
        // 每次计算一行，共计算dealRowCount行
        for (uint32_t i = 0; i < dealRowCount; i++) {
            // 计算一行中的dLoop个repeat, 每个repeat计算256/block_size 个data_block
            Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum, dLoop,
                repeatParams);
            //  计算一行中的尾块
            if (dRemain > 0) {
                Mul(dstUb[i * columnCount + dLoop * repeatElementNum],
                    src0Ub[i * columnCount + dLoop * repeatElementNum], src1Ub[i * blockElementNum], dRemain, 1,
                    repeatParams);
            }
        }
    }
}
} // namespace SASKernel
#endif // TURBO_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_VECTOR_H
