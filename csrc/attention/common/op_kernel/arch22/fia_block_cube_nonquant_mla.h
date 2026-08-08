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
 * \file fia_block_cube_nonquant_mla.h
 * \brief use 7 buffer for matmul l1, better pipeline
 */
#ifndef FIA_BLOCK_CUBE_NONQUANT_MLA_H
#define FIA_BLOCK_CUBE_NONQUANT_MLA_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../fia_public_define.h"
#include "../matmul.h"
#include "../memory_copy.h"

using namespace fa_base_matmul;
using namespace AttentionCommon;

template <typename FIAT> 
class FiaBlockCubeNonQuantMla {
public:
    // 中间计算数据类型为float, 高精度模式
    using T = float;

    using Q_T = typename FIAT::queryType;
    using KV_T = typename FIAT::kvType;
    using OUT_T = typename FIAT::outputType;
    static constexpr bool PAGE_ATTENTION = FIAT::pageAttention;
    static constexpr FIA_LAYOUT LAYOUT_T = FIAT::layout;
    static constexpr FIA_LAYOUT KV_LAYOUT_T = FIAT::kvLayout;

    static constexpr bool KVINT4 = IsSameType<KV_T, int4b_t>::value;
    using MM_OUT_T = T;
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    static constexpr GmFormat KV_FORMAT = GetKVFormat<KV_LAYOUT_T, PAGE_ATTENTION>();

    __aicore__ inline FiaBlockCubeNonQuantMla(){};
    __aicore__ inline void InitParams(const AttentionCommon::ConstInfo &constInfo);
    __aicore__ inline void Init(
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
        __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
        __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
        __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
        __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
        __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
        __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
        __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse);
    __aicore__ inline void InitMm1GlobalTensor(GlobalTensor<MM_OUT_T> mm1ResGm);
    __aicore__ inline void InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm,
                                               GlobalTensor<MM_OUT_T> mm2ResGm);
    __aicore__ inline void InitBuffers(TPipe *pipe);

    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const AttentionCommon::RunInfo &info);
    __aicore__ inline void ComputeMm2(const AttentionCommon::RunInfo &info);

protected:
    template <typename T> __aicore__ inline size_t BlockAlign(size_t s)
    {
        if constexpr (IsSameType<T, int4b_t>::value) {
            return (s + 63) / 64 * 64;
        }
        size_t n = (32 / sizeof(T));
        return (s + n - 1) / n * n;
    }

    __aicore__ inline uint32_t GetQPL1RealIdx(uint32_t mIdx, uint32_t k1Idx)
    {
        uint32_t idxMap[] = {0, 2}; // 确保0块和1块连在一起, 2和3块连在一起, 来保证同一m块的地址相连
        return idxMap[mIdx % 2] + k1Idx;
    }

    __aicore__ inline void InitValueGm(uint32_t bIdx);
    __aicore__ inline void InitKeyGm(uint32_t bIdx);
    __aicore__ inline void ProcessMm1(const AttentionCommon::RunInfo &info, const MSplitInfo mSplitInfo);
    __aicore__ inline void ProcessMm2(const AttentionCommon::RunInfo &info, const MSplitInfo mSplitInfo);
    __aicore__ inline void UpdateKey(GlobalTensor<KV_T> keyGm);
    __aicore__ inline void UpdateValue(GlobalTensor<KV_T> valueGm);
    __aicore__ inline void CopyGmToL1(LocalTensor<KV_T> &l1Tensor, GlobalTensor<KV_T> &gmSrcTensor, uint32_t srcN,
                                      uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyInMm2AToL1(LocalTensor<KV_T> &aL1Tensor, const AttentionCommon::RunInfo &info, uint32_t mSeqIdx,
                                          uint32_t subMSizeAct, uint32_t nSize, uint32_t nOffset);

protected:
    // key和value的TensorList原始地址
    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;
    // mm1
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<Q_T> qRopeGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> kRopeGm;
    GlobalTensor<MM_OUT_T> mm1ResGm;

    // mm2
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<MM_OUT_T> mm2ResGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<int32_t> mm2ResInt32Gm;

    // pageAttention
    uint32_t maxBlockNumPerBatch = 0;
    // block_table
    GlobalTensor<int32_t> blockTableGm;

    AttentionCommon::ConstInfo constInfo{};

    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGm;
private:
    FaGmTensor<Q_T, Q_FORMAT> queryGmTensor;
    FaGmTensor<Q_T, Q_FORMAT> queryRopeGmTensor;
    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1;

    FaGmTensor<KV_T, KV_FORMAT> keyGmTensor;
    FaGmTensor<KV_T, KV_FORMAT> keyRopeGmTensor;
    FaGmTensor<KV_T, KV_FORMAT> valueGmTensor;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1;
    CopyKKropePAGmToL1<KV_T, KV_FORMAT> copyKKropePAGmToL1;

    static constexpr uint32_t M_L1_SPLIT_SIZE = 128; // m方向切分
    static constexpr uint32_t N_L1_SPLIT_SIZE = 128; // n方向切分
    static constexpr uint32_t MM1_K_L0_SPLIT_SIZE = 96;
    static constexpr uint32_t MM1_K_L1_SPLIT_SIZE = (512 + 64) / 2; // 288
    static constexpr uint32_t MM2_K_L0_SPLIT_SIZE = 128;

    // L1 buffer size
    static constexpr uint32_t L1_BLOCK_SIZE = M_L1_SPLIT_SIZE * MM1_K_L1_SPLIT_SIZE * sizeof(Q_T);
    static constexpr uint32_t L1_BLOCK_OFFSET = M_L1_SPLIT_SIZE * MM1_K_L1_SPLIT_SIZE; // 72K的元素个数

    // L0 buffer size
    static constexpr uint32_t L0A_BLOCK_SIZE = (32 * 1024);
    static constexpr uint32_t L0B_BLOCK_SIZE = (32 * 1024);
    static constexpr uint32_t L0C_BLOCK_SIZE = (64 * 1024);

    // mte2 <> mte1 EventID
    // L1 3buf, 使用3个eventId
    static constexpr uint32_t L1_EVENT0 = EVENT_ID2;
    static constexpr uint32_t L1_EVENT1 = EVENT_ID3;
    static constexpr uint32_t L1_EVENT2 = EVENT_ID4;
    static constexpr uint32_t L1_EVENT3 = EVENT_ID5;
    static constexpr uint32_t L1_EVENT4 = EVENT_ID6;
    static constexpr uint32_t L1_EVENT5 = EVENT_ID7;
    static constexpr uint32_t L1_EVENT6 = EVENT_ID1;

    static constexpr uint32_t mte21QPIds[4] = {L1_EVENT0, L1_EVENT1, L1_EVENT2, L1_EVENT3}; // mte12复用
    static constexpr uint32_t mte21KVIds[3] = {L1_EVENT4, L1_EVENT5, L1_EVENT6};

#ifdef BASE_MM
    fa_base_matmul::BufferManager<fa_base_matmul::BufferType::L1> l1BufferManager;
    fa_base_matmul::BuffersPolicy3buff<fa_base_matmul::BufferType::L1> l1KV3Buffers;
    fa_base_matmul::Matrix2x2BufferPolicy<fa_base_matmul::BufferType::L1> l1QP4Buffers;
#endif

    fa_base_matmul::BufferManager<fa_base_matmul::BufferType::L0A> l0aBufferManager;
    fa_base_matmul::BufferManager<fa_base_matmul::BufferType::L0B> l0bBufferManager;
    // L0A
    fa_base_matmul::BuffersPolicyDB<fa_base_matmul::BufferType::L0A> mmL0ABuffers;
    // L0B
    fa_base_matmul::BuffersPolicyDB<fa_base_matmul::BufferType::L0B> mmL0BBuffers;

    // L1
    TBuf<TPosition::A1> bufQPL1;
    LocalTensor<Q_T> l1QPTensor;

    TBuf<TPosition::A1> bufKVL1;
    LocalTensor<Q_T> l1KVTensor;

    // L0_C
    TBuf<TPosition::CO1> tmpBufL0C;
    LocalTensor<MM_OUT_T> cL0TensorPingPong;

    // L1分成3块buf, 用于记录
    uint32_t qpL1BufIter = 0;
    uint32_t kvL1BufIter = 0;
    uint32_t abL0BufIter = 0;
    uint32_t cL0BufIter = 0;
};

template <typename FIAT> 
class FiaBlockCubeNonQuantMlaDummy {
public:
    using T = float;
    using KV_T = typename FIAT::kvType;
    using MM_OUT_T = T;

    __aicore__ inline FiaBlockCubeNonQuantMlaDummy(){};
    __aicore__ inline void InitParams(const AttentionCommon::ConstInfo &constInfo);
    __aicore__ inline void Init(
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
        __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
        __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
        __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
        __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
        __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
        __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
        __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse);
    __aicore__ inline void InitMm1GlobalTensor(GlobalTensor<MM_OUT_T> mm1ResGm);
    __aicore__ inline void InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm,
                                               GlobalTensor<MM_OUT_T> mm2ResGm);
    __aicore__ inline void InitBuffers(TPipe *pipe);

    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const AttentionCommon::RunInfo &info);
    __aicore__ inline void ComputeMm2(const AttentionCommon::RunInfo &info);
};

template <typename FIAT> __aicore__ inline void FiaBlockCubeNonQuantMla<FIAT>::InitParams(const ConstInfo &constInfo)
{
    this->constInfo = constInfo;
}

template <typename FIAT> __aicore__ inline void FiaBlockCubeNonQuantMla<FIAT>::Init(
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
        __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
        __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
        __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
        __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
        __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
        __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
        __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse)
{
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    qRopeGm.SetGlobalBuffer((__gm__ Q_T *)queryRope);
    kRopeGm.SetGlobalBuffer((__gm__ KV_T *)keyRope);

    keyPtr = key;
    valuePtr = value;

    if constexpr (PAGE_ATTENTION) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        this->blockTableGm = blockTableGm;
    }

    if (constInfo.actualLenQDims != 0) {
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
    }

    if (constInfo.actualLenDims != 0) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, constInfo.actualLenDims);
    }

    // init Q params
    uint32_t qkTensorD = constInfo.ropeSplitMode ? constInfo.headDim : (constInfo.headDim + constInfo.headDimRope);
    queryGmTensor.gmTensor = queryGm;
    if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
        queryGmTensor.offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.gSize,
                                            constInfo.qSeqSize, qkTensorD, actualSeqLengthsGmQ,
                                            constInfo.actualLenQDims);
        queryRopeGmTensor.offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.gSize,
                                            constInfo.qSeqSize, constInfo.headDimRope, actualSeqLengthsGmQ,
                                            constInfo.actualLenQDims);
    } else if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
        queryGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.gSize, qkTensorD, actualSeqLengthsGmQ,
                                            constInfo.actualLenQDims);
        queryRopeGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.gSize, constInfo.headDimRope,
                                                actualSeqLengthsGmQ, constInfo.actualLenQDims);
    }

    // init kv params
    if (constInfo.batchContinuous) {
        InitKeyGm(0);
        InitValueGm(0);
        {
            keyGmTensor.gmTensor = keyGm;
            valueGmTensor.gmTensor = valueGm;
            if constexpr (PAGE_ATTENTION) {
                if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
                    keyGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.kvCacheBlockSize, qkTensorD,
                                                      blockTableGm, constInfo.maxBlockNumPerBatch,
                                                      constInfo.keyBnStride, constInfo.keyN2Stride);
                } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
                    uint32_t d0 = 32 / sizeof(KV_T);
                    uint32_t d1 = qkTensorD / d0;
                    keyGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.kvCacheBlockSize, d1, d0,
                                                      blockTableGm, constInfo.maxBlockNumPerBatch,
                                                      constInfo.keyBnStride, constInfo.keyN2Stride);
                }
            } else {
                if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
                    keyGmTensor.offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.kvSeqSize,
                                                      qkTensorD, actualSeqLengthsGm, constInfo.actualLenDims);
                } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
                    keyGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, qkTensorD, actualSeqLengthsGm,
                                                      constInfo.actualLenDims);
                }
            }
        }
        valueGmTensor.gmTensor = valueGm;
        if constexpr (PAGE_ATTENTION) {
            if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
                valueGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.kvCacheBlockSize,
                                                    constInfo.headDim, blockTableGm, constInfo.maxBlockNumPerBatch,
                                                    constInfo.valueBnStride, constInfo.valueN2Stride);
            } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
                uint32_t d0 = 32 / sizeof(KV_T);
                uint32_t d1 = constInfo.headDim / d0;
                valueGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.kvCacheBlockSize, d1, d0,
                                                    blockTableGm, constInfo.maxBlockNumPerBatch,
                                                    constInfo.valueBnStride, constInfo.valueN2Stride);
            }
        } else {
            if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
                valueGmTensor.offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.kvSeqSize,
                                                    constInfo.headDim, actualSeqLengthsGm, constInfo.actualLenDims);
            } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
                valueGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.headDim, actualSeqLengthsGm,
                                                    constInfo.actualLenDims);
            }
        }
    }

    // init rope params
        if (constInfo.ropeSplitMode) {
        // query rope
            queryRopeGmTensor.gmTensor = qRopeGm;
            if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
                queryRopeGmTensor.offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.gSize,
                                                        constInfo.qSeqSize, constInfo.headDimRope, actualSeqLengthsGmQ,
                                                        constInfo.actualLenQDims);
            } else if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
                queryRopeGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.gSize, constInfo.headDimRope,
                                                        actualSeqLengthsGmQ, constInfo.actualLenQDims);
            }

        // key rope
        keyRopeGmTensor.gmTensor = kRopeGm;
        if constexpr (PAGE_ATTENTION) {
            if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
                keyRopeGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.kvCacheBlockSize,
                                                      constInfo.headDimRope, blockTableGm,
                                                      constInfo.maxBlockNumPerBatch,
                                                      constInfo.keyRopeBnStride, constInfo.keyRopeN2Stride);
            } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
                uint32_t d0 = 32 / sizeof(KV_T);
                uint32_t d1 = constInfo.headDimRope / d0;
                keyRopeGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.kvCacheBlockSize, d1, d0,
                                                      blockTableGm, constInfo.maxBlockNumPerBatch,
                                                      constInfo.keyRopeBnStride, constInfo.keyRopeN2Stride);
            }
        } else {
            if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
                keyRopeGmTensor.offsetCalculator.Init(constInfo.batchSize, constInfo.kvHeadNum, constInfo.kvSeqSize,
                                                      constInfo.headDimRope, actualSeqLengthsGm, constInfo.actualLenDims);
            } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
                keyRopeGmTensor.offsetCalculator.Init(constInfo.kvHeadNum, constInfo.headDimRope,
                                                      actualSeqLengthsGm, constInfo.actualLenDims);
            }
        }
    }
}

template <typename FIAT>
__aicore__ inline void
FiaBlockCubeNonQuantMla<FIAT>::InitMm1GlobalTensor(GlobalTensor<MM_OUT_T> mm1ResGm)
{
    // mm1
    this->mm1ResGm = mm1ResGm;
}

template <typename FIAT>
__aicore__ inline void
FiaBlockCubeNonQuantMla<FIAT>::InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm, GlobalTensor<MM_OUT_T> mm2ResGm)
{
    // mm2
    this->vec1ResGm = vec1ResGm;
    this->mm2ResGm = mm2ResGm;
}

template <typename FIAT> __aicore__ inline void FiaBlockCubeNonQuantMla<FIAT>::InitBuffers(TPipe *pipe)
{
#ifdef BASE_MM
    l1BufferManager.Init(pipe, 524288);                // L1 total size
    l1KV3Buffers.Init(l1BufferManager, L1_BLOCK_SIZE); // 3buf KV复用
    l1QP4Buffers.Init(l1BufferManager, L1_BLOCK_SIZE); // 4buf QP复用
#else
    pipe->InitBuffer(bufQPL1, L1_BLOCK_SIZE * 4); // (64K + 8K) * 4
    l1QPTensor = bufQPL1.Get<Q_T>();

    pipe->InitBuffer(bufKVL1, L1_BLOCK_SIZE * 3); // (64K + 8K) * 3
    l1KVTensor = bufKVL1.Get<KV_T>();
#endif

    l0aBufferManager.Init(pipe, L0A_BLOCK_SIZE * 2);
    l0bBufferManager.Init(pipe, L0B_BLOCK_SIZE * 2);

    mmL0ABuffers.Init(l0aBufferManager, L0A_BLOCK_SIZE); // buffer + event
    mmL0BBuffers.Init(l0bBufferManager, L0B_BLOCK_SIZE);

    // L0C
    pipe->InitBuffer(tmpBufL0C, L0C_BLOCK_SIZE * 2); // 128K
    cL0TensorPingPong = tmpBufL0C.Get<MM_OUT_T>();
}

template <typename FIAT> __aicore__ inline void FiaBlockCubeNonQuantMla<FIAT>::InitKeyGm(uint32_t bIdx)
{
    ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
    __gm__ uint8_t *key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

    keyGm.SetGlobalBuffer((__gm__ KV_T *)key_);
}
 
template <typename FIAT> __aicore__ inline void FiaBlockCubeNonQuantMla<FIAT>::InitValueGm(uint32_t bIdx)
{
    ListTensorDesc valueListTensorDesc((__gm__ void *)valuePtr);
    __gm__ uint8_t *value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

    valueGm.SetGlobalBuffer((__gm__ KV_T *)value_);
}

template <typename FIAT> __aicore__ inline void FiaBlockCubeNonQuantMla<FIAT>::UpdateKey(GlobalTensor<KV_T> keyGm)
{
    this->keyGm = keyGm;
}

template <typename FIAT> __aicore__ inline void FiaBlockCubeNonQuantMla<FIAT>::UpdateValue(GlobalTensor<KV_T> valueGm)
{
    this->valueGm = valueGm;
}

template <typename FIAT> __aicore__ inline void FiaBlockCubeNonQuantMla<FIAT>::AllocEventID()
{
    CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V1);
#ifdef BASE_MM
#else
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT2);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT3);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT4);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT5);
    SetFlag<HardEvent::MTE1_MTE2>(L1_EVENT6);
#endif
}

template <typename FIAT> __aicore__ inline void FiaBlockCubeNonQuantMla<FIAT>::FreeEventID()
{
#ifdef BASE_MM
    l1KV3Buffers.Uninit(l1BufferManager);
    l1QP4Buffers.Uninit(l1BufferManager);
#else
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT2);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT3);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT4);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT5);
    WaitFlag<HardEvent::MTE1_MTE2>(L1_EVENT6);
#endif

    mmL0ABuffers.Uninit(l0aBufferManager);
    mmL0BBuffers.Uninit(l0bBufferManager);
}

template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuantMla<FIAT>::CopyGmToL1(LocalTensor<KV_T> &l1Tensor,
                                                                 GlobalTensor<KV_T> &gmSrcTensor, uint32_t srcN,
                                                                 uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN; // 行数
    if constexpr (KVINT4) {
        nd2nzPara.dValue = srcD / 2;
        nd2nzPara.srcDValue = srcDstride / 2;
    } else {
        nd2nzPara.dValue = srcD;
        nd2nzPara.srcDValue = srcDstride;
    }
    nd2nzPara.dstNzC0Stride = (srcN + 15) / 16 * 16; // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuantMla<FIAT>::CopyInMm2AToL1(LocalTensor<KV_T> &aL1Tensor, const AttentionCommon::RunInfo &info,
                                                                     uint32_t mSeqIdx, uint32_t subMSizeAct,
                                                                     uint32_t nSize, uint32_t nOffset)
{
    // 全量拷贝 确认是否是紧密排布, actualSingleProcessSInnerSize是否需要32B对齐
    auto srcGm = vec1ResGm[(info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                           mSeqIdx * info.actualSingleProcessSInnerSizeAlign + nOffset];
    CopyGmToL1(aL1Tensor, srcGm, subMSizeAct, nSize, info.actualSingleProcessSInnerSizeAlign);
}

template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuantMla<FIAT>::ComputeMm1(const RunInfo &info)
{
    if (info.isChangeBatch) {
        InitKeyGm(info.bIdx);
        UpdateKey(keyGm);
    }
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;
        ProcessMm1(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuantMla<FIAT>::ComputeMm2(const RunInfo &info)
{
    if (info.isChangeBatch) {
        InitValueGm(info.bIdx);
        UpdateValue(valueGm);
    }
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;
        CrossCoreWaitFlag(constInfo.syncV1C2);
        ProcessMm2(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V2);
        CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V1);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuantMla<FIAT>::ProcessMm1(const AttentionCommon::RunInfo &info, const MSplitInfo mSplitInfo)
{
    // 最外层还需要一层m的循环
    uint32_t mSize = mSplitInfo.nBufferDealM;
    uint32_t mL1Size = M_L1_SPLIT_SIZE;
    uint32_t mL1SizeAlign = AttentionCommon::Align(M_L1_SPLIT_SIZE, 16U);
    uint32_t mL1Loops = (mSize + M_L1_SPLIT_SIZE - 1) / M_L1_SPLIT_SIZE;

    uint32_t nSize = info.actualSingleProcessSInnerSize;
    uint32_t nL1Size = N_L1_SPLIT_SIZE;
    uint32_t nL1SizeAlign = AttentionCommon::Align(N_L1_SPLIT_SIZE, 16U);
    uint32_t nL1Loops = (nSize + N_L1_SPLIT_SIZE - 1) / N_L1_SPLIT_SIZE;

    uint32_t kSize = 576;
    uint32_t kL1Size = MM1_K_L1_SPLIT_SIZE; // 288
    uint32_t kL1Loops = 2; // 2 : 576/288, mla专用 这里不考虑d泛化

    uint32_t kL0Size = 96;
    uint32_t kL0Loops = (kL1Size + kL0Size - 1) / kL0Size; // 288 / 96 = 3 kloops

    LocalTensor<KV_T> bL1Tensor;
    LocalTensor<KV_T> kRopeTensor;
#ifdef BASE_MM
    fa_base_matmul::Buffer<fa_base_matmul::BufferType::L1> mm1B;
    fa_base_matmul::Buffer<fa_base_matmul::BufferType::L1> mm1A;
    l1QP4Buffers.SetMExtent(mL1Loops);
#endif
    // ka表示左矩阵4buf选择哪一块buf, kb表示右矩阵3buf选择哪一块buf
    uint32_t ka = 0, kb = 0;

    // L1 切n切k切m
    for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) { // L1切n, 512/128
        if (nL1 == (nL1Loops - 1)) {
            // 尾块重新计算size
            nL1Size = nSize - (nL1Loops - 1) * N_L1_SPLIT_SIZE;
            nL1SizeAlign = AttentionCommon::Align(nL1Size, 16U);
        }

        for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) { // L1切k, 576/288, 这里不考虑d泛化
#ifdef BASE_MM
            mm1B = l1KV3Buffers.Get();
            mm1B.Wait<HardEvent::MTE1_MTE2>();
            bL1Tensor = mm1B.GetTensor<KV_T>();
#else
            uint32_t kb = kvL1BufIter % 3;
            kvL1BufIter++;
            WaitFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]);
            // 从k当中取当前的块
            bL1Tensor = l1KVTensor[kb * L1_BLOCK_OFFSET];
#endif
            if (kL1 == 0) {
                FaL1Tensor<KV_T, L1Format::NZ> dstTensor {
                    .tensor = bL1Tensor,
                    .rowCount = nL1SizeAlign
                };
                GmKvCoord gmCoord {
                    .bIdx = constInfo.batchContinuous ? info.bIdx : 0,
                    .n2Idx = info.n2Idx,
                    .s2Idx = info.s2BatchOffset + nL1 * N_L1_SPLIT_SIZE,
                    .dIdx = kL1 * 256, // D方向上切256
                    .s2DealSize = nL1Size,
                    .dDealSize = 256 // D方向上切256
                };
                FaL1Tensor<KV_T, L1Format::NZ> dstRopeTensor {
                    .tensor = bL1Tensor[nL1SizeAlign * (BlockAlign<KV_T>(constInfo.headDim) >> 1)],
                    .rowCount = nL1SizeAlign
                };
                GmKvCoord gmCoordKRope {
                    .bIdx = constInfo.batchContinuous ? info.bIdx : 0,
                    .n2Idx = info.n2Idx,
                    .s2Idx = info.s2BatchOffset + nL1 * N_L1_SPLIT_SIZE,
                    .dIdx = kL1 * 32U, // D方向上切32
                    .s2DealSize = nL1Size,
                    .dDealSize = 32U // D方向上切32
                };
                if (PAGE_ATTENTION) {
                    copyKKropePAGmToL1(dstTensor, dstRopeTensor, keyGmTensor, keyRopeGmTensor, gmCoord, gmCoordKRope);
                } else {
                    copyKvGmToL1(dstTensor, keyGmTensor, gmCoord);
                    copyKvGmToL1(dstRopeTensor, keyRopeGmTensor, gmCoordKRope);
                }
            } else {
                FaL1Tensor<KV_T, L1Format::NZ> dstRopeTensor {
                    .tensor = bL1Tensor,
                    .rowCount = nL1SizeAlign
                };
                GmKvCoord gmCoordKRope {
                    .bIdx = constInfo.batchContinuous ? info.bIdx : 0,
                    .n2Idx = info.n2Idx,
                    .s2Idx = info.s2BatchOffset + nL1 * N_L1_SPLIT_SIZE,
                    .dIdx = kL1 * 32U, // D方向上切32
                    .s2DealSize = nL1Size,
                    .dDealSize = 32U // D方向上切32
                };
                FaL1Tensor<KV_T, L1Format::NZ> dstTensor {
                    .tensor = bL1Tensor[32U * nL1SizeAlign],
                    .rowCount = nL1SizeAlign
                };
                GmKvCoord gmCoord {
                    .bIdx = constInfo.batchContinuous ? info.bIdx : 0,
                    .n2Idx = info.n2Idx,
                    .s2Idx = info.s2BatchOffset + nL1 * N_L1_SPLIT_SIZE,
                    .dIdx = kL1 * 256U, // D方向上切256
                    .s2DealSize = nL1Size,
                    .dDealSize = 256U // D方向上切256
                };
                if (PAGE_ATTENTION) {
                    copyKKropePAGmToL1(dstTensor, dstRopeTensor, keyGmTensor, keyRopeGmTensor, gmCoord, gmCoordKRope);
                } else {
                    copyKvGmToL1(dstRopeTensor, keyRopeGmTensor, gmCoordKRope);
                    copyKvGmToL1(dstTensor, keyGmTensor, gmCoord);
                }
            }
#ifdef BASE_MM
            mm1B.Set<HardEvent::MTE2_MTE1>();
            mm1B.Wait<HardEvent::MTE2_MTE1>();
#else
            SetFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
            WaitFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
#endif
            mL1Size = M_L1_SPLIT_SIZE;
            mL1SizeAlign = AttentionCommon::Align(M_L1_SPLIT_SIZE, 16U);
            for (uint32_t mL1 = 0; mL1 < mL1Loops; mL1++) {
                uint32_t aL1PaddingSize = 0; // 用于使左矩阵对齐到尾部, 以保证两块32K内存连续
                if (mL1 == (mL1Loops - 1)) {
                    // 尾块重新计算size
                    mL1Size = mSize - (mL1Loops - 1) * M_L1_SPLIT_SIZE;
                    mL1SizeAlign = AttentionCommon::Align(mL1Size, 16U);
                    aL1PaddingSize = (M_L1_SPLIT_SIZE - mL1SizeAlign) *
                                     288; // mL1SizeAlign<128 kL1=0时需要偏移, 确保qRope能一半拷贝到当前tensor,
                                          // 一半拷贝到下一个tensor
                }

                // 左矩阵L1选择12块还是34块的index, 由m l1 index决定
                // 左矩阵L1选择12块或34块的前一块还是后一块, 由k l1 index决定
                uint32_t mIdx = qpL1BufIter + mL1;
                ka = GetQPL1RealIdx(mIdx, kL1);
#ifdef BASE_MM
                LocalTensor<Q_T> aL1Tensor;
                if (nL1 == 0) {
                    mm1A = l1QP4Buffers.AllocNext();
                    // kL1=0时需要偏移
                    aL1Tensor = mm1A.GetTensor<Q_T>()[(1 - kL1) * aL1PaddingSize];
#else
                LocalTensor<Q_T> aL1Tensor =
                    l1QPTensor[ka * L1_BLOCK_OFFSET + (1 - kL1) * aL1PaddingSize]; // kL1=0时需要偏移
                if (nL1 == 0) { // mL1=0, mL1=1两次
#endif
                    if (kL1 == 0) {
                        FaL1Tensor<Q_T, L1Format::NZ> dstTensor {
                            .tensor = aL1Tensor,
                            .rowCount = AttentionCommon::Align(mL1Size, 16U)
                        };
                        GmCoord gmCoord {
                            .bIdx = info.bIdx,
                            .n2Idx = info.n2Idx,
                            .gS1Idx = info.gS1Idx + mSplitInfo.nBufferStartM + mL1 * M_L1_SPLIT_SIZE,
                            .dIdx = kL1 * 256,
                            .gS1DealSize = mL1Size,
                            .dDealSize = 256
                        };
                        LocalTensor<Q_T> qRopeTensor = aL1Tensor[mL1SizeAlign * 256];
                        FaL1Tensor<Q_T, L1Format::NZ> dstRopeTensor {
                            .tensor = aL1Tensor[mL1SizeAlign * 256],
                            .rowCount = AttentionCommon::Align(mL1Size, 16U)
                        };
                        GmCoord gmCoordQRope {
                            .bIdx = info.bIdx,
                            .n2Idx = info.n2Idx,
                            .gS1Idx = info.gS1Idx + mSplitInfo.nBufferStartM + mL1 * M_L1_SPLIT_SIZE,
                            .dIdx = 0,
                            .gS1DealSize = mL1Size,
                            .dDealSize = 64
                        };
#ifdef BASE_MM
                        auto mm1ANextK = l1QP4Buffers.PeekNextK();
                        mm1A.Wait<HardEvent::MTE1_MTE2>();
                        mm1ANextK.Wait<HardEvent::MTE1_MTE2>(); // 这里通过mm1A越界访问mm1ANext，所以需要提前wait
                        copyQueryGmToL1(dstTensor, queryGmTensor, gmCoord);

                        copyQueryGmToL1(dstRopeTensor, queryRopeGmTensor, gmCoordQRope);
#else
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]);
                        WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka + 1]);
                        copyQueryGmToL1(dstTensor, queryGmTensor, gmCoord);
                        // 由于L1里面是NZ, 这里q rope的偏移为整块q nope切k的后大小, 256为headDim的一半
                        copyQueryGmToL1(dstRopeTensor, queryRopeGmTensor, gmCoordQRope); 
#endif
                    } else {
                        FaL1Tensor<Q_T, L1Format::NZ> dstTensor {
                            .tensor = aL1Tensor[mL1SizeAlign * 32], // 32 : rope headDim的一半
                            .rowCount = AttentionCommon::Align(mL1Size, 16U)
                        };
                        GmCoord gmCoord {
                            .bIdx = info.bIdx,
                            .n2Idx = info.n2Idx,
                            .gS1Idx = info.gS1Idx + mSplitInfo.nBufferStartM + mL1 * M_L1_SPLIT_SIZE,
                            .dIdx = kL1 * 256,
                            .gS1DealSize = mL1Size,
                            .dDealSize = 256
                        };
                        copyQueryGmToL1(dstTensor, queryGmTensor, gmCoord);
                    }
#ifdef BASE_MM
                    mm1A.Set<HardEvent::MTE2_MTE1>();
                    mm1A.Wait<HardEvent::MTE2_MTE1>();
#else
                    SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
                    WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
#endif
                }
#ifdef BASE_MM
                if (nL1 != 0) {
                    // 常驻，做一次拷贝，在循环过程中一直复用
                    // kL1 = 0时需要偏移
                    aL1Tensor = l1QP4Buffers.ReuseNext().GetTensor<Q_T>()[(1 - kL1) * aL1PaddingSize];
                }
#else
#endif

                // 使用unitflag同步
                if (kL1 == 0) {
                    // WaitFlag<HardEvent::FIX_M>(MmFixpEventId(cL0BufIter % 2));
                }
                LocalTensor cL0Tensor =
                    cL0TensorPingPong[(cL0BufIter % 2) *
                                      (L0C_BLOCK_SIZE / sizeof(MM_OUT_T))]; // 需要保证cL0BufIter和m步调一致
#ifdef BASE_MM
                fa_base_matmul::MMParam param;
                param.singleM = mL1SizeAlign;
                param.singleN = nL1SizeAlign;
                param.singleK = kL1Size;
                param.isLeftTranspose = false;
                param.isRightTranspose = false;
                param.isOutKFisrt = (kL1 == 0) ? true : false;
                param.unitFlag = (kL1 == kL1Loops - 1) ? fa_base_matmul::UNITFLAG_EN_OUTER_LAST :
                                 fa_base_matmul::UNITFLAG_ENABLE;
                fa_base_matmul::MatmulKPP<Q_T, KV_T, MM_OUT_T, M_L1_SPLIT_SIZE, N_L1_SPLIT_SIZE, MM1_K_L0_SPLIT_SIZE,
                                          fa_base_matmul::ABLayout::MK, fa_base_matmul::ABLayout::NK>(
                                          aL1Tensor, bL1Tensor, mmL0ABuffers, mmL0BBuffers, cL0Tensor, param);
#else
                fa_base_matmul::MMParam param;
                param.singleM = mL1SizeAlign;
                param.singleN = nL1SizeAlign;
                param.singleK = kL1Size;
                param.isLeftTranspose = false;
                param.isRightTranspose = false;
                param.isOutKFisrt = (kL1 == 0) ? true : false;
                param.unitFlag = (kL1 == kL1Loops - 1) ? fa_base_matmul::UNITFLAG_EN_OUTER_LAST :
                                 fa_base_matmul::UNITFLAG_ENABLE;
                fa_base_matmul::MatmulKPP<Q_T, KV_T, MM_OUT_T, M_L1_SPLIT_SIZE, N_L1_SPLIT_SIZE, MM1_K_L0_SPLIT_SIZE,
                                          fa_base_matmul::ABLayout::MK, fa_base_matmul::ABLayout::NK>(
                                          aL1Tensor, bL1Tensor, mmL0ABuffers, mmL0BBuffers, cL0Tensor, param);
#endif

                if (nL1 == (nL1Loops - 1)) {
#ifdef BASE_MM
                    l1QP4Buffers.FreeNext().Set<HardEvent::MTE1_MTE2>();
#else
                    SetFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]); // 反向同步, 表示L1中的A已经被mte1消费完
#endif
                }

                if (kL1 == 1) { // 最后一轮kL1循环
                    FixpipeParamsV220 fixParams;
                    fixParams.nSize = nL1SizeAlign;
                    fixParams.mSize = mL1SizeAlign;
                    fixParams.srcStride = mL1SizeAlign;
                    fixParams.dstStride = info.actualSingleProcessSInnerSizeAlign; // mm1ResGm两行之间的间隔
                    fixParams.unitFlag = 0b11;
                    fixParams.ndNum = 1; // 输出ND

                    // 输出偏移info.loop % (constInfo.preLoadNum)) * mmResUbSize是否在matmul里计算
                    Fixpipe(mm1ResGm[(info.loop % (constInfo.preLoadNum)) * constInfo.mmResUbSize + nL1 * N_L1_SPLIT_SIZE +
                                     (mSplitInfo.nBufferStartM + mL1 * M_L1_SPLIT_SIZE) *
                                         info.actualSingleProcessSInnerSizeAlign],
                            cL0Tensor, fixParams);
                }
                if (mL1Loops == 2) {
                    cL0BufIter++;
                }
            }
#ifdef BASE_MM
            mm1B.Set<HardEvent::MTE1_MTE2>();
#else
            SetFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]); // 反向同步, 表示L1已经被mte1消费完
#endif
        }
        if (mL1Loops == 1) {
            cL0BufIter++;
        }
    }
    qpL1BufIter += mL1Loops;
}

template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuantMla<FIAT>::ProcessMm2(const AttentionCommon::RunInfo &info, const MSplitInfo mSplitInfo)
{
    uint32_t mSize = mSplitInfo.nBufferDealM;
    uint32_t mSizeAlign = (mSize + 16 - 1) / 16;
    uint32_t mL1Loops = (mSize + M_L1_SPLIT_SIZE - 1) / M_L1_SPLIT_SIZE;
    uint32_t mL1SizeAlign = M_L1_SPLIT_SIZE; // 16对齐
    uint32_t mL1Size = M_L1_SPLIT_SIZE;      // m的实际大小

    uint32_t nSize = BlockAlign<KV_T>(constInfo.headDim);
    uint32_t nL1Loops = (nSize + N_L1_SPLIT_SIZE - 1) / N_L1_SPLIT_SIZE;
    uint32_t nL1SizeAlign = N_L1_SPLIT_SIZE; // 16对齐
    uint32_t nL1Size = N_L1_SPLIT_SIZE;      // n的实际大小

    uint32_t kSize = info.actualSingleProcessSInnerSize;
    uint32_t kL1Size = 256;
    uint32_t kL1SizeAlign = AttentionCommon::Align(kL1Size, 16U);
    uint32_t kL1Loops = (kSize + kL1Size - 1) / kL1Size;
    uint32_t kL0Size = 128;
    uint32_t kL0Loops = (kL1Size + kL0Size - 1) / kL0Size;
    uint32_t kL0SizeAlign = kL0Size;
    LocalTensor<KV_T> bL1Tensor;  // bL1Tensor需要跨m复用, 需要定义在外面?
    LocalTensor<KV_T> subvTensor; // subvTensor需要跨m复用, 需要定义在外面?
#ifdef BASE_MM
    fa_base_matmul::Buffer<fa_base_matmul::BufferType::L1> mm2A;
    fa_base_matmul::Buffer<fa_base_matmul::BufferType::L1> mm2B;
    l1QP4Buffers.SetMExtent(mL1Loops);
#else
#endif

    // ka表示左矩阵4buf选择哪一块buf, kb表示右矩阵3buf选择哪一块buf
    uint32_t ka = 0, kb = 0;
    uint32_t mBaseIdx = qpL1BufIter;
    for (uint32_t nL1 = 0; nL1 < nL1Loops; nL1++) { // n切L1
        if (nL1 == (nL1Loops - 1)) {
            // 尾块
            nL1Size = nSize - (nL1Loops - 1) * N_L1_SPLIT_SIZE;
            nL1SizeAlign = AttentionCommon::Align(nL1Size, 16U);
        }

        // k l1写成一个循环, 和mm1保持一致
        kL1Size = 256;
        kL1SizeAlign = AttentionCommon::Align(kL1Size, 16U);
        for (uint32_t k1 = 0; k1 < kL1Loops; k1++) { // k切L1, 这里套了一层l0来操作
            if (k1 == (kL1Loops - 1)) {
                // 尾块
                kL1Size = kSize - (kL1Loops - 1) * 256;
                kL1SizeAlign = AttentionCommon::Align(kL1Size, 16U);
            }
#ifdef BASE_MM
            mm2A = l1KV3Buffers.Get();
            mm2A.Wait<HardEvent::MTE1_MTE2>();
            bL1Tensor = mm2A.GetTensor<KV_T>();
#else
            uint32_t kb = kvL1BufIter % 3U;
            kvL1BufIter++;
            WaitFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]);
            bL1Tensor = l1KVTensor[kb * L1_BLOCK_OFFSET];
#endif
            uint32_t kOffset = k1 * kL0Loops;
            kL0Size = 128U;
            // 此处必须先初始化kL0Size, 再求kL0Loops, 否则由于循环会改变kL0Size大小, 导致kL0Loops错误
            kL0Loops = (kL1Size + kL0Size - 1U) / kL0Size;
            kL0SizeAlign = kL0Size;
            for (uint32_t kL1 = kOffset; kL1 < kL0Loops + kOffset; kL1++) { // 128 循环搬pa
                if (kL1 == kOffset + kL0Loops - 1U) {
                    // 尾块
                    kL0Size = kL1Size - (kL0Loops - 1U) * kL0Size;
                    kL0SizeAlign = AttentionCommon::Align(kL0Size, 16U);
                }
                FaL1Tensor<KV_T, L1Format::NZ> dstTensor {
                    .tensor = bL1Tensor[(kL1 - kOffset) * 128U * N_L1_SPLIT_SIZE],
                    .rowCount = kL0SizeAlign
                };
                GmKvCoord gmCoord {
                    .bIdx = constInfo.batchContinuous ? info.bIdx : 0,
                    .n2Idx = info.n2Idx,
                    .s2Idx = info.s2BatchOffset + (kL1 - kOffset) * 128U + k1 * 256U,
                    .dIdx = nL1 * N_L1_SPLIT_SIZE,
                    .s2DealSize = kL0Size,
                    .dDealSize = N_L1_SPLIT_SIZE
                };
                copyKvGmToL1(dstTensor, valueGmTensor, gmCoord);
            }
#ifdef BASE_MM
            mm2A.Set<HardEvent::MTE2_MTE1>();
            mm2A.Wait<HardEvent::MTE2_MTE1>();
#else
            SetFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
            WaitFlag<HardEvent::MTE2_MTE1>(mte21KVIds[kb]);
#endif
            mL1SizeAlign = M_L1_SPLIT_SIZE;
            mL1Size = M_L1_SPLIT_SIZE; // m的实际大小
            for (uint32_t mL1 = 0; mL1 < mL1Loops; mL1++) {
                if (mL1 == (mL1Loops - 1)) {
                    // 尾块
                    mL1Size = mSize - (mL1Loops - 1) * M_L1_SPLIT_SIZE;
                    mL1SizeAlign = AttentionCommon::Align(mL1Size, 16U);
                }

                uint32_t mIdx = mBaseIdx + mL1;
                ka = GetQPL1RealIdx(mIdx, k1);
#ifdef BASE_MM
                LocalTensor<KV_T> aL1Tensor;
#else
                LocalTensor<KV_T> aL1Tensor = l1QPTensor[ka * L1_BLOCK_OFFSET];
#endif
                if (nL1 == 0) {
#ifdef BASE_MM
                    mm2B = l1QP4Buffers.AllocNext();
                    if (kL1Loops == 1) {
                        // 保持k循环两次，实际不使用
                        auto next = l1QP4Buffers.AllocNext();
                    }
                    aL1Tensor = mm2B.GetTensor<Q_T>();
                    mm2B.Wait<HardEvent::MTE1_MTE2>();
                    CopyInMm2AToL1(aL1Tensor, info, mSplitInfo.nBufferStartM + mL1 * M_L1_SPLIT_SIZE, mL1Size, kL1Size,
                                   256 * k1);
                    mm2B.Set<HardEvent::MTE2_MTE1>();
                    mm2B.Wait<HardEvent::MTE2_MTE1>();
#else
                    WaitFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]);
                    CopyInMm2AToL1(aL1Tensor, info, mSplitInfo.nBufferStartM + mL1 * M_L1_SPLIT_SIZE, mL1Size, kL1Size,
                                   256 * k1);
                    SetFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
                    WaitFlag<HardEvent::MTE2_MTE1>(mte21QPIds[ka]);
#endif
                }
#ifdef BASE_MM
                if (nL1 != 0) {
                    // 常驻，做一次拷贝，在循环过程中一直复用
                    aL1Tensor = l1QP4Buffers.ReuseNext().GetTensor<Q_T>();
                    if (kL1Loops == 1) {
                        // 保持k循环两次，实际不使用
                        auto next = l1QP4Buffers.ReuseNext();
                    }
                }
#else
#endif
                LocalTensor cL0Tensor =
                    cL0TensorPingPong[(cL0BufIter % 2) *
                                      (L0C_BLOCK_SIZE / sizeof(MM_OUT_T))]; // 需要保证cL0BufIter和m步调一致
                uint32_t baseK = 128;
                uint32_t baseN = 128;
                kL0Size = 128;
                kL0SizeAlign = kL0Size;

                fa_base_matmul::MMParam param;
                param.singleM = mL1SizeAlign;
                param.singleN = nL1SizeAlign;
                param.singleK = kL1Size;
                param.isLeftTranspose = false;
                param.isRightTranspose = false;
                param.isOutKFisrt = (k1 == 0) ? true : false;
                param.unitFlag = (k1 == kL1Loops - 1) ? fa_base_matmul::UNITFLAG_EN_OUTER_LAST :
                                 fa_base_matmul::UNITFLAG_ENABLE;
                fa_base_matmul::MatmulKPP<KV_T, KV_T, MM_OUT_T, M_L1_SPLIT_SIZE, N_L1_SPLIT_SIZE, MM2_K_L0_SPLIT_SIZE,
                                          fa_base_matmul::ABLayout::MK, fa_base_matmul::ABLayout::KN>(
                                          aL1Tensor, bL1Tensor, mmL0ABuffers, mmL0BBuffers, cL0Tensor, param);

                if (nL1 == (nL1Loops - 1)) {
#ifdef BASE_MM
                    l1QP4Buffers.FreeNext().Set<HardEvent::MTE1_MTE2>();
                    if (kL1Loops == 1) {
                        // 保持k循环两次，实际不使用
                        auto next = l1QP4Buffers.FreeNext();
                    }
#else
                    SetFlag<HardEvent::MTE1_MTE2>(mte21QPIds[ka]); // 反向同步, 表示L1中的A已经被mte1消费完
#endif
                }

                if (k1 == (kL1Loops - 1)) {
                    if (nL1 == 0 && mL1 == 0) { // 第一次Fixpipe前等待
                        CrossCoreWaitFlag(constInfo.syncV1NupdateC2);
                    }

                    if (!info.isFirstSInnerLoop) {
                        SetAtomicAdd<MM_OUT_T>();
                    }
                    // ND
                    FixpipeParamsV220 fixParams;
                    fixParams.nSize = nL1SizeAlign;
                    fixParams.mSize = mL1SizeAlign;
                    fixParams.srcStride = mL1SizeAlign;
                    fixParams.dstStride = nSize; // mm2ResGm两行之间的间隔
                    fixParams.unitFlag = 0b11;
                    fixParams.ndNum = 1; // 输出ND
                    uint32_t mStart = mSplitInfo.nBufferStartM + mL1 * M_L1_SPLIT_SIZE;
                    uint64_t mm2Offset = (mSplitInfo.nBufferStartM + mL1 * M_L1_SPLIT_SIZE) * nSize + nL1 * N_L1_SPLIT_SIZE;
                    Fixpipe(mm2ResGm[(info.bn2IdxInCurCore % (constInfo.preLoadNum)) * constInfo.bmm2ResUbSize + mm2Offset],
                            cL0Tensor, fixParams);

                    if (!info.isFirstSInnerLoop) {
                        SetAtomicNone();
                    }
                }

                if (mL1Loops == 2) {
                    cL0BufIter++;
                }
            }
#ifdef BASE_MM
            mm2A.Set<HardEvent::MTE1_MTE2>();
#else
            SetFlag<HardEvent::MTE1_MTE2>(mte21KVIds[kb]);
#endif
        }

        if (mL1Loops == 1) {
            cL0BufIter++;
        }
    }

    qpL1BufIter += mL1Loops;
}

#endif
