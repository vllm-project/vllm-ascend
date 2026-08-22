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
 * \file fia_block_cube_nonquant.h
 * \brief support all headDim and M
 */
#ifndef FIA_BLOCK_CUBE_NONQUANT_H
#define FIA_BLOCK_CUBE_NONQUANT_H

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
#include "../memory_copy.h"

using namespace AttentionCommon;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename FIAT> class FiaBlockCubeNonQuant {
public:

    using T = float;

    using Q_T = typename FIAT::queryType;
    using KV_T = typename FIAT::kvType;
    using OUT_T = typename FIAT::outputType;
    static constexpr bool PAGE_ATTENTION = FIAT::pageAttention;
    static constexpr bool KV_CONTINUOUS = FIAT::kvContinuous;
    static constexpr bool FLASH_DECODE = FIAT::flashDecode;
    static constexpr FIA_LAYOUT LAYOUT_T = FIAT::layout;
    static constexpr FIA_LAYOUT KV_LAYOUT_T = FIAT::kvLayout;

    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    static constexpr GmFormat KV_FORMAT = GetKVFormat<KV_LAYOUT_T, PAGE_ATTENTION>();

    static constexpr bool ANTIQUANT = !IsSameType<Q_T, KV_T>::value;
    static constexpr bool QUANT = (IsSameType<Q_T, KV_T>::value && IsSameType<KV_T, int8_t>::value);
    // define pse datetype
    using pseShiftType = typename AscendC::Conditional<AscendC::IsSameType<Q_T, int8_t>::value, half, Q_T>::type;
    // 后接量化的条件需要重新审视
    static constexpr bool POST_QUANT = IsSameType<OUT_T, int8_t>::value;
    using MM_OUT_T = typename AscendC::Conditional<(ANTIQUANT || QUANT), int32_t, T>::type;
    __aicore__ inline FiaBlockCubeNonQuant(){};
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
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
    __aicore__ inline void InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm, GlobalTensor<MM_OUT_T> mm2ResGm);

    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

    __aicore__ inline void ComputeMm1(const RunInfo &info);
    __aicore__ inline void ComputeMm2(const RunInfo &info);

protected:
    template <typename T> __aicore__ inline size_t BlockAlign(size_t s)
    {
        if constexpr (IsSameType<T, int4b_t>::value) {
            return (s + 63) / 64 * 64;
        }
        size_t n = (32 / sizeof(T));
        return (s + n - 1) / n * n;
    }

    __aicore__ inline void InitValueGm(uint32_t bIdx);
    __aicore__ inline void InitKeyGm(uint32_t bIdx);
    __aicore__ inline void UpdateKey(uint32_t bIdx);
    __aicore__ inline void UpdateValue(uint32_t bIdx);

    __aicore__ inline void InitQBuffer(uint32_t batchSize, uint32_t kvHeadNum, uint32_t gSize, uint32_t qSeqSize,
                                       uint32_t headDim, GlobalTensor<uint64_t> actualSeqLengthsGmQ,
                                       uint32_t actualLenQDims, FaGmTensor<Q_T, Q_FORMAT> &qGmTensor,
                                       GlobalTensor<Q_T> &qGm);
    __aicore__ inline void InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize,
                                        GlobalTensor<uint64_t> actualSeqLengthsGmQ, uint32_t actualLenDims,
                                        uint32_t kvHeadNum, uint32_t kvCacheBlockSize, uint32_t headDim,
                                        FaGmTensor<KV_T, KV_FORMAT> &kvGmTensor, GlobalTensor<KV_T> &kvGm,
                                        uint64_t bn2Stride, uint64_t n2Stride);

    __aicore__ inline void CopyKeyToL1(const RunInfo &info,
                                       uint32_t kSize, uint32_t kStart, uint32_t kDealSize,
                                       uint32_t nSize, uint32_t nStart, uint32_t nDealSize,
                                       uint32_t nopeDealSize, uint32_t ropeDealSize, uint64_t l1Offset);
    __aicore__ inline void CopyValueToL1(const RunInfo &info, uint32_t kSize, uint32_t kStart, uint32_t kDealSize,
                                         uint32_t nSize, uint32_t nStart, uint32_t nDealSize, uint64_t l1Offset);
    __aicore__ inline void DealMm1SingleMKN(const RunInfo &info,
                                            uint32_t mSize, uint32_t mStart, uint32_t mDealSize,
                                            uint32_t kSize, uint32_t kStart, uint32_t kDealSize,
                                            uint32_t nSize, uint32_t nStart, uint32_t nDealSize,
                                            uint32_t nopeDealSize, uint32_t ropeDealSize);
    __aicore__ inline void DealMm2SingleMKN(const RunInfo &info,
                                            uint32_t mSize, uint32_t mStart, uint32_t mDealSize,
                                            uint32_t kSize, uint32_t kStart, uint32_t kDealSize,
                                            uint32_t nSize, uint32_t nStart, uint32_t nDealSize);
    __aicore__ inline void CopyQGmToL1(uint32_t nopeDealSize, uint32_t ropeDealSize, uint32_t mActSizeAlign, uint32_t bIdx, 
                                       uint32_t n2Idx, uint32_t gS1Idx, uint32_t kStart, uint32_t mActSize);
    __aicore__ inline void CopyQDealSizeToL1(uint32_t DealSize, uint64_t queryL1Offset, uint32_t mActSizeAlign, uint32_t bIdx, uint32_t n2Idx, 
                                             uint32_t gS1Idx, uint32_t dIdx, uint32_t mActSize, FaGmTensor<Q_T, Q_FORMAT> &qGmTensor);
    __aicore__ inline void CopyPrefixAndKeyToL1(const RunInfo &info, uint32_t kSize, uint32_t kStart,
                                                uint32_t kDealSize, uint32_t nSize, uint32_t nStart, uint32_t nDealSize,
                                                uint32_t nopeDealSize, uint64_t l1Offset);
    __aicore__ inline void CopyPrefixAndValueToL1(const RunInfo &info, uint32_t kSize, uint32_t kStart,
                                                  uint32_t kDealSize, uint32_t nSize, uint32_t nStart,
                                                  uint32_t nDealSize, uint64_t l1Offset);

protected :
    // key和value的TensorList原始地址
    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;
    // mm1
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<Q_T> qRopeGm; 
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> keyRopeGm;
    GlobalTensor<MM_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> keyPrefixGm;
    // mm2
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<MM_OUT_T> mm2ResGm;
    GlobalTensor<KV_T> valuePrefixGm;

    GlobalTensor<uint64_t> actualSeqLengthsGmQ;
    GlobalTensor<uint64_t> actualSeqLengthsGm;
    __gm__ uint8_t *actualSeqLengths = nullptr;

    // block_table
    GlobalTensor<int32_t> blockTableGm;

    ConstInfo constInfo{};

    // =================================L1 Buffer=================================
    static constexpr uint32_t L1_QP_SIZE = 128 * 1024;
    static constexpr uint32_t L1_KV_SIZE = 128 * 1024;
    // QP复用buffer,组成DB
    TBuf<TPosition::A1> qpBufL1;
    LocalTensor<Q_T> qpL1Tensor;
    // QP复用buffer,组成DB
    TBuf<TPosition::A1> kvBufL1;
    LocalTensor<KV_T> kvL1Tensor;

    // =================================L0 Buffer=================================
    // L0 buffer size
    static constexpr uint32_t L0A_PP_SIZE = 32 * 1024;
    static constexpr uint32_t L0B_PP_SIZE = 32 * 1024;
    static constexpr uint32_t L0C_PP_SIZE = 64 * 1024;
    // L0_A
    TBuf<TPosition::A2> tmpBufL0A;
    LocalTensor<KV_T> aL0TensorPingPong;
    // L0_B
    TBuf<TPosition::B2> tmpBufL0B;
    LocalTensor<KV_T> bL0TensorPingPong;
    // L0_C
    TBuf<TPosition::CO1> tmpBufL0C;
    LocalTensor<T> cL0TensorPingPong;

    // =================================Event&Buffer ID===========================
    // mte2 <> mte1 EventID
    static constexpr uint32_t QP_EVENT0 = EVENT_ID2;
    static constexpr uint32_t QP_EVENT1 = EVENT_ID3;
    uint32_t qpBufId = 0;
    static constexpr uint32_t KV_EVENT0 = EVENT_ID4;
    static constexpr uint32_t KV_EVENT1 = EVENT_ID5;
    uint32_t kvBufId = 0;
    // mte1 <> mmad EventID
    static constexpr uint32_t L0AB_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0AB_EVENT1 = EVENT_ID4;
    uint32_t l0abBufId = 0;
    // mmad <> fixpipe EventID
    static constexpr uint32_t L0C_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0C_EVENT1 = EVENT_ID4;
    uint32_t l0cBufId = 0;

    __gm__ uint8_t *actualSequenceLengthsQ = nullptr;

private:
    FaGmTensor<Q_T, Q_FORMAT> queryGmTensor;
    FaGmTensor<Q_T, Q_FORMAT> queryRopeGmTensor;
    CopyQueryGmToL1<Q_T, Q_FORMAT> copyQueryGmToL1;

    FaGmTensor<KV_T, KV_FORMAT> keyGmTensor;
    FaGmTensor<KV_T, KV_FORMAT> keyRopeGmTensor;
    FaGmTensor<KV_T, KV_FORMAT> valueGmTensor;
    CopyKvGmToL1<KV_T, KV_FORMAT> copyKvGmToL1;

    FaGmTensor<KV_T, KV_FORMAT> keyPrefixGmTensor;
    FaGmTensor<KV_T, KV_FORMAT> valuePrefixGmTensor;
};


template <typename FIAT> class FiaBlockCubeNonQuantDummy {
public:
    using T = float;
    using Q_T = typename FIAT::queryType;
    using KV_T = typename FIAT::kvType;
    using OUT_T = typename FIAT::outputType;

    static constexpr bool ANTIQUANT = !IsSameType<Q_T, KV_T>::value;
    static constexpr bool QUANT = (IsSameType<Q_T, KV_T>::value && IsSameType<KV_T, int8_t>::value);
    using pseShiftType = typename AscendC::Conditional<AscendC::IsSameType<Q_T, int8_t>::value, half, Q_T>::type;
    static constexpr bool POST_QUANT = IsSameType<OUT_T, int8_t>::value;
    using MM_OUT_T = typename AscendC::Conditional<(ANTIQUANT || QUANT), int32_t, T>::type;
    __aicore__ inline FiaBlockCubeNonQuantDummy(){};
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void Init(
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift, __gm__ uint8_t *attenMask,
        __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *deqScale1, 
        __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2, __gm__ uint8_t *quantOffset2,
        __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset, __gm__ uint8_t *blockTable,
        __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *keyAntiquantScale,
        __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffset,
        __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen,
        __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *keyRopeAntiquantScale,
        __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse);
    __aicore__ inline void InitMm1GlobalTensor(GlobalTensor<MM_OUT_T> mm1ResGm);
    __aicore__ inline void InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm, GlobalTensor<MM_OUT_T> mm2ResGm);
    __aicore__ inline void ComputeMm1(const RunInfo &info);
    __aicore__ inline void ComputeMm2(const RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
};

template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuant<FIAT>::InitKeyGm(uint32_t bIdx)
{
    ListTensorDesc keyListTensorDesc((__gm__ void *)keyPtr);
    __gm__ uint8_t *key_ = (__gm__ uint8_t *)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

    keyGm.SetGlobalBuffer((__gm__ KV_T *)key_);
    if (constInfo.l2CacheOffFlag) {
        // 关闭K、V的L2 Cache
#ifndef ASCENDC_OOM
        keyGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuant<FIAT>::InitValueGm(uint32_t bIdx)
{
    ListTensorDesc valueListTensorDesc((__gm__ void *)valuePtr);
    __gm__ uint8_t *value_ = (__gm__ uint8_t *)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(bIdx);

    valueGm.SetGlobalBuffer((__gm__ KV_T *)value_);
    if (constInfo.l2CacheOffFlag) {
        // 关闭K、V的L2 Cache
#ifndef ASCENDC_OOM
        valueGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuant<FIAT>::InitParams(const ConstInfo &constInfo)
{
    this->constInfo = constInfo;
}

template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuant<FIAT>::Init(
        __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
        __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
        __gm__ uint8_t *deqScale1, __gm__ uint8_t *quantScale1, __gm__ uint8_t *deqScale2, __gm__ uint8_t *quantScale2,
        __gm__ uint8_t *quantOffset2, __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset,
        __gm__ uint8_t *blockTable, __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize,
        __gm__ uint8_t *keyAntiquantScale, __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale,
        __gm__ uint8_t *valueAntiquantOffset, __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix,
        __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
        __gm__ uint8_t *keyRopeAntiquantScale, __gm__ uint8_t *attentionOut,
        __gm__ uint8_t *softmaxLse)
{
    // 先初始化基础参数
    if (constInfo.actualLenQDims != 0) {
        actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengthsQ, constInfo.actualLenQDims);
        this->actualSequenceLengthsQ = actualSeqLengthsQ;
    }
    if (constInfo.actualLenDims != 0) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t *)actualSeqLengths, constInfo.actualLenDims);
        this->actualSeqLengths = actualSeqLengths;
    }
    if constexpr (PAGE_ATTENTION) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    }

    // 再初始化复杂参数
    uint32_t qkTensorD = constInfo.ropeSplitMode ? (constInfo.headDim) : (constInfo.headDim + constInfo.headDimRope);
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    InitQBuffer(constInfo.batchSize, constInfo.kvHeadNum, constInfo.gSize, constInfo.qSeqSize, qkTensorD, actualSeqLengthsGmQ,
                constInfo.actualLenQDims, queryGmTensor, queryGm);

    if (constInfo.ropeSplitMode) {
        // query rope
        qRopeGm.SetGlobalBuffer((__gm__ Q_T *)queryRope);
        InitQBuffer(constInfo.batchSize, constInfo.kvHeadNum, constInfo.gSize, constInfo.qSeqSize,
                constInfo.headDimRope, actualSeqLengthsGmQ, constInfo.actualLenQDims, queryRopeGmTensor, qRopeGm);
        // key rope
        keyRopeGm.SetGlobalBuffer((__gm__ KV_T *)keyRope);
        InitKVBuffer(constInfo.batchSize, constInfo.kvSeqSize, actualSeqLengthsGm, constInfo.actualLenDims,
                    constInfo.kvHeadNum, constInfo.kvCacheBlockSize, constInfo.headDimRope, keyRopeGmTensor, keyRopeGm,
                    constInfo.keyRopeBnStride, constInfo.keyRopeN2Stride);
    }

    keyPtr = key;
    valuePtr = value;
    // batch连续时,只需要初始化一次;不连续时,需要在使用时根据batchIdx初始化
    if (constInfo.batchContinuous) {
        InitKeyGm(0);
        InitKVBuffer(constInfo.batchSize, constInfo.kvSeqSize, actualSeqLengthsGm, constInfo.actualLenDims,
                    constInfo.kvHeadNum, constInfo.kvCacheBlockSize, qkTensorD, keyGmTensor, keyGm,
                    constInfo.keyBnStride, constInfo.keyN2Stride);

        InitValueGm(0);
        InitKVBuffer(constInfo.batchSize, constInfo.kvSeqSize, actualSeqLengthsGm, constInfo.actualLenDims,
                    constInfo.kvHeadNum, constInfo.kvCacheBlockSize, constInfo.headDim, valueGmTensor, valueGm,
                    constInfo.valueBnStride, constInfo.valueN2Stride);
    }

    if (constInfo.systemPrefixFlag) {
        keyPrefixGm.SetGlobalBuffer((__gm__ KV_T *)keySharedPrefix);
        keyPrefixGmTensor.gmTensor = keyPrefixGm;
        valuePrefixGm.SetGlobalBuffer((__gm__ KV_T *)valueSharedPrefix);
        valuePrefixGmTensor.gmTensor = valuePrefixGm;
        if constexpr (!PAGE_ATTENTION) {
            if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
                // 前缀batch为1, 直接初始化为1
                keyPrefixGmTensor.offsetCalculator.Init(1, constInfo.kvHeadNum, constInfo.systemPrefixMaxLen,
                                                        constInfo.headDim);
                valuePrefixGmTensor.offsetCalculator.Init(1, constInfo.kvHeadNum, constInfo.systemPrefixMaxLen,
                                                          constInfo.headDim);
            }
        }
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuant<FIAT>::InitQBuffer(uint32_t batchSize, uint32_t kvHeadNum, uint32_t gSize,
    uint32_t qSeqSize, uint32_t headDim, GlobalTensor<uint64_t> actualSeqLengthsGmQ, 
    uint32_t actualLenQDims, FaGmTensor<Q_T, Q_FORMAT> &qGmTensor, GlobalTensor<Q_T> &qGm)
{
    qGmTensor.gmTensor = qGm;
    if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_BNGSD) {
        qGmTensor.offsetCalculator.Init(batchSize, kvHeadNum, gSize, qSeqSize, headDim, actualSeqLengthsGmQ,
                                        actualLenQDims, constInfo.isQHasLeftPadding, constInfo.qLeftPaddingSize);
    } else if constexpr (GmLayoutParams<Q_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
        qGmTensor.offsetCalculator.Init(kvHeadNum, gSize, headDim, actualSeqLengthsGmQ, actualLenQDims);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuant<FIAT>::InitKVBuffer(uint32_t batchSize, uint32_t kvSeqSize,
    GlobalTensor<uint64_t> actualSeqLengthsGmQ, uint32_t actualLenDims, uint32_t kvHeadNum, 
    uint32_t kvCacheBlockSize, uint32_t headDim, FaGmTensor<KV_T, KV_FORMAT> &kvGmTensor, GlobalTensor<KV_T> &kvGm,
    uint64_t bn2Stride, uint64_t n2Stride)
{
    kvGmTensor.gmTensor = kvGm;
    if constexpr (PAGE_ATTENTION) {
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_BNBD) {
            kvGmTensor.offsetCalculator.Init(kvHeadNum, kvCacheBlockSize, headDim, blockTableGm,
                                             constInfo.maxBlockNumPerBatch, bn2Stride, n2Stride);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_PA_NZ) {
            uint32_t d0 = 32 / sizeof(KV_T);
            uint32_t d1 = headDim / d0;
            kvGmTensor.offsetCalculator.Init(kvHeadNum, kvCacheBlockSize, d1, d0, blockTableGm,
                                             constInfo.maxBlockNumPerBatch, bn2Stride, n2Stride);
        }
    } else {
        if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
            kvGmTensor.offsetCalculator.Init(batchSize, kvHeadNum, kvSeqSize, headDim, actualSeqLengthsGm, actualLenDims,
                                             constInfo.isKVHasLeftPadding, constInfo.kvLeftPaddingSize);
        } else if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_TND) {
            kvGmTensor.offsetCalculator.Init(kvHeadNum, headDim, actualSeqLengthsGm, actualLenDims);
        }
    }
}

template <typename FIAT>
__aicore__ inline void
FiaBlockCubeNonQuant<FIAT>::InitMm1GlobalTensor(GlobalTensor<MM_OUT_T> mm1ResGm)
{
    // mm1
    this->mm1ResGm = mm1ResGm;
}

template <typename FIAT>
__aicore__ inline void
FiaBlockCubeNonQuant<FIAT>::InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm, GlobalTensor<MM_OUT_T> mm2ResGm)
{
    // mm2
    this->vec1ResGm = vec1ResGm;
    this->mm2ResGm = mm2ResGm;
}

template <typename FIAT> __aicore__ inline void FiaBlockCubeNonQuant<FIAT>::UpdateKey(uint32_t bIdx)
{
    static_assert(GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD,
                  "when only KV TensorList, support update KeyGm");
    InitKeyGm(bIdx);
    uint64_t s2Size = fa_base_kernel::SeqLenFromTensorList<KV_LAYOUT_T>(keyPtr, bIdx);
    keyGmTensor.gmTensor = keyGm;
    keyGmTensor.offsetCalculator.Init(0, constInfo.kvHeadNum, s2Size, constInfo.headDim, actualSeqLengthsGm, constInfo.actualLenDims);
}

template <typename FIAT> __aicore__ inline void FiaBlockCubeNonQuant<FIAT>::UpdateValue(uint32_t bIdx)
{
    static_assert(GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD,
                  "when only KV TensorList, support update ValueGm");
    InitValueGm(bIdx);
    uint64_t s2Size = fa_base_kernel::SeqLenFromTensorList<KV_LAYOUT_T>(valuePtr, bIdx);
    valueGmTensor.gmTensor = valueGm;
    valueGmTensor.offsetCalculator.Init(0, constInfo.kvHeadNum, s2Size, constInfo.headDim, actualSeqLengthsGm, constInfo.actualLenDims);
}

template <typename FIAT> __aicore__ inline void FiaBlockCubeNonQuant<FIAT>::InitBuffers(TPipe *pipe)
{
    // L1
    pipe->InitBuffer(qpBufL1, L1_QP_SIZE * 2);
    qpL1Tensor = qpBufL1.Get<Q_T>();

    pipe->InitBuffer(kvBufL1, L1_KV_SIZE * 2);
    kvL1Tensor = kvBufL1.Get<KV_T>();

    // L0
    pipe->InitBuffer(tmpBufL0A, L0A_PP_SIZE * 2);
    aL0TensorPingPong = tmpBufL0A.Get<KV_T>();

    pipe->InitBuffer(tmpBufL0B, L0B_PP_SIZE * 2);
    bL0TensorPingPong = tmpBufL0B.Get<KV_T>();

    pipe->InitBuffer(tmpBufL0C, L0C_PP_SIZE * 2);
    cL0TensorPingPong = tmpBufL0C.Get<T>();
}

template <typename FIAT> __aicore__ inline void FiaBlockCubeNonQuant<FIAT>::AllocEventID()
{
    SetFlag<HardEvent::MTE1_MTE2>(QP_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(QP_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT1);

    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT1);

    SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT1);
}

template <typename FIAT> __aicore__ inline void FiaBlockCubeNonQuant<FIAT>::FreeEventID()
{
    WaitFlag<HardEvent::MTE1_MTE2>(QP_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(QP_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT1);

    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT1);

    WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
    WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);
}

template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuant<FIAT>::CopyKeyToL1(const RunInfo &info,
                                                           uint32_t kSize, uint32_t kStart, uint32_t kDealSize,
                                                           uint32_t nSize, uint32_t nStart, uint32_t nDealSize,
                                                           uint32_t nopeDealSize, uint32_t ropeDealSize,
                                                           uint64_t l1Offset)
{
    if (nopeDealSize > 0) {
        FaL1Tensor<KV_T, L1Format::NZ> dstTensor {
            .tensor = kvL1Tensor[l1Offset],
            .rowCount = AttentionCommon::Align(nDealSize, 16U)
        };
        GmKvCoord gmCoord {
            .bIdx = constInfo.batchContinuous ? info.bIdx : 0,
            .n2Idx = info.n2Idx,
            .s2Idx = info.s2Idx * constInfo.s2BaseSize + nStart,
            .dIdx = kStart,
            .s2DealSize = nDealSize,
            .dDealSize = nopeDealSize
        };
        copyKvGmToL1(dstTensor, keyGmTensor, gmCoord);
    }
    if (ropeDealSize > 0) {
        // nopeDealSize需要按照32B对齐, 否则这里取偏移的方式错误, 并且当前分段拷贝存在问题
        FaL1Tensor<KV_T, L1Format::NZ> dstTensor {
            .tensor = kvL1Tensor[l1Offset + AttentionCommon::Align(nopeDealSize, 16U) * AttentionCommon::Align(nDealSize, 16U)],
            .rowCount = AttentionCommon::Align(nDealSize, 16U)
        };
        GmKvCoord gmCoord {
            .bIdx = constInfo.batchContinuous ? info.bIdx : 0,
            .n2Idx = info.n2Idx,
            .s2Idx = info.s2Idx * constInfo.s2BaseSize + nStart,
            .dIdx = kStart + nopeDealSize - static_cast<uint32_t>(constInfo.headDim),
            .s2DealSize = nDealSize,
            .dDealSize = ropeDealSize
        };
        copyKvGmToL1(dstTensor, keyRopeGmTensor, gmCoord);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuant<FIAT>::CopyValueToL1(const RunInfo &info, uint32_t kSize, uint32_t kStart, uint32_t kDealSize,
                                         uint32_t nSize, uint32_t nStart, uint32_t nDealSize, uint64_t l1Offset)
{
    FaL1Tensor<KV_T, L1Format::NZ> dstTensor {
        .tensor = kvL1Tensor[l1Offset],
        .rowCount = AttentionCommon::Align(kDealSize, 16U)
    };
    GmKvCoord gmCoord {
        .bIdx = constInfo.batchContinuous ? info.bIdx : 0U,
        .n2Idx = info.n2Idx,
        .s2Idx = info.s2Idx * constInfo.s2BaseSize + kStart,
        .dIdx = nStart,
        .s2DealSize = kDealSize,
        .dDealSize = nDealSize
    };
    copyKvGmToL1(dstTensor, valueGmTensor, gmCoord);
}

//在公共前缀情况下拷贝
template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuant<FIAT>::CopyPrefixAndKeyToL1(const RunInfo &info, uint32_t kSize, uint32_t kStart,
                                                uint32_t kDealSize, uint32_t nSize, uint32_t nStart, uint32_t nDealSize,
                                                uint32_t nopeDealSize, uint64_t l1Offset)
{
    FaL1Tensor<KV_T, L1Format::NZ> dstTensor {
        .tensor = kvL1Tensor[l1Offset],
        .rowCount = AttentionCommon::Align(nDealSize, 16U)
    };
    // 得出前缀和后缀的处理长度
    uint32_t s2StartIdx = info.s2Idx * constInfo.s2BaseSize + nStart;
    uint32_t prefixDealSize = constInfo.systemPrefixLen > s2StartIdx ? (constInfo.systemPrefixLen - s2StartIdx) : 0U;
    prefixDealSize = nDealSize < prefixDealSize ? nDealSize : prefixDealSize;
    uint32_t s2EndIdx = info.s2Idx * constInfo.s2BaseSize + nStart + nDealSize;
    uint32_t normalDealSize = s2EndIdx > constInfo.systemPrefixLen ? (s2EndIdx - constInfo.systemPrefixLen) : 0U;
    normalDealSize = nDealSize < normalDealSize ? nDealSize : normalDealSize;

    // prefix
    if (prefixDealSize > 0) {
        GmKvCoord gmCoord {
            .bIdx = 0U,
            .n2Idx = info.n2Idx,
            .s2Idx = s2StartIdx,
            .dIdx = kStart,
            .s2DealSize = prefixDealSize,
            .dDealSize = nopeDealSize
        };
        copyKvGmToL1(dstTensor, keyPrefixGmTensor, gmCoord);
    }

    // normal
    if (normalDealSize > 0) {
        GmKvCoord gmCoord {
            .bIdx = constInfo.batchContinuous ? info.bIdx : 0U,
            .n2Idx = info.n2Idx,
            .s2Idx = constInfo.systemPrefixLen > s2StartIdx ? 0U : (s2StartIdx - constInfo.systemPrefixLen),
            .dIdx = kStart,
            .s2DealSize = normalDealSize,
            .dDealSize = nopeDealSize
        };
        dstTensor.tensor = kvL1Tensor[l1Offset + BUFFER_SIZE_BYTE_32B / sizeof(KV_T) * prefixDealSize];
        copyKvGmToL1(dstTensor, keyGmTensor, gmCoord);
    } 
}

template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuant<FIAT>::CopyPrefixAndValueToL1(const RunInfo &info, uint32_t kSize, uint32_t kStart,
                                                  uint32_t kDealSize, uint32_t nSize, uint32_t nStart,
                                                  uint32_t nDealSize, uint64_t l1Offset)
{
    FaL1Tensor<KV_T, L1Format::NZ> dstTensor {
        .tensor = kvL1Tensor[l1Offset],
        .rowCount = AttentionCommon::Align(kDealSize, 16U)
    };
    uint32_t s2StartIdx = info.s2Idx * constInfo.s2BaseSize + kStart;
    uint32_t prefixDealSize = s2StartIdx > constInfo.systemPrefixLen ? 0U : (constInfo.systemPrefixLen - s2StartIdx);
    prefixDealSize = kDealSize < prefixDealSize ? kDealSize : prefixDealSize;
    uint32_t s2EndIdx = info.s2Idx * constInfo.s2BaseSize + kStart + kDealSize;
    uint32_t normalDealSize = s2EndIdx > constInfo.systemPrefixLen ? (s2EndIdx - constInfo.systemPrefixLen) : 0U;
    normalDealSize = kDealSize < normalDealSize ? kDealSize : normalDealSize;
    // prefix 
    if (prefixDealSize > 0) {
        GmKvCoord gmCoord {
            .bIdx = 0U,
            .n2Idx = info.n2Idx,
            .s2Idx = info.s2Idx * constInfo.s2BaseSize + kStart,
            .dIdx = nStart,
            .s2DealSize = prefixDealSize,
            .dDealSize = nDealSize
        };
        copyKvGmToL1(dstTensor, valuePrefixGmTensor, gmCoord);
    }
    // normal
    if (normalDealSize > 0) {
        GmKvCoord gmCoord {
            .bIdx = constInfo.batchContinuous ? info.bIdx : 0U,
            .n2Idx = info.n2Idx,
            .s2Idx = (constInfo.systemPrefixLen > s2StartIdx ? constInfo.systemPrefixLen : s2StartIdx) - constInfo.systemPrefixLen,
            .dIdx = nStart,
            .s2DealSize = normalDealSize,
            .dDealSize = nDealSize
        };
        dstTensor.tensor = kvL1Tensor[l1Offset + BUFFER_SIZE_BYTE_32B / sizeof(KV_T) * prefixDealSize];
        copyKvGmToL1(dstTensor, valueGmTensor, gmCoord);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuant<FIAT>::CopyQDealSizeToL1(uint32_t DealSize, uint64_t queryL1Offset, uint32_t mActSizeAlign, uint32_t bIdx, uint32_t n2Idx, 
                                                                 uint32_t gS1Idx, uint32_t dIdx, uint32_t mActSize, FaGmTensor<Q_T, Q_FORMAT> &qGmTensor)
{
    FaL1Tensor<Q_T, L1Format::NZ> dstTensor {
        .tensor = qpL1Tensor[queryL1Offset],
        .rowCount = mActSizeAlign
    };
    GmCoord gmCoord {
        .bIdx = bIdx,
        .n2Idx = n2Idx,
        .gS1Idx = gS1Idx,
        .dIdx = dIdx,
        .gS1DealSize = mActSize,
        .dDealSize = DealSize
    };
    copyQueryGmToL1(dstTensor, qGmTensor, gmCoord); 
}

template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuant<FIAT>::CopyQGmToL1(uint32_t nopeDealSize, uint32_t ropeDealSize, uint32_t mActSizeAlign, uint32_t bIdx, 
                                                           uint32_t n2Idx, uint32_t gS1Idx, uint32_t kStart, uint32_t mActSize)
{
    uint64_t queryL1BaseOffset = (qpBufId % 2) * (L1_QP_SIZE / sizeof(Q_T));
    if(nopeDealSize > 0) {
        CopyQDealSizeToL1(nopeDealSize, queryL1BaseOffset, mActSizeAlign, bIdx, n2Idx, gS1Idx, kStart, mActSize, queryGmTensor);
    }
    uint64_t queryRopeL1Offset = queryL1BaseOffset + AttentionCommon::Align(nopeDealSize, 16U) * mActSizeAlign;
    if(ropeDealSize > 0) {
        uint32_t ropeKStart = kStart + nopeDealSize - static_cast<uint32_t>(constInfo.headDim);
        CopyQDealSizeToL1(ropeDealSize, queryRopeL1Offset, mActSizeAlign, bIdx, n2Idx, gS1Idx, ropeKStart, mActSize, queryRopeGmTensor);
    }
}

template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuant<FIAT>::DealMm1SingleMKN(const RunInfo &info,
                                                                uint32_t mSize, uint32_t mStart, uint32_t mDealSize,
                                                                uint32_t kSize, uint32_t kStart, uint32_t kDealSize,
                                                                uint32_t nSize, uint32_t nStart, uint32_t nDealSize,
                                                                uint32_t nopeDealSize, uint32_t ropeDealSize)
{
    constexpr uint32_t M_BASE = 128;
    constexpr uint32_t K_BASE = 128;

    WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kvBufId % 2);

    uint64_t l1Offset = (kvBufId % 2) * (L1_KV_SIZE / sizeof(KV_T));
    if (constInfo.systemPrefixFlag) {
        CopyPrefixAndKeyToL1(info, kSize, kStart, kDealSize, nSize, nStart, nDealSize, nopeDealSize, l1Offset);
    } else {
        CopyKeyToL1(info, kSize, kStart, kDealSize, nSize, nStart, nDealSize, nopeDealSize, ropeDealSize, l1Offset);
    }

    SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvBufId % 2);
    WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvBufId % 2);
    for (uint32_t m = 0; m < mDealSize; m += M_BASE) {
        uint32_t mActSize = M_BASE;
        if (m + M_BASE > mDealSize) {
            mActSize = mDealSize - m;
        }
        uint32_t mActSizeAlign = AttentionCommon::Align(mActSize, 16U);
        // QK 首次S2循环拷贝Q
        if (nStart == 0) {
            WaitFlag<HardEvent::MTE1_MTE2>(QP_EVENT0 + qpBufId % 2);
            uint64_t queryL1BaseOffset = (qpBufId % 2) * (L1_QP_SIZE / sizeof(Q_T));
            // 根据nope和rope上需要拷贝的数据长度分别执行Copy动作
            uint32_t gS1Idx = info.gS1Idx + mStart + m;
            CopyQGmToL1(nopeDealSize, ropeDealSize, mActSizeAlign, info.bIdx, info.n2Idx, gS1Idx, kStart, mActSize);
            SetFlag<HardEvent::MTE2_MTE1>(QP_EVENT0 + qpBufId % 2);
            WaitFlag<HardEvent::MTE2_MTE1>(QP_EVENT0 + qpBufId % 2);
        } else {
            if (m == 0) {
                qpBufId = (qpBufId + ((mDealSize + M_BASE - 1) / M_BASE)) % 2;
            }
        }

        // matmul k
        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + l0cBufId % 2);
        LocalTensor<T> cL0Tensor = cL0TensorPingPong[(l0cBufId % 2) * (L0C_PP_SIZE / sizeof(MM_OUT_T))];
        uint32_t kDealSizeAlign = AttentionCommon::Align(nopeDealSize, 16U) + AttentionCommon::Align(ropeDealSize, 16U);
        for (uint32_t k = 0; k < kDealSizeAlign; k += K_BASE) {
            uint32_t kActSize = K_BASE;
            if (k + K_BASE > kDealSizeAlign) {
                kActSize = kDealSizeAlign - k;
            }
            uint32_t kActSizeAlign = AttentionCommon::Align(kActSize, 16U);
            WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + l0abBufId % 2);
            LocalTensor<Q_T> aL0Tensor = aL0TensorPingPong[(l0abBufId % 2) * (L0A_PP_SIZE / sizeof(Q_T))];
            LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(l0abBufId % 2) * (L0B_PP_SIZE / sizeof(KV_T))];
            // LoadAToL0
            {
                uint64_t qL1Offset = (qpBufId % 2) * (L1_QP_SIZE / sizeof(Q_T)) + k * mActSizeAlign;
                uint32_t mLoop = mActSizeAlign / 16;
                for (uint32_t i = 0; i < mLoop; i++) {
                    LoadData2DParams loadData2DParams;
                    loadData2DParams.startIndex = i;
                    loadData2DParams.repeatTimes = kActSizeAlign / 16;
                    loadData2DParams.srcStride = mActSizeAlign / 16;
                    loadData2DParams.dstGap = 0;
                    loadData2DParams.ifTranspose = false;;
                    LoadData(aL0Tensor[16 * i * kActSizeAlign], qpL1Tensor[qL1Offset], loadData2DParams);
                }
            }
            // LoadBToL0
            {
                uint64_t kL1Offset = (kvBufId % 2) * (L1_KV_SIZE / sizeof(KV_T)) + k * AttentionCommon::Align(nDealSize, 16U);
                LoadData2DParams loadData2DParams;
                loadData2DParams.startIndex = 0;
                loadData2DParams.repeatTimes = (kActSizeAlign / 16) * (AttentionCommon::Align(nDealSize, 16U) / 16);
                loadData2DParams.srcStride = 1;
                loadData2DParams.dstGap = 0;
                loadData2DParams.ifTranspose = false;
                LoadData(bL0Tensor, kvL1Tensor[kL1Offset], loadData2DParams);
            }
            SetFlag<HardEvent::MTE1_M>(L0AB_EVENT0 + l0abBufId % 2);
            WaitFlag<HardEvent::MTE1_M>(L0AB_EVENT0 + l0abBufId % 2);
            // MMAD
            {
                MmadParams mmadParams;
                mmadParams.m = mActSize;
                if (mmadParams.m == 1) {
                    mmadParams.m = 16;
                }
                mmadParams.n = nDealSize;
                mmadParams.k = kActSize;
                mmadParams.cmatrixSource = false;
                mmadParams.cmatrixInitVal = (k == 0);
                Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
            }
            AscendC::PipeBarrier<PIPE_M>();
            SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + l0abBufId % 2);
            l0abBufId = (l0abBufId + 1) % 2;
        }
        SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + l0cBufId % 2);
        WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + l0cBufId % 2);
        // FIXPIPE
        {
            if (kStart != 0) {
                SetAtomicAdd<MM_OUT_T>();
            }
            FixpipeParamsV220 fixParams;
            fixParams.mSize = mActSize;
            fixParams.nSize = nDealSize;
            fixParams.srcStride = mActSizeAlign;
            fixParams.ndNum = 1;
            fixParams.dstStride = info.actualSingleProcessSInnerSizeAlign; // mm1ResGm两行之间的间隔
            
            uint64_t mm1ResGmOffset = (info.loop % (constInfo.preLoadNum)) * constInfo.mmResUbSize +
                                      (mStart + m) * info.actualSingleProcessSInnerSizeAlign + // m方向上偏移
                                      nStart; // n方向上偏移nStart个元素
            Fixpipe(mm1ResGm[mm1ResGmOffset], cL0Tensor, fixParams);
            if (kStart != 0) {
                SetAtomicNone();
            }
        }
        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + l0cBufId % 2);
        l0cBufId = (l0cBufId + 1) % 2;

        if (nStart + nDealSize >= nSize) {
            SetFlag<HardEvent::MTE1_MTE2>(QP_EVENT0 + qpBufId % 2);
        }
        qpBufId = (qpBufId + 1) % 2;
    }
    SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kvBufId % 2);
    kvBufId = (kvBufId + 1) % 2;
}

template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuant<FIAT>::DealMm2SingleMKN(const RunInfo &info,
                                                                uint32_t mSize, uint32_t mStart, uint32_t mDealSize,
                                                                uint32_t kSize, uint32_t kStart, uint32_t kDealSize,
                                                                uint32_t nSize, uint32_t nStart, uint32_t nDealSize)
{
    constexpr uint32_t M_BASE = 128;
    constexpr uint32_t K_BASE = 128;

    WaitFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kvBufId % 2);

    uint64_t l1Offset = (kvBufId % 2) * (L1_KV_SIZE / sizeof(KV_T));
    
    if (constInfo.systemPrefixFlag) {
        CopyPrefixAndValueToL1(info, kSize, kStart, kDealSize, nSize, nStart, nDealSize, l1Offset);
    } else {
        CopyValueToL1(info, kSize, kStart, kDealSize, nSize, nStart, nDealSize, l1Offset);
    }

    SetFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvBufId % 2);
    WaitFlag<HardEvent::MTE2_MTE1>(KV_EVENT0 + kvBufId % 2);
    for (uint32_t m = 0; m < mDealSize; m += M_BASE) {
        uint32_t mActSize = M_BASE;
        if (m + M_BASE > mDealSize) {
            mActSize = mDealSize - m;
        }
        uint32_t mActSizeAlign = AttentionCommon::Align(mActSize, 16U);
        if (nStart == 0) {
            WaitFlag<HardEvent::MTE1_MTE2>(QP_EVENT0 + qpBufId % 2);
            // CopySoftmaxResGmToL1
            {
                Nd2NzParams nd2nzPara;
                nd2nzPara.ndNum = 1;
                nd2nzPara.nValue = mActSize; // 行数
                nd2nzPara.dValue = kDealSize;
                nd2nzPara.srcDValue = info.actualSingleProcessSInnerSizeAlign;
                nd2nzPara.dstNzC0Stride = (mActSize + 15) / 16 * 16; // 对齐到16 单位block
                nd2nzPara.dstNzNStride = 1;
                nd2nzPara.srcNdMatrixStride = 0;
                nd2nzPara.dstNzMatrixStride = 0;

                uint64_t softmaxResL1BaseOffset = (qpBufId % 2) * (L1_QP_SIZE / sizeof(Q_T));
                uint64_t offset = (info.loop % (constInfo.preLoadNum)) * constInfo.mmResUbSize +
                    (mStart + m) * info.actualSingleProcessSInnerSizeAlign + kStart;
                DataCopy(qpL1Tensor[softmaxResL1BaseOffset], vec1ResGm[offset], nd2nzPara);
            }
            SetFlag<HardEvent::MTE2_MTE1>(QP_EVENT0 + qpBufId % 2);
            WaitFlag<HardEvent::MTE2_MTE1>(QP_EVENT0 + qpBufId % 2);
        } else {
            if (m == 0) {
                qpBufId = (qpBufId + ((mDealSize + M_BASE - 1) / M_BASE)) % 2;
            }
        }

        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + l0cBufId % 2);
        LocalTensor<T> cL0Tensor = cL0TensorPingPong[(l0cBufId % 2) * (L0C_PP_SIZE / sizeof(MM_OUT_T))];
        // matmul k
        for (uint32_t k = 0; k < kDealSize; k += K_BASE) {
            uint32_t kActSize = K_BASE;
            if (k + K_BASE > kDealSize) {
                kActSize = kDealSize - k;
            }

            WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + l0abBufId % 2);
            uint32_t kActSizeAlign = AttentionCommon::Align(kActSize, 16U);
            LocalTensor<Q_T> aL0Tensor = aL0TensorPingPong[(l0abBufId % 2) * (L0A_PP_SIZE / sizeof(Q_T))];
            LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(l0abBufId % 2) * (L0B_PP_SIZE / sizeof(KV_T))];
            // LoadAToL0
            {
                uint64_t qL1Offset = (qpBufId % 2) * (L1_QP_SIZE / sizeof(Q_T)) + k * mActSizeAlign;
                uint32_t mLoop = mActSizeAlign / 16;
                for (uint32_t i = 0; i < mLoop; i++) {
                    LoadData2DParams loadData2DParams;
                    loadData2DParams.startIndex = i;
                    loadData2DParams.repeatTimes = kActSizeAlign / 16;
                    loadData2DParams.srcStride = mActSizeAlign / 16;
                    loadData2DParams.dstGap = 0;
                    loadData2DParams.ifTranspose = false;
                    LoadData(aL0Tensor[16 * i * kActSizeAlign], qpL1Tensor[qL1Offset], loadData2DParams);
                }
            }
            // LoadBToL0
            {
                uint64_t kL1Offset = (kvBufId % 2) * (L1_KV_SIZE / sizeof(KV_T)) + k * 16U;
                uint32_t kLoop = kActSizeAlign / 16;
                for (uint32_t i = 0; i < kLoop; i++) {
                    LoadData2DParams loadData2DParams;
                    loadData2DParams.startIndex = i;
                    loadData2DParams.repeatTimes = AttentionCommon::Align(nDealSize, 16U) / 16;
                    loadData2DParams.srcStride = AttentionCommon::Align(kDealSize, 16U) / 16;
                    loadData2DParams.dstGap = 0;
                    loadData2DParams.ifTranspose = true;
                    LoadData(bL0Tensor[16 * i * AttentionCommon::Align(nDealSize, 16U)], kvL1Tensor[kL1Offset], loadData2DParams);
                }
            }
            SetFlag<HardEvent::MTE1_M>(L0AB_EVENT0 + l0abBufId % 2);
            WaitFlag<HardEvent::MTE1_M>(L0AB_EVENT0 + l0abBufId % 2);
            // MMAD
            {
                MmadParams mmadParams;
                mmadParams.m = mActSize;
                if (mmadParams.m == 1) {
                    mmadParams.m = 16;
                }
                mmadParams.n = nDealSize;
                mmadParams.k = kActSize;
                mmadParams.cmatrixInitVal = (k == 0);
                mmadParams.cmatrixSource = false;
                Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
            }
            AscendC::PipeBarrier<PIPE_M>();
            SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + l0abBufId % 2);
            l0abBufId = (l0abBufId + 1) % 2;
        }
        SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + l0cBufId % 2);
        WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + l0cBufId % 2);
        // FIXPIPE
        {
            if (kStart != 0) {
                SetAtomicAdd<MM_OUT_T>();
            }
            FixpipeParamsV220 fixParams;
            fixParams.mSize = mActSize;
            fixParams.nSize = nDealSize;
            fixParams.srcStride = mActSizeAlign;
            fixParams.dstStride = constInfo.headDimAlign; // mm2ResGm两行之间的间隔
            fixParams.ndNum = 1;

            uint64_t mm2ResGmOffset = (info.loop % (constInfo.preLoadNum)) * constInfo.bmm2ResUbSize +
                                      (mStart + m) * constInfo.headDimAlign + // m方向上偏移
                                      nStart; // n方向上偏移nStart个元素
            Fixpipe(mm2ResGm[mm2ResGmOffset], cL0Tensor, fixParams);
            if (kStart != 0) {
                SetAtomicNone();
            }
        }
        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + l0cBufId % 2);
        l0cBufId = (l0cBufId + 1) % 2;

        if (nStart + nDealSize >= nSize) {
            SetFlag<HardEvent::MTE1_MTE2>(QP_EVENT0 + qpBufId % 2);
        }
        qpBufId = (qpBufId + 1) % 2;
    }
    SetFlag<HardEvent::MTE1_MTE2>(KV_EVENT0 + kvBufId % 2);
    kvBufId = (kvBufId + 1) % 2;
}

template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuant<FIAT>::ComputeMm1(const RunInfo &info)
{
    if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
        if (info.isChangeBatch) {
            UpdateKey(info.bIdx);
        }
    }
    constexpr uint32_t M_SPLIT_SIZE = 256;
    constexpr uint32_t K_SPLIT_SIZE = 512;
    constexpr uint32_t N_SPLIT_SIZE = 128;

    uint32_t mSize = info.actMBaseSize;
    uint32_t kSize = constInfo.headDim + constInfo.headDimRope;
    uint32_t nSize = info.actualSingleProcessSInnerSize;

    if (constInfo.ropeSplitMode) {
        kSize = AttentionCommon::Align(constInfo.headDim, 16UL) + constInfo.headDimRope;
    }
    for (uint32_t kStart = 0; kStart < kSize; kStart += K_SPLIT_SIZE) {
        uint32_t kDealSize = K_SPLIT_SIZE;
        if (kStart + K_SPLIT_SIZE > kSize) {
            kDealSize = kSize - kStart;
        }

        // 适配ROPE分离模式, 需要根据D的起始点和D方向拷贝长度, 拆分出需要在nope和rope上分别在D方向上拷贝的数据长度
        uint32_t nopeDealSize = kDealSize;
        uint32_t ropeDealSize = 0;
        if (constInfo.ropeSplitMode) {
            if ((kStart < constInfo.headDim) && (kStart + kDealSize > constInfo.headDim)) {
                nopeDealSize = constInfo.headDim - kStart;
                // kDealSize -= AttentionCommon::Align(constInfo.headDim, 16UL) - constInfo.headDim;
                ropeDealSize = kStart + kDealSize - AttentionCommon::Align(constInfo.headDim, 16UL);
            } else if (kStart >= constInfo.headDim) {
                nopeDealSize = 0;
                ropeDealSize = kDealSize;
            }
        }

        for (uint32_t mStart = 0; mStart < mSize; mStart += M_SPLIT_SIZE) {
            uint32_t mDealSize = M_SPLIT_SIZE;
            if (mStart + M_SPLIT_SIZE > mSize) {
                mDealSize = mSize - mStart;
            }

            for (uint32_t nStart = 0; nStart < nSize; nStart += N_SPLIT_SIZE) {
                uint32_t nDealSize = N_SPLIT_SIZE;
                if (nStart + N_SPLIT_SIZE > nSize) {
                    nDealSize = nSize - nStart;
                }

                DealMm1SingleMKN(info, mSize, mStart, mDealSize, kSize, kStart, kDealSize, nSize, nStart, nDealSize,
                                 nopeDealSize, ropeDealSize);
            }
        }
    }
    CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
}

template <typename FIAT>
__aicore__ inline void FiaBlockCubeNonQuant<FIAT>::ComputeMm2(const RunInfo &info)
{
    if constexpr (GmLayoutParams<KV_FORMAT>::CATEGORY == FormatCategory::GM_KV_BNSD) {
        if (info.isChangeBatch) {
            UpdateValue(info.bIdx);
        }
    }
    CrossCoreWaitFlag(constInfo.syncV1C2);

    constexpr uint32_t M_SPLIT_SIZE = 256;
    constexpr uint32_t K_SPLIT_SIZE = 512;
    constexpr uint32_t N_SPLIT_SIZE = 128;

    uint32_t mSize = info.actMBaseSize;
    uint32_t kSize = info.actualSingleProcessSInnerSize;
    uint32_t nSize = constInfo.headDim;
    for (uint32_t kStart = 0; kStart < kSize; kStart += K_SPLIT_SIZE) {
        uint32_t kDealSize = K_SPLIT_SIZE;
        if (kStart + K_SPLIT_SIZE > kSize) {
            kDealSize = kSize - kStart;
        }

        for (uint32_t mStart = 0; mStart < mSize; mStart += M_SPLIT_SIZE) {
            uint32_t mDealSize = M_SPLIT_SIZE;
            if (mStart + M_SPLIT_SIZE > mSize) {
                mDealSize = mSize - mStart;
            }

            for (uint32_t nStart = 0; nStart < nSize; nStart += N_SPLIT_SIZE) {
                uint32_t nDealSize = N_SPLIT_SIZE;
                if (nStart + N_SPLIT_SIZE > nSize) {
                    nDealSize = nSize - nStart;
                }

                DealMm2SingleMKN(info, mSize, mStart, mDealSize, kSize, kStart, kDealSize, nSize, nStart, nDealSize);
            }
        }
    }

    CrossCoreSetFlag<ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V2);
}

#endif
