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
 * \file prompt_flash_attention_s1s2_bns1_x310_base.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_S1S2_BNS1_X310_BASE_H
#define PROMPT_FLASH_ATTENTION_S1S2_BNS1_X310_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"
#include "kernel_data_copy_transpose.h"
#include "kernel_operator_softmax_compute_nz.h"

using namespace matmul;
constexpr uint32_t BATCH_NUM_MAX_NZ = 300; // 300 is batch size high limit
constexpr int32_t MAX_REPEATS_PER_BATCH = 255; // max repeatTime in InitConstValue
constexpr int32_t REPEAT_DATASIZE_EACH_TIME = 512; // processing a fixed amount of data per iteration
constexpr static uint32_t NEGATIVE_MIN_VAULE_FP32 = 0xFF7FFFFF;
constexpr static uint32_t NEGATIVE_MIN_VAULE_FP16 = 0xC61C4000;

enum ModeNZ {
    HighPrecisionNZ = 0,
    HighPerformanceNZ
};

enum class PFALayoutNZ {
    BSH = 0,
    BNSD,
};

template <PFALayoutNZ L, typename T, typename U, typename O = T, typename KV_T = T, ModeNZ M = ModeNZ::HighPerformanceNZ, typename...Args>
struct PFATypeNZ {
    using inputType = T;
    using maskType = U;
    using outputType = O;
    using kvInputType = KV_T;
    static constexpr PFALayoutNZ layout = L;
    static constexpr ModeNZ calcMode = M;
};

template<typename T, ModeNZ M = ModeNZ::HighPerformanceNZ>
struct PromptFlashAttentionTypeTraitsNZ
{
    using mmInputType = T;
    using mmBiasType = T;
    using mmOutputType = T;
    using softmaxType = std::conditional_t<M == ModeNZ::HighPrecisionNZ, float, T>;
    using mm1OutputType = std::conditional_t<M == ModeNZ::HighPrecisionNZ, float, T>;
};

constexpr uint32_t BOOLBYTENUM_NZ = 32U;
constexpr uint32_t UB_ALIGN_NZ = 32U;
constexpr uint32_t FP16BYTENUM = 16U;

template<typename PFAT>
class PromptFlashAttentionS1s2Bns1X310Base {
public:
    __aicore__ inline PromptFlashAttentionS1s2Bns1X310Base() {};
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
                                __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* blocktable,
                                __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                                __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                                __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
                                const PromptFlashAttentionTilingData* __restrict tiling, __gm__ uint8_t* gmTiling, TPipe* tPipe);
    __aicore__ inline void InitMsd(__gm__ uint8_t* key_antiquant_scale, __gm__ uint8_t* key_antiquant_offset, __gm__ uint8_t* value_antiquant_scale, __gm__ uint8_t* value_antiquant_offset);
    using T = typename PFAT::inputType;
    using U = typename PFAT::maskType;
    using O = typename PFAT::outputType;

    __aicore__ inline void Process();
    // define datatype
    using mmInputType = typename PromptFlashAttentionTypeTraitsNZ<T, PFAT::calcMode>::mmInputType;
    using mmBiasType = typename PromptFlashAttentionTypeTraitsNZ<T, PFAT::calcMode>::mmBiasType;
    using mmOutputType = typename PromptFlashAttentionTypeTraitsNZ<T, PFAT::calcMode>::mmOutputType;
    using mm1OutputType = typename PromptFlashAttentionTypeTraitsNZ<T, PFAT::calcMode>::mm1OutputType;
    using softmaxType = typename PromptFlashAttentionTypeTraitsNZ<T, PFAT::calcMode>::softmaxType;
    // define matmul
    using a1Type = MatmulType<TPosition::A1, CubeFormat::NZ, mmInputType, false>;
    using b1Type = MatmulType<TPosition::A1, CubeFormat::NZ, mmInputType, true>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, mmBiasType>;
    using c1Type = MatmulType<TPosition::VECCALC, CubeFormat::NZ, mm1OutputType>;
    Matmul<a1Type, b1Type, c1Type, bias1Type> mm;
    // define batchmatmul
    using a2Type = MatmulType<TPosition::VECCALC, CubeFormat::NZ, mmInputType, false>;
    using b2Type = MatmulType<TPosition::A1, CubeFormat::NZ, mmInputType, false>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, mmBiasType>;
#if defined(__CCE_KT_TEST__)
    using c2Type = MatmulType<TPosition::VECCALC, CubeFormat::NZ, mmOutputType>; // cpu
#else
    using c2Type = MatmulType<TPosition::VECCALC, CubeFormat::NZ, mmOutputType>; // npu
#endif
    Matmul<a2Type, b2Type, c2Type, bias2Type> bmm2;

protected:
    const PromptFlashAttentionTilingData* __restrict tilingData;
    TPipe* pipe;
    // define the que
    TBuf<> attenMaskUb_;
    TQue<QuePosition::VECIN, 1> eleWiseInQueue;
    TQue<QuePosition::VECOUT, 1> tempBmm2Queue;
    TQue<QuePosition::VECOUT, 1> Bmm1Queue;
    TQue<QuePosition::VECOUT, 1> softmaxOutQueue;
    TBuf<QuePosition::A1> a1Buf_;
    TBuf<QuePosition::A1> b1Buf_;
    TBuf<QuePosition::A1> c1Buf_;
    TBuf<> selectSpaceUb;

    TBuf<> softmaxExpUb_;
    TBuf<> tempBmm2Ub;

    TBuf<> tmpSoftmaxFlashV2Ub_;
    TBuf<> tmpmm2Ub_;
    LocalTensor<mmOutputType> softmaxExpUb16_;
    LocalTensor<softmaxType> softmaxExpUb;
    LocalTensor<mmInputType> a1Local_;
    LocalTensor<mmInputType> b1Local_;
    LocalTensor<mmInputType> c1Local_;

    LocalTensor<mmOutputType> softmaxMaxUb16_;
    LocalTensor<mmOutputType> softmaxSumUb16_;
    LocalTensor<float> softmaxMaxUb32_;
    LocalTensor<float> softmaxSumUb32_;
    GlobalTensor<T> queryGm;
    GlobalTensor<T> keyGm;
    GlobalTensor<T> valueGm;
    GlobalTensor<U> attenMaskGm;
    GlobalTensor<O> attentionOutGm;
    GlobalTensor<int64_t> actualSeqLengthsGm;
    GlobalTensor<int64_t> actualSeqLengthsKVGm;
    GlobalTensor<softmaxType> workspaceGm;
    GlobalTensor<mmOutputType> workspaceGmProcessT;

    // quant: define quant variable
    uint64_t dequantScale1;
    float quantScale1;
    uint64_t dequantScale2;
    float quantScale2;
    float quantOffset2;

    bool isInnerLoopLast_ = false;
    bool isNextInnerLoopLast_ = false; // record second last inner loop
    bool isOuterLoopLast_ = false;
    bool isOuterLoopStart_ = false;
    bool isOuterTail_ = false; // record outerloop tail
    bool isNextOuterLoopLast_ = false; // record second last outer loop
    int32_t fetchOuterSize_ = -1; // record next loop outer size

    bool useMask;
    bool needCalMask_ = false;
    bool isHighPrecision_ = true;
    uint32_t tmp_block_idx;
    uint32_t loopSNum;
    int32_t sOuterOffset;
    uint32_t batchNOffset;
    uint32_t maskOffset;
    uint32_t maskCoreOffset;
    uint64_t attenMaskCoreOffset;
    uint64_t valueOffset;
    uint64_t valueCoreOffset;
    uint64_t attenMaskOffset;
    uint64_t tensorAOffset;
    uint64_t tensorBOffset;
    uint64_t tensorACoreOffset;
    uint64_t tensorBCoreOffset;
    uint64_t attentionOutOffset;
    uint64_t offsetSS;
    uint64_t offsetSH;
    uint32_t offsetSTypeNum;
    uint32_t offsetNSTypeNum;
    uint64_t offsetNSS;
    uint64_t offsetNSH;
    uint32_t maskDataType;
    uint32_t attenMaskBatch;
    uint32_t maskCopyInCol;

    // tilingdata
    uint32_t singleProcessSOuterSize;
    uint32_t singleProcessSOuterSizeTail;
    uint32_t singleProcessSOuterSizeTailAlign;

    uint32_t singleProcessSInnerSize;
    uint32_t singleProcessSInnerSizeTail;
    uint32_t singleProcessSInnerSizeTailAlign;

    uint32_t mmResUbSize;
    uint32_t attenMaskUbSize;
    uint32_t maskSize;
    uint32_t softmaxMaxSize;
    uint32_t softmaxSumSize;
    uint32_t softmaxExpSize;
    uint32_t spmTmpSize;
    uint32_t scmTmpSize;
    uint32_t bmm2ResUbSize;
    uint32_t tmpMMResBmm2PreUbSize;
    uint32_t tmpSoftmaxBmm2UbSize;
    uint32_t padSize;
    uint32_t typeByteNum;
    uint32_t outputTypeByteNum;
    uint32_t softmaxTypeByteNum;
    uint32_t headNumRatio;
    uint32_t maskTypeByteNum;
    uint32_t selectSpaceUbSize;
    uint32_t softMaxV2Size_;
    uint32_t mm2TmpUbSize_;
    uint32_t splitS2;
    uint32_t layoutType;
    uint32_t maskInnerTailAlign;
    uint32_t negativeScalar = NEGATIVE_MIN_VAULE_FP32;
    bool isSoftmaxResNeedUpdate;

    SoftMaxTiling softmaxTilingData;
    SoftMaxTiling softmaxFlashTilingData;
    CopyTransposeTiling transposeTilingData;
    uint32_t MultiHeadQ;
    uint32_t MultiHeadKV;
    uint32_t startInBlock;
    uint32_t maxInnerLoopTimes;
    uint32_t seqListOffset;
    int64_t multiSeqOffset;

    uint32_t attentionMaskStride;
    int32_t attentionMaskType;
    uint32_t attentionMaskMaxSize;

    bool isActualLenDimsNull;
    bool isActualLenDimsKVNull;
    int32_t actualSeqLengthPerBatch;
    int32_t actualSeqLengthKVPerBatch;
    uint32_t accumSOuterTilingNums[BATCH_NUM_MAX_NZ];
    uint32_t actualSeqOffsets[BATCH_NUM_MAX_NZ];

    uint32_t queryStride;
    uint32_t keyValueStride;

    __aicore__ inline void ElewiseCompute310P(LocalTensor<mm1OutputType>& mmResUb, uint32_t SInnerSize, uint32_t SOuterSize);

    __aicore__ inline void AttenMaskCopyIn(uint64_t offset, uint32_t sinnerSize, uint32_t sInnerIdx);

    __aicore__ inline void AttenMaskTransND2NZ(uint32_t SInnerSize, uint32_t SOuterSize);

    __aicore__ inline void SoftmaxBasicComputeFirstTail(LocalTensor<mm1OutputType>& mmResUb,
                                                          LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                                          LocalTensor<softmaxType>& softmaxExpUb, LocalTensor<uint8_t>& sharedTmpUb,
                                                          const SoftMaxShapeInfo &softmaxShapeInfo);
    __aicore__ inline void SoftmaxBasicComputeTail(LocalTensor<mm1OutputType>& mmResUb,
                                                     LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                                     LocalTensor<softmaxType>& softmaxExpUb, LocalTensor<uint8_t>& sharedTmpUb,
                                                     const SoftMaxShapeInfo &softmaxShapeInfo);
    __aicore__ inline void SoftmaxBasicComputeFirstTailTmp(LocalTensor<mmOutputType>& mmResUb,
                                                          LocalTensor<mmOutputType>& softmaxMaxUb, LocalTensor<mmOutputType>& softmaxSumUb,
                                                          LocalTensor<mmOutputType>& softmaxExpUb, LocalTensor<uint8_t>& sharedTmpUb,
                                                          const SoftMaxShapeInfo &softmaxShapeInfo);
    __aicore__ inline void SoftmaxBasicComputeTailTmp(LocalTensor<mmOutputType>& mmResUb,
                                                     LocalTensor<mmOutputType>& softmaxMaxUb, LocalTensor<mmOutputType>& softmaxSumUb,
                                                     LocalTensor<mmOutputType>& softmaxExpUb, LocalTensor<uint8_t>& sharedTmpUb,
                                                     const SoftMaxShapeInfo &softmaxShapeInfo);
    __aicore__ inline void QuantCompute(LocalTensor<int8_t> quantResUb, LocalTensor<mmOutputType> mmResUb, float scale, float offset, uint32_t computeSize);

    __aicore__ inline void CopyND2NZOnTheFly(const LocalTensor<mmOutputType>& dst,  const GlobalTensor<mmOutputType>& src,
        const int height, const int width, const int gCol, const bool isA1);

    __aicore__ inline void Bmm2UpdateDivNoTail310P(LocalTensor<mmOutputType>& bmm2ResPreUb,
                                               LocalTensor<float>& softmaxSumUb, LocalTensor<softmaxType>& softmaxExpUb);

    __aicore__ inline void Bmm2UpdateDivNoTail310PTmp(LocalTensor<mmOutputType>& bmm2ResPreUb,
                                               LocalTensor<mmOutputType>& softmaxSumUb, LocalTensor<softmaxType>& softmaxExpUb);

    __aicore__ inline void Bmm2Compute(LocalTensor<mmOutputType>& bmm2ResL1);

    __aicore__ inline void UpdateVmul(LocalTensor<softmaxType>& softmaxExpUb);

    __aicore__ inline void Bmm2UpdateAdd(LocalTensor<mmOutputType>& bmm2ResUb);

    __aicore__ inline void DataCopyTransposeOut(LocalTensor<mmOutputType>& bmm2ResUb);

    __aicore__ inline void ComputeAttenMaskOffset(uint32_t sInnerLoopIdx, bool isLast);

    __aicore__ inline void ComputeOffset(uint32_t sInnerLoopIdx, bool isLast);

    __aicore__ inline void LoopSOuterOffsetInitWithBSH(uint64_t seqListOffsetSize, int sIdx);

    __aicore__ inline void LoopSOuterOffsetInitWithBNSD(uint64_t seqListOffsetSize, int sIdx);

    __aicore__ inline void LoopSOuterOffsetInit(uint64_t seqListOffsetSize, int sIdx);

    __aicore__ inline void initOffset();

    __aicore__ inline void InitTensorSize(const PromptAttentionSingleCoreTensorSize* tensorSizeTiling);

    __aicore__ inline void GetSingleCoreParam(int sIdx);

    __aicore__ inline void InitOutputSingleCore();

    __aicore__ inline void Bmm1Compute(LocalTensor<mmInputType>& a1Local, LocalTensor<mmInputType>& b1Local, int32_t singleM, int32_t singleN, int32_t singleK);
};

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::Init(__gm__ uint8_t* query, __gm__ uint8_t* key,
                                        __gm__ uint8_t* value, __gm__ uint8_t* pseShift,
                                        __gm__ uint8_t* attenMask, __gm__ uint8_t* actualSeqLengths,
                                        __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* blocktable,
                                        __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                                        __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                                        __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
                                        const PromptFlashAttentionTilingData* __restrict tiling, __gm__ uint8_t* gmTiling,
                                        TPipe* tPipe) {
    tmp_block_idx = GetBlockIdx();
    // init global buffer
    tilingData = tiling;
    queryGm.SetGlobalBuffer((__gm__ T*)query);
    keyGm.SetGlobalBuffer((__gm__ T*)key);
    valueGm.SetGlobalBuffer((__gm__ T*)value);
    attentionOutGm.SetGlobalBuffer((__gm__ O*)attentionOut);
    workspaceGm.SetGlobalBuffer((__gm__ softmaxType*)workspace);

    pipe = tPipe;
    typeByteNum = tilingData->promptAttentionBaseParams.typeByteNum;
    outputTypeByteNum = tilingData->promptAttentionBaseParams.outputTypeByteNum;
    softmaxTypeByteNum = tilingData->promptAttentionBaseParams.softmaxTypeByteNum;
    headNumRatio = tilingData->promptAttentionBaseParams.headNumRatio;
    maskDataType = tilingData->promptAttentionBaseParams.attenMaskElemType;
    maskTypeByteNum = tilingData->promptAttentionBaseParams.maskTypeByteNum;
    attenMaskBatch = tilingData->promptAttentionSingleCoreParams.attenMaskBatch;
    layoutType = tilingData->promptAttentionBaseParams.layoutType;
    initOffset();

    isActualLenDimsNull = true;
    isActualLenDimsKVNull = true;
    if (!tilingData->promptAttentionBaseParams.isActualSeqLengthsNull) { // actual seq length is null
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengths, tilingData->promptAttentionBaseParams.batchSize);
        isActualLenDimsNull = false;
    }
    if (!tilingData->promptAttentionBaseParams.isActualSeqLengthsKVNull) {
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengthsKV, tilingData->promptAttentionBaseParams.batchSize);
        isActualLenDimsKVNull = false;
    }

    uint32_t preAccumSOuter = 0;
    uint64_t h = static_cast<uint64_t>(tilingData->promptAttentionBaseParams.headNumSize) *
        tilingData->promptAttentionBaseParams.headSize;
    uint64_t s = tilingData->promptAttentionBaseParams.seqSize;
    uint64_t middle_actualSeqLengths = 0;
    uint32_t actualSeqLengthsIdx = 0;

    if constexpr (IsSameType<T, half>::value) {
        this->negativeScalar = NEGATIVE_MIN_VAULE_FP16;
    }

    for (int i = 0; i < tilingData->promptAttentionBaseParams.batchSize; i++) {
        actualSeqLengthsIdx = isActualLenDimsNull ? tilingData->promptAttentionBaseParams.seqSize : actualSeqLengthsGm.GetValue(i);
        if (tilingData->promptAttentionBaseParams.isActualSeqLengthsNull) {
            actualSeqOffsets[i] = static_cast<uint64_t>(i) * s * h;
        } else {
            if (tilingData->promptAttentionBaseParams.isLayoutSH) {
                actualSeqOffsets[i] = middle_actualSeqLengths * h;
                middle_actualSeqLengths += actualSeqLengthsIdx;
            } else {
                actualSeqOffsets[i] = static_cast<uint64_t>(i) * s * h;
            }
        }

        actualSeqLengthsIdx = ((int64_t)actualSeqLengthsIdx >
                               (int64_t)tilingData->promptAttentionBaseParams.seqInnerSize +
                               (int64_t)tilingData->promptAttentionBaseParams.preTokens) ?
                               tilingData->promptAttentionBaseParams.seqInnerSize + tilingData->promptAttentionBaseParams.preTokens :
                               actualSeqLengthsIdx;
        accumSOuterTilingNums[i] = (((actualSeqLengthsIdx + tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize - 1) /
                            tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize) *
                            tilingData->promptAttentionBaseParams.headNumSize) +
                            preAccumSOuter;
        preAccumSOuter = accumSOuterTilingNums[i];
    }
    accumSOuterTilingNums[0] = (headNumRatio != 1 ||
                                tilingData->promptAttentionInitOutputParams.needInit ||
                                tilingData->promptAttentionBaseParams.batchSize != 1) ?
                                0 : accumSOuterTilingNums[0];

    if (tilingData->promptAttentionBaseParams.sparseMode == 99 && PFAT::calcMode == ModeNZ::HighPerformanceNZ) { // approximate calculation
        isHighPrecision_ = false;
    }
    pipe->InitBuffer(a1Buf_, tilingData->promptAttentionTensorSizeRect.scmTmpSize * sizeof(mmInputType));
    pipe->InitBuffer(b1Buf_, tilingData->promptAttentionTensorSizeRect.scmTmpSize * sizeof(mmInputType));
    pipe->InitBuffer(c1Buf_, tilingData->promptAttentionTensorSizeRect.scmTmpSize * sizeof(mmInputType));
    pipe->InitBuffer(attenMaskUb_, 2 * (tilingData->promptAttentionTensorSizeRect.attenMaskUbSize) * sizeof(U));
    if (!isHighPrecision_) {
        pipe->InitBuffer(softmaxOutQueue, 1, 2 * tilingData->promptAttentionTensorSizeRect.softmaxMaxSize * sizeof(mmOutputType));
    } else {
        pipe->InitBuffer(softmaxOutQueue, 1, 2 * tilingData->promptAttentionTensorSizeRect.softmaxMaxSize * sizeof(float));
    }
    pipe->InitBuffer(softmaxExpUb_, tilingData->promptAttentionTensorSizeRect.softmaxExpSize * sizeof(softmaxType));
    softmaxExpUb = softmaxExpUb_.Get<softmaxType>(tilingData->promptAttentionTensorSizeRect.softmaxExpSize);

    pipe->InitBuffer(tempBmm2Ub, tilingData->promptAttentionTensorSizeRect.bmm2ResUbSize * sizeof(mmOutputType));
    pipe->InitBuffer(tempBmm2Queue, 1, tilingData->promptAttentionTensorSizeRect.bmm2ResUbSize * sizeof(mmOutputType));

    pipe->InitBuffer(Bmm1Queue, 1, tilingData->promptAttentionTensorSizeRect.mmResUbSize * sizeof(mm1OutputType));
    if ((tilingData->promptAttentionTensorSizeRect.tmpSoftMaxV2Size) != 0 && (tilingData->promptAttentionTensorSizeRect.mm2TmpUbSize != 0)) {
        pipe->InitBuffer(tmpSoftmaxFlashV2Ub_, tilingData->promptAttentionTensorSizeRect.tmpSoftMaxV2Size / UB_ALIGN_NZ * UB_ALIGN_NZ * sizeof(uint8_t));
        pipe->InitBuffer(tmpmm2Ub_, tilingData->promptAttentionTensorSizeRect.softmaxExpSize * sizeof(mmOutputType) * 2);
    }

    attentionMaskType = tilingData->promptAttentionBaseParams.sparseMode;
    if ((attenMask != NULL) && (tilingData->promptAttentionBaseParams.useMask == 1)) {
        needCalMask_ = true;
        attenMaskGm.SetGlobalBuffer((__gm__ U*)attenMask);
        attentionMaskStride = tilingData->promptAttentionBaseParams.maskKVsSize;
    }

    if (tilingData->promptAttentionInitOutputParams.needInit == 1) {
        InitOutputSingleCore();
    }

    if constexpr (PFAT::layout == PFALayoutNZ::BSH) {
        // MultiHeadQ
        queryStride = tilingData->promptAttentionBaseParams.headSize * tilingData->promptAttentionBaseParams.headNumSize;
        // MultiHeadKV
        keyValueStride = queryStride / headNumRatio;
    } else { // BNSD
        queryStride = tilingData->promptAttentionBaseParams.headSize;
        keyValueStride = tilingData->promptAttentionBaseParams.headSize;
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::InitMsd(__gm__ uint8_t* key_antiquant_scale, __gm__ uint8_t* key_antiquant_offset, __gm__ uint8_t* value_antiquant_scale, __gm__ uint8_t* value_antiquant_offset) {
    return;
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::InitOutputSingleCore()
{
    auto &initParams = tilingData->promptAttentionInitOutputParams;
    uint32_t tailSize = initParams.totalOutputSize - tmp_block_idx * initParams.singleCoreSize;
    uint32_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;
    InitOutput<O>(attentionOutGm[tmp_block_idx * initParams.singleCoreSize], singleInitOutputSize, 0);
    SyncAll();
}

template<>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFATypeNZ<PFALayoutNZ::BNSD, int8_t, bool, int8_t>>::InitOutputSingleCore() {}
template<>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFATypeNZ<PFALayoutNZ::BSH, int8_t, bool, int8_t>>::InitOutputSingleCore() {}

template<>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFATypeNZ<PFALayoutNZ::BNSD, int8_t, half, int8_t>>::InitOutputSingleCore() {}
template<>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFATypeNZ<PFALayoutNZ::BSH, int8_t, half, int8_t>>::InitOutputSingleCore() {}

template<>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFATypeNZ<PFALayoutNZ::BNSD, int8_t, float, int8_t>>::InitOutputSingleCore() {}
template<>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFATypeNZ<PFALayoutNZ::BSH, int8_t, float, int8_t>>::InitOutputSingleCore() {}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::initOffset() {
    offsetSS = static_cast<uint64_t>(tilingData->promptAttentionBaseParams.seqSize) *
        tilingData->promptAttentionBaseParams.seqSize;
    offsetSH = static_cast<uint64_t>(tilingData->promptAttentionBaseParams.seqSize) *
        tilingData->promptAttentionBaseParams.headSize;
    offsetSTypeNum = static_cast<uint64_t>(tilingData->promptAttentionBaseParams.seqSize) * typeByteNum;
    offsetNSTypeNum = static_cast<uint64_t>(tilingData->promptAttentionBaseParams.headNumSize) * offsetSTypeNum;
    offsetNSS = static_cast<uint64_t>(tilingData->promptAttentionBaseParams.headNumSize) * offsetSS;
    offsetNSH = static_cast<uint64_t>(tilingData->promptAttentionBaseParams.headNumSize) * offsetSH;
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::InitTensorSize(
                const PromptAttentionSingleCoreTensorSize* tensorSizeTiling) {
    mmResUbSize = tensorSizeTiling->mmResUbSize;
    attenMaskUbSize = tensorSizeTiling->attenMaskUbSize;
    maskSize = tensorSizeTiling->maskSize;
    softmaxMaxSize = tensorSizeTiling->softmaxMaxSize;
    softmaxSumSize = tensorSizeTiling->softmaxSumSize;
    softmaxExpSize = tensorSizeTiling->softmaxExpSize;
    spmTmpSize = tensorSizeTiling->spmTmpSize;
    scmTmpSize = tensorSizeTiling->scmTmpSize;
    bmm2ResUbSize = tensorSizeTiling->bmm2ResUbSize;
    tmpMMResBmm2PreUbSize = tensorSizeTiling->tmpMMResBmm2PreUbSize;
    tmpSoftmaxBmm2UbSize = tensorSizeTiling->tmpSoftmaxBmm2UbSize;
    selectSpaceUbSize = tensorSizeTiling->selectSpaceUbSize;
    softMaxV2Size_ = tensorSizeTiling->tmpSoftMaxV2Size / UB_ALIGN_NZ * UB_ALIGN_NZ;
    mm2TmpUbSize_ = tensorSizeTiling->mm2TmpUbSize;
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::AttenMaskCopyIn(uint64_t offset,
                                                                                             uint32_t sinnerSize,
                                                                                             uint32_t sInnerLoopIdx) {
    LocalTensor<U> attenMaskUb = this->attenMaskUb_.template Get<U>(this->attenMaskUbSize);
    attenMaskUb.SetSize(this->singleProcessSOuterSize * sinnerSize);
    DataCopyParams intriParams;
    intriParams.blockCount = this->singleProcessSOuterSize;
    intriParams.blockLen = sinnerSize / this->maskTypeByteNum;
    intriParams.srcStride = (this->tilingData->promptAttentionBaseParams.seqInnerSize - sinnerSize) /
                            this->maskTypeByteNum;
    intriParams.dstStride = 0;

    DataCopy(attenMaskUb, this->attenMaskGm[offset], intriParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    this->AttenMaskTransND2NZ(this->singleProcessSInnerSize, this->singleProcessSOuterSize);
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::AttenMaskTransND2NZ(uint32_t SInnerSize, uint32_t SOuterSize)
{
    struct DataCopyParams dataCopyParams;
    LocalTensor<int8_t> tmpUb2 = this->attenMaskUb_.template Get<int8_t>(this->attenMaskUbSize);
    LocalTensor<mmOutputType> tmpUb = this->tmpSoftmaxFlashV2Ub_.template Get<mmOutputType>(this->attenMaskUbSize);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(tmpUb, tmpUb2, RoundMode::CAST_NONE, tmpUb.GetSize());
    PipeBarrier<PIPE_V>();
    int32_t calHigh = (SOuterSize + BLOCK_CUBE - 1) / BLOCK_CUBE;
    dataCopyParams.blockCount = SOuterSize;
    dataCopyParams.blockLen = 1;
    dataCopyParams.srcStride = SInnerSize / BLOCK_CUBE - 1;
    dataCopyParams.dstStride = 0;
    LocalTensor<mmOutputType> attenMaskUb = this->attenMaskUb_.template Get<mmOutputType>(this->attenMaskUbSize);
    for(int i = 0; i < calHigh; i++) {
        DataCopy(attenMaskUb[i * BLOCK_CUBE * singleProcessSOuterSize], tmpUb[i * BLOCK_CUBE], dataCopyParams);
    }
    PipeBarrier<PIPE_V>();
    Muls(attenMaskUb, attenMaskUb, static_cast<mmOutputType>(-10000.0), SOuterSize * SInnerSize);
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::ElewiseCompute310P(LocalTensor<mm1OutputType>& mmResUb, uint32_t SInnerSize, uint32_t SOuterSize) {
    uint32_t computeSize = SInnerSize * SOuterSize;

    Muls(mmResUb, mmResUb, static_cast<mm1OutputType>(tilingData->promptAttentionBaseParams.scaleValue), computeSize);
    PipeBarrier<PIPE_V>();
    if (needCalMask_) {
        if constexpr (IsSameType<mm1OutputType, float>::value) {
            LocalTensor<mmOutputType> attenMaskUb = this->attenMaskUb_.template Get<mmOutputType>(this->attenMaskUbSize);
            LocalTensor<mm1OutputType> tmpUb = this->tmpSoftmaxFlashV2Ub_.template Get<mm1OutputType>(this->attenMaskUbSize);
            Cast(tmpUb, attenMaskUb, RoundMode::CAST_NONE, tmpUb.GetSize());
            PipeBarrier<PIPE_V>();
            Add(mmResUb, mmResUb, tmpUb, computeSize);
            PipeBarrier<PIPE_V>();
        }
        else{
            LocalTensor<mmOutputType> tmpUb = this->attenMaskUb_.template Get<mmOutputType>(this->attenMaskUbSize);
            Add(mmResUb, mmResUb, tmpUb, computeSize);
            PipeBarrier<PIPE_V>();
        }
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::SoftmaxBasicComputeFirstTail(LocalTensor<mm1OutputType>& mmResUb,
                                                        LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                                        LocalTensor<softmaxType>& softmaxExpUb, LocalTensor<uint8_t>& sharedTmpUb,
                                                        const SoftMaxShapeInfo &softmaxShapeInfo) {
    SoftmaxFlashV2<softmaxType, false, true, true, true>(mmResUb, softmaxSumUb, softmaxMaxUb,
                                         mmResUb, softmaxExpUb, softmaxSumUb, softmaxMaxUb, sharedTmpUb, softmaxFlashTilingData, softmaxShapeInfo);
    if (this->isSoftmaxResNeedUpdate) {
        SoftMaxShapeInfo softmaxShapeInfo{
            static_cast<uint32_t>(tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize),
            static_cast<uint32_t>(tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize),
            static_cast<uint32_t>(tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize),
            static_cast<uint32_t>(tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize)
        };
        this->isSoftmaxResNeedUpdate = AdjustSoftMaxRes<mm1OutputType, float, true>(mmResUb,
            softmaxMaxUb, this->negativeScalar, 0.0, softmaxShapeInfo);
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::SoftmaxBasicComputeTail(LocalTensor<mm1OutputType>& mmResUb,
                                                    LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                                    LocalTensor<softmaxType>& softmaxExpUb, LocalTensor<uint8_t>& sharedTmpUb,
                                                    const SoftMaxShapeInfo &softmaxShapeInfo) {
    SoftmaxFlashV2<softmaxType, true, true, true, true>(mmResUb, softmaxSumUb, softmaxMaxUb,
                                        mmResUb, softmaxExpUb, softmaxSumUb, softmaxMaxUb, sharedTmpUb, softmaxFlashTilingData, softmaxShapeInfo);
    if (this->isSoftmaxResNeedUpdate) {
        // Updates the softmaxSapeInfo using parameters from tilingData
        SoftMaxShapeInfo softmaxShapeInfo{
            static_cast<uint32_t>(tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize),
            static_cast<uint32_t>(tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize),
            static_cast<uint32_t>(tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize),
            static_cast<uint32_t>(tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize)
        };
        // Invokes the AdjustSoftMaxRes function to update the softmax results
        this->isSoftmaxResNeedUpdate = AdjustSoftMaxRes<mm1OutputType, float, true>(mmResUb,
            softmaxMaxUb, this->negativeScalar, 0.0, softmaxShapeInfo);
    }
                                                    }
template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::SoftmaxBasicComputeFirstTailTmp(LocalTensor<mmOutputType>& mmResUb,
                                                        LocalTensor<mmOutputType>& softmaxMaxUb, LocalTensor<mmOutputType>& softmaxSumUb,
                                                        LocalTensor<mmOutputType>& softmaxExpUb, LocalTensor<uint8_t>& sharedTmpUb,
                                                        const SoftMaxShapeInfo &softmaxShapeInfo) {
    SoftmaxFlashV2Tmp<mmOutputType, false, true, true, true>(mmResUb, softmaxSumUb, softmaxMaxUb,
                                         mmResUb, softmaxExpUb, softmaxSumUb, softmaxMaxUb, sharedTmpUb, softmaxFlashTilingData, softmaxShapeInfo);
    if (this->isSoftmaxResNeedUpdate) {
        SoftMaxShapeInfo softmaxShapeInfo{
            static_cast<uint32_t>(tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize),
            static_cast<uint32_t>(tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize),
            static_cast<uint32_t>(tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize),
            static_cast<uint32_t>(tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize)
        };
        this->isSoftmaxResNeedUpdate = AdjustSoftMaxRes<mmOutputType,mmOutputType, true>(mmResUb,
                                                                                         softmaxMaxUb,
                                                                                         this->negativeScalar,
                                                                                         0.0,
                                                                                         softmaxShapeInfo);
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::SoftmaxBasicComputeTailTmp(LocalTensor<mmOutputType>& mmResUb,
                                                    LocalTensor<mmOutputType>& softmaxMaxUb, LocalTensor<mmOutputType>& softmaxSumUb,
                                                    LocalTensor<mmOutputType>& softmaxExpUb, LocalTensor<uint8_t>& sharedTmpUb,
                                                    const SoftMaxShapeInfo &softmaxShapeInfo) {
    SoftmaxFlashV2Tmp<mmOutputType, true, true, true, true>(mmResUb, softmaxSumUb, softmaxMaxUb,
                                        mmResUb, softmaxExpUb, softmaxSumUb, softmaxMaxUb, sharedTmpUb, softmaxFlashTilingData, softmaxShapeInfo);
    if (this->isSoftmaxResNeedUpdate) {
        SoftMaxShapeInfo softmaxShapeInfo{
            static_cast<uint32_t>(tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize),
            static_cast<uint32_t>(tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize),
            static_cast<uint32_t>(tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize),
            static_cast<uint32_t>(tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize)
        };
        this->isSoftmaxResNeedUpdate = AdjustSoftMaxRes<mmOutputType, mmOutputType, true>(mmResUb,
            softmaxMaxUb, this->negativeScalar, 0.0, softmaxShapeInfo);
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::Bmm2UpdateDivNoTail310P(LocalTensor<mmOutputType>& bmm2ResPreUb,
                                            LocalTensor<float>& softmaxSumUb, LocalTensor<softmaxType>& softmaxExpUb) {
    int32_t headLoop = tilingData->promptAttentionBaseParams.headSize / softmaxTypeByteNum;
    constexpr int32_t REPEAT_DATA_NUM = 256 / sizeof(softmaxType);
    BinaryRepeatParams repeatParams;
    // Default values of continuous calculation
    repeatParams.dstBlkStride = 1;
    repeatParams.dstRepStride = 8;
    repeatParams.src0BlkStride = 1;
    repeatParams.src0RepStride = 8;
    repeatParams.src1BlkStride = 1;
    repeatParams.src1RepStride = 8;


    int32_t loop = tilingData->promptAttentionBaseParams.headSize / BLOCK_CUBE;
    int32_t outerSize = isOuterTail_ ? singleProcessSOuterSizeTailAlign : singleProcessSOuterSize;
    int32_t repeat = 16 * outerSize * sizeof(mmOutputType) / 256;

	constexpr int32_t FP32_BLOCK_NUM = 8;
    int32_t calcSize = outerSize * FP32_BLOCK_NUM;
    LocalTensor<float> tmpBuffer = tmpmm2Ub_.template Get<float>();
	DataCopy(tmpBuffer, softmaxSumUb, {static_cast<uint16_t>(outerSize), 1, 0, 1});
    DataCopy(tmpBuffer[FP32_BLOCK_NUM], softmaxSumUb, {static_cast<uint16_t>(outerSize), 1, 0, 1});
    PipeBarrier<PIPE_V>();
	if constexpr (IsSameType<softmaxType, half>::value) {
        Cast(softmaxExpUb, tmpBuffer, RoundMode::CAST_ODD, calcSize * 2);
        PipeBarrier<PIPE_V>();

        for (int i = 0; i < loop; i++) {
            PipeBarrier<PIPE_V>();
            Div(bmm2ResPreUb[i * BLOCK_CUBE * outerSize], bmm2ResPreUb[i * BLOCK_CUBE * outerSize], softmaxExpUb,
                REPEAT_DATA_NUM, repeat, repeatParams);
            PipeBarrier<PIPE_V>();
        }

    } else if constexpr (IsSameType<softmaxType, float>::value){
        LocalTensor<half> tmpSoftmaxSum = tmpmm2Ub_.template Get<half>();
        Cast(tmpSoftmaxSum, tmpBuffer, RoundMode::CAST_NONE, calcSize * 2);
        PipeBarrier<PIPE_V>();
        for (int i = 0; i < loop; i++) {
            PipeBarrier<PIPE_V>();
            Div(bmm2ResPreUb[i * BLOCK_CUBE * outerSize], bmm2ResPreUb[i * BLOCK_CUBE * outerSize], tmpSoftmaxSum, BLOCK_CUBE * outerSize);
            PipeBarrier<PIPE_V>();
        }
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::Bmm2UpdateDivNoTail310PTmp(LocalTensor<mmOutputType>& bmm2ResPreUb,
                                            LocalTensor<mmOutputType>& softmaxSumUb, LocalTensor<softmaxType>& softmaxExpUb) {
    int32_t headLoop = tilingData->promptAttentionBaseParams.headSize / softmaxTypeByteNum;
    constexpr int32_t REPEAT_DATA_NUM = 256 / sizeof(softmaxType);
    BinaryRepeatParams repeatParams;
    // Default values of continuous calculation
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstBlkStride = 1;

    repeatParams.src0RepStride = 8;
    repeatParams.src1RepStride = 8;
    repeatParams.dstRepStride = 8;

    int32_t loop = tilingData->promptAttentionBaseParams.headSize / BLOCK_CUBE;
    int32_t outerSize = isOuterTail_ ? singleProcessSOuterSizeTailAlign : singleProcessSOuterSize;
    int32_t repeat = 16 * outerSize * sizeof(mmOutputType) / 256;
    if constexpr (IsSameType<softmaxType, half>::value) {
        for (int i = 0; i < loop; i++) {
            PipeBarrier<PIPE_V>();
            Div(bmm2ResPreUb[i * BLOCK_CUBE * outerSize], bmm2ResPreUb[i * BLOCK_CUBE * outerSize], softmaxSumUb,
                REPEAT_DATA_NUM, repeat, repeatParams);
            PipeBarrier<PIPE_V>();
        }
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::Bmm2Compute(LocalTensor<mmOutputType>& bmm2ResL1) {
    bmm2.SetTensorA(bmm2ResL1);
    bmm2.SetTensorB(c1Local_);

    int32_t singleM = isOuterTail_ ? singleProcessSOuterSizeTailAlign : singleProcessSOuterSize;
    int32_t singleN = tilingData->promptAttentionBaseParams.headSize;
    int32_t singleK = isInnerLoopLast_ ? singleProcessSInnerSizeTailAlign : singleProcessSInnerSize;

    bmm2.SetOrgShape(singleM, singleN, singleK);
    bmm2.SetTail(singleM, singleN, singleK);
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::UpdateVmul(LocalTensor<softmaxType>& softmaxExpUb) {
    LocalTensor<mmOutputType> bmm2ResPreUb = tempBmm2Ub.Get<mmOutputType>(bmm2ResUbSize);
    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = 8;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1RepStride = 8;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = 8;
    constexpr int32_t REPEAT_DATA_NUM = 256 / sizeof(softmaxType);
    int32_t outerSize = isOuterTail_ ? singleProcessSOuterSizeTailAlign : singleProcessSOuterSize;
    int32_t repeat = 16 * outerSize * sizeof(mmOutputType) / 256;
    // only support singleProcessSOuterSize <=255, headsize 32B align
    int32_t numOneRep = 256 / sizeof(softmaxType);
    int32_t loop = tilingData->promptAttentionBaseParams.headSize / BLOCK_CUBE;

    if constexpr (IsSameType<softmaxType, float>::value) {
        constexpr int32_t FP32_BLOCK_NUM = 8;
        int32_t calcSize = outerSize * FP32_BLOCK_NUM;
        LocalTensor<float> tmpBuffer = tmpmm2Ub_.template Get<float>();
        LocalTensor<half> tmpSoftmaxExp = tmpmm2Ub_.template Get<half>();
        DataCopy(tmpBuffer, softmaxExpUb, {static_cast<uint16_t>(outerSize), 1, 0, 1});
        DataCopy(tmpBuffer[FP32_BLOCK_NUM], softmaxExpUb, {static_cast<uint16_t>(outerSize), 1, 0, 1});
        PipeBarrier<PIPE_V>();
        Cast(tmpSoftmaxExp, tmpBuffer, RoundMode::CAST_ODD, calcSize * 2);
        PipeBarrier<PIPE_V>();


        for (int i = 0; i < loop; i++) {
            PipeBarrier<PIPE_V>();
            Mul(bmm2ResPreUb[i * BLOCK_CUBE * outerSize], tmpSoftmaxExp, bmm2ResPreUb[i * BLOCK_CUBE * outerSize], BLOCK_CUBE * outerSize);
            PipeBarrier<PIPE_V>();
        }
    } else {
        for (int i = 0; i < loop; i++) {
            PipeBarrier<PIPE_V>();
            Mul(bmm2ResPreUb[i * BLOCK_CUBE * outerSize], softmaxExpUb, bmm2ResPreUb[i * BLOCK_CUBE * outerSize],
                REPEAT_DATA_NUM, repeat, repeatParams);
            PipeBarrier<PIPE_V>();
        }
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::ComputeAttenMaskOffset(uint32_t sInnerLoopIdx, bool isLast) {
    uint64_t attenMaskOffsetDateSize = 0;
    if (!isLast) {
        attenMaskOffsetDateSize =
            static_cast<uint64_t>(sInnerLoopIdx + 1) * singleProcessSInnerSize;
        attenMaskOffset = attenMaskCoreOffset + attenMaskOffsetDateSize;
    } else {
        attenMaskOffset = attenMaskCoreOffset +
            this->singleProcessSOuterSize *
            (uint64_t)tilingData->promptAttentionBaseParams.maskKVsSize;
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::ComputeOffset(uint32_t sInnerLoopIdx, bool isLast) {
    if constexpr (PFAT::layout == PFALayoutNZ::BSH) {
        uint64_t sInnerOffsetDataSize =
            static_cast<uint64_t>(sInnerLoopIdx) * singleProcessSInnerSize;
        ComputeAttenMaskOffset(sInnerLoopIdx, isLast);
        // tensorBOffset cannot be updated here, as it will erase the previously set values
        valueOffset = valueCoreOffset + sInnerOffsetDataSize * MultiHeadKV;
        tensorAOffset = tensorACoreOffset;
    } else { // BNSD
        uint64_t sInnerOffsetDataSize =
            static_cast<uint64_t>(sInnerLoopIdx) * singleProcessSInnerSize;
        valueOffset = valueCoreOffset + sInnerOffsetDataSize * tilingData->promptAttentionBaseParams.headSize;
        ComputeAttenMaskOffset(sInnerLoopIdx, isLast);
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::DataCopyTransposeOut(LocalTensor<mmOutputType>& bmm2ResUb) {
    if ((PFAT::layout == PFALayoutNZ::BSH) ||
        (PFAT::layout == PFALayoutNZ::BNSD && this->tilingData->promptAttentionBaseParams.isBSNDOut == 1)) {
        // Copying here requires consideration of BNSD to BSH conversion and NZ to ND conversion
        int32_t outerSize = isOuterTail_ ? singleProcessSOuterSizeTailAlign : singleProcessSOuterSize;
        struct DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = isOuterTail_ ? singleProcessSOuterSizeTail : singleProcessSOuterSize;;
        dataCopyParams.blockLen = 1;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = MultiHeadQ / BLOCK_CUBE - 1;
        int32_t loop = tilingData->promptAttentionBaseParams.headSize / BLOCK_CUBE;
        event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(enQueEvtID);
        WaitFlag<HardEvent::V_MTE3>(enQueEvtID);
        int64_t startAddr = this->multiSeqOffset +
            static_cast<uint64_t>(batchNOffset) * tilingData->promptAttentionBaseParams.headSize +
            static_cast<uint64_t>(sOuterOffset) * MultiHeadQ;
        for(int i = 0; i < loop; i++) {
            DataCopy(attentionOutGm[startAddr + i * BLOCK_CUBE], bmm2ResUb[i * BLOCK_CUBE * outerSize], dataCopyParams);
        }
    } else { // BNSD
        int32_t outerSize = isOuterTail_ ? singleProcessSOuterSizeTailAlign : singleProcessSOuterSize;
        struct DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = isOuterTail_ ? singleProcessSOuterSizeTail : singleProcessSOuterSize;;
        dataCopyParams.blockLen = 1;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = tilingData->promptAttentionBaseParams.headSize / BLOCK_CUBE - 1;
        int32_t loop = tilingData->promptAttentionBaseParams.headSize / BLOCK_CUBE;
        event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(enQueEvtID);
        WaitFlag<HardEvent::V_MTE3>(enQueEvtID);
        for(int i = 0; i < loop; i++) {
            DataCopy(attentionOutGm[attentionOutOffset + i * BLOCK_CUBE], bmm2ResUb[i * BLOCK_CUBE * outerSize], dataCopyParams);
        }
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::Bmm1Compute(LocalTensor<mmInputType>& a1Local,
    LocalTensor<mmInputType>& b1Local, int32_t singleM, int32_t singleN, int32_t singleK) {
    mm.SetTensorA(a1Local);
    mm.SetTensorB(b1Local, true);
    mm.SetOrgShape(singleM, singleN, singleK);
    mm.SetTail(singleM, singleN, singleK);
    event_t eventIdM_MTE1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_MTE1));
    SetFlag<HardEvent::M_MTE1>(eventIdM_MTE1);
    WaitFlag<HardEvent::M_MTE1>(eventIdM_MTE1);
    mm.template Iterate<false>();
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::LoopSOuterOffsetInitWithBSH(
    uint64_t seqListOffsetSize, int sIdx) {
    uint64_t attenMaskBatchOffset = 0;
    if (attenMaskBatch != 1) {
        attenMaskBatchOffset = (uint64_t)sIdx * (uint64_t)tilingData->promptAttentionBaseParams.maskKVsSize *
                        (uint64_t)tilingData->promptAttentionBaseParams.maskQsSize;
    }

    // mask offset of core
    attenMaskCoreOffset = (uint64_t)sOuterOffset * (uint64_t)tilingData->promptAttentionBaseParams.maskKVsSize + attenMaskBatchOffset;

    tensorACoreOffset = seqListOffsetSize +
                        static_cast<uint64_t>(sOuterOffset) * MultiHeadQ +
                        static_cast<uint64_t>(batchNOffset) *
                        tilingData->promptAttentionBaseParams.headSize;

    uint64_t seqInnerOffsetSize =
        tilingData->promptAttentionBaseParams.seqSize == tilingData->promptAttentionBaseParams.seqInnerSize ?
        seqListOffsetSize / headNumRatio :
        static_cast<uint64_t>(sIdx) * tilingData->promptAttentionBaseParams.seqInnerSize * MultiHeadKV;
    // calculate the offset for tensor B (key or value tensor).
    tensorBCoreOffset = seqInnerOffsetSize +
                        static_cast<uint64_t>(batchNOffset) / headNumRatio *
                        tilingData->promptAttentionBaseParams.headSize;

    valueCoreOffset = tensorBCoreOffset;
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::LoopSOuterOffsetInitWithBNSD(
    uint64_t seqListOffsetSize, int sIdx) {
    uint64_t head_stride_q = static_cast<uint64_t>(tilingData->promptAttentionBaseParams.headSize) *
                             tilingData->promptAttentionBaseParams.seqSize;
    uint64_t head_stride_kv = static_cast<uint64_t>(tilingData->promptAttentionBaseParams.headSize) *
                              tilingData->promptAttentionBaseParams.seqInnerSize;
    uint64_t seq_stride = tilingData->promptAttentionBaseParams.headSize;

    uint64_t attenMaskBatchOffset = 0;
    if (attenMaskBatch != 1) {
        attenMaskBatchOffset =
            (uint64_t)sIdx * (uint64_t)tilingData->promptAttentionBaseParams.maskKVsSize *
            (uint64_t)tilingData->promptAttentionBaseParams.maskQsSize;
    }
    attenMaskCoreOffset =
        (uint64_t)sOuterOffset * (uint64_t)tilingData->promptAttentionBaseParams.maskKVsSize +
        attenMaskBatchOffset;

    tensorACoreOffset = seqListOffsetSize +
        static_cast<uint64_t>(batchNOffset) * head_stride_q +
        static_cast<uint64_t>(sOuterOffset) * seq_stride;

    uint64_t seqInnerOffsetSize =
        tilingData->promptAttentionBaseParams.seqSize == tilingData->promptAttentionBaseParams.seqInnerSize ?
        seqListOffsetSize / headNumRatio :
        static_cast<uint64_t>(sIdx) * head_stride_kv *
        tilingData->promptAttentionBaseParams.headNumSize / headNumRatio;

    tensorBCoreOffset = seqInnerOffsetSize +
        static_cast<uint64_t>(batchNOffset) / headNumRatio * head_stride_kv;

    valueCoreOffset = tensorBCoreOffset;

    attentionOutOffset = seqListOffsetSize +
        static_cast<uint64_t>(batchNOffset) * head_stride_q +
        static_cast<uint64_t>(sOuterOffset) * seq_stride;
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::LoopSOuterOffsetInit(
    uint64_t seqListOffsetSize, int sIdx) {
    if constexpr (PFAT::layout == PFALayoutNZ::BSH) {
        LoopSOuterOffsetInitWithBSH(seqListOffsetSize, sIdx);
    } else { // BNSD
        LoopSOuterOffsetInitWithBNSD(seqListOffsetSize, sIdx);
    }
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::Bmm2UpdateAdd(LocalTensor<mmOutputType>& bmm2ResUb) {
    LocalTensor<mmOutputType> bmm2ResPreUb = tempBmm2Ub.Get<mmOutputType>(bmm2ResUbSize);
    Add(bmm2ResPreUb, bmm2ResUb, bmm2ResPreUb, bmm2ResUbSize);
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::GetSingleCoreParam(int sIdx) {
    singleProcessSInnerSize = tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize;
    singleProcessSOuterSize = tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize;
    MultiHeadQ = tilingData->promptAttentionBaseParams.headSize * tilingData->promptAttentionBaseParams.headNumSize;
    MultiHeadKV = MultiHeadQ / headNumRatio;

    actualSeqLengthPerBatch = isActualLenDimsNull ? tilingData->promptAttentionBaseParams.seqSize :
                              actualSeqLengthsGm.GetValue(sIdx);
    actualSeqLengthPerBatch = ((int64_t)actualSeqLengthPerBatch >
                               (int64_t)tilingData->promptAttentionBaseParams.seqInnerSize +
                               (int64_t)tilingData->promptAttentionBaseParams.preTokens) ?
                               tilingData->promptAttentionBaseParams.seqInnerSize + tilingData->promptAttentionBaseParams.preTokens :
                               actualSeqLengthPerBatch;
    actualSeqLengthKVPerBatch = isActualLenDimsKVNull ? tilingData->promptAttentionBaseParams.seqInnerSize : actualSeqLengthsKVGm.GetValue(sIdx);
    singleProcessSOuterSizeTail = (actualSeqLengthPerBatch % singleProcessSOuterSize != 0) ?
                                   actualSeqLengthPerBatch % singleProcessSOuterSize : singleProcessSOuterSize;
    singleProcessSOuterSizeTailAlign = (singleProcessSOuterSizeTail + typeByteNum - 1) / typeByteNum * typeByteNum;
    maxInnerLoopTimes = (actualSeqLengthKVPerBatch + singleProcessSInnerSize - 1) / singleProcessSInnerSize;
    singleProcessSInnerSizeTail = (actualSeqLengthKVPerBatch % singleProcessSInnerSize != 0) ?
                     actualSeqLengthKVPerBatch % singleProcessSInnerSize : singleProcessSInnerSize;
    singleProcessSInnerSizeTailAlign = (singleProcessSInnerSizeTail + typeByteNum - 1) / typeByteNum * typeByteNum;
    maskInnerTailAlign = (singleProcessSInnerSizeTail + maskTypeByteNum - 1) / maskTypeByteNum * maskTypeByteNum;
    padSize = maskInnerTailAlign - singleProcessSInnerSizeTail;

    InitTensorSize(&tilingData->promptAttentionTensorSizeRect);
    transposeTilingData = tilingData->transposeTilingDataRect;
    softmaxTilingData = tilingData->softmaxTilingDataRect;
    softmaxFlashTilingData = tilingData->softmaxFlashTilingDataRect;
}

template<typename PFAT>
__aicore__ inline void PromptFlashAttentionS1s2Bns1X310Base<PFAT>::CopyND2NZOnTheFly(
    const LocalTensor<mmOutputType>& dst,  const GlobalTensor<mmOutputType>& src, const int height,
    const int width, const int gCol, const bool isA1) {
    ASSERT(gCol >= width && "Copy ND block gm->ub width larger than origin matrix width.");
    int32_t dstOffset = 0;
    int32_t srcOffset = 0;
    int32_t calcWidth = width / BLOCK_CUBE; // cube block numbers that do not need to be pad zero
    int32_t calcHeightAlign = (height + BLOCK_CUBE - 1) / BLOCK_CUBE;
    if (height % BLOCK_CUBE != 0) {
        int32_t totalRepeat = calcWidth * calcHeightAlign;
        int32_t initConstRepeatTimes = (totalRepeat + MAX_REPEATS_PER_BATCH - 1) / MAX_REPEATS_PER_BATCH;
        for (int32_t i = 0; i < initConstRepeatTimes; i++) {
            uint16_t curRepeat = (i == initConstRepeatTimes - 1) ?
                static_cast<uint16_t>(totalRepeat - i * MAX_REPEATS_PER_BATCH) : static_cast<uint16_t>(MAX_REPEATS_PER_BATCH);
            InitConstValue(dst[i * MAX_REPEATS_PER_BATCH * REPEAT_DATASIZE_EACH_TIME / sizeof(mmOutputType)],
                {static_cast<uint16_t>(curRepeat), 0, 0, 0});
        }
        PipeBarrier<PIPE_MTE2>();
    }
    // gCol unaligned ,can not use dma copy repeat stride
    int src_gap = gCol * sizeof(mmOutputType) / UB_ALIGN_NZ - 1;
    for (int i = 0; i < calcWidth; i++) {
        dstOffset = i * calcHeightAlign * CUBE_MAX_SIZE;
        srcOffset = i * BLOCK_CUBE;
        DataCopy(dst[dstOffset], src[srcOffset],
                 { static_cast<uint16_t>(height), 1, static_cast<uint16_t>(src_gap), 0});
    }
    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID3);
}

#endif  // PROMPT_FLASH_ATTENTION_S1S2_BNS1_X310_BASE_H