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
 * \file prompt_flash_attention_base.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_BASE_H
#define PROMPT_FLASH_ATTENTION_BASE_H

#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/quantization/ascend_quant.h"
#include "adv_api/activation/softmax.h"
#include "adv_api/activation/softmaxflash.h"
#include "adv_api/activation/softmaxflashv2.h"
#include "lib/matmul_intf.h"
#include "../kernel_data_copy_transpose.h"
#include "kernel_operator_list_tensor_intf.h"

using namespace matmul;

constexpr uint32_t BATCH_NUM_MAX = 128;

enum OptimizationMode {
    HighPrecision,
    HighPerformance
};

template<typename T, OptimizationMode M = OptimizationMode::HighPerformance>
struct PromptFlashAttentionTypeTraits
{
    using mmInputType = T;
    using mmBiasType = T;
    using mmOutputType = T;
    using softmaxType = T;
    using pseShiftType = T;
    using pseShiftCastType = half;
};

template<>
struct PromptFlashAttentionTypeTraits<half, OptimizationMode::HighPerformance>
{
    using mmInputType = half;
    using mmBiasType = float;
    using mmOutputType = half;
    using softmaxType = half;
    using pseShiftType = half;
    using pseShiftCastType = half;
};

#if (__CCE_AICORE__ > 200)

template<>
struct PromptFlashAttentionTypeTraits<half, OptimizationMode::HighPrecision>
{
    using mmInputType = half;
    using mmBiasType = float;
    using mmOutputType = float;
    using softmaxType = float;
    using pseShiftType = half;
    using pseShiftCastType = float;  // pseShiftCastType is only fp32 in the case of high precision and bf16
};

template<>
struct PromptFlashAttentionTypeTraits<bfloat16_t>
{
    using mmInputType = bfloat16_t;
    using mmBiasType = float;
    using mmOutputType = float;
    using softmaxType = float;
    using pseShiftType = bfloat16_t;
    using pseShiftCastType = float;  // pseShiftCastType is only fp32 in the case of high precision and bf16
};
#endif

template<>
struct PromptFlashAttentionTypeTraits<int8_t>
{
    using mmInputType = int8_t;
    using mmBiasType = int32_t;
    using mmOutputType = half;
    using softmaxType = half;
    using pseShiftType = half;
    using pseShiftCastType = half;
};

constexpr uint32_t BOOLBYTENUM = 32;
constexpr uint32_t UB_ALIGN = 32U;
template<typename T, typename U, CubeFormat FORMAT, typename O, OptimizationMode M = OptimizationMode::HighPerformance>
class PromptFlashAttentionBase {
public:
    __aicore__ inline PromptFlashAttentionBase() {};
    __aicore__ inline void Init(__gm__ uint8_t* query,
                                __gm__ uint8_t* key,
                                __gm__ uint8_t* value,
                                __gm__ uint8_t* pseShift,
                                __gm__ uint8_t* attenMask,
                                __gm__ uint8_t* actualSeqLengths,
                                __gm__ uint8_t* actualSeqLengthsKV,
                                __gm__ uint8_t* blocktable,
                                __gm__ uint8_t* queryPaddingSize,
                                __gm__ uint8_t* kvPaddingSize,
                                __gm__ uint8_t* keySharedPrefix,
                                __gm__ uint8_t* valueSharedPrefix,
                                __gm__ uint8_t* actualSharedPrefixLen,
                                __gm__ uint8_t* attentionOut,
                                __gm__ uint8_t* softmaxLse,
                                __gm__ uint8_t* workspace,
                                const PromptFlashAttentionTilingData* __restrict tiling,
                                __gm__ uint8_t* gmTiling,
                                TPipe* tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void InitQuant(__gm__ uint8_t* deq_scale1, __gm__ uint8_t* scale1, __gm__ uint8_t* deq_scale2,
                                     __gm__ uint8_t* scale2, __gm__ uint8_t* offset2);
    __aicore__ inline void InitMsd(__gm__ uint8_t* key_antiquant_scale, __gm__ uint8_t* key_antiquant_offset, __gm__ uint8_t* value_antiquant_scale, __gm__ uint8_t* value_antiquant_offset);

    // define datatype
    using mmInputType = typename PromptFlashAttentionTypeTraits<T, M>::mmInputType;
    using mmBiasType = typename PromptFlashAttentionTypeTraits<T, M>::mmBiasType;
    using mmOutputType = typename PromptFlashAttentionTypeTraits<T, M>::mmOutputType;
    using softmaxType = typename PromptFlashAttentionTypeTraits<T, M>::softmaxType;
    using pseShiftType = typename PromptFlashAttentionTypeTraits<T, M>::pseShiftType;
    using pseShiftCastType = typename PromptFlashAttentionTypeTraits<T, M>::pseShiftCastType;
    // define matmul
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, mmInputType, false>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, mmInputType, true>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, mmBiasType>;
    using c1Type = MatmulType<TPosition::VECCALC, FORMAT, mmOutputType>;
    Matmul<a1Type, b1Type, c1Type, bias1Type> mm;
    // define batchmatmul
    using a2Type = MatmulType<TPosition::VECCALC, FORMAT, mmInputType, false>;
    using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, mmInputType, false>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, mmBiasType>;
#if defined(__CCE_KT_TEST__)
    using c2Type = MatmulType<TPosition::VECCALC, CubeFormat::ND_ALIGN, mmOutputType>; // cpu
#else
    using c2Type = MatmulType<TPosition::VECCALC, CubeFormat::ND, mmOutputType>; // npu
#endif
    Matmul<a2Type, b2Type, c2Type, bias2Type> bmm2;

protected:
    const PromptFlashAttentionTilingData* __restrict tilingData;
    TPipe* pipe;
    // define the que
    TQue<QuePosition::VECIN, 1> attenMaskQueue;
    TQue<QuePosition::VECIN, 1> eleWiseInQueue;
    TQue<QuePosition::VECOUT, 1> tempBmm2Queue;
    TQue<QuePosition::VECOUT, 1> Bmm1Queue;
    TQue<QuePosition::VECOUT, 1> softmaxOutQueue;
    TBuf<> selectSpaceUb;

    TBuf<> pseShiftCastUb;
    TBuf<> softmaxExpUb_;
    TBuf<> tempBmm2Ub;
    TBuf<> tmpTypeCastUb; // for cast
    TBuf<> tmpTypeCastUb2222; // for cast
    TBuf<> tmpSoftmaxFlashV2Ub_;
    TBuf<> tmpmm2Ub_;

    LocalTensor<softmaxType> softmaxExpUb;

    GlobalTensor<T> queryGm;
    GlobalTensor<T> keyGm;
    GlobalTensor<T> valueGm;
    GlobalTensor<U> attenMaskGm;
    GlobalTensor<O> attentionOutGm;
    GlobalTensor<int64_t> actualSeqLengthsGm;
    GlobalTensor<int64_t> actualSeqLengthsKVGm;
    GlobalTensor<softmaxType> workspaceGm;
    GlobalTensor<pseShiftType> pseShiftGm;
    GlobalTensor<mmOutputType> workspaceGmProcessT;
    GlobalTensor<int64_t> queryPaddingSizeGm;
    GlobalTensor<int64_t> kvPaddingSizeGm;

    // quant: define quant variable
    uint64_t dequantScale1 = 0;
    float quantScale1 = 0;
    uint64_t dequantScale2 = 0;
    float quantScale2 = 0;
    float quantOffset2 = 0;

    GlobalTensor<uint32_t> deqScale1Fp32Gm;
    GlobalTensor<uint32_t> deqScale2Fp32Gm;

    __gm__ uint8_t* key_ptr;
    __gm__ uint8_t* value_ptr;
    __gm__ uint8_t* currentKey;
    __gm__ uint8_t* currentValue;

    bool useMask = false;
    bool usePseShift = false;
    bool needCalMask_ = false;
    uint32_t tmp_block_idx = 0;
    uint32_t loopSNum = 0;
    int32_t sOuterOffset = 0;
    int32_t preTokensOffset = 0;
    int32_t nextTokensOffset = 0;
    uint32_t batchNOffset = 0;
    uint32_t maskOffset = 0;
    uint32_t maskCoreOffset = 0;
    uint64_t attenMaskCoreOffset = 0;
    uint64_t pseShiftCoreOffset = 0;
    uint32_t valueOffset = 0;
    uint32_t valueCoreOffset = 0;
    uint64_t attenMaskOffset = 0;
    uint64_t attenMaskOffsetPre = 0;
    uint64_t pseShiftOffset = 0;
    uint32_t tensorAOffset = 0;
    uint32_t tensorBOffset = 0;
    uint32_t tensorACoreOffset = 0;
    uint32_t tensorBCoreOffset = 0;
    uint32_t attentionOutOffset = 0;
    uint64_t offsetSS = 0;
    uint64_t offsetSH = 0;
    uint64_t offsetSTypeNum = 0;
    uint64_t offsetNSTypeNum = 0;
    uint64_t offsetNSS = 0;
    uint64_t offsetNSH = 0;
    uint32_t maskDataType = 0;
    uint32_t attenMaskBatch = 0;
    uint32_t pseShiftBatch = 0;
    uint32_t maskCopyInCol = 0;
    uint32_t pseShiftCopyInCol = 0;

    // tilingdata
    uint32_t singleProcessSOuterSizeWhole = 0;
    uint32_t singleProcessSOuterSize = 0;
    uint32_t singleProcessSOuterSizeTail = 0;
    uint32_t singleProcessSInnerSize = 0;
    uint32_t singleProcessSInnerSizeNow = 0;
    uint32_t singleProcessSInnerSizeTail = 0;
    uint32_t singleProcessSInnerBmmTail = 0;
    uint32_t mmResUbSize = 0;
    uint32_t attenMaskUbSize = 0;
    uint32_t pseShiftUbSize = 0;
    uint32_t maskSize = 0;
    uint32_t softmaxMaxSize = 0;
    uint32_t softmaxSumSize = 0;
    uint32_t softmaxExpSize = 0;
    uint32_t spmTmpSize = 0;
    uint32_t scmTmpSize = 0;
    uint32_t bmm2ResUbSize = 0;
    uint32_t tmpMMResBmm2PreUbSize = 0;
    uint32_t tmpSoftmaxBmm2UbSize = 0;
    uint32_t padSize = 0;
    uint32_t pseShiftPadSize = 0;
    uint32_t unalignSInner = 0;
    uint32_t typeByteNum = 0;
    uint32_t outputTypeByteNum = 0;
    uint32_t softmaxTypeByteNum = 0;
    uint32_t headNumRatio = 0;
    uint32_t maskTypeByteNum = 0;
    uint32_t pseShiftTypeByteNum = 0;
    uint32_t selectSpaceUbSize = 0;
    uint32_t softMaxV2Size_ = 0;
    uint32_t mm2TmpUbSize_ = 0;
    uint32_t splitS2 = 0;
    uint32_t layoutType = 0;
    uint32_t maskInnerTailAlign = 0;
    uint32_t pseShiftInnerTailAlign = 0;

    SoftMaxTiling softmaxTilingData;
    SoftMaxTiling softmaxFlashTilingData;
    CopyTransposeTiling transposeTilingData;
    uint32_t bmm2_settail_m  = 0;
    uint32_t sOuterSizeTail_tmp = 0;
    uint32_t tailCoresOuterSizeTail_tmp = 0;
    uint32_t MultiHeadQ = 0;
    uint32_t MultiHeadKV = 0;
    uint32_t startInBlock = 0;
    uint32_t maxInnerLoopTimes = 0;
    uint32_t seqListOffset = 0;
    int64_t multiSeqOffset = 0;

    int32_t preTokensPerBatch = 0;
    int32_t nextTokensPerBatch = 0;
    uint32_t attentionMaskStride = 0;
    int32_t attentionMaskType = 0;
    uint32_t attentionMaskMaxSize = 0;

    uint32_t pseShiftStride = 0;

    bool isActualLenDimsNull;
    bool isActualLenDimsKVNull;
    int32_t actualSeqLengthPerBatch = 0;
    int32_t actualSeqLengthKVPerBatch = 0;
    uint32_t accumSOuterTilingNums[BATCH_NUM_MAX];
    uint32_t actualSeqOffsets[BATCH_NUM_MAX];

    uint32_t isKvContinuous = 0;
    uint32_t fromFused = 0;

    __aicore__ inline void ElewiseCompute(LocalTensor<mmOutputType>& mmResUb, uint32_t computeSize, uint32_t type);

    __aicore__ inline void SoftmaxBasicComputeFirst(LocalTensor<mmOutputType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
                                                    LocalTensor<float>& softmaxSumUb, SoftMaxShapeInfo& shapeInfo);

    __aicore__ inline void SoftmaxComputeFirst(LocalTensor<mmOutputType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
                                               LocalTensor<float>& softmaxSumUb, SoftMaxShapeInfo& shapeInfo);

    __aicore__ inline bool IsSoftmaxBasic();

    __aicore__ inline void SoftmaxBasicCompute(LocalTensor<mmOutputType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
                                               LocalTensor<float>& softmaxSumUb, LocalTensor<softmaxType>& softmaxExpUb,
                                               SoftMaxShapeInfo& shapeInfo);

    __aicore__ inline void SoftmaxCompute(LocalTensor<mmOutputType>& mmResUb, LocalTensor<float>& softmaxMaxUb,
                                          LocalTensor<float>& softmaxSumUb, LocalTensor<softmaxType>& softmaxExpUb,
                                          SoftMaxShapeInfo& shapeInfo);

    __aicore__ inline void SoftmaxBasicComputeFirstNoTail(LocalTensor<mmOutputType>& mmResUb,
                                                          LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                                          LocalTensor<softmaxType>& softmaxExpUb, SoftMaxShapeInfo& shapeInfo);

    __aicore__ inline void SoftmaxBasicComputeNoTail(LocalTensor<mmOutputType>& mmResUb,
                                                     LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                                     LocalTensor<softmaxType>& softmaxExpUb, SoftMaxShapeInfo& shapeInfo);

    __aicore__ inline void SoftmaxBasicComputeFirstNoTail(LocalTensor<mmOutputType>& mmResUb,
                                                          LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                                          LocalTensor<softmaxType>& softmaxExpUb, LocalTensor<uint8_t>& sharedTmpUb,
                                                          SoftMaxShapeInfo& shapeInfo);

    __aicore__ inline void SoftmaxBasicComputeNoTail(LocalTensor<mmOutputType>& mmResUb,
                                                     LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                                     LocalTensor<softmaxType>& softmaxExpUb, LocalTensor<uint8_t>& sharedTmpUb,
                                                     SoftMaxShapeInfo& shapeInfo);

    __aicore__ inline void SoftmaxComputeFirstTail(LocalTensor<mmOutputType>& mmResUb,
                                                   LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                                   LocalTensor<softmaxType>& softmaxExpUb, SoftMaxShapeInfo& shapeInfo);

    __aicore__ inline void SoftmaxComputeTail(LocalTensor<mmOutputType>& mmResUb,
                                              LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                              LocalTensor<softmaxType>& softmaxExpUb, SoftMaxShapeInfo& shapeInfo);

    __aicore__ inline bool IsSoftmaxFlashBasic();

    __aicore__ inline void QuantCompute(LocalTensor<int8_t> quantResUb, LocalTensor<mmOutputType> mmResUb, float scale, float offset, uint32_t computeSize);

    __aicore__ inline void Bmm2UpdateDivNoTail(LocalTensor<mmOutputType>& bmm2ResPreUb,
                                               LocalTensor<float>& softmaxSumUb, LocalTensor<softmaxType>& softmaxExpUb);

    __aicore__ inline void Bmm2Compute(uint32_t offset, LocalTensor<mmOutputType>& bmm2ResL1);

    __aicore__ inline void UpdateVmul(LocalTensor<softmaxType>& softmaxExpUb);

    __aicore__ inline void Bmm2UpdateAdd(LocalTensor<mmOutputType>& bmm2ResUb);

    __aicore__ inline void DataCopyTransposeOut(LocalTensor<mmOutputType>& bmm2ResUb);

    __aicore__ inline void DataCopyOutWithBNSD(LocalTensor<mmOutputType>& bmm2ResUb);

    __aicore__ inline void ComputePseShiftOffset(int sInnerOffsetDataSize);

    __aicore__ inline void ComputeAttenMaskOffset(int sInnerOffsetDataSize);
    __aicore__ inline void ComputeAttenMaskOffsetPre(int sInnerOffsetDataSize);

    __aicore__ inline void ComputeOffset(uint32_t sInnerLoopIdx);

    __aicore__ inline void ComputeOffsetWithBNSD(uint32_t sInnerLoopIdx);

    __aicore__ inline void CalPseShiftOffset(int sIdx);

    __aicore__ inline void LoopSOuterOffsetInit(uint32_t seqListOffsetSize, int sIdx);

    __aicore__ inline void LoopSOuterOffsetInitWithBNSD(uint32_t seqListOffsetSize, int sIdx);

    __aicore__ inline void initOffset();

    __aicore__ inline void InitTensorSize(const PromptAttentionSingleCoreTensorSize* tensorSizeTiling);

    __aicore__ inline void GetSingleCoreParam(int sIdx);

    __aicore__ inline void GetSparseParam(int32_t* preTokens, int32_t* nextTokens);

    __aicore__ inline void ComputeTokenOffset();

    __aicore__ inline void InitOutputSingleCore();
};

template<typename T, typename U, CubeFormat FORMAT, typename O, OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::Init(__gm__ uint8_t* query, __gm__ uint8_t* key,
                                        __gm__ uint8_t* value, __gm__ uint8_t* pseShift, __gm__ uint8_t* attenMask,
                                        __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV, __gm__ uint8_t* blocktable,
                                        __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
                                        __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix, __gm__ uint8_t* actualSharedPrefixLen,
                                        __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace,
                                        const PromptFlashAttentionTilingData* __restrict tiling,
                                        __gm__ uint8_t* gmTiling, TPipe* tPipe) {
    tmp_block_idx = GetBlockIdx();
    // init global buffer
    tilingData = tiling;
    queryGm.SetGlobalBuffer((__gm__ T*)query);
    key_ptr = key;
    value_ptr = value;
    attentionOutGm.SetGlobalBuffer((__gm__ O*)attentionOut);
    workspaceGm.SetGlobalBuffer((__gm__ softmaxType*)workspace);

    pipe = tPipe;
    typeByteNum = tilingData->promptAttentionBaseParams.typeByteNum;
    outputTypeByteNum = tilingData->promptAttentionBaseParams.outputTypeByteNum;
    softmaxTypeByteNum = tilingData->promptAttentionBaseParams.softmaxTypeByteNum;
    headNumRatio = tilingData->promptAttentionBaseParams.headNumRatio;
    maskDataType = tilingData->promptAttentionBaseParams.attenMaskElemType;
    maskTypeByteNum = tilingData->promptAttentionBaseParams.maskTypeByteNum;
    preTokensPerBatch = 0;
    nextTokensPerBatch = 0;
    preTokensOffset = 0;
    nextTokensOffset = 0;
    attenMaskBatch = tilingData->promptAttentionSingleCoreParams.attenMaskBatch;
    pseShiftTypeByteNum = tilingData->promptAttentionBaseParams.pseShiftTypeByteNum;
    pseShiftBatch = tilingData->promptAttentionSingleCoreParams.pseShiftBatch;
    layoutType = tilingData->promptAttentionBaseParams.layoutType;
    isKvContinuous = tilingData->promptAttentionBaseParams.isKvContinuous;
    fromFused = tilingData->promptAttentionBaseParams.fromFused;

    if (fromFused) {
        ListTensorDesc keyListTensorDescInit((__gm__ void*)key_ptr);
        ListTensorDesc valueListTensorDescInit((__gm__ void*)value_ptr);
        currentKey = (__gm__ uint8_t*)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
        currentValue = (__gm__ uint8_t*)valueListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);

        keyGm.SetGlobalBuffer((__gm__ T*)currentKey);
        valueGm.SetGlobalBuffer((__gm__ T*)currentValue);
    } else {
        keyGm.SetGlobalBuffer((__gm__ T*)key);
        valueGm.SetGlobalBuffer((__gm__ T*)value);
    }

    initOffset();

    isActualLenDimsKVNull = true;
    if (!tilingData->promptAttentionBaseParams.isActualSeqLengthsKVNull) {
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengthsKV, tilingData->promptAttentionBaseParams.batchSize);
        isActualLenDimsKVNull = false;
    }

    isActualLenDimsNull = true;
    if (!tilingData->promptAttentionBaseParams.isActualSeqLengthsNull) {
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengths, tilingData->promptAttentionBaseParams.batchSize);
        isActualLenDimsNull = false;
    }


    uint32_t preAccumSOuter = 0;
    uint32_t h = tilingData->promptAttentionBaseParams.headNumSize * tilingData->promptAttentionBaseParams.headSize;
    uint32_t s = tilingData->promptAttentionBaseParams.seqSize;
    uint32_t actualSeqLengthsIdx = 0;
    uint32_t middle_actualSeqLengths = 0;
    for (int i = 0; i < tilingData->promptAttentionBaseParams.batchSize; i++) {
        actualSeqLengthsIdx = isActualLenDimsNull ?
            tilingData->promptAttentionBaseParams.seqSize : actualSeqLengthsGm.GetValue(i);
        if (tilingData->promptAttentionBaseParams.isActualSeqLengthsNull) {
            actualSeqOffsets[i] = i * s * h;
        } else {
            if (tilingData->promptAttentionBaseParams.isLayoutSH) {
                actualSeqOffsets[i] = middle_actualSeqLengths * h;
                middle_actualSeqLengths += actualSeqLengthsIdx;
            } else {
                actualSeqOffsets[i] = i * s * h;
            }
        }

        actualSeqLengthsIdx = ((int64_t)actualSeqLengthsIdx >
                            (int64_t)tilingData->promptAttentionBaseParams.seqInnerSize +
                            (int64_t)tilingData->promptAttentionBaseParams.preTokens) && (attentionMaskType != 4)?
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

    uint32_t pseMaskMaxSize = tilingData->promptAttentionBaseParams.pseMaskMaxSize;
    pipe->InitBuffer(attenMaskQueue, 1, (tilingData->promptAttentionTensorSizeRect.attenMaskUbSize) * pseMaskMaxSize);
    pipe->InitBuffer(softmaxOutQueue, 1, 2 * tilingData->promptAttentionTensorSizeRect.softmaxMaxSize * sizeof(float));
    if (tilingData->promptAttentionBaseParams.splitS2 == 1) {
        pipe->InitBuffer(softmaxExpUb_, tilingData->promptAttentionTensorSizeRect.softmaxExpSize * sizeof(softmaxType));
        softmaxExpUb = softmaxExpUb_.Get<softmaxType>(tilingData->promptAttentionTensorSizeRect.softmaxExpSize);
        if (tilingData->promptAttentionBaseParams.splitD == 1) {
            pipe->InitBuffer(eleWiseInQueue, 1, tilingData->promptAttentionTensorSizeRect.mmResUbSize * sizeof(mmOutputType));
        } else {
            pipe->InitBuffer(tempBmm2Ub, tilingData->promptAttentionTensorSizeRect.bmm2ResUbSize * sizeof(mmOutputType));
            pipe->InitBuffer(tempBmm2Queue, 1, tilingData->promptAttentionTensorSizeRect.bmm2ResUbSize * sizeof(mmOutputType));
        }
    }
    pipe->InitBuffer(Bmm1Queue, 1, tilingData->promptAttentionTensorSizeRect.mmResUbSize * sizeof(mmOutputType));
    if ((tilingData->promptAttentionTensorSizeRect.tmpSoftMaxV2Size) != 0 && (tilingData->promptAttentionTensorSizeRect.mm2TmpUbSize != 0)) {
        pipe->InitBuffer(tmpSoftmaxFlashV2Ub_, tilingData->promptAttentionTensorSizeRect.tmpSoftMaxV2Size / UB_ALIGN * UB_ALIGN * sizeof(uint8_t));
        pipe->InitBuffer(tmpmm2Ub_, tilingData->promptAttentionTensorSizeRect.mm2TmpUbSize * sizeof(uint8_t));
    }
    if (tilingData->promptAttentionTensorSizeRect.selectSpaceUbSize != 0) {
        pipe->InitBuffer(selectSpaceUb, tilingData->promptAttentionTensorSizeRect.selectSpaceUbSize);
    }
    useMask = false;
    attentionMaskType = tilingData->promptAttentionBaseParams.sparseMode;
    if ((attenMask != NULL) && (tilingData->promptAttentionBaseParams.useMask == 1)) {
        useMask = true;
        attenMaskGm.SetGlobalBuffer((__gm__ U*)attenMask);
        attentionMaskStride = tilingData->promptAttentionBaseParams.maskKVsSize;
    }

    usePseShift = false;
    if ((pseShift != NULL) && (tilingData->promptAttentionBaseParams.usePseShift == 1)) {
        usePseShift = true;
        pseShiftGm.SetGlobalBuffer((__gm__ pseShiftType*)pseShift);
        pseShiftStride = tilingData->promptAttentionBaseParams.pseShiftS2Size;

        if constexpr (AscendC::IsSameType<pseShiftCastType, float>::value) {
            pipe->InitBuffer(pseShiftCastUb,
                             (tilingData->promptAttentionTensorSizeRect.pseShiftUbSize) * sizeof(float));
        }
    }

    if (tilingData->promptAttentionInitOutputParams.needInit == 1) {
        InitOutputSingleCore();
    }
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::InitQuant(__gm__ uint8_t* deq_scale1,
                                                             __gm__ uint8_t* scale1, __gm__ uint8_t* deq_scale2,
                                                             __gm__ uint8_t* scale2, __gm__ uint8_t* offset2) {
    if (deq_scale1 != nullptr) {
        if(tilingData->promptAttentionBaseParams.deqScaleFlag == 1){
            deqScale1Fp32Gm.SetGlobalBuffer((__gm__ uint32_t*)deq_scale1);
            dequantScale1 = deqScale1Fp32Gm(0);
        } else {
            dequantScale1 = *(reinterpret_cast<__gm__ uint64_t*>(deq_scale1));
        }
    }
    if (scale1 != nullptr) { quantScale1 = *(reinterpret_cast<__gm__ float*>(scale1));}
    if (deq_scale2 != nullptr) {
        if(tilingData->promptAttentionBaseParams.deqScale2Flag == 1){
            deqScale2Fp32Gm.SetGlobalBuffer((__gm__ uint32_t*)deq_scale2);
            dequantScale2 = deqScale2Fp32Gm(0);
        } else {
            dequantScale2 = *(reinterpret_cast<__gm__ uint64_t*>(deq_scale2));
        }
    }
    if (scale2 != nullptr) { quantScale2 = *(reinterpret_cast<__gm__ float*>(scale2));}
    if (offset2 != nullptr) { quantOffset2 = *(reinterpret_cast<__gm__ float*>(offset2));}
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::InitMsd(__gm__ uint8_t* key_antiquant_scale, __gm__ uint8_t* key_antiquant_offset,
                                                                             __gm__ uint8_t* value_antiquant_scale, __gm__ uint8_t* value_antiquant_offset){
    return;
}


template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::InitOutputSingleCore()
{
    auto &initParams = tilingData->promptAttentionInitOutputParams;
    uint32_t tailSize = initParams.totalOutputSize - tmp_block_idx * initParams.singleCoreSize;
    uint32_t singleInitOutputSize = tailSize < initParams.singleCoreSize ? tailSize : initParams.singleCoreSize;
    InitOutput<O>(attentionOutGm[tmp_block_idx * initParams.singleCoreSize], singleInitOutputSize, 0);
    SyncAll();
}
#ifndef PFA_UT
template<>
__aicore__ inline void PromptFlashAttentionBase<int8_t, bool, CubeFormat::ND, int8_t>::InitOutputSingleCore() {}

template<>
__aicore__ inline void PromptFlashAttentionBase<int8_t, half, CubeFormat::ND, int8_t>::InitOutputSingleCore() {}

template<>
__aicore__ inline void PromptFlashAttentionBase<int8_t, float, CubeFormat::ND, int8_t>::InitOutputSingleCore() {}
#endif
template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::initOffset() {
    offsetSS = static_cast<uint64_t>(tilingData->promptAttentionBaseParams.seqSize) *
        tilingData->promptAttentionBaseParams.seqSize;
    offsetSH = static_cast<uint64_t>(tilingData->promptAttentionBaseParams.seqSize) *
        tilingData->promptAttentionBaseParams.headSize;
    offsetSTypeNum = static_cast<uint64_t>(tilingData->promptAttentionBaseParams.seqSize) * typeByteNum;
    offsetNSTypeNum = static_cast<uint64_t>(tilingData->promptAttentionBaseParams.headNumSize) * offsetSTypeNum;
    offsetNSS = static_cast<uint64_t>(tilingData->promptAttentionBaseParams.headNumSize) * offsetSS;
    offsetNSH = static_cast<uint64_t>(tilingData->promptAttentionBaseParams.headNumSize) * offsetSH;
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::InitTensorSize(
                const PromptAttentionSingleCoreTensorSize* tensorSizeTiling) {
    mmResUbSize = tensorSizeTiling->mmResUbSize;            // Matrix mu result UB size
    attenMaskUbSize = tensorSizeTiling->attenMaskUbSize;    // Attention mask UB size
    pseShiftUbSize = tensorSizeTiling->pseShiftUbSize;      // PSE shift UB size
    maskSize = tensorSizeTiling->maskSize;                  // Mask size
    softmaxMaxSize = tensorSizeTiling->softmaxMaxSize;      // Softmax max UB size
    softmaxSumSize = tensorSizeTiling->softmaxSumSize;      // Softmax sum UB size
    softmaxExpSize = tensorSizeTiling->softmaxExpSize;      // Softmax exp UB size
    spmTmpSize = tensorSizeTiling->spmTmpSize;              // SPM temporary size
    scmTmpSize = tensorSizeTiling->scmTmpSize;              // SCM temporary size
    bmm2ResUbSize = tensorSizeTiling->bmm2ResUbSize;        // Second matric mul result UB size
    tmpMMResBmm2PreUbSize = tensorSizeTiling->tmpMMResBmm2PreUbSize;
    tmpSoftmaxBmm2UbSize = tensorSizeTiling->tmpSoftmaxBmm2UbSize;
    selectSpaceUbSize = tensorSizeTiling->selectSpaceUbSize;
    softMaxV2Size_ = tensorSizeTiling->tmpSoftMaxV2Size / UB_ALIGN * UB_ALIGN;
    mm2TmpUbSize_ = tensorSizeTiling->mm2TmpUbSize;
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::ElewiseCompute(LocalTensor<mmOutputType>& mmResUb,
                                                                                    uint32_t computeSize, uint32_t type) {
    if (useMask) {
        if (maskTypeByteNum == BOOLBYTENUM) {
            LocalTensor<U> attenMaskUb = attenMaskQueue.DeQue<U>();
            attenMaskUb.SetSize(this->singleProcessSOuterSize * this->maskCopyInCol);
            LocalTensor<uint8_t> selectSpace = selectSpaceUb.Get<uint8_t>(selectSpaceUbSize);
            mmOutputType scalar;
            if constexpr (IsSameType<mmOutputType, float>::value) {
                uint32_t tmp = 0xFF7FFFFF;  // Minimum of float
                scalar = *((float *)&tmp);
            } else {
                uint16_t tmp = 0xFBFF;   // Minimum of FP16
                scalar = *((half *)&tmp);
            }
            SelectWithBytesMaskShapeInfo selectWithBytesMaskShapeInfo;
            selectWithBytesMaskShapeInfo.firstAxis = this->singleProcessSOuterSize;
            selectWithBytesMaskShapeInfo.srcLastAxis = this->singleProcessSInnerSizeNow;
            selectWithBytesMaskShapeInfo.maskLastAxis = this->maskCopyInCol;
            if(type == 0){
                SelectWithBytesMask(mmResUb, mmResUb, static_cast<mmOutputType>(scalar), attenMaskUb, selectSpace,
                                selectWithBytesMaskShapeInfo);
            } else if(type == 1) {
                SelectWithBytesMask(mmResUb, static_cast<mmOutputType>(scalar), mmResUb, attenMaskUb, selectSpace,
                                selectWithBytesMaskShapeInfo); // swape param 2 and param 3 of SelectWithBytesMask to compute attenMaskPre for band mode
            }

            PipeBarrier<PIPE_V>();
            attenMaskQueue.FreeTensor(attenMaskUb);
        } else {
            LocalTensor<mmOutputType> attenMaskUb = attenMaskQueue.DeQue<mmOutputType>();
            Muls(attenMaskUb, attenMaskUb, static_cast<mmOutputType>(-10000.0), computeSize);
            PipeBarrier<PIPE_V>();
            Add(mmResUb, mmResUb, attenMaskUb, computeSize);
            attenMaskQueue.FreeTensor(attenMaskUb);
        }
    }
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::SoftmaxBasicComputeFirst(LocalTensor<mmOutputType>& mmResUb,
                                        LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb, SoftMaxShapeInfo& shapeInfo) {
    SoftMax<softmaxType, true, true> (mmResUb, softmaxSumUb, softmaxMaxUb, mmResUb, softmaxTilingData, shapeInfo);
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::SoftmaxComputeFirst(LocalTensor<mmOutputType>& mmResUb,
                                        LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb, SoftMaxShapeInfo& shapeInfo) {
    SoftMax<softmaxType, true> (mmResUb, softmaxSumUb, softmaxMaxUb, mmResUb, softmaxTilingData, shapeInfo);
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline bool PromptFlashAttentionBase<T, U, FORMAT, O, M>::IsSoftmaxBasic() {
    return ((this->softmaxTilingData.splitM % 8 ==0) && (this->softmaxTilingData.splitK % 64 ==0));
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::SoftmaxBasicCompute(LocalTensor<mmOutputType>& mmResUb,
                                        LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                        LocalTensor<softmaxType>& softmaxExpUb, SoftMaxShapeInfo& shapeInfo) {
    SoftmaxFlash<softmaxType, true, true> (mmResUb, softmaxSumUb, softmaxMaxUb,
                                mmResUb, softmaxExpUb, softmaxSumUb,
                                softmaxMaxUb, softmaxFlashTilingData, true, shapeInfo);
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::SoftmaxCompute(LocalTensor<mmOutputType>& mmResUb,
                                        LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                        LocalTensor<softmaxType>& softmaxExpUb, SoftMaxShapeInfo& shapeInfo) {
    SoftmaxFlash<softmaxType, true> (mmResUb, softmaxSumUb, softmaxMaxUb,
                          mmResUb, softmaxExpUb, softmaxSumUb,
                          softmaxMaxUb, softmaxFlashTilingData, true, shapeInfo);
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::SoftmaxBasicComputeFirstNoTail(LocalTensor<mmOutputType>& mmResUb,
                                                        LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                                        LocalTensor<softmaxType>& softmaxExpUb, LocalTensor<uint8_t>& sharedTmpUb,
                                                        SoftMaxShapeInfo& shapeInfo) {
    SoftmaxFlashV2<softmaxType, false, true, true, true>(mmResUb, softmaxSumUb, softmaxMaxUb,
                                         mmResUb, softmaxExpUb, softmaxSumUb, softmaxMaxUb, sharedTmpUb, softmaxFlashTilingData, shapeInfo);
                                                        }

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::SoftmaxBasicComputeNoTail(LocalTensor<mmOutputType>& mmResUb,
                                                    LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                                    LocalTensor<softmaxType>& softmaxExpUb, LocalTensor<uint8_t>& sharedTmpUb,
                                                    SoftMaxShapeInfo& shapeInfo) {
    SoftmaxFlashV2<softmaxType, true, true, true, true>(mmResUb, softmaxSumUb, softmaxMaxUb,
                                        mmResUb, softmaxExpUb, softmaxSumUb, softmaxMaxUb, sharedTmpUb, softmaxFlashTilingData, shapeInfo);
                                                    }

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::SoftmaxBasicComputeFirstNoTail(LocalTensor<mmOutputType>& mmResUb,
                                            LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                            LocalTensor<softmaxType>& softmaxExpUb, SoftMaxShapeInfo& shapeInfo) {
    SoftmaxFlashV2<softmaxType, false, true, true>(mmResUb, softmaxSumUb, softmaxMaxUb,
                                         mmResUb, softmaxExpUb, softmaxSumUb, softmaxMaxUb, softmaxFlashTilingData, shapeInfo);
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::SoftmaxBasicComputeNoTail(LocalTensor<mmOutputType>& mmResUb,
                                            LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                            LocalTensor<softmaxType>& softmaxExpUb, SoftMaxShapeInfo& shapeInfo) {
    SoftmaxFlashV2<softmaxType, true, true, true>(mmResUb, softmaxSumUb, softmaxMaxUb,
                                        mmResUb, softmaxExpUb, softmaxSumUb, softmaxMaxUb, softmaxFlashTilingData, shapeInfo);
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::SoftmaxComputeFirstTail(LocalTensor<mmOutputType>& mmResUb,
                                            LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                            LocalTensor<softmaxType>& softmaxExpUb, SoftMaxShapeInfo& shapeInfo) {
    SoftmaxFlashV2<softmaxType, false, true, false>(mmResUb, softmaxSumUb, softmaxMaxUb,
                                          mmResUb, softmaxExpUb, softmaxSumUb, softmaxMaxUb, softmaxFlashTilingData, shapeInfo);
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::SoftmaxComputeTail(LocalTensor<mmOutputType>& mmResUb,
                                            LocalTensor<float>& softmaxMaxUb, LocalTensor<float>& softmaxSumUb,
                                            LocalTensor<softmaxType>& softmaxExpUb, SoftMaxShapeInfo& shapeInfo) {
    SoftmaxFlashV2<softmaxType, true, true, false>(mmResUb, softmaxSumUb, softmaxMaxUb,
                                         mmResUb, softmaxExpUb, softmaxSumUb, softmaxMaxUb, softmaxFlashTilingData, shapeInfo);
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline bool PromptFlashAttentionBase<T, U, FORMAT, O, M>::IsSoftmaxFlashBasic() {
    return ((this->softmaxFlashTilingData.splitM % 8 ==0) && (this->softmaxFlashTilingData.splitK % 64 ==0));
}

// quant: add quant functions
template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::QuantCompute(LocalTensor<int8_t> quantResUb, LocalTensor<mmOutputType> mmResUb,
                                                                                    float scale, float offset, uint32_t computeSize) {
    AscendQuant(quantResUb, mmResUb, scale, offset, computeSize);
}


template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::Bmm2UpdateDivNoTail(LocalTensor<mmOutputType>& bmm2ResPreUb,
                                            LocalTensor<float>& softmaxSumUb, LocalTensor<softmaxType>& softmaxExpUb) {
    int32_t headLoop = (tilingData->promptAttentionBaseParams.headSize + softmaxTypeByteNum - 1) / softmaxTypeByteNum;
    constexpr int32_t REPEAT_DATA_NUM = 256 / sizeof(softmaxType);

    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 1;
    repeatParams.src0RepStride = headLoop;
    repeatParams.src1BlkStride = 0;
    repeatParams.src1RepStride = 1;
    repeatParams.dstRepStride = headLoop;

    int32_t loop = tilingData->promptAttentionBaseParams.headSize / REPEAT_DATA_NUM;
    int32_t remain = tilingData->promptAttentionBaseParams.headSize % REPEAT_DATA_NUM;
    if constexpr (IsSameType<softmaxType, half>::value) {
        constexpr int32_t FP32_BLOCK_NUM = 8;
        constexpr int32_t FP32_MASK_NUM = 64;
        CopyRepeatParams copyRepeatParams{2, 1, 16, 8};
        int32_t calcSize = singleProcessSOuterSize * FP32_BLOCK_NUM;
        LocalTensor<float> tmpBuffer = attenMaskQueue.template AllocTensor<float>();

        int32_t repeat = (calcSize + FP32_MASK_NUM - 1) / FP32_MASK_NUM;
        AscendC::Copy(tmpBuffer, softmaxSumUb, FP32_MASK_NUM, repeat, copyRepeatParams);
        AscendC::Copy(tmpBuffer[FP32_BLOCK_NUM], softmaxSumUb, FP32_MASK_NUM, repeat, copyRepeatParams);
        PipeBarrier<PIPE_V>();
        Cast(softmaxExpUb, tmpBuffer, RoundMode::CAST_ROUND, calcSize * 2);
        PipeBarrier<PIPE_V>();
        attenMaskQueue.FreeTensor(tmpBuffer);

        for (int i = 0; i < loop; i++) {
            Div(bmm2ResPreUb[i * REPEAT_DATA_NUM], bmm2ResPreUb[i * REPEAT_DATA_NUM], softmaxExpUb,
                REPEAT_DATA_NUM, singleProcessSOuterSize, repeatParams);
        }
        if (remain) {
            Div(bmm2ResPreUb[loop * REPEAT_DATA_NUM], bmm2ResPreUb[loop * REPEAT_DATA_NUM], softmaxExpUb,
                remain, singleProcessSOuterSize, repeatParams);
        }
    } else {
        for (int i = 0; i < loop; i++) {
            Div(bmm2ResPreUb[i * REPEAT_DATA_NUM], bmm2ResPreUb[i * REPEAT_DATA_NUM], softmaxSumUb,
                REPEAT_DATA_NUM, singleProcessSOuterSize, repeatParams);
        }
        if (remain) {
            Div(bmm2ResPreUb[loop * REPEAT_DATA_NUM], bmm2ResPreUb[loop * REPEAT_DATA_NUM], softmaxSumUb,
                remain, singleProcessSOuterSize, repeatParams);
        }
    }
}

template<typename T, typename U, CubeFormat FORMAT, typename O, OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::Bmm2Compute(uint32_t offset, LocalTensor<mmOutputType>& bmm2ResL1) {
#if (__CCE_AICORE__ > 200)
    if constexpr (IsSameType<mmInputType, bfloat16_t>::value ||
                  (IsSameType<mmInputType, half>::value &&
                  IsSameType<mmOutputType, float>::value)) {
        PipeBarrier<PIPE_V>();
        LocalTensor<mmInputType> tmpBmm2ResCastTensor; // The same ub buffer is used before and after the cast.
        tmpBmm2ResCastTensor = bmm2ResL1.template ReinterpretCast<mmInputType>();
        tmpBmm2ResCastTensor.SetSize(bmm2ResL1.GetSize());
        Cast(tmpBmm2ResCastTensor, bmm2ResL1, RoundMode::CAST_ROUND, bmm2ResL1.GetSize());
        bmm2.SetTensorA(tmpBmm2ResCastTensor);
    } else if constexpr (IsSameType<mmInputType, int8_t>::value) {
        LocalTensor<int8_t> softmaxQuantResUb;
        softmaxQuantResUb = bmm2ResL1.template ReinterpretCast<int8_t>();
        softmaxQuantResUb.SetSize(bmm2ResL1.GetSize());
        QuantCompute(softmaxQuantResUb, bmm2ResL1, quantScale1, 0, mmResUbSize);
        bmm2.SetTensorA(softmaxQuantResUb);
    } else {
        bmm2.SetTensorA(bmm2ResL1);
    }
#else
    bmm2.SetTensorA(bmm2ResL1);
#endif
    bmm2.SetTensorB(valueGm[offset]);
    if constexpr (IsSameType<mmInputType, int8_t>::value) {
        bmm2.SetQuantScalar(dequantScale2);
    }
    bmm2.SetTail(-1, -1, singleProcessSInnerBmmTail);
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::UpdateVmul(LocalTensor<softmaxType>& softmaxExpUb) {
    LocalTensor<mmOutputType> bmm2ResPreUb = tempBmm2Ub.Get<mmOutputType>(bmm2ResUbSize);

    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = 1;
    repeatParams.src0BlkStride = 0;
    repeatParams.src1RepStride = (
        tilingData->promptAttentionBaseParams.headSize + softmaxTypeByteNum - 1) / softmaxTypeByteNum;
    repeatParams.dstRepStride = (
        tilingData->promptAttentionBaseParams.headSize + softmaxTypeByteNum - 1) / softmaxTypeByteNum;

    // only support singleProcessSOuterSize <=255, headsize 32B align
    int32_t numOneRep = 256 / sizeof(softmaxType);
    int32_t loop = tilingData->promptAttentionBaseParams.headSize / numOneRep;
    int32_t remain =  tilingData->promptAttentionBaseParams.headSize % numOneRep;

    for (int i = 0; i < loop; i++) {
        Mul(bmm2ResPreUb[i * numOneRep], softmaxExpUb, bmm2ResPreUb[i * numOneRep],
            numOneRep, singleProcessSOuterSize, repeatParams);
    }
    if (remain) {
        Mul(bmm2ResPreUb[loop * numOneRep], softmaxExpUb, bmm2ResPreUb[loop * numOneRep],
            remain, singleProcessSOuterSize, repeatParams);
    }
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::ComputePseShiftOffset(int sInnerOffsetDataSize) {
    if (!usePseShift) {
        return;
    }

    pseShiftOffset = pseShiftCoreOffset + (uint64_t)sInnerOffsetDataSize;
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::ComputeAttenMaskOffset(int sInnerOffsetDataSize) {
    int32_t delta;
    if (attentionMaskType == 2 || attentionMaskType == 3 || attentionMaskType == 4) { // 2:leftUp mode of sparseMode, 3:rightdown mode of sparseMode, 4:band mode of sparseMode
        if (attentionMaskType == 2) {
            delta = sOuterOffset - sInnerOffsetDataSize + tilingData->promptAttentionBaseParams.nextTokens;
        } else {
            delta = sOuterOffset - sInnerOffsetDataSize + nextTokensPerBatch;
        }

        if (delta < 0) {
            attenMaskOffset = ((int32_t)singleProcessSOuterSizeWhole + delta) > 0 ? (-delta) : singleProcessSOuterSizeWhole;
        }
        else {
            attenMaskOffset = (((int32_t)singleProcessSInnerSize - delta) > 0 ? delta : singleProcessSInnerSize) * attentionMaskStride;
        }
    } else {
        attenMaskOffset = attenMaskCoreOffset + (uint64_t)sInnerOffsetDataSize;
    }
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::ComputeAttenMaskOffsetPre(int sInnerOffsetDataSize) {
    if (attentionMaskType == 0 || attentionMaskType == 1) {
        return;
    }
    int32_t delta;
    delta = sOuterOffset - sInnerOffsetDataSize - preTokensPerBatch -1;
    if (delta < 0) {
        attenMaskOffsetPre = ((int32_t)singleProcessSOuterSizeWhole + delta) > 0 ? (-delta) : singleProcessSOuterSizeWhole;
    }
    else {
        attenMaskOffsetPre = (((int32_t)singleProcessSInnerSize - delta) > 0 ? delta : singleProcessSInnerSize) * attentionMaskStride;
    }
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::ComputeOffset(uint32_t sInnerLoopIdx) {
    int sInnerOffsetDataSize = sInnerLoopIdx * singleProcessSInnerSize;
    ComputePseShiftOffset(sInnerOffsetDataSize);
    ComputeAttenMaskOffset(sInnerOffsetDataSize);
    ComputeAttenMaskOffsetPre(sInnerOffsetDataSize);
    valueOffset = valueCoreOffset + sInnerOffsetDataSize * MultiHeadKV;
    tensorAOffset = tensorACoreOffset;
    tensorBOffset = tensorBCoreOffset + sInnerOffsetDataSize * MultiHeadKV;
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::ComputeOffsetWithBNSD(uint32_t sInnerLoopIdx) {
    int sInnerOffsetDataSize = sInnerLoopIdx * singleProcessSInnerSize;
    ComputePseShiftOffset(sInnerOffsetDataSize);
    ComputeAttenMaskOffset(sInnerOffsetDataSize);
    ComputeAttenMaskOffsetPre(sInnerOffsetDataSize);
    valueOffset = valueCoreOffset + sInnerOffsetDataSize * tilingData->promptAttentionBaseParams.headSize;
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::DataCopyTransposeOut(LocalTensor<mmOutputType>& bmm2ResUb) {
    TransposeParams transposeParams;
    transposeParams.bIndex = 0;
    transposeParams.nIndex = batchNOffset;
    transposeParams.sIndex = sOuterOffset;
    transposeParams.hNIndex = 0;
    if (preTokensPerBatch < 0) {
        int32_t preTokenLength = actualSeqLengthKVPerBatch + preTokensPerBatch;
        if (sOuterOffset < preTokenLength && (sOuterOffset + this->singleProcessSOuterSize) > preTokenLength) {
            preTokensOffset = sOuterOffset + this->singleProcessSOuterSize - preTokenLength;
        } else {
            preTokensOffset = 0;
        }
    }
    CopyTransposeTiling transposeTilingData22 = tilingData->transposeTilingDataRect;
    transposeTilingData22.srcShapeS = singleProcessSOuterSize - preTokensOffset - nextTokensOffset;
    transposeTilingData22.invalidParamCopyTransposeTiling = 0;
    transposeParams.sIndex = transposeParams.sIndex + nextTokensOffset;

    if constexpr (IsSameType<O, half>::value && IsSameType<mmOutputType, half>::value) {
        event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(enQueEvtID);
        WaitFlag<HardEvent::V_MTE3>(enQueEvtID);
        if (tilingData->promptAttentionBaseParams.headSize == tilingData->promptAttentionBaseParams.alignedHeadSize) {
            DataCopyTranspose2<O> (attentionOutGm, bmm2ResUb[nextTokensOffset * tilingData->promptAttentionBaseParams.headSize],
                                   CopyTransposeType::TRANSPOSE_ND_UB_GM, transposeParams, transposeTilingData22, multiSeqOffset);
        } else {
            DataCopyParams intriParams;
            intriParams.blockCount = 1;
            intriParams.blockLen = tilingData->promptAttentionBaseParams.headSize * sizeof(O); // This should be unaligned
            intriParams.srcStride = 0;
            intriParams.dstStride = 0;
            int startAddr = multiSeqOffset + transposeParams.nIndex * transposeTilingData22.dstShapeHN +
                (transposeParams.sIndex + nextTokensOffset) * transposeTilingData22.dstShapeH + transposeParams.hNIndex;
            for (int i = 0; i < transposeTilingData22.srcShapeB; i++) {
                for (int j = 0; j < transposeTilingData22.srcShapeN; j++) {
                    for (int k = nextTokensOffset; k < transposeTilingData22.srcShapeS; k++) {
                        DataCopyPad(attentionOutGm[startAddr + i * (transposeTilingData22.shapeSHValue) +
                            j * transposeTilingData22.dstShapeHN + k * transposeTilingData22.dstShapeH],
                            bmm2ResUb[k * tilingData->promptAttentionBaseParams.alignedHeadSize +
                            j * transposeTilingData22.shapeNsValue + i * transposeTilingData22.shapeNsnValue],
                            intriParams);
                    }
                }
            }
        }
    }
#if (__CCE_AICORE__ > 200)
    // Execute the code when core version is greater than 200.
    if constexpr (IsSameType<O, bfloat16_t>::value ||
                  (IsSameType<O, half>::value &&
                  IsSameType<mmOutputType, float>::value)) {
        LocalTensor<O> tmpBmm2ResCastTensor; // 原地Cast
        tmpBmm2ResCastTensor = bmm2ResUb.template ReinterpretCast<O>();
        tmpBmm2ResCastTensor.SetSize(bmm2ResUb.GetSize());  // Set the size of the converted
        PipeBarrier<PIPE_V>();
        if (tilingData->promptAttentionBaseParams.headSize == tilingData->promptAttentionBaseParams.alignedHeadSize) {
            Cast(tmpBmm2ResCastTensor, bmm2ResUb, RoundMode::CAST_ROUND, bmm2ResUb.GetSize());
            event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(enQueEvtID);
            WaitFlag<HardEvent::V_MTE3>(enQueEvtID);
            DataCopyTranspose2<O> (attentionOutGm, tmpBmm2ResCastTensor[nextTokensOffset * tilingData->promptAttentionBaseParams.headSize],
                                   CopyTransposeType::TRANSPOSE_ND_UB_GM, transposeParams, transposeTilingData22, multiSeqOffset);
        } else {
            DataCopyParams intriParams;
            intriParams.blockCount = 1;
            intriParams.blockLen = tilingData->promptAttentionBaseParams.headSize * sizeof(O); // This should be unaligned
            intriParams.srcStride = 0;
            intriParams.dstStride = 0;

            int32_t headLoop = tilingData->promptAttentionBaseParams.alignedHeadSize / outputTypeByteNum;
            constexpr int32_t REPEAT_DATA_NUM = 256 / sizeof(outputTypeByteNum);
            int32_t loop = tilingData->promptAttentionBaseParams.headSize / REPEAT_DATA_NUM;
            int32_t remain = tilingData->promptAttentionBaseParams.headSize % REPEAT_DATA_NUM;
            UnaryRepeatParams repeatParams;
            repeatParams.srcBlkStride = 1;
            repeatParams.srcRepStride = (tilingData->promptAttentionBaseParams.headSize * 4 + 32 -1) / 32;
            repeatParams.dstBlkStride = 1;
            repeatParams.dstRepStride = (tilingData->promptAttentionBaseParams.headSize * 2 + 32 -1) / 32;
            for (int i = 0; i < loop; i++) {
                Cast(tmpBmm2ResCastTensor[i * REPEAT_DATA_NUM], bmm2ResUb[i * REPEAT_DATA_NUM], RoundMode::CAST_ROUND, REPEAT_DATA_NUM,
                    this->singleProcessSOuterSize, repeatParams);
            }
            if (remain) {
                Cast(tmpBmm2ResCastTensor[loop * REPEAT_DATA_NUM], bmm2ResUb[loop * REPEAT_DATA_NUM], RoundMode::CAST_ROUND, remain,
                    this->singleProcessSOuterSize, repeatParams);
            }
            event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(enQueEvtID);
            WaitFlag<HardEvent::V_MTE3>(enQueEvtID);

            int startAddr = multiSeqOffset + transposeParams.nIndex * transposeTilingData22.dstShapeHN +
                (transposeParams.sIndex + nextTokensOffset) * transposeTilingData22.dstShapeH + transposeParams.hNIndex;
            for (int i = 0; i < transposeTilingData22.srcShapeB; i++) {
                for (int j = 0; j < transposeTilingData22.srcShapeN; j++) {
                    for (int k = nextTokensOffset; k < transposeTilingData22.srcShapeS; k++) {
                        DataCopyPad(attentionOutGm[startAddr + i * (transposeTilingData22.shapeSHValue) +
                            j * transposeTilingData22.dstShapeHN + k * transposeTilingData22.dstShapeH],
                            tmpBmm2ResCastTensor[k * tilingData->promptAttentionBaseParams.alignedHeadSize +
                            j * transposeTilingData22.shapeNsValue + i * transposeTilingData22.shapeNsnValue],
                            intriParams);
                    }
                }
            }
        }
    }
#endif
    if constexpr (IsSameType<O, int8_t>::value) {
        LocalTensor<int8_t> outputQuantRes;
        outputQuantRes = bmm2ResUb.template ReinterpretCast<int8_t>();
        outputQuantRes.SetSize(bmm2ResUb.GetSize());
        QuantCompute(outputQuantRes, bmm2ResUb, quantScale2, quantOffset2, bmm2ResUbSize);
        // Quantifying vector computation and interpolating synchronization between datacopy to ensure timing
        event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(enQueEvtID);
        WaitFlag<HardEvent::V_MTE3>(enQueEvtID);
        DataCopyTranspose2<O> (attentionOutGm, outputQuantRes[nextTokensOffset * tilingData->promptAttentionBaseParams.headSize],
                               CopyTransposeType::TRANSPOSE_ND_UB_GM, transposeParams, transposeTilingData22, multiSeqOffset);
    }
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::DataCopyOutWithBNSD(LocalTensor<mmOutputType>& bmm2ResUb) {
    uint32_t copySize = (this->singleProcessSOuterSize - nextTokensOffset) * tilingData->promptAttentionBaseParams.headSize;
    if (preTokensPerBatch < 0) {
        int32_t preTokenLength = actualSeqLengthKVPerBatch + preTokensPerBatch;
        if (sOuterOffset < preTokenLength && (sOuterOffset + this->singleProcessSOuterSize) > preTokenLength) {
            preTokensOffset = sOuterOffset + this->singleProcessSOuterSize - preTokenLength;
            copySize = copySize - preTokensOffset * tilingData->promptAttentionBaseParams.headSize;
        } else {
            preTokensOffset = 0;
        }
    }
    struct DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = copySize / outputTypeByteNum;
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    if constexpr (IsSameType<O, half>::value && IsSameType<mmOutputType, half>::value) {
        event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(enQueEvtID);
        WaitFlag<HardEvent::V_MTE3>(enQueEvtID);
        if (tilingData->promptAttentionBaseParams.headSize == tilingData->promptAttentionBaseParams.alignedHeadSize) {
            DataCopy(attentionOutGm[attentionOutOffset],
                     bmm2ResUb[nextTokensOffset * tilingData->promptAttentionBaseParams.headSize], dataCopyParams);
        } else {
            dataCopyParams.blockCount = this->singleProcessSOuterSize - preTokensOffset;
            dataCopyParams.blockLen = tilingData->promptAttentionBaseParams.headSize * sizeof(O); // This should be unaligned
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPad(attentionOutGm[attentionOutOffset],
                        bmm2ResUb[nextTokensOffset * tilingData->promptAttentionBaseParams.headSize], dataCopyParams);
        }
    }
#if (__CCE_AICORE__ > 200)
    if constexpr (IsSameType<O, bfloat16_t>::value ||
                  (IsSameType<O, half>::value &&
                  IsSameType<mmOutputType, float>::value)) {
        LocalTensor<O> tmpBmm2ResCastTensor; // The same ub buffer is used before and after the cast.
        tmpBmm2ResCastTensor = bmm2ResUb.template ReinterpretCast<O>();
        tmpBmm2ResCastTensor.SetSize(bmm2ResUb.GetSize());
        PipeBarrier<PIPE_V>();
        if (tilingData->promptAttentionBaseParams.headSize == tilingData->promptAttentionBaseParams.alignedHeadSize) {
            Cast(tmpBmm2ResCastTensor, bmm2ResUb, RoundMode::CAST_ROUND, bmm2ResUb.GetSize());
            event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(enQueEvtID);
            WaitFlag<HardEvent::V_MTE3>(enQueEvtID);
            DataCopy(attentionOutGm[attentionOutOffset],
                     tmpBmm2ResCastTensor[nextTokensOffset * tilingData->promptAttentionBaseParams.headSize], dataCopyParams);
        } else {
            int32_t headLoop = tilingData->promptAttentionBaseParams.alignedHeadSize / outputTypeByteNum;
            constexpr int32_t REPEAT_DATA_NUM = 256 / sizeof(outputTypeByteNum);
            int32_t loop = tilingData->promptAttentionBaseParams.headSize / REPEAT_DATA_NUM;
            int32_t remain = tilingData->promptAttentionBaseParams.headSize % REPEAT_DATA_NUM;
            UnaryRepeatParams repeatParams;
            repeatParams.srcBlkStride = 1;
            repeatParams.srcRepStride = (tilingData->promptAttentionBaseParams.headSize * 4 + 32 -1) / 32;
            repeatParams.dstBlkStride = 1;
            repeatParams.dstRepStride = (tilingData->promptAttentionBaseParams.headSize * 2 + 32 -1) / 32;
            for (int i = 0; i < loop; i++) {
                Cast(tmpBmm2ResCastTensor[i * REPEAT_DATA_NUM], bmm2ResUb[i * REPEAT_DATA_NUM], RoundMode::CAST_ROUND, REPEAT_DATA_NUM,
                    this->singleProcessSOuterSize, repeatParams);
            }
            if (remain) {
                Cast(tmpBmm2ResCastTensor[loop * REPEAT_DATA_NUM], bmm2ResUb[loop * REPEAT_DATA_NUM], RoundMode::CAST_ROUND, remain,
                    this->singleProcessSOuterSize, repeatParams);
            }
            event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(enQueEvtID);
            WaitFlag<HardEvent::V_MTE3>(enQueEvtID);

            dataCopyParams.blockCount = this->singleProcessSOuterSize;
            dataCopyParams.blockLen = tilingData->promptAttentionBaseParams.headSize * sizeof(O); // This should be unaligned
            dataCopyParams.srcStride = 0;
            dataCopyParams.dstStride = 0;
            DataCopyPad(attentionOutGm[attentionOutOffset],
                        tmpBmm2ResCastTensor[nextTokensOffset * tilingData->promptAttentionBaseParams.headSize], dataCopyParams);
        }
    }
#endif
    if constexpr (IsSameType<O, int8_t>::value) {
        LocalTensor<int8_t> outputQuantRes;
        outputQuantRes = bmm2ResUb.template ReinterpretCast<int8_t>();
        outputQuantRes.SetSize(bmm2ResUb.GetSize());
        QuantCompute(outputQuantRes, bmm2ResUb, quantScale2, quantOffset2, bmm2ResUbSize);
        // Insert synchronization between quantization calculation and datacopy to ensure accurate timing.
        event_t enQueEvtID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(enQueEvtID);
        WaitFlag<HardEvent::V_MTE3>(enQueEvtID);
        DataCopy(attentionOutGm[attentionOutOffset],
                 outputQuantRes[nextTokensOffset * tilingData->promptAttentionBaseParams.headSize], dataCopyParams);
    }
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::CalPseShiftOffset(int sIdx) {
    if (!usePseShift) {
        return;
    }

    uint64_t pseShiftBatchOffset = 0;
    uint64_t pseShiftN = (uint64_t)tilingData->promptAttentionBaseParams.headNumSize;
    uint64_t pseShiftS1 = (uint64_t)tilingData->promptAttentionBaseParams.pseShiftS1Size;
    uint64_t pseShiftS2 = (uint64_t)tilingData->promptAttentionBaseParams.pseShiftS2Size;

    if (pseShiftBatch != 1) {
        pseShiftBatchOffset = (uint64_t)sIdx * pseShiftN * pseShiftS1 * pseShiftS2;
    }

    pseShiftCoreOffset = pseShiftBatchOffset + (uint64_t)batchNOffset * pseShiftS1 * pseShiftS2 +
                         (uint64_t)sOuterOffset * pseShiftS2;
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::LoopSOuterOffsetInit(uint32_t seqListOffsetSize, int sIdx) {
    uint64_t attenMaskBatchOffset = 0;
    if (attenMaskBatch != 1) {
        attenMaskBatchOffset = (uint64_t)sIdx * (uint64_t)tilingData->promptAttentionBaseParams.maskKVsSize *
                               (uint64_t)tilingData->promptAttentionBaseParams.maskQsSize;
    }

    // mask offset of core
    attenMaskCoreOffset = (uint64_t)sOuterOffset * (uint64_t)tilingData->promptAttentionBaseParams.maskKVsSize + attenMaskBatchOffset;

    CalPseShiftOffset(sIdx);

    tensorACoreOffset = seqListOffsetSize +
                        sOuterOffset * MultiHeadQ +
                        batchNOffset * tilingData->promptAttentionBaseParams.headSize;

    uint32_t seqInnerOffsetSize =
        tilingData->promptAttentionBaseParams.seqSize == tilingData->promptAttentionBaseParams.seqInnerSize ?
        seqListOffsetSize / headNumRatio : sIdx * tilingData->promptAttentionBaseParams.seqInnerSize * MultiHeadKV;
    tensorBCoreOffset = seqInnerOffsetSize +
                        batchNOffset / headNumRatio * tilingData->promptAttentionBaseParams.headSize;

    valueCoreOffset = tensorBCoreOffset;
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::LoopSOuterOffsetInitWithBNSD(uint32_t seqListOffsetSize,
                                                                                    int sIdx) {
    uint64_t attenMaskBatchOffset = 0;
    if (attenMaskBatch != 1) {
        attenMaskBatchOffset = (uint64_t)sIdx * (uint64_t)tilingData->promptAttentionBaseParams.maskKVsSize *
                               (uint64_t)tilingData->promptAttentionBaseParams.maskQsSize;
    }
    attenMaskCoreOffset = (uint64_t)sOuterOffset * (uint64_t)tilingData->promptAttentionBaseParams.maskKVsSize + attenMaskBatchOffset;

    uint64_t head_stride_q = static_cast<uint64_t>(tilingData->promptAttentionBaseParams.headSize) *
                             tilingData->promptAttentionBaseParams.seqSize;
    uint64_t head_stride_kv = static_cast<uint64_t>(tilingData->promptAttentionBaseParams.headSize) *
                              tilingData->promptAttentionBaseParams.seqInnerSize;
    uint64_t seq_stride = tilingData->promptAttentionBaseParams.headSize;

    CalPseShiftOffset(sIdx);

    tensorACoreOffset =
        seqListOffsetSize + batchNOffset * head_stride_q + static_cast<uint64_t>(sOuterOffset) * seq_stride;

    uint64_t seqInnerOffsetSize =
        tilingData->promptAttentionBaseParams.seqSize == tilingData->promptAttentionBaseParams.seqInnerSize ?
        seqListOffsetSize / headNumRatio : static_cast<uint64_t>(sIdx) * head_stride_kv *
        tilingData->promptAttentionBaseParams.headNumSize / headNumRatio;

    tensorBCoreOffset = seqInnerOffsetSize + batchNOffset / headNumRatio * head_stride_kv;

    valueCoreOffset = tensorBCoreOffset;

    attentionOutOffset = seqListOffsetSize + batchNOffset * head_stride_q +
        static_cast<uint64_t>(sOuterOffset + nextTokensOffset) * seq_stride;
}

template<typename T, typename U, CubeFormat FORMAT, typename O,  OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::Bmm2UpdateAdd(LocalTensor<mmOutputType>& bmm2ResUb) {
    LocalTensor<mmOutputType> bmm2ResPreUb = tempBmm2Ub.Get<mmOutputType>(bmm2ResUbSize);
    Add(bmm2ResPreUb, bmm2ResUb, bmm2ResPreUb, bmm2ResUbSize);
}

template<typename T, typename U, CubeFormat FORMAT, typename O, OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::GetSingleCoreParam(int sIdx) {
    actualSeqLengthPerBatch = isActualLenDimsNull ? tilingData->promptAttentionBaseParams.seqSize :
                              actualSeqLengthsGm.GetValue(sIdx);
    actualSeqLengthKVPerBatch = isActualLenDimsKVNull ? tilingData->promptAttentionBaseParams.seqInnerSize :
                                actualSeqLengthsKVGm.GetValue(sIdx);

    singleProcessSInnerSize = tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize;
    singleProcessSOuterSizeWhole = tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize;
    MultiHeadQ = tilingData->promptAttentionBaseParams.headSize * tilingData->promptAttentionBaseParams.headNumSize;
    MultiHeadKV = MultiHeadQ / headNumRatio;

    actualSeqLengthPerBatch = ((int64_t)actualSeqLengthPerBatch >
                            (int64_t)tilingData->promptAttentionBaseParams.seqInnerSize +
                            (int64_t)tilingData->promptAttentionBaseParams.preTokens) && (attentionMaskType != 4)?
                            tilingData->promptAttentionBaseParams.seqInnerSize + tilingData->promptAttentionBaseParams.preTokens :
                            actualSeqLengthPerBatch;
    singleProcessSOuterSizeTail = (actualSeqLengthPerBatch % singleProcessSOuterSizeWhole != 0) ?
                                   actualSeqLengthPerBatch % singleProcessSOuterSizeWhole : singleProcessSOuterSizeWhole;
    unalignSInner = (actualSeqLengthKVPerBatch % singleProcessSInnerSize != 0) ?
                     actualSeqLengthKVPerBatch % singleProcessSInnerSize : singleProcessSInnerSize;
    maxInnerLoopTimes = (actualSeqLengthKVPerBatch + singleProcessSInnerSize - 1) / singleProcessSInnerSize;
    singleProcessSInnerSizeTail = (unalignSInner + typeByteNum - 1) / typeByteNum * typeByteNum;
    maskInnerTailAlign = (unalignSInner + maskTypeByteNum - 1) / maskTypeByteNum * maskTypeByteNum;
    padSize = maskInnerTailAlign - unalignSInner;

    pseShiftInnerTailAlign = (unalignSInner + pseShiftTypeByteNum - 1) / pseShiftTypeByteNum * pseShiftTypeByteNum;
    pseShiftPadSize = pseShiftInnerTailAlign - unalignSInner;

    InitTensorSize(&tilingData->promptAttentionTensorSizeRect);
    transposeTilingData = tilingData->transposeTilingDataRect;
    softmaxTilingData = tilingData->softmaxTilingDataRect;
    softmaxFlashTilingData = tilingData->softmaxFlashTilingDataRect;
}

template<typename T, typename U, CubeFormat FORMAT, typename O, OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::GetSparseParam(int32_t* preTokens,
                                                                                 int32_t* nextTokens) {
    if (attentionMaskType == 3) {  // SPARSE_MODE_RIGHT_DOWN : 3
        *preTokens = 214748647;
        *nextTokens = actualSeqLengthKVPerBatch - actualSeqLengthPerBatch;
    }
    if (attentionMaskType == 4) {
        *preTokens = (int32_t)tilingData->promptAttentionBaseParams.preTokens - actualSeqLengthKVPerBatch + actualSeqLengthPerBatch;
        *nextTokens = (int32_t)tilingData->promptAttentionBaseParams.nextTokens + actualSeqLengthKVPerBatch - actualSeqLengthPerBatch;
    }
    preTokensPerBatch = *preTokens;
    nextTokensPerBatch = *nextTokens;
}

template<typename T, typename U, CubeFormat FORMAT, typename O, OptimizationMode M>
__aicore__ inline void PromptFlashAttentionBase<T, U, FORMAT, O, M>::ComputeTokenOffset() {
    if (sOuterOffset < nextTokensPerBatch * (-1) &&
        (sOuterOffset + this->singleProcessSOuterSize) > nextTokensPerBatch * (-1)) {
        nextTokensOffset = nextTokensPerBatch * (-1) - sOuterOffset;
    } else {
        nextTokensOffset = 0;
    }
}

#endif  // PROMPT_FLASH_ATTENTION_BASE_H