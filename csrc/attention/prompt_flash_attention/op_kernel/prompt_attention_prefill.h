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
 * \file prompt_attention_prefill.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_PREFILL_H
#define PROMPT_FLASH_ATTENTION_PREFILL_H
#include "common_func.h"
#include "simd.h"
#include "iterator.h"
#include "mma.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "unpad_flash_attention_common.h"

template <typename PFATypeNZ, PrecType prec_type>
class PromptAttentionPrefill {
public:
    __aicore__ inline PromptAttentionPrefill(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *attenMask, __gm__ uint8_t* _actualSeqLengths, __gm__ uint8_t* _actualSeqLengthsKV, __gm__ uint8_t *attentionOut,
                                __gm__ uint8_t *workspace, const PromptFlashAttentionBaseApiTilingData* __restrict tiling);
    __aicore__ inline void Process();

    using T = typename PFATypeNZ::inputType;

    using Q_T = typename PFATypeNZ::inputType;
    using KV_T = typename PFATypeNZ::inputType;
    using OUT_T = typename PFATypeNZ::outputType;

protected:
    const PromptFlashAttentionBaseApiTilingData *__restrict tilingData = nullptr;

    uint32_t long_seq = 0;
    uint32_t is_triu=0;
    uint32_t alibi_compress_offset=0;
    uint32_t q_tight=0;
    uint32_t is_sqrt=0;
    uint32_t alibi_left_align=0;

    uint32_t embeddingSize = 0;
    half tor = 0;
    uint32_t batchSize = 0;
    uint32_t scaleType = 0;
    uint32_t headSplit = 0;
    uint32_t maskType = 0;
    uint32_t maskStride = 0;
    uint32_t runMode = 0;
    int64_t headMaskStride = 0;
    int64_t batchMaskStride = 0;
    int32_t kv_copy_stride = 0;
    uint32_t qTokens = 0;
    uint32_t kvHead = 0;
    int64_t heads = 0;
    int32_t kv_real_heads=0;
    uint32_t maxSeqlen = 0;
    uint32_t windowLen = 0;
    uint32_t cacheType = 0;
    uint32_t maxKvSeqlen = 0;
    int32_t group_num=0;
    bool batchContinuous = true;

    uint32_t startBlk = 0;
    uint32_t endBlk = 0;
    uint32_t startBatch = 0;
    uint32_t endBatch = 0;
    uint32_t headNum = 0;

    uint32_t addrKHigh32 = 0;
    uint32_t addrKLow32 = 0;
    int64_t addrKScalar = 0;
    uint32_t addrVHigh32 = 0;
    uint32_t addrVLow32 = 0;
    int64_t addrVScalar = 0;
    uint32_t inputLayout = 1;

    __gm__ uint8_t *query;
    __gm__ uint8_t *key;
    __gm__ uint8_t *value;
    __gm__ uint8_t *attenMask;
    __gm__ uint8_t *attentionOut;
    __gm__ uint8_t *actualSeqLengths;
    __gm__ uint8_t *actualSeqLengthsKV;
    AscendC::GlobalTensor<int64_t> actualSeqLengthsGm;
    AscendC::GlobalTensor<int64_t> actualSeqLengthsKVGm;

    template <typename T> __aicore__ inline T Align(T num, T rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
    }
    __aicore__ inline void InitTilingData();
};

template <typename T, PrecType prec_type2>
__aicore__ inline void PromptAttentionPrefill<T, prec_type2>::Init(__gm__ uint8_t *_query, __gm__ uint8_t *_key,
                                __gm__ uint8_t *_value, __gm__ uint8_t *_attenMask, __gm__ uint8_t* _actualSeqLengths, __gm__ uint8_t* _actualSeqLengthsKV, __gm__ uint8_t *_attentionOut, __gm__ uint8_t *workspace,
                                const PromptFlashAttentionBaseApiTilingData* __restrict tiling)
{
    tilingData = tiling;

    InitTilingData();
    this->query = _query;
    this->key = _key;
    this->value = _value;
    this->attenMask = _attenMask;
    this->attentionOut = _attentionOut;
    this->actualSeqLengths = _actualSeqLengths;
    this->actualSeqLengthsKV = _actualSeqLengthsKV;
}


template <typename T, PrecType prec_type2>
__aicore__ inline void PromptAttentionPrefill<T, prec_type2>::Process()
{
    AscendC::SetMaskNorm();
    AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    AscendC::SetLoadDataPaddingValue<uint64_t>(uint16_t(0));
    AscendC::SetAtomicNone();

    uint64_t stride_qo = maxSeqlen * embeddingSize;
    uint64_t stride_kv = maxSeqlen * embeddingSize;
    int32_t group_num = heads / kv_real_heads;
    int32_t kv_copy_stride = qTokens;

    headMaskStride = headMaskStride * maxSeqlen;
    batchMaskStride = batchMaskStride * maxSeqlen;
    if (prec_type2 == PrecType::BMM1_FP16_EXP_FP32) {
        UnpadFlashAttentionCommon<float, half, PrecType::BMM1_FP16_EXP_FP32, PrecType::BMM1_FP16_EXP_FP32> flashAttentionEncoder(query, key, value, attenMask, nullptr, nullptr, nullptr, attentionOut);
        flashAttentionEncoder.Run(
            tilingData,
            nullptr, //alibi_coeff_gm,
            actualSeqLengthsGm, actualSeqLengthsKVGm,
            maskType, windowLen, long_seq,
            stride_qo, stride_kv, headMaskStride, batchMaskStride,
            startBatch, endBatch, startBlk, endBlk,
            is_triu, alibi_compress_offset, group_num,
            maskStride, qTokens, embeddingSize, q_tight, scaleType,
            tor, kv_copy_stride, is_sqrt, heads,
            maxSeqlen, batchSize, kv_real_heads, alibi_left_align, inputLayout);
    }
}

template <typename T, PrecType prec_type2>
__aicore__ inline void PromptAttentionPrefill<T, prec_type2>::InitTilingData()
{
    uint32_t tmpBlockIdx = AscendC::GetBlockIdx();

    batchSize = tilingData->promptAttentionBaseApiBaseParams.batchSize;
    embeddingSize = tilingData->promptAttentionBaseApiBaseParams.headSize;
    tor = tilingData->promptAttentionBaseApiBaseParams.tor;

    startBlk = tilingData->promptAttentionSplitCoreParams.startBlkArray[tmpBlockIdx];
    endBlk = tilingData->promptAttentionSplitCoreParams.endBlkArray[tmpBlockIdx];
    headNum = tilingData->promptAttentionBaseApiBaseParams.headNumSize;
    startBatch = static_cast<uint32_t>(startBlk / headNum);
    endBatch = static_cast<uint32_t>((endBlk- 1) / headNum);

    kvHead = tilingData->promptAttentionBaseApiBaseParams.kvHeadNumSize;
    heads = tilingData->promptAttentionBaseApiBaseParams.headNumSize;
    kv_real_heads = kvHead > 0 ? kvHead : heads;
    maxSeqlen = tilingData->promptAttentionBaseApiBaseParams.maxSeqLen;
    windowLen = 0;
    cacheType = 0;
    maxKvSeqlen = tilingData->promptAttentionBaseApiBaseParams.maxKvSeqLen;
    qTokens = maxSeqlen;

    long_seq = tilingData->promptAttentionBaseApiBaseParams.isLongSeq;
    is_triu=tilingData->promptAttentionBaseApiBaseParams.isTriuMask;
    alibi_compress_offset=tilingData->promptAttentionBaseApiBaseParams.alibiCompressOffset;

    alibi_left_align=tilingData->promptAttentionBaseApiBaseParams.alibiLeftAlign;

    maskType = tilingData->promptAttentionBaseApiBaseParams.maskType;
    maskStride = maxSeqlen;
    headMaskStride = 0;
    batchMaskStride = maxSeqlen;
    scaleType = 0;
    inputLayout = tilingData->promptAttentionBaseApiBaseParams.inputLayoutType;

    actualSeqLengthsGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengths, batchSize);
    actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLengthsKV, batchSize);
}
#endif