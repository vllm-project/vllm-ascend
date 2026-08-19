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
 * \file mla_prolog_v3.cpp
 * \brief
 */

#define MLA_PROLOG_VERSION 3
#define GLOBAL_OVERFLOW_MODE_CTRL 60

#if __has_include("arch35/kernel_mla_prolog_split_n.h")
#include "arch35/kernel_mla_prolog_split_n.h"
#include "arch35/kernel_mla_prolog_split_m.h"
#else
#include "arch35/kernel_mla_prolog_split_n.h"
#include "arch35/kernel_mla_prolog_split_m.h"
#endif
using namespace MlaProlog;

template<uint8_t CacheMode, uint8_t Scenario, uint8_t QuantMode,
         bool EnableDequantOpt, bool EnableGroupComputeOpt, uint8_t EmptyTensorMode,
         uint8_t ActualSeqLenMode, uint8_t SplitMMode, uint8_t CvMode, bool EnableRope>
__global__ __aicore__ void mla_prolog_v3(
    __gm__ uint8_t *tokenX,
    __gm__ uint8_t *weightDq,
    __gm__ uint8_t *weightUqQr,
    __gm__ uint8_t *weightUk,
    __gm__ uint8_t *weightDkvKr,
    __gm__ uint8_t *rmsnormGammaCq,
    __gm__ uint8_t *rmsnormGammaCkv,
    __gm__ uint8_t *ropeSin,
    __gm__ uint8_t *ropeCos,
    __gm__ uint8_t *kvCache,
    __gm__ uint8_t *krCache,
    __gm__ uint8_t *cacheIndex,
    __gm__ uint8_t *dequantScaleX,
    __gm__ uint8_t *dequantScaleWDq,
    __gm__ uint8_t *dequantScaleWUqQr,
    __gm__ uint8_t *dequantScaleWDkvKr,
    __gm__ uint8_t *quantScaleCkv,
    __gm__ uint8_t *quantScaleCkr,
    __gm__ uint8_t *smoothScalesCq,
    __gm__ uint8_t *actualSeqLen,
    __gm__ uint8_t *kNopeClipAlpha,
    __gm__ uint8_t *queryOut,
    __gm__ uint8_t *queryRopeOut,
    __gm__ uint8_t *kvCacheOut,
    __gm__ uint8_t *krCacheOut,
    __gm__ uint8_t *dequantScaleQNopeOut,
    __gm__ uint8_t *queryNormOut,
    __gm__ uint8_t *dequantScaleQNormOut,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling) 
{
#if (__NPU_ARCH__ == 3510)
    int64_t globalOriOverflowMode = AscendC::GetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>();
#endif

    REGISTER_TILING_DEFAULT(optiling::MlaPrologTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    constexpr auto emptyMode = static_cast<EMPTY_TENSOR_MODE>(EmptyTensorMode);
    if constexpr (emptyMode == EMPTY_TENSOR_MODE::EMPTY_QUERY) {
        return;
    }
    constexpr auto cacheMode = static_cast<CACHE_MODE>(CacheMode);
    constexpr auto actualSeqLenMode = static_cast<ACTUAL_SEQ_MODE>(ActualSeqLenMode);
    constexpr auto splitMMode = static_cast<SPLIT_M_MODE>(SplitMMode);
    constexpr uint32_t cvRatio = CvMode == ASCENDC_TPL_MIX_AIC_1_1 ? 1 : 2;

    GET_TILING_DATA_WITH_STRUCT(optiling::MlaPrologTilingData, tilingDataIn, tiling);
    const optiling::MlaPrologTilingData *__restrict tilingData = nullptr;
    const optiling::MlaPrologBaseParams *__restrict tilingDataBaseParams = &tilingDataIn.baseParams;

    TPipe pipe;
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>(0);
#endif
    if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::NO_QUANT) {
        MlaPrologVecS1CubS2<MLAPType<bfloat16_t, bfloat16_t, bfloat16_t, float, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, 
            emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && 
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::PARTIAL_QUANT_KV_NO_QUANT) {
        MlaPrologVecS1CubS2<MLAPType<bfloat16_t, int8_t, bfloat16_t, float, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, 
            emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && 
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PER_CHANNEL) {
        MlaPrologVecS1CubS2<MLAPType<bfloat16_t, int8_t, int8_t, float, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, 
            emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && 
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::FULL_QUANT_KV_NO_QUANT) {
        MlaPrologVecS1CubS2<MLAPType<int8_t, int8_t, bfloat16_t, float, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, 
            emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && 
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::FULL_QUANT_KV_QUANT_PER_TENSOR) {
        MlaPrologVecS1CubS2<MLAPType<int8_t, int8_t, int8_t, float, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, 
            emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && 
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::PARTIAL_QUANT_KV_QUANT_PERTILE) {
        MlaPrologVecS1CubS2<MLAPType<bfloat16_t, int8_t, int8_t, float, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, 
            emptyMode, actualSeqLenMode, true, cvRatio, EnableRope>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && 
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::FULL_QUANT_KV_QUANT_PERTILE) {
        MlaPrologVecS1CubS2<MLAPType<int8_t, int8_t, int8_t, float, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, 
            emptyMode, actualSeqLenMode, true, cvRatio, EnableRope>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    }
#if __CCE_AICORE__ == 310
    else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT &&
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::MXFP8_FULL_QUANT_KV_NO_QUANT) {
        if constexpr (splitMMode == SPLIT_M_MODE::ENABLED) {
            MlaPrologV3SplitM<MLAPType<FP8E4M3, FP8E4M3, bfloat16_t, FP8E8M0, cacheMode,
                EnableDequantOpt, EnableGroupComputeOpt,
                emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>> op(&pipe, tilingData, tilingDataBaseParams);
            op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                    ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                    dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                    queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
            op.Process();
        } else {
            MlaPrologVecS1CubS2<MLAPType<FP8E4M3, FP8E4M3, bfloat16_t, FP8E8M0, cacheMode,
                EnableDequantOpt, EnableGroupComputeOpt,
                emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>> op(&pipe, tilingData, tilingDataBaseParams);
            op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                    ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                    dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                    queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
            op.Process();
        }
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && 
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::MXFP8_FULL_QUANT_KV_QUANT_PER_TENSOR) {
        if constexpr (splitMMode == SPLIT_M_MODE::ENABLED) {
            MlaPrologV3SplitM<MLAPType<FP8E4M3, FP8E4M3, FP8E4M3, FP8E8M0, cacheMode,
                EnableDequantOpt, EnableGroupComputeOpt,
                emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>> op(&pipe, tilingData, tilingDataBaseParams);
            op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                    ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                    dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                    queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
            op.Process();
        } else {
            MlaPrologVecS1CubS2<MLAPType<FP8E4M3, FP8E4M3, FP8E4M3, FP8E8M0, cacheMode,
                EnableDequantOpt, EnableGroupComputeOpt,
                emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>> op(&pipe, tilingData, tilingDataBaseParams);
            op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                    ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                    dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                    queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
            op.Process();
        }
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && 
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::MXFP8_FULL_QUANT_KV_QUANT_PER_TILE) {
        if constexpr (splitMMode == SPLIT_M_MODE::ENABLED) {
            MlaPrologV3SplitM<MLAPType<FP8E4M3, FP8E4M3, FP8E4M3, FP8E8M0, cacheMode,
                EnableDequantOpt, EnableGroupComputeOpt,
                emptyMode, actualSeqLenMode, true, cvRatio, EnableRope>> op(&pipe, tilingData, tilingDataBaseParams);
            op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                    ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                    dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                    queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
            op.Process();
        } else {
            MlaPrologVecS1CubS2<MLAPType<FP8E4M3, FP8E4M3, FP8E4M3, FP8E8M0, cacheMode,
                EnableDequantOpt, EnableGroupComputeOpt,
                emptyMode, actualSeqLenMode, true, cvRatio, EnableRope>> op(&pipe, tilingData, tilingDataBaseParams);
            op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                    ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                    dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                    queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
            op.Process();
        }
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && 
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::FP8_FULL_QUANT_KV_NO_QUANT) {
        MlaPrologVecS1CubS2<MLAPType<FP8E4M3, FP8E4M3, bfloat16_t, float, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, 
            emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && 
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::FP8_FULL_QUANT_KV_QUANT_PER_TENSOR) {
        MlaPrologVecS1CubS2<MLAPType<FP8E4M3, FP8E4M3, FP8E4M3, float, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, 
            emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && 
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::HIF8_FULL_QUANT_KV_NO_QUANT) {
        MlaPrologVecS1CubS2<MLAPType<HIF8, HIF8, bfloat16_t, float, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, 
            emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && 
                         static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::HIF8_FULL_QUANT_KV_QUANT_PER_TENSOR) {
        MlaPrologVecS1CubS2<MLAPType<HIF8, HIF8, HIF8, float, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, 
            emptyMode, actualSeqLenMode, false, cvRatio, EnableRope>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && 
                        static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::FP8_FULL_QUANT_KV_QUANT_PER_TILE) {
        MlaPrologVecS1CubS2<MLAPType<FP8E4M3, FP8E4M3, FP8E4M3, float, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, 
            emptyMode, actualSeqLenMode, true, cvRatio, EnableRope>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    } else if constexpr (static_cast<SCENARIO>(Scenario) == SCENARIO::QUANT && 
                        static_cast<QUANT_MODE>(QuantMode) == QUANT_MODE::HIF8_FULL_QUANT_KV_QUANT_PER_TILE) {
        MlaPrologVecS1CubS2<MLAPType<HIF8, HIF8, HIF8, float, cacheMode,
            EnableDequantOpt, EnableGroupComputeOpt, 
            emptyMode, actualSeqLenMode, true, cvRatio, EnableRope>> op(&pipe, tilingData, tilingDataBaseParams);
        op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin,
                ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq, dequantScaleWUqQr,
                dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq, actualSeqLen, kNopeClipAlpha,
                queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut, dequantScaleQNormOut, workspace);
        op.Process();
    }
#endif
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>(globalOriOverflowMode);
#endif
}
