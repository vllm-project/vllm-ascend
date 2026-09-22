// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#define MLA_PROLOG_VERSION 3
#define GLOBAL_OVERFLOW_MODE_CTRL 60
#include "mla_prolog_template_tiling_key.h"
#if __has_include("../../mla_prolog_v3/op_kernel/arch35/kernel_mla_prolog_split_n.h")
#include "../../mla_prolog_v3/op_kernel/arch35/kernel_mla_prolog_split_n.h"
#else
#include "../mla_prolog_v3/arch35/kernel_mla_prolog_split_n.h"
#endif
using namespace MlaProlog;

template<uint8_t CacheMode, uint8_t Scenario, uint8_t QuantMode,
         bool EnableDequantOpt, bool EnableGroupComputeOpt, uint8_t EmptyTensorMode,
         uint8_t ActualSeqLenMode, uint8_t SplitMMode, uint8_t CvMode, bool EnableRope>
__global__ __aicore__ void mla_prolog_dcp_c8(
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
    __gm__ uint8_t *kvDescale,
    __gm__ uint8_t *queryOut,
    __gm__ uint8_t *queryRopeOut,
    __gm__ uint8_t *kvCacheOut,
    __gm__ uint8_t *krCacheOut,
    __gm__ uint8_t *dequantScaleQNopeOut,
    __gm__ uint8_t *queryNormOut,
    __gm__ uint8_t *dequantScaleQNormOut,
    __gm__ uint8_t *queryC8,
    __gm__ uint8_t *queryRopeC8,
    __gm__ uint8_t *queryScaleC8,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(optiling::MlaPrologTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(optiling::MlaPrologTilingData, tilingDataIn, tiling);
    TPipe pipe;
    MlaPrologVecS1CubS2<MLAPType<FP8E4M3, FP8E4M3, bfloat16_t, FP8E8M0,
        static_cast<CACHE_MODE>(CacheMode), EnableDequantOpt, EnableGroupComputeOpt,
        static_cast<EMPTY_TENSOR_MODE>(EmptyTensorMode), static_cast<ACTUAL_SEQ_MODE>(ActualSeqLenMode),
        false, 2, EnableRope, true>> op(&pipe, nullptr, &tilingDataIn.baseParams);
    const auto overflow = AscendC::GetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>();
    AscendC::SetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>(0);
    op.Init(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv,
        ropeSin, ropeCos, cacheIndex, kvCacheOut, krCacheOut, dequantScaleX, dequantScaleWDq,
        dequantScaleWUqQr, dequantScaleWDkvKr, quantScaleCkv, quantScaleCkr, smoothScalesCq,
        actualSeqLen, kNopeClipAlpha, queryOut, queryRopeOut, dequantScaleQNopeOut, queryNormOut,
        dequantScaleQNormOut, workspace);
    op.InitDcpOutputs(queryC8, queryRopeC8, queryScaleC8, kvDescale);
    op.Process();
    AscendC::SetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>(overflow);
}
