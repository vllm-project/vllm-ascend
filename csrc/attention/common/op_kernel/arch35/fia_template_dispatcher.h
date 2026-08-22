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
 * \file fia_template_dispatcher.h
 * \brief
*/

#ifndef FIA_TEMPLATE_DISPATCHER_H
#define FIA_TEMPLATE_DISPATCHER_H

#include "fia_kernel_noquant_gqa.h"
#include "fia_kernel_fullquant_mx.h"
#include "fia_kernel_fullquant_gqa.h"

template <typename INPUT_T, typename OUT_T, uint8_t inOutLayoutType, uint16_t config, uint8_t pseMode,
          uint8_t quantMode, bool hasAttenMask, bool hasRope, uint8_t KvLayoutType, bool isFd, bool emptyTensor,
          bool enableKVPrefix, bool enableS1OutSplit>
inline __aicore__ void run_fia_noquant_gqa_kernel(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKV,
    __gm__ uint8_t *postQuantScale, __gm__ uint8_t *postQuantOffset, __gm__ uint8_t *blocktable,
    __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *keySharedPrefix,
    __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen, __gm__ uint8_t *queryRope,
    __gm__ uint8_t *keyRope, __gm__ uint8_t *learnableSink, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    PARSE_PARAMS_NoQuant(inOutLayoutType, config, pseMode, ...);

    fa_base_matmul::idCounterNum = 0;

    // 默认先不按DN处理
    constexpr bool useDn = false;
    constexpr bool bmm2Write2Ub = true;
    constexpr bool splitD = false;

    // CubBlockType
    using CubBlockNormal = BaseApi::FANoQuantGqaBlockCube<INPUT_T, float, inputLayoutType,
        s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, hasRope, KvLayoutType,
        enableKVPrefix, useDn, bmm2Write2Ub, splitD>;
    using CubBlockDummy = BaseApi::FANoQuantGqaBlockCubeDummy<INPUT_T, float, inputLayoutType,
        s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, hasRope, KvLayoutType,
        enableKVPrefix, useDn, bmm2Write2Ub, splitD>;
    using CubBlock = typename std::conditional<g_coreType == AscendC::AIC, CubBlockNormal, CubBlockDummy>::type;

    // VecFaBlockType
    using VecFaBlockNormal = BaseApi::FANoQuantGqaBlockVec<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
        s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode),
        hasAttenMask, false, hasRope, KvLayoutType, isFd, enableKVPrefix, useDn, bmm2Write2Ub, splitD>;
    using VecFaBlockDummy = BaseApi::FANoQuantGqaBlockVecDummy<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
        s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask,
        false, hasRope, KvLayoutType, isFd, enableKVPrefix, useDn, bmm2Write2Ub, splitD>;
    using VecFaBlock = typename std::conditional<g_coreType == AscendC::AIV, VecFaBlockNormal, VecFaBlockDummy>::type;

    // VecFdBlockType
    using VecFdBlockNormal = BaseApi::FiaBlockVecFlashDecode<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
        s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask,
        false, hasRope, KvLayoutType, enableKVPrefix, useDn, bmm2Write2Ub, splitD>;
    using VecFdBlockDummy = BaseApi::FiaBlockVecFlashDecodeDummy<INPUT_T, float, OUT_T,
        inputLayoutType, outputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
        static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, KvLayoutType, enableKVPrefix,
        useDn, bmm2Write2Ub, splitD>;
    using VecFdBlock = typename std::conditional<g_coreType == AscendC::AIV, VecFdBlockNormal, VecFdBlockDummy>::type;

    // KernelType
    using Kernel = FlashAttentionNoQuantGqaKernel<CubBlock, VecFaBlock, VecFdBlock>;

    GET_TILING_DATA_MEMBER(FusedInferAttentionScoreTilingData, baseTiling, baseTilingIn, tiling);
    const NoQuantTilingArch35 *__restrict tilingData = &baseTilingIn;
    __gm__ uint8_t *fiaMetaData = tiling + offsetof(FusedInferAttentionScoreTilingData, fiaMetaData);
    TPipe tPipe;
    Kernel op;
    op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blocktable, queryPaddingSize,
            kvPaddingSize, postQuantScale, postQuantOffset, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen,
            queryRope, keyRope, learnableSink, softmaxLse, attentionOut, workspace, fiaMetaData, tilingData, &tPipe);
    op.Process();
}

template <typename INPUT_T, typename OUT_T, uint8_t inOutLayoutType, uint16_t config, uint8_t pseMode,
          uint8_t quantMode, bool hasAttenMask, bool hasRope, uint8_t KvLayoutType, bool isFd, bool emptyTensor,
          bool enableKVPrefix, bool enableS1OutSplit>
inline __aicore__ void run_fia_fullquant_mx_kernel(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKV,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
    __gm__ uint8_t *dequantScaleValue, __gm__ uint8_t *pScale, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    PARSE_PARAMS_FullQuant(inOutLayoutType, config, pseMode, ...);
    fa_base_matmul::idCounterNum = 0;

    constexpr TPosition bmm2OutPos =
        GetC2Position(dVTemplateType,
                      UbOutCondition<INPUT_T>(false, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope,
                                              (uint32_t)s1TemplateType == 64),
                      ((uint32_t)s2TemplateType == 256 && (uint32_t)s1TemplateType == 64), false);
    constexpr bool useDn = (quantMode == FULLQUANT_MODE_QKV_MXFP8_PREFILL);
    constexpr bool bmm2Write2Ub = bmm2OutPos == TPosition::VECCALC;
    constexpr bool splitD = (uint16_t)dVTemplateType > (uint16_t)DTemplateType::Aligned256;

    // CubBlockType
    using CubBlockNormal = BaseApi::FAFullQuantMxBlockCube<INPUT_T, float, inputLayoutType,
        s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, hasRope, KvLayoutType,
        enableKVPrefix, useDn, bmm2Write2Ub, splitD>;
    using CubBlockDummy = BaseApi::FAFullQuantMxBlockCubeDummy<INPUT_T, float, inputLayoutType,
        s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, hasRope, KvLayoutType,
        enableKVPrefix, useDn, bmm2Write2Ub, splitD>;
    using CubBlock = typename std::conditional<g_coreType == AscendC::AIC, CubBlockNormal, CubBlockDummy>::type;

    // VecFaBlockType
    using VecFaBlockNormal = BaseApi::FAFullQuantMxBlockVec<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
        s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode),
        hasAttenMask, false, hasRope, KvLayoutType, isFd, enableKVPrefix, useDn, bmm2Write2Ub, splitD>;
    using VecFaBlockDummy = BaseApi::FAFullQuantMxBlockVecDummy<
        INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
        s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask,
        false, hasRope, KvLayoutType, isFd, enableKVPrefix, useDn, bmm2Write2Ub, splitD>;
    using VecFaBlock = typename std::conditional<g_coreType == AscendC::AIV, VecFaBlockNormal, VecFaBlockDummy>::type;

    // VecFdBlockType
    using VecFdBlockNormal = BaseApi::FiaBlockVecFlashDecodeFullQuant<
        INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
        s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask,
        false, hasRope, KvLayoutType, enableKVPrefix, useDn, bmm2Write2Ub, splitD>;
    using VecFdBlockDummy = BaseApi::FiaBlockVecFlashDecodeFullQuantDummy<INPUT_T, float, OUT_T,
        inputLayoutType, outputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
        static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, KvLayoutType, enableKVPrefix,
        useDn, bmm2Write2Ub, splitD>;
    using VecFdBlock = typename std::conditional<g_coreType == AscendC::AIV, VecFdBlockNormal, VecFdBlockDummy>::type;

    // KernelType
    using Kernel = FlashAttentionFullQuantMxKernel<CubBlock, VecFaBlock, VecFdBlock>;

    GET_TILING_DATA_MEMBER(FusedInferAttentionScoreFullQuantTilingData, baseTiling, baseTilingIn, tiling);
    const FullQuantTiling *__restrict tilingData = &baseTilingIn;
    __gm__ uint8_t *fiaMetaData = tiling + offsetof(FusedInferAttentionScoreFullQuantTilingData, fiaMetaData);
    TPipe tPipe;
    Kernel op;
    op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blockTable,
        dequantScaleQuery, dequantScaleKey, dequantScaleValue, pScale, queryRope, keyRope,
        softmaxLse, attentionOut, workspace, fiaMetaData, tilingData, &tPipe);
    op.Process();
}

template <typename INPUT_T, typename OUT_T, uint8_t inOutLayoutType, uint16_t config, uint8_t pseMode,
          uint8_t quantMode, bool hasAttenMask, bool hasRope, uint8_t KvLayoutType, bool isFd, bool emptyTensor,
          bool enableKVPrefix, bool enableS1OutSplit>
inline __aicore__ void run_fia_fullquant_gqa_kernel(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *pseShift,
    __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *actualSeqLengthsKV,
    __gm__ uint8_t *blockTable, __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *dequantScaleKey,
    __gm__ uint8_t *dequantScaleValue, __gm__ uint8_t *pScale, __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse,
    __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    PARSE_PARAMS_FullQuant(inOutLayoutType, config, pseMode, ...);
    fa_base_matmul::idCounterNum = 0;

    constexpr TPosition bmm2OutPos =
        GetC2Position(dVTemplateType,
                      UbOutCondition<INPUT_T>(false, static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope,
                                              (uint32_t)s1TemplateType == 64),
                      ((uint32_t)s2TemplateType == 256 && (uint32_t)s1TemplateType == 64), false);
    constexpr bool useDn = (quantMode == FULLQUANT_MODE_QK_PER_TOKEN_HEAD_V_PER_HEAD);
    constexpr bool bmm2Write2Ub = bmm2OutPos == TPosition::VECCALC;
    constexpr bool splitD = (uint16_t)dVTemplateType > (uint16_t)DTemplateType::Aligned256;

    // CubBlockType
    using CubBlockNormal = BaseApi::FAFullQuantGqaBlockCube<INPUT_T, float, inputLayoutType,
        s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, hasRope, KvLayoutType,
        enableKVPrefix, useDn, bmm2Write2Ub, splitD>;
    using CubBlockDummy = BaseApi::FAFullQuantGqaBlockCubeDummy<INPUT_T, float, inputLayoutType,
        s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, hasRope, KvLayoutType,
        enableKVPrefix, useDn, bmm2Write2Ub, splitD>;
    using CubBlock = typename std::conditional<g_coreType == AscendC::AIC, CubBlockNormal, CubBlockDummy>::type;

    // VecFaBlockType
    using VecFaBlockNormal = BaseApi::FAFullQuantGqaBlockVec<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
        s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode),
        hasAttenMask, false, hasRope, KvLayoutType, isFd, enableKVPrefix, useDn, bmm2Write2Ub, splitD>;
    using VecFaBlockDummy = BaseApi::FAFullQuantGqaBlockVecDummy<
        INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
        s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask,
        false, hasRope, KvLayoutType, isFd, enableKVPrefix, useDn, bmm2Write2Ub, splitD>;
    using VecFaBlock = typename std::conditional<g_coreType == AscendC::AIV, VecFaBlockNormal, VecFaBlockDummy>::type;

    // VecFdBlockType
    using VecFdBlockNormal = BaseApi::FiaBlockVecFlashDecodeFullQuant<
        INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
        s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, static_cast<PseTypeEnum>(pseMode), hasAttenMask,
        false, hasRope, KvLayoutType, enableKVPrefix, useDn, bmm2Write2Ub, splitD>;
    using VecFdBlockDummy = BaseApi::FiaBlockVecFlashDecodeFullQuantDummy<INPUT_T, float, OUT_T,
        inputLayoutType, outputLayoutType, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
        static_cast<PseTypeEnum>(pseMode), hasAttenMask, false, hasRope, KvLayoutType, enableKVPrefix,
        useDn, bmm2Write2Ub, splitD>;
    using VecFdBlock = typename std::conditional<g_coreType == AscendC::AIV, VecFdBlockNormal, VecFdBlockDummy>::type;

    // KernelType
    using Kernel = FlashAttentionFullQuantGqaKernel<CubBlock, VecFaBlock, VecFdBlock>;

    GET_TILING_DATA_MEMBER(FusedInferAttentionScoreFullQuantTilingData, baseTiling, baseTilingIn, tiling);
    const FullQuantTiling *__restrict tilingData = &baseTilingIn;
    __gm__ uint8_t *fiaMetaData = tiling + offsetof(FusedInferAttentionScoreFullQuantTilingData, fiaMetaData);
    TPipe tPipe;
    Kernel op;
    op.Init(query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKV, blockTable,
        dequantScaleQuery, dequantScaleKey, dequantScaleValue, pScale, queryRope, keyRope,
        softmaxLse, attentionOut, workspace, fiaMetaData, tilingData, &tPipe);
    op.Process();
}
#endif