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
 * \file fused_infer_attention_score_tiling_impl.cpp
 * \brief
 */

#include "fused_infer_attention_score_tiling_impl.h"
#include "../fused_infer_attention_score_tiling_info_parser.h"

#include "log/log.h"
#include "err/ops_err.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "../fused_infer_attention_score_tiling_utils.h"
#include "../fused_infer_attention_score_tiling_constants.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
using namespace arch35FIA;
ge::graphStatus FusedInferAttentionScoreTilingImpl::SetPlatMemoryInfo(const gert::TilingContext *context,
                                                                      const FiaTilingInfo &fiaInfo)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(fiaInfo.opName, "The platformInfoPtr is null!"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    platformInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    platformInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
    platformInfo_.coreNum = platformInfo_.aivNum;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformInfo_.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo_.l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo_.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo_.l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, platformInfo_.l2Size);

    platformInfo_.defaultSysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGI(fiaInfo.opName, "AIV:%u AIC:%u L0A:%lu L0B:%lu L0C:%lu UB:%lu L1:%lu L2:%lu", platformInfo_.aivNum,
            platformInfo_.aicNum, platformInfo_.l0aSize, platformInfo_.l0bSize, platformInfo_.l0cSize,
            platformInfo_.ubSize, platformInfo_.l1Size, platformInfo_.l2Size);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SetEmptyTensor(gert::TilingContext *context,
                                                                   const FiaTilingInfo &fiaInfo)
{
    auto &initOutputParams = faRunTilingAdapter_.initOutputParams;
    int64_t outSize = fiaInfo.opParamInfo.attenOut.shape->GetStorageShape().GetShapeSize();
    int64_t lseSize = fiaInfo.softmaxLseFlag ? fiaInfo.opParamInfo.lseOut.shape->GetStorageShape().GetShapeSize() : 0;
    uint32_t singleCoreSize = (outSize + platformInfo_.aivNum - 1) / (platformInfo_.aivNum);
    if (fiaInfo.isOutQuantEnable) {
        singleCoreSize = AlignUp(singleCoreSize, uint32_t(2));
    }
    initOutputParams.set_singleCoreSize(singleCoreSize);
    initOutputParams.set_totalOutputSize(outSize);
    initOutputParams.set_totalSoftMaxLseOutputSize(lseSize);
    initOutputParams.set_needInit(1);

    OP_CHECK_IF(GenTilingKey(context, fiaInfo) != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo.opName, "GenTilingKey fail."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetBlockDim(context, fiaInfo) != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo.opName, "SetBlockDim fail."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspace(context, fiaInfo) != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo.opName, "GetWorkspace fail."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetTilingData(context, fiaInfo) != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo.opName, "SetTilingData fail."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool FusedInferAttentionScoreTilingImpl::CheckTransposeLayout(const FiaTilingInfo &fiaInfo) const
{
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    if(layoutStr == "BNSD_BSND" || layoutStr == "BSND_BNSD" || layoutStr == "BSH_BNSD" ||
        layoutStr == "NTD" || layoutStr == "NTD_TND" || layoutStr == "TND_NTD"){
        return true;
    }
    return false;
}

void FusedInferAttentionScoreTilingImpl::SetIsIFA(const FiaTilingInfo &fiaInfo)
{
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    bool isTransposeLayout = layoutStr == "BNSD_BSND" || layoutStr == "BSND_BNSD" || layoutStr == "BSH_BNSD" ||
            layoutStr == "NTD" || layoutStr == "NTD_TND";
    if (fiaInfo.s1Size == 1 && !fiaInfo.enableAlibiPse && !isTransposeLayout &&
        fiaInfo.fullQuantMode != FiaFullQuantMode::QKV_PER_BLOCK_FULL_QUANT) {
        isIFAFlag_ =true;
        return;
    }
}

void FusedInferAttentionScoreTilingImpl::SetGSMerge(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) {
        gsMergeFlag_ = true;
        return;
    }
    if (!fiaInfo.antiQuantFlag && fiaInfo.qkHeadDim > 256U && (fiaInfo.qkHeadDim % DSIZE_64) != 0) {
        gsMergeFlag_ = false;
        return;
    }

    pfaMergeFlag_ = CheckPFAMergeSupport(fiaInfo);

    bool actualSeqLenUnequal = !CheckActualSeqLenAligned(fiaInfo);
    if (actualSeqLenUnequal) {
        pfaMergeFlag_ = false;
    }

    if (fiaInfo.antiQuantFlag) {
        gsMergeFlag_ = CheckAntiQuantGSMerge(fiaInfo, actualSeqLenUnequal);
    } else {
        gsMergeFlag_ = pfaMergeFlag_;
    }
}

bool FusedInferAttentionScoreTilingImpl::CheckPFAMergeSupport(const FiaTilingInfo &fiaInfo) const
{
    if (fiaInfo.s1Size * fiaInfo.gSize >= NUM_64) {
        return false;
    }
    bool isTransposeLayout = CheckTransposeLayout(fiaInfo);
    return !(fiaInfo.pseShiftFlag || fiaInfo.enableAlibiPse ||
             fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D128 || fiaInfo.isOutQuantEnable ||
             fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag ||
             fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST ||
             fiaInfo.quantMode == FiaQuantMode::FULL_QUANT || isTransposeLayout);
}

bool FusedInferAttentionScoreTilingImpl::CheckActualSeqLenAligned(const FiaTilingInfo &fiaInfo) const
{
    if (!actualSeqLenQFlag_ || fiaInfo.antiQuantFlag) {
        return true;
    }

    const gert::Tensor *actSeqLenQ = fiaInfo.opParamInfo.actualSeqLengthsQ.tensor;
    if (actSeqLenQ == nullptr) {
        return true;
    }

    uint32_t actSeqLenQDims = actSeqLenQ->GetShapeSize();
    uint32_t actSeqLengthSize = std::min(actSeqLenQDims, fiaInfo.bSize);
    bool isCumulativeLayout = (fiaInfo.qLayout == FiaLayout::TND || fiaInfo.qLayout == FiaLayout::NTD);

    for (uint32_t i = 0; i < actSeqLengthSize; ++i) {
        int64_t actSeqTmp = actSeqLenQ->GetData<int64_t>()[i];
        if (isCumulativeLayout && i >= 1) {
            actSeqTmp -= actSeqLenQ->GetData<int64_t>()[i - 1];
        }
        if (actSeqTmp < fiaInfo.s1Size) {
            return false;
        }
    }
    return true;
}

bool FusedInferAttentionScoreTilingImpl::CheckAntiQuantGSMerge(const FiaTilingInfo &fiaInfo,
                                                               bool actualSeqLenUnequal) const
{
    if (fiaInfo.s1Size == 1 && !fiaInfo.enableAlibiPse) {
        return true;
    }

    int64_t totalSize = static_cast<int64_t>(fiaInfo.gSize) * fiaInfo.s1Size;
    if (totalSize <= 0 || totalSize > NUM_32) {
        return false;
    }

    bool isBnsdBsndLayout = (std::string(fiaInfo.opParamInfo.layOut) == "BNSD_BSND");
    return !(actualSeqLenUnequal || fiaInfo.qPaddingSizeFlag || fiaInfo.attenMaskFlag ||
             fiaInfo.pseShiftFlag || fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION ||
             fiaInfo.enableAlibiPse || fiaInfo.isOutQuantEnable || fiaInfo.softmaxLseFlag ||
             isBnsdBsndLayout);
}

static inline bool IsTensorValid(const gert::Tensor *tensor, uint32_t dims)
{
    return dims > 0 && tensor != nullptr && tensor->GetData<int64_t>() != nullptr;
}

static inline int64_t ParseSeqLength(const gert::Tensor *tensor, uint32_t dims, uint32_t bIdx, FiaLayout layout)
{
    if (dims == 1) {
        return tensor->GetData<int64_t>()[0];
    }

    bool isCumulative = (layout == FiaLayout::TND || layout == FiaLayout::NTD);
    if (!isCumulative) {
        return tensor->GetData<int64_t>()[bIdx];
    }

    int64_t currentVal = tensor->GetData<int64_t>()[bIdx];
    return (bIdx == 0) ? currentVal : (currentVal - tensor->GetData<int64_t>()[bIdx - 1]);
}

void FusedInferAttentionScoreTilingImpl::InitImplParam(const FiaTilingInfo &fiaInfo)
{
    const auto &params = fiaInfo.opParamInfo;
    uint32_t qDims = params.actualSeqLengthsQ.tensor ? params.actualSeqLengthsQ.tensor->GetShapeSize() : 0;
    uint32_t kvDims = params.actualSeqLengths.tensor ? params.actualSeqLengths.tensor->GetShapeSize() : 0;
    uint32_t prefixDims = params.actualSharedPrefixLen.tensor ? params.actualSharedPrefixLen.tensor->GetShapeSize() : 0;

    InitTilingFlags(fiaInfo, params.actualSeqLengthsQ.tensor, qDims,
                    params.actualSeqLengths.tensor, kvDims,
                    params.actualSharedPrefixLen.tensor, prefixDims);
    ComputeLoopParams(fiaInfo);
    ParseActualSeqLengths(fiaInfo);
}

void FusedInferAttentionScoreTilingImpl::InitTilingFlags(const FiaTilingInfo &fiaInfo,
                                                         const gert::Tensor *actSeqLenQ, uint32_t qDims,
                                                         const gert::Tensor *actSeqLenKV, uint32_t kvDims,
                                                         const gert::Tensor *actSharedPrefixLen, uint32_t prefixDims)
{
    if (fiaInfo.isMaxWorkspace) {
        actualSeqLenQFlag_ = false;
        actualSeqLenKVFlag_ = false;
        actualSharedPrefixLenFlag_ = false;
        return;
    }

    actualSeqLenQFlag_ = IsTensorValid(actSeqLenQ, qDims);
    actualSeqLenKVFlag_ = IsTensorValid(actSeqLenKV, kvDims);

    if (fiaInfo.antiQuantFlag && fiaInfo.qLayout == FiaLayout::TND) {
        actualSeqLenQFlag_ = true;
    }
    if (fiaInfo.antiQuantFlag && fiaInfo.s1Size == 1 && fiaInfo.qLayout != FiaLayout::TND) {
        actualSeqLenQFlag_ = false;
    }

    if (!fiaInfo.sysPrefixFlag) {
        actualSharedPrefixLenFlag_ = false;
    } else {
        bool isValid = IsTensorValid(actSharedPrefixLen, prefixDims);
        actualSharedPrefixLenFlag_ = fiaInfo.antiQuantFlag ? isValid : !fiaInfo.isMaxWorkspace && isValid;
    }
}

void FusedInferAttentionScoreTilingImpl::ComputeLoopParams(const FiaTilingInfo &fiaInfo)
{
    SetGSMerge(fiaInfo);
    SetIsIFA(fiaInfo);
    if (!fiaInfo.antiQuantFlag) {
        gsMergeFlag_ = gsMergeFlag_ || isIFAFlag_;
    }

    nLoopTimes_ = gsMergeFlag_ ? fiaInfo.n2Size : fiaInfo.n1Size;
    gsSize_ = gsMergeFlag_ ? fiaInfo.gSize * fiaInfo.s1Size : fiaInfo.s1Size;
    headDimAlign_ = AlignUp(fiaInfo.qkHeadDim, NUM_32);

    if (fiaInfo.quantMode == FiaQuantMode::ANTI_QUANT && fiaInfo.s1Size > 1) {
        isPFAFlag_ = true;
    }

    OP_LOGI(fiaInfo.opName, "gsMergeFlag:%u nLoopTimes:%u gsSize:%u ", gsMergeFlag_, nLoopTimes_, gsSize_);
}

void FusedInferAttentionScoreTilingImpl::ParseActualSeqLengths(const FiaTilingInfo &fiaInfo)
{
    const gert::Tensor *actSeqLenQ = fiaInfo.opParamInfo.actualSeqLengthsQ.tensor;
    const gert::Tensor *actSeqLenKV = fiaInfo.opParamInfo.actualSeqLengths.tensor;
    uint32_t qDims = actSeqLenQ ? actSeqLenQ->GetShapeSize() : 0;
    uint32_t kvDims = actSeqLenKV ? actSeqLenKV->GetShapeSize() : 0;

    actualSeqLengthsQ_.reserve(fiaInfo.bSize);
    actualSeqLengthsKV_.reserve(fiaInfo.bSize);

    for (uint32_t bIdx = 0; bIdx < fiaInfo.bSize; bIdx++) {
        if (!actualSeqLenQFlag_) {
            actualSeqLengthsQ_.push_back(gsSize_);
        } else if (gsMergeFlag_ && fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512 && fiaInfo.qLayout == FiaLayout::BNSD) {
            actualSeqLengthsQ_.push_back(fiaInfo.s1Size * fiaInfo.gSize);
        } else {
            int64_t qLen = ParseSeqLength(actSeqLenQ, qDims, bIdx, fiaInfo.qLayout);
            actualSeqLengthsQ_.push_back(gsMergeFlag_ ? qLen * fiaInfo.gSize : qLen);
        }

        int64_t kvLen;
        if (actualSeqLenKVFlag_) {
            kvLen = ParseSeqLength(actSeqLenKV, kvDims, bIdx, fiaInfo.kvLayout);
        } else if (fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST) {
            kvLen = fiaInfo.kvListSeqLens[bIdx];
        } else {
            kvLen = fiaInfo.s2Size;
        }
        actualSeqLengthsKV_.push_back(kvLen);
    }
}

static inline bool IsSupportedQLayout(FiaLayout qLayout)
{
    return qLayout == FiaLayout::BSH || qLayout == FiaLayout::BSND || qLayout == FiaLayout::TND;
}

bool FusedInferAttentionScoreTilingImpl::CheckPfaMergeLayout(const FiaTilingInfo &fiaInfo) const
{
    return IsSupportedQLayout(fiaInfo.qLayout) && pfaMergeFlag_;
}

bool FusedInferAttentionScoreTilingImpl::CheckSmallHeadOptimization(const FiaTilingInfo &fiaInfo) const
{
    if (fiaInfo.quantMode != FiaQuantMode::NO_QUANT) {
        return false;
    }
    if (fiaInfo.s1Size > SOUTER_64 || fiaInfo.s2Size <= SINNER_128) {
        return false;
    }
    if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D128) {
        return false;
    }
    if (fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_PER_BLOCK_FULL_QUANT) {
        return false;
    }

    int32_t preTokens = fiaInfo.preToken;
    int32_t nextTokens = fiaInfo.nextToken;
    if (fiaInfo.sparseMode == 0) {
        preTokens = (preTokens > 0) ? 0 : preTokens;
    } else if (fiaInfo.sparseMode == 4) {
        nextTokens = (nextTokens > 0) ? 0 : nextTokens;
    }
    if (fiaInfo.sparseMode == 2 || preTokens + nextTokens <= 128) {
        return false;
    }

    return true;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::AdjustSinnerAndSouter(gert::TilingContext *context,
                                                                          const FiaTilingInfo &fiaInfo)
{
    sOuterFactor_ = SOUTER_64;
    sInnerFactor_ = SINNER_128;
    uint32_t softmaxSOuterFactor = SOUTER_64;

    if (fiaInfo.vHeadDim <= DSIZE_128 && fiaInfo.mlaMode != MlaMode::ROPE_COMBINE_D128) {
        if (CheckSmallHeadOptimization(fiaInfo)) {
            sOuterFactor_ = SOUTER_32;
            sInnerFactor_ = SINNER_256;
            softmaxSOuterFactor = SOUTER_32;
        } else if (CheckPfaMergeLayout(fiaInfo)) {
            sOuterFactor_ = SOUTER_32;
            sInnerFactor_ = SINNER_256;
        }
    } else if (fiaInfo.vHeadDim > DSIZE_128 && fiaInfo.mlaMode != MlaMode::ROPE_SPLIT_D512 &&
               fiaInfo.s1Size != 1) {
        softmaxSOuterFactor = SOUTER_32;
        if (CheckPfaMergeLayout(fiaInfo) && fiaInfo.vHeadDim <= DSIZE_256) {
            sOuterFactor_ = SOUTER_32;
            sInnerFactor_ = SINNER_256;
        }
    } else if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512 ||
               (fiaInfo.s1Size == 1 && fiaInfo.vHeadDim > DSIZE_128)) {
        sOuterFactor_ = (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) ? SOUTER_32 : SOUTER_64;
    }

    if (fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_MXFP8_FULL_QUANT) {
        sOuterFactor_ = SOUTER_64;
        sInnerFactor_ = SINNER_512;
    }

    OP_LOGI(fiaInfo.opName, "Souter:%u SInner:%u softmaxSOuterFactor %u", sOuterFactor_, sInnerFactor_,
            softmaxSOuterFactor);

    return ge::GRAPH_SUCCESS;
}
void FusedInferAttentionScoreTilingImpl::GetPreNextTokensLeftUp(const FiaTilingInfo &fiaInfo, int64_t actualSeqLength,
                                                                int64_t actualSeqLengthKV, int64_t &preTokensLeftUp,
                                                                int64_t &nextTokensLeftUp) const
{
    bool isRopeSplit512 = (fiaInfo.ropeMode == RopeMode::ROPE_SPLIT &&
                           fiaInfo.vHeadDim == DSIZE_512);
    bool isOptLayout = IsSupportedQLayout(fiaInfo.qLayout);
    if (isRopeSplit512 && !isOptLayout) {
        preTokensLeftUp = SPARSE_MODE_INT_MAX;
        nextTokensLeftUp = SPARSE_MODE_INT_MAX;
        return;
    }

    if (fiaInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) {
        preTokensLeftUp = SPARSE_MODE_INT_MAX;
        if (isRopeSplit512) {
            nextTokensLeftUp = actualSeqLengthKV * fiaInfo.gSize - actualSeqLength;
        } else {
            nextTokensLeftUp = actualSeqLengthKV - actualSeqLength;
        }
    } else if (fiaInfo.sparseMode == SPARSE_MODE_BAND) {
        if (isRopeSplit512) {
            preTokensLeftUp = fiaInfo.preToken * fiaInfo.gSize - actualSeqLengthKV * fiaInfo.gSize + actualSeqLength;
            nextTokensLeftUp = fiaInfo.nextToken * fiaInfo.gSize + actualSeqLengthKV * fiaInfo.gSize - actualSeqLength;
        } else {
            preTokensLeftUp = fiaInfo.preToken - actualSeqLengthKV + actualSeqLength;
            nextTokensLeftUp = fiaInfo.nextToken + actualSeqLengthKV - actualSeqLength;
        }
    } else {
        if (isRopeSplit512) {
            preTokensLeftUp = fiaInfo.preToken * fiaInfo.gSize;
            nextTokensLeftUp = fiaInfo.nextToken * fiaInfo.gSize;
        } else {
            preTokensLeftUp = fiaInfo.preToken;
            nextTokensLeftUp = fiaInfo.nextToken;
        }
    }
}

void FusedInferAttentionScoreTilingImpl::FixParamWithRowInvalid(const FiaTilingInfo &fiaInfo, int64_t &actualSeqLength,
                                                                int64_t actualSeqLengthKV, int64_t &preTokensLeftUp,
                                                                int64_t &nextTokensLeftUp)
{
    // 若出现行无效，需要重新计算nexttokens，pretokens，actualseqlen，以便正确计算分核核数
    int64_t nextTokensError = (nextTokensLeftUp < 0) ? -nextTokensLeftUp : 0;
    nextTokensError = nextTokensError > actualSeqLength ? actualSeqLength : nextTokensError;
    int64_t preTokensError = 0;
    if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) {
        preTokensError = (actualSeqLength > actualSeqLengthKV * fiaInfo.gSize + preTokensLeftUp) ?
                (actualSeqLength - actualSeqLengthKV * fiaInfo.gSize- preTokensLeftUp) : 0;
    } else {
        preTokensError = (actualSeqLength > actualSeqLengthKV + preTokensLeftUp) ?
                (actualSeqLength - actualSeqLengthKV - preTokensLeftUp) : 0;
    }
    preTokensError = preTokensError > actualSeqLength ? actualSeqLength : preTokensError;

    if (nextTokensError > 0 || preTokensError > 0) {
        isRowInvalid_ = true;
    }

    // 若出现上方行无效，需要重新计算nexttokens，pretokens，actualseqlen
    nextTokensLeftUp += nextTokensError;
    preTokensLeftUp -= nextTokensError;
    actualSeqLength -= nextTokensError;

    // 若出现下方行无效，需要重新计算actualseqlen
    actualSeqLength -= preTokensError;
}

int64_t FusedInferAttentionScoreTilingImpl::SumOfArithmeticSeries(int64_t an, int64_t d) const
{
    // 等差数列求和，an：等差数列第n项，d：等差数列公差
    if (d == 0) {
        return 0;
    }
    return (an > 0) ? (an % d + an) * (an / d  + 1) / 2 : 0; // 2：等差数列求和公式分母 
}

int64_t FusedInferAttentionScoreTilingImpl::GetCutBlockNums(int64_t blockSeqLengthKV, int64_t blockSeqLength,
                                                            int64_t sInner, int64_t sOuter, int64_t token)
{
    // 以nextToken视角计算完全被nextToken掩盖的基本块数
    int64_t blockNums = 0;
    int64_t blockToken = token > 0 ? ((token + sInner - 1) / sInner * sInner) : (token / sInner * sInner);
    int64_t outDivIn = sOuter > sInner ? sOuter / sInner : 1;
    int64_t inDivOut = sInner > sOuter ? sInner / sOuter : 1;
    int64_t tolerance = 0;
    int64_t smallSize = 0;
    if (outDivIn >= static_cast<int64_t>(NUM1)) {
        tolerance = outDivIn;
        smallSize = sInner;
    } else {
        tolerance = inDivOut;
        smallSize = sOuter;
    }

    // nextToken与上边右边构成的大三角形
    int64_t innerCutBlockNums = (blockSeqLengthKV - blockToken) / smallSize - tolerance;
    blockNums += SumOfArithmeticSeries(innerCutBlockNums, tolerance);

    // nextToken与上边左边构成的左侧三角形，需要减去
    int64_t innerCutBlockLeftNums = -blockToken / smallSize - tolerance;
    blockNums -= SumOfArithmeticSeries(innerCutBlockLeftNums, tolerance);

    // nextToken与下边右边构成的下侧三角形，需要减去
    int64_t innerCutBlockDownNums = (blockSeqLengthKV - blockSeqLength - blockToken) / smallSize - tolerance;
    blockNums -= SumOfArithmeticSeries(innerCutBlockDownNums, tolerance);

    // nextToken与下边左边构成的小三角形，需要加上
    int64_t innerCutBlockLeftDownNums = (-blockToken - blockSeqLength) / smallSize - tolerance;
    blockNums += SumOfArithmeticSeries(innerCutBlockLeftDownNums, tolerance);
    return blockNums;
} 

int64_t FusedInferAttentionScoreTilingImpl::GetCalcBlockNumsOneHead(const FiaTilingInfo &fiaInfo,
                                                                    int64_t actualSeqLength, int64_t actualSeqLengthKV,
                                                                    uint32_t sOuterSize, uint32_t sInnerSize)
{
    int64_t preTokensLeftUp = 0;
    int64_t nextTokensLeftUp = 0;
    GetPreNextTokensLeftUp(fiaInfo, actualSeqLength, actualSeqLengthKV + fiaInfo.systemPrefixLen,
        preTokensLeftUp, nextTokensLeftUp);
    FixParamWithRowInvalid(fiaInfo, actualSeqLength, actualSeqLengthKV + fiaInfo.systemPrefixLen,
        preTokensLeftUp, nextTokensLeftUp);
    if (!fiaInfo.attenMaskFlag) {
        int64_t outerBlockNums = (actualSeqLength + sOuterSize - 1) / sOuterSize;
        int64_t innerBlockNums =
            (actualSeqLengthKV + sInnerSize - 1) / sInnerSize + (fiaInfo.systemPrefixLen + sInnerSize - 1) / sInnerSize;
        int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
        return toCalcBlockNums;
    } else {
        int64_t innerBlockNumsPrefix = (fiaInfo.systemPrefixLen + sInnerSize - 1) / sInnerSize;
        int64_t outerBlockNums = (actualSeqLength + sOuterSize - 1) / sOuterSize;
        int64_t innerBlockNums = (actualSeqLengthKV + sInnerSize - 1) / sInnerSize;
        int64_t toCalcBlockNums = 0;
        for (uint32_t sOuterIndex = 0; sOuterIndex < outerBlockNums; sOuterIndex++) {
            // 非prefix部分计算，去除prefix影响
            int64_t preTokensNoPrefix = preTokensLeftUp + fiaInfo.systemPrefixLen;
            int64_t nextTokensNoPrefix = nextTokensLeftUp - fiaInfo.systemPrefixLen;
            int64_t sInnerIndexStart = -(preTokensNoPrefix > 0 ? (preTokensNoPrefix + static_cast<int64_t>(sInnerSize) - 1) /
                static_cast<int64_t>(sInnerSize) : preTokensNoPrefix / static_cast<int64_t>(sInnerSize));
            int64_t sInnerIndexEnd = nextTokensNoPrefix > 0 ? (nextTokensNoPrefix + static_cast<int64_t>(sInnerSize) - 1) /
                static_cast<int64_t>(sInnerSize) : nextTokensNoPrefix / static_cast<int64_t>(sInnerSize);
            // prefix部分单独计算
            int64_t sInnerIndexStartPrefix = -(preTokensLeftUp > 0 ? (preTokensLeftUp + static_cast<int64_t>(sInnerSize) - 1) /
                static_cast<int64_t>(sInnerSize) : preTokensLeftUp / static_cast<int64_t>(sInnerSize));
            int64_t sInnerIndexEndPrefix = nextTokensLeftUp > 0 ? (nextTokensLeftUp + static_cast<int64_t>(sInnerSize) - 1) /
                static_cast<int64_t>(sInnerSize) : nextTokensLeftUp / static_cast<int64_t>(sInnerSize);
            // 当前这一行有多少基本块需要计算
            toCalcBlockNums += GetActualInnerBlockNums(sInnerIndexStart, sInnerIndexEnd, innerBlockNums) +
                GetActualInnerBlockNums(sInnerIndexStartPrefix, sInnerIndexEndPrefix, innerBlockNumsPrefix);
            preTokensLeftUp -= sOuterSize;
            nextTokensLeftUp += sOuterSize;
        }
        return toCalcBlockNums;
    }
}

int64_t FusedInferAttentionScoreTilingImpl::GetSInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd,
                                                               int64_t innerBlockNums) const
{
    int64_t sInnerBlockNums = 0;
    if (sInnerIndexEnd < 0) {
        sInnerBlockNums = 0;
    } else if (sInnerIndexEnd < innerBlockNums) {
        sInnerBlockNums = (sInnerIndexStart < 0) ? (sInnerIndexEnd + 1) : (sInnerIndexEnd - sInnerIndexStart + 1);
    } else {
        sInnerBlockNums = (sInnerIndexStart < 0) ?
                              innerBlockNums :
                              (sInnerIndexStart < innerBlockNums ? innerBlockNums - sInnerIndexStart : 0);
    }
    return sInnerBlockNums;
}

void FusedInferAttentionScoreTilingImpl::ComputeSplitNBSeq(const FiaTilingInfo &fiaInfo, const size_t maxCoreNums,
                                                           uint32_t sOuterSize, uint32_t sInnerSize,
                                                           double coreWeightTarget, uint32_t &curCore)
{
    std::vector<uint32_t> coreSposEnd(maxCoreNums, 0U);
    std::vector<uint32_t> coreSposStart(maxCoreNums, 0U);
    std::vector<uint32_t> coreSidEnd(maxCoreNums, 0U);
    std::vector<uint32_t> coreSidStart(maxCoreNums, 0U);
    std::vector<uint32_t> coreNidEnd(maxCoreNums, 0U);
    std::vector<uint32_t> coreNidStart(maxCoreNums, 0U);
    std::vector<int64_t> sparseStartIdx(maxCoreNums, 0L);
    std::vector<uint32_t> bnStartIdx(maxCoreNums, 0U);
    std::vector<int64_t> gS1StartIdx(maxCoreNums, 0L);

    double curWeight = 0;
    curCore = 0;
    uint32_t tmpCoreNidEnd = 0;
    uint32_t tmpCoreSidEnd = 0;
    uint32_t tmpCoreSposEnd = 0;
    int64_t innerBlockNumsPrefix = (fiaInfo.systemPrefixLen + sInnerSize - 1) / sInnerSize;
    for (uint32_t bIdx = 0; bIdx < fiaInfo.bSize; bIdx++) {
        for (uint32_t nIdx = 0; nIdx < nLoopTimes_; nIdx++) {
            int64_t preTokensLeftUp = 0;
            int64_t nextTokensLeftUp = 0;
            GetPreNextTokensLeftUp(fiaInfo, actualSeqLengthsQ_[bIdx],
                                   actualSeqLengthsKV_[bIdx] + fiaInfo.systemPrefixLen, preTokensLeftUp,
                                   nextTokensLeftUp);
            int64_t actualSeqLength = actualSeqLengthsQ_[bIdx];
            int64_t actualSeqLengthKV = actualSeqLengthsKV_[bIdx];
            FixParamWithRowInvalid(fiaInfo, actualSeqLength, actualSeqLengthKV + fiaInfo.systemPrefixLen, preTokensLeftUp,
                                   nextTokensLeftUp);

            int64_t outerBlockNums = (actualSeqLength + sOuterSize - 1) / sOuterSize;
            int64_t innerBlockNums = (actualSeqLengthKV + sInnerSize - 1) / sInnerSize;
            for (uint32_t sOuterIndex = 0; sOuterIndex < outerBlockNums; sOuterIndex++) {
                double dif = coreWeightTarget * double(curCore + 1) - curWeight;
                // 非prefix部分计算，去除prefix影响
                int64_t preTokensNoPrefix = preTokensLeftUp + fiaInfo.systemPrefixLen;
                int64_t nextTokensNoPrefix = nextTokensLeftUp - fiaInfo.systemPrefixLen;
                int64_t sInnerIndexStart = -(preTokensNoPrefix > 0 ?
                                                 (preTokensNoPrefix + static_cast<int64_t>(sInnerSize) - 1) /
                                                     static_cast<int64_t>(sInnerSize) :
                                                 preTokensNoPrefix / static_cast<int64_t>(sInnerSize));
                int64_t sInnerIndexEnd = nextTokensNoPrefix > 0 ?
                                             (nextTokensNoPrefix + static_cast<int64_t>(sInnerSize) - 1) /
                                                 static_cast<int64_t>(sInnerSize) :
                                             nextTokensNoPrefix / static_cast<int64_t>(sInnerSize);

                // prefix部分单独计算
                int64_t sInnerIndexStartPrefix = -(preTokensLeftUp > 0 ?
                                                       (preTokensLeftUp + static_cast<int64_t>(sInnerSize) - 1) /
                                                           static_cast<int64_t>(sInnerSize) :
                                                       preTokensLeftUp / static_cast<int64_t>(sInnerSize));
                int64_t sInnerIndexEndPrefix = nextTokensLeftUp > 0 ?
                                                   (nextTokensLeftUp + static_cast<int64_t>(sInnerSize) - 1) /
                                                       static_cast<int64_t>(sInnerSize) :
                                                   nextTokensLeftUp / static_cast<int64_t>(sInnerSize);

                // The number of innerBlock blocks in each outBlock row represents the calculation amount of each outBlock row.
                int64_t actualInnerBlockNums =
                    GetActualInnerBlockNums(sInnerIndexStart, sInnerIndexEnd, innerBlockNums) +
                    GetActualInnerBlockNums(sInnerIndexStartPrefix, sInnerIndexEndPrefix, innerBlockNumsPrefix);
                if (double(actualInnerBlockNums) - dif > dif &&
                    !(tmpCoreNidEnd == 0 && tmpCoreSidEnd == 0 && tmpCoreSposEnd == 0)) {
                    coreNidEnd[curCore] = tmpCoreNidEnd;
                    coreSidEnd[curCore] = tmpCoreSidEnd;
                    coreSposEnd[curCore] = tmpCoreSposEnd;
                    curCore += 1;
                    coreNidStart[curCore] = nIdx;
                    coreSidStart[curCore] = bIdx;
                    coreSposStart[curCore] = sOuterIndex;
                    bnStartIdx[curCore] = bIdx * nLoopTimes_ + nIdx;
                    gS1StartIdx[curCore] = sOuterIndex;
                }

                tmpCoreNidEnd = nIdx + 1;
                tmpCoreSidEnd = bIdx + 1;
                tmpCoreSposEnd = sOuterIndex + 1;

                curWeight += actualInnerBlockNums;
                preTokensLeftUp -= sOuterSize;
                nextTokensLeftUp += sOuterSize;
            }
        }
    }

    coreNidEnd[curCore] = tmpCoreNidEnd;
    coreSidEnd[curCore] = tmpCoreSidEnd;
    coreSposEnd[curCore] = tmpCoreSposEnd;
    if (curCore + 1 < maxCoreNums) {
        bnStartIdx[curCore + 1] = fiaInfo.bSize * nLoopTimes_;
        gS1StartIdx[curCore + 1] = tmpCoreSposEnd;
    }

    PromptAttentionSeqParams *seqParams = &pfaTilingData_.promptAttentionSeqParams;
    seqParams->set_CoreHeadNumTail(coreNidStart.data());
    seqParams->set_actualS1(coreNidEnd.data());
    seqParams->set_actualCoreNums(coreSidStart.data());
    seqParams->set_singleCoreHeadNumSize(coreSidEnd.data());
    seqParams->set_coreSeqPosStart(coreSposStart.data());
    seqParams->set_coreSeqPosEnd(coreSposEnd.data());
    faRunTilingAdapter_.multiCoreParamsRegbase.set_bnStartIdx(bnStartIdx.data());
    faRunTilingAdapter_.multiCoreParamsRegbase.set_sparseStartIdx(gS1StartIdx.data());
}


/*
 * 计算同时参与计算所需的L2 cache大小
 * 算法思路：
 * 1. 同时最多只有aicNum个核在计算，每个核处理一个(batch, head)组合
 * 2. 每个batch有n2Size个head，因此每个batch最多有n2Size个核同时处理
 * 3. 为最大化L2占用，优先给actualSeqLenKV最大的batch分配核
 * 4. 分配策略：按actualSeqLenKV降序排列，每个batch最多分配n2Size个核，累加到aicNum核为止
 * 5. 如果同时参与计算的KV块超过L2CacheSize的一半就开启S1外切分核
 * 计算公式：L2Size = sum(分配核数 * actualSeqLenKV) * (qkHeadDim + vHeadDim) * dataTypeSize
 */
int64_t FusedInferAttentionScoreTilingImpl::CalcRequiredL2Size(const FiaTilingInfo &fiaInfo)
{
    const int64_t dataTypeSize = 2U;
    // 按actualSeqLenKV降序排列，每个batch最多分配n2Size个核，累加到aicNum核为止
    std::vector<int64_t> sortedSeqLenKV = actualSeqLengthsKV_;
    std::sort(sortedSeqLenKV.begin(), sortedSeqLenKV.end(), std::greater<int64_t>());
    uint32_t remainingCores = platformInfo_.aicNum;
    int64_t totalSeqLenKV = 0;
    for (size_t i = 0; i < sortedSeqLenKV.size() && remainingCores > 0; i++) {
        uint32_t coresForThisBatch = std::min(remainingCores, fiaInfo.n2Size);
        totalSeqLenKV += coresForThisBatch * sortedSeqLenKV[i];
        remainingCores -= coresForThisBatch;
    }
    int64_t requiredL2Size = totalSeqLenKV * (fiaInfo.qkHeadDim + fiaInfo.vHeadDim) * dataTypeSize;
    return requiredL2Size;
}

bool FusedInferAttentionScoreTilingImpl::CheckS1OutSplit(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.antiQuantFlag || fiaInfo.quantFlag || fiaInfo.isOutQuantEnable ||
       fiaInfo.quantMode == FiaQuantMode::FULL_QUANT) {
        return false;
    }

    if (fiaInfo.sysPrefixFlag || fiaInfo.kvPaddingSizeFlag || fiaInfo.qPaddingSizeFlag || dnFlag_ ||
        fiaInfo.learnableSinkFlag || fiaInfo.enableAlibiPse || gsMergeFlag_ || pfaMergeFlag_) {
        return false;
    }

    if (fiaInfo.sparseMode == SPARSE_MODE_BAND ||
       (fiaInfo.sparseMode == SPARSE_MODE_NO_MASK && fiaInfo.attenMaskFlag)) {
        return false;
    }

    if (fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST) {
        return false;
    }

    if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) {
        return false;
    }

    // S1外切分核优化主要针对prefill阶段，需要Q序列长度足够大才能有效复用L2 cache
    // sOuterSize = sOuterFactor_ * CV_RATIO，是S1外切分的最小块大小
    int64_t maxActualSeqLenQ = 0;
    for (size_t i = 0; i < actualSeqLengthsQ_.size(); i++) {
        maxActualSeqLenQ = std::max(maxActualSeqLenQ, actualSeqLengthsQ_[i]);
    }
    uint32_t sOuterSize = sOuterFactor_ * CV_RATIO;
    if (maxActualSeqLenQ < static_cast<int64_t>(sOuterSize)) {
        return false;
    }

    int64_t requiredL2Size = CalcRequiredL2Size(fiaInfo);

    // 当同时参与计算所需的L2Size超过系统配置一半时，开启S1外切分核优化L2cache复用率
    // *2表示超过一半就开启，后续可根据实际情况调整此比例
    return requiredL2Size * 2 >= platformInfo_.l2Size;
}

void FusedInferAttentionScoreTilingImpl::SplitOutSeq(const FiaTilingInfo &fiaInfo)
{
    uint32_t curCoreNum = platformInfo_.aicNum;
    uint32_t sOuterSize = sOuterFactor_ * CV_RATIO;
    int64_t totalSize = 0;
    for (uint32_t bIdx = 0; bIdx < fiaInfo.bSize; bIdx++) {
        int64_t actualSeqLengthsTmp = actualSeqLengthsQ_[bIdx];  // 用于存放减去行无效后，真实的actseqlen
        int64_t preTokensLeftUp = 0;
        int64_t nextTokensLeftUp = 0;
        GetPreNextTokensLeftUp(fiaInfo, actualSeqLengthsQ_[bIdx], actualSeqLengthsKV_[bIdx] + fiaInfo.systemPrefixLen,
                               preTokensLeftUp, nextTokensLeftUp);
        FixParamWithRowInvalid(fiaInfo, actualSeqLengthsTmp, actualSeqLengthsKV_[bIdx] + fiaInfo.systemPrefixLen,
                               preTokensLeftUp, nextTokensLeftUp);

        int64_t outerBlockNums = (actualSeqLengthsTmp + static_cast<int64_t>(sOuterSize) - 1) / static_cast<int64_t>(sOuterSize);
        totalSize += outerBlockNums * fiaInfo.n1Size;
    }

    int64_t s1OuterSize = (fiaInfo.s1Size + sOuterSize - 1) / sOuterSize;
    faRunTilingAdapter_.multiCoreParamsRegbase.set_s1OuterSize(s1OuterSize);

    int64_t actualUsedCoreNum = std::min(totalSize, static_cast<int64_t>(curCoreNum));
    faRunTilingAdapter_.multiCoreParamsRegbase.set_coreNum(static_cast<int32_t>(actualUsedCoreNum));
    faRunTilingAdapter_.multiCoreParamsRegbase.set_totalSize(totalSize);
}

void FusedInferAttentionScoreTilingImpl::SplitNBSeq(const FiaTilingInfo &fiaInfo)
{
    uint32_t curCoreNum = platformInfo_.aicNum;
    uint32_t sOuterSize = sOuterFactor_ * CV_RATIO;
    uint32_t sInnerSize = sInnerFactor_;
    int64_t totalBlockNumsOneHead = 0;
    uint32_t multiSmaxsInnerLoopTimes = 0U;
    std::vector<uint32_t> sInnerLoopTimes(fiaInfo.bSize);
    for (uint32_t bIdx = 0; bIdx < fiaInfo.bSize; bIdx++) {
        int64_t actualSeqLengthsTmp = actualSeqLengthsQ_[bIdx];  // 用于存放减去行无效后，真实的actseqlen
        int64_t preTokensLeftUp = 0;
        int64_t nextTokensLeftUp = 0;
        GetPreNextTokensLeftUp(fiaInfo, actualSeqLengthsQ_[bIdx], actualSeqLengthsKV_[bIdx] + fiaInfo.systemPrefixLen,
                               preTokensLeftUp, nextTokensLeftUp);
        FixParamWithRowInvalid(fiaInfo, actualSeqLengthsTmp, actualSeqLengthsKV_[bIdx] + fiaInfo.systemPrefixLen,
                               preTokensLeftUp, nextTokensLeftUp);

        // sinner方向块数，prefix和origin是分开切的。
        sInnerLoopTimes[bIdx] = (actualSeqLengthsKV_[bIdx] + sInnerSize - 1) / sInnerSize +
                                (fiaInfo.systemPrefixLen + sInnerSize - 1) / sInnerSize;
        multiSmaxsInnerLoopTimes = std::max(multiSmaxsInnerLoopTimes, sInnerLoopTimes[bIdx]);

        totalBlockNumsOneHead += GetCalcBlockNumsOneHead(fiaInfo, actualSeqLengthsQ_[bIdx], actualSeqLengthsKV_[bIdx],
                                                         sOuterSize, sInnerSize);
    }
    pfaTilingData_.promptAttentionSingleCoreParams.set_multiSmaxsInnerLoopTimes(multiSmaxsInnerLoopTimes);

    double coreWeightTarget = (double(totalBlockNumsOneHead * nLoopTimes_) / double(curCoreNum));

    int64_t s1OuterSize = (gsSize_ + sOuterSize - 1) / sOuterSize;
    faRunTilingAdapter_.multiCoreParamsRegbase.set_s1OuterSize(s1OuterSize);

    // The tiling structure element needs to have a length greater than or equal to the length specified
    // by TILING_DATA_FIELD_DEF_ARR. If the tiling structure definition specifies a length of 64,
    // the vector definition needs to compare its size with coreNum and take the larger value
    const size_t tilingElementArrayLen = (static_cast<size_t>(curCoreNum) > 64UL) ? static_cast<size_t>(curCoreNum) :
                                                                                    64UL;
    uint32_t curIndx = 0;

    ComputeSplitNBSeq(fiaInfo, tilingElementArrayLen, sOuterSize, sInnerSize, coreWeightTarget, curIndx);

    uint32_t actualCoreNums = (curIndx + 1) * CV_RATIO;
    pfaTilingData_.promptAttentionSingleCoreParams.set_actualCoreNums(actualCoreNums);
    int64_t sinnerBlocknum = (fiaInfo.s2Size + sInnerSize - 1) / sInnerSize;
    int64_t totalSize = (sinnerBlocknum == 0) ? 0 : (totalBlockNumsOneHead / sinnerBlocknum) * nLoopTimes_;
    int64_t aicUsedCoreNum = curIndx + 1;

    faRunTilingAdapter_.multiCoreParamsRegbase.set_coreNum(static_cast<int32_t>(aicUsedCoreNum));
    OP_LOGI(fiaInfo.opName, "aicUsedCoreNum: %ld", aicUsedCoreNum);
    faRunTilingAdapter_.multiCoreParamsRegbase.set_totalSize(totalSize);
    faRunTilingAdapter_.multiCoreParamsRegbase.set_splitFactorSize(CeilDivision(totalSize, aicUsedCoreNum));
    faRunTilingAdapter_.multiCoreParamsRegbase.set_splitFactorTailSize(
        CalcTailSize(totalSize, faRunTilingAdapter_.multiCoreParamsRegbase.get_splitFactorSize()));
}

bool FusedInferAttentionScoreTilingImpl::CheckFlashDecode(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.s1Size == 1 && fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_PER_BLOCK_FULL_QUANT) {
        return false;
    }
    float flashDecodeBNRatio = 0.4F;  // 0.4, 经验值
    if (fiaInfo.quantMode == FiaQuantMode::ANTI_QUANT) {
        if (fiaInfo.sysPrefixFlag) {
            return false;
        }
        // 如果s2方向上最长还不超过两个sinnersize，不生效FD
        if (fiaInfo.maxActualseq < sInnerFactor_ * 2) {
            return false;
        }
    } else {
        if (fiaInfo.s1Size != 1 || fiaInfo.ropeMode != RopeMode::NO_ROPE) {
            return false;
        }
        if (fiaInfo.maxActualseq < NUM_256) {
            return false;
        }
    }
    uint64_t loopTimes = fiaInfo.bSize * fiaInfo.n2Size * ((fiaInfo.gSize + sOuterFactor_ - 1) / sOuterFactor_);
    if (loopTimes < flashDecodeBNRatio * platformInfo_.aicNum && fiaInfo.gSize == 1) {
        return true;
    }
    if (loopTimes < flashDecodeBNRatio * platformInfo_.aicNum &&
        (fiaInfo.maxActualseq >= NUM_2048)) {  // 2048, 在flash decode + gqa时的经验值
        return true;
    }
    return false;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SplitS2(const FiaTilingInfo &fiaInfo)
{
    uint64_t batchSize = fiaInfo.bSize;
    uint64_t headNumSize = fiaInfo.n1Size;
    uint32_t kvSplitLimit =
        sInnerFactor_ <= 256U ?
            256U :
            sInnerFactor_;  // 256: 经验值 这里的门限值也需要调整，否则会分的很碎，每个核分到的还不到一个基本块
    uint64_t loopTimes = fiaInfo.bSize * fiaInfo.n2Size * (fiaInfo.gSize + sOuterFactor_ - 1) / sOuterFactor_;
    int64_t kvSplitPart = platformInfo_.aicNum / loopTimes;
    while (((fiaInfo.s2Size / kvSplitPart) < kvSplitLimit) && (kvSplitPart > 1)) {
        kvSplitPart--;
    }
    auto vHeadDimAlign = AlignUp(fiaInfo.vHeadDim, NUM_32);

    faRunTilingAdapter_.inputParamsRegbase.set_kvSplitPart(kvSplitPart);
    faRunTilingAdapter_.inputParamsRegbase.set_accumOutSize(batchSize * headNumSize * kvSplitPart * vHeadDimAlign);
    faRunTilingAdapter_.inputParamsRegbase.set_logSumExpSize(batchSize * headNumSize * kvSplitPart *
                                                             (BYTE_BLOCK / sizeof(float)));

    return ge::GRAPH_SUCCESS;
}

// 伪量化
void FusedInferAttentionScoreTilingImpl::SetDequantBaseSize(const FiaTilingInfo &fiaInfo)
{
    sOuterFactor_ = NUM_16;
    if (fiaInfo.s1Size > 1 && gsMergeFlag_) {  // pfa gs1合轴时 s1base=32
        sOuterFactor_ = NUM_32;
    }
    if (fiaInfo.qkHeadDim <= NUM_64) {
        sInnerFactor_ = NUM_1024;
        if (fiaInfo.pseShiftFlag ||
            (fiaInfo.s1Size > 1 && gsMergeFlag_)) {  // pfa gs1合轴 s1base=32 s2base=512 dbase=64
            sInnerFactor_ = NUM_512;
        }
    } else if (fiaInfo.qkHeadDim <= NUM_128) {
        sInnerFactor_ = NUM_512;
    } else if (fiaInfo.qkHeadDim <= NUM_256) {
        sInnerFactor_ = NUM_256;
    } else {
        sInnerFactor_ = NUM_128;
    }
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::CalcInnerSize(const FiaTilingInfo &fiaInfo, uint32_t seqSize)
{
    /**
     * sInnerFactor：s2的切分大小，直接决定了MM的singleN/K和vector的切块大小，但当前切分也并非适用所有case。
     * GQA场景-伪量化：vector比较重，尽量较少vector的循环次数, 因此，cube发小块，期望vector尽量被cube的mte2掩盖。sInnerSize=1024
     */
    SetDequantBaseSize(fiaInfo);
    sInnerFactorSize_ = sInnerFactor_;
    if (seqSize != 0U && sInnerFactor_ > seqSize) {
        sInnerFactor_ = seqSize;
    }
    sInnerSizeAlign_ = AlignUp(sInnerFactor_, BYTE_BLOCK);  // 元素个数按照基本块大小对齐
    return ge::GRAPH_SUCCESS;
}

int64_t FusedInferAttentionScoreTilingImpl::GetActualInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd,
                                                                    int64_t innerBlockNums) const
{
    int64_t sInnerBlockNums = 0;
    if (sInnerIndexEnd < 0) {
        sInnerBlockNums = 0;
    } else if (sInnerIndexEnd < innerBlockNums) {
        sInnerBlockNums = (sInnerIndexStart < 0) ? (sInnerIndexEnd + 1) : (sInnerIndexEnd - sInnerIndexStart + 1);
    } else {
        sInnerBlockNums = (sInnerIndexStart < 0) ?
                              innerBlockNums :
                              (sInnerIndexStart < innerBlockNums ? innerBlockNums - sInnerIndexStart : 0);
    }

    return sInnerBlockNums;
}

void FusedInferAttentionScoreTilingImpl::GetAntiQuantPreNextTokensLeftUp(const FiaTilingInfo &fiaInfo,
                                                                         int64_t actualSeqLength,
                                                                         int64_t actualSeqLengthKV,
                                                                         int64_t &preTokensLeftUp,
                                                                         int64_t &nextTokensLeftUp) const
{
    preTokensLeftUp = fiaInfo.preToken;
    nextTokensLeftUp = fiaInfo.nextToken;
    if (fiaInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) {
        nextTokensLeftUp = actualSeqLengthKV - actualSeqLength;
    } else if (fiaInfo.sparseMode == SPARSE_MODE_BAND) {
        preTokensLeftUp = fiaInfo.preToken - actualSeqLengthKV + actualSeqLength;
        nextTokensLeftUp = fiaInfo.nextToken + actualSeqLengthKV - actualSeqLength;
    }
}

void FusedInferAttentionScoreTilingImpl::FixAntiQuantParamWithRowInvalid(const FiaTilingInfo &fiaInfo,
                                                                         int64_t &actualSeqLength, 
                                                                         int64_t actualSeqLengthKV,
                                                                         int64_t &preTokensLeftUp,
                                                                         int64_t &nextTokensLeftUp)
{
    // 若出现行无效，需要重新计算nexttokens，pretokens，actualseqlen，以便正确计算分核核数
    int64_t nextTokensError = (nextTokensLeftUp < 0) ? -nextTokensLeftUp : 0;
    nextTokensError = nextTokensError > actualSeqLength ? actualSeqLength : nextTokensError;
    
    int64_t preTokensError = (actualSeqLength > actualSeqLengthKV + preTokensLeftUp) ?
                (actualSeqLength - actualSeqLengthKV - preTokensLeftUp) : 0;
    preTokensError = preTokensError > actualSeqLength ? actualSeqLength : preTokensError;

    // 若出现上方行无效，需要重新计算nexttokens，pretokens，actualseqlen
    nextTokensLeftUp += nextTokensError;
    preTokensLeftUp -= nextTokensError;
    actualSeqLength -= nextTokensError;

    // 若出现下方行无效，需要重新计算actualseqlen
    actualSeqLength -= preTokensError;
    if (nextTokensError != 0 || preTokensError != 0) {
        needInit_ = true;
    }
}

void FusedInferAttentionScoreTilingImpl::ComputeDequantSplitNBSeq(const FiaTilingInfo &fiaInfo,
                                                                  std::vector<int64_t> sOuterLoopTimes,
                                                                  std::vector<int64_t> sInnerLoopTimes,
                                                                  int64_t sInnerLoopTimesPrefix, double coreWeightTarget,
                                                                  uint32_t &curCore, const size_t tilingElementArrayLen)
{
    std::vector<int64_t> sparseStartIdx(tilingElementArrayLen, 0L);
    std::vector<uint32_t> bnStartIdx(tilingElementArrayLen, 0U);
    std::vector<int64_t> gS1StartIdx(tilingElementArrayLen, 0L);
    // Temporary algorithm to be optimized
    int64_t curWeight = 0;
    uint32_t tmpCoreNidEnd = 0;  // actual seq为0时不分配核
    uint32_t tmpCoreSidEnd = 0;
    uint32_t tmpCoreSposEnd = 0;
    int64_t actualSeqLengthsTmp= 0;
    int64_t actualSeqLengthsKVTmp = 0;
    for (uint32_t bIdx = 0; bIdx < fiaInfo.bSize; bIdx++) {
        GetActualSeqLength(fiaInfo, actualSeqLengthsTmp, actualSeqLengthsKVTmp, bIdx);
        for (uint32_t headNum = 0; headNum < nLoopTimes_; headNum++) {
            int64_t preTokensLeftUp = 0;
            int64_t nextTokensLeftUp = 0;
            GetAntiQuantPreNextTokensLeftUp(fiaInfo, actualSeqLengthsTmp,
                                   actualSeqLengthsKVTmp + fiaInfo.systemPrefixLen, preTokensLeftUp,
                                   nextTokensLeftUp);
            FixAntiQuantParamWithRowInvalid(fiaInfo, actualSeqLengthsTmp,
                                            actualSeqLengthsKVTmp + fiaInfo.systemPrefixLen,
                                            preTokensLeftUp, nextTokensLeftUp);
            int64_t outerBlockNums = sOuterLoopTimes[bIdx];
            int64_t innerBlockNums = sInnerLoopTimes[bIdx];
            for (uint32_t sOuterIndex = 0; sOuterIndex < outerBlockNums; sOuterIndex++) {
                int64_t dif = static_cast<int64_t>(coreWeightTarget * double(curCore + 1)) - curWeight;
                // 非prefix部分计算，去除prefix影响
                int64_t preTokensNoPrefix = preTokensLeftUp + fiaInfo.systemPrefixLen;
                int64_t nextTokensNoPrefix = nextTokensLeftUp - fiaInfo.systemPrefixLen;
                int64_t sInnerIndexStart = -(preTokensNoPrefix > 0 ?
                                                 (preTokensNoPrefix + static_cast<int64_t>(sInnerFactor_) - 1) /
                                                     static_cast<int64_t>(sInnerFactor_) :
                                                 preTokensNoPrefix / static_cast<int64_t>(sInnerFactor_));
                int64_t sInnerIndexEnd = nextTokensNoPrefix > 0 ?
                                             (nextTokensNoPrefix + static_cast<int64_t>(sInnerFactor_) - 1) /
                                                 static_cast<int64_t>(sInnerFactor_) :
                                             nextTokensNoPrefix / static_cast<int64_t>(sInnerFactor_);

                // prefix部分单独计算
                int64_t sInnerIndexStartPrefix = -(preTokensLeftUp > 0 ?
                                                       (preTokensLeftUp + static_cast<int64_t>(sInnerFactor_) - 1) /
                                                           static_cast<int64_t>(sInnerFactor_) :
                                                       preTokensLeftUp / static_cast<int64_t>(sInnerFactor_));
                int64_t sInnerIndexEndPrefix = nextTokensLeftUp > 0 ?
                                                   (nextTokensLeftUp + static_cast<int64_t>(sInnerFactor_) - 1) /
                                                       static_cast<int64_t>(sInnerFactor_) :
                                                   nextTokensLeftUp / static_cast<int64_t>(sInnerFactor_);

                // 当前这一行有多少基本块需要计算
                int64_t actualInnerBlockNums =
                    GetActualInnerBlockNums(sInnerIndexStart, sInnerIndexEnd, innerBlockNums) +
                    GetActualInnerBlockNums(sInnerIndexStartPrefix, sInnerIndexEndPrefix, sInnerLoopTimesPrefix);
                if (actualInnerBlockNums - dif > dif &&
                    !(tmpCoreNidEnd == 0 && tmpCoreSidEnd == 0 && tmpCoreSposEnd == 0)) {
                    curCore += 1;
                    bnStartIdx[curCore] = bIdx * nLoopTimes_ + headNum;
                    gS1StartIdx[curCore] = sOuterIndex;
                }
                tmpCoreNidEnd = headNum + 1;
                tmpCoreSidEnd = bIdx + 1;
                tmpCoreSposEnd = sOuterIndex + 1;

                curWeight += actualInnerBlockNums;
                preTokensLeftUp -= sOuterFactor_;
                nextTokensLeftUp += sOuterFactor_;
            }
        }
    }
    bnStartIdx[curCore + 1] = fiaInfo.bSize * nLoopTimes_;
    gS1StartIdx[curCore + 1] = tmpCoreSposEnd;

    faRunTilingAdapter_.multiCoreParamsRegbase.set_bnStartIdx(bnStartIdx.data());
    faRunTilingAdapter_.multiCoreParamsRegbase.set_sparseStartIdx(gS1StartIdx.data());
}

void FusedInferAttentionScoreTilingImpl::GetActualSeqLength(const FiaTilingInfo &fiaInfo, int64_t &actualSeqLengths,
                                                            int64_t &actualSeqLengthsKV, uint32_t bIdx)
{
    if (fiaInfo.isMaxWorkspace) {
        actualSeqLengths = fiaInfo.s1Size;
        actualSeqLengthsKV = fiaInfo.s2Size;
        return;
    }
    uint32_t nNumOfQInOneGroup = 1;
    nNumOfQInOneGroup = fiaInfo.gSize;
    if (fiaInfo.qLayout == FiaLayout::TND) {
        actualSeqLengths = bIdx == 0 ? fiaInfo.opParamInfo.actualSeqLengthsQ.tensor->GetData<int64_t>()[0] :
                                       fiaInfo.opParamInfo.actualSeqLengthsQ.tensor->GetData<int64_t>()[bIdx] -
                                       fiaInfo.opParamInfo.actualSeqLengthsQ.tensor->GetData<int64_t>()[bIdx - 1];
        if (gsMergeFlag_) {
            actualSeqLengths *= nNumOfQInOneGroup;
        }
        actualSeqLengthsKV = fiaInfo.opParamInfo.actualSeqLengths.tensor->GetData<int64_t>()[bIdx];
        if (fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION && bIdx > 0) {
            actualSeqLengthsKV -= fiaInfo.opParamInfo.actualSeqLengths.tensor->GetData<int64_t>()[bIdx - 1];
        }
        if (actualSeqLengthsKV <= 0) {
            needInit_ = true;
        }
    } else {
        if (fiaInfo.actualSeqLenFlag && fiaInfo.actualLenKvDims > 0 &&
            fiaInfo.opParamInfo.actualSeqLengths.tensor->GetData<int64_t>() != nullptr) { // kvLengths
            actualSeqLengthsKV = fiaInfo.actualLenKvDims == NUM1 ?
                                    fiaInfo.opParamInfo.actualSeqLengths.tensor->GetData<int64_t>()[0] :
                                    fiaInfo.opParamInfo.actualSeqLengths.tensor->GetData<int64_t>()[bIdx];
        } else {
            actualSeqLengthsKV = fiaInfo.kvListSeqLens.size() == NUM1 ?
                                    fiaInfo.kvListSeqLens[0] :
                                    fiaInfo.kvListSeqLens[bIdx];
        }
        if (actualSeqLengthsKV < fiaInfo.s2Size) {
            needInit_ = true;
        }
        if (actualSeqLenQFlag_) { // qLengths
            actualSeqLengths = fiaInfo.actualLenQDims == NUM1 ?
                                    fiaInfo.opParamInfo.actualSeqLengthsQ.tensor->GetData<int64_t>()[0] :
                                    fiaInfo.opParamInfo.actualSeqLengthsQ.tensor->GetData<int64_t>()[bIdx];
        } else {
            actualSeqLengths = fiaInfo.s1Size;
            if (gsMergeFlag_) {
                actualSeqLengths = fiaInfo.s1Size * nNumOfQInOneGroup;
            }
        }
    }
}

int64_t FusedInferAttentionScoreTilingImpl::GetAntiQuantCalcBlockNumsOneHead(
    const FiaTilingInfo &fiaInfo, int64_t outerBlockNums, int64_t innerBlockNums, int64_t sInnerLoopTimesPrefix,
    int64_t preTokensLeftUp, int64_t nextTokensLeftUp)
{
    if (!fiaInfo.attenMaskFlag) {
        return (innerBlockNums + sInnerLoopTimesPrefix) * outerBlockNums;
    } else {
        int64_t blockSeqLength = outerBlockNums * static_cast<int64_t>(sOuterFactor_);
        int64_t blockSeqLengthKV = innerBlockNums * static_cast<int64_t>(sInnerFactorSize_);
        int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
        // 必须满足pretoken + nexttoken > 0，否则会减出小于0的块数，这里需要去除prefix影响
        toCalcBlockNums -= GetCutBlockNums(blockSeqLengthKV, blockSeqLength,
            static_cast<int64_t>(sInnerFactorSize_), static_cast<int64_t>(sOuterFactor_),
            nextTokensLeftUp - fiaInfo.systemPrefixLen);
        toCalcBlockNums -= GetCutBlockNums(blockSeqLengthKV, blockSeqLength,
            static_cast<int64_t>(sInnerFactorSize_), static_cast<int64_t>(sOuterFactor_),
            blockSeqLengthKV - blockSeqLength + preTokensLeftUp + fiaInfo.systemPrefixLen);

        // prefix部分单独计算
        int64_t blockSharedPrefix = sInnerLoopTimesPrefix * static_cast<int64_t>(sInnerFactorSize_);
        toCalcBlockNums += sInnerLoopTimesPrefix * outerBlockNums;
        toCalcBlockNums -= GetCutBlockNums(blockSharedPrefix, blockSeqLength,
            static_cast<int64_t>(sInnerFactorSize_), static_cast<int64_t>(sOuterFactor_), nextTokensLeftUp);
        toCalcBlockNums -= GetCutBlockNums(blockSharedPrefix, blockSeqLength,
            static_cast<int64_t>(sInnerFactorSize_), static_cast<int64_t>(sOuterFactor_),
            blockSharedPrefix - blockSeqLength + preTokensLeftUp);
        return toCalcBlockNums;
    }
}

void FusedInferAttentionScoreTilingImpl::DequantCubeSplitBNSeq(const FiaTilingInfo &fiaInfo)  // 伪量化只用Cube视角分核
{
    uint32_t curCoreNum = platformInfo_.aicNum;
    int64_t totalBlockNumsOneHead = 0;
    std::vector<int64_t> sOuterLoopTimes(fiaInfo.bSize, 0U);
    std::vector<int64_t> sInnerLoopTimes(fiaInfo.bSize, 0U);
    int64_t multiSmaxsInnerLoopTimes = 0U;
    int64_t sInnerLoopTimesPrefix =
        (fiaInfo.systemPrefixLen + static_cast<int64_t>(sInnerFactor_) - 1) / static_cast<int64_t>(sInnerFactor_);
    for (uint32_t bIdx = 0; bIdx < fiaInfo.bSize; bIdx++) {
        int64_t actualSeqLengthsTmp = 0;
        int64_t actualSeqLengthsKVTmp = 0;
        GetActualSeqLength(fiaInfo, actualSeqLengthsTmp, actualSeqLengthsKVTmp, bIdx);
        int64_t preTokensLeftUp = 0;
        int64_t nextTokensLeftUp = 0;
        GetAntiQuantPreNextTokensLeftUp(fiaInfo, actualSeqLengthsTmp, actualSeqLengthsKVTmp + fiaInfo.systemPrefixLen,
                               preTokensLeftUp, nextTokensLeftUp);
        FixAntiQuantParamWithRowInvalid(fiaInfo, actualSeqLengthsTmp, actualSeqLengthsKVTmp + fiaInfo.systemPrefixLen,
                               preTokensLeftUp, nextTokensLeftUp);

        sOuterLoopTimes[bIdx] =
            (actualSeqLengthsTmp + static_cast<int64_t>(sOuterFactor_) - 1) / static_cast<int64_t>(sOuterFactor_);
        sInnerLoopTimes[bIdx] =
            (actualSeqLengthsKVTmp + static_cast<int64_t>(sInnerFactor_) - 1) / static_cast<int64_t>(sInnerFactor_);
        multiSmaxsInnerLoopTimes = std::max(multiSmaxsInnerLoopTimes, sInnerLoopTimes[bIdx] + sInnerLoopTimesPrefix);

        totalBlockNumsOneHead += GetAntiQuantCalcBlockNumsOneHead(fiaInfo, sOuterLoopTimes[bIdx],sInnerLoopTimes[bIdx],
                                                         sInnerLoopTimesPrefix, preTokensLeftUp,
                                                         nextTokensLeftUp);
    }
    double coreWeightTarget = (double(totalBlockNumsOneHead * nLoopTimes_) / double(curCoreNum));

    int64_t s1OuterSize = (fiaInfo.s1Size + sOuterFactor_ - 1) / sOuterFactor_;
    faRunTilingAdapter_.multiCoreParamsRegbase.set_s1OuterSize(s1OuterSize);
    const size_t tilingElementArrayLen = (static_cast<size_t>(curCoreNum) > 64UL) ? static_cast<size_t>(curCoreNum) :
                                                                                    64UL;
    uint32_t curIndx = 0;
    ComputeDequantSplitNBSeq(fiaInfo, sOuterLoopTimes, sInnerLoopTimes, sInnerLoopTimesPrefix, coreWeightTarget, curIndx,
                             tilingElementArrayLen);
    int64_t sinnerBlocknum = (fiaInfo.maxActualseq + sInnerFactor_ - 1) / sInnerFactor_;
    int64_t totalSize = ((sinnerBlocknum == 0) ? 0 : (totalBlockNumsOneHead / sinnerBlocknum) * nLoopTimes_);
    int64_t actualUsedCoreNum = static_cast<int64_t>((curIndx + 1));
    faRunTilingAdapter_.multiCoreParamsRegbase.set_coreNum(static_cast<int32_t>(actualUsedCoreNum));
    faRunTilingAdapter_.multiCoreParamsRegbase.set_totalSize(totalSize);
    faRunTilingAdapter_.multiCoreParamsRegbase.set_splitFactorSize(CeilDivision(totalSize, actualUsedCoreNum));
    faRunTilingAdapter_.multiCoreParamsRegbase.set_splitFactorTailSize(
        CalcTailSize(totalSize, faRunTilingAdapter_.multiCoreParamsRegbase.get_splitFactorSize()));
}

void FusedInferAttentionScoreTilingImpl::SplitDequant(const FiaTilingInfo &fiaInfo)
{
    CalcInnerSize(fiaInfo, fiaInfo.maxActualseq);
    DequantCubeSplitBNSeq(fiaInfo);
    if (!isPFAFlag_) {
        if (CheckFlashDecode(fiaInfo)) {
            flashDecodeFlag_ = true;
            SplitS2(fiaInfo);
        }
    }
}

bool FusedInferAttentionScoreTilingImpl::CheckEnableDN(const FiaTilingInfo &fiaInfo) const
{
    constexpr uint32_t dLimitDN = DSIZE_128;
    constexpr uint32_t sOuterLimitDN = SOUTER_64;
    bool res = !fiaInfo.attenMaskFlag && !fiaInfo.pseShiftFlag && !fiaInfo.enableAlibiPse &&
               fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION &&
               fiaInfo.ropeMode == RopeMode::NO_ROPE && fiaInfo.qkHeadDim <= dLimitDN &&
               fiaInfo.vHeadDim <= dLimitDN && !fiaInfo.sysPrefixFlag &&
               (fiaInfo.quantMode == FiaQuantMode::NO_QUANT ||
                fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_PER_BLOCK_FULL_QUANT) &&
                sOuterFactor_ * CV_RATIO > sOuterLimitDN;
    return res;
}

bool FusedInferAttentionScoreTilingImpl::CheckQKVActualSeqLengthsRight(const FiaTilingInfo &fiaInfo)
{
    for (uint32_t bIdx = 0; bIdx < fiaInfo.bSize; bIdx++) {
        if ((actualSeqLengthsQ_[bIdx] % SOUTER_32 > 0) || (actualSeqLengthsKV_[bIdx] <= SINNER_128)) {
            return false;
        }
    }
    return true;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SplitPolicy(gert::TilingContext *context,
                                                                const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.quantMode == FiaQuantMode::ANTI_QUANT) {
        SplitDequant(fiaInfo);
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(AdjustSinnerAndSouter(context, fiaInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(fiaInfo.opName, "Get sinner and souter fail!"), return ge::GRAPH_FAILED);

    ApplySinnerSouterOverrides(fiaInfo);
    ApplySplitStrategy(fiaInfo);
    ApplyRowInvalidCheck(fiaInfo);

    return ge::GRAPH_SUCCESS;
}

void FusedInferAttentionScoreTilingImpl::ApplySinnerSouterOverrides(const FiaTilingInfo &fiaInfo)
{
    dnFlag_ = CheckEnableDN(fiaInfo);
    isQKVActualSeqLengthsRightFlag_ = CheckQKVActualSeqLengthsRight(fiaInfo);
    if (dnFlag_ && isQKVActualSeqLengthsRightFlag_ && fiaInfo.qkHeadDim == fiaInfo.vHeadDim &&
        fiaInfo.qkHeadDim == DSIZE_64) {
        sInnerFactor_ = SINNER_256;
    }
    if (dnFlag_ && fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_PER_BLOCK_FULL_QUANT &&
        fiaInfo.qkHeadDim == fiaInfo.vHeadDim && fiaInfo.qkHeadDim <= DSIZE_128) {
        sInnerFactor_ = SINNER_256;
    }
    if (fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_MXFP8_FULL_QUANT) {
        sOuterFactor_ = SOUTER_64;
        sInnerFactor_ = SINNER_512;
    }
}

void FusedInferAttentionScoreTilingImpl::ApplySplitStrategy(const FiaTilingInfo &fiaInfo)
{
    enableS1OutSplit = CheckS1OutSplit(fiaInfo);
    if (enableS1OutSplit) {
        SplitOutSeq(fiaInfo);
        return;
    }

    SplitNBSeq(fiaInfo);
    if (isIFAFlag_ && !pfaMergeFlag_ && CheckFlashDecode(fiaInfo)) {
        flashDecodeFlag_ = true;
        OP_LOGI(fiaInfo.opName, "FlashDecode is enable.");
        SplitS2(fiaInfo);
    }
}

void FusedInferAttentionScoreTilingImpl::ApplyRowInvalidCheck(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.sparseMode == SPARSE_MODE_NO_MASK && fiaInfo.attenMaskFlag &&
        (fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag)) {
        isRowInvalid_ = true;
    }
}

static uint32_t SnapDimToBucket(uint32_t dim)
{
    if (dim <= DSIZE_64)  return DSIZE_64;
    if (dim <= DSIZE_128) return DSIZE_128;
    if (dim <= DSIZE_256) return DSIZE_256;
    if (dim <= DSIZE_512) return DSIZE_512;
    if (dim <= DSIZE_576) return DSIZE_576;
    return dim;
}

struct TilingConfigRule {
    bool isAntiQuant;
    uint32_t sOuter;
    uint32_t sInner;
    uint32_t dSize;
    uint32_t dVsize;
    uint64_t config;
    bool requireRopeSplit = false;
};

void FusedInferAttentionScoreTilingImpl::UpdateTilingKeyConfig(const FiaTilingInfo &fiaInfo)
{
    uint32_t dSize = fiaInfo.qkHeadDim;
    if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) {
        dSize += fiaInfo.ropeHeadDim;
    }
    dSize = SnapDimToBucket(dSize);
    uint32_t dVsize = SnapDimToBucket(fiaInfo.vHeadDim);

    uint32_t sOuter = sOuterFactor_ * 2;
    uint32_t sInner = sInnerFactor_;
    bool isAntiQuant = (fiaInfo.quantMode == FiaQuantMode::ANTI_QUANT);
    if (isAntiQuant) {
        sInner = sInnerFactorSize_;
        sOuter = sOuterFactor_;
    }

    static const std::vector<TilingConfigRule> configRules = {
        {true, 16, 1024, DSIZE_64, DSIZE_64, Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64},
        {true, 16, 512, DSIZE_64, DSIZE_64, Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64},
        {true, 16, 512, DSIZE_128, DSIZE_128, Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128},
        {true, 16, 256, DSIZE_256, DSIZE_256, Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256},
        {true, 16, 128, DSIZE_512, DSIZE_512, Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512},
        {true, 32, 512, DSIZE_64, DSIZE_64, Config_S1Aligned32_S2Aligned512_DAligned64_DVAligned64},
        {true, 32, 512, DSIZE_128, DSIZE_128, Config_S1Aligned32_S2Aligned512_DAligned128_DVAligned128},
        {true, 32, 256, DSIZE_256, DSIZE_256, Config_S1Aligned32_S2Aligned256_DAligned256_DVAligned256},
        {true, 32, 128, DSIZE_512, DSIZE_512, Config_S1Aligned32_S2Aligned128_DAligned512_DVAligned512},

        {false, SOUTER_64, SINNER_64, DSIZE_256, DSIZE_256, Config_S1Aligned64_S2Aligned64_DAligned256_DVAligned256},
        {false, SOUTER_64, SINNER_64, DSIZE_512, DSIZE_512, Config_S1Aligned64_S2Aligned64_DAligned512_DVAligned512},
        {false, SOUTER_64, SINNER_256, DSIZE_64, DSIZE_64, Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64},
        {false, SOUTER_64, SINNER_256, DSIZE_128, DSIZE_128, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128},
        {false, SOUTER_128, SINNER_128, DSIZE_64, DSIZE_64, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64},
        {false, SOUTER_128, SINNER_128, DSIZE_128, DSIZE_128, Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128, true},
        {false, SOUTER_128, SINNER_128, DSIZE_128, DSIZE_128, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128},
        {false, SOUTER_128, SINNER_128, DSIZE_192, DSIZE_128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128},
        {false, SOUTER_128, SINNER_128, DSIZE_256, DSIZE_128, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128},
        {false, SOUTER_128, SINNER_128, DSIZE_256, DSIZE_256, Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256},
        {false, SOUTER_128, SINNER_128, DSIZE_512, DSIZE_512, Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512},
        {false, SOUTER_128, SINNER_256, DSIZE_64, DSIZE_64, Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64},
        {false, SOUTER_64, SINNER_128, DSIZE_576, DSIZE_512, Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512},
        {false, SOUTER_64, SINNER_256, DSIZE_256, DSIZE_256, Config_S1Aligned64_S2Aligned256_DAligned256_DVAligned256},
        {false, SOUTER_128, SINNER_256, DSIZE_128, DSIZE_128, Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128},
        {false, SOUTER_128, SINNER_128, DSIZE_128, DSIZE_64, Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned64},
        {false, SOUTER_128, SINNER_128, DSIZE_64, DSIZE_128, Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned128},
        {false, SOUTER_64, SINNER_256, DSIZE_128, DSIZE_64, Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned64},
        {false, SOUTER_64, SINNER_256, DSIZE_64, DSIZE_128, Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned128},
        {false, SOUTER_128, SINNER_512, DSIZE_64, DSIZE_64, Config_S1Aligned128_S2Aligned512_DAligned64_DVAligned64},
        {false, SOUTER_128, SINNER_512, DSIZE_128, DSIZE_128, Config_S1Aligned128_S2Aligned512_DAligned128_DVAligned128},
    };

    for (const auto &rule : configRules) {
        if (rule.isAntiQuant != isAntiQuant) continue;
        if (rule.sOuter != sOuter || rule.sInner != sInner) continue;
        if (rule.dSize != dSize || rule.dVsize != dVsize) continue;
        if (rule.requireRopeSplit && fiaInfo.ropeMode != RopeMode::ROPE_SPLIT) continue;
        tilingKeyInfo_.config = rule.config;
        return;
    }
}

void FusedInferAttentionScoreTilingImpl::UpdateTilingKeyLayout(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.qLayout == FiaLayout::BNSD) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BNSD_BNSD;
    } else if (fiaInfo.qLayout == FiaLayout::TND) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_TND_TND;
    } else if (fiaInfo.qLayout == FiaLayout::NTD) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_NTD_NTD;
    } else if (fiaInfo.qLayout == FiaLayout::BSH || fiaInfo.qLayout == FiaLayout::BSND) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BSH_BSH;
    }
}

void FusedInferAttentionScoreTilingImpl::UpdateTilingKeyPseMode(const FiaTilingInfo &fiaInfo)
{
    if (!fiaInfo.pseShiftFlag && !fiaInfo.enableAlibiPse) {
        tilingKeyInfo_.pseMode = PSE_MODE_PSE_NONE_TYPE;
    } else {
        tilingKeyInfo_.pseMode = *(fiaInfo.opParamInfo.pseType);
    }
}

void FusedInferAttentionScoreTilingImpl::UpdateTilingKeyQuantMode(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.quantMode == FiaQuantMode::NO_QUANT) {
        tilingKeyInfo_.quantMode = NoQuantMode;
    }

    if (fiaInfo.quantMode == FiaQuantMode::ANTI_QUANT) {
        uint32_t antiQuantMode = fiaInfo.keyAntiquantMode;
        tilingKeyInfo_.quantMode = 0; // 默认处理，与重构前保持一致
        switch (antiQuantMode) {
            case 0:
                tilingKeyInfo_.quantMode = (fiaInfo.valueAntiquantMode == 1) ? AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN :
                                                                               AntiquantMode_PER_CHANNEL;
                break;
            case 2:
                tilingKeyInfo_.quantMode =
                    AntiquantMode_PER_CHANNEL;  // perTensorHead,使用同perChannel,通过perHeadFlag来区分
                break;
            case 1:                 // AntiquantMode_PER_TOKEN
            case 3:                 // AntiquantMode_PER_TOKEN_HEAD
            case 4:                 // AntiquantMode_PER_TOKEN_PAGE_ATTENTION
            case 5:                 // AntiquantMode_PER_TOKEN_HEAD_PAGE_ATTENTION
                tilingKeyInfo_.quantMode = AntiquantMode_K_PER_TOKEN;
                break;
            default:
                break;
        }
    }

    if (fiaInfo.quantMode == FiaQuantMode::FULL_QUANT) {
        if (fiaInfo.ropeMode == RopeMode::ROPE_SPLIT) {
            tilingKeyInfo_.quantMode = FULLQUANT_MODE_Q_PER_TOKEN_HEAD_KV_PER_TENSOR;
        } else if (*fiaInfo.opParamInfo.keyAntiquantMode == 7 && *fiaInfo.opParamInfo.valueAntiquantMode == 7 &&
                   *fiaInfo.opParamInfo.queryQuantMode == 7) {
            tilingKeyInfo_.quantMode = FULLQUANT_MODE_QKV_PERBLOCK;
        } else if (*fiaInfo.opParamInfo.keyAntiquantMode == 6 && *fiaInfo.opParamInfo.valueAntiquantMode == 8 &&
                   *fiaInfo.opParamInfo.queryQuantMode == 6){ // 6: per_token_group, 8: per_channel_group
            tilingKeyInfo_.quantMode = FULLQUANT_MODE_QKV_MXFP8_PREFILL;
        } else {
            tilingKeyInfo_.quantMode = FullQuantMode;
        }
    }
}

void FusedInferAttentionScoreTilingImpl::UpdateTilingKeyHasRope(const FiaTilingInfo &fiaInfo)
{
    tilingKeyInfo_.hasRope = false;
    if (fiaInfo.quantMode == FiaQuantMode::NO_QUANT) {
        if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D128) {
            tilingKeyInfo_.hasRope = true;
        }
    }
    if (fiaInfo.quantMode == FiaQuantMode::FULL_QUANT) {
        if (fiaInfo.ropeMode == RopeMode::ROPE_SPLIT) {
            tilingKeyInfo_.hasRope = true;
        }
    }
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::UpdateTilingKeyInfo(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.emptyTensorFlag) {
        tilingKeyInfo_.emptyTensor = fiaInfo.emptyTensorFlag;
    } else {
        UpdateTilingKeyLayout(fiaInfo);
        UpdateTilingKeyConfig(fiaInfo);
        UpdateTilingKeyPseMode(fiaInfo);
        UpdateTilingKeyQuantMode(fiaInfo);
        tilingKeyInfo_.isFd = flashDecodeFlag_;
        if (fiaInfo.sysPrefixFlag) {
            tilingKeyInfo_.isFd = false;
        }
        tilingKeyInfo_.hasAttenMask = fiaInfo.attenMaskFlag;
        UpdateTilingKeyHasRope(fiaInfo);
        if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
            // 重构前模板，KvLayout无需做常量化，为减少tilingkey，PA场景下，KvLayout统一取值KvLayoutType_ENABLE_PA
            tilingKeyInfo_.KvLayoutType = 1;
        } else {
            tilingKeyInfo_.KvLayoutType = 0;
        }
        tilingKeyInfo_.emptyTensor = fiaInfo.emptyTensorFlag;
        tilingKeyInfo_.enableKvPrefix = fiaInfo.sysPrefixFlag;
        tilingKeyInfo_.enableS1OutSplit = enableS1OutSplit;
        tilingKeyInfo_.isReconstructTemp = false;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::GenTilingKey(gert::TilingContext *context,
                                                                 const FiaTilingInfo &fiaInfo)
{
    UpdateTilingKeyInfo(fiaInfo);
    uint64_t genTilingkey = GET_TPL_TILING_KEY(
        tilingKeyInfo_.inputLayout, tilingKeyInfo_.config, tilingKeyInfo_.pseMode, tilingKeyInfo_.quantMode,
        tilingKeyInfo_.hasAttenMask, tilingKeyInfo_.hasRope, tilingKeyInfo_.KvLayoutType, tilingKeyInfo_.isFd,
        tilingKeyInfo_.emptyTensor, tilingKeyInfo_.enableKvPrefix, tilingKeyInfo_.enableS1OutSplit,
        tilingKeyInfo_.isReconstructTemp);
    context->SetTilingKey(genTilingkey);
    OP_LOGI(fiaInfo.opName, "The tilingkey is %llu.", genTilingkey);
    OP_LOGI(fiaInfo.opName,
            "The tilingkey param is inOutLayoutType: %llu, config: %llu, pseMode: %llu, quantMode: %llu, "
            "hasAttenMask: %u, hasRope: %u, KvLayoutType: %llu, isFd: %u, emptyTensor: %u, "
            "enableKvPrefix: %u, enableS1OutSplit: %u, isReconstructTemp: %u",
            tilingKeyInfo_.inputLayout, tilingKeyInfo_.config, tilingKeyInfo_.pseMode, tilingKeyInfo_.quantMode,
            tilingKeyInfo_.hasAttenMask, tilingKeyInfo_.hasRope, tilingKeyInfo_.KvLayoutType, tilingKeyInfo_.isFd,
            tilingKeyInfo_.emptyTensor, tilingKeyInfo_.enableKvPrefix, tilingKeyInfo_.enableS1OutSplit,
            tilingKeyInfo_.isReconstructTemp);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SetBlockDim(gert::TilingContext *context,
                                                                const FiaTilingInfo &fiaInfo)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    blockDim_ = ascendcPlatform.CalcTschBlockDim(platformInfo_.aivNum, platformInfo_.aicNum, platformInfo_.aivNum);
    context->SetBlockDim(blockDim_);
    OP_LOGI(fiaInfo.opName, "blockDim: %d", blockDim_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SetWorkspaceNormal(const FiaTilingInfo &fiaInfo,
                                                                       int64_t &curWorkspaceSize)
{
    size_t sysWorkspaceSize = platformInfo_.defaultSysWorkspaceSize;
    size_t accumOutSize = 0;
    size_t logSumExpSize = 0;
    if (fiaInfo.isMaxWorkspace) {  // 计算maxWorkSpaceSize时默认开启FD且使用最大核数进行归约
        auto vHeadSize = AlignUp(fiaInfo.vHeadDim, NUM_64);
        accumOutSize = platformInfo_.aicNum * vHeadSize * sizeof(float);
        logSumExpSize = platformInfo_.aicNum * BYTE_BLOCK * CV_RATIO;
    } else if (flashDecodeFlag_) {
        accumOutSize = faRunTilingAdapter_.inputParamsRegbase.get_accumOutSize() * sizeof(float);
        logSumExpSize = faRunTilingAdapter_.inputParamsRegbase.get_logSumExpSize() * sizeof(float);
    }

    int64_t bmm2Bytes = 0;
    int64_t vec2Bytes = 0;
    uint32_t dSize = fiaInfo.vHeadDim;
    uint32_t dVBasicBlock = 0;
    if (dSize <= DSIZE_64) {
        dVBasicBlock = DSIZE_64;
    } else if (dSize <= DSIZE_128) {
        dVBasicBlock = DSIZE_128;
    } else if (dSize <= DSIZE_256) {
        dVBasicBlock = DSIZE_256;
    } else if (dSize <= DSIZE_512) {
        dVBasicBlock = DSIZE_512;
    }
    int64_t bmm2ResBlockSize = dVBasicBlock;
    if (dVBasicBlock > DSIZE_256) {
        bmm2ResBlockSize = DSIZE_512;
    }
    uint32_t s1BasicBlock = sOuterFactor_ * CV_RATIO;
    uint32_t calcTypeSize = sizeof(float);
    if ((!dnFlag_ && dSize > DSIZE_128) || (dnFlag_ && dSize > DSIZE_192)) {
        bmm2Bytes = s1BasicBlock * bmm2ResBlockSize * calcTypeSize;
        if (dVBasicBlock > DSIZE_256) {
            vec2Bytes = s1BasicBlock * dVBasicBlock * calcTypeSize;
        }
    }
    curWorkspaceSize = (bmm2Bytes + vec2Bytes) * 3 * platformInfo_.coreNum + // 3: perload 2次 需要2+1
                        sysWorkspaceSize + accumOutSize + logSumExpSize * 2;

    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        // 2 bmm, db, ensure alignment of each structure 64B, dcci cacheline needs
        curWorkspaceSize += static_cast<uint64_t>(platformInfo_.coreNum) * 2 * 2 * 64;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SetWorkspaceAntiQuant(const FiaTilingInfo &fiaInfo,
                                                                          int64_t &workspaceSize_)
{
    uint32_t cubeL1UbSize = (512 / 2) * 1024;   // david L1 512K,提供给两个Vec使用,单个vector占用256K
    uint32_t cubeL0CUbSize = (256 / 2) * 1024;  // david L0C 256K,提供给两个Vec使用,单个vector占用128K

    workspaceSize_ = platformInfo_.defaultSysWorkspaceSize;
    // L1
    workspaceSize_ += cubeL1UbSize * platformInfo_.coreNum;
    // L0C
    workspaceSize_ += cubeL0CUbSize * platformInfo_.coreNum;
    if (fiaInfo.isMaxWorkspace) {  // 计算maxWorkSpaceSize时默认开启FD且使用最大核数进行归约
        uint32_t maxAccumOutSize = platformInfo_.aicNum * fiaInfo.qkHeadDim;
        uint32_t maxLogSumExpSize = platformInfo_.aicNum * (BYTE_BLOCK / sizeof(float));
        workspaceSize_ += (maxAccumOutSize + maxLogSumExpSize * 2) * sizeof(float);  // 2 : sMax 和 sSum
    } else if (flashDecodeFlag_) {
        workspaceSize_ += (faRunTilingAdapter_.inputParamsRegbase.get_accumOutSize() +
                           faRunTilingAdapter_.inputParamsRegbase.get_logSumExpSize() * 2) *
                          sizeof(float);  // 2 : sMax 和 sSum
    }

    workspaceSize_ += 100 * 1024 * 1024;  // 100*1024*1024: extra workspace for dump in david

    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        workspaceSize_ += platformInfo_.coreNum * 64 * 2;  // bmm1 bmm2 2份
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SetWorkspacePTQuant(const FiaTilingInfo &fiaInfo,
                                                                        int64_t &curWorkspaceSize)
{
    uint64_t maxSpmSize = 0;  // 待处理 tilingData.promptAttentionTensorSizeRect.get_spmTmpSize();
    int64_t mm1ResSize = sOuterFactor_ * CV_RATIO * sInnerFactor_;
    int64_t mm2ResSize = sOuterFactor_ * CV_RATIO * fiaInfo.vHeadDim;
    curWorkspaceSize = platformInfo_.defaultSysWorkspaceSize +
                       platformInfo_.coreNum * 2 * (maxSpmSize + mm1ResSize * 2 + mm2ResSize * 2);
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        // 2 bmm, db, ensure alignment of each structure 64B, dcci cacheline needs
        curWorkspaceSize += static_cast<uint64_t>(platformInfo_.coreNum) * 2 * 2 * 64;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::GetWorkspace(gert::TilingContext *context,
                                                                 const FiaTilingInfo &fiaInfo)
{
    int64_t workspace = 0;
    if (fiaInfo.emptyTensorFlag) {
        workspace = platformInfo_.defaultSysWorkspaceSize;
    } else if (fiaInfo.quantMode == FiaQuantMode::ANTI_QUANT) {
        OP_CHECK_IF(SetWorkspaceAntiQuant(fiaInfo, workspace) != ge::GRAPH_SUCCESS,
                    OP_LOGE(fiaInfo.opName, "Get workspace failed ."), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(SetWorkspaceNormal(fiaInfo, workspace) != ge::GRAPH_SUCCESS,
                    OP_LOGE(fiaInfo.opName, "Get workspace failed ."), return ge::GRAPH_FAILED);
    }

    OP_LOGI(fiaInfo.opName, "Workspaces: %ld", workspace);
    size_t *workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = workspace;
    return ge::GRAPH_SUCCESS;
}

bool FusedInferAttentionScoreTilingImpl::EnableMTE2BmmPipe(const FiaTilingInfo &fiaInfo,
                                                           matmul_tiling::MatmulApiTiling &bmm,
                                                           TCubeTiling &bmmTilingData)
{
    if (fiaInfo.s1Size > NUM_16) {
        return true;
    }
    uint32_t baseK = 32U;
    uint32_t headSize = fiaInfo.qkHeadDim;
    if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) {
        headSize = fiaInfo.qkHeadDim + fiaInfo.ropeHeadDim;
    }
    if (headSize % baseK != 0) {
        return true;
    }

    uint32_t baseM = std::min(uint32_t(128), sOuterFactor_);
    uint32_t baseN = std::min(uint32_t(512), sInnerFactor_);
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        baseN = 128;  //128
    }
    bool res = (bmm.SetFixSplit(baseM, baseN, baseK) == ge::GRAPH_SUCCESS);
    OP_CHECK_IF(!res, OP_LOGE(fiaInfo.opName, "set fix split fail"), return ge::GRAPH_FAILED);
    //check
    res = bmm.GetTiling(bmmTilingData) != -1;
    return res;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SetDequantMMTilingData(const gert::TilingContext *context,
                                                                           const FiaTilingInfo &fiaInfo)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    matmul_tiling::MatmulApiTiling bmm1(ascendcPlatform);
    matmul_tiling::MatmulApiTiling bmm2(ascendcPlatform);

    matmul_tiling::DataType qType;
    matmul_tiling::DataType kvType;

    uint32_t baseN = 512;  // antiquant to split K;
    uint32_t singleM;
    if (isPFAFlag_) {
        singleM = sOuterFactor_;
    } else {
        singleM = fiaInfo.gSize;
    }
    bmm1.SetShape(singleM, sInnerFactor_, fiaInfo.qkHeadDim);
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, kvType, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, kvType, true);
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, qType, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, qType, true);
    bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN, matmul_tiling::DataType::DT_FLOAT);
    bmm1.SetOrgShape(singleM, sInnerFactor_, fiaInfo.qkHeadDim, headDimAlign_);
    bmm1.SetBias(false);

    uint32_t bmm1BaseN = std::min(AlignUp(sInnerFactor_, NUM_16), baseN);
    // 向下对齐保证M*N不超过L0C，且由于bmm1BaseN有最大限制，L0C_SIZE / sizeof(float) / bmm1BaseN不会小于16
    uint32_t bmm1MaxBaseM =
        AlignUp(static_cast<uint32_t>(platformInfo_.l0cSize / sizeof(float) / bmm1BaseN) - NUM_16, NUM_16);

    OP_CHECK_IF((bmm1.SetFixSplit(std::min(AlignUp(singleM, NUM_16), bmm1MaxBaseM), AlignUp(bmm1BaseN, NUM_16)) == -1),
                OP_LOGE(fiaInfo.opName, "Bmm1 SetFixSplit fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((bmm1.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTN) == -1),
                OP_LOGE(fiaInfo.opName, "Bmm1 SetTraverse fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((bmm1.GetTiling(ifaTilingData_.bmm1TilingData) == -1), OP_LOGE(fiaInfo.opName, "Bmm1 get tiling fail."),
                return ge::GRAPH_FAILED);

    bmm2.SetShape(singleM, fiaInfo.qkHeadDim, sInnerFactor_);
    bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, qType, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, qType, false);
    bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN, matmul_tiling::DataType::DT_FLOAT);
    bmm2.SetOrgShape(singleM, headDimAlign_, sInnerSizeAlign_, sInnerFactor_);
    // 存在输入query是BNSD格式，但使能PA，需要按BSH SetOrgShape
    bmm2.SetBias(false);
    OP_CHECK_IF((bmm2.SetFixSplit(std::min(AlignUp(singleM, NUM_16), NUM_128)) == -1),
                OP_LOGE(fiaInfo.opName, "Bmm2 SetFixSplit fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((bmm2.GetTiling(ifaTilingData_.bmm2TilingData) == -1), OP_LOGE(fiaInfo.opName, "Bmm2 get tiling fail."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

template <typename K, typename V>
static inline V GetMapValueOrDefault(const std::map<K, V> &map, const K &key, V defaultVal = V())
{
    auto it = map.find(key);
    return (it != map.end()) ? it->second : defaultVal;
}

static inline int64_t ReadFirstTensorValue(const gert::Tensor *tensor)
{
    if (tensor == nullptr || tensor->GetShapeSize() < 1) {
        return 0;
    }
    const int64_t *value = tensor->GetData<int64_t>();
    return (value != nullptr) ? value[0] : 0;
}

static void SetSparseCompressMode(const FiaTilingInfo &fiaInfo,
                                  InputParamsRegbase &inputParams)
{
    static const std::map<uint32_t, uint8_t> sparseCompressModeMap = {{SPARSE_MODE_NO_MASK, 0},
                                                                      {SPARSE_MODE_ALL_MASK, 0},
                                                                      {SPARSE_MODE_LEFT_UP, 1},
                                                                      {SPARSE_MODE_RIGHT_DOWN, 2},
                                                                      {SPARSE_MODE_BAND, 3}};
    inputParams.set_attenMaskCompressMode(
        GetMapValueOrDefault(sparseCompressModeMap, static_cast<uint32_t>(fiaInfo.sparseMode)));
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::ComputeTilingData(const FiaTilingInfo &fiaInfo)
{
    auto &inputParams = faRunTilingAdapter_.inputParamsRegbase;
    auto &baseParams = pfaTilingData_.promptAttentionBaseParams;

    if (SetMaskTilingData(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SetSparseCompressMode(fiaInfo, inputParams);
    inputParams.set_sparseType(ComputeSparseType(fiaInfo));
    SetAlibiStartIdx(fiaInfo);
    SetLayoutType(fiaInfo);
    SetPaLayoutTilingData(fiaInfo);
    SetTransposeLayout(fiaInfo);
    SetAntiQuantInfo(fiaInfo);
    ComputeNeedInit(fiaInfo);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SetMaskTilingData(const FiaTilingInfo &fiaInfo)
{
    auto &inputParams = faRunTilingAdapter_.inputParamsRegbase;
    auto &baseParams = pfaTilingData_.promptAttentionBaseParams;
    auto &singleCoreParams = pfaTilingData_.promptAttentionSingleCoreParams;

    if (!fiaInfo.attenMaskFlag) {
        inputParams.set_attenMaskS1Size(0);
        inputParams.set_attenMaskS2Size(0);
        inputParams.set_attenMaskShapeType(0);
        baseParams.set_maskQsSize(0);
        baseParams.set_maskKVsSize(0);
        return ge::GRAPH_SUCCESS;
    }

    auto attenMaskTensor = fiaInfo.opParamInfo.attenMask.tensor;
    if (attenMaskTensor == nullptr) {
        OP_LOGE(fiaInfo.opName, "attenMaskFlag is true but attenMask tensor is null.");
        return ge::GRAPH_FAILED;
    }
    const auto &shape = attenMaskTensor->GetStorageShape();
    uint64_t maskDimNum = shape.GetDimNum();
    uint64_t maskBatch = (maskDimNum != 2 || fiaInfo.s1Size == 1) ? shape.GetDim(0) : 1;
    uint64_t maskS1Size = shape.GetDim(maskDimNum - 2);
    uint64_t maskS2Size = shape.GetDim(maskDimNum - 1);
    inputParams.set_attenMaskShapeType(maskBatch > 1 ? 1 : 2);
    singleCoreParams.set_attenMaskBatch(maskBatch);
    inputParams.set_attenMaskS1Size(maskS1Size);
    inputParams.set_attenMaskS2Size(maskS2Size);
    baseParams.set_maskQsSize(maskS1Size);
    baseParams.set_maskKVsSize(maskS2Size);
    return ge::GRAPH_SUCCESS;
}


uint8_t FusedInferAttentionScoreTilingImpl::ComputeSparseType(const FiaTilingInfo &fiaInfo) const
{
    if (fiaInfo.quantMode == FiaQuantMode::ANTI_QUANT && !isPFAFlag_) {
        return 0;
    }

    switch (fiaInfo.sparseMode) {
        case SPARSE_MODE_NO_MASK:
            if (fiaInfo.preToken >= fiaInfo.s1Size && fiaInfo.nextToken == 0) {
                return 3;
            }
            if (fiaInfo.preToken >= fiaInfo.s1Size &&
                fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION &&
                fiaInfo.nextToken >= fiaInfo.s2Size) {
                return 0;
            }
            return 4;
        case SPARSE_MODE_ALL_MASK:
            return 0;
        case SPARSE_MODE_LEFT_UP:
            return 3;
        case SPARSE_MODE_RIGHT_DOWN:
            return (fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION &&
                    fiaInfo.s1Size == fiaInfo.s2Size) ? 3 : 4;
        case SPARSE_MODE_BAND:
            return 4;
        default:
            return 0;
    }
}

void FusedInferAttentionScoreTilingImpl::SetAlibiStartIdx(const FiaTilingInfo &fiaInfo)
{
    auto &inputParams = faRunTilingAdapter_.inputParamsRegbase;
    inputParams.set_qStartIdx(ReadFirstTensorValue(fiaInfo.opParamInfo.qStartIdx.tensor));
    inputParams.set_kvStartIdx(ReadFirstTensorValue(fiaInfo.opParamInfo.kvStartIdx.tensor));
}

void FusedInferAttentionScoreTilingImpl::SetLayoutType(const FiaTilingInfo &fiaInfo)
{
    static const std::map<FiaLayout, uint8_t> layoutTypeMap = {
        {FiaLayout::BSH, 1}, {FiaLayout::TND, 4}, {FiaLayout::BSND, 1},
        {FiaLayout::BNSD, 3}, {FiaLayout::NTD, 5},
    };
    auto &inputParams = faRunTilingAdapter_.inputParamsRegbase;
    auto &baseParams = pfaTilingData_.promptAttentionBaseParams;
    uint8_t layoutType = GetMapValueOrDefault(layoutTypeMap, fiaInfo.qLayout);
    inputParams.set_layoutType(layoutType);
    baseParams.set_layoutType(layoutType);
}

void FusedInferAttentionScoreTilingImpl::SetPaLayoutTilingData(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION) {
        return;
    }

    auto &inputParams = faRunTilingAdapter_.inputParamsRegbase;
    auto &baseParams = pfaTilingData_.promptAttentionBaseParams;

    if (fiaInfo.antiQuantFlag) {
        uint32_t dimNum = fiaInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum();
        uint8_t paType = (dimNum == 3) ? 0 : (dimNum == 4) ? 1 : (dimNum == 5) ? 2 : 0;
        inputParams.set_paLayoutType(paType);
        baseParams.set_PAlayoutType(paType);
        return;
    }

    if (fiaInfo.kvLayout == FiaLayout::BnBsH) {
        inputParams.set_paLayoutType(1);
        baseParams.set_PAlayoutType(1);
    } else if (fiaInfo.kvLayout == FiaLayout::BnNBsD) {
        inputParams.set_paLayoutType(0);
        baseParams.set_PAlayoutType(0);
    } else if (fiaInfo.kvLayout == FiaLayout::NZ) {
        inputParams.set_paLayoutType(2);
        baseParams.set_PAlayoutType(2);
    }
}

void FusedInferAttentionScoreTilingImpl::SetTransposeLayout(const FiaTilingInfo &fiaInfo)
{
    static const std::map<std::string, uint32_t> transposeLayoutMap = {
        {"BNSD_BSND", 1}, {"BSND_BNSD", 2}, {"BSH_BNSD", 3}, {"BNSD_NBSD", 4},
        {"BSND_NBSD", 5}, {"BSH_NBSD", 6}, {"NTD_TND", 7},   {"TND_NTD", 8},
    };
    auto &inputParams = faRunTilingAdapter_.inputParamsRegbase;
    auto &baseParams = pfaTilingData_.promptAttentionBaseParams;
    uint32_t transposeType = GetMapValueOrDefault(transposeLayoutMap, std::string(fiaInfo.opParamInfo.layOut));
    inputParams.set_transposeLayout(transposeType);
    baseParams.set_transposeLayout(transposeType);
}

void FusedInferAttentionScoreTilingImpl::SetAntiQuantInfo(const FiaTilingInfo &fiaInfo)
{
    auto &inputParams = faRunTilingAdapter_.inputParamsRegbase;

    if (fiaInfo.quantMode != FiaQuantMode::ANTI_QUANT) {
        inputParams.set_antiquantPerTensorFlag(0);
        return;
    }

    auto scaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    uint32_t scaleDimNum = scaleTensor->GetStorageShape().GetDimNum();
    uint32_t scaleShape = scaleTensor->GetStorageShape().GetDim(0);
    bool perTensor = (scaleDimNum == 1 && scaleShape == 1);
    inputParams.set_antiquantPerTensorFlag(perTensor ? 1 : 0);
}

void FusedInferAttentionScoreTilingImpl::ComputeNeedInit(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.quantMode == FiaQuantMode::ANTI_QUANT) {
        return;
    }

    for (uint32_t i = 0; i < fiaInfo.bSize; i++) {
        int64_t preTokensPerbatch = fiaInfo.preToken;
        int64_t nextTokensPerbatch = fiaInfo.nextToken;

        if (fiaInfo.sparseMode == SPARSE_MODE_RIGHT_DOWN) {
            preTokensPerbatch = SPARSE_MODE_INT_MAX;
            nextTokensPerbatch = actualSeqLengthsKV_[i] + fiaInfo.systemPrefixLen;
            if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) {
                nextTokensPerbatch -= actualSeqLengthsQ_[i] / fiaInfo.gSize;
            } else {
                nextTokensPerbatch -= actualSeqLengthsQ_[i];
            }
        } else if (fiaInfo.sparseMode == SPARSE_MODE_BAND) {
            int64_t kvLen = actualSeqLengthsKV_[i] + fiaInfo.systemPrefixLen;
            preTokensPerbatch = fiaInfo.preToken - kvLen + actualSeqLengthsQ_[i];
            nextTokensPerbatch = fiaInfo.nextToken + kvLen - actualSeqLengthsQ_[i];
        }

        if ((nextTokensPerbatch < 0) ||
            (actualSeqLengthsQ_[i] > (actualSeqLengthsKV_[i] + fiaInfo.systemPrefixLen + preTokensPerbatch))) {
            needInit_ = true;
        }
        OP_LOGI(fiaInfo.opName, "preTokensPerbatch[%u] is %ld, nextTokensPerbatch[%u] is %ld",
                i, preTokensPerbatch, i, nextTokensPerbatch);
        OP_LOGI(fiaInfo.opName,
                "actualSeqLengths[%u] is %ld, actualSeqLengthsKV[%u] is %ld, "
                "actualSharedPrefixLen is %ld, needInit is %u.",
                i, actualSeqLengthsQ_[i], i, actualSeqLengthsKV_[i], fiaInfo.systemPrefixLen, needInit_);
    }
}

ge::graphStatus FusedInferAttentionScoreTilingImpl::SetFATilingData(const FiaTilingInfo &fiaInfo)
{
    auto &inputParams = faRunTilingAdapter_.inputParamsRegbase;
    inputParams.set_bSize(fiaInfo.bSize);
    inputParams.set_t1Size(fiaInfo.qTSize);
    inputParams.set_t2Size(fiaInfo.kTSize);
    inputParams.set_n2Size(fiaInfo.n2Size);
    inputParams.set_gSize(fiaInfo.gSize);
    inputParams.set_s1Size(fiaInfo.s1Size);
    inputParams.set_s2Size(fiaInfo.s2Size);
    inputParams.set_dSize(fiaInfo.qkHeadDim);
    inputParams.set_dSizeV(fiaInfo.vHeadDim);
    // Some 9.1 opbase paths report 3D PA inputs as TensorV1 even though the
    // dynamic-input stride is available.  Prefer the stride supplied by the
    // context whenever it exists; otherwise retain the TensorV1 defaults.
    if (fiaInfo.keyStrides != nullptr && fiaInfo.valueStrides != nullptr) {
        inputParams.keyStrides.set_bnStride(fiaInfo.keyStrides->GetStride(0));
        inputParams.keyStrides.set_n2Stride(fiaInfo.keyStrides->GetStride(1));
        inputParams.valueStrides.set_bnStride(fiaInfo.valueStrides->GetStride(0));
        inputParams.valueStrides.set_n2Stride(fiaInfo.valueStrides->GetStride(1));
        if (fiaInfo.kRopeStrides != nullptr) {
            inputParams.kRopeStrides.set_bnStride(fiaInfo.kRopeStrides->GetStride(0));
            inputParams.kRopeStrides.set_n2Stride(fiaInfo.kRopeStrides->GetStride(1));
        }
    }
    inputParams.set_dSizeRope(DSIZE_64);
    inputParams.set_scaleValue(fiaInfo.scaleValue);
    inputParams.set_preTokens(fiaInfo.preToken);
    inputParams.set_nextTokens(fiaInfo.nextToken);
    inputParams.set_pseShapeType(fiaInfo.enableAlibiPse ? 3 : 0);
    inputParams.set_pseS1Size(fiaInfo.pseShiftS1);
    inputParams.set_pseS2Size(fiaInfo.pseShiftS2);
    inputParams.set_pseBSize(fiaInfo.pseShiftByBatch);
    inputParams.set_implMode(0);
    inputParams.set_needDropMaskOp(0);
    inputParams.set_dropMaskOuter(0);
    inputParams.set_pseEncodeType(0);
    inputParams.set_remain(0);
    inputParams.set_rsv1(0);
    inputParams.set_seed(0);
    SetFATilingDataInputParams(fiaInfo);
    SetFATilingDataInitOutput(fiaInfo);
    return ge::GRAPH_SUCCESS;
}

void FusedInferAttentionScoreTilingImpl::SetFATilingDataInputParams(const FiaTilingInfo &fiaInfo)
{
    auto &inputParams = faRunTilingAdapter_.inputParamsRegbase;
    inputParams.set_offset(0);
    inputParams.set_keepProbUint8(0);
    inputParams.set_pseType(*fiaInfo.opParamInfo.pseType);
    inputParams.set_deqScaleFlag(0);   // per-tensor全量化场景才会用到
    inputParams.set_deqScale2Flag(0);  // per-tensor全量化场景才会用到
    inputParams.set_actualSeqLengthsSize(fiaInfo.actualLenQDims);
    inputParams.set_actualSeqLengthsKVSize(fiaInfo.actualLenKvDims);
    inputParams.set_isKvContinuous(fiaInfo.kvStorageMode != KvStorageMode::TENSOR_LIST);
    inputParams.set_fromFused(!fromPFA_);
    inputParams.set_isBSNDOut(fiaInfo.qLayout == FiaLayout::BNSD && fiaInfo.outLayout == FiaLayout::BSND);
    if (fiaInfo.antiQuantFlag) {
        inputParams.set_isGqa(gsMergeFlag_);
    } else {
        inputParams.set_isGqa(gsMergeFlag_ && fiaInfo.mlaMode != MlaMode::ROPE_SPLIT_D512);
    }
    inputParams.set_isSoftMaxLseEnable(fiaInfo.softmaxLseFlag);
    inputParams.set_isQHasLeftPadding(fiaInfo.qPaddingSizeFlag);
    inputParams.set_isKVHasLeftPadding(fiaInfo.kvPaddingSizeFlag);
    inputParams.set_ropeHeadSize(fiaInfo.ropeHeadDim);
    inputParams.set_prefixSeqInnerSize(fiaInfo.systemPrefixMaxLen);
    inputParams.set_headNumRatio(gsMergeFlag_ ? 1 : fiaInfo.gSize);
    inputParams.set_blockSize(fiaInfo.blockSize);
    uint32_t blockTableDim2 = 0;
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        blockTableDim2 = fiaInfo.opParamInfo.blockTable.tensor->GetStorageShape().GetDim(1);
    }
    inputParams.set_blockTableDim2(blockTableDim2);
    inputParams.set_paBlockNumSum(fiaInfo.totalBlockNum);
    inputParams.set_isRowInvalid(((static_cast<uint32_t>(fiaInfo.innerPrecise) >> 1) & 1) || isRowInvalid_);
    inputParams.set_isPostQuantPerChnl(fiaInfo.isOutQuantPerChnOut);
    inputParams.set_isPostQuantBF16(fiaInfo.isOutQuantTypeBf16);
    inputParams.set_antiquantParaSeqSize(fiaInfo.antiqSeqSize);
    inputParams.set_antiquantPerHeadFlag(fiaInfo.keyAntiquantMode == 2 || fiaInfo.keyAntiquantMode == 3 ||
                                         fiaInfo.keyAntiquantMode == 5);
    inputParams.set_antiquantPageAttentionFlag(fiaInfo.keyAntiquantMode == 4 || fiaInfo.keyAntiquantMode == 5);
    inputParams.set_isActualSeqLengthsNull(!actualSeqLenQFlag_);
    inputParams.set_isActualSeqLengthsKVNull(!actualSeqLenKVFlag_);
    inputParams.set_isActualSharedPrefixLenNull(!actualSharedPrefixLenFlag_);
    inputParams.set_keepProb(0);           // 默认值, 未使用
    inputParams.set_bandIndex(0);          // 默认值，未使用
    inputParams.set_attenMaskDataType(1);  // 默认值，未使用
    inputParams.set_alignedS2(0);          // 默认值, 未使用
    inputParams.set_s1SparseValidSize(0);  // 默认值，未使用
    inputParams.set_s2SparseValidSize(0);  // 默认值，未使用
    inputParams.set_pseAlibiBaseS1(0);     // 默认值，未使用
    inputParams.set_pseAlibiBaseS2(0);     // 默认值，未使用
}

void FusedInferAttentionScoreTilingImpl::SetFATilingDataInitOutput(const FiaTilingInfo &fiaInfo)
{
    auto &initOutputParams = faRunTilingAdapter_.initOutputParams;
    int64_t outSize = fiaInfo.opParamInfo.attenOut.shape->GetStorageShape().GetShapeSize();
    int64_t lseSize = fiaInfo.softmaxLseFlag ? fiaInfo.opParamInfo.lseOut.shape->GetStorageShape().GetShapeSize() : 0;
    uint32_t singleCoreSize = (outSize + platformInfo_.aivNum - 1) / (platformInfo_.aivNum);
    if (fiaInfo.isOutQuantEnable) {
        singleCoreSize = ((singleCoreSize + 1) / 2) * 2;
    }
    if (fiaInfo.antiQuantFlag && !(fiaInfo.needInit || needInit_)) {
        initOutputParams.set_singleCoreSize(0);
    } else {
        initOutputParams.set_singleCoreSize(singleCoreSize);
    }
    initOutputParams.set_totalOutputSize(outSize);
    initOutputParams.set_totalSoftMaxLseOutputSize(lseSize);
    initOutputParams.set_needInit(fiaInfo.needInit || needInit_);
    initOutputParams.set_isOneN(0);  // 默认值,当前未使用
}


ge::graphStatus FusedInferAttentionScoreTilingImpl::SetTilingData(gert::TilingContext *context,
                                                                  const FiaTilingInfo &fiaInfo)
{
    if (!fiaInfo.emptyTensorFlag) {
        if (fiaInfo.quantMode == FiaQuantMode::ANTI_QUANT) {
            SetDequantMMTilingData(context, fiaInfo);
        }
        SetFATilingData(fiaInfo);
    }

    int64_t cap = context->GetRawTilingData()->GetCapacity();
    OP_LOGI(fiaInfo.opName, "Tiling Data context GetCapacity: %lu.", cap);

    PrintAllTilingData(fiaInfo);
    FlashAttentionScoreSimplifiedTilingData *tiling =
        context->GetTilingData<FlashAttentionScoreSimplifiedTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE(fiaInfo.opName, "The tiling data is nullptr"), return ge::GRAPH_FAILED);
    *tiling = faRunTilingAdapter_;
    return ge::GRAPH_SUCCESS;
}

void FusedInferAttentionScoreTilingImpl::PrintAllTilingData(const FiaTilingInfo &fiaInfo)
{
    OP_LOGD(fiaInfo.opName, "bSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_bSize());
    OP_LOGD(fiaInfo.opName, "t1Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_t1Size());
    OP_LOGD(fiaInfo.opName, "t2Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_t2Size());
    OP_LOGD(fiaInfo.opName, "n2Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_n2Size());
    OP_LOGD(fiaInfo.opName, "gSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_gSize());
    OP_LOGD(fiaInfo.opName, "s1Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_s1Size());
    OP_LOGD(fiaInfo.opName, "s2Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_s2Size());
    OP_LOGD(fiaInfo.opName, "alignedS2:%d", faRunTilingAdapter_.inputParamsRegbase.get_alignedS2());
    OP_LOGD(fiaInfo.opName, "dSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_dSize());
    OP_LOGD(fiaInfo.opName, "dSizeV:%d", faRunTilingAdapter_.inputParamsRegbase.get_dSizeV());
    OP_LOGD(fiaInfo.opName, "dSizeRope:%d", faRunTilingAdapter_.inputParamsRegbase.get_dSizeRope());
    OP_LOGD(fiaInfo.opName, "keepProb:%d", faRunTilingAdapter_.inputParamsRegbase.get_keepProb());
    OP_LOGD(fiaInfo.opName, "scaleValue:%d", faRunTilingAdapter_.inputParamsRegbase.get_scaleValue());
    OP_LOGD(fiaInfo.opName, "preTokens:%d", faRunTilingAdapter_.inputParamsRegbase.get_preTokens());
    OP_LOGD(fiaInfo.opName, "nextTokens:%d", faRunTilingAdapter_.inputParamsRegbase.get_nextTokens());
    OP_LOGD(fiaInfo.opName, "pseS1Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_pseS1Size());
    OP_LOGD(fiaInfo.opName, "pseS2Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_pseS2Size());
    OP_LOGD(fiaInfo.opName, "pseBSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_pseBSize());
    OP_LOGD(fiaInfo.opName, "bandIndex:%d", faRunTilingAdapter_.inputParamsRegbase.get_bandIndex());
    OP_LOGD(fiaInfo.opName, "layoutType:%d", faRunTilingAdapter_.inputParamsRegbase.get_layoutType());
    OP_LOGD(fiaInfo.opName, "pseShapeType:%d", faRunTilingAdapter_.inputParamsRegbase.get_pseShapeType());
    OP_LOGD(fiaInfo.opName, "attenMaskShapeType:%d", faRunTilingAdapter_.inputParamsRegbase.get_attenMaskShapeType());
    OP_LOGD(fiaInfo.opName, "attenMaskDataType:%d", faRunTilingAdapter_.inputParamsRegbase.get_attenMaskDataType());
    OP_LOGD(fiaInfo.opName, "attenMaskCompressMode:%d", faRunTilingAdapter_.inputParamsRegbase.get_attenMaskCompressMode());
    OP_LOGD(fiaInfo.opName, "implMode:%d", faRunTilingAdapter_.inputParamsRegbase.get_implMode());
    OP_LOGD(fiaInfo.opName, "sparseType:%d", faRunTilingAdapter_.inputParamsRegbase.get_sparseType());
    OP_LOGD(fiaInfo.opName, "needDropMaskOp:%d", faRunTilingAdapter_.inputParamsRegbase.get_needDropMaskOp());
    OP_LOGD(fiaInfo.opName, "dropMaskOuter:%d", faRunTilingAdapter_.inputParamsRegbase.get_dropMaskOuter());
    OP_LOGD(fiaInfo.opName, "pseEncodeType:%d", faRunTilingAdapter_.inputParamsRegbase.get_pseEncodeType());
    OP_LOGD(fiaInfo.opName, "remain:%d", faRunTilingAdapter_.inputParamsRegbase.get_remain());
    OP_LOGD(fiaInfo.opName, "attenMaskS2Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_attenMaskS2Size());
    OP_LOGD(fiaInfo.opName, "pseType:%d", faRunTilingAdapter_.inputParamsRegbase.get_pseType());
    OP_LOGD(fiaInfo.opName, "rsv1:%d", faRunTilingAdapter_.inputParamsRegbase.get_rsv1());
    OP_LOGD(fiaInfo.opName, "qStartIdx:%d", faRunTilingAdapter_.inputParamsRegbase.get_qStartIdx());
    OP_LOGD(fiaInfo.opName, "kvStartIdx:%d", faRunTilingAdapter_.inputParamsRegbase.get_kvStartIdx());
    OP_LOGD(fiaInfo.opName, "s1SparseValidSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_s1SparseValidSize());
    OP_LOGD(fiaInfo.opName, "s2SparseValidSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_s2SparseValidSize());
    OP_LOGD(fiaInfo.opName, "seed:%d", faRunTilingAdapter_.inputParamsRegbase.get_seed());
    OP_LOGD(fiaInfo.opName, "offset:%d", faRunTilingAdapter_.inputParamsRegbase.get_offset());
    OP_LOGD(fiaInfo.opName, "keepProbUint8:%d", faRunTilingAdapter_.inputParamsRegbase.get_keepProbUint8());
    OP_LOGD(fiaInfo.opName, "pseAlibiBaseS1:%d", faRunTilingAdapter_.inputParamsRegbase.get_pseAlibiBaseS1());
    OP_LOGD(fiaInfo.opName, "pseAlibiBaseS2:%d", faRunTilingAdapter_.inputParamsRegbase.get_pseAlibiBaseS2());
    OP_LOGD(fiaInfo.opName, "deqScaleFlag:%d", faRunTilingAdapter_.inputParamsRegbase.get_deqScaleFlag());
    OP_LOGD(fiaInfo.opName, "deqScale2Flag:%d", faRunTilingAdapter_.inputParamsRegbase.get_deqScale2Flag());
    OP_LOGD(fiaInfo.opName, "isActualSeqLengthsNull:%d", faRunTilingAdapter_.inputParamsRegbase.get_isActualSeqLengthsNull());
    OP_LOGD(fiaInfo.opName, "isActualSeqLengthsKVNull:%d", faRunTilingAdapter_.inputParamsRegbase.get_isActualSeqLengthsKVNull());
    PrintInputParams(fiaInfo);
}

void FusedInferAttentionScoreTilingImpl::PrintInputParams(const FiaTilingInfo &fiaInfo)
{
    OP_LOGD(fiaInfo.opName, "actualSeqLengthsSize:%d",
            faRunTilingAdapter_.inputParamsRegbase.get_actualSeqLengthsSize());
    OP_LOGD(fiaInfo.opName, "actualSeqLengthsKVSize:%d",
            faRunTilingAdapter_.inputParamsRegbase.get_actualSeqLengthsKVSize());
    OP_LOGD(fiaInfo.opName, "isKvContinuous:%d", faRunTilingAdapter_.inputParamsRegbase.get_isKvContinuous());
    OP_LOGD(fiaInfo.opName, "fromFused:%d", faRunTilingAdapter_.inputParamsRegbase.get_fromFused());
    OP_LOGD(fiaInfo.opName, "isBSNDOut:%d", faRunTilingAdapter_.inputParamsRegbase.get_isBSNDOut());
    OP_LOGD(fiaInfo.opName, "transposeLayout:%d", faRunTilingAdapter_.inputParamsRegbase.get_transposeLayout());
    OP_LOGD(fiaInfo.opName, "isGqa:%d", faRunTilingAdapter_.inputParamsRegbase.get_isGqa());
    OP_LOGD(fiaInfo.opName, "isSoftMaxLseEnable:%d", faRunTilingAdapter_.inputParamsRegbase.get_isSoftMaxLseEnable());
    OP_LOGD(fiaInfo.opName, "isActualSharedPrefixLenNull:%d",
            faRunTilingAdapter_.inputParamsRegbase.get_isActualSharedPrefixLenNull());
    OP_LOGD(fiaInfo.opName, "isQHasLeftPadding:%d", faRunTilingAdapter_.inputParamsRegbase.get_isQHasLeftPadding());
    OP_LOGD(fiaInfo.opName, "isKVHasLeftPadding:%d", faRunTilingAdapter_.inputParamsRegbase.get_isKVHasLeftPadding());
    OP_LOGD(fiaInfo.opName, "prefixSeqInnerSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_prefixSeqInnerSize());
    OP_LOGD(fiaInfo.opName, "headNumRatio:%d", faRunTilingAdapter_.inputParamsRegbase.get_headNumRatio());
    OP_LOGD(fiaInfo.opName, "blockSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_blockSize());
    OP_LOGD(fiaInfo.opName, "blockTableDim2:%d", faRunTilingAdapter_.inputParamsRegbase.get_blockTableDim2());
    OP_LOGD(fiaInfo.opName, "paBlockNumSum:%d", faRunTilingAdapter_.inputParamsRegbase.get_paBlockNumSum());
    OP_LOGD(fiaInfo.opName, "paLayoutType:%d", faRunTilingAdapter_.inputParamsRegbase.get_paLayoutType());
    OP_LOGD(fiaInfo.opName, "attenMaskS1Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_attenMaskS1Size());
    OP_LOGD(fiaInfo.opName, "isRowInvalid:%d", faRunTilingAdapter_.inputParamsRegbase.get_isRowInvalid());
    OP_LOGD(fiaInfo.opName, "kvSplitPart:%d", faRunTilingAdapter_.inputParamsRegbase.get_kvSplitPart());
    OP_LOGD(fiaInfo.opName, "accumOutSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_accumOutSize());
    OP_LOGD(fiaInfo.opName, "logSumExpSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_logSumExpSize());
    OP_LOGD(fiaInfo.opName, "isPostQuantPerChnl:%d", faRunTilingAdapter_.inputParamsRegbase.get_isPostQuantPerChnl());
    OP_LOGD(fiaInfo.opName, "isPostQuantBF16:%d", faRunTilingAdapter_.inputParamsRegbase.get_isPostQuantBF16());
    OP_LOGD(fiaInfo.opName, "antiquantPerTensorFlag:%d",
            faRunTilingAdapter_.inputParamsRegbase.get_antiquantPerTensorFlag());
    OP_LOGD(fiaInfo.opName, "antiquantPerHeadFlag:%d",
            faRunTilingAdapter_.inputParamsRegbase.get_antiquantPerHeadFlag());
    OP_LOGD(fiaInfo.opName, "antiquantParaSeqSize:%d",
            faRunTilingAdapter_.inputParamsRegbase.get_antiquantParaSeqSize());

    OP_LOGD(fiaInfo.opName, "coreNum:%d", faRunTilingAdapter_.multiCoreParamsRegbase.get_coreNum());
    OP_LOGD(fiaInfo.opName, "totalSize:%d", faRunTilingAdapter_.multiCoreParamsRegbase.get_totalSize());
    OP_LOGD(fiaInfo.opName, "s1OuterSize:%d", faRunTilingAdapter_.multiCoreParamsRegbase.get_s1OuterSize());
    OP_LOGD(fiaInfo.opName, "splitFactorSize:%d", faRunTilingAdapter_.multiCoreParamsRegbase.get_splitFactorSize());
    OP_LOGD(fiaInfo.opName, "splitFactorTailSize:%d",
            faRunTilingAdapter_.multiCoreParamsRegbase.get_splitFactorTailSize());
    for (uint32_t i = 0; i < 48; i++) { // 48 cores
        OP_LOGD(fiaInfo.opName, "bnStartIdx[%d]:%d", i,
                faRunTilingAdapter_.multiCoreParamsRegbase.get_bnStartIdxPtr()[i]);
    }
    for (uint32_t i = 0; i < 48; i++) { // 48 cores
        OP_LOGD(fiaInfo.opName, "sparseStartIdx[%d]:%d", i,
                faRunTilingAdapter_.multiCoreParamsRegbase.get_sparseStartIdxPtr()[i]);
    }
    OP_LOGD(fiaInfo.opName, "firstFullLoadS1OuterIdx:%d",
            faRunTilingAdapter_.multiCoreParamsRegbase.get_firstFullLoadS1OuterIdx());
    OP_LOGD(fiaInfo.opName, "splitCoreMode:%d", faRunTilingAdapter_.multiCoreParamsRegbase.get_splitCoreMode());

    OP_LOGD(fiaInfo.opName, "singleCoreSize:%d", faRunTilingAdapter_.initOutputParams.get_singleCoreSize());
    OP_LOGD(fiaInfo.opName, "needInit:%d", faRunTilingAdapter_.initOutputParams.get_needInit());
    OP_LOGD(fiaInfo.opName, "isOneN:%d", faRunTilingAdapter_.initOutputParams.get_isOneN());
    OP_LOGD(fiaInfo.opName, "totalOutputSize:%d", faRunTilingAdapter_.initOutputParams.get_totalOutputSize());
    OP_LOGD(fiaInfo.opName, "totalSoftMaxLseOutputSize:%d",
            faRunTilingAdapter_.initOutputParams.get_totalSoftMaxLseOutputSize());
}
ge::graphStatus FusedInferAttentionScoreTilingImpl::DoOpTiling(gert::TilingContext *context,
                                                               const FiaTilingInfo &fiaInfo)
{
    OP_CHECK_IF(SetPlatMemoryInfo(context, fiaInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(fiaInfo.opName, "Set plat memory info fail."), return ge::GRAPH_FAILED);

    if (fiaInfo.emptyTensorFlag) {
        OP_CHECK_IF(SetEmptyTensor(context, fiaInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(fiaInfo.opName, "Set emptyt ensor fail."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    InitImplParam(fiaInfo);

    OP_CHECK_IF(SplitPolicy(context, fiaInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(fiaInfo.opName, "Excute split policy fail."), return ge::GRAPH_FAILED);

    if (fiaInfo.quantMode == FiaQuantMode::NO_QUANT && fiaInfo.mlaMode != MlaMode::ROPE_SPLIT_D512 &&
        gsMergeFlag_ && !dnFlag_ && (((fiaInfo.s1Size * fiaInfo.gSize + 1) / 2) + fiaInfo.gSize - 1) /
        fiaInfo.gSize * fiaInfo.gSize > sOuterFactor_ && fiaInfo.s1Size > 1) {
        pfaMergeFlag_ = false;
        gsMergeFlag_ = false;
    }

    OP_CHECK_IF(ComputeTilingData(fiaInfo) != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo.opName, "Compute tilingData fail."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GenTilingKey(context, fiaInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(fiaInfo.opName, "Generate tilingKey fail."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(SetBlockDim(context, fiaInfo) != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo.opName, "Set blockDim fail."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspace(context, fiaInfo) != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo.opName, "Get workspace fail."),
                return ge::GRAPH_FAILED);
    // 使用SyncAll，需要设置为batchmode模式，所有核同时启动，否则多流方式下执行可能会卡死
    constexpr uint32_t BATCH_MODE_SCHEDULE = 1;
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    OP_CHECK_IF(SetTilingData(context, fiaInfo) != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo.opName, "Set tiling data fail."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
