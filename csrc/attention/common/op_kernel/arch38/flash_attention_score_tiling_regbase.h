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
 * \file flash_attention_score_tiling_regbase.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_TILING_REGBASE_H_
#define FLASH_ATTENTION_SCORE_GRAD_TILING_REGBASE_H_

#include <cstdint>

namespace optiling {
class FlashAttentionScoreEmptyInputTilingDataRegbase {
public:
    uint32_t coreNum;
    uint32_t attentionOutFormerNum;
    uint32_t attentionOutTailNum;
    uint32_t softmaxMaxFormerNum;
    uint32_t softmaxMaxTailNum;
    uint64_t attentionOutSingleCoreDataSize;
    uint64_t attentionOutTailCoreDataSize;
    uint64_t softmaxMaxSingleCoreDataSize;
    uint64_t softmaxMaxTailCoreDataSize;
    uint64_t attentionOutLastCoreDataSize;
    uint64_t attentionOutLastCoreIndex;

    uint32_t get_coreNum() const {return coreNum;}
    uint32_t get_attentionOutFormerNum() const {return attentionOutFormerNum;}
    uint32_t get_attentionOutTailNum() const {return attentionOutTailNum;}
    uint32_t get_softmaxMaxFormerNum() const {return softmaxMaxFormerNum;}
    uint32_t get_softmaxMaxTailNum() const {return softmaxMaxTailNum;}
    uint64_t get_attentionOutSingleCoreDataSize() const {return attentionOutSingleCoreDataSize;}
    uint64_t get_attentionOutTailCoreDataSize() const {return attentionOutTailCoreDataSize;}
    uint64_t get_softmaxMaxSingleCoreDataSize() const {return softmaxMaxSingleCoreDataSize;}
    uint64_t get_softmaxMaxTailCoreDataSize() const {return softmaxMaxTailCoreDataSize;}
    uint64_t get_attentionOutLastCoreDataSize() const {return attentionOutLastCoreDataSize;}
    uint64_t get_attentionOutLastCoreIndex() const {return attentionOutLastCoreIndex;}

    void set_coreNum(uint32_t coreNumParam) {this->coreNum = coreNumParam;}
    void set_attentionOutFormerNum(uint32_t attentionOutFormerNumParam) {this->attentionOutFormerNum = attentionOutFormerNumParam;}
    void set_attentionOutTailNum(uint32_t attentionOutTailNumParam) {this->attentionOutTailNum = attentionOutTailNumParam;}
    void set_softmaxMaxFormerNum(uint32_t softmaxMaxFormerNumParam) {this->softmaxMaxFormerNum = softmaxMaxFormerNumParam;}
    void set_softmaxMaxTailNum(uint32_t softmaxMaxTailNumParam) {this->softmaxMaxTailNum = softmaxMaxTailNumParam;}
    void set_attentionOutSingleCoreDataSize(uint64_t attentionOutSingleCoreDataSizeParam) {this->attentionOutSingleCoreDataSize = attentionOutSingleCoreDataSizeParam;}
    void set_attentionOutTailCoreDataSize(uint64_t attentionOutTailCoreDataSizeParam) {this->attentionOutTailCoreDataSize = attentionOutTailCoreDataSizeParam;}
    void set_softmaxMaxSingleCoreDataSize(uint64_t softmaxMaxSingleCoreDataSizeParam) {this->softmaxMaxSingleCoreDataSize = softmaxMaxSingleCoreDataSizeParam;}
    void set_softmaxMaxTailCoreDataSize(uint64_t softmaxMaxTailCoreDataSizeParam) {this->softmaxMaxTailCoreDataSize = softmaxMaxTailCoreDataSizeParam;}
    void set_attentionOutLastCoreDataSize(uint64_t attentionOutLastCoreDataSizeParam) {this->attentionOutLastCoreDataSize = attentionOutLastCoreDataSizeParam;}
    void set_attentionOutLastCoreIndex(uint64_t attentionOutLastCoreIndexParam) {this->attentionOutLastCoreIndex = attentionOutLastCoreIndexParam;}
};

class InputParamsRegbase {
public:
    int64_t bSize;
    int64_t n2Size;
    int64_t gSize;
    int64_t s1Size;
    int64_t s2Size;
    int64_t alignedS2;
    int64_t dSize;
    int64_t dSizeV;
    int64_t dSizeRope;
    float keepProb;
    float scaleValue;
    int64_t preTokens;
    int64_t nextTokens;
    int64_t pseS1Size;
    int64_t pseS2Size;
    uint32_t pseBSize;
    uint32_t bandIndex;
    uint8_t layoutType;  // 1: BSH/BSND, 2: SBH, 3: BNSD
    uint8_t pseShapeType;  // 0: (B,N2,G,S1,S2), 1: (B,N2,G,1,S2)
    uint8_t attenMaskShapeType;  // 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
    uint8_t attenMaskDataType;  // 0: fp16, 1: bool(uint8)
    uint8_t attenMaskCompressMode;  // ALL: 0, NONE: 1, ANY: 2, CAUSAL: 3, BAND: 4
    uint8_t implMode;  // 0: high precise, 1: high performance, 2: invalid line high precise
    uint8_t sparseType;
    uint8_t needDropMaskOp;
    uint8_t dropMaskOuter;
    uint8_t pseEncodeType;
    uint16_t remain;
    uint32_t attenMaskS2Size;
    uint32_t pseType;
    uint32_t rsv1;
    int64_t qStartIdx;
    int64_t kvStartIdx;
    int64_t s1SparseValidSize;
    int64_t s2SparseValidSize;
    int64_t seed;
    int64_t offset;
    int64_t keepProbUint8;
    int64_t pseAlibiBaseS1;
    int64_t pseAlibiBaseS2;

    // PFA
    uint8_t deqScaleFlag;  // 0: uint64  1: float32
    uint8_t deqScale2Flag;  // 0: uint64  1: float32
    uint8_t isActualSeqLengthsNull;
    uint8_t isActualSeqLengthsKVNull;
    uint32_t actualSeqLengthsSize;
    uint32_t actualSeqLengthsKVSize;
    uint8_t isKvContinuous;
    uint8_t fromFused;
    uint8_t isBSNDOut;
    uint8_t isGqa;
    uint8_t isSoftMaxLseEnable;
    uint8_t isActualSharedPrefixLenNull;
    uint8_t isQHasLeftPadding;
    uint8_t isKVHasLeftPadding;
    uint32_t ropeHeadSize;
    uint32_t prefixSeqInnerSize;
    uint32_t headNumRatio;
    int32_t blockSize;
    int32_t blockTableDim2;
    int32_t paBlockNumSum;
    int32_t attenMaskS1Size;
    uint32_t kvSplitPart;
    uint32_t accumOutSize;
    uint32_t logSumExpSize;
    uint8_t paLayoutType;
    uint8_t isRowInvalid;
    uint8_t isPostQuantPerChnl;
    uint8_t isPostQuantBF16;
    uint16_t antiquantPerTensorFlag;
    uint16_t antiquantPerHeadFlag;
    uint32_t antiquantParaSeqSize;

    int64_t get_bSize() const {return bSize;}
    void set_bSize(int64_t bSizeParam) {this->bSize = bSizeParam;}
    int64_t get_n2Size() const {return n2Size;}
    void set_n2Size(int64_t n2SizeParam) {this->n2Size = n2SizeParam;}
    int64_t get_gSize() const {return gSize;}
    void set_gSize(int64_t gSizeParam) {this->gSize = gSizeParam;}
    int64_t get_s1Size() const {return s1Size;}
    void set_s1Size(int64_t s1SizeParam) {this->s1Size = s1SizeParam;}
    int64_t get_s2Size() const {return s2Size;}
    void set_s2Size(int64_t s2SizeParam) {this->s2Size = s2SizeParam;}
    int64_t get_alignedS2() const {return alignedS2;}
    void set_alignedS2(int64_t alignedS2Param) {this->alignedS2 = alignedS2Param;}
    int64_t get_dSize() const {return dSize;}
    void set_dSize(int64_t dSizeParam) {this->dSize = dSizeParam;}
    int64_t get_dSizeV() const {return dSizeV;}
    void set_dSizeV(int64_t dSizeVParam) {this->dSizeV = dSizeVParam;}
    int64_t get_dSizeRope() const {return dSizeRope;}
    void set_dSizeRope(int64_t dSizeRopeParam) {this->dSizeRope = dSizeRopeParam;}
    float get_keepProb() const {return keepProb;}
    void set_keepProb(float keepProbParam) {this->keepProb = keepProbParam;}
    float get_scaleValue() const {return scaleValue;}
    void set_scaleValue(float scaleValueParam) {this->scaleValue = scaleValueParam;}
    int64_t get_preTokens() const {return preTokens;}
    void set_preTokens(int64_t preTokensParam) {this->preTokens = preTokensParam;}
    int64_t get_nextTokens() const {return nextTokens;}
    void set_nextTokens(int64_t nextTokensParam) {this->nextTokens = nextTokensParam;}
    int64_t get_pseS1Size() const {return pseS1Size;}
    void set_pseS1Size(int64_t pseS1SizeParam) {this->pseS1Size = pseS1SizeParam;}
    int64_t get_pseS2Size() const {return pseS2Size;}
    void set_pseS2Size(int64_t pseS2SizeParam) {this->pseS2Size = pseS2SizeParam;}
    uint32_t get_pseBSize() const {return pseBSize;}
    void set_pseBSize(uint32_t pseBSizeParam) {this->pseBSize = pseBSizeParam;}
    uint32_t get_bandIndex() const {return bandIndex;}
    void set_bandIndex(uint32_t bandIndexParam) {this->bandIndex = bandIndexParam;}
    uint8_t get_layoutType() const {return layoutType;}
    void set_layoutType(uint8_t layoutTypeParam) {this->layoutType = layoutTypeParam;}
    uint8_t get_pseShapeType() const {return pseShapeType;}
    void set_pseShapeType(uint8_t pseShapeTypeParam) {this->pseShapeType = pseShapeTypeParam;}
    uint8_t get_attenMaskShapeType() const {return attenMaskShapeType;}
    void set_attenMaskShapeType(uint8_t attenMaskShapeTypeParam) {this->attenMaskShapeType = attenMaskShapeTypeParam;}
    uint8_t get_attenMaskDataType() const {return attenMaskDataType;}
    void set_attenMaskDataType(uint8_t attenMaskDataTypeParam) {this->attenMaskDataType = attenMaskDataTypeParam;}
    uint8_t get_attenMaskCompressMode() const {return attenMaskCompressMode;}
    void set_attenMaskCompressMode(uint8_t attenMaskCompressModeParam) {this->attenMaskCompressMode = attenMaskCompressModeParam;}
    uint8_t get_implMode() const {return implMode;}
    void set_implMode(uint8_t implModeParam) {this->implMode = implModeParam;}
    uint8_t get_sparseType() const {return sparseType;}
    void set_sparseType(uint8_t sparseTypeParam) {this->sparseType = sparseTypeParam;}
    uint8_t get_needDropMaskOp() const {return needDropMaskOp;}
    void set_needDropMaskOp(uint8_t needDropMaskOpParam) {this->needDropMaskOp = needDropMaskOpParam;}
    uint8_t get_dropMaskOuter() const {return dropMaskOuter;}
    void set_dropMaskOuter(uint8_t dropMaskOuterParam) {this->dropMaskOuter = dropMaskOuterParam;}
    uint8_t get_pseEncodeType() const {return pseEncodeType;}
    void set_pseEncodeType(uint8_t pseEncodeTypeParam) {this->pseEncodeType = pseEncodeTypeParam;}
    uint16_t get_remain() const {return remain;}
    void set_remain(uint16_t remainParam) {this->remain = remainParam;}
    uint32_t get_attenMaskS2Size() const {return attenMaskS2Size;}
    void set_attenMaskS2Size(uint32_t attenMaskS2SizeParam) {this->attenMaskS2Size = attenMaskS2SizeParam;}
    uint32_t get_pseType() const {return pseType;}
    void set_pseType(uint32_t pseTypeParam) {this->pseType = pseTypeParam;}
    uint32_t get_rsv1() const {return rsv1;}
    void set_rsv1(uint32_t rsv1Param) {this->rsv1 = rsv1Param;}
    int64_t get_qStartIdx() const {return qStartIdx;}
    void set_qStartIdx(int64_t qStartIdxParam) {this->qStartIdx = qStartIdxParam;}
    int64_t get_kvStartIdx() const {return kvStartIdx;}
    void set_kvStartIdx(int64_t kvStartIdxParam) {this->kvStartIdx = kvStartIdxParam;}
    int64_t get_s1SparseValidSize() const {return s1SparseValidSize;}
    void set_s1SparseValidSize(int64_t s1SparseValidSizeParam) {this->s1SparseValidSize = s1SparseValidSizeParam;}
    int64_t get_s2SparseValidSize() const {return s2SparseValidSize;}
    void set_s2SparseValidSize(int64_t s2SparseValidSizeParam) {this->s2SparseValidSize = s2SparseValidSizeParam;}
    int64_t get_seed() const {return seed;}
    void set_seed(int64_t seedParam) {this->seed = seedParam;}
    int64_t get_offset() const {return offset;}
    void set_offset(int64_t offsetParam) {this->offset = offsetParam;}
    int64_t get_keepProbUint8() const {return keepProbUint8;}
    void set_keepProbUint8(int64_t keepProbUint8Param) {this->keepProbUint8 = keepProbUint8Param;}
    int64_t get_pseAlibiBaseS1() const {return pseAlibiBaseS1;}
    void set_pseAlibiBaseS1(int64_t pseAlibiBaseS1Param) {this->pseAlibiBaseS1 = pseAlibiBaseS1Param;}
    int64_t get_pseAlibiBaseS2() const {return pseAlibiBaseS2;}
    void set_pseAlibiBaseS2(int64_t pseAlibiBaseS2Param) {this->pseAlibiBaseS2 = pseAlibiBaseS2Param;}
    uint8_t get_deqScaleFlag() const {return deqScaleFlag;}
    void set_deqScaleFlag(uint8_t deqScaleFlagParam) {this->deqScaleFlag = deqScaleFlagParam;}
    uint8_t get_deqScale2Flag() const {return deqScale2Flag;}
    void set_deqScale2Flag(uint8_t deqScale2FlagParam) {this->deqScale2Flag = deqScale2FlagParam;}
    uint8_t get_isActualSeqLengthsNull() const {return isActualSeqLengthsNull;}
    void set_isActualSeqLengthsNull(uint8_t isActualSeqLengthsNullParam) {this->isActualSeqLengthsNull = isActualSeqLengthsNullParam;}
    uint8_t get_isActualSeqLengthsKVNull() const {return isActualSeqLengthsKVNull;}
    void set_isActualSeqLengthsKVNull(uint8_t isActualSeqLengthsKVNullParam) {this->isActualSeqLengthsKVNull = isActualSeqLengthsKVNullParam;}
    uint32_t get_actualSeqLengthsSize() const {return actualSeqLengthsSize;}
    void set_actualSeqLengthsSize(uint32_t actualSeqLengthsSizeParam) {this->actualSeqLengthsSize = actualSeqLengthsSizeParam;}
    uint32_t get_actualSeqLengthsKVSize() const {return actualSeqLengthsKVSize;}
    void set_actualSeqLengthsKVSize(uint32_t actualSeqLengthsKVSizeParam) {this->actualSeqLengthsKVSize = actualSeqLengthsKVSizeParam;}
    uint8_t get_isKvContinuous() const {return isKvContinuous;}
    void set_isKvContinuous(uint8_t isKvContinuousParam) {this->isKvContinuous = isKvContinuousParam;}
    uint8_t get_fromFused() const {return fromFused;}
    void set_fromFused(uint8_t fromFusedParam) {this->fromFused = fromFusedParam;}
    uint8_t get_isBSNDOut() const {return isBSNDOut;}
    void set_isBSNDOut(uint8_t isBSNDOutParam) {this->isBSNDOut = isBSNDOutParam;}
    uint8_t get_isGqa() const {return isGqa;}
    void set_isGqa(uint8_t isGqaParam) {this->isGqa = isGqaParam;}
    uint8_t get_isSoftMaxLseEnable() const {return isSoftMaxLseEnable;}
    void set_isSoftMaxLseEnable(uint8_t isSoftMaxLseEnableParam) {this->isSoftMaxLseEnable = isSoftMaxLseEnableParam;}
    uint8_t get_isActualSharedPrefixLenNull() const {return isActualSharedPrefixLenNull;}
    void set_isActualSharedPrefixLenNull(uint8_t isActualSharedPrefixLenNullParam) {this->isActualSharedPrefixLenNull = isActualSharedPrefixLenNullParam;}
    uint8_t get_isQHasLeftPadding() const {return isQHasLeftPadding;}
    void set_isQHasLeftPadding(uint8_t isQHasLeftPaddingParam) {this->isQHasLeftPadding = isQHasLeftPaddingParam;}
    uint8_t get_isKVHasLeftPadding() const {return isKVHasLeftPadding;}
    void set_isKVHasLeftPadding(uint8_t isKVHasLeftPaddingParam) {this->isKVHasLeftPadding = isKVHasLeftPaddingParam;}
    uint32_t get_ropeHeadSize() const {return ropeHeadSize;}
    void set_ropeHeadSize(uint32_t ropeHeadSizeParam) {this->ropeHeadSize = ropeHeadSizeParam;}
    uint32_t get_prefixSeqInnerSize() const {return prefixSeqInnerSize;}
    void set_prefixSeqInnerSize(uint32_t prefixSeqInnerSizeParam) {this->prefixSeqInnerSize = prefixSeqInnerSizeParam;}
    uint32_t get_headNumRatio() const {return headNumRatio;}
    void set_headNumRatio(uint32_t headNumRatioParam) {this->headNumRatio = headNumRatioParam;}
    int32_t get_blockSize() const {return blockSize;}
    void set_blockSize(int32_t blockSizeParam) {this->blockSize = blockSizeParam;}
    int32_t get_blockTableDim2() const {return blockTableDim2;}
    void set_blockTableDim2(int32_t blockTableDim2Param) {this->blockTableDim2 = blockTableDim2Param;}
    int32_t get_paBlockNumSum() const {return paBlockNumSum;}
    void set_paBlockNumSum(int32_t paBlockNumSumParam) {this->paBlockNumSum = paBlockNumSumParam;}
    uint8_t get_paLayoutType() const {return paLayoutType;}
    void set_paLayoutType(uint8_t paLayoutTypeParam) {this->paLayoutType = paLayoutTypeParam;}
    uint32_t get_attenMaskS1Size() const {return attenMaskS1Size;}
    void set_attenMaskS1Size(int32_t attenMaskS1SizeParam) {this->attenMaskS1Size = attenMaskS1SizeParam;}
    uint8_t get_isRowInvalid() const {return isRowInvalid;}
    void set_isRowInvalid(uint8_t isRowInvalidParam) {this->isRowInvalid = isRowInvalidParam;}
    uint8_t get_isPostQuantPerChnl() const {return isPostQuantPerChnl;}
    void set_isPostQuantPerChnl(uint8_t isPostQuantPerChnlParam) {this->isPostQuantPerChnl = isPostQuantPerChnlParam;}
    uint8_t get_isPostQuantBF16() const {return isPostQuantBF16;}
    void set_isPostQuantBF16(uint8_t isPostQuantBF16Param) {this->isPostQuantBF16 = isPostQuantBF16Param;}
    uint16_t get_antiquantPerTensorFlag() const {return antiquantPerTensorFlag;}
    void set_antiquantPerTensorFlag(uint16_t antiquantPerTensorFlagParam) {this->antiquantPerTensorFlag = antiquantPerTensorFlagParam;}
    uint16_t get_antiquantPerHeadFlag() const {return antiquantPerHeadFlag;}
    void set_antiquantPerHeadFlag(uint16_t antiquantPerHeadFlagParam) {this->antiquantPerHeadFlag = antiquantPerHeadFlagParam;}
    uint32_t get_antiquantParaSeqSize() const {return antiquantParaSeqSize;}
    void set_antiquantParaSeqSize(uint32_t antiquantParaSeqSizeParam) {this->antiquantParaSeqSize = antiquantParaSeqSizeParam;}
};

class MultiCoreParamsRegbase {
public:
    int32_t coreNum;
    int64_t totalSize;
    int64_t s1OuterSize;
    int64_t splitFactorSize;
    int64_t splitFactorTailSize;
    uint32_t bnStartIdx[48];
    int64_t sparseStartIdx[48];

    int32_t get_coreNum() const {return coreNum;}
    void set_coreNum(int32_t coreNumParam) {this->coreNum = coreNumParam;}
    int64_t get_totalSize() const {return totalSize;}
    void set_totalSize(int64_t totalSizeParam) {this->totalSize = totalSizeParam;}
    int64_t get_s1OuterSize() const {return s1OuterSize;}
    void set_s1OuterSize(int64_t s1OuterSizeParam) {this->s1OuterSize = s1OuterSizeParam;}
    int64_t get_splitFactorSize() const {return splitFactorSize;}
    void set_splitFactorSize(int64_t splitFactorSizeParam) {this->splitFactorSize = splitFactorSizeParam;}
    int64_t get_splitFactorTailSize() const {return splitFactorTailSize;}
    void set_splitFactorTailSize(int64_t splitFactorTailSizeParam) {this->splitFactorTailSize = splitFactorTailSizeParam;}
    uint32_t *get_bnStartIdxPtr() {return bnStartIdx;}
    int64_t *get_sparseStartIdxPtr() {return sparseStartIdx;}
};

class DropmaskParamsRegbase {
public:
    int32_t multiCoreFactorSize;
    int32_t baseUbCalSize;
    int64_t multiCoreTotalSize;
    int64_t shapeTotalSize;

    int32_t get_multiCoreFactorSize() const {return multiCoreFactorSize;}
    void set_multiCoreFactorSize(int32_t multiCoreFactorSizeParam) {this->multiCoreFactorSize = multiCoreFactorSizeParam;}
    int32_t get_baseUbCalSize() const {return baseUbCalSize;}
    void set_baseUbCalSize(int32_t baseUbCalSizeParam) {this->baseUbCalSize = baseUbCalSizeParam;}
    int64_t get_multiCoreTotalSize() const {return multiCoreTotalSize;}
    void set_multiCoreTotalSize(int64_t multiCoreTotalSizeParam) {this->multiCoreTotalSize = multiCoreTotalSizeParam;}
    int64_t get_shapeTotalSize() const {return shapeTotalSize;}
    void set_shapeTotalSize(int64_t shapeTotalSizeParam) {this->shapeTotalSize = shapeTotalSizeParam;}
};

class InitOutputParams {
public:
    uint32_t singleCoreSize;
    uint8_t needInit;
    uint8_t isOneN;
    uint8_t rsvd[2];
    int64_t totalOutputSize;
    int64_t totalSoftMaxLseOutputSize;

    uint32_t get_singleCoreSize() const {return singleCoreSize;}
    void set_singleCoreSize(uint32_t singleCoreSizeParam) {this->singleCoreSize = singleCoreSizeParam;}
    uint8_t get_needInit() const {return needInit;}
    void set_needInit(uint8_t needInitParam) {this->needInit = needInitParam;}
    uint8_t get_isOneN() const {return isOneN;}
    void set_isOneN(uint8_t isOneNParam) {this->isOneN = isOneNParam;}
    uint8_t *get_rsvdPtr() {return rsvd;}
    int64_t get_totalOutputSize() const {return totalOutputSize;}
    void set_totalOutputSize(int64_t totalOutputSizeParam) {this->totalOutputSize = totalOutputSizeParam;}
    int64_t get_totalSoftMaxLseOutputSize() const {return totalSoftMaxLseOutputSize;}
    void set_totalSoftMaxLseOutputSize(int64_t totalSoftMaxLseOutputSizeParam) {this->totalSoftMaxLseOutputSize = totalSoftMaxLseOutputSizeParam;}
};

class FlashAttentionScoreSimplifiedTilingData {
public:
    InputParamsRegbase inputParamsRegbase;
    MultiCoreParamsRegbase multiCoreParamsRegbase;
    DropmaskParamsRegbase dropmaskParamsRegbase;
    InitOutputParams initOutputParams;
};
}  // namespace optiling
#endif