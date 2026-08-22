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
 * \file prompt_flash_attention_tiling_regbase.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_TILING_REGBASE_H
#define PROMPT_FLASH_ATTENTION_TILING_REGBASE_H

#include "kernel_tiling/kernel_tiling.h"

namespace pfatilingdata {
class TCubeTiling {
public:
    int usedCoreNum = 0;
    int get_usedCoreNum() const { return this->usedCoreNum; }
    void set_usedCoreNum(int usedCoreNumParam) { this->usedCoreNum = usedCoreNumParam; }

    int M = 0;
    int get_M() const { return this->M; }
    void set_M(int MParam) { this->M = MParam; }

    int N = 0;
    int get_N() const { return this->N; }
    void set_N(int NParam) { this->N = NParam; }

    int Ka = 0;
    int get_Ka() const { return this->Ka; }
    void set_Ka(int KaParam) { this->Ka = KaParam; }

    int Kb = 0;
    int get_Kb() const { return this->Kb; }
    void set_Kb(int KbParam) { this->Kb = KbParam; }

    int singleCoreM = 0;
    int get_singleCoreM() const { return this->singleCoreM; }
    void set_singleCoreM(int singleCoreMParam) { this->singleCoreM = singleCoreMParam; }

    int singleCoreN = 0;
    int get_singleCoreN() const { return this->singleCoreN; }
    void set_singleCoreN(int singleCoreNParam) { this->singleCoreN = singleCoreNParam; }

    int singleCoreK = 0;
    int get_singleCoreK() const { return this->singleCoreK; }
    void set_singleCoreK(int singleCoreKParam) { this->singleCoreK = singleCoreKParam; }

    int baseM = 0;
    int get_baseM() const { return this->baseM; }
    void set_baseM(int baseMParam) { this->baseM = baseMParam; }

    int baseN = 0;
    int get_baseN() const { return this->baseN; }
    void set_baseN(int baseNParam) { this->baseN = baseNParam; }

    int baseK = 0;
    int get_baseK() const { return this->baseK; }
    void set_baseK(int baseKParam) { this->baseK = baseKParam; }

    int depthA1 = 0;
    int get_depthA1() const { return this->depthA1; }
    void set_depthA1(int depthA1Param) { this->depthA1 = depthA1Param; }

    int depthB1 = 0;
    int get_depthB1() const { return this->depthB1; }
    void set_depthB1(int depthB1Param) { this->depthB1 = depthB1Param; }

    int stepM = 0;
    int get_stepM() const { return this->stepM; }
    void set_stepM(int stepMParam) { this->stepM = stepMParam; }

    int stepN = 0;
    int get_stepN() const { return this->stepN; }
    void set_stepN(int stepNParam) { this->stepN = stepNParam; }

    int stepKa = 0;
    int get_stepKa() const { return this->stepKa; }
    void set_stepKa(int stepKaParam) { this->stepKa = stepKaParam; }

    int stepKb = 0;
    int get_stepKb() const { return this->stepKb; }
    void set_stepKb(int stepKbParam) { this->stepKb = stepKbParam; }

    int isBias = 0;
    int get_isBias() const { return this->isBias; }
    void set_isBias(int isBiasParam) { this->isBias = isBiasParam; }

    int transLength = 0;
    int get_transLength() const { return this->transLength; }
    void set_transLength(int transLengthParam) { this->transLength = transLengthParam; }

    int iterateOrder = 0;
    int get_iterateOrder() const { return this->iterateOrder; }
    void set_iterateOrder(int iterateOrderParam) { this->iterateOrder = iterateOrderParam; }

    int dbL0A = 0;
    int get_dbL0A() const { return this->dbL0A; }
    void set_dbL0A(int dbL0AParam) { this->dbL0A = dbL0AParam; }

    int dbL0B = 0;
    int get_dbL0B() const { return this->dbL0B; }
    void set_dbL0B(int dbL0BParam) { this->dbL0B = dbL0BParam; }

    int dbL0C = 0;
    int get_dbL0C() const { return this->dbL0C; }
    void set_dbL0C(int dbL0CParam) { this->dbL0C = dbL0CParam; }

    int shareMode = 0;
    int get_shareMode() const { return this->shareMode; }
    void set_shareMode(int shareModeParam) { this->shareMode = shareModeParam; }

    int shareL1Size = 0;
    int get_shareL1Size() const { return this->shareL1Size; }
    void set_shareL1Size(int shareL1SizeParam) { this->shareL1Size = shareL1SizeParam; }

    int shareL0CSize = 0;
    int get_shareL0CSize() const { return this->shareL0CSize; }
    void set_shareL0CSize(int shareL0CSizeParam) { this->shareL0CSize = shareL0CSizeParam; }

    int shareUbSize = 0;
    int get_shareUbSize() const { return this->shareUbSize; }
    void set_shareUbSize(int shareUbSizeParam) { this->shareUbSize = shareUbSizeParam; }

    int batchM = 0;
    int get_batchM() const { return this->batchM; }
    void set_batchM(int batchMParam) { this->batchM = batchMParam; }

    int batchN = 0;
    int get_batchN() const { return this->batchN; }
    void set_batchN(int batchNParam) { this->batchN = batchNParam; }

    int singleBatchM = 0;
    int get_singleBatchM() const { return this->singleBatchM; }
    void set_singleBatchM(int singleBatchMParam) { this->singleBatchM = singleBatchMParam; }

    int singleBatchN = 0;
    int get_singleBatchN() const { return this->singleBatchN; }
    void set_singleBatchN(int singleBatchNParam) { this->singleBatchN = singleBatchNParam; }
};

class SoftMaxTiling {
public:
    uint32_t srcM = 0;
    uint32_t get_srcM() const { return this->srcM; }
    void set_srcM(uint32_t srcMParam) { this->srcM = srcMParam; }

    uint32_t srcK = 0;
    uint32_t get_srcK() const { return this->srcK; }
    void set_srcK(uint32_t srcKParam) { this->srcK = srcKParam; }

    uint32_t srcSize = 0;
    uint32_t get_srcSize() const { return this->srcSize; }
    void set_srcSize(uint32_t srcSizeParam) { this->srcSize = srcSizeParam; }

    uint32_t outMaxM = 0;
    uint32_t get_outMaxM() const { return this->outMaxM; }
    void set_outMaxM(uint32_t outMaxMParam) { this->outMaxM = outMaxMParam; }

    uint32_t outMaxK = 0;
    uint32_t get_outMaxK() const { return this->outMaxK; }
    void set_outMaxK(uint32_t outMaxKParam) { this->outMaxK = outMaxKParam; }

    uint32_t outMaxSize = 0;
    uint32_t get_outMaxSize() const { return this->outMaxSize; }
    void set_outMaxSize(uint32_t outMaxSizeParam) { this->outMaxSize = outMaxSizeParam; }

    uint32_t splitM = 0;
    uint32_t get_splitM() const { return this->splitM; }
    void set_splitM(uint32_t splitMParam) { this->splitM = splitMParam; }

    uint32_t splitK = 0;
    uint32_t get_splitK() const { return this->splitK; }
    void set_splitK(uint32_t splitKParam) { this->splitK = splitKParam; }

    uint32_t splitSize = 0;
    uint32_t get_splitSize() const { return this->splitSize; }
    void set_splitSize(uint32_t splitSizeParam) { this->splitSize = splitSizeParam; }

    uint32_t reduceM = 0;
    uint32_t get_reduceM() const { return this->reduceM; }
    void set_reduceM(uint32_t reduceMParam) { this->reduceM = reduceMParam; }

    uint32_t reduceK = 0;
    uint32_t get_reduceK() const { return this->reduceK; }
    void set_reduceK(uint32_t reduceKParam) { this->reduceK = reduceKParam; }

    uint32_t reduceSize = 0;
    uint32_t get_reduceSize() const { return this->reduceSize; }
    void set_reduceSize(uint32_t reduceSizeParam) { this->reduceSize = reduceSizeParam; }

    uint32_t rangeM = 0;
    uint32_t get_rangeM() const { return this->rangeM; }
    void set_rangeM(uint32_t rangeMParam) { this->rangeM = rangeMParam; }

    uint32_t tailM = 0;
    uint32_t get_tailM() const { return this->tailM; }
    void set_tailM(uint32_t tailMParam) { this->tailM = tailMParam; }

    uint32_t tailSplitSize = 0;
    uint32_t get_tailSplitSize() const { return this->tailSplitSize; }
    void set_tailSplitSize(uint32_t tailSplitSizeParam) { this->tailSplitSize = tailSplitSizeParam; }

    uint32_t tailReduceSize = 0;
    uint32_t get_tailReduceSize() const { return this->tailReduceSize; }
    void set_tailReduceSize(uint32_t tailReduceSizeParam) { this->tailReduceSize = tailReduceSizeParam; }
};

}

#define TCubeTilingCopy(dstTCubeTiling, srcTCubeTiling)  \
do{  \
    dstTCubeTiling.usedCoreNum = srcTCubeTiling.usedCoreNum;  \
    dstTCubeTiling.M = srcTCubeTiling.M;  \
    dstTCubeTiling.N = srcTCubeTiling.N;  \
    dstTCubeTiling.Ka = srcTCubeTiling.Ka;  \
    dstTCubeTiling.Kb = srcTCubeTiling.Kb;  \
    dstTCubeTiling.singleCoreM = srcTCubeTiling.singleCoreM;  \
    dstTCubeTiling.singleCoreN = srcTCubeTiling.singleCoreN;  \
    dstTCubeTiling.singleCoreK = srcTCubeTiling.singleCoreK;  \
    dstTCubeTiling.baseM = srcTCubeTiling.baseM;  \
    dstTCubeTiling.baseN = srcTCubeTiling.baseN;  \
    dstTCubeTiling.baseK = srcTCubeTiling.baseK;  \
    dstTCubeTiling.depthA1 = srcTCubeTiling.depthA1;  \
    dstTCubeTiling.depthB1 = srcTCubeTiling.depthB1;  \
    dstTCubeTiling.stepM = srcTCubeTiling.stepM;  \
    dstTCubeTiling.stepN = srcTCubeTiling.stepN;  \
    dstTCubeTiling.stepKa = srcTCubeTiling.stepKa;  \
    dstTCubeTiling.stepKb = srcTCubeTiling.stepKb;  \
    dstTCubeTiling.isBias = srcTCubeTiling.isBias;  \
    dstTCubeTiling.transLength = srcTCubeTiling.transLength;  \
    dstTCubeTiling.iterateOrder = srcTCubeTiling.iterateOrder;  \
    dstTCubeTiling.dbL0A = srcTCubeTiling.dbL0A;  \
    dstTCubeTiling.dbL0B = srcTCubeTiling.dbL0B;  \
    dstTCubeTiling.dbL0C = srcTCubeTiling.dbL0C;  \
    dstTCubeTiling.shareMode = srcTCubeTiling.shareMode;  \
    dstTCubeTiling.shareL1Size = srcTCubeTiling.shareL1Size;  \
    dstTCubeTiling.shareL0CSize = srcTCubeTiling.shareL0CSize;  \
    dstTCubeTiling.shareUbSize = srcTCubeTiling.shareUbSize;  \
    dstTCubeTiling.batchM = srcTCubeTiling.batchM;  \
    dstTCubeTiling.batchN = srcTCubeTiling.batchN;  \
    dstTCubeTiling.singleBatchM = srcTCubeTiling.singleBatchM;  \
    dstTCubeTiling.singleBatchN = srcTCubeTiling.singleBatchN;  \
}while(0)

#define SoftMaxTilingCopy(dstSoftMaxTiling, srcSoftMaxTiling)  \
do{  \
    dstSoftMaxTiling.srcM = srcSoftMaxTiling.srcM;  \
    dstSoftMaxTiling.srcK = srcSoftMaxTiling.srcK;  \
    dstSoftMaxTiling.srcSize = srcSoftMaxTiling.srcSize;  \
    dstSoftMaxTiling.outMaxM = srcSoftMaxTiling.outMaxM;  \
    dstSoftMaxTiling.outMaxK = srcSoftMaxTiling.outMaxK;  \
    dstSoftMaxTiling.outMaxSize = srcSoftMaxTiling.outMaxSize;  \
    dstSoftMaxTiling.splitM = srcSoftMaxTiling.splitM;  \
    dstSoftMaxTiling.splitK = srcSoftMaxTiling.splitK;  \
    dstSoftMaxTiling.splitSize = srcSoftMaxTiling.splitSize;  \
    dstSoftMaxTiling.reduceM = srcSoftMaxTiling.reduceM;  \
    dstSoftMaxTiling.reduceK = srcSoftMaxTiling.reduceK;  \
    dstSoftMaxTiling.reduceSize = srcSoftMaxTiling.reduceSize;  \
    dstSoftMaxTiling.rangeM = srcSoftMaxTiling.rangeM;  \
    dstSoftMaxTiling.tailM = srcSoftMaxTiling.tailM;  \
    dstSoftMaxTiling.tailSplitSize = srcSoftMaxTiling.tailSplitSize;  \
    dstSoftMaxTiling.tailReduceSize = srcSoftMaxTiling.tailReduceSize;  \
}while(0)

#define TilingDataCopy(dstTilingData, srcTilingData) \
do{  \
    TCubeTilingCopy(dstTilingData.bmm1TilingDataRect, srcTilingData.bmm1TilingDataRect);  \
    TCubeTilingCopy(dstTilingData.bmm2TilingDataRect, srcTilingData.bmm2TilingDataRect);  \
    dstTilingData.promptAttentionBaseParams = srcTilingData.promptAttentionBaseParams;  \
    dstTilingData.promptAttentionSeqParams = srcTilingData.promptAttentionSeqParams;  \
    dstTilingData.promptAttentionSingleCoreParams = srcTilingData.promptAttentionSingleCoreParams;  \
    dstTilingData.promptAttentionTensorSizeRect = srcTilingData.promptAttentionTensorSizeRect;  \
    dstTilingData.promptAttentionInitOutputParams = srcTilingData.promptAttentionInitOutputParams;  \
    SoftMaxTilingCopy(dstTilingData.softmaxTilingDataRect, srcTilingData.softmaxTilingDataRect);  \
    SoftMaxTilingCopy(dstTilingData.softmaxFlashTilingDataRect, srcTilingData.softmaxFlashTilingDataRect);  \
}while(0)

namespace optiling {
class PromptAttentionBaseParams {
public:
    uint32_t batchSize = 0;
    uint32_t get_batchSize() const { return this->batchSize; }
    void set_batchSize(uint32_t batchSizeParam) { this->batchSize = batchSizeParam; }

    uint32_t headNumSize = 0;
    uint32_t get_headNumSize() const { return this->headNumSize; }
    void set_headNumSize(uint32_t headNumSizeParam) { this->headNumSize = headNumSizeParam; }

    uint32_t seqSize = 0;
    uint32_t get_seqSize() const { return this->seqSize; }
    void set_seqSize(uint32_t seqSizeParam) { this->seqSize = seqSizeParam; }

    uint32_t headSize = 0;
    uint32_t get_headSize() const { return this->headSize; }
    void set_headSize(uint32_t headSizeParam) { this->headSize = headSizeParam; }

    float scaleValue = 0;
    float get_scaleValue() const { return this->scaleValue; }
    void set_scaleValue(float scaleValueParam) { this->scaleValue = scaleValueParam; }

    int32_t preTokens = 0;
    int32_t get_preTokens() const { return this->preTokens; }
    void set_preTokens(int32_t preTokensParam) { this->preTokens = preTokensParam; }

    int32_t nextTokens = 0;
    int32_t get_nextTokens() const { return this->nextTokens; }
    void set_nextTokens(int32_t nextTokensParam) { this->nextTokens = nextTokensParam; }

    int32_t blockSize = 0;
    int32_t get_blockSize() const { return this->blockSize; }
    void set_blockSize(int32_t blockSizeParam) { this->blockSize = blockSizeParam; }

    int32_t blockTableDim2 = 0;
    int32_t get_blockTableDim2() const { return this->blockTableDim2; }
    void set_blockTableDim2(int32_t blockTableDim2Param) { this->blockTableDim2 = blockTableDim2Param; }

    int32_t PABlockNumSum = 0;
    int32_t get_PABlockNumSum() const { return this->PABlockNumSum; }
    void set_PABlockNumSum(int32_t PABlockNumSumParam) { this->PABlockNumSum = PABlockNumSumParam; }

    uint32_t dimNumOfseq = 0;
    uint32_t get_dimNumOfseq() const { return this->dimNumOfseq; }
    void set_dimNumOfseq(uint32_t dimNumOfseqParam) { this->dimNumOfseq = dimNumOfseqParam; }

    uint32_t typeByteNum = 0;
    uint32_t get_typeByteNum() const { return this->typeByteNum; }
    void set_typeByteNum(uint32_t typeByteNumParam) { this->typeByteNum = typeByteNumParam; }

    uint32_t seqInnerSize = 0;
    uint32_t get_seqInnerSize() const { return this->seqInnerSize; }
    void set_seqInnerSize(uint32_t seqInnerSizeParam) { this->seqInnerSize = seqInnerSizeParam; }

    uint32_t prefixSeqInnerSize = 0;
    uint32_t get_prefixSeqInnerSize() const { return this->prefixSeqInnerSize; }
    void set_prefixSeqInnerSize(uint32_t prefixSeqInnerSizeParam) { this->prefixSeqInnerSize = prefixSeqInnerSizeParam; }

    uint32_t usePseShift = 0;
    uint32_t get_usePseShift() const { return this->usePseShift; }
    void set_usePseShift(uint32_t usePseShiftParam) { this->usePseShift = usePseShiftParam; }

    uint32_t useMask = 0;
    uint32_t get_useMask() const { return this->useMask; }
    void set_useMask(uint32_t useMaskParam) { this->useMask = useMaskParam; }

    uint32_t headNumRatio = 0;
    uint32_t get_headNumRatio() const { return this->headNumRatio; }
    void set_headNumRatio(uint32_t headNumRatioParam) { this->headNumRatio = headNumRatioParam; }

    uint32_t attenMaskElemType = 0;
    uint32_t get_attenMaskElemType() const { return this->attenMaskElemType; }
    void set_attenMaskElemType(uint32_t attenMaskElemTypeParam) { this->attenMaskElemType = attenMaskElemTypeParam; }

    uint32_t pseShiftTypeByteNum = 0;
    uint32_t get_pseShiftTypeByteNum() const { return this->pseShiftTypeByteNum; }
    void set_pseShiftTypeByteNum(uint32_t pseShiftTypeByteNumParam) { this->pseShiftTypeByteNum = pseShiftTypeByteNumParam; }

    uint32_t pseMaskMaxSize = 0;
    uint32_t get_pseMaskMaxSize() const { return this->pseMaskMaxSize; }
    void set_pseMaskMaxSize(uint32_t pseMaskMaxSizeParam) { this->pseMaskMaxSize = pseMaskMaxSizeParam; }

    uint32_t maskTypeByteNum = 0;
    uint32_t get_maskTypeByteNum() const { return this->maskTypeByteNum; }
    void set_maskTypeByteNum(uint32_t maskTypeByteNumParam) { this->maskTypeByteNum = maskTypeByteNumParam; }

    uint32_t outputTypeByteNum = 0;
    uint32_t get_outputTypeByteNum() const { return this->outputTypeByteNum; }
    void set_outputTypeByteNum(uint32_t outputTypeByteNumParam) { this->outputTypeByteNum = outputTypeByteNumParam; }

    uint32_t softmaxTypeByteNum = 0;
    uint32_t get_softmaxTypeByteNum() const { return this->softmaxTypeByteNum; }
    void set_softmaxTypeByteNum(uint32_t softmaxTypeByteNumParam) { this->softmaxTypeByteNum = softmaxTypeByteNumParam; }

    uint32_t sparseMode = 0;
    uint32_t get_sparseMode() const { return this->sparseMode; }
    void set_sparseMode(uint32_t sparseModeParam) { this->sparseMode = sparseModeParam; }

    uint32_t alignedHeadSize = 0;
    uint32_t get_alignedHeadSize() const { return this->alignedHeadSize; }
    void set_alignedHeadSize(uint32_t alignedHeadSizeParam) { this->alignedHeadSize = alignedHeadSizeParam; }

    uint32_t splitS2 = 0;
    uint32_t get_splitS2() const { return this->splitS2; }
    void set_splitS2(uint32_t splitS2Param) { this->splitS2 = splitS2Param; }

    uint32_t splitD = 0;
    uint32_t get_splitD() const { return this->splitD; }
    void set_splitD(uint32_t splitDParam) { this->splitD = splitDParam; }

    uint32_t layoutType = 0;
    uint32_t get_layoutType() const { return this->layoutType; }
    void set_layoutType(uint32_t layoutTypeParam) { this->layoutType = layoutTypeParam; }

    uint32_t PAlayoutType = 0;
    uint32_t get_PAlayoutType() const { return this->PAlayoutType; }
    void set_PAlayoutType(uint32_t PAlayoutTypeParam) { this->PAlayoutType = PAlayoutTypeParam; }

    uint32_t pseShiftS1Size = 0;
    uint32_t get_pseShiftS1Size() const { return this->pseShiftS1Size; }
    void set_pseShiftS1Size(uint32_t pseShiftS1SizeParam) { this->pseShiftS1Size = pseShiftS1SizeParam; }

    uint32_t pseShiftS2Size = 0;
    uint32_t get_pseShiftS2Size() const { return this->pseShiftS2Size; }
    void set_pseShiftS2Size(uint32_t pseShiftS2SizeParam) { this->pseShiftS2Size = pseShiftS2SizeParam; }

    uint32_t maskKVsSize = 0;
    uint32_t get_maskKVsSize() const { return this->maskKVsSize; }
    void set_maskKVsSize(uint32_t maskKVsSizeParam) { this->maskKVsSize = maskKVsSizeParam; }

    uint32_t maskQsSize = 0;
    uint32_t get_maskQsSize() const { return this->maskQsSize; }
    void set_maskQsSize(uint32_t maskQsSizeParam) { this->maskQsSize = maskQsSizeParam; }

    uint32_t isLayoutSH = 0;
    uint32_t get_isLayoutSH() const { return this->isLayoutSH; }
    void set_isLayoutSH(uint32_t isLayoutSHParam) { this->isLayoutSH = isLayoutSHParam; }

    uint32_t isActualSeqLengthsNull = 0;
    uint32_t get_isActualSeqLengthsNull() const { return this->isActualSeqLengthsNull; }
    void set_isActualSeqLengthsNull(uint32_t isActualSeqLengthsNullParam) { this->isActualSeqLengthsNull = isActualSeqLengthsNullParam; }

    uint32_t isActualSeqLengthsKVNull = 0;
    uint32_t get_isActualSeqLengthsKVNull() const { return this->isActualSeqLengthsKVNull; }
    void set_isActualSeqLengthsKVNull(uint32_t isActualSeqLengthsKVNullParam) { this->isActualSeqLengthsKVNull = isActualSeqLengthsKVNullParam; }

    uint32_t actualSeqLengthsSize = 0;
    uint32_t get_actualSeqLengthsSize() const { return this->actualSeqLengthsSize; }
    void set_actualSeqLengthsSize(uint32_t actualSeqLengthsSizeParam) { this->actualSeqLengthsSize = actualSeqLengthsSizeParam; }

    uint32_t actualSeqLengthsKVSize = 0;
    uint32_t get_actualSeqLengthsKVSize() const { return this->actualSeqLengthsKVSize; }
    void set_actualSeqLengthsKVSize(uint32_t actualSeqLengthsKVSizeParam) { this->actualSeqLengthsKVSize = actualSeqLengthsKVSizeParam; }

    uint32_t deqScaleFlag = 0;
    uint32_t get_deqScaleFlag() const { return this->deqScaleFlag; }
    void set_deqScaleFlag(uint32_t deqScaleFlagParam) { this->deqScaleFlag = deqScaleFlagParam; }

    uint32_t deqScale2Flag = 0;
    uint32_t get_deqScale2Flag() const { return this->deqScale2Flag; }
    void set_deqScale2Flag(uint32_t deqScale2FlagParam) { this->deqScale2Flag = deqScale2FlagParam; }

    uint32_t isAntiPerchannel = 0;
    uint32_t get_isAntiPerchannel() const { return this->isAntiPerchannel; }
    void set_isAntiPerchannel(uint32_t isAntiPerchannelParam) { this->isAntiPerchannel = isAntiPerchannelParam; }

    uint32_t isRowInvalid = 0;
    uint32_t get_isRowInvalid() const { return this->isRowInvalid; }
    void set_isRowInvalid(uint32_t isRowInvalidParam) { this->isRowInvalid = isRowInvalidParam; }

    uint32_t softmaxOuterSize = 0;
    uint32_t get_softmaxOuterSize() const { return this->softmaxOuterSize; }
    void set_softmaxOuterSize(uint32_t softmaxOuterSizeParam) { this->softmaxOuterSize = softmaxOuterSizeParam; }

    uint32_t isQuant2Perchannel = 0;
    uint32_t get_isQuant2Perchannel() const { return this->isQuant2Perchannel; }
    void set_isQuant2Perchannel(uint32_t isQuant2PerchannelParam) { this->isQuant2Perchannel = isQuant2PerchannelParam; }

    uint32_t isQuant2BF16 = 0;
    uint32_t get_isQuant2BF16() const { return this->isQuant2BF16; }
    void set_isQuant2BF16(uint32_t isQuant2BF16Param) { this->isQuant2BF16 = isQuant2BF16Param; }

    uint32_t isKvContinuous = 0;
    uint32_t get_isKvContinuous() const { return this->isKvContinuous; }
    void set_isKvContinuous(uint32_t isKvContinuousParam) { this->isKvContinuous = isKvContinuousParam; }

    uint32_t fromFused = 0;
    uint32_t get_fromFused() const { return this->fromFused; }
    void set_fromFused(uint32_t fromFusedParam) { this->fromFused = fromFusedParam; }

    uint32_t isBSNDOut = 0;
    uint32_t get_isBSNDOut() const { return this->isBSNDOut; }
    void set_isBSNDOut(uint32_t isBSNDOutParam) { this->isBSNDOut = isBSNDOutParam; }

    uint32_t transposeLayout = 0;
    uint32_t get_transposeLayout() const { return this->transposeLayout; }
    void set_transposeLayout(uint32_t transposeLayoutParam) { this->transposeLayout = transposeLayoutParam; }

    int64_t t1Size = 0;
    int64_t get_t1Size() const { return this->t1Size; }
    void set_t1Size(int64_t t1SizeParam) { this->t1Size = t1SizeParam; }

    int64_t t2Size = 0;
    int64_t get_t2Size() const { return this->t2Size; }
    void set_t2Size(int64_t t2SizeParam) { this->t2Size = t2SizeParam; }

    uint32_t isIFA = 0;
    uint32_t get_isIFA() const { return this->isIFA; }
    void set_isIFA(uint32_t isIFAParam) { this->isIFA = isIFAParam; }

    uint32_t isSoftMaxLseEnable = 0;
    uint32_t get_isSoftMaxLseEnable() const { return this->isSoftMaxLseEnable; }
    void set_isSoftMaxLseEnable(uint32_t isSoftMaxLseEnableParam) { this->isSoftMaxLseEnable = isSoftMaxLseEnableParam; }

    uint32_t isActualSharedPrefixLenNull = 0;
    uint32_t get_isActualSharedPrefixLenNull() const { return this->isActualSharedPrefixLenNull; }
    void set_isActualSharedPrefixLenNull(uint32_t isActualSharedPrefixLenNullParam) { this->isActualSharedPrefixLenNull = isActualSharedPrefixLenNullParam; }

    uint32_t isQHasLeftPadding = 0;
    uint32_t get_isQHasLeftPadding() const { return this->isQHasLeftPadding; }
    void set_isQHasLeftPadding(uint32_t isQHasLeftPaddingParam) { this->isQHasLeftPadding = isQHasLeftPaddingParam; }

    uint32_t isKVHasLeftPadding = 0;
    uint32_t get_isKVHasLeftPadding() const { return this->isKVHasLeftPadding; }
    void set_isKVHasLeftPadding(uint32_t isKVHasLeftPaddingParam) { this->isKVHasLeftPadding = isKVHasLeftPaddingParam; }

    int64_t keyAntiquantMode = 0;
    int64_t get_keyAntiquantMode() const { return this->keyAntiquantMode; }
    void set_keyAntiquantMode(int64_t keyAntiquantModeParam) { this->keyAntiquantMode = keyAntiquantModeParam; }

    int64_t valueAntiquantMode = 0;
    int64_t get_valueAntiquantMode() const { return this->valueAntiquantMode; }
    void set_valueAntiquantMode(int64_t valueAntiquantModeParam) { this->valueAntiquantMode = valueAntiquantModeParam; }

    uint32_t hasKeyAntiquantOffset = 0;
    uint32_t get_hasKeyAntiquantOffset() const { return this->hasKeyAntiquantOffset; }
    void set_hasKeyAntiquantOffset(uint32_t hasKeyAntiquantOffsetParam) { this->hasKeyAntiquantOffset = hasKeyAntiquantOffsetParam; }

    uint32_t isMsd = 0;
    uint32_t get_isMsd() const { return this->isMsd; }
    void set_isMsd(uint32_t isMsdParam) { this->isMsd = isMsdParam; }

    uint32_t isQuant2FP16 = 0;
    uint32_t get_isQuant2FP16() const { return this->isQuant2FP16; }
    void set_isQuant2FP16(uint32_t isQuant2FP16Param) { this->isQuant2FP16 = isQuant2FP16Param; }

    uint32_t ropeHeadSize = 0;
    uint32_t get_ropeHeadSize() const { return this->ropeHeadSize; }
    void set_ropeHeadSize(uint32_t ropeHeadSizeParam) { this->ropeHeadSize = ropeHeadSizeParam; }

    uint32_t qkHeadSize = 0;
    uint32_t get_qkHeadSize() const { return this->qkHeadSize; }
    void set_qkHeadSize(uint32_t qkHeadSizeParam) { this->qkHeadSize = qkHeadSizeParam; }

    uint32_t vHeadSize = 0;
    uint32_t get_vHeadSize() const { return this->vHeadSize; }
    void set_vHeadSize(uint32_t vHeadSizeParam) { this->vHeadSize = vHeadSizeParam; }

    uint32_t gOfMla = 0;
    uint32_t get_gOfMla() const { return this->gOfMla; }
    void set_gOfMla(uint32_t gOfMlaParam) { this->gOfMla = gOfMlaParam; }
};

#define CORE_NUM 64 // 64: default core num

class PromptAttentionSeqParams {
public:
    uint32_t CoreHeadNumTail[CORE_NUM] = {0};
    uint32_t *get_CoreHeadNumTailPtr()
    {
        return CoreHeadNumTail;
    }
    void set_CoreHeadNumTail(const uint32_t *val)
    {
        for (int i = 0; i < CORE_NUM; ++i) {
            this->CoreHeadNumTail[i] = val[i];
        }
    }
    uint32_t actualS1[CORE_NUM] = {0};
    uint32_t *get_actualS1Ptr()
    {
        return actualS1;
    }
    void set_actualS1(const uint32_t *val)
    {
        for (int i = 0; i < CORE_NUM; ++i) {
            this->actualS1[i] = val[i];
        }
    }
    uint32_t actualCoreNums[CORE_NUM] = {0};
    uint32_t *get_actualCoreNumsPtr()
    {
        return actualCoreNums;
    }
    void set_actualCoreNums(const uint32_t *val)
    {
        for (int i = 0; i < CORE_NUM; ++i) {
            this->actualCoreNums[i] = val[i];
        }
    }
    uint32_t singleCoreHeadNumSize[CORE_NUM] = {0};
    uint32_t *get_singleCoreHeadNumSizePtr()
    {
        return singleCoreHeadNumSize;
    }
    void set_singleCoreHeadNumSize(const uint32_t *val)
    {
        for (int i = 0; i < CORE_NUM; ++i) {
            this->singleCoreHeadNumSize[i] = val[i];
        }
    }
    uint32_t coreSeqPosStart[CORE_NUM] = {0};
    uint32_t *get_coreSeqPosStartPtr()
    {
        return coreSeqPosStart;
    }
    void set_coreSeqPosStart(const uint32_t *val)
    {
        for (int i = 0; i < CORE_NUM; ++i) {
            this->coreSeqPosStart[i] = val[i];
        }
    }
    uint32_t coreSeqPosEnd[CORE_NUM] = {0};
    uint32_t *get_coreSeqPosEndPtr()
    {
        return coreSeqPosEnd;
    }
    void set_coreSeqPosEnd(const uint32_t *val)
    {
        for (int i = 0; i < CORE_NUM; ++i) {
            this->coreSeqPosEnd[i] = val[i];
        }
    }
};

class PromptAttentionSingleCoreParams {
public:
    uint32_t singleProcessSInnerSize = 0;
    uint32_t get_singleProcessSInnerSize() const { return this->singleProcessSInnerSize; }
    void set_singleProcessSInnerSize(uint32_t singleProcessSInnerSizeParam) { this->singleProcessSInnerSize = singleProcessSInnerSizeParam; }

    uint32_t singleProcessSOuterSize = 0;
    uint32_t get_singleProcessSOuterSize() const { return this->singleProcessSOuterSize; }
    void set_singleProcessSOuterSize(uint32_t singleProcessSOuterSizeParam) { this->singleProcessSOuterSize = singleProcessSOuterSizeParam; }

    uint32_t multiSmaxsInnerLoopTimes = 0;
    uint32_t get_multiSmaxsInnerLoopTimes() const { return this->multiSmaxsInnerLoopTimes; }
    void set_multiSmaxsInnerLoopTimes(uint32_t multiSmaxsInnerLoopTimesParam) { this->multiSmaxsInnerLoopTimes = multiSmaxsInnerLoopTimesParam; }

    uint32_t actualCoreNums = 0;
    uint32_t get_actualCoreNums() const { return this->actualCoreNums; }
    void set_actualCoreNums(uint32_t actualCoreNumsParam) { this->actualCoreNums = actualCoreNumsParam; }

    uint32_t pseShiftBatch = 0;
    uint32_t get_pseShiftBatch() const { return this->pseShiftBatch; }
    void set_pseShiftBatch(uint32_t pseShiftBatchParam) { this->pseShiftBatch = pseShiftBatchParam; }

    uint32_t attenMaskBatch = 0;
    uint32_t get_attenMaskBatch() const { return this->attenMaskBatch; }
    void set_attenMaskBatch(uint32_t attenMaskBatchParam) { this->attenMaskBatch = attenMaskBatchParam; }

    uint32_t kvAntiquantSInnerSize = 0;
    uint32_t get_kvAntiquantSInnerSize() const { return this->kvAntiquantSInnerSize; }
    void set_kvAntiquantSInnerSize(uint32_t kvAntiquantSInnerSizeParam) { this->kvAntiquantSInnerSize = kvAntiquantSInnerSizeParam; }
};

class PromptAttentionSingleCoreTensorSize {
public:
    uint32_t mmResUbSize = 0;
    uint32_t get_mmResUbSize() const { return this->mmResUbSize; }
    void set_mmResUbSize(uint32_t mmResUbSizeParam) { this->mmResUbSize = mmResUbSizeParam; }

    uint32_t pseShiftUbSize = 0;
    uint32_t get_pseShiftUbSize() const { return this->pseShiftUbSize; }
    void set_pseShiftUbSize(uint32_t pseShiftUbSizeParam) { this->pseShiftUbSize = pseShiftUbSizeParam; }

    uint32_t attenMaskUbSize = 0;
    uint32_t get_attenMaskUbSize() const { return this->attenMaskUbSize; }
    void set_attenMaskUbSize(uint32_t attenMaskUbSizeParam) { this->attenMaskUbSize = attenMaskUbSizeParam; }

    uint32_t maskSize = 0;
    uint32_t get_maskSize() const { return this->maskSize; }
    void set_maskSize(uint32_t maskSizeParam) { this->maskSize = maskSizeParam; }

    uint32_t softmaxMaxSize = 0;
    uint32_t get_softmaxMaxSize() const { return this->softmaxMaxSize; }
    void set_softmaxMaxSize(uint32_t softmaxMaxSizeParam) { this->softmaxMaxSize = softmaxMaxSizeParam; }

    uint32_t softmaxSumSize = 0;
    uint32_t get_softmaxSumSize() const { return this->softmaxSumSize; }
    void set_softmaxSumSize(uint32_t softmaxSumSizeParam) { this->softmaxSumSize = softmaxSumSizeParam; }

    uint32_t softmaxExpSize = 0;
    uint32_t get_softmaxExpSize() const { return this->softmaxExpSize; }
    void set_softmaxExpSize(uint32_t softmaxExpSizeParam) { this->softmaxExpSize = softmaxExpSizeParam; }

    uint32_t softmaxValueSize = 0;
    uint32_t get_softmaxValueSize() const { return this->softmaxValueSize; }
    void set_softmaxValueSize(uint32_t softmaxValueSizeParam) { this->softmaxValueSize = softmaxValueSizeParam; }

    uint32_t spmTmpSize = 0;
    uint32_t get_spmTmpSize() const { return this->spmTmpSize; }
    void set_spmTmpSize(uint32_t spmTmpSizeParam) { this->spmTmpSize = spmTmpSizeParam; }

    uint32_t scmTmpSize = 0;
    uint32_t get_scmTmpSize() const { return this->scmTmpSize; }
    void set_scmTmpSize(uint32_t scmTmpSizeParam) { this->scmTmpSize = scmTmpSizeParam; }

    uint32_t bmm2ResUbSize = 0;
    uint32_t get_bmm2ResUbSize() const { return this->bmm2ResUbSize; }
    void set_bmm2ResUbSize(uint32_t bmm2ResUbSizeParam) { this->bmm2ResUbSize = bmm2ResUbSizeParam; }

    uint32_t tmpMMResBmm2PreUbSize = 0;
    uint32_t get_tmpMMResBmm2PreUbSize() const { return this->tmpMMResBmm2PreUbSize; }
    void set_tmpMMResBmm2PreUbSize(uint32_t tmpMMResBmm2PreUbSizeParam) { this->tmpMMResBmm2PreUbSize = tmpMMResBmm2PreUbSizeParam; }

    uint32_t tmpSoftmaxBmm2UbSize = 0;
    uint32_t get_tmpSoftmaxBmm2UbSize() const { return this->tmpSoftmaxBmm2UbSize; }
    void set_tmpSoftmaxBmm2UbSize(uint32_t tmpSoftmaxBmm2UbSizeParam) { this->tmpSoftmaxBmm2UbSize = tmpSoftmaxBmm2UbSizeParam; }

    uint32_t selectSpaceUbSize = 0;
    uint32_t get_selectSpaceUbSize() const { return this->selectSpaceUbSize; }
    void set_selectSpaceUbSize(uint32_t selectSpaceUbSizeParam) { this->selectSpaceUbSize = selectSpaceUbSizeParam; }

    uint32_t tmpSoftMaxV2Size = 0;
    uint32_t get_tmpSoftMaxV2Size() const { return this->tmpSoftMaxV2Size; }
    void set_tmpSoftMaxV2Size(uint32_t tmpSoftMaxV2SizeParam) { this->tmpSoftMaxV2Size = tmpSoftMaxV2SizeParam; }

    uint32_t mm1TmpUbSize = 0;
    uint32_t get_mm1TmpUbSize() const { return this->mm1TmpUbSize; }
    void set_mm1TmpUbSize(uint32_t mm1TmpUbSizeParam) { this->mm1TmpUbSize = mm1TmpUbSizeParam; }

    uint32_t mm2TmpUbSize = 0;
    uint32_t get_mm2TmpUbSize() const { return this->mm2TmpUbSize; }
    void set_mm2TmpUbSize(uint32_t mm2TmpUbSizeParam) { this->mm2TmpUbSize = mm2TmpUbSizeParam; }

    uint32_t kvAntiquantUbSize = 0;
    uint32_t get_kvAntiquantUbSize() const { return this->kvAntiquantUbSize; }
    void set_kvAntiquantUbSize(uint32_t kvAntiquantUbSizeParam) { this->kvAntiquantUbSize = kvAntiquantUbSizeParam; }

    uint32_t bmm2ResUbMsdSize = 0;
    uint32_t get_bmm2ResUbMsdSize() const { return this->bmm2ResUbMsdSize; }
    void set_bmm2ResUbMsdSize(uint32_t bmm2ResUbMsdSizeParam) { this->bmm2ResUbMsdSize = bmm2ResUbMsdSizeParam; }

    uint32_t tempBmm2QueueMsdSize = 0;
    uint32_t get_tempBmm2QueueMsdSize() const { return this->tempBmm2QueueMsdSize; }
    void set_tempBmm2QueueMsdSize(uint32_t tempBmm2QueueMsdSizeParam) { this->tempBmm2QueueMsdSize = tempBmm2QueueMsdSizeParam; }

    uint32_t msdInQueueSize = 0;
    uint32_t get_msdInQueueSize() const { return this->msdInQueueSize; }
    void set_msdInQueueSize(uint32_t msdInQueueSizeParam) { this->msdInQueueSize = msdInQueueSizeParam; }

    uint32_t msdQRowSumBuffSize = 0;
    uint32_t get_msdQRowSumBuffSize() const { return this->msdQRowSumBuffSize; }
    void set_msdQRowSumBuffSize(uint32_t msdQRowSumBuffSizeParam) { this->msdQRowSumBuffSize = msdQRowSumBuffSizeParam; }

    uint32_t msdAMaxTmpBuffSize = 0;
    uint32_t get_msdAMaxTmpBuffSize() const { return this->msdAMaxTmpBuffSize; }
    void set_msdAMaxTmpBuffSize(uint32_t msdAMaxTmpBuffSizeParam) { this->msdAMaxTmpBuffSize = msdAMaxTmpBuffSizeParam; }

    uint32_t msdAMaxResBuffSize = 0;
    uint32_t get_msdAMaxResBuffSize() const { return this->msdAMaxResBuffSize; }
    void set_msdAMaxResBuffSize(uint32_t msdAMaxResBuffSizeParam) { this->msdAMaxResBuffSize = msdAMaxResBuffSizeParam; }

    uint32_t msdSoftmaxResAmaxBuffSize = 0;
    uint32_t get_msdSoftmaxResAmaxBuffSize() const { return this->msdSoftmaxResAmaxBuffSize; }
    void set_msdSoftmaxResAmaxBuffSize(uint32_t msdSoftmaxResAmaxBuffSizeParam) { this->msdSoftmaxResAmaxBuffSize = msdSoftmaxResAmaxBuffSizeParam; }

    uint32_t msdSoftmaxRowSumScaleBuffSize = 0;
    uint32_t get_msdSoftmaxRowSumScaleBuffSize() const { return this->msdSoftmaxRowSumScaleBuffSize; }
    void set_msdSoftmaxRowSumScaleBuffSize(uint32_t msdSoftmaxRowSumScaleBuffSizeParam) { this->msdSoftmaxRowSumScaleBuffSize = msdSoftmaxRowSumScaleBuffSizeParam; }

    uint32_t msdScaleBuffSize = 0;
    uint32_t get_msdScaleBuffSize() const { return this->msdScaleBuffSize; }
    void set_msdScaleBuffSize(uint32_t msdScaleBuffSizeParam) { this->msdScaleBuffSize = msdScaleBuffSizeParam; }

    uint32_t msdOffsetBuffSize = 0;
    uint32_t get_msdOffsetBuffSize() const { return this->msdOffsetBuffSize; }
    void set_msdOffsetBuffSize(uint32_t msdOffsetBuffSizeParam) { this->msdOffsetBuffSize = msdOffsetBuffSizeParam; }

    uint32_t msdTmpMm1BuffSize = 0;
    uint32_t get_msdTmpMm1BuffSize() const { return this->msdTmpMm1BuffSize; }
    void set_msdTmpMm1BuffSize(uint32_t msdTmpMm1BuffSizeParam) { this->msdTmpMm1BuffSize = msdTmpMm1BuffSizeParam; }

    uint32_t msdTmpMm2BuffSize = 0;
    uint32_t get_msdTmpMm2BuffSize() const { return this->msdTmpMm2BuffSize; }
    void set_msdTmpMm2BuffSize(uint32_t msdTmpMm2BuffSizeParam) { this->msdTmpMm2BuffSize = msdTmpMm2BuffSizeParam; }

    uint32_t msdOutQueueSize = 0;
    uint32_t get_msdOutQueueSize() const { return this->msdOutQueueSize; }
    void set_msdOutQueueSize(uint32_t msdOutQueueSizeParam) { this->msdOutQueueSize = msdOutQueueSizeParam; }

    uint32_t msdComputeLines = 0;
    uint32_t get_msdComputeLines() const { return this->msdComputeLines; }
    void set_msdComputeLines(uint32_t msdComputeLinesParam) { this->msdComputeLines = msdComputeLinesParam; }
};

class PromptAttentionInitOutputParams {
public:
    uint32_t singleCoreSize = 0;
    uint32_t get_singleCoreSize() const { return this->singleCoreSize; }
    void set_singleCoreSize(uint32_t singleCoreSizeParam) { this->singleCoreSize = singleCoreSizeParam; }

    int64_t totalOutputSize = 0;
    int64_t get_totalOutputSize() const { return this->totalOutputSize; }
    void set_totalOutputSize(int64_t totalOutputSizeParam) { this->totalOutputSize = totalOutputSizeParam; }

    int64_t totalSoftMaxLseOutputSize = 0;
    int64_t get_totalSoftMaxLseOutputSize() const { return this->totalSoftMaxLseOutputSize; }
    void set_totalSoftMaxLseOutputSize(int64_t totalSoftMaxLseOutputSizeParam) { this->totalSoftMaxLseOutputSize = totalSoftMaxLseOutputSizeParam; }

    uint32_t needInit = 0;
    uint32_t get_needInit() const { return this->needInit; }
    void set_needInit(uint32_t needInitParam) { this->needInit = needInitParam; }

    uint32_t isOneN = 0;
    uint32_t get_isOneN() const { return this->isOneN; }
    void set_isOneN(uint32_t isOneNParam) { this->isOneN = isOneNParam; }
};


class PromptFlashAttentionTilingDataV2 {
public:
    TCubeTiling bmm1TilingDataRect;
    TCubeTiling bmm2TilingDataRect;

    PromptAttentionBaseParams promptAttentionBaseParams;
    PromptAttentionSeqParams promptAttentionSeqParams;
    PromptAttentionSingleCoreParams promptAttentionSingleCoreParams;
    PromptAttentionSingleCoreTensorSize promptAttentionTensorSizeRect;
    PromptAttentionInitOutputParams promptAttentionInitOutputParams;

    SoftMaxTiling softmaxTilingDataRect;
    SoftMaxTiling softmaxFlashTilingDataRect;
    // CopyTransposeTiling transposeTilingDataRect;
};

}
#endif