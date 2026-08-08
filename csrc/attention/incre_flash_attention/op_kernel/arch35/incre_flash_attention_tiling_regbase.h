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
 * \file incre_flash_attention_tiling_regbase.h
 * \brief
 */
 

#ifndef INCRE_FLASH_ATTENTION_TILING_REGBASE_H
#define INCRE_FLASH_ATTENTION_TILING_REGBASE_H

#include "kernel_tiling/kernel_tiling.h"
namespace optiling {

class IncreFlashAttentionBaseParams {
public:
    uint32_t batchSize = 0;
    uint32_t get_batchSize() const { return this->batchSize; }
    void set_batchSize(uint32_t batchSizeParam) { this->batchSize = batchSizeParam; }

    uint32_t seqSize = 0;
    uint32_t get_seqSize() const { return this->seqSize; }
    void set_seqSize(uint32_t seqSizeParam) { this->seqSize = seqSizeParam; }

    uint32_t qSeqSize = 0;
    uint32_t get_qSeqSize() const { return this->qSeqSize; }
    void set_qSeqSize(uint32_t qSeqSizeParam) { this->qSeqSize = qSeqSizeParam; }

    uint32_t headSize = 0;
    uint32_t get_headSize() const { return this->headSize; }
    void set_headSize(uint32_t headSizeParam) { this->headSize = headSizeParam; }

    uint32_t headSizeV = 0;
    uint32_t get_headSizeV() const { return this->headSizeV; }
    void set_headSizeV(uint32_t headSizeVParam) { this->headSizeV = headSizeVParam; }

    uint32_t blockSize = 0;
    uint32_t get_blockSize() const { return this->blockSize; }
    void set_blockSize(uint32_t blockSizeParam) { this->blockSize = blockSizeParam; }

    uint32_t maxBlockNumPerBatch = 0;
    uint32_t get_maxBlockNumPerBatch() const { return this->maxBlockNumPerBatch; }
    void set_maxBlockNumPerBatch(uint32_t maxBlockNumPerBatchParam) { this->maxBlockNumPerBatch = maxBlockNumPerBatchParam; }

    uint32_t maxBlockNumPerSeq = 0;
    uint32_t get_maxBlockNumPerSeq() const { return this->maxBlockNumPerSeq; }
    void set_maxBlockNumPerSeq(uint32_t maxBlockNumPerSeqParam) { this->maxBlockNumPerSeq = maxBlockNumPerSeqParam; }

    float scaleValue = 0;
    float get_scaleValue() const { return this->scaleValue; }
    void set_scaleValue(float scaleValueParam) { this->scaleValue = scaleValueParam; }

    uint32_t kvHeadNum = 0;
    uint32_t get_kvHeadNum() const { return this->kvHeadNum; }
    void set_kvHeadNum(uint32_t kvHeadNumParam) { this->kvHeadNum = kvHeadNumParam; }

    uint32_t headNumRatio = 0;
    uint32_t get_headNumRatio() const { return this->headNumRatio; }
    void set_headNumRatio(uint32_t headNumRatioParam) { this->headNumRatio = headNumRatioParam; }

    uint32_t qHeadNum = 0;
    uint32_t get_qHeadNum() const { return this->qHeadNum; }
    void set_qHeadNum(uint32_t qHeadNumParam) { this->qHeadNum = qHeadNumParam; }

    uint32_t nNumOfQInOneGroup = 0;
    uint32_t get_nNumOfQInOneGroup() const { return this->nNumOfQInOneGroup; }
    void set_nNumOfQInOneGroup(uint32_t nNumOfQInOneGroupParam) { this->nNumOfQInOneGroup = nNumOfQInOneGroupParam; }

    uint32_t batchContinuousFlag = 0;
    uint32_t get_batchContinuousFlag() const { return this->batchContinuousFlag; }
    void set_batchContinuousFlag(uint32_t batchContinuousFlagParam) { this->batchContinuousFlag = batchContinuousFlagParam; }

    uint32_t pseShiftFlag = 0;
    uint32_t get_pseShiftFlag() const { return this->pseShiftFlag; }
    void set_pseShiftFlag(uint32_t pseShiftFlagParam) { this->pseShiftFlag = pseShiftFlagParam; }

    uint32_t pseShiftB = 0;
    uint32_t get_pseShiftB() const { return this->pseShiftB; }
    void set_pseShiftB(uint32_t pseShiftBParam) { this->pseShiftB = pseShiftBParam; }

    uint32_t pseShiftS = 0;
    uint32_t get_pseShiftS() const { return this->pseShiftS; }
    void set_pseShiftS(uint32_t pseShiftSParam) { this->pseShiftS = pseShiftSParam; }

    uint32_t pseShiftS0 = 0;
    uint32_t get_pseShiftS0() const { return this->pseShiftS0; }
    void set_pseShiftS0(uint32_t pseShiftS0Param) { this->pseShiftS0 = pseShiftS0Param; }

    uint32_t selectWithByteMaskTmpMinSize = 0;
    uint32_t get_selectWithByteMaskTmpMinSize() const { return this->selectWithByteMaskTmpMinSize; }
    void set_selectWithByteMaskTmpMinSize(uint32_t selectWithByteMaskTmpMinSizeParam) { this->selectWithByteMaskTmpMinSize = selectWithByteMaskTmpMinSizeParam; }

    uint32_t actualLenQDims = 0;
    uint32_t get_actualLenQDims() const { return this->actualLenQDims; }
    void set_actualLenQDims(uint32_t actualLenQDimsParam) { this->actualLenQDims = actualLenQDimsParam; }

    uint32_t actualLenDims = 0;
    uint32_t get_actualLenDims() const { return this->actualLenDims; }
    void set_actualLenDims(uint32_t actualLenDimsParam) { this->actualLenDims = actualLenDimsParam; }

    uint32_t qPaddingFlag = 0;
    uint32_t get_qPaddingFlag() const { return this->qPaddingFlag; }
    void set_qPaddingFlag(uint32_t qPaddingFlagParam) { this->qPaddingFlag = qPaddingFlagParam; }

    uint32_t kvPaddingFlag = 0;
    uint32_t get_kvPaddingFlag() const { return this->kvPaddingFlag; }
    void set_kvPaddingFlag(uint32_t kvPaddingFlagParam) { this->kvPaddingFlag = kvPaddingFlagParam; }

    uint32_t msdIterNum = 0;
    uint32_t get_msdIterNum() const { return this->msdIterNum; }
    void set_msdIterNum(uint32_t msdIterNumParam) { this->msdIterNum = msdIterNumParam; }

    uint32_t l2CacheOffFlag = 0;
    uint32_t get_l2CacheOffFlag() const { return this->l2CacheOffFlag; }
    void set_l2CacheOffFlag(uint32_t l2CacheOffFlagParam) { this->l2CacheOffFlag = l2CacheOffFlagParam; }

    uint32_t antiquantPerTensorFlag = 0;
    uint32_t get_antiquantPerTensorFlag() const { return this->antiquantPerTensorFlag; }
    void set_antiquantPerTensorFlag(uint32_t antiquantPerTensorFlagParam) { this->antiquantPerTensorFlag = antiquantPerTensorFlagParam; }

    uint32_t antiquantPerHeadFlag = 0;
    uint32_t get_antiquantPerHeadFlag() const { return this->antiquantPerHeadFlag; }
    void set_antiquantPerHeadFlag(uint32_t antiquantPerHeadFlagParam) { this->antiquantPerHeadFlag = antiquantPerHeadFlagParam; }

    uint32_t antiquantParamsInPagedAttentionFlag = 0;
    uint32_t get_antiquantParamsInPagedAttentionFlag() const { return this->antiquantParamsInPagedAttentionFlag; }
    void set_antiquantParamsInPagedAttentionFlag(uint32_t antiquantParamsInPagedAttentionFlagParam) { this->antiquantParamsInPagedAttentionFlag = antiquantParamsInPagedAttentionFlagParam; }

    uint32_t attenMaskFlag = 0;
    uint32_t get_attenMaskFlag() const { return this->attenMaskFlag; }
    void set_attenMaskFlag(uint32_t attenMaskFlagParam) { this->attenMaskFlag = attenMaskFlagParam; }

    uint32_t attenMaskBatch = 0;
    uint32_t get_attenMaskBatch() const { return this->attenMaskBatch; }
    void set_attenMaskBatch(uint32_t attenMaskBatchParam) { this->attenMaskBatch = attenMaskBatchParam; }

    uint32_t attenMaskQSize = 0;
    uint32_t get_attenMaskQSize() const { return this->attenMaskQSize; }
    void set_attenMaskQSize(uint32_t attenMaskQSizeParam) { this->attenMaskQSize = attenMaskQSizeParam; }

    uint32_t attenMaskSize = 0;
    uint32_t get_attenMaskSize() const { return this->attenMaskSize; }
    void set_attenMaskSize(uint32_t attenMaskSizeParam) { this->attenMaskSize = attenMaskSizeParam; }

    uint32_t softmaxLseFlag = 0;
    uint32_t get_softmaxLseFlag() const { return this->softmaxLseFlag; }
    void set_softmaxLseFlag(uint32_t softmaxLseFlagParam) { this->softmaxLseFlag = softmaxLseFlagParam; }

    uint32_t totalBlockNum = 0;
    uint32_t get_totalBlockNum() const { return this->totalBlockNum; }
    void set_totalBlockNum(uint32_t totalBlockNumParam) { this->totalBlockNum = totalBlockNumParam; }

    uint32_t paKvShapeType = 0;
    uint32_t get_paKvShapeType() const { return this->paKvShapeType; }
    void set_paKvShapeType(uint32_t paKvShapeTypeParam) { this->paKvShapeType = paKvShapeTypeParam; }

    uint32_t antiqSeqSize = 0;
    uint32_t get_antiqSeqSize() const { return this->antiqSeqSize; }
    void set_antiqSeqSize(uint32_t antiqSeqSizeParam) { this->antiqSeqSize = antiqSeqSizeParam; }

    int32_t preToken = 0;
    int32_t get_preToken() const { return this->preToken; }
    void set_preToken(int32_t preTokenParam) { this->preToken = preTokenParam; }

    int32_t nextToken = 0;
    int32_t get_nextToken() const { return this->nextToken; }
    void set_nextToken(int32_t nextTokenParam) { this->nextToken = nextTokenParam; }

    uint32_t isRowInvalid = 0;
    uint32_t get_isRowInvalid() const { return this->isRowInvalid; }
    void set_isRowInvalid(uint32_t isRowInvalidParam) { this->isRowInvalid = isRowInvalidParam; }

    uint32_t sparseMode = 0;
    uint32_t get_sparseMode() const { return this->sparseMode; }
    void set_sparseMode(uint32_t sparseModeParam) { this->sparseMode = sparseModeParam; }

    uint32_t slidingFlag = 0;
    uint32_t get_slidingFlag() const { return this->slidingFlag; }
    void set_slidingFlag(uint32_t slidingFlagParam) { this->slidingFlag = slidingFlagParam; }

    int64_t windowSize = 0;
    int64_t get_windowSize() const { return this->windowSize; }
    void set_windowSize(int64_t windowSizeParam) { this->windowSize = windowSizeParam; }
};

class IncreFlashAttentionSplitKVParams {
public:
    uint32_t s2 = 0;
    uint32_t get_s2() const { return this->s2; }
    void set_s2(uint32_t s2Param) { this->s2 = s2Param; }

    uint32_t sInnerLoopSize = 0;
    uint32_t get_sInnerLoopSize() const { return this->sInnerLoopSize; }
    void set_sInnerLoopSize(uint32_t sInnerLoopSizeParam) { this->sInnerLoopSize = sInnerLoopSizeParam; }

    uint32_t accumOutSize = 0;
    uint32_t get_accumOutSize() const { return this->accumOutSize; }
    void set_accumOutSize(uint32_t accumOutSizeParam) { this->accumOutSize = accumOutSizeParam; }

    uint32_t logSumExpSize = 0;
    uint32_t get_logSumExpSize() const { return this->logSumExpSize; }
    void set_logSumExpSize(uint32_t logSumExpSizeParam) { this->logSumExpSize = logSumExpSizeParam; }
};

class IncreFlashAttentionCoreParams {
public:
    uint32_t coreSidxEnd[50] = {0}; // 50:MAX_CORE_NUM of 910b coreSidxEnd数组首地址要保证8字节对齐
    uint32_t* get_coreSidxEndPtr() { return coreSidxEnd; }
    void set_coreSidxEnd(const uint32_t* val) {
        for (int i = 0; i < 50; ++i) {
            this->coreSidxEnd[i] = val[i];
        }
    }

    uint32_t coreSidxEndRegbase[66] = {0}; // 66:MAX_CORE_NUM of Ascend 950
    uint32_t* get_coreSidxEndRegbasePtr() { return coreSidxEndRegbase; }
    void set_coreSidxEndRegbase(const uint32_t* val) {
        for (int i = 0; i < 66; ++i) {
            this->coreSidxEndRegbase[i] = val[i];
        }
    }

    uint32_t coreSposStartRegbase[66] = {0}; // 66:MAX_CORE_NUM of Ascend 950
    uint32_t* get_coreSposStartRegbasePtr() { return coreSposStartRegbase; }
    void set_coreSposStartRegbase(const uint32_t* val) {
        for (int i = 0; i < 66; ++i) {
            this->coreSposStartRegbase[i] = val[i];
        }
    }
};

class IncreFlashAttentionSingleCoreParams {
public:
    uint32_t sInnerLoopTimes = 0;
    uint32_t get_sInnerLoopTimes() const { return this->sInnerLoopTimes; }
    void set_sInnerLoopTimes(uint32_t sInnerLoopTimesParam) { this->sInnerLoopTimes = sInnerLoopTimesParam; }

    uint32_t singleProcessSInnerSize = 0;
    uint32_t get_singleProcessSInnerSize() const { return this->singleProcessSInnerSize; }
    void set_singleProcessSInnerSize(uint32_t singleProcessSInnerSizeParam) { this->singleProcessSInnerSize = singleProcessSInnerSizeParam; }

    uint32_t singleProcessSInnerSizeTail = 0;
    uint32_t get_singleProcessSInnerSizeTail() const { return this->singleProcessSInnerSizeTail; }
    void set_singleProcessSInnerSizeTail(uint32_t singleProcessSInnerSizeTailParam) { this->singleProcessSInnerSizeTail = singleProcessSInnerSizeTailParam; }

    uint32_t usedCoreNum = 0;
    uint32_t get_usedCoreNum() const { return this->usedCoreNum; }
    void set_usedCoreNum(uint32_t usedCoreNumParam) { this->usedCoreNum = usedCoreNumParam; }

    uint32_t formerCoreNum = 0;
    uint32_t get_formerCoreNum() const { return this->formerCoreNum; }
    void set_formerCoreNum(uint32_t formerCoreNumParam) { this->formerCoreNum = formerCoreNumParam; }

    uint32_t blockSplitBn2Range = 0;
    uint32_t get_blockSplitBn2Range() const { return this->blockSplitBn2Range; }
    void set_blockSplitBn2Range(uint32_t blockSplitBn2RangeParam) { this->blockSplitBn2Range = blockSplitBn2RangeParam; }

    uint32_t tailSplitedBatchRange = 0;
    uint32_t get_tailSplitedBatchRange() const { return this->tailSplitedBatchRange; }
    void set_tailSplitedBatchRange(uint32_t tailSplitedBatchRangeParam) { this->tailSplitedBatchRange = tailSplitedBatchRangeParam; }

    uint32_t groupSplitSize = 0;
    uint32_t get_groupSplitSize() const { return this->groupSplitSize; }
    void set_groupSplitSize(uint32_t groupSplitSizeParam) { this->groupSplitSize = groupSplitSizeParam; }

    uint32_t s1SplitSize = 0;
    uint32_t get_s1SplitSize() const { return this->s1SplitSize; }
    void set_s1SplitSize(uint32_t s1SplitSizeParam) { this->s1SplitSize = s1SplitSizeParam; }
};

class IncreFlashAttentionSingleCoreTensorSize {
public:
    uint32_t mmResUbSize = 0;
    uint32_t get_mmResUbSize() const { return this->mmResUbSize; }
    void set_mmResUbSize(uint32_t mmResUbSizeParam) { this->mmResUbSize = mmResUbSizeParam; }

    uint32_t bmm2ResUbSize = 0;
    uint32_t get_bmm2ResUbSize() const { return this->bmm2ResUbSize; }
    void set_bmm2ResUbSize(uint32_t bmm2ResUbSizeParam) { this->bmm2ResUbSize = bmm2ResUbSizeParam; }
};

class IncreFlashAttentionInitOutputParams {
public:
    uint32_t isPerChnOut = 0;
    uint32_t get_isPerChnOut() const { return this->isPerChnOut; }
    void set_isPerChnOut(uint32_t isPerChnOutParam) { this->isPerChnOut = isPerChnOutParam; }

    uint32_t isOutQuantTypeBf16 = 0;
    uint32_t get_isOutQuantTypeBf16() const { return this->isOutQuantTypeBf16; }
    void set_isOutQuantTypeBf16(uint32_t isOutQuantTypeBf16Param) { this->isOutQuantTypeBf16 = isOutQuantTypeBf16Param; }

    uint32_t singleCoreSize = 0;
    uint32_t get_singleCoreSize() const { return this->singleCoreSize; }
    void set_singleCoreSize(uint32_t singleCoreSizeParam) { this->singleCoreSize = singleCoreSizeParam; }

    uint32_t singleCoreLseSize = 0;
    uint32_t get_singleCoreLseSize() const { return this->singleCoreLseSize; }
    void set_singleCoreLseSize(uint32_t singleCoreLseSizeParam) { this->singleCoreLseSize = singleCoreLseSizeParam; }

    int64_t totalOutputSize = 0;
    int64_t get_totalOutputSize() const { return this->totalOutputSize; }
    void set_totalOutputSize(int64_t totalOutputSizeParam) { this->totalOutputSize = totalOutputSizeParam; }

    int64_t totalLseOutputSize = 0;
    int64_t get_totalLseOutputSize() const { return this->totalLseOutputSize; }
    void set_totalLseOutputSize(int64_t totalLseOutputSizeParam) { this->totalLseOutputSize = totalLseOutputSizeParam; }

    uint32_t needInit = 0;
    uint32_t get_needInit() const { return this->needInit; }
    void set_needInit(uint32_t needInitParam) { this->needInit = needInitParam; }

    uint32_t isBSNDOut = 0;
    uint32_t get_isBSNDOut() const { return this->isBSNDOut; }
    void set_isBSNDOut(uint32_t isBSNDOutParam) { this->isBSNDOut = isBSNDOutParam; }
};

class IncreFlashAttentionTilingDataRegbase {
public:
    TCubeTiling bmm1TilingData;
    TCubeTiling bmm2TilingData;
    IncreFlashAttentionBaseParams baseParams;
    IncreFlashAttentionSplitKVParams splitKVParams;
    IncreFlashAttentionCoreParams increFlashAttentionCoreParams;
    IncreFlashAttentionSingleCoreParams increFlashAttentionSingleCoreParams;
    IncreFlashAttentionSingleCoreTensorSize increFlashAttentionSingleCoreTensorSize;
    SoftMaxTiling softmaxFlashTilingData;
    IncreFlashAttentionInitOutputParams outputParams;
};

class IncreFlashAttentionTilingDataPrefix {
public:
    IncreFlashAttentionTilingDataRegbase base;
    uint64_t prefixAttenOutOffset = 0; // 临时输出偏移;
    uint64_t get_prefixAttenOutOffset() const { return this->prefixAttenOutOffset; }
    void set_prefixAttenOutOffset(uint64_t prefixAttenOutOffsetParam) { this->prefixAttenOutOffset = prefixAttenOutOffsetParam; }

    uint64_t userPromptAttenOutOffset = 0;
    uint64_t get_userPromptAttenOutOffset() const { return this->userPromptAttenOutOffset; }
    void set_userPromptAttenOutOffset(uint64_t userPromptAttenOutOffsetParam) { this->userPromptAttenOutOffset = userPromptAttenOutOffsetParam; }

    uint64_t tmpLseOffset = 0;
    uint64_t get_tmpLseOffset() const { return this->tmpLseOffset; }
    void set_tmpLseOffset(uint64_t tmpLseOffsetParam) { this->tmpLseOffset = tmpLseOffsetParam; }

    uint64_t prefixLen = 0; // prefix 长度;
    uint64_t get_prefixLen() const { return this->prefixLen; }
    void set_prefixLen(uint64_t prefixLenParam) { this->prefixLen  = prefixLenParam; }

    uint32_t formerCoreNum = 0; // combine 分核参数，参考普通bn分核流程，总数不超过NumBlocks;
    uint32_t get_formerCoreNum() const { return this->formerCoreNum; }
    void set_formerCoreNum(uint32_t formerCoreNumParam) { this->formerCoreNum = formerCoreNumParam; }

    uint32_t blockSplitBn2Range = 0;
    uint32_t get_blockSplitBn2Range() const { return this->blockSplitBn2Range; }
    void set_blockSplitBn2Range(uint32_t blockSplitBn2RangeParam) { this->blockSplitBn2Range = blockSplitBn2RangeParam; }

    uint32_t tailSplitedBatchRange = 0;
    uint32_t get_tailSplitedBatchRange() const { return this->tailSplitedBatchRange; }
    void set_tailSplitedBatchRange(uint32_t tailSplitedBatchRangeParam) { this->tailSplitedBatchRange = tailSplitedBatchRangeParam; }

    uint32_t usedCoreNum = 0;
    uint32_t get_usedCoreNum() const { return this->usedCoreNum; }
    void set_usedCoreNum(uint32_t usedCoreNumParam) { this->usedCoreNum = usedCoreNumParam; }

    uint32_t batchSizeQ = 0;
    uint32_t get_batchSizeQ() const { return this->batchSizeQ; }
    void set_batchSizeQ(uint32_t batchSizeQParam) { this->batchSizeQ = batchSizeQParam; }
};

class IncreFlashAttentionTilingDataV2 {
public:
    IncreFlashAttentionTilingDataRegbase tilingBase;
    IncreFlashAttentionTilingDataPrefix tilingPrefix;
};
}
#endif