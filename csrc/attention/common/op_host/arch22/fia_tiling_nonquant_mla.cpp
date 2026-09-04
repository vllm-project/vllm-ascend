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
 * \file fia_tiling_nonquant_mla.cpp
 * \brief
 */

#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "err/ops_err.h"
#include "fia_tiling_nonquant_mla.h"
#include "../fia_tiling_templates_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

constexpr uint64_t PRE_LOAD_NUM_MLA = 2;

constexpr uint64_t FIA_TILINGKEYOFFSET = uint64_t(100000000000000000UL); // 10^17
constexpr uint64_t FIA_PERF_MODE_TILINGKEYOFFSET = uint64_t(1000000000000000UL); // 10^15

template <typename T> 
inline auto Align(T num, T rnd) -> T
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

constexpr uint64_t RecursiveSum()
{
    return 0;
}

template <typename T, typename... Args> 
constexpr uint64_t RecursiveSum(T templateId, Args... templateIds)
{
    return static_cast<uint64_t>(templateId) + 10U * RecursiveSum(templateIds...);
}
template <typename... Args> 
constexpr uint64_t FIA_GET_TILINGKEY(Args... templateIds)
{
    return RecursiveSum(templateIds...);
}

void FiaTilingNonQuantMla::InitTilingInfo(TilingInfo *tilingInfo)
{
    fiaInfo_ = static_cast<FiaTilingInfo *>(tilingInfo);
}

ge::graphStatus FiaTilingNonQuantMla::GetPlatformInfo()
{
    OP_CHECK_IF(fiaInfo_->platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(fiaInfo_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(fiaInfo_->platformInfo);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();

    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0,
        OPS_REPORT_VECTOR_INNER_ERR(fiaInfo_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    // 设置CV1:1模式
    cvRatio_ = aivNum_ / aicNum_;
    OP_LOGI(fiaInfo_->opName, "FIA aicNum: %u, aivNum:%u, cvRatio:%u.", aicNum_, aivNum_, cvRatio_);

    return ge::GRAPH_SUCCESS;
}

bool FiaTilingNonQuantMla::IsCapable()
{
    // 不支持解析为空或空Tensor
    if ((fiaInfo_ == nullptr) || (fiaInfo_->emptyTensorFlag)) {
        return false;
    }

    ge::DataType qDataType = fiaInfo_->inputQType;
    ge::DataType kDataType = fiaInfo_->inputKvType;
    // 仅支持非量化
    if ((qDataType != ge::DT_FLOAT16 && qDataType != ge::DT_BF16) || (qDataType != kDataType)) {
        return false;
    }

    // 不支持的特性
    if (fiaInfo_->learnableSinkFlag || fiaInfo_->sysPrefixFlag || fiaInfo_->pseShiftFlag ||
        fiaInfo_->qPaddingSizeFlag || fiaInfo_->kvPaddingSizeFlag) {
        return false;
    }

    // 不支持后量化
    if (fiaInfo_->outputType == ge::DT_INT8) {
        return false;
    }

    // 不支持TensorList
    if (fiaInfo_->kvStorageMode == KvStorageMode::TENSOR_LIST) {
        return false;
    }

    // 仅支持ROPE分开传输
    if (fiaInfo_->ropeMode != RopeMode::ROPE_SPLIT) {
        return false;
    }

    // 仅支持Q&K&V的HeadDim为512, ROPE的HeadDim为64
    constexpr uint32_t QK_HEAD_DIM_512 = 512U;
    constexpr uint32_t ROPE_HEAD_DIM_64 = 64U;
    if ((fiaInfo_->qkHeadDim != fiaInfo_->vHeadDim) ||
        (fiaInfo_->qkHeadDim != QK_HEAD_DIM_512) ||
        (fiaInfo_->ropeHeadDim != ROPE_HEAD_DIM_64)) {
        return false;
    }

    // 仅支持KV_N为1
    if (fiaInfo_->n2Size != 1U) {
        return false;
    }

    // 支持的input_layout范围
    std::string layout = fiaInfo_->opParamInfo.layOut;
    const std::vector<std::string> layoutSupportList = {
        "BSH", "BSND", "BNSD", "TND", "BSH_NBSD", "BSND_NBSD", "BNSD_NBSD", "TND_NTD"
    };
    if (std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end()) {
        return false;
    }

    // 支持的sparse_mode值

    if ((fiaInfo_->sparseMode != SPARSE_MODE_NO_MASK) && (fiaInfo_->sparseMode != SPARSE_MODE_RIGHT_DOWN) && 
        (fiaInfo_->sparseMode != SPARSE_MODE_BAND) && (fiaInfo_->sparseMode != SPARSE_MODE_TREE)) {
        return false;
    }

    // 支持的G的取值范围
    std::vector<uint32_t> gSizeSupportList = {1, 2, 4, 8, 16, 32, 64, 128};
    if (std::find(gSizeSupportList.begin(), gSizeSupportList.end(), fiaInfo_->gSize) == gSizeSupportList.end()) {
        return false;
    }

    return true;
}

void FiaTilingNonQuantMla::GenTilingKey()
{
    uint8_t layoutVal{0};
    uint8_t inputQVal{0};
    uint8_t inputKvVal{0};
    uint8_t outputVal{0};
    uint8_t originVal{0};
    uint8_t splitKvVal = kvSplit_ > 0U ? 1U : 0U;
    uint8_t paVal = (fiaInfo_->pageAttentionFlag && fiaInfo_->s2Size != static_cast<int64_t>(0)) ?
        static_cast<uint8_t>(1 * 2) : static_cast<uint8_t>(0);
    uint8_t antiquantModeVal = 0;
    uint64_t modeVal = fiaInfo_->sysPrefixFlag ? 2U : 1U;
    uint8_t kvLayoutVal = 0;
    uint8_t cvRatioVal = (cvRatio_ == 1) ? 1 : 0; // CV1:1场景为1，其他场景为0
    uint8_t enableTreeVal = (fiaInfo_->sparseMode == SPARSE_MODE_TREE) ? 2 : 0; // sparse_mode为 tree时为2，其他场景为0

    const std::map<TilingKeyLayout, uint8_t> kvLayoutMap = {
        {TilingKeyLayout::BNSD, 0U}, {TilingKeyLayout::BSH_BSND, 1U}, {TilingKeyLayout::NZ, 2U},
        {TilingKeyLayout::TND, 3U}
    };

    const std::map<TilingKeyLayout, uint8_t> qLayoutMap = {
        {TilingKeyLayout::BNSD, 0U}, {TilingKeyLayout::BSH_BSND, 1U}, {TilingKeyLayout::TND, 2U}
    };

    const std::map<ge::DataType, uint8_t> typeMap = {
        {ge::DT_FLOAT16, 0U}, {ge::DT_BF16, 2U}, {ge::DT_INT8, 3U}, {ge::DT_INT4, 4U},
    };

    if (qLayoutMap.find(fiaInfo_->inputLayout) != qLayoutMap.end()) {
        layoutVal = qLayoutMap.at(fiaInfo_->inputLayout);
    }
    if (kvLayoutMap.find(fiaInfo_->inputKvLayout) != kvLayoutMap.end()) {
        kvLayoutVal = kvLayoutMap.at(fiaInfo_->inputKvLayout);
    }
    if (typeMap.find(fiaInfo_->inputQType) != typeMap.end()) {
        inputQVal = typeMap.at(fiaInfo_->inputQType);
    }
    if (typeMap.find(fiaInfo_->inputKvType) != typeMap.end()) {
        inputKvVal = typeMap.at(fiaInfo_->inputKvType);
    }
    if (typeMap.find(fiaInfo_->outputType) != typeMap.end()) {
        outputVal = typeMap.at(fiaInfo_->outputType);
    }

    originVal = inputQVal;
    uint64_t baseOffset =
        modeVal * FIA_TILINGKEYOFFSET + (static_cast<uint64_t>(perfMode_)) * FIA_PERF_MODE_TILINGKEYOFFSET;
    tilingKey_ = baseOffset + FIA_GET_TILINGKEY(layoutVal, inputQVal, inputKvVal, outputVal, originVal,
        (paVal + splitKvVal), antiquantModeVal, kvLayoutVal, (cvRatioVal + enableTreeVal));

    OP_LOGI(fiaInfo_->opName, "FIA tilingKey_: %lu.", tilingKey_);
}

bool FiaTilingNonQuantMla::IsFlashDecode()
{
    uint32_t tndFDCoreArrLen = tilingData_.fdParams.get_numOfFdHead();
    return tndFDCoreArrLen > 0U;
}

bool FiaTilingNonQuantMla::DealSameSeqEachBatch() const
{
    if (!fiaInfo_->batchContinuousFlag) {
        if (fiaInfo_->actualSeqLenFlag) {
            return fiaInfo_->isSameActualseq;
        } else {
            return fiaInfo_->isSameSeqAllKVTensor;
        }
    } else {
        return fiaInfo_->isSameActualseq;
    }
}

void FiaTilingNonQuantMla::EmptyTensorProcess() const
{
    if (fiaInfo_->s2Size == 0) {
        /*
         * 1024，空tensor场景下，作为默认值完成后续计算
         * 避免matmal tiling  softmax tiling异常
         * kernel计算使用真实的seqSize=0, 与actuseq_len流程归一
         */
        fiaInfo_->s2Size = 1024;
    }
}

void FiaTilingNonQuantMla::InitParams()
{
    perfMode_ = FiaTemplateId::HIGH_PERFORMANCE_MLA;
    coreNum_ = aicNum_;
    numBlocks_ = aicNum_; // Tiling下沉首次Tiling也会校验numBlocks_是否为0，为避免拦截报错，将numBlocks_设置为aicNum_，实际不生效

    headDimAlign_ = Align(fiaInfo_->qkHeadDim, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    EmptyTensorProcess();
}

void FiaTilingNonQuantMla::CalcInnerSize(uint32_t seqSize)
{
    sInnerSize_ = 512U;
    // FlashDecode时，如果S2的计算量>=256(确保切分后不小于128)但又不足以分2次计算时，则修改sInnerSize_，均分为2份进行计算，确保Nbuffer=2
    if (splitKVFlag_ && fiaInfo_->inputLayout != TilingKeyLayout::TND) {
        if (seqSize == 256U) {
            sInnerSize_ = 128U;
        } else if (seqSize > 256U && seqSize <= sInnerSize_) {
            sInnerSize_ = (sInnerSize_ + 1U) / 2U;
        }
    }

    // PA特性泛化场景，blockSize可能为112等值，无法被sInnerSize_整除，当step*base跨block时，搬运处理复杂，通过向下对齐避免
    if (fiaInfo_->pageAttentionFlag && fiaInfo_->blockSize != 0) {
        uint32_t blockSize = static_cast<uint32_t>(fiaInfo_->blockSize);
        if ((sInnerSize_ > blockSize) && (sInnerSize_ % blockSize != 0)) {
            sInnerSize_ = (sInnerSize_ / blockSize) * blockSize;
        }
    }

    sInnerLoopTimes_ = (seqSize + sInnerSize_ - 1U) / sInnerSize_;
    sInnerSizeTail_ = seqSize - (sInnerLoopTimes_ - 1U) * sInnerSize_;
    // tiling下沉 && flash decoder场景时，sInnerSize_基块大小不按照真实值修改
    // 否则会导致 tiling下沉 && flash decoder 场景时开辟workspace空间大小小于真实运行时所需的workspace大小
    if (sInnerSize_ > seqSize) {
        sInnerSize_ = seqSize;
    }
    sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐

    CalcMBaseSize();
}

/**
 * @brief 根据shape调整mBaseSize，在一定范围内可以不开FD
 */
void FiaTilingNonQuantMla::CalcMBaseSize()
{
    uint32_t bN2 = fiaInfo_->bSize * fiaInfo_->n2Size;
    uint32_t s1G = fiaInfo_->s1Size * fiaInfo_->gSize;
    if (bN2 == 0U || aicNum_ == 0U) {
        return;
    }
    // 若BN2能整除核数或者KVS不等长，不考虑调整mBaseSize
    if (bN2 % aicNum_ == 0U || !DealSameSeqEachBatch()) {
        return;
    }
    
    // 求B*N2与核数的最小公倍数
    uint32_t originalA = bN2;
    uint32_t originalB = aicNum_;
    // 欧几里得算法计算最大公约数
    while (originalB != 0U) {
        originalA %= originalB;
        std::swap(originalA, originalB);
    }
    
    // 计算最小公倍数
    uint32_t lcm = (bN2 / originalA) * aicNum_;
    uint32_t splitOfS1G = lcm / bN2;    // 求S1G被切多少份，可以不开FD
    // 如果不能等分S1G，则不切S2无法做到负载不均衡，不修改mBaseSize，直接按照最大mBaseSize进行切分
    if (splitOfS1G == 0U || s1G % splitOfS1G != 0U) {
        return;
    }
    uint32_t mBaseSizeTmp = s1G / splitOfS1G;   // 计算可以等分S1G的mBaseSize
    // 128：最小的mBaseSize，计算的理论mBaseSize需要在原定mBaseSize和最小mBaseSize区间内
    if (mBaseSizeTmp > mBaseSize_ || mBaseSizeTmp < 128U) {
        return;
    }

    mBaseSize_ = mBaseSizeTmp;  // 满足以上条件则更新mBaseSize
}

void FiaTilingNonQuantMla::CreateSplitInput(BaseInfo &baseInfo, SplitParam &splitParam) const
{
    //构造分核输入参数
    baseInfo.bSize = fiaInfo_->bSize;
    baseInfo.n2Size = fiaInfo_->n2Size;
    baseInfo.gSize = fiaInfo_->gSize;
    baseInfo.s2Size = static_cast<uint32_t>(fiaInfo_->s2Size);
    baseInfo.s1Size = fiaInfo_->s1Size;
    baseInfo.actualLenQDims = fiaInfo_->actualLenQDims;
    baseInfo.actualLenKvDims = fiaInfo_->actualLenKvDims;
    baseInfo.preToken = fiaInfo_->preToken;
    baseInfo.nextToken = fiaInfo_->nextToken;
    baseInfo.isS1G = fiaInfo_->inputLayout == TilingKeyLayout::TND ||
        fiaInfo_->inputLayout == TilingKeyLayout::BSH_BSND; // 使用枚举映射
    baseInfo.sparseMode = fiaInfo_->sparseMode;
    baseInfo.attenMaskFlag = fiaInfo_->attenMaskFlag;

    if (fiaInfo_->opParamInfo.actualSeqLengthsQ.tensor != nullptr) {
        baseInfo.isAccumSeqS1 = fiaInfo_->isAccumQSeq;
        baseInfo.actualSeqS1Size.reserve(fiaInfo_->bSize);
        const int64_t *s1Ptr = fiaInfo_->opParamInfo.actualSeqLengthsQ.tensor->GetData<int64_t>();
        for (uint32_t i = 0; i < fiaInfo_->bSize; ++i) {
            baseInfo.actualSeqS1Size.emplace_back(s1Ptr[i]);
        }
    }
    if (fiaInfo_->opParamInfo.actualSeqLengths.tensor != nullptr) {
        baseInfo.isAccumSeqS2 = fiaInfo_->isAccumKVSeq;
        baseInfo.actualSeqS2Size.reserve(fiaInfo_->bSize);
        const int64_t *s2Ptr = fiaInfo_->opParamInfo.actualSeqLengths.tensor->GetData<int64_t>();
        for (uint32_t i = 0; i < fiaInfo_->bSize; ++i) {
            baseInfo.actualSeqS2Size.emplace_back(s2Ptr[i]);
        }
    }
    splitParam.mBaseSize = mBaseSize_;
    splitParam.s2BaseSize = sInnerSize_;
    splitParam.gS1BaseSizeOfFd = mFdBaseSize_;
}

void FiaTilingNonQuantMla::SetSplitOutput(SplitResult &res)
{
    uint32_t *bN2EndPtr = tilingData_.outerSplitParams.get_bN2End();
    uint32_t *gS1EndPtr = tilingData_.outerSplitParams.get_gS1End();
    uint32_t *s2EndPtr = tilingData_.outerSplitParams.get_s2End();
    uint32_t *bN2IdxOfFdHead = tilingData_.fdParams.get_bN2IdxOfFdHead();
    uint32_t *gS1IdxOfFdHead = tilingData_.fdParams.get_gS1IdxOfFdHead();
    uint32_t *s2SplitNumOfFdHead = tilingData_.fdParams.get_s2SplitNumOfFdHead();
    uint32_t *s2SplitStartIdxOfCore = tilingData_.fdParams.get_s2SplitStartIdxOfCore();
    uint32_t *gS1SplitNumOfFdHead = tilingData_.fdParams.get_gS1SplitNumOfFdHead();
    uint32_t *gS1LastPartSizeOfFdHead = tilingData_.fdParams.get_gS1LastPartSizeOfFdHead();
    uint32_t *gS1IdxEndOfFdHeadSplit = tilingData_.fdParams.get_gS1IdxEndOfFdHeadSplit();
    uint32_t *gS1IdxEndOfFdHead = tilingData_.fdParams.get_gS1IdxEndOfFdHead();

    for (uint32_t i = 0; i < res.usedCoreNum; ++i) {
        bN2EndPtr[i] = res.bN2End[i];
        gS1EndPtr[i] = res.gS1End[i];
        s2EndPtr[i] = res.s2End[i];
        bN2IdxOfFdHead[i] = res.fdRes.bN2IdxOfFdHead[i];
        gS1IdxOfFdHead[i] = res.fdRes.gS1IdxOfFdHead[i];
        s2SplitNumOfFdHead[i] = res.fdRes.s2SplitNumOfFdHead[i];
        s2SplitStartIdxOfCore[i] = res.fdRes.s2SplitStartIdxOfCore[i];
        gS1SplitNumOfFdHead[i] = res.fdRes.gS1SplitNumOfFdHead[i];
        gS1LastPartSizeOfFdHead[i] = res.fdRes.gS1LastPartSizeOfFdHead[i];
    }

    for (uint32_t i = 0; i < res.usedCoreNum * 2; ++i) { // 2: cube : vector = 1:2
        gS1IdxEndOfFdHead[i] = res.fdRes.gS1IdxEndOfFdHead[i];
        gS1IdxEndOfFdHeadSplit[i] = res.fdRes.gS1IdxEndOfFdHeadSplit[i];
    }
}

void FiaTilingNonQuantMla::Split()
{
    CalcInnerSize(static_cast<uint32_t>(fiaInfo_->s2Size));

    //构造分核输入参数
    BaseInfo baseInfo {};
    SplitParam splitParam {};
    CreateSplitInput(baseInfo, splitParam);

    //构造分核输出参数
    SplitResult res {aicNum_, cvRatio_};
    SplitCore(aicNum_, baseInfo, splitParam, res);
    if (res.numOfFdHead > aicNum_ || res.usedCoreNum > aicNum_ || res.maxS2SplitNum > aicNum_ + 1U) {
        OP_LOGE(fiaInfo_->opName, "used_core_num: %u, num_of_fd_head: %u, max_s2_split_num: %u, aic_num: %u", 
            res.usedCoreNum, res.numOfFdHead, res.maxS2SplitNum, aicNum_);
    }
    SetSplitOutput(res);
    tilingData_.innerSplitParams.set_mBaseSize(mBaseSize_);
    tilingData_.innerSplitParams.set_s2BaseSize(sInnerSize_);
    tilingData_.fdParams.set_numOfFdHead(res.numOfFdHead);
    tilingData_.fdParams.set_gS1BaseSizeOfFd(mFdBaseSize_);
    usedCoreNum_ = res.usedCoreNum; 

    //kvSplitPart_,用于lse out workspace计算
    if (IsFlashDecode()) {
        splitKVFlag_ = true;
        kvSplit_++;
        kvSplitPart_ = res.maxS2SplitNum;
        tilingData_.fdParams.set_usedVecNumOfFd(res.usedVecNumOfFd);
    }
    CalcMmResSize();
}

void FiaTilingNonQuantMla::FillTilingBaseParams()
{
    tilingData_.baseParams.set_bSize(fiaInfo_->bSize);
    tilingData_.baseParams.set_s2Size(fiaInfo_->s2Size);
    tilingData_.baseParams.set_s1Size(fiaInfo_->s1Size);
    tilingData_.baseParams.set_n2Size(fiaInfo_->n2Size);
    tilingData_.baseParams.set_headDim(fiaInfo_->vHeadDim);
    tilingData_.baseParams.set_headDimRope(fiaInfo_->ropeHeadDim);
    tilingData_.baseParams.set_scaleValue(fiaInfo_->scaleValue);
    tilingData_.baseParams.set_gSize(fiaInfo_->n1Size / fiaInfo_->n2Size);
    tilingData_.baseParams.set_batchContinuous((fiaInfo_->kvStorageMode == KvStorageMode::TENSOR_LIST) ? 0 : 1);
    tilingData_.baseParams.set_actualSeqS1Dims(fiaInfo_->actualLenQDims);
    tilingData_.baseParams.set_actualSeqS2Dims(fiaInfo_->actualLenKvDims);
    tilingData_.baseParams.set_accumQSeqFlag(fiaInfo_->isAccumQSeq ? 1 : 0);
    tilingData_.baseParams.set_accumKVSeqFlag(fiaInfo_->isAccumKVSeq ? 1 : 0);
    tilingData_.baseParams.set_outputLayout(static_cast<uint32_t>(fiaInfo_->outputLayout));
    tilingData_.baseParams.set_needInit(fiaInfo_->needInit);
    tilingData_.baseParams.set_usedCoreNum(usedCoreNum_);
    tilingData_.baseParams.set_softmaxLseFlag(fiaInfo_->softmaxLseFlag ? 1 : 0);
    if (fiaInfo_->keyStrides != nullptr && fiaInfo_->keyStrides->GetDimNum() >= 2 &&
        fiaInfo_->valueStrides != nullptr && fiaInfo_->valueStrides->GetDimNum() >= 2) {
        tilingData_.baseParams.set_keyBnStride(fiaInfo_->keyStrides->GetStride(0));
        tilingData_.baseParams.set_keyN2Stride(fiaInfo_->keyStrides->GetStride(1));
        tilingData_.baseParams.set_valueBnStride(fiaInfo_->valueStrides->GetStride(0));
        tilingData_.baseParams.set_valueN2Stride(fiaInfo_->valueStrides->GetStride(1));
    }
    if (fiaInfo_->kRopeStrides != nullptr && fiaInfo_->kRopeStrides->GetDimNum() >= 2) {
        tilingData_.baseParams.set_keyRopeBnStride(fiaInfo_->kRopeStrides->GetStride(0));
        tilingData_.baseParams.set_keyRopeN2Stride(fiaInfo_->kRopeStrides->GetStride(1));
    }
}

void FiaTilingNonQuantMla::FillTilingPageAttenParams()
{
    tilingData_.pageAttenParams.set_blockSize(fiaInfo_->blockSize);
    tilingData_.pageAttenParams.set_maxBlockNumPerBatch(fiaInfo_->maxBlockNumPerBatch);
}

void FiaTilingNonQuantMla::FillTilingMaskParams()
{
    tilingData_.maskParams.set_attenMaskFlag(fiaInfo_->attenMaskFlag ? 1 : 0);
    tilingData_.maskParams.set_attenMaskBatchStride(fiaInfo_->attenMaskBatchStride);
    tilingData_.maskParams.set_attenMaskStride(fiaInfo_->attenMaskStride);
    tilingData_.maskParams.set_sparseMode(fiaInfo_->sparseMode);
    tilingData_.maskParams.set_preToken(fiaInfo_->preToken);
    tilingData_.maskParams.set_nextToken(fiaInfo_->nextToken);
    tilingData_.baseParams.set_slidingFlag(fiaInfo_->slidingFlag);
    uint32_t isRowInvalid = static_cast<uint32_t>(fiaInfo_->innerPrecise) >> 1;
    tilingData_.maskParams.set_isRowInvalid(isRowInvalid);
}

void FiaTilingNonQuantMla::FillTilingWorkspaceParams()
{
    uint32_t accumOutElemNumOfOneFdTask = fiaInfo_->n2Size * mBaseSize_ * headDimAlign_;
    // max和sum总共两份
    uint32_t sumAndMaxElemNumOfOneFdTask = 2 * fiaInfo_->n2Size * mBaseSize_ * (BYTE_BLOCK / sizeof(float));
    // 每个核可能有头规约和尾规约，一共两份规约信息
    constexpr uint32_t MAX_FD_TASK_NUMS = 2;
    tilingData_.workspaceParams.set_fdAccumOutSize(aicNum_ * MAX_FD_TASK_NUMS * accumOutElemNumOfOneFdTask);
    tilingData_.workspaceParams.set_fdLogSumExpSize(aicNum_ * MAX_FD_TASK_NUMS * sumAndMaxElemNumOfOneFdTask);
    tilingData_.workspaceParams.set_mm1ResSize(mm1ResSize_);
    tilingData_.workspaceParams.set_mm2ResSize(mm2ResSize_);
}

void FiaTilingNonQuantMla::CalcMmResSize()
{
    int64_t mSize = std::min(fiaInfo_->gSize * fiaInfo_->s1Size, mBaseSize_);
    mSize = Align(mSize, static_cast<int64_t>(16U));
    mm1ResSize_ = static_cast<int64_t>(sInnerSizeAlign_) * mSize;
    mm2ResSize_ = static_cast<int64_t>(headDimAlign_) * mSize;
}

void FiaTilingNonQuantMla::CalcMaxMmResSize()
{
    constexpr int64_t S2_SIZE_MAX = 512;
    mm1ResSize_ = S2_SIZE_MAX * static_cast<int64_t>(mBaseSize_);
    mm2ResSize_ = static_cast<int64_t>(headDimAlign_) * static_cast<int64_t>(mBaseSize_);
}

void FiaTilingNonQuantMla::FillTiling()
{
    FillTilingBaseParams();
    FillTilingPageAttenParams();
    FillTilingMaskParams();
    FillTilingWorkspaceParams();
}

uint32_t FiaTilingNonQuantMla::CalcFlashDecodeParamNums(const uint32_t coreNum) const
{
    return coreNum * 2U * fiaInfo_->n2Size * mBaseSize_; // 每个核可能有头规约和尾规约，一共两份规约信息
}

uint64_t FiaTilingNonQuantMla::CalcNormalWorkspaceSize(uint32_t coreNum, int64_t mm1ResSize,
    int64_t mm2ResSize, uint32_t mBaseSize) const
{
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;      // 4: fp32
    constexpr uint32_t V1_RES_ELEM_SIZE = 2;       // 2: fp16/bf16
    constexpr uint32_t MM2_RES_ELEM_SIZE = 4;      // 4: fp32
    constexpr uint32_t V2_RES_ELEM_SIZE = 4;       // 4: fp32
    constexpr uint32_t N_UPDATE_ELEM_SIZE = 4;     // 4: int32
    constexpr uint32_t SOFTMAX_SUM_ELEM_SIZE = 4;  // 4: int32

    uint64_t workspaceSize = 0;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mm1ResSize * MM1_RES_ELEM_SIZE;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mm1ResSize * V1_RES_ELEM_SIZE;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mm2ResSize * MM2_RES_ELEM_SIZE;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mm2ResSize * V2_RES_ELEM_SIZE;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mBaseSize * N_UPDATE_ELEM_SIZE; // aMla nUpdate, mBaseSize=128
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mBaseSize * SOFTMAX_SUM_ELEM_SIZE; // aMla softmaxSum, mBaseSize=128
    return workspaceSize;
}

uint64_t FiaTilingNonQuantMla::CalcFlashDecodeWorkspace(const uint32_t coreNum) const
{
    uint64_t flashDecodeParamNums = static_cast<uint64_t>(CalcFlashDecodeParamNums(coreNum));
    uint64_t accumOutSize = flashDecodeParamNums * static_cast<uint64_t>(headDimAlign_);
    uint64_t logSumExpSize = 2 * flashDecodeParamNums * (BYTE_BLOCK / sizeof(float)); // log和sum的存储空间一致,需要2份
    uint64_t workspaceSize = (accumOutSize + logSumExpSize) * sizeof(float);
    return workspaceSize;
}

void FiaTilingNonQuantMla::CalcWorkspaceSize()
{
    workspaceSize_ = libapiSize_;
    workspaceSize_ += CalcNormalWorkspaceSize(coreNum_, mm1ResSize_, mm2ResSize_, mBaseSize_);
    if (splitKVFlag_) {
        workspaceSize_ += CalcFlashDecodeWorkspace(coreNum_);
    }
}

void FiaTilingNonQuantMla::CalcMaxWorkspaceSize()
{
    CalcMaxMmResSize();
    workspaceSize_ = libapiSize_;
    workspaceSize_ += CalcNormalWorkspaceSize(coreNum_, mm1ResSize_, mm2ResSize_, mBaseSize_);
    workspaceSize_ += CalcFlashDecodeWorkspace(aicNum_);
}

void FiaTilingNonQuantMla::CalcNumBlocks(uint32_t coreNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(fiaInfo_->platformInfo);
    auto aicNum = coreNum;
    auto aivNum = aicNum * cvRatio_;

    numBlocks_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum); // 暂时与当前代码一致
    OP_LOGI(fiaInfo_->opName, "FIA block dim: %u aiv Num: %u aic Num: %u.", numBlocks_, aivNum, aicNum);
}

void FiaTilingNonQuantMla::CalcScheduleMode()
{
    scheduleMode_ = ScheduleMode::BATCH_MODE;
    OP_LOGI(fiaInfo_->opName, "FIA schedule mode: %u.", static_cast<uint32_t>(scheduleMode_));
}

ge::graphStatus FiaTilingNonQuantMla::DoOpTiling()
{
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    InitParams();
    if (fiaInfo_->isMaxWorkspace) {
        CalcMaxWorkspaceSize();
        GenTilingKey();
    } else {
        Split();
        FillTiling();
        CalcNumBlocks(usedCoreNum_);
        CalcScheduleMode();
        CalcWorkspaceSize();
        GenTilingKey();
    }

    if ((SetNumBlocks(numBlocks_) != ge::GRAPH_SUCCESS) || (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) ||
        (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS) ||
        (SetScheduleMode(scheduleMode_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// 值越小表示优先级越高. 对于FIA, 使用3位数表示优先级, 优先级编码含义为:
// 1. 百位代表非量化、伪量化、全量化等场景, 即: 0xx-非量化，1xx-伪量化, 2xx-全量化
// 2. 十位表示gqa、mla、泛化，即: x0x-mla, x1x-gpa, x2x-泛化
// 3. 个位代表特化模板到泛化模板的优先级排序
REGISTER_TILING_TEMPLATE_FIA(FusedInferAttentionScore, FiaTilingNonQuantMla,
    std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}), 9);
} // namespace optiling
