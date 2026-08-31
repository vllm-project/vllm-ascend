/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_tiling_nonquant_sink.cpp
 * \brief
 */

#include "fia_tiling_nonquant_sink.h"
#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "fia_tiling_templates_registry.h"
#include "split_core.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

constexpr uint64_t PRE_LOAD_NUM_MLA = 2;
constexpr uint32_t BLOCK_TABLE_ELEM_BYTE = 4;

constexpr uint64_t FIA_TILINGKEYOFFSET = uint64_t(100000000000000000UL);
constexpr uint64_t FIA_PERF_MODE_TILINGKEYOFFSET = uint64_t(1000000000000000UL);

constexpr uint32_t G_SIZE_128 = 128;
constexpr uint32_t S1_SIZE_16 = 16;

constexpr uint32_t QK_HEAD_DIM_64 = 64;
constexpr uint32_t QK_HEAD_DIM_128 = 128;
constexpr uint32_t QK_HEAD_DIM_192 = 192;

constexpr uint32_t ROPE_HEAD_DIM_0 = 0;
constexpr uint32_t ROPE_HEAD_DIM_64 = 64;

constexpr uint32_t V_HEAD_DIM_64 = 64;
constexpr uint32_t V_HEAD_DIM_128 = 128;

constexpr uint32_t S_INNER_SIZE_512 = 512;
constexpr uint32_t S_INNER_SIZE_1024 = 1024;

constexpr uint32_t S_INNER_SIZE_ALIGN_512 = 512;
constexpr uint32_t S_INNER_SIZE_ALIGN_1024 = 1024;
constexpr uint32_t S_INNER_SIZE_ALIGN_2048 = 2048;
constexpr uint32_t S_INNER_SIZE_ALIGN_4096 = 4096;

constexpr int32_t SPARSE_MODE_2 = 2;
constexpr int32_t SPARSE_MODE_3 = 3;
constexpr int32_t SPARSE_MODE_4 = 4;

constexpr uint32_t M_BASE_SIZE_32 = 32;
constexpr uint32_t M_BASE_SIZE_64 = 64;
constexpr uint32_t M_BASE_SIZE_128 = 128;
constexpr uint32_t M_BASE_SIZE_256 = 256;
constexpr uint32_t M_BASE_SIZE_512 = 512;

template <typename T> inline auto Align(T num, T rnd) -> T
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

constexpr uint64_t RecursiveSum()
{
    return 0;
}

template <typename T, typename... Args> constexpr uint64_t RecursiveSum(T templateId, Args... templateIds)
{
    return static_cast<uint64_t>(templateId) + 10U * RecursiveSum(templateIds...);
}

template <typename... Args> constexpr uint64_t FIA_GET_TILINGKEY(Args... templateIds)
{
    return RecursiveSum(templateIds...);
}

void FiaTilingNonQuant::InitTilingInfo(TilingInfo *tilingInfo)
{
    fiaInfo_ = static_cast<FiaTilingInfo *>(tilingInfo);
}

ge::graphStatus FiaTilingNonQuant::GetPlatformInfo()
{
    OPS_CHECK(fiaInfo_->platformInfo == nullptr,
              OPS_REPORT_VECTOR_INNER_ERR(fiaInfo_->opName, "GetPlatformInfo is nullptr."),
              return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(fiaInfo_->platformInfo);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();

    OPS_CHECK(aicNum_ == 0 || aivNum_ == 0,
              OPS_REPORT_VECTOR_INNER_ERR(fiaInfo_->opName, "num of core obtained is 0."),
              return GRAPH_FAILED);

    // 设置CV1:1模式
    cvRatio_ = aivNum_ / aicNum_;
    OPS_LOG_I(fiaInfo_->opName, "FIA aicNum: %u, aivNum:%u, cvRatio:%u.", aicNum_, aivNum_, cvRatio_);

    return ge::GRAPH_SUCCESS;
}

bool FiaTilingNonQuant::IsCapable()
{
    if (fiaInfo_ == nullptr) {
        return false;
    }

    // 不支持空Tensor
    if (fiaInfo_->emptyTensorFlag) {
        return false;
    }

    ge::DataType qDataType = fiaInfo_->inputQType;
    ge::DataType kDataType = fiaInfo_->inputKvType;
    // 仅支持非量化
    if ((qDataType != ge::DT_FLOAT16 && qDataType != ge::DT_BF16) || (qDataType != kDataType)) {
        return false;
    }
    return true;
}

void FiaTilingNonQuant::GenTilingKey()
{
    uint8_t layoutVal{0};
    uint8_t inputQVal{0};
    uint8_t inputKvVal{0};
    uint8_t outputVal{0};
    uint8_t originVal{0};
    uint8_t splitKvVal = static_cast<uint8_t>(kvSplit_ > static_cast<uint32_t>(0) ? 1 : 0);
    uint8_t paVal = static_cast<uint8_t>((fiaInfo_->pageAttentionFlag && fiaInfo_->s2Size != 0) ? 1 * 2 : 0);
    uint8_t softmaxBrcbFlagVal = static_cast<uint8_t>((softmaxWithBrcbFlag_) ? 1 * 4 : 0);
    uint8_t antiquantModeVal = 0;
    uint64_t modeVal = 1;
    uint8_t kvLayoutVal = 0;

    const std::map<TilingKeyLayout, uint8_t> kvLayoutMap = {{TilingKeyLayout::BNSD, 0U},
                                                            {TilingKeyLayout::BSH_BSND, 1U},
                                                            {TilingKeyLayout::NZ, 2U},
                                                            {TilingKeyLayout::TND, 3U},
                                                            {TilingKeyLayout::NTD, 5U}};

    const std::map<TilingKeyLayout, uint8_t> qLayoutMap = {{TilingKeyLayout::BNSD, 0U},
                                                           {TilingKeyLayout::BSH_BSND, 1U},
                                                           {TilingKeyLayout::TND, 3U},
                                                           {TilingKeyLayout::NTD, 5U}};

    const std::map<ge::DataType, uint8_t> typeMap = {
        {ge::DT_FLOAT16, 0U},
        {ge::DT_BF16, 2U},
        {ge::DT_INT8, 3U},
        {ge::DT_INT4, 4U},
    };

    if (kvLayoutMap.find(fiaInfo_->inputKvLayout) != kvLayoutMap.end()) {
        kvLayoutVal = kvLayoutMap.at(fiaInfo_->inputKvLayout);
    }
    if (qLayoutMap.find(fiaInfo_->inputLayout) != qLayoutMap.end()) {
        layoutVal = qLayoutMap.at(fiaInfo_->inputLayout);
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
    tilingKey_ = baseOffset + FIA_GET_TILINGKEY(layoutVal,
                                                inputQVal,
                                                inputKvVal,
                                                outputVal,
                                                originVal,
                                                (softmaxBrcbFlagVal + paVal + splitKvVal),
                                                antiquantModeVal,
                                                kvLayoutVal);

    OPS_LOG_I(fiaInfo_->opName, "FIA tilingKey_: %lu.", tilingKey_);
}

bool FiaTilingNonQuant::IsFlashDecode()
{
    return true;
}

void FiaTilingNonQuant::ZeroTensorProcess() const
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

void FiaTilingNonQuant::InitParams()
{
    perfMode_ = FiaTemplateId::GENERAL_GQA;
    if ((fiaInfo_->qkHeadDim == QK_HEAD_DIM_128 && fiaInfo_->ropeHeadDim == ROPE_HEAD_DIM_0 &&
         fiaInfo_->vHeadDim == V_HEAD_DIM_128) ||
        (fiaInfo_->qkHeadDim == QK_HEAD_DIM_64 && fiaInfo_->ropeHeadDim == ROPE_HEAD_DIM_0 &&
         fiaInfo_->vHeadDim == V_HEAD_DIM_64) ||
        (fiaInfo_->qkHeadDim == QK_HEAD_DIM_192 && fiaInfo_->ropeHeadDim == ROPE_HEAD_DIM_64 &&
         fiaInfo_->vHeadDim == V_HEAD_DIM_128) ||
        (fiaInfo_->qkHeadDim == QK_HEAD_DIM_128 && fiaInfo_->ropeHeadDim == ROPE_HEAD_DIM_64 &&
         fiaInfo_->vHeadDim == V_HEAD_DIM_128)) {
        if (!(fiaInfo_->sysPrefixFlag || fiaInfo_->pseShiftFlag || fiaInfo_->kvPaddingSizeFlag ||
              fiaInfo_->qPaddingSizeFlag)) {
            perfMode_ = FiaTemplateId::HIGH_PERFORMANCE_GQA;
        }
    }

    coreNum_ = aicNum_;
    blockDim_ = aicNum_; // Tiling下沉首次Tiling也会校验blockDim_是否为0，将blockDim_设置为aicNum_，实际不生效

    headDimAlign_ = Align(fiaInfo_->qkHeadDim, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    ZeroTensorProcess();
}

void FiaTilingNonQuant::CalcInnerSize(uint32_t s2Size)
{
    /*
     * CANN reduceSum函数在规约长度S  S>=512、S<512两种场景内部实现的累加顺序不同，
     * 为了保障batchInvariant场景计算结果的一致性，将算子在S2方向的基本块大小固定为256，
     * 从而确保不同shape下reduceSum函数的累加序的一致性
    */
    if (fiaInfo_->batchInvariant) {
        sInnerSize_ = 256U;
    } else {
        if (fiaInfo_->inputLayout == TilingKeyLayout::TND || fiaInfo_->inputLayout == TilingKeyLayout::NTD) {
            sInnerSize_ = S_INNER_SIZE_512;
            /*
             * BAND场景，带上sinkToken后，分块位置发生变化，从而导致计算band边界覆盖的基本块增加，导致无效计算量增大
             * 将基本块大小减小至256，减少无效计算量，缓解极端场景下分核劣势
            */
            if ((fiaInfo_->sinkNumber > 0 && fiaInfo_->sinkNumber <= 256 && fiaInfo_->s1Size > S1_SIZE_16
                && fiaInfo_->sparseMode == SPARSE_MODE_4)) {
                sInnerSize_ = 256U;
            }

            /*
             * prefill阶段，SWA场景(窗口大小为512)，针对sinkTensor场景做了小块跳过性能优化操作，需要将s2基本块修改为256
             * 由于prefill阶段数据量较大，在长序列场景为计算bound，跳过操作的搬运收益拿不到
             * 且将s2基本块修改为256后，scale操作增多，性能反而劣化，实验测试边界值为qTSize=32k
             *
             * kernel侧使用的sInnerSize_是通过aicpu传递的，该处的sInnerSize_不起作用
             * 该处和aicpu的差异在于，该处使用qTSize，aicpu使用actseqlen[-1]
            */
            if ((fiaInfo_->keySinkNumber > 0 && fiaInfo_->sparseMode == SPARSE_MODE_BAND &&
                fiaInfo_->qTSize < 32 * 1024 && (fiaInfo_->preToken + fiaInfo_->nextToken < 512))) {
                sInnerSize_ = 256U;
            }
        } else {
            if (fiaInfo_->s1Size <= S1_SIZE_16) {
                /**
                * V1阶段分配用于存放mm1结果的UB大小为32K, 当计算的数据类型为float时，其可以存放8192个元素.
                * 另外, 需要保证单次计算不会切分S2, 那么S2的内切大小最大为8192, 所以将默认值设置为8192
                */
                sInnerSize_ = MAX_SPLIT_SIZE; // 8192

                /** 当前版本限制workspace大小不超过32MB，否则会影响网络中前后算子性能，
                *  GQA场景下 nNumOfQInOneGroup和sInnerSize_切分大小直接影响workspace大小,
                *  具体计算参考CalcWorkSpace函数，这里根据nNumOfQInOneGroup将sInnerSize_
                *  分为8192，4096，2048三档，nNumOfQInOneGroup增大时减小sInnerSize_，
                *   保证最终workspace大小不超过32MB。
                */
                uint32_t sInnerSize[3U] = {8192U, 4096U, 2048U};
                uint32_t idx = std::min(fiaInfo_->gSize / 5U, 2U);
                sInnerSize_ = sInnerSize[idx];
            } else {
                sInnerSize_ = S_INNER_SIZE_512;
            }
        }
        if (fiaInfo_->attenMaskFlag && (fiaInfo_->sparseMode == SPARSE_MODE_2 ||
            fiaInfo_->sparseMode == SPARSE_MODE_3 || fiaInfo_->sparseMode == SPARSE_MODE_4)) {
            sInnerSize_ = std::min(sInnerSize_, S_INNER_SIZE_1024); // attention mask压缩场景，基本块最大支持1024*1024
        }
        // PA特性泛化场景，blockSize可能为112等值，无法被sInnerSize_整除，当step*base跨block时，搬运处理复杂，通过向下对齐避免
        if (fiaInfo_->pageAttentionFlag && fiaInfo_->blockSize != 0) {
            uint32_t blockSize = static_cast<uint32_t>(fiaInfo_->blockSize);
            if ((sInnerSize_ > blockSize) && (sInnerSize_ % blockSize != 0)) {
                sInnerSize_ = (sInnerSize_ / blockSize) * blockSize;
            }
        }
        sInnerLoopTimes_ = (s2Size + sInnerSize_ - static_cast<uint32_t>(1)) / sInnerSize_;
        sInnerSizeTail_ = s2Size - (sInnerLoopTimes_ - static_cast<uint32_t>(1)) * sInnerSize_;
        // tiling下沉 && flash decoder场景时，sInnerSize_基块大小不按照真实值修改
        // 否则会导致 tiling下沉 && flash decoder 场景时开辟workspace空间大小小于真实运行时所需的workspace大小
        if (sInnerSize_ > s2Size) {
            sInnerSize_ = s2Size;
        }
    }
    sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    CalcMBaseSize();
}

void FiaTilingNonQuant::CalcMBaseSize()
{
    if (fiaInfo_->inputLayout == TilingKeyLayout::TND || fiaInfo_->inputLayout == TilingKeyLayout::NTD) {
        mBaseSize_ = M_BASE_SIZE_512;
        /*
         * BAND场景，带上sinkToken后，分块位置发生变化，从而导致计算band边界覆盖的基本块增加，导致无效计算量增大
         * 将基本块大小减小至256，减少无效计算量，缓解极端场景下分核劣势
        */
        if (fiaInfo_->sinkNumber > 0 && fiaInfo_->sinkNumber <= 256 && fiaInfo_->s1Size > S1_SIZE_16 &&
            fiaInfo_->sparseMode == SPARSE_MODE_4 && fiaInfo_->nextToken == 0 &&
            fiaInfo_->preToken % 512 >= fiaInfo_->sinkNumber) {
            mBaseSize_ = M_BASE_SIZE_256;
        }
    } else {
        if (fiaInfo_->s1Size <= S1_SIZE_16) {
            if (sInnerSizeAlign_ <= S_INNER_SIZE_ALIGN_512) {
                mBaseSize_ = M_BASE_SIZE_512;
            } else if (sInnerSizeAlign_ <= S_INNER_SIZE_ALIGN_1024) {
                mBaseSize_ = M_BASE_SIZE_256;
            } else if (sInnerSizeAlign_ <= S_INNER_SIZE_ALIGN_2048) {
                mBaseSize_ = M_BASE_SIZE_128;
            } else if (sInnerSizeAlign_ <= S_INNER_SIZE_ALIGN_4096) {
                mBaseSize_ = M_BASE_SIZE_64;
            } else { // sInnerSizeAlign_最大值为8192
                mBaseSize_ = M_BASE_SIZE_32;
            }
        } else {
            mBaseSize_ = M_BASE_SIZE_512;
        }
    }

    if (fiaInfo_->batchInvariant) {
        mBaseSize_ = M_BASE_SIZE_256;
    }
    softmaxWithBrcbFlag_ = (mBaseSize_ <= M_BASE_SIZE_128);

    OPS_LOG_I(fiaInfo_->opName,
              "FIA sInnerSize_:%u sInnerSizeAlign_:%u mBaseSize_:%u softmaxWithBrcbFlag_:%u.",
              sInnerSize_,
              sInnerSizeAlign_,
              mBaseSize_,
              softmaxWithBrcbFlag_);
}

void FiaTilingNonQuant::CreateSplitInput(BaseInfo &baseInfo, SplitParam &splitParam) const
{
    // 构造分核输入参数
    baseInfo.bSize = fiaInfo_->bSize;
    baseInfo.n2Size = fiaInfo_->n2Size;
    baseInfo.gSize = fiaInfo_->gSize;
    baseInfo.s2Size = fiaInfo_->s2Size;
    baseInfo.s1Size = fiaInfo_->s1Size;
    baseInfo.actualLenQDims = fiaInfo_->actualLenQDims;
    baseInfo.actualLenKvDims = fiaInfo_->actualLenDims;
    baseInfo.preToken = fiaInfo_->preToken;
    baseInfo.nextToken = fiaInfo_->nextToken;
    baseInfo.isS1G = fiaInfo_->inputLayout == TilingKeyLayout::TND ||
                     fiaInfo_->inputLayout == TilingKeyLayout::BSH_BSND; // 使用枚举映射
    baseInfo.sparseMode = fiaInfo_->sparseMode;
    baseInfo.attenMaskFlag = fiaInfo_->attenMaskFlag;
    baseInfo.sinkNumber = fiaInfo_->sinkNumber;
    baseInfo.keySinkNumber = fiaInfo_->keySinkNumber;
    baseInfo.supportFD = (baseInfo.keySinkNumber == 0); // sink做为独立tensor传入时,prefill不走FD
    baseInfo.batchInvariant = fiaInfo_->batchInvariant;

    splitParam.mBaseSize = mBaseSize_;
    splitParam.s2BaseSize = sInnerSize_;
    splitParam.gS1BaseSizeOfFd = mFdBaseSize_;
}

void FiaTilingNonQuant::SetSplitOutput(const SplitResult &res)
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
    uint32_t *gS1IdxEndOfFdHead = tilingData_.fdParams.get_gS1IdxEndOfFdHead();
    uint32_t *gS1IdxEndOfFdHeadSplit = tilingData_.fdParams.get_gS1IdxEndOfFdHeadSplit();
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
    tilingData_.innerSplitParams.set_mBaseSize(mBaseSize_);
    tilingData_.innerSplitParams.set_s2BaseSize(sInnerSize_);
    tilingData_.fdParams.set_gS1BaseSizeOfFd(mFdBaseSize_);
    tilingData_.fdParams.set_numOfFdHead(res.numOfFdHead);
    usedCoreNum_ = res.usedCoreNum;
}

void FiaTilingNonQuant::Split()
{
    uint32_t s2SizeInput = static_cast<uint32_t>(fiaInfo_->s2Size);
    CalcInnerSize(s2SizeInput);
    BaseInfo baseInfo{};
    SplitParam splitParam{};
    CreateSplitInput(baseInfo, splitParam);

    usedCoreNum_ = aicNum_;
    fiaInfo_->isExistRowInvalid = IsExistRowInvalid(baseInfo);

    splitKVFlag_ = true;
    kvSplit_++;
    CalcMaxMmResSize();
}

uint32_t FiaTilingNonQuant::GetL2CacheOffFlag()
{
    uint64_t kvTypeSize = 2;
    uint64_t kvSize = 0;
    if (fiaInfo_->kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        kvSize = fiaInfo_->opParamInfo.key.shape->GetStorageShape().GetShapeSize();
    } else if (fiaInfo_->kvStorageMode == KvStorageMode::TENSOR_LIST) {
        for (int64_t size = 0; size < fiaInfo_->bSize; ++size) {
            auto keyTensorInList = fiaInfo_->kCache[size];
            kvSize += keyTensorInList->GetStorageShape().GetShapeSize();
        }
    } else {
        kvSize = fiaInfo_->opParamInfo.key.shape->GetStorageShape().GetShapeSize();
    }

    uint64_t l2CacheSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(fiaInfo_->platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize);

    // 之前路由到IFA的GQA场景才需要考虑关闭L2Cache
    if ((fiaInfo_->ropeMode == RopeMode::NO_ROPE) && (fiaInfo_->s1Size == 1) &&
        (fiaInfo_->gSize <= 64)) { // 1:qs=1 64:g<=64 IFA的GQA场景
        // 1. 连续访存时, 即KV的layout为BNSD或者BnNBsD, 不涉及数据预取, 可以直接关闭L2Cache
        // 2. 考虑K和V数据的总大小超过一定值后, 关闭L2Cache, 当前系数确定为1.2
        if (fiaInfo_->kvLayout == FiaLayout::BNSD || fiaInfo_->kvLayout == FiaLayout::BnNBsD) {
            l2CacheOffFlag_ = 1U;
        } else if (static_cast<double>(kvSize) * kvTypeSize * 2.0f >=
                   l2CacheSize * 1.2) { // 2:K和V数据的总大小    1.2:阈值系数
            l2CacheOffFlag_ = 1U;
        } else {
            l2CacheOffFlag_ = 0;
        }
    } else {
        l2CacheOffFlag_ = 0;
    }

    OPS_LOG_D(fiaInfo_->opName,
              "l2CacheOffFlag_: %u, kvSize: %lu, kvTypeSize: %u, l2CacheSize: %lu",
              l2CacheOffFlag_,
              kvSize,
              kvTypeSize,
              l2CacheSize);
    return l2CacheOffFlag_;
}

void FiaTilingNonQuant::FillTilingBaseParams()
{
    tilingData_.baseParams.set_bSize(fiaInfo_->bSize);
    tilingData_.baseParams.set_s2Size(fiaInfo_->s2Size);
    tilingData_.baseParams.set_s1Size(fiaInfo_->s1Size);
    tilingData_.baseParams.set_n2Size(fiaInfo_->n2Size);
    tilingData_.baseParams.set_headDim(fiaInfo_->vHeadDim);
    tilingData_.baseParams.set_headDimRope(fiaInfo_->ropeHeadDim);
    tilingData_.baseParams.set_scaleValue(fiaInfo_->scaleValue);
    tilingData_.baseParams.set_gSize(fiaInfo_->n1Size / fiaInfo_->n2Size);
    tilingData_.baseParams.set_qTSize(fiaInfo_->qTSize);
    tilingData_.baseParams.set_kTSize(fiaInfo_->kTSize);
    tilingData_.baseParams.set_batchContinuous((fiaInfo_->kvStorageMode == KvStorageMode::TENSOR_LIST) ? 0 : 1);
    tilingData_.baseParams.set_batchInvariant(fiaInfo_->batchInvariant ? 1 : 0);
    tilingData_.baseParams.set_actualSeqS1Dims(fiaInfo_->actualLenQDims);
    tilingData_.baseParams.set_actualSeqS2Dims(fiaInfo_->actualLenDims);
    tilingData_.baseParams.set_accumQSeqFlag(fiaInfo_->isAccumQSeq ? 1 : 0);
    tilingData_.baseParams.set_accumKVSeqFlag(fiaInfo_->isAccumKVSeq ? 1 : 0);
    tilingData_.baseParams.set_outputLayout(static_cast<uint32_t>(fiaInfo_->outputLayout));
    tilingData_.baseParams.set_softmaxLseFlag(fiaInfo_->softmaxLseFlag ? 1 : 0);
    tilingData_.baseParams.set_usedCoreNum(usedCoreNum_);
    l2CacheOffFlag_ = GetL2CacheOffFlag();
    tilingData_.baseParams.set_l2CacheOffFlag(l2CacheOffFlag_);
    tilingData_.baseParams.set_isLegacyIfa(fiaInfo_->isLegacyIfa);
}

void FiaTilingNonQuant::FillTilingPageAttenParams()
{
    tilingData_.pageAttenParams.set_blockSize(fiaInfo_->blockSize);
    tilingData_.pageAttenParams.set_maxBlockNumPerBatch(fiaInfo_->maxBlockNumPerBatch);
    // 获取偏移值
    uint64_t *keyStridePtr = tilingData_.pageAttenParams.get_keyStride();
    uint64_t *valueStridePtr = tilingData_.pageAttenParams.get_valueStride();
    uint64_t *keyRopeStridePtr = tilingData_.pageAttenParams.get_keyRopeStride();

    for (uint32_t i = 0; i < 5U; ++i) {
        keyStridePtr[i] = fiaInfo_->keyStride[i];
        valueStridePtr[i] = fiaInfo_->valueStride[i];
        keyRopeStridePtr[i] = fiaInfo_->keyRopeStride[i];
    }
}

void FiaTilingNonQuant::FillTilingMaskParams()
{
    tilingData_.maskParams.set_attenMaskFlag(fiaInfo_->attenMaskFlag ? 1 : 0);
    tilingData_.maskParams.set_attenMaskSize(fiaInfo_->attenMaskSize);
    tilingData_.maskParams.set_attenMaskStride(fiaInfo_->attenMaskStride);
    tilingData_.maskParams.set_sparseMode(fiaInfo_->sparseMode);
    tilingData_.maskParams.set_preToken(fiaInfo_->preToken);
    tilingData_.maskParams.set_sinkNumber(fiaInfo_->sinkNumber);
    tilingData_.maskParams.set_keySinkNumber(fiaInfo_->keySinkNumber);
    if (fiaInfo_->sinkNumber > 0 && fiaInfo_->sparseMode == SPARSE_MODE_2) {
        fiaInfo_->nextToken = fiaInfo_->sinkNumber;
    }
    tilingData_.maskParams.set_nextToken(fiaInfo_->nextToken);
    uint32_t isRowInvalid = static_cast<uint32_t>(fiaInfo_->innerPrecise) >> 1;
    tilingData_.maskParams.set_isRowInvalid(isRowInvalid);
    tilingData_.maskParams.set_isExistRowInvalid(static_cast<uint32_t>(fiaInfo_->isExistRowInvalid));
}

// for flash decode
void FiaTilingNonQuant::FillTilingWorkspaceParams()
{
    // 每个核可能有头规约和尾规约，一共两份规约信息
    tilingData_.workspaceParams.set_fdAccumOutSize(aicNum_ * 2 * mBaseSize_ * headDimAlign_);
    // 每个核可能有头规约和尾规约，一共两份规约信息; 另外sum和max各一份
    tilingData_.workspaceParams.set_fdLogSumExpSize(2 * aicNum_ * 2 * mBaseSize_ *
                                                    (BYTE_BLOCK / BLOCK_TABLE_ELEM_BYTE));
    tilingData_.workspaceParams.set_mm1ResSize(mm1ResSize_);
    tilingData_.workspaceParams.set_mm2ResSize(mm2ResSize_);
}

void FiaTilingNonQuant::CalcMmResSize()
{
    int64_t mSize = std::min(fiaInfo_->gSize * fiaInfo_->s1Size, mBaseSize_);
    mm1ResSize_ = static_cast<int64_t>(sInnerSizeAlign_) * mSize;
    mm2ResSize_ = static_cast<int64_t>(headDimAlign_) * mSize;
}

void FiaTilingNonQuant::CalcMaxMmResSize()
{
    mBaseSize_ = M_BASE_SIZE_512;
    mm1ResSize_ = 512 * 512; // mm1的结果最大为512*512个元素
    mm2ResSize_ = static_cast<int64_t>(headDimAlign_) * 512; // mBaseSize最大值为512
}

void FiaTilingNonQuant::FillTiling()
{
    FillTilingBaseParams();
    FillTilingPageAttenParams();
    FillTilingMaskParams();
    FillTilingWorkspaceParams();
}

uint32_t FiaTilingNonQuant::CalcFlashDecodeParamNums(const uint32_t coreNum) const
{
    return coreNum * 2U * mBaseSize_; // 每个核可能有头规约和尾规约，一共两份规约信息
}

uint64_t FiaTilingNonQuant::CalcNormalWorkspaceSize(uint32_t coreNum, int64_t mm1ResSize, int64_t mm2ResSize) const
{
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4; // 4: fp32
    constexpr uint32_t V1_RES_ELEM_SIZE = 2; // 2: fp16/bf16
    constexpr uint32_t MM2_RES_ELEM_SIZE = 4; // 4: fp32
    constexpr uint32_t V2_RES_ELEM_SIZE = 4; // 4: fp32

    uint64_t workspaceSize = 0;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mm1ResSize * MM1_RES_ELEM_SIZE;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mm1ResSize * V1_RES_ELEM_SIZE;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mm2ResSize * MM2_RES_ELEM_SIZE;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mm2ResSize * V2_RES_ELEM_SIZE;
    return workspaceSize;
}

uint64_t FiaTilingNonQuant::CalcFlashDecodeWorkspace(const uint32_t coreNum) const
{
    uint64_t flashDecodeParamNums = static_cast<uint64_t>(CalcFlashDecodeParamNums(coreNum));
    uint64_t accumOutSize = flashDecodeParamNums * static_cast<uint64_t>(headDimAlign_);
    uint64_t logSumExpSize =
        2 * flashDecodeParamNums *
        (BYTE_BLOCK / static_cast<uint64_t>(BLOCK_TABLE_ELEM_BYTE)); // log和sum的存储空间一致，共需要2份内存
    uint64_t workspaceSize = (accumOutSize + logSumExpSize) * static_cast<uint64_t>(BLOCK_TABLE_ELEM_BYTE);
    return workspaceSize;
}

void FiaTilingNonQuant::CalcWorkspaceSize()
{
    workspaceSize_ = libapiSize_;
    workspaceSize_ += CalcNormalWorkspaceSize(coreNum_, mm1ResSize_, mm2ResSize_);
    if (splitKVFlag_) {
        workspaceSize_ += CalcFlashDecodeWorkspace(coreNum_);
    }
}

void FiaTilingNonQuant::CalcMaxWorkspaceSize()
{
    CalcMaxMmResSize();
    workspaceSize_ = libapiSize_;
    workspaceSize_ += CalcNormalWorkspaceSize(coreNum_, mm1ResSize_, mm2ResSize_);
    workspaceSize_ += CalcFlashDecodeWorkspace(aicNum_);
}

void FiaTilingNonQuant::CalcBlockDim(uint32_t coreNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(fiaInfo_->platformInfo);
    auto aicNum = coreNum;
    auto aivNum = aicNum * cvRatio_;

    blockDim_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OPS_LOG_I(fiaInfo_->opName, "FIA block dim: %u aiv Num: %u aic Num: %u.", blockDim_, aivNum, aicNum);
}

void FiaTilingNonQuant::CalcScheduleMode()
{
    scheduleMode_ = ScheduleMode::BATCH_MODE;
    OPS_LOG_I(fiaInfo_->opName, "FIA schedule mode: %u.", static_cast<uint32_t>(scheduleMode_));
}

void FiaTilingNonQuant::GetSafeActToken(
    SparseMode mode, int64_t actSeqLensQ, int64_t actSeqLensKv, int64_t &safePreToken, int64_t &safeNextToken)
{
    if (mode == SparseMode::DEFAULT_MASK) {
        safePreToken = std::max(-actSeqLensKv, safePreToken);
        safePreToken = std::min(safePreToken, actSeqLensQ);
        safeNextToken = std::max(-actSeqLensQ, safeNextToken);
        safeNextToken = std::min(safeNextToken, actSeqLensKv);
    } else if (mode == SparseMode::BAND) {
        safePreToken = std::max(-actSeqLensQ, safePreToken);
        safePreToken = std::min(safePreToken, actSeqLensKv);
        safeNextToken = std::max(-actSeqLensKv, safeNextToken);
        safeNextToken = std::min(safeNextToken, actSeqLensQ);
    }
}

bool FiaTilingNonQuant::IsExistRowInvalid(const BaseInfo &baseInfo)
{
    if (!baseInfo.attenMaskFlag) {
        return false;
    }

    auto mode = static_cast<SparseMode>(baseInfo.sparseMode);
    if (mode == SparseMode::LEFT_UP_CAUSAL) {
        return false;
    }

    if (mode == SparseMode::ALL_MASK) {
        return true;
    }

    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        uint32_t s1Size = GetS1SeqSize(bIdx, baseInfo);
        uint32_t s2Size = GetS2SeqSize(bIdx, baseInfo);
        if ((s1Size == 0) || (s2Size == 0)) {
            // 空tensor认为不会有无效行
            continue;
        }

        int64_t safePreToken = baseInfo.preToken;
        int64_t safeNextToken = baseInfo.nextToken;
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;

        GetSafeActToken(mode, s1Size, s2Size, safePreToken, safeNextToken);
        if (mode == SparseMode::BAND) {
            preTokenLeftUp = safePreToken;
            nextTokenLeftUp = static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + safeNextToken;
        } else if (mode == SparseMode::DEFAULT_MASK) {
            preTokenLeftUp = static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + safePreToken;
            nextTokenLeftUp = safeNextToken;
        } else {
            preTokenLeftUp = 0;
            nextTokenLeftUp = static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        }

        if ((preTokenLeftUp < 0) || (nextTokenLeftUp < 0)) {
            return true;
        }
    }
    return false;
}

ge::graphStatus FiaTilingNonQuant::DoOpTiling()
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
        CalcBlockDim(usedCoreNum_);
        CalcScheduleMode();
        CalcWorkspaceSize();
        GenTilingKey();
    }

    if ((SetBlockDim(blockDim_) != ge::GRAPH_SUCCESS) || (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) || (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS) ||
        (SetScheduleMode(scheduleMode_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// 值越小表示优先级越高. 对于FIA, 使用3位数表示优先级, 优先级编码含义为:
// 1. 百位代表非量化、伪量化、全量化等场景, 即: 0xx-非量化，1xx-伪量化, 2xx-全量化
// 2. 十位表示gqa、mla、泛化，即: x0x-mla, x1x-gpa, x2x-泛化
// 3. 个位代表特化模板到泛化模板的优先级排序
REGISTER_TILING_TEMPLATE_FIA(FusedInferAttentionScoreV2Sink,
                             FiaTilingNonQuant,
                             std::vector<int32_t>({
                                 (int32_t)platform_ascendc::SocVersion::ASCEND910B,
                                 (int32_t)platform_ascendc::SocVersion::ASCEND910_93,
                                 (int32_t)platform_ascendc::SocVersion::ASCEND950,
                             }),
                             19);
} // namespace optiling
