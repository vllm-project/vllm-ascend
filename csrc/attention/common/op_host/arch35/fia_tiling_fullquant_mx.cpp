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
 * \file fia_tiling_fullquant_mx.cpp
 * \brief
 */

#include "fia_tiling_fullquant_mx.h"
#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "../fia_tiling_templates_registry.h"
#include "../split_core_v2.h"
#include "../../../fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_utils.h"
#include "../../../fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_constants.h"
#include "../../../prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_template_tiling_key_enum.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
using namespace arch35FIA;
constexpr uint64_t PRE_LOAD_NUM_GQA_ARCH35 = 3;

void FiaTilingFullQuantMxArch35::InitTilingInfo(TilingInfo *tilingInfo)
{
    fiaInfo_ = static_cast<FiaTilingInfo *>(tilingInfo);
}

bool FiaTilingFullQuantMxArch35::IsCapable()
{
    if (fiaInfo_ == nullptr) {
        return false;
    }

    // 不支持空Tensor
    if (fiaInfo_->emptyTensorFlag) {
        return false;
    }
    if (fiaInfo_->fullQuantMode != FiaFullQuantMode::QKV_MXFP8_FULL_QUANT) {
        return false;
    }

    return true;
}

void FiaTilingFullQuantMxArch35::CalcMaxWorkspaceSize()
{
    constexpr int32_t mSize = 128;
    constexpr int32_t dVSize = 512;

    size_t sysWorkspaceSize = platformInfo_.defaultSysWorkspaceSize;
    workspaceSize_ = sysWorkspaceSize;
    workspaceSize_ += platformInfo_.coreNum * PRE_LOAD_NUM_GQA_ARCH35 * (mSize * dVSize) * sizeof(float);

    // 2 bmm, db, ensure alignment of each structure 64B, dcci cacheline needs
    workspaceSize_ += static_cast<uint64_t>(platformInfo_.coreNum) * 2 * 2 * 64;

    uint32_t faTmpAttenGmSize = platformInfo_.coreNum * 2 * mSize * dVSize; // 每个核最多有2次写到workspace
    uint32_t fatmpResLseGmSize = platformInfo_.coreNum * 2 * mSize * 8;
    workspaceSize_ += (faTmpAttenGmSize + 2 * fatmpResLseGmSize) * sizeof(float); // ResLse有2份，sum和max
}

void FiaTilingFullQuantMxArch35::CalcScheduleMode()
{
    scheduleMode_ = ScheduleMode::BATCH_MODE;
    OP_LOGI(fiaInfo_->opName, "FIA schedule mode: %u.", static_cast<uint32_t>(scheduleMode_));
}

ge::graphStatus FiaTilingFullQuantMxArch35::DoOpTiling()
{
    OP_CHECK_IF(SetPlatMemoryInfo() != ge::GRAPH_SUCCESS, OP_LOGE(fiaInfo_->opName, "Set plat memory info fail."),
                return ge::GRAPH_FAILED);

    if (fiaInfo_->emptyTensorFlag) {
        int64_t outSize = fiaInfo_->opParamInfo.attenOut.shape->GetStorageShape().GetShapeSize();
        int64_t lseSize =
            fiaInfo_->softmaxLseFlag ? fiaInfo_->opParamInfo.lseOut.shape->GetStorageShape().GetShapeSize() : 0;
        uint32_t singleCoreSize = (outSize + platformInfo_.aivNum - 1) / (platformInfo_.aivNum);
        if (fiaInfo_->isOutQuantEnable) {
            singleCoreSize = AlignUp(singleCoreSize, uint32_t(2));
        }
        tilingData_.baseTiling.fiaEmptyTensorParams.singleCoreSize = singleCoreSize;
        tilingData_.baseTiling.fiaEmptyTensorParams.totalOutputSize =
            static_cast<uint64_t>(std::max(outSize, static_cast<int64_t>(0)));
        tilingData_.baseTiling.fiaEmptyTensorParams.totalSoftMaxLseOutputSize =
            static_cast<uint64_t>(std::max(lseSize, static_cast<int64_t>(0)));
        tilingData_.baseTiling.fiaEmptyTensorParams.needInit = 1;

        tilingKeyInfo_.emptyTensor = true;
        tilingKey_ = GET_TPL_TILING_KEY(tilingKeyInfo_.inputLayout, tilingKeyInfo_.config, tilingKeyInfo_.pseMode,
                                        tilingKeyInfo_.quantMode, tilingKeyInfo_.hasAttenMask, tilingKeyInfo_.hasRope,
                                        tilingKeyInfo_.kvLayoutType, tilingKeyInfo_.isFd, tilingKeyInfo_.emptyTensor,
                                        tilingKeyInfo_.enableKvPrefix, tilingKeyInfo_.enableS1OutSplit);

        workspaceSize_ = platformInfo_.defaultSysWorkspaceSize;
        CalcNumBlocks(platformInfo_.aicNum);

        if ((SetNumBlocks(numBlocks_) != ge::GRAPH_SUCCESS) || (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
            (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) ||
            (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS)) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    if (fiaInfo_->isMaxWorkspace) {
        // tiling下沉场景，无法获取到actual_seq，分核结果未知，workspace设置成最大
        CalcMaxWorkspaceSize();
        GenTilingKey();
        CalcNumBlocks(platformInfo_.aicNum);

        if ((SetNumBlocks(numBlocks_) != ge::GRAPH_SUCCESS) || (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
            (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) ||
            (SetScheduleMode(scheduleMode_) != ge::GRAPH_SUCCESS)) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    InitImplParam();
    SplitPolicy();
    FillTiling();
    CalcScheduleMode();
    CalcWorkspaceSize();
    GenTilingKey();

    if ((SetNumBlocks(numBlocks_) != ge::GRAPH_SUCCESS) || (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) || (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS) ||
        (SetScheduleMode(scheduleMode_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingFullQuantMxArch35::SetPlatMemoryInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(fiaInfo_->opName, "The platformInfoPtr is null!"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    platformInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    platformInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
    platformInfo_.cvRatio = platformInfo_.aivNum / platformInfo_.aicNum;
    platformInfo_.coreNum = platformInfo_.aivNum;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformInfo_.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo_.l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo_.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo_.l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, platformInfo_.l2Size);

    platformInfo_.defaultSysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGI(fiaInfo_->opName, "AIV:%u AIC:%u L0A:%lu L0B:%lu L0C:%lu UB:%lu L1:%lu L2:%lu", platformInfo_.aivNum,
            platformInfo_.aicNum, platformInfo_.l0aSize, platformInfo_.l0bSize, platformInfo_.l0cSize,
            platformInfo_.ubSize, platformInfo_.l1Size, platformInfo_.l2Size);

    return ge::GRAPH_SUCCESS;
}

void FiaTilingFullQuantMxArch35::InitImplParam()
{
    const gert::Tensor *actSeqLenQ = fiaInfo_->opParamInfo.actualSeqLengthsQ.tensor;
    const gert::Tensor *actSeqLenKV = fiaInfo_->opParamInfo.actualSeqLengths.tensor;
    const gert::Tensor *actSharedPrefixLen = fiaInfo_->opParamInfo.actualSharedPrefixLen.tensor;
    uint32_t actSeqLenQDims = 0;
    uint32_t actSeqLenKVDims = 0;
    uint32_t actSharedPrefixLenDims = 0;
    if (fiaInfo_->isMaxWorkspace) {
        actualSeqLenQFlag_ = false;
        actualSeqLenKVFlag_ = false;
    } else {
        actSeqLenQDims = (actSeqLenQ != nullptr && actSeqLenQ->GetShapeSize() > 0)
                            ? static_cast<uint32_t>(actSeqLenQ->GetShapeSize()) : 0U;
        actSeqLenKVDims = (actSeqLenKV != nullptr && actSeqLenKV->GetShapeSize() > 0)
                            ? static_cast<uint32_t>(actSeqLenKV->GetShapeSize()) : 0U;
        actSharedPrefixLenDims = (actSharedPrefixLen != nullptr && actSharedPrefixLen->GetShapeSize() > 0)
                                    ? static_cast<uint32_t>(actSharedPrefixLen->GetShapeSize()) : 0U;
        actualSeqLenQFlag_ =
            !((actSeqLenQDims == 0) || (actSeqLenQ == nullptr) || (actSeqLenQ->GetData<int64_t>() == nullptr));
        actualSeqLenKVFlag_ =
            !((actSeqLenKVDims == 0) || (actSeqLenKV == nullptr) || (actSeqLenKV->GetData<int64_t>() == nullptr));
    }

    if (!fiaInfo_->sysPrefixFlag) {
        actualSharedPrefixLenFlag_ = false;
    } else {
        actualSharedPrefixLenFlag_ = !((actSharedPrefixLenDims == 0) || (actSharedPrefixLen == nullptr) ||
                                       (actSharedPrefixLen->GetData<int64_t>() == nullptr));
    }

    for (uint32_t bIdx = 0; bIdx < fiaInfo_->bSize; bIdx++) {
        if (actualSeqLenQFlag_) {
            int64_t actSeqLenQData = 0;
            if (actSeqLenQDims == 1) {
                actSeqLenQData = actSeqLenQ->GetData<int64_t>()[0];
            } else {
                if (fiaInfo_->qLayout != FiaLayout::TND && fiaInfo_->qLayout != FiaLayout::NTD) {
                    actSeqLenQData = actSeqLenQ->GetData<int64_t>()[bIdx];
                } else {
                    if (bIdx == 0) {
                        actSeqLenQData = actSeqLenQ->GetData<int64_t>()[bIdx];
                    } else {
                        actSeqLenQData =
                            actSeqLenQ->GetData<int64_t>()[bIdx] - actSeqLenQ->GetData<int64_t>()[bIdx - 1];
                    }
                }
            }
            actualSeqLengthsQ_.push_back(actSeqLenQData);
        } else {
            actualSeqLengthsQ_.push_back(fiaInfo_->s1Size);
        }

        if (actualSeqLenKVFlag_) {
            if (actSeqLenKVDims == 1) {
                actualSeqLengthsKV_.push_back(actSeqLenKV->GetData<int64_t>()[0]);
            } else {
                if (fiaInfo_->kvLayout != FiaLayout::TND && fiaInfo_->kvLayout != FiaLayout::NTD) {
                    actualSeqLengthsKV_.push_back(actSeqLenKV->GetData<int64_t>()[bIdx]);
                } else {
                    if (bIdx == 0) {
                        actualSeqLengthsKV_.push_back(actSeqLenKV->GetData<int64_t>()[bIdx]);
                    } else {
                        actualSeqLengthsKV_.push_back(actSeqLenKV->GetData<int64_t>()[bIdx] -
                                                      actSeqLenKV->GetData<int64_t>()[bIdx - 1]);
                    }
                }
            }
        } else if (fiaInfo_->kvStorageMode == KvStorageMode::TENSOR_LIST) {
            actualSeqLengthsKV_.push_back(fiaInfo_->kvListSeqLens[bIdx]);
        } else {
            actualSeqLengthsKV_.push_back(fiaInfo_->s2Size);
        }
    }
}

void FiaTilingFullQuantMxArch35::AdjustSinnerAndSouter()
{
    // mxfp8 基本块大小为 128*512
    sOuterFactor_ = SOUTER_64;
    sInnerFactor_ = SINNER_512;
    OP_LOGI(fiaInfo_->opName, "Souter:%u SInner:%u", sOuterFactor_, sInnerFactor_);
}

void FiaTilingFullQuantMxArch35::GetPreNextTokensLeftUp(int64_t actualSeqLength, int64_t actualSeqLengthKV,
                                                        int64_t &preTokensLeftUp, int64_t &nextTokensLeftUp)
{
    if (fiaInfo_->sparseMode == SPARSE_MODE_RIGHT_DOWN) {
        preTokensLeftUp = SPARSE_MODE_INT_MAX;
        if (fiaInfo_->ropeMode == RopeMode::ROPE_SPLIT && fiaInfo_->vHeadDim == 512) {
            if (fiaInfo_->qLayout == FiaLayout::BSND || fiaInfo_->qLayout == FiaLayout::BSH ||
                fiaInfo_->qLayout == FiaLayout::TND) {
                nextTokensLeftUp = actualSeqLengthKV * fiaInfo_->gSize - actualSeqLength;
            } else { // BNSD场景下分核不做优化
                nextTokensLeftUp = SPARSE_MODE_INT_MAX;
            }
        } else {
            nextTokensLeftUp = actualSeqLengthKV - actualSeqLength;
        }
    } else if (fiaInfo_->sparseMode == SPARSE_MODE_BAND) {
        preTokensLeftUp = fiaInfo_->preToken - actualSeqLengthKV + actualSeqLength;
        nextTokensLeftUp = fiaInfo_->nextToken + actualSeqLengthKV - actualSeqLength;
    } else {
        preTokensLeftUp = fiaInfo_->preToken;
        nextTokensLeftUp = fiaInfo_->nextToken;
    }
}

void FiaTilingFullQuantMxArch35::FixParamWithRowInvalid(int64_t &actualSeqLength, int64_t actualSeqLengthKV,
                                                        int64_t &preTokensLeftUp, int64_t &nextTokensLeftUp)
{
    // 若出现行无效，需要重新计算nexttokens，pretokens，actualseqlen，以便正确计算分核核数
    int64_t nextTokensError = (nextTokensLeftUp < 0) ? -nextTokensLeftUp : 0;
    nextTokensError = nextTokensError > actualSeqLength ? actualSeqLength : nextTokensError;
    int64_t preTokensError = 0;
    if (fiaInfo_->mlaMode == MlaMode::ROPE_SPLIT_D512) {
        preTokensError = (actualSeqLength > actualSeqLengthKV * fiaInfo_->gSize + preTokensLeftUp) ?
                             (actualSeqLength - actualSeqLengthKV * fiaInfo_->gSize - preTokensLeftUp) :
                             0;
    } else {
        preTokensError = (actualSeqLength > actualSeqLengthKV + preTokensLeftUp) ?
                             (actualSeqLength - actualSeqLengthKV - preTokensLeftUp) :
                             0;
    }
    preTokensError = preTokensError > actualSeqLength ? actualSeqLength : preTokensError;

    // 若出现上方行无效，需要重新计算nexttokens，pretokens，actualseqlen
    nextTokensLeftUp += nextTokensError;
    preTokensLeftUp -= nextTokensError;
    actualSeqLength -= nextTokensError;

    // 若出现下方行无效，需要重新计算actualseqlen
    actualSeqLength -= preTokensError;
}

bool FiaTilingFullQuantMxArch35::CheckS1OutSplit()
{
    return false;
    if (fiaInfo_->isOutQuantEnable) {
        return false;
    }

    if (fiaInfo_->sysPrefixFlag || fiaInfo_->kvPaddingSizeFlag || fiaInfo_->qPaddingSizeFlag ||
        fiaInfo_->learnableSinkFlag || fiaInfo_->enableAlibiPse) {
        return false;
    }

    if (fiaInfo_->sparseMode == SPARSE_MODE_BAND ||
        (fiaInfo_->sparseMode == SPARSE_MODE_NO_MASK && fiaInfo_->attenMaskFlag)) {
        return false;
    }

    // 仅支持非量化，占用2B
    const int64_t dataTypeSize = 2U;
    int64_t bnSize = std::min(fiaInfo_->bSize * fiaInfo_->n2Size, platformInfo_.aicNum);

    // 当所需的L2cache资源的超过系统配置一半时，开启S1外切分核优化L2cache复用率，乘2是经验值，后续进行优化
    return bnSize * fiaInfo_->s2Size * (fiaInfo_->qkHeadDim + fiaInfo_->vHeadDim) * dataTypeSize * 2 >=
           platformInfo_.l2Size;
}

void FiaTilingFullQuantMxArch35::SplitOutSeq()
{
    uint32_t curCoreNum = platformInfo_.aicNum;
    uint32_t sOuterSize = sOuterFactor_ * CV_RATIO;
    int64_t totalSize = 0;
    for (uint32_t bIdx = 0; bIdx < fiaInfo_->bSize; bIdx++) {
        int64_t actualSeqLengthsTmp = actualSeqLengthsQ_[bIdx]; // 用于存放减去行无效后，真实的actseqlen
        int64_t preTokensLeftUp = 0;
        int64_t nextTokensLeftUp = 0;
        GetPreNextTokensLeftUp(actualSeqLengthsQ_[bIdx], actualSeqLengthsKV_[bIdx] + fiaInfo_->systemPrefixLen,
                               preTokensLeftUp, nextTokensLeftUp);
        FixParamWithRowInvalid(actualSeqLengthsTmp, actualSeqLengthsKV_[bIdx] + fiaInfo_->systemPrefixLen,
                               preTokensLeftUp, nextTokensLeftUp);

        int64_t outerBlockNums = (actualSeqLengthsTmp * fiaInfo_->gSize + static_cast<int64_t>(sOuterSize) - 1) /
                                 static_cast<int64_t>(sOuterSize) * fiaInfo_->n2Size;
        if (actualSeqLengthsTmp == 0 || actualSeqLengthsKV_[bIdx] == 0) {
            outerBlockNums = fiaInfo_->n2Size;
        }
        totalSize += outerBlockNums;
        OP_LOGD(fiaInfo_->opName,
                "bIdx:%u, sOuterSize:%u, sactualSeqLengthsQ_[bIdx]:%lld, actualSeqLengthsKV_[bIdx]:%lld, "
                "outerBlockNums:%lld, totalSize:%lld\n",
                bIdx, sOuterSize, actualSeqLengthsQ_[bIdx], actualSeqLengthsKV_[bIdx], outerBlockNums, totalSize);
    }

    int64_t actualUsedCoreNum = std::min(totalSize, static_cast<int64_t>(curCoreNum));
    CalcNumBlocks(actualUsedCoreNum);
    tilingData_.baseTiling.fiaS1OuterSplitCoreParams.totalSize = totalSize;
    tilingData_.baseTiling.fiaS1OuterSplitCoreParams.enableS1OutSplit = true;
    OP_LOGD(fiaInfo_->opName, "actualUsedCoreNum:%lld\n", actualUsedCoreNum);
    OP_LOGD(fiaInfo_->opName, "totalSize:%lld\n", totalSize);
    OP_LOGD(fiaInfo_->opName, "gqa enableS1OutSplit\n");
}

bool FiaTilingFullQuantMxArch35::CheckQKVActualSeqLengthsRight()
{
    for (uint32_t bIdx = 0; bIdx < fiaInfo_->bSize; bIdx++) {
        if ((actualSeqLengthsQ_[bIdx] % SOUTER_32 > 0) || (actualSeqLengthsKV_[bIdx] <= SINNER_128)) {
            return false;
        }
    }
    return true;
}

void FiaTilingFullQuantMxArch35::CreateSplitInput(split_core_v2::BaseInfo &baseInfo,
                                                  split_core_v2::SplitParam &splitParam)
{
    baseInfo.bSize = fiaInfo_->bSize;
    baseInfo.n2Size = decodeS1GMerge_ ? fiaInfo_->n2Size : fiaInfo_->n1Size;
    baseInfo.gSize = decodeS1GMerge_ ? fiaInfo_->gSize : 1;
    baseInfo.s1Size = fiaInfo_->s1Size;
    baseInfo.s2Size = fiaInfo_->s2Size;
    baseInfo.actualLenQDims = fiaInfo_->actualLenQDims;
    baseInfo.actualLenKvDims = fiaInfo_->actualLenKvDims;
    baseInfo.preToken = fiaInfo_->preToken;
    baseInfo.nextToken = fiaInfo_->nextToken;
    baseInfo.isS1G = (fiaInfo_->qLayout == FiaLayout::BSH) || (fiaInfo_->qLayout == FiaLayout::BSND) ||
                     (fiaInfo_->qLayout == FiaLayout::TND);
    baseInfo.sparseMode = fiaInfo_->sparseMode;
    baseInfo.attenMaskFlag = fiaInfo_->attenMaskFlag;

    if (fiaInfo_->qLayout == FiaLayout::TND) {
        baseInfo.isAccumSeqS1 = true;
        baseInfo.isAccumSeqS2 = !fiaInfo_->pageAttentionFlag;
    } else {
        baseInfo.isAccumSeqS1 = false;
        baseInfo.isAccumSeqS2 = false;
    }
    const gert::Tensor *actSeqLenData = fiaInfo_->opParamInfo.actualSeqLengthsQ.tensor;
    const gert::Tensor *actSeqLenDataKV = fiaInfo_->opParamInfo.actualSeqLengths.tensor;
    if (actSeqLenData != nullptr) {
        baseInfo.actualSeqS1Size.reserve(baseInfo.bSize);
        const int64_t *s1Ptr = actSeqLenData->GetData<int64_t>();
        for (uint32_t i = 0; i < baseInfo.bSize; i++) {
            baseInfo.actualSeqS1Size.emplace_back(s1Ptr[i]);
        }
    }
    if (actSeqLenDataKV != nullptr) {
        baseInfo.actualSeqS2Size.reserve(baseInfo.bSize);
        const int64_t *s2Ptr = actSeqLenDataKV->GetData<int64_t>();
        for (uint32_t i = 0; i < baseInfo.bSize; i++) {
            baseInfo.actualSeqS2Size.emplace_back(s2Ptr[i]);
        }
    }
    if (fiaInfo_->sysPrefixFlag) {
        baseInfo.actualSeqPrefixSize = fiaInfo_->systemPrefixLen;
    }
    splitParam.mBaseSize = sOuterFactor_ * CV_RATIO;
    splitParam.s2BaseSize = sInnerFactor_;
    splitParam.gS1BaseSizeOfFd = 8;
    splitParam.streamK = false;
}

void FiaTilingFullQuantMxArch35::SetSplitOutput(const split_core_v2::FAMetaData &splitRes)
{
    // FA Metadata Generate
    for (size_t i = 0; i < NPU_AIC_CORE_NUM; ++i) {
        if (i >= splitRes.usedCoreNum) {
            tilingData_.fiaMetaData.FAMetadata[i][FA_CORE_ENABLE_INDEX] = 0; // AIC disenable
            continue;
        }
        tilingData_.fiaMetaData.FAMetadata[i][FA_CORE_ENABLE_INDEX] = 1; // AIC enable
        // FA START
        tilingData_.fiaMetaData.FAMetadata[i][FA_BN2_START_INDEX] = (i == 0) ? 0 : splitRes.bN2End[i - 1];
        tilingData_.fiaMetaData.FAMetadata[i][FA_M_START_INDEX] = (i == 0) ? 0 : splitRes.mEnd[i - 1];
        tilingData_.fiaMetaData.FAMetadata[i][FA_S2_START_INDEX] = (i == 0) ? 0 : splitRes.s2End[i - 1];
        // FA END
        tilingData_.fiaMetaData.FAMetadata[i][FA_BN2_END_INDEX] = splitRes.bN2End[i];
        tilingData_.fiaMetaData.FAMetadata[i][FA_M_END_INDEX] = splitRes.mEnd[i];
        tilingData_.fiaMetaData.FAMetadata[i][FA_S2_END_INDEX] = splitRes.s2End[i];
        // FA idx
        tilingData_.fiaMetaData.FAMetadata[i][FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX] =
            splitRes.firstFdDataWorkspaceIdx[i];
    }

    // FD Metadata Generate
    const split_core_v2::FlashDecodeResult &fdRes = splitRes.fdRes;
    for (size_t i = 0; i < NPU_AIV_CORE_NUM; ++i) {
        if (i >= fdRes.fdUsedVecNum) {
            tilingData_.fiaMetaData.FDMetadata[i][FD_CORE_ENABLE_INDEX] = 0; // AIV disenable
            continue;
        }
        tilingData_.fiaMetaData.FDMetadata[i][FD_CORE_ENABLE_INDEX] = 1; // AIV enable
        uint32_t curFdIdx = fdRes.taskIdx[i];
        tilingData_.fiaMetaData.FDMetadata[i][FD_BN2_IDX_INDEX] = fdRes.fdBN2Idx[curFdIdx];
        tilingData_.fiaMetaData.FDMetadata[i][FD_M_IDX_INDEX] = fdRes.fdMIdx[curFdIdx];
        tilingData_.fiaMetaData.FDMetadata[i][FD_S2_SPLIT_NUM_INDEX] = fdRes.fdS2SplitNum[curFdIdx];
        tilingData_.fiaMetaData.FDMetadata[i][FD_WORKSPACE_IDX_INDEX] = fdRes.fdWorkspaceIdx[curFdIdx];
        tilingData_.fiaMetaData.FDMetadata[i][FD_M_START_INDEX] = fdRes.mStart[i];
        tilingData_.fiaMetaData.FDMetadata[i][FD_M_NUM_INDEX] = fdRes.mLen[i];
    }
}

void FiaTilingFullQuantMxArch35::SplitPolicy()
{
    AdjustSinnerAndSouter(); // 确定tiling切块

    // decode场景qscale shape为[N2, T, G, D//64, 2]；prefill qscale shape为[T, N, D//64, 2]
    decodeS1GMerge_ = (fiaInfo_->opParamInfo.dequantScaleQuery.tensor->GetStorageShape().GetDimNum() == 5);

    split_core_v2::BaseInfo baseInfo{};
    split_core_v2::SplitParam splitParam{};
    CreateSplitInput(baseInfo, splitParam);

    enableS1OutSplit = CheckS1OutSplit();
    if (enableS1OutSplit) {
        SplitOutSeq();
    } else {
        split_core_v2::FAMetaData result{platformInfo_.aicNum, platformInfo_.cvRatio};
        split_core_v2::SplitCore(platformInfo_.aicNum, baseInfo, splitParam, result);
        SetSplitOutput(result);
        CalcNumBlocks(result.usedCoreNum);
        flashDecodeFlag_ = (result.fdRes.fdNum > 0);
    }
    fiaInfo_->isExistRowInvalid = (fiaInfo_->needInit || IsExistRowInvalid(baseInfo) ||
                                   IsActualSeqLengthsKVHasZero(baseInfo));
}

bool FiaTilingFullQuantMxArch35::IsActualSeqLengthsKVHasZero(const split_core_v2::BaseInfo &baseInfo)
{
    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        uint32_t s2Size = split_core_v2::GetS2SeqSize(bIdx, baseInfo);
        if (s2Size == 0) {
            return true;
        }
    }
    return false;
}

bool FiaTilingFullQuantMxArch35::IsExistRowInvalid(const split_core_v2::BaseInfo &baseInfo)
{
    if (!baseInfo.attenMaskFlag) {
        return false;
    }

    auto mode = static_cast<split_core_v2::SparseMode>(baseInfo.sparseMode);
    if (mode == split_core_v2::SparseMode::LEFT_UP_CAUSAL) {
        return false;
    }

    if (mode == split_core_v2::SparseMode::ALL_MASK) {
        return true;
    }

    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        int32_t s1Size = split_core_v2::GetS1SeqSize(bIdx, baseInfo);
        int32_t s2Size = split_core_v2::GetS2SeqSize(bIdx, baseInfo);
        if ((s1Size == 0) || (s2Size == 0)) {
            // 空tensor认为不会有无效行
            continue;
        }

        int64_t safePreToken = baseInfo.preToken;
        int64_t safeNextToken = baseInfo.nextToken;
        int64_t preTokenLeftUp = 0;
        int64_t nextTokenLeftUp = 0;

        GetSafeActToken(mode, s1Size, s2Size, safePreToken, safeNextToken);
        if (mode == split_core_v2::SparseMode::BAND) {
            preTokenLeftUp = safePreToken;
            nextTokenLeftUp = s2Size - s1Size + safeNextToken;
        } else if (mode == split_core_v2::SparseMode::DEFAULT_MASK) {
            preTokenLeftUp = s2Size - s1Size + safePreToken;
            nextTokenLeftUp = safeNextToken;
        } else {
            preTokenLeftUp = 0;
            nextTokenLeftUp = s2Size - s1Size;
        }

        if ((preTokenLeftUp < 0) || (nextTokenLeftUp < 0)) {
            return true;
        }
    }
    return false;
}

void FiaTilingFullQuantMxArch35::GetSafeActToken(split_core_v2::SparseMode mode, int64_t actSeqLensQ,
                                                 int64_t actSeqLensKv, int64_t &safePreToken, int64_t &safeNextToken)
{
    if (mode == split_core_v2::SparseMode::DEFAULT_MASK) {
        safePreToken = std::max(-actSeqLensKv, safePreToken);
        safePreToken = std::min(safePreToken, actSeqLensQ);
        safeNextToken = std::max(-actSeqLensQ, safeNextToken);
        safeNextToken = std::min(safeNextToken, actSeqLensKv);
    } else if (mode == split_core_v2::SparseMode::BAND) {
        safePreToken = std::max(-actSeqLensQ, safePreToken);
        safePreToken = std::min(safePreToken, actSeqLensKv);
        safeNextToken = std::max(-actSeqLensKv, safeNextToken);
        safeNextToken = std::min(safeNextToken, actSeqLensQ);
    }
}

void FiaTilingFullQuantMxArch35::UpdateTilingKeyConfig()
{
    auto sOuter = sOuterFactor_ * platformInfo_.cvRatio;
    auto sInner = sInnerFactor_;
    auto dSize = fiaInfo_->qkHeadDim;
    auto dVsize = fiaInfo_->vHeadDim;

    if (dSize <= DSIZE_64) {
        dSize = DSIZE_64;
    } else if (dSize <= DSIZE_128) {
        dSize = DSIZE_128;
    } else if (dSize <= DSIZE_256) {
        dSize = DSIZE_256;
    } else if (dSize <= DSIZE_512) {
        dSize = DSIZE_512;
    } else if (dSize <= DSIZE_576) {
        dSize = DSIZE_576;
    }

    if (dVsize <= DSIZE_64) {
        dVsize = DSIZE_64;
    } else if (dVsize <= DSIZE_128) {
        dVsize = DSIZE_128;
    } else if (dVsize <= DSIZE_256) {
        dVsize = DSIZE_256;
    } else if (dVsize <= DSIZE_512) {
        dVsize = DSIZE_512;
    }

    if (sOuter == SOUTER_128 && sInner == SINNER_512 && dSize == DSIZE_64 && dVsize == DSIZE_64) {
        tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned512_DAligned64_DVAligned64;
    } else if (sOuter == SOUTER_128 && sInner == SINNER_512 && dSize == DSIZE_128 && dVsize == DSIZE_128) {
        tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned512_DAligned128_DVAligned128;
    } else {
        tilingKeyInfo_.config = Config_S1Aligned128_S2Aligned512_DAligned128_DVAligned128;
    }
}

void FiaTilingFullQuantMxArch35::UpdateTilingKeyLayout()
{
    if (fiaInfo_->qLayout == FiaLayout::BNSD) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BNSD_BNSD;
    } else if (fiaInfo_->qLayout == FiaLayout::TND) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_TND_TND;
    } else if (fiaInfo_->qLayout == FiaLayout::NTD) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_NTD_NTD;
    } else if (fiaInfo_->qLayout == FiaLayout::BSH || fiaInfo_->qLayout == FiaLayout::BSND) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BSH_BSH;
    }
}

void FiaTilingFullQuantMxArch35::UpdateTilingKeyPseMode()
{
    if (!fiaInfo_->pseShiftFlag && !fiaInfo_->enableAlibiPse) {
        tilingKeyInfo_.pseMode = PSE_MODE_PSE_NONE_TYPE;
    } else {
        tilingKeyInfo_.pseMode = *(fiaInfo_->opParamInfo.pseType);
    }
}

void FiaTilingFullQuantMxArch35::UpdateTilingKeyQuantMode()
{
    if (decodeS1GMerge_) {
        tilingKeyInfo_.quantMode = FULLQUANT_MODE_QKV_MXFP8_DECODE;
    } else {
        tilingKeyInfo_.quantMode = FULLQUANT_MODE_QKV_MXFP8_PREFILL;
    }
}

void FiaTilingFullQuantMxArch35::UpdateTilingKeyHasRope()
{
    tilingKeyInfo_.hasRope = false;
    if (fiaInfo_->ropeMode == RopeMode::ROPE_SPLIT) {
        tilingKeyInfo_.hasRope = true;
    }
}

void FiaTilingFullQuantMxArch35::UpdateTilingKeyInfo()
{
    if (fiaInfo_->emptyTensorFlag) {
        tilingKeyInfo_.emptyTensor = fiaInfo_->emptyTensorFlag;
    } else {
        UpdateTilingKeyLayout();
        UpdateTilingKeyConfig();
        UpdateTilingKeyPseMode();
        UpdateTilingKeyQuantMode();
        tilingKeyInfo_.isFd = flashDecodeFlag_;
        if (fiaInfo_->sysPrefixFlag) {
            tilingKeyInfo_.isFd = false;
        }
        tilingKeyInfo_.hasAttenMask = fiaInfo_->attenMaskFlag;
        UpdateTilingKeyHasRope();

        tilingKeyInfo_.kvLayoutType = tilingData_.baseTiling.fiaPageAttentionParams.paLayoutType;

        if (fiaInfo_->pageAttentionFlag) {
            if (tilingData_.baseTiling.fiaPageAttentionParams.paLayoutType == 0) { // BNBD
                tilingKeyInfo_.kvLayoutType = 2;
            } else if (tilingData_.baseTiling.fiaPageAttentionParams.paLayoutType == 1) { // BBH
                tilingKeyInfo_.kvLayoutType = 1;
            } else { // PA NZ
                tilingKeyInfo_.kvLayoutType = 3;
            }
        } else {
            tilingKeyInfo_.kvLayoutType = 0;
        }

        tilingKeyInfo_.emptyTensor = fiaInfo_->emptyTensorFlag;
        tilingKeyInfo_.enableKvPrefix = fiaInfo_->sysPrefixFlag;
        // tilingKeyInfo_.enableS1OutSplit = enableS1OutSplit;
        tilingKeyInfo_.isReconstructTemp = true;
    }
}

void FiaTilingFullQuantMxArch35::GenTilingKey()
{
    UpdateTilingKeyInfo();
    tilingKey_ = GET_TPL_TILING_KEY(tilingKeyInfo_.inputLayout, tilingKeyInfo_.config, tilingKeyInfo_.pseMode,
                                    tilingKeyInfo_.quantMode, tilingKeyInfo_.hasAttenMask, tilingKeyInfo_.hasRope,
                                    tilingKeyInfo_.kvLayoutType, tilingKeyInfo_.isFd, tilingKeyInfo_.emptyTensor,
                                    tilingKeyInfo_.enableKvPrefix, tilingKeyInfo_.enableS1OutSplit,
                                    tilingKeyInfo_.isReconstructTemp);

    OP_LOGI(fiaInfo_->opName, "The tilingkey is %llu.", tilingKey_);
    OP_LOGI(fiaInfo_->opName,
            "The tilingkey param is inOutLayoutType: %llu, config: %llu, pseMode: %llu, quantMode: %llu, "
            "hasAttenMask: %u, hasRope: %u, kvLayoutType: %llu, isFd: %u, emptyTensor: %u, "
            "enableKvPrefix: %u, enableS1OutSplit: %u, isReconstructTemp: %u.",
            tilingKeyInfo_.inputLayout, tilingKeyInfo_.config, tilingKeyInfo_.pseMode, tilingKeyInfo_.quantMode,
            tilingKeyInfo_.hasAttenMask, tilingKeyInfo_.hasRope, tilingKeyInfo_.kvLayoutType, tilingKeyInfo_.isFd,
            tilingKeyInfo_.emptyTensor, tilingKeyInfo_.enableKvPrefix, tilingKeyInfo_.enableS1OutSplit,
            tilingKeyInfo_.isReconstructTemp);
}

void FiaTilingFullQuantMxArch35::CalcNumBlocks(uint32_t aicNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(fiaInfo_->platformInfo);
    auto aivNum = aicNum * platformInfo_.cvRatio;

    numBlocks_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_LOGI(fiaInfo_->opName, "FIA block dim: %u aiv Num: %u aic Num: %u.", numBlocks_, aivNum, aicNum);
}

void FiaTilingFullQuantMxArch35::CalcWorkspaceSize()
{
    size_t sysWorkspaceSize = platformInfo_.defaultSysWorkspaceSize;
    uint32_t mSize = sOuterFactor_ * platformInfo_.cvRatio;
    uint32_t dSize = fiaInfo_->vHeadDim;
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

    workspaceSize_ = sysWorkspaceSize;

    int64_t bmm2Bytes = 0;
    int64_t vec2Bytes = 0;
    int64_t bmm2ResBlockSize = dVBasicBlock;
    if (dVBasicBlock > DSIZE_256) {
        bmm2ResBlockSize = DSIZE_512;
    }
    if (dSize > DSIZE_128) {
        bmm2Bytes = mSize * bmm2ResBlockSize * sizeof(float);
        if (dVBasicBlock > DSIZE_256) {
            vec2Bytes = mSize * dVBasicBlock * sizeof(float);
        }
    }
    workspaceSize_ += (bmm2Bytes + vec2Bytes) * 3 * numBlocks_; // 3: perload 2次 需要2+1

    if (flashDecodeFlag_) {
        uint32_t faTmpAttenGmSize = numBlocks_ * 2 * mSize * dSize; // 每个核最多有2次写到workspace
        uint32_t fatmpResLseGmSize = numBlocks_ * 2 * mSize * 8;
        workspaceSize_ += (faTmpAttenGmSize + 2 * fatmpResLseGmSize) * sizeof(float); // ResLse有2份，sum和max
        tilingData_.baseTiling.fiaWorkspaceParams.accumOutSize = faTmpAttenGmSize;
        tilingData_.baseTiling.fiaWorkspaceParams.logSumExpSize = fatmpResLseGmSize;
    }

    OP_LOGI(fiaInfo_->opName, "Workspaces: %ld", workspaceSize_);
}

void FiaTilingFullQuantMxArch35::FillTiling()
{
    ComputeTilingData();
    SetFATilingData();
    PrintAllTilingData();
}

void FiaTilingFullQuantMxArch35::ComputeTilingData()
{
    // 处理不能直接从fiaInfo赋值到tiling data
    /*
     *  mask,sparse mode 相关tiling data
     */
    if (fiaInfo_->attenMaskFlag) {
        uint64_t maskBatch = 1;
        uint64_t maskDimNum = fiaInfo_->opParamInfo.attenMask.tensor->GetStorageShape().GetDimNum();
        uint64_t maskS1Size = 2048;
        uint64_t maskS2Size = 2048;
        if (maskDimNum != 2 || fiaInfo_->s1Size == 1) {
            maskBatch = fiaInfo_->opParamInfo.attenMask.tensor->GetStorageShape().GetDim(0);
        }
        tilingData_.baseTiling.fiaAttenMaskParams.attenMaskBatch = maskBatch;
        maskS2Size = static_cast<uint64_t>(
            std::max(fiaInfo_->opParamInfo.attenMask.tensor->GetStorageShape().GetDim(maskDimNum - 1),
                     static_cast<int64_t>(0)));
        maskS1Size = static_cast<uint64_t>(
            std::max(fiaInfo_->opParamInfo.attenMask.tensor->GetStorageShape().GetDim(maskDimNum - 2),
                     static_cast<int64_t>(0)));
        tilingData_.baseTiling.fiaAttenMaskParams.attenMaskS1Size = static_cast<uint32_t>(maskS1Size);
        tilingData_.baseTiling.fiaAttenMaskParams.attenMaskS2Size = static_cast<uint32_t>(maskS2Size);
    } else {
        tilingData_.baseTiling.fiaAttenMaskParams.attenMaskS1Size = 0;
        tilingData_.baseTiling.fiaAttenMaskParams.attenMaskS2Size = 0;
    }
    tilingData_.baseTiling.fiaAttenMaskParams.sparseMode = fiaInfo_->sparseMode;

    if (fiaInfo_->pageAttentionFlag) {
        uint32_t keyCacheDimNum = fiaInfo_->opParamInfo.key.shape->GetStorageShape().GetDimNum();
        if (keyCacheDimNum == 3) { // 3: BBH
            tilingData_.baseTiling.fiaPageAttentionParams.paLayoutType = 1;
        } else if (keyCacheDimNum == 4) { // 4: BNBD
            tilingData_.baseTiling.fiaPageAttentionParams.paLayoutType = 0;
        } else if (keyCacheDimNum == 5) { // 5: PA NZ
            tilingData_.baseTiling.fiaPageAttentionParams.paLayoutType = 2;
        }
    }
}

void FiaTilingFullQuantMxArch35::SetFATilingData()
{
    tilingData_.baseTiling.fiaBaseParams.bSize = fiaInfo_->bSize;
    tilingData_.baseTiling.fiaBaseParams.t1Size = fiaInfo_->qTSize;
    tilingData_.baseTiling.fiaBaseParams.t2Size = fiaInfo_->kTSize;
    tilingData_.baseTiling.fiaBaseParams.n2Size = fiaInfo_->n2Size;
    tilingData_.baseTiling.fiaBaseParams.gSize = fiaInfo_->gSize;
    tilingData_.baseTiling.fiaBaseParams.s1Size = fiaInfo_->s1Size;
    tilingData_.baseTiling.fiaBaseParams.s2Size = fiaInfo_->s2Size;
    tilingData_.baseTiling.fiaBaseParams.dSize = fiaInfo_->qkHeadDim;
    tilingData_.baseTiling.fiaBaseParams.dSizeV = fiaInfo_->vHeadDim;
    tilingData_.baseTiling.fiaBaseParams.dSizeRope = fiaInfo_->ropeHeadDim;
    tilingData_.baseTiling.fiaBaseParams.scaleValue = fiaInfo_->scaleValue;
    tilingData_.baseTiling.fiaBaseParams.actualSeqLengthsQSize = fiaInfo_->actualLenQDims;
    tilingData_.baseTiling.fiaBaseParams.actualSeqLengthsKVSize = fiaInfo_->actualLenKvDims;
    tilingData_.baseTiling.fiaBaseParams.isKvContinuous = fiaInfo_->kvStorageMode != KvStorageMode::TENSOR_LIST;
    tilingData_.baseTiling.fiaBaseParams.isSoftMaxLseEnable = fiaInfo_->softmaxLseFlag;
    tilingData_.baseTiling.fiaBaseParams.coreNum = numBlocks_;
    tilingData_.baseTiling.fiaBaseParams.outputLayout = static_cast<uint32_t>(fiaInfo_->outputLayout);
    if (!fiaInfo_->isTensorV1) { // 只有tensorv2才会包含strides信息
        // fiaBaseParams增加strides参数，从fiaInfo获取strides值
        tilingData_.baseTiling.fiaBaseParams.keyStrides.bnStride = fiaInfo_->keyStrides->GetStride(0);
        tilingData_.baseTiling.fiaBaseParams.keyStrides.n2Stride = fiaInfo_->keyStrides->GetStride(1);
        tilingData_.baseTiling.fiaBaseParams.valueStrides.bnStride = fiaInfo_->valueStrides->GetStride(0);
        tilingData_.baseTiling.fiaBaseParams.valueStrides.n2Stride = fiaInfo_->valueStrides->GetStride(1);
        if (fiaInfo_->ropeHeadDim != 0) {  // 传入了rope才赋值
            tilingData_.baseTiling.fiaBaseParams.kRopeStrides.bnStride = fiaInfo_->kRopeStrides->GetStride(0);
            tilingData_.baseTiling.fiaBaseParams.kRopeStrides.n2Stride = fiaInfo_->kRopeStrides->GetStride(1);
        }
        tilingData_.baseTiling.fiaBaseParams.kScaleStrides.bnStride = fiaInfo_->kScaleStrides->GetStride(0);
        tilingData_.baseTiling.fiaBaseParams.kScaleStrides.n2Stride = fiaInfo_->kScaleStrides->GetStride(1);
        tilingData_.baseTiling.fiaBaseParams.vScaleStrides.bnStride = fiaInfo_->vScaleStrides->GetStride(0);
        tilingData_.baseTiling.fiaBaseParams.vScaleStrides.n2Stride = fiaInfo_->vScaleStrides->GetStride(1);
    }
    tilingData_.baseTiling.fiaAttenMaskParams.preTokens = fiaInfo_->preToken;
    tilingData_.baseTiling.fiaAttenMaskParams.nextTokens = fiaInfo_->nextToken;
    tilingData_.baseTiling.fiaAttenMaskParams.isRowInvalidOpen = (fiaInfo_->innerPrecise >> 1) & 1;
    tilingData_.baseTiling.fiaAttenMaskParams.isExistRowInvalid = fiaInfo_->isExistRowInvalid;

    tilingData_.baseTiling.fiaPageAttentionParams.blockSize = fiaInfo_->blockSize;
    uint32_t maxBlockNumPerBatch = 0;
    if (fiaInfo_->kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        maxBlockNumPerBatch = fiaInfo_->opParamInfo.blockTable.tensor->GetStorageShape().GetDim(1);
    }
    tilingData_.baseTiling.fiaPageAttentionParams.maxBlockNumPerBatch = maxBlockNumPerBatch;

    int64_t outSize = fiaInfo_->opParamInfo.attenOut.shape->GetStorageShape().GetShapeSize();
    int64_t lseSize =
        fiaInfo_->softmaxLseFlag ? fiaInfo_->opParamInfo.lseOut.shape->GetStorageShape().GetShapeSize() : 0;
    uint32_t singleCoreSize = (outSize + platformInfo_.aivNum - 1) / (platformInfo_.aivNum);
    if (fiaInfo_->isOutQuantEnable) {
        singleCoreSize = ((singleCoreSize + 1) / 2) * 2;
    }
    tilingData_.baseTiling.fiaEmptyTensorParams.singleCoreSize = singleCoreSize;
    tilingData_.baseTiling.fiaEmptyTensorParams.totalOutputSize = outSize;
    tilingData_.baseTiling.fiaEmptyTensorParams.totalSoftMaxLseOutputSize = lseSize;
    tilingData_.baseTiling.fiaEmptyTensorParams.needInit = fiaInfo_->needInit || needInit_;
}

ge::graphStatus FiaTilingFullQuantMxArch35::SetTilingData(FusedInferAttentionScoreFullQuantTilingData &tilingData)
{
    FusedInferAttentionScoreFullQuantTilingData *tiling =
        context_->GetTilingData<FusedInferAttentionScoreFullQuantTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE(fiaInfo_->opName, "The tiling data is nullptr"), return ge::GRAPH_FAILED);
    *tiling = tilingData;
    return ge::GRAPH_SUCCESS;
}

void FiaTilingFullQuantMxArch35::PrintAllTilingData()
{
    FullQuantTiling &baseTiling = tilingData_.baseTiling;
    FiaBaseParams &fiaBaseParams = baseTiling.fiaBaseParams;
    FiaAttenMaskParams &fiaAttenMaskParams = baseTiling.fiaAttenMaskParams;
    FiaPageAttentionParams &fiaPageAttentionParams = baseTiling.fiaPageAttentionParams;
    FiaWorkspaceParams &fiaWorkspaceParams = baseTiling.fiaWorkspaceParams;
    FiaS1OuterSplitCoreParams &fiaS1OuterSplitCoreParams = baseTiling.fiaS1OuterSplitCoreParams;
    FiaEmptyTensorParams &fiaEmptyTensorParams = baseTiling.fiaEmptyTensorParams;
    FiaMetaData &fiaMetaData = tilingData_.fiaMetaData;

    OP_LOGD(fiaInfo_->opName, "bSize:%d", fiaBaseParams.bSize);
    OP_LOGD(fiaInfo_->opName, "t1Size:%d", fiaBaseParams.t1Size);
    OP_LOGD(fiaInfo_->opName, "t2Size:%d", fiaBaseParams.t2Size);
    OP_LOGD(fiaInfo_->opName, "n2Size:%d", fiaBaseParams.n2Size);
    OP_LOGD(fiaInfo_->opName, "gSize:%d", fiaBaseParams.gSize);
    OP_LOGD(fiaInfo_->opName, "s1Size:%d", fiaBaseParams.s1Size);
    OP_LOGD(fiaInfo_->opName, "s2Size:%d", fiaBaseParams.s2Size);
    OP_LOGD(fiaInfo_->opName, "dSize:%d", fiaBaseParams.dSize);
    OP_LOGD(fiaInfo_->opName, "dSizeV:%d", fiaBaseParams.dSizeV);
    OP_LOGD(fiaInfo_->opName, "dSizeRope:%d", fiaBaseParams.dSizeRope);
    OP_LOGD(fiaInfo_->opName, "actualSeqLengthsQSize:%d", fiaBaseParams.actualSeqLengthsQSize);
    OP_LOGD(fiaInfo_->opName, "actualSeqLengthsKVSize:%d", fiaBaseParams.actualSeqLengthsKVSize);
    OP_LOGD(fiaInfo_->opName, "scaleValue:%f", fiaBaseParams.scaleValue);
    OP_LOGD(fiaInfo_->opName, "isKvContinuous:%d", fiaBaseParams.isKvContinuous);
    OP_LOGD(fiaInfo_->opName, "isSoftMaxLseEnable:%d", fiaBaseParams.isSoftMaxLseEnable);
    OP_LOGD(fiaInfo_->opName, "coreNum:%d", fiaBaseParams.coreNum);

    OP_LOGD(fiaInfo_->opName, "sparseMode:%d", fiaAttenMaskParams.sparseMode);
    OP_LOGD(fiaInfo_->opName, "preTokens:%d", fiaAttenMaskParams.preTokens);
    OP_LOGD(fiaInfo_->opName, "nextTokens:%d", fiaAttenMaskParams.nextTokens);
    OP_LOGD(fiaInfo_->opName, "attenMaskS1Size:%d", fiaAttenMaskParams.attenMaskS1Size);
    OP_LOGD(fiaInfo_->opName, "attenMaskS2Size:%d", fiaAttenMaskParams.attenMaskS2Size);
    OP_LOGD(fiaInfo_->opName, "isRowInvalidOpen:%d", fiaAttenMaskParams.isRowInvalidOpen);
    OP_LOGD(fiaInfo_->opName, "isExistRowInvalid:%d", fiaAttenMaskParams.isExistRowInvalid);

    OP_LOGD(fiaInfo_->opName, "paLayoutType:%d", fiaPageAttentionParams.paLayoutType);
    OP_LOGD(fiaInfo_->opName, "blockSize:%d", fiaPageAttentionParams.blockSize);
    OP_LOGD(fiaInfo_->opName, "maxBlockNumPerBatch:%d", fiaPageAttentionParams.maxBlockNumPerBatch);

    OP_LOGD(fiaInfo_->opName, "accumOutSize:%d", fiaWorkspaceParams.accumOutSize);
    OP_LOGD(fiaInfo_->opName, "logSumExpSize:%d", fiaWorkspaceParams.logSumExpSize);

    OP_LOGD(fiaInfo_->opName, "enableS1OutSplit:%d", fiaS1OuterSplitCoreParams.enableS1OutSplit);
    OP_LOGD(fiaInfo_->opName, "totalSize:%d", fiaS1OuterSplitCoreParams.totalSize);

    OP_LOGD(fiaInfo_->opName, "singleCoreSize:%d", fiaEmptyTensorParams.singleCoreSize);
    OP_LOGD(fiaInfo_->opName, "needInit:%d", fiaEmptyTensorParams.needInit);
    OP_LOGD(fiaInfo_->opName, "totalOutputSize:%d", fiaEmptyTensorParams.totalOutputSize);
    OP_LOGD(fiaInfo_->opName, "totalSoftMaxLseOutputSize:%d", fiaEmptyTensorParams.totalSoftMaxLseOutputSize);

    for (int aicIdx = 0; aicIdx < NPU_AIC_CORE_NUM; ++aicIdx) {
        OP_LOGD(fiaInfo_->opName, "FAMetadata[%d], [0]:%d, [1]:%d, [2]:%d, [3]:%d, [4]:%d, [5]:%d, [6]:%d, [7]:%d",
                aicIdx, fiaMetaData.FAMetadata[aicIdx][0], fiaMetaData.FAMetadata[aicIdx][1],
                fiaMetaData.FAMetadata[aicIdx][2], fiaMetaData.FAMetadata[aicIdx][3], fiaMetaData.FAMetadata[aicIdx][4],
                fiaMetaData.FAMetadata[aicIdx][5], fiaMetaData.FAMetadata[aicIdx][6],
                fiaMetaData.FAMetadata[aicIdx][7]);
    }

    for (int aivIdx = 0; aivIdx < NPU_AIV_CORE_NUM; ++aivIdx) {
        OP_LOGD(fiaInfo_->opName, "FDMetadata[%d], [0]:%d, [1]:%d, [2]:%d, [3]:%d, [4]:%d, [5]:%d, [6]:%d, [7]:%d",
                aivIdx, fiaMetaData.FDMetadata[aivIdx][0], fiaMetaData.FDMetadata[aivIdx][1],
                fiaMetaData.FDMetadata[aivIdx][2], fiaMetaData.FDMetadata[aivIdx][3], fiaMetaData.FDMetadata[aivIdx][4],
                fiaMetaData.FDMetadata[aivIdx][5], fiaMetaData.FDMetadata[aivIdx][6],
                fiaMetaData.FDMetadata[aivIdx][7]);
    }

    int64_t cap = context_->GetRawTilingData()->GetCapacity();
    OP_LOGD(fiaInfo_->opName, "Tiling Data context_ GetCapacity: %lu.", cap);
}

// 值越小表示优先级越高. 对于FIA, 使用3位数表示优先级, 优先级编码含义为:
// 1. 百位代表非量化、伪量化、全量化等场景, 即: 0xx-非量化，1xx-伪量化, 2xx-全量化
// 2. 十位表示gqa、mla、泛化，即: x0x-mla, x1x-gpa, x2x-泛化
// 3. 个位代表特化模板到泛化模板的优先级排序
REGISTER_TILING_TEMPLATE_FIA(FusedInferAttentionScore, FiaTilingFullQuantMxArch35,
                             std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_3510)}), 210);
} // namespace optiling