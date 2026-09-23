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
 * \file flash_attn_tiling.cpp
 * \brief
 */

#include "flash_attn_tiling_arch35.h"
#include "../flash_attn_tiling.h"
#include "../fa_adjust_sinner_souter.h"
#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "../flash_attn_tiling_utils.h"
#include "../../op_kernel/arch35/flash_attn_template_tiling_key.h"
#include "../../../a5_mla_common/op_host/fia_tiling_templates_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
namespace flash_attn {
constexpr uint64_t PRE_LOAD_NUM_GQA_ARCH35 = 3;

void FlashAttnTilingImpl::InitTilingInfo(TilingInfo *tilingInfo)
{
    faInfo_ = static_cast<FaTilingInfo *>(tilingInfo);
}

bool FlashAttnTilingImpl::IsCapable()
{
    return true;
}

void FlashAttnTilingImpl::CalcScheduleMode()
{
    scheduleMode_ = ScheduleMode::BATCH_MODE;
}

ge::graphStatus FlashAttnTilingImpl::DoOpTiling()
{
    OP_CHECK_IF(SetPlatMemoryInfo() != ge::GRAPH_SUCCESS, OP_LOGE(faInfo_->opName, "Set plat memory info fail."),
                return ge::GRAPH_FAILED);

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

ge::graphStatus FlashAttnTilingImpl::SetPlatMemoryInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(faInfo_->opName, "The platformInfoPtr is null!"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    platformInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    platformInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
    platformInfo_.cvRatio = platformInfo_.aivNum / platformInfo_.aicNum;
    platformInfo_.coreNum = platformInfo_.aivNum;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformInfo_.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo_.l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo_.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo_.l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, platformInfo_.l2Size);

    platformInfo_.defaultSysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

void FlashAttnTilingImpl::InitImplParam()
{
    const gert::Tensor *cuSeqLenQ = faInfo_->opParamInfo.cuSeqlensQ.tensor;
    const gert::Tensor *cuSeqLenKV = faInfo_->opParamInfo.cuSeqlensKv.tensor;
    uint32_t cuSeqLenQDims = 0;
    uint32_t cuSeqLenKVDims = 0;
    cuSeqLenQDims = (cuSeqLenQ != nullptr) ? cuSeqLenQ->GetShapeSize() : 0;
    cuSeqLenKVDims = (cuSeqLenKV != nullptr) ? cuSeqLenKV->GetShapeSize() : 0;
    cuSeqLenQFlag_ = !((cuSeqLenQDims == 0) || (cuSeqLenQ == nullptr));
    cuSeqLenKVFlag_ = !((cuSeqLenKVDims == 0) || (cuSeqLenKV == nullptr));

    const gert::Tensor *seqUsedQ = faInfo_->opParamInfo.sequsedQ.tensor;
    const gert::Tensor *seqUsedKv = faInfo_->opParamInfo.sequsedKv.tensor;
    uint32_t seqUsedQDims = (seqUsedQ != nullptr) ? seqUsedQ->GetShapeSize() : 0;
    uint32_t seqUsedKvDims = (seqUsedKv != nullptr) ? seqUsedKv->GetShapeSize() : 0;
    seqUsedQFlag_ = !((seqUsedQDims == 0) || (seqUsedQ == nullptr));
    seqUsedKvFlag_ = !((seqUsedKvDims == 0) || (seqUsedKv == nullptr));
}

void FlashAttnTilingImpl::SplitPolicy()
{
    int64_t winLeft = faInfo_->winLeft;
    int64_t winRight = faInfo_->winRight;
    if (faInfo_->maskMode == static_cast<int64_t>(MaskMode::NO_MASK) ||
        faInfo_->maskMode == static_cast<int64_t>(MaskMode::CAUSAL)) {
        winLeft = MASK_MODE_INT_MAX;
        winRight = MASK_MODE_INT_MAX;
    }
    if (winLeft == -1) {
        winLeft = MASK_MODE_INT_MAX;
    }
    if (winRight == -1) {
        winRight = MASK_MODE_INT_MAX;
    }
    fa_tiling_util::AdjustSinnerAndSouter(static_cast<uint32_t>(faInfo_->vHeadDim),
                                          static_cast<uint32_t>(faInfo_->gSize), faInfo_->maxSeqQ, faInfo_->maxSeqKv,
                                          static_cast<int32_t>(faInfo_->maskMode), winLeft, winRight,
                                          static_cast<uint32_t>(faInfo_->qLayout), sOuterFactor_, sInnerFactor_);
    CalcNumBlocks(platformInfo_.aicNum);
    flashDecodeFlag_ = true;
}

void FlashAttnTilingImpl::UpdateTilingKeyConfig()
{
    if (faInfo_->qkHeadDim == faInfo_->vHeadDim) {
        // 等长 (QK D == V DV) 按 D 值映射:
        //   config=0: Dk=64,  Dv=64,  sOuter=64,  sInner=128
        //   config=1: Dk=64,  Dv=64,  sOuter=32,  sInner=256
        //   config=2: Dk=128, Dv=128, sOuter=64,  sInner=128 (Dk=72 复用, kernel 内 pad 到 128)
        //   config=3: Dk=128, Dv=128, sOuter=32,  sInner=256 (Dk=72 复用, kernel 内 pad 到 128)
        //   config=4: Dk=256, Dv=256, sOuter=64, sInner=128
        //   config=5: Dk=256, Dv=256, sOuter=32, sInner=256
        if (faInfo_->qkHeadDim == 64) {
            tilingKeyInfo_.config = (sOuterFactor_ == fa_tiling_util::SOUTER_64) ? 0 : 1;
        } else if (faInfo_->qkHeadDim == 128 || faInfo_->qkHeadDim == 72) {
            tilingKeyInfo_.config = (sOuterFactor_ == fa_tiling_util::SOUTER_64) ? 2 : 3;
        } else if (faInfo_->qkHeadDim == 256) {
            tilingKeyInfo_.config = (sOuterFactor_ == fa_tiling_util::SOUTER_64) ? 4 : 5;
        }
    } else {
        // 不等长 (qkHeadDim != vHeadDim) 按 (QK D, V DV) 组合映射:
        if (faInfo_->qkHeadDim == 192 && faInfo_->vHeadDim == 128) {
            // QK D=192 且 V DV=128 (MLA 非吸收形态) 进入此分支:
            //   config=6: Dk=192, Dv=128, sOuter=64, sInner=128
            //   config=7: Dk=192, Dv=128, sOuter=32, sInner=256
            tilingKeyInfo_.config = (sOuterFactor_ == fa_tiling_util::SOUTER_64) ? 6 : 7;
        }
    }
}

void FlashAttnTilingImpl::UpdateTilingKeyLayout()
{
    // q_out layout: 0=BSND, 1=BNSD, 2=TND, 3=BNSD_BSND
    if (faInfo_->outLayout == FaLayout::BSND && faInfo_->qLayout == FaLayout::BNSD) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BNSD_BSND;
    } else if (faInfo_->outLayout == FaLayout::BNSD) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BNSD;
    } else if (faInfo_->outLayout == FaLayout::TND) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_TND;
    } else {
        // outLayout == BSND (default)
        tilingKeyInfo_.inputLayout = InOutLayoutType_BSND;
    }
}

void FlashAttnTilingImpl::UpdateTilingKeyKvLayout()
{
    if (faInfo_->kvLayout == FaLayout::PA_BBND) {
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_BBH;
    } else if (faInfo_->kvLayout == FaLayout::PA_BNBD) {
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_BNBD;
    } else if (faInfo_->kvLayout == FaLayout::PA_NZ) {
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_NZ;
    }
}

void FlashAttnTilingImpl::UpdateTilingKeyTemplateId()
{
    // templateId: 0=ND模板, 1=DN模板
    // DN模板仅支持无attenMask且config=0/2/6（sOuter=64, D=64/128, QK
    // D=192/DV=128）场景，与kernel侧模板实例化范围保持一致
    dnFlag_ = (!tilingKeyInfo_.hasAttenMask) &&
              ((tilingKeyInfo_.config == 0) || (tilingKeyInfo_.config == 2) || (tilingKeyInfo_.config == 6));
    tilingKeyInfo_.templateId = dnFlag_ ? 1 : 0;
}

void FlashAttnTilingImpl::UpdateTilingKeyInfo()
{
    UpdateTilingKeyLayout();
    UpdateTilingKeyConfig();
    UpdateTilingKeyKvLayout();
    tilingKeyInfo_.hasAttenMask = (faInfo_->maskMode == static_cast<int64_t>(MaskMode::NO_MASK)) ? 0 : 1;
    UpdateTilingKeyTemplateId();
}

void FlashAttnTilingImpl::GenTilingKey()
{
    UpdateTilingKeyInfo();
    tilingKey_ = GET_TPL_TILING_KEY(tilingKeyInfo_.inputLayout, tilingKeyInfo_.kvLayoutType,
                                    tilingKeyInfo_.hasAttenMask, tilingKeyInfo_.templateId, tilingKeyInfo_.config);
}

void FlashAttnTilingImpl::CalcNumBlocks(uint32_t aicNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(faInfo_->platformInfo);
    auto aivNum = aicNum * platformInfo_.cvRatio;

    numBlocks_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
}

void FlashAttnTilingImpl::CalcWorkspaceSize()
{
    size_t sysWorkspaceSize = platformInfo_.defaultSysWorkspaceSize;
    uint32_t mSize = sOuterFactor_ * platformInfo_.cvRatio;
    uint32_t dSize = faInfo_->vHeadDim;
    uint32_t dVBasicBlock = 0;
    if (dSize <= arch35FA::DSIZE_64) {
        dVBasicBlock = arch35FA::DSIZE_64;
    } else if (dSize <= arch35FA::DSIZE_128) {
        dVBasicBlock = arch35FA::DSIZE_128;
    } else if (dSize <= arch35FA::DSIZE_256) {
        dVBasicBlock = arch35FA::DSIZE_256;
    } else if (dSize <= arch35FA::DSIZE_512) {
        dVBasicBlock = arch35FA::DSIZE_512;
    }

    workspaceSize_ = sysWorkspaceSize;

    // TODO，确认这段workspace分配逻辑
    int64_t bmm2Bytes = 0;
    int64_t vec2Bytes = 0;
    int64_t bmm2ResBlockSize = dVBasicBlock;
    if (dVBasicBlock > arch35FA::DSIZE_256) {
        bmm2ResBlockSize = arch35FA::DSIZE_512;
    }
    if ((!dnFlag_ && dSize > arch35FA::DSIZE_128) || (dnFlag_ && dSize > arch35FA::DSIZE_192)) {
        bmm2Bytes = mSize * bmm2ResBlockSize * sizeof(float);
        if (dVBasicBlock > arch35FA::DSIZE_256) {
            vec2Bytes = mSize * dVBasicBlock * sizeof(float);
        }
    }
    workspaceSize_ += (bmm2Bytes + vec2Bytes) * 3 * platformInfo_.coreNum; // 3: perload 2次 需要2+1

    if (faInfo_->pageAttentionFlag) {
        // 2 bmm, db, ensure alignment of each structure 64B, dcci cacheline needs
        workspaceSize_ += static_cast<uint64_t>(platformInfo_.coreNum) * 2 * 2 * 64;
    }

    if (flashDecodeFlag_) {
        uint32_t faTmpAttenGmSize = platformInfo_.coreNum * 2 * mSize * dSize; // 每个核最多有2次写到workspace
        uint32_t fatmpResLseGmSize = platformInfo_.coreNum * 2 * mSize * 8;
        workspaceSize_ += (faTmpAttenGmSize + 2 * fatmpResLseGmSize) * sizeof(float); // ResLse有2份，sum和max
        tilingData_.baseTiling.flashAttnWorkspaceParams.accumOutSize = faTmpAttenGmSize;
        tilingData_.baseTiling.flashAttnWorkspaceParams.logSumExpSize = fatmpResLseGmSize;
    }

}

void FlashAttnTilingImpl::FillTiling()
{
    ComputeTilingData();
    SetFATilingData();
    PrintAllTilingData();
}

void FlashAttnTilingImpl::ComputeTilingData()
{
    tilingData_.baseTiling.flashAttnAttenMaskParams.sparseMode = faInfo_->maskMode;
    tilingKeyInfo_.hasAttenMask = (faInfo_->maskMode == static_cast<int64_t>(MaskMode::NO_MASK)) ? 0 : 1;

    if (tilingKeyInfo_.hasAttenMask) {
        uint64_t maskBatch = 1;
        uint64_t maskDimNum = faInfo_->opParamInfo.attnMask.tensor->GetStorageShape().GetDimNum();
        uint64_t maskS1Size = 2048;
        uint64_t maskS2Size = 2048;
        if (maskDimNum != 2 || faInfo_->s1Size == 1) {
            maskBatch = faInfo_->opParamInfo.attnMask.tensor->GetStorageShape().GetDim(0);
        }
        maskS2Size = faInfo_->opParamInfo.attnMask.tensor->GetStorageShape().GetDim(maskDimNum - 1);
        maskS1Size = faInfo_->opParamInfo.attnMask.tensor->GetStorageShape().GetDim(maskDimNum - 2);
        tilingData_.baseTiling.flashAttnAttenMaskParams.attenMaskS1Size = maskS1Size;
        tilingData_.baseTiling.flashAttnAttenMaskParams.attenMaskS2Size = maskS2Size;
    }

    if (faInfo_->pageAttentionFlag) {
        if (faInfo_->kvLayout == FaLayout::PA_BBND) {
            tilingData_.baseTiling.flashAttnPageAttentionParams.paLayoutType = 1;
        } else if (faInfo_->kvLayout == FaLayout::PA_BNBD) {
            tilingData_.baseTiling.flashAttnPageAttentionParams.paLayoutType = 0;
        } else if (faInfo_->kvLayout == FaLayout::PA_NZ) {
            tilingData_.baseTiling.flashAttnPageAttentionParams.paLayoutType = 2;
        }
    }
}

void FlashAttnTilingImpl::SetFATilingData()
{
    tilingData_.baseTiling.flashAttnBaseParams.bSize = faInfo_->bSize;
    tilingData_.baseTiling.flashAttnBaseParams.t1Size = faInfo_->qTSize;
    tilingData_.baseTiling.flashAttnBaseParams.t2Size = faInfo_->kTSize;
    tilingData_.baseTiling.flashAttnBaseParams.n2Size = faInfo_->n2Size;
    tilingData_.baseTiling.flashAttnBaseParams.gSize = faInfo_->gSize;
    tilingData_.baseTiling.flashAttnBaseParams.s1Size = faInfo_->s1Size;
    tilingData_.baseTiling.flashAttnBaseParams.s2Size = faInfo_->s2Size;
    tilingData_.baseTiling.flashAttnBaseParams.dSize = faInfo_->qkHeadDim;
    tilingData_.baseTiling.flashAttnBaseParams.dSizeV = faInfo_->vHeadDim;
    tilingData_.baseTiling.flashAttnBaseParams.scaleValue = faInfo_->softmaxScale;
    tilingData_.baseTiling.flashAttnBaseParams.cuSeqLensQSize =
        faInfo_->qLayout == FaLayout::TND ? (faInfo_->bSize + 1) : 0;
    tilingData_.baseTiling.flashAttnBaseParams.cuSeqLensKVSize =
        faInfo_->kvLayout == FaLayout::TND ? (faInfo_->bSize + 1) : 0;
    tilingData_.baseTiling.flashAttnBaseParams.seqUsedQSize = seqUsedQFlag_ ? faInfo_->bSize : 0;
    tilingData_.baseTiling.flashAttnBaseParams.seqUsedKvSize = seqUsedKvFlag_ ? faInfo_->bSize : 0;
    tilingData_.baseTiling.flashAttnBaseParams.isSoftMaxLseEnable = faInfo_->softmaxLseFlag;
    tilingData_.baseTiling.flashAttnBaseParams.iscuSeqLengthsNull = !cuSeqLenQFlag_;
    tilingData_.baseTiling.flashAttnBaseParams.iscuSeqLengthsKVNull = !cuSeqLenKVFlag_;
    tilingData_.baseTiling.flashAttnBaseParams.coreNum = numBlocks_;

    tilingData_.baseTiling.flashAttnAttenMaskParams.winLefts = faInfo_->winLeft;
    tilingData_.baseTiling.flashAttnAttenMaskParams.winRights = faInfo_->winRight;
    if (faInfo_->winLeft == -1) {
        tilingData_.baseTiling.flashAttnAttenMaskParams.winLefts = MASK_MODE_INT_MAX;
    }
    if (faInfo_->winRight == -1) {
        tilingData_.baseTiling.flashAttnAttenMaskParams.winRights = MASK_MODE_INT_MAX;
    }
    tilingData_.baseTiling.flashAttnPageAttentionParams.blockSize = faInfo_->blockSize;
    uint32_t maxBlockNumPerBatch = 0;
    if (faInfo_->pageAttentionFlag) {
        maxBlockNumPerBatch = faInfo_->opParamInfo.blockTable.tensor->GetStorageShape().GetDim(1);
    }
    tilingData_.baseTiling.flashAttnPageAttentionParams.maxBlockNumPerBatch = maxBlockNumPerBatch;
    tilingData_.baseTiling.flashAttnBaseParams.needInitOutput = CheckNeedInitOutput();

    // kvcache非连续场景: 从tensor stride取偏移值, 连续时stride值即默认值, 不影响
    if (faInfo_->hasViewStride) {
        tilingData_.baseTiling.flashAttnBaseParams.keyBnStride = faInfo_->keyBnStride;
        tilingData_.baseTiling.flashAttnBaseParams.keyN2Stride = faInfo_->keyN2Stride;
        tilingData_.baseTiling.flashAttnBaseParams.valueBnStride = faInfo_->valueBnStride;
        tilingData_.baseTiling.flashAttnBaseParams.valueN2Stride = faInfo_->valueN2Stride;
    }
}

bool FlashAttnTilingImpl::CheckNeedInitOutput() const
{
    // varlen场景可能存在长度为0的batch, 对应行无计算, 必须清零
    if (seqUsedQFlag_ || seqUsedKvFlag_) {
        return true;
    }
    // 传入cuSeqLenKv时默认清零，可能会有空batch
    if (cuSeqLenKVFlag_) {
        return true;
    }
    // 非TND固定shape: 按mask模式判断是否存在整行被mask掉的query块
    if (faInfo_->maskMode == static_cast<int64_t>(MaskMode::NO_MASK)) {
        return false;
    }
    if (faInfo_->maskMode == static_cast<int64_t>(MaskMode::CAUSAL)) {
        // 下三角: s1>s2时首个query块的窗口全在s2负侧
        return faInfo_->s1Size > faInfo_->s2Size;
    }
    if (faInfo_->maskMode == static_cast<int64_t>(MaskMode::BAND)) {
        // winRight==-1表示右窗口无限大(被转为MASK_MODE_INT_MAX传给kernel), 每行必有有效S2块
        if (faInfo_->winRight == -1) {
            return false;
        }
        // 窗口右端=s2Size-s1Size+winRight, s1远大于s2时右端为负, 整行被mask
        return (faInfo_->s1Size - faInfo_->s2Size) > faInfo_->winRight;
    }
    return false;
}

ge::graphStatus FlashAttnTilingImpl::SetTilingData(FlashAttnTilingData &tilingData)
{
    FlashAttnTilingData *tiling = context_->GetTilingData<FlashAttnTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE(faInfo_->opName, "The tiling data is nullptr"), return ge::GRAPH_FAILED);
    *tiling = tilingData;
    return ge::GRAPH_SUCCESS;
}

void FlashAttnTilingImpl::PrintAllTilingData()
{
    FlashAttnNoQuantTilingArch35 &baseTiling = tilingData_.baseTiling;
    FlashAttnBaseParams &flashAttnBaseParams = baseTiling.flashAttnBaseParams;
    FlashAttnAttenMaskParams &flashAttnAttenMaskParams = baseTiling.flashAttnAttenMaskParams;
    FlashAttnPageAttentionParams &flashAttnPageAttentionParams = baseTiling.flashAttnPageAttentionParams;
    FlashAttnWorkspaceParams &flashAttnWorkspaceParams = baseTiling.flashAttnWorkspaceParams;
    FlashAttnS1OuterSplitCoreParams &flashAttnS1OuterSplitCoreParams = baseTiling.flashAttnS1OuterSplitCoreParams;

    OP_LOGD(faInfo_->opName, "bSize:%d", flashAttnBaseParams.bSize);
    OP_LOGD(faInfo_->opName, "t1Size:%d", flashAttnBaseParams.t1Size);
    OP_LOGD(faInfo_->opName, "t2Size:%d", flashAttnBaseParams.t2Size);
    OP_LOGD(faInfo_->opName, "n2Size:%d", flashAttnBaseParams.n2Size);
    OP_LOGD(faInfo_->opName, "gSize:%d", flashAttnBaseParams.gSize);
    OP_LOGD(faInfo_->opName, "s1Size:%d", flashAttnBaseParams.s1Size);
    OP_LOGD(faInfo_->opName, "s2Size:%d", flashAttnBaseParams.s2Size);
    OP_LOGD(faInfo_->opName, "dSize:%d", flashAttnBaseParams.dSize);
    OP_LOGD(faInfo_->opName, "dSizeV:%d", flashAttnBaseParams.dSizeV);
    OP_LOGD(faInfo_->opName, "dSizeRope:%d", flashAttnBaseParams.dSizeRope);
    OP_LOGD(faInfo_->opName, "cuSeqLensQSize:%d", flashAttnBaseParams.cuSeqLensQSize);
    OP_LOGD(faInfo_->opName, "cuSeqLensKVSize:%d", flashAttnBaseParams.cuSeqLensKVSize);
    OP_LOGD(faInfo_->opName, "seqUsedQSize:%d", flashAttnBaseParams.seqUsedQSize);
    OP_LOGD(faInfo_->opName, "seqUsedKvSize:%d", flashAttnBaseParams.seqUsedKvSize);
    OP_LOGD(faInfo_->opName, "scaleValue:%f", flashAttnBaseParams.scaleValue);
    OP_LOGD(faInfo_->opName, "iscuSeqLengthsNull:%d", flashAttnBaseParams.iscuSeqLengthsNull);
    OP_LOGD(faInfo_->opName, "iscuSeqLengthsKVNull:%d", flashAttnBaseParams.iscuSeqLengthsKVNull);
    OP_LOGD(faInfo_->opName, "isSoftMaxLseEnable:%d", flashAttnBaseParams.isSoftMaxLseEnable);
    OP_LOGD(faInfo_->opName, "coreNum:%d", flashAttnBaseParams.coreNum);

    OP_LOGD(faInfo_->opName, "maskMode:%d", flashAttnAttenMaskParams.sparseMode);
    OP_LOGD(faInfo_->opName, "winLefts:%d", flashAttnAttenMaskParams.winLefts);
    OP_LOGD(faInfo_->opName, "winRights:%d", flashAttnAttenMaskParams.winRights);
    OP_LOGD(faInfo_->opName, "attenMaskS1Size:%d", flashAttnAttenMaskParams.attenMaskS1Size);
    OP_LOGD(faInfo_->opName, "attenMaskS2Size:%d", flashAttnAttenMaskParams.attenMaskS2Size);

    OP_LOGD(faInfo_->opName, "paLayoutType:%d", flashAttnPageAttentionParams.paLayoutType);
    OP_LOGD(faInfo_->opName, "blockSize:%d", flashAttnPageAttentionParams.blockSize);
    OP_LOGD(faInfo_->opName, "maxBlockNumPerBatch:%d", flashAttnPageAttentionParams.maxBlockNumPerBatch);

    OP_LOGD(faInfo_->opName, "accumOutSize:%d", flashAttnWorkspaceParams.accumOutSize);
    OP_LOGD(faInfo_->opName, "logSumExpSize:%d", flashAttnWorkspaceParams.logSumExpSize);

    OP_LOGD(faInfo_->opName, "totalSize:%d", flashAttnS1OuterSplitCoreParams.totalSize);

    int64_t cap = context_->GetRawTilingData()->GetCapacity();
    OP_LOGD(faInfo_->opName, "Tiling Data context_ GetCapacity: %lu.", cap);
}

} // namespace flash_attn

using flash_attn::FlashAttnTilingImpl;

// 值越小表示优先级越高
REGISTER_TILING_TEMPLATE_FIA(FlashAttn, FlashAttnTilingImpl,
                             std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_3510)}), 1);
} // namespace optiling
