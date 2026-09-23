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
 * \file flash_mla_with_kvcache_tiling.cpp
 * \brief FlashMlaWithKvcache arch35 tiling implementation.
 *        Parameter validation is performed by the checker; task partitioning
 *        is supplied by FlashMlaWithKvcacheMetadata.
 */

#include "flash_mla_with_kvcache_tiling.h"
#include "../flash_mla_with_kvcache_tiling.h"
#include <cstring>
#include <vector>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "../flash_mla_with_kvcache_tiling_utils.h"
#include "../../op_kernel/arch35/flash_mla_with_kvcache_template_tiling_key.h"
#include "../../../a5_mla_common/op_host/fia_tiling_templates_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
namespace flash_mla_with_kvcache {

constexpr uint64_t PRE_LOAD_NUM_MLA_ARCH35 = 2;

void FlashMlaWithKvcacheTilingImpl::InitTilingInfo(TilingInfo *tilingInfo)
{
    faInfo_ = static_cast<FlashMlaWithKvcacheTilingInfo *>(tilingInfo);
}

bool FlashMlaWithKvcacheTilingImpl::IsCapable()
{
    return true;
}

void FlashMlaWithKvcacheTilingImpl::CalcScheduleMode()
{
    scheduleMode_ = ScheduleMode::BATCH_MODE;
    OP_LOGI(faInfo_->opName, "FlashMlaWithKvcache schedule mode: %u.", static_cast<uint32_t>(scheduleMode_));
}

ge::graphStatus FlashMlaWithKvcacheTilingImpl::DoOpTiling()
{
    OP_CHECK_IF(SetPlatMemoryInfo() != ge::GRAPH_SUCCESS, OP_LOGE(faInfo_->opName, "Set plat memory info fail."),
                return ge::GRAPH_FAILED);

    // Clear unused fields and padding before filling the kernel tiling data.
    memset(&tilingData_, 0, sizeof(tilingData_));

    InitImplParam();
    SplitPolicy();
    FillTiling();
    CalcScheduleMode();
    CalcWorkspaceSize();
    PrintAllTilingData();
    GenTilingKey();

    if ((SetNumBlocks(numBlocks_) != ge::GRAPH_SUCCESS) || (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) || (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS) ||
        (SetScheduleMode(scheduleMode_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheTilingImpl::SetPlatMemoryInfo()
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
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo_.l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo_.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo_.l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, platformInfo_.l2Size);

    platformInfo_.defaultSysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGI(faInfo_->opName, "AIV:%u AIC:%u L0A:%lu L0B:%lu L0C:%lu UB:%lu L1:%lu L2:%lu", platformInfo_.aivNum,
            platformInfo_.aicNum, platformInfo_.l0aSize, platformInfo_.l0bSize, platformInfo_.l0cSize,
            platformInfo_.ubSize, platformInfo_.l1Size, platformInfo_.l2Size);

    return ge::GRAPH_SUCCESS;
}

void FlashMlaWithKvcacheTilingImpl::InitImplParam()
{
    // Tiling only reads tensor shapes; sequence values are consumed on device.
    const gert::Tensor *cuSeqLenQ = faInfo_->opParamInfo.cuSeqlensQ.tensor;
    const gert::Tensor *seqUsedQ = faInfo_->opParamInfo.sequsedQ.tensor;
    const gert::Tensor *cacheSeqlens = faInfo_->opParamInfo.cacheSeqlens.tensor;
    cuSeqLenQFlag_ = cuSeqLenQ != nullptr && cuSeqLenQ->GetShapeSize() > 0;
    seqUsedQFlag_ = seqUsedQ != nullptr && seqUsedQ->GetShapeSize() > 0;
    cacheSeqlensFlag_ = cacheSeqlens != nullptr && cacheSeqlens->GetShapeSize() > 0;
}

void FlashMlaWithKvcacheTilingImpl::SplitPolicy()
{
    // MLA uses M96/S2=112; each AIV handles 48 rows. The metadata producer
    // owns task partitioning, and the kernel executes its FA/FD sections.
    sOuterFactor_ = faInfo_->isC8 ? 32 : 48;
    sInnerFactor_ = faInfo_->isC8 ? 128 : 112;
    CalcNumBlocks(platformInfo_.aicNum);
    flashDecodeFlag_ = true;
}

void FlashMlaWithKvcacheTilingImpl::UpdateTilingKeyConfig()
{
    // MLA D512 唯一路径: S1=64, S2=128, D=576, DV=512 -> Config index 0
    tilingKeyInfo_.config = Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512;
}

void FlashMlaWithKvcacheTilingImpl::UpdateTilingKeyLayout()
{
    // q/out 布局 -> InOutLayoutType（取值以 op_kernel/utils/flash_mla_with_kvcache_common_def.h
    // 为准：BSND=0 / BNSD=1 / TND=2 / TND_NTD=4）
    if (faInfo_->qLayout == FlashMlaWithKvcacheLayout::TND) {
        if (faInfo_->outLayout == FlashMlaWithKvcacheLayout::NTD) {
            tilingKeyInfo_.inputLayout = InOutLayoutType_TND_NTD;
        } else {
            tilingKeyInfo_.inputLayout = InOutLayoutType_TND;
        }
    } else if (faInfo_->qLayout == FlashMlaWithKvcacheLayout::BSND) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BSND;
    } else if (faInfo_->qLayout == FlashMlaWithKvcacheLayout::BNSD) {
        tilingKeyInfo_.inputLayout = InOutLayoutType_BNSD;
    }
}

void FlashMlaWithKvcacheTilingImpl::UpdateTilingKeyKvLayout()
{
    if (faInfo_->kvLayout == FlashMlaWithKvcacheLayout::PA_BBND) {
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_BBND;
    } else if (faInfo_->kvLayout == FlashMlaWithKvcacheLayout::PA_BNBD) {
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_BNBD;
    } else if (faInfo_->kvLayout == FlashMlaWithKvcacheLayout::PA_NZ) {
        tilingKeyInfo_.kvLayoutType = KvLayoutType_PA_NZ;
    }
}

void FlashMlaWithKvcacheTilingImpl::UpdateTilingKeyInfo()
{
    UpdateTilingKeyLayout();
    UpdateTilingKeyConfig();
    UpdateTilingKeyKvLayout();
    tilingKeyInfo_.hasAttenMask = faInfo_->maskMode != static_cast<int64_t>(MaskMode::NO_MASK);
}

void FlashMlaWithKvcacheTilingImpl::GenTilingKey()
{
    UpdateTilingKeyInfo();
    // 实参序 = ASCENDC_TPL_ARGS_DECL 声明序：InOutLayoutType, KvLayoutType, HasAttenMask, Config
    tilingKey_ = GET_TPL_TILING_KEY(tilingKeyInfo_.inputLayout, tilingKeyInfo_.kvLayoutType,
                                    tilingKeyInfo_.hasAttenMask, tilingKeyInfo_.config);

    OP_LOGI(faInfo_->opName, "The tilingkey is %llu.", tilingKey_);
    OP_LOGI(faInfo_->opName,
            "The tilingkey param is inOutLayoutType: %llu, kvLayoutType: %llu, hasAttenMask: %u, config: %llu.",
            tilingKeyInfo_.inputLayout, tilingKeyInfo_.kvLayoutType, tilingKeyInfo_.hasAttenMask,
            tilingKeyInfo_.config);
}

void FlashMlaWithKvcacheTilingImpl::CalcNumBlocks(uint32_t aicNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(faInfo_->platformInfo);
    auto aivNum = aicNum * platformInfo_.cvRatio;

    numBlocks_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_LOGI(faInfo_->opName, "FlashMlaWithKvcache block dim: %u aiv Num: %u aic Num: %u.", numBlocks_, aivNum, aicNum);
}

void FlashMlaWithKvcacheTilingImpl::CalcWorkspaceSize()
{
    // Each AIC owns two AIV row tiles and two partial-result slots.
    const uint64_t mSize = static_cast<uint64_t>(sOuterFactor_) * platformInfo_.cvRatio;
    const uint64_t dSize = faInfo_->vHeadDim;
    constexpr uint64_t lseSize = 8;

    workspaceSize_ = platformInfo_.defaultSysWorkspaceSize;

    if (flashDecodeFlag_) {
        const uint64_t faTmpAttenGmSize =
            static_cast<uint64_t>(numBlocks_) * PRE_LOAD_NUM_MLA_ARCH35 * mSize * dSize; // 每个核最多有2次写到workspace
        const uint64_t faTmpResLseGmSize =
            static_cast<uint64_t>(numBlocks_) * PRE_LOAD_NUM_MLA_ARCH35 * mSize * lseSize;
        workspaceSize_ += (faTmpAttenGmSize + 2 * faTmpResLseGmSize) * sizeof(float); // ResLse有2份，sum和max
        tilingData_.baseTiling.flashMlaWithKvcacheWorkspaceParams.accumOutSize =
            static_cast<uint32_t>(faTmpAttenGmSize);
        tilingData_.baseTiling.flashMlaWithKvcacheWorkspaceParams.logSumExpSize =
            static_cast<uint32_t>(faTmpResLseGmSize);
    }

    OP_LOGI(faInfo_->opName, "Workspaces: %lu", workspaceSize_);
}

void FlashMlaWithKvcacheTilingImpl::FillTiling()
{
    ComputeTilingData();
    SetFATilingData();
}

void FlashMlaWithKvcacheTilingImpl::ComputeTilingData()
{
    auto &maskParams = tilingData_.baseTiling.flashMlaWithKvcacheAttenMaskParams;
    maskParams.sparseMode = static_cast<uint8_t>(faInfo_->maskMode);
    maskParams.preTokens = static_cast<int32_t>(faInfo_->preTokens);
    maskParams.nextTokens = static_cast<int32_t>(faInfo_->nextTokens);
    tilingKeyInfo_.hasAttenMask = faInfo_->maskMode != static_cast<int64_t>(MaskMode::NO_MASK);
    if (tilingKeyInfo_.hasAttenMask) {
        // The checker requires one shared [2048, 2048] mask for all batches.
        const auto &maskShape = faInfo_->opParamInfo.attnMask.tensor->GetStorageShape();
        maskParams.attenMaskBatch = 1;
        maskParams.attenMaskS1Size = maskShape.GetDim(0);
        maskParams.attenMaskS2Size = maskShape.GetDim(1);
    }

    auto &pageParams = tilingData_.baseTiling.flashMlaWithKvcachePageAttentionParams;
    if (faInfo_->kvLayout == FlashMlaWithKvcacheLayout::PA_BBND) {
        pageParams.paLayoutType = 1;
    } else if (faInfo_->kvLayout == FlashMlaWithKvcacheLayout::PA_BNBD) {
        pageParams.paLayoutType = 0;
    } else if (faInfo_->kvLayout == FlashMlaWithKvcacheLayout::PA_NZ) {
        pageParams.paLayoutType = 2;
    }
}

void FlashMlaWithKvcacheTilingImpl::SetFATilingData()
{
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.bSize = faInfo_->bSize;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.t1Size = faInfo_->qTSize;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.t2Size = faInfo_->kTSize;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.n2Size = faInfo_->n2Size;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.gSize = faInfo_->gSize;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.s1Size = faInfo_->s1Size;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.s2Size = faInfo_->s2Size;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.dSize = faInfo_->qkHeadDim;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.dSizeV = faInfo_->vHeadDim;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.dSizeRope = faInfo_->ropeHeadDim;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.scaleValue = faInfo_->softmaxScale;
    // kernel ActualSeqLensParser<ACTLEN_T=uint32_t> 的 actualLenDims（INT32 buffer 元素数）
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.actualSeqLengthsQSize =
        cuSeqLenQFlag_ ? static_cast<uint32_t>(faInfo_->actualLenQDims) : 0;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.actualSeqLengthsKVSize =
        cacheSeqlensFlag_ ? static_cast<uint32_t>(faInfo_->actualLenKvDims) : 0;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.isKvContinuous = 0;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.isSoftMaxLseEnable = faInfo_->softmaxLseFlag;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.coreNum = numBlocks_;
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.outputLayout =
        static_cast<uint32_t>(faInfo_->kernelOutputLayout);
    // 单 token / 小 batch decode 场景：K/V 数据量远超 L2 缓存容量，关闭 L2 cache 避免无意义的缓存填充/驱逐开销
    // （阈值 128 = 2*mBaseSize，由性能实验验证：gSize*s1Size > 128 时可能存在跨 batch 复用，保持 L2 开启）
    tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.l2CacheOffFlag =
        (faInfo_->gSize * faInfo_->s1Size) <= 128 ? 1U : 0U;

    tilingData_.baseTiling.flashMlaWithKvcachePageAttentionParams.blockSize = faInfo_->blockSize;
    uint32_t maxBlockNumPerBatch = faInfo_->opParamInfo.blockTable.tensor->GetStorageShape().GetDim(1);
    tilingData_.baseTiling.flashMlaWithKvcachePageAttentionParams.maxBlockNumPerBatch = maxBlockNumPerBatch;

    tilingData_.baseTiling.flashMlaWithKvcacheSystemPrefixParams.isActualSharedPrefixLenNull = 1;
    tilingData_.baseTiling.flashMlaWithKvcacheSystemPrefixParams.prefixSeqInnerSize = 0;

    int64_t outSize = faInfo_->opParamInfo.attnOut.shape->GetStorageShape().GetShapeSize();
    int64_t lseSize = faInfo_->softmaxLseFlag ? faInfo_->opParamInfo.lseOut.shape->GetStorageShape().GetShapeSize() : 0;
    uint32_t singleCoreSize = (outSize + platformInfo_.aivNum - 1) / (platformInfo_.aivNum);
    tilingData_.baseTiling.flashMlaWithKvcacheEmptyTensorParams.singleCoreSize = singleCoreSize;
    tilingData_.baseTiling.flashMlaWithKvcacheEmptyTensorParams.totalOutputSize = outSize;
    tilingData_.baseTiling.flashMlaWithKvcacheEmptyTensorParams.totalSoftMaxLseOutputSize = lseSize;
    tilingData_.baseTiling.flashMlaWithKvcacheEmptyTensorParams.needInit = CheckNeedInitOutput() ? 1 : 0;

    // Key, value and rope share k_cache and its physical strides.
    if (faInfo_->hasViewStride) {
        tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.keyStrides.bnStride = faInfo_->keyBnStride;
        tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.keyStrides.n2Stride = faInfo_->keyN2Stride;
        tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.valueStrides.bnStride = faInfo_->valueBnStride;
        tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.valueStrides.n2Stride = faInfo_->valueN2Stride;
        tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.kRopeStrides.bnStride = faInfo_->isC8 ? faInfo_->kRopeBnStride : faInfo_->keyBnStride;
        tilingData_.baseTiling.flashMlaWithKvcacheBaseParams.kRopeStrides.n2Stride = faInfo_->isC8 ? faInfo_->kRopeN2Stride : faInfo_->keyN2Stride;
    }
}

bool FlashMlaWithKvcacheTilingImpl::CheckNeedInitOutput() const
{
    if (seqUsedQFlag_ || cacheSeqlensFlag_) {
        return true;
    }
    if (faInfo_->maskMode == static_cast<int64_t>(MaskMode::CAUSAL)) {
        return faInfo_->s1Size > faInfo_->s2Size;
    }
    return false;
}

ge::graphStatus FlashMlaWithKvcacheTilingImpl::SetTilingData(FlashMlaWithKvcacheTilingData &tilingData)
{
    FlashMlaWithKvcacheTilingData *tiling = context_->GetTilingData<FlashMlaWithKvcacheTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE(faInfo_->opName, "The tiling data is nullptr"), return ge::GRAPH_FAILED);
    *tiling = tilingData;
    return ge::GRAPH_SUCCESS;
}

void FlashMlaWithKvcacheTilingImpl::PrintAllTilingData()
{
    const auto &base = tilingData_.baseTiling.flashMlaWithKvcacheBaseParams;
    const auto &mask = tilingData_.baseTiling.flashMlaWithKvcacheAttenMaskParams;
    const auto &page = tilingData_.baseTiling.flashMlaWithKvcachePageAttentionParams;
    const auto &workspace = tilingData_.baseTiling.flashMlaWithKvcacheWorkspaceParams;
    const auto &output = tilingData_.baseTiling.flashMlaWithKvcacheEmptyTensorParams;
    OP_LOGD(faInfo_->opName, "B:%u T:%u N2:%u G:%u S1:%u S2:%u D:%u DV:%u Rope:%u", base.bSize, base.t1Size,
            base.n2Size, base.gSize, base.s1Size, base.s2Size, base.dSize, base.dSizeV, base.dSizeRope);
    OP_LOGD(faInfo_->opName, "QSeqDims:%u KVSeqDims:%u scale:%f LSE:%u cores:%u outputLayout:%u L2Off:%u",
            base.actualSeqLengthsQSize, base.actualSeqLengthsKVSize, base.scaleValue, base.isSoftMaxLseEnable,
            base.coreNum, base.outputLayout, base.l2CacheOffFlag);
    OP_LOGD(faInfo_->opName, "maskMode:%u maskS1:%u maskS2:%u preTokens:%d nextTokens:%d", mask.sparseMode,
            mask.attenMaskS1Size, mask.attenMaskS2Size, mask.preTokens, mask.nextTokens);
    OP_LOGD(faInfo_->opName, "paLayout:%u blockSize:%u maxBlocks:%u", page.paLayoutType, page.blockSize,
            page.maxBlockNumPerBatch);
    OP_LOGD(faInfo_->opName, "accumOutSize:%u logSumExpSize:%u needInit:%u", workspace.accumOutSize,
            workspace.logSumExpSize, output.needInit);
}

} // namespace flash_mla_with_kvcache

using flash_mla_with_kvcache::FlashMlaWithKvcacheTilingImpl;

// 值越小表示优先级越高
REGISTER_TILING_TEMPLATE_FIA(FlashMlaWithKvcache, FlashMlaWithKvcacheTilingImpl,
                             std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_3510)}), 1);
} // namespace optiling
