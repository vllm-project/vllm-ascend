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
 * \file fia_tiling_empty_tensor.cpp
 * \brief
 */

#include "fia_tiling_empty_tensor.h"
#include <vector>
#include "log/log.h"
#include "../fia_tiling_templates_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
constexpr uint64_t FIA_TILINGKEYOFFSET = uint64_t(100000000000000000UL);          
constexpr uint64_t FIA_PERF_MODE_TILINGKEYOFFSET = uint64_t(1000000000000000UL); 

constexpr uint64_t EMPTY_KV_TILING_KEY = 20;

void FiaTilingEmptyTensor::InitTilingInfo(TilingInfo *tilingInfo)
{
    fiaInfo_ = static_cast<FiaTilingInfo *>(tilingInfo);
}

ge::graphStatus FiaTilingEmptyTensor::GetPlatformInfo()
{
    OP_CHECK_IF(fiaInfo_->platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(fiaInfo_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(fiaInfo_->platformInfo);
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();

    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0,
        OPS_REPORT_VECTOR_INNER_ERR(fiaInfo_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool FiaTilingEmptyTensor::IsCapable()
{
    if (fiaInfo_ == nullptr) {
        return false;
    }
    if (fiaInfo_->emptyTensorFlag) {
        return true;
    }
    return false;
}

void FiaTilingEmptyTensor::GenTilingKey()
{
    uint64_t baseOffset = FIA_TILINGKEYOFFSET;
    tilingKey_ = baseOffset + EMPTY_KV_TILING_KEY;

    OP_LOGI(fiaInfo_->opName, "FIA tiling Key is: %lu.", tilingKey_);
}

void FiaTilingEmptyTensor::InitParams()
{
    usedCoreNum_ = aicNum_;
}

void FiaTilingEmptyTensor::FillTiling()
{
    uint64_t totalOutputSize = 0;
    uint64_t singleCoreSize = 0;
    uint64_t totalLseSize = 0;
    uint64_t singleCoreLseSize = 0;
    uint64_t tSize = static_cast<uint64_t>(fiaInfo_->bSize) * static_cast<uint64_t>(fiaInfo_->s1Size);
    if (fiaInfo_->outLayout == FiaLayout::TND || fiaInfo_->outLayout == FiaLayout::NTD) {
        tSize = fiaInfo_->qTSize;
    }
    totalOutputSize = tSize * fiaInfo_->n1Size * fiaInfo_->vHeadDim;
    if (totalOutputSize > fiaInfo_->totalOutputSize) {
        totalOutputSize = fiaInfo_->totalOutputSize;
    }
    if (fiaInfo_->isOutQuantEnable) {
        totalOutputSize = totalOutputSize / 2UL;
    }
    singleCoreSize = (totalOutputSize + (2UL * usedCoreNum_) - 1UL) / (2UL * usedCoreNum_);
    if (fiaInfo_->softmaxLseFlag) {
        totalLseSize = tSize * fiaInfo_->n1Size;
        if (totalLseSize > fiaInfo_->totalLseSize) {
            totalLseSize = fiaInfo_->totalLseSize;
        }
        singleCoreLseSize = (totalLseSize + (2UL * usedCoreNum_) - 1UL) / (2UL * usedCoreNum_);
    }
    
    tilingData_.set_softmaxLseFlag(fiaInfo_->softmaxLseFlag ? 1 : 0);
    tilingData_.set_headDim(fiaInfo_->vHeadDim);
    tilingData_.set_totalOutputSize(totalOutputSize);
    tilingData_.set_singleCoreSize(singleCoreSize);
    tilingData_.set_totalLseSize(totalLseSize);
    tilingData_.set_singleCoreLseSize(singleCoreLseSize);
    tilingData_.set_usedCoreNum(usedCoreNum_);
}

void FiaTilingEmptyTensor::CalcWorkspaceSize()
{
    workspaceSize_ = 16UL * 1024UL * 1024UL; // 16 * 1024 * 1024:min size required by workspace
}

void FiaTilingEmptyTensor::CalcNumBlocks(uint32_t coreNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(fiaInfo_->platformInfo);
    auto aicNum = coreNum;
    auto aivNum = aicNum * (aivNum_ / aicNum_);

    numBlocks_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum); 
    OP_LOGI(fiaInfo_->opName, "FIA block dim: %u aiv Num: %u aic Num: %u.", numBlocks_, aivNum, aicNum);
}

ge::graphStatus FiaTilingEmptyTensor::DoOpTiling()
{
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    InitParams();

    FillTiling();
    CalcNumBlocks(usedCoreNum_);
    CalcWorkspaceSize();
    GenTilingKey();

    if ((SetNumBlocks(numBlocks_) != ge::GRAPH_SUCCESS) ||
        (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) ||
        (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// 值越小表示优先级越高. 对于FIA, 使用3位数表示优先级, 优先级编码含义为:
// 1. 百位代表非量化、伪量化、全量化等场景, 即: 0xx-非量化，1xx-伪量化, 2xx-全量化
// 2. 十位表示gqa、mla、泛化，即: x0x-mla, x1x-gpa, x2x-泛化
// 3. 个位代表特化模板到泛化模板的优先级排序
REGISTER_TILING_TEMPLATE_FIA(FusedInferAttentionScore, FiaTilingEmptyTensor,
    std::vector<int32_t>({static_cast<int32_t>(NpuArch::DAV_2201)}), 0);
} // namespace optiling
