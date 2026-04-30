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
 * \file apply_top_k_top_p_custom_tiling.cpp
 * \brief
 */

#include <iostream>
#include <map>
#include "error_log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include "apply_top_k_top_p_custom_tiling.h"

namespace {
    constexpr uint32_t SYS_RESERVED_UB = uint32_t(16 * 1024);
    constexpr uint32_t SELECT_RESERVED_UB = uint32_t(8 * 1024);
    constexpr uint32_t DIM_ONE = 1;
    constexpr uint32_t DIM_TWO = 2;
    constexpr int32_t SORTED_VALUE_INPUT_INDEX = 0;
    constexpr int32_t SORTED_INDICES_INPUT_INDEX = 1;
    constexpr int32_t P_INPUT_INDEX = 2;
    constexpr int32_t K_INPUT_INDEX = 3;
    constexpr uint32_t DIM_INDEX0 = 0;
    constexpr uint32_t FLOAT_BYTES = 4;
    static std::map<ge::DataType, uint32_t> DTYPE_MAP = {{ge::DT_BF16, 2}, {ge::DT_FLOAT16, 1}, {ge::DT_FLOAT, 0}};
    static std::map<ge::DataType, uint32_t> DATATYPE_LEN_MAP = {
        {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}, {ge::DT_FLOAT, 4}};
    const static uint32_t SYS_WORKSPACESIZE = uint32_t(16 * 1024 * 1024);

    constexpr uint32_t DATA_PER_BLOCK_B32 = 8;
    constexpr uint32_t BYTES_B32 = 4;
    constexpr uint32_t BLOCK_BYTES = 32;
    constexpr uint32_t K_VALUE_MAX = 1024;
    constexpr uint32_t ONLY_TOP_P_KEY = 2;
    constexpr uint32_t ONLY_TOP_K_KEY = 1;
    constexpr uint32_t BATCH_MODE = 1;
} // namespace

namespace optiling {
class ApplyTopKTopPCustomTiling {
public:
    explicit ApplyTopKTopPCustomTiling(gert::TilingContext* context) : tilingcontext(context){};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
private:
    ApplyTopKTopPCustomTilingData tilingData;
    gert::TilingContext* tilingcontext = nullptr;
    ge::graphStatus CheckShape();
    void SetTilingKey();
    void GetUsedCore();
    void CalDataPerCore();
    void FillTilingData();
    void PrintTilingData();
    template <typename T1>
    inline auto CeilAlign(T1 a, T1 b) const -> T1
    {
        return b == 0 ? a : (a + b - 1) / b * b;
    }
    template <typename T1>
    inline auto FloorAlign(T1 a, T1 b) const -> T1
    {
        return b == 0 ? a : a / b * b;
    }

    const char *opName_ = nullptr;
    uint32_t coreNum_ = 0;
    uint32_t calUbSize_ = 0;
    uint32_t batchSize_ = 0;
    uint32_t vocabSize_ = 0;
    uint32_t tilingKey_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t batchPerCore_ = 1;
    uint32_t tailBatch_ = 0;
    uint32_t dataNumInit_ = 0;
    uint32_t dataNumInitAligned_ = 0;
    uint32_t ubFactorElement_ = 0;
    uint32_t ubFactorElementAligned_ = 0;
    uint32_t tailUbFactorElement_ = 0;
    uint32_t tailUbFactorElementAligned_ = 0;
    uint32_t iterateTimes_ = 0;
    uint32_t onlyTopK_ = 0;
    uint32_t onlyTopP_ = 0;
    uint64_t platformUbSize_ = 0;
};

ge::graphStatus ApplyTopKTopPCustomTiling::CheckShape() {
    auto sortedValueShapePtr = tilingcontext->GetInputShape(SORTED_VALUE_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingcontext, sortedValueShapePtr);
    auto sortedValueShape = sortedValueShapePtr->GetStorageShape();
    if (sortedValueShape.GetDimNum() != DIM_TWO) {
        OP_LOGE(opName_, "the dimNum of sorted_value should be 2, but got %u.", sortedValueShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    auto sortedIndicesShapePtr = tilingcontext->GetInputShape(SORTED_INDICES_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingcontext, sortedIndicesShapePtr);
    auto sortedIndicesShape = sortedIndicesShapePtr->GetStorageShape();
    if (sortedIndicesShape.GetDimNum() != DIM_TWO) {
        OP_LOGE(opName_, "the dimNum of sorted_indices should be 2, but got %u.", sortedIndicesShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    batchSize_ = sortedValueShape.GetDim(DIM_INDEX0);
    vocabSize_ = sortedValueShape.GetDim(DIM_ONE);
    if (sortedIndicesShape.GetDim(DIM_INDEX0) != batchSize_ || sortedIndicesShape.GetDim(DIM_ONE) != vocabSize_) {
        OP_LOGE(opName_, "the shape of sorted_indices should be equal to sorted_value.");
        return ge::GRAPH_FAILED;
    }

    auto pShapePtr = tilingcontext->GetOptionalInputShape(P_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingcontext, pShapePtr);
    auto pShape = pShapePtr->GetStorageShape();
    auto pDimNum = pShape.GetDimNum();
    if (pDimNum != DIM_ONE && pDimNum != 0) {
        OP_LOGE(opName_, "the dimNum of p should be 1 or 0, but got %u.", pDimNum);
        return ge::GRAPH_FAILED;
    }
    if (pDimNum != 0 && batchSize_ != pShape.GetDim(DIM_INDEX0)) {
        OP_LOGE(opName_, "p.shape[0] should be equal to logits.shape[0].");
        return ge::GRAPH_FAILED;
    }

    auto kShapePtr = tilingcontext->GetOptionalInputShape(K_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(tilingcontext, kShapePtr);
    auto kShape = kShapePtr->GetStorageShape();
    auto kDimNum = kShape.GetDimNum();
    if (kDimNum != DIM_ONE && kDimNum != 0) {
        OP_LOGE(opName_, "the dimNum of k should be 1 or 0, but got %u.", kShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    if (kDimNum != 0 && batchSize_ != kShape.GetDim(DIM_INDEX0)) {
        OP_LOGE(opName_, "k.shape[0] should be equal to logits.shape[0].");
        return ge::GRAPH_FAILED;
    }
    if (kDimNum == 0 && pDimNum == 0) {
        OP_LOGE(opName_, "the dimNum of q and k should be 0 at the same time.");
        return ge::GRAPH_FAILED;
    }
    onlyTopK_ = (kDimNum != 0 && pDimNum == 0) ? ONLY_TOP_K_KEY : 0;
    onlyTopP_ = (pDimNum != 0 && kDimNum == 0) ? ONLY_TOP_P_KEY : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ApplyTopKTopPCustomTiling::Init() {
    opName_ = tilingcontext->GetNodeName();
    OP_LOGD(opName_, "TilingForApplyTopKTopPCustom init.");
    auto platformInfo = platform_ascendc::PlatformAscendC(tilingcontext->GetPlatformInfo());
    coreNum_ = platformInfo.GetCoreNumAiv();
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformUbSize_);
    OP_LOGD(opName_, "platformUbSize: %lu.", platformUbSize_);
    uint32_t avaliableUb = static_cast<uint32_t>(platformUbSize_) - SYS_RESERVED_UB - SELECT_RESERVED_UB;
    calUbSize_ = FloorAlign(avaliableUb, BLOCK_BYTES);
    if (CheckShape() == ge::GRAPH_FAILED) {
        OP_LOGE(opName_, "check shape failed.");
        return ge::GRAPH_FAILED;
    }
    uint32_t tempValue = 1;
    while (tempValue < vocabSize_) {
        tempValue <<= 1;
        iterateTimes_++;
    } // ceil(log2(vocabSize_))
    return ge::GRAPH_SUCCESS;
}

void ApplyTopKTopPCustomTiling::SetTilingKey() {
    tilingKey_ += onlyTopK_;
    tilingKey_ += onlyTopP_;
    tilingcontext->SetTilingKey(tilingKey_);
    if (tilingKey_ == ONLY_TOP_P_KEY){
        tilingcontext->SetScheduleMode(BATCH_MODE);
    }
}

void ApplyTopKTopPCustomTiling::GetUsedCore()
{
    if (coreNum_ > 0) {
        batchPerCore_ = coreNum_ == uint32_t(0) ? batchSize_ : batchSize_ / coreNum_;
        tailBatch_ = batchSize_ % coreNum_;
        usedCoreNum_ = coreNum_;
    }
}

void ApplyTopKTopPCustomTiling::CalDataPerCore()
{
    uint32_t inputDataTypeByte = DATATYPE_LEN_MAP[tilingcontext->GetInputDesc(SORTED_VALUE_INPUT_INDEX)->GetDataType()];
    uint32_t dataPerBlock = BLOCK_BYTES / inputDataTypeByte;
    dataNumInit_ = vocabSize_ < K_VALUE_MAX ? vocabSize_ : K_VALUE_MAX;
    dataNumInitAligned_ = vocabSize_ < K_VALUE_MAX ? vocabSize_ : K_VALUE_MAX;
    ubFactorElement_ = vocabSize_ < K_VALUE_MAX ? vocabSize_ : K_VALUE_MAX;
    ubFactorElementAligned_ = CeilAlign(ubFactorElement_, dataPerBlock);
    tailUbFactorElement_ = vocabSize_ % ubFactorElement_;
    tailUbFactorElement_ = tailUbFactorElement_ == uint32_t(0) ? ubFactorElement_ : tailUbFactorElement_;
    tailUbFactorElementAligned_ = CeilAlign(tailUbFactorElement_, dataPerBlock);

    uint32_t sortedValueBytes = ubFactorElementAligned_ * inputDataTypeByte + K_VALUE_MAX  * inputDataTypeByte;
    uint32_t sortedIndicesBytes = ubFactorElementAligned_ * BYTES_B32 + K_VALUE_MAX  * BYTES_B32;
    uint32_t pBytes = dataPerBlock * inputDataTypeByte;
    uint32_t kBytes = DATA_PER_BLOCK_B32 * BYTES_B32;
    uint32_t outTensorBytes = ubFactorElementAligned_ * inputDataTypeByte;

    calUbSize_ = calUbSize_ - sortedValueBytes - sortedIndicesBytes - pBytes - kBytes - outTensorBytes;
    if (onlyTopP_ > 0) {
        calUbSize_ =  static_cast<uint32_t>(platformUbSize_);
    }
}

void ApplyTopKTopPCustomTiling::FillTilingData()
{
    tilingData.set_batchSize(batchSize_);
    tilingData.set_vocabSize(vocabSize_);
    tilingData.set_batchPerCore(batchPerCore_);
    tilingData.set_tailBatch(tailBatch_);
    tilingData.set_blockNum(usedCoreNum_);
    tilingData.set_dataNumInit(dataNumInit_);
    tilingData.set_dataNumInitAligned(dataNumInitAligned_);
    tilingData.set_ubFactorElement(ubFactorElement_);
    tilingData.set_ubFactorElementAligned(ubFactorElementAligned_);
    tilingData.set_tailUbFactorElement(tailUbFactorElement_);
    tilingData.set_tailUbFactorElementAligned(tailUbFactorElementAligned_);
    tilingData.set_calUbSize(calUbSize_);
    tilingData.set_iterateTimes(iterateTimes_);
}

void ApplyTopKTopPCustomTiling::PrintTilingData()
{
    OP_LOGD(opName_, "batchSize: %u.", tilingData.get_batchSize());
    OP_LOGD(opName_, "vocabSize: %u.", tilingData.get_vocabSize());
    OP_LOGD(opName_, "batchPerCore: %u.", tilingData.get_batchPerCore());
    OP_LOGD(opName_, "tailBatch: %u.", tilingData.get_tailBatch());
    OP_LOGD(opName_, "usedCoreNum: %u.", tilingData.get_blockNum());
    OP_LOGD(opName_, "dataNumInit_: %u.", tilingData.get_dataNumInit());
    OP_LOGD(opName_, "dataNumInitAligned_: %u.", tilingData.get_dataNumInitAligned());
    OP_LOGD(opName_, "ubFactorElement: %u.", tilingData.get_ubFactorElement());
    OP_LOGD(opName_, "ubFactorElementAligned: %u.", tilingData.get_ubFactorElementAligned());
    OP_LOGD(opName_, "tailUbFactorElement: %u.", tilingData.get_tailUbFactorElement());
    OP_LOGD(opName_, "tailUbFactorElementAligned: %u.", tilingData.get_tailUbFactorElementAligned());
    OP_LOGD(opName_, "calUbSize: %u.", tilingData.get_calUbSize());
    OP_LOGD(opName_, "iterateTimes: %u.", tilingData.get_iterateTimes());
}

ge::graphStatus ApplyTopKTopPCustomTiling::RunKernelTiling()
{
    OP_LOGD(opName_, "TilingForApplyTopKTopPCustom start.");

    SetTilingKey();
    GetUsedCore();
    CalDataPerCore();
    FillTilingData();
    PrintTilingData();

    OP_LOGD(opName_, "tilingKey: %u.", tilingKey_);
    uint32_t syncWorkspaceSize = SYS_WORKSPACESIZE;
    size_t* currentWorkspace = tilingcontext->GetWorkspaceSizes(1);
    currentWorkspace[0] = onlyTopP_ > 0 ? syncWorkspaceSize + batchSize_ * vocabSize_ * FLOAT_BYTES : syncWorkspaceSize;

    tilingData.SaveToBuffer(tilingcontext->GetRawTilingData()->GetData(),
                            tilingcontext->GetRawTilingData()->GetCapacity());
    tilingcontext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    tilingcontext->SetBlockDim(usedCoreNum_);

    OP_LOGD(opName_, "TilingForApplyTopKTopPCustom end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForApplyTopKTopPCustom(gert::TilingContext* context)
{
    ApplyTopKTopPCustomTiling tilingObject(context);
    auto ret = tilingObject.Init();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "tiling Init failed.");
        return ge::GRAPH_FAILED;
    }
    ret = tilingObject.RunKernelTiling();
    OP_LOGD(context->GetNodeName(), "TilingForApplyTopKTopPCustom end.");
    return ret;
}

static ge::graphStatus TilingPrepareForApplyTopKTopPCustom(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForApplyTopKTopPCustom start");
    auto compileInfo = context->GetCompiledInfo<TilingForApplyTopKTopPCustomCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(compileInfo->ubSizePlatForm <= 0,
                OP_LOGE(context->GetNodeName(), "Failed to get ub size"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "ub_size_platform is %lu", compileInfo->ubSizePlatForm);
    uint64_t totalUbSize = 0;
    platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
    OP_LOGD(context->GetNodeName(), "total ub size is %lu", totalUbSize);
    OP_LOGD(context->GetNodeName(), "TilingPrepareForApplyTopKTopPCustom end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ApplyTopKTopPCustom)
    .Tiling(TilingForApplyTopKTopPCustom)
    .TilingParse<TilingForApplyTopKTopPCustomCompileInfo>(TilingPrepareForApplyTopKTopPCustom);
} // namespace optiling