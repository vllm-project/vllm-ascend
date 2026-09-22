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
 * \file compressor_tiling.cpp
 * \file compressor_tiling.cpp
 * \brief
 */

#include <functional>
#include <algorithm>
#include <algorithm>
#include <unordered_map>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "register/op_def_registry.h"
#include "compressor_v2_tiling_arch22.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
namespace {

ge::graphStatus CompressorV2Tiling::ConvertRequiredParams(gert::TilingContext &context,
                                                          CompressorV2Context &compressorContext)
{
    compressorContext.x.desc = context.GetRequiredInputDesc(TOKEN_X_INPUT_INDEX);
    compressorContext.x.shape = context.GetRequiredInputShape(TOKEN_X_INPUT_INDEX);
    OP_CHECK_IF(compressorContext.x.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(compressorContext.opName, X_NAME, "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(compressorContext.x.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(compressorContext.opName, X_NAME, "desc is nullptr"),
                return ge::GRAPH_FAILED);
    compressorContext.wkv.desc = context.GetRequiredInputDesc(WEIGHT_KV_INPUT_INDEX);
    compressorContext.wkv.shape = context.GetRequiredInputShape(WEIGHT_KV_INPUT_INDEX);
    compressorContext.wgate.desc = context.GetRequiredInputDesc(WEIGHT_WGATE_INPUT_INDEX);
    compressorContext.wgate.shape = context.GetRequiredInputShape(WEIGHT_WGATE_INPUT_INDEX);
    compressorContext.stateCache.desc = context.GetRequiredInputDesc(STATE_CACHE_INPUT_INDEX);
    compressorContext.stateCache.shape = context.GetRequiredInputShape(STATE_CACHE_INPUT_INDEX);

    compressorContext.cmpKv.desc = context.GetOutputDesc(CMP_KV_OUTPUT_INDEX);
    compressorContext.cmpKv.shape = context.GetOutputShape(CMP_KV_OUTPUT_INDEX);

    compressorContext.dtype = compressorContext.x.desc->GetDataType();
    auto xDimNum = compressorContext.x.shape->GetStorageShape().GetDimNum();
    if (xDimNum == COMPRESSOR_DIM_NUM_3) {
        compressorContext.layout = LayoutType::LAYOUT_BSH;
    } else if (xDimNum == COMPRESSOR_DIM_NUM_2) {
        compressorContext.layout = LayoutType::LAYOUT_TH;
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(compressorContext.opName, X_NAME, std::to_string(xDimNum),
                                                 "x dimension should be 2 or 3");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void CompressorV2Tiling::ConvertOptionalParams(gert::TilingContext &context, CompressorV2Context &compressorContext)
{
    compressorContext.stateBlockTable.desc = context.GetOptionalInputDesc(STATE_BLOCK_TABLE_INPUT_INDEX);
    compressorContext.stateBlockTable.shape = context.GetOptionalInputShape(STATE_BLOCK_TABLE_INPUT_INDEX);
    compressorContext.cuSeqlens.desc = context.GetOptionalInputDesc(CU_SEQ_LEN_INPUT_INDEX);
    compressorContext.cuSeqlens.shape = context.GetOptionalInputShape(CU_SEQ_LEN_INPUT_INDEX);
    compressorContext.seqUsed.desc = context.GetOptionalInputDesc(SEQ_USED_INPUT_INDEX);
    compressorContext.seqUsed.shape = context.GetOptionalInputShape(SEQ_USED_INPUT_INDEX);
    compressorContext.startPos.desc = context.GetOptionalInputDesc(START_POS_INPUT_INDEX);
    compressorContext.startPos.shape = context.GetOptionalInputShape(START_POS_INPUT_INDEX);
}

ge::graphStatus CompressorV2Tiling::ConvertContext(gert::TilingContext &context, CompressorV2Context &compressorContext)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("CompressorV2", "opName", "got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }

    OP_LOGI("Getting Context");

    compressorContext.opName = context.GetNodeName();
    compressorContext.opType = context.GetNodeType();
    compressorContext.platformInfo = context.GetPlatformInfo();
    OP_CHECK_IF(ConvertRequiredParams(context, compressorContext) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    ConvertOptionalParams(context, compressorContext);

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.GetNodeName(), "attrs", "got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    compressorContext.cmpRatio = attrs->GetAttrPointer<int>(CMP_RATIO_ATTR_INDEX);
    compressorContext.stateCacheStrideDim0 = attrs->GetAttrPointer<int>(STATE_CACHE_STRIDE_DIM0_ATTR_INDEX);
    auto *stride = context.GetInputStride(STATE_CACHE_INPUT_INDEX);
    const auto &shape = compressorContext.stateCache.shape->GetStorageShape();
    if (stride != nullptr && stride->GetDimNum() == shape.GetDimNum()) {
        int64_t expectedStride = 1;
        for (int64_t axis = shape.GetDimNum() - 1; axis >= 1; --axis) {
            OP_CHECK_IF(stride->GetStride(axis) != expectedStride,
                        OP_LOGE(context.GetNodeName(), "state_cache axes 1 and 2 must be contiguous"),
                        return ge::GRAPH_FAILED);
            expectedStride *= shape.GetDim(axis);
        }
    }
    OP_CHECK_IF(
        context.GetWorkspaceSizes(1) == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.GetNodeName(), "workSpaceSize", "got from ge is nullptr"),
        return ge::GRAPH_FAILED);
    compressorContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::GetNpuInfo()
{
    OP_CHECK_IF(context_->platformInfo == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "platformInfo", "is nullptr"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();

    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);

    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(
        aicNum_ == 0 || aivNum_ == 0,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "aicNum/aivNum", "num of core obtained is 0"),
        return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::SetBaseInfo()
{
    if (context_->x.shape->GetStorageShape().GetDimNum() == COMPRESSOR_DIM_NUM_3) {
        baseParams_->batchSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0);
        baseParams_->seqSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1);
        baseParams_->hiddenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_2);
        baseParams_->tokenSize = baseParams_->batchSize * baseParams_->seqSize;
    } else {
        baseParams_->batchSize = context_->cuSeqlens.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0) - 1;
        baseParams_->tokenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0);
        baseParams_->hiddenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1);
    }

    baseParams_->headDim = context_->wkv.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0);
    baseParams_->cmpRatio = static_cast<uint32_t>(*context_->cmpRatio);
    baseParams_->csSize = baseParams_->seqSize - (baseParams_->seqSize % baseParams_->cmpRatio);
    baseParams_->stateCacheStrideDim0 = static_cast<uint64_t>(*context_->stateCacheStrideDim0);
    baseParams_->nSize = 2; // 2:每个核处理两个基本块后做全核同步
    baseParams_->usedCoreNum = aicNum_;
    OP_LOGI(context_->opName, "[TILING] bSize:%u  tSize:%u cmpRatio:%u", baseParams_->batchSize, baseParams_->tokenSize,
            baseParams_->cmpRatio);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::SetPageAttentionInfo()
{
    pageAttentionParams_->blockNum = context_->stateCache.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0);
    pageAttentionParams_->blockSize = context_->stateCache.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::SetWorkSpaceInfo()
{
    constexpr uint8_t coff = 1;
    workspaceParams_->dbWorkspaceRatio = 2;
    workspaceParams_->mm1KvResSize = innerSplitParams_->mBaseSize * baseParams_->headDim * coff;
    workspaceParams_->mm1ScoreResSize = innerSplitParams_->mBaseSize * baseParams_->headDim * coff;
    workspaceParams_->vec1ResSize = innerSplitParams_->mBaseSize * baseParams_->headDim * baseParams_->nSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::SetScenarioInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::SetTemplateId()
{
    if (context_->templateId == TemplateId::EMPTY_X) {
        return ge::GRAPH_SUCCESS;
    }
    // 设置高性能模板
    if (context_->layout == LayoutType::LAYOUT_BSH && baseParams_->seqSize <= 4 && baseParams_->tokenSize <= 128) {
        context_->templateId = TemplateId::FULL_LOAD;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::SetInnerSplitInfo()
{
    constexpr uint32_t coff = 1;
    if (context_->templateId == TemplateId::FULL_LOAD) {
        uint32_t kAlignNum = baseParams_->hiddenSize / 128;
        innerSplitParams_->mBaseSize = 128;              // 256:核间切分，M轴基本块大小
        innerSplitParams_->dBaseSize = 256 / (coff * 2); // nBase = dBase * coff * 2
        uint32_t dBaseNum = baseParams_->headDim / innerSplitParams_->dBaseSize;
        uint32_t mBaseNum = (baseParams_->tokenSize + innerSplitParams_->mBaseSize - 1) / innerSplitParams_->mBaseSize;
        baseParams_->coreGroupNum = baseParams_->usedCoreNum / dBaseNum;
        baseParams_->kBaseNum = 1;
        baseParams_->kBaseSize = baseParams_->hiddenSize;
        if ((dBaseNum * mBaseNum) < baseParams_->usedCoreNum) {
            baseParams_->kBaseNum = std::min(kAlignNum, baseParams_->usedCoreNum / dBaseNum);
            baseParams_->kBaseSize = kAlignNum / baseParams_->kBaseNum * 128;
        }
        for (uint32_t i = 0; i < baseParams_->usedCoreNum; i++) {
            baseParams_->splitCoreParam[i].nStart = (i % dBaseNum) * innerSplitParams_->dBaseSize;
            baseParams_->splitCoreParam[i].nEnd = baseParams_->splitCoreParam[i].nStart + innerSplitParams_->dBaseSize;
            if (baseParams_->kBaseNum > 1) {
                uint32_t kStartIdx = i / dBaseNum;
                uint32_t dealKSize = baseParams_->kBaseSize;
                if (kStartIdx < kAlignNum % baseParams_->kBaseNum) {
                    dealKSize += 128;
                    baseParams_->splitCoreParam[i].kStart = kStartIdx * dealKSize;
                } else if (kStartIdx < baseParams_->kBaseNum) {
                    baseParams_->splitCoreParam[i].kStart =
                        kStartIdx * baseParams_->kBaseSize + (kAlignNum % baseParams_->kBaseNum) * 128;
                } else {
                    dealKSize = 0;
                    baseParams_->splitCoreParam[i].kStart = 0;
                }
                baseParams_->splitCoreParam[i].kEnd = baseParams_->splitCoreParam[i].kStart + dealKSize;
                baseParams_->splitCoreParam[i].mStart = 0;
                baseParams_->splitCoreParam[i].mEnd = baseParams_->tokenSize;
                baseParams_->mLoopNum = 1;
            } else {
                baseParams_->splitCoreParam[i].kStart = 0;
                baseParams_->splitCoreParam[i].kEnd = baseParams_->hiddenSize;
                baseParams_->splitCoreParam[i].mStart = (i / dBaseNum) * innerSplitParams_->mBaseSize;
                baseParams_->splitCoreParam[i].mEnd =
                    baseParams_->splitCoreParam[i].mStart + innerSplitParams_->mBaseSize;
                baseParams_->mLoopNum = mBaseNum / baseParams_->coreGroupNum;
            }
        }
    } else {
        innerSplitParams_->mBaseSize = 256;
        innerSplitParams_->dBaseSize = 64;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CalcWorkSpace()
{
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4; // 4: fp32
    constexpr uint32_t V1_RES_ELEM_SIZE = 4;  // 4: fp32
    uint32_t maxGroupNum = aicNum_ / (baseParams_->headDim / innerSplitParams_->dBaseSize);
    workspaceSize_ = libapiSize_;
    workspaceSize_ +=
        workspaceParams_->mm1KvResSize * maxGroupNum * MM1_RES_ELEM_SIZE * workspaceParams_->dbWorkspaceRatio;
    workspaceSize_ +=
        workspaceParams_->mm1ScoreResSize * maxGroupNum * MM1_RES_ELEM_SIZE * workspaceParams_->dbWorkspaceRatio;
    workspaceSize_ +=
        workspaceParams_->vec1TailCacheSize * MM1_RES_ELEM_SIZE * workspaceParams_->dbWorkspaceRatio * 2; // 2 kv和score
    workspaceSize_ +=
        workspaceParams_->vec1ResSize * maxGroupNum * V1_RES_ELEM_SIZE * workspaceParams_->dbWorkspaceRatio;

    if (context_->workSpaces) {
        context_->workSpaces[0] = workspaceSize_;
    }

    OP_LOGI(context_->opName, "Tiling info: workspaceSize_ = %zu", workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckEmptyTensor() const
{
    if ((context_->layout == LayoutType::LAYOUT_BSH &&
         context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0) == 0) ||
        (context_->layout == LayoutType::LAYOUT_BSH &&
         context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1) == 0) ||
        (context_->layout == LayoutType::LAYOUT_TH &&
         context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0) == 0)) {
        context_->templateId = TemplateId::EMPTY_X;
    } else {
        if (context_->x.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->wkv.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->wgate.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->stateCache.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->stateBlockTable.shape->GetStorageShape().GetShapeSize() == 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->opName, "x", "0",
                                                  "Only input tensor x dim B or S or T supports to be 0");
            return ge::GRAPH_FAILED;
        }
        context_->templateId = TemplateId::NORMAL;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::RunBigKernelTiling(CompressorV2TilingData *tilingData)
{
    this->baseParams_ = &tilingData->baseParams;
    this->pageAttentionParams_ = &tilingData->pageAttentionParams;
    this->innerSplitParams_ = &tilingData->innerSplitParams;
    this->workspaceParams_ = &tilingData->workspaceParams;
    using StatusFunction = std::function<ge::graphStatus()>;
    std::vector<StatusFunction> requiredTilingFuncs{std::bind(&CompressorV2Tiling::GetNpuInfo, this),
                                                    std::bind(&CompressorV2Tiling::CheckRequiredParaExistence, this),
                                                    std::bind(&CompressorV2Tiling::CheckEmptyTensor, this),
                                                    std::bind(&CompressorV2Tiling::CheckSinglePara, this),
                                                    std::bind(&CompressorV2Tiling::SetBaseInfo, this),
                                                    std::bind(&CompressorV2Tiling::SetPageAttentionInfo, this),
                                                    std::bind(&CompressorV2Tiling::CheckFeature, this),
                                                    std::bind(&CompressorV2Tiling::CheckMultiParaConsistency, this),
                                                    std::bind(&CompressorV2Tiling::CheckBlockDimConstrain, this),
                                                    std::bind(&CompressorV2Tiling::SetTemplateId, this),
                                                    std::bind(&CompressorV2Tiling::SetInnerSplitInfo, this),
                                                    std::bind(&CompressorV2Tiling::SetWorkSpaceInfo, this),
                                                    std::bind(&CompressorV2Tiling::SetScenarioInfo, this)};
    for (const auto &func : requiredTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    if (context_->templateId == TemplateId::EMPTY_X) {
        workspaceSize_ = libapiSize_;
        if (context_->workSpaces) {
            context_->workSpaces[0] = workspaceSize_;
        }
        GenTilingKey();
        context_->blockDim = 1U;
        return ge::GRAPH_SUCCESS;
    }
    std::vector<StatusFunction> optionalTilingFuncs{std::bind(&CompressorV2Tiling::CalcWorkSpace, this),
                                                    std::bind(&CompressorV2Tiling::GenTilingKey, this)};
    for (const auto &func : optionalTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    context_->blockDim = aicNum_;

    OP_LOGI("Run big kernel");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::GenTilingKey() const
{
    // 0:BF16, 1:FP16
    uint8_t dtype = 0;
    // 0: BSH 1:TH
    uint8_t layout = 0;
    uint8_t templateId = static_cast<uint8_t>(context_->templateId);

    auto xDtype = context_->x.desc->GetDataType();
    if (xDtype == ge::DT_BF16) {
        dtype = 0;
    } else if (xDtype == ge::DT_FLOAT16) {
        dtype = 1;
    }
    auto xDimNum = context_->x.shape->GetStorageShape().GetDimNum();
    if (xDimNum == COMPRESSOR_DIM_NUM_3) {
        layout = 0;
    } else {
        layout = 1;
    }

    context_->tilingKey = GET_TPL_TILING_KEY(layout, dtype, templateId);
    OP_LOGI(context_->opName, "CompressorV2 dtype:%hhu layout:%hhu template_id:%hhu", dtype, layout, templateId);
    OP_LOGI(context_->opName, "CompressorV2 tilingKey:%lu", context_->tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaCmpRatio() || ge::GRAPH_SUCCESS != CheckSingleParaX() ||
        ge::GRAPH_SUCCESS != CheckSingleParaWkv() || ge::GRAPH_SUCCESS != CheckSingleParaWgate() ||
        ge::GRAPH_SUCCESS != CheckSingleParaStateCache() || ge::GRAPH_SUCCESS != CheckSingleParaStateBlockTable() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCuSeqlens() || ge::GRAPH_SUCCESS != CheckSingleParaSeqused() ||
        ge::GRAPH_SUCCESS != CheckSingleParaStartPos() || ge::GRAPH_SUCCESS != CheckSingleParaCmpKv()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus CompressorV2Tiling::CheckFeatureValueSupport(const T *featureValue,
                                                             const std::vector<T> &expectFeatureValList,
                                                             const std::string &name) const
{
    if (std::find(expectFeatureValList.begin(), expectFeatureValList.end(), *featureValue) ==
        expectFeatureValList.end()) {
        LogErrorNumberSupport(expectFeatureValList, *featureValue, name, "feature value");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus CompressorV2Tiling::CheckAttrValueSupport(const T *attrValue, const std::vector<T> &expectAttrValList,
                                                          const std::string &name) const
{
    if (attrValue == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectAttrValList.begin(), expectAttrValList.end(), *attrValue) == expectAttrValList.end()) {
        LogErrorNumberSupport(expectAttrValList, *attrValue, name, "attr value");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

template <typename T>
std::string to_string(const T &value)
{
    if (std::is_same_v<T, bool>) {
        return value ? "true" : "false";
    } else {
        return std::to_string(value);
    }
}

template <typename T>
void CompressorV2Tiling::LogErrorNumberSupport(const std::vector<T> &expectNumberList, const T &actualValue,
                                               const std::string &name, const std::string subName) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }

    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, name, to_string(actualValue),
                                          subName + " only supports " + oss.str());
}

static std::string LayoutTypeToStr(LayoutType layout)
{
    switch (layout) {
        case LayoutType::LAYOUT_BSH:
            return "BSH";
        case LayoutType::LAYOUT_TH:
            return "TH";
        default:
            return "UNKNOWN_LAYOUT";
    }
}

ge::graphStatus CompressorV2Tiling::CheckDimNumInLayoutSupport(const std::string &layout,
                                                               const gert::StorageShape *shape,
                                                               const std::string &name) const
{
    const auto &dimIt = LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(
        shape->GetStorageShape().GetDimNum() != dimIt->second,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->opName, name, std::to_string(shape->GetStorageShape().GetDimNum()),
                                     std::to_string(dimIt->second)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
                                                      const std::string &name) const
{
    if (desc != nullptr) {
        const auto &it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(it == DTYPE_SUPPORT_MAP.end(),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context_->opName, name, "datatype support list should be specify in DTYPE_SUPPORT_MAP"),
                    return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(
            std::find(expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end(),
            LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void CompressorV2Tiling::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
                                              const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << DataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE_FOR_INVALID_DTYPE(context_->opName, name, DataTypeToSerialString(actualDtype), oss.str());
}

static std::string DataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("CompressorV2", "datatype", std::to_string(static_cast<int32_t>(type)),
                                              "not support");
        return "UNDEFINED";
    }
}

ge::graphStatus CompressorV2Tiling::CheckDimNumSupport(const gert::StorageShape *shape, const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto &it = DIM_NUM_MAP.find(name);
    OP_CHECK_IF(it == DIM_NUM_MAP.end(),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, name,
                                                         "dim number support list should be specify in DIM_NUM_MAP"),
                return ge::GRAPH_FAILED);
    auto &expectDimNumList = it->second;
    OP_CHECK_IF(
        std::find(expectDimNumList.begin(), expectDimNumList.end(), shape->GetStorageShape().GetDimNum()) ==
            expectDimNumList.end(),
        [&]() {
            std::ostringstream oss;
            for (size_t i = 0; i < expectDimNumList.size(); ++i) {
                oss << expectDimNumList[i];
                if (i < expectDimNumList.size() - 1) {
                    oss << " or ";
                }
            }
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->opName, name,
                                                     std::to_string(shape->GetStorageShape().GetDimNum()),
                                                     name + " dimension should be " + oss.str());
        }(),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckSingleParaX() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->x.desc, X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->x.shape, X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(LayoutTypeToStr(context_->layout), context_->x.shape, X_NAME)) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(context_->x.shape->GetStorageShape().GetDim(context_->x.shape->GetStorageShape().GetDimNum() - 1) >
                        MAX_HIDDEN_SIZE ||
                    context_->x.shape->GetStorageShape().GetDim(context_->x.shape->GetStorageShape().GetDimNum() - 1) <
                        MIN_HIDDEN_SIZE ||
                    context_->x.shape->GetStorageShape().GetDim(context_->x.shape->GetStorageShape().GetDimNum() - 1) %
                            ALIGN_FACTOR_HIDDEN_SIZE !=
                        0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->opName, X_NAME,
                    "dim " + std::to_string(context_->x.shape->GetStorageShape().GetDimNum() - 1) + "=" +
                        std::to_string(context_->x.shape->GetStorageShape().GetDim(
                            context_->x.shape->GetStorageShape().GetDimNum() - 1)),
                    "hiddenSize (x dim" + std::to_string(context_->x.shape->GetStorageShape().GetDimNum() - 1) +
                        ") should be within [" + std::to_string(MIN_HIDDEN_SIZE) + ", " +
                        std::to_string(MAX_HIDDEN_SIZE) + "] and be 512-aligned"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckSingleParaWkv() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->wkv.desc, WKV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->wkv.shape, WKV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    constexpr uint32_t coffVal = 1;
    uint32_t headDim = context_->wkv.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0);
    OP_CHECK_IF(
        std::find(HEAD_DIM.begin(), HEAD_DIM.end(), headDim) == HEAD_DIM.end(),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->opName, WKV_NAME,
            "dim " + std::to_string(COMPRESSOR_DIM_INDEX_0) + "=" +
                std::to_string(context_->wkv.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0)),
            "headDim (wkv dim0) should be " + std::to_string(HEAD_DIM[0]) + " or " + std::to_string(HEAD_DIM[1])),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckSingleParaWgate() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->wgate.desc, WGATE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->wgate.shape, WGATE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckSingleParaStateCache() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->stateCache.desc, STATE_CACHE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->stateCache.shape, STATE_CACHE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    uint32_t blockSize = context_->stateCache.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1);
    int64_t contiguousStride = context_->stateCache.shape->GetStorageShape().GetDim(1) *
                               context_->stateCache.shape->GetStorageShape().GetDim(2);
    OP_CHECK_IF(context_->stateCacheStrideDim0 == nullptr || *context_->stateCacheStrideDim0 < contiguousStride,
                OP_LOGE(context_->opName, "state_cache stride0 must cover its contiguous rows"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        blockSize > MAX_BLOCK_SIZE || blockSize < MIN_BLOCK_SIZE,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->opName, STATE_CACHE_NAME, "dim 1=" + std::to_string(blockSize),
                                              "state_cache dim 1 should be within [" + std::to_string(MIN_BLOCK_SIZE) +
                                                  ", " + std::to_string(MAX_BLOCK_SIZE) + "]"),
        return ge::GRAPH_FAILED);

    uint32_t blockNum = context_->stateCache.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0);
    uint32_t batchSize = context_->x.shape->GetStorageShape().GetDimNum() == COMPRESSOR_DIM_NUM_3 ?
                             context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0) :
                             context_->cuSeqlens.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0) - 1;
    OP_CHECK_IF(blockNum < batchSize,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->opName, STATE_CACHE_NAME, "dim 0=" + std::to_string(blockNum),

                    ", state_cache dim 0 should not be less than batchSize(" + std::to_string(batchSize) + ")"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context_->stateBlockTable.shape->GetStorageShape().GetDimNum() != COMPRESSOR_DIM_NUM_1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->opName, "state_block_table",
                                     std::to_string(context_->stateBlockTable.shape->GetStorageShape().GetDimNum()),
                                     std::to_string(COMPRESSOR_DIM_NUM_1)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckSingleParaStateBlockTable() const
{
    if (context_->stateBlockTable.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->stateBlockTable.desc, STATE_BLOCK_TABLE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->stateBlockTable.shape, STATE_BLOCK_TABLE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckSingleParaCuSeqlens() const
{
    if (context_->cuSeqlens.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->cuSeqlens.desc, CU_SEQLENS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->cuSeqlens.shape, CU_SEQLENS_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckSingleParaSeqused() const
{
    if (context_->seqUsed.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->seqUsed.desc, SEQUSED_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->seqUsed.shape, SEQUSED_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckSingleParaStartPos() const
{
    if (context_->startPos.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->startPos.desc, START_POS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->startPos.shape, START_POS_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckSingleParaCmpKv() const
{
    if (context_->cmpKv.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->cmpKv.desc, CMP_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->cmpKv.shape, CMP_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckSingleParaCmpRatio() const
{
    int64_t cmpRatio = *context_->cmpRatio;
    OP_CHECK_IF(cmpRatio > MAX_CMPRATIO_SIZE || cmpRatio < MIN_CMPRATIO_SIZE || (cmpRatio & (cmpRatio - 1)) != 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, "cmp_ratio", std::to_string(cmpRatio),
                                                      "cmp_ratio must be a power of two within [" +
                                                          std::to_string(MIN_CMPRATIO_SIZE) + ", " +
                                                          std::to_string(MAX_CMPRATIO_SIZE) + "]"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(context_->x.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "x", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->x.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "x", "desc is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wkv.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "wkv", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wkv.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "wkv", "desc is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wgate.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "wgate", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wgate.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "wgate", "desc is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->stateCache.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "state_cache", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->stateCache.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "state_cache", "desc is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->stateBlockTable.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "state_block_table", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->stateBlockTable.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "state_block_table", "desc is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->cmpKv.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cmp_kv", "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->cmpKv.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cmp_kv", "desc is nullptr"),
                return ge::GRAPH_FAILED);
    if (context_->layout == LayoutType::LAYOUT_TH) {
        OP_CHECK_IF(context_->cuSeqlens.desc == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cu_seqlens",
                                                             "cu_seqlens should not be nullptr in TH layout"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->cuSeqlens.shape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cu_seqlens",
                                                             "cu_seqlens should not be nullptr in TH layout"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(context_->cuSeqlens.desc != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cu_seqlens",
                                                             "cu_seqlens must be nullptr in BSH layout"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->cuSeqlens.shape != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cu_seqlens",
                                                             "cu_seqlens must be nullptr in BSH layout"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(context_->cmpRatio == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cmp_ratio", "attr is nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckFeature() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::LogErrorShapeConsistency(const std::string &name, const gert::StorageShape *shape,
                                                             const uint32_t &dimNum, const std::string &subName,
                                                             const uint32_t &expectNum) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const uint32_t actualNum = shape->GetStorageShape().GetDim(dimNum);
    OP_CHECK_IF(actualNum != expectNum,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->opName, name, "dim " + std::to_string(dimNum) + "=" + std::to_string(actualNum),
                    name + " should be equal to " + subName + ": " + std::to_string(expectNum)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckShapeConsistency() const
{
    if (ge::GRAPH_SUCCESS != LogErrorShapeConsistency(STATE_BLOCK_TABLE_NAME, context_->stateBlockTable.shape,
                                                      COMPRESSOR_DIM_INDEX_0, "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency(CU_SEQLENS_NAME, context_->cuSeqlens.shape,
                                                      COMPRESSOR_DIM_INDEX_0, "batchSize+1",
                                                      baseParams_->batchSize + 1) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency(SEQUSED_NAME, context_->seqUsed.shape, COMPRESSOR_DIM_INDEX_0,
                                                      "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency(START_POS_NAME, context_->startPos.shape, COMPRESSOR_DIM_INDEX_0,
                                                      "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency(WKV_NAME, context_->wkv.shape, COMPRESSOR_DIM_INDEX_1, "x",
                                                      baseParams_->hiddenSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency(WGATE_NAME, context_->wgate.shape, COMPRESSOR_DIM_INDEX_1, "x",
                                                      baseParams_->hiddenSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency(WKV_NAME, context_->wkv.shape, COMPRESSOR_DIM_INDEX_0, "headDim",
                                                      baseParams_->headDim) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency(WGATE_NAME, context_->wgate.shape, COMPRESSOR_DIM_INDEX_0,
                                                      "headDim", baseParams_->headDim)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckDtypeConsistencyX(const gert::CompileTimeTensorDesc *desc,
                                                           const std::string &name) const
{
    const auto actualDtype = desc->GetDataType();
    OP_CHECK_IF(actualDtype != context_->dtype,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context_->opName, name, DataTypeToSerialString(actualDtype),
                    name + " should be same with x: " + DataTypeToSerialString(context_->dtype)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckDtypeConsistency() const
{
    if (CheckDtypeConsistencyX(context_->wkv.desc, WKV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->wgate.desc, WGATE_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->cmpKv.desc, CMP_KV_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckDimNumConsistency() const
{
    auto xDimNum = context_->x.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        xDimNum != context_->cmpKv.shape->GetStorageShape().GetDimNum(),
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context_->opName, "cmp_kv, x",
            std::to_string(context_->cmpKv.shape->GetStorageShape().GetDimNum()) + ", " + std::to_string(xDimNum),
            "dim num of cmp_kv should be equal to x"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckBlockDimConstrain() const
{
    uint32_t minBlockNum = baseParams_->headDim / 64; // 64 is the largest dBaseSize
    OP_CHECK_IF(aicNum_ < minBlockNum,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, "aicNum", std::to_string(aicNum_),
                                                      "aicNum should not be less than " + std::to_string(minBlockNum)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorV2Tiling::CheckMultiParaConsistency() const
{
    if (CheckShapeConsistency() != ge::GRAPH_SUCCESS || CheckDtypeConsistency() != ge::GRAPH_SUCCESS ||
        CheckDimNumConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace

CMP_EXTERN_C ge::graphStatus TilingCompressorV2Arch22(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("CompressorV2", "context", "is nullptr"),
                return ge::GRAPH_FAILED);

    OP_LOGI("Getting Tiling");

    CompressorV2Context compressorContext{};
    if (CompressorV2Tiling::ConvertContext(*context, compressorContext) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context->GetNodeName(), "context", "error occurred while converting tilingContext to CompressorV2 context");
        return ge::GRAPH_FAILED;
    }
    CompressorV2Tiling compressorTiling(&compressorContext);
    CompressorV2TilingData *tilingData = context->GetTilingData<CompressorV2TilingData>();
    OP_CHECK_IF(tilingData == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(compressorContext.opName, "tilingData", "is nullptr"),
                return ge::GRAPH_FAILED);
    // 使用SyncAll，需要设置为batchmode模式，所有核同时启动，否则多流方式下执行可能会卡死
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    if (compressorTiling.RunBigKernelTiling(tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(compressorContext.tilingKey);
    context->SetBlockDim(compressorContext.blockDim);
    OP_LOGI(compressorContext.opName, "block dim: %u.", compressorContext.blockDim);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
