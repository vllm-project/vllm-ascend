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

#include <numeric>
#include <functional>
#include <algorithm>
#include <unordered_map>
#include <graph/utils/type_utils.h>
#include "err/ops_err.h"
#include "register/op_def_registry.h"
#include "compressor_tiling.h"

using namespace ge;
using namespace AscendC;
namespace optiling {


void CompressorTiling::ConvertRequiredParams(gert::TilingContext &context, CompressorContext &compressorContext)
{
    compressorContext.x.desc = context.GetRequiredInputDesc(TOKEN_X_INPUT_INDEX);
    compressorContext.x.shape = context.GetRequiredInputShape(TOKEN_X_INPUT_INDEX);
    compressorContext.wkv.desc = context.GetRequiredInputDesc(WEIGHT_KV_INPUT_INDEX);
    compressorContext.wkv.shape = context.GetRequiredInputShape(WEIGHT_KV_INPUT_INDEX);
    compressorContext.wgate.desc = context.GetRequiredInputDesc(WEIGHT_WGATE_INPUT_INDEX);
    compressorContext.wgate.shape = context.GetRequiredInputShape(WEIGHT_WGATE_INPUT_INDEX);
    compressorContext.stateCache.desc = context.GetRequiredInputDesc(STATE_CACHE_INPUT_INDEX);
    compressorContext.stateCache.shape = context.GetRequiredInputShape(STATE_CACHE_INPUT_INDEX);
    compressorContext.ape.desc = context.GetRequiredInputDesc(APE_INPUT_INDEX);
    compressorContext.ape.shape = context.GetRequiredInputShape(APE_INPUT_INDEX);
    compressorContext.normWeight.desc = context.GetRequiredInputDesc(NORM_WEIGHT_INPUT_INDEX);
    compressorContext.normWeight.shape = context.GetRequiredInputShape(NORM_WEIGHT_INPUT_INDEX);
    compressorContext.ropeSin.desc = context.GetRequiredInputDesc(ROPE_SIN_INPUT_INDEX);
    compressorContext.ropeSin.shape = context.GetRequiredInputShape(ROPE_SIN_INPUT_INDEX);
    compressorContext.ropeCos.desc = context.GetRequiredInputDesc(ROPE_COS_INPUT_INDEX);
    compressorContext.ropeCos.shape = context.GetRequiredInputShape(ROPE_COS_INPUT_INDEX);

    compressorContext.cmpKv.desc = context.GetOutputDesc(CMP_KV_OUTPUT_INDEX);
    compressorContext.cmpKv.shape = context.GetOutputShape(CMP_KV_OUTPUT_INDEX);

    compressorContext.dtype = compressorContext.x.desc->GetDataType();
    auto xDimNum = compressorContext.x.shape->GetStorageShape().GetDimNum();
    if (xDimNum == COMPRESSOR_DIM_NUM_3) {
        compressorContext.layout = LayoutType::LAYOUT_BSH;
    } else if (xDimNum == COMPRESSOR_DIM_NUM_2) {
        compressorContext.layout = LayoutType::LAYOUT_TH;
    }
}

void CompressorTiling::ConvertOptionalParams(gert::TilingContext &context, CompressorContext &compressorContext)
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

ge::graphStatus CompressorTiling::ConvertContext(gert::TilingContext &context, CompressorContext &compressorContext)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE("Compressor", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }

    OP_LOGI("Getting Context");

    compressorContext.opName = context.GetNodeName();
    compressorContext.opType = context.GetNodeType();
    compressorContext.platformInfo = context.GetPlatformInfo();
    ConvertRequiredParams(context, compressorContext);
    ConvertOptionalParams(context, compressorContext);

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context.GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    compressorContext.ropeHeadDim = attrs->GetAttrPointer<int>(ROPE_HEAD_DIM_ATTR_INDEX);
    compressorContext.coff = attrs->GetAttrPointer<int>(COFF_ATTR_INDEX);
    compressorContext.cmpRatio = attrs->GetAttrPointer<int>(CMP_RATIO_ATTR_INDEX);
    compressorContext.normEps = attrs->GetAttrPointer<float>(NORM_EPS_ATTR_INDEX);
    compressorContext.rotaryMode = attrs->GetAttrPointer<int>(ROTARY_MODE_ATTR_INDEX);
    compressorContext.cacheMode = attrs->GetAttrPointer<int>(CACHE_MODE_ATTR_INDEX);
    compressorContext.stateCacheStrideDim0 = attrs->GetAttrPointer<int>(STATE_CACHE_STRIDE_DIM0_ATTR_INDEX);

    OP_CHECK_IF(context.GetWorkspaceSizes(1) == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "workSpaceSize got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    compressorContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::GetNpuInfo()
{
    OP_CHECK_IF(context_->platformInfo == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();

    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);

    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();

    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0,
                OPS_REPORT_VECTOR_INNER_ERR(context_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::SetBaseInfo()
{
    if (context_->x.shape->GetStorageShape().GetDimNum() == COMPRESSOR_DIM_NUM_3) {
        baseParams_->batchSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0);
        baseParams_->seqSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1);
        baseParams_->hiddenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_2);
        baseParams_->tokenSize = baseParams_->batchSize * baseParams_->seqSize;
        baseParams_->cgSize = context_->ropeSin.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1);
    } else {
        baseParams_->batchSize = context_->cuSeqlens.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0) - 1;
        baseParams_->tokenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0);
        baseParams_->hiddenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1);
        baseParams_->cgSize = context_->ropeSin.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0);
    }

    baseParams_->headDim = context_->normWeight.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0);
    baseParams_->cmpRatio = static_cast<uint32_t>(*context_->cmpRatio);
    baseParams_->csSize = baseParams_->seqSize - (baseParams_->seqSize % baseParams_->cmpRatio);
    baseParams_->ropeHeadDim = static_cast<uint32_t>(*context_->ropeHeadDim);
    baseParams_->normEps = static_cast<float>(*context_->normEps);
    baseParams_->reciprocalD = 1.0 / baseParams_->headDim;
    baseParams_->cgSize =
        (baseParams_->seqSize + baseParams_->cmpRatio - 1) / baseParams_->cmpRatio; // number of token after compress
    coff = static_cast<uint8_t>(*context_->coff);
    baseParams_->stateCacheStrideDim0 = static_cast<uint64_t>(*context_->stateCacheStrideDim0);
    baseParams_->nSize = 2; // 2:每个核处理两个基本块后做全核同步
    baseParams_->usedCoreNum = aicNum_;
    OP_LOGI(context_->opName, "[TILING] bSize:%u  tSize:%u cmpRatio:%u coff:%u, stateCacheStrideDim0:%u",
            baseParams_->batchSize, baseParams_->tokenSize, baseParams_->cmpRatio, coff,
            baseParams_->stateCacheStrideDim0);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::SetPageAttentionInfo()
{
    pageAttentionParams_->blockNum = context_->stateCache.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0);
    pageAttentionParams_->blockSize = context_->stateCache.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1);
    if (static_cast<uint8_t>(*context_->cacheMode) == static_cast<uint8_t>(CACHE_MODE::CONTINUOUS)) {
        pageAttentionParams_->maxBlockNumPerBatch =
            context_->stateBlockTable.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::SetWorkSpaceInfo()
{
    workspaceParams_->dbWorkspaceRatio = 2;
    workspaceParams_->mm1KvResSize = innerSplitParams_->mBaseSize * baseParams_->headDim * coff;
    workspaceParams_->mm1ScoreResSize = innerSplitParams_->mBaseSize * baseParams_->headDim * coff;
    if (coff == 2) {
        workspaceParams_->vec1TailCacheSize = baseParams_->cmpRatio * baseParams_->headDim;
    }
    workspaceParams_->vec1ResSize = innerSplitParams_->mBaseSize * baseParams_->headDim * baseParams_->nSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::SetScenarioInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::SetTemplateId()
{
    if (context_->templateId == TemplateId::EMPTY_X) {
        return ge::GRAPH_SUCCESS;
    }
    if (socVersion_ == platform_ascendc::SocVersion::ASCEND950) {
        // 设置高性能模板
        if (context_->layout == LayoutType::LAYOUT_BSH && baseParams_->seqSize <= 4 && baseParams_->tokenSize <= 256) {
            context_->templateId = TemplateId::FULL_LOAD;
        }
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::SetInnerSplitInfo()
{
    if (context_->templateId == TemplateId::FULL_LOAD) {
        innerSplitParams_->mBaseSize = 256;              // 256:核间切分，M轴基本块大小
        innerSplitParams_->dBaseSize = 256 / (coff * 2); // nBase = dBase * coff * 2
        uint32_t dBaseNum = baseParams_->headDim / innerSplitParams_->dBaseSize;
        uint32_t mBaseNum = (baseParams_->tokenSize + innerSplitParams_->mBaseSize - 1) / innerSplitParams_->mBaseSize;
        baseParams_->coreGroupNum = baseParams_->usedCoreNum / dBaseNum;
        baseParams_->kBaseNum = 1;
        baseParams_->kBaseSize = baseParams_->hiddenSize;
        if ((dBaseNum * mBaseNum) < baseParams_->usedCoreNum) {
            baseParams_->kBaseNum = baseParams_->usedCoreNum / dBaseNum;
            baseParams_->kBaseSize = (baseParams_->hiddenSize + baseParams_->kBaseNum - 1) / baseParams_->kBaseNum;
        }
        for (uint32_t i = 0; i < baseParams_->usedCoreNum; i++) {
            baseParams_->splitCoreParam[i].nStart = (i % dBaseNum) * innerSplitParams_->dBaseSize;
            baseParams_->splitCoreParam[i].nEnd = baseParams_->splitCoreParam[i].nStart + innerSplitParams_->dBaseSize;
            if (baseParams_->kBaseNum > 1) {
                baseParams_->splitCoreParam[i].kStart = (i / dBaseNum) * baseParams_->kBaseSize;
                baseParams_->splitCoreParam[i].kEnd = baseParams_->splitCoreParam[i].kStart + baseParams_->kBaseSize;
                baseParams_->splitCoreParam[i].mStart = 0;
                baseParams_->splitCoreParam[i].mEnd = baseParams_->tokenSize;
                baseParams_->mLoopNum = 1;
            } else {
                baseParams_->splitCoreParam[i].kStart = 0;
                baseParams_->splitCoreParam[i].kEnd = baseParams_->splitCoreParam[i].kStart + baseParams_->kBaseSize;
                baseParams_->splitCoreParam[i].mStart = (i / dBaseNum) * innerSplitParams_->mBaseSize;
                baseParams_->splitCoreParam[i].mEnd =
                    baseParams_->splitCoreParam[i].mStart + innerSplitParams_->mBaseSize;
                baseParams_->mLoopNum = mBaseNum / baseParams_->coreGroupNum;
            }
        }
    } else {
        innerSplitParams_->mBaseSize = 256;        // 256:核间切分，M轴基本块大小
        innerSplitParams_->dBaseSize = 128 / coff; // 128：核间切分，D轴基本块大小
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CalcWorkSpace()
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

ge::graphStatus CompressorTiling::CheckEmptyTensor() const
{
    if (context_->layout == LayoutType::LAYOUT_BSH &&
            context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0) == 0 ||
        context_->layout == LayoutType::LAYOUT_BSH &&
            context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1) == 0 ||
        context_->layout == LayoutType::LAYOUT_TH &&
            context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0) == 0) {
        context_->templateId = TemplateId::EMPTY_X;
    } else {
        if (context_->x.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->wkv.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->wgate.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->stateCache.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->ape.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->normWeight.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->ropeSin.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->ropeCos.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->stateBlockTable.shape->GetStorageShape().GetShapeSize() == 0) {
            OP_LOGE(context_->opName, "Only input tensor x dim B or S or T supports to be 0");
            return ge::GRAPH_FAILED;
        }
        context_->templateId = TemplateId::NORMAL;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::RunBigKernelTiling(CompressorTilingData *tilingData)
{
    this->baseParams_ = &tilingData->baseParams;
    this->pageAttentionParams_ = &tilingData->pageAttentionParams;
    this->innerSplitParams_ = &tilingData->innerSplitParams;
    this->workspaceParams_ = &tilingData->workspaceParams;
    using StatusFunction = std::function<ge::graphStatus()>;
    std::vector<StatusFunction> requiredTilingFuncs {
        std::bind(&CompressorTiling::GetNpuInfo, this),
        std::bind(&CompressorTiling::CheckRequiredParaExistence, this),
        std::bind(&CompressorTiling::CheckEmptyTensor, this),
        std::bind(&CompressorTiling::CheckSinglePara, this),
        std::bind(&CompressorTiling::SetBaseInfo, this),
        std::bind(&CompressorTiling::SetPageAttentionInfo, this),
        std::bind(&CompressorTiling::CheckFeature, this),
        std::bind(&CompressorTiling::CheckMultiParaConsistency, this),
        std::bind(&CompressorTiling::CheckBlockDimConstrain, this),
        std::bind(&CompressorTiling::SetTemplateId, this),
        std::bind(&CompressorTiling::SetInnerSplitInfo, this),
        std::bind(&CompressorTiling::SetWorkSpaceInfo, this),
        std::bind(&CompressorTiling::SetScenarioInfo, this)
    };
    for (const auto &func: requiredTilingFuncs) {
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
    std::vector<StatusFunction> optionalTilingFuncs {
        std::bind(&CompressorTiling::CalcWorkSpace, this),
        std::bind(&CompressorTiling::GenTilingKey, this)
    };
    for (const auto &func : optionalTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    context_->blockDim = aicNum_;

    OP_LOGI("Run big kernel");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::GenTilingKey() const
{
    // 0:BF16, 1:FP16
    uint8_t dtype = 0;
    // 0: BSH 1:TH
    uint8_t layout = 0;
    uint8_t rotaryMode = static_cast<uint8_t>(*context_->rotaryMode);
    uint8_t templateId = static_cast<uint8_t>(context_->templateId);
    uint8_t cacheMode = static_cast<uint8_t>(*context_->cacheMode);

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

    context_->tilingKey = GET_TPL_TILING_KEY(
        layout,
        dtype,
        coff,
        rotaryMode,
        cacheMode,
        templateId
    );
    OP_LOGI(context_->opName,
            "Compressor dtype:%hhu layout:%hhu  coff:%hhu rotary_mode:%hhu, cacheMode: %u, template_id:%hhu", dtype,
            layout, coff, rotaryMode, cacheMode, templateId);
    OP_LOGI(context_->opName, "Compressor tilingKey:%lu", context_->tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaX() ||
        ge::GRAPH_SUCCESS != CheckSingleParaWkv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaWgate() ||
        ge::GRAPH_SUCCESS != CheckSingleParaStateCache() ||
        ge::GRAPH_SUCCESS != CheckSingleParaApe() ||
        ge::GRAPH_SUCCESS != CheckSingleParaNormWeight() ||
        ge::GRAPH_SUCCESS != CheckSingleParaRopeSin() ||
        ge::GRAPH_SUCCESS != CheckSingleParaRopeCos() ||
        ge::GRAPH_SUCCESS != CheckSingleParaStateBlockTable() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCuSeqlens() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSeqused() ||
        ge::GRAPH_SUCCESS != CheckSingleParaStartPos() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpKv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaRopeHeadDim() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpRatio() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCoff() ||
        ge::GRAPH_SUCCESS != CheckSingleParaNormEps() ||
        ge::GRAPH_SUCCESS != CheckSingleParaRotaryMode() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCacheMode()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus CompressorTiling::CheckFeatureValueSupport(const T *featureValue,
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
ge::graphStatus CompressorTiling::CheckAttrValueSupport(const T *attrValue, const std::vector<T> &expectAttrValList,
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
void CompressorTiling::LogErrorNumberSupport(const std::vector<T> &expectNumberList, const T &actualValue,
                                             const std::string &name, const std::string subName) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }

    OP_LOGE(context_->opName, "%s %s only supports %s, but got %s", name.c_str(), subName.c_str(), oss.str().c_str(),
            to_string(actualValue).c_str());
}

std::string LayoutTypeToStr(LayoutType layout)
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

ge::graphStatus CompressorTiling::CheckDimNumInLayoutSupport(const std::string &layout, const gert::StorageShape *shape,
                                                             const std::string &name) const
{
    const auto &dimIt = LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
                OP_LOGE(context_->opName, "When layout is %s, %s dimension should be %zu, but it's %zu", layout.c_str(),
                        name.c_str(), dimIt->second, shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
                                                    const std::string &name) const
{
    if (desc != nullptr) {
        const auto &it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(
            it == DTYPE_SUPPORT_MAP.end(),
            OP_LOGE(context_->opName, "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
            return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(std::find(expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) ==
                        expectDtypeList.end(),
                    LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void CompressorTiling::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
                                            const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << DataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(context_->opName, "Tensor %s only supports dtype %s, but got %s", name.c_str(), oss.str().c_str(),
            DataTypeToSerialString(actualDtype).c_str());
}

static std::string DataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("Compressor", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

ge::graphStatus CompressorTiling::CheckDimNumSupport(const gert::StorageShape *shape, const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto &it = DIM_NUM_MAP.find(name);
    OP_CHECK_IF(it == DIM_NUM_MAP.end(),
                OP_LOGE(context_->opName, "%s dim number support list should be specify in DIM_NUM_MAP", name.c_str()),
                return ge::GRAPH_FAILED);
    auto &expectDimNumList = it->second;
    OP_CHECK_IF(std::find(expectDimNumList.begin(), expectDimNumList.end(), shape->GetStorageShape().GetDimNum()) ==
                    expectDimNumList.end(),
                LogErrorNumberSupport(expectDimNumList, static_cast<uint32_t>(shape->GetStorageShape().GetDimNum()),
                                      name, "dimension"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaX() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->x.desc, X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->x.shape, X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(LayoutTypeToStr(context_->layout), context_->x.shape, X_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaWkv() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->wkv.desc, WKV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->wkv.shape, WKV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaWgate() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->wgate.desc, WGATE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->wgate.shape, WGATE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaStateCache() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->stateCache.desc, STATE_CACHE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->stateCache.shape, STATE_CACHE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaApe() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->ape.desc, APE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->ape.shape, APE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaNormWeight() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->normWeight.desc, NORM_WEIGHT_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->normWeight.shape, NORM_WEIGHT_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaRopeSin() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->ropeSin.desc, ROPE_SIN_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->ropeSin.shape, ROPE_SIN_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumInLayoutSupport(LayoutTypeToStr(context_->layout), context_->ropeSin.shape, ROPE_SIN_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaRopeCos() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->ropeCos.desc, ROPE_COS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->ropeCos.shape, ROPE_COS_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumInLayoutSupport(LayoutTypeToStr(context_->layout), context_->ropeCos.shape, ROPE_COS_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaStateBlockTable() const
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

ge::graphStatus CompressorTiling::CheckSingleParaCuSeqlens() const
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

ge::graphStatus CompressorTiling::CheckSingleParaSeqused() const
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

ge::graphStatus CompressorTiling::CheckSingleParaStartPos() const
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

ge::graphStatus CompressorTiling::CheckSingleParaCmpKv() const
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

ge::graphStatus CompressorTiling::CheckSingleParaRopeHeadDim() const
{
    if (CheckAttrValueSupport(context_->ropeHeadDim, ROPE_HEAD_DIM, ROPE_HEAD_DIM_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaCmpRatio() const
{
    if (CheckAttrValueSupport(context_->cmpRatio, CMP_RATIO, CMP_RATIO_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaCoff() const
{
    if (CheckAttrValueSupport(context_->coff, COFF, COFF_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaNormEps() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaRotaryMode() const
{
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupport(context_->rotaryMode, ROTARY_MODE, ROTARY_MODE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaCacheMode() const
{
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupport(context_->cacheMode, CACHE_MODE, CACHE_MODE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(context_->x.shape == nullptr, OP_LOGE(context_->opName, "tensor x is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->x.desc == nullptr, OP_LOGE(context_->opName, "tensor x is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wkv.shape == nullptr, OP_LOGE(context_->opName, "tensor wkv is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wkv.desc == nullptr, OP_LOGE(context_->opName, "tensor wkv is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wgate.shape == nullptr, OP_LOGE(context_->opName, "tensor wgate is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wgate.desc == nullptr, OP_LOGE(context_->opName, "tensor wgate is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->stateCache.shape == nullptr, OP_LOGE(context_->opName, "tensor stateCache is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->stateCache.desc == nullptr, OP_LOGE(context_->opName, "tensor stateCache is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ape.shape == nullptr, OP_LOGE(context_->opName, "tensor ape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ape.desc == nullptr, OP_LOGE(context_->opName, "tensor ape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->normWeight.shape == nullptr, OP_LOGE(context_->opName, "tensor normWeight is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->normWeight.desc == nullptr, OP_LOGE(context_->opName, "tensor normWeight is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ropeSin.shape == nullptr, OP_LOGE(context_->opName, "tensor ropeSin is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ropeSin.desc == nullptr, OP_LOGE(context_->opName, "tensor ropeSin is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ropeCos.shape == nullptr, OP_LOGE(context_->opName, "tensor ropeCos is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ropeCos.desc == nullptr, OP_LOGE(context_->opName, "tensor ropeCos is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->stateBlockTable.shape == nullptr,
                OP_LOGE(context_->opName, "tensor stateBlockTable is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->stateBlockTable.desc == nullptr,
                OP_LOGE(context_->opName, "tensor stateBlockTable is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->cmpKv.shape == nullptr, OP_LOGE(context_->opName, "tensor cmpKv is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->cmpKv.desc == nullptr, OP_LOGE(context_->opName, "tensor cmpKv is nullptr"),
                return ge::GRAPH_FAILED);
    if (context_->layout == LayoutType::LAYOUT_TH) {
        OP_CHECK_IF(context_->cuSeqlens.desc == nullptr,
                    OP_LOGE(context_->opName, "In TH layout, tensor cuSeqlens should not be nullptr"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->cuSeqlens.shape == nullptr,
                    OP_LOGE(context_->opName, "In TH layout, tensor cuSeqlens should not be nullptr"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(context_->cuSeqlens.desc != nullptr,
                    OP_LOGE(context_->opName, "In BSH layout, tensor cuSeqlens must be nullptr"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->cuSeqlens.shape != nullptr,
                    OP_LOGE(context_->opName, "In TH layout, tensor cuSeqlens must be nullptr"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(context_->ropeHeadDim == nullptr, OP_LOGE(context_->opName, "attr ropeHeadDim is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(context_->cmpRatio == nullptr, OP_LOGE(context_->opName, "attr cmpRatio is nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckFeature() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureValueSupport(&baseParams_->headDim, HEAD_DIM, "headDim")) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(baseParams_->hiddenSize > MAX_HIDDEN_SIZE || baseParams_->hiddenSize < MIN_HIDDEN_SIZE ||
                    baseParams_->hiddenSize % ALIGN_FACTOR_HIDDEN_SIZE != 0,
                OP_LOGE(context_->opName, "hiddenSize should be whthin [1k, 10k] and be 512-aligned, but got %u",
                        baseParams_->hiddenSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        pageAttentionParams_->blockSize < MIN_BLOCK_SIZE,
        OP_LOGE(context_->opName, "blockSize should not be less than 1, but got %u", pageAttentionParams_->blockSize),
        return ge::GRAPH_FAILED);
    if (static_cast<uint8_t>(*context_->cacheMode) == static_cast<uint8_t>(CACHE_MODE::CYCLE)) {
        OP_CHECK_IF(pageAttentionParams_->blockNum < baseParams_->batchSize,
                    OP_LOGE(context_->opName,
                            "when cacheMode is %u, blockNum should not be less than batchSize(%u), but got %u",
                            static_cast<uint8_t>(CACHE_MODE::CYCLE), baseParams_->batchSize,
                            pageAttentionParams_->blockSize),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::LogErrorShapeConsistency(const std::string &name, const gert::StorageShape *shape,
                                                           const uint32_t &dimNum, const std::string &subName,
                                                           const uint32_t &expectNum) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const uint32_t actualNum = shape->GetStorageShape().GetDim(dimNum);
    OP_CHECK_IF(actualNum != expectNum,
                OP_LOGE(context_->opName, "%s shape dim %u, should be equal to %s: %u, but got %u", name.c_str(),
                        dimNum, subName.c_str(), expectNum, actualNum),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckShapeConsistency() const
{
    if (CheckShapeConsistencyRope() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto coffD = coff * baseParams_->headDim;
    uint32_t stateNum = 2;
    if (ge::GRAPH_SUCCESS != LogErrorShapeConsistency("stateBlockTable", context_->stateBlockTable.shape,
                                                      COMPRESSOR_DIM_INDEX_0, "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("cuSeqlens", context_->cuSeqlens.shape, COMPRESSOR_DIM_INDEX_0,
                                                      "batchSize+1", baseParams_->batchSize + 1) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("seqUsed", context_->seqUsed.shape, COMPRESSOR_DIM_INDEX_0,
                                                      "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("startPos", context_->startPos.shape, COMPRESSOR_DIM_INDEX_0,
                                                      "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("wkv", context_->wkv.shape, COMPRESSOR_DIM_INDEX_1, "hiddenSize",
                                                      baseParams_->hiddenSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("wgate", context_->wgate.shape, COMPRESSOR_DIM_INDEX_1,
                                                      "hiddenSize", baseParams_->hiddenSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("wkv", context_->wkv.shape, COMPRESSOR_DIM_INDEX_0,
                                                      "coff*headDim", static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("wgate", context_->wgate.shape, COMPRESSOR_DIM_INDEX_0,
                                                      "coff*headDim", static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("stateCache", context_->stateCache.shape, COMPRESSOR_DIM_INDEX_2,
                                                      "2*coff*headDim", stateNum * static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ape", context_->ape.shape, COMPRESSOR_DIM_INDEX_1,
                                                      "coff*headDim", static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ape", context_->ape.shape, COMPRESSOR_DIM_INDEX_0, "cmpRatio",
                                                      baseParams_->cmpRatio)) {
        return ge::GRAPH_FAILED;
    }
    if (static_cast<uint8_t>(*context_->cacheMode) == static_cast<uint8_t>(CACHE_MODE::CONTINUOUS) &&
        (ge::GRAPH_SUCCESS != LogErrorShapeConsistency("stateCache", context_->stateCache.shape, COMPRESSOR_DIM_INDEX_0,
                                                       "blockNum", pageAttentionParams_->blockNum) ||
         ge::GRAPH_SUCCESS != LogErrorShapeConsistency("stateCache", context_->stateCache.shape, COMPRESSOR_DIM_INDEX_1,
                                                       "blockSize", pageAttentionParams_->blockSize))) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckShapeConsistencyRope() const
{
    auto cmpT =
        std::min(baseParams_->tokenSize, baseParams_->tokenSize / baseParams_->cmpRatio + baseParams_->batchSize);
    if (context_->layout == LayoutType::LAYOUT_BSH) {
        if (ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeSin", context_->ropeSin.shape, COMPRESSOR_DIM_INDEX_0,
                                                          "batchSize", baseParams_->batchSize) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeCos", context_->ropeCos.shape, COMPRESSOR_DIM_INDEX_0,
                                                          "batchSize", baseParams_->batchSize) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeSin", context_->ropeSin.shape, COMPRESSOR_DIM_INDEX_1,
                                                          "ceil(seqSize/cmpRatio)", baseParams_->cgSize) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeCos", context_->ropeCos.shape, COMPRESSOR_DIM_INDEX_1,
                                                          "ceil(seqSize/cmpRatio)", baseParams_->cgSize) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeSin", context_->ropeSin.shape, COMPRESSOR_DIM_INDEX_2,
                                                          "ropeHeadDim", baseParams_->ropeHeadDim) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeCos", context_->ropeCos.shape, COMPRESSOR_DIM_INDEX_2,
                                                          "ropeHeadDim", baseParams_->ropeHeadDim)) {
            return ge::GRAPH_FAILED;
        }
    } else {
        if (ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeSin", context_->ropeSin.shape, COMPRESSOR_DIM_INDEX_0,
                                                          "min(tokenSize, tokenSize/cmpRatio+batchSize)",
                                                          static_cast<uint32_t>(cmpT)) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeCos", context_->ropeCos.shape, COMPRESSOR_DIM_INDEX_0,
                                                          "min(tokenSize, tokenSize/cmpRatio+batchSize)",
                                                          static_cast<uint32_t>(cmpT)) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeSin", context_->ropeSin.shape, COMPRESSOR_DIM_INDEX_1,
                                                          "ropeHeadDim", baseParams_->ropeHeadDim) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeCos", context_->ropeCos.shape, COMPRESSOR_DIM_INDEX_1,
                                                          "ropeHeadDim", baseParams_->ropeHeadDim)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckDtypeConsistencyX(const gert::CompileTimeTensorDesc *desc,
                                                         const std::string &name) const
{
    const auto actualDtype = desc->GetDataType();
    OP_CHECK_IF(actualDtype != context_->dtype,
                OP_LOGE(context_->opName, "%s datatype should be same with x: %s, but got %s", name.c_str(),
                        DataTypeToSerialString(actualDtype).c_str(), DataTypeToSerialString(context_->dtype).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckDtypeConsistency() const
{
    if (CheckDtypeConsistencyX(context_->wkv.desc, WKV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->wgate.desc, WGATE_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->normWeight.desc, NORM_WEIGHT_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->ropeSin.desc, ROPE_SIN_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->ropeCos.desc, ROPE_COS_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->cmpKv.desc, CMP_KV_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckDimNumConsistency() const
{
    auto xDimNum = context_->x.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(xDimNum != context_->ropeSin.shape->GetStorageShape().GetDimNum(),
                OP_LOGE(context_->opName, "ropeSin dim num should be equal to x: %u, but got %u", xDimNum,
                        context_->ropeSin.shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDimNum != context_->ropeCos.shape->GetStorageShape().GetDimNum(),
                OP_LOGE(context_->opName, "ropeCos dim num should be equal to x: %u, but got %u", xDimNum,
                        context_->ropeCos.shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDimNum != context_->cmpKv.shape->GetStorageShape().GetDimNum(),
                OP_LOGE(context_->opName, "cmpKv dim num should be equal to x: %u, but got %u", xDimNum,
                        context_->cmpKv.shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckScenarioConsistency() const
{
    auto curCmpratio = baseParams_->cmpRatio;
    auto curHeaddim = baseParams_->headDim;
    auto curCoff = static_cast<uint8_t>(*context_->coff);
    std::vector<uint32_t> curScenario{curCmpratio, curCoff, curHeaddim};
    const std::vector<std::vector<uint32_t>> allowdScenarios = {{4, 2, 512}, {4, 2, 128}, {128, 1, 512}};

    OP_CHECK_IF(std::find(allowdScenarios.begin(), allowdScenarios.end(), curScenario) == allowdScenarios.end(),
                OP_LOGE(context_->opName,
                        "Cmpratio Coff Headdim should be equal to {4, 2, 512}, {4, 2, 128}, {128, 1, 512}, \
                        but now cmpratio=%u, coff=%u, headdim=%u", curCmpratio, curCoff, curHeaddim),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckBlockDimConstrain() const
{
    uint32_t minBlockNum = baseParams_->headDim / 64; // 64 is the largest dBaseSize
    OP_CHECK_IF(aicNum_ < minBlockNum,
                OP_LOGE(context_->opName, "aicNum is %d, which should not be less than %d", aicNum_, minBlockNum),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckMultiParaConsistency() const
{
    if (CheckShapeConsistency() != ge::GRAPH_SUCCESS || CheckDtypeConsistency() != ge::GRAPH_SUCCESS ||
        CheckDimNumConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
#ifdef DAY0_SCOPE
    if (CheckScenarioConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
#endif
    return ge::GRAPH_SUCCESS;
}

CMP_EXTERN_C ge::graphStatus TilingCompressor(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("Compressor", "Context is nullptr."),
                return ge::GRAPH_FAILED);

    OP_LOGI("Getting Tiling");

    CompressorContext compressorContext{};
    if (CompressorTiling::ConvertContext(*context, compressorContext) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Error occurred while converting tilingContext to Compressor context");
        return ge::GRAPH_FAILED;
    }
    CompressorTiling compressorTiling(&compressorContext);
    CompressorTilingData *tilingData = context->GetTilingData<CompressorTilingData>();
    OP_CHECK_IF(tilingData == nullptr, OPS_REPORT_VECTOR_INNER_ERR(compressorContext.opName, "TilingData is nullptr."),
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

ge::graphStatus TilingPrepareForCompressor(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Compressor)
    .Tiling(TilingCompressor)
    .TilingParse<CompressorCompileInfo>(TilingPrepareForCompressor);
} // namespace optiling
