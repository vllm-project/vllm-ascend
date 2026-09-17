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
 * \file key_pool_tiling.cpp
 * \file key_pool_tiling.cpp
 * \brief
 */

#include <functional>
#include <algorithm>
#include <sstream>
#include <graph/utils/type_utils.h>
#include "err/ops_err.h"
#include "register/op_def_registry.h"
#include "key_pool_tiling_arch35.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
void KeyPoolTiling::ConvertRequiredParams(gert::TilingContext &context, KeyPoolContext &key_poolContext)
{
    key_poolContext.hidden_states.desc = context.GetRequiredInputDesc(HIDDEN_STATES_INPUT_INDEX);
    key_poolContext.hidden_states.shape = context.GetRequiredInputShape(HIDDEN_STATES_INPUT_INDEX);
    key_poolContext.wk.desc = context.GetRequiredInputDesc(WK_INPUT_INDEX);
    key_poolContext.wk.shape = context.GetRequiredInputShape(WK_INPUT_INDEX);
    key_poolContext.gate_weight.desc = context.GetRequiredInputDesc(GATE_WEIGHT_INPUT_INDEX);
    key_poolContext.gate_weight.shape = context.GetRequiredInputShape(GATE_WEIGHT_INPUT_INDEX);
    key_poolContext.ape.desc = context.GetRequiredInputDesc(APE_INPUT_INDEX);
    key_poolContext.ape.shape = context.GetRequiredInputShape(APE_INPUT_INDEX);
    key_poolContext.stateCache.desc = context.GetRequiredInputDesc(STATE_CACHE_INPUT_INDEX);
    key_poolContext.stateCache.shape = context.GetRequiredInputShape(STATE_CACHE_INPUT_INDEX);

    key_poolContext.pooledKey.desc = context.GetOutputDesc(POOLED_KEY_OUTPUT_INDEX);
    key_poolContext.pooledKey.shape = context.GetOutputShape(POOLED_KEY_OUTPUT_INDEX);

    key_poolContext.dtype = key_poolContext.hidden_states.desc->GetDataType();
    auto hiddenStatesDimNum = key_poolContext.hidden_states.shape->GetStorageShape().GetDimNum();
    if (hiddenStatesDimNum == KEY_POOL_DIM_NUM_3) {
        key_poolContext.layout = LayoutType::LAYOUT_BSH;
    } else if (hiddenStatesDimNum == KEY_POOL_DIM_NUM_2) {
        key_poolContext.layout = LayoutType::LAYOUT_TH;
    }
}

void KeyPoolTiling::ConvertOptionalParams(gert::TilingContext &context, KeyPoolContext &key_poolContext)
{
    key_poolContext.cacheBlockTable.desc = context.GetRequiredInputDesc(CACHE_BLOCK_TABLE_INPUT_INDEX);
    key_poolContext.cacheBlockTable.shape = context.GetRequiredInputShape(CACHE_BLOCK_TABLE_INPUT_INDEX);
    key_poolContext.startPos.desc = context.GetRequiredInputDesc(START_POS_INPUT_INDEX);
    key_poolContext.startPos.shape = context.GetRequiredInputShape(START_POS_INPUT_INDEX);
    key_poolContext.normWeight.desc = context.GetOptionalInputDesc(NORM_WEIGHT_INPUT_INDEX);
    key_poolContext.normWeight.shape = context.GetOptionalInputShape(NORM_WEIGHT_INPUT_INDEX);
    key_poolContext.normBias.desc = context.GetOptionalInputDesc(NORM_BIAS_INPUT_INDEX);
    key_poolContext.normBias.shape = context.GetOptionalInputShape(NORM_BIAS_INPUT_INDEX);
    key_poolContext.cos.desc = context.GetOptionalInputDesc(COS_INPUT_INDEX);
    key_poolContext.cos.shape = context.GetOptionalInputShape(COS_INPUT_INDEX);
    key_poolContext.sin.desc = context.GetOptionalInputDesc(SIN_INPUT_INDEX);
    key_poolContext.sin.shape = context.GetOptionalInputShape(SIN_INPUT_INDEX);
    key_poolContext.seqLens.desc = context.GetOptionalInputDesc(CU_SEQLENS_INPUT_INDEX);
    key_poolContext.seqLens.shape = context.GetOptionalInputShape(CU_SEQLENS_INPUT_INDEX);
    key_poolContext.seqUsed.desc = context.GetOptionalInputDesc(SEQUSED_INPUT_INDEX);
    key_poolContext.seqUsed.shape = context.GetOptionalInputShape(SEQUSED_INPUT_INDEX);
}

ge::graphStatus KeyPoolTiling::ConvertContext(gert::TilingContext &context, KeyPoolContext &key_poolContext)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE("KeyPool", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }

    OP_LOGI("Getting Context");

    key_poolContext.opName = context.GetNodeName();
    key_poolContext.opType = context.GetNodeType();
    key_poolContext.platformInfo = context.GetPlatformInfo();
    ConvertRequiredParams(context, key_poolContext);
    ConvertOptionalParams(context, key_poolContext);

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context.GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    key_poolContext.cmpRatio = attrs->GetAttrPointer<int>(CMP_RATIO_ATTR_INDEX);
    key_poolContext.normEps = attrs->GetAttrPointer<float>(NORM_EPS_ATTR_INDEX);
    key_poolContext.rotaryMode = attrs->GetAttrPointer<int>(ROTARY_MODE_ATTR_INDEX);
    key_poolContext.stateCacheStrideDim0 = attrs->GetAttrPointer<int>(STATE_CACHE_STRIDE_DIM0_ATTR_INDEX);
    key_poolContext.batchConsistency = context.GetDeterministicLevel();
    OP_LOGD(context.GetNodeName(), "deterministic_level=%d", context.GetDeterministicLevel());
    OP_CHECK_IF(context.GetWorkspaceSizes(1) == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context.GetNodeName(), "workSpaceSize got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    key_poolContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::GetNpuInfo()
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

ge::graphStatus KeyPoolTiling::SetBaseInfo()
{
    if (context_->hidden_states.shape->GetStorageShape().GetDimNum() == KEY_POOL_DIM_NUM_3) {
        baseParams_->batchSize = context_->hidden_states.shape->GetStorageShape().GetDim(KEY_POOL_DIM_INDEX_0);
        baseParams_->seqSize = context_->hidden_states.shape->GetStorageShape().GetDim(KEY_POOL_DIM_INDEX_1);
        baseParams_->hiddenSize = context_->hidden_states.shape->GetStorageShape().GetDim(KEY_POOL_DIM_INDEX_2);
        baseParams_->tokenSize = baseParams_->batchSize * baseParams_->seqSize;
    } else {
        baseParams_->batchSize = context_->seqLens.shape->GetStorageShape().GetDim(KEY_POOL_DIM_INDEX_0) - 1;
        baseParams_->tokenSize = context_->hidden_states.shape->GetStorageShape().GetDim(KEY_POOL_DIM_INDEX_0);
        // TH request lengths are read from cu_seqlens; use T as the static bound.
        baseParams_->seqSize = baseParams_->tokenSize;
        baseParams_->hiddenSize = context_->hidden_states.shape->GetStorageShape().GetDim(KEY_POOL_DIM_INDEX_1);
    }

    baseParams_->headDim = context_->wk.shape->GetStorageShape().GetDim(KEY_POOL_DIM_INDEX_0);
    baseParams_->cmpRatio = static_cast<uint32_t>(*context_->cmpRatio);
    baseParams_->csSize = baseParams_->seqSize - (baseParams_->seqSize % baseParams_->cmpRatio);
    baseParams_->normEps = context_->normEps == nullptr ? 1e-6f : *context_->normEps;
    baseParams_->stateCacheStrideDim0 = static_cast<uint64_t>(*context_->stateCacheStrideDim0);
    baseParams_->nSize = 2; // 2:每个核处理两个基本块后做全核同步
    baseParams_->usedCoreNum = aicNum_;
    baseParams_->batchConsistency = static_cast<uint32_t>(context_->batchConsistency);
    OP_LOGI(context_->opName, "[TILING] bSize:%u  tSize:%u cmpRatio:%u, stateCacheStrideDim0:%u",
            baseParams_->batchSize, baseParams_->tokenSize, baseParams_->cmpRatio, baseParams_->stateCacheStrideDim0);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::SetPageAttentionInfo()
{
    pageAttentionParams_->blockNum = context_->stateCache.shape->GetStorageShape().GetDim(KEY_POOL_DIM_INDEX_0);
    pageAttentionParams_->blockSize = context_->stateCache.shape->GetStorageShape().GetDim(KEY_POOL_DIM_INDEX_1);
    pageAttentionParams_->maxBlockNumPerBatch =
        context_->cacheBlockTable.shape->GetStorageShape().GetDim(KEY_POOL_DIM_INDEX_1);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::SetWorkSpaceInfo()
{
    workspaceParams_->dbWorkspaceRatio = 2;
    workspaceParams_->mm1KvResSize = innerSplitParams_->mBaseSize * baseParams_->headDim;
    workspaceParams_->mm1ScoreResSize = innerSplitParams_->mBaseSize * baseParams_->headDim;
    workspaceParams_->vec1ResSize = innerSplitParams_->mBaseSize * baseParams_->headDim * baseParams_->nSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::SetTemplateId()
{
    if (context_->templateId == TemplateId::EMPTY_HIDDEN_STATES) {
        return ge::GRAPH_SUCCESS;
    }
    if (socVersion_ == platform_ascendc::SocVersion::ASCEND950) {
        // 设置高性能模板
        if (context_->normWeight.desc == nullptr && context_->layout == LayoutType::LAYOUT_BSH &&
            baseParams_->seqSize <= 4 && baseParams_->tokenSize <= 256) {
            context_->templateId = TemplateId::FULL_LOAD;
        }
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::SetInnerSplitInfo()
{
    if (context_->templateId == TemplateId::FULL_LOAD) {
        innerSplitParams_->mBaseSize = 256; // 256:核间切分，M轴基本块大小
        innerSplitParams_->dBaseSize = 128; // nBase = dBase * 2
        uint32_t dBaseNum = baseParams_->headDim / innerSplitParams_->dBaseSize;
        uint32_t mBaseNum = (baseParams_->tokenSize + innerSplitParams_->mBaseSize - 1) / innerSplitParams_->mBaseSize;
        baseParams_->coreGroupNum = baseParams_->usedCoreNum / dBaseNum;
        baseParams_->kBaseNum = 1;
        baseParams_->kBaseSize = baseParams_->hiddenSize;
        if ((dBaseNum * mBaseNum) < baseParams_->usedCoreNum && baseParams_->batchConsistency != BATCH_CONSISTENCY) {
            baseParams_->kBaseNum = baseParams_->usedCoreNum / dBaseNum;
            uint32_t kAlignSize = (baseParams_->hiddenSize + baseParams_->kBaseNum - 1) / baseParams_->kBaseNum;
            baseParams_->kBaseSize = kAlignSize / 16 * 16; // 切k的size需要16对齐
        }
        for (uint32_t i = 0; i < baseParams_->usedCoreNum; i++) {
            baseParams_->splitCoreParam[i].nStart = (i % dBaseNum) * innerSplitParams_->dBaseSize;
            baseParams_->splitCoreParam[i].nEnd = baseParams_->splitCoreParam[i].nStart + innerSplitParams_->dBaseSize;
            if (baseParams_->kBaseNum > 1) {
                uint32_t kStartIdx = i / dBaseNum;
                if (kStartIdx + 1 < baseParams_->coreGroupNum) {
                    uint32_t dealKSize = baseParams_->kBaseSize;
                    baseParams_->splitCoreParam[i].kStart = kStartIdx * baseParams_->kBaseSize;
                    baseParams_->splitCoreParam[i].kEnd = baseParams_->splitCoreParam[i].kStart + dealKSize;
                } else {
                    uint32_t dealKSize = kStartIdx < baseParams_->coreGroupNum ?
                                             baseParams_->hiddenSize - kStartIdx * baseParams_->kBaseSize :
                                             0;
                    baseParams_->splitCoreParam[i].kStart = kStartIdx * baseParams_->kBaseSize;
                    baseParams_->splitCoreParam[i].kEnd = baseParams_->splitCoreParam[i].kStart + dealKSize;
                }
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
        innerSplitParams_->mBaseSize = 256; // 256:核间切分，M轴基本块大小
        innerSplitParams_->dBaseSize = 128; // 128：核间切分，D轴基本块大小
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CalcWorkSpace()
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
    if (context_->normWeight.desc != nullptr) {
        workspaceSize_ += workspaceParams_->mm1KvResSize * baseParams_->usedCoreNum * 2U * MM1_RES_ELEM_SIZE *
                          workspaceParams_->dbWorkspaceRatio;
    }

    if (context_->workSpaces) {
        context_->workSpaces[0] = workspaceSize_;
    }

    OP_LOGI(context_->opName, "Tiling info: workspaceSize_ = %zu", workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckEmptyTensor() const
{
    if ((context_->layout == LayoutType::LAYOUT_BSH &&
         context_->hidden_states.shape->GetStorageShape().GetDim(KEY_POOL_DIM_INDEX_0) == 0) ||
        (context_->layout == LayoutType::LAYOUT_BSH &&
         context_->hidden_states.shape->GetStorageShape().GetDim(KEY_POOL_DIM_INDEX_1) == 0) ||
        (context_->layout == LayoutType::LAYOUT_TH &&
         context_->hidden_states.shape->GetStorageShape().GetDim(KEY_POOL_DIM_INDEX_0) == 0)) {
        context_->templateId = TemplateId::EMPTY_HIDDEN_STATES;
    } else {
        if (context_->hidden_states.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->wk.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->gate_weight.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->stateCache.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->ape.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->cacheBlockTable.shape->GetStorageShape().GetShapeSize() == 0) {
            OP_LOGE(context_->opName, "Only input tensor hidden_states dim B or S or T supports to be 0");
            return ge::GRAPH_FAILED;
        }
        context_->templateId = TemplateId::NORMAL;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::RunBigKernelTiling(KeyPoolTilingData *tilingData)
{
    this->baseParams_ = &tilingData->baseParams;
    this->pageAttentionParams_ = &tilingData->pageAttentionParams;
    this->innerSplitParams_ = &tilingData->innerSplitParams;
    this->workspaceParams_ = &tilingData->workspaceParams;
    using StatusFunction = std::function<ge::graphStatus()>;
    std::vector<StatusFunction> requiredTilingFuncs{std::bind(&KeyPoolTiling::GetNpuInfo, this),
                                                    std::bind(&KeyPoolTiling::CheckRequiredParaExistence, this),
                                                    std::bind(&KeyPoolTiling::CheckEmptyTensor, this),
                                                    std::bind(&KeyPoolTiling::CheckSinglePara, this),
                                                    std::bind(&KeyPoolTiling::SetBaseInfo, this),
                                                    std::bind(&KeyPoolTiling::SetPageAttentionInfo, this),
                                                    std::bind(&KeyPoolTiling::CheckFeature, this),
                                                    std::bind(&KeyPoolTiling::CheckMultiParaConsistency, this),
                                                    std::bind(&KeyPoolTiling::CheckBlockDimConstrain, this),
                                                    std::bind(&KeyPoolTiling::SetTemplateId, this),
                                                    std::bind(&KeyPoolTiling::SetInnerSplitInfo, this),
                                                    std::bind(&KeyPoolTiling::SetWorkSpaceInfo, this)};
    for (const auto &func : requiredTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    if (context_->templateId == TemplateId::EMPTY_HIDDEN_STATES) {
        workspaceSize_ = libapiSize_;
        if (context_->workSpaces) {
            context_->workSpaces[0] = workspaceSize_;
        }
        GenTilingKey();
        context_->blockDim = 1U;
        return ge::GRAPH_SUCCESS;
    }
    std::vector<StatusFunction> optionalTilingFuncs{std::bind(&KeyPoolTiling::CalcWorkSpace, this),
                                                    std::bind(&KeyPoolTiling::GenTilingKey, this)};
    for (const auto &func : optionalTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    context_->blockDim = aicNum_;

    OP_LOGI("Run big kernel");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::GenTilingKey() const
{
    // 0:BF16, 1:FP16
    uint8_t dtype = 0;
    // 0: BSH 1:TH
    uint8_t layout = 0;
    uint8_t templateId = static_cast<uint8_t>(context_->templateId);
    auto hiddenStatesDtype = context_->hidden_states.desc->GetDataType();
    if (hiddenStatesDtype == ge::DT_BF16) {
        dtype = 0;
    } else if (hiddenStatesDtype == ge::DT_FLOAT16) {
        dtype = 1;
    }
    auto hiddenStatesDimNum = context_->hidden_states.shape->GetStorageShape().GetDimNum();
    if (hiddenStatesDimNum == KEY_POOL_DIM_NUM_3) {
        layout = 0;
    } else {
        layout = 1;
    }

    context_->tilingKey = GET_TPL_TILING_KEY(layout, dtype, templateId);
    OP_LOGI(context_->opName, "KeyPool dtype:%hhu layout:%hhu template_id:%hhu", dtype, layout, templateId);
    OP_LOGI(context_->opName, "KeyPool tilingKey:%lu", context_->tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaHiddenStates() || ge::GRAPH_SUCCESS != CheckSingleParaWk() ||
        ge::GRAPH_SUCCESS != CheckSingleParaGateWeight() || ge::GRAPH_SUCCESS != CheckSingleParaStateCache() ||
        ge::GRAPH_SUCCESS != CheckSingleParaApe() || ge::GRAPH_SUCCESS != CheckSingleParaNormWeight() ||
        ge::GRAPH_SUCCESS != CheckSingleParaNormBias() || ge::GRAPH_SUCCESS != CheckSingleParaCacheBlockTable() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSeqLens() || ge::GRAPH_SUCCESS != CheckSingleParaSeqUsed() ||
        ge::GRAPH_SUCCESS != CheckSingleParaStartPos() || ge::GRAPH_SUCCESS != CheckSingleParaPooledKey() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpRatio()) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(context_->normEps != nullptr && *context_->normEps <= 0.0f,
                OP_LOGE(context_->opName, "norm_eps must be greater than 0, but got %f", *context_->normEps),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus KeyPoolTiling::CheckFeatureValueSupport(const T *featureValue,
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
ge::graphStatus KeyPoolTiling::CheckAttrValueSupport(const T *attrValue, const std::vector<T> &expectAttrValList,
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
void KeyPoolTiling::LogErrorNumberSupport(const std::vector<T> &expectNumberList, const T &actualValue,
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

ge::graphStatus KeyPoolTiling::CheckDimNumInLayoutSupport(const std::string &layout, const gert::StorageShape *shape,
                                                          const std::string &name) const
{
    const auto &dimIt = LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
                OP_LOGE(context_->opName, "When layout is %s, %s dimension should be %u, but it's %zu", layout.c_str(),
                        name.c_str(), dimIt->second, shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc, const std::string &name) const
{
    if (desc != nullptr) {
        const auto &it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(
            it == DTYPE_SUPPORT_MAP.end(),
            OP_LOGE(context_->opName, "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
            return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(
            std::find(expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end(),
            LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void KeyPoolTiling::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
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
        OP_LOGE("KeyPool", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

ge::graphStatus KeyPoolTiling::CheckDimNumSupport(const gert::StorageShape *shape, const std::string &name) const
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

ge::graphStatus KeyPoolTiling::CheckSingleParaHiddenStates() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->hidden_states.desc, HIDDEN_STATES_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->hidden_states.shape, HIDDEN_STATES_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(LayoutTypeToStr(context_->layout),
                                                        context_->hidden_states.shape, HIDDEN_STATES_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckSingleParaWk() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->wk.desc, WK_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->wk.shape, WK_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckSingleParaGateWeight() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->gate_weight.desc, GATE_WEIGHT_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->gate_weight.shape, GATE_WEIGHT_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckSingleParaStateCache() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->stateCache.desc, STATE_CACHE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->stateCache.shape, STATE_CACHE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckSingleParaApe() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->ape.desc, APE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->ape.shape, APE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckSingleParaCacheBlockTable() const
{
    if (context_->cacheBlockTable.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->cacheBlockTable.desc, CACHE_BLOCK_TABLE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->cacheBlockTable.shape, CACHE_BLOCK_TABLE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckSingleParaSeqLens() const
{
    if (context_->seqLens.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->seqLens.desc, CU_SEQLENS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->seqLens.shape, CU_SEQLENS_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckSingleParaSeqUsed() const
{
    OP_CHECK_IF(context_->seqUsed.desc != nullptr || context_->seqUsed.shape != nullptr,
                OP_LOGE(context_->opName, "seqused is reserved and must be nullptr in this stage"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckSingleParaStartPos() const
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

ge::graphStatus KeyPoolTiling::CheckSingleParaNormWeight() const
{
    bool hasWeight = context_->normWeight.desc != nullptr;
    bool hasBias = context_->normBias.desc != nullptr;
    OP_CHECK_IF(hasWeight != hasBias, OP_LOGE(context_->opName, "norm_weight and norm_bias must be provided together"),
                return ge::GRAPH_FAILED);
    if (!hasWeight) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->normWeight.desc, NORM_WEIGHT_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->normWeight.shape, NORM_WEIGHT_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckSingleParaNormBias() const
{
    if (context_->normBias.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->normBias.desc, NORM_BIAS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->normBias.shape, NORM_BIAS_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckSingleParaPooledKey() const
{
    if (context_->pooledKey.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->pooledKey.desc, POOLED_KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->pooledKey.shape, POOLED_KEY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckSingleParaCmpRatio() const
{
    uint32_t cmpRatio = static_cast<uint32_t>(*context_->cmpRatio);
    OP_CHECK_IF(cmpRatio > MAX_CMPRATIO_SIZE || cmpRatio < MIN_CMPRATIO_SIZE,
                OP_LOGE(context_->opName, "cmpRatio should be within [%u, %u], but got %u", MIN_CMPRATIO_SIZE,
                        MAX_CMPRATIO_SIZE, cmpRatio),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(context_->hidden_states.shape == nullptr, OP_LOGE(context_->opName, "tensor hidden_states is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->hidden_states.desc == nullptr, OP_LOGE(context_->opName, "tensor hidden_states is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wk.shape == nullptr, OP_LOGE(context_->opName, "tensor wk is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wk.desc == nullptr, OP_LOGE(context_->opName, "tensor wk is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->gate_weight.shape == nullptr, OP_LOGE(context_->opName, "tensor gate_weight is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->gate_weight.desc == nullptr, OP_LOGE(context_->opName, "tensor gate_weight is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->stateCache.shape == nullptr, OP_LOGE(context_->opName, "tensor stateCache is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->stateCache.desc == nullptr, OP_LOGE(context_->opName, "tensor stateCache is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ape.shape == nullptr, OP_LOGE(context_->opName, "tensor ape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ape.desc == nullptr, OP_LOGE(context_->opName, "tensor ape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->cacheBlockTable.shape == nullptr,
                OP_LOGE(context_->opName, "tensor cacheBlockTable is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->cacheBlockTable.desc == nullptr,
                OP_LOGE(context_->opName, "tensor cacheBlockTable is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->startPos.shape == nullptr, OP_LOGE(context_->opName, "tensor startPos is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->startPos.desc == nullptr, OP_LOGE(context_->opName, "tensor startPos is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->pooledKey.shape == nullptr, OP_LOGE(context_->opName, "tensor pooledKey is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->pooledKey.desc == nullptr, OP_LOGE(context_->opName, "tensor pooledKey is nullptr"),
                return ge::GRAPH_FAILED);
    if (context_->layout == LayoutType::LAYOUT_TH) {
        OP_CHECK_IF(context_->seqLens.desc == nullptr,
                    OP_LOGE(context_->opName, "In TH layout, tensor seqLens should not be nullptr"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->seqLens.shape == nullptr,
                    OP_LOGE(context_->opName, "In TH layout, tensor seqLens should not be nullptr"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(context_->seqLens.desc != nullptr,
                    OP_LOGE(context_->opName, "In BSH layout, tensor seqLens must be nullptr"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->seqLens.shape != nullptr,
                    OP_LOGE(context_->opName, "In BSH layout, tensor seqLens must be nullptr"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(context_->cmpRatio == nullptr, OP_LOGE(context_->opName, "attr cmpRatio is nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckFeature() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureValueSupport(&baseParams_->headDim, HEAD_DIM, "headDim")) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(baseParams_->hiddenSize > MAX_HIDDEN_SIZE || baseParams_->hiddenSize < MIN_HIDDEN_SIZE ||
                    baseParams_->hiddenSize % ALIGN_FACTOR_HIDDEN_SIZE != 0,
                OP_LOGE(context_->opName, "hiddenSize should be within [%u, %u] and be 512-aligned, but got %u",
                        MIN_HIDDEN_SIZE, MAX_HIDDEN_SIZE, baseParams_->hiddenSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(pageAttentionParams_->blockSize > MAX_BLOCK_SIZE || pageAttentionParams_->blockSize < MIN_BLOCK_SIZE,
                OP_LOGE(context_->opName, "blockSize should be within [%u, %u], but got %u", MIN_BLOCK_SIZE,
                        MAX_BLOCK_SIZE, pageAttentionParams_->blockSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->cacheBlockTable.shape->GetStorageShape().GetDimNum() != KEY_POOL_DIM_NUM_2,
                OP_LOGE(context_->opName, "cacheBlockTable dimension should be 2 for linear cache"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::LogErrorShapeConsistency(const std::string &name, const gert::StorageShape *shape,
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

ge::graphStatus KeyPoolTiling::CheckShapeConsistency() const
{
    uint32_t stateNum = 2;
    if (ge::GRAPH_SUCCESS != LogErrorShapeConsistency("cacheBlockTable", context_->cacheBlockTable.shape,
                                                      KEY_POOL_DIM_INDEX_0, "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("seqLens", context_->seqLens.shape, KEY_POOL_DIM_INDEX_0,
                                                      "batchSize+1", baseParams_->batchSize + 1) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("startPos", context_->startPos.shape, KEY_POOL_DIM_INDEX_0,
                                                      "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("wk", context_->wk.shape, KEY_POOL_DIM_INDEX_1, "hiddenSize",
                                                      baseParams_->hiddenSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("gate_weight", context_->gate_weight.shape, KEY_POOL_DIM_INDEX_1,
                                                      "hiddenSize", baseParams_->hiddenSize) ||
        ge::GRAPH_SUCCESS !=
            LogErrorShapeConsistency("wk", context_->wk.shape, KEY_POOL_DIM_INDEX_0, "headDim", baseParams_->headDim) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("gate_weight", context_->gate_weight.shape, KEY_POOL_DIM_INDEX_0,
                                                      "headDim", baseParams_->headDim) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("stateCache", context_->stateCache.shape, KEY_POOL_DIM_INDEX_2,
                                                      "2*headDim", stateNum * baseParams_->headDim) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ape", context_->ape.shape, KEY_POOL_DIM_INDEX_1, "headDim",
                                                      baseParams_->headDim) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ape", context_->ape.shape, KEY_POOL_DIM_INDEX_0, "cmpRatio",
                                                      baseParams_->cmpRatio)) {
        return ge::GRAPH_FAILED;
    }
    if (context_->normWeight.shape != nullptr &&
        (ge::GRAPH_SUCCESS != LogErrorShapeConsistency("norm_weight", context_->normWeight.shape, KEY_POOL_DIM_INDEX_0,
                                                       "headDim", baseParams_->headDim) ||
         ge::GRAPH_SUCCESS != LogErrorShapeConsistency("norm_bias", context_->normBias.shape, KEY_POOL_DIM_INDEX_0,
                                                       "headDim", baseParams_->headDim))) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != LogErrorShapeConsistency("stateCache", context_->stateCache.shape, KEY_POOL_DIM_INDEX_0,
                                                      "blockNum", pageAttentionParams_->blockNum) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("stateCache", context_->stateCache.shape, KEY_POOL_DIM_INDEX_1,
                                                      "blockSize", pageAttentionParams_->blockSize)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckDtypeConsistencyHiddenStates(const gert::CompileTimeTensorDesc *desc,
                                                                 const std::string &name) const
{
    const auto actualDtype = desc->GetDataType();
    OP_CHECK_IF(actualDtype != context_->dtype,
                OP_LOGE(context_->opName, "%s datatype should be same with hidden_states: %s, but got %s", name.c_str(),
                        DataTypeToSerialString(actualDtype).c_str(), DataTypeToSerialString(context_->dtype).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckDtypeConsistency() const
{
    if (CheckDtypeConsistencyHiddenStates(context_->wk.desc, WK_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyHiddenStates(context_->gate_weight.desc, GATE_WEIGHT_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyHiddenStates(context_->pooledKey.desc, POOLED_KEY_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckDimNumConsistency() const
{
    const auto &outputShape = context_->pooledKey.shape->GetStorageShape();
    const bool isBsh = context_->layout == LayoutType::LAYOUT_BSH;
    const uint32_t rank = isBsh ? KEY_POOL_DIM_NUM_3 : KEY_POOL_DIM_NUM_2;
    OP_CHECK_IF(outputShape.GetDimNum() != rank,
                OP_LOGE(context_->opName, "pooledKey rank must match hidden_states rank"), return ge::GRAPH_FAILED);
    const uint32_t capacity = isBsh ? (baseParams_->seqSize + baseParams_->cmpRatio - 1) / baseParams_->cmpRatio :
                                      std::min(baseParams_->tokenSize,
                                               baseParams_->tokenSize / baseParams_->cmpRatio + baseParams_->batchSize);
    OP_CHECK_IF(outputShape.GetDim(rank - 1) != baseParams_->headDim || outputShape.GetDim(rank - 2) != capacity ||
                    (isBsh && outputShape.GetDim(0) != baseParams_->batchSize),
                OP_LOGE(context_->opName, "pooledKey shape must use current-input capacity"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckBlockDimConstrain() const
{
    uint32_t minBlockNum = baseParams_->headDim / 64; // 64 is the largest dBaseSize
    OP_CHECK_IF(aicNum_ < minBlockNum,
                OP_LOGE(context_->opName, "aicNum is %u, which should not be less than %u", aicNum_, minBlockNum),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus KeyPoolTiling::CheckMultiParaConsistency() const
{
    if (CheckShapeConsistency() != ge::GRAPH_SUCCESS || CheckDtypeConsistency() != ge::GRAPH_SUCCESS ||
        CheckDimNumConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

CMP_EXTERN_C ge::graphStatus TilingKeyPool(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("KeyPool", "Context is nullptr."),
                return ge::GRAPH_FAILED);

    OP_LOGI("Getting Tiling");

    KeyPoolContext key_poolContext{};
    if (KeyPoolTiling::ConvertContext(*context, key_poolContext) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Error occurred while converting tilingContext to KeyPool context");
        return ge::GRAPH_FAILED;
    }
    KeyPoolTiling key_poolTiling(&key_poolContext);
    KeyPoolTilingData *tilingData = context->GetTilingData<KeyPoolTilingData>();
    OP_CHECK_IF(tilingData == nullptr, OPS_REPORT_VECTOR_INNER_ERR(key_poolContext.opName, "TilingData is nullptr."),
                return ge::GRAPH_FAILED);
    // 使用SyncAll，需要设置为batchmode模式，所有核同时启动，否则多流方式下执行可能会卡死
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    if (key_poolTiling.RunBigKernelTiling(tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(key_poolContext.tilingKey);
    context->SetBlockDim(key_poolContext.blockDim);
    OP_LOGI(key_poolContext.opName, "block dim: %u.", key_poolContext.blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForKeyPool(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(KeyPool).Tiling(TilingKeyPool).TilingParse<KeyPoolCompileInfo>(TilingPrepareForKeyPool);
} // namespace optiling
