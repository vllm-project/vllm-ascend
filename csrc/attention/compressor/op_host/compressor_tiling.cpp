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

static const std::string X_NAME = "query";
static const std::string WKV_NAME = "wkv";
static const std::string WGATE_NAME = "wgate";
static const std::string KV_STATE_NAME = "kv_state";
static const std::string SCORE_STATE_NAME = "score_state";
static const std::string APE_NAME = "ape";
static const std::string NORM_WEIGHT_NAME = "norm_weight";
static const std::string ROPE_SIN_NAME = "rope_sin";
static const std::string ROPE_COS_NAME = "rope_cos";
static const std::string KV_BLOCK_TABLE_NAME = "kv_block_table";
static const std::string SCORE_BLOCK_TABLE_NAME = "score_block_table";
static const std::string CU_SEQLENS_NAME = "cu_seqlens";
static const std::string SEQUSED_NAME = "seq_used";
static const std::string START_POS_NAME = "start_pos";
static const std::string ROPE_HEAD_DIM_NAME = "rope_head_dim";
static const std::string CMP_RATIO_NAME = "cmp_ratio";
static const std::string COFF_NAME = "coff";
static const std::string NORM_EPS_NAME = "nrom_eps";
static const std::string ROTARY_MODE_NAME = "rotary_mode";
static const std::string CMP_KV_NAME = "cmp_kv";
static std::string DataTypeToSerialString(ge::DataType type);
const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {X_NAME,                {ge::DT_BF16, ge::DT_FLOAT16}},
    {WKV_NAME,              {ge::DT_BF16, ge::DT_FLOAT16}},
    {WGATE_NAME,            {ge::DT_BF16, ge::DT_FLOAT16}},
    {KV_STATE_NAME,         {ge::DT_FLOAT}},
    {SCORE_STATE_NAME,      {ge::DT_FLOAT}},
    {APE_NAME,              {ge::DT_FLOAT}},
    {NORM_WEIGHT_NAME,      {ge::DT_BF16, ge::DT_FLOAT16}},
    {ROPE_SIN_NAME,         {ge::DT_BF16, ge::DT_FLOAT16}},
    {ROPE_COS_NAME,         {ge::DT_BF16, ge::DT_FLOAT16}},
    {KV_BLOCK_TABLE_NAME,   {ge::DT_INT32}},
    {SCORE_BLOCK_TABLE_NAME, {ge::DT_INT32}},
    {CU_SEQLENS_NAME,       {ge::DT_INT32}},
    {SEQUSED_NAME,          {ge::DT_INT32}},
    {START_POS_NAME,       {ge::DT_INT32}},
    {CMP_KV_NAME,           {ge::DT_BF16, ge::DT_FLOAT16}}
};
const std::map<std::string, std::vector<uint32_t>> DIM_NUM_MAP = {
    {X_NAME,                {COMPRESSOR_DIM_NUM_2, COMPRESSOR_DIM_NUM_3}},
    {WKV_NAME,              {COMPRESSOR_DIM_NUM_2}},
    {WGATE_NAME,            {COMPRESSOR_DIM_NUM_2}},
    {KV_STATE_NAME,         {COMPRESSOR_DIM_NUM_3}},
    {SCORE_STATE_NAME,      {COMPRESSOR_DIM_NUM_3}},
    {APE_NAME,              {COMPRESSOR_DIM_NUM_2}},
    {NORM_WEIGHT_NAME,      {COMPRESSOR_DIM_NUM_1}},
    {ROPE_SIN_NAME,         {COMPRESSOR_DIM_NUM_2, COMPRESSOR_DIM_NUM_3}},
    {ROPE_COS_NAME,         {COMPRESSOR_DIM_NUM_2, COMPRESSOR_DIM_NUM_3}},
    {KV_BLOCK_TABLE_NAME,   {COMPRESSOR_DIM_NUM_2}},
    {SCORE_BLOCK_TABLE_NAME, {COMPRESSOR_DIM_NUM_2}},
    {CU_SEQLENS_NAME,       {COMPRESSOR_DIM_NUM_1}},
    {SEQUSED_NAME,          {COMPRESSOR_DIM_NUM_1}},
    {START_POS_NAME,       {COMPRESSOR_DIM_NUM_1}},
    {CMP_KV_NAME,           {COMPRESSOR_DIM_NUM_2, COMPRESSOR_DIM_NUM_3}}
};
static const std::map<std::string, uint32_t> LAYOUT_DIM_MAP = {
    {"BSH", COMPRESSOR_DIM_NUM_3},
    {"TH", COMPRESSOR_DIM_NUM_2},
};
const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},           // Used to indicate a DataType field has not been set.
    {ge::DT_FLOAT, "DT_FLOAT"},                   // float type
    {ge::DT_FLOAT16, "DT_FLOAT16"},               // fp16 type
    {ge::DT_INT8, "DT_INT8"},                     // int8 type
    {ge::DT_INT16, "DT_INT16"},                   // int16 type
    {ge::DT_UINT16, "DT_UINT16"},                 // uint16 type
    {ge::DT_UINT8, "DT_UINT8"},                   // uint8 type
    {ge::DT_INT32, "DT_INT32"},                   // uint32 type
    {ge::DT_INT64, "DT_INT64"},                   // int64 type
    {ge::DT_UINT32, "DT_UINT32"},                 // unsigned int32
    {ge::DT_UINT64, "DT_UINT64"},                 // unsigned int64
    {ge::DT_BOOL, "DT_BOOL"},                     // bool type
    {ge::DT_DOUBLE, "DT_DOUBLE"},                 // double type
    {ge::DT_DUAL, "DT_DUAL"},                     // dual output type
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},   // dual output int8 type
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"}, // dual output uint8 type
    {ge::DT_COMPLEX32, "DT_COMPLEX32"},           // complex32 type
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},           // complex64 type
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},         // complex128 type
    {ge::DT_QINT8, "DT_QINT8"},                   // qint8 type
    {ge::DT_QINT16, "DT_QINT16"},                 // qint16 type
    {ge::DT_QINT32, "DT_QINT32"},                 // qint32 type
    {ge::DT_QUINT8, "DT_QUINT8"},                 // quint8 type
    {ge::DT_QUINT16, "DT_QUINT16"},               // quint16 type
    {ge::DT_RESOURCE, "DT_RESOURCE"},             // resource type
    {ge::DT_STRING_REF, "DT_STRING_REF"},         // string ref type
    {ge::DT_STRING, "DT_STRING"},                 // string type
    {ge::DT_VARIANT, "DT_VARIANT"},               // dt_variant type
    {ge::DT_BF16, "DT_BFLOAT16"},                 // dt_bfloat16 type
    {ge::DT_INT4, "DT_INT4"},                     // dt_variant type
    {ge::DT_UINT1, "DT_UINT1"},                   // dt_variant type
    {ge::DT_INT2, "DT_INT2"},                     // dt_variant type
    {ge::DT_UINT2, "DT_UINT2"}                    // dt_variant type
};

void CompressorTiling::ConvertRequiredParams(gert::TilingContext &context, CompressorContext &compressorContext)
{
    compressorContext.x.desc = context.GetRequiredInputDesc(TOKEN_X_INPUT_INDEX);
    compressorContext.x.shape = context.GetRequiredInputShape(TOKEN_X_INPUT_INDEX);
    compressorContext.wkv.desc = context.GetRequiredInputDesc(WEIGHT_KV_INPUT_INDEX);
    compressorContext.wkv.shape = context.GetRequiredInputShape(WEIGHT_KV_INPUT_INDEX);
    compressorContext.wgate.desc = context.GetRequiredInputDesc(WEIGHT_WGATE_INPUT_INDEX);
    compressorContext.wgate.shape = context.GetRequiredInputShape(WEIGHT_WGATE_INPUT_INDEX);
    compressorContext.kvState.desc = context.GetRequiredInputDesc(KV_STATE_INPUT_INDEX);
    compressorContext.kvState.shape = context.GetRequiredInputShape(KV_STATE_INPUT_INDEX);
    compressorContext.scoreState.desc = context.GetRequiredInputDesc(SCORE_STATE_INPUT_INDEX);
    compressorContext.scoreState.shape = context.GetRequiredInputShape(SCORE_STATE_INPUT_INDEX);
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
    compressorContext.kvBlockTable.desc = context.GetOptionalInputDesc(KV_BLOCK_TABLE_INPUT_INDEX);
    compressorContext.kvBlockTable.shape = context.GetOptionalInputShape(KV_BLOCK_TABLE_INPUT_INDEX);
    compressorContext.scoreBlockTable.desc = context.GetOptionalInputDesc(SCORE_BLOCK_TABLE_INPUT_INDEX);
    compressorContext.scoreBlockTable.shape = context.GetOptionalInputShape(SCORE_BLOCK_TABLE_INPUT_INDEX);
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
    socVersion_ = ascendcPlatform.GetSocVersion();

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
    baseParams_->csSize = baseParams_->seqSize - (baseParams_->seqSize %  baseParams_->cmpRatio);
    baseParams_->ropeHeadDim = static_cast<uint32_t>(*context_->ropeHeadDim);
    baseParams_->normEps = static_cast<float>(*context_->normEps);
    baseParams_->reciprocalD = 1.0 / baseParams_->headDim;
    baseParams_->cgSize = (baseParams_->seqSize + baseParams_->cmpRatio - 1) / baseParams_->cmpRatio; // number of token after compress
    coff = static_cast<uint8_t>(*context_->coff);
    baseParams_->nSize = 2; // 2:每个核处理两个基本块后做全核同步

    OP_LOGI(context_->opName, "[TILING] bSize:%u  tSize:%u cmpRatio:%u coff:%u", baseParams_->batchSize, baseParams_->tokenSize, baseParams_->cmpRatio, coff);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::SetPageAttentionInfo()
{
    pageAttentionParams_->blockNum = context_->kvState.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0);
    pageAttentionParams_->blockSize = context_->kvState.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1);
    pageAttentionParams_->maxBlockNumPerBatch = context_->kvBlockTable.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::SetWorkSpaceInfo()
{
    workspaceParams_->preMm1ResSize = 0;
    if (coff == 2) { // 2:需要做overlap
        workspaceParams_->preMm1ResSize = innerSplitParams_->mBaseSize * innerSplitParams_->dBaseSize * 2;      // 2 wkv和score合一起
    }
    workspaceParams_->curMm1ResSize = innerSplitParams_->mBaseSize * innerSplitParams_->dBaseSize * 2;          // 2 wkv和score合一起
    if (context_->templateId == TemplateId::PERF) {
        workspaceParams_->vec1ResSize = innerSplitParams_->mBaseSize * innerSplitParams_->dBaseSize * baseParams_->nSize;
    } else {
        workspaceParams_->vec1ResSize = innerSplitParams_->mBaseSize / baseParams_->cmpRatio * innerSplitParams_->dBaseSize * baseParams_->nSize;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::SetScenarioInfo()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::SetTemplateId()
{
    if (socVersion_ == platform_ascendc::SocVersion::ASCEND910_95) {
        return ge::GRAPH_SUCCESS;
    }
    if (context_->seqUsed.desc != nullptr || context_->seqUsed.shape != nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (context_->layout == LayoutType::LAYOUT_BSH) {
        return ge::GRAPH_SUCCESS;
    }
    // 设置高性能模板
    context_->templateId = TemplateId::PERF;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::SetInnerSplitInfo()
{
    innerSplitParams_->mBaseSize = 256; // 256:核间切分，M轴基本块大小
    innerSplitParams_->dBaseSize = 128 / coff; // 128：核间切分，D轴基本块大小
    if (context_->templateId == TemplateId::PERF) {
        if (coff == 2) {
            innerSplitParams_->mBaseSize = 128;
        } else {
            innerSplitParams_->mBaseSize = 256;
        }
        innerSplitParams_->dBaseSize = 64;
    } else {
        innerSplitParams_->mBaseSize = 256; // 256:核间切分，M轴基本块大小
        innerSplitParams_->dBaseSize = 128 / coff; // 128：核间切分，D轴基本块大小
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CalcWorkSpace()
{
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;      // 4: fp32
    constexpr uint32_t V1_RES_ELEM_SIZE = 2;       // 2: fp16/bf16
    workspaceSize_ = libapiSize_;
    workspaceSize_ += aicNum_ * workspaceParams_->preMm1ResSize * MM1_RES_ELEM_SIZE;
    workspaceSize_ += aicNum_ * workspaceParams_->curMm1ResSize * MM1_RES_ELEM_SIZE;
    workspaceSize_ += aicNum_ * workspaceParams_->vec1ResSize * V1_RES_ELEM_SIZE;
    
    workspaceSize_ += 1024 * 1024 * 1024; // 1024:申请workspace大小
    if (context_->workSpaces) {
        context_->workSpaces[0] = workspaceSize_;
    }
    
    OP_LOGI(context_->opName, "Tiling info: workspaceSize_ = %zu", workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckEmptyTensor() const
{
    if (context_->layout == LayoutType::LAYOUT_BSH && context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_1) == 0 ||
        context_->layout == LayoutType::LAYOUT_TH && context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_DIM_INDEX_0) == 0) {
        context_->templateId = TemplateId::EMPTY_X;
    } else {
        if (context_->x.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->wkv.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->wgate.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->kvState.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->scoreState.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->ape.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->normWeight.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->ropeSin.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->ropeCos.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->kvBlockTable.shape->GetStorageShape().GetShapeSize() == 0 ||
            context_->scoreBlockTable.shape->GetStorageShape().GetShapeSize() == 0) {
            return ge::GRAPH_FAILED;
        }
        context_->templateId = TemplateId::NORMAL;
        OP_LOGI(context_->opName, "Only input tensor x supports empty state");
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::RunBigKernelTiling(CompressorTilingData* tilingData)
{
    this->baseParams_ = &tilingData->baseParams;
    this->pageAttentionParams_ = &tilingData->pageAttentionParams;
    this->innerSplitParams_ = &tilingData->innerSplitParams;
    this->workspaceParams_ = &tilingData->workspaceParams;
    using StatusFunction = std::function<ge::graphStatus()>;
    std::vector<StatusFunction> requiredTilingFuncs {
        std::bind(&CompressorTiling::CheckRequiredParaExistence, this),
        std::bind(&CompressorTiling::CheckEmptyTensor, this),
        std::bind(&CompressorTiling::CheckSinglePara, this),
        std::bind(&CompressorTiling::GetNpuInfo, this),
        std::bind(&CompressorTiling::SetBaseInfo, this),
        std::bind(&CompressorTiling::SetPageAttentionInfo, this),
        std::bind(&CompressorTiling::CheckFeature, this),
        std::bind(&CompressorTiling::CheckMultiParaConsistency, this),
        std::bind(&CompressorTiling::SetTemplateId, this),
        std::bind(&CompressorTiling::SetInnerSplitInfo, this),
        std::bind(&CompressorTiling::SetWorkSpaceInfo, this),
        std::bind(&CompressorTiling::SetScenarioInfo, this)
    };
    for (const auto &func: requiredTilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        if (context_->templateId == TemplateId::EMPTY_X) {
            GenTilingKey();
            context_->blockDim = 1U;
            return ge::GRAPH_SUCCESS;
        }
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

    baseParams_->usedCoreNum = aicNum_;

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
    
    auto xDtype = context_->x.desc->GetDataType();
    if (xDtype == ge::DT_BF16) {
        dtype = 0;
    } else if (xDtype == ge::DT_FLOAT16) {
        dtype = 1;
    }
    auto xDimNum = context_->x.shape->GetStorageShape().GetDimNum();
    if (xDimNum == COMPRESSOR_DIM_NUM_3) {
        layout = 0;
    }else {
        layout = 1;
    }
    
    context_->tilingKey = GET_TPL_TILING_KEY(
        layout,
        dtype,
        coff,
        rotaryMode,
        templateId
    );

    OP_LOGI(context_->opName, "Compressor dtype:%hhu layout:%hhu  coff:%hhu rotary_mode:%hhu, template_id:%hhu", dtype, layout, coff, rotaryMode, templateId);
    OP_LOGI(context_->opName, "Compressor tilingKey:%lu", context_->tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaX() ||
        ge::GRAPH_SUCCESS != CheckSingleParaWkv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaWgate() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKvState() ||
        ge::GRAPH_SUCCESS != CheckSingleParaScoreState() ||
        ge::GRAPH_SUCCESS != CheckSingleParaApe() ||
        ge::GRAPH_SUCCESS != CheckSingleParaNormWeight() ||
        ge::GRAPH_SUCCESS != CheckSingleParaRopeSin() ||
        ge::GRAPH_SUCCESS != CheckSingleParaRopeCos() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKvBlockTable() ||
        ge::GRAPH_SUCCESS != CheckSingleParaScoreBlockTable() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCuSeqlens() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSeqused() ||
        ge::GRAPH_SUCCESS != CheckSingleParaStartPos() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpKv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaRopeHeadDim() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpRatio() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCoff() ||
        ge::GRAPH_SUCCESS != CheckSingleParaNormEps() ||
        ge::GRAPH_SUCCESS != CheckSingleParaRotaryMode()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus CompressorTiling::CheckFeatureValueSupport(const T *featureValue,
    const std::vector<T> &expectFeatureValList, const std::string &name) const
{
    if (std::find(expectFeatureValList.begin(), expectFeatureValList.end(), *featureValue) == expectFeatureValList.end()) {
        LogErrorNumberSupport(expectFeatureValList, *featureValue, name, "feature value");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus CompressorTiling::CheckAttrValueSupport(const T *attrValue,
    const std::vector<T> &expectAttrValList, const std::string &name) const
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
void CompressorTiling::LogErrorNumberSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name, const std::string subName) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << std::to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }

    OP_LOGE("Compressor", "%s %s only supports %s, but got %s",
              name.c_str(), subName.c_str(), oss.str().c_str(), std::to_string(actualValue).c_str());
}

std::string LayoutTypeToStr(LayoutType layout) {
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
    const auto& dimIt = LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
        OP_LOGE("Compressor", "When layout is %s, %s dimension should be %zu, but it's %zu",
            layout.c_str(), name.c_str(), dimIt->second,
            shape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
                                                   const std::string &name) const
{
    if (desc != nullptr) {
        const auto &it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(it == DTYPE_SUPPORT_MAP.end(),
                    OP_LOGE("Compressor", "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
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
    OP_LOGE("Compressor", "Tensor %s only supports dtype %s, but got %s", name.c_str(), oss.str().c_str(),
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
                OP_LOGE("Compressor", "%s dim number support list should be specify in DIM_NUM_MAP", name.c_str()),
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

ge::graphStatus CompressorTiling::CheckSingleParaKvState() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->kvState.desc, KV_STATE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->kvState.shape, KV_STATE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaScoreState() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->scoreState.desc, SCORE_STATE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->scoreState.shape, SCORE_STATE_NAME)) {
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
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(LayoutTypeToStr(context_->layout), context_->ropeSin.shape, ROPE_SIN_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaRopeCos() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->ropeCos.desc, ROPE_COS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->ropeCos.shape, ROPE_COS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(LayoutTypeToStr(context_->layout), context_->ropeCos.shape, ROPE_COS_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaKvBlockTable() const
{
    if (context_->kvBlockTable.desc == nullptr){
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->kvBlockTable.desc, KV_BLOCK_TABLE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->kvBlockTable.shape, KV_BLOCK_TABLE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaScoreBlockTable() const
{
    if (context_->scoreBlockTable.desc == nullptr){
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->scoreBlockTable.desc, SCORE_BLOCK_TABLE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->scoreBlockTable.shape, SCORE_BLOCK_TABLE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaCuSeqlens() const
{
    if (context_->cuSeqlens.desc == nullptr){
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
    if (context_->seqUsed.desc == nullptr){
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
    if (context_->startPos.desc == nullptr){
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
    if (context_->cmpKv.desc == nullptr){
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->cmpKv.desc, CMP_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->cmpKv.shape, CMP_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaRopeHeadDim()const
{
    if (CheckAttrValueSupport(context_->ropeHeadDim, ROPE_HEAD_DIM, ROPE_HEAD_DIM_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaCmpRatio()const
{
    if (CheckAttrValueSupport(context_->cmpRatio, CMP_RATIO, CMP_RATIO_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaCoff()const
{
    if (CheckAttrValueSupport(context_->coff, COFF, COFF_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaNormEps()const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckSingleParaRotaryMode()const
{
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupport(context_->rotaryMode, ROTARY_MODE, ROTARY_MODE_NAME)) {
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
    OP_CHECK_IF(context_->x.shape == nullptr, OP_LOGE("Compressor", "Shape of tensor x is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->x.desc == nullptr, OP_LOGE("Compressor", "Desc of tensor x is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wkv.shape == nullptr, OP_LOGE("Compressor", "Shape of tensor wkv is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wkv.desc == nullptr, OP_LOGE("Compressor", "Desc of tensor wkv is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wgate.shape == nullptr, OP_LOGE("Compressor", "Shape of tensor wgate is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->wgate.desc == nullptr, OP_LOGE("Compressor", "Desc of tensor wgate is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->kvState.shape == nullptr, OP_LOGE("Compressor", "Shape of tensor kvState is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->kvState.desc == nullptr, OP_LOGE("Compressor", "Desc of tensor kvState is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->scoreState.shape == nullptr, OP_LOGE("Compressor", "Shape of tensor scoreState is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->scoreState.desc == nullptr, OP_LOGE("Compressor", "Desc of tensor scoreState is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ape.shape == nullptr, OP_LOGE("Compressor", "Shape of tensor ape is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ape.desc == nullptr, OP_LOGE("Compressor", "Desc of tensor ape is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->normWeight.shape == nullptr, OP_LOGE("Compressor", "Shape of tensor normWeight is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->normWeight.desc == nullptr, OP_LOGE("Compressor", "Desc of tensor normWeight is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ropeSin.shape == nullptr, OP_LOGE("Compressor", "Shape of tensor ropeSin is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ropeSin.desc == nullptr, OP_LOGE("Compressor", "Desc of tensor ropeSin is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ropeCos.shape == nullptr, OP_LOGE("Compressor", "Shape of tensor ropeCos is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ropeCos.desc == nullptr, OP_LOGE("Compressor", "Desc of tensor ropeCos is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->kvBlockTable.shape == nullptr, OP_LOGE("Compressor", "Shape of tensor kvBlockTable is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->kvBlockTable.desc == nullptr, OP_LOGE("Compressor", "Desc of tensor kvBlockTable is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->scoreBlockTable.shape == nullptr, OP_LOGE("Compressor", "Shape of tensor scoreBlockTable is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->scoreBlockTable.desc == nullptr, OP_LOGE("Compressor", "Desc of tensor scoreBlockTable is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->cmpKv.shape == nullptr, OP_LOGE("Compressor", "Shape of tensor cmpKv is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->cmpKv.desc == nullptr, OP_LOGE("Compressor", "Desc of tensor cmpKv is nullptr"), return ge::GRAPH_FAILED);
    if (context_->layout == LayoutType::LAYOUT_TH){
        OP_CHECK_IF(context_->cuSeqlens.desc == nullptr, 
        OP_LOGE("Compressor", "In TH situation, desc of tensor cuSeqlens should not be nullptr"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->cuSeqlens.shape == nullptr, 
        OP_LOGE("Compressor", "In TH situation, shape of tensor cuSeqlens should not be nullptr"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(context_->ropeHeadDim == nullptr, OP_LOGE("Compressor", "attr ropeHeadDim is nullptr"),
               return ge::GRAPH_FAILED);

    OP_CHECK_IF(context_->cmpRatio == nullptr, OP_LOGE("Compressor", "attr cmpRatio is nullptr"),
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
                OP_LOGE("Compressor", "hiddenSize should be whthin [1k, 10k] and be 512-aligned, but got %u",
                        baseParams_->hiddenSize),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(pageAttentionParams_->blockSize > MAX_BLOCK_SIZE || pageAttentionParams_->blockSize < MIN_BLOCK_SIZE ||
                    pageAttentionParams_->blockSize % ALIGN_FACTOR_BLOCK_SIZE != 0,
                OP_LOGE("Compressor", "blockSize should be whthin [16, 1024] and be 16-aligned, but got %u",
                        pageAttentionParams_->blockSize),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::LogErrorShapeConsistency(const std::string &name,
    const gert::StorageShape *shape, const uint32_t &dimNum, const std::string &subName, const uint32_t &expectNum) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const uint32_t actualNum = shape->GetStorageShape().GetDim(dimNum);
    OP_CHECK_IF(actualNum != expectNum,
                OP_LOGE("Compressor", 
                        "%s shape dim [%s] should be equal to %u, but got %u",
                        name.c_str(), subName.c_str(), expectNum, actualNum),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckShapeConsistency() const
{
    if (CheckShapeConsistencyRope() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto coffD = coff * baseParams_->headDim;
    if (ge::GRAPH_SUCCESS != LogErrorShapeConsistency("kvBlockTable", context_->kvBlockTable.shape, COMPRESSOR_DIM_INDEX_0, "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("scoreBlockTable", context_->scoreBlockTable.shape, COMPRESSOR_DIM_INDEX_0, "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("cuSeqlens", context_->cuSeqlens.shape, COMPRESSOR_DIM_INDEX_0, "batchSize+1", baseParams_->batchSize + 1) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("seqUsed", context_->seqUsed.shape, COMPRESSOR_DIM_INDEX_0, "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("startPos", context_->startPos.shape, COMPRESSOR_DIM_INDEX_0, "batchSize", baseParams_->batchSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("wkv", context_->wkv.shape, COMPRESSOR_DIM_INDEX_1, "hiddenSize", baseParams_->hiddenSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("wgate", context_->wgate.shape, COMPRESSOR_DIM_INDEX_1, "hiddenSize", baseParams_->hiddenSize) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("wkv", context_->wkv.shape, COMPRESSOR_DIM_INDEX_0, "coff*headDim", static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("wgate", context_->wgate.shape, COMPRESSOR_DIM_INDEX_0, "coff*headDim", static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("kvState", context_->kvState.shape, COMPRESSOR_DIM_INDEX_2, "coff*headDim", static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("scoreState", context_->scoreState.shape, COMPRESSOR_DIM_INDEX_2, "coff*headDim", static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ape", context_->ape.shape, COMPRESSOR_DIM_INDEX_1, "coff*headDim", static_cast<uint32_t>(coffD)) ||
        ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ape", context_->ape.shape, COMPRESSOR_DIM_INDEX_0, "cmpRatio", baseParams_->cmpRatio)) { 
        return ge::GRAPH_FAILED;
    }
    const auto& scoreStateShape = context_->scoreState.shape->GetStorageShape();
    uint32_t actualDim0 = scoreStateShape.GetDim(COMPRESSOR_DIM_INDEX_0);
    uint32_t actualDim1 = scoreStateShape.GetDim(COMPRESSOR_DIM_INDEX_1);
    const uint32_t expectDim0 = pageAttentionParams_->blockNum;
    const uint32_t expectDim1 = pageAttentionParams_->blockSize;
    OP_CHECK_IF(actualDim0 != expectDim0 || actualDim1 != expectDim1,
        OP_LOGE("Compressor", "scoreState shape dim0 should be blockNum(%u), dim1 should be blockSize(%u), but got dim0=%u, dim1=%u",
                expectDim0, expectDim1, actualDim0, actualDim1), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckShapeConsistencyRope() const
{
    auto cmpT = std::min(baseParams_->tokenSize, baseParams_->tokenSize / baseParams_->cmpRatio + baseParams_->batchSize);
    if (context_->layout == LayoutType::LAYOUT_BSH) {
        if (ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeSin", context_->ropeSin.shape, COMPRESSOR_DIM_INDEX_0, "0:batchSize", baseParams_->batchSize) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeCos", context_->ropeCos.shape, COMPRESSOR_DIM_INDEX_0, "0:batchSize", baseParams_->batchSize) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeSin", context_->ropeSin.shape, COMPRESSOR_DIM_INDEX_1, "1:ceil(seqSize/cmpRatio)", baseParams_->cgSize) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeCos", context_->ropeCos.shape, COMPRESSOR_DIM_INDEX_1, "1:ceil(seqSize/cmpRatio)", baseParams_->cgSize) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeSin", context_->ropeSin.shape, COMPRESSOR_DIM_INDEX_2, "2:ropeHeadDim", baseParams_->ropeHeadDim) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeCos", context_->ropeCos.shape, COMPRESSOR_DIM_INDEX_2, "2:ropeHeadDim", baseParams_->ropeHeadDim)) {
            return ge::GRAPH_FAILED;
        }
    } else {
        if (ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeSin", context_->ropeSin.shape, COMPRESSOR_DIM_INDEX_0, "0:min(tokenSize, tokenSize/cmpRatio+batchSize)", static_cast<uint32_t>(cmpT)) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeCos", context_->ropeCos.shape, COMPRESSOR_DIM_INDEX_0, "0:min(tokenSize, tokenSize/cmpRatio+batchSize)", static_cast<uint32_t>(cmpT)) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeSin", context_->ropeSin.shape, COMPRESSOR_DIM_INDEX_1, "1:ropeHeadDim", baseParams_->ropeHeadDim) ||
            ge::GRAPH_SUCCESS != LogErrorShapeConsistency("ropeCos", context_->ropeCos.shape, COMPRESSOR_DIM_INDEX_1, "1:ropeHeadDim", baseParams_->ropeHeadDim)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorTiling::CheckDtypeConsistencyX(const gert::CompileTimeTensorDesc *desc,
                                                         const std::string &name) const
{
    const auto actualDtype = desc->GetDataType();
    OP_CHECK_IF(
        actualDtype != context_->dtype,
        OP_LOGE("Compressor", "%s datatype should be same with x: %s, but got %s", name.c_str(),
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
                OP_LOGE("Compressor", "ropeSin dim num should be equal to x: %u, but got %u", xDimNum,
                        context_->ropeSin.shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDimNum != context_->ropeCos.shape->GetStorageShape().GetDimNum(),
                OP_LOGE("Compressor", "ropeCos dim num should be equal to x: %u, but got %u", xDimNum,
                        context_->ropeCos.shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(xDimNum != context_->cmpKv.shape->GetStorageShape().GetDimNum(),
                OP_LOGE("Compressor", "cmpKv dim num should be equal to x: %u, but got %u", xDimNum,
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
                OP_LOGE("Compressor", "Cmpratio Coff Headdim should be equal to {4, 2, 512}, {4, 2, 128}, {128, 1, 512},\
 but now cmpratio=%u, coff=%u, headdim=%u", curCmpratio, curCoff, curHeaddim), return ge::GRAPH_FAILED);
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
    CompressorTilingData* tilingData = context->GetTilingData<CompressorTilingData>();
    OP_CHECK_IF(tilingData == nullptr,
            OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "TilingData is nullptr."),
            return ge::GRAPH_FAILED);
    // 使用SyncAll，需要设置为batchmode模式，所有核同时启动，否则多流方式下执行可能会卡死
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);
    if (compressorTiling.RunBigKernelTiling(tilingData) == ge::SUCCESS) {
        context->SetTilingKey(compressorContext.tilingKey);
        context->SetBlockDim(compressorContext.blockDim);
        OP_LOGI(context->GetNodeName(), "Compressor block dim: %u.", compressorContext.blockDim);
        return ge::GRAPH_SUCCESS;
    }

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