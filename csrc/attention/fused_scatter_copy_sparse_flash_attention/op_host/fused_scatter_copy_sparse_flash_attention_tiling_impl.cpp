/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_scatter_copy_sparse_flash_attention_tiling_impl.cpp
 * \brief
 */

#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "../op_kernel/sparse_flash_attention_impl/fused_scatter_copy_sparse_flash_attention_template_tiling_key.h"
#include "fused_scatter_copy_sparse_flash_attention_tiling_impl.h"

using std::map;
using std::string;
using std::pair;

using namespace ge;
using namespace AscendC;
namespace optiling {

constexpr uint32_t PRE_LOAD_NUM = 2;
constexpr uint32_t BLOCK_TABLE_ELEM_BYTE = 4;
constexpr int32_t SPARSE_MODE_BAND = 4;

static const std::string QUERY_NAME = "query";
static const std::string KEY_NAME = "key";
static const std::string VALUE_NAME = "value";
static const std::string BLOCK_TABLE_NAME = "block_table";
static const std::string SPARSE_INDICES_NAME = "sparse_indices";
static const std::string QUERY_ROPE_NAME = "query_rope";
static const std::string KEY_ROPE_NAME = "key_rope";
static const std::string ATTEN_OUT_NAME = "attention_out";

const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME,                  {ge::DT_FLOAT16, ge::DT_BF16}},
    {KEY_NAME,                    {ge::DT_FLOAT16, ge::DT_BF16}},
    {VALUE_NAME,                  {ge::DT_FLOAT16, ge::DT_BF16}},
    {QUERY_ROPE_NAME,             {ge::DT_FLOAT16, ge::DT_BF16}},
    {KEY_ROPE_NAME,               {ge::DT_FLOAT16, ge::DT_BF16}},
    {ATTEN_OUT_NAME,              {ge::DT_FLOAT16, ge::DT_BF16}},
    {SPARSE_INDICES_NAME,         {ge::DT_INT32}}
};

const std::map<std::string, std::vector<FusedScatterCopySparseFlashAttentionLayout>> LAYOUT_SUPPORT_MAP = {
    {QUERY_NAME,             {FusedScatterCopySparseFlashAttentionLayout::BSND, FusedScatterCopySparseFlashAttentionLayout::TND}},
    {KEY_NAME,               {FusedScatterCopySparseFlashAttentionLayout::BSND, FusedScatterCopySparseFlashAttentionLayout::TND, FusedScatterCopySparseFlashAttentionLayout::PA_BSND}},
    {VALUE_NAME,             {FusedScatterCopySparseFlashAttentionLayout::BSND, FusedScatterCopySparseFlashAttentionLayout::TND, FusedScatterCopySparseFlashAttentionLayout::PA_BSND}},
    {ATTEN_OUT_NAME,         {FusedScatterCopySparseFlashAttentionLayout::BSND, FusedScatterCopySparseFlashAttentionLayout::TND}},
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

struct FusedScatterCopySparseFlashAttentionCompileInfo {
    int64_t core_num;
};

static const std::map<FusedScatterCopySparseFlashAttentionLayout, std::vector<FusedScatterCopySparseFlashAttentionAxis>> FusedScatterCopySparseFlashAttention_LAYOUT_AXIS_MAP = {
    {FusedScatterCopySparseFlashAttentionLayout::BSND, {FusedScatterCopySparseFlashAttentionAxis::B, FusedScatterCopySparseFlashAttentionAxis::S, FusedScatterCopySparseFlashAttentionAxis::N, FusedScatterCopySparseFlashAttentionAxis::D}},
    {FusedScatterCopySparseFlashAttentionLayout::TND, {FusedScatterCopySparseFlashAttentionAxis::T, FusedScatterCopySparseFlashAttentionAxis::N, FusedScatterCopySparseFlashAttentionAxis::D}},
    {FusedScatterCopySparseFlashAttentionLayout::PA_BSND, {FusedScatterCopySparseFlashAttentionAxis::Bn, FusedScatterCopySparseFlashAttentionAxis::Bs, FusedScatterCopySparseFlashAttentionAxis::N, FusedScatterCopySparseFlashAttentionAxis::D}},
};

static const std::map<FusedScatterCopySparseFlashAttentionLayout, size_t> FusedScatterCopySparseFlashAttention_LAYOUT_DIM_MAP = {
    {FusedScatterCopySparseFlashAttentionLayout::BSND, DIM_NUM_FOUR},
    {FusedScatterCopySparseFlashAttentionLayout::TND, DIM_NUM_THREE},
    {FusedScatterCopySparseFlashAttentionLayout::PA_BSND, DIM_NUM_FOUR},
};

static std::string GetShapeStr(gert::Shape shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

static std::string FusedScatterCopySparseFlashAttentionDataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OPS_LOG_E("FusedScatterCopySparseFlashAttention", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

string FusedScatterCopySparseFlashAttentionTensorDesc2String(const gert::StorageShape *shape, const gert::CompileTimeTensorDesc *tensor)
{
    if (shape == nullptr || tensor == nullptr) {
        return "nil ";
    }

    std::ostringstream oss;
    oss << "(dtype: " << ge::TypeUtils::DataTypeToAscendString(tensor->GetDataType()).GetString() << "),";
    oss << "(shape:" << FusedScatterCopySparseFlashAttentionShape2String(shape->GetStorageShape()) << "),";
    oss << "(ori_shape:" << FusedScatterCopySparseFlashAttentionShape2String(shape->GetOriginShape()) << "),";
    oss << "(format: "
        << ge::TypeUtils::FormatToAscendString(
               static_cast<ge::Format>(ge::GetPrimaryFormat(tensor->GetStorageFormat())))
               .GetString()
        << "),";
    oss << "(ori_format: " << ge::TypeUtils::FormatToAscendString(tensor->GetOriginFormat()).GetString() << ") ";

    return oss.str();
}

string FusedScatterCopySparseFlashAttentionDebugTilingContext(const gert::TilingContext *context)
{
    std::ostringstream oss;
    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetInputsNum(); ++i) {
        oss << "input" << i << ": ";
        oss << FusedScatterCopySparseFlashAttentionTensorDesc2String(context->GetInputShape(i), context->GetInputDesc(i));
    }

    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
        oss << "output" << i << ": ";
        oss << FusedScatterCopySparseFlashAttentionTensorDesc2String(context->GetOutputShape(i), context->GetOutputDesc(i));
    }
    return oss.str();
}

std::string FusedScatterCopySparseFlashAttentionLayoutToSerialString(FusedScatterCopySparseFlashAttentionLayout layout)
{
    switch (layout) {
        case FusedScatterCopySparseFlashAttentionLayout::BSND: return "BSND";
        case FusedScatterCopySparseFlashAttentionLayout::TND: return "TND";
        case FusedScatterCopySparseFlashAttentionLayout::PA_BSND: return "PA_BSND";
        default: return "UNKNOWN";
    }
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTiling::SetBlockDim(uint32_t blockDim)
{
    context_->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTiling::SetTilingKey(uint64_t tilingKey)
{
    context_->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTiling::SetWorkspaceSize(uint64_t workspaceSize)
{
    OPS_ERR_IF(context_->GetWorkspaceSizes(1) == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "workSpaceSize got from ge is nullptr"),
        return ge::GRAPH_FAILED);
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTiling::SetTilingData(TilingDef &tilingData)
{
    OPS_ERR_IF(context_->GetRawTilingData() == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
        return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTiling::GetPlatformInfo()
{
    OPS_ERR_IF(sfaInfo_->platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(sfaInfo_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(sfaInfo_->platformInfo);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();

    OPS_ERR_IF(aicNum_ == 0 || aivNum_ == 0,
        OPS_REPORT_VECTOR_INNER_ERR(sfaInfo_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void FusedScatterCopySparseFlashAttentionTiling::GenTilingKey()
{
    tilingKey_ = GET_TPL_TILING_KEY(0U, static_cast<uint32_t>(FusedScatterCopySparseFlashAttentionLayout::TND),
                                   static_cast<uint32_t>(FusedScatterCopySparseFlashAttentionLayout::PA_BSND), 1U);

    OPS_LOG_I(sfaInfo_->opName, "FusedScatterCopySparseFlashAttention tilingKey_: %lu.", tilingKey_);
}

void FusedScatterCopySparseFlashAttentionTiling::ZeroTensorProcess()
{
    if (sfaInfo_->s2Size == 0) {
        sfaInfo_->s2Size = 1024;
    }
}

void FusedScatterCopySparseFlashAttentionTiling::InitParams()
{
    perfMode_ = FusedScatterCopySparseFlashAttentionPerfMode::V_TEMPLATE_MODE;

    coreNum_ = aicNum_;

    headDimAlign_ = FusedScatterCopySparseFlashAttentionAlign(sfaInfo_->qkHeadDim, BYTE_BLOCK);
    ZeroTensorProcess();
}

void FusedScatterCopySparseFlashAttentionTiling::CalcUbBmm()
{
    uint32_t cubeMSize = sfaInfo_->gSize * sfaInfo_->s1Size;
    uint32_t maxMSize = mBaseSize_;
    if (cubeMSize > maxMSize) {
        cubeMSize = maxMSize;
    }
    mmResUbSize_ = sInnerSizeAlign_ * FusedScatterCopySparseFlashAttentionAlign(cubeMSize, 16U);
    bmm2ResUbSize_ = headDimAlign_ * FusedScatterCopySparseFlashAttentionAlign(cubeMSize, 16U);

    qPreSizeMla_ = sfaInfo_->gSize * (headDimAlign_ + 64U) * sfaInfo_->s1Size;
}

void FusedScatterCopySparseFlashAttentionTiling::CheckUbSpace()
{
    CalcUbBmm();
}

void FusedScatterCopySparseFlashAttentionTiling::CalcInnerSize(uint32_t s2Size)
{
    sInnerSize_ = 512;
    if (splitKVFlag_ && sfaInfo_->qLayout != FusedScatterCopySparseFlashAttentionLayout::TND) {
        if (s2Size == 256) {
            sInnerSize_ = 128;
        } else if (s2Size > 256 && s2Size <= sInnerSize_) {
            sInnerSize_ = (sInnerSize_ + 1) / 2;
        }
    }

    sInnerLoopTimes_ = (s2Size + sInnerSize_ - 1) / sInnerSize_;
    sInnerSizeTail_ = s2Size - (sInnerLoopTimes_ - 1) * sInnerSize_;
    if (sInnerSize_ > s2Size) {
        sInnerSize_ = s2Size;
    }
    sInnerSizeAlign_ = FusedScatterCopySparseFlashAttentionAlign(sInnerSize_, BYTE_BLOCK);

    CheckUbSpace();
}

void FusedScatterCopySparseFlashAttentionTiling::SplitBalanced()
{
    CalcInnerSize(sfaInfo_->s2Size);

    FusedScatterCopySparseFlashAttentionInnerSplitParams innerSplitParams;
    innerSplitParams.s1GBaseSize = sfaInfo_->gSize;
    innerSplitParams.s2BaseSize = sInnerSize_;
    tilingData_.innerSplitParams.set_mBaseSize(innerSplitParams.s1GBaseSize);
    tilingData_.innerSplitParams.set_s2BaseSize(innerSplitParams.s2BaseSize);

    usedCoreNum_ = aicNum_;
}

void FusedScatterCopySparseFlashAttentionTiling::Split()
{
    SplitBalanced();
}

void FusedScatterCopySparseFlashAttentionTiling::FillTilingBaseParamsMla()
{
    tilingData_.baseParams.set_batchSize(sfaInfo_->bSize);
    tilingData_.baseParams.set_seqSize(sfaInfo_->s2Size);
    tilingData_.baseParams.set_qSeqSize(sfaInfo_->s1Size);
    tilingData_.baseParams.set_blockSize(sfaInfo_->blockSize);
    tilingData_.baseParams.set_maxBlockNumPerBatch(sfaInfo_->maxBlockNumPerBatch);
    tilingData_.baseParams.set_scaleValue(sfaInfo_->scaleValue);
    tilingData_.baseParams.set_nNumOfQInOneGroup(sfaInfo_->n1Size / sfaInfo_->n2Size);
    tilingData_.baseParams.set_actualLenDimsQ(sfaInfo_->actualLenDimsQ);
    tilingData_.baseParams.set_actualLenDimsKV(sfaInfo_->actualLenDimsKV);
    tilingData_.baseParams.set_outputLayout(static_cast<uint32_t>(sfaInfo_->outLayout));
    tilingData_.baseParams.set_sparseMode(sfaInfo_->sparseMode);
    tilingData_.baseParams.set_sparseBlockSize(sfaInfo_->sparseBlockSize);
    tilingData_.baseParams.set_sparseBlockCount(OFFLOAD_SPARSE_COMPUTE_COUNT);
}

// for flash decode
void FusedScatterCopySparseFlashAttentionTiling::FillTilingSplitKVMla()
{
    tilingData_.splitKVParams.set_s2(kvSplitPart_);

    tilingData_.splitKVParams.set_accumOutSize(aicNum_ * 2 * sfaInfo_->n2Size * mBaseSize_ * headDimAlign_);
    tilingData_.splitKVParams.set_logSumExpSize(2 * aicNum_ * 2 * sfaInfo_->n2Size * mBaseSize_ *
                                                (BYTE_BLOCK / BLOCK_TABLE_ELEM_BYTE));

    if (!splitKVFlag_) {
        tilingData_.splitKVParams.set_s2(0);
    }
}

void FusedScatterCopySparseFlashAttentionTiling::FillTilingSingleCoreParamsMla()
{
    tilingData_.singleCoreParams.set_usedCoreNum(usedCoreNum_);
}

void FusedScatterCopySparseFlashAttentionTiling::FillTilingSingleCoreTensorSizeMla()
{
    tilingData_.singleCoreTensorSize.set_mmResUbSize(mmResUbSize_);
    tilingData_.singleCoreTensorSize.set_bmm2ResUbSize(bmm2ResUbSize_);
}

void FusedScatterCopySparseFlashAttentionTiling::FillTiling()
{
    FillTilingBaseParamsMla();
    FillTilingSplitKVMla();
    FillTilingSingleCoreParamsMla();
    FillTilingSingleCoreTensorSizeMla();
}

uint32_t FusedScatterCopySparseFlashAttentionTiling::CalcBalanceFDParamNums(const uint32_t actCoreNum)
{
    return actCoreNum * 2 * sfaInfo_->n2Size * mBaseSize_;
}

void FusedScatterCopySparseFlashAttentionTiling::NormalCalcFDWorkSpace(const uint32_t actCoreNum)
{
    if (splitKVFlag_) {
        uint32_t accumOutSize = 0;
        uint32_t logSumExpSize = 0;
        uint32_t FDParamNums = CalcBalanceFDParamNums(actCoreNum);
        accumOutSize = FDParamNums * headDimAlign_;
        logSumExpSize = 2 * FDParamNums * (BYTE_BLOCK / sfaInfo_->blockTypeSize);
        workspaceSize_ += (accumOutSize + logSumExpSize) * sfaInfo_->blockTypeSize;
        if (sfaInfo_->socVersion == platform_ascendc::SocVersion::ASCEND310P) {
            workspaceSize_ += static_cast<size_t>(actCoreNum) * 32;
        }
    }
}

void FusedScatterCopySparseFlashAttentionTiling::CalcFDWorkSpace(const uint32_t actCoreNum)
{
    NormalCalcFDWorkSpace(actCoreNum);
}

void FusedScatterCopySparseFlashAttentionTiling::GetWorkspaceSize()
{
    uint32_t mmResElemSize = 4;
    uint32_t vec1ResElemSize = 2;
    uint32_t bmm2ResElemSize = 4;
    uint32_t qPreProcResElemSize = 0;
    uint32_t nUpdateElemSize = 4;
    uint32_t softmaxSumElemSize = 4;
    float kvDtypeRatio = 1.0;

    workspaceSize_ = libapiSize_;
    uint32_t preLoadNum = 1;
    uint32_t actCoreNum = coreNum_;
    preLoadNum = PRE_LOAD_NUM;

    workspaceSize_ += preLoadNum * (mmResUbSize_ * actCoreNum * mmResElemSize);
    workspaceSize_ += preLoadNum * static_cast<size_t>(static_cast<float>(mmResUbSize_ * actCoreNum * vec1ResElemSize) * kvDtypeRatio);
    workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * bmm2ResElemSize;
    workspaceSize_ += preLoadNum * static_cast<size_t>(static_cast<float>(qPreSizeMla_ * actCoreNum * qPreProcResElemSize) * kvDtypeRatio);
    workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * nUpdateElemSize;
    workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * softmaxSumElemSize;
    workspaceSize_ += 4 * 512 * (512 + 64) * 2 * actCoreNum;
    workspaceSize_ += 4 * 128 * 4 * (2 * actCoreNum);

    CalcFDWorkSpace(actCoreNum);
}

void FusedScatterCopySparseFlashAttentionTiling::CalcBlockDim()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(sfaInfo_->platformInfo);
    auto aicNum = usedCoreNum_;
    auto aivNum = 2 * usedCoreNum_;

    blockDim_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OPS_LOG_I(sfaInfo_->opName, "FusedScatterCopySparseFlashAttention block dim: %u aiv Num: %u aic Num: %u.", blockDim_, aivNum, aicNum);
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTiling::DoOpTiling(FusedScatterCopySparseFlashAttentionTilingInfo *sfaInfo)
{
    sfaInfo_ = sfaInfo;
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    InitParams();
    Split();
    FillTiling();
    CalcBlockDim();
    GetWorkspaceSize();
    GenTilingKey();

    if ((SetBlockDim(blockDim_) != ge::GRAPH_SUCCESS) ||
        (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) ||
        (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingFusedScatterCopySparseFlashAttention(gert::TilingContext *context)
{
    FusedScatterCopySparseFlashAttentionTilingInfo sfaInfo;
    FusedScatterCopySparseFlashAttentionInfoParser sfaInfoParser(context);
    if (sfaInfoParser.Parse(sfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    FusedScatterCopySparseFlashAttentionTilingCheck tilingChecker(sfaInfo);
    if (tilingChecker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    FusedScatterCopySparseFlashAttentionTiling tiling(context);
    return tiling.DoOpTiling(&sfaInfo);
}

ge::graphStatus TilingFusedScatterCopySparseFlashAttentionMtp(gert::TilingContext *context)
{
    FusedScatterCopySparseFlashAttentionTilingInfo sfaInfo;
    FusedScatterCopySparseFlashAttentionInfoParser sfaInfoParser(context);
    if (sfaInfoParser.Parse(sfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OPS_ERR_IF(sfaInfo.bSize == 0 || sfaInfo.qTSize < sfaInfo.bSize ||
                   sfaInfo.qTSize > sfaInfo.bSize * 16U,
               OPS_LOG_E(context->GetNodeName(),
                         "MTP TND query rows must satisfy B <= T <= 16B."),
               return ge::GRAPH_FAILED);

    // The standalone non-MTP operator keeps the strict single-query checker.
    // MTP uses the same variable-width TND contract and device-side prefix
    // validation as standardized COPYFusedScatterCopySparseFlashAttention-MTP: B <= T <= 16B.
    FusedScatterCopySparseFlashAttentionTiling tiling(context);
    const ge::graphStatus status = tiling.DoOpTiling(&sfaInfo);
    if (status == ge::GRAPH_SUCCESS) {
        // Variable-width MTP has one TND/PA_BSND V-template specialization.
        context->SetTilingKey(1U);
        // Match the official FusedScatterCopySparseFlashAttention batch scheduler for one AIC plus two AIVs.
        context->SetScheduleMode(1);
    }
    return status;
}

ge::graphStatus TilingPrepareForFusedScatterCopySparseFlashAttention(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::GetExpectedShape(gert::Shape &shapeExpected,
    const FusedScatterCopySparseFlashAttentionTilingShapeCompareParam &param, const FusedScatterCopySparseFlashAttentionLayout &layout) const
{
    if (layout == FusedScatterCopySparseFlashAttentionLayout::BSND) {
        shapeExpected = gert::Shape({param.B, param.S, param.N, param.D});
    } else if (layout == FusedScatterCopySparseFlashAttentionLayout::TND) {
        shapeExpected = gert::Shape({param.T, param.N, param.D});
    } else if (layout == FusedScatterCopySparseFlashAttentionLayout::PA_BSND) {
        shapeExpected = gert::Shape({param.Bn, param.Bs, param.N, param.D});
    } else {
        OPS_LOG_E(opName_, "layout %s is unsupported", FusedScatterCopySparseFlashAttentionLayoutToSerialString(layout).c_str());
        return ge::GRAPH_FAILED;
    }
    if (shapeExpected.GetDim(0) == 0) {
        OPS_LOG_E(opName_, "expected shape is %s, the first dim should not be 0.", GetShapeStr(shapeExpected).c_str());
        return ge::GRAPH_PARAM_INVALID;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CompareShape(FusedScatterCopySparseFlashAttentionTilingShapeCompareParam &param,
    const gert::Shape &shape, const FusedScatterCopySparseFlashAttentionLayout &layout, const std::string &name) const
{
    gert::Shape shapeExpected;
    if (GetExpectedShape(shapeExpected, param, layout) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (shape.GetDimNum() != shapeExpected.GetDimNum()) {
        OPS_LOG_E(opName_,
            "%s dimension is %zu, expected dimension is %zu.",
            name.c_str(), shape.GetDimNum(), shapeExpected.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) != shapeExpected.GetDim(i)) {
            OPS_LOG_E(opName_, "%s layout is %s, shape is %s, expected shape is %s.",
                name.c_str(), FusedScatterCopySparseFlashAttentionLayoutToSerialString(layout).c_str(),
                GetShapeStr(shape).c_str(), GetShapeStr(shapeExpected).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

void FusedScatterCopySparseFlashAttentionTilingCheck::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
    const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << FusedScatterCopySparseFlashAttentionDataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OPS_LOG_E(opName_, "Tensor %s only supports dtype %s, but got %s",
        name.c_str(), oss.str().c_str(), FusedScatterCopySparseFlashAttentionDataTypeToSerialString(actualDtype).c_str());
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
    const std::string &name) const
{
    if (desc != nullptr) {
        const auto& it = DTYPE_SUPPORT_MAP.find(name);
        OPS_ERR_IF(it == DTYPE_SUPPORT_MAP.end(),
            OPS_LOG_E(opName_, "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
            return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OPS_ERR_IF(std::find(
            expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end(),
            LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
void FusedScatterCopySparseFlashAttentionTilingCheck::LogErrorNumberSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name, const std::string subName) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << std::to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }

    OPS_LOG_E(opName_, "%s %s only supports %s, but got %s",
              name.c_str(), subName.c_str(), oss.str().c_str(), std::to_string(actualValue).c_str());
}

template <typename T>
void FusedScatterCopySparseFlashAttentionTilingCheck::LogErrorDimNumSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name) const
{
    LogErrorNumberSupport(expectNumberList, actualValue, name, "dimension");
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckDimNumInLayoutSupport(const FusedScatterCopySparseFlashAttentionLayout &layout,
    const gert::StorageShape *shape, const std::string &name) const
{
    const auto& dimIt = FusedScatterCopySparseFlashAttention_LAYOUT_DIM_MAP.find(layout);
    OPS_ERR_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
        OPS_LOG_E(opName_, "When layout is %s, %s dimension should be %zu, but it's %zu",
            FusedScatterCopySparseFlashAttentionLayoutToSerialString(layout).c_str(), name.c_str(), dimIt->second,
            shape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckDimNumSupport(const gert::StorageShape *shape,
    const std::vector<size_t> &expectDimNumList, const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectDimNumList.begin(), expectDimNumList.end(),
        shape->GetStorageShape().GetDimNum()) == expectDimNumList.end()) {
        LogErrorDimNumSupport(expectDimNumList, shape->GetStorageShape().GetDimNum(), name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}


void FusedScatterCopySparseFlashAttentionTilingCheck::LogErrorLayoutSupport(const std::vector<FusedScatterCopySparseFlashAttentionLayout> &expectLayoutList,
    const FusedScatterCopySparseFlashAttentionLayout &actualLayout, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectLayoutList.size(); ++i) {
        oss << FusedScatterCopySparseFlashAttentionLayoutToSerialString(expectLayoutList[i]);
        if (i < expectLayoutList.size() - 1) {
            oss << ", ";
        }
    }
    OPS_LOG_E(opName_, "Tensor %s only supports layout %s, but got %s",
        name.c_str(), oss.str().c_str(), FusedScatterCopySparseFlashAttentionLayoutToSerialString(actualLayout).c_str());
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckLayoutSupport(const FusedScatterCopySparseFlashAttentionLayout &actualLayout, const std::string &name) const
{
    const auto& it = LAYOUT_SUPPORT_MAP.find(name);
    OPS_ERR_IF(it == LAYOUT_SUPPORT_MAP.end(),
        OPS_LOG_E(opName_, "%s layout support list should be specify in LAYOUT_SUPPORT_MAP", name.c_str()),
        return ge::GRAPH_FAILED);
    auto &expectLayoutList = it->second;
    OPS_ERR_IF(std::find(
        expectLayoutList.begin(), expectLayoutList.end(), actualLayout) == expectLayoutList.end(),
        LogErrorLayoutSupport(expectLayoutList, actualLayout, name),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckSingleParaQuery() const
{
    const std::vector<size_t> queryDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.query.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(qLayout_, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.query.shape, queryDimNumList, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(qLayout_, opParamInfo_.query.shape, QUERY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckSingleParaKey() const
{
    const std::vector<size_t> keyDimNumList = {DIM_NUM_FOUR, DIM_NUM_THREE};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.key.desc, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.key.shape, keyDimNumList, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(kvLayout_, opParamInfo_.key.shape, KEY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckSingleParaNumHeads() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckSingleParaKvHeadNums() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckSingleParaSparseMode() const
{
    OPS_ERR_IF((*opParamInfo_.sparseMode != 3 && *opParamInfo_.sparseMode != 0),
        OPS_LOG_E(opName_, "sparseMode must == 0/3, but got: %ld.", *opParamInfo_.sparseMode),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckSingleParaSparseBlockSize() const
{
    OPS_ERR_IF((*opParamInfo_.sparseBlockSize <= 0),
        OPS_LOG_E(opName_, "sparseBlockSize should be greater than 0, but got: %ld.", *opParamInfo_.sparseBlockSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckSingleParaSparseIndices() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.sparseIndices.desc, SPARSE_INDICES_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaQuery() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKey() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseIndices() ||
        ge::GRAPH_SUCCESS != CheckSingleParaNumHeads() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKvHeadNums() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseMode() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseBlockSize()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckRopeExistence()
{
    OPS_ERR_IF((opParamInfo_.queryRope.tensor != nullptr && opParamInfo_.keyRope.tensor == nullptr),
        OPS_LOG_E(opName_, "KeyRope is null, but queryRope exists, they should be both null or exist."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF((opParamInfo_.queryRope.tensor == nullptr && opParamInfo_.keyRope.tensor != nullptr),
        OPS_LOG_E(opName_, "QueryRope is null, but keyRope exists, they should be both null or exist."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.keyRope.desc == nullptr || opParamInfo_.queryRope.desc == nullptr,
        OPS_LOG_E(opName_, "In Mla situation, desc of keyRope and queryRope should not be null"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckExists(const void *pointer, const std::string &name) const
{
    OPS_ERR_IF(pointer == nullptr,
        OPS_LOG_E(opName_, "%s should not be null", name.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckNotExists(const void *pointer, const std::string &name) const
{
    OPS_ERR_IF(pointer != nullptr,
        OPS_LOG_E(opName_, "%s should be null", name.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto& kv : paramMap) {
        if (CheckExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckNotExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto& kv : paramMap) {
        if (CheckNotExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckExistenceByMap(std::map<std::string, const void *> &existMap,
    std::map<std::string, const void *> &notExistMap) const
{
    if (CheckExistsByMap(existMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNotExistsByMap(notExistMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckAttrValueByMap(std::map<std::string, std::pair<const T *, T>> &attrMap) const
{
    for (auto const &kv : attrMap) {
        const std::string &name = kv.first;
        const std::pair<const T *, T> &pointerValuePair = kv.second;
        if (pointerValuePair.first == nullptr) {
            OPS_LOG_E(opName_, "Attr %s should not be nullptr", name.c_str());
            return ge::GRAPH_FAILED;
        }

        if (*(pointerValuePair.first) != pointerValuePair.second) {
            std::ostringstream ossExpect;
            ossExpect << std::to_string(pointerValuePair.second);
            std::ostringstream ossActual;
            ossActual << std::to_string(*(pointerValuePair.first));
            OPS_LOG_E(opName_,
                "%s value should be %s, but got %s",
                name.c_str(),
                ossExpect.str().c_str(),
                ossActual.str().c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckParaExistenceMlaNoquant() const
{
    if (kvStorageMode_ != FusedScatterCopySparseFlashAttentionKvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }
    std::map<std::string, const void *> mlaNoquantParamExistMap = {
        {"actualSeqLengths", opParamInfo_.actualSeqLengths.tensor},
        {"blockTable", opParamInfo_.blockTable.tensor},
    };
    std::map<std::string, const void *> mlaNoquantParamNotExistMap = {};
    if (CheckExistenceByMap(mlaNoquantParamExistMap, mlaNoquantParamNotExistMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckParaExistenceMla() const
{
    return CheckParaExistenceMlaNoquant();
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckParaExistence()
{
    if (ge::GRAPH_SUCCESS != CheckRopeExistence()) {
        return ge::GRAPH_FAILED;
    }

    return CheckParaExistenceMla();
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    const FusedScatterCopySparseFlashAttentionLayout &layoutQuery, const std::string &name)
{
    if (tensor == nullptr) {
        OPS_LOG_E(opName_, "when layout of query is %s, %s must be provided.",
            FusedScatterCopySparseFlashAttentionLayoutToSerialString(layoutQuery).c_str(), name.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OPS_LOG_E(opName_, "the shape size of %s is %ld, it should be greater than 0.",
            name.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

void FusedScatterCopySparseFlashAttentionTilingCheck::SetFusedScatterCopySparseFlashAttentionShapeCompare()
{
    queryShapeCmp_ = opParamInfo_.query.shape->GetStorageShape();
    topkShapeCmp_ = opParamInfo_.sparseIndices.shape->GetStorageShape();
    keyShapeCmp_ = opParamInfo_.key.shape->GetStorageShape();
    valueShapeCmp_ = opParamInfo_.value.shape->GetStorageShape();
    attenOutShapeCmp_ = opParamInfo_.attenOut.shape->GetStorageShape();
    queryRopeShapeCmp_ = opParamInfo_.queryRope.tensor->GetStorageShape();
    keyRopeShapeCmp_ = opParamInfo_.keyRope.tensor->GetStorageShape();
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckBlockTable() const
{
    if (kvStorageMode_ != FusedScatterCopySparseFlashAttentionKvStorageMode::PAGE_ATTENTION) {
        OPS_ERR_IF(opParamInfo_.blockTable.tensor != nullptr,
            OPS_LOG_E(opName_, "when the layout_kv is %s, %s should be null",
                FusedScatterCopySparseFlashAttentionLayoutToSerialString(kvLayout_).c_str(), BLOCK_TABLE_NAME.c_str()),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    uint32_t blockTableBatch = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0);
    OPS_ERR_IF(blockTableBatch != bSize_,
        OPS_LOG_E(opName_, "%s's first dimension(%u) should be equal to batch size(%u)",
            BLOCK_TABLE_NAME.c_str(), blockTableBatch, bSize_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckDTypeConsistency(const ge::DataType &actualDtype,
    const ge::DataType &expectDtype, const std::string &name) const
{
    if (actualDtype != expectDtype) {
        OPS_LOG_E(opName_, "%s dtype should be %s, but it's %s.", name.c_str(),
            FusedScatterCopySparseFlashAttentionDataTypeToSerialString(expectDtype).c_str(),
            FusedScatterCopySparseFlashAttentionDataTypeToSerialString(actualDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckQRopeShape()
{
    FusedScatterCopySparseFlashAttentionTilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n1Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = ropeHeadDim_;
    shapeParams.T = qTSize_;
    return CompareShape(shapeParams, queryRopeShapeCmp_, qLayout_, QUERY_ROPE_NAME);
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckTopkShape()
{
    FusedScatterCopySparseFlashAttentionTilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n2Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = sparseBlockCount_;
    shapeParams.T = qTSize_;
    return CompareShape(shapeParams, topkShapeCmp_, topkLayout_, SPARSE_INDICES_NAME);
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckAttenOutShape()
{
    FusedScatterCopySparseFlashAttentionTilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n1Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = vHeadDim_;
    shapeParams.T = qTSize_;
    if (CompareShape(shapeParams, attenOutShapeCmp_, outLayout_, ATTEN_OUT_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckAttenOut()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.attenOut.desc->GetDataType(),
        inputQType_, ATTEN_OUT_NAME) ||
        ge::GRAPH_SUCCESS != CheckAttenOutShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckQRope()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.queryRope.desc->GetDataType(),
        inputQType_, QUERY_ROPE_NAME) ||
        ge::GRAPH_SUCCESS != CheckQRopeShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckTopK()
{
    if (ge::GRAPH_SUCCESS != CheckTopkShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckVAndKRopeShapeForBatchContinuous()
{
    FusedScatterCopySparseFlashAttentionTilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n2Size_;
    shapeParams.S = s2Size_;
    shapeParams.T = kvTSize_;
    shapeParams.D = qkHeadDim_;
    if (CompareShape(shapeParams, keyShapeCmp_, kvLayout_, KEY_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    shapeParams.D = vHeadDim_;
    if (CompareShape(shapeParams, valueShapeCmp_, kvLayout_, VALUE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    shapeParams.D = ropeHeadDim_;
    if (CompareShape(shapeParams, keyRopeShapeCmp_, kvLayout_, KEY_ROPE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

uint32_t FusedScatterCopySparseFlashAttentionTilingCheck::GetTypeSize(ge::DataType dtype) const
{
    uint32_t typeSize = NUM_BYTES_FLOAT16;
    switch (dtype) {
        case ge::DT_FLOAT16:
            typeSize = NUM_BYTES_FLOAT16;
            break;
        case ge::DT_BF16:
            typeSize = NUM_BYTES_BF16;
            break;
        default:
            typeSize = NUM_BYTES_FLOAT16;
    }
    return typeSize;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckVAndKRopeShapeForPageAttention()
{
    int64_t blockNum = keyShapeCmp_.GetDim(0);
    OPS_ERR_IF(blockNum <= 0,
        OPS_LOG_E(opName_, "The first dim(%ld) of key should be greater than 0", blockNum),
        return ge::GRAPH_FAILED);
    FusedScatterCopySparseFlashAttentionTilingShapeCompareParam shapeParams;
    shapeParams.Bn = blockNum;
    shapeParams.N = n2Size_;
    shapeParams.Bs = blockSize_;
    shapeParams.D = vHeadDim_;
    shapeParams.T = kvTSize_;
    if (CompareShape(shapeParams, valueShapeCmp_, kvLayout_, VALUE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    shapeParams.D = ropeHeadDim_;
    if (CompareShape(shapeParams, keyRopeShapeCmp_, kvLayout_, KEY_ROPE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckVAndKRopeShape()
{
    if (kvStorageMode_ == FusedScatterCopySparseFlashAttentionKvStorageMode::BATCH_CONTINUOUS) {
        return CheckVAndKRopeShapeForBatchContinuous();
    }

    if (kvStorageMode_ == FusedScatterCopySparseFlashAttentionKvStorageMode::PAGE_ATTENTION) {
        return CheckVAndKRopeShapeForPageAttention();
    }

    OPS_LOG_E(opName_, "storage mode of key and value is %u, it is incorrect.", static_cast<uint32_t>(kvStorageMode_));
    return ge::GRAPH_FAILED;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckVAndKRope()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.value.desc->GetDataType(),
        inputKvType_, VALUE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.keyRope.desc->GetDataType(),
        inputKvType_, KEY_ROPE_NAME) || ge::GRAPH_SUCCESS != CheckVAndKRopeShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckActualSeqLensQ()
{
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensQDType() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckActualSeqLensQDType()
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.actualSeqLengthsQ.desc == nullptr) {
        OPS_LOG_E(opName_, "actualSeqLengthsQ is not empty,"
            "but actualSeqLengthsQ's dtype is nullptr.");
            return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.actualSeqLengthsQ.desc->GetDataType() != ge::DT_INT32) {
        OPS_LOG_E(opName_, "actualSeqLengthsQ's dtype is %s, it should be DT_INT32.",
            FusedScatterCopySparseFlashAttentionDataTypeToSerialString(opParamInfo_.actualSeqLengthsQ.desc->GetDataType()).c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckActualSeqLensQShape()
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t shapeSize = 0;
    if (GetActualSeqLenSize(shapeSize, opParamInfo_.actualSeqLengthsQ.tensor, qLayout_, "actualSeqLengthsQ") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (shapeSize != bSize_) {
        OPS_LOG_E(opName_, "actualSeqLengthsQ shape size is %u, it should be equal to batch size[%u]",
            shapeSize, bSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckActualSeqLens()
{
    if (std::string(opParamInfo_.layoutKV) == "TND" && opParamInfo_.actualSeqLengths.tensor == nullptr) {
        OPS_LOG_E(opName_,
                  "when the layout of key and value is TND, "
                  "the actualSeqLengths of key and value should not be empty.");
        return ge::GRAPH_PARAM_INVALID;
    }
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensDType() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckActualSeqLensDType()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.actualSeqLengths.desc == nullptr) {
        OPS_LOG_E(opName_, "actualSeqLengths is not empty,"
            "but actualSeqLengths's dtype is nullptr.");
            return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.actualSeqLengths.desc->GetDataType() != ge::DT_INT32) {
        OPS_LOG_E(opName_, "actualSeqLengths's dtype is %s, it should be DT_INT32.",
            FusedScatterCopySparseFlashAttentionDataTypeToSerialString(opParamInfo_.actualSeqLengths.desc->GetDataType()).c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckActualSeqLensShape()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t shapeSize = 0;
    if(GetActualSeqLenSize(shapeSize, opParamInfo_.actualSeqLengths.tensor, kvLayout_, "actualSeqLengths") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (shapeSize != bSize_) {
        OPS_LOG_E(opName_, "actualSeqLengths shape size is %u, it should be equal to batch size[%u].",
            shapeSize, bSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckMultiParaConsistency()
{
    SetFusedScatterCopySparseFlashAttentionShapeCompare();
    if (ge::GRAPH_SUCCESS != CheckVAndKRope() ||
        ge::GRAPH_SUCCESS != CheckQRope() ||
        ge::GRAPH_SUCCESS != CheckTopK() ||
        ge::GRAPH_SUCCESS != CheckAttenOut() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQ() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckBlockTable()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckFeatureMlaNoQuantShape() const
{
    OPS_ERR_IF(bSize_ <= 0,
        OPS_LOG_E(opName_, "batch_size should be greater than 0, but got %u", bSize_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(qTSize_ <= 0 && (qLayout_ == FusedScatterCopySparseFlashAttentionLayout::TND),
        OPS_LOG_E(opName_, "T_size of query should be greater than 0, but got %u", qTSize_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(n1Size_ <= 0,
        OPS_LOG_E(opName_, "q_head_num should be greater than 0, but got %u", n1Size_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(n2Size_ != 1,
        OPS_LOG_E(opName_, "kv_head_num should be 1, but got %u", n2Size_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(n1Size_ % n2Size_ != 0,
        OPS_LOG_E(opName_, "q_head_num(%u) must be divisible by kv_head_num(%u)", n1Size_, n2Size_),
        return ge::GRAPH_FAILED);

    std::vector<uint32_t> gSizeSupportList = {1, 2, 4, 8, 16, 32, 64, 128};
    OPS_ERR_IF(std::find(gSizeSupportList.begin(), gSizeSupportList.end(), gSize_) == gSizeSupportList.end(),
        OPS_LOG_E(opName_, "group num should be in 1, 2, 4, 8, 16, 32, 64, 128, but got %u", gSize_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(qkHeadDim_ != 512,
        OPS_LOG_E(opName_, "qk_head_dim only support 512, but got %u", qkHeadDim_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(qkHeadDim_ != vHeadDim_,
        OPS_LOG_E(opName_, "qk_head_dim[%u] should be equal to v_head_dim[%u]", qkHeadDim_, vHeadDim_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(ropeHeadDim_ != 64,
        OPS_LOG_E(opName_, "rope_head_dim should be 64, but got %u", ropeHeadDim_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckFeatureMlaNoQuantLayout() const
{
    const std::vector<std::string> layoutSupportList = {
        "BSND",
        "TND"
    };
    std::string layoutQuery = opParamInfo_.layoutQuery;
    OPS_ERR_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layoutQuery) == layoutSupportList.end(),
        OPS_LOG_E(opName_, "layoutQuery only supports BSND/TND, but got %s", layoutQuery.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckFeatureMlaNoQuantDtype() const
{
    OPS_ERR_IF(inputQType_ != ge::DT_BF16 && inputQType_ != ge::DT_FLOAT16,
        OPS_LOG_E(opName_, "query dtype only support %s and %s, but got %s",
            FusedScatterCopySparseFlashAttentionDataTypeToSerialString(ge::DT_BF16).c_str(), FusedScatterCopySparseFlashAttentionDataTypeToSerialString(ge::DT_FLOAT16).c_str(),
            FusedScatterCopySparseFlashAttentionDataTypeToSerialString(inputQType_).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckFeatureMlaNoquantPa() const
{
    if (kvStorageMode_ != FusedScatterCopySparseFlashAttentionKvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }

    OPS_ERR_IF(blockSize_ <= 0 || blockSize_ > static_cast<int32_t>(MAX_BLOCK_SIZE),
        OPS_LOG_E(opName_, "when page attention is enabled, block_size(%d) should be in range (0, %u].",
        blockSize_, MAX_BLOCK_SIZE), return ge::GRAPH_FAILED);

    OPS_ERR_IF(blockSize_ % 16 > 0,
        OPS_LOG_E(opName_, "when page attention is enabled, block_size(%d) should be 16-aligned.",
        blockSize_), return ge::GRAPH_FAILED);

    OPS_ERR_IF(blockSize_ % sparseBlockSize_ > 0,
        OPS_LOG_E(opName_, "when page attention is enabled, block_size(%d) must be divided by sparse_block_size(%d), but now the remainder is %d.",
        blockSize_, sparseBlockSize_, blockSize_ % sparseBlockSize_), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckFeatureMlaNoquant() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantShape() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoquantPa()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckFeatureMla() const
{
    return CheckFeatureMlaNoquant();
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckFeature() const
{
    return CheckFeatureMla();
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::CheckDecodeOnly() const
{
    OPS_ERR_IF(qLayout_ != FusedScatterCopySparseFlashAttentionLayout::TND || kvLayout_ != FusedScatterCopySparseFlashAttentionLayout::PA_BSND,
        OPS_LOG_E(opName_, "only layout_query=TND and layout_kv=PA_BSND are supported"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(sparseBlockSize_ != 1 || *opParamInfo_.sparseMode != 3,
        OPS_LOG_E(opName_, "sparse_block_size must be 1 and sparse_mode must be 3"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(n1Size_ == 0 || n1Size_ > 128 || n2Size_ != 1 || gSize_ != n1Size_,
        OPS_LOG_E(opName_, "decode requires 1 <= N1 <= 128, N2=1 and group size=N1"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(qkHeadDim_ != 512 || vHeadDim_ != 512 || ropeHeadDim_ != 64,
        OPS_LOG_E(opName_, "decode requires qk/value head dim 512 and rope head dim 64"),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(blockSize_ != 128,
        OPS_LOG_E(opName_, "PA block size must be 128, but got %ld", blockSize_),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(sparseBlockCount_ != OFFLOAD_SPARSE_INDICES_CAPACITY,
        OPS_LOG_E(opName_, "sparse_indices last dim must be %u, but got %u",
                  OFFLOAD_SPARSE_INDICES_CAPACITY, sparseBlockCount_),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(qTSize_ != bSize_ &&
               qTSize_ != bSize_ * OFFLOAD_MTP3_QUERY_COUNT,
        OPS_LOG_E(opName_, "decode TND query token count must be B or 4B"),
        return ge::GRAPH_FAILED);

    const gert::Shape &cacheTokensShape = opParamInfo_.cacheTokens.shape->GetStorageShape();
    OPS_ERR_IF(cacheTokensShape.GetDimNum() != 1 || cacheTokensShape.GetDim(0) != bSize_,
        OPS_LOG_E(opName_, "cache_tokens must have shape [B]"), return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.cacheTokens.desc->GetDataType() != ge::DT_INT32,
        OPS_LOG_E(opName_, "cache_tokens must be int32"), return ge::GRAPH_FAILED);

    const ge::DataType dtype = opParamInfo_.query.desc->GetDataType();
    OPS_ERR_IF(opParamInfo_.key.desc->GetDataType() != dtype ||
               opParamInfo_.value.desc->GetDataType() != dtype ||
               opParamInfo_.queryRope.desc->GetDataType() != dtype ||
               opParamInfo_.keyRope.desc->GetDataType() != dtype || outputType_ != dtype,
        OPS_LOG_E(opName_, "all floating-point inputs and output must have the same dtype"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void FusedScatterCopySparseFlashAttentionTilingCheck::Init()
{
    opName_ = sfaInfo_.opName;
    platformInfo_ = sfaInfo_.platformInfo;
    opParamInfo_ = sfaInfo_.opParamInfo;
    socVersion_ = sfaInfo_.socVersion;

    bSize_ = sfaInfo_.bSize;
    n1Size_ = sfaInfo_.n1Size;
    n2Size_ = sfaInfo_.n2Size;
    s1Size_ = sfaInfo_.s1Size;
    s2Size_ = sfaInfo_.s2Size;
    gSize_ = sfaInfo_.gSize;
    qkHeadDim_ = sfaInfo_.qkHeadDim;
    vHeadDim_ = sfaInfo_.vHeadDim;
    ropeHeadDim_ = sfaInfo_.ropeHeadDim;
    maxBlockNumPerBatch_ = sfaInfo_.maxBlockNumPerBatch;
    qTSize_ = sfaInfo_.qTSize;
    kvTSize_ = sfaInfo_.kvTSize;
    blockSize_ = sfaInfo_.blockSize;
    sparseBlockCount_ = sfaInfo_.sparseBlockCount;
    sparseBlockSize_ = sfaInfo_.sparseBlockSize;

    inputQType_ = sfaInfo_.inputQType;
    inputKvType_ = sfaInfo_.inputKvType;
    inputQRopeType_ = sfaInfo_.inputQRopeType;
    inputKRopeType_ = sfaInfo_.inputKRopeType;
    outputType_ = sfaInfo_.outputType;

    qLayout_ = sfaInfo_.qLayout;
    topkLayout_ = sfaInfo_.topkLayout;
    kvLayout_ = sfaInfo_.kvLayout;
    outLayout_ = sfaInfo_.outLayout;

    kvStorageMode_ = sfaInfo_.kvStorageMode;
    l2CacheSize_ = sfaInfo_.l2CacheSize;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionTilingCheck::Process()
{
    Init();
    if (CheckDecodeOnly() != ge::GRAPH_SUCCESS ||
        CheckSinglePara() != ge::GRAPH_SUCCESS ||
        CheckParaExistence() != ge::GRAPH_SUCCESS ||
        CheckFeature() != ge::GRAPH_SUCCESS ||
        CheckMultiParaConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool FusedScatterCopySparseFlashAttentionInfoParser::HasAxis(const FusedScatterCopySparseFlashAttentionAxis &axis, const FusedScatterCopySparseFlashAttentionLayout &layout, const gert::Shape &shape) const
{
    const auto& layoutIt = FusedScatterCopySparseFlashAttention_LAYOUT_AXIS_MAP.find(layout);
    if (layoutIt == FusedScatterCopySparseFlashAttention_LAYOUT_AXIS_MAP.end()) {
        return false;
    }

    const std::vector<FusedScatterCopySparseFlashAttentionAxis>& axes = layoutIt->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    if (axisIt == axes.end()) {
        return false;
    }
    const auto& dimIt = FusedScatterCopySparseFlashAttention_LAYOUT_DIM_MAP.find(layout);
    if (dimIt == FusedScatterCopySparseFlashAttention_LAYOUT_DIM_MAP.end() || dimIt->second != shape.GetDimNum()) {
        return false;
    }
    return true;
}

size_t FusedScatterCopySparseFlashAttentionInfoParser::GetAxisIdx(const FusedScatterCopySparseFlashAttentionAxis &axis, const FusedScatterCopySparseFlashAttentionLayout &layout) const
{
    const std::vector<FusedScatterCopySparseFlashAttentionAxis>& axes = FusedScatterCopySparseFlashAttention_LAYOUT_AXIS_MAP.find(layout)->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    return std::distance(axes.begin(), axisIt);
}

uint32_t FusedScatterCopySparseFlashAttentionInfoParser::GetAxisNum(const gert::Shape &shape, const FusedScatterCopySparseFlashAttentionAxis &axis,const FusedScatterCopySparseFlashAttentionLayout &layout) const
{
    return HasAxis(axis, layout, shape) ? shape.GetDim(GetAxisIdx(axis, layout)) : invalidDimValue_;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::CheckRequiredInOutExistence() const
{
    OPS_ERR_IF(opParamInfo_.query.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.query.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.key.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.key.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.value.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.value.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.sparseIndices.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor sparseIndices is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.sparseIndices.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor sparseIndices is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.cacheTokens.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor cacheTokens is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.cacheTokens.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor cacheTokens is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.attenOut.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.attenOut.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.queryRope.tensor == nullptr, OPS_LOG_E(opName_, "Shape of queryRope is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.queryRope.desc == nullptr, OPS_LOG_E(opName_, "Desc of queryRope is nullptr"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::CheckRequiredAttrExistence() const
{
    OPS_ERR_IF(opParamInfo_.layoutQuery == nullptr, OPS_LOG_E(opName_, "attr layoutQuery is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.layoutKV == nullptr, OPS_LOG_E(opName_, "attr layoutKV is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.sparseBlockSize == nullptr, OPS_LOG_E(opName_, "attr sparseBlockSize is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.scaleValue == nullptr, OPS_LOG_E(opName_, "attr scaleValue is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.sparseMode == nullptr, OPS_LOG_E(opName_, "attr sparseMode is nullptr"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS ||
        CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    FusedScatterCopySparseFlashAttentionLayout &layout, const std::string &name)
{
    if ((tensor == nullptr)) {
        OPS_LOG_E(opName_, "when layout of query is %s, %s must be provided.",
            FusedScatterCopySparseFlashAttentionLayoutToSerialString(layout).c_str(), name.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OPS_LOG_E(opName_, "the shape size of %s is %ld, it should be greater than 0.",
            name.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetActualSeqLenQSize(uint32_t &size)
{
    return GetActualSeqLenSize(size, opParamInfo_.actualSeqLengthsQ.tensor, qLayout_, "actualSeqLengthsQ");
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OPS_LOG_E("FusedScatterCopySparseFlashAttention", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo_ == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OPS_ERR_IF(aicNum == 0 || aivNum == 0,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "num of core obtained is 0."), return GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    if (socVersion_ != platform_ascendc::SocVersion::ASCEND910B &&
        socVersion_ != platform_ascendc::SocVersion::ASCEND910_93) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "SOC Version[%d] is not support.", (int32_t)socVersion_);
        return GRAPH_FAILED;
    }

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize_);

    return ge::GRAPH_SUCCESS;
}

void FusedScatterCopySparseFlashAttentionInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.blockTable.tensor = context_->GetInputTensor(BLOCK_TABLE_INPUT_INDEX);
    opParamInfo_.actualSeqLengthsQ.tensor = context_->GetInputTensor(ACT_SEQ_LEN_Q_INPUT_INDEX);
    opParamInfo_.actualSeqLengthsQ.desc = context_->GetInputDesc(ACT_SEQ_LEN_Q_INPUT_INDEX);
    opParamInfo_.actualSeqLengths.tensor = context_->GetInputTensor(ACT_SEQ_LEN_KV_INPUT_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetInputDesc(ACT_SEQ_LEN_KV_INPUT_INDEX);
    opParamInfo_.queryRope.tensor = context_->GetInputTensor(QUERY_ROPE_INPUT_INDEX);
    opParamInfo_.queryRope.desc = context_->GetInputDesc(QUERY_ROPE_INPUT_INDEX);
    opParamInfo_.keyRope.tensor = context_->GetInputTensor(KEY_ROPE_INPUT_INDEX);
    opParamInfo_.keyRope.desc = context_->GetInputDesc(KEY_ROPE_INPUT_INDEX);
}

void FusedScatterCopySparseFlashAttentionInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INPUT_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INPUT_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INPUT_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INPUT_INDEX);
    opParamInfo_.value.desc = context_->GetInputDesc(VALUE_INPUT_INDEX);
    opParamInfo_.value.shape = context_->GetInputShape(VALUE_INPUT_INDEX);
    opParamInfo_.sparseIndices.desc = context_->GetInputDesc(SPARSE_INDICES_INPUT_INDEX);
    opParamInfo_.sparseIndices.shape = context_->GetInputShape(SPARSE_INDICES_INPUT_INDEX);
    opParamInfo_.cacheTokens.desc = context_->GetInputDesc(CACHE_TOKENS_INPUT_INDEX);
    opParamInfo_.cacheTokens.shape = context_->GetInputShape(CACHE_TOKENS_INPUT_INDEX);
    GetOptionalInputParaInfo();
}

void FusedScatterCopySparseFlashAttentionInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(OUTPUT_INDEX);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(OUTPUT_INDEX);
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);

    opParamInfo_.layoutQuery = attrs->GetStr(LAYOUT_QUERY_ATTR_INDEX);
    opParamInfo_.layoutKV = attrs->GetStr(LAYOUT_KV_ATTR_INDEX);
    opParamInfo_.sparseBlockSize = attrs->GetAttrPointer<int64_t>(SPARSE_BLOCK_SIZE_ATTR_INDEX);
    opParamInfo_.scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    opParamInfo_.sparseMode = attrs->GetAttrPointer<int64_t>(SPARSE_MODE_ATTR_INDEX);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKvType_ = opParamInfo_.key.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();
    if (opParamInfo_.queryRope.desc != nullptr) {
        inputQRopeType_ = opParamInfo_.queryRope.desc->GetDataType();
    }
    if (opParamInfo_.keyRope.desc != nullptr) {
        inputKRopeType_ = opParamInfo_.keyRope.desc->GetDataType();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetBatchSize()
{
    if (qLayout_ == FusedScatterCopySparseFlashAttentionLayout::TND) {
        return GetActualSeqLenQSize(bSize_);
    } else { // BSND
        bSize_ = GetAxisNum(queryShape_, FusedScatterCopySparseFlashAttentionAxis::B, qLayout_);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetQTSize()
{
    qTSize_ = (qLayout_ == FusedScatterCopySparseFlashAttentionLayout::TND) ? GetAxisNum(queryShape_, FusedScatterCopySparseFlashAttentionAxis::T, qLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetKVTSize()
{
    kvTSize_ = (kvLayout_ == FusedScatterCopySparseFlashAttentionLayout::TND) ? GetAxisNum(keyShape_, FusedScatterCopySparseFlashAttentionAxis::T, kvLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetQkHeadDim()
{
    qkHeadDim_ = GetAxisNum(queryShape_, FusedScatterCopySparseFlashAttentionAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetS1Size()
{
    if (qLayout_ == FusedScatterCopySparseFlashAttentionLayout::TND) {
        s1Size_ = GetAxisNum(queryShape_, FusedScatterCopySparseFlashAttentionAxis::T, qLayout_);
        return ge::GRAPH_SUCCESS;
    } else { // BSND
        s1Size_ = GetAxisNum(queryShape_, FusedScatterCopySparseFlashAttentionAxis::S, qLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetFusedScatterCopySparseFlashAttentionKvStorageMode()
{
    if (kvLayout_ == FusedScatterCopySparseFlashAttentionLayout::PA_BSND) {
        kvStorageMode_ = FusedScatterCopySparseFlashAttentionKvStorageMode::PAGE_ATTENTION;
    } else {
        kvStorageMode_ = FusedScatterCopySparseFlashAttentionKvStorageMode::BATCH_CONTINUOUS;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetKvLayout()
{
    const map<string, FusedScatterCopySparseFlashAttentionLayout> layoutKVMap = {
        {"BSND",        FusedScatterCopySparseFlashAttentionLayout::BSND},
        {"PA_BSND",     FusedScatterCopySparseFlashAttentionLayout::PA_BSND},
        {"TND",         FusedScatterCopySparseFlashAttentionLayout::TND}
    };

    std::string layout(opParamInfo_.layoutKV);
    auto it = layoutKVMap.find(layout);
    if (it != layoutKVMap.end()) {
        kvLayout_ = it->second;
    } else {
        OPS_LOG_E(opName_, "layoutKV is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    if (kvLayout_ != FusedScatterCopySparseFlashAttentionLayout::PA_BSND && qLayout_ != kvLayout_) {
        OPS_LOG_E(opName_, "When layoutKV is not PA_BSND, layoutKV must be the same as layoutQ.");
        return ge::GRAPH_FAILED;
    }
    uint32_t keyDimNum = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    if (kvLayout_ == FusedScatterCopySparseFlashAttentionLayout::PA_BSND && keyDimNum != 4U) {
        OPS_LOG_E(opName_, "When layoutKV is PA_BSND, kvDimNum must be 4, but now is %d.", keyDimNum);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetS2SizeForBatchContinuous()
{
    if (kvLayout_ == FusedScatterCopySparseFlashAttentionLayout::BSND) { // BSND
        s2Size_ = GetAxisNum(keyShape_, FusedScatterCopySparseFlashAttentionAxis::S, kvLayout_);
    } else if (kvLayout_ == FusedScatterCopySparseFlashAttentionLayout::TND) {
        s2Size_ = GetAxisNum(keyShape_, FusedScatterCopySparseFlashAttentionAxis::T, kvLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetMaxBlockNumPerBatch()
{
    if (opParamInfo_.blockTable.tensor == nullptr) {
        OPS_LOG_E(opName_, "the layout_kv is %s, blockTable must be provided.", FusedScatterCopySparseFlashAttentionLayoutToSerialString(kvLayout_).c_str());
        return ge::GRAPH_FAILED;
    }
    uint32_t dimNum = opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum();
    if (dimNum != DIM_NUM_TWO) {
        OPS_LOG_E(opName_, "the dim num of block_table is %u, it should be %u.", dimNum, DIM_NUM_TWO);
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1) <= 0) {
        OPS_LOG_E(opName_, "%s's second dimension(%ld) should be greater than 0",
            BLOCK_TABLE_NAME.c_str(), opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1));
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetBlockSize()
{
    blockSize_ = GetAxisNum(keyShape_, FusedScatterCopySparseFlashAttentionAxis::Bs, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetSparseBlockCount()
{
    sparseBlockCount_ = GetAxisNum(sparseIndicesShape_, FusedScatterCopySparseFlashAttentionAxis::K, qLayout_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetS2SizeForPageAttention()
{
    if (GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS || GetBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    s2Size_ = maxBlockNumPerBatch_ * blockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetS2Size()
{
    if (kvStorageMode_ == FusedScatterCopySparseFlashAttentionKvStorageMode::BATCH_CONTINUOUS) {
        return GetS2SizeForBatchContinuous();
    }
    return GetS2SizeForPageAttention();
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetValueHeadDim()
{
    vHeadDim_ = GetAxisNum(valueShape_, FusedScatterCopySparseFlashAttentionAxis::D, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetRopeHeadDim()
{
    ropeHeadDim_ = GetAxisNum(queryRopeShape_, FusedScatterCopySparseFlashAttentionAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetQueryAndOutLayout()
{
    const map<string, pair<FusedScatterCopySparseFlashAttentionLayout, FusedScatterCopySparseFlashAttentionLayout>> layoutMap = {
        {"BSND",        {FusedScatterCopySparseFlashAttentionLayout::BSND,    FusedScatterCopySparseFlashAttentionLayout::BSND}},
        {"TND",         {FusedScatterCopySparseFlashAttentionLayout::TND,     FusedScatterCopySparseFlashAttentionLayout::TND }},
    };

    std::string layout(opParamInfo_.layoutQuery);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second.first;
        outLayout_ = it->second.second;
    } else {
        OPS_LOG_E(opName_, "layoutQuery is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetTopkLayout()
{
    topkLayout_ = qLayout_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetN1Size()
{
    n1Size_ = GetAxisNum(queryShape_, FusedScatterCopySparseFlashAttentionAxis::N, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetN2Size()
{
    n2Size_ = GetAxisNum(keyShape_, FusedScatterCopySparseFlashAttentionAxis::N, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

void FusedScatterCopySparseFlashAttentionInfoParser::SetFusedScatterCopySparseFlashAttentionShape()
{
    queryShape_ = opParamInfo_.query.shape->GetStorageShape();
    keyShape_ = opParamInfo_.key.shape->GetStorageShape();
    valueShape_ = opParamInfo_.value.shape->GetStorageShape();
    sparseIndicesShape_ = opParamInfo_.sparseIndices.shape->GetStorageShape();
    queryRopeShape_ = opParamInfo_.queryRope.tensor->GetStorageShape();
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetGSize()
{
    if (n2Size_ != 0) {
        gSize_ = n1Size_ / n2Size_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::GetActualseqInfo()
{
    maxActualseq_ = static_cast<uint32_t>(s2Size_);
    if (opParamInfo_.actualSeqLengths.tensor != nullptr) {
        actualLenDimsKV_ = opParamInfo_.actualSeqLengths.tensor->GetShapeSize();
    }
    if (opParamInfo_.actualSeqLengthsQ.tensor != nullptr) {
        actualLenDimsQ_ = opParamInfo_.actualSeqLengthsQ.tensor->GetShapeSize();
    }
    return ge::GRAPH_SUCCESS;
}

void FusedScatterCopySparseFlashAttentionInfoParser::GenerateInfo(FusedScatterCopySparseFlashAttentionTilingInfo &sfaInfo)
{
    sfaInfo.opName = opName_;
    sfaInfo.platformInfo = platformInfo_;
    sfaInfo.opParamInfo = opParamInfo_;
    sfaInfo.socVersion = socVersion_;

    sfaInfo.bSize = bSize_;
    sfaInfo.n1Size = n1Size_;
    sfaInfo.n2Size = n2Size_;
    sfaInfo.s1Size = s1Size_;
    sfaInfo.s2Size = s2Size_;
    sfaInfo.gSize = gSize_;
    sfaInfo.qkHeadDim = qkHeadDim_;
    sfaInfo.vHeadDim = vHeadDim_;
    sfaInfo.ropeHeadDim = ropeHeadDim_;
    sfaInfo.qTSize = qTSize_;
    sfaInfo.kvTSize = kvTSize_;
    sfaInfo.sparseBlockSize = *opParamInfo_.sparseBlockSize;
    sfaInfo.sparseBlockCount = sparseBlockCount_;

    sfaInfo.inputQType = inputQType_;
    sfaInfo.inputKvType = inputKvType_;
    sfaInfo.inputQRopeType = inputQRopeType_;
    sfaInfo.inputKRopeType = inputKRopeType_;
    sfaInfo.outputType = outputType_;

    sfaInfo.kvStorageMode = kvStorageMode_;
    sfaInfo.l2CacheSize = l2CacheSize_;

    sfaInfo.totalBlockNum = opParamInfo_.key.shape->GetStorageShape().GetDim(0);
    sfaInfo.scaleValue = *opParamInfo_.scaleValue;
    sfaInfo.pageAttentionFlag = (kvStorageMode_ == FusedScatterCopySparseFlashAttentionKvStorageMode::PAGE_ATTENTION);
    sfaInfo.blockSize = blockSize_;
    sfaInfo.blockTypeSize =  sizeof(float);
    sfaInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    sfaInfo.actualLenDimsQ = actualLenDimsQ_;
    sfaInfo.actualLenDimsKV = actualLenDimsKV_;
    sfaInfo.maxActualseq = maxActualseq_;
    sfaInfo.actualSeqLenFlag = (opParamInfo_.actualSeqLengths.tensor != nullptr);
    sfaInfo.isSameSeqAllKVTensor = isSameSeqAllKVTensor_;
    sfaInfo.isSameActualseq = isSameActualseq_;

    sfaInfo.sparseMode = *opParamInfo_.sparseMode;

    sfaInfo.qLayout = qLayout_;
    sfaInfo.topkLayout = topkLayout_;
    sfaInfo.kvLayout = kvLayout_;
    sfaInfo.outLayout = outLayout_;
}

ge::graphStatus FusedScatterCopySparseFlashAttentionInfoParser::Parse(FusedScatterCopySparseFlashAttentionTilingInfo &sfaInfo)
{
    if (context_ == nullptr) {
        OPS_LOG_E("FusedScatterCopySparseFlashAttention", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }
    OPS_LOG_FULL(DLOG_INFO, "FusedScatterCopySparseFlashAttention", "TilingContext: %s", FusedScatterCopySparseFlashAttentionDebugTilingContext(context_).c_str());
    if (ge::GRAPH_SUCCESS != GetOpName() ||
        ge::GRAPH_SUCCESS != GetNpuInfo() ||
        ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetInOutDataType() ||
        ge::GRAPH_SUCCESS != GetQueryAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetTopkLayout() ||
        ge::GRAPH_SUCCESS != GetKvLayout() ||
        ge::GRAPH_SUCCESS != GetFusedScatterCopySparseFlashAttentionKvStorageMode()) {
        return ge::GRAPH_FAILED;
    }

    SetFusedScatterCopySparseFlashAttentionShape();
    if (
        ge::GRAPH_SUCCESS != GetN1Size() ||
        ge::GRAPH_SUCCESS != GetN2Size() ||
        ge::GRAPH_SUCCESS != GetGSize() ||
        ge::GRAPH_SUCCESS != GetBatchSize() ||
        ge::GRAPH_SUCCESS != GetQTSize() ||
        ge::GRAPH_SUCCESS != GetKVTSize() ||
        ge::GRAPH_SUCCESS != GetS1Size() ||
        ge::GRAPH_SUCCESS != GetQkHeadDim() ||
        ge::GRAPH_SUCCESS != GetS2Size() ||
        ge::GRAPH_SUCCESS != GetValueHeadDim() ||
        ge::GRAPH_SUCCESS != GetRopeHeadDim() ||
        ge::GRAPH_SUCCESS != GetSparseBlockCount()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetActualseqInfo()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(sfaInfo);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
