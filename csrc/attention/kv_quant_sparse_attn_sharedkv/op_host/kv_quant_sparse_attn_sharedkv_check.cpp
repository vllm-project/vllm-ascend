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
 * \file kv_quant_sparse_attn_sharedkv_check.cpp
 * \brief
 */

#include "kv_quant_sparse_attn_sharedkv_check.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
using std::pair;
namespace optiling {

std::string SASDataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("KvQuantSparseAttnSharedkv ", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

std::string KvQuantSASLayoutToSerialString(SASLayout layout)
{
    switch (layout) {
        case SASLayout::BSND: return "BSND";
        case SASLayout::TND: return "TND";
        case SASLayout::PA_ND: return "PA_ND";
        default: return "UNKNOWN";
    }
}

std::string GetShapeStr(gert::Shape shape)
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

bool KvQuantSASTilingCheck::HasAxis(const SASAxis &axis, const SASLayout &layout, const gert::Shape &shape) const
{
    const auto& layoutIt = SAS_LAYOUT_AXIS_MAP.find(layout);
    if (layoutIt == SAS_LAYOUT_AXIS_MAP.end()) {
        return false;
    }

    const std::vector<SASAxis>& axes = layoutIt->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    if (axisIt == axes.end()) {
        return false;
    }
    const auto& dimIt = SAS_LAYOUT_DIM_MAP.find(layout);
    if (dimIt == SAS_LAYOUT_DIM_MAP.end() || dimIt->second != shape.GetDimNum()) {
        return false;
    }
    return true;
}

size_t KvQuantSASTilingCheck::GetAxisIdx(const SASAxis &axis, const SASLayout &layout) const
{
    const std::vector<SASAxis>& axes = SAS_LAYOUT_AXIS_MAP.find(layout)->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    return std::distance(axes.begin(), axisIt);
}

uint32_t KvQuantSASTilingCheck::GetAxisNum(const gert::Shape &shape, const SASAxis &axis,const SASLayout &layout) const
{
    return HasAxis(axis, layout, shape) ? shape.GetDim(GetAxisIdx(axis, layout)) : invalidDimValue_;
}

void KvQuantSASTilingCheck::Init()
{
    opName_ = sasInfo_.opName;
    platformInfo_ = sasInfo_.platformInfo;
    opParamInfo_ = sasInfo_.opParamInfo;
    socVersion_ = sasInfo_.socVersion;

    bSize_ = sasInfo_.bSize;
    bSize_ = opParamInfo_.oriBlockTable.tensor->GetShape().GetStorageShape().GetDim(0);
    n1Size_ = sasInfo_.n1Size;
    n2Size_ = sasInfo_.n2Size;
    s1Size_ = sasInfo_.s1Size;
    s2Size_ = sasInfo_.s2Size;
    gSize_ = sasInfo_.gSize;
    qkHeadDim_ = sasInfo_.qkHeadDim;
    qTSize_ = sasInfo_.qTSize;

    dSize_ = sasInfo_.dSize;
    dSizeV_ = sasInfo_.dSizeV;
    if (opParamInfo_.oriKv.tensor != nullptr) {
        dSizeOriKvInput_ = GetAxisNum(opParamInfo_.oriKv.tensor->GetStorageShape(), SASAxis::D, kvLayout_);
    }
    if (opParamInfo_.cmpKv.tensor != nullptr) {
        dSizeCmpKvInput_ = GetAxisNum(opParamInfo_.cmpKv.tensor->GetStorageShape(), SASAxis::D, kvLayout_);
    }

    actualLenDimsQ_ = sasInfo_.actualLenDimsQ;
    maxActualseq_ = sasInfo_.maxActualseq;

    ropeHeadDim_ = sasInfo_.ropeHeadDim;
    oriMaxBlockNumPerBatch_ = sasInfo_.oriMaxBlockNumPerBatch;
    cmpMaxBlockNumPerBatch_ = sasInfo_.cmpMaxBlockNumPerBatch;

    oriBlockSize_ = sasInfo_.oriBlockSize;
    cmpBlockSize_ = sasInfo_.cmpBlockSize;

    sparseBlockCount_ = sasInfo_.sparseBlockCount;
    sparseBlockSize_ = sasInfo_.sparseBlockSize;

    tileSize_ = sasInfo_.tileSize;

    cmpRatio_ = sasInfo_.cmpRatio;
    oriWinLeft_ = sasInfo_.oriWinLeft;
    oriWinRight_ = sasInfo_.oriWinRight;

    oriMaskMode_ = sasInfo_.oriMaskMode;
    cmpMaskMode_ = sasInfo_.cmpMaskMode;

    qType_ = sasInfo_.qType;
    oriKvType_ = sasInfo_.oriKvType;
    cmpKvType_ = sasInfo_.cmpKvType;
    outputType_ = sasInfo_.outputType;

    qLayout_ = sasInfo_.qLayout;
    kvLayout_ = sasInfo_.kvLayout;
    outLayout_ = sasInfo_.outLayout;

    if (opParamInfo_.cmpKv.tensor == nullptr) {
        perfMode_ = SASTemplateMode::SWA_TEMPLATE_MODE;
    } else if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
        perfMode_ = SASTemplateMode::SCFA_TEMPLATE_MODE;
    } else {
        perfMode_ = SASTemplateMode::CFA_TEMPLATE_MODE;
    }
}

ge::graphStatus KvQuantSASTilingCheck::Process()
{
    Init();
    if (CheckSinglePara() != ge::GRAPH_SUCCESS ||
        CheckParaExistence() != ge::GRAPH_SUCCESS ||
        CheckFeature() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

}