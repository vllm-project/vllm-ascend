/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_attn_sharedkv_tiling.cpp
 * \brief
 */

#include "sparse_attn_sharedkv_tiling.h"
#include "../op_kernel/sparse_attn_sharedkv_template_tiling_key.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
using std::pair;
namespace optiling {

static const std::string QUERY_NAME = "query";
static const std::string ORI_KV_NAME = "ori_kv";
static const std::string CMP_KV_NAME = "cmp_kv";
static const std::string ORI_SPARSE_INDICES = "ori_sparse_indices";
static const std::string CMP_SPARSE_INDICES = "cmp_sparse_indices";
static const std::string ORI_BLOCK_TABLE_NAME = "ori_block_table";
static const std::string CMP_BLOCK_TABLE_NAME = "cmp_block_table";
static const std::string SINKS_NAME = "sinks";
static const std::string METADATA_NAME = "metadata";
static const std::string ATTEN_OUT_NAME = "attn_out";
const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME,                  {ge::DT_FLOAT16, ge::DT_BF16}},
    {ORI_KV_NAME,                    {ge::DT_FLOAT16, ge::DT_BF16}},
    {CMP_KV_NAME,                  {ge::DT_FLOAT16, ge::DT_BF16}},
    {ORI_SPARSE_INDICES,             {ge::DT_INT32}},
    {CMP_SPARSE_INDICES,               {ge::DT_INT32}},
    {ATTEN_OUT_NAME,              {ge::DT_FLOAT16, ge::DT_BF16}},
    {ORI_BLOCK_TABLE_NAME,         {ge::DT_INT32}},
    {CMP_BLOCK_TABLE_NAME,            {ge::DT_INT32}},
    {SINKS_NAME,                    {ge::DT_FLOAT}},
    {METADATA_NAME,                    {ge::DT_INT32}}
};

const std::map<std::string, std::vector<SASLayout>> LAYOUT_SUPPORT_MAP = {
    {QUERY_NAME,            {SASLayout::BSND, SASLayout::TND}},
    {ORI_KV_NAME,               {SASLayout::PA_ND, SASLayout::TND}},
    {CMP_KV_NAME,             {SASLayout::PA_ND, SASLayout::TND}},
    {ATTEN_OUT_NAME,         {SASLayout::BSND, SASLayout::TND}},
    {ORI_SPARSE_INDICES,         {SASLayout::BSND, SASLayout::TND}},
    {CMP_SPARSE_INDICES,         {SASLayout::BSND, SASLayout::TND}},
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

std::string SASLayoutToSerialString(SASLayout layout)
{
    switch (layout) {
        case SASLayout::BSND: return "BSND";
        case SASLayout::TND: return "TND";
        case SASLayout::PA_ND: return "PA_ND";
        default: return "UNKNOWN";
    }
}

struct SASCompileInfo {
    int64_t core_num;
};

static const std::map<SASLayout, std::vector<SASAxis>> SAS_LAYOUT_AXIS_MAP = {
    {SASLayout::BSND, {SASAxis::B, SASAxis::S, SASAxis::N, SASAxis::D}},
    {SASLayout::TND, {SASAxis::T, SASAxis::N, SASAxis::D}},
    {SASLayout::PA_ND, {SASAxis::Bn, SASAxis::Bs, SASAxis::N, SASAxis::D}},
};

static const std::map<SASLayout, size_t> SAS_LAYOUT_DIM_MAP = {
    {SASLayout::BSND, DIM_NUM_FOUR},
    {SASLayout::TND, DIM_NUM_THREE},
    {SASLayout::PA_ND, DIM_NUM_FOUR},
};

static std::string SASDataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("sparseAttnSharedkv", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

// --------------------------SASInfoParser类成员函数定义-------------------------------------
ge::graphStatus SASInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.q.shape == nullptr, OP_LOGE(opName_, "Shape of tensor query is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.q.desc == nullptr, OP_LOGE(opName_, "Desc of tensor query is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.oriKv.tensor == nullptr, OP_LOGE(opName_, "tensor of oriKv is nullptr"),
                return ge::GRAPH_FAILED);
    if (perfMode_ == SASTemplateMode::CFA_TEMPLATE_MODE){
        OP_CHECK_IF(opParamInfo_.cmpKv.tensor == nullptr, OP_LOGE(opName_, "tensor of cmpKv is nullptr"),
                    return ge::GRAPH_FAILED);
    }
    if (perfMode_ == SASTemplateMode::SCFA_TEMPLATE_MODE){
        OP_CHECK_IF(opParamInfo_.cmpKv.tensor == nullptr, OP_LOGE(opName_, "tensor of cmpKv is nullptr"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.cmpSparseIndices.tensor == nullptr, OP_LOGE(opName_, "cmpSparseIndices is nullptr"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::CheckRequiredAttrExistence() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS ||
        CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::CheckUnrequiredParaExistence() const
{
    OP_CHECK_IF(opParamInfo_.oriSparseIndices.tensor != nullptr || opParamInfo_.oriSparseIndices.desc != nullptr,
                OP_LOGE(opName_, "Currently, oriSparseIndices must be a nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("SparseAttnSharedkv", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0, OP_LOGE(opName_, "num of core obtained is 0."), return ge::GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    if ((socVersion_ != platform_ascendc::SocVersion::ASCEND910B) &&
        (socVersion_ != platform_ascendc::SocVersion::ASCEND910_93)) {
        OP_LOGE(opName_, "SOC Version[%d] is not support.", (int32_t)socVersion_);
        return GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void SASInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.oriKv.tensor = context_->GetOptionalInputTensor(ORI_KV_INDEX);
    opParamInfo_.oriKv.desc = context_->GetOptionalInputDesc(ORI_KV_INDEX);
    opParamInfo_.cmpKv.tensor = context_->GetOptionalInputTensor(CMP_KV_INDEX);
    opParamInfo_.cmpKv.desc = context_->GetOptionalInputDesc(CMP_KV_INDEX);
    opParamInfo_.oriSparseIndices.tensor = context_->GetOptionalInputTensor(ORI_SPARSE_INDICES_INDEX);
    opParamInfo_.oriSparseIndices.desc = context_->GetOptionalInputDesc(ORI_SPARSE_INDICES_INDEX);
    opParamInfo_.cmpSparseIndices.tensor = context_->GetOptionalInputTensor(CMP_SPARSE_INDICES_INDEX);
    opParamInfo_.cmpSparseIndices.desc = context_->GetOptionalInputDesc(CMP_SPARSE_INDICES_INDEX);
    opParamInfo_.oriBlockTable.tensor = context_->GetOptionalInputTensor(ORI_BLOCK_TABLE_INDEX);
    opParamInfo_.oriBlockTable.desc = context_->GetOptionalInputDesc(ORI_BLOCK_TABLE_INDEX);
    opParamInfo_.cmpBlockTable.tensor = context_->GetOptionalInputTensor(CMP_BLOCK_TABLE_INDEX);
    opParamInfo_.cmpBlockTable.desc = context_->GetOptionalInputDesc(CMP_BLOCK_TABLE_INDEX);
    opParamInfo_.sinks.tensor = context_->GetOptionalInputTensor(SINKS_INDEX);
    opParamInfo_.sinks.desc = context_->GetOptionalInputDesc(SINKS_INDEX);
    opParamInfo_.cuSeqLensQ.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_Q_INDEX);
    opParamInfo_.cuSeqLensQ.desc = context_->GetOptionalInputDesc(CU_SEQLENS_Q_INDEX);
    opParamInfo_.seqUsedQ.tensor = context_->GetOptionalInputTensor(SEQUSED_Q_INDEX);
    opParamInfo_.seqUsedQ.desc = context_->GetOptionalInputDesc(SEQUSED_Q_INDEX);
    opParamInfo_.cuSeqLensKv.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_KV_INDEX);
    opParamInfo_.cuSeqLensKv.desc = context_->GetOptionalInputDesc(CU_SEQLENS_KV_INDEX);
    opParamInfo_.cuSeqLensCmpKv.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_CMP_KV_INDEX);
    opParamInfo_.cuSeqLensCmpKv.desc = context_->GetOptionalInputDesc(CU_SEQLENS_CMP_KV_INDEX);
    opParamInfo_.sequsedKv.tensor = context_->GetOptionalInputTensor(SEQUSED_KV_INDEX);
    opParamInfo_.sequsedKv.desc = context_->GetOptionalInputDesc(SEQUSED_KV_INDEX);
    opParamInfo_.metadata.desc = context_->GetOptionalInputDesc(METADATA_INDEX);
    opParamInfo_.metadata.tensor = context_->GetOptionalInputTensor(METADATA_INDEX);
}

void SASInfoParser::GetInputParaInfo()
{
    opParamInfo_.q.desc = context_->GetInputDesc(Q_INDEX);
    opParamInfo_.q.shape = context_->GetInputShape(Q_INDEX);
    GetOptionalInputParaInfo();
}

void SASInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attnOut.desc = context_->GetOutputDesc(ATTN_OUT_INDEX);
    opParamInfo_.attnOut.shape = context_->GetOutputShape(ATTN_OUT_INDEX);
}

ge::graphStatus SASInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo start");
    opParamInfo_.softmaxScale = attrs->GetAttrPointer<float>(ATTR_SOFTMAX_SCALE_INDEX);
    opParamInfo_.cmpRatio = attrs->GetAttrPointer<uint32_t>(ATTR_CMP_RATIO_INDEX);
    opParamInfo_.oriMaskMode = attrs->GetAttrPointer<uint32_t>(ATTR_ORI_MASK_MODE_INDEX);
    opParamInfo_.cmpMaskMode = attrs->GetAttrPointer<uint32_t>(ATTR_CMP_MASK_MODE_INDEX);
    opParamInfo_.oriWinLeft = attrs->GetAttrPointer<uint32_t>(ATTR_ORI_WIN_LEFT_INDEX);
    opParamInfo_.oriWinRight = attrs->GetAttrPointer<uint32_t>(ATTR_ORI_WIN_RIGHT_INDEX);
    opParamInfo_.layoutQ = attrs->GetStr(ATTR_LAYOUT_Q_INDEX);
    opParamInfo_.layoutKv = attrs->GetStr(ATTR_LAYOUT_KV_INDEX);
    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo end");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetInOutDataType()
{
    qType_ = opParamInfo_.q.desc->GetDataType();
    outputType_ = opParamInfo_.attnOut.desc->GetDataType();
    if (opParamInfo_.oriKv.desc != nullptr) {
        oriKvType_ = opParamInfo_.oriKv.desc->GetDataType();
    }
    if (opParamInfo_.cmpKv.desc != nullptr) {
        cmpKvType_ = opParamInfo_.cmpKv.desc->GetDataType();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetSASTemplateMode(SASTilingInfo &sasInfo)
{
    if (opParamInfo_.oriKv.desc != nullptr) {
        if (opParamInfo_.cmpKv.desc != nullptr && opParamInfo_.cmpSparseIndices.tensor != nullptr) {
            sasInfo.perfMode = SASTemplateMode::SCFA_TEMPLATE_MODE;
        } else if (opParamInfo_.cmpKv.desc != nullptr && opParamInfo_.cmpSparseIndices.tensor == nullptr) {
            sasInfo.perfMode = SASTemplateMode::CFA_TEMPLATE_MODE;
        } else if (opParamInfo_.cmpKv.desc == nullptr && opParamInfo_.cmpSparseIndices.tensor == nullptr) {
            sasInfo.perfMode = SASTemplateMode::SWA_TEMPLATE_MODE;
        } else {
            OP_LOGE(opName_, "When cmpSparseIndices is not nullptr, cmpKv cannot be nullptr.");
            return ge::GRAPH_FAILED;
        }
        if (sasInfo.perfMode == SASTemplateMode::CFA_TEMPLATE_MODE || sasInfo.perfMode == SASTemplateMode::SCFA_TEMPLATE_MODE) {
            if (kvLayout_ == SASLayout::TND && opParamInfo_.cuSeqLensCmpKv.tensor == nullptr) {
                OP_LOGE(opName_, "the layout_kv is %s, seqlens_cmp_kv must be provided.", SASLayoutToSerialString(kvLayout_).c_str());
                return ge::GRAPH_FAILED;
            }
        }
        return ge::GRAPH_SUCCESS;
    } else {
        OP_LOGE(opName_, "oriKv is nullptr");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetQueryAndOutLayout()
{
    // 获取q和attnOut的Layout基准值
    // layoutQuery: {qLayout, outLayout}
    const map<string, pair<SASLayout, SASLayout>> layoutMap = {
        {"BSND",        {SASLayout::BSND, SASLayout::BSND}},
        {"TND",         {SASLayout::TND, SASLayout::TND }},
    };
    std::string layout(opParamInfo_.layoutQ);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second.first;
        outLayout_ = it->second.second;
        oriSparseIndicesLayout_ = qLayout_;
        cmpSparseIndicesLayout_ = qLayout_;
    } else {
        OP_LOGE(opName_, "layout of Q is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    if (qLayout_ == SASLayout::BSND){
        OP_CHECK_IF(opParamInfo_.cuSeqLensQ.tensor != nullptr,
                    OP_LOGE(opName_, "when query's layout is BSND, cuSeqLensQ is null."),
                    return ge::GRAPH_FAILED);
    }
    if (qLayout_ == SASLayout::TND){
        OP_CHECK_IF(opParamInfo_.seqUsedQ.tensor != nullptr,
                    OP_LOGE(opName_, "when query's layout is TND, seqUsedQ is null."),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetKvLayout()
{
    const map<string, SASLayout> layoutKVMap = {
        {"PA_ND",     SASLayout::PA_ND},
        {"TND",     SASLayout::TND}
    };
    std::string layout(opParamInfo_.layoutKv);
    auto it = layoutKVMap.find(layout);
    if (it != layoutKVMap.end()) {
        kvLayout_ = it->second;
    } else {
        OP_LOGE(opName_, "layoutKV is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// =============Parser function====================
bool SASInfoParser::HasAxis(const SASAxis &axis, const SASLayout &layout, const gert::Shape &shape) const
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

size_t SASInfoParser::GetAxisIdx(const SASAxis &axis, const SASLayout &layout) const
{
    const std::vector<SASAxis>& axes = SAS_LAYOUT_AXIS_MAP.find(layout)->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    return std::distance(axes.begin(), axisIt);
}

uint32_t SASInfoParser::GetAxisNum(const gert::Shape &shape, const SASAxis &axis,const SASLayout &layout) const
{
    return HasAxis(axis, layout, shape) ? shape.GetDim(GetAxisIdx(axis, layout)) : invalidDimValue_;
}

void SASInfoParser::SetSASShape()
{
    qShape_ = opParamInfo_.q.shape->GetStorageShape();
    if (opParamInfo_.oriKv.tensor != nullptr) {
        oriKvShape_ = opParamInfo_.oriKv.tensor->GetStorageShape();
    } else {
        OP_LOGE(opName_, "query tensor is nullptr, please check input parameters.");
    }
    if (opParamInfo_.cmpKv.tensor != nullptr) {
        cmpKvShape_ = opParamInfo_.cmpKv.tensor->GetStorageShape();
    }
    if (perfMode_ == SASTemplateMode::SCFA_TEMPLATE_MODE)
    {
        if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
            cmpSparseIndicesShape_ = opParamInfo_.cmpSparseIndices.tensor->GetStorageShape();
            uint32_t cmpSparseIndicesT = GetAxisNum(cmpSparseIndicesShape_, SASAxis::T, cmpSparseIndicesLayout_);
        } else {
            OP_LOGE(opName_, "cmpSparseIndices tensor is nullptr, please check input parameters.");
        }
    }
}

ge::graphStatus SASInfoParser::GetN1Size()
{
    n1Size_ = GetAxisNum(qShape_, SASAxis::N, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetN2Size()
{
    if (opParamInfo_.oriKv.tensor != nullptr) {
        n2Size_ = GetAxisNum(oriKvShape_, SASAxis::N, kvLayout_);
    } 
    if (opParamInfo_.cmpKv.tensor != nullptr) {
        uint32_t cmpKvN2Size_ = GetAxisNum(cmpKvShape_, SASAxis::N, kvLayout_);
        if (perfMode_ == SASTemplateMode::SCFA_TEMPLATE_MODE){
            uint32_t cmpSparseIndicesN2Size_ = GetAxisNum(cmpSparseIndicesShape_, SASAxis::N, cmpSparseIndicesLayout_);
            OP_CHECK_IF(cmpKvN2Size_ != n2Size_ || n2Size_ != cmpSparseIndicesN2Size_, 
            OP_LOGE(opName_, "N2 size check failed! Expected oriKvN2 == cmpSparseIndicesN2."),
            return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(cmpKvN2Size_ != n2Size_, 
                    OP_LOGE(opName_, "N2 size check failed! Expected cmpKvN2 ==oriKvN2."),
                    return ge::GRAPH_FAILED);
        n2Size_ = cmpKvN2Size_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetGSize()
{
    if (n2Size_ != 0) {
        gSize_ = n1Size_ / n2Size_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    SASLayout &layout, const std::string &name) const
{
    if ((tensor == nullptr)) {
        OP_LOGE(opName_, "when layout of q is %s, %s must be provided.",
            SASLayoutToSerialString(layout).c_str(), name.c_str());
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OP_LOGE(opName_, "the shape size of %s is %ld, it should be greater than 0.",
            name.c_str(), shapeSize);
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetActualSeqLenQSize(uint32_t &size)
{
    return GetActualSeqLenSize(size, opParamInfo_.cuSeqLensQ.tensor, qLayout_, "cuSeqLensQ");
}

ge::graphStatus SASInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if (qLayout_ == SASLayout::TND) {
        return GetActualSeqLenQSize(bSize_);
    } else { // BSND
        bSize_ = GetAxisNum(qShape_, SASAxis::B, qLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetQTSize()
{
    // 获取query的T基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    qTSize_ = (qLayout_ == SASLayout::TND) ? GetAxisNum(qShape_, SASAxis::T, qLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetKVTSize()
{
    // 获取KV的T基准值
    // 1、非TND时, 以KV的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_ori_kv和actual_seq_lens_cmp_kv必须传入, 以actual_seq_lens_ori_kv数组的长度为B轴大小(当前接口只传入oriseq，先以oriseq算出cmpseq)
    orikvTSize_ = (kvLayout_ == SASLayout::TND) ? GetAxisNum(oriKvShape_, SASAxis::T, kvLayout_) : 0;
    // 入参接口信息可以从GetOptionalInputParaInfo()函数中获取
    // cmpkvTSize_ = (kvLayout_ == SASLayout::TND) ? GetAxisNum(cmpKvShape_, SASAxis::T, kvLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus SASInfoParser::GetS1Size()
{
    // 获取S1基准值
    // 1、非TND时, 以query的S维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组中的最大值为基准
    if (qLayout_ == SASLayout::TND) {
        s1Size_ = GetAxisNum(qShape_, SASAxis::T, qLayout_);
    } else { // BSND
        s1Size_ = GetAxisNum(qShape_, SASAxis::S, qLayout_);
    }
    if (perfMode_ == SASTemplateMode::SCFA_TEMPLATE_MODE){
        if (cmpSparseIndicesLayout_ == SASLayout::TND) {
            uint32_t cmpSparseIndicesT = GetAxisNum(cmpSparseIndicesShape_, SASAxis::T, cmpSparseIndicesLayout_);
            OP_CHECK_IF(cmpSparseIndicesT != s1Size_,
            OP_LOGE(opName_, "T size check failed !"),
            return ge::GRAPH_FAILED);
        } else{
            uint32_t cmpSparseIndicesS1 = GetAxisNum(cmpSparseIndicesShape_, SASAxis::S, cmpSparseIndicesLayout_);
            OP_CHECK_IF(cmpSparseIndicesS1 != s1Size_,
                        OP_LOGE(opName_, "s1 size check failed !"),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetMaxBlockNumPerBatch()
{
    if (opParamInfo_.oriBlockTable.tensor == nullptr) {
        OP_LOGE(opName_, "the layout_kv is %s, blockTable must be provided.", SASLayoutToSerialString(kvLayout_).c_str());
        return ge::GRAPH_FAILED;
    }
    uint32_t oriDimNum = opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDimNum();
    if (oriDimNum != DIM_NUM_TWO) {
        OP_LOGE(opName_, "the dim num of ori_block_table is %u, it should be %u.", oriDimNum, DIM_NUM_TWO);
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(1) < 0) {
        OP_LOGE(opName_, "%s's second dimension(%lld) should be non-negative number.",
            ORI_BLOCK_TABLE_NAME.c_str(), opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(1));
        return ge::GRAPH_FAILED;
    }
    oriMaxBlockNumPerBatch_ = opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(1);

    if (opParamInfo_.cmpBlockTable.tensor != nullptr) {
        uint32_t cmpDimNum = opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDimNum();
        if (cmpDimNum != DIM_NUM_TWO) {
            OP_LOGE(opName_, "the dim num of cmp_block_table is %u, it should be %u.", cmpDimNum, DIM_NUM_TWO);
            return ge::GRAPH_FAILED;
        }
        if (opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(1) <= 0) {
            OP_LOGE(opName_, "%s's second dimension(%lld) should be greater than 0",
                CMP_BLOCK_TABLE_NAME.c_str(), opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(1));
            return ge::GRAPH_FAILED;
        }
        cmpMaxBlockNumPerBatch_ = opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetBlockSize()
{
    oriBlockSize_ = GetAxisNum(oriKvShape_, SASAxis::Bs, kvLayout_);
    cmpBlockSize_ = GetAxisNum(cmpKvShape_, SASAxis::Bs, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetS2SizeForPageAttention()
{
    if (GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS || GetBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    s2Size_ = oriMaxBlockNumPerBatch_ * oriBlockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetS2SizeForTND()
{
    if (opParamInfo_.cuSeqLensKv.tensor == nullptr) {
        OP_LOGE(opName_, "the layout_kv is %s, seqlens_ori_kv must be provided.", SASLayoutToSerialString(kvLayout_).c_str());
        return ge::GRAPH_FAILED;
    } 
    // if (opParamInfo_.sequsedKv.tensor == nullptr) {
    //     OP_LOGE(opName_, "the layout_kv is %s, sequsedKv must be provided.", SASLayoutToSerialString(kvLayout_).c_str());
    //     return ge::GRAPH_FAILED;
    // }
    // 这里返回累加和的最大值
    s2Size_ = GetAxisNum(oriKvShape_, SASAxis::T, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetS2Size()
{
    // 获取S2基准值:PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    return (kvLayout_ == SASLayout::PA_ND) ? GetS2SizeForPageAttention() : GetS2SizeForTND();
}

ge::graphStatus SASInfoParser::GetQHeadDim()
{
    qHeadDim_ = GetAxisNum(qShape_, SASAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetValueHeadDim()
{
    if (opParamInfo_.oriKv.tensor != nullptr) {
        oriKvHeadDim_ = GetAxisNum(oriKvShape_, SASAxis::D, kvLayout_);
    }
    if (opParamInfo_.cmpKv.tensor != nullptr) {
        cmpKvHeadDim_ = GetAxisNum(cmpKvShape_, SASAxis::D, kvLayout_);
    }
    return ge::GRAPH_SUCCESS;
}


ge::graphStatus SASInfoParser::GetSparseBlockCount()
{
    if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
        sparseBlockCount_ = GetAxisNum(cmpSparseIndicesShape_, SASAxis::K, cmpSparseIndicesLayout_);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetSinks()
{
    if (opParamInfo_.sinks.tensor == nullptr) {
        OP_LOGE(opName_, "%s must be provided!", SINKS_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASInfoParser::GetActualseqInfo()
{
    maxActualseq_ = static_cast<uint32_t>(s2Size_);
    if (qLayout_ == SASLayout::TND) {
        if (opParamInfo_.cuSeqLensQ.tensor != nullptr) {
            if (opParamInfo_.cuSeqLensQ.tensor->GetShapeSize() != bSize_) {
                OP_LOGE(opName_, "cu_seqlens_q's dimension should be equal to %u.", bSize_);
                return ge::GRAPH_FAILED;
            }
            actualLenDimsQ_ = opParamInfo_.cuSeqLensQ.tensor->GetShapeSize() - 1; // cuSeqLensQ shape is B+1
        } else {
            OP_LOGE(opName_, "When qLayout is TND,  input cu_seqlens_q must be provided");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (opParamInfo_.seqUsedQ.tensor != nullptr) {
            actualLenDimsQ_ = opParamInfo_.seqUsedQ.tensor->GetShapeSize();
        } else {
            OP_LOGE(opName_, "When qLayout is BSND,  input seqUsedQ must be provided");
            return ge::GRAPH_FAILED;
        }
    }
    if (kvLayout_ == SASLayout::PA_ND) {
        if (opParamInfo_.sequsedKv.tensor != nullptr) {
            if (qLayout_ == SASLayout::BSND){
                if (opParamInfo_.sequsedKv.tensor->GetShapeSize() != bSize_) {
                    OP_LOGE(opName_, "sequsedKv's dimension should be equal to %u, but got %ld.", 
                        bSize_, opParamInfo_.sequsedKv.tensor->GetShapeSize());
                    return ge::GRAPH_FAILED;
                }
            } else {
                if (opParamInfo_.sequsedKv.tensor->GetShapeSize() != (bSize_ - 1)) {
                    OP_LOGE(opName_, "sequsedKv's dimension should be equal to %u (bSize - 1), but got %ld.", 
                        (bSize_ - 1), opParamInfo_.sequsedKv.tensor->GetShapeSize());
                    return ge::GRAPH_FAILED;
                }
            }
            actualLenDimsKV_ = opParamInfo_.sequsedKv.tensor->GetShapeSize();
        } else {
            OP_LOGE(opName_, "When kv layout is PA_ND, input sequsedKv must be provided");
            return ge::GRAPH_FAILED;
        }
    } else if (kvLayout_ == SASLayout::TND) {
        if (opParamInfo_.cuSeqLensKv.tensor != nullptr) {
            if (qLayout_ == SASLayout::BSND){
                if (opParamInfo_.cuSeqLensKv.tensor->GetShapeSize() != bSize_ + 1) {
                    OP_LOGE(opName_, "cuSeqLensKv's dimension should be equal to %u (bSize + 1), but got %ld.", 
                        (bSize_ + 1), opParamInfo_.sequsedKv.tensor->GetShapeSize());
                    return ge::GRAPH_FAILED;
                }
            } else {
                if (opParamInfo_.cuSeqLensKv.tensor->GetShapeSize() != (bSize_)) {
                    OP_LOGE(opName_, "cuSeqLensKv's dimension should be equal to %u, but got %ld.", 
                        bSize_, opParamInfo_.sequsedKv.tensor->GetShapeSize());
                    return ge::GRAPH_FAILED;
                }
            }
            actualLenDimsKV_ = opParamInfo_.cuSeqLensKv.tensor->GetShapeSize();
        } else {
            OP_LOGE(opName_, "When kv layout is TND, input cuSeqLensKv must be provided");
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE(opName_, "oriKV and cmpKv only support PA_ND or TND layout.");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.seqUsedQ.tensor != nullptr) {
        actualLenDimsQ_ = opParamInfo_.seqUsedQ.tensor->GetShapeSize();
    } else if (opParamInfo_.cuSeqLensQ.tensor != nullptr) {
        actualLenDimsQ_ = opParamInfo_.cuSeqLensQ.tensor->GetShapeSize() - 1; // cuSeqLensQ shape is B+1
    }
    return ge::GRAPH_SUCCESS;
}

void SASInfoParser::GenerateInfo(SASTilingInfo &sasInfo)
{
    sasInfo.opName = opName_;
    sasInfo.platformInfo = platformInfo_;
    sasInfo.opParamInfo = opParamInfo_;
    sasInfo.socVersion = socVersion_;

    sasInfo.bSize = bSize_;
    sasInfo.n1Size = n1Size_;
    sasInfo.n2Size = n2Size_;
    sasInfo.s1Size = s1Size_;
    sasInfo.s2Size = s2Size_;
    sasInfo.gSize = gSize_;
    sasInfo.qHeadDim = qHeadDim_;
    sasInfo.oriKvHeadDim = oriKvHeadDim_;
    sasInfo.cmpKvHeadDim = cmpKvHeadDim_;
    sasInfo.qTSize = qTSize_;
    sasInfo.sparseBlockCount = sparseBlockCount_;
    sasInfo.oriWinLeft = oriWinLeft_;
    sasInfo.oriWinRight = oriWinRight_;
    sasInfo.qType = qType_;
    sasInfo.oriKvType = oriKvType_;
    sasInfo.cmpKvType = cmpKvType_;
    sasInfo.outputType = outputType_;

    sasInfo.totalBlockNum = (opParamInfo_.oriKv.tensor != nullptr) ?
        opParamInfo_.oriKv.tensor->GetStorageShape().GetDim(0) : 0;
    sasInfo.sparseBlockSize = 1;
    sasInfo.oriBlockSize = oriBlockSize_;
    sasInfo.cmpBlockSize = cmpBlockSize_;
    sasInfo.blockTypeSize = sizeof(float);
    sasInfo.oriMaxBlockNumPerBatch = oriMaxBlockNumPerBatch_;
    sasInfo.cmpMaxBlockNumPerBatch = cmpMaxBlockNumPerBatch_;

    sasInfo.actualLenDimsQ = actualLenDimsQ_;
    sasInfo.actualLenDimsKV = actualLenDimsKV_;
    sasInfo.maxActualseq = maxActualseq_;
    sasInfo.actualSeqLenFlag = (opParamInfo_.sequsedKv.tensor != nullptr);
    sasInfo.isSameSeqAllKVTensor = isSameSeqAllKVTensor_;

    sasInfo.softmaxScale = *opParamInfo_.softmaxScale;
    sasInfo.cmpRatio = *opParamInfo_.cmpRatio;
    sasInfo.oriMaskMode = *opParamInfo_.oriMaskMode;
    sasInfo.cmpMaskMode = *opParamInfo_.cmpMaskMode;
    sasInfo.oriWinLeft = *opParamInfo_.oriWinLeft;
    sasInfo.oriWinRight = *opParamInfo_.oriWinRight;

    sasInfo.qLayout = qLayout_;
    sasInfo.oriSparseIndicesLayout = oriSparseIndicesLayout_;
    sasInfo.cmpSparseIndicesLayout = cmpSparseIndicesLayout_;
    sasInfo.kvLayout = kvLayout_;
    sasInfo.outLayout = outLayout_;
}

ge::graphStatus SASInfoParser::Parse(SASTilingInfo &sasInfo)
{
    if (context_ == nullptr) {
        OP_LOGE("SparseFlashAttention", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetOpName() ||
        ge::GRAPH_SUCCESS != GetNpuInfo() ||
        ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence() ||
        ge::GRAPH_SUCCESS != CheckUnrequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetInOutDataType() ||
        ge::GRAPH_SUCCESS != GetQueryAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetKvLayout() ||
        ge::GRAPH_SUCCESS != GetSASTemplateMode(sasInfo)) {
        return ge::GRAPH_FAILED;
    }

    SetSASShape();
    if (
        ge::GRAPH_SUCCESS != GetN1Size() ||
        ge::GRAPH_SUCCESS != GetN2Size() ||
        ge::GRAPH_SUCCESS != GetGSize() ||
        ge::GRAPH_SUCCESS != GetBatchSize() ||
        ge::GRAPH_SUCCESS != GetQTSize() ||
        ge::GRAPH_SUCCESS != GetKVTSize() ||
        ge::GRAPH_SUCCESS != GetS1Size() ||
        ge::GRAPH_SUCCESS != GetS2Size() ||
        ge::GRAPH_SUCCESS != GetQHeadDim() ||
        ge::GRAPH_SUCCESS != GetValueHeadDim() ||
        ge::GRAPH_SUCCESS != GetSparseBlockCount() ||
        ge::GRAPH_SUCCESS != GetSinks()
        ) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != GetActualseqInfo()) {
        return ge::GRAPH_FAILED;
    }
    GenerateInfo(sasInfo);
    return ge::GRAPH_SUCCESS;
}

void SASTilingCheck::Init()
{
    opName_ = sasInfo_.opName;
    platformInfo_ = sasInfo_.platformInfo;
    opParamInfo_ = sasInfo_.opParamInfo;
    socVersion_ = sasInfo_.socVersion;
    bSize_ = sasInfo_.bSize;
    n1Size_ = sasInfo_.n1Size;
    n2Size_ = sasInfo_.n2Size;
    s1Size_ = sasInfo_.s1Size;
    s2Size_ = sasInfo_.s2Size;
    gSize_ = sasInfo_.gSize;
    qHeadDim_ = sasInfo_.qHeadDim;
    oriKvHeadDim_ = sasInfo_.oriKvHeadDim;
    cmpKvHeadDim_ = sasInfo_.cmpKvHeadDim;
    oriBlockSize_ = sasInfo_.oriBlockSize;
    cmpBlockSize_ = sasInfo_.cmpBlockSize;
    qTSize_ = sasInfo_.qTSize;
    qType_ = sasInfo_.qType;
    oriKvType_ = sasInfo_.oriKvType;
    cmpKvType_ = sasInfo_.cmpKvType;
    outputType_ = sasInfo_.outputType;
    cmpRatio_ = sasInfo_.cmpRatio;
    qLayout_ = sasInfo_.qLayout;
    oriSparseIndicesLayout_ = sasInfo_.oriSparseIndicesLayout;
    cmpSparseIndicesLayout_ = sasInfo_.cmpSparseIndicesLayout;
    oriWinLeft_ = sasInfo_.oriWinLeft;
    oriWinRight_ = sasInfo_.oriWinRight;
    kvLayout_ = sasInfo_.kvLayout;
    outLayout_ = sasInfo_.outLayout;
}

void SASTilingCheck::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
    const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << SASDataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName_, "Tensor %s only supports dtype %s, but got %s",
        name.c_str(), oss.str().c_str(), SASDataTypeToSerialString(actualDtype).c_str());
}

ge::graphStatus SASTilingCheck::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
    const std::string &name) const
{
   if (desc != nullptr) {
        const auto& it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(it == DTYPE_SUPPORT_MAP.end(),
                    OP_LOGE(opName_, "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
                    return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(std::find(
                    expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end(),
                    LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void SASTilingCheck::LogErrorLayoutSupport(const std::vector<SASLayout> &expectLayoutList,
    const SASLayout &actualLayout, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectLayoutList.size(); ++i) {
        oss << SASLayoutToSerialString(expectLayoutList[i]);
        if (i < expectLayoutList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName_, "Tensor %s only supports layout %s, but got %s",
        name.c_str(), oss.str().c_str(), SASLayoutToSerialString(actualLayout).c_str());
}

ge::graphStatus SASTilingCheck::CheckLayoutSupport(const SASLayout &actualLayout, const std::string &name) const
{
    const auto& it = LAYOUT_SUPPORT_MAP.find(name);
    OP_CHECK_IF(it == LAYOUT_SUPPORT_MAP.end(),
                OP_LOGE(opName_, "%s layout support list should be specify in LAYOUT_SUPPORT_MAP", name.c_str()),
                return ge::GRAPH_FAILED);
    auto &expectLayoutList = it->second;
    OP_CHECK_IF(std::find(
        expectLayoutList.begin(), expectLayoutList.end(), actualLayout) == expectLayoutList.end(),
        LogErrorLayoutSupport(expectLayoutList, actualLayout, name),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

template <typename T>
void SASTilingCheck::LogErrorNumberSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name, const std::string subName) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << std::to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName_, "%s %s only supports %s, but got %s",
              name.c_str(), subName.c_str(), oss.str().c_str(), std::to_string(actualValue).c_str());
}

template <typename T> 
void SASTilingCheck::LogErrorDimNumSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name) const
{
    LogErrorNumberSupport(expectNumberList, actualValue, name, "dimension");
}

ge::graphStatus SASTilingCheck::CheckDimNumSupport(const gert::StorageShape *shape,
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

ge::graphStatus SASTilingCheck::CheckDimNumInLayoutSupport(const SASLayout &layout,
    const gert::StorageShape *shape, const std::string &name) const
{
    const auto& dimIt = SAS_LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
                OP_LOGE(opName_, "When layout is %s, %s dimension should be %zu, but it's %zu",
                SASLayoutToSerialString(layout).c_str(), name.c_str(), dimIt->second,
                shape->GetStorageShape().GetDimNum()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckSingleParaQuery() const
{
    if (opParamInfo_.q.desc == nullptr) {
        OP_LOGE(opName_, "%s must be provided!", QUERY_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    const std::vector<size_t> queryDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (
        ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.q.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(qLayout_, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.q.shape, queryDimNumList, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(qLayout_, opParamInfo_.q.shape, QUERY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckSingleParaOriKv() const
{
    const std::vector<size_t> oriKvDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (
        ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.oriKv.desc, ORI_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, ORI_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.oriKv.tensor->GetShape(), oriKvDimNumList, ORI_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(kvLayout_, &opParamInfo_.oriKv.tensor->GetShape(), ORI_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckSingleParaCmpKv() const
{
    if (sasInfo_.perfMode == SASTemplateMode::SCFA_TEMPLATE_MODE || sasInfo_.perfMode == SASTemplateMode::CFA_TEMPLATE_MODE) {
        const std::vector<size_t> cmpKvDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
        if (
            ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpKv.desc, CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cmpKv.tensor->GetShape(), cmpKvDimNumList, CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(kvLayout_, &opParamInfo_.cmpKv.tensor->GetShape(), CMP_KV_NAME)) {
            return ge::GRAPH_FAILED;
            }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckSingleParaNumHeads() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckSingleParaKvHeadNums() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckSingleParaCmpSparseIndices() const
{
    if (sasInfo_.perfMode == optiling::SASTemplateMode::SCFA_TEMPLATE_MODE){
        const std::vector<size_t> cmpSparseIndicesDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
        if (
            ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpSparseIndices.desc, CMP_SPARSE_INDICES) ||
            ge::GRAPH_SUCCESS != CheckLayoutSupport(cmpSparseIndicesLayout_, CMP_SPARSE_INDICES) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cmpSparseIndices.tensor->GetShape(), cmpSparseIndicesDimNumList, CMP_SPARSE_INDICES) ||
            ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(cmpSparseIndicesLayout_, &opParamInfo_.cmpSparseIndices.tensor->GetShape(), CMP_SPARSE_INDICES)) {
            return ge::GRAPH_FAILED;
        }
        if (cmpSparseIndicesLayout_ == SASLayout::TND)
        {
            OP_CHECK_IF((opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_NUM_THREE - 1) != TOPK_LIMIT),
                        OP_LOGE(opName_, "K should be %u, but got: %lld ",TOPK_LIMIT,
                        opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_NUM_THREE - 1)),
                        return ge::GRAPH_FAILED);
        } else{
            OP_CHECK_IF((opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_NUM_FOUR - 1) != TOPK_LIMIT),
                        OP_LOGE(opName_, "K should be %u, but got: %lld ",TOPK_LIMIT,
                        opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(DIM_NUM_FOUR - 1)),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckSingleParaOriBlockTable() const
{   
    if(kvLayout_ == SASLayout::TND) {
        return ge::GRAPH_SUCCESS;
    }
    const std::vector<size_t> oriBlockTableDimNumList = {DIM_NUM_TWO};
    if (
        ge::GRAPH_SUCCESS != CheckDtypeSupport( opParamInfo_.oriBlockTable.desc, ORI_BLOCK_TABLE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.oriBlockTable.tensor->GetShape(), oriBlockTableDimNumList, ORI_BLOCK_TABLE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF((oriBlockSize_ <= 0 || oriBlockSize_ > BLOCK_SIZE_LIMIT ||
                (static_cast<uint64_t>(oriBlockSize_) % 16 != 0UL)),
                OP_LOGE(opName_, "oriBlockSize_ should be in range [1, 1024], and be aligned to 16, but got: %d.",
                oriBlockSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckSingleParaCmpBlockTable() const
{
    if(kvLayout_ == SASLayout::TND) {
        return ge::GRAPH_SUCCESS;
    }
    if (sasInfo_.perfMode == optiling::SASTemplateMode::SCFA_TEMPLATE_MODE ||
        sasInfo_.perfMode == optiling::SASTemplateMode::CFA_TEMPLATE_MODE){
            const std::vector<size_t> cmpBlockTableDimNumList = {DIM_NUM_TWO};
            if (
                ge::GRAPH_SUCCESS != CheckDtypeSupport( opParamInfo_.cmpBlockTable.desc, CMP_BLOCK_TABLE_NAME) ||
                ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cmpBlockTable.tensor->GetShape(),
                cmpBlockTableDimNumList, CMP_BLOCK_TABLE_NAME)) {
                return ge::GRAPH_FAILED;
                }
            OP_CHECK_IF((cmpBlockSize_ <= 0 || cmpBlockSize_ > BLOCK_SIZE_LIMIT ||
                        (static_cast<uint64_t>(cmpBlockSize_) % 16 != 0UL)),
                        OP_LOGE(opName_, "cmpBlockSize should be in [1, 1024], and be aligned to 16, but got: %d.",
                        cmpBlockSize_),
                        return ge::GRAPH_FAILED);
        }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckSingleParaSinks() const
{
    if (opParamInfo_.sinks.tensor->GetStorageShape().GetDimNum() != DIM_NUM_ONE) {
        OP_LOGE(opName_, "the dim num of %s is %u, it should be %u.", SINKS_NAME.c_str(),
            opParamInfo_.sinks.tensor->GetStorageShape().GetDimNum(), DIM_NUM_ONE);
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.sinks.tensor->GetStorageShape().GetDim(0) != n1Size_) {
        OP_LOGE(opName_, "%s's dimension(%ld) should be equal to query head num(%u).", SINKS_NAME.c_str(),
            opParamInfo_.sinks.tensor->GetStorageShape().GetDim(0), n1Size_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckSingleParaMetadata() const
{
    if (opParamInfo_.metadata.tensor == nullptr) {
        OP_LOGE(opName_, "%s must be provided!", METADATA_NAME.c_str());
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF((opParamInfo_.metadata.tensor->GetShapeSize() != METADATA_LIMIT),
 	            OP_LOGE(opName_, "input metadata dim 0 must be %u.", METADATA_LIMIT),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckSingleParaCmpRatio() const
{
    if (sasInfo_.perfMode == optiling::SASTemplateMode::CFA_TEMPLATE_MODE){
        OP_CHECK_IF(cmpRatio_ != 128,
                    OP_LOGE(opName_, "cmpRatio should be 128, but got %u", cmpRatio_),
                    return ge::GRAPH_FAILED);
    }
    if (sasInfo_.perfMode == optiling::SASTemplateMode::SCFA_TEMPLATE_MODE){
        OP_CHECK_IF(cmpRatio_ != 4,
                    OP_LOGE(opName_, "cmpRatio should be 4, but got %u", cmpRatio_),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckSingleParaOriMaskMode() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckSingleParaCmpMaskMode() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckSingleParaOriWinLeft() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckSingleParaOriWinRight() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckSinglePara() const
{
    if (
        ge::GRAPH_SUCCESS != CheckSingleParaQuery() ||
        ge::GRAPH_SUCCESS != CheckSingleParaOriKv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpKv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaNumHeads() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKvHeadNums() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpSparseIndices() ||
        ge::GRAPH_SUCCESS != CheckSingleParaOriBlockTable() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpBlockTable() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSinks() ||
        ge::GRAPH_SUCCESS != CheckSingleParaMetadata() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpRatio() ||
        ge::GRAPH_SUCCESS != CheckSingleParaOriMaskMode() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpMaskMode() ||
        ge::GRAPH_SUCCESS != CheckSingleParaOriWinLeft() ||
        ge::GRAPH_SUCCESS != CheckSingleParaOriWinRight()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckExists(const void *pointer, const std::string &name) const
{
    OP_CHECK_IF(pointer == nullptr,
                OP_LOGE(opName_, "%s should not be null", name.c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckNotExists(const void *pointer, const std::string &name) const
{
    OP_CHECK_IF(pointer != nullptr,
                OP_LOGE(opName_, "%s should be null", name.c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto& kv : paramMap) {
        if (CheckExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckNotExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto& kv : paramMap) {
        if (CheckNotExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckExistenceByMap(std::map<std::string, const void *> &existMap,
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

ge::graphStatus SASTilingCheck::CheckParaExistence() const
{   
    std::map<std::string, const void *> ParamExistMap = {};
    std::map<std::string, const void *> ParamNotExistMap = {};
    if (CheckExistenceByMap(ParamExistMap, ParamNotExistMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckFeatureShape() const
{
    OP_CHECK_IF(bSize_ <= 0,
                OP_LOGE(opName_, "batch_size should be greater than 0, but got %u", bSize_),
                return ge::GRAPH_FAILED);
        
    OP_CHECK_IF(qTSize_ <= 0 && (qLayout_ == SASLayout::TND),
                OP_LOGE(opName_, "T_size of query should be greater than 0, but got %u", qTSize_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(n1Size_ % 4 != 0,
                OP_LOGE(opName_, "q_head_num should be multiple of 4, but got %u", n1Size_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(n2Size_ != 1,
                OP_LOGE(opName_, "kv_head_num should be 1, but got %u", n2Size_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(n1Size_ % n2Size_ != 0,
                OP_LOGE(opName_, "q_head_num(%u) must be divisible by kv_head_num(%u)", n1Size_, n2Size_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(gSize_ % 4 != 0,
                OP_LOGE(opName_, "group num should be multiple of 4, but got %u", gSize_),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(qHeadDim_ != DIM_LIMIT,
                OP_LOGE(opName_, "q_head_dim only support %u, but got %u", DIM_LIMIT, qHeadDim_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(oriKvHeadDim_ != DIM_LIMIT,
                OP_LOGE(opName_, "oriKvHeadDim only support %u, but got %u", DIM_LIMIT, oriKvHeadDim_),
                return ge::GRAPH_FAILED);
    if (!(sasInfo_.perfMode == SASTemplateMode::SWA_TEMPLATE_MODE)){
        OP_CHECK_IF(cmpKvHeadDim_ != DIM_LIMIT,
                    OP_LOGE(opName_, "cmpKvHeadDim only support %u, but got %u", DIM_LIMIT, cmpKvHeadDim_),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(!(qType_ == oriKvType_),
                OP_LOGE(opName_, "Head dimension data type check failed! qType[%s] must be the same with oriKvType[%s].",
                SASDataTypeToSerialString(qType_).c_str(),
                SASDataTypeToSerialString(oriKvType_).c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(*opParamInfo_.oriMaskMode != 4,
                OP_LOGE(opName_, "oriMaskMode should be 4, but got %d", *opParamInfo_.oriMaskMode),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.cmpMaskMode != 3,
                OP_LOGE(opName_, "cmpMaskMode should be 3, but got %d", *opParamInfo_.cmpMaskMode),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(oriWinLeft_ != 127,
                OP_LOGE(opName_, "oriWinLeft_ should be 127, but got %d", oriWinLeft_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(oriWinRight_ != 0,
                OP_LOGE(opName_, "oriWinRight_ should be 0, but got %d", oriWinRight_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckFeatureLayout() const
{
    const std::vector<std::string> layoutQuerySupportList = {
        "BSND",
        "TND"
    };
    std::string layoutQuery = opParamInfo_.layoutQ;
    OP_CHECK_IF(std::find(layoutQuerySupportList.begin(), layoutQuerySupportList.end(), layoutQuery) ==
                layoutQuerySupportList.end(),
                OP_LOGE(opName_, "layoutQuery only supports BSND/TND, but got %s", layoutQuery.c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckFeatureDtype() const
{
    OP_CHECK_IF(qType_ != ge::DT_BF16 && qType_ != ge::DT_FLOAT16,
                OP_LOGE(opName_, "query dtype only support %s and %s, but got %s",
                SASDataTypeToSerialString(ge::DT_BF16).c_str(), SASDataTypeToSerialString(ge::DT_FLOAT16).c_str(),
                SASDataTypeToSerialString(qType_).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckFeaturePa() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckFeature() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureShape() ||
        ge::GRAPH_SUCCESS != CheckFeatureLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureDtype() ||
        ge::GRAPH_SUCCESS != CheckFeaturePa()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void SASTilingCheck::SetSASShapeCompare()
{
    queryShapeCmp_ = opParamInfo_.q.shape->GetStorageShape();
    oriKvShapeCmp_= opParamInfo_.oriKv.tensor->GetShape().GetStorageShape();
    attenOutShapeCmp_ = opParamInfo_.attnOut.shape->GetStorageShape();
    if (sasInfo_.perfMode == SASTemplateMode::CFA_TEMPLATE_MODE || 
        sasInfo_.perfMode == SASTemplateMode::SCFA_TEMPLATE_MODE) {
        cmpKvShapeCmp_= opParamInfo_.cmpKv.tensor->GetShape().GetStorageShape();
    }
    if (sasInfo_.perfMode == SASTemplateMode::SCFA_TEMPLATE_MODE) {
        cmpKvSparseIndicesCmp_ = opParamInfo_.cmpSparseIndices.tensor->GetShape().GetStorageShape();
    }
}

ge::graphStatus SASTilingCheck::CheckDTypeConsistency(const ge::DataType &actualDtype,
    const ge::DataType &expectDtype, const std::string &name) const
{
    if (actualDtype != expectDtype) {
        OP_LOGE(opName_, "%s dtype should be the same to %s, but it's %s.", name.c_str(),
            SASDataTypeToSerialString(expectDtype).c_str(),
            SASDataTypeToSerialString(actualDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckOriAndCmpKv() const
{
    if (sasInfo_.perfMode == SASTemplateMode::CFA_TEMPLATE_MODE ||
        sasInfo_.perfMode == SASTemplateMode::SCFA_TEMPLATE_MODE)
    {
        if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(cmpKvType_,
            oriKvType_, CMP_KV_NAME)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckAttenOut() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckActualSeqLensQ() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckActualSeqLens() const
{
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus SASTilingCheck::CheckBlockTable() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::CheckMultiParaConsistency()
{
    SetSASShapeCompare();
    if (
        ge::GRAPH_SUCCESS != CheckOriAndCmpKv() ||
        ge::GRAPH_SUCCESS != CheckAttenOut() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQ() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckBlockTable()) 
        {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SASTilingCheck::Process()
{
    Init();
    if (
        CheckSinglePara() != ge::GRAPH_SUCCESS ||
        CheckParaExistence() != ge::GRAPH_SUCCESS ||
        CheckFeature() != ge::GRAPH_SUCCESS ||
        CheckMultiParaConsistency() != ge::GRAPH_SUCCESS
        ) 
    {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// --------------------------TilingPrepare函数定义-------------------------------------
static ge::graphStatus TilingPrepareForSparseAttnSharedkv(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

void SparseAttnSharedkvTiling::CalcUbBmm(SASTilingInfo *tilingInfo)
{
    uint32_t cubeMSize = tilingInfo->gSize * tilingInfo->s1Size;
    uint32_t maxMSize = mBaseSize_;
    if (cubeMSize > maxMSize) {
        cubeMSize = maxMSize;
    }
    mmResUbSize_ = sInnerSizeAlign_ * Align(cubeMSize, 16U);// kernel按照16对齐写出，tiling按照这个原则分配内存
    bmm2ResUbSize_ = headDimAlign_ * Align(cubeMSize, 16U);// kernel按照16对齐写出，tiling按照这个原则分配内存
}

void SparseAttnSharedkvTiling::SplitBalanced(SASTilingInfo *tilingInfo)
{
    uint32_t s2Size = tilingInfo->s2Size;
    sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    headDimAlign_ = Align(tilingInfo->qHeadDim, BYTE_BLOCK);
    CalcUbBmm(tilingInfo);

    tilingData_.baseParams.set_mBaseSize(mBaseSize_);
    tilingData_.baseParams.set_s2BaseSize(sInnerSize_);
    tilingData_.baseParams.set_mmResUbSize(mmResUbSize_);
    tilingData_.baseParams.set_bmm2ResUbSize(bmm2ResUbSize_);
}

// --------------------------SparseAttnSharedkvTiling类成员函数定义-----------------------
ge::graphStatus SparseAttnSharedkvTiling::DoOpTiling(SASTilingInfo *tilingInfo)
{
    // -------------set blockdim-----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);
    OP_LOGI(tilingInfo->opName, "SAS block dim: %u aiv Num: %u aic Num: %u.", blockDim, aivNum, aicNum);

    SplitBalanced(tilingInfo);
    // -------------set workspacesize-----------------
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;         // 4: fp32
    constexpr uint32_t VEC1_RES_ELEM_SIZE = 2;        // 2: fp16/bf16
    constexpr uint32_t MM2_RES_ELEM_SIZE = 4;         // 4: fp32
    constexpr uint32_t VEC2_RES_ELEM_SIZE = 4;        // 4: fp32
    constexpr uint32_t PRELOAD_NUM = 2;               // preload数量

    uint32_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // 主流程需Workspace大小
    workspaceSize += PRELOAD_NUM * mmResUbSize_ * MM1_RES_ELEM_SIZE * aicNum;
    workspaceSize += PRELOAD_NUM * mmResUbSize_ * VEC1_RES_ELEM_SIZE * aicNum;
    workspaceSize += PRELOAD_NUM * bmm2ResUbSize_ * MM2_RES_ELEM_SIZE * aicNum;
    workspaceSize += PRELOAD_NUM * bmm2ResUbSize_ * VEC2_RES_ELEM_SIZE * aicNum;
    if (tilingInfo->perfMode == SASTemplateMode::SCFA_TEMPLATE_MODE) {
        workspaceSize += 4 * 512 * 512 * 2 * aicNum; // 4:bufNum 512:s2Size  512:D 2:sizeof(half)
        workspaceSize += 4 * 128 * 4 * (2 * aicNum); // 4:缓存有效mte2 size长度 128:份数 4:512B对齐长度 2:aiv数量
    }
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    // -------------set tilingdata-----------------
    tilingData_.baseParams.set_batchSize(tilingInfo->bSize);
    tilingData_.baseParams.set_kvSeqSize(tilingInfo->s2Size);
    tilingData_.baseParams.set_qSeqSize(tilingInfo->s1Size);
    tilingData_.baseParams.set_nNumOfQInOneGroup(tilingInfo->gSize);
    tilingData_.baseParams.set_paBlockSize(tilingInfo->blockSize);
    tilingData_.baseParams.set_oriBlockSize(tilingInfo->oriBlockSize);
    tilingData_.baseParams.set_cmpBlockSize(tilingInfo->cmpBlockSize);
    tilingData_.baseParams.set_oriMaxBlockNumPerBatch(tilingInfo->oriMaxBlockNumPerBatch);
    tilingData_.baseParams.set_actualLenDimsQ(tilingInfo->actualLenDimsQ);
    tilingData_.baseParams.set_actualLenDimsKV(tilingInfo->actualLenDimsKV);

    tilingData_.baseParams.set_softmaxScale(tilingInfo->softmaxScale);
    tilingData_.baseParams.set_outputLayout(static_cast<uint32_t>(tilingInfo->outLayout));
    tilingData_.baseParams.set_oriMaskMode(tilingInfo->oriMaskMode);
    tilingData_.baseParams.set_oriWinLeft(tilingInfo->oriWinLeft);
    tilingData_.baseParams.set_oriWinRight(tilingInfo->oriWinRight);
    tilingData_.baseParams.set_sparseBlockSize(tilingInfo->sparseBlockSize);

    tilingData_.cmpParams.set_cmpMaxBlockNumPerBatch(tilingInfo->cmpMaxBlockNumPerBatch);
    tilingData_.cmpParams.set_sparseBlockCount(tilingInfo->sparseBlockCount);
    tilingData_.cmpParams.set_cmpRatio(tilingInfo->cmpRatio);
    tilingData_.cmpParams.set_cmpMaskMode(tilingInfo->cmpMaskMode);

    usedCoreNum_ = aicNum;
    tilingData_.baseParams.set_usedCoreNum(usedCoreNum_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // -------------set tilingkey-----------------
    // FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T, TEMPLATE_MODE
    uint32_t qLayout = static_cast<uint32_t>(tilingInfo->qLayout);
    uint32_t inputKvLayout = static_cast<uint32_t>(tilingInfo->kvLayout);

    uint32_t tilingKey =
        GET_TPL_TILING_KEY(0U, qLayout, inputKvLayout, static_cast<uint32_t>(tilingInfo->perfMode));
    context_->SetScheduleMode(1);
    context_->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

// --------------------------Tiling函数定义---------------------------
ge::graphStatus TilingSparseAttnSharedkv(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SparseAttnSharedkv", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    SASTilingInfo sasInfo;
    SASInfoParser sasInfoParser(context);
    if (sasInfoParser.Parse(sasInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SASTilingCheck sasTilingChecker(sasInfo);
    if (sasTilingChecker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    SparseAttnSharedkvTiling tiling(context);
    return tiling.DoOpTiling(&sasInfo);
}
// --------------------------Tiling函数及TilingPrepare函数注册--------
IMPL_OP_OPTILING(SparseAttnSharedkv)
    .Tiling(TilingSparseAttnSharedkv)
    .TilingParse<SASCompileInfo>(TilingPrepareForSparseAttnSharedkv);
} // namespace optiling
