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
 * \file sparse_flash_mla_tiling.cpp
 * \brief
 */

#include "sparse_flash_mla_tiling.h"
#include "checkers/checker_adapter.h"
#include "checkers/sparse_flash_mla_checker.h"
#include "../op_kernel/sparse_flash_mla_template_tiling_key.h"
#include "register/op_def_registry.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::pair;
using std::string;

namespace optiling {

static const std::string QUERY_NAME = "query";
static const std::string ORI_KV_NAME = "ori_kv";
static const std::string CMP_KV_NAME = "cmp_kv";
static const std::string CU_SEQLENS_ORI_KV_NAME = "cu_seqlens_ori_kv";
static const std::string CU_SEQLENS_CMP_KV_NAME = "cu_seqlens_cmp_kv";
static const std::string ORI_SPARSE_INDICES = "ori_sparse_indices";
static const std::string CMP_SPARSE_INDICES = "cmp_sparse_indices";
static const std::string ORI_BLOCK_TABLE_NAME = "ori_block_table";
static const std::string CMP_BLOCK_TABLE_NAME = "cmp_block_table";
static const std::string SINKS_NAME = "sinks";
static const std::string METADATA_NAME = "metadata";
static const std::string ATTEN_OUT_NAME = "attn_out";
static const std::string CU_SEQLENS_Q_NAME = "cu_seqlens_q";
static const std::string SEQUSED_Q_NAME = "seqused_q";
static const std::string SEQUSED_ORI_KV_NAME = "seqused_ori_kv";
static const std::string SEQUSED_CMP_KV_NAME = "seqused_cmp_kv";
static const std::string CMP_RESIDUAL_KV_NAME = "cmp_residual_kv";
static const std::string ORI_TOPK_LENGTH_NAME = "ori_topk_length";
static const std::string CMP_TOPK_LENGTH_NAME = "cmp_topk_length";
static const std::string A2_A3_PLATFORM_LOG = "A2/A3";
static const std::string A5_PLATFORM_LOG = "A5";
constexpr uint32_t FD_MAX_S2_SPLIT_NUM = 2U;
constexpr uint32_t BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM = 33U;
constexpr uint32_t FD_BROADCAST_ELEMS = 8U;
constexpr int64_t BATCH_CONSISTENCY_LEVEL = 3;

static std::vector<int64_t> ToVector(const gert::Shape &shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static std::string ToStringRaw(const gert::Shape &shape)
{
    std::ostringstream oss;
    auto v = ToVector(shape);
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << v[i] << ", ";
        }
        oss << v[v.size() - 1];
    }
    return oss.str();
}

static bool IsNonEmptyOptionalTensor(const gert::Tensor *tensor)
{
    return tensor != nullptr && tensor->GetShapeSize() > 0;
}

static bool IsPowerOfTwoInRange(uint32_t value, uint32_t minValue, uint32_t maxValue)
{
    return value >= minValue && value <= maxValue && (value & (value - 1U)) == 0U;
}

static bool IsA5Arch(NpuArch npuArch)
{
    return npuArch == NpuArch::DAV_3510;
}

static bool IsPaBlockSizeSupport(NpuArch npuArch, int32_t blockSize)
{
    if (IsA5Arch(npuArch)) {
        return blockSize >= 1 && blockSize <= static_cast<int32_t>(BLOCK_SIZE_LIMIT);
    }
    return blockSize >= 16U && blockSize <= static_cast<int32_t>(BLOCK_SIZE_LIMIT) && blockSize % 16U == 0;
}

static const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME, {ge::DT_FLOAT16, ge::DT_BF16}},
    {ORI_KV_NAME, {ge::DT_FLOAT16, ge::DT_BF16}},
    {CMP_KV_NAME, {ge::DT_FLOAT16, ge::DT_BF16}},
    {CU_SEQLENS_ORI_KV_NAME, {ge::DT_INT32}},
    {CU_SEQLENS_CMP_KV_NAME, {ge::DT_INT32}},
    {ORI_SPARSE_INDICES, {ge::DT_INT32}},
    {CMP_SPARSE_INDICES, {ge::DT_INT32}},
    {ATTEN_OUT_NAME, {ge::DT_FLOAT16, ge::DT_BF16}},
    {ORI_BLOCK_TABLE_NAME, {ge::DT_INT32}},
    {CMP_BLOCK_TABLE_NAME, {ge::DT_INT32}},
    {SINKS_NAME, {ge::DT_FLOAT}},
    {METADATA_NAME, {ge::DT_INT32}},
    {CU_SEQLENS_Q_NAME, {ge::DT_INT32}},
    {SEQUSED_Q_NAME, {ge::DT_INT32}},
    {SEQUSED_ORI_KV_NAME, {ge::DT_INT32}},
    {SEQUSED_CMP_KV_NAME, {ge::DT_INT32}},
    {CMP_RESIDUAL_KV_NAME, {ge::DT_INT32}},
    {ORI_TOPK_LENGTH_NAME, {ge::DT_INT32}},
    {CMP_TOPK_LENGTH_NAME, {ge::DT_INT32}}};

static const std::map<std::string, std::vector<SMLALayout>> LAYOUT_SUPPORT_MAP = {
    {QUERY_NAME, {SMLALayout::BSND, SMLALayout::TND}},
    {ORI_KV_NAME, {SMLALayout::PA_BBND, SMLALayout::TND, SMLALayout::BSND}},
    {CMP_KV_NAME, {SMLALayout::PA_BBND, SMLALayout::TND, SMLALayout::BSND}},
    {ATTEN_OUT_NAME, {SMLALayout::BSND, SMLALayout::TND}},
    {ORI_SPARSE_INDICES, {SMLALayout::BSND, SMLALayout::TND}},
    {CMP_SPARSE_INDICES, {SMLALayout::BSND, SMLALayout::TND}},
};

static const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
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

static uint64_t GetStorageShapeStride0(const gert::Shape &storageShape)
{
    if (storageShape.GetDimNum() <= DIM_NUM_ONE) {
        return 0ULL;
    }

    uint64_t stride0 = 1ULL;
    for (size_t i = 1; i < storageShape.GetDimNum(); ++i) {
        int64_t dim = storageShape.GetDim(i);
        if (dim <= 0) {
            return 0ULL;
        }
        stride0 *= static_cast<uint64_t>(dim);
    }
    return stride0;
}

template <typename StrideT>
static auto GetStride0FromStrideObject(const StrideT &stride, int) -> decltype(stride.GetDimNum(), stride.GetStride(0),
                                                                               uint64_t())
{
    if (stride.GetDimNum() <= 0) {
        return 0ULL;
    }
    int64_t stride0 = stride.GetStride(0);
    return stride0 > 0 ? static_cast<uint64_t>(stride0) : 0ULL;
}

template <typename StrideT>
static uint64_t GetStride0FromStrideObject(const StrideT &, ...)
{
    return 0ULL;
}

template <typename StrideT>
static auto GetStride0FromStrideScalar(const StrideT &stride, int) -> decltype(stride > 0,
                                                                               static_cast<uint64_t>(stride))
{
    return stride > 0 ? static_cast<uint64_t>(stride) : 0ULL;
}

template <typename StrideT>
static uint64_t GetStride0FromStrideScalar(const StrideT &, ...)
{
    return 0ULL;
}

template <typename StrideT>
static uint64_t GetStride0FromStrideElement(const StrideT &stride)
{
    // CANN stride APIs return a dimension-wise stride array. In newer headers, stride[0] is scalar stride0.
    // In compatibility headers it may be a stride object. Non-positive stride is treated as unavailable and
    // falls back to the storage-shape contiguous calculation.
    uint64_t stride0 = GetStride0FromStrideScalar(stride, 0);
    if (stride0 > 0) {
        return stride0;
    }
    return GetStride0FromStrideObject(stride, 0);
}

template <typename StrideT>
static uint64_t GetStride0FromStrideArray(const StrideT *stride)
{
    if (stride == nullptr) {
        return 0ULL;
    }
    return GetStride0FromStrideElement(stride[0]);
}

template <typename ContextT>
static auto TryGetOptionalInputStride0(ContextT *context, uint32_t inputIndex,
                                       int) -> decltype(context->GetOptionalInputStride(inputIndex), uint64_t())
{
    return GetStride0FromStrideArray(context->GetOptionalInputStride(inputIndex));
}

template <typename ContextT>
static uint64_t TryGetOptionalInputStride0(ContextT *, uint32_t, ...)
{
    return 0ULL;
}

// Compatibility path for CANN headers that do not expose GetOptionalInputStride.
// Some tiling contexts only provide real stride for view inputs through InputIsView/GetInputStride.
// Returning 0 means the stride is unavailable; the caller then falls back to storage-shape contiguous stride.
template <typename ContextT>
static auto TryGetInputViewStride0(ContextT *context, uint32_t inputIndex,
                                   int) -> decltype(context->InputIsView(inputIndex),
                                                    context->GetInputStride(inputIndex), uint64_t())
{
    if (!context->InputIsView(inputIndex)) {
        return 0ULL;
    }
    return GetStride0FromStrideArray(context->GetInputStride(inputIndex));
}

template <typename ContextT>
static uint64_t TryGetInputViewStride0(ContextT *, uint32_t, ...)
{
    return 0ULL;
}

std::string SMLALayoutToSerialString(SMLALayout layout)
{
    switch (layout) {
        case SMLALayout::BSND:
            return "BSND";
        case SMLALayout::TND:
            return "TND";
        case SMLALayout::PA_BBND:
            return "PA_BBND";
        default:
            return "UNKNOWN";
    }
}

struct SMLACompileInfo {
    int64_t core_num;
};

static const std::map<SMLALayout, std::vector<SMLAAxis>> SMLA_LAYOUT_AXIS_MAP = {
    {SMLALayout::BSND, {SMLAAxis::B, SMLAAxis::S, SMLAAxis::N, SMLAAxis::D}},
    {SMLALayout::TND, {SMLAAxis::T, SMLAAxis::N, SMLAAxis::D}},
    {SMLALayout::PA_BBND, {SMLAAxis::Bn, SMLAAxis::Bs, SMLAAxis::N, SMLAAxis::D}},
};

static const std::map<SMLALayout, size_t> SMLA_LAYOUT_DIM_MAP = {
    {SMLALayout::BSND, DIM_NUM_FOUR},
    {SMLALayout::TND, DIM_NUM_THREE},
    {SMLALayout::PA_BBND, DIM_NUM_FOUR},
};

static std::string SMLADataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("sparseFlashMla", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

// --------------------------SMLAInfoParser类成员函数定义------------------------------------
ge::graphStatus SMLAInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.q.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "query", "The shape of query is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.q.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "query", "The desc of query is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.oriKv.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "ori_kv", "The tensor of ori_kv is nullptr"),
                return ge::GRAPH_FAILED);
    if (std::string(opParamInfo_.layoutKv) == "PA_BBND") {
        OP_CHECK_IF(
            opParamInfo_.oriBlockTable.tensor == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                opName_, "ori_block_table", "The tensor of ori_block_table is nullptr when layoutKv is PA_BBND"),
            return ge::GRAPH_FAILED);
    }
    if (perfMode_ == SMLATemplateMode::HCA_TEMPLATE_MODE) {
        OP_CHECK_IF(opParamInfo_.cmpKv.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "cmp_kv", "The tensor of cmp_kv is nullptr"),
                    return ge::GRAPH_FAILED);
    }
    if (perfMode_ == SMLATemplateMode::CSA_TEMPLATE_MODE) {
        OP_CHECK_IF(opParamInfo_.cmpKv.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "cmp_kv", "The tensor of cmp_kv is nullptr"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.cmpSparseIndices.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "cmp_sparse_indices",
                                                             "The tensor of cmp_sparse_indices is nullptr"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::CheckRequiredAttrExistence() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::CheckUnrequiredParaExistence() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("SparseFlashMla", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0, OP_LOGE(opName_, "num of core obtained is 0."), return ge::GRAPH_FAILED);

    npuArch_ = ascendcPlatform.GetCurNpuArch();
    if (npuArch_ != NpuArch::DAV_2201 && npuArch_ != NpuArch::DAV_3510) {
        OP_LOGE(opName_, "Npu Arch Version[%d] is not support.", static_cast<int32_t>(npuArch_));
        return ge::GRAPH_FAILED;
    }
    batchConsistency_ = (context_->GetDeterministicLevel() == BATCH_CONSISTENCY_LEVEL);
    OP_LOGD(opName_, "deterministic_level=%d", context_->GetDeterministicLevel());

    return ge::GRAPH_SUCCESS;
}

void SMLAInfoParser::GetOptionalInputParaInfo()
{
    sparse_mla_checker::PopulateOptionalTensorParam(context_, ORI_KV_INDEX, opParamInfo_.oriKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, CMP_KV_INDEX, opParamInfo_.cmpKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, ORI_SPARSE_INDICES_INDEX, opParamInfo_.oriSparseIndices);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, CMP_SPARSE_INDICES_INDEX, opParamInfo_.cmpSparseIndices);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, ORI_BLOCK_TABLE_INDEX, opParamInfo_.oriBlockTable);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, CMP_BLOCK_TABLE_INDEX, opParamInfo_.cmpBlockTable);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, SINKS_INDEX, opParamInfo_.sinks);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, CU_SEQLENS_Q_INDEX, opParamInfo_.cuSeqLensQ);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, CU_SEQLENS_ORI_KV_INDEX, opParamInfo_.cuSeqLensOriKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, CU_SEQLENS_CMP_KV_INDEX, opParamInfo_.cuSeqLensCmpKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, SEQUSED_Q_INDEX, opParamInfo_.seqUsedQ);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, SEQUSED_ORI_KV_INDEX, opParamInfo_.sequsedOriKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, SEQUSED_CMP_KV_INDEX, opParamInfo_.sequsedCmpKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, CMP_RESIDUAL_KV_INDEX, opParamInfo_.cmpResidualKv);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, ORI_TOPK_LENGTH_INDEX, opParamInfo_.oriTopkLength);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, CMP_TOPK_LENGTH_INDEX, opParamInfo_.cmpTopkLength);
    sparse_mla_checker::PopulateOptionalTensorParam(context_, METADATA_INDEX, opParamInfo_.metadata);
}

void SMLAInfoParser::GetInputParaInfo()
{
    opParamInfo_.q.desc = context_->GetInputDesc(Q_INDEX);
    opParamInfo_.q.shape = context_->GetInputShape(Q_INDEX);
    GetOptionalInputParaInfo();
}

void SMLAInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attnOut.desc = context_->GetOutputDesc(ATTN_OUT_INDEX);
    opParamInfo_.attnOut.shape = context_->GetOutputShape(ATTN_OUT_INDEX);
    opParamInfo_.softmaxLse.desc = context_->GetOutputDesc(SOFTMAX_LSE_INDEX);
    opParamInfo_.softmaxLse.shape = context_->GetOutputShape(SOFTMAX_LSE_INDEX);
}

ge::graphStatus SMLAInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo start");
    opParamInfo_.softmaxScale = attrs->GetAttrPointer<float>(ATTR_SOFTMAX_SCALE_INDEX);
    opParamInfo_.cmpRatio = attrs->GetAttrPointer<uint32_t>(ATTR_CMP_RATIO_INDEX);
    opParamInfo_.oriMaskMode = attrs->GetAttrPointer<uint32_t>(ATTR_ORI_MASK_MODE_INDEX);
    opParamInfo_.cmpMaskMode = attrs->GetAttrPointer<uint32_t>(ATTR_CMP_MASK_MODE_INDEX);
    opParamInfo_.oriWinLeft = attrs->GetAttrPointer<int32_t>(ATTR_ORI_WIN_LEFT_INDEX);
    opParamInfo_.oriWinRight = attrs->GetAttrPointer<int32_t>(ATTR_ORI_WIN_RIGHT_INDEX);
    opParamInfo_.layoutQ = attrs->GetStr(ATTR_LAYOUT_Q_INDEX);
    opParamInfo_.layoutKv = attrs->GetStr(ATTR_LAYOUT_KV_INDEX);
    opParamInfo_.topkValueMode = attrs->GetAttrPointer<uint32_t>(ATTR_TOPK_VALUE_MODE_INDEX);
    opParamInfo_.returnSoftmaxLse = attrs->GetAttrPointer<bool>(ATTR_RETURN_SOFTMAX_LSE_INDEX);

    auto oriKeyStrides = context_->GetDynamicInputStride(ORI_KV_INDEX, 0);
    if (oriKeyStrides != nullptr && oriKeyStrides->GetDimNum() > 0) {
        for (size_t i = 0; i < oriKeyStrides->GetDimNum(); i++) {
            oriKeyStridesVec_.push_back(oriKeyStrides->GetStride(i));
        }
    }
    auto cmpKeyStrides = context_->GetDynamicInputStride(CMP_KV_INDEX, 0);
    if (cmpKeyStrides != nullptr && cmpKeyStrides->GetDimNum() > 0) {
        for (size_t i = 0; i < cmpKeyStrides->GetDimNum(); i++) {
            cmpKeyStridesVec_.push_back(cmpKeyStrides->GetStride(i));
        }
    }

    OP_LOGI(context_->GetNodeName(), "GetAttrParaInfo end");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t SMLAInfoParser::GetOptionalInputStride0(uint32_t inputIndex) const
{
    const gert::Tensor *inputTensor = nullptr;
    if (inputIndex == ORI_KV_INDEX) {
        inputTensor = opParamInfo_.oriKv.tensor;
    } else if (inputIndex == CMP_KV_INDEX) {
        inputTensor = opParamInfo_.cmpKv.tensor;
    }
    if (inputTensor == nullptr) {
        return 0ULL;
    }

    uint64_t stride0 = TryGetOptionalInputStride0(context_, inputIndex, 0);
    if (stride0 > 0) {
        return stride0;
    }

    // Compatible with CANN packages that only expose view stride by normal input index.
    stride0 = TryGetInputViewStride0(context_, inputIndex, 0);
    if (stride0 > 0) {
        return stride0;
    }

    const gert::Shape &storageShape = inputTensor->GetStorageShape();
    stride0 = GetStorageShapeStride0(storageShape);
    const char *inputName = inputIndex == ORI_KV_INDEX ? "ori_kv" : "cmp_kv";
    OP_LOGW(context_->GetNodeName(),
            "Cannot get %s stride0 from tiling context stride APIs. Use storage shape to infer contiguous "
            "stride0(%lu). Non-contiguous %s requires GetOptionalInputStride or GetInputStride support.",
            inputName, stride0, inputName);
    return stride0;
}
ge::graphStatus SMLAInfoParser::GetInOutDataType()
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

ge::graphStatus SMLAInfoParser::GetSMLATemplateMode()
{
    if (opParamInfo_.oriKv.desc != nullptr) {
        if (opParamInfo_.cmpKv.desc != nullptr && opParamInfo_.cmpSparseIndices.tensor != nullptr) {
            if (opParamInfo_.oriSparseIndices.tensor != nullptr) {
                perfMode_ = SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE;
            } else {
                perfMode_ = SMLATemplateMode::CSA_TEMPLATE_MODE;
            }
        } else if (opParamInfo_.cmpKv.desc != nullptr && opParamInfo_.cmpSparseIndices.tensor == nullptr) {
            perfMode_ = SMLATemplateMode::HCA_TEMPLATE_MODE;
        } else if (opParamInfo_.cmpKv.desc == nullptr && opParamInfo_.cmpSparseIndices.tensor == nullptr) {
            if (opParamInfo_.oriSparseIndices.tensor != nullptr) {
                // A2A3此处dspark用 SWA_TEMPLATE_MODE+hasOri判断；给A5留ORI_SPARSE_TEMPLATE_MODE分支
                if (IsA5Arch(npuArch_)) {
                    perfMode_ = SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE;
                } else {
                    // DSpark: oriMaskMode=0 + ori_sparse_indices on SWA kernel path.
                    if (opParamInfo_.oriMaskMode == nullptr || *opParamInfo_.oriMaskMode != 0U) {
                        OP_LOGE(opName_, "SWA ori sparse (DSpark) requires oriMaskMode 0, but got %u.",
                                opParamInfo_.oriMaskMode != nullptr ? *opParamInfo_.oriMaskMode : UINT32_MAX);
                        return ge::GRAPH_FAILED;
                    }
                    perfMode_ = SMLATemplateMode::SWA_TEMPLATE_MODE;
                }
            } else {
                perfMode_ = SMLATemplateMode::SWA_TEMPLATE_MODE;
            }
        } else {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                opName_, "cmp_kv", "When cmp_sparse_indices is not nullptr, cmp_kv cannot be nullptr");
            return ge::GRAPH_FAILED;
        }
        if (perfMode_ == SMLATemplateMode::HCA_TEMPLATE_MODE || perfMode_ == SMLATemplateMode::CSA_TEMPLATE_MODE) {
            if (kvLayout_ == SMLALayout::TND && opParamInfo_.cuSeqLensCmpKv.tensor == nullptr) {
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    opName_, "cu_seqlens_cmp_kv",
                    "The layout_kv is" + SMLALayoutToSerialString(kvLayout_) + ", seqlens_cmp_kv must be provided");
                return ge::GRAPH_FAILED;
            }
        }
    } else {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "ori_kv", "ori_kv is nullptr");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::GetQueryAndOutLayout()
{
    const map<string, pair<SMLALayout, SMLALayout>> layoutMap = {
        {"BSND", {SMLALayout::BSND, SMLALayout::BSND}},
        {"TND", {SMLALayout::TND, SMLALayout::TND}},
    };
    std::string layout(opParamInfo_.layoutQ);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second.first;
        outLayout_ = it->second.second;
        oriSparseIndicesLayout_ = qLayout_;
        cmpSparseIndicesLayout_ = qLayout_;
    } else {
        OP_LOGE_FOR_INVALID_VALUE(opName_, "layout_q", layout.c_str(), "BSND or TND");
        return ge::GRAPH_FAILED;
    }
    if (qLayout_ == SMLALayout::BSND) {
        OP_CHECK_IF(opParamInfo_.cuSeqLensQ.tensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "cu_seqlens_q",
                                                             "When layout_q is BSND, cu_seqlens_q should be null"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::GetKvLayout()
{
    const map<string, SMLALayout> layoutKVMap = {
        {"PA_BBND", SMLALayout::PA_BBND},
        {"TND", SMLALayout::TND},
        {"BSND", SMLALayout::BSND},
    };
    std::string layout(opParamInfo_.layoutKv);
    auto it = layoutKVMap.find(layout);
    if (it != layoutKVMap.end()) {
        kvLayout_ = it->second;
    } else {
        OP_LOGE_FOR_INVALID_VALUE(opName_, "layout_kv", layout.c_str(), "BSND, PA_BBND or TND");
        return ge::GRAPH_FAILED;
    }
    if (kvLayout_ != SMLALayout::PA_BBND && qLayout_ != kvLayout_) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName_, "layout_q and layout_kv",
            SMLALayoutToSerialString(qLayout_) + " and " + SMLALayoutToSerialString(kvLayout_),
            "Layout_q and layout_kv only support BSND/BSND, TND/TND, BSND/PA_BBND or TND/PA_BBND");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// =============Parser function====================
bool SMLAInfoParser::HasAxis(const SMLAAxis &axis, const SMLALayout &layout, const gert::Shape &shape) const
{
    const auto &layoutIt = SMLA_LAYOUT_AXIS_MAP.find(layout);
    if (layoutIt == SMLA_LAYOUT_AXIS_MAP.end()) {
        return false;
    }

    const std::vector<SMLAAxis> &axes = layoutIt->second;
    const auto &axisIt = std::find(axes.begin(), axes.end(), axis);
    if (axisIt == axes.end()) {
        return false;
    }
    const auto &dimIt = SMLA_LAYOUT_DIM_MAP.find(layout);
    if (dimIt == SMLA_LAYOUT_DIM_MAP.end() || dimIt->second != shape.GetDimNum()) {
        return false;
    }
    return true;
}

size_t SMLAInfoParser::GetAxisIdx(const SMLAAxis &axis, const SMLALayout &layout) const
{
    const std::vector<SMLAAxis> &axes = SMLA_LAYOUT_AXIS_MAP.find(layout)->second;
    const auto &axisIt = std::find(axes.begin(), axes.end(), axis);
    return std::distance(axes.begin(), axisIt);
}

uint32_t SMLAInfoParser::GetAxisNum(const gert::Shape &shape, const SMLAAxis &axis, const SMLALayout &layout) const
{
    return HasAxis(axis, layout, shape) ? shape.GetDim(GetAxisIdx(axis, layout)) : invalidDimValue_;
}

void SMLAInfoParser::SetSMLAShape()
{
    qShape_ = opParamInfo_.q.shape->GetStorageShape();
    if (opParamInfo_.oriKv.tensor != nullptr) {
        oriKvShape_ = opParamInfo_.oriKv.tensor->GetStorageShape();
    } else {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "q", "Q is nullptr, please check input parameters");
    }
    if (opParamInfo_.cmpKv.tensor != nullptr) {
        cmpKvShape_ = opParamInfo_.cmpKv.tensor->GetStorageShape();
    }
    if (opParamInfo_.oriSparseIndices.tensor != nullptr) {
        oriSparseIndicesShape_ = opParamInfo_.oriSparseIndices.tensor->GetStorageShape();
        hasOriSparseIndices_ = true;
        oriSparseIndexWidth_ = GetAxisNum(oriSparseIndicesShape_, SMLAAxis::K, oriSparseIndicesLayout_);
    }
    if (perfMode_ == SMLATemplateMode::CSA_TEMPLATE_MODE ||
        perfMode_ == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
            cmpSparseIndicesShape_ = opParamInfo_.cmpSparseIndices.tensor->GetStorageShape();
        } else {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "cmp_sparse_indices",
                                                     "Cmp_sparse_indices is nullptr, please check input parameters");
        }
    }

    if (perfMode_ == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
        perfMode_ == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (opParamInfo_.oriSparseIndices.tensor != nullptr) {
            oriSparseIndicesShape_ = opParamInfo_.oriSparseIndices.tensor->GetStorageShape();
        }
    }
}

// 根据layout计算期望的连续stride
std::vector<uint64_t> SMLAInfoParser::GetKvstride(const gert::Shape &shape, const SMLALayout &layout) const
{
    std::vector<uint64_t> expectedStrides;
    if (layout == SMLALayout::BSND || layout == SMLALayout::PA_BBND) {
        uint64_t dim1 = static_cast<uint64_t>(shape.GetDim(1));
        uint64_t dim2 = static_cast<uint64_t>(shape.GetDim(2));
        uint64_t dim3 = static_cast<uint64_t>(shape.GetDim(3));
        expectedStrides = {dim1 * dim2 * dim3, dim2 * dim3, dim3, 1};
    } else if (layout == SMLALayout::TND) {
        uint64_t dim1 = static_cast<uint64_t>(shape.GetDim(1));
        uint64_t dim2 = static_cast<uint64_t>(shape.GetDim(2));
        expectedStrides = {dim1 * dim2, dim2, 1};
    }
    return expectedStrides;
}

// 非连续校验：通过shape计算expected stride进行校验
// PA_BBND时，只允许0轴非连续，其余轴必须连续
// 非PA_BBND时，所有轴都必须连续
ge::graphStatus SMLAInfoParser::CheckContiguous() const
{
    bool oriKeyNonContiguous = false;
    bool cmpKeyNonContiguous = false;
    size_t checkStartIdx = (kvLayout_ == SMLALayout::PA_BBND) ? 1 : 0;
    if (opParamInfo_.oriKv.tensor != nullptr && !oriKeyStridesVec_.empty() &&
        opParamInfo_.oriKv.tensor->GetShapeSize() > 0) {
        std::vector<uint64_t> oriExpectedStrides = GetKvstride(oriKvShape_, kvLayout_);
        OP_CHECK_IF(oriKeyStridesVec_.size() != oriExpectedStrides.size(),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "ori_kv",
                        "Ori_kv strideVec size[" + std::to_string(oriKeyStridesVec_.size()) +
                            "] not match layout_kv expect len[" + std::to_string(oriExpectedStrides.size()) + "]"),
                    return ge::GRAPH_FAILED);
        oriKeyNonContiguous =
            static_cast<uint64_t>(oriKeyStridesVec_[checkStartIdx]) != oriExpectedStrides[checkStartIdx];
    }
    if (opParamInfo_.cmpKv.tensor != nullptr && !cmpKeyStridesVec_.empty() &&
        opParamInfo_.cmpKv.tensor->GetShapeSize() > 0) {
        std::vector<uint64_t> cmpExpectedStrides = GetKvstride(cmpKvShape_, kvLayout_);
        OP_CHECK_IF(cmpKeyStridesVec_.size() != cmpExpectedStrides.size(),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "cmp_kv",
                        "Cmp_kv strideVec size[" + std::to_string(cmpKeyStridesVec_.size()) +
                            "] not match kvLayout expect len[" + std::to_string(cmpExpectedStrides.size()) + "]"),
                    return ge::GRAPH_FAILED);
        cmpKeyNonContiguous =
            static_cast<uint64_t>(cmpKeyStridesVec_[checkStartIdx]) != cmpExpectedStrides[checkStartIdx];
    }

    OP_CHECK_IF(oriKeyNonContiguous,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "ori_kv",
                                                         "Ori_kv only support non-continuous keying on the 0-axis"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(cmpKeyNonContiguous,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "cmp_kv",
                                                         "Cmp_kv only support non-continuous keying on the 0-axis"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::GetN1Size()
{
    n1Size_ = GetAxisNum(qShape_, SMLAAxis::N, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::GetN2Size()
{
    if (opParamInfo_.oriKv.tensor != nullptr) {
        n2Size_ = GetAxisNum(oriKvShape_, SMLAAxis::N, kvLayout_);
    }
    if (opParamInfo_.cmpKv.tensor != nullptr) {
        n2Size_ = GetAxisNum(cmpKvShape_, SMLAAxis::N, kvLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::GetGSize()
{
    if (n2Size_ != 0) {
        gSize_ = n1Size_ / n2Size_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor, SMLALayout &layout,
                                                    const std::string &name) const
{
    if ((tensor == nullptr)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName_, name.c_str(),
            "When layout_q is " + SMLALayoutToSerialString(layout) + ", " + name + " must be provided");
        return ge::GRAPH_FAILED;
    }
    int64_t shapeSize = tensor->GetShapeSize();
    if (shapeSize <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, name.c_str(), std::to_string(shapeSize).c_str(),
                                                  "The shape size of " + name + " should be greater than 0");
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(shapeSize) - 1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::GetActualSeqLenQSize(uint32_t &size)
{
    return GetActualSeqLenSize(size, opParamInfo_.cuSeqLensQ.tensor, qLayout_, "cu_seqlens_q");
}

ge::graphStatus SMLAInfoParser::GetBatchSize()
{
    // 获取B基准    // 1、非TND: 以query的batch_size维度为基
    // 2、TND: actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if (qLayout_ == SMLALayout::TND) {
        return GetActualSeqLenQSize(bSize_);
    } else { // BSND
        bSize_ = GetAxisNum(qShape_, SMLAAxis::B, qLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::GetQTSize()
{
    // 获取query的T基准    // 1、非TND: 以query的batch_size维度为基准
    // 2、TND: actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    qTSize_ = (qLayout_ == SMLALayout::TND) ? GetAxisNum(qShape_, SMLAAxis::T, qLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::GetS1Size()
{
    // 获取S1基准    // 1、非TND: 以query的S维度为基准
    // 2、TND: actual_seq_lens_q必须传入, 以actual_seq_lens_q数组中的最大值为基准
    if (qLayout_ == SMLALayout::TND) {
        s1Size_ = GetAxisNum(qShape_, SMLAAxis::T, qLayout_);
    } else { // BSND
        s1Size_ = GetAxisNum(qShape_, SMLAAxis::S, qLayout_);
    }
    if (perfMode_ == SMLATemplateMode::CSA_TEMPLATE_MODE ||
        perfMode_ == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (cmpSparseIndicesLayout_ == SMLALayout::TND) {
            uint32_t cmpSparseIndicesT = GetAxisNum(cmpSparseIndicesShape_, SMLAAxis::T, cmpSparseIndicesLayout_);
            OP_CHECK_IF(
                cmpSparseIndicesT != s1Size_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "cmp_sparse_indices", Ops::Base::ToString(cmpSparseIndicesShape_), "T size check failed"),
                return ge::GRAPH_FAILED);
        } else {
            uint32_t cmpSparseIndicesS1 = GetAxisNum(cmpSparseIndicesShape_, SMLAAxis::S, cmpSparseIndicesLayout_);
            OP_CHECK_IF(
                cmpSparseIndicesS1 != s1Size_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "cmp_sparse_indices", Ops::Base::ToString(cmpSparseIndicesShape_), "S1 size check failed"),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::GetMaxBlockNumPerBatch()
{
    if (opParamInfo_.oriBlockTable.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t oriDimNum = opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDimNum();
    if (oriDimNum != DIM_NUM_TWO) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "ori_block_table", std::to_string(oriDimNum).c_str(),
                                     std::to_string(DIM_NUM_TWO).c_str());
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(1) < 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName_, "ori_block_table", ToStringRaw(opParamInfo_.oriBlockTable.tensor->GetStorageShape()).c_str(),
            "Ori_block_table's second dimension(" +
                std::to_string(opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(1)) +
                ") should be non-negative number");
        return ge::GRAPH_FAILED;
    }
    oriMaxBlockNumPerBatch_ = opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(1);

    if (opParamInfo_.cmpBlockTable.tensor != nullptr) {
        uint32_t cmpDimNum = opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDimNum();
        if (cmpDimNum != DIM_NUM_TWO) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "cmp_block_table", std::to_string(cmpDimNum).c_str(),
                                         std::to_string(DIM_NUM_TWO).c_str());
            return ge::GRAPH_FAILED;
        }
        if (qLayout_ == SMLALayout::TND || qLayout_ == SMLALayout::BSND) {
            if (opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(0) != bSize_) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "cmp_block_table",
                    ToStringRaw(opParamInfo_.cmpBlockTable.tensor->GetStorageShape()).c_str(),
                    "Cmp_block_table's first dimension(" +
                        std::to_string(opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(0)) +
                        ") should be equal to query's B(" + std::to_string(bSize_) + ")");
                return ge::GRAPH_FAILED;
            }
        }
        if (opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(1) <= 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                opName_, "cmp_block_table", ToStringRaw(opParamInfo_.cmpBlockTable.tensor->GetStorageShape()).c_str(),
                "cmp_block_table's second dimension(" +
                    std::to_string(opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(1)) +
                    ") should be greater than 0");
            return ge::GRAPH_FAILED;
        }
        cmpMaxBlockNumPerBatch_ = opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::GetBlockSize()
{
    oriBlockSize_ = GetAxisNum(oriKvShape_, SMLAAxis::Bs, kvLayout_);
    cmpBlockSize_ = GetAxisNum(cmpKvShape_, SMLAAxis::Bs, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::GetS2SizeForPageAttention()
{
    if (GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS || GetBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    s2Size_ = oriMaxBlockNumPerBatch_ * oriBlockSize_;
    cmpS2Size_ = cmpMaxBlockNumPerBatch_ * cmpBlockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::GetS2Size()
{
    if (kvLayout_ == SMLALayout::TND) {
        s2Size_ = GetAxisNum(oriKvShape_, SMLAAxis::T, kvLayout_);
        cmpS2Size_ = GetAxisNum(cmpKvShape_, SMLAAxis::T, kvLayout_);
        return ge::GRAPH_SUCCESS;
    } else if (kvLayout_ == SMLALayout::BSND) {
        s2Size_ = GetAxisNum(oriKvShape_, SMLAAxis::S, kvLayout_);
        cmpS2Size_ = GetAxisNum(cmpKvShape_, SMLAAxis::S, kvLayout_);
        return ge::GRAPH_SUCCESS;
    } else if (kvLayout_ == SMLALayout::PA_BBND) {
        // 获取S2基准PAGE_ATTENTION S2 = block_table.dim1 * block_size
        return GetS2SizeForPageAttention();
    }
    return ge::GRAPH_FAILED;
}

ge::graphStatus SMLAInfoParser::GetQHeadDim()
{
    qHeadDim_ = GetAxisNum(qShape_, SMLAAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::GetValueHeadDim()
{
    if (opParamInfo_.oriKv.tensor != nullptr) {
        oriKvHeadDim_ = GetAxisNum(oriKvShape_, SMLAAxis::D, kvLayout_);
    }
    if (opParamInfo_.cmpKv.tensor != nullptr) {
        cmpKvHeadDim_ = GetAxisNum(cmpKvShape_, SMLAAxis::D, kvLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::GetSparseBlockCount()
{
    if (opParamInfo_.oriSparseIndices.tensor != nullptr) {
        oriSparseBlockCount_ = GetAxisNum(oriSparseIndicesShape_, SMLAAxis::K, oriSparseIndicesLayout_);
    }
    if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
        cmpSparseBlockCount_ = GetAxisNum(cmpSparseIndicesShape_, SMLAAxis::K, cmpSparseIndicesLayout_);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::GetSinks()
{
    if (opParamInfo_.sinks.tensor == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sinks", "sinks must be provided");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLAInfoParser::GetActualseqInfo()
{
    if (qLayout_ == SMLALayout::TND) {
        if (opParamInfo_.cuSeqLensQ.tensor != nullptr) {
            if (opParamInfo_.cuSeqLensQ.tensor->GetShapeSize() != bSize_ + 1) {
                OP_LOGE_FOR_INVALID_SHAPESIZE(opName_, "cu_seqlens_q",
                                              std::to_string(opParamInfo_.cuSeqLensQ.tensor->GetShapeSize()).c_str(),
                                              std::to_string(bSize_ + 1));
                return ge::GRAPH_FAILED;
            }
            actualLenDimsQ_ = opParamInfo_.cuSeqLensQ.tensor->GetShapeSize() - 1; // cuSeqLensQ shape is B+1
        } else {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "cu_seqlens_q",
                                                     "When layout_q is TND, cu_seqlens_q must be provided");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (opParamInfo_.seqUsedQ.tensor != nullptr) {
            actualLenDimsQ_ = opParamInfo_.seqUsedQ.tensor->GetShapeSize();
        }
    }
    if (kvLayout_ != SMLALayout::PA_BBND && kvLayout_ != SMLALayout::BSND && kvLayout_ != SMLALayout::TND) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_kv", SMLALayoutToSerialString(kvLayout_),
                                              "Ori_kv and cmp_kv only support PA_BBND, BSND and TND");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.sequsedOriKv.tensor != nullptr) {
        actualLenDimsOriKV_ = opParamInfo_.sequsedOriKv.tensor->GetShapeSize();
    }
    if (opParamInfo_.sequsedCmpKv.tensor != nullptr) {
        actualLenDimsCmpKV_ = opParamInfo_.sequsedCmpKv.tensor->GetShapeSize();
        if (opParamInfo_.sequsedCmpKv.tensor->GetShapeSize() != bSize_) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                opName_, "seqused_cmp_kv", std::to_string(opParamInfo_.sequsedCmpKv.tensor->GetShapeSize()),
                "Seqused_cmp_kv's dimension should be equal to " + std::to_string(bSize_));
            return ge::GRAPH_FAILED;
        }
    }
    if (opParamInfo_.cmpResidualKv.tensor != nullptr) {
        cmpResidualKVSize_ = opParamInfo_.cmpResidualKv.tensor->GetShapeSize();
        if (opParamInfo_.cmpResidualKv.tensor->GetShapeSize() != bSize_) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                opName_, "cmp_residual_kv", std::to_string(opParamInfo_.cmpResidualKv.tensor->GetShapeSize()),
                "Cmp_residual_kv's dimension should be equal to " + std::to_string(bSize_));
            return ge::GRAPH_FAILED;
        }
    }
    if (!IsA5Arch(npuArch_) && IsNonEmptyOptionalTensor(opParamInfo_.cmpTopkLength.tensor)) {
        OP_LOGE(opName_, "cmp_topk_length is reserved and does not support non-empty tensor on %s.",
                A2_A3_PLATFORM_LOG.c_str());
        return ge::GRAPH_FAILED;
    }
    if (kvLayout_ == SMLALayout::PA_BBND) {
        if (opParamInfo_.sequsedOriKv.tensor != nullptr) {
            if (qLayout_ == SMLALayout::BSND) {
                if (opParamInfo_.sequsedOriKv.tensor->GetShapeSize() != bSize_) {
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        opName_, "seqused_ori_kv", std::to_string(opParamInfo_.sequsedOriKv.tensor->GetShapeSize()),
                        "Seqused_ori_kv's dimension should be equal to " + std::to_string(bSize_));
                    return ge::GRAPH_FAILED;
                }
            } else {
                if (opParamInfo_.sequsedOriKv.tensor->GetShapeSize() != bSize_) {
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        opName_, "seqused_ori_kv", std::to_string(opParamInfo_.sequsedOriKv.tensor->GetShapeSize()),
                        "Seqused_ori_kv's dimension should be equal to " + std::to_string(bSize_));
                    return ge::GRAPH_FAILED;
                }
            }
            actualLenDimsKV_ = opParamInfo_.sequsedOriKv.tensor->GetShapeSize();
        } else if (perfMode_ != SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE &&
                   perfMode_ != SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "seqused_ori_kv",
                                                     "Seqused_ori_kv must be provided when layout_kv is PA_BBND");
            return ge::GRAPH_FAILED;
        }
    } else if (kvLayout_ == SMLALayout::TND) {
    } else if (kvLayout_ == SMLALayout::BSND) {
        actualLenDimsKV_ = actualLenDimsOriKV_;
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_kv", SMLALayoutToSerialString(kvLayout_),
                                              "Ori_kv and cmp_kv only support PA_BBND, TND and BSND");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void SMLAInfoParser::GenerateInfo(SMLATilingInfo &smlaInfo)
{
    smlaInfo.opName = opName_;
    smlaInfo.platformInfo = platformInfo_;
    smlaInfo.opParamInfo = opParamInfo_;
    smlaInfo.npuArch = npuArch_;

    smlaInfo.bSize = bSize_;
    smlaInfo.n1Size = n1Size_;
    smlaInfo.n2Size = n2Size_;
    smlaInfo.s1Size = s1Size_;
    smlaInfo.s2Size = s2Size_;
    smlaInfo.cmpS2Size = cmpS2Size_;
    smlaInfo.gSize = gSize_;
    smlaInfo.qHeadDim = qHeadDim_;
    smlaInfo.oriKvHeadDim = oriKvHeadDim_;
    smlaInfo.cmpKvHeadDim = cmpKvHeadDim_;
    smlaInfo.qTSize = qTSize_;
    smlaInfo.oriSparseBlockCount = oriSparseBlockCount_;
    smlaInfo.cmpSparseBlockCount = cmpSparseBlockCount_;
    smlaInfo.sparseBlockCount = cmpSparseBlockCount_;
    smlaInfo.hasOriSparseIndices = hasOriSparseIndices_;
    smlaInfo.oriSparseIndexWidth = oriSparseIndexWidth_;
    smlaInfo.oriWinLeft = oriWinLeft_;
    smlaInfo.oriWinRight = oriWinRight_;
    smlaInfo.qType = qType_;
    smlaInfo.oriKvType = oriKvType_;
    smlaInfo.cmpKvType = cmpKvType_;
    smlaInfo.outputType = outputType_;
    smlaInfo.perfMode = perfMode_;

    smlaInfo.sparseBlockSize = 1;
    smlaInfo.oriBlockSize = oriBlockSize_;
    smlaInfo.cmpBlockSize = cmpBlockSize_;
    smlaInfo.oriMaxBlockNumPerBatch = oriMaxBlockNumPerBatch_;
    smlaInfo.cmpMaxBlockNumPerBatch = cmpMaxBlockNumPerBatch_;

    smlaInfo.actualLenDimsQ = actualLenDimsQ_;
    smlaInfo.actualLenDimsKV = actualLenDimsKV_;

    smlaInfo.softmaxScale = *opParamInfo_.softmaxScale;
    smlaInfo.cmpRatio = *opParamInfo_.cmpRatio;
    smlaInfo.oriMaskMode = *opParamInfo_.oriMaskMode;
    smlaInfo.cmpMaskMode = *opParamInfo_.cmpMaskMode;
    smlaInfo.oriKvStride0 = GetOptionalInputStride0(ORI_KV_INDEX);
    smlaInfo.cmpKvStride0 = GetOptionalInputStride0(CMP_KV_INDEX);
    smlaInfo.oriWinLeft = *opParamInfo_.oriWinLeft;
    smlaInfo.oriWinRight = *opParamInfo_.oriWinRight;

    smlaInfo.topkValueMode = *opParamInfo_.topkValueMode;

    smlaInfo.qLayout = qLayout_;
    smlaInfo.oriSparseIndicesLayout = oriSparseIndicesLayout_;
    smlaInfo.cmpSparseIndicesLayout = cmpSparseIndicesLayout_;
    smlaInfo.kvLayout = kvLayout_;
    smlaInfo.outLayout = outLayout_;
    smlaInfo.returnSoftmaxLse = *opParamInfo_.returnSoftmaxLse;
    smlaInfo.batchConsistency = batchConsistency_;

    smlaInfo.actualLenDimsOriKV = actualLenDimsOriKV_;
    smlaInfo.actualLenDimsCmpKV = actualLenDimsCmpKV_;
    smlaInfo.cmpResidualKVSize = cmpResidualKVSize_;

    if (!IsA5Arch(npuArch_)) {
        smlaInfo.oriKeyStride0 = 0;
        smlaInfo.cmpKeyStride0 = 0;
    } else {
        if (!oriKeyStridesVec_.empty()) {
            smlaInfo.oriKeyStride0 = static_cast<uint32_t>(oriKeyStridesVec_[0]);
        } else {
            smlaInfo.oriKeyStride0 = GetKvstride(oriKvShape_, kvLayout_)[0];
        }

        if (!cmpKeyStridesVec_.empty()) {
            smlaInfo.cmpKeyStride0 = static_cast<uint32_t>(cmpKeyStridesVec_[0]);
        } else {
            smlaInfo.cmpKeyStride0 = GetKvstride(cmpKvShape_, kvLayout_)[0];
        }
    }
}

ge::graphStatus SMLAInfoParser::Parse(SMLATilingInfo &smlaInfo)
{
    if (context_ == nullptr) {
        OP_LOGE("SparseFlashAttention", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo() || ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence() || ge::GRAPH_SUCCESS != CheckUnrequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetInOutDataType() || ge::GRAPH_SUCCESS != GetQueryAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetKvLayout() || ge::GRAPH_SUCCESS != GetSMLATemplateMode()) {
        return ge::GRAPH_FAILED;
    }

    // Match the model-facing template selection on both A2/A3 and A5.
    OP_CHECK_IF(qLayout_ != SMLALayout::TND || kvLayout_ != SMLALayout::PA_BBND,
                OP_LOGE(opName_, "Aurora SparseFlashMla only compiles TND Q with PA_BBND KV."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(perfMode_ != SMLATemplateMode::SWA_TEMPLATE_MODE &&
                    perfMode_ != SMLATemplateMode::CSA_TEMPLATE_MODE,
                OP_LOGE(opName_, "Aurora SparseFlashMla only compiles SWA and CSA templates."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qType_ != ge::DT_BF16,
                OP_LOGE(opName_, "Aurora SparseFlashMla only compiles BF16 Q/KV."), return ge::GRAPH_FAILED);

    SetSMLAShape();
    if (ge::GRAPH_SUCCESS != GetN1Size() || ge::GRAPH_SUCCESS != GetN2Size() || ge::GRAPH_SUCCESS != GetGSize() ||
        ge::GRAPH_SUCCESS != GetBatchSize() || ge::GRAPH_SUCCESS != GetQTSize() || ge::GRAPH_SUCCESS != GetS1Size() ||
        ge::GRAPH_SUCCESS != GetS2Size() || ge::GRAPH_SUCCESS != GetQHeadDim() ||
        ge::GRAPH_SUCCESS != GetValueHeadDim() || ge::GRAPH_SUCCESS != GetSparseBlockCount() ||
        ge::GRAPH_SUCCESS != GetSinks()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != GetActualseqInfo()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckContiguous()) {
        return ge::GRAPH_FAILED;
    }
    GenerateInfo(smlaInfo);
    return ge::GRAPH_SUCCESS;
}

void SMLATilingCheck::Init()
{
    opName_ = smlaInfo_.opName;
    platformInfo_ = smlaInfo_.platformInfo;
    opParamInfo_ = smlaInfo_.opParamInfo;
    npuArch_ = smlaInfo_.npuArch;
    bSize_ = smlaInfo_.bSize;
    n1Size_ = smlaInfo_.n1Size;
    n2Size_ = smlaInfo_.n2Size;
    s1Size_ = smlaInfo_.s1Size;
    s2Size_ = smlaInfo_.s2Size;
    cmpS2Size_ = smlaInfo_.cmpS2Size;
    gSize_ = smlaInfo_.gSize;
    qHeadDim_ = smlaInfo_.qHeadDim;
    oriKvHeadDim_ = smlaInfo_.oriKvHeadDim;
    cmpKvHeadDim_ = smlaInfo_.cmpKvHeadDim;
    oriBlockSize_ = smlaInfo_.oriBlockSize;
    cmpBlockSize_ = smlaInfo_.cmpBlockSize;
    qTSize_ = smlaInfo_.qTSize;
    qType_ = smlaInfo_.qType;
    oriKvType_ = smlaInfo_.oriKvType;
    cmpKvType_ = smlaInfo_.cmpKvType;
    outputType_ = smlaInfo_.outputType;
    cmpRatio_ = smlaInfo_.cmpRatio;
    qLayout_ = smlaInfo_.qLayout;
    oriSparseIndicesLayout_ = smlaInfo_.oriSparseIndicesLayout;
    cmpSparseIndicesLayout_ = smlaInfo_.cmpSparseIndicesLayout;
    oriWinLeft_ = smlaInfo_.oriWinLeft;
    oriWinRight_ = smlaInfo_.oriWinRight;
    hasOriSparseIndices_ = smlaInfo_.hasOriSparseIndices;
    oriSparseIndexWidth_ = smlaInfo_.oriSparseIndexWidth;
    kvLayout_ = smlaInfo_.kvLayout;
    outLayout_ = smlaInfo_.outLayout;
    topkValueMode_ = smlaInfo_.topkValueMode;
}

void SMLATilingCheck::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
                                           const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << SMLADataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, name.c_str(), SMLADataTypeToSerialString(actualDtype).c_str(),
                                          "The dtype of " + name + " only supports " + oss.str());
}

ge::graphStatus SMLATilingCheck::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
                                                   const std::string &name) const
{
    if (desc != nullptr) {
        const auto &it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(it == DTYPE_SUPPORT_MAP.end(),
                    OP_LOGE(opName_, "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
                    return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(
            std::find(expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end(),
            LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void SMLATilingCheck::LogErrorLayoutSupport(const std::vector<SMLALayout> &expectLayoutList,
                                            const SMLALayout &actualLayout, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectLayoutList.size(); ++i) {
        oss << SMLALayoutToSerialString(expectLayoutList[i]);
        if (i < expectLayoutList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, name.c_str(), SMLALayoutToSerialString(actualLayout).c_str(),
                                          "Tensor " + name + " only supports layout " + oss.str());
}

ge::graphStatus SMLATilingCheck::CheckLayoutSupport(const SMLALayout &actualLayout, const std::string &name) const
{
    const auto &it = LAYOUT_SUPPORT_MAP.find(name);
    OP_CHECK_IF(it == LAYOUT_SUPPORT_MAP.end(),
                OP_LOGE(opName_, "%s layout support list should be specify in LAYOUT_SUPPORT_MAP", name.c_str()),
                return ge::GRAPH_FAILED);
    auto &expectLayoutList = it->second;
    OP_CHECK_IF(std::find(expectLayoutList.begin(), expectLayoutList.end(), actualLayout) == expectLayoutList.end(),
                LogErrorLayoutSupport(expectLayoutList, actualLayout, name), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

template <typename T>
void SMLATilingCheck::LogErrorNumberSupport(const std::vector<T> &expectNumberList, const T &actualValue,
                                            const std::string &name, const std::string subName) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << std::to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName_, "%s %s only supports %s, but got %s", name.c_str(), subName.c_str(), oss.str().c_str(),
            std::to_string(actualValue).c_str());
}

template <typename T>
void SMLATilingCheck::LogErrorDimNumSupport(const std::vector<T> &expectNumberList, const T &actualValue,
                                            const std::string &name) const
{
    LogErrorNumberSupport(expectNumberList, actualValue, name, "dimension");
}

ge::graphStatus SMLATilingCheck::CheckDimNumSupport(const gert::StorageShape *shape,
                                                    const std::vector<size_t> &expectDimNumList,
                                                    const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectDimNumList.begin(), expectDimNumList.end(), shape->GetStorageShape().GetDimNum()) ==
        expectDimNumList.end()) {
        LogErrorDimNumSupport(expectDimNumList, shape->GetStorageShape().GetDimNum(), name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckDimNumInLayoutSupport(const SMLALayout &layout, const gert::StorageShape *shape,
                                                            const std::string &name) const
{
    const auto &dimIt = SMLA_LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName_, name.c_str(), std::to_string(shape->GetStorageShape().GetDimNum()).c_str(),
                    "When layout is " + SMLALayoutToSerialString(layout) + ", the shape dim of " + name +
                        " should be " + std::to_string(dimIt->second)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckSingleParaQuery() const
{
    if (opParamInfo_.q.desc == nullptr || opParamInfo_.q.shape == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "q", "Q must be provided");
        return ge::GRAPH_FAILED;
    }
    const std::vector<size_t> queryDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.q.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(qLayout_, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.q.shape, queryDimNumList, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(qLayout_, opParamInfo_.q.shape, QUERY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckSingleParaOriKv() const
{
    const std::vector<size_t> oriKvDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.oriKv.desc, ORI_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, ORI_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.oriKv.tensor->GetShape(), oriKvDimNumList, ORI_KV_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumInLayoutSupport(kvLayout_, &opParamInfo_.oriKv.tensor->GetShape(), ORI_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    if (kvLayout_ == SMLALayout::BSND) {
        OP_CHECK_IF(
            opParamInfo_.oriKv.tensor->GetStorageShape().GetDim(0) != bSize_,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                opName_, "ori_kv", ToStringRaw(opParamInfo_.oriKv.tensor->GetStorageShape()),
                "Ori_kv's batch dimension(" + std::to_string(opParamInfo_.oriKv.tensor->GetStorageShape().GetDim(0)) +
                    ") should be equal to B(" + std::to_string(bSize_) + ")"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckSingleParaCmpKv() const
{
    if (smlaInfo_.perfMode == SMLATemplateMode::CSA_TEMPLATE_MODE ||
        smlaInfo_.perfMode == SMLATemplateMode::HCA_TEMPLATE_MODE ||
        smlaInfo_.perfMode == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        const std::vector<size_t> cmpKvDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpKv.desc, CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS !=
                CheckDimNumSupport(&opParamInfo_.cmpKv.tensor->GetShape(), cmpKvDimNumList, CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS !=
                CheckDimNumInLayoutSupport(kvLayout_, &opParamInfo_.cmpKv.tensor->GetShape(), CMP_KV_NAME)) {
            return ge::GRAPH_FAILED;
        }
        if (kvLayout_ == SMLALayout::BSND) {
            OP_CHECK_IF(opParamInfo_.cmpKv.tensor->GetStorageShape().GetDim(0) != bSize_,
                        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                            opName_, "cmp_kv", ToStringRaw(opParamInfo_.cmpKv.tensor->GetStorageShape()),
                            "Cmp_kv's batch dimension(" +
                                std::to_string(opParamInfo_.cmpKv.tensor->GetStorageShape().GetDim(0)) +
                                ") should be equal to B(" + std::to_string(bSize_) + ")"),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckSingleParaCuSeqLensQ() const
{
    if (opParamInfo_.cuSeqLensQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const std::vector<size_t> dimNumList = {DIM_NUM_ONE};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cuSeqLensQ.desc, CU_SEQLENS_Q_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumSupport(&opParamInfo_.cuSeqLensQ.tensor->GetShape(), dimNumList, CU_SEQLENS_Q_NAME)) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(opParamInfo_.cuSeqLensQ.tensor->GetShapeSize() != bSize_ + 1,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    opName_, "cu_seqlens_q", std::to_string(opParamInfo_.cuSeqLensQ.tensor->GetShapeSize()).c_str(),
                    "The shape size of cu_seqlens_q is not equal to B + 1:" + std::to_string(bSize_ + 1)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckSingleParaCuSeqLensOriKv() const
{
    if (opParamInfo_.cuSeqLensOriKv.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const std::vector<size_t> dimNumList = {DIM_NUM_ONE};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cuSeqLensOriKv.desc, CU_SEQLENS_ORI_KV_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumSupport(&opParamInfo_.cuSeqLensOriKv.tensor->GetShape(), dimNumList, CU_SEQLENS_ORI_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        opParamInfo_.cuSeqLensOriKv.tensor->GetShapeSize() != bSize_ + 1,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            opName_, "cu_seqlens_ori_kv", std::to_string(opParamInfo_.cuSeqLensOriKv.tensor->GetShapeSize()).c_str(),
            "The shape size of cu_seqlens_ori_kv is not equal to B + 1:" + std::to_string(bSize_ + 1)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckSingleParaCuSeqLensCmpKv() const
{
    if (opParamInfo_.cuSeqLensCmpKv.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const std::vector<size_t> dimNumList = {DIM_NUM_ONE};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cuSeqLensCmpKv.desc, CU_SEQLENS_CMP_KV_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumSupport(&opParamInfo_.cuSeqLensCmpKv.tensor->GetShape(), dimNumList, CU_SEQLENS_CMP_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        opParamInfo_.cuSeqLensCmpKv.tensor->GetShapeSize() != bSize_ + 1,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            opName_, "cu_seqlens_cmp_kv", std::to_string(opParamInfo_.cuSeqLensCmpKv.tensor->GetShapeSize()).c_str(),
            "The shape size of cu_seqlens_cmp_kv is not equal to B + 1:" + std::to_string(bSize_ + 1)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckSingleParaNumHeads() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckSingleParaKvHeadNums() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckSingleParaOriSparseIndices() const
{
    if (opParamInfo_.oriSparseIndices.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (!IsA5Arch(npuArch_)) {
        OP_CHECK_IF(smlaInfo_.perfMode != SMLATemplateMode::SWA_TEMPLATE_MODE,
                    OP_LOGE(opName_, "ori_sparse_indices is only supported in SWA mode."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.oriMaskMode == nullptr || *opParamInfo_.oriMaskMode != 0U,
                    OP_LOGE(opName_, "ori_sparse_indices SWA (DSpark) requires oriMaskMode 0."),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(opParamInfo_.oriSparseIndices.tensor->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "ori_sparse_indices",
                                                         "Ori_sparse_indices cannot be empty tensor"),
                return ge::GRAPH_FAILED);
    const std::vector<size_t> oriSparseIndicesDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.oriSparseIndices.desc, ORI_SPARSE_INDICES) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(oriSparseIndicesLayout_, ORI_SPARSE_INDICES) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.oriSparseIndices.tensor->GetShape(),
                                                oriSparseIndicesDimNumList, ORI_SPARSE_INDICES) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(oriSparseIndicesLayout_,
                                                        &opParamInfo_.oriSparseIndices.tensor->GetShape(),
                                                        ORI_SPARSE_INDICES)) {
        return ge::GRAPH_FAILED;
    }
    if (oriSparseIndicesLayout_ == SMLALayout::BSND) {
        OP_CHECK_IF(
            opParamInfo_.oriSparseIndices.tensor->GetStorageShape().GetDim(0) != bSize_,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                opName_, "ori_sparse_indices", ToStringRaw(opParamInfo_.oriSparseIndices.tensor->GetStorageShape()),
                "Ori_sparse_indices's batch dimension(" +
                    std::to_string(opParamInfo_.oriSparseIndices.tensor->GetStorageShape().GetDim(0)) +
                    ") should be equal to B(" + std::to_string(bSize_) + ")"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckSingleParaCmpSparseIndices() const
{
    if (smlaInfo_.perfMode == optiling::SMLATemplateMode::CSA_TEMPLATE_MODE ||
        smlaInfo_.perfMode == optiling::SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        OP_CHECK_IF(opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetShapeSize() == 0,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "cmp_sparse_indices",
                        "When cmp_sparse_indices is not nullptr(CSA), cmp_sparse_indices cannot be empty tensor"),
                    return ge::GRAPH_FAILED);
        const std::vector<size_t> cmpSparseIndicesDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpSparseIndices.desc, CMP_SPARSE_INDICES) ||
            ge::GRAPH_SUCCESS != CheckLayoutSupport(cmpSparseIndicesLayout_, CMP_SPARSE_INDICES) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cmpSparseIndices.tensor->GetShape(),
                                                    cmpSparseIndicesDimNumList, CMP_SPARSE_INDICES) ||
            ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(cmpSparseIndicesLayout_,
                                                            &opParamInfo_.cmpSparseIndices.tensor->GetShape(),
                                                            CMP_SPARSE_INDICES)) {
            return ge::GRAPH_FAILED;
        }
        if (cmpSparseIndicesLayout_ == SMLALayout::BSND) {
            OP_CHECK_IF(
                opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(0) != bSize_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "cmp_sparse_indices", ToStringRaw(opParamInfo_.cmpSparseIndices.tensor->GetStorageShape()),
                    "Cmp_sparse_indices's batch dimension(" +
                        std::to_string(opParamInfo_.cmpSparseIndices.tensor->GetStorageShape().GetDim(0)) +
                        ") should be equal to B(" + std::to_string(bSize_) + ")"),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckSingleParaOriBlockTable() const
{
    if (kvLayout_ != SMLALayout::PA_BBND) {
        return ge::GRAPH_SUCCESS;
    }
    const std::vector<size_t> oriBlockTableDimNumList = {DIM_NUM_TWO};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.oriBlockTable.desc, ORI_BLOCK_TABLE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.oriBlockTable.tensor->GetShape(), oriBlockTableDimNumList,
                                                ORI_BLOCK_TABLE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(0) != bSize_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "ori_block_table", ToStringRaw(opParamInfo_.oriBlockTable.tensor->GetStorageShape()),
                    "Ori_block_table's first dimension(" +
                        std::to_string(opParamInfo_.oriBlockTable.tensor->GetStorageShape().GetDim(0)) +
                        ") should be equal to B(" + std::to_string(bSize_) + ")"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsPaBlockSizeSupport(npuArch_, oriBlockSize_),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "ori_kv", ToStringRaw(opParamInfo_.oriKv.tensor->GetStorageShape()).c_str(),
                    "OriBlockSize_ should be in [1, 1024] on " + A5_PLATFORM_LOG + " or 16-aligned [16, 1024] on " +
                        A2_A3_PLATFORM_LOG + ", but got: " + std::to_string(oriBlockSize_)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckSingleParaCmpBlockTable() const
{
    if (kvLayout_ != SMLALayout::PA_BBND) {
        return ge::GRAPH_SUCCESS;
    }
    if (smlaInfo_.perfMode == optiling::SMLATemplateMode::CSA_TEMPLATE_MODE ||
        smlaInfo_.perfMode == optiling::SMLATemplateMode::HCA_TEMPLATE_MODE ||
        smlaInfo_.perfMode == optiling::SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        OP_CHECK_IF(opParamInfo_.cmpBlockTable.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "cmp_block_table",
                        "Cmp_block_table must be provided when layout_kv is PA_BBND in CSA/HCA/ORI_CMP_SPARSE mode"),
                    return ge::GRAPH_FAILED);
        const std::vector<size_t> cmpBlockTableDimNumList = {DIM_NUM_TWO};
        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpBlockTable.desc, CMP_BLOCK_TABLE_NAME) ||
            ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cmpBlockTable.tensor->GetShape(),
                                                    cmpBlockTableDimNumList, CMP_BLOCK_TABLE_NAME)) {
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(0) != bSize_,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "cmp_block_table", ToStringRaw(opParamInfo_.cmpBlockTable.tensor->GetStorageShape()),
                        "cmp_block_table's first dimension(" +
                            std::to_string(opParamInfo_.cmpBlockTable.tensor->GetStorageShape().GetDim(0)) +
                            ") should be equal to B(" + std::to_string(bSize_) + ")"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(!IsPaBlockSizeSupport(npuArch_, cmpBlockSize_),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "cmp_kv", ToStringRaw(opParamInfo_.oriKv.tensor->GetStorageShape()).c_str(),
                        "CmpBlockSize should be in [1, 1024] on " + A5_PLATFORM_LOG + " or 16-aligned [16, 1024] on " +
                            A2_A3_PLATFORM_LOG + ", but got: " + std::to_string(cmpBlockSize_)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckSingleParaSinks() const
{
    OP_CHECK_IF(opParamInfo_.sinks.tensor->GetStorageShape().GetShapeSize() == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sinks", "Sinks cannot be empty tensor"),
                return ge::GRAPH_FAILED);
    if (opParamInfo_.sinks.tensor->GetStorageShape().GetDimNum() != DIM_NUM_ONE) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, SINKS_NAME.c_str(),
                                     std::to_string(opParamInfo_.sinks.tensor->GetStorageShape().GetDimNum()).c_str(),
                                     std::to_string(DIM_NUM_ONE).c_str());
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.sinks.tensor->GetStorageShape().GetDim(0) != n1Size_) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName_, "sinks", ToStringRaw(opParamInfo_.sinks.tensor->GetStorageShape()),
            "Sinks's dimension(" + std::to_string(opParamInfo_.sinks.tensor->GetStorageShape().GetDim(0)) +
                ") should be equal to the head num of query(" + std::to_string(n1Size_) + ")");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(opParamInfo_.sinks.desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName_, "sinks", SMLADataTypeToSerialString(opParamInfo_.sinks.desc->GetDataType()).c_str(),
                    "The dtype of sinks must be DT_FLOAT"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckSingleParaMetadata() const
{
    if (opParamInfo_.metadata.tensor == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "metadata", "Metadata must be provided");
        return ge::GRAPH_FAILED;
    }
    const std::vector<size_t> dimNumList = {DIM_NUM_ONE};
    if (ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.metadata.tensor->GetShape(), dimNumList, METADATA_NAME)) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF((opParamInfo_.metadata.tensor->GetShapeSize() != METADATA_LIMIT),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "metadata",
                                                      ToStringRaw(opParamInfo_.metadata.tensor->GetStorageShape()),
                                                      "metadata dim 0 must be" + std::to_string(METADATA_LIMIT)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.metadata.desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName_, "metadata", SMLADataTypeToSerialString(opParamInfo_.metadata.desc->GetDataType()).c_str(),
                    "The dtype of metadata must be DT_INT32"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckSingleParaCmpRatio() const
{
    if (IsA5Arch(npuArch_)) {
        if (opParamInfo_.cmpKv.tensor != nullptr) {
            OP_CHECK_IF(
                cmpRatio_ < 1 || cmpRatio_ > 128,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "cmp_ratio", std::to_string(cmpRatio_).c_str(),
                                                      "Cmp_ratio should be in range [1, 128] on " + A5_PLATFORM_LOG),
                return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }

    const auto checkRatio = [this](bool isSupported, const char *expectedRatios, const char *modeName,
                                   const char *modeReason) {
        OP_CHECK_IF(!isSupported,
                    OP_LOGE(opName_, "cmpRatio should be %s in %s on %s %s, but got %ld.", expectedRatios, modeName,
                            A2_A3_PLATFORM_LOG.c_str(), modeReason, cmpRatio_),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    };

    switch (smlaInfo_.perfMode) {
        case SMLATemplateMode::CSA_TEMPLATE_MODE:
            return checkRatio(cmpRatio_ == 1 || cmpRatio_ == 2 || cmpRatio_ == 4, "1, 2 or 4", "CSA",
                              "when cmp_sparse_indices is provided");
        case SMLATemplateMode::HCA_TEMPLATE_MODE:
            return checkRatio(cmpRatio_ == 128, "128", "HCA", "when cmp_sparse_indices is not provided");
        default:
            return checkRatio(cmpRatio_ == 0, "0", "SWA", "when cmp_kv is not provided");
    }
}

ge::graphStatus SMLATilingCheck::CheckSingleParaOriMaskMode() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckSingleParaCmpMaskMode() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckSingleParaOriWinLeft() const
{
    OP_CHECK_IF(oriWinLeft_ < -1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "ori_win_left", std::to_string(oriWinLeft_).c_str(),
                                                      "Ori_win_left should be -1(unlimited) or non-negative"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckSingleParaOriWinRight() const
{
    OP_CHECK_IF(oriWinRight_ < -1,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "ori_win_right", std::to_string(oriWinRight_).c_str(),
                                                      "Ori_win_right should be -1(unlimited) or non-negative"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckSingleParaCmpResidualKv() const
{
    bool isCmpTemplate = smlaInfo_.perfMode == SMLATemplateMode::HCA_TEMPLATE_MODE ||
                         smlaInfo_.perfMode == SMLATemplateMode::CSA_TEMPLATE_MODE;
    if (isCmpTemplate && *opParamInfo_.cmpMaskMode == 3 && cmpRatio_ != 1) { // 3: RightDownCausal模式
        OP_CHECK_IF(
            opParamInfo_.cmpResidualKv.tensor == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                opName_, "cmp_residual_kv", "Cmp_residual_kv is required when cmp_mask_mode=3 and cmp_ratio != 1"),
            return ge::GRAPH_FAILED);
    }
    if (opParamInfo_.cmpResidualKv.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const std::vector<size_t> dimNumList = {DIM_NUM_ONE};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpResidualKv.desc, CMP_RESIDUAL_KV_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumSupport(&opParamInfo_.cmpResidualKv.tensor->GetShape(), dimNumList, CMP_RESIDUAL_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        opParamInfo_.cmpResidualKv.tensor->GetShapeSize() != bSize_,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName_, "cmp_residual_kv", ToStringRaw(opParamInfo_.cmpResidualKv.tensor->GetStorageShape()),
            "Cmp_residual_kv's first dimension(" + std::to_string(opParamInfo_.cmpResidualKv.tensor->GetShapeSize()) +
                ") should be equal to B(" + std::to_string(bSize_) + ")"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckSingleParaTopkLength() const
{
    if (IsA5Arch(npuArch_)) {
        if (opParamInfo_.oriTopkLength.tensor != nullptr) {
            const std::vector<size_t> dimNumList = {DIM_NUM_TWO, DIM_NUM_THREE};
            if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.oriTopkLength.desc, ORI_TOPK_LENGTH_NAME) ||
                ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.oriTopkLength.tensor->GetShape(), dimNumList,
                                                        ORI_TOPK_LENGTH_NAME)) {
                return ge::GRAPH_FAILED;
            }
            OP_CHECK_IF(opParamInfo_.oriTopkLength.desc->GetDataType() != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            opName_, "ori_topk_length",
                            SMLADataTypeToSerialString(opParamInfo_.oriTopkLength.desc->GetDataType()).c_str(),
                            "The dtype of ori_topk_length must be DT_INT32"),
                        return ge::GRAPH_FAILED);
        }
        if (opParamInfo_.cmpTopkLength.tensor != nullptr) {
            const std::vector<size_t> dimNumList = {DIM_NUM_TWO, DIM_NUM_THREE};
            if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.cmpTopkLength.desc, CMP_TOPK_LENGTH_NAME) ||
                ge::GRAPH_SUCCESS != CheckDimNumSupport(&opParamInfo_.cmpTopkLength.tensor->GetShape(), dimNumList,
                                                        CMP_TOPK_LENGTH_NAME)) {
                return ge::GRAPH_FAILED;
            }
            OP_CHECK_IF(opParamInfo_.cmpTopkLength.desc->GetDataType() != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            opName_, "cmp_topk_length",
                            SMLADataTypeToSerialString(opParamInfo_.cmpTopkLength.desc->GetDataType()).c_str(),
                            "The dtype of cmp_topk_length must be DT_INT32"),
                        return ge::GRAPH_FAILED);
        }
        return ge::GRAPH_SUCCESS;
    }
    if (IsNonEmptyOptionalTensor(opParamInfo_.cmpTopkLength.tensor)) {
        OP_CHECK_IF(smlaInfo_.perfMode != SMLATemplateMode::CSA_TEMPLATE_MODE,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "cmp_topk_length",
                                                             "Cmp_topk_length is only supported in CSA mode"),
                    return ge::GRAPH_FAILED);
    }
    if (!IsNonEmptyOptionalTensor(opParamInfo_.oriTopkLength.tensor)) {
        OP_CHECK_IF(hasOriSparseIndices_ && smlaInfo_.perfMode == SMLATemplateMode::SWA_TEMPLATE_MODE,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "ori_topk_length",
                                                             "Ori_topk_length is required for SWA ori sparse"),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(opParamInfo_.oriSparseIndices.tensor == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "ori_sparse_indices",
                                                         "Ori_topk_length requires ori_sparse_indices"),
                return ge::GRAPH_FAILED);
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.oriTopkLength.desc, ORI_TOPK_LENGTH_NAME)) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(opParamInfo_.oriTopkLength.desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName_, "ori_topk_length",
                    SMLADataTypeToSerialString(opParamInfo_.oriTopkLength.desc->GetDataType()).c_str(),
                    "The dtype of ori_topk_length must be DT_INT32"),
                return ge::GRAPH_FAILED);
    const gert::Shape &topkLenShape = opParamInfo_.oriTopkLength.tensor->GetStorageShape();
    const gert::Shape &sparseShape = opParamInfo_.oriSparseIndices.tensor->GetStorageShape();
    if (oriSparseIndicesLayout_ == SMLALayout::TND) {
        OP_CHECK_IF(topkLenShape.GetDimNum() != 2 || sparseShape.GetDimNum() != 3,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "ori_topk_length and ori_sparse_indices",
                        Ops::Base::ToString(topkLenShape) + " and " + Ops::Base::ToString(sparseShape),
                        "TND ori_topk_length shape must be [T, N2], ori_sparse_indices [T, N2, K]"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(topkLenShape.GetDim(0) != sparseShape.GetDim(0) || topkLenShape.GetDim(1) != sparseShape.GetDim(1),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "ori_topk_length and ori_sparse_indices",
                        Ops::Base::ToString(topkLenShape) + " and " + Ops::Base::ToString(sparseShape),
                        "Ori_topk_length shape must match ori_sparse_indices without K dim"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            topkLenShape.GetDimNum() != 3 || sparseShape.GetDimNum() != 4,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "ori_topk_length", Ops::Base::ToString(topkLenShape),
                                                   "BSND ori_topk_length shape must be [B, S1, N2]"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(topkLenShape.GetDim(0) != sparseShape.GetDim(0) ||
                        topkLenShape.GetDim(1) != sparseShape.GetDim(1) ||
                        topkLenShape.GetDim(2) != sparseShape.GetDim(2),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "ori_topk_length and ori_sparse_indices",
                        Ops::Base::ToString(topkLenShape) + " and " + Ops::Base::ToString(sparseShape),
                        "Ori_topk_length shape must match ori_sparse_indices without K dim"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaQuery() || ge::GRAPH_SUCCESS != CheckSingleParaOriKv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpKv() || ge::GRAPH_SUCCESS != CheckSingleParaCuSeqLensQ() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCuSeqLensOriKv() || ge::GRAPH_SUCCESS != CheckSingleParaCuSeqLensCmpKv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpRatio() || ge::GRAPH_SUCCESS != CheckSingleParaCmpResidualKv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaTopkLength() || ge::GRAPH_SUCCESS != CheckSingleParaNumHeads() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKvHeadNums() || ge::GRAPH_SUCCESS != CheckSingleParaOriSparseIndices() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpSparseIndices() || ge::GRAPH_SUCCESS != CheckSingleParaOriBlockTable() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpBlockTable() || ge::GRAPH_SUCCESS != CheckSingleParaSinks() ||
        ge::GRAPH_SUCCESS != CheckSingleParaMetadata() || ge::GRAPH_SUCCESS != CheckSingleParaOriMaskMode() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCmpMaskMode() || ge::GRAPH_SUCCESS != CheckSingleParaOriWinLeft() ||
        ge::GRAPH_SUCCESS != CheckSingleParaOriWinRight()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckExists(const void *pointer, const std::string &name) const
{
    OP_CHECK_IF(pointer == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, name.c_str(), name + " should not be null"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckNotExists(const void *pointer, const std::string &name) const
{
    OP_CHECK_IF(pointer != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, name.c_str(), name + " should not be null"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto &kv : paramMap) {
        if (CheckExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckNotExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto &kv : paramMap) {
        if (CheckNotExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckExistenceByMap(std::map<std::string, const void *> &existMap,
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

ge::graphStatus SMLATilingCheck::CheckParaExistence() const
{
    if (npuArch_ == NpuArch::DAV_3510) {
        OP_CHECK_IF((kvLayout_ == SMLALayout::TND && opParamInfo_.cuSeqLensOriKv.tensor == nullptr),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName_, "cu_seqlens_ori_kv", "cu_seqlens_ori_kv must be provided when kv layout is TND"),
                    return ge::GRAPH_FAILED);
    } else {
        if (kvLayout_ == SMLALayout::PA_BBND) {
            std::map<std::string, const void *> ParamExistMap = {
                {"actualSeqLengths", opParamInfo_.sequsedOriKv.tensor},
                {"oriBlockTable", opParamInfo_.oriBlockTable.tensor},
            };
            std::map<std::string, const void *> ParamNotExistMap = {};
            if (CheckExistenceByMap(ParamExistMap, ParamNotExistMap) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckFeatureShape() const
{
    if (qLayout_ == SMLALayout::TND) {
        OP_CHECK_IF(bSize_ <= 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "cu_seqlens_q", ToStringRaw(opParamInfo_.cuSeqLensQ.tensor->GetStorageShape()).c_str(),
                        "Batch_size should be greater than 0, but got " + std::to_string(bSize_)),
                    return ge::GRAPH_FAILED);
    } else { // BSND
        OP_CHECK_IF(bSize_ <= 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "q", ToStringRaw(opParamInfo_.q.shape->GetStorageShape()).c_str(),
                        "Batch_size should be greater than 0, but got " + std::to_string(bSize_)),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(qTSize_ <= 0 && (qLayout_ == SMLALayout::TND),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "q", ToStringRaw(opParamInfo_.q.shape->GetStorageShape()).c_str(),
                    "T_size of q should be greater than 0, but got " + std::to_string(qTSize_)),
                return ge::GRAPH_FAILED);

    if (IsA5Arch(npuArch_)) {
        OP_CHECK_IF(n1Size_ < 1 || n1Size_ > 128,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "q",
                                                          ToStringRaw(opParamInfo_.q.shape->GetStorageShape()).c_str(),
                                                          "The head num of q should be in [1, 128] on" +
                                                              A5_PLATFORM_LOG + ", but got " + std::to_string(n1Size_)),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(!IsPowerOfTwoInRange(n1Size_, 1, 128),
                    OP_LOGE(opName_, "q_head_num should be power of two in [1, 128] on %s, but got %u",
                            A2_A3_PLATFORM_LOG.c_str(), n1Size_),
                    return ge::GRAPH_FAILED);
    }

    if (opParamInfo_.oriKv.tensor != nullptr) {
        OP_CHECK_IF(n2Size_ != 1,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "ori_kv", ToStringRaw(opParamInfo_.oriKv.tensor->GetStorageShape()).c_str(),
                        "The head num of ori_kv should be 1, but got " + std::to_string(n2Size_)),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(n1Size_ % n2Size_ != 0,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "q and ori_kv",
                        Ops::Base::ToString(opParamInfo_.q.shape->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.oriKv.tensor->GetStorageShape()),
                        "The head num of q(" + std::to_string(n1Size_) +
                            ") must be divisible by the head num of ori_kv(" + std::to_string(n2Size_) + ")"),
                    return ge::GRAPH_FAILED);
    }
    if (opParamInfo_.cmpKv.tensor != nullptr) {
        OP_CHECK_IF(n2Size_ != 1,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "cmp_kv", ToStringRaw(opParamInfo_.cmpKv.tensor->GetStorageShape()).c_str(),
                        "The head num of cmp_kv should be 1, but got " + std::to_string(n2Size_)),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(n1Size_ % n2Size_ != 0,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "q and cmp_kv",
                        Ops::Base::ToString(opParamInfo_.q.shape->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.cmpKv.tensor->GetStorageShape()),
                        "The head num of q(" + std::to_string(n1Size_) +
                            ") must be divisible by the head num of cmp_kv(" + std::to_string(n2Size_) + ")"),
                    return ge::GRAPH_FAILED);
    }

    if (IsA5Arch(npuArch_)) {
        if (opParamInfo_.oriKv.tensor != nullptr) {
            OP_CHECK_IF(gSize_ < 1 || gSize_ > 128,
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            opName_, "q and ori_kv",
                            Ops::Base::ToString(opParamInfo_.q.shape->GetStorageShape()) + " and " +
                                Ops::Base::ToString(opParamInfo_.oriKv.tensor->GetStorageShape()),
                            "The value of (the head num of q ceildivided by the head num of ori_kv) "
                            "should be in [1, 128] on " +
                                A5_PLATFORM_LOG + ", but got " + std::to_string(gSize_)),
                        return ge::GRAPH_FAILED);
        }
        if (opParamInfo_.cmpKv.tensor != nullptr) {
            OP_CHECK_IF(gSize_ < 1 || gSize_ > 128,
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            opName_, "q and cmp_kv",
                            Ops::Base::ToString(opParamInfo_.q.shape->GetStorageShape()) + " and " +
                                Ops::Base::ToString(opParamInfo_.cmpKv.tensor->GetStorageShape()),
                            "The value of (the head num of q ceildivided by the head num of cmp_kv) "
                            "should be in [1, 128] on " +
                                A5_PLATFORM_LOG + ", but got " + std::to_string(gSize_)),
                        return ge::GRAPH_FAILED);
        }
    } else {
        OP_CHECK_IF(!IsPowerOfTwoInRange(gSize_, 1, 128),
                    OP_LOGE(opName_, "group num should be power of two in [1, 128] on %s, but got %u",
                            A2_A3_PLATFORM_LOG.c_str(), gSize_),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(
        qHeadDim_ != DIM_LIMIT,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName_, "q", ToStringRaw(opParamInfo_.q.shape->GetStorageShape()).c_str(),
            "The head num of q only support " + std::to_string(DIM_LIMIT) + ", but got " + std::to_string(qHeadDim_)),
        return ge::GRAPH_FAILED);

    if (opParamInfo_.oriKv.tensor != nullptr) {
        OP_CHECK_IF(oriKvHeadDim_ != DIM_LIMIT,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "ori_kv", ToStringRaw(opParamInfo_.oriKv.tensor->GetStorageShape()).c_str(),
                        "The head num of ori_kv only support " + std::to_string(DIM_LIMIT) + ", but got " +
                            std::to_string(oriKvHeadDim_)),
                    return ge::GRAPH_FAILED);
    }
    if (!(smlaInfo_.perfMode == SMLATemplateMode::SWA_TEMPLATE_MODE ||
          smlaInfo_.perfMode == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) &&
        (opParamInfo_.cmpKv.tensor != nullptr)) {
        OP_CHECK_IF(cmpKvHeadDim_ != DIM_LIMIT,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        opName_, "cmp_kv", ToStringRaw(opParamInfo_.cmpKv.tensor->GetStorageShape()).c_str(),
                        "The head num of cmp_kv only support " + std::to_string(DIM_LIMIT) + ", but got " +
                            std::to_string(cmpKvHeadDim_)),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(!(qType_ == oriKvType_),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    opName_, "q and ori_kv",
                    SMLADataTypeToSerialString(qType_) + " and " + SMLADataTypeToSerialString(oriKvType_),
                    "The dtype of q and ori_kv must be same"),
                return ge::GRAPH_FAILED);

    if (IsA5Arch(npuArch_)) {
        OP_CHECK_IF(*opParamInfo_.oriMaskMode != 0 && *opParamInfo_.oriMaskMode != 3 && *opParamInfo_.oriMaskMode != 4,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "ori_mask_mode",
                                                          std::to_string(*opParamInfo_.oriMaskMode).c_str(),
                                                          "Ori_mask_mode should be {0, 3, 4} on " + A5_PLATFORM_LOG),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(*opParamInfo_.cmpMaskMode != 0 && *opParamInfo_.cmpMaskMode != 3,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "cmp_mask_mode",
                                                          std::to_string(*opParamInfo_.cmpMaskMode).c_str(),
                                                          "Cmp_mask_mode should be {0, 3} on " + A5_PLATFORM_LOG),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            topkValueMode_ != 1,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "topk_value_mode", std::to_string(topkValueMode_).c_str(),
                                                  "Topk_value_mode should be 1"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(oriWinLeft_ < -1,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        opName_, "ori_win_left", std::to_string(oriWinLeft_).c_str(),
                        "Ori_win_left should be -1(unlimited) or non-negative on " + A5_PLATFORM_LOG),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(oriWinRight_ < -1,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        opName_, "ori_win_right", std::to_string(oriWinRight_).c_str(),
                        "Ori_win_right should be -1(unlimited) or non-negative on " + A5_PLATFORM_LOG),
                    return ge::GRAPH_FAILED);
    } else {
        if (smlaInfo_.perfMode == SMLATemplateMode::CSA_TEMPLATE_MODE ||
            smlaInfo_.perfMode == SMLATemplateMode::HCA_TEMPLATE_MODE) {
            OP_CHECK_IF(*opParamInfo_.cmpMaskMode != 3,
                        OP_LOGE(opName_, "cmpMaskMode should be 3 on %s, but got %u", A2_A3_PLATFORM_LOG.c_str(),
                                *opParamInfo_.cmpMaskMode),
                        return ge::GRAPH_FAILED);
        }
        if (hasOriSparseIndices_) {
            OP_CHECK_IF(*opParamInfo_.oriMaskMode != 0U,
                        OP_LOGE(opName_, "oriMaskMode must be 0 for SWA ori sparse (DSpark) on %s, but got %u.",
                                A2_A3_PLATFORM_LOG.c_str(), *opParamInfo_.oriMaskMode),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(oriWinLeft_ < 0 || oriWinRight_ < 0,
                        OP_LOGE(opName_, "ori_win_left/right should be non-negative for ori sparse SWA on %s.",
                                A2_A3_PLATFORM_LOG.c_str()),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(*opParamInfo_.oriMaskMode != 4U,
                        OP_LOGE(opName_, "oriMaskMode should be 4 on %s when ori_sparse_indices is empty, but got %u",
                                A2_A3_PLATFORM_LOG.c_str(), *opParamInfo_.oriMaskMode),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                oriWinLeft_ != 127,
                OP_LOGE(opName_, "oriWinLeft_ should be 127 on %s when ori_sparse_indices is empty, but got %ld",
                        A2_A3_PLATFORM_LOG.c_str(), oriWinLeft_),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(oriWinRight_ != 0,
                        OP_LOGE(opName_, "oriWinRight_ should be 0 on %s when ori_sparse_indices is empty, but got %ld",
                                A2_A3_PLATFORM_LOG.c_str(), oriWinRight_),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckFeatureLayout() const
{
    const std::vector<std::string> layoutQuerySupportList = {"BSND", "TND"};
    std::string layoutQuery = opParamInfo_.layoutQ;
    OP_CHECK_IF(std::find(layoutQuerySupportList.begin(), layoutQuerySupportList.end(), layoutQuery) ==
                    layoutQuerySupportList.end(),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_q", layoutQuery.c_str(),
                                                      "Layout_q only supports BSND or TND"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckFeatureDtype() const
{
    OP_CHECK_IF(qType_ != ge::DT_BF16 && qType_ != ge::DT_FLOAT16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "q", SMLADataTypeToSerialString(qType_).c_str(),
                                                      "The dtype of q only supports " +
                                                          SMLADataTypeToSerialString(ge::DT_BF16) + " and " +
                                                          SMLADataTypeToSerialString(ge::DT_FLOAT16)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckFeaturePa() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckFeature() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureShape() || ge::GRAPH_SUCCESS != CheckFeatureLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureDtype() || ge::GRAPH_SUCCESS != CheckFeaturePa()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckDTypeConsistency(const ge::DataType &actualDtype, const ge::DataType &expectDtype,
                                                       const std::string &name) const
{
    if (actualDtype != expectDtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            opName_, name.c_str(), SMLADataTypeToSerialString(actualDtype).c_str(),
            "The dtype of " + name + " should be " + SMLADataTypeToSerialString(expectDtype));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckOriAndCmpKv() const
{
    if (smlaInfo_.perfMode == SMLATemplateMode::HCA_TEMPLATE_MODE ||
        smlaInfo_.perfMode == SMLATemplateMode::CSA_TEMPLATE_MODE ||
        smlaInfo_.perfMode == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(cmpKvType_, oriKvType_, CMP_KV_NAME)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckAttenOut() const
{
    if (opParamInfo_.attnOut.desc == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "attn_out", "Attn_out must be provided");
        return ge::GRAPH_FAILED;
    }
    const std::vector<size_t> dimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.attnOut.desc, ATTEN_OUT_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(outLayout_, ATTEN_OUT_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.attnOut.shape, dimNumList, ATTEN_OUT_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(outLayout_, opParamInfo_.attnOut.shape, ATTEN_OUT_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckActualSeqLensQ() const
{
    if (opParamInfo_.seqUsedQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const std::vector<size_t> dimNumList = {DIM_NUM_ONE};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.seqUsedQ.desc, SEQUSED_Q_NAME) ||
        ge::GRAPH_SUCCESS !=
            CheckDimNumSupport(&opParamInfo_.seqUsedQ.tensor->GetShape(), dimNumList, SEQUSED_Q_NAME)) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(opParamInfo_.seqUsedQ.tensor->GetShapeSize() != bSize_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "seqused_q", ToStringRaw(opParamInfo_.seqUsedQ.tensor->GetStorageShape()),
                    "Seqused_q's first dimension(" + std::to_string(opParamInfo_.seqUsedQ.tensor->GetShapeSize()) +
                        ") should be equal to B(" + std::to_string(bSize_) + ")"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::CheckActualSeqLens() const
{
    if (opParamInfo_.sequsedOriKv.tensor != nullptr) {
        const std::vector<size_t> dimNumList = {DIM_NUM_ONE};
        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.sequsedOriKv.desc, SEQUSED_ORI_KV_NAME) ||
            ge::GRAPH_SUCCESS !=
                CheckDimNumSupport(&opParamInfo_.sequsedOriKv.tensor->GetShape(), dimNumList, SEQUSED_ORI_KV_NAME)) {
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(
            opParamInfo_.sequsedOriKv.tensor->GetShapeSize() != bSize_,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                opName_, "seqused_ori_kv", ToStringRaw(opParamInfo_.sequsedOriKv.tensor->GetStorageShape()),
                "Seqused_ori_kv's first dimension(" + std::to_string(opParamInfo_.sequsedOriKv.tensor->GetShapeSize()) +
                    ") should be equal to B(" + std::to_string(bSize_) + ")"),
            return ge::GRAPH_FAILED);
    }
    if (opParamInfo_.sequsedCmpKv.tensor != nullptr) {
        const std::vector<size_t> dimNumList = {DIM_NUM_ONE};
        if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.sequsedCmpKv.desc, SEQUSED_CMP_KV_NAME) ||
            ge::GRAPH_SUCCESS !=
                CheckDimNumSupport(&opParamInfo_.sequsedCmpKv.tensor->GetShape(), dimNumList, SEQUSED_CMP_KV_NAME)) {
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(
            opParamInfo_.sequsedCmpKv.tensor->GetShapeSize() != bSize_,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                opName_, "seqused_cmp_kv", ToStringRaw(opParamInfo_.sequsedCmpKv.tensor->GetStorageShape()),
                "Seqused_cmp_kv's first dimension(" + std::to_string(opParamInfo_.sequsedCmpKv.tensor->GetShapeSize()) +
                    ") should be equal to B(" + std::to_string(bSize_) + ")"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus SMLATilingCheck::CheckMultiParaConsistency()
{
    if (ge::GRAPH_SUCCESS != CheckOriAndCmpKv() || ge::GRAPH_SUCCESS != CheckAttenOut() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQ() || ge::GRAPH_SUCCESS != CheckActualSeqLens()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SMLATilingCheck::Process()
{
    Init();
    if (CheckSinglePara() != ge::GRAPH_SUCCESS || CheckParaExistence() != ge::GRAPH_SUCCESS ||
        CheckFeature() != ge::GRAPH_SUCCESS || CheckMultiParaConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void SparseFlashMlaTiling::CalcUbBmm(const SMLATilingInfo *tilingInfo)
{
    uint32_t cubeMSize = tilingInfo->gSize * tilingInfo->s1Size;
    uint32_t maxMSize = mBaseSize_;
    if (cubeMSize > maxMSize) {
        cubeMSize = maxMSize;
    }
    mmResUbSize_ = sInnerSizeAlign_ * Align(cubeMSize, 16U); // kernel按照16对齐写出，tiling按照这个原则分配内存
    bmm2ResUbSize_ = headDimAlign_ * Align(cubeMSize, 16U); // kernel按照16对齐写出，tiling按照这个原则分配内存
}

void SparseFlashMlaTiling::SplitBalanced(SMLATilingInfo *tilingInfo)
{
    sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK);
    if (tilingInfo->npuArch == NpuArch::DAV_2201) {
        mBaseSize_ = tilingInfo->perfMode == SMLATemplateMode::CSA_TEMPLATE_MODE ?
                         tilingInfo->gSize :
                         (256 / tilingInfo->gSize) * tilingInfo->gSize; // 256：DAV_2201架构下的基础计算块大小
        // DSpark ori_sparse_indices are per query token; keep one S1 row per M block.
        if (tilingInfo->hasOriSparseIndices && tilingInfo->perfMode == SMLATemplateMode::SWA_TEMPLATE_MODE) {
            mBaseSize_ = tilingInfo->gSize;
        }
    }
    headDimAlign_ = Align(tilingInfo->qHeadDim, BYTE_BLOCK);
    CalcUbBmm(tilingInfo);

    tilingData_.baseParams.set_mBaseSize(mBaseSize_);
    tilingData_.baseParams.set_s2BaseSize(sInnerSize_);
    tilingData_.baseParams.set_mmResUbSize(mmResUbSize_);
    tilingData_.baseParams.set_bmm2ResUbSize(bmm2ResUbSize_);
}

uint32_t SparseFlashMlaTiling::CalcFdLogicalSlotCount(const SMLATilingInfo *tilingInfo, uint32_t aicNum) const
{
    bool isSplitG = IsA5Arch(tilingInfo->npuArch) && tilingInfo->gSize > 64;
    if (isSplitG) {
        return aicNum >> 1U;
    }
    return aicNum;
}

uint64_t SparseFlashMlaTiling::CalcFdStagingWorkspaceSize(const SMLATilingInfo *tilingInfo, uint32_t aicNum) const
{
    if (!IsA5Arch(tilingInfo->npuArch) || aicNum == 0U) {
        return 0ULL;
    }

    const uint64_t logicalCoreSlots = CalcFdLogicalSlotCount(tilingInfo, aicNum);
    const uint64_t bytesPerSlot = tilingInfo->gSize * sizeof(float) * (headDimAlign_ + 2ULL * FD_BROADCAST_ELEMS);
    if (tilingInfo->batchConsistency) {
        // Intra-core reduction ping-pongs by multiCoreIdxMod2. Cross-core staging
        // keeps one reduce-block range for every physical AIC.
        const uint64_t intraCoreSlots = 2ULL * logicalCoreSlots;
        const uint64_t crossCoreSlots = static_cast<uint64_t>(aicNum) * BATCH_CONSISTENCY_MAX_REDUCE_BLOCK_NUM;
        return (intraCoreSlots + crossCoreSlots) * bytesPerSlot;
    }
    return logicalCoreSlots * FD_MAX_S2_SPLIT_NUM * bytesPerSlot;
}

uint64_t SparseFlashMlaTiling::CalcVectorizeKvPhyAddrWorkspaceSize(const SMLATilingInfo *tilingInfo,
                                                                   uint32_t &vectorizeFlag) const
{
    vectorizeFlag = 0U;
    if (tilingInfo->npuArch != NpuArch::DAV_3510) {
        return 0ULL;
    }
    constexpr uint32_t SPARSE_BLOCK_ALIGN_NUM = 128;
    constexpr uint32_t UB_SIZE = 184 * 1024;
    uint32_t alignedOriSparseBlockCount = (tilingInfo->oriSparseBlockCount + SPARSE_BLOCK_ALIGN_NUM - 1) /
                                          SPARSE_BLOCK_ALIGN_NUM * SPARSE_BLOCK_ALIGN_NUM;
    uint32_t alignedCmpSparseBlockCount = (tilingInfo->cmpSparseBlockCount + SPARSE_BLOCK_ALIGN_NUM - 1) /
                                          SPARSE_BLOCK_ALIGN_NUM * SPARSE_BLOCK_ALIGN_NUM;
    bool isPa = (tilingInfo->kvLayout == SMLALayout::PA_BBND);
    uint32_t oriBlocksizeFlag =
        static_cast<uint32_t>(tilingInfo->oriBlockSize & static_cast<uint32_t>(tilingInfo->oriBlockSize - 1)) == 0;
    uint32_t cmpBlocksizeFlag =
        static_cast<uint32_t>(tilingInfo->cmpBlockSize & static_cast<uint32_t>(tilingInfo->cmpBlockSize - 1)) == 0;
    uint32_t blocksizeFlag = isPa ? ((tilingInfo->perfMode == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE) ?
                                         oriBlocksizeFlag :
                                         (oriBlocksizeFlag != 0 && cmpBlocksizeFlag != 0)) :
                                    1U;
    uint64_t paExtraUb = std::max(static_cast<uint64_t>(tilingInfo->oriMaxBlockNumPerBatch) * sizeof(int32_t),
                                  static_cast<uint64_t>(tilingInfo->cmpMaxBlockNumPerBatch) * sizeof(int32_t));
    uint64_t cmpUbSize = (isPa ? paExtraUb : 0U) + static_cast<uint64_t>(alignedCmpSparseBlockCount) * sizeof(int32_t) +
                         static_cast<uint64_t>(alignedCmpSparseBlockCount) * sizeof(int64_t);
    uint64_t oriUbSize = (isPa ? paExtraUb : 0U) + static_cast<uint64_t>(alignedOriSparseBlockCount) * sizeof(int32_t) +
                         static_cast<uint64_t>(alignedOriSparseBlockCount) * sizeof(int64_t);
    uint64_t vectorizeUbSize = std::max(cmpUbSize, oriUbSize);
    vectorizeFlag = static_cast<uint32_t>((tilingInfo->perfMode == SMLATemplateMode::CSA_TEMPLATE_MODE ||
                                           tilingInfo->perfMode == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
                                           tilingInfo->perfMode == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) &&
                                          (vectorizeUbSize <= UB_SIZE) && (blocksizeFlag != 0));

    if (vectorizeFlag == 0U) {
        return 0ULL;
    }
    uint32_t totalBS1 =
        (tilingInfo->qLayout == SMLALayout::TND) ? tilingInfo->s1Size : (tilingInfo->bSize * tilingInfo->s1Size);
    uint64_t oriPhyAddrSize = 0;
    if (tilingInfo->perfMode == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
        tilingInfo->perfMode == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        oriPhyAddrSize = static_cast<uint64_t>(totalBS1) * alignedOriSparseBlockCount * sizeof(int64_t);
    }
    uint64_t cmpPhyAddrSize = 0;
    if (tilingInfo->perfMode == SMLATemplateMode::CSA_TEMPLATE_MODE ||
        tilingInfo->perfMode == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
        cmpPhyAddrSize = static_cast<uint64_t>(totalBS1) * alignedCmpSparseBlockCount * sizeof(int64_t);
    }
    return oriPhyAddrSize + cmpPhyAddrSize;
}

// --------------------------SparseFlashMlaTiling类成员函数定义----------------------
ge::graphStatus SparseFlashMlaTiling::DoOpTiling(SMLATilingInfo *tilingInfo)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);
    OP_LOGI(tilingInfo->opName, "SMLA block dim: %u aiv Num: %u aic Num: %u.", blockDim, aivNum, aicNum);

    SplitBalanced(tilingInfo);

    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;
    constexpr uint32_t VEC1_RES_ELEM_SIZE = 2;
    constexpr uint32_t MM2_RES_ELEM_SIZE = 4;
    constexpr uint32_t VEC2_RES_ELEM_SIZE = 4;
    constexpr uint32_t PRELOAD_NUM = 2;

    size_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (tilingInfo->npuArch == NpuArch::DAV_3510) {
        bool isSplitG = tilingInfo->gSize > 64;
        if (isSplitG || tilingInfo->perfMode == SMLATemplateMode::CSA_TEMPLATE_MODE ||
            tilingInfo->perfMode == SMLATemplateMode::ORI_SPARSE_TEMPLATE_MODE ||
            tilingInfo->perfMode == SMLATemplateMode::ORI_CMP_SPARSE_TEMPLATE_MODE) {
            constexpr uint32_t TRIPLE_BUFFER_NUM = 3;
            constexpr uint32_t S2_BASE_SIZE = 128;
            constexpr uint32_t D_SIZE = 512;
            constexpr uint32_t VEC_RES_ELEM_SIZE = 2;
            uint32_t gmSize = S2_BASE_SIZE * D_SIZE * VEC_RES_ELEM_SIZE * TRIPLE_BUFFER_NUM;
            if (isSplitG) {
                gmSize *= (aicNum >> 1U);
            } else {
                gmSize *= aicNum;
            }
            workspaceSize += gmSize;
            constexpr uint32_t S2_REAL_BUF_LEN = 128;
            workspaceSize += TRIPLE_BUFFER_NUM * S2_REAL_BUF_LEN * sizeof(int32_t) * aivNum;
        }
    } else {
        workspaceSize += PRELOAD_NUM * mmResUbSize_ * MM1_RES_ELEM_SIZE * aicNum;
        workspaceSize += PRELOAD_NUM * mmResUbSize_ * VEC1_RES_ELEM_SIZE * aicNum;
        workspaceSize += PRELOAD_NUM * bmm2ResUbSize_ * MM2_RES_ELEM_SIZE * aicNum;
        workspaceSize += PRELOAD_NUM * bmm2ResUbSize_ * VEC2_RES_ELEM_SIZE * aicNum;
        if (tilingInfo->perfMode == SMLATemplateMode::CSA_TEMPLATE_MODE ||
            (tilingInfo->perfMode == SMLATemplateMode::SWA_TEMPLATE_MODE && tilingInfo->hasOriSparseIndices)) {
            constexpr uint32_t MERGE_CACHE_GM_BUF_NUM = 3;                    // 3：缓存缓冲区数量
            workspaceSize += MERGE_CACHE_GM_BUF_NUM * 512 * 512 * 2 * aicNum; // 缓冲区的尺寸为512x512字节，2表示双缓冲
        }
    }

    // 计算vectorizeFlag (稀疏KV物理地址向量化)
    uint32_t vectorizeFlag = 0U;
    workspaceSize += CalcVectorizeKvPhyAddrWorkspaceSize(tilingInfo, vectorizeFlag);

    workspaceSize += CalcFdStagingWorkspaceSize(tilingInfo, aicNum);
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

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
    tilingData_.baseParams.set_oriKvStride0(tilingInfo->oriKvStride0);
    tilingData_.baseParams.set_oriWinLeft(tilingInfo->oriWinLeft);
    tilingData_.baseParams.set_oriWinRight(tilingInfo->oriWinRight);
    tilingData_.baseParams.set_sparseBlockSize(tilingInfo->sparseBlockSize);
    tilingData_.baseParams.set_returnSoftmaxLse(tilingInfo->returnSoftmaxLse);

    tilingData_.cmpParams.set_cmpMaxBlockNumPerBatch(tilingInfo->cmpMaxBlockNumPerBatch);
    tilingData_.cmpParams.set_cmpRatio(tilingInfo->cmpRatio);
    tilingData_.cmpParams.set_cmpMaskMode(tilingInfo->cmpMaskMode);
    tilingData_.cmpParams.set_cmpKvStride0(tilingInfo->cmpKvStride0);
    tilingData_.cmpParams.set_cmpKvSeqSize(tilingInfo->cmpS2Size);
    tilingData_.baseParams.set_actualLenDimsOriKV(tilingInfo->actualLenDimsOriKV);
    tilingData_.baseParams.set_actualLenDimsCmpKV(tilingInfo->actualLenDimsCmpKV);
    tilingData_.baseParams.set_cmpResidualKVSize(tilingInfo->cmpResidualKVSize);
    tilingData_.baseParams.set_oriKeyStride0(tilingInfo->oriKeyStride0);
    tilingData_.baseParams.set_hasOriSparseIndices(tilingInfo->hasOriSparseIndices ? 1U : 0U);
    tilingData_.baseParams.set_oriSparseIndexWidth(tilingInfo->oriSparseIndexWidth);
    tilingData_.cmpParams.set_cmpKeyStride0(tilingInfo->cmpKeyStride0);

    if (tilingInfo->npuArch == NpuArch::DAV_3510) {
        tilingData_.baseParams.set_oriSparseBlockCount(tilingInfo->oriSparseBlockCount);
        tilingData_.baseParams.set_topkValueMode(tilingInfo->topkValueMode);
        tilingData_.cmpParams.set_cmpSparseBlockCount(tilingInfo->cmpSparseBlockCount);
    } else {
        tilingData_.cmpParams.set_sparseBlockCount(tilingInfo->sparseBlockCount);
    }

    usedCoreNum_ = aicNum;
    tilingData_.baseParams.set_usedCoreNum(usedCoreNum_);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    uint32_t qLayout = static_cast<uint32_t>(tilingInfo->qLayout);
    uint32_t inputKvLayout = static_cast<uint32_t>(tilingInfo->kvLayout);

    uint64_t tilingKey;
    uint32_t splitG = 0U;
    uint32_t headRatioOne =
        static_cast<uint32_t>(tilingInfo->npuArch == NpuArch::DAV_2201 &&
                              tilingInfo->perfMode == SMLATemplateMode::CSA_TEMPLATE_MODE && tilingInfo->gSize == 1U);
    if (tilingInfo->npuArch == NpuArch::DAV_3510) {
        splitG = static_cast<uint32_t>(tilingInfo->gSize > 64); // 64：分组拆分阈值
    }
    tilingKey = GET_TPL_TILING_KEY(0U, qLayout, inputKvLayout, static_cast<uint32_t>(tilingInfo->perfMode), splitG,
                                   headRatioOne, static_cast<uint32_t>(tilingInfo->batchConsistency), vectorizeFlag);
    context_->SetScheduleMode(1);
    context_->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace optiling {
// --------------------------TilingPrepare函数定义-------------------------------------
static ge::graphStatus TilingPrepareForSparseFlashMla(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// --------------------------Tiling函数定义---------------------------
ge::graphStatus TilingForSparseFlashMla(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SparseFlashMla", "Tiling context is null."),
                return ge::GRAPH_FAILED);

    SMLATilingInfo smlaInfo;
    SMLAInfoParser smlaInfoParser(context);
    if (smlaInfoParser.Parse(smlaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (smlaInfo.npuArch == NpuArch::DAV_2201) {
        SMLATilingCheck smlaTilingChecker(smlaInfo);
        if (smlaTilingChecker.Process() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        SparseFlashMlaChecker smlaTilingChecker(smlaInfo);
        if (smlaTilingChecker.Process() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    SparseFlashMlaTiling tiling(context);
    return tiling.DoOpTiling(&smlaInfo);
}
// --------------------------Tiling函数及TilingPrepare函数注册--------
IMPL_OP_OPTILING(SparseFlashMla)
    .Tiling(TilingForSparseFlashMla)
    .TilingParse<SMLACompileInfo>(TilingPrepareForSparseFlashMla);
} // namespace optiling
