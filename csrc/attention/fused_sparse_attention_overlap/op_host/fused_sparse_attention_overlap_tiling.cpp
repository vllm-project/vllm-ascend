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
 * \file fused_sparse_attention_overlap_tiling.cpp
 * \brief
 */

#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "err/ops_err.h"
#include "register/op_def_registry.h"
#include "../op_kernel/fused_sparse_attention_overlap_template_tiling_key.h"
#include "fused_sparse_attention_overlap_tiling.h"

using std::map;
using std::string;
using std::pair;

using namespace ge;
using namespace AscendC;
namespace optiling {

constexpr uint32_t PRE_LOAD_NUM = 2;
constexpr uint32_t BLOCK_TABLE_ELEM_BYTE = 4;
constexpr int32_t SPARSE_MODE_BAND = 4;
constexpr uint32_t SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT = 16400;

static const std::string QUERY_NAME = "query";
static const std::string KEY_NAME = "key";
static const std::string VALUE_NAME = "value";
static const std::string BLOCK_TABLE_NAME = "block_table";
static const std::string SPARSE_INDICES_NAME = "sparse_indices";
static const std::string QUERY_ROPE_NAME = "query_rope";
static const std::string KEY_ROPE_NAME = "key_rope";
static const std::string ATTEN_OUT_NAME = "attention_out";
static const std::string SOFTMAX_MAX_NAME = "softmax_max";
static const std::string SOFTMAX_SUM_NAME = "softmax_sum";

static bool IsFusedSparseAttentionOverlapStandaloneOp(const char *opName)
{
    return opName != nullptr && std::string(opName) == "FusedSparseAttentionOverlap";
}

const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME,                  {ge::DT_FLOAT16, ge::DT_BF16}},
    {KEY_NAME,                    {ge::DT_FLOAT16, ge::DT_BF16}},
    {VALUE_NAME,                  {ge::DT_FLOAT16, ge::DT_BF16}},
    {QUERY_ROPE_NAME,             {ge::DT_FLOAT16, ge::DT_BF16}},
    {KEY_ROPE_NAME,               {ge::DT_FLOAT16, ge::DT_BF16}},
    {ATTEN_OUT_NAME,              {ge::DT_FLOAT16, ge::DT_BF16}},
    {SOFTMAX_MAX_NAME,            {ge::DT_FLOAT}},
    {SOFTMAX_SUM_NAME,            {ge::DT_FLOAT}},
    {SPARSE_INDICES_NAME,         {ge::DT_INT32}},
    {BLOCK_TABLE_NAME,            {ge::DT_INT32}},
};

const std::map<std::string, std::vector<FusedSparseAttentionOverlapLayout>> LAYOUT_SUPPORT_MAP = {
    {QUERY_NAME,             {FusedSparseAttentionOverlapLayout::BSND, FusedSparseAttentionOverlapLayout::TND}},
    {KEY_NAME,               {FusedSparseAttentionOverlapLayout::BSND, FusedSparseAttentionOverlapLayout::TND, FusedSparseAttentionOverlapLayout::PA_BSND}},
    {VALUE_NAME,             {FusedSparseAttentionOverlapLayout::BSND, FusedSparseAttentionOverlapLayout::TND, FusedSparseAttentionOverlapLayout::PA_BSND}},
    {ATTEN_OUT_NAME,         {FusedSparseAttentionOverlapLayout::BSND, FusedSparseAttentionOverlapLayout::TND}},
    {SOFTMAX_MAX_NAME,       {FusedSparseAttentionOverlapLayout::BNSG, FusedSparseAttentionOverlapLayout::NTG}},
    {SOFTMAX_SUM_NAME,       {FusedSparseAttentionOverlapLayout::BNSG, FusedSparseAttentionOverlapLayout::NTG}},
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

struct FusedSparseAttentionOverlapCompileInfo {
    int64_t core_num;
};

static const std::map<FusedSparseAttentionOverlapLayout, std::vector<FusedSparseAttentionOverlapAxis>> FUSED_SPARSE_ATTENTION_OVERLAP_LAYOUT_AXIS_MAP = {
    {FusedSparseAttentionOverlapLayout::BSND, {FusedSparseAttentionOverlapAxis::B, FusedSparseAttentionOverlapAxis::S, FusedSparseAttentionOverlapAxis::N, FusedSparseAttentionOverlapAxis::D}},
    {FusedSparseAttentionOverlapLayout::TND, {FusedSparseAttentionOverlapAxis::T, FusedSparseAttentionOverlapAxis::N, FusedSparseAttentionOverlapAxis::D}},
    {FusedSparseAttentionOverlapLayout::PA_BSND, {FusedSparseAttentionOverlapAxis::Bn, FusedSparseAttentionOverlapAxis::Bs, FusedSparseAttentionOverlapAxis::N, FusedSparseAttentionOverlapAxis::D}},
    {FusedSparseAttentionOverlapLayout::BNSG, {FusedSparseAttentionOverlapAxis::B, FusedSparseAttentionOverlapAxis::N, FusedSparseAttentionOverlapAxis::S, FusedSparseAttentionOverlapAxis::G}},
    {FusedSparseAttentionOverlapLayout::NTG, {FusedSparseAttentionOverlapAxis::N, FusedSparseAttentionOverlapAxis::T, FusedSparseAttentionOverlapAxis::G}},
};

static const std::map<FusedSparseAttentionOverlapLayout, size_t> FUSED_SPARSE_ATTENTION_OVERLAP_LAYOUT_DIM_MAP = {
    {FusedSparseAttentionOverlapLayout::BSND, DIM_NUM_FOUR},
    {FusedSparseAttentionOverlapLayout::TND, DIM_NUM_THREE},
    {FusedSparseAttentionOverlapLayout::PA_BSND, DIM_NUM_FOUR},
    {FusedSparseAttentionOverlapLayout::BNSG, DIM_NUM_FOUR},
    {FusedSparseAttentionOverlapLayout::NTG, DIM_NUM_THREE},
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

static std::string FusedSparseAttentionOverlapDataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("FusedSparseAttentionOverlap", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

string FusedSparseAttentionOverlapTensorDesc2String(const gert::StorageShape *shape, const gert::CompileTimeTensorDesc *tensor)
{
    if (shape == nullptr || tensor == nullptr) {
        return "nil ";
    }

    std::ostringstream oss;
    oss << "(dtype: " << ge::TypeUtils::DataTypeToAscendString(tensor->GetDataType()).GetString() << "),";
    oss << "(shape:" << FusedSparseAttentionOverlapShape2String(shape->GetStorageShape()) << "),";
    oss << "(ori_shape:" << FusedSparseAttentionOverlapShape2String(shape->GetOriginShape()) << "),";
    oss << "(format: "
        << ge::TypeUtils::FormatToAscendString(
               static_cast<ge::Format>(ge::GetPrimaryFormat(tensor->GetStorageFormat())))
               .GetString()
        << "),";
    oss << "(ori_format: " << ge::TypeUtils::FormatToAscendString(tensor->GetOriginFormat()).GetString() << ") ";

    return oss.str();
}

string FusedSparseAttentionOverlapDebugTilingContext(const gert::TilingContext *context)
{
    std::ostringstream oss;
    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetInputsNum(); ++i) {
        oss << "input" << i << ": ";
        oss << FusedSparseAttentionOverlapTensorDesc2String(context->GetInputShape(i), context->GetInputDesc(i));
    }

    for (size_t i = 0; i < context->GetComputeNodeInfo()->GetOutputsNum(); ++i) {
        oss << "output" << i << ": ";
        oss << FusedSparseAttentionOverlapTensorDesc2String(context->GetOutputShape(i), context->GetOutputDesc(i));
    }
    return oss.str();
}

std::string FusedSparseAttentionOverlapLayoutToSerialString(FusedSparseAttentionOverlapLayout layout)
{
    switch (layout) {
        case FusedSparseAttentionOverlapLayout::BSND: return "BSND";
        case FusedSparseAttentionOverlapLayout::TND: return "TND";
        case FusedSparseAttentionOverlapLayout::PA_BSND: return "PA_BSND";
        case FusedSparseAttentionOverlapLayout::BNSG: return "BNSG";
        case FusedSparseAttentionOverlapLayout::NTG: return "NTG";
        default: return "UNKNOWN";
    }
}

ge::graphStatus FusedSparseAttentionOverlapMlaTiling::SetBlockDim(uint32_t blockDim) const
{
    context_->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapMlaTiling::SetTilingKey(uint64_t tilingKey) const
{
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1);     // 1: batchmode模式
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapMlaTiling::SetWorkspaceSize(uint64_t workspaceSize) const
{
    OP_CHECK_IF(context_->GetWorkspaceSizes(1) == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "workSpaceSize got from ge is nullptr"),
        return ge::GRAPH_FAILED);
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapMlaTiling::SetTilingData(TilingDef &tilingData) const
{
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
        return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapMlaTiling::GetPlatformInfo()
{
    OP_CHECK_IF(sfaInfo_->platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(sfaInfo_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(sfaInfo_->platformInfo);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();

    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0,
        OPS_REPORT_VECTOR_INNER_ERR(sfaInfo_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void FusedSparseAttentionOverlapMlaTiling::GenTilingKey()
{
    uint32_t inputQType = static_cast<uint32_t>(sfaInfo_->inputQType);
    uint32_t inputKvType = static_cast<uint32_t>(sfaInfo_->inputKvType);
    uint32_t outputType = static_cast<uint32_t>(sfaInfo_->outputType);
    uint32_t layoutQuery = static_cast<uint32_t>(sfaInfo_->qLayout);
    uint32_t layoutKV = static_cast<uint32_t>(sfaInfo_->kvLayout);
    uint32_t pageAttention = 0U;
    if (sfaInfo_->kvLayout == FusedSparseAttentionOverlapLayout::PA_BSND) {
        pageAttention = 1U;
    }

    if (IsFusedSparseAttentionOverlapStandaloneOp(sfaInfo_->opName)) {
        // The migrated SelectionUpdate kernel was built from the older SFA template family.
        // Its registered kernels use the 4-argument key layout:
        //   flash_decode bit0, layout_query bits[1:4], layout_kv bits[5:8], template_mode bit9.
        // Keep the normal FusedSparseAttentionOverlap CANN9 key path unchanged.
        tilingKey_ = static_cast<uint64_t>(splitKVFlag_ ? 1U : 0U) |
            (static_cast<uint64_t>(layoutQuery) << 1U) |
            (static_cast<uint64_t>(layoutKV) << 5U) |
            (static_cast<uint64_t>(perfMode_ == FusedSparseAttentionOverlapPerfMode::V_TEMPLATE_MODE) << 9U);
    } else {
        tilingKey_ = GET_TPL_TILING_KEY(0U, pageAttention, layoutQuery, layoutKV,
            perfMode_ == FusedSparseAttentionOverlapPerfMode::V_TEMPLATE_MODE, static_cast<uint32_t>(sfaInfo_->gSize > 64)); // N1 > 128????G
    }

    OP_LOGI(sfaInfo_->opName, "SFA tilingKey_: %lu.", tilingKey_);
}

void FusedSparseAttentionOverlapMlaTiling::ZeroTensorProcess()  const
{
    if (sfaInfo_->s2Size == 0) {
        /*
         * 1024，空tensor场景下，作为默认值完成后续计算
         * 避免matmal tiling  softmax tiling异常
         * kernel计算使用真实的seqSize=0, 与actuseq_len流程归一
         */
        sfaInfo_->s2Size = 1024;
    }
}

void FusedSparseAttentionOverlapMlaTiling::InitParams()
{
    if (sfaInfo_->s2Size != 0 && sfaInfo_->sparseBlockSize <= 4) { // 4:当前支持范围
        perfMode_ = FusedSparseAttentionOverlapPerfMode::V_TEMPLATE_MODE;
    } else {
        perfMode_ = FusedSparseAttentionOverlapPerfMode::C_TEMPLATE_MODE;
    }

    coreNum_ = aicNum_;

    // splitKV：解码时一核一请求（kernel 侧 usedCoreNum = 总 query token 数），
    // bs 小的时候大部分核在空转。把每个请求的 KV 切成 kvSplitPart_ 段分给多个核，
    // 各算一份局部结果，最后归约。
    // ⚠ 核已经用满时保持 false —— tiling key 与原来逐位相同、编出来的代码一行不变。
    //    bs=16 起就落在这一侧，它同时是这一刀的零对照臂。
    // ⚠ 这里给的是上限，kernel 侧还会按真实的 S2 块数再收一次（host 此时拿不到）。
    // 【wht/opt-lru0723 合并线】本分支暂时不带 splitKV：强制关闭。tiling key
    // bit0=0 → FLASH_DECODE=1 变体不进编译，编出代码与 splitKV 之前逐位相同。
    // 恢复方式：把下一行的 false 换回 git 历史 8ac44f83 里的原条件表达式。
    splitKVFlag_ = false;
    kvSplitPart_ = 1U;
    if (splitKVFlag_) {
        kvSplitPart_ = aicNum_ / sfaInfo_->bSize;
        // 4:生产形状 topk=2048、S2 基本块 512，一个请求只有 4 块，再切就出空片
        if (kvSplitPart_ > 4U) {
            kvSplitPart_ = 4U;
        }
    }

    headDimAlign_ = Align(sfaInfo_->qkHeadDim, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    ZeroTensorProcess();
}

void FusedSparseAttentionOverlapMlaTiling::CalcUbBmm()
{
    uint32_t cubeMSize = sfaInfo_->gSize * sfaInfo_->s1Size;
    uint32_t maxMSize = mBaseSize_;
    if (cubeMSize > maxMSize) {
        cubeMSize = maxMSize;
    }
    mmResUbSize_ = sInnerSizeAlign_ * Align(cubeMSize, 16U);// kernel按照16对齐写出，tiling按照这个原则分配内存
    bmm2ResUbSize_ = headDimAlign_ * Align(cubeMSize, 16U);// kernel按照16对齐写出，tiling按照这个原则分配内存

    qPreSizeMla_ = sfaInfo_->gSize * (headDimAlign_ + 64U) * sfaInfo_->s1Size;
}

void FusedSparseAttentionOverlapMlaTiling::CheckUbSpace()
{
    CalcUbBmm();
}

void FusedSparseAttentionOverlapMlaTiling::CalcInnerSize(uint32_t s2Size)
{
    sInnerSize_ = 512; // 512:s2默认切分大小
    // FlashDecode时，如果S2的计算量>=256(确保切分后不小于128)但又不足以分2次计算时，则修改sInnerSize_，均分为2份进行计算，确保Nbuffer=2
    if (splitKVFlag_ && sfaInfo_->qLayout != FusedSparseAttentionOverlapLayout::TND) {
        if (s2Size == 256) {   // 256:s2Size的阈值，判断sInnerSize_是否切分
            sInnerSize_ = 128; // 128:sInnerSize_值为s2Size的一半，均分为2份进行计算，
        } else if (s2Size > 256 && s2Size <= sInnerSize_) { // 256:s2Size的阈值，判断sInnerSize_是否切分
            sInnerSize_ = (sInnerSize_ + 1) / 2; // 2:减半
        }
    }

    sInnerLoopTimes_ = (s2Size + sInnerSize_ - 1) / sInnerSize_;
    sInnerSizeTail_ = s2Size - (sInnerLoopTimes_ - 1) * sInnerSize_;
    if (sInnerSize_ > s2Size) {
        sInnerSize_ = s2Size;
    }
    sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐

    CheckUbSpace();
}

void FusedSparseAttentionOverlapMlaTiling::SplitBalanced()
{
    CalcInnerSize(sfaInfo_->s2Size);

    FusedOverlapInnerSplitParams innerSplitParams;
    innerSplitParams.s1GBaseSize = sfaInfo_->gSize;
    innerSplitParams.s2BaseSize = sInnerSize_;
    tilingData_.innerSplitParams.set_mBaseSize(innerSplitParams.s1GBaseSize);
    tilingData_.innerSplitParams.set_s2BaseSize(innerSplitParams.s2BaseSize);

    usedCoreNum_ = aicNum_;
}

void FusedSparseAttentionOverlapMlaTiling::Split()
{
    SplitBalanced();
}

void FusedSparseAttentionOverlapMlaTiling::FillTilingBaseParamsMla()
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
    tilingData_.baseParams.set_preTokens(sfaInfo_->preTokens);
    tilingData_.baseParams.set_nextTokens(sfaInfo_->nextTokens);
    tilingData_.baseParams.set_sparseBlockSize(sfaInfo_->sparseBlockSize);
    tilingData_.baseParams.set_sparseBlockCount(sfaInfo_->sparseBlockCount);
    uint32_t selectionBlockTableStride =
        static_cast<uint32_t>((sfaInfo_->sparseBlockCount + sfaInfo_->blockSize - 1) /
                              sfaInfo_->blockSize);
    if (IsFusedSparseAttentionOverlapStandaloneOp(sfaInfo_->opName)) {
        const gert::StorageShape *selectionBlockTableShape =
            context_->GetInputShape(SELECTION_KV_BLOCK_TABLE_INPUT_INDEX);
        if (selectionBlockTableShape != nullptr &&
            selectionBlockTableShape->GetStorageShape().GetDimNum() > 0) {
            int64_t blockTableLastDim = selectionBlockTableShape->GetStorageShape().GetDim(
                selectionBlockTableShape->GetStorageShape().GetDimNum() - 1);
            if (blockTableLastDim >= static_cast<int64_t>(selectionBlockTableStride)) {
                selectionBlockTableStride = static_cast<uint32_t>(blockTableLastDim);
            }
        }
    }
    tilingData_.baseParams.set_selectionBlockTableStride(selectionBlockTableStride);
    uint32_t selectionStatusStride = sfaInfo_->sparseBlockCount + 1;
    if (IsFusedSparseAttentionOverlapStandaloneOp(sfaInfo_->opName)) {
        const gert::StorageShape *selectionStatusShape =
            context_->GetInputShape(SELECTION_KV_BLOCK_STATUS_INPUT_INDEX);
        if (selectionStatusShape != nullptr &&
            selectionStatusShape->GetStorageShape().GetDimNum() > 0) {
            int64_t statusLastDim = selectionStatusShape->GetStorageShape().GetDim(
                selectionStatusShape->GetStorageShape().GetDimNum() - 1);
            if (statusLastDim >= static_cast<int64_t>(selectionStatusStride)) {
                selectionStatusStride = static_cast<uint32_t>(statusLastDim);
            }
        }
    }
    tilingData_.baseParams.set_selectionStatusStride(selectionStatusStride);
    uint32_t selectionMembershipStride = SELECTION_MEMBERSHIP_STORAGE_INT16_COUNT;
    if (IsFusedSparseAttentionOverlapStandaloneOp(sfaInfo_->opName)) {
        const gert::StorageShape *selectionMembershipShape =
            context_->GetInputShape(SELECTION_MEMBERSHIP_MAP_INPUT_INDEX);
        if (selectionMembershipShape != nullptr &&
            selectionMembershipShape->GetStorageShape().GetDimNum() > 0) {
            int64_t membershipLastDim = selectionMembershipShape->GetStorageShape().GetDim(
                selectionMembershipShape->GetStorageShape().GetDimNum() - 1);
            if (membershipLastDim >= static_cast<int64_t>(selectionMembershipStride)) {
                selectionMembershipStride = static_cast<uint32_t>(membershipLastDim);
            }
        }
    }
    tilingData_.baseParams.set_selectionMembershipStride(selectionMembershipStride);
    tilingData_.baseParams.set_attentionMode(sfaInfo_->attentionMode);
    tilingData_.baseParams.set_returnSoftmaxLse(sfaInfo_->returnSoftmaxLse);
    tilingData_.baseParams.set_isActualLenDimsNull(sfaInfo_->actualQSeqLenFlag ? 0U : 1U);
    tilingData_.baseParams.set_isActualLenDimsKVNull(sfaInfo_->actualSeqLenFlag ? 0U : 1U);
}

// for flash decode
void FusedSparseAttentionOverlapMlaTiling::FillTilingSplitKVMla()
{
    tilingData_.splitKVParams.set_s2(kvSplitPart_);

    tilingData_.splitKVParams.set_accumOutSize(aicNum_ * 2 * sfaInfo_->n2Size * mBaseSize_ * headDimAlign_);   // 2:每个核可能有头规约和尾规约，一共两份规约信息
    tilingData_.splitKVParams.set_logSumExpSize(2 * aicNum_ * 2 * sfaInfo_->n2Size * mBaseSize_ *  // 2:每个核可能有头规约和尾规约，一共两份规约信息;sum + max
                                                (BYTE_BLOCK / BLOCK_TABLE_ELEM_BYTE));

    if (!splitKVFlag_) {
        tilingData_.splitKVParams.set_s2(0);
    }
}

void FusedSparseAttentionOverlapMlaTiling::FillTilingSingleCoreParamsMla()
{
    tilingData_.singleCoreParams.set_usedCoreNum(usedCoreNum_);
}

void FusedSparseAttentionOverlapMlaTiling::FillTilingSingleCoreTensorSizeMla()
{
    tilingData_.singleCoreTensorSize.set_mmResUbSize(mmResUbSize_);
    tilingData_.singleCoreTensorSize.set_bmm2ResUbSize(bmm2ResUbSize_);
}

void FusedSparseAttentionOverlapMlaTiling::FillTiling()
{
    FillTilingBaseParamsMla();
    FillTilingSplitKVMla();
    FillTilingSingleCoreParamsMla();
    FillTilingSingleCoreTensorSizeMla();
}

uint32_t FusedSparseAttentionOverlapMlaTiling::CalcBalanceFDParamNums(const uint32_t actCoreNum)  const
{
    return actCoreNum * 2 * sfaInfo_->n2Size * mBaseSize_; // 2:每个核可能有头规约和尾规约，一共两份规约信息
}

void FusedSparseAttentionOverlapMlaTiling::NormalCalcFDWorkSpace(const uint32_t actCoreNum)
{
    if (splitKVFlag_) {
        uint32_t accumOutSize = 0;
        uint32_t logSumExpSize = 0;
        uint32_t FDParamNums = CalcBalanceFDParamNums(actCoreNum); //balanceModeFlag_ ? CalcBalanceFDParamNums(actCoreNum) : CalcUnbalanceFDParamNums();
        accumOutSize = FDParamNums * headDimAlign_;
        logSumExpSize = 2 * FDParamNums * (BYTE_BLOCK / sfaInfo_->blockTypeSize);  // log和sum的存储空间一致，共需要2份内存
        workspaceSize_ += (accumOutSize + logSumExpSize) * sfaInfo_->blockTypeSize;
    }
}

void FusedSparseAttentionOverlapMlaTiling::CalcFDWorkSpace(const uint32_t actCoreNum)
{
    NormalCalcFDWorkSpace(actCoreNum);
}

void FusedSparseAttentionOverlapMlaTiling::GetWorkspaceSize()
{
    uint32_t actCoreNum = coreNum_;
    if (sfaInfo_->isA5) {
        workspaceSize_ = libapiSize_;
        constexpr uint32_t TRIPLE_BUFFER_NUM = 3;
        constexpr uint32_t S2_BASE_SIZE = 128;            // S2轴基本块大小
        constexpr uint32_t D_SIZE = 576;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(sfaInfo_->platformInfo);
        uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
        if (sfaInfo_->gSize > 64) { // N1大于64时切G，相邻两个cube核处理同一个s2Base
            aicNum = aicNum >> 1;
        }
        workspaceSize_ += (S2_BASE_SIZE * D_SIZE * GetTypeSize(sfaInfo_->inputQType) \
            * TRIPLE_BUFFER_NUM * aicNum);
    } else {
        uint32_t mmResElemSize = 4;         // 4:fp32
        uint32_t vec1ResElemSize = 2;       // 2:fp16/bf16
        uint32_t bmm2ResElemSize = 4;       // 4:fp32
        uint32_t qPreProcResElemSize = 0;   // 普通场景不涉及Q预处理
        uint32_t nUpdateElemSize = 4;   // 4:int32
        uint32_t softmaxSumElemSize = 4;   // 4:int32
        float kvDtypeRatio = 1.0;

        workspaceSize_ = libapiSize_;
        uint32_t preLoadNum = 1;
        preLoadNum = PRE_LOAD_NUM;

        workspaceSize_ += preLoadNum * (mmResUbSize_ * actCoreNum * mmResElemSize);
        workspaceSize_ += preLoadNum * static_cast<size_t>(static_cast<float>(mmResUbSize_ * \
            actCoreNum * vec1ResElemSize) * kvDtypeRatio);
        workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * bmm2ResElemSize;
        workspaceSize_ += preLoadNum * static_cast<size_t>(static_cast<float>(qPreSizeMla_ * \
            actCoreNum * qPreProcResElemSize) * kvDtypeRatio);
        workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * nUpdateElemSize;
        workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * softmaxSumElemSize;
        // topk BlkSize == 1场景, 需要额外空间缓存离散聚合的值
        //              bufNum  s2Base   D   dRope  sizeOf(half)
        // 4:bufNum  512:s2Base  512:D  64:dRope  2:sizeOf(half)
        workspaceSize_ += 4 * 512 * (512 + 64) * 2 * actCoreNum;
        // 缓存有效mte2 size的长度 份数  512B对齐的长度  sizeof(int32_t)   aiv核数
        workspaceSize_ += 4 * 128 * 4 * (2 * actCoreNum); // 4:缓存有效mte2 size的长度 128:份数  4:512B对齐的长度  2:aiv核数
        // helper 握手区独立选址扩容：done/credit 两环，每核 64 槽×8 int32。
        // 原借空闲核的 merge 槽，波次满波 24 核全写 merge 会把握手区踩脏、
        // 尾波前被迫双 SyncAll+清零；独立选址后满波不再触碰（kernel 侧同步改址）。
        workspaceSize_ += 2 * actCoreNum * 64 * 8 * 4;
    }

    CalcFDWorkSpace(actCoreNum);
}

void FusedSparseAttentionOverlapMlaTiling::CalcBlockDim()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(sfaInfo_->platformInfo);
    auto aicNum = usedCoreNum_;
    auto aivNum = 2 * usedCoreNum_;

    blockDim_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    OP_LOGI(sfaInfo_->opName, "SFA block dim: %u aiv Num: %u aic Num: %u.", blockDim_, aivNum, aicNum);
}

ge::graphStatus FusedSparseAttentionOverlapMlaTiling::DoOpTiling(FusedSparseAttentionOverlapTilingInfo *sfaInfo)
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

ge::graphStatus TilingFusedSparseAttentionOverlap(gert::TilingContext *context)
{
    FusedSparseAttentionOverlapTilingInfo sfaInfo;
    FusedSparseAttentionOverlapInfoParser sfaInfoParser(context);
    if (sfaInfoParser.Parse(sfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    FusedSparseAttentionOverlapTilingCheck tilingChecker(sfaInfo);
    if (tilingChecker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    FusedSparseAttentionOverlapMlaTiling tiling(context);
    return tiling.DoOpTiling(&sfaInfo);
}

ge::graphStatus TilingPrepareForFusedSparseAttentionOverlap(gert::TilingParseContext* const context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::GetExpectedShape(gert::Shape &shapeExpected,
    const FusedSparseAttentionOverlapTilingShapeCompareParam &param, const FusedSparseAttentionOverlapLayout &layout) const
{
    if (layout == FusedSparseAttentionOverlapLayout::BSND) {
        shapeExpected = gert::Shape({param.B, param.S, param.N, param.D});
    } else if (layout == FusedSparseAttentionOverlapLayout::TND) {
        shapeExpected = gert::Shape({param.T, param.N, param.D});
    } else if (layout == FusedSparseAttentionOverlapLayout::PA_BSND) {
        shapeExpected = gert::Shape({param.Bn, param.Bs, param.N, param.D});
    } else if (layout == FusedSparseAttentionOverlapLayout::BNSG) {
        shapeExpected = gert::Shape({param.B, param.N, param.S, param.G});
    } else if (layout == FusedSparseAttentionOverlapLayout::NTG) {
        shapeExpected = gert::Shape({param.N, param.T, param.G});
    } else {
        OP_LOGE(opName_, "layout %s is unsupported", FusedSparseAttentionOverlapLayoutToSerialString(layout).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CompareShape(FusedSparseAttentionOverlapTilingShapeCompareParam &param,
    const gert::Shape &shape, const FusedSparseAttentionOverlapLayout &layout, const std::string &name) const
{
    gert::Shape shapeExpected;
    if (GetExpectedShape(shapeExpected, param, layout) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (shape.GetDimNum() != shapeExpected.GetDimNum()) {
        OP_LOGE(opName_,
            "%s dimension is %zu, expected dimension is %zu.",
            name.c_str(), shape.GetDimNum(), shapeExpected.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) != shapeExpected.GetDim(i)) {
            OP_LOGE(opName_, "%s layout is %s, shape is %s, expected shape is %s.",
                name.c_str(), FusedSparseAttentionOverlapLayoutToSerialString(layout).c_str(),
                GetShapeStr(shape).c_str(), GetShapeStr(shapeExpected).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

void FusedSparseAttentionOverlapTilingCheck::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
    const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << FusedSparseAttentionOverlapDataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName_, "Tensor %s only supports dtype %s, but got %s",
        name.c_str(), oss.str().c_str(), FusedSparseAttentionOverlapDataTypeToSerialString(actualDtype).c_str());
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
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

template <typename T>
void FusedSparseAttentionOverlapTilingCheck::LogErrorNumberSupport(const std::vector<T> &expectNumberList,
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
void FusedSparseAttentionOverlapTilingCheck::LogErrorDimNumSupport(const std::vector<T> &expectNumberList,
    const T &actualValue, const std::string &name) const
{
    LogErrorNumberSupport(expectNumberList, actualValue, name, "dimension");
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckDimNumInLayoutSupport(const FusedSparseAttentionOverlapLayout &layout,
    const gert::StorageShape *shape, const std::string &name) const
{
    const auto& dimIt = FUSED_SPARSE_ATTENTION_OVERLAP_LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
        OP_LOGE(opName_, "When layout is %s, %s dimension should be %zu, but it's %zu",
            FusedSparseAttentionOverlapLayoutToSerialString(layout).c_str(), name.c_str(), dimIt->second,
            shape->GetStorageShape().GetDimNum()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckDimNumSupport(const gert::StorageShape *shape,
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


void FusedSparseAttentionOverlapTilingCheck::LogErrorLayoutSupport(const std::vector<FusedSparseAttentionOverlapLayout> &expectLayoutList,
    const FusedSparseAttentionOverlapLayout &actualLayout, const std::string &name) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectLayoutList.size(); ++i) {
        oss << FusedSparseAttentionOverlapLayoutToSerialString(expectLayoutList[i]);
        if (i < expectLayoutList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE(opName_, "Tensor %s only supports layout %s, but got %s",
        name.c_str(), oss.str().c_str(), FusedSparseAttentionOverlapLayoutToSerialString(actualLayout).c_str());
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckLayoutSupport(const FusedSparseAttentionOverlapLayout &actualLayout, const std::string &name) const
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

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckSingleParaQuery() const
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

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckSingleParaKey() const
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

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckSingleParaNumHeads() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckSingleParaKvHeadNums() const
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckSingleParaSparseMode() const
{
    OP_CHECK_IF((*opParamInfo_.sparseMode != 3 && *opParamInfo_.sparseMode != 0),
        OP_LOGE(opName_, "sparseMode must == 0/3, but got: %ld.", *opParamInfo_.sparseMode),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckSingleParaSparseBlockSize() const
{
    OP_CHECK_IF((*opParamInfo_.sparseBlockSize <= 0 || *opParamInfo_.sparseBlockSize > 128 ||
        (static_cast<uint64_t>(*opParamInfo_.sparseBlockSize) & static_cast<uint64_t>(*opParamInfo_.sparseBlockSize - 1L)) != 0UL),
        OP_LOGE(opName_, "sparseBlockSize should be be in range [1, 128] and be a power of 2, but got: %ld.",
            *opParamInfo_.sparseBlockSize),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((npuArch_ == NpuArch::DAV_3510 && *opParamInfo_.sparseBlockSize != 1),
        OP_LOGE(opName_, "when soc version is Ascend950, sparse_block_size only support 1, but now is %d.",
        *opParamInfo_.sparseBlockSize), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckSingleParaSparseIndices() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.sparseIndices.desc, SPARSE_INDICES_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckSingleParaPreTokens() const
{
    OP_CHECK_IF((*opParamInfo_.preTokens != INT64_MAX),
        OP_LOGE(opName_, "preTokens should be 9223372036854775807, but got: %ld.", *opParamInfo_.preTokens),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckSingleParaNextTokens() const
{
    OP_CHECK_IF((*opParamInfo_.nextTokens != INT64_MAX),
        OP_LOGE(opName_, "nextTokens should be 9223372036854775807, but got: %ld.", *opParamInfo_.nextTokens),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaQuery() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKey() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseIndices() ||
        ge::GRAPH_SUCCESS != CheckSingleParaNumHeads() ||
        ge::GRAPH_SUCCESS != CheckSingleParaKvHeadNums() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseMode() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseBlockSize() ||
        ge::GRAPH_SUCCESS != CheckSingleParaPreTokens() ||
        ge::GRAPH_SUCCESS != CheckSingleParaNextTokens()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckRopeExistence()
{
    OP_CHECK_IF((opParamInfo_.queryRope.tensor != nullptr || opParamInfo_.keyRope.tensor != nullptr)
        && *opParamInfo_.attentionMode == 0,
        OP_LOGE(opName_, "In MHA/GQA situation(attentionMode=0), queryRope and keyRope should be null."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(*opParamInfo_.attentionMode != 2,
        OP_LOGE(opName_, "attentionMode only support 2."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((opParamInfo_.queryRope.tensor != nullptr && opParamInfo_.keyRope.tensor == nullptr),
        OP_LOGE(opName_, "KeyRope is null, but queryRope exists, they should be both null or exist."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((opParamInfo_.queryRope.tensor == nullptr && opParamInfo_.keyRope.tensor != nullptr),
        OP_LOGE(opName_, "QueryRope is null, but keyRope exists, they should be both null or exist."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.keyRope.desc == nullptr || opParamInfo_.queryRope.desc == nullptr,
        OP_LOGE(opName_, "In Mla situation, desc of keyRope and queryRope should not be null"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckExists(const void *pointer, const std::string &name) const
{
    OP_CHECK_IF(pointer == nullptr,
        OP_LOGE(opName_, "%s should not be null", name.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckNotExists(const void *pointer, const std::string &name) const
{
    OP_CHECK_IF(pointer != nullptr,
        OP_LOGE(opName_, "%s should be null", name.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto& kv : paramMap) {
        if (CheckExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckNotExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto& kv : paramMap) {
        if (CheckNotExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckExistenceByMap(std::map<std::string, const void *> &existMap,
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
ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckAttrValueByMap(std::map<std::string, std::pair<const T *, T>> &attrMap) const
{
    for (auto const &kv : attrMap) {
        const std::string &name = kv.first;
        const std::pair<const T *, T> &pointerValuePair = kv.second;
        if (pointerValuePair.first == nullptr) {
            OP_LOGE(opName_, "Attr %s should not be nullptr", name.c_str());
            return ge::GRAPH_FAILED;
        }

        if (*(pointerValuePair.first) != pointerValuePair.second) {
            std::ostringstream ossExpect;
            ossExpect << std::to_string(pointerValuePair.second);
            std::ostringstream ossActual;
            ossActual << std::to_string(*(pointerValuePair.first));
            OP_LOGE(opName_,
                "%s value should be %s, but got %s",
                name.c_str(),
                ossExpect.str().c_str(),
                ossActual.str().c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckParaExistenceMlaNoquant() const
{
    if (kvStorageMode_ != FusedOverlapKvStorageMode::PAGE_ATTENTION) {
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

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckParaExistenceMla() const
{
    return CheckParaExistenceMlaNoquant();
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckParaExistence()
{
    if (ge::GRAPH_SUCCESS != CheckRopeExistence()) {
        return ge::GRAPH_FAILED;
    }

    return CheckParaExistenceMla();
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    const FusedSparseAttentionOverlapLayout &layout, const std::string &name) const
{
    if (tensor == nullptr) {
        OP_LOGE(opName_, "when layout of query is %s, %s must be provided.",
            FusedSparseAttentionOverlapLayoutToSerialString(layout).c_str(), name.c_str());
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

void FusedSparseAttentionOverlapTilingCheck::SetSFAShapeCompare()
{
    queryShapeCmp_ = opParamInfo_.query.shape->GetStorageShape();
    topkShapeCmp_ = opParamInfo_.sparseIndices.shape->GetStorageShape();
    keyShapeCmp_ = opParamInfo_.key.shape->GetStorageShape();
    valueShapeCmp_ = opParamInfo_.value.shape->GetStorageShape();
    attenOutShapeCmp_ = opParamInfo_.attenOut.shape->GetStorageShape();
    queryRopeShapeCmp_ = opParamInfo_.queryRope.tensor->GetStorageShape();
    keyRopeShapeCmp_ = opParamInfo_.keyRope.tensor->GetStorageShape();
    if (!IsFusedSparseAttentionOverlapStandaloneOp(opName_)) {
        softmaxMaxShapeCmp_ = opParamInfo_.softmaxMax.shape->GetStorageShape();
        softmaxSumShapeCmp_ = opParamInfo_.softmaxSum.shape->GetStorageShape();
    }
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckBlockTable() const
{
    if (kvStorageMode_ != FusedOverlapKvStorageMode::PAGE_ATTENTION) {
        OP_CHECK_IF(opParamInfo_.blockTable.tensor != nullptr,
            OP_LOGE(opName_, "when the layout_kv is %s, %s should be null",
                FusedSparseAttentionOverlapLayoutToSerialString(kvLayout_).c_str(), BLOCK_TABLE_NAME.c_str()),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.blockTable.desc, BLOCK_TABLE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    uint32_t blockTableBatch = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(blockTableBatch != bSize_,
        OP_LOGE(opName_, "%s's first dimension(%u) should be equal to batch size(%u)",
            BLOCK_TABLE_NAME.c_str(), blockTableBatch, bSize_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckDTypeConsistency(const ge::DataType &actualDtype,
    const ge::DataType &expectDtype, const std::string &name) const
{
    if (actualDtype != expectDtype) {
        OP_LOGE(opName_, "%s dtype should be %s, but it's %s.", name.c_str(),
            FusedSparseAttentionOverlapDataTypeToSerialString(expectDtype).c_str(),
            FusedSparseAttentionOverlapDataTypeToSerialString(actualDtype).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckQRopeShape()
{
    FusedSparseAttentionOverlapTilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n1Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = ropeHeadDim_;
    shapeParams.T = qTSize_;
    return CompareShape(shapeParams, queryRopeShapeCmp_, qLayout_, QUERY_ROPE_NAME);
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckTopkShape()
{
    FusedSparseAttentionOverlapTilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n2Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = sparseBlockCount_;
    shapeParams.T = qTSize_;
    return CompareShape(shapeParams, topkShapeCmp_, topkLayout_, SPARSE_INDICES_NAME);
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckAttenOutShape()
{
    FusedSparseAttentionOverlapTilingShapeCompareParam shapeParams;
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

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckSoftmaxMaxShape()
{
    if (*opParamInfo_.returnSoftmaxLse) {
        FusedSparseAttentionOverlapTilingShapeCompareParam shapeParams;
        shapeParams.B = bSize_;
        shapeParams.N = n2Size_;
        shapeParams.S = s1Size_;
        shapeParams.T = qTSize_;
        shapeParams.G = n1Size_/n2Size_;
        if (CompareShape(shapeParams, softmaxMaxShapeCmp_, softmaxMaxLayout_, SOFTMAX_MAX_NAME) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
    }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckSoftmaxSumShape()
{
    if (*opParamInfo_.returnSoftmaxLse) {
        FusedSparseAttentionOverlapTilingShapeCompareParam shapeParams;
        shapeParams.B = bSize_;
        shapeParams.N = n2Size_;
        shapeParams.S = s1Size_;
        shapeParams.D = vHeadDim_;
        shapeParams.T = qTSize_;
        shapeParams.G = n1Size_/n2Size_;
        if (CompareShape(shapeParams, softmaxSumShapeCmp_, softmaxSumLayout_, SOFTMAX_SUM_NAME) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckAttenOut()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.attenOut.desc->GetDataType(),
        inputQType_, ATTEN_OUT_NAME) ||
        ge::GRAPH_SUCCESS != CheckAttenOutShape()) {
        return ge::GRAPH_FAILED;
    }
    if (!IsFusedSparseAttentionOverlapStandaloneOp(opName_) &&
        (ge::GRAPH_SUCCESS != CheckSoftmaxMaxShape() ||
         ge::GRAPH_SUCCESS != CheckSoftmaxSumShape())) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckSoftmaxMax()
{
    if (*opParamInfo_.returnSoftmaxLse) {
        OP_CHECK_IF(opParamInfo_.softmaxMax.shape->GetStorageShape().GetShapeSize() == 0,
            OP_LOGE(opName_, "When return_softmax_lse is true, SoftmaxMax tensor cannot be empty tensor."),
                return ge::GRAPH_FAILED);
        // type类型校验
        OP_CHECK_IF(opParamInfo_.softmaxMax.desc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE(opName_, "SoftmaxMax's dtype must be FLOAT."),
                return ge::GRAPH_FAILED);
        // shape和维度校验
        if (ge::GRAPH_SUCCESS != CheckAttenOutShape()) {
                return ge::GRAPH_FAILED;
        }
    } else {
        if (opParamInfo_.softmaxMax.shape->GetStorageShape().GetShapeSize() != 0) {
            OP_LOGW(opName_, "When return_softmax_lse is false, SoftmaxMax tensor must be empty tensor.");
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckSoftmaxSum()
{
    if (*opParamInfo_.returnSoftmaxLse) {
        OP_CHECK_IF(opParamInfo_.softmaxSum.shape->GetStorageShape().GetShapeSize() == 0,
            OP_LOGE(opName_, "When return_softmax_lse is true, softmaxSum tensor cannot be empty tensor."),
                return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.softmaxSum.desc->GetDataType() != ge::DT_FLOAT,
            OP_LOGE(opName_, "softmaxSum's dtype must be FLOAT."),
                return ge::GRAPH_FAILED);
    } else {
        if (opParamInfo_.softmaxSum.shape->GetStorageShape().GetShapeSize() != 0) {
            OP_LOGW(opName_, "When return_softmax_lse is false, softmaxSum tensor must be empty tensor.");
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckQRope()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.queryRope.desc->GetDataType(),
        inputQType_, QUERY_ROPE_NAME) ||
        ge::GRAPH_SUCCESS != CheckQRopeShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckTopK()
{
    if (ge::GRAPH_SUCCESS != CheckTopkShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckVAndKRopeShapeForBatchContinuous()
{
    FusedSparseAttentionOverlapTilingShapeCompareParam shapeParams;
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

uint32_t FusedSparseAttentionOverlapMlaTiling::GetTypeSize(ge::DataType dtype) const
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

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckVAndKRopeShapeForPageAttention()
{
    int64_t blockNum = keyShapeCmp_.GetDim(0);
    OP_CHECK_IF(blockNum <= 0,
        OP_LOGE(opName_, "The first dim(%ld) of key should be greater than 0", blockNum),
        return ge::GRAPH_FAILED);
    FusedSparseAttentionOverlapTilingShapeCompareParam shapeParams;
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

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckVAndKRopeShape()
{
    if (kvStorageMode_ == FusedOverlapKvStorageMode::BATCH_CONTINUOUS) {
        return CheckVAndKRopeShapeForBatchContinuous();
    }

    if (kvStorageMode_ == FusedOverlapKvStorageMode::PAGE_ATTENTION) {
        return CheckVAndKRopeShapeForPageAttention();
    }

    OP_LOGE(opName_, "storage mode of key and value is %u, it is incorrect.", static_cast<uint32_t>(kvStorageMode_));
    return ge::GRAPH_FAILED;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckVAndKRope()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.value.desc->GetDataType(),
        inputKvType_, VALUE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.keyRope.desc->GetDataType(),
        inputKvType_, KEY_ROPE_NAME) || ge::GRAPH_SUCCESS != CheckVAndKRopeShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckActualSeqLensQ()
{
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensQDType() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckActualSeqLensQDType()
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.actualSeqLengthsQ.desc == nullptr) {
        OP_LOGE(opName_, "actualSeqLengthsQ is not empty,"
            "but actualSeqLengthsQ's dtype is nullptr.");
            return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.actualSeqLengthsQ.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE(opName_, "actualSeqLengthsQ's dtype is %s, it should be DT_INT32.",
            FusedSparseAttentionOverlapDataTypeToSerialString(opParamInfo_.actualSeqLengthsQ.desc->GetDataType()).c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckActualSeqLensQShape()
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t shapeSize = 0;
    if (GetActualSeqLenSize(shapeSize, opParamInfo_.actualSeqLengthsQ.tensor, qLayout_, "actualSeqLengthsQ") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (shapeSize != bSize_) {
        OP_LOGE(opName_, "actualSeqLengthsQ shape size is %u, it should be equal to batch size[%u]",
            shapeSize, bSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckActualSeqLens()
{
    if (std::string(opParamInfo_.layoutKV) == "TND" && opParamInfo_.actualSeqLengths.tensor == nullptr) {
        OP_LOGE(opName_,
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

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckActualSeqLensDType()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.actualSeqLengths.desc == nullptr) {
        OP_LOGE(opName_, "actualSeqLengths is not empty,"
            "but actualSeqLengths's dtype is nullptr.");
            return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.actualSeqLengths.desc->GetDataType() != ge::DT_INT32) {
        OP_LOGE(opName_, "actualSeqLengths's dtype is %s, it should be DT_INT32.",
            FusedSparseAttentionOverlapDataTypeToSerialString(opParamInfo_.actualSeqLengths.desc->GetDataType()).c_str());
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckActualSeqLensShape()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t shapeSize = 0;
    if(GetActualSeqLenSize(shapeSize, opParamInfo_.actualSeqLengths.tensor, kvLayout_, "actualSeqLengths") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (shapeSize != bSize_) {
        OP_LOGE(opName_, "actualSeqLengths shape size is %u, it should be equal to batch size[%u].",
            shapeSize, bSize_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckMultiParaConsistency()
{
    SetSFAShapeCompare();
    if (ge::GRAPH_SUCCESS != CheckVAndKRope() ||
        ge::GRAPH_SUCCESS != CheckQRope() ||
        ge::GRAPH_SUCCESS != CheckTopK() ||
        ge::GRAPH_SUCCESS != CheckAttenOut() ||
        (!IsFusedSparseAttentionOverlapStandaloneOp(opName_) && ge::GRAPH_SUCCESS != CheckSoftmaxMax()) ||
        (!IsFusedSparseAttentionOverlapStandaloneOp(opName_) && ge::GRAPH_SUCCESS != CheckSoftmaxSum()) ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQ() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckBlockTable()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckFeatureMlaNoQuantShape() const
{
    OP_CHECK_IF(bSize_ <= 0,
        OP_LOGE(opName_, "batch_size should be greater than 0, but got %u", bSize_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(qTSize_ <= 0 && (qLayout_ == FusedSparseAttentionOverlapLayout::TND),
        OP_LOGE(opName_, "T_size of query should be greater than 0, but got %u", qTSize_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(n1Size_ <= 0,
        OP_LOGE(opName_, "q_head_num should be greater than 0, but got %u", n1Size_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(n2Size_ != 1,
        OP_LOGE(opName_, "kv_head_num should be 1, but got %u", n2Size_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(n1Size_ % n2Size_ != 0,
        OP_LOGE(opName_, "q_head_num(%u) must be divisible by kv_head_num(%u)", n1Size_, n2Size_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(gSize_ < 1 || (gSize_ > 64 && gSize_ != 128),
        OP_LOGE(opName_, "group num should be in 1 ~ 64, 128, but got %u", gSize_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(qkHeadDim_ != 512,
        OP_LOGE(opName_, "qk_head_dim only support 512, but got %u", qkHeadDim_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(qkHeadDim_ != vHeadDim_,
        OP_LOGE(opName_, "qk_head_dim[%u] should be equal to v_head_dim[%u]", qkHeadDim_, vHeadDim_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(ropeHeadDim_ != 64,
        OP_LOGE(opName_, "rope_head_dim should be 64, but got %u", ropeHeadDim_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckFeatureMlaNoQuantLayout() const
{
    const std::vector<std::string> layoutSupportList = {
        "BSND",
        "TND"
    };
    std::string layoutQuery = opParamInfo_.layoutQuery;
    OP_CHECK_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layoutQuery) == layoutSupportList.end(),
        OP_LOGE(opName_, "layoutQuery only supports BSND/TND, but got %s", layoutQuery.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckFeatureMlaNoQuantDtype() const
{
    OP_CHECK_IF(inputQType_ != ge::DT_BF16 && inputQType_ != ge::DT_FLOAT16,
        OP_LOGE(opName_, "query dtype only support %s and %s, but got %s",
            FusedSparseAttentionOverlapDataTypeToSerialString(ge::DT_BF16).c_str(), FusedSparseAttentionOverlapDataTypeToSerialString(ge::DT_FLOAT16).c_str(),
            FusedSparseAttentionOverlapDataTypeToSerialString(inputQType_).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckFeatureMlaNoquantPa() const
{
    if (kvStorageMode_ != FusedOverlapKvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(blockSize_ <= 0 || blockSize_ > static_cast<int32_t>(MAX_BLOCK_SIZE),
        OP_LOGE(opName_, "when page attention is enabled, block_size(%d) should be in range (0, %u].",
        blockSize_, MAX_BLOCK_SIZE), return ge::GRAPH_FAILED);

    OP_CHECK_IF(blockSize_ % 16 > 0,
        OP_LOGE(opName_, "when page attention is enabled, block_size(%d) should be 16-aligned.",
        blockSize_), return ge::GRAPH_FAILED);

    OP_CHECK_IF(blockSize_ % sparseBlockSize_ > 0,
        OP_LOGE(opName_, "when page attention is enabled, block_size(%d) must be divided by sparse_block_size(%d), but now the remainder is %d.",
        blockSize_, sparseBlockSize_, blockSize_ % sparseBlockSize_), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckFeatureMlaNoquant() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantShape() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoQuantDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaNoquantPa()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckFeatureMla() const
{
    return CheckFeatureMlaNoquant();
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::CheckFeature() const
{
    return CheckFeatureMla();
}

void FusedSparseAttentionOverlapTilingCheck::Init()
{
    opName_ = sfaInfo_.opName;
    platformInfo_ = sfaInfo_.platformInfo;
    opParamInfo_ = sfaInfo_.opParamInfo;
    npuArch_ = sfaInfo_.npuArch;
    isA5_ = sfaInfo_.isA5;

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
    softmaxMaxLayout_ = sfaInfo_.softmaxMaxLayout;
    softmaxSumLayout_ = sfaInfo_.softmaxSumLayout;

    kvStorageMode_ = sfaInfo_.kvStorageMode;
    l2CacheSize_ = sfaInfo_.l2CacheSize;
}

ge::graphStatus FusedSparseAttentionOverlapTilingCheck::Process()
{
    Init();
    if (CheckSinglePara() != ge::GRAPH_SUCCESS ||
        CheckParaExistence() != ge::GRAPH_SUCCESS ||
        CheckFeature() != ge::GRAPH_SUCCESS ||
        CheckMultiParaConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool FusedSparseAttentionOverlapInfoParser::HasAxis(const FusedSparseAttentionOverlapAxis &axis, const FusedSparseAttentionOverlapLayout &layout, const gert::Shape &shape) const
{
    const auto& layoutIt = FUSED_SPARSE_ATTENTION_OVERLAP_LAYOUT_AXIS_MAP.find(layout);
    if (layoutIt == FUSED_SPARSE_ATTENTION_OVERLAP_LAYOUT_AXIS_MAP.end()) {
        return false;
    }

    const std::vector<FusedSparseAttentionOverlapAxis>& axes = layoutIt->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    if (axisIt == axes.end()) {
        return false;
    }
    const auto& dimIt = FUSED_SPARSE_ATTENTION_OVERLAP_LAYOUT_DIM_MAP.find(layout);
    if (dimIt == FUSED_SPARSE_ATTENTION_OVERLAP_LAYOUT_DIM_MAP.end() || dimIt->second != shape.GetDimNum()) {
        return false;
    }
    return true;
}

size_t FusedSparseAttentionOverlapInfoParser::GetAxisIdx(const FusedSparseAttentionOverlapAxis &axis, const FusedSparseAttentionOverlapLayout &layout) const
{
    const std::vector<FusedSparseAttentionOverlapAxis>& axes = FUSED_SPARSE_ATTENTION_OVERLAP_LAYOUT_AXIS_MAP.find(layout)->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    return std::distance(axes.begin(), axisIt);
}

uint32_t FusedSparseAttentionOverlapInfoParser::GetAxisNum(const gert::Shape &shape, const FusedSparseAttentionOverlapAxis &axis,const FusedSparseAttentionOverlapLayout &layout) const
{
    return HasAxis(axis, layout, shape) ? shape.GetDim(GetAxisIdx(axis, layout)) : invalidDimValue_;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.query.shape == nullptr, OP_LOGE(opName_, "Shape of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query.desc == nullptr, OP_LOGE(opName_, "Desc of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.shape == nullptr, OP_LOGE(opName_, "Shape of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.desc == nullptr, OP_LOGE(opName_, "Desc of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.value.shape == nullptr, OP_LOGE(opName_, "Shape of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.value.desc == nullptr, OP_LOGE(opName_, "Desc of tensor value is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseIndices.shape == nullptr, OP_LOGE(opName_, "Shape of tensor sparseIndices is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseIndices.desc == nullptr, OP_LOGE(opName_, "Desc of tensor sparseIndices is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attenOut.shape == nullptr, OP_LOGE(opName_, "Shape of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attenOut.desc == nullptr, OP_LOGE(opName_, "Desc of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    if (!IsFusedSparseAttentionOverlapStandaloneOp(opName_)) {
        OP_CHECK_IF(opParamInfo_.softmaxMax.shape == nullptr, OP_LOGE(opName_, "Shape of tensor softmaxMax is nullptr"),
                return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.softmaxMax.desc == nullptr, OP_LOGE(opName_, "Desc of tensor softmaxMax is nullptr"),
                return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.softmaxSum.shape == nullptr, OP_LOGE(opName_, "Shape of tensor softmaxSum is nullptr"),
                return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.softmaxSum.desc == nullptr, OP_LOGE(opName_, "Desc of tensor softmaxSum is nullptr"),
                return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(opParamInfo_.queryRope.tensor == nullptr, OP_LOGE(opName_, "Shape of queryRope is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.queryRope.desc == nullptr, OP_LOGE(opName_, "Desc of queryRope is nullptr"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.layoutQuery == nullptr, OP_LOGE(opName_, "attr layoutQuery is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.layoutKV == nullptr, OP_LOGE(opName_, "attr layoutKV is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseBlockSize == nullptr, OP_LOGE(opName_, "attr sparseBlockSize is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.scaleValue == nullptr, OP_LOGE(opName_, "attr scaleValue is nullptr"),
               return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseMode == nullptr, OP_LOGE(opName_, "attr sparseMode is nullptr"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS ||
        CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
    FusedSparseAttentionOverlapLayout &layout, const std::string &name) const
{
    if ((tensor == nullptr)) {
        OP_LOGE(opName_, "when layout of query is %s, %s must be provided.",
            FusedSparseAttentionOverlapLayoutToSerialString(layout).c_str(), name.c_str());
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

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetActualSeqLenQSize(uint32_t &size)
{
    return GetActualSeqLenSize(size, opParamInfo_.actualSeqLengthsQ.tensor, qLayout_, "actualSeqLengthsQ");
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("FusedSparseAttentionOverlap", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0,
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "num of core obtained is 0."), return GRAPH_FAILED);

    npuArch_ = ascendcPlatform.GetCurNpuArch();
    isA5_ = (npuArch_ == NpuArch::DAV_3510);
    if (npuArch_ != NpuArch::DAV_2201 && npuArch_ != NpuArch::DAV_3510) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "Npu Arch Version[%d] is not support.", static_cast<int32_t>(npuArch_));
        return GRAPH_FAILED;
    }

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize_);

    return ge::GRAPH_SUCCESS;
}

void FusedSparseAttentionOverlapInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INPUT_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INPUT_INDEX);
    opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACT_SEQ_LEN_Q_INPUT_INDEX);
    opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACT_SEQ_LEN_Q_INPUT_INDEX);
    opParamInfo_.actualSeqLengths.tensor = context_->GetOptionalInputTensor(ACT_SEQ_LEN_KV_INPUT_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetOptionalInputDesc(ACT_SEQ_LEN_KV_INPUT_INDEX);
    opParamInfo_.queryRope.tensor = context_->GetOptionalInputTensor(QUERY_ROPE_INPUT_INDEX);
    opParamInfo_.queryRope.desc = context_->GetOptionalInputDesc(QUERY_ROPE_INPUT_INDEX);
    opParamInfo_.keyRope.tensor = context_->GetOptionalInputTensor(KEY_ROPE_INPUT_INDEX);
    opParamInfo_.keyRope.desc = context_->GetOptionalInputDesc(KEY_ROPE_INPUT_INDEX);
}

void FusedSparseAttentionOverlapInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INPUT_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INPUT_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INPUT_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INPUT_INDEX);
    opParamInfo_.value.desc = context_->GetInputDesc(VALUE_INPUT_INDEX);
    opParamInfo_.value.shape = context_->GetInputShape(VALUE_INPUT_INDEX);
    opParamInfo_.sparseIndices.desc = context_->GetInputDesc(SPARSE_INDICES_INPUT_INDEX);
    opParamInfo_.sparseIndices.shape = context_->GetInputShape(SPARSE_INDICES_INPUT_INDEX);
    GetOptionalInputParaInfo();
}

void FusedSparseAttentionOverlapInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(OUTPUT_INDEX);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(OUTPUT_INDEX);
    if (!IsFusedSparseAttentionOverlapStandaloneOp(opName_)) {
        opParamInfo_.softmaxMax.desc = context_->GetOutputDesc(SOFTMAXMAX_INDEX);
        opParamInfo_.softmaxMax.shape = context_->GetOutputShape(SOFTMAXMAX_INDEX);
        opParamInfo_.softmaxSum.desc = context_->GetOutputDesc(SOFTMAXSUM_INDEX);
        opParamInfo_.softmaxSum.shape = context_->GetOutputShape(SOFTMAXSUM_INDEX);
    }
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);

    opParamInfo_.layoutQuery = attrs->GetStr(LAYOUT_QUERY_ATTR_INDEX);
    opParamInfo_.layoutKV = attrs->GetStr(LAYOUT_KV_ATTR_INDEX);
    opParamInfo_.sparseBlockSize = attrs->GetAttrPointer<int64_t>(SPARSE_BLOCK_SIZE_ATTR_INDEX);
    opParamInfo_.scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    opParamInfo_.sparseMode = attrs->GetAttrPointer<int64_t>(SPARSE_MODE_ATTR_INDEX);
    opParamInfo_.preTokens = attrs->GetAttrPointer<int64_t>(PRE_TOKENS_ATTR_INDEX);
    opParamInfo_.nextTokens = attrs->GetAttrPointer<int64_t>(NEXT_TOKENS_ATTR_INDEX);
    opParamInfo_.attentionMode = attrs->GetAttrPointer<int64_t>(ATTENTION_MODE_ATTR_INDEX);
    opParamInfo_.returnSoftmaxLse = attrs->GetAttrPointer<bool>(RETURN_SOFTMAX_LSE_ATTR_INDEX);
    static const int64_t defaultPreTokens = INT64_MAX;
    static const int64_t defaultNextTokens = INT64_MAX;
    static const int64_t defaultAttentionMode = 2;
    static const bool defaultReturnSoftmaxLse = false;
    if (IsFusedSparseAttentionOverlapStandaloneOp(opName_)) {
        if (opParamInfo_.preTokens == nullptr) {
            opParamInfo_.preTokens = &defaultPreTokens;
        }
        if (opParamInfo_.nextTokens == nullptr) {
            opParamInfo_.nextTokens = &defaultNextTokens;
        }
        if (opParamInfo_.attentionMode == nullptr) {
            opParamInfo_.attentionMode = &defaultAttentionMode;
        }
        if (opParamInfo_.returnSoftmaxLse == nullptr) {
            opParamInfo_.returnSoftmaxLse = &defaultReturnSoftmaxLse;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetInOutDataType()
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

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    if (qLayout_ == FusedSparseAttentionOverlapLayout::TND) {
        return GetActualSeqLenQSize(bSize_);
    } else { // BSND
        bSize_ = GetAxisNum(queryShape_, FusedSparseAttentionOverlapAxis::B, qLayout_);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetQTSize()
{
    // 获取query的T基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    qTSize_ = (qLayout_ == FusedSparseAttentionOverlapLayout::TND) ? GetAxisNum(queryShape_, FusedSparseAttentionOverlapAxis::T, qLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetKVTSize()
{
    // 获取query的T基准值
    // 1、非TND时, 以key的batch_size维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组的长度为B轴大小
    kvTSize_ = (kvLayout_ == FusedSparseAttentionOverlapLayout::TND) ? GetAxisNum(keyShape_, FusedSparseAttentionOverlapAxis::T, kvLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetQkHeadDim()
{
    // 获取qkHeadDim基准值
    // 以query的D维度为基准
    qkHeadDim_ = GetAxisNum(queryShape_, FusedSparseAttentionOverlapAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetS1Size()
{
    // 获取S1基准值
    // 1、非TND时, 以query的S维度为基准;
    // 2、TND时, actual_seq_lens_q必须传入, 以actual_seq_lens_q数组中的最大值为基准
    if (qLayout_ == FusedSparseAttentionOverlapLayout::TND) {
        s1Size_ = GetAxisNum(queryShape_, FusedSparseAttentionOverlapAxis::T, qLayout_);
        return ge::GRAPH_SUCCESS;
    } else { // BSND
        s1Size_ = GetAxisNum(queryShape_, FusedSparseAttentionOverlapAxis::S, qLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetKvStorageMode()
{
    if (kvLayout_ == FusedSparseAttentionOverlapLayout::PA_BSND) {
        kvStorageMode_ = FusedOverlapKvStorageMode::PAGE_ATTENTION;
    } else {
        kvStorageMode_ = FusedOverlapKvStorageMode::BATCH_CONTINUOUS;
    }
    // kv存储模式基准值
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetKvLayout()
{
    const map<string, FusedSparseAttentionOverlapLayout> layoutKVMap = {
        {"BSND",        FusedSparseAttentionOverlapLayout::BSND},
        {"PA_BSND",     FusedSparseAttentionOverlapLayout::PA_BSND},
        {"TND",         FusedSparseAttentionOverlapLayout::TND}
    };

    std::string layout(opParamInfo_.layoutKV);
    auto it = layoutKVMap.find(layout);
    if (it != layoutKVMap.end()) {
        kvLayout_ = it->second;
    } else {
        OP_LOGE(opName_, "layoutKV is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    if (kvLayout_ != FusedSparseAttentionOverlapLayout::PA_BSND && qLayout_ != kvLayout_) {
        OP_LOGE(opName_, "When layoutKV is not PA_BSND, layoutKV must be the same as layoutQ.");
        return ge::GRAPH_FAILED;
    }
    uint32_t keyDimNum = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    if (kvLayout_ == FusedSparseAttentionOverlapLayout::PA_BSND && keyDimNum != 4U) {
        OP_LOGE(opName_, "When layoutKV is PA_BSND, kvDimNum must be 4, but now is %d.", keyDimNum);
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(*opParamInfo_.returnSoftmaxLse && kvLayout_ == FusedSparseAttentionOverlapLayout::PA_BSND,
                OP_LOGE(opName_, "when return_softmax_lse is true, key layout do not support PA_BSND."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetS2SizeForBatchContinuous()
{
    if (kvLayout_ == FusedSparseAttentionOverlapLayout::BSND) { // BSND
        s2Size_ = GetAxisNum(keyShape_, FusedSparseAttentionOverlapAxis::S, kvLayout_);
    } else if (kvLayout_ == FusedSparseAttentionOverlapLayout::TND) {
        s2Size_ = GetAxisNum(keyShape_, FusedSparseAttentionOverlapAxis::T, kvLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetMaxBlockNumPerBatch()
{
    if (opParamInfo_.blockTable.tensor == nullptr) {
        OP_LOGE(opName_, "the layout_kv is %s, blockTable must be provided.", FusedSparseAttentionOverlapLayoutToSerialString(kvLayout_).c_str());
        return ge::GRAPH_FAILED;
    }
    uint32_t dimNum = opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum();
    if (dimNum != DIM_NUM_TWO) {
        OP_LOGE(opName_, "the dim num of block_table is %u, it should be %u.", dimNum, DIM_NUM_TWO);
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1) <= 0) {
        OP_LOGE(opName_, "%s's second dimension(%ld) should be greater than 0",
            BLOCK_TABLE_NAME.c_str(), opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1));
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetBlockSize()
{
    blockSize_ = GetAxisNum(keyShape_, FusedSparseAttentionOverlapAxis::Bs, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetSparseBlockCount()
{
    sparseBlockCount_ = GetAxisNum(sparseIndicesShape_, FusedSparseAttentionOverlapAxis::K, qLayout_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetS2SizeForPageAttention()
{
    if (GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS || GetBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    s2Size_ = maxBlockNumPerBatch_ * blockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetS2Size()
{
    // 获取S2基准值
    // 1、BATCH_CONTINUOUS时, 从key的S轴获取
    // 2、PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    if (kvStorageMode_ == FusedOverlapKvStorageMode::BATCH_CONTINUOUS) {
        return GetS2SizeForBatchContinuous();
    }
    return GetS2SizeForPageAttention();
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetValueHeadDim()
{
    // 获取vHeadDim基准值
    // 以value的D维度为基准
    vHeadDim_ = GetAxisNum(valueShape_, FusedSparseAttentionOverlapAxis::D, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetRopeHeadDim()
{
    if (queryShape_.GetDimNum() != queryRopeShape_.GetDimNum()) {
        OP_LOGE(opName_, "The dimensions of query and query_rope should be equal, but query has dimension %zu while query_rope has dimension %zu.",
                queryShape_.GetDimNum(), queryRopeShape_.GetDimNum());
        return ge::GRAPH_PARAM_INVALID;
    }
    ropeHeadDim_ = GetAxisNum(queryRopeShape_, FusedSparseAttentionOverlapAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetQueryAndOutLayout()
{
    // 获取query和attentionOut的Layout基准值
    // layoutQuery: {qLayout, outLayout}
    const map<string, pair<FusedSparseAttentionOverlapLayout, FusedSparseAttentionOverlapLayout>> layoutMap = {
        {"BSND",        {FusedSparseAttentionOverlapLayout::BSND,    FusedSparseAttentionOverlapLayout::BSND}},
        {"TND",         {FusedSparseAttentionOverlapLayout::TND,     FusedSparseAttentionOverlapLayout::TND }},
    };

    std::string layout(opParamInfo_.layoutQuery);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second.first;
        outLayout_ = it->second.second;
    } else {
        OP_LOGE(opName_, "layoutQuery is %s, it is unsupported.", layout.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetTopkLayout()
{
    topkLayout_ = qLayout_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetSoftmaxMaxAndSumLayout()
{
    if (qLayout_ == FusedSparseAttentionOverlapLayout::BSND) {
        softmaxMaxLayout_ = FusedSparseAttentionOverlapLayout::BNSG;
        softmaxSumLayout_ = FusedSparseAttentionOverlapLayout::BNSG;
    } else if (qLayout_ == FusedSparseAttentionOverlapLayout::TND) {
        softmaxMaxLayout_ = FusedSparseAttentionOverlapLayout::NTG;
        softmaxSumLayout_ = FusedSparseAttentionOverlapLayout::NTG;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetN1Size()
{
    n1Size_ = GetAxisNum(queryShape_, FusedSparseAttentionOverlapAxis::N, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetN2Size()
{
    n2Size_ = GetAxisNum(keyShape_, FusedSparseAttentionOverlapAxis::N, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

void FusedSparseAttentionOverlapInfoParser::SetSFAShape()
{
    queryShape_ = opParamInfo_.query.shape->GetStorageShape();
    keyShape_ = opParamInfo_.key.shape->GetStorageShape();
    valueShape_ = opParamInfo_.value.shape->GetStorageShape();
    sparseIndicesShape_ = opParamInfo_.sparseIndices.shape->GetStorageShape();
    queryRopeShape_ = opParamInfo_.queryRope.tensor->GetStorageShape();
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetGSize()
{
    if (n2Size_ != 0) {
        gSize_ = n1Size_ / n2Size_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::GetActualseqInfo()
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

void FusedSparseAttentionOverlapInfoParser::GenerateInfo(FusedSparseAttentionOverlapTilingInfo &sfaInfo)
{
    sfaInfo.opName = opName_;
    sfaInfo.platformInfo = platformInfo_;
    sfaInfo.opParamInfo = opParamInfo_;
    sfaInfo.npuArch = npuArch_;
    sfaInfo.isA5 = isA5_;

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
    sfaInfo.pageAttentionFlag = (kvStorageMode_ == FusedOverlapKvStorageMode::PAGE_ATTENTION);
    sfaInfo.blockSize = blockSize_;
    sfaInfo.blockTypeSize =  sizeof(float);
    sfaInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    sfaInfo.actualLenDimsQ = actualLenDimsQ_;
    sfaInfo.actualLenDimsKV = actualLenDimsKV_;
    sfaInfo.maxActualseq = maxActualseq_;
    sfaInfo.isSameSeqAllKVTensor = isSameSeqAllKVTensor_;
    sfaInfo.isSameActualseq = isSameActualseq_;

    sfaInfo.actualQSeqLenFlag = (opParamInfo_.actualSeqLengthsQ.tensor != nullptr);
    sfaInfo.actualSeqLenFlag = (opParamInfo_.actualSeqLengths.tensor != nullptr);

    sfaInfo.sparseMode = *opParamInfo_.sparseMode;
    sfaInfo.preTokens = *opParamInfo_.preTokens;
    sfaInfo.nextTokens = *opParamInfo_.nextTokens;
    sfaInfo.attentionMode = *opParamInfo_.attentionMode;
    sfaInfo.returnSoftmaxLse = *opParamInfo_.returnSoftmaxLse;

    sfaInfo.qLayout = qLayout_;
    sfaInfo.topkLayout = topkLayout_;
    sfaInfo.kvLayout = kvLayout_;
    sfaInfo.outLayout = outLayout_;
    sfaInfo.softmaxMaxLayout = softmaxMaxLayout_;
    sfaInfo.softmaxSumLayout = softmaxSumLayout_;
}

ge::graphStatus FusedSparseAttentionOverlapInfoParser::Parse(FusedSparseAttentionOverlapTilingInfo &sfaInfo)
{
    if (context_ == nullptr) {
        OP_LOGE("FusedSparseAttentionOverlap", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetOpName() ||
        ge::GRAPH_SUCCESS != GetNpuInfo() ||
        ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetInOutDataType() ||
        ge::GRAPH_SUCCESS != GetQueryAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetTopkLayout() ||
        ge::GRAPH_SUCCESS != GetSoftmaxMaxAndSumLayout() ||
        ge::GRAPH_SUCCESS != GetKvLayout() ||
        ge::GRAPH_SUCCESS != GetKvStorageMode()) {
        return ge::GRAPH_FAILED;
    }

    SetSFAShape();
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

IMPL_OP_OPTILING(FusedSparseAttentionOverlap)
    .Tiling(TilingFusedSparseAttentionOverlap)
    .TilingParse<FusedSparseAttentionOverlapCompileInfo>(TilingPrepareForFusedSparseAttentionOverlap);
} // namespace optiling
