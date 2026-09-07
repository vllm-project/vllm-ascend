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
 * \file turbo_quant_sparse_flash_attention_tiling.cpp
 * \brief
 */

#include <map>
#include <vector>
#include <algorithm>
#include <sstream>
#include <graph/utils/type_utils.h>
#include "err/ops_err.h"
#include "register/op_def_registry.h"
#include "../op_kernel/turbo_quant_sparse_flash_attention_template_tiling_key.h"
#include "turbo_quant_sparse_flash_attention_tiling.h"

using std::map;
using std::string;
using std::pair;

using namespace ge;
using namespace AscendC;
namespace optiling {

inline std::string TQSFAErrorToString(const char *value)
{
    return value == nullptr ? std::string() : std::string(value);
}

inline std::string TQSFAErrorToString(char *value)
{
    return TQSFAErrorToString(static_cast<const char *>(value));
}

inline std::string TQSFAErrorToString(const std::string &value)
{
    return value;
}

template <typename T>
std::string TQSFAErrorToString(const T &value)
{
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

#define TQSFA_LOG_INVALID_WITH_EXPECTED(opname, kind, param, actual, expected)                 \
    do {                                                                                       \
        const auto tqsfaParam = ::optiling::TQSFAErrorToString(param);                          \
        const auto tqsfaActual = ::optiling::TQSFAErrorToString(actual);                        \
        const auto tqsfaExpected = ::optiling::TQSFAErrorToString(expected);                    \
        OP_LOGE(opname, "Invalid %s for %s, actual: %s, expected: %s.", kind,                  \
                tqsfaParam.c_str(), tqsfaActual.c_str(), tqsfaExpected.c_str());                \
    } while (0)

#define TQSFA_LOG_INVALID_WITH_REASON(opname, kind, param, actual, reason)                     \
    do {                                                                                       \
        const auto tqsfaParam = ::optiling::TQSFAErrorToString(param);                          \
        const auto tqsfaActual = ::optiling::TQSFAErrorToString(actual);                        \
        const auto tqsfaReason = ::optiling::TQSFAErrorToString(reason);                        \
        OP_LOGE(opname, "Invalid %s for %s, actual: %s, reason: %s.", kind,                    \
                tqsfaParam.c_str(), tqsfaActual.c_str(), tqsfaReason.c_str());                  \
    } while (0)

#ifndef OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON
#define OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opname, param, actual, reason)                   \
    TQSFA_LOG_INVALID_WITH_REASON(opname, "dtype", param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opname, param, actual, reason)                   \
    TQSFA_LOG_INVALID_WITH_REASON(opname, "shape", param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opname, param, actual, reason)                \
    TQSFA_LOG_INVALID_WITH_REASON(opname, "shape dim", param, actual, reason)
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUE
#define OP_LOGE_FOR_INVALID_VALUE(opname, param, actual, expected)                             \
    TQSFA_LOG_INVALID_WITH_EXPECTED(opname, "value", param, actual, expected)
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
#define OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opname, param, actual, reason)                   \
    TQSFA_LOG_INVALID_WITH_REASON(opname, "value", param, actual, reason)
#endif

constexpr uint32_t PRE_LOAD_NUM = 2;
constexpr uint32_t BLOCK_TABLE_ELEM_BYTE = 4;

static const std::string QUERY_NAME = "query";
static const std::string KEY_NAME = "key";
static const std::string VALUE_NAME = "value";
static const std::string SPARSE_INDICES_NAME = "sparse_indices";
static const std::string ATTEN_OUT_NAME = "attention_out";

const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {{QUERY_NAME, {ge::DT_BF16}},
                                                                            {KEY_NAME, {ge::DT_INT8}},
                                                                            {VALUE_NAME, {ge::DT_INT8}},
                                                                            {ATTEN_OUT_NAME, {ge::DT_BF16}},
                                                                            {SPARSE_INDICES_NAME, {ge::DT_INT32}}};

// query 与 attention_out 仅支持 TND：PyTorch launcher 强制 query 为 3D，
// 且输出固定按 TND 构造，BSND 无对应的 launcher / Meta / 输出 shape 实现。
// KV 仅支持 PA_BSND：GetKvLayout() 要求非 PA_BSND 的 KV layout 必须与 query 相同，
// 而 query 已限定为 TND，故 BSND 的 KV 不可达，TND 的 KV 亦无对应实现。
const std::map<std::string, std::vector<TQSFALayout>> LAYOUT_SUPPORT_MAP = {
    {QUERY_NAME, {TQSFALayout::TND}},
    {KEY_NAME, {TQSFALayout::PA_BSND}},
    {VALUE_NAME, {TQSFALayout::PA_BSND}},
    {ATTEN_OUT_NAME, {TQSFALayout::TND}},
};

const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {ge::DT_FLOAT, "DT_FLOAT"},                   // float type
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},           // Used to indicate a DataType field has not been set.
    {ge::DT_FLOAT16, "DT_FLOAT16"},               // fp16 type
    {ge::DT_INT8, "DT_INT8"},                     // int8 type
    {ge::DT_INT16, "DT_INT16"},                   // int16 type
    {ge::DT_UINT16, "DT_UINT16"},                 // uint16 type
    {ge::DT_UINT8, "DT_UINT8"},                   // uint8 type
    {ge::DT_INT64, "DT_INT64"},                   // int64 type
    {ge::DT_INT32, "DT_INT32"},                   // int32 type
    {ge::DT_UINT64, "DT_UINT64"},                 // unsigned int64
    {ge::DT_UINT32, "DT_UINT32"},                 // unsigned int32
    {ge::DT_BOOL, "DT_BOOL"},                     // bool type
    {ge::DT_DOUBLE, "DT_DOUBLE"},                 // double type
    {ge::DT_DUAL, "DT_DUAL"},                     // dual output type
    {ge::DT_COMPLEX32, "DT_COMPLEX32"},           // complex32 type
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},           // complex64 type
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},         // complex128 type
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},   // dual output int8 type
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"}, // dual output uint8 type
    {ge::DT_QUINT8, "DT_QUINT8"},                 // quint8 type
    {ge::DT_QUINT16, "DT_QUINT16"},               // quint16 type
    {ge::DT_QINT8, "DT_QINT8"},                   // qint8 type
    {ge::DT_QINT16, "DT_QINT16"},                 // qint16 type
    {ge::DT_QINT32, "DT_QINT32"},                 // qint32 type
    {ge::DT_RESOURCE, "DT_RESOURCE"},             // resource type
    {ge::DT_STRING_REF, "DT_STRING_REF"},         // string ref type
    {ge::DT_BF16, "DT_BFLOAT16"},                 // dt_bfloat16 type
    {ge::DT_STRING, "DT_STRING"},                 // string type
    {ge::DT_VARIANT, "DT_VARIANT"},               // dt_variant type
    {ge::DT_INT2, "DT_INT2"},                     // dt_variant type
    {ge::DT_UINT2, "DT_UINT2"},                   // dt_variant type
    {ge::DT_INT4, "DT_INT4"},                     // dt_variant type
    {ge::DT_UINT1, "DT_UINT1"}                    // dt_variant type
};

struct TurboQuantSparseFlashAttentionCompileInfo {
    int64_t coreNum;
};

static const std::map<TQSFALayout, std::vector<TQSFAAxis>> TQSFA_LAYOUT_AXIS_MAP = {
    {TQSFALayout::BSND, {TQSFAAxis::B, TQSFAAxis::S, TQSFAAxis::N, TQSFAAxis::D}},
    {TQSFALayout::TND, {TQSFAAxis::T, TQSFAAxis::N, TQSFAAxis::D}},
    {TQSFALayout::PA_BSND, {TQSFAAxis::Bn, TQSFAAxis::Bs, TQSFAAxis::N, TQSFAAxis::D}},
};

static const std::map<TQSFALayout, size_t> TQSFA_LAYOUT_DIM_MAP = {
    {TQSFALayout::BSND, DIM_NUM_FOUR},
    {TQSFALayout::TND, DIM_NUM_THREE},
    {TQSFALayout::PA_BSND, DIM_NUM_FOUR},
};

static std::string TQSFADataTypeToSerialString(ge::DataType type)
{
    const auto qsfaIt = DATATYPE_TO_STRING_MAP.find(type);
    if (qsfaIt != DATATYPE_TO_STRING_MAP.end()) {
        return qsfaIt->second;
    } else {
        OP_LOGE("SparseFlashAttention", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

std::string TQSFALayoutToSerialString(TQSFALayout layout)
{
    switch (layout) {
        case TQSFALayout::BSND:
            return "BSND";
        case TQSFALayout::TND:
            return "TND";
        case TQSFALayout::PA_BSND:
            return "PA_BSND";
        default:
            return "UNKNOWN";
    }
}

ge::graphStatus TQSFAMlaTiling::SetBlockDim(uint32_t blockDim) const
{
    context_->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAMlaTiling::SetTilingKey(uint64_t tilingKey) const
{
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1); // 1: batchmode模式
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAMlaTiling::SetWorkspaceSize(uint64_t workspaceSize) const
{
    OP_CHECK_IF(context_->GetWorkspaceSizes(1) == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "workSpaceSize got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAMlaTiling::SetTilingData(TilingDef &tilingData) const
{
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
                return ge::GRAPH_FAILED);

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAMlaTiling::GetPlatformInfo()
{
    OP_CHECK_IF(qsfaInfo_->platformInfo == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(qsfaInfo_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto qsfaAscendcPlatform = platform_ascendc::PlatformAscendC(qsfaInfo_->platformInfo);
    libapiSize_ = qsfaAscendcPlatform.GetLibApiWorkSpaceSize();
    aivNum_ = qsfaAscendcPlatform.GetCoreNumAiv();
    aicNum_ = qsfaAscendcPlatform.GetCoreNumAic();

    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0,
                OPS_REPORT_VECTOR_INNER_ERR(qsfaInfo_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void TQSFAMlaTiling::GenTilingKey()
{
    uint32_t layoutQuery = static_cast<uint32_t>(qsfaInfo_->qLayout);
    uint32_t layoutKV = static_cast<uint32_t>(qsfaInfo_->kvLayout);
    uint32_t pageAttention = 0U;
    if (qsfaInfo_->kvLayout == TQSFALayout::PA_BSND) {
        pageAttention = 1U;
    }

    tilingKey_ =
        GET_TPL_TILING_KEY(0U, pageAttention, layoutQuery, layoutKV, perfMode_ == TQSFAPerfMode::V_TEMPLATE_MODE,
                           static_cast<uint32_t>(qsfaInfo_->gSize > 64)); // G大于64时核间切G
}

void TQSFAMlaTiling::InitParams()
{
    perfMode_ = TQSFAPerfMode::V_TEMPLATE_MODE;
    coreNum_ = aicNum_;

    headDimAlign_ = Align(qsfaInfo_->qHeadDim, BYTE_BLOCK); // 元素个数按照基本块大小对齐
}

void TQSFAMlaTiling::CalcUbBmm()
{
    uint32_t qsfaCubeMSize = qsfaInfo_->gSize * qsfaInfo_->s1Size;
    uint32_t qsfaMaxMSize = mBaseSize_;
    if (qsfaCubeMSize > qsfaMaxMSize) {
        qsfaCubeMSize = qsfaMaxMSize;
    }
    mmResUbSize_ = sInnerSizeAlign_ * Align(qsfaCubeMSize, 16U); // kernel按照16对齐写出，tiling按照这个原则分配内存
    bmm2ResUbSize_ = headDimAlign_ * Align(qsfaCubeMSize, 16U); // kernel按照16对齐写出，tiling按照这个原则分配内存

    qPreSizeMla_ = qsfaInfo_->gSize * (headDimAlign_ + qsfaInfo_->ropeHeadDim) * qsfaInfo_->s1Size;
}

void TQSFAMlaTiling::CheckUbSpace() { CalcUbBmm(); }

void TQSFAMlaTiling::CalcInnerSize(uint32_t qsfaS2Size)
{
    sInnerSize_ = 512; // 512:s2默认切分大小
    // FlashDecode时，如果S2的计算量>=256(确保切分后不小于128)但又不足以分2次计算时，则修改sInnerSize_，均分为2份进行计算，确保Nbuffer=2
    if (splitKVFlag_ && qsfaInfo_->qLayout != TQSFALayout::TND) {
        if (qsfaS2Size == 256) { // 256:s2Size的阈值，判断sInnerSize_是否切分
            sInnerSize_ = 128;   // 128:sInnerSize_值为s2Size的一半，均分两份进行计算，
        } else if (qsfaS2Size > 256 && qsfaS2Size <= sInnerSize_) { // 256:s2Size的阈值，判断sInnerSize_是否切分
            sInnerSize_ = (sInnerSize_ + 1) / 2;                    // 2:减半
        }
    }

    sInnerLoopTimes_ = (qsfaS2Size + sInnerSize_ - 1) / sInnerSize_;
    if (sInnerSize_ > qsfaS2Size) {
        sInnerSize_ = qsfaS2Size;
    }
    sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    CheckUbSpace();
}

void TQSFAMlaTiling::SplitBalanced()
{
    CalcInnerSize(qsfaInfo_->s2Size);
    InnerSplitParams qsfaInnerSplitParams;
    qsfaInnerSplitParams.s1GBaseSize = qsfaInfo_->gSize;
    tilingData_.innerSplitParams.set_mBaseSize(qsfaInnerSplitParams.s1GBaseSize);

    qsfaInnerSplitParams.s2BaseSize = sInnerSize_;
    tilingData_.innerSplitParams.set_s2BaseSize(qsfaInnerSplitParams.s2BaseSize);

    usedCoreNum_ = aicNum_;
}

void TQSFAMlaTiling::Split() { SplitBalanced(); }

void TQSFAMlaTiling::FillTilingBaseParamsMla()
{
    tilingData_.baseParams.set_batchSize(qsfaInfo_->bSize);
    tilingData_.baseParams.set_seqSize(qsfaInfo_->s2Size);
    tilingData_.baseParams.set_qSeqSize(qsfaInfo_->s1Size);
    tilingData_.baseParams.set_blockSize(qsfaInfo_->blockSize);
    tilingData_.baseParams.set_maxBlockNumPerBatch(qsfaInfo_->maxBlockNumPerBatch);
    tilingData_.baseParams.set_scaleValue(qsfaInfo_->scaleValue);
    tilingData_.baseParams.set_nNumOfQInOneGroup(qsfaInfo_->n1Size / qsfaInfo_->n2Size);
    tilingData_.baseParams.set_actualLenDimsQ(qsfaInfo_->actualLenDimsQ);
    tilingData_.baseParams.set_actualLenDimsKV(qsfaInfo_->actualLenDimsKV);
    tilingData_.baseParams.set_outputLayout(static_cast<uint32_t>(qsfaInfo_->outLayout));
    tilingData_.baseParams.set_sparseMode(qsfaInfo_->sparseMode);
    tilingData_.baseParams.set_sparseBlockSize(qsfaInfo_->sparseBlockSize);
    tilingData_.baseParams.set_sparseBlockCount(qsfaInfo_->sparseBlockCount);
    tilingData_.baseParams.set_dSizeVInput(qsfaInfo_->dSizeVInput);
    tilingData_.baseParams.set_headDim(qsfaInfo_->qHeadDim - qsfaInfo_->ropeHeadDim);
    tilingData_.baseParams.set_ropeHeadDim(qsfaInfo_->ropeHeadDim);
    tilingData_.baseParams.set_keyQuantMode(qsfaInfo_->keyQuantMode);
    tilingData_.baseParams.set_valueQuantMode(qsfaInfo_->valueQuantMode);
    tilingData_.baseParams.set_tileSize(qsfaInfo_->tileSize);
    tilingData_.baseParams.set_isActualLenDimsNull(qsfaInfo_->actualQSeqLenFlag ? 0U : 1U);
    tilingData_.baseParams.set_isActualLenDimsKVNull(qsfaInfo_->actualSeqLenFlag ? 0U : 1U);
    tilingData_.baseParams.set_returnSoftmaxLse(qsfaInfo_->returnSoftmaxLse ? 1U : 0U);
}

// for flash decode
void TQSFAMlaTiling::FillTilingSplitKVMla()
{
    tilingData_.splitKVParams.set_s2(kvSplitPart_);
    // 2:每个核可能有头规约和尾规约，一共两份规约信息
    tilingData_.splitKVParams.set_accumOutSize(aicNum_ * 2 * qsfaInfo_->n2Size * mBaseSize_ * headDimAlign_);
    // 2:每个核可能有头规约和尾规约，一共两份规约信息：sum + max
    tilingData_.splitKVParams.set_logSumExpSize(2 * aicNum_ * 2 * qsfaInfo_->n2Size * mBaseSize_ *
                                                (BYTE_BLOCK / BLOCK_TABLE_ELEM_BYTE));

    if (!splitKVFlag_) {
        tilingData_.splitKVParams.set_s2(0);
    }
}

void TQSFAMlaTiling::FillTilingSingleCoreParamsMla() { tilingData_.singleCoreParams.set_usedCoreNum(usedCoreNum_); }

void TQSFAMlaTiling::FillTilingSingleCoreTensorSizeMla()
{
    tilingData_.singleCoreTensorSize.set_mmResUbSize(mmResUbSize_);
    tilingData_.singleCoreTensorSize.set_bmm2ResUbSize(bmm2ResUbSize_);
}

void TQSFAMlaTiling::FillTiling()
{
    FillTilingBaseParamsMla();
    FillTilingSplitKVMla();
    FillTilingSingleCoreParamsMla();
    FillTilingSingleCoreTensorSizeMla();
}

uint32_t TQSFAMlaTiling::CalcBalanceFDParamNums(const uint32_t actCoreNum) const
{
    return actCoreNum * 2 * qsfaInfo_->n2Size * mBaseSize_; // 2:每个核可能有头规约和尾规约，一共两份规约信息
}

void TQSFAMlaTiling::NormalCalcFDWorkSpace(const uint32_t actCoreNum)
{
    if (splitKVFlag_) {
        uint32_t accumOutSize = 0;
        uint32_t logSumExpSize = 0;
        uint32_t FDParamNums = CalcBalanceFDParamNums(actCoreNum);
        accumOutSize = FDParamNums * headDimAlign_;
        logSumExpSize =
            2 * FDParamNums * (BYTE_BLOCK / qsfaInfo_->blockTypeSize); // log和sum的存储空间一致，共需两份内存
        workspaceSize_ += (accumOutSize + logSumExpSize) * qsfaInfo_->blockTypeSize;
    }
}

void TQSFAMlaTiling::CalcFDWorkSpace(const uint32_t actCoreNum) { NormalCalcFDWorkSpace(actCoreNum); }

void TQSFAMlaTiling::GetWorkspaceSize()
{
    uint32_t actCoreNum = coreNum_;
    uint32_t mmResElemSize = 4;   // 4:fp32
    uint32_t vec1ResElemSize = 2; // 2:fp16/bf16
    uint32_t bmm2ResElemSize = 4; // 4:fp32
    uint32_t qPreProcResElemSize = 0;
    uint32_t softmaxSumElemSize = 4; // 4:int32
    float kvDtypeRatio = 1.0;

    workspaceSize_ = libapiSize_;
    uint32_t preLoadNum = PRE_LOAD_NUM;

    workspaceSize_ += preLoadNum * (mmResUbSize_ * actCoreNum * mmResElemSize);
    workspaceSize_ += preLoadNum * static_cast<size_t>(static_cast<float>(mmResUbSize_ * actCoreNum * vec1ResElemSize) *
                                                       kvDtypeRatio);
    workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * bmm2ResElemSize;
    workspaceSize_ +=
        preLoadNum *
        static_cast<size_t>(static_cast<float>(qPreSizeMla_ * actCoreNum * qPreProcResElemSize) * kvDtypeRatio);
    workspaceSize_ += preLoadNum * mBaseSize_ * actCoreNum * softmaxSumElemSize;
    workspaceSize_ += preLoadNum * bmm2ResUbSize_ * actCoreNum * bmm2ResElemSize; // vec2ResGm
    workspaceSize_ += 4 * 512 * qsfaInfo_->qHeadDim * NUM_BYTES_FLOAT16 * actCoreNum;
    // kvValidSize 区：每核 4槽 × 2个AIV × 128 int32 = 1024 int32。
    // [TQ4] 再扩一倍，上半区用作 per-column scale 传递（4槽 × s2BaseSize(512) half = 1024 int32），
    // 复用 kvValidSize 已验证的“各 AIV 按 GetSubBlockIdx 分区写、vec1 整体读回”通路，
    // 不新增 GlobalTensor、不改函数签名。
    workspaceSize_ += 4 * 128 * 4 * (4 * actCoreNum);

    CalcFDWorkSpace(actCoreNum);
}

void TQSFAMlaTiling::CalcBlockDim()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(qsfaInfo_->platformInfo);
    auto aicNum = usedCoreNum_;
    auto aivNum = 2 * usedCoreNum_;

    blockDim_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
}

ge::graphStatus TQSFAMlaTiling::DoOpTiling(TQSFATilingInfo *qsfaInfo)
{
    qsfaInfo_ = qsfaInfo;
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    InitParams();
    Split();
    FillTiling();
    CalcBlockDim();
    GetWorkspaceSize();
    GenTilingKey();

    if ((SetBlockDim(blockDim_) != ge::GRAPH_SUCCESS) || (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) || (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingTurboQuantSparseFlashAttention(gert::TilingContext *context)
{
    TQSFATilingInfo qsfaInfo;
    TQSFAInfoParser qsfaInfoParser(context);
    if (qsfaInfoParser.Parse(qsfaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    TQSFATilingCheck tilingChecker(qsfaInfo);
    if (tilingChecker.Process() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    TQSFAMlaTiling tiling(context);
    return tiling.DoOpTiling(&qsfaInfo);
}

ge::graphStatus TilingPrepareForTurboQuantSparseFlashAttention(gert::TilingParseContext *const context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::GetExpectedShape(gert::Shape &shapeExpected,
                                                   const TQSFATilingShapeCompareParam &param,
                                                   const TQSFALayout &layout) const
{
    if (layout == TQSFALayout::BSND) {
        shapeExpected = gert::Shape({param.B, param.S, param.N, param.D});
    } else if (layout == TQSFALayout::TND) {
        shapeExpected = gert::Shape({param.T, param.N, param.D});
    } else if (layout == TQSFALayout::PA_BSND) {
        shapeExpected = gert::Shape({param.Bn, param.Bs, param.N, param.D});
    } else {
        OP_LOGE(opName_, "layout %s is unsupported", TQSFALayoutToSerialString(layout).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CompareShape(TQSFATilingShapeCompareParam &param, const gert::Shape &shape,
                                               const TQSFALayout &layout, const std::string &name) const
{
    gert::Shape qsfaShapeExpected;
    if (GetExpectedShape(qsfaShapeExpected, param, layout) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (shape.GetDimNum() != qsfaShapeExpected.GetDimNum()) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            opName_, name.c_str(), std::to_string(shape.GetDimNum()).c_str(),
            "The shape dim of " + name + " should be " + std::to_string(qsfaShapeExpected.GetDimNum()));
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) != qsfaShapeExpected.GetDim(i)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, name.c_str(), std::to_string(shape.GetDim(i)).c_str(),
                                                  "Dim " + std::to_string(i) + " of " + name + " should be " +
                                                      std::to_string(qsfaShapeExpected.GetDim(i)));
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

void TQSFATilingCheck::LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
                                            const ge::DataType &actualDtype, const std::string &name) const
{
    std::ostringstream qsfaOss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        qsfaOss << TQSFADataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            qsfaOss << ", ";
        }
    }
    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, name.c_str(), TQSFADataTypeToSerialString(actualDtype).c_str(),
                                          "The dtype of " + name + " must be " + qsfaOss.str());
}

ge::graphStatus TQSFATilingCheck::CheckDtypeSupport(const gert::CompileTimeTensorDesc *qsfaDesc,
                                                    const std::string &name) const
{
    if (qsfaDesc != nullptr) {
        const auto &qsfaIt = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(qsfaIt == DTYPE_SUPPORT_MAP.end(),
                    OP_LOGE(opName_, "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
                    return ge::GRAPH_FAILED);
        auto &qsfaExpectDtypeList = qsfaIt->second;
        OP_CHECK_IF(std::find(qsfaExpectDtypeList.begin(), qsfaExpectDtypeList.end(), qsfaDesc->GetDataType()) ==
                        qsfaExpectDtypeList.end(),
                    LogErrorDtypeSupport(qsfaExpectDtypeList, qsfaDesc->GetDataType(), name), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
void TQSFATilingCheck::LogErrorNumberSupport(const std::vector<T> &expectNumberList, const T &actualValue,
                                             const std::string &name, const std::string subName) const
{
    std::ostringstream qsfaOssNum;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        qsfaOssNum << std::to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            qsfaOssNum << ", ";
        }
    }

    OP_LOGE_FOR_INVALID_VALUE(opName_, (name + " " + subName).c_str(), std::to_string(actualValue).c_str(),
                              qsfaOssNum.str());
}

template <typename T>
void TQSFATilingCheck::LogErrorDimNumSupport(const std::vector<T> &expectNumberList, const T &actualValue,
                                             const std::string &name) const
{
    LogErrorNumberSupport(expectNumberList, actualValue, name, "dimension");
}

ge::graphStatus TQSFATilingCheck::CheckDimNumInLayoutSupport(const TQSFALayout &layout, const gert::StorageShape *shape,
                                                             const std::string &name) const
{
    const auto &qsfaDimIt = TQSFA_LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != qsfaDimIt->second,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    opName_, name.c_str(), std::to_string(shape->GetStorageShape().GetDimNum()).c_str(),
                    "When layout is " + TQSFALayoutToSerialString(layout) + ", the shape dim of " + name +
                        " should be " + std::to_string(qsfaDimIt->second)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckDimNumSupport(const gert::StorageShape *shape,
                                                     const std::vector<size_t> &qsfaExpectDimNumList,
                                                     const std::string &name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(qsfaExpectDimNumList.begin(), qsfaExpectDimNumList.end(), shape->GetStorageShape().GetDimNum()) ==
        qsfaExpectDimNumList.end()) {
        LogErrorDimNumSupport(qsfaExpectDimNumList, shape->GetStorageShape().GetDimNum(), name);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void TQSFATilingCheck::LogErrorLayoutSupport(const std::vector<TQSFALayout> &expectLayoutList,
                                             const TQSFALayout &actualLayout, const std::string &name) const
{
    std::ostringstream qsfaOssLayout;
    for (size_t i = 0; i < expectLayoutList.size(); ++i) {
        qsfaOssLayout << TQSFALayoutToSerialString(expectLayoutList[i]);
        if (i < expectLayoutList.size() - 1) {
            qsfaOssLayout << ", ";
        }
    }
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, name.c_str(), TQSFALayoutToSerialString(actualLayout).c_str(),
                                          "Tensor " + name + " only supports layout " + qsfaOssLayout.str());
}

ge::graphStatus TQSFATilingCheck::CheckLayoutSupport(const TQSFALayout &actualLayout, const std::string &name) const
{
    const auto &qsfaItLayout = LAYOUT_SUPPORT_MAP.find(name);
    OP_CHECK_IF(qsfaItLayout == LAYOUT_SUPPORT_MAP.end(),
                OP_LOGE(opName_, "%s layout support list should be specify in LAYOUT_SUPPORT_MAP", name.c_str()),
                return ge::GRAPH_FAILED);
    auto &qsfaExpectLayoutList = qsfaItLayout->second;
    OP_CHECK_IF(
        std::find(qsfaExpectLayoutList.begin(), qsfaExpectLayoutList.end(), actualLayout) == qsfaExpectLayoutList.end(),
        LogErrorLayoutSupport(qsfaExpectLayoutList, actualLayout, name), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckSingleParaQuery() const
{
    const std::vector<size_t> qsfaQueryDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.query.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(qLayout_, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.query.shape, qsfaQueryDimNumList, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(qLayout_, opParamInfo_.query.shape, QUERY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckSingleParaKey() const
{
    const std::vector<size_t> qsfaKeyDimNumList = {DIM_NUM_THREE, DIM_NUM_FOUR};
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.key.desc, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckLayoutSupport(kvLayout_, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(opParamInfo_.key.shape, qsfaKeyDimNumList, KEY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(kvLayout_, opParamInfo_.key.shape, KEY_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckSingleParaSparseMode() const
{
    OP_CHECK_IF((*opParamInfo_.sparseMode != 3 && *opParamInfo_.sparseMode != 0),
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "sparseMode invalid"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckSingleParaSparseBlockSize() const
{
    OP_CHECK_IF(((*opParamInfo_.sparseBlockSize <= 0 || *opParamInfo_.sparseBlockSize > 16) ||
                 (static_cast<uint64_t>(*opParamInfo_.sparseBlockSize) &
                  static_cast<uint64_t>(*opParamInfo_.sparseBlockSize - 1L)) != 0UL),
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "sparseBlockSize invalid"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckSingleParaSparseIndices() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(opParamInfo_.sparseIndices.desc, SPARSE_INDICES_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaQuery() || ge::GRAPH_SUCCESS != CheckSingleParaKey() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseIndices() || ge::GRAPH_SUCCESS != CheckSingleParaSparseMode() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSparseBlockSize()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckParaExistenceMlaAntiquant() const
{
    // KV 仅支持 PA_BSND，两个输入均为必需。
    OP_CHECK_IF(opParamInfo_.actualSeqLengths.tensor == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "actualSeqLengthsKv invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.blockTable.tensor == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "blockTable invalid"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckParaExistenceMla() const { return CheckParaExistenceMlaAntiquant(); }

ge::graphStatus TQSFATilingCheck::CheckParaExistence() { return CheckParaExistenceMla(); }

static ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor, const std::string &name,
                                           const char *opName)
{
    if (tensor == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, name.c_str(), "null", "Tensor " + name + " should not be null");
        return ge::GRAPH_FAILED;
    }
    int64_t qsfaShapeSize = tensor->GetShapeSize();
    if (qsfaShapeSize <= 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, name.c_str(), std::to_string(qsfaShapeSize).c_str(),
                                              "The shape size of " + name + " should be greater than 0");
        return ge::GRAPH_FAILED;
    }
    size = static_cast<uint32_t>(qsfaShapeSize);
    return ge::GRAPH_SUCCESS;
}

void TQSFATilingCheck::SetTQSFAShapeCompare()
{
    queryShapeCmp_ = opParamInfo_.query.shape->GetStorageShape();
    topkShapeCmp_ = opParamInfo_.sparseIndices.shape->GetStorageShape();
    keyShapeCmp_ = opParamInfo_.key.shape->GetStorageShape();
    valueShapeCmp_ = opParamInfo_.value.shape->GetStorageShape();
    attenOutShapeCmp_ = opParamInfo_.attenOut.shape->GetStorageShape();
}

ge::graphStatus TQSFATilingCheck::CheckBlockTable() const
{
    // block_table 的非空校验已在 CheckParaExistenceMlaAntiquant 中完成，且其调用
    // 先于本函数所在的 CheckMultiParaConsistency，故此处可直接解引用。
    uint32_t blockTableBatch = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0);
    OP_CHECK_IF(blockTableBatch != bSize_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "block_table", std::to_string(blockTableBatch).c_str(),
                    "The first dim of block_table should be equal to batch size " + std::to_string(bSize_)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckDTypeConsistency(const ge::DataType &actualDtype,
                                                        const ge::DataType &expectDtype, const std::string &name) const
{
    if (actualDtype != expectDtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            opName_, name.c_str(), TQSFADataTypeToSerialString(actualDtype).c_str(),
            "The dtype of " + name + " should be " + TQSFADataTypeToSerialString(expectDtype));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckTopkShape()
{
    TQSFATilingShapeCompareParam qsfaShapeParams;
    qsfaShapeParams.B = bSize_;
    qsfaShapeParams.N = n2Size_;
    qsfaShapeParams.S = s1Size_;
    qsfaShapeParams.D = sparseBlockCount_;
    qsfaShapeParams.T = qTSize_;
    return CompareShape(qsfaShapeParams, topkShapeCmp_, topkLayout_, SPARSE_INDICES_NAME);
}

ge::graphStatus TQSFATilingCheck::CheckAttenOutShape()
{
    TQSFATilingShapeCompareParam shapeParams;
    shapeParams.B = bSize_;
    shapeParams.N = n1Size_;
    shapeParams.S = s1Size_;
    shapeParams.D = qHeadDim_ - ropeHeadDim_;
    shapeParams.T = qTSize_;
    if (CompareShape(shapeParams, attenOutShapeCmp_, outLayout_, ATTEN_OUT_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckAttenOut()
{
    if (ge::GRAPH_SUCCESS !=
            CheckDTypeConsistency(opParamInfo_.attenOut.desc->GetDataType(), inputQType_, ATTEN_OUT_NAME) ||
        ge::GRAPH_SUCCESS != CheckAttenOutShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckTopK()
{
    if (ge::GRAPH_SUCCESS != CheckTopkShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckKV()
{
    if (ge::GRAPH_SUCCESS != CheckDTypeConsistency(opParamInfo_.value.desc->GetDataType(), inputKvType_, VALUE_NAME)) {
        return ge::GRAPH_FAILED;
    }

    // MLA 场景下 K 与 V 为同一份 latent，kernel 的 MM2 复用已反量化的合并缓冲，
    // 不单独读取 value。此处要求两者 shape 完全一致：既是该语义的直接表达，
    // 也可拦住误传不匹配 value 的调用（原先仅校验 dtype 会放行）。
    OP_CHECK_IF(valueShapeCmp_.GetDimNum() != keyShapeCmp_.GetDimNum(),
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "value dim num invalid"), return ge::GRAPH_FAILED);
    for (size_t i = 0U; i < keyShapeCmp_.GetDimNum(); ++i) {
        OP_CHECK_IF(valueShapeCmp_.GetDim(i) != keyShapeCmp_.GetDim(i),
                    OPS_REPORT_VECTOR_INNER_ERR(opName_, "value shape invalid"), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckActualSeqLensQ()
{
    if (opParamInfo_.actualSeqLengthsQ.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensQDType() || ge::GRAPH_SUCCESS != CheckActualSeqLensQShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckActualSeqLensQDType()
{
    if (opParamInfo_.actualSeqLengthsQ.desc == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "actualSeqLengthsQ's dtype invalid");
        return ge::GRAPH_FAILED;
    }

    if (opParamInfo_.actualSeqLengthsQ.desc->GetDataType() != ge::DT_INT32) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "actualSeqLengthsQ invalid");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckActualSeqLensQShape()
{
    uint32_t qsfaShapeSize = 0;
    if (GetActualSeqLenSize(qsfaShapeSize, opParamInfo_.actualSeqLengthsQ.tensor, "actualSeqLengthsQ", opName_) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (qsfaShapeSize != bSize_) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "actualSeqLengthsQ invalid");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckActualSeqLens()
{
    if (ge::GRAPH_SUCCESS != CheckActualSeqLensDType() || ge::GRAPH_SUCCESS != CheckActualSeqLensShape()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckActualSeqLensDType()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (opParamInfo_.actualSeqLengths.desc == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "actualSeqLengths's dtype invalid");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.actualSeqLengths.desc->GetDataType() != ge::DT_INT32) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "actualSeqLengths invalid");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckActualSeqLensShape()
{
    if (opParamInfo_.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    uint32_t qsfaShapeSizeKv = 0;
    if (GetActualSeqLenSize(qsfaShapeSizeKv, opParamInfo_.actualSeqLengths.tensor, "actualSeqLengths", opName_) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (qsfaShapeSizeKv != bSize_) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "actualSeqLengths invalid");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckMultiParaConsistency()
{
    SetTQSFAShapeCompare();
    if (ge::GRAPH_SUCCESS != CheckKV() || ge::GRAPH_SUCCESS != CheckTopK() || ge::GRAPH_SUCCESS != CheckAttenOut() ||
        ge::GRAPH_SUCCESS != CheckActualSeqLensQ() || ge::GRAPH_SUCCESS != CheckActualSeqLens() ||
        ge::GRAPH_SUCCESS != CheckBlockTable()) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckFeatureMlaAntiquantShape() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantShapeSizes() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantShapeSparseAndHeadDim()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckFeatureMlaAntiquantShapeSizes() const
{
    OP_CHECK_IF(bSize_ <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName_, "batch_size invalid"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(qTSize_ <= 0 && (qLayout_ == TQSFALayout::TND),
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "T_size of query invalid"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(n1Size_ <= 0, OPS_REPORT_VECTOR_INNER_ERR(opName_, "q_head_num invalid"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(n2Size_ != 1, OPS_REPORT_VECTOR_INNER_ERR(opName_, "kv_head_num invalid"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(n1Size_ % n2Size_ != 0, OPS_REPORT_VECTOR_INNER_ERR(opName_, "q_head_num and kv_head_num invalid"),
                return ge::GRAPH_FAILED);

    std::vector<uint32_t> gSizeSupportList = {1, 2, 4, 8, 16, 32, 64, 128};
    OP_CHECK_IF(std::find(gSizeSupportList.begin(), gSizeSupportList.end(), gSize_) == gSizeSupportList.end(),
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "group num invalid"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckFeatureMlaAntiquantShapeSparseAndHeadDim() const
{
    OP_CHECK_IF(sparseBlockSize_ <= 0 || (sparseBlockSize_ & (sparseBlockSize_ - 1)) != 0 || sparseBlockSize_ > 16,
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "sparseBlockSize_ invalid"), return ge::GRAPH_FAILED);

    // kernel 侧 MM1 的 K 宽度与 vector scratch 容量均按 576 / 512 写死，
    // 故此处必须按同一口径拦截，不能只做 qHeadDim_ > ropeHeadDim_ 的弱校验：
    // 例如 qD=640/ropeD=64 可通过弱校验，却会让 vector scratch 越界。
    OP_CHECK_IF(qHeadDim_ != TQ4_Q_HEAD_DIM, OPS_REPORT_VECTOR_INNER_ERR(opName_, "qHeadDim_ invalid"),
                return ge::GRAPH_FAILED);

    uint32_t kvLoraRank = qHeadDim_ - ropeHeadDim_;
    OP_CHECK_IF(kvLoraRank != TQ4_NOPE_HEAD_DIM || (kvLoraRank % 2U) != 0U,
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "kvLoraRank invalid"), return ge::GRAPH_FAILED);

    // slot 字节数由上述头维派生，三者必须自洽；不等即说明布局协议被改动。
    OP_CHECK_IF(kvLoraRank / 2U + static_cast<uint32_t>(ropeHeadDim_) * 2U + 2U != TQ4_KV_SLOT_BYTES,
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "kv slot layout invalid"), return ge::GRAPH_FAILED);
    uint32_t tq4SlotBytes = kvLoraRank / 2 + ropeHeadDim_ * NUM_BYTES_BF16 + NUM_BYTES_FLOAT16;
    uint32_t expectedKHeadDim = tq4SlotBytes;
    OP_CHECK_IF(kHeadDim_ != expectedKHeadDim, OPS_REPORT_VECTOR_INNER_ERR(opName_, "kHeadDim_ invalid"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckFeatureMlaAntiquantLayout() const
{
    // 与 LAYOUT_SUPPORT_MAP 保持一致：query 仅支持 TND（PyTorch launcher 强制
    // query 为 3D，且输出固定按 TND 构造，BSND 无对应实现）。
    const std::vector<std::string> qsfaLayoutSupportList = {"TND"};
    std::string layoutQuery = opParamInfo_.layoutQuery;
    OP_CHECK_IF(std::find(qsfaLayoutSupportList.begin(), qsfaLayoutSupportList.end(), layoutQuery) ==
                    qsfaLayoutSupportList.end(),
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "query invalid"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckFeatureMlaAntiquantDtype() const
{
    // 算子声明与 DTYPE_SUPPORT_MAP 均仅支持 BFLOAT16，此处同口径收窄，
    // 避免 feature checker 放行 FP16 而由上游 dtype 校验兜底的语义矛盾。
    OP_CHECK_IF(inputQType_ != ge::DT_BF16,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, QUERY_NAME.c_str(),
                                                      TQSFADataTypeToSerialString(inputQType_).c_str(),
                                                      "The dtype of query must be DT_BF16"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(inputKvType_ != ge::DT_INT8, OPS_REPORT_VECTOR_INNER_ERR(opName_, "key and value invalid"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckFeatureMlaAntiquantAttr() const
{
    OP_CHECK_IF(attentionMode_ != 2, // 2:MLA-absorb
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "attention_mode invalid"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(keyQuantMode_ != 3, // 3:TQ4 codebook (Phase B fused-in-SFA)
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "key_quant_mode invalid"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(valueQuantMode_ != 3, // 3:TQ4 codebook (Phase B fused-in-SFA)
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "value_quant_mode invalid"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(quantScaleRepoMode_ != 1, // 1:combine
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "quant_scale_repo_mode invalid"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(preTokens_ != INT64_MAX, OPS_REPORT_VECTOR_INNER_ERR(opName_, "preTokens_ invalid"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(nextTokens_ != INT64_MAX, OPS_REPORT_VECTOR_INNER_ERR(opName_, "nextTokens_ invalid"),
                return ge::GRAPH_FAILED);

    // kernel 只把 tile_size 存进 tiling 而从不读取，slot 布局与 512 维 nope 均按
    // 128 写死，故非 128 的取值不会产生对应语义，此处直接拒绝而非放行。
    OP_CHECK_IF(tileSize_ != TQ4_TILE_SIZE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "tile_size", std::to_string(tileSize_).c_str(),
                                                      "Tile_size should be 128"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ropeHeadDim_ != static_cast<int64_t>(TQ4_ROPE_HEAD_DIM),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "rope_head_dim", std::to_string(ropeHeadDim_).c_str(),
                                                      "Rope_head_dim should be 64"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckFeatureMlaAntiquantPa() const
{
    OP_CHECK_IF(blockSize_ <= 0 || blockSize_ > static_cast<int32_t>(MAX_BLOCK_SIZE),
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "block_size invalid"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(blockSize_ % 16 > 0, OPS_REPORT_VECTOR_INNER_ERR(opName_, "block_size invalid"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(blockSize_ % sparseBlockSize_ > 0, OPS_REPORT_VECTOR_INNER_ERR(opName_, "block_size invalid"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckFeatureMlaAntiquant() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantAttr() || ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantShape() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantLayout() || ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureMlaAntiquantPa()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFATilingCheck::CheckFeatureMla() const { return CheckFeatureMlaAntiquant(); }

ge::graphStatus TQSFATilingCheck::CheckFeature() const { return CheckFeatureMla(); }

void TQSFATilingCheck::Init()
{
    opName_ = qsfaInfo_.opName;
    platformInfo_ = qsfaInfo_.platformInfo;
    opParamInfo_ = qsfaInfo_.opParamInfo;

    bSize_ = qsfaInfo_.bSize;
    n1Size_ = qsfaInfo_.n1Size;
    n2Size_ = qsfaInfo_.n2Size;
    s1Size_ = qsfaInfo_.s1Size;
    s2Size_ = qsfaInfo_.s2Size;
    gSize_ = qsfaInfo_.gSize;
    qHeadDim_ = qsfaInfo_.qHeadDim;
    kHeadDim_ = qsfaInfo_.kHeadDim;
    vHeadDim_ = qsfaInfo_.vHeadDim;
    ropeHeadDim_ = qsfaInfo_.ropeHeadDim;
    maxBlockNumPerBatch_ = qsfaInfo_.maxBlockNumPerBatch;
    qTSize_ = qsfaInfo_.qTSize;
    kvTSize_ = qsfaInfo_.kvTSize;
    blockSize_ = qsfaInfo_.blockSize;
    sparseBlockCount_ = qsfaInfo_.sparseBlockCount;
    sparseBlockSize_ = qsfaInfo_.sparseBlockSize;

    attentionMode_ = qsfaInfo_.attentionMode;
    keyQuantMode_ = qsfaInfo_.keyQuantMode;
    valueQuantMode_ = qsfaInfo_.valueQuantMode;
    quantScaleRepoMode_ = qsfaInfo_.quantScaleRepoMode;
    tileSize_ = qsfaInfo_.tileSize;
    preTokens_ = qsfaInfo_.preTokens;
    nextTokens_ = qsfaInfo_.nextTokens;

    inputQType_ = qsfaInfo_.inputQType;
    inputKvType_ = qsfaInfo_.inputKvType;
    outputType_ = qsfaInfo_.outputType;

    qLayout_ = qsfaInfo_.qLayout;
    topkLayout_ = qsfaInfo_.topkLayout;
    kvLayout_ = qsfaInfo_.kvLayout;
    outLayout_ = qsfaInfo_.outLayout;

    l2CacheSize_ = qsfaInfo_.l2CacheSize;
}

ge::graphStatus TQSFATilingCheck::Process()
{
    Init();
    if (CheckSinglePara() != ge::GRAPH_SUCCESS || CheckParaExistence() != ge::GRAPH_SUCCESS ||
        CheckFeature() != ge::GRAPH_SUCCESS || CheckMultiParaConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static constexpr int64_t kInvalidDimValue = std::numeric_limits<int64_t>::min();

static bool HasAxis(const TQSFAAxis &axis, const TQSFALayout &layout, const gert::Shape &shape)
{
    const auto &qsfaLayoutIt = TQSFA_LAYOUT_AXIS_MAP.find(layout);
    if (qsfaLayoutIt == TQSFA_LAYOUT_AXIS_MAP.end()) {
        return false;
    }

    const std::vector<TQSFAAxis> &qsfaAxes = qsfaLayoutIt->second;
    const auto &qsfaAxisIt = std::find(qsfaAxes.begin(), qsfaAxes.end(), axis);
    if (qsfaAxisIt == qsfaAxes.end()) {
        return false;
    }

    const auto &qsfaDimIt = TQSFA_LAYOUT_DIM_MAP.find(layout);
    if (qsfaDimIt == TQSFA_LAYOUT_DIM_MAP.end() || qsfaDimIt->second != shape.GetDimNum()) {
        return false;
    }

    return true;
}

static size_t GetAxisIdx(const TQSFAAxis &axis, const TQSFALayout &layout)
{
    const std::vector<TQSFAAxis> &axes = TQSFA_LAYOUT_AXIS_MAP.find(layout)->second;
    const auto &axisIt = std::find(axes.begin(), axes.end(), axis);

    return std::distance(axes.begin(), axisIt);
}

static uint32_t GetAxisNum(const gert::Shape &shape, const TQSFAAxis &axis, const TQSFALayout &layout)
{
    return HasAxis(axis, layout, shape) ? shape.GetDim(GetAxisIdx(axis, layout)) : kInvalidDimValue;
}

ge::graphStatus TQSFAInfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.query.shape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "Shape of tensor query invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query.desc == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "Desc of tensor query invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.shape == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "Shape of tensor k invalid"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.desc == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "Desc of tensor k invalid"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.value.shape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "Shape of tensor value invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.value.desc == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "Desc of tensor value invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseIndices.shape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "Shape of tensor sparseIndices invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseIndices.desc == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "Desc of tensor sparseIndices invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attenOut.shape == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "Shape of tensor output invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.attenOut.desc == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "Desc of tensor output invalid"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.layoutQuery == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "layoutQuery invalid"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.layoutKV == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "layoutKV invalid"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseBlockSize == nullptr,
                OPS_REPORT_VECTOR_INNER_ERR(opName_, "sparseBlockSize invalid"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.scaleValue == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "scaleValue invalid"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.sparseMode == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "sparseMode invalid"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::GetActualSeqLenQSize(uint32_t &size)
{
    return GetActualSeqLenSize(size, opParamInfo_.actualSeqLengthsQ.tensor, "actualSeqLengthsQ", opName_);
}

ge::graphStatus TQSFAInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR("TurboQuantSparseFlashAttention", "opName invalid");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName_, "GetPlatformInfo is nullptr."),
                return ge::GRAPH_FAILED);

    auto qsfaAscendcPlat = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t qsfaAivNum = qsfaAscendcPlat.GetCoreNumAiv();
    uint32_t qsfaAicNum = qsfaAscendcPlat.GetCoreNumAic();
    OP_CHECK_IF(qsfaAicNum == 0 || qsfaAivNum == 0, OPS_REPORT_VECTOR_INNER_ERR(opName_, "num of core obtained is 0."),
                return GRAPH_FAILED);

    qsfaAscendcPlat.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize_);

    return ge::GRAPH_SUCCESS;
}

void TQSFAInfoParser::GetOptionalInputParaInfo()
{
    // 注意：即便 block_table 与两个 actual_seq_lengths 已在 OpDef 中声明为 REQUIRED，
    // 取值仍须走 GetOptionalInputTensor / GetOptionalInputDesc——实测改用
    // GetInputTensor / GetInputDesc 会取不到这三个输入，导致 tiling 直接失败
    // （IR 顺序中它们位于两个可选的 dequant scale 之后，索引解析方式不同）。
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INPUT_INDEX);
    opParamInfo_.actualSeqLengthsQ.tensor = context_->GetOptionalInputTensor(ACT_SEQ_LEN_Q_INPUT_INDEX);
    opParamInfo_.actualSeqLengthsQ.desc = context_->GetOptionalInputDesc(ACT_SEQ_LEN_Q_INPUT_INDEX);
    opParamInfo_.actualSeqLengths.tensor = context_->GetOptionalInputTensor(ACT_SEQ_LEN_KV_INPUT_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetOptionalInputDesc(ACT_SEQ_LEN_KV_INPUT_INDEX);
    opParamInfo_.keyDequantScale.tensor = context_->GetOptionalInputTensor(KEY_DEQUANT_SCALE_INPUT_INDEX);
    opParamInfo_.valueDequantScale.tensor = context_->GetOptionalInputTensor(VALUE_DEQUANT_SCALE_INPUT_INDEX);
}

void TQSFAInfoParser::GetInputParaInfo()
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

void TQSFAInfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(OUTPUT_INDEX);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(OUTPUT_INDEX);
}

ge::graphStatus TQSFAInfoParser::GetAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);

    opParamInfo_.layoutQuery = attrs->GetStr(LAYOUT_QUERY_ATTR_INDEX);
    opParamInfo_.layoutKV = attrs->GetStr(LAYOUT_KV_ATTR_INDEX);
    opParamInfo_.sparseBlockSize = attrs->GetAttrPointer<int64_t>(SPARSE_BLOCK_SIZE_ATTR_INDEX);
    opParamInfo_.scaleValue = attrs->GetAttrPointer<float>(SCALE_VALUE_ATTR_INDEX);
    opParamInfo_.sparseMode = attrs->GetAttrPointer<int64_t>(SPARSE_MODE_ATTR_INDEX);
    opParamInfo_.keyQuantMode = attrs->GetAttrPointer<int64_t>(KEY_QUANT_MODE_ATTR_INDEX);
    opParamInfo_.valueQuantMode = attrs->GetAttrPointer<int64_t>(VALUE_QUANT_MODE_ATTR_INDEX);
    opParamInfo_.attentionMode = attrs->GetAttrPointer<int64_t>(ATTENTION_MODE_ATTR_INDEX);
    opParamInfo_.preTokens = attrs->GetAttrPointer<int64_t>(PRE_TOKENS_ATTR_INDEX);
    opParamInfo_.nextTokens = attrs->GetAttrPointer<int64_t>(NEXT_TOKENS_ATTR_INDEX);
    opParamInfo_.quantScaleRepoMode = attrs->GetAttrPointer<int64_t>(QUANT_SCALE_REPO_MODE_ATTR_INDEX);
    opParamInfo_.tileSize = attrs->GetAttrPointer<int64_t>(TILE_SIZE_ATTR_INDEX);
    opParamInfo_.ropeHeadDim = attrs->GetAttrPointer<int64_t>(ROPE_HEAD_DIM_ATTR_INDEX);
    opParamInfo_.returnSoftmaxLse = attrs->GetAttrPointer<bool>(RETURN_SOFTMAX_LSE_ATTR_INDEX);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::GetInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKvType_ = opParamInfo_.key.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND时：以query的batch_size维度为基准
    // 2、TND时：actual_seq_lens_q必须传入，以actual_seq_lens_q数组的长度为B轴大小
    if (qLayout_ == TQSFALayout::TND) {
        return GetActualSeqLenQSize(bSize_);
    } else { // BSND
        bSize_ = GetAxisNum(queryShape_, TQSFAAxis::B, qLayout_);
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus TQSFAInfoParser::GetQTSize()
{
    // 获取query的T基准值
    // 1、非TND时：以query的batch_size维度为基准
    // 2、TND时：actual_seq_lens_q必须传入，以actual_seq_lens_q数组的长度为B轴大小
    qTSize_ = (qLayout_ == TQSFALayout::TND) ? GetAxisNum(queryShape_, TQSFAAxis::T, qLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::GetKVTSize()
{
    // 获取query的T基准值
    // 1、非TND时：以key的batch_size维度为基准
    // 2、TND时：actual_seq_lens_q必须传入，以actual_seq_lens_q数组的长度为B轴大小
    kvTSize_ = (kvLayout_ == TQSFALayout::TND) ? GetAxisNum(keyShape_, TQSFAAxis::T, kvLayout_) : 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::GetQHeadDim()
{
    // 获取qHeadDim基准值
    // 以query的D维度为基准
    qHeadDim_ = GetAxisNum(queryShape_, TQSFAAxis::D, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::GetKHeadDim()
{
    // 获取kHeadDim基准值
    // 以key的D维度为基准
    kHeadDim_ = GetAxisNum(keyShape_, TQSFAAxis::D, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::GetS1Size()
{
    // 获取S1基准值
    // 1、非TND时：以query的S维度为基准
    // 2、TND时：actual_seq_lens_q必须传入，以actual_seq_lens_q数组中的最大值为基准
    if (qLayout_ == TQSFALayout::TND) {
        s1Size_ = GetAxisNum(queryShape_, TQSFAAxis::T, qLayout_);
        return ge::GRAPH_SUCCESS;
    } else { // BSND
        s1Size_ = GetAxisNum(queryShape_, TQSFAAxis::S, qLayout_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::GetKvLayout()
{
    // 仅 PA_BSND 可达：非 PA_BSND 的 KV 要求与 query 同 layout，而 query 限定为 TND，
    // TND 的 KV 又无对应实现，故此处不再接受其它取值。
    const map<string, TQSFALayout> layoutKVMap = {{"PA_BSND", TQSFALayout::PA_BSND}};

    std::string layout(opParamInfo_.layoutKV);
    auto it = layoutKVMap.find(layout);
    if (it != layoutKVMap.end()) {
        kvLayout_ = it->second;
    } else {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "KV invalid");
        return ge::GRAPH_FAILED;
    }
    uint32_t keyDimNum = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    if (kvLayout_ == TQSFALayout::PA_BSND && keyDimNum != 4U) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "key invalid");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::GetMaxBlockNumPerBatch()
{
    if (opParamInfo_.blockTable.tensor == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "blockTable invalid");
        return ge::GRAPH_FAILED;
    }
    uint32_t qsfaDimNum = opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum();
    if (qsfaDimNum != DIM_NUM_TWO) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "block_table invalid");
        return ge::GRAPH_FAILED;
    }
    if (opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1) <= 0) {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "block_table invalid");
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::GetSparseBlockCount()
{
    sparseBlockCount_ = GetAxisNum(sparseIndicesShape_, TQSFAAxis::K, qLayout_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::GetBlockSize()
{
    blockSize_ = GetAxisNum(keyShape_, TQSFAAxis::Bs, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::GetS2SizeForPageAttention()
{
    if (GetMaxBlockNumPerBatch() != ge::GRAPH_SUCCESS || GetBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    s2Size_ = maxBlockNumPerBatch_ * blockSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::GetS2Size()
{
    // KV 仅支持 PA_BSND：S2 = block_table.dim1 * block_size
    return GetS2SizeForPageAttention();
}

ge::graphStatus TQSFAInfoParser::GetValueHeadDim()
{
    // 获取vHeadDim基准值
    // 以value的D维度为基准
    vHeadDim_ = GetAxisNum(valueShape_, TQSFAAxis::D, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::GetDSizeKV()
{
    dSizeKV_ = GetAxisNum(keyShape_, TQSFAAxis::D, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::GetQueryAndOutLayout()
{
    // 获取query和attentionOut的Layout基准值
    // layoutQuery: {qLayout, outLayout}
    const std::map<std::string, std::pair<TQSFALayout, TQSFALayout>> qsfaLayoutMap = {
        {"TND", {TQSFALayout::TND, TQSFALayout::TND}},
    };

    std::string qsfaLayout(opParamInfo_.layoutQuery);
    auto qsfaLayoutIt = qsfaLayoutMap.find(qsfaLayout);
    if (qsfaLayoutIt != qsfaLayoutMap.end()) {
        qLayout_ = qsfaLayoutIt->second.first;
        outLayout_ = qsfaLayoutIt->second.second;
    } else {
        OPS_REPORT_VECTOR_INNER_ERR(opName_, "query invalid");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::GetTopkLayout()
{
    topkLayout_ = qLayout_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::GetN1Size()
{
    n1Size_ = GetAxisNum(queryShape_, TQSFAAxis::N, qLayout_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::GetN2Size()
{
    n2Size_ = GetAxisNum(keyShape_, TQSFAAxis::N, kvLayout_);
    return ge::GRAPH_SUCCESS;
}

void TQSFAInfoParser::SetTQSFAShape()
{
    queryShape_ = opParamInfo_.query.shape->GetStorageShape();
    keyShape_ = opParamInfo_.key.shape->GetStorageShape();

    valueShape_ = opParamInfo_.value.shape->GetStorageShape();
    sparseIndicesShape_ = opParamInfo_.sparseIndices.shape->GetStorageShape();
}

ge::graphStatus TQSFAInfoParser::GetGSize()
{
    if (n2Size_ != 0) {
        gSize_ = n1Size_ / n2Size_;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQSFAInfoParser::GetActualseqInfo()
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

ge::graphStatus TQSFAInfoParser::GetShapeAndSizeInfo()
{
    SetTQSFAShape();
    if (ge::GRAPH_SUCCESS != GetN1Size() || ge::GRAPH_SUCCESS != GetN2Size() || ge::GRAPH_SUCCESS != GetGSize() ||
        ge::GRAPH_SUCCESS != GetBatchSize() || ge::GRAPH_SUCCESS != GetQTSize() || ge::GRAPH_SUCCESS != GetKVTSize() ||
        ge::GRAPH_SUCCESS != GetS1Size() || ge::GRAPH_SUCCESS != GetQHeadDim() || ge::GRAPH_SUCCESS != GetKHeadDim() ||
        ge::GRAPH_SUCCESS != GetS2Size() || ge::GRAPH_SUCCESS != GetValueHeadDim() ||
        ge::GRAPH_SUCCESS != GetDSizeKV() || ge::GRAPH_SUCCESS != GetSparseBlockCount()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void TQSFAInfoParser::GenerateInfo(TQSFATilingInfo &qsfaInfo)
{
    qsfaInfo.opName = opName_;
    qsfaInfo.platformInfo = platformInfo_;
    qsfaInfo.opParamInfo = opParamInfo_;

    qsfaInfo.bSize = bSize_;
    qsfaInfo.n1Size = n1Size_;
    qsfaInfo.n2Size = n2Size_;
    qsfaInfo.s1Size = s1Size_;
    qsfaInfo.s2Size = s2Size_;
    qsfaInfo.gSize = gSize_;
    qsfaInfo.qHeadDim = qHeadDim_;
    qsfaInfo.kHeadDim = kHeadDim_;
    qsfaInfo.vHeadDim = vHeadDim_;
    qsfaInfo.qTSize = qTSize_;
    qsfaInfo.kvTSize = kvTSize_;
    qsfaInfo.sparseBlockSize = *opParamInfo_.sparseBlockSize;
    qsfaInfo.sparseBlockCount = sparseBlockCount_;

    qsfaInfo.inputQType = inputQType_;
    qsfaInfo.inputKvType = inputKvType_;
    qsfaInfo.outputType = outputType_;

    qsfaInfo.l2CacheSize = l2CacheSize_;

    qsfaInfo.totalBlockNum = opParamInfo_.key.shape->GetStorageShape().GetDim(0);
    qsfaInfo.scaleValue = *opParamInfo_.scaleValue;
    qsfaInfo.blockSize = blockSize_;
    qsfaInfo.blockTypeSize = sizeof(float);
    qsfaInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    FillTilingInfoAttrsAndLayouts(qsfaInfo);
}

void TQSFAInfoParser::FillTilingInfoAttrsAndLayouts(TQSFATilingInfo &qsfaInfo)
{
    qsfaInfo.actualLenDimsQ = actualLenDimsQ_;
    qsfaInfo.actualLenDimsKV = actualLenDimsKV_;
    qsfaInfo.maxActualseq = maxActualseq_;

    qsfaInfo.actualQSeqLenFlag = (opParamInfo_.actualSeqLengthsQ.tensor != nullptr);
    qsfaInfo.actualSeqLenFlag = (opParamInfo_.actualSeqLengths.tensor != nullptr);

    qsfaInfo.isSameSeqAllKVTensor = isSameSeqAllKVTensor_;
    qsfaInfo.isSameActualseq = isSameActualseq_;

    qsfaInfo.sparseMode = *opParamInfo_.sparseMode;
    qsfaInfo.attentionMode = *opParamInfo_.attentionMode;
    qsfaInfo.keyQuantMode = *opParamInfo_.keyQuantMode;
    qsfaInfo.valueQuantMode = *opParamInfo_.valueQuantMode;
    qsfaInfo.quantScaleRepoMode = *opParamInfo_.quantScaleRepoMode;
    qsfaInfo.preTokens = *opParamInfo_.preTokens;
    qsfaInfo.nextTokens = *opParamInfo_.nextTokens;
    qsfaInfo.tileSize = *opParamInfo_.tileSize;
    qsfaInfo.ropeHeadDim = *opParamInfo_.ropeHeadDim;
    qsfaInfo.returnSoftmaxLse = (opParamInfo_.returnSoftmaxLse != nullptr) ? *opParamInfo_.returnSoftmaxLse : false;

    qsfaInfo.qLayout = qLayout_;
    qsfaInfo.topkLayout = topkLayout_;
    qsfaInfo.kvLayout = kvLayout_;
    qsfaInfo.outLayout = outLayout_;
    uint32_t tileSize = static_cast<uint32_t>(qsfaInfo.tileSize);
    if (qHeadDim_ > qsfaInfo.ropeHeadDim && tileSize > 0) {
        uint32_t kvLoraRank = qHeadDim_ - qsfaInfo.ropeHeadDim;
        qsfaInfo.dSizeVInput = kvLoraRank / 2 + qsfaInfo.ropeHeadDim * NUM_BYTES_BF16 + NUM_BYTES_FLOAT16;
    } else {
        qsfaInfo.dSizeVInput = dSizeKV_;
    }
}

ge::graphStatus TQSFAInfoParser::Parse(TQSFATilingInfo &qsfaInfo)
{
    if (context_ == nullptr) {
        OPS_REPORT_VECTOR_INNER_ERR("TurboQuantSparseFlashAttention", "tiling context invalid");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo() || ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetInOutDataType() || ge::GRAPH_SUCCESS != GetQueryAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetTopkLayout() || ge::GRAPH_SUCCESS != GetKvLayout()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetShapeAndSizeInfo()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetActualseqInfo()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(qsfaInfo);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TurboQuantSparseFlashAttention)
    .Tiling(TilingTurboQuantSparseFlashAttention)
    .TilingParse<TurboQuantSparseFlashAttentionCompileInfo>(TilingPrepareForTurboQuantSparseFlashAttention);
} // namespace optiling
