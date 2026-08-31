/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_v2_sink_tiling_check.cpp
 * \brief
 */

#include <map>
#include <vector>
#include <string>
#include <utility>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "register/op_def_registry.h"
#include "fused_infer_attention_score_v2_sink_tiling_v3.h"
#include "fused_infer_attention_score_v2_sink_tiling_check.h"
#include "fused_infer_attention_score_v2_sink_tiling_compile_info.h"
#include "fused_infer_attention_score_v2_sink_tiling_index.h"

using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
namespace optiling {

std::string RopeModeToSerialString(const RopeMode &ropeMode)
{
    switch (ropeMode) {
        case RopeMode::NO_ROPE:
            return "GQA";
        case RopeMode::ROPE_SPLIT:
            return "MLA";
        case RopeMode::ROPE_COMBINE:
            return "MLA";
        default: return "UNKNOWN";
    }
}

std::string FusedDataTypeToSerialString(ge::DataType type)
{
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

    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OPS_LOG_E("FusedInferAttentionScoreV2Sink", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

void FiaTilingCheck::Init()
{
    opName_ = fiaInfo_.opName;
    platformInfo_ = fiaInfo_.platformInfo;
    opParamInfo_ = fiaInfo_.opParamInfo;
    socVersion_ = fiaInfo_.socVersion;

    bSize_ = fiaInfo_.bSize;
    n1Size_ = fiaInfo_.n1Size;
    n2Size_ = fiaInfo_.n2Size;
    s1Size_ = fiaInfo_.s1Size;
    s2Size_ = fiaInfo_.s2Size;
    gSize_ = fiaInfo_.gSize;
    qkHeadDim_ = fiaInfo_.qkHeadDim;
    vHeadDim_ = fiaInfo_.vHeadDim;
    ropeHeadDim_ = fiaInfo_.ropeHeadDim;
    maxBlockNumPerBatch_ = fiaInfo_.maxBlockNumPerBatch;
    qTSize_ = fiaInfo_.qTSize;
    kTSize_ = fiaInfo_.kTSize;
    blockSize_ = fiaInfo_.blockSize;

    inputQType_ = fiaInfo_.inputQType;
    inputKvType_ = fiaInfo_.inputKvType;
    inputQRopeType_ = fiaInfo_.inputQRopeType;
    inputKRopeType_ = fiaInfo_.inputKRopeType;
    outputType_ = fiaInfo_.outputType;

    qLayout_ = fiaInfo_.qLayout;
    kvLayout_ = fiaInfo_.kvLayout;
    outLayout_ = fiaInfo_.outLayout;

    kvStorageMode_ = fiaInfo_.kvStorageMode;
    ropeMode_ = fiaInfo_.ropeMode;
    l2CacheSize_ = fiaInfo_.l2CacheSize;
    attenMaskFlag_ = fiaInfo_.attenMaskFlag;
    kCache_ = fiaInfo_.kCache;
    vCache_ = fiaInfo_.vCache;
}

ge::graphStatus FiaTilingCheck::Process()
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

ge::graphStatus TilingCheck::Check(const FiaTilingInfo &fiaInfo)
{
    // Check函数只做校验，不能修改fiaInfo中的信息
    FiaTilingCheck tilingChecker(fiaInfo);
    return tilingChecker.Process();
}
} // namespace optiling
