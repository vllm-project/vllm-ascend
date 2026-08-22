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
 * \file fused_infer_attention_score_tiling_check.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "fused_infer_attention_score_tiling_v3.h"
#include "fused_infer_attention_score_tiling_check.h"
#include "../fused_infer_attention_score_tiling_compile_info.h"
#include "../fused_infer_attention_score_tiling_index.h"
#include "../../../incre_flash_attention/op_host/incre_flash_attention_tiling_base.h"

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
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("FusedInferAttentionScore", "datatype %d not support", type);
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
    if (fiaInfo_.emptyTensorFlag) {
        return ge::GRAPH_SUCCESS;
    }
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
