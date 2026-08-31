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
 * \file fused_infer_attention_score_v2_sink_tiling_check_existence.cpp
 * \brief
 */

#include <map>
#include <vector>
#include <string>
#include <utility>
#include <sstream>
#include <numeric>
#include <algorithm>
#include "tiling/tiling_api.h"
#include "fused_infer_attention_score_v2_sink_tiling_check.h"

using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
namespace optiling {
constexpr int64_t ANTI_QUANT_MODE_DEFAULT_VALUE = 0;
constexpr int64_t KEY_ANTI_QUANT_MODE_DEFAULT_VALUE = 0;
constexpr int64_t VALUE_ANTI_QUANT_MODE_DEFAULT_VALUE = 0;
constexpr int64_t QUERY_QUANT_MODE_DEFAULT_VALUE = 0;

ge::graphStatus FiaTilingCheck::CheckRopeExistence() const
{
    OPS_CHECK((opParamInfo_.queryRope.tensor != nullptr && opParamInfo_.keyRope.tensor == nullptr),
              OPS_LOG_E(opName_,
                        "%s is null, but queryRope exists, they should be both null or exist.",
                        KEY_ROPE_NAME.c_str()),
              return ge::GRAPH_FAILED);
    OPS_CHECK((opParamInfo_.queryRope.tensor == nullptr && opParamInfo_.keyRope.tensor != nullptr),
              OPS_LOG_E(opName_,
                        "%s is null, but keyRope exists, they should be both null or exist.",
                        QUERY_ROPE_NAME.c_str()),
              return ge::GRAPH_FAILED);

    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        OPS_CHECK(opParamInfo_.keyRope.desc == nullptr || opParamInfo_.queryRope.desc == nullptr,
                  OPS_LOG_E(opName_,
                            "In %s situation and rope exists, desc of %s and %s should not be null",
                            QuantModeToSerialString(quantMode_).c_str(),
                            KEY_ROPE_NAME.c_str(),
                            QUERY_ROPE_NAME.c_str()),
                  return ge::GRAPH_FAILED);
    } else if (ropeMode_ == RopeMode::ROPE_COMBINE) {
        OPS_CHECK(opParamInfo_.keyRope.desc != nullptr || opParamInfo_.queryRope.desc != nullptr,
                  OPS_LOG_E(opName_,
                            "In %s situation and rope exists, desc of %s and %s should be null",
                            QuantModeToSerialString(quantMode_).c_str(),
                            KEY_ROPE_NAME.c_str(),
                            QUERY_ROPE_NAME.c_str()),
                  return ge::GRAPH_FAILED);
    }

    OPS_LOG_I(opName_, "rope mode is %s", RopeModeToSerialString(ropeMode_).c_str());
    return ge::GRAPH_SUCCESS;
}

static std::string DtypeListToStr(const std::vector<DataType> &dtypeList)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < dtypeList.size(); ++i) {
        oss << FusedDataTypeToSerialString(dtypeList[i]);
        if (i < dtypeList.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]";

    return oss.str();
}

static std::string DtypeDoubleListToStr(const std::vector<std::vector<DataType>> &dtypeDoubleList)
{
    std::ostringstream oss;
    for (size_t i = 0; i < dtypeDoubleList.size(); ++i) {
        oss << DtypeListToStr(dtypeDoubleList[i]);
        if (i < dtypeDoubleList.size() - 1) {
            oss << ", ";
        }
    }
    return oss.str();
}

ge::graphStatus FiaTilingCheck::CheckDtypeAndSetQuantFlagMla()
{
    const std::vector<std::vector<ge::DataType>> mlaNoquantDtypeList = {
        // queryDtype,   kvDtype,        queryRopeDtype, keyRopeDtype
        {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
        {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16},
    };

    ge::DataType queryRopeDtype = opParamInfo_.queryRope.desc->GetDataType();
    ge::DataType keyRopeDtype = opParamInfo_.keyRope.desc->GetDataType();
    std::vector<ge::DataType> actualDtypeList = {inputQType_, inputKvType_, queryRopeDtype, keyRopeDtype};
    if (VecContains(mlaNoquantDtypeList, actualDtypeList)) {
        quantMode_ = FiaQuantMode::NO_QUANT;
    } else {
        OPS_LOG_E(opName_,
                  "In %s situation and rope exists, only supports [query_dtype, kv_dtype, query_rope_dtype, "
                  "key_rope_dtype] as %s, but got %s",
                  QuantModeToSerialString(quantMode_).c_str(),
                  DtypeDoubleListToStr(mlaNoquantDtypeList).c_str(),
                  DtypeListToStr(actualDtypeList).c_str());
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_I(opName_, "quant mode is %s", QuantModeToSerialString(quantMode_).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckDtypeAndSetQuantFlagGqa()
{
    const std::vector<std::vector<ge::DataType>> gqaNoquantDtypeList = {
        // queryDtype,   kvDtype
        {ge::DT_FLOAT16, ge::DT_FLOAT16},
        {ge::DT_BF16, ge::DT_BF16},
    };

    std::vector<ge::DataType> actualDtypeList = {inputQType_, inputKvType_};
    if (VecContains(gqaNoquantDtypeList, actualDtypeList)) {
        quantMode_ = FiaQuantMode::NO_QUANT;
    } else {
        OPS_LOG_E(opName_,
                  "In %s situation, only supports [query_dtype, kv_dtype] as %s, %s, %s, but got %s",
                  QuantModeToSerialString(quantMode_).c_str(),
                  DtypeDoubleListToStr(gqaNoquantDtypeList).c_str(),
                  DtypeListToStr(actualDtypeList).c_str());
        return ge::GRAPH_FAILED;
    }

    OPS_LOG_I(opName_, "quant mode is %s", QuantModeToSerialString(quantMode_).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckDtypeAndSetQuantFlag()
{
    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        return CheckDtypeAndSetQuantFlagMla();
    } else {
        return CheckDtypeAndSetQuantFlagGqa();
    }
}

ge::graphStatus FiaTilingCheck::CheckExists(const void *pointer, const std::string &name) const
{
    OPS_CHECK(pointer == nullptr,
              OPS_LOG_E(opName_,
                        "In %s, %s situation, %s should not be null",
                        QuantModeToSerialString(quantMode_).c_str(),
                        SituationToSerialString(ropeMode_).c_str(),
                        name.c_str()),
              return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckNotExists(const void *pointer, const std::string &name) const
{
    OPS_CHECK(pointer != nullptr,
              OPS_LOG_E(opName_,
                        "In %s, %s situation, %s should be null",
                        QuantModeToSerialString(quantMode_).c_str(),
                        SituationToSerialString(ropeMode_).c_str(),
                        name.c_str()),
              return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto &kv : paramMap) {
        if (CheckExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckNotExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto &kv : paramMap) {
        if (CheckNotExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckExistenceByMap(std::map<std::string, const void *> &existMap,
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
ge::graphStatus FiaTilingCheck::CheckAttrValueByMap(std::map<std::string, std::pair<const T *, T>> &attrMap) const
{
    for (auto const &kv : attrMap) {
        const std::string &name = kv.first;
        const std::pair<const T *, T> &pointerValuePair = kv.second;
        if (pointerValuePair.first == nullptr) {
            OPS_LOG_E(opName_, "%s should not be nullptr", name.c_str());
            return ge::GRAPH_FAILED;
        }

        if (*(pointerValuePair.first) != pointerValuePair.second) {
            std::ostringstream ossExpect;
            ossExpect << std::to_string(pointerValuePair.second);
            std::ostringstream ossActual;
            ossActual << std::to_string(*(pointerValuePair.first));
            OPS_LOG_E(opName_,
                      "In %s situation, %s value should be %s, but got %s",
                      QuantModeToSerialString(quantMode_).c_str(),
                      name.c_str(),
                      ossExpect.str().c_str(),
                      ossActual.str().c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckParaExistenceMlaNoquant() const
{
    std::map<std::string, const void *> mlaNoquantParamExistMap = {};
    std::map<std::string, const void *> mlaNoquantParamNotExistMap = {
        // antiquantParam
        {ANTIQUANT_SCALE_NAME, opParamInfo_.antiquantScale.tensor},
        {ANTIQUANT_OFFSET_NAME, opParamInfo_.antiquantOffset.tensor},
        {KEY_ANTIQUANT_SCALE_NAME, opParamInfo_.keyAntiquantScale.tensor},
        {KEY_ANTIQUANT_OFFSET_NAME, opParamInfo_.keyAntiquantOffset.tensor},
        {VALUE_ANTIQUANT_SCALE_NAME, opParamInfo_.valueAntiquantScale.tensor},
        {VALUE_ANTIQUANT_OFFSET_NAME, opParamInfo_.valueAntiquantOffset.tensor},
        {KEY_ROPE_ANTIQUANT_SCALE_NAME, opParamInfo_.keyRopeAntiquantScale.tensor},
        // fullquantParam
        {DEQUANT_SCALE1_NAME, opParamInfo_.deqScale1.tensor},
        {QUANT_SCALE1_NAME, opParamInfo_.quantScale1.tensor},
        {DEQUANT_SCALE2_NAME, opParamInfo_.deqScale2.tensor},
        {DEQUANT_SCALE_QUERY_NAME, opParamInfo_.dequantScaleQuery.tensor},
        // postquantParam
        {QUANT_SCALE2_NAME, opParamInfo_.quantScale2.tensor},
        {QUANT_OFFSET2_NAME, opParamInfo_.quantOffset2.tensor},
        // unsupportedFeaturesParam
        {PSE_SHIFT_NAME, opParamInfo_.pseShift.tensor},
        {QUERY_PADDING_SIZE_NAME, opParamInfo_.queryPaddingSize.tensor},
        {KV_PADDING_SIZE_NAME, opParamInfo_.kvPaddingSize.tensor},
    };

    std::map<std::string, std::pair<const int64_t *, int64_t>> attrDefaultValueMap = {
        {ANTIQUANT_MODE_NAME, {opParamInfo_.antiquantMode, ANTI_QUANT_MODE_DEFAULT_VALUE}},
        {KEY_ANTIQUANT_MODE_NAME, {opParamInfo_.keyAntiquantMode, KEY_ANTI_QUANT_MODE_DEFAULT_VALUE}},
        {VALUE_ANTIQUANT_MODE_NAME, {opParamInfo_.valueAntiquantMode, VALUE_ANTI_QUANT_MODE_DEFAULT_VALUE}},
        {QUERY_QUANT_MODE_NAME, {opParamInfo_.queryQuantMode, QUERY_QUANT_MODE_DEFAULT_VALUE}},
    };
    if (CheckExistenceByMap(mlaNoquantParamExistMap, mlaNoquantParamNotExistMap) != ge::GRAPH_SUCCESS ||
        CheckAttrValueByMap(attrDefaultValueMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus FiaTilingCheck::CheckParaExistenceGqaNoquant() const
{
    std::map<std::string, const void *> gqaNoquantParamExistMap = {};

    std::map<std::string, const void *> gqaNoquantParamNotExistMap = {
        // antiquantParam
        {ANTIQUANT_SCALE_NAME, opParamInfo_.antiquantScale.tensor},
        {ANTIQUANT_OFFSET_NAME, opParamInfo_.antiquantOffset.tensor},
        {KEY_ANTIQUANT_SCALE_NAME, opParamInfo_.keyAntiquantScale.tensor},
        {KEY_ANTIQUANT_OFFSET_NAME, opParamInfo_.keyAntiquantOffset.tensor},
        {VALUE_ANTIQUANT_SCALE_NAME, opParamInfo_.valueAntiquantScale.tensor},
        {VALUE_ANTIQUANT_OFFSET_NAME, opParamInfo_.valueAntiquantOffset.tensor},
        {KEY_ROPE_ANTIQUANT_SCALE_NAME, opParamInfo_.keyRopeAntiquantScale.tensor},
        // fullquantParam
        {DEQUANT_SCALE1_NAME, opParamInfo_.deqScale1.tensor},
        {QUANT_SCALE1_NAME, opParamInfo_.quantScale1.tensor},
        {DEQUANT_SCALE2_NAME, opParamInfo_.deqScale2.tensor},
        {DEQUANT_SCALE_QUERY_NAME, opParamInfo_.dequantScaleQuery.tensor},
    };

    std::map<std::string, std::pair<const int64_t *, int64_t>> attrDefaultValueMap = {
        {ANTIQUANT_MODE_NAME, {opParamInfo_.antiquantMode, ANTI_QUANT_MODE_DEFAULT_VALUE}},
        {KEY_ANTIQUANT_MODE_NAME, {opParamInfo_.keyAntiquantMode, KEY_ANTI_QUANT_MODE_DEFAULT_VALUE}},
        {VALUE_ANTIQUANT_MODE_NAME, {opParamInfo_.valueAntiquantMode, VALUE_ANTI_QUANT_MODE_DEFAULT_VALUE}},
        {QUERY_QUANT_MODE_NAME, {opParamInfo_.queryQuantMode, QUERY_QUANT_MODE_DEFAULT_VALUE}},
    };
    if (CheckExistenceByMap(gqaNoquantParamExistMap, gqaNoquantParamNotExistMap) != ge::GRAPH_SUCCESS ||
        CheckAttrValueByMap(attrDefaultValueMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckParaExistenceMla() const
{
    if (quantMode_ == FiaQuantMode::NO_QUANT) {
        return CheckParaExistenceMlaNoquant();
    } else {
        OPS_LOG_E(opName_, "fiaSink Only Support NoQuant, but got %s", QuantModeToSerialString(quantMode_).c_str());
    }

    return ge::GRAPH_SUCCESS;
}
ge::graphStatus FiaTilingCheck::CheckParaExistenceGqa() const
{
    if (quantMode_ == FiaQuantMode::NO_QUANT) {
        return CheckParaExistenceGqaNoquant();
    } else {
        OPS_LOG_E(opName_, "fiaSink Only Support NoQuant, but got %s", QuantModeToSerialString(quantMode_).c_str());
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckParaExistence()
{
    if (ge::GRAPH_SUCCESS != CheckRopeExistence() || ge::GRAPH_SUCCESS != CheckDtypeAndSetQuantFlag()) {
        return ge::GRAPH_FAILED;
    }

    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        return CheckParaExistenceMla();
    } else {
        return CheckParaExistenceGqa();
    }
}
} // namespace optiling
