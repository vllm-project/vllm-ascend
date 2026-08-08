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
 * \file fused_infer_attention_score_tiling_check_existence.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "fused_infer_attention_score_tiling_check.h"

using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
namespace optiling {
constexpr int64_t ANTI_QUANT_MODE_DEFAULT_VALUE = 0;
constexpr int64_t KEY_ANTI_QUANT_MODE_DEFAULT_VALUE = 0;
constexpr int64_t VALUE_ANTI_QUANT_MODE_DEFAULT_VALUE = 0;
constexpr int64_t QUERY_QUANT_MODE_DEFAULT_VALUE = 0;

ge::graphStatus FiaTilingCheck::CheckRopeExistence() const
{
    OP_CHECK_IF((opParamInfo_.queryRope.tensor != nullptr && opParamInfo_.keyRope.tensor == nullptr),
        OP_LOGE(opName_, "%s is null, but queryRope exists, they should be both null or exist.", KEY_ROPE_NAME.c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((opParamInfo_.queryRope.tensor == nullptr && opParamInfo_.keyRope.tensor != nullptr),
        OP_LOGE(opName_, "%s is null, but keyRope exists, they should be both null or exist.", QUERY_ROPE_NAME.c_str()),
        return ge::GRAPH_FAILED);

    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        OP_CHECK_IF(opParamInfo_.keyRope.desc == nullptr || opParamInfo_.queryRope.desc == nullptr,
            OP_LOGE(opName_, "In %s situation and rope exsists, desc of %s and %s should not be null",
                QuantModeToSerialString(quantMode_).c_str(),
                KEY_ROPE_NAME.c_str(), QUERY_ROPE_NAME.c_str()),
            return ge::GRAPH_FAILED);
    } else if (ropeMode_ == RopeMode::ROPE_COMBINE) {
        OP_CHECK_IF(opParamInfo_.keyRope.desc != nullptr || opParamInfo_.queryRope.desc != nullptr,
            OP_LOGE(opName_, "In %s situation and rope exsists, desc of %s and %s should be null",
                QuantModeToSerialString(quantMode_).c_str(),
                KEY_ROPE_NAME.c_str(), QUERY_ROPE_NAME.c_str()),
            return ge::GRAPH_FAILED);
    }

    OP_LOGI(opName_, "rope mode is %s", RopeModeToSerialString(ropeMode_).c_str());
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
        {ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16},
    };
    const std::vector<std::vector<ge::DataType>> mlaAntiquantDtypeList = {
        {ge::DT_FLOAT16, ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_INT8},
        {ge::DT_BF16,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_INT8},
    };
    const std::vector<std::vector<ge::DataType>> mlaFullquantDtypeList = {
        {ge::DT_INT8,    ge::DT_INT8,    ge::DT_FLOAT16, ge::DT_FLOAT16},
        {ge::DT_INT8,    ge::DT_INT8,    ge::DT_BF16,    ge::DT_BF16},
    };

    ge::DataType queryRopeDtype = opParamInfo_.queryRope.desc->GetDataType();
    ge::DataType keyRopeDtype = opParamInfo_.keyRope.desc->GetDataType();
    std::vector<ge::DataType> actualDtypeList = {
        inputQType_, inputKvType_, queryRopeDtype, keyRopeDtype
    };
    if (VecContains(mlaNoquantDtypeList, actualDtypeList)) {
        quantMode_ = FiaQuantMode::NO_QUANT;
    } else if (VecContains(mlaAntiquantDtypeList, actualDtypeList)) {
        quantMode_ = FiaQuantMode::ANTI_QUANT;
    } else if (VecContains(mlaFullquantDtypeList, actualDtypeList)) {
        quantMode_ = FiaQuantMode::FULL_QUANT;
    } else {
        OP_LOGE(opName_, "In %s situation and rope exsists, only supports [query_dtype, kv_dtype, query_rope_dtype, key_rope_dtype] as %s, %s, %s, but got %s",
            QuantModeToSerialString(quantMode_).c_str(),
            DtypeDoubleListToStr(mlaNoquantDtypeList).c_str(),
            DtypeDoubleListToStr(mlaAntiquantDtypeList).c_str(),
            DtypeDoubleListToStr(mlaFullquantDtypeList).c_str(),
            DtypeListToStr(actualDtypeList).c_str());
        return ge::GRAPH_FAILED;
    }

    OP_LOGI(opName_, "quant mode is %s", QuantModeToSerialString(quantMode_).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckDtypeAndSetQuantFlagGqa()
{
    const std::vector<std::vector<ge::DataType>> gqaNoquantDtypeList = {
        // queryDtype,   kvDtype
        {ge::DT_FLOAT16, ge::DT_FLOAT16},
        {ge::DT_BF16,    ge::DT_BF16},
    };
    const std::vector<std::vector<ge::DataType>> gqaAntiquantDtypeList = {
        {ge::DT_FLOAT16, ge::DT_INT8},
        {ge::DT_BF16,    ge::DT_INT8},
        {ge::DT_FLOAT16, ge::DT_INT4},
        {ge::DT_BF16,    ge::DT_INT4},
    };
    const std::vector<std::vector<ge::DataType>> gqaFullquantDtypeList = {
        {ge::DT_INT8,    ge::DT_INT8},
    };

    std::vector<ge::DataType> actualDtypeList = {
        inputQType_, inputKvType_
    };
    if (VecContains(gqaNoquantDtypeList, actualDtypeList)) {
        quantMode_ = FiaQuantMode::NO_QUANT;
    } else if (VecContains(gqaAntiquantDtypeList, actualDtypeList)) {
        quantMode_ = FiaQuantMode::ANTI_QUANT;
    } else if (VecContains(gqaFullquantDtypeList, actualDtypeList)) {
        quantMode_ = FiaQuantMode::FULL_QUANT;
    } else {
        OP_LOGE(opName_, "In %s situation, only supports [query_dtype, kv_dtype] as %s, %s, %s, but got %s",
            QuantModeToSerialString(quantMode_).c_str(),
            DtypeDoubleListToStr(gqaNoquantDtypeList).c_str(),
            DtypeDoubleListToStr(gqaAntiquantDtypeList).c_str(),
            DtypeDoubleListToStr(gqaFullquantDtypeList).c_str(),
            DtypeListToStr(actualDtypeList).c_str());
        return ge::GRAPH_FAILED;
    }

    OP_LOGI(opName_, "quant mode is %s", QuantModeToSerialString(quantMode_).c_str());
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
    OP_CHECK_IF(pointer == nullptr,
        OP_LOGE(opName_, "In %s, %s situation, %s should not be null",
            QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), name.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckNotExists(const void *pointer, const std::string &name) const
{
    OP_CHECK_IF(pointer != nullptr,
        OP_LOGE(opName_, "In %s, %s situation, %s should be null",
            QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str(), name.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto& kv : paramMap) {
        if (CheckExists(kv.second, kv.first) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckNotExistsByMap(const std::map<std::string, const void *> &paramMap) const
{
    for (const auto& kv : paramMap) {
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

void FiaTilingCheck::LogErrorExistenceEqual(std::map<std::string, const void *> &paramMap) const
{
    std::ostringstream oss;
    for (auto it = paramMap.begin(); it != paramMap.end(); ++it) {
        oss << it->first;
        if (it->second == nullptr) {
            oss << "(null)";
        } else {
            oss << "(not null)";
        }
        if (std::next(it) != paramMap.end()) {
            oss << ", ";
        }
    }

    OP_LOGE(opName_, "In %s situation, %s's existence status should be same",
        QuantModeToSerialString(quantMode_).c_str(), oss.str().c_str());
}

ge::graphStatus FiaTilingCheck::CheckParaExistenceEqual(std::map<std::string, const void *> &paramMap) const
{
    if (paramMap.empty()) {
        return ge::GRAPH_SUCCESS;
    }

    auto it = paramMap.begin();
    bool firstVal = (it->second != nullptr);
    bool isAllEqual = true;
    for (++it; it != paramMap.end(); ++it) {
        bool currentVal = (it->second != nullptr);
        if (currentVal != firstVal) {
            isAllEqual = false;
            break;
        }
    }

    if (!isAllEqual) {
        LogErrorExistenceEqual(paramMap);
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
            OP_LOGE(opName_, "%s should not be nullptr", name.c_str());
            return ge::GRAPH_FAILED;
        }

        if (*(pointerValuePair.first) != pointerValuePair.second) {
            std::ostringstream ossExpect;
            ossExpect << std::to_string(pointerValuePair.second);
            std::ostringstream ossActual;
            ossActual << std::to_string(*(pointerValuePair.first));
            OP_LOGE(opName_,
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
        // unsupportedFeaturesParam
        {PSE_SHIFT_NAME, opParamInfo_.pseShift.tensor},
        {QUERY_PADDING_SIZE_NAME, opParamInfo_.queryPaddingSize.tensor},
        {KV_PADDING_SIZE_NAME, opParamInfo_.kvPaddingSize.tensor},
        {KEY_SHARED_PREFIX_NAME, opParamInfo_.keySharedPrefix.tensor},
        {VALUE_SHARED_PREFIX_NAME, opParamInfo_.valueSharedPrefix.tensor},
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
    // torch_npu==2.1, 当actualSharedPrefixLen不传的时候, pta会把actualSharedPrefixLen传入一个shape为0的tensor
    OP_CHECK_IF(opParamInfo_.actualSharedPrefixLen.tensor != nullptr && opParamInfo_.actualSharedPrefixLen.tensor->GetStorageShape().GetShapeSize() != 0,
        OP_LOGE(opName_, "In %s, %s situation, actualSharedPrefixLen should be null",
            QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckParaExistenceMlaAntiquant() const
{
    std::map<std::string, const void *> mlaAntiquantParamExistMap = {
        // paParamMap
        {ACTUAL_SEQ_KV_LEN_NAME, opParamInfo_.actualSeqLengths.tensor},
        {BLOCK_TABLE_NAME, opParamInfo_.blockTable.tensor},
        // antiquantParam
        {KEY_ANTIQUANT_SCALE_NAME, opParamInfo_.keyAntiquantScale.tensor},
        {VALUE_ANTIQUANT_SCALE_NAME, opParamInfo_.valueAntiquantScale.tensor},
        {KEY_ROPE_ANTIQUANT_SCALE_NAME, opParamInfo_.keyRopeAntiquantScale.tensor},
    };

    std::map<std::string, const void *> mlaAntiquantParamNotExistMap = {
        // antiquantParam
        {ANTIQUANT_SCALE_NAME, opParamInfo_.antiquantScale.tensor},
        {ANTIQUANT_OFFSET_NAME, opParamInfo_.antiquantOffset.tensor},
        {KEY_ANTIQUANT_OFFSET_NAME, opParamInfo_.keyAntiquantOffset.tensor},
        {VALUE_ANTIQUANT_OFFSET_NAME, opParamInfo_.valueAntiquantOffset.tensor},
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
        {KEY_SHARED_PREFIX_NAME, opParamInfo_.keySharedPrefix.tensor},
        {VALUE_SHARED_PREFIX_NAME, opParamInfo_.valueSharedPrefix.tensor},
    };

    std::map<std::string, std::pair<const int64_t *, int64_t>> attrDefaultValueMap = {
        {ANTIQUANT_MODE_NAME, {opParamInfo_.antiquantMode, ANTI_QUANT_MODE_DEFAULT_VALUE}},
        {QUERY_QUANT_MODE_NAME, {opParamInfo_.queryQuantMode, QUERY_QUANT_MODE_DEFAULT_VALUE}},
    };
    if (CheckExistenceByMap(mlaAntiquantParamExistMap, mlaAntiquantParamNotExistMap) != ge::GRAPH_SUCCESS ||
        CheckAttrValueByMap(attrDefaultValueMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(opParamInfo_.actualSharedPrefixLen.tensor != nullptr && opParamInfo_.actualSharedPrefixLen.tensor->GetStorageShape().GetShapeSize() != 0,
        OP_LOGE(opName_, "In %s, %s situation, actualSharedPrefixLen should be null",
            QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckParaExistenceMlaFullquant() const
{
    std::map<std::string, const void *> mlaFullquantParamExistMap = {
        // paParamMap
        {ACTUAL_SEQ_KV_LEN_NAME, opParamInfo_.actualSeqLengths.tensor},
        {BLOCK_TABLE_NAME, opParamInfo_.blockTable.tensor},
        // fullquantParam
        {KEY_ANTIQUANT_SCALE_NAME, opParamInfo_.keyAntiquantScale.tensor},
        {VALUE_ANTIQUANT_SCALE_NAME, opParamInfo_.valueAntiquantScale.tensor},
        {DEQUANT_SCALE_QUERY_NAME, opParamInfo_.dequantScaleQuery.tensor},
    };

    std::map<std::string, const void *> mlaFullquantParamNotExistMap = {
        // antiquantParam
        {ANTIQUANT_SCALE_NAME, opParamInfo_.antiquantScale.tensor},
        {ANTIQUANT_OFFSET_NAME, opParamInfo_.antiquantOffset.tensor},
        {KEY_ANTIQUANT_OFFSET_NAME, opParamInfo_.keyAntiquantOffset.tensor},
        {VALUE_ANTIQUANT_OFFSET_NAME, opParamInfo_.valueAntiquantOffset.tensor},
        {KEY_ROPE_ANTIQUANT_SCALE_NAME, opParamInfo_.keyRopeAntiquantScale.tensor},
        // fullquantParam
        {DEQUANT_SCALE1_NAME, opParamInfo_.deqScale1.tensor},
        {QUANT_SCALE1_NAME, opParamInfo_.quantScale1.tensor},
        {DEQUANT_SCALE2_NAME, opParamInfo_.deqScale2.tensor},
        // postquantParam
        {QUANT_SCALE2_NAME, opParamInfo_.quantScale2.tensor},
        {QUANT_OFFSET2_NAME, opParamInfo_.quantOffset2.tensor},
        // unsupportedFeaturesParam
        {PSE_SHIFT_NAME, opParamInfo_.pseShift.tensor},
        {QUERY_PADDING_SIZE_NAME, opParamInfo_.queryPaddingSize.tensor},
        {KV_PADDING_SIZE_NAME, opParamInfo_.kvPaddingSize.tensor},
        {KEY_SHARED_PREFIX_NAME, opParamInfo_.keySharedPrefix.tensor},
        {VALUE_SHARED_PREFIX_NAME, opParamInfo_.valueSharedPrefix.tensor},
    };

    std::map<std::string, std::pair<const int64_t *, int64_t>> attrDefaultValueMap = {
        {ANTIQUANT_MODE_NAME, {opParamInfo_.antiquantMode, ANTI_QUANT_MODE_DEFAULT_VALUE}},
    };
    if (CheckExistenceByMap(mlaFullquantParamExistMap, mlaFullquantParamNotExistMap) != ge::GRAPH_SUCCESS ||
        CheckAttrValueByMap(attrDefaultValueMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(opParamInfo_.actualSharedPrefixLen.tensor != nullptr && opParamInfo_.actualSharedPrefixLen.tensor->GetStorageShape().GetShapeSize() != 0,
        OP_LOGE(opName_, "In %s, %s situation, actualSharedPrefixLen should be null",
            QuantModeToSerialString(quantMode_).c_str(), SituationToSerialString(ropeMode_).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckParaExistenceGqaNoquantForFullquant() const
{
    std::string layout = opParamInfo_.layOut;
    const std::vector<std::string> layoutSupportList = {
        "BSH", "BSND", "BNSD", "BNSD_BSND"
    };
    if((std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) != layoutSupportList.end()) && (s1Size_ > 1) && (inputQType_ == ge::DT_INT8 || outputType_ != ge::DT_INT8)) {
        return ge::GRAPH_SUCCESS;
    } else {
        std::map<std::string, const void *> gqaNoquantParamNotExistMap = {
            // fullquantParam
            {DEQUANT_SCALE1_NAME, opParamInfo_.deqScale1.tensor},
            {QUANT_SCALE1_NAME, opParamInfo_.quantScale1.tensor},
            {DEQUANT_SCALE2_NAME, opParamInfo_.deqScale2.tensor},
        };
        if (CheckNotExistsByMap(gqaNoquantParamNotExistMap) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
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
        {DEQUANT_SCALE_QUERY_NAME, opParamInfo_.dequantScaleQuery.tensor},
    };

    std::map<std::string, std::pair<const int64_t *, int64_t>> attrDefaultValueMap = {
        {ANTIQUANT_MODE_NAME, {opParamInfo_.antiquantMode, ANTI_QUANT_MODE_DEFAULT_VALUE}},
        {KEY_ANTIQUANT_MODE_NAME, {opParamInfo_.keyAntiquantMode, KEY_ANTI_QUANT_MODE_DEFAULT_VALUE}},
        {VALUE_ANTIQUANT_MODE_NAME, {opParamInfo_.valueAntiquantMode, VALUE_ANTI_QUANT_MODE_DEFAULT_VALUE}},
        {QUERY_QUANT_MODE_NAME, {opParamInfo_.queryQuantMode, QUERY_QUANT_MODE_DEFAULT_VALUE}},
    };
    if (CheckExistenceByMap(gqaNoquantParamExistMap, gqaNoquantParamNotExistMap) != ge::GRAPH_SUCCESS ||
        CheckAttrValueByMap(attrDefaultValueMap) != ge::GRAPH_SUCCESS ||
        CheckExistenceSystemPrefix() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if(CheckParaExistenceGqaNoquantForFullquant() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckParaExistenceGqaAntiquantInt8Inner() const
{
    if (opParamInfo_.keyAntiquantScale.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(opParamInfo_.keyAntiquantOffset.tensor != nullptr,
        OP_LOGE(opName_,
            "In %s situation, %s is null, but %s exists", 
            QuantModeToSerialString(quantMode_).c_str(),
            KEY_ANTIQUANT_SCALE_NAME.c_str(),
            KEY_ANTIQUANT_OFFSET_NAME.c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.antiquantScale.tensor == nullptr,
        OP_LOGE(opName_,
            "In %s situation, when %s is null, %s should not be null", 
            QuantModeToSerialString(quantMode_).c_str(),
            KEY_ANTIQUANT_SCALE_NAME.c_str(),
            ANTIQUANT_SCALE_NAME.c_str()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((*opParamInfo_.keyAntiquantMode != 0) || (*opParamInfo_.valueAntiquantMode != 0),
        OP_LOGE(opName_,
            "In %s situation, when %s is null, %s(%ld) and %s(%ld) should both be 0", 
            QuantModeToSerialString(quantMode_).c_str(),
            KEY_ANTIQUANT_SCALE_NAME.c_str(),
            KEY_ANTIQUANT_MODE_NAME.c_str(),
            *opParamInfo_.keyAntiquantMode,
            VALUE_ANTIQUANT_MODE_NAME.c_str(),
            *opParamInfo_.valueAntiquantMode),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckParaExistenceGqaAntiquantInt8() const
{
    std::map<std::string, const void *> kvAntiquantScaleMap = {
        {KEY_ANTIQUANT_SCALE_NAME, opParamInfo_.keyAntiquantScale.tensor},
        {VALUE_ANTIQUANT_SCALE_NAME, opParamInfo_.valueAntiquantScale.tensor},
    };
    std::map<std::string, const void *> kvAntiquantOffsetMap = {
        {KEY_ANTIQUANT_OFFSET_NAME, opParamInfo_.keyAntiquantOffset.tensor},
        {VALUE_ANTIQUANT_OFFSET_NAME, opParamInfo_.valueAntiquantOffset.tensor},
    };
    if (CheckParaExistenceEqual(kvAntiquantScaleMap) != ge::GRAPH_SUCCESS ||
        CheckParaExistenceEqual(kvAntiquantOffsetMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    std::map<std::string, const void *> gqaAntiquantInt8ParamExistMap = {};
    std::map<std::string, const void *> gqaAntiquantInt8ParamNotExistMap = {
        // antiquantParam
        {KEY_ROPE_ANTIQUANT_SCALE_NAME, opParamInfo_.keyRopeAntiquantScale.tensor},
        // fullquantParam
        {DEQUANT_SCALE1_NAME, opParamInfo_.deqScale1.tensor},
        {QUANT_SCALE1_NAME, opParamInfo_.quantScale1.tensor},
        {DEQUANT_SCALE2_NAME, opParamInfo_.deqScale2.tensor},
        {DEQUANT_SCALE_QUERY_NAME, opParamInfo_.dequantScaleQuery.tensor},
    };

    std::map<std::string, std::pair<const int64_t *, int64_t>> attrDefaultValueMap = {
        {QUERY_QUANT_MODE_NAME, {opParamInfo_.queryQuantMode, QUERY_QUANT_MODE_DEFAULT_VALUE}},
    };
    if (CheckExistenceByMap(gqaAntiquantInt8ParamExistMap, gqaAntiquantInt8ParamNotExistMap) != ge::GRAPH_SUCCESS ||
        CheckAttrValueByMap(attrDefaultValueMap) != ge::GRAPH_SUCCESS ||
        CheckParaExistenceGqaAntiquantInt8Inner() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckParaExistenceGqaAntiquantInt4() const
{
    std::map<std::string, const void *> kvAntiquantScaleMap = {
        {KEY_ANTIQUANT_OFFSET_NAME, opParamInfo_.keyAntiquantOffset.tensor},
        {VALUE_ANTIQUANT_OFFSET_NAME, opParamInfo_.valueAntiquantOffset.tensor},
    };
    if (CheckParaExistenceEqual(kvAntiquantScaleMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    std::map<std::string, const void *> gqaAntiquantInt4ParamExistMap = {};
    std::map<std::string, const void *> gqaAntiquantInt4ParamNotExistMap = {
        // antiquantParam
        {KEY_ROPE_ANTIQUANT_SCALE_NAME, opParamInfo_.keyRopeAntiquantScale.tensor},
        // fullquantParam
        {DEQUANT_SCALE1_NAME, opParamInfo_.deqScale1.tensor},
        {QUANT_SCALE1_NAME, opParamInfo_.quantScale1.tensor},
        {DEQUANT_SCALE2_NAME, opParamInfo_.deqScale2.tensor},
        {DEQUANT_SCALE_QUERY_NAME, opParamInfo_.dequantScaleQuery.tensor},
        // postquantParam
        {QUANT_SCALE2_NAME, opParamInfo_.quantScale2.tensor},
        {QUANT_OFFSET2_NAME, opParamInfo_.quantOffset2.tensor},
    };

    std::map<std::string, std::pair<const int64_t *, int64_t>> attrDefaultValueMap = {
        {QUERY_QUANT_MODE_NAME, {opParamInfo_.queryQuantMode, QUERY_QUANT_MODE_DEFAULT_VALUE}},
    };
    if (CheckExistenceByMap(gqaAntiquantInt4ParamExistMap, gqaAntiquantInt4ParamNotExistMap) != ge::GRAPH_SUCCESS ||
        CheckAttrValueByMap(attrDefaultValueMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckParaExistenceGqaAntiquant() const
{
    if (CheckParaExistenceGqaAntiquantInt8() != ge::GRAPH_SUCCESS ||
        CheckParaExistenceGqaAntiquantInt4() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckParaExistenceGqaFullquant() const
{
    std::map<std::string, const void *> gqaFullquantParamExistMap = {
        // fullquantParam
        {DEQUANT_SCALE1_NAME, opParamInfo_.deqScale1.tensor},
        {QUANT_SCALE1_NAME, opParamInfo_.quantScale1.tensor},
        {DEQUANT_SCALE2_NAME, opParamInfo_.deqScale2.tensor},
    };

    std::map<std::string, const void *> gqaFullquantParamNotExistMap = {
        // antiquantParam
        {ANTIQUANT_SCALE_NAME, opParamInfo_.antiquantScale.tensor},
        {ANTIQUANT_OFFSET_NAME, opParamInfo_.antiquantOffset.tensor},
        {KEY_ANTIQUANT_OFFSET_NAME, opParamInfo_.keyAntiquantOffset.tensor},
        {VALUE_ANTIQUANT_OFFSET_NAME, opParamInfo_.valueAntiquantOffset.tensor},
        {KEY_ROPE_ANTIQUANT_SCALE_NAME, opParamInfo_.keyRopeAntiquantScale.tensor},
        {KEY_ANTIQUANT_SCALE_NAME, opParamInfo_.keyAntiquantScale.tensor},
        {VALUE_ANTIQUANT_SCALE_NAME, opParamInfo_.valueAntiquantScale.tensor},
        // fullquantParam
        {DEQUANT_SCALE_QUERY_NAME, opParamInfo_.dequantScaleQuery.tensor},
        // syetemprefix
        {KEY_SHARED_PREFIX_NAME, opParamInfo_.keySharedPrefix.tensor},
        {VALUE_SHARED_PREFIX_NAME, opParamInfo_.valueSharedPrefix.tensor},
        {ACTUAL_SHARED_PREFIX_LEN_NAME, opParamInfo_.actualSharedPrefixLen.tensor}
    };

    std::map<std::string, std::pair<const int64_t *, int64_t>> attrDefaultValueMap = {
        {ANTIQUANT_MODE_NAME, {opParamInfo_.antiquantMode, ANTI_QUANT_MODE_DEFAULT_VALUE}},
        {KEY_ANTIQUANT_MODE_NAME, {opParamInfo_.keyAntiquantMode, KEY_ANTI_QUANT_MODE_DEFAULT_VALUE}},
        {VALUE_ANTIQUANT_MODE_NAME, {opParamInfo_.valueAntiquantMode, VALUE_ANTI_QUANT_MODE_DEFAULT_VALUE}},
        {QUERY_QUANT_MODE_NAME, {opParamInfo_.queryQuantMode, QUERY_QUANT_MODE_DEFAULT_VALUE}},
    };
    if (CheckExistenceByMap(gqaFullquantParamExistMap, gqaFullquantParamNotExistMap) != ge::GRAPH_SUCCESS ||
        CheckAttrValueByMap(attrDefaultValueMap) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckExistenceSystemPrefix() const
{
    if (!fiaInfo_.sysPrefixFlag && (opParamInfo_.keySharedPrefix.tensor != nullptr || opParamInfo_.valueSharedPrefix.tensor != nullptr)) {
        if (opParamInfo_.keySharedPrefix.tensor == nullptr) {
            OP_LOGE(opName_, "When valueSharedPrefix exists, keySharedPrefix should also exist.");
            return ge::GRAPH_FAILED;
        }
        if (opParamInfo_.valueSharedPrefix.tensor == nullptr) {
            OP_LOGE(opName_, "When keySharedPrefix exists, valueSharedPrefix should also exist.");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckParaExistenceMla() const
{
    if (quantMode_ == FiaQuantMode::NO_QUANT) {
        return CheckParaExistenceMlaNoquant();
    } else if (quantMode_ == FiaQuantMode::ANTI_QUANT) {
        return CheckParaExistenceMlaAntiquant();
    } else if (quantMode_ == FiaQuantMode::FULL_QUANT) {
        return CheckParaExistenceMlaFullquant();
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckParaExistenceGqa() const
{
    if (quantMode_ == FiaQuantMode::NO_QUANT) {
        return CheckParaExistenceGqaNoquant();
    } else if (quantMode_ == FiaQuantMode::ANTI_QUANT) {
        return CheckParaExistenceGqaAntiquant();
    } else if (quantMode_ == FiaQuantMode::FULL_QUANT) {
        return CheckParaExistenceGqaFullquant();
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaTilingCheck::CheckParaExistence()
{
    if (ge::GRAPH_SUCCESS != CheckRopeExistence() ||
        ge::GRAPH_SUCCESS != CheckDtypeAndSetQuantFlag()) {
        return ge::GRAPH_FAILED;
    }

    if (ropeMode_ == RopeMode::ROPE_SPLIT) {
        return CheckParaExistenceMla();
    } else {
        return CheckParaExistenceGqa();
    }
}
} // namespace optiling
