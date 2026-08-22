/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file system_prefix_checker.cpp
 * \brief
 */

#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "system_prefix_checker.h"

namespace optiling {
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// singlepara
ge::graphStatus SystemPrefixChecker::CheckSharedPrefixDim(const FiaTilingInfo &fiaInfo)
{
    // 校验keySharedPrefix和valueSharedPrefix的的维度
    auto &keySharedPrefixTensor = fiaInfo.opParamInfo.keySharedPrefix.tensor;
    auto &valueSharedPrefixTensor = fiaInfo.opParamInfo.valueSharedPrefix.tensor;
    if (!fiaInfo.sysPrefixFlag) {
        // 若keySharedPrefix或valueSharedPrefix的shape为空，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    // keySharedPrefix、valueSharedPrefix、key、value的维度相同
    uint32_t keySharedPrefixDimNum = keySharedPrefixTensor->GetStorageShape().GetDimNum();
    uint32_t valueSharedPrefixDimNum = valueSharedPrefixTensor->GetStorageShape().GetDimNum();
    uint32_t keyDimNum = fiaInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum();
    if (keySharedPrefixDimNum != keyDimNum || valueSharedPrefixDimNum != keyDimNum) {
        std::string dimsStr = std::to_string(keySharedPrefixDimNum) + ", " + std::to_string(valueSharedPrefixDimNum) +
            " and " + std::to_string(keyDimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(fiaInfo.opName, "key_shared_prefix, value_shared_prefix and key",
            dimsStr.c_str(),
            "The shape dims of key_shared_prefix, value_shared_prefix and key/value must be the same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SystemPrefixChecker::CheckSharedPrefixDataType(const FiaTilingInfo &fiaInfo)
{
    // 校验keySharedPrefix和valueSharedPrefix的数据类型
    auto &keySharedPrefixDesc = fiaInfo.opParamInfo.keySharedPrefix.desc;
    auto &valueSharedPrefixDesc = fiaInfo.opParamInfo.valueSharedPrefix.desc;
    if (keySharedPrefixDesc != nullptr && valueSharedPrefixDesc != nullptr) {
        // keySharedPrefix和valueSharedPrefix的数据类型与key/value相同
        auto keySharedPrefixType = keySharedPrefixDesc->GetDataType();
        auto valueSharedPrefixType = valueSharedPrefixDesc->GetDataType();
        if (keySharedPrefixType != fiaInfo.inputKvType) {
            std::string dtypeMsg = ToString(keySharedPrefixType) + " and " + ToString(fiaInfo.inputKvType);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, "key_shared_prefix and key/value",
                dtypeMsg.c_str(), "The dtypes of key_shared_prefix and key/value must be the same");
            return ge::GRAPH_FAILED;
        }
        if (keySharedPrefixType != valueSharedPrefixType) {
            std::string dtypeMsg = ToString(keySharedPrefixType) + " and " + ToString(valueSharedPrefixType);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, "key_shared_prefix and value_shared_prefix",
                dtypeMsg.c_str(), "The dtypes of key_shared_prefix and value_shared_prefix must be the same");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SystemPrefixChecker::CheckSharedPrefixShape(const FiaTilingInfo &fiaInfo)
{
    // 校验keySharedPrefix和valueSharedPrefix的shape合法性
    auto &keySharedPrefixTensor = fiaInfo.opParamInfo.keySharedPrefix.tensor;
    auto &valueSharedPrefixTensor = fiaInfo.opParamInfo.valueSharedPrefix.tensor;
    if (keySharedPrefixTensor != nullptr && valueSharedPrefixTensor != nullptr) {
        auto &keySharedPrefixShape = fiaInfo.opParamInfo.keySharedPrefix.tensor->GetStorageShape();
        auto &valueSharedPrefixShape = fiaInfo.opParamInfo.valueSharedPrefix.tensor->GetStorageShape();
        auto &keyShape = fiaInfo.opParamInfo.key.shape->GetStorageShape();
        // 第一维batch必须为1
        if (keySharedPrefixShape.GetDim(DIM_NUM_0) != 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key_shared_prefix",
                ToString(keySharedPrefixShape).c_str(), "The 1st axis of key_shared_prefix must be equal to 1");
            return ge::GRAPH_FAILED;
        }
        if (valueSharedPrefixShape.GetDim(DIM_NUM_0) != 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "value_shared_prefix",
                ToString(valueSharedPrefixShape).c_str(), "The 1st axis of value_shared_prefix must be equal to 1");
            return ge::GRAPH_FAILED;
        }
        // layout为BNSD和BSND情况下，N、D轴与key一致
        if (fiaInfo.kvLayout == FiaLayout::BNSD) {
            uint32_t keySharedPrefixN = keySharedPrefixShape.GetDim(DIM_NUM_1);
            uint32_t keySharedPrefixD = keySharedPrefixShape.GetDim(DIM_NUM_3);
            uint32_t valueSharedPrefixN = valueSharedPrefixShape.GetDim(DIM_NUM_1);
            uint32_t valueSharedPrefixD = valueSharedPrefixShape.GetDim(DIM_NUM_3);
            uint32_t keyN = keyShape.GetDim(DIM_NUM_1);
            uint32_t keyD = keyShape.GetDim(DIM_NUM_3);
            uint32_t keySharedPrefixS = keySharedPrefixShape.GetDim(DIM_NUM_2);
            uint32_t valueSharedPrefixS = valueSharedPrefixShape.GetDim(DIM_NUM_2);
            if (keySharedPrefixN != keyN) {
                std::string shapeStr = ToString(keySharedPrefixShape) + " and " + ToString(keyShape);
                std::string reason = "N of key_shared_prefix must be equal to the same axis "
                    "of key when layout of key is BNSD or BSND";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key_shared_prefix and key",
                    shapeStr.c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
            if (keySharedPrefixN != valueSharedPrefixN) {
                std::string shapeStr = ToString(keySharedPrefixShape) + " and " + ToString(valueSharedPrefixShape);
                std::string reason = "N of key_shared_prefix must be equal to the same axis of "
                    "valueSharedPrefix when layout of key is BNSD or BSND";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key_shared_prefix and value_shared_prefix",
                    shapeStr.c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
            if (keySharedPrefixD != keyD) {
                std::string shapeStr = ToString(keySharedPrefixShape) + " and " + ToString(keyShape);
                std::string reason = "D of key_shared_prefix must be equal to the same axis of "
                    "key when layout of key is BNSD or BSND";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key_shared_prefix and key",
                    shapeStr.c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
            if (keySharedPrefixD != valueSharedPrefixD) {
                std::string shapeStr = ToString(keySharedPrefixShape) + " and " + ToString(valueSharedPrefixShape);
                std::string reason = "D of key_shared_prefix must be equal to the same axis of "
                    "valueSharedPrefix when layout of key is BNSD or BSND";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key_shared_prefix and value_shared_prefix",
                    shapeStr.c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
            // keySharedPrefix和valueSharedPrefix的S应相等
            if (keySharedPrefixS != valueSharedPrefixS) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key_shared_prefix and value_shared_prefix",
                    (ToString(keySharedPrefixShape) + " and " + ToString(valueSharedPrefixShape)).c_str(),
                    "S of key_shared_prefix must be equal to the same axis of value_shared_prefix");
                return ge::GRAPH_FAILED;
            }
        }
        if (fiaInfo.kvLayout == FiaLayout::BSND) {
            uint32_t keySharedPrefixS = keySharedPrefixShape.GetDim(DIM_NUM_1);
            uint32_t valueSharedPrefixS = valueSharedPrefixShape.GetDim(DIM_NUM_1);
            uint32_t keySharedPrefixD = keySharedPrefixShape.GetDim(DIM_NUM_3);
            uint32_t valueSharedPrefixD = valueSharedPrefixShape.GetDim(DIM_NUM_3);
            uint32_t keyD = keyShape.GetDim(DIM_NUM_3);
            uint32_t keySharedPrefixN = keySharedPrefixShape.GetDim(DIM_NUM_2);
            uint32_t valueSharedPrefixN = valueSharedPrefixShape.GetDim(DIM_NUM_2);
            uint32_t keyN = keyShape.GetDim(DIM_NUM_2);
            if (keySharedPrefixD != keyD) {
                std::string shapeStr = ToString(keySharedPrefixShape) + " and " + ToString(keyShape);
                std::string reason = "D of key_shared_prefix must be equal to the same axis of key "
                    "when layout of key is BNSD or BSND";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key_shared_prefix and key",
                    shapeStr.c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
            if (keySharedPrefixD != valueSharedPrefixD) {
                std::string shapeStr = ToString(keySharedPrefixShape) + " and " + ToString(valueSharedPrefixShape);
                std::string reason = "D of key_shared_prefix must be equal to the same axis of "
                    "valueSharedPrefix when layout of key is BNSD or BSND";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key_shared_prefix and value_shared_prefix",
                    shapeStr.c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
            if (keySharedPrefixN != keyN) {
                std::string shapeStr = ToString(keySharedPrefixShape) + " and " + ToString(keyShape);
                std::string reason = "N of key_shared_prefix must be equal to the same axis of "
                    "key when layout of key is BNSD or BSND";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key_shared_prefix and key",
                    shapeStr.c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
            if (keySharedPrefixN != valueSharedPrefixN) {
                std::string shapeStr = ToString(keySharedPrefixShape) + " and " + ToString(valueSharedPrefixShape);
                std::string reason = "N of key_shared_prefix must be equal to the same axis of "
                    "valueSharedPrefix when layout of key is BNSD or BSND";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key_shared_prefix and value_shared_prefix",
                    shapeStr.c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
            // keySharedPrefix和valueSharedPrefix的S应相等
            if (keySharedPrefixS != valueSharedPrefixS) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key_shared_prefix and value_shared_prefix",
                    (ToString(keySharedPrefixShape) + " and " + ToString(valueSharedPrefixShape)).c_str(),
                    "S of key_shared_prefix must be equal to the same axis of value_shared_prefix");
                return ge::GRAPH_FAILED;
            }
        }
        // layout为BSH情况下H与key一致
        if (fiaInfo.kvLayout == FiaLayout::BSH) {
            uint32_t keySharedPrefixH = keySharedPrefixShape.GetDim(DIM_NUM_2);
            uint32_t valueSharedPrefixH = valueSharedPrefixShape.GetDim(DIM_NUM_2);
            uint32_t keySharedPrefixS = keySharedPrefixShape.GetDim(DIM_NUM_1);
            uint32_t valueSharedPrefixS = valueSharedPrefixShape.GetDim(DIM_NUM_1);
            uint32_t keyH = keyShape.GetDim(DIM_NUM_2);
            if (keySharedPrefixH != keyH) {
                std::string shapeStr = ToString(keySharedPrefixShape) + " and " + ToString(keyShape);
                std::string reason = "H of key_shared_prefix must be equal to the same axis of key "
                    "when layout of key is BSH";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key_shared_prefix and key",
                    shapeStr.c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
            if (keySharedPrefixH != valueSharedPrefixH) {
                std::string shapeStr = ToString(keySharedPrefixShape) + " and " + ToString(valueSharedPrefixShape);
                std::string reason = "H of key_shared_prefix must be equal to the same axis of "
                    "valueSharedPrefix when layout of key is BSH";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key_shared_prefix and value_shared_prefix",
                    shapeStr.c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
            // keySharedPrefix和valueSharedPrefix的S应相等
            if (keySharedPrefixS != valueSharedPrefixS) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key_shared_prefix and value_shared_prefix",
                    (ToString(keySharedPrefixShape) + " and " + ToString(valueSharedPrefixShape)).c_str(),
                    "S of key_shared_prefix must be equal to the same axis of value_shared_prefix");
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SystemPrefixChecker::CheckActualSharedPrefixLenData(const FiaTilingInfo &fiaInfo)
{
    // 校验actualSharedPrefixLen的数值，其值不能大于keySharedPrefix和valueSharedPrefix的S
    auto &keySharedPrefixTensor = fiaInfo.opParamInfo.keySharedPrefix.tensor;
    auto &valueSharedPrefixTensor = fiaInfo.opParamInfo.valueSharedPrefix.tensor;
    auto &actualSharedPrefixLenTensor = fiaInfo.opParamInfo.actualSharedPrefixLen.tensor;
    if (keySharedPrefixTensor == nullptr || valueSharedPrefixTensor == nullptr) {
        // 若keySharedPrefix、valueSharedPrefix不存在，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    if (actualSharedPrefixLenTensor == nullptr || actualSharedPrefixLenTensor->GetData<int64_t>() == nullptr) {
        // 若没有传入actualSharedPrefixLen的shape或者data，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    uint32_t keySharedPrefixS = 0;
    uint32_t valueSharedPrefixS = 0;
    const std::vector<std::string> layoutSupportList = {
        "BSND", "BNSD", "BSH", "BNSD_BSND"
    };
    std::string layout(fiaInfo.opParamInfo.layOut);
    if (std::find(layoutSupportList.begin(), layoutSupportList.end(), layout) == layoutSupportList.end()) {
        std::string reason = "The value of input_layout must be in BSH, BSND, BNSD, and BNSD_BSND "
            "when system prefix is enabled";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout",
            layout.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    if (layout == "BNSD" || layout == "BNSD_BSND") {
        keySharedPrefixS = keySharedPrefixTensor->GetStorageShape().GetDim(DIM_NUM_2);
        valueSharedPrefixS = valueSharedPrefixTensor->GetStorageShape().GetDim(DIM_NUM_2);
    } else if (layout == "BSND") {
        keySharedPrefixS = keySharedPrefixTensor->GetStorageShape().GetDim(DIM_NUM_1);
        valueSharedPrefixS = valueSharedPrefixTensor->GetStorageShape().GetDim(DIM_NUM_1);
    } else if (layout == "BSH") {
        keySharedPrefixS = keySharedPrefixTensor->GetStorageShape().GetDim(DIM_NUM_1);
        valueSharedPrefixS = valueSharedPrefixTensor->GetStorageShape().GetDim(DIM_NUM_1);
    }
    int64_t actualSharedPrefixLenData = actualSharedPrefixLenTensor->GetData<int64_t>()[0];
    if (actualSharedPrefixLenData > keySharedPrefixS) {
        std::string reason = "The value of actual_shared_prefix_len cannot be greater than S of "
            "key_shared_prefix: " + std::to_string(keySharedPrefixS);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "actual_shared_prefix_len",
            std::to_string(actualSharedPrefixLenData).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (actualSharedPrefixLenData > valueSharedPrefixS) {
        std::string reason = "The value of actual_shared_prefix_len cannot be greater than S of "
            "value_shared_prefix(" + std::to_string(valueSharedPrefixS) + ")";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "actual_shared_prefix_len",
            std::to_string(actualSharedPrefixLenData).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (actualSharedPrefixLenData < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "actual_shared_prefix_len",
            std::to_string(actualSharedPrefixLenData).c_str(),
            "The value of actual_shared_prefix_len cannot be less than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// existence
ge::graphStatus SystemPrefixChecker::CheckSharedPrefixExistence(const FiaTilingInfo &fiaInfo)
{
    // 校验keySharedPrefix和valueSharedPrefix的存在性
    auto &keySharedPrefixTensor = fiaInfo.opParamInfo.keySharedPrefix.tensor;
    auto &valueSharedPrefixTensor = fiaInfo.opParamInfo.valueSharedPrefix.tensor;
    // 两者要么都为空，要么都不为空
    OP_CHECK_IF((keySharedPrefixTensor != nullptr && valueSharedPrefixTensor == nullptr),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "valueSharedPrefix",
            "valueSharedPrefix cannot be empty when keySharedPrefix is not empty"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((keySharedPrefixTensor == nullptr && valueSharedPrefixTensor != nullptr),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "keySharedPrefix",
            "keySharedPrefix cannot be empty when valueSharedPrefix is not empty"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// feature
ge::graphStatus SystemPrefixChecker::CheckUnSupportFeature(const FiaTilingInfo &fiaInfo)
{
    if (!fiaInfo.sysPrefixFlag) {
        // 若不使能systemPrefix，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    // 校验不支持带prefix的feautre
    bool enableIFAMLA = fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512;
    bool enablePFARope = fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D128;
    bool enablePFAMLA = fiaInfo.mlaMode == MlaMode::ROPE_COMBINE_D128;
    // 不支持page attention场景
    OP_CHECK_IF((fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "prefix",
            "prefix must be empty when page attention is enabled"),
        return ge::GRAPH_FAILED);
    // 不支持tensorlist
    if ((fiaInfo.s1Size > 1U && fiaInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B) ||
        fiaInfo.npuArch == NpuArch::DAV_3510) {
        OP_CHECK_IF((fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "prefix",
                "prefix must be empty when tensorlist is enabled"),
            return ge::GRAPH_FAILED);
    }
    // 不支持左padding场景
    OP_CHECK_IF((fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "prefix",
            "prefix must be empty when left padding is enabled"),
        return ge::GRAPH_FAILED);
    // 不支持alibi场景
    OP_CHECK_IF((fiaInfo.enableAlibiPse),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "prefix",
            "prefix must be empty when pseType is 2 or 3"),
        return ge::GRAPH_FAILED);
     OP_CHECK_IF(layoutStr == "BSH_BNSD" ||
                    layoutStr == "BSND_BNSD" ||
                    layoutStr == "TND" ||
                    layoutStr == "NTD" ||
                    layoutStr == "NTD_TND" ||
                    layoutStr == "TND_NTD",
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "prefix",
            "prefix must be empty when the inputLayout is BSH_BNSD/BSND_BNSD/TND/NTD/TND_NTD/NTD_TND"),
        return ge::GRAPH_FAILED);
    // 不支持PFA MLA场景
    OP_CHECK_IF((enablePFAMLA || enablePFARope),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "prefix",
            "prefix must be empty in PFA MLA scenario"),
        return ge::GRAPH_FAILED);
    // 不支持IFA MLA场景
    OP_CHECK_IF((enableIFAMLA),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "prefix",
            "prefix must be empty in IFA MLA scenario"),
        return ge::GRAPH_FAILED);
    // 不支持qkv全部为INT8/FP8/HiF8(per-block/per-tensor全量化)的情况
    OP_CHECK_IF((fiaInfo.inputQType == ge::DT_INT8 && fiaInfo.inputKvType == ge::DT_INT8),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "prefix",
            "prefix must be empty when the dtype of query and key/value is INT8"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((fiaInfo.inputQType == ge::DT_FLOAT8_E4M3FN && fiaInfo.inputKvType == ge::DT_FLOAT8_E4M3FN),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "prefix",
            "prefix must be empty when the dtype of query and key/value is FP8"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((fiaInfo.inputQType == ge::DT_HIFLOAT8 && fiaInfo.inputKvType == ge::DT_HIFLOAT8),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "prefix",
            "prefix must be empty when the dtype of query and key/value is HiF8"),
        return ge::GRAPH_FAILED);
    // 后量化仅支持INT8
    if (fiaInfo.isOutQuantEnable) {
        if (fiaInfo.outputType != ge::DT_INT8) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "output", ToString(fiaInfo.outputType).c_str(),
                "The dtype of output must be INT8 when prefix is enabled");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SystemPrefixChecker::CheckFeatureAntiquantS1Gt1(const FiaTilingInfo &fiaInfo,
    int64_t keyAntiquantMode, int64_t valueAntiquantMode)
{ // Q_S > 1，per-channel(per-tensor)模式，key/value，仅支持INT8
    if (keyAntiquantMode == PER_CHANNEL_MODE && valueAntiquantMode == PER_CHANNEL_MODE) {
        if (fiaInfo.inputKvType != ge::DT_INT8) {
            std::string reason = "The dtype of key/value must be INT8 when prefix is enabled, S of query > 1, "
                "keyAntiquantMode is per-channel(per-tensor) mode and valueAntiquantMode is "
                "per-channel(per-tensor) mode";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key/value",
                ToString(fiaInfo.inputKvType).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (keyAntiquantMode == PER_TOKEN_MODE && valueAntiquantMode == PER_TOKEN_MODE) {
        if (fiaInfo.inputKvType != ge::DT_INT8) {
            std::string reason = "The dtype of key/value must be INT8 when prefix is enabled, S of query > 1, "
                "keyAntiquantMode is per-token mode and valueAntiquantMode is per-token mode";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key/value",
                ToString(fiaInfo.inputKvType).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    } else { // 其余量化模式在Q_S > 1时均为非法
        std::string valueStr = std::to_string(keyAntiquantMode) + " and " + std::to_string(valueAntiquantMode);
        std::string reason = "key_antiquant_mode(" + std::to_string(keyAntiquantMode) +
            ") and value_antiquant_mode(" + std::to_string(valueAntiquantMode) + ") is "
            "not supported when prefix is enabled and S of query > 1";
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(fiaInfo.opName, "key_antiquant_mode and value_antiquant_mode",
            valueStr.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SystemPrefixChecker::CheckFeatureAntiquantS1Eq1(const FiaTilingInfo &fiaInfo,
    int64_t keyAntiquantMode, int64_t valueAntiquantMode)
{ // Q_S = 1
    auto &keyAntiquantScaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
    auto &valueAntiquantScaleTensor = fiaInfo.opParamInfo.valueAntiquantScale.tensor;
    if (keyAntiquantScaleTensor == nullptr || valueAntiquantScaleTensor == nullptr) {
        // 若不存在keyAntiquantScaleTensor和valueAntiquantScaleTensor，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    gert::Shape keyAntiquantScaleTensorShape = keyAntiquantScaleTensor->GetStorageShape();
    uint32_t keyAntiquantScaleTensorDimNum = keyAntiquantScaleTensorShape.GetDimNum();
    if (keyAntiquantMode == PER_CHANNEL_MODE && valueAntiquantMode == PER_CHANNEL_MODE) {
        if (keyAntiquantScaleTensorDimNum == DIM_NUM_1) {
            if (fiaInfo.inputKvType != ge::DT_INT8) {
                std::string reason = "The dtype of key/value must be INT8 when prefix is enabled, S of query is 1, "
                    "keyAntiquantMode is per-tensor mode and valueAntiquantMode is per-tensor mode";
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key/value",
                    ToString(fiaInfo.inputKvType).c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
        } else {
            if (fiaInfo.inputKvType != ge::DT_INT8 && fiaInfo.inputKvType != ge::DT_INT4) {
                std::string reason = "The dtype of key/value must be INT8 or INT4(INT32) when prefix is enabled"
                    ", S of query is 1, keyAntiquantMode is per-channel mode and valueAntiquantMode "
                    "is per-channel mode";
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key/value",
                    ToString(fiaInfo.inputKvType).c_str(), reason.c_str());
                return ge::GRAPH_FAILED;
            }
        }
    } else if (keyAntiquantMode == PER_TOKEN_MODE && valueAntiquantMode == PER_TOKEN_MODE) { // per-token模式，支持INT8或INT4(INT32)
        if (fiaInfo.inputKvType != ge::DT_INT8 && fiaInfo.inputKvType != ge::DT_INT4) {
            std::string reason = "The dtype of key/value must be INT8 or INT4(INT32) when prefix is enabled"
                ", S of query is 1, keyAntiquantMode is per-token mode and valueAntiquantMode is per-token mode";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key/value",
                ToString(fiaInfo.inputKvType).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (keyAntiquantMode == PER_TENSOR_HEAD_MODE && valueAntiquantMode == PER_TENSOR_HEAD_MODE) { // per-tensor-head模式，仅支持INT8
        if (fiaInfo.inputKvType != ge::DT_INT8) {
            std::string reason = "The dtype of key/value must be INT8 when prefix is enabled, S of query is 1, "
                "keyAntiquantMode is per-tensor-head mode and valueAntiquantMode is per-tensor-head mode";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key/value",
                ToString(fiaInfo.inputKvType).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (keyAntiquantMode == PER_TOKEN_HEAD_MODE && valueAntiquantMode == PER_TOKEN_HEAD_MODE) {
        if (fiaInfo.inputKvType != ge::DT_INT8 && fiaInfo.inputKvType != ge::DT_INT4) {
            std::string reason = "The dtype of key/value must be INT8 or INT4(INT32) when prefix is enabled"
                ", S of query is 1, keyAntiquantMode is per-token-head mode and valueAntiquantMode "
                "is per-token-head mode";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key/value",
                ToString(fiaInfo.inputKvType).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (keyAntiquantMode == PER_TOKEN_PA_MODE && valueAntiquantMode == PER_TOKEN_PA_MODE) {
        if (fiaInfo.inputKvType != ge::DT_INT8) {
            std::string reason = "The dtype of key/value must be INT8 when prefix is enabled, S of query is 1, "
                "keyAntiquantMode is per-tensor-PA mode and valueAntiquantMode is per-tensor-PA mode";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key/value",
                ToString(fiaInfo.inputKvType).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (keyAntiquantMode == PER_TOKEN_HEAD_PA_MODE && valueAntiquantMode == PER_TOKEN_HEAD_PA_MODE) {
        if (fiaInfo.inputKvType != ge::DT_INT8) {
            std::string reason = "The dtype of key/value must be INT8 when prefix is enabled, S of query is 1, "
                "keyAntiquantMode is per-token-head-PA mode and valueAntiquantMode is per-token-head-PA mode";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key/value",
                ToString(fiaInfo.inputKvType).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (keyAntiquantMode == PER_CHANNEL_MODE && valueAntiquantMode == PER_TOKEN_MODE) {
        if (fiaInfo.inputKvType != ge::DT_INT8 && fiaInfo.inputKvType != ge::DT_INT4) {
            std::string reason = "The dtype of key/value must be INT8 or INT4(INT32) when prefix is enabled"
                ", S of query is 1, keyAntiquantMode is per-channel mode and valueAntiquantMode is per-token mode";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key/value",
                ToString(fiaInfo.inputKvType).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        std::string valueStr = std::to_string(keyAntiquantMode) + " and " + std::to_string(valueAntiquantMode);
        std::string reason = "key_antiquant_mode(" + std::to_string(keyAntiquantMode) + ") and "
            "value_antiquant_mode(" + std::to_string(valueAntiquantMode) + ") is not supported "
            "when prefix is enabled and S of query is 1";
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(fiaInfo.opName, "key_antiquant_mode and value_antiquant_mode",
            valueStr.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SystemPrefixChecker::CheckFeatureAntiquant(const FiaTilingInfo &fiaInfo)
{ // 校验prefix叠加伪量化的支持场景
    if (!fiaInfo.sysPrefixFlag) {
        // 若不使能systemPrefix，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    int64_t keyAntiquantMode = 0;
    int64_t valueAntiquantMode = 0;
    if (fiaInfo.opParamInfo.keyAntiquantMode != nullptr) {
        keyAntiquantMode = *fiaInfo.opParamInfo.keyAntiquantMode;
    }
    if (fiaInfo.opParamInfo.valueAntiquantMode != nullptr) {
        valueAntiquantMode = *fiaInfo.opParamInfo.valueAntiquantMode;
    }
    if (fiaInfo.s1Size > 1) {
        return CheckFeatureAntiquantS1Gt1(fiaInfo, keyAntiquantMode, valueAntiquantMode);
    }
    if (fiaInfo.s1Size == 1) {
        return CheckFeatureAntiquantS1Eq1(fiaInfo, keyAntiquantMode, valueAntiquantMode);
    }
    return ge::GRAPH_SUCCESS;
}

// multipara
ge::graphStatus SystemPrefixChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SystemPrefixChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SystemPrefixChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckUnSupportFeature(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }

    if (enableAntiQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFeatureAntiquant(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SystemPrefixChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckSharedPrefixDim(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckSharedPrefixDataType(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckSharedPrefixShape(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckActualSharedPrefixLenData(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckSharedPrefixExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
