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
 * \file common_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "common_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// 公共校验函数
ge::graphStatus CommonChecker::CheckInputFormat(const FiaTilingInfo &fiaInfo)
{
    if (CheckFormatSupport(fiaInfo.opParamInfo.query.desc, "query") != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(fiaInfo.opParamInfo.key.desc, "key") != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(fiaInfo.opParamInfo.value.desc, "value") != ge::GRAPH_SUCCESS ||
        CheckFormatSupport(fiaInfo.opParamInfo.attenOut.desc, "attentionOut") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckParaExistenceImpl(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.opParamInfo.query.desc == nullptr || fiaInfo.opParamInfo.query.shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(fiaInfo.opName, "query");
        return ge::GRAPH_FAILED;
    }
    if (fiaInfo.opParamInfo.key.desc == nullptr || fiaInfo.opParamInfo.key.shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(fiaInfo.opName, "key");
        return ge::GRAPH_FAILED;
    }
    if (fiaInfo.opParamInfo.value.desc == nullptr || fiaInfo.opParamInfo.value.shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(fiaInfo.opName, "value");
        return ge::GRAPH_FAILED;
    }
    if (fiaInfo.opParamInfo.attenOut.desc == nullptr || fiaInfo.opParamInfo.attenOut.shape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(fiaInfo.opName, "attention_out");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckDtypeCommon(const gert::CompileTimeTensorDesc *desc,
    const std::string &name, std::map<std::string, std::vector<ge::DataType>> dataMap)
{
    if (desc != nullptr) {
        const auto& it = dataMap.find(name);
        OP_CHECK_IF(it == dataMap.end(),
            OP_LOGE("FIA", "%s datatype support list should be specify in map", name.c_str()),
            return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        OP_CHECK_IF(std::find(expectDtypeList.begin(), expectDtypeList.end(),
         desc->GetDataType()) == expectDtypeList.end(),
            "", // LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name), //公共打印函数
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckPAKeyValue(const FiaTilingInfo &fiaInfo)
{
    const gert::StorageShape *keyShape = fiaInfo.opParamInfo.key.shape;
    const gert::StorageShape *valueShape = fiaInfo.opParamInfo.value.shape;
    uint32_t keyDimNum = keyShape->GetStorageShape().GetDimNum();
    uint32_t keyBlockNum = fiaInfo.totalBlockNum;
    uint32_t keyHeadNum = fiaInfo.n2Size;
    uint32_t keyBlockSize = fiaInfo.blockSize;
    uint32_t keyHeadDim = fiaInfo.qkHeadDim;
    uint32_t keyH = fiaInfo.qkHeadDim * fiaInfo.n2Size;
    uint32_t keyD0 = 16;
    uint32_t keyD1 = fiaInfo.qkHeadDim / 16; // 当前PA NZ场景D 16

    uint32_t valueDimNum = valueShape->GetStorageShape().GetDimNum();
    uint32_t valueBlockNum = fiaInfo.totalBlockNum;
    uint32_t valueHeadNum = fiaInfo.n2Size;
    uint32_t valueBlockSize = fiaInfo.blockSize;
    uint32_t valueHeadDim = fiaInfo.vHeadDim;
    uint32_t valueH = fiaInfo.vHeadDim * fiaInfo.n2Size;
    uint32_t valueD0 = 16;
    uint32_t valueD1 = fiaInfo.vHeadDim / 16; // 当前PA NZ场景D 16
    if (keyDimNum != valueDimNum) {
        std::string dimMsg = std::to_string(keyDimNum) + " and " + std::to_string(valueDimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(fiaInfo.opName, "key and value",
            dimMsg.c_str(), "The shape dim of key must be equal to that of value");
        return ge::GRAPH_FAILED;
    }

    if (keyDimNum == 3) {                    // BBH
        keyBlockNum = keyShape->GetStorageShape().GetDim(0);
        keyBlockSize = keyShape->GetStorageShape().GetDim(1);
        keyH = keyShape->GetStorageShape().GetDim(2);
        valueBlockNum = valueShape->GetStorageShape().GetDim(0);
        valueBlockSize = valueShape->GetStorageShape().GetDim(1);
        valueH = valueShape->GetStorageShape().GetDim(2);
    } else if (keyDimNum == 4) {            // BNBD
        keyBlockNum = keyShape->GetStorageShape().GetDim(0);
        keyHeadNum = keyShape->GetStorageShape().GetDim(1);
        keyBlockSize = keyShape->GetStorageShape().GetDim(2);
        keyHeadDim = keyShape->GetStorageShape().GetDim(3);
        valueBlockNum = valueShape->GetStorageShape().GetDim(0);
        valueHeadNum = valueShape->GetStorageShape().GetDim(1);
        valueBlockSize = valueShape->GetStorageShape().GetDim(2);
        valueHeadDim = valueShape->GetStorageShape().GetDim(3);
    } else if (keyDimNum == 5) {           // BND1BD0
        keyBlockNum = keyShape->GetStorageShape().GetDim(0);
        keyHeadNum = keyShape->GetStorageShape().GetDim(1);
        keyD1 = keyShape->GetStorageShape().GetDim(2);
        keyBlockSize = keyShape->GetStorageShape().GetDim(3);
        keyD0 = keyShape->GetStorageShape().GetDim(4);

        valueBlockNum = valueShape->GetStorageShape().GetDim(0);
        valueHeadNum = valueShape->GetStorageShape().GetDim(1);
        valueD1 = valueShape->GetStorageShape().GetDim(2);
        valueBlockSize = valueShape->GetStorageShape().GetDim(3);
        valueD0 = valueShape->GetStorageShape().GetDim(4);
    } else {
        std::string dimStr = std::to_string(keyDimNum);
        std::string reason = "The shape dim of key must be within the range of [" +
            std::to_string(INPUT_KV_SHAPE_MIN_DIMS) + ", " + std::to_string(INPUT_KV_SHAPE_MAX_DIMS) + "]";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "key",
            dimStr.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    if (fiaInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B && fiaInfo.kvLayout == FiaLayout::NZ && fiaInfo.ropeMode == RopeMode::NO_ROPE) {
        const std::vector<std::int32_t> nzNoRopeDSupportList = {64, 128};
        if (std::find(nzNoRopeDSupportList.begin(), nzNoRopeDSupportList.end(), keyHeadDim) ==
                nzNoRopeDSupportList.end() ||
            std::find(nzNoRopeDSupportList.begin(), nzNoRopeDSupportList.end(), valueHeadDim) ==
                nzNoRopeDSupportList.end()) {
            OP_LOGE(fiaInfo.opName,
                    "In %s %s situation, when the dim of key&value is 5, and the headDim shared by query and key is "
                    "equal to that of value. The headDim of query|key|value should be 64 | "
                    "128, but got valueHeadDim:%u, queryHeadDim and keyHeadDim:%u",
                    QuantModeToSerialString(fiaInfo.quantMode).c_str(), SituationToSerialString(fiaInfo.ropeMode).c_str(), valueHeadDim,
                    keyHeadDim);
            return ge::GRAPH_FAILED;
        }
    }

    if ((keyDimNum == 5) && ((keyBlockNum != valueBlockNum) || (keyHeadNum != valueHeadNum) ||
        ((keyD1 != valueD1) && (fiaInfo.mlaMode != MlaMode::ROPE_COMBINE_D128)) || (keyBlockSize != valueBlockSize) ||
        ((keyD0 != valueD0) && (fiaInfo.mlaMode != MlaMode::ROPE_COMBINE_D128)))) {
        std::string shapeMsg = ToString(keyShape->GetStorageShape()) + " and " +
            ToString(valueShape->GetStorageShape());
        std::string reason = "The dim num of key and value are inconsistent when page attention is enabled";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key and value", shapeMsg.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if ((keyDimNum == 4) && ((keyBlockNum != valueBlockNum) || (keyHeadNum != valueHeadNum) ||
        (keyBlockSize != valueBlockSize) || ((keyHeadDim != valueHeadDim) &&
        (fiaInfo.mlaMode != MlaMode::ROPE_COMBINE_D128)))) {
        std::string shapeMsg = ToString(keyShape->GetStorageShape()) + " and " +
            ToString(valueShape->GetStorageShape());
        std::string reason = "The dim num of key and value are inconsistent when page attention is enabled";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key and value", shapeMsg.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if ((keyDimNum == 3) && ((keyBlockNum != valueBlockNum) || (keyBlockSize != valueBlockSize) ||
        ((keyH != valueH) && (fiaInfo.mlaMode != MlaMode::ROPE_COMBINE_D128)))) {
        std::string shapeMsg = ToString(keyShape->GetStorageShape()) + " and " +
            ToString(valueShape->GetStorageShape());
        std::string reason = "The dim num of key and value are inconsistent when page attention is enabled";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key and value", shapeMsg.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    if (keyHeadDim != fiaInfo.qkHeadDim) {
        std::string reason = "D of key must be equal to " + std::to_string(fiaInfo.qkHeadDim);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key",
            ToString(keyShape->GetStorageShape()).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (valueHeadDim != fiaInfo.vHeadDim) {
        std::string reason = "D of value must be equal to " + std::to_string(fiaInfo.vHeadDim);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "value",
            ToString(valueShape->GetStorageShape()).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (keyH != fiaInfo.qkHeadDim * fiaInfo.n2Size) {
        std::string reason = "H of key must be equal to " + std::to_string(fiaInfo.qkHeadDim * fiaInfo.n2Size);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key",
            ToString(keyShape->GetStorageShape()).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (valueH != fiaInfo.vHeadDim * fiaInfo.n2Size) {
        std::string reason = "H of value must be equal to " + std::to_string(fiaInfo.vHeadDim * fiaInfo.n2Size);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "value",
            ToString(valueShape->GetStorageShape()).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (enableNonQuant_) {
        if (keyD0 != 16 || valueD0 != 16) {
            std::string valMsg = ToString(keyShape->GetStorageShape()) + " and " +
                ToString(valueShape->GetStorageShape());
            std::string reason = "The last axis of key and value must both be equal "
                "to 16 when layout is PA_NZ";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key and value",
                valMsg.c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
        if (keyD1 != fiaInfo.qkHeadDim  / 16) {
            std::string reason = "D1 of key must be equal to " + std::to_string(fiaInfo.qkHeadDim / 16);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key",
                ToString(keyShape->GetStorageShape()).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
        if (valueD1 != fiaInfo.vHeadDim / 16) {
            std::string reason = "D1 of value must be equal to " + std::to_string(fiaInfo.vHeadDim / 16);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "value",
                ToString(valueShape->GetStorageShape()).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

bool CommonChecker::CheckEmptyTensorList(const FiaTilingInfo &fiaInfo)
{
    for (int64_t tmpIdx = 0; tmpIdx < fiaInfo.kCache.size(); ++tmpIdx) {
        if (fiaInfo.kCache[tmpIdx]->GetStorageShape().GetShapeSize() != 0) {
            return false;
        }
        if (fiaInfo.vCache[tmpIdx]->GetStorageShape().GetShapeSize() != 0) {
            return false;
        }
    }
    return true;
}

bool CommonChecker::CheckNormalTensorListBSH(const FiaTilingInfo &fiaInfo)
{ // check all H across batches and KVs are the same under BSH layout
    auto standardKH = fiaInfo.kCache[0]->GetStorageShape().GetDim(2);
    auto standardVH = fiaInfo.vCache[0]->GetStorageShape().GetDim(2);
    int64_t tmpNKv = (fiaInfo.n2Size != 0) ? fiaInfo.n2Size : fiaInfo.n1Size;
    int64_t keyRopeS = 0;
    if (fiaInfo.opParamInfo.keyRope.tensor != nullptr) {
        keyRopeS = fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(1);
        if (fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(0) != fiaInfo.kCache.size()) {
            std::string valuesStr = std::to_string(fiaInfo.kCache.size()) + " and " +
                std::to_string(fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(0));
            std::string reason = "The values of Batch of key and B of key_rope must be the "
                "same in the tensorlist scenario";
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(fiaInfo.opName, "Batch of key and B of key_rope",
                valuesStr.c_str(), reason.c_str());
            return false;
        }
    }

    for (int64_t tmpIdx = 0; tmpIdx < fiaInfo.kCache.size(); ++tmpIdx) {
        // 2: The second dimension of the tensorlist represents n, in order to check whether all n in the tensorlist are the same.
        if (fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(2) != standardKH) {
            std::string paramName = "key in the " + std::to_string(tmpIdx + 1) + "th tensor";
            std::string shapeStr = ToStringRaw(fiaInfo.kCache[tmpIdx]->GetStorageShape());
            std::string reason = "H of key in the " + std::to_string(tmpIdx + 1) +
                "th tensor must be equal to axis H of key in the 1st tensor in the tensorlist scenario";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, paramName.c_str(),
                shapeStr.c_str(), reason.c_str());
            return false;
        }
        if (fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(2) != standardVH) {
            std::string paramName = "value in the " + std::to_string(tmpIdx + 1) + "th tensor";
            std::string shapeStr = ToStringRaw(fiaInfo.vCache[tmpIdx]->GetStorageShape());
            std::string reason = "H of value in the " + std::to_string(tmpIdx + 1) +
                "th tensor must be equal to axis H of value in the 1st tensor in the tensorlist scenario";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, paramName.c_str(),
                shapeStr.c_str(), reason.c_str());
            return false;
        }
        if (fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1) != fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(1)) { // k_s != v_s
            std::string shapesStr = ToString(fiaInfo.kCache[tmpIdx]->GetStorageShape()) + " and " +
                ToString(fiaInfo.vCache[tmpIdx]->GetStorageShape());
            std::string reason = "S of key must be equal to S of value";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key and value",
                shapesStr.c_str(), reason.c_str());
            return false;
        }
        if (fiaInfo.opParamInfo.keyRope.tensor != nullptr) {
            if (fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1) != keyRopeS) { // k_s != krope_s
                std::string shapesStr = ToString(fiaInfo.kCache[tmpIdx]->GetStorageShape()) + " and " +
                    ToString(fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape());
                std::string reason = "S of key must be equal to S of key_rope";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key and key_rope",
                    shapesStr.c_str(), reason.c_str());
                return false;
            }
        }
    }
    return true;
}

bool CommonChecker::CheckNormalTensorListBNSD(const FiaTilingInfo &fiaInfo)
{ // check N and D, respectively, are the same
    // across batches and KVs under BNSD/BNSD_BSND/BNSD_NBSD
    auto standardN = fiaInfo.kCache[0]->GetStorageShape().GetDim(1);
    auto standardKD = fiaInfo.kCache[0]->GetStorageShape().GetDim(3);
    auto standardVD = fiaInfo.vCache[0]->GetStorageShape().GetDim(3);
    int64_t tmpNKv = (fiaInfo.n2Size != 0) ? fiaInfo.n2Size : fiaInfo.n1Size;
    int64_t keyRopeS = 0;

    if (fiaInfo.opParamInfo.keyRope.tensor != nullptr) {
        keyRopeS = fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(2);
        if (fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(0) != fiaInfo.kCache.size()) {
            std::string valuesStr = std::to_string(fiaInfo.kCache.size()) + " and " +
                std::to_string(fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(0));
            std::string reason = "The values of Batch of key and B of key_rope must be the "
                "same in the tensorlist scenario";
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(fiaInfo.opName, "Batch of key and B of key_rope",
                valuesStr.c_str(), reason.c_str());
            return false;
        }
    }

    if (tmpNKv != standardN) {
        std::string reason = "N of key 1st tensor must be equal to numKeyValueHeads: " +
            std::to_string(tmpNKv);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key 1st tensor",
            ToString(fiaInfo.kCache[0]->GetStorageShape()).c_str(), reason.c_str());
        return false;
    }

    for (int64_t tmpIdx = 0; tmpIdx < fiaInfo.kCache.size(); ++tmpIdx) {
        if ((fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1) != standardN) ||
            (fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(1) != standardN)) {
            std::string paramName = "key in the " + std::to_string(tmpIdx + 1) + "th tensor and value in the " +
                std::to_string(tmpIdx + 1) + "th tensor";
            std::string shapeStr = ToString(fiaInfo.kCache[tmpIdx]->GetStorageShape()) + " and " +
                ToString(fiaInfo.vCache[tmpIdx]->GetStorageShape());
            std::string reason = "N of key and value in the " + std::to_string(tmpIdx + 1) +
                "th tensor must be equal to num_key_value_heads: " + std::to_string(standardN) +
                " in the tensorlist scenario";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, paramName.c_str(),
                shapeStr.c_str(), reason.c_str());
            return false;
        }
        if (fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(3) != standardKD) {
            std::string paramName = "key in the " + std::to_string(tmpIdx + 1) + "th tensor";
            std::string shapeStr = ToStringRaw(fiaInfo.kCache[tmpIdx]->GetStorageShape());
            std::string reason = "D of key in the " + std::to_string(tmpIdx + 1) +
                "th tensor must be equal to axis D of key in the 1st tensor in the tensorlist scenario";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, paramName.c_str(),
                shapeStr.c_str(), reason.c_str());
            return false;
        }
        if (fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(3) != standardVD) {
            std::string paramName = "value in the " + std::to_string(tmpIdx + 1) + "th tensor";
            std::string shapeStr = ToStringRaw(fiaInfo.vCache[tmpIdx]->GetStorageShape());
            std::string reason = "D of value in the " + std::to_string(tmpIdx + 1) +
                "th tensor must be equal to axis D of value in the 1st tensor in the tensorlist scenario";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, paramName.c_str(),
                shapeStr.c_str(), reason.c_str());
            return false;
        }
        if (fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(2) != // 2: Obtain the second dimension
            fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(2)) { // 2: Obtain the second dimension
            std::string shapesStr = ToString(fiaInfo.kCache[tmpIdx]->GetStorageShape()) + " and " +
                ToString(fiaInfo.vCache[tmpIdx]->GetStorageShape());
            std::string reason = "S of key must be equal to S of value";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key and value",
                shapesStr.c_str(), reason.c_str());
            return false;
        }
        if (fiaInfo.opParamInfo.keyRope.tensor != nullptr) {
            if (fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(2) != keyRopeS) { // k_s != krope_s
                std::string shapesStr = ToString(fiaInfo.kCache[tmpIdx]->GetStorageShape()) + " and " +
                    ToString(fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape());
                std::string reason = "S of key must be equal to S of key_rope";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key and key_rope",
                    shapesStr.c_str(), reason.c_str());
                return false;
            }
        }
    }
    return true;
}

bool CommonChecker::CheckNormalTensorListBSND(const FiaTilingInfo &fiaInfo)
{ // check N and D, respectively, are the same across batches and KVs under BSND
    auto standardN = fiaInfo.kCache[0]->GetStorageShape().GetDim(2);
    auto standardKD = fiaInfo.kCache[0]->GetStorageShape().GetDim(3);
    auto standardVD = fiaInfo.vCache[0]->GetStorageShape().GetDim(3);
    int64_t tmpNKv = (fiaInfo.n2Size != 0) ? fiaInfo.n2Size : fiaInfo.n1Size;
    int64_t keyRopeS = 0;

    if (fiaInfo.opParamInfo.keyRope.tensor != nullptr) {
        keyRopeS = fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(1);
        if (fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(0) != fiaInfo.kCache.size()) {
            std::string valuesStr = std::to_string(fiaInfo.kCache.size()) + " and " +
                std::to_string(fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape().GetDim(0));
            std::string reason = "The values of Batch of key and B of key_rope must be the "
                "same in the tensorlist scenario";
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(fiaInfo.opName, "Batch of key and B of key_rope",
                valuesStr.c_str(), reason.c_str());
            return false;
        }
    }

    if (tmpNKv != standardN) {
        std::string reason = "N of key 1st tensor must be equal to num_key_value_heads: " +
            std::to_string(tmpNKv);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key 1st tensor",
            ToString(fiaInfo.kCache[0]->GetStorageShape()).c_str(), reason.c_str());
        return false;
    }

    for (int64_t tmpIdx = 0; tmpIdx < fiaInfo.kCache.size(); ++tmpIdx) {
        if ((fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(2) != standardN) || // 2: The second dimension of the tensorlist represents n, in order to check whether all n in the tensorlist are the same.
            (fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(2) != standardN)) { // 2: The second dimension of the tensorlist represents n, in order to check whether all n in the tensorlist are the same.
            std::string paramName = "key in the " + std::to_string(tmpIdx + 1) + "th tensor and value in the " +
                std::to_string(tmpIdx + 1) + "th tensor";
            std::string shapeStr = ToString(fiaInfo.kCache[tmpIdx]->GetStorageShape()) + " and " +
                ToString(fiaInfo.vCache[tmpIdx]->GetStorageShape());
            std::string reason = "N of key and value in the " + std::to_string(tmpIdx + 1) +
                "th tensor must be equal to num_key_value_heads: " + std::to_string(standardN) +
                " in the tensorlist scenario";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, paramName.c_str(),
                shapeStr.c_str(), reason.c_str());
            return false;
        }
        if (fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(3) != standardKD) {
            std::string paramName = "key in the " + std::to_string(tmpIdx + 1) + "th tensor";
            std::string shapeStr = ToStringRaw(fiaInfo.kCache[tmpIdx]->GetStorageShape());
            std::string reason = "D of key in the " + std::to_string(tmpIdx + 1) +
                "th tensor must be equal to axis D of key in the 1st tensor in the tensorlist scenario";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, paramName.c_str(),
                shapeStr.c_str(), reason.c_str());
            return false;
        }
        if (fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(3) != standardVD) {
            std::string paramName = "value in the " + std::to_string(tmpIdx + 1) + "th tensor";
            std::string shapeStr = ToStringRaw(fiaInfo.vCache[tmpIdx]->GetStorageShape());
            std::string reason = "D of value in the " + std::to_string(tmpIdx + 1) +
                "th tensor must be equal to axis D of value in the 1st tensor in the tensorlist scenario";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, paramName.c_str(),
                shapeStr.c_str(), reason.c_str());
            return false;
        }
        if (fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1) != fiaInfo.vCache[tmpIdx]->GetStorageShape().GetDim(1)) {
            std::string shapesStr = ToString(fiaInfo.kCache[tmpIdx]->GetStorageShape()) + " and " +
                ToString(fiaInfo.vCache[tmpIdx]->GetStorageShape());
            std::string reason = "S of key must be equal to S of value";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key and value",
                shapesStr.c_str(), reason.c_str());
            return false;
        }
        if (fiaInfo.opParamInfo.keyRope.tensor != nullptr) {
            if (fiaInfo.kCache[tmpIdx]->GetStorageShape().GetDim(1) != keyRopeS) { // k_s != krope_s
                std::string shapesStr = ToString(fiaInfo.kCache[tmpIdx]->GetStorageShape()) + " and " +
                    ToString(fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape());
                std::string reason = "S of key must be equal to S of key_rope";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key and key_rope",
                    shapesStr.c_str(), reason.c_str());
                return false;
            }
        }
    }
    return true;
}

bool CommonChecker::CheckNormalTensorList(const FiaTilingInfo &fiaInfo)
{
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    if (layoutStr == "BSH") {
        return CheckNormalTensorListBSH(fiaInfo);
    }
    if (layoutStr == "BNSD" || layoutStr == "BNSD_BSND" || layoutStr == "BNSD_NBSD") {
        return CheckNormalTensorListBNSD(fiaInfo);
    }
    return CheckNormalTensorListBSND(fiaInfo);
}

ge::graphStatus CommonChecker::CheckTensorList(const FiaTilingInfo &fiaInfo)
{
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    OP_CHECK_IF((fiaInfo.opParamInfo.blockTable.tensor != nullptr),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "blockTable",
            "blockTable must be empty in the tensorlist scenario"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((layoutStr == "TND" || layoutStr == "NTD" || layoutStr == "NTD_TND" || layoutStr == "TND_NTD"),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "input_layout",
            "input_layout TND/NTD/TND_NTD/NTD_TND is not supported in the tensorlist scenario"),
        return ge::GRAPH_FAILED);
    if (fiaInfo.bSize > B_LIMIT) {
        std::string reason = "B of query must be less than or equal to " + std::to_string(B_LIMIT) +
            " in the tensorlist scenario";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "query",
            ToStringRaw(fiaInfo.opParamInfo.query.shape->GetStorageShape()).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (CheckEmptyTensorList(fiaInfo)) {
        return ge::GRAPH_SUCCESS;
    }
    if (!CheckNormalTensorList(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    if ((CheckQueryKeyTensorlistConsistency(fiaInfo)) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMultiDtype(const FiaTilingInfo &fiaInfo)
{
    const std::map<std::string, std::vector<ge::DataType>> QKVD_Different_MAP = {
        {"query",                  {ge::DT_FLOAT16, ge::DT_BF16}},
        {"key",                    {ge::DT_FLOAT16, ge::DT_BF16}},
        {"value",                  {ge::DT_FLOAT16, ge::DT_BF16}},
        {"attentionOut",           {ge::DT_FLOAT16, ge::DT_BF16}},
    };
    OP_CHECK_IF(fiaInfo.isQKVDDifferent &&
        ge::GRAPH_SUCCESS != CheckDtypeCommon(fiaInfo.opParamInfo.query.desc, "query", QKVD_Different_MAP),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "query",
            ToString(fiaInfo.opParamInfo.query.desc->GetDataType()).c_str(),
            "The dtype of query must be within the range of FLOAT16 or BF16 "
            "when D of query and key is not equal to D of value"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(fiaInfo.isQKVDDifferent &&
        ge::GRAPH_SUCCESS != CheckDtypeCommon(fiaInfo.opParamInfo.attenOut.desc, "attentionOut", QKVD_Different_MAP),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "attention_out",
            ToString(fiaInfo.opParamInfo.attenOut.desc->GetDataType()).c_str(),
            "The dtype of attention_out must be within the range of FLOAT16 or BF16 "
            "when D of query and key is not equal to D of value"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(fiaInfo.isQKVDDifferent &&
        ge::GRAPH_SUCCESS != CheckDtypeCommon(fiaInfo.opParamInfo.key.desc, "key", QKVD_Different_MAP),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key",
            ToString(fiaInfo.opParamInfo.key.desc->GetDataType()).c_str(),
            "The dtype of key must be within the range of FLOAT16 or BF16 "
            "when D of query and key is not equal to D of value"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(fiaInfo.isQKVDDifferent &&
        ge::GRAPH_SUCCESS != CheckDtypeCommon(fiaInfo.opParamInfo.value.desc, "value", QKVD_Different_MAP),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "value",
 	        ToString(fiaInfo.opParamInfo.value.desc->GetDataType()).c_str(),
 	        "The dtype of value must be within the range of FLOAT16 or BF16 "
            "when D of query and key is not equal to D of value"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckAxis(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.bSize > B_LIMIT || fiaInfo.bSize <= 0) {
        std::string reason = "The value of B must be within the range (0, " + std::to_string(B_LIMIT) + "]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "axis B",
            std::to_string(fiaInfo.bSize).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (fiaInfo.s1Size < 0) {
        std::string reason = "S of query must be greater than or equal to 0";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "query",
            ToStringRaw(fiaInfo.opParamInfo.query.shape->GetStorageShape()).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (fiaInfo.s2Size < 0) {
        std::string reason = "S of key must be greater than or equal to 0";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key",
            ToStringRaw(fiaInfo.opParamInfo.key.shape->GetStorageShape()).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (fiaInfo.qkHeadDim > D_LIMIT || fiaInfo.qkHeadDim < 0) {
        std::string reason = "D of query must be within the range [0, " + std::to_string(D_LIMIT) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "query",
            ToStringRaw(fiaInfo.opParamInfo.query.shape->GetStorageShape()).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (fiaInfo.vHeadDim > D_LIMIT || fiaInfo.vHeadDim < 0) {
        std::string reason = "D of value must be within the range [0, " + std::to_string(D_LIMIT) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "value",
            ToStringRaw(fiaInfo.opParamInfo.value.shape->GetStorageShape()).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (fiaInfo.isQKVDDifferent && (fiaInfo.qkHeadDim > 128)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "query",
            ToStringRaw(fiaInfo.opParamInfo.query.shape->GetStorageShape()).c_str(),
            "D of query must be <= 128 when D of query and key is not equal to D of value");
        return ge::GRAPH_FAILED;
    }
    if (fiaInfo.isQKVDDifferent && (fiaInfo.vHeadDim > 128)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "value",
            ToStringRaw(fiaInfo.opParamInfo.value.shape->GetStorageShape()).c_str(),
            "D of value must be <= 128 when D of query and key is not equal to D of value");
        return ge::GRAPH_FAILED;
    }
    if ((fiaInfo.qLayout == FiaLayout::TND || fiaInfo.qLayout == FiaLayout::NTD) && fiaInfo.qTSize < 0) {
        std::string reason = "T of query must be greater than or equal to 0 when input_layout is TND/NTD";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "query",
            ToStringRaw(fiaInfo.opParamInfo.query.shape->GetStorageShape()).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if ((fiaInfo.kvLayout == FiaLayout::TND || fiaInfo.qLayout == FiaLayout::NTD) && fiaInfo.kTSize < 0) {
        std::string reason = "T of key must be greater than or equal to 0 when input_layout is TND/NTD";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key",
            ToStringRaw(fiaInfo.opParamInfo.key.shape->GetStorageShape()).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) {
        if (fiaInfo.s1Size < 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "query",
                ToStringRaw(fiaInfo.opParamInfo.query.shape->GetStorageShape()).c_str(),
                "S of query must be >=1 in the Decode MLA scenario");
            return ge::GRAPH_FAILED;
        }
        static const std::set<uint32_t> SUPPORT_G_IN_IFAMLA = {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U}; // ifa mla场景g轴支持范围
        if ((SUPPORT_G_IN_IFAMLA.find(fiaInfo.gSize) == SUPPORT_G_IN_IFAMLA.end())) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "axis G",
                std::to_string(fiaInfo.n1Size / fiaInfo.n2Size).c_str(),
                "The value of axis G must be in the range of {1, 2, 4, 8, 16, 32, 64, 128} "
                "in the Decode MLA scenario");
            return ge::GRAPH_FAILED;
        }
    }
    OP_LOGI(fiaInfo.opName, "The axis B(%u), qkD(%u), vD(%u), G(%u), qT(%u), kT(%u).",
            fiaInfo.bSize, fiaInfo.qkHeadDim, fiaInfo.vHeadDim, fiaInfo.gSize, fiaInfo.qTSize, fiaInfo.kTSize);
    return ge::GRAPH_SUCCESS;
}

void CommonChecker::GetQueryDimAndOutDim(const gert::StorageShape* queryShape, const gert::StorageShape* outShape,
    const std::string &layoutStr, int64_t &tmpQueryDim, int64_t &outDim, uint32_t i) {
    if (layoutStr == "BNSD_BSND" || layoutStr == "BSND_BNSD") {
        if (i == 1) { // BNSD_BSND：query:N, output:S; BSND_BNSD：query:S, output:N
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i + 1);
        } else if (i == 2) { // BNSD_BSND：query:S, output:N; BSND_BNSD：query:N, output:S
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i - 1);
        }
    } else if (layoutStr == "BSH_BNSD") {
        if (i == 2) { // BSH_BNSD：query:H, output:ND
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i + 1) * outShape->GetStorageShape().GetDim(i - 1);
        }
    } else if (layoutStr == "BSND_NBSD") {
        if (i == 1) { // BSND_NBSD：query:S, output:B
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i + 1);
        } else if (i == 2) { // BSND_NBSD：query:N, output:S
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i - 2);
        }
    } else if (layoutStr == "BNSD_NBSD") {
        if (i == 1) { // BNSD_NBSD：query:N, output:B
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i - 1);
        } else if (i == 2) { // BNSD_NBSD：query:S, output:S
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i);
        }
    } else if (layoutStr == "BSH_NBSD") {
        if (i == 2) { // BSH_NBSD：query:H, output:ND
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i + 1) * outShape->GetStorageShape().GetDim(i - 2);
        }
    } else if (layoutStr == "NTD_TND" || layoutStr == "TND_NTD") {
        if (i == 0) { // query:N/T, output:T/N
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i + 1);
        } else if (i == 1) { // 2 for current queryDimNum; Q:T/N, Output:N/T
            tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i - 1);
        }
    } else {
        tmpQueryDim = queryShape->GetStorageShape().GetDim(i);
        outDim = outShape->GetStorageShape().GetDim(i);
    }
}

ge::graphStatus CommonChecker::CheckQueryOutConsistency(const FiaTilingInfo &fiaInfo)
{
    const gert::StorageShape *queryShape = fiaInfo.opParamInfo.query.shape;
    const gert::StorageShape *attentionOutShape = fiaInfo.opParamInfo.attenOut.shape;
    size_t dimNumQ = queryShape->GetStorageShape().GetDimNum();
    size_t dimNumOut = attentionOutShape->GetStorageShape().GetDimNum();
    int64_t tmpQueryDim = 0;
    int64_t tmpOutDim = 0;
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    bool isLayoutShapeSupport = layoutStr == "BSH_BNSD" || layoutStr == "BSH_NBSD";
    if (dimNumQ != dimNumOut && !isLayoutShapeSupport) {
        std::string dimsMsg = std::to_string(dimNumQ) + " and " + std::to_string(dimNumOut);
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(fiaInfo.opName, "query and attention_out",
            dimsMsg.c_str(), "The shape dim of query must be equal to that of attention_out");
        return ge::GRAPH_FAILED;
    }
    if (dimNumQ < INPUT_Q_SHAPE_MIN_DIMS || dimNumQ > INPUT_Q_SHAPE_MAX_DIMS) {
        std::string dimStr = std::to_string(dimNumQ);
        std::string reason = "The shape dim of query must be within the range of [" +
            std::to_string(INPUT_Q_SHAPE_MIN_DIMS) + ", " + std::to_string(INPUT_Q_SHAPE_MAX_DIMS) + "]";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "query",
            dimStr.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    for (uint32_t queryDimIdx = 0; queryDimIdx < dimNumQ; ++queryDimIdx) {
        if ((queryDimIdx == dimNumQ - 1) && fiaInfo.mlaMode == MlaMode::ROPE_COMBINE_D128) {
            continue;
        }
        GetQueryDimAndOutDim(queryShape, attentionOutShape, layoutStr, tmpQueryDim, tmpOutDim, queryDimIdx);
        if (!fiaInfo.isQKVDDifferent && (tmpQueryDim != tmpOutDim)) {
            std::string shapeMsg = ToString(queryShape->GetStorageShape()) + " and " +
                ToString(attentionOutShape->GetStorageShape());
            std::string reason = std::to_string(queryDimIdx) + "th axis of query must be equal to "
                "the corresponding axis of attention_out when input_layout is " + layoutStr;
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "query and attention_out",
                shapeMsg.c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
        if (fiaInfo.isQKVDDifferent && (queryDimIdx != dimNumQ - 1) && (tmpQueryDim != tmpOutDim)) {
            std::string shapeMsg = ToString(queryShape->GetStorageShape()) + " and " +
                ToString(attentionOutShape->GetStorageShape());
            std::string reason = std::to_string(queryDimIdx) + "th axis of query must be equal to "
                "the corresponding axis of attention_out when input_layout is " + layoutStr;
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "query and attention_out",
                shapeMsg.c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckKeyValueConsistency(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION || fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::StorageShape *keyShape = fiaInfo.opParamInfo.key.shape;
    const gert::StorageShape *valueShape = fiaInfo.opParamInfo.value.shape;
    ge::DataType keyDataType = fiaInfo.opParamInfo.key.desc->GetDataType();
    ge::DataType valueDataType = fiaInfo.opParamInfo.value.desc->GetDataType();

    const size_t keyDimNum = keyShape->GetStorageShape().GetDimNum();
    const size_t valueDimNum = valueShape->GetStorageShape().GetDimNum();
    if (keyDataType != valueDataType) {
        std::string dtypeMsg = ToString(keyDataType) + " and " + ToString(valueDataType);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, "key and value",
            dtypeMsg.c_str(), "The dtypes of key and value must be the same");
        return ge::GRAPH_FAILED;
    }
    if (keyDimNum != valueDimNum) {
        std::string dimMsg = std::to_string(keyDimNum) + " and " + std::to_string(valueDimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(fiaInfo.opName, "key and value",
            dimMsg.c_str(), "The shape dim of key must be equal to that of value");
        return ge::GRAPH_FAILED;
    }

    if (keyDimNum < INPUT_KV_SHAPE_MIN_DIMS || keyDimNum > INPUT_KV_SHAPE_MAX_DIMS) {
        std::string dimStr = std::to_string(keyDimNum);
        std::string reason = "The shape dim of key must be within the range of [" +
            std::to_string(INPUT_KV_SHAPE_MIN_DIMS) + ", " + std::to_string(INPUT_KV_SHAPE_MAX_DIMS) + "]";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "key",
            dimStr.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }

    for (uint32_t i = 0; i < keyDimNum; ++i) {
        if ((i == keyDimNum - 1) && fiaInfo.mlaMode == MlaMode::ROPE_COMBINE_D128) {
            continue;
        }
        int64_t tmpKeyDim = keyShape->GetStorageShape().GetDim(i);
        int64_t tmpValueDim = valueShape->GetStorageShape().GetDim(i);
        if (!fiaInfo.isQKVDDifferent && (tmpKeyDim != tmpValueDim)) {
            std::string shapesStr = ToString(keyShape->GetStorageShape()) + " and " +
                ToString(valueShape->GetStorageShape());
            std::string reason = std::to_string(i) + "th axis of key must be equal to " +
                std::to_string(i) + "th axis of value";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key and value", shapesStr.c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
        if (fiaInfo.isQKVDDifferent && (i != keyDimNum - 1) && (tmpKeyDim != tmpValueDim)) {
            std::string shapesStr = ToString(keyShape->GetStorageShape()) + " and " +
                ToString(valueShape->GetStorageShape());
            std::string reason = std::to_string(i) + "th axis of key must be equal to " +
                std::to_string(i) + "th axis of value";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "key and value", shapesStr.c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckValueOutDConsistency(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION || fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::StorageShape *valueShape = fiaInfo.opParamInfo.value.shape;
    const gert::StorageShape *attentionOutShape = fiaInfo.opParamInfo.attenOut.shape;
    size_t dimNumValue = valueShape->GetStorageShape().GetDimNum();
    size_t dimNumOut = attentionOutShape->GetStorageShape().GetDimNum();
    int64_t valueHeadDim;
    int64_t outHeadDim;
    if (fiaInfo.kvLayout != FiaLayout::BSH) {
        valueHeadDim = valueShape->GetStorageShape().GetDim(dimNumValue -1);
    } else {
        valueHeadDim = valueShape->GetStorageShape().GetDim(dimNumValue - 1) / fiaInfo.n2Size;
    }
    if (fiaInfo.outLayout != FiaLayout::BSH) {
        outHeadDim = attentionOutShape->GetStorageShape().GetDim(dimNumOut -1);
    } else {
        outHeadDim = attentionOutShape->GetStorageShape().GetDim(dimNumOut - 1) / fiaInfo.n1Size;
    }
    if (valueHeadDim != outHeadDim) {
        std::string dimsStr = std::to_string(valueHeadDim) + " and " + std::to_string(outHeadDim);
        std::string reason = "The shape dim of value must be equal to that of attention_out";
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(fiaInfo.opName, "value and attention_out",
            dimsStr.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckQueryShape(const FiaTilingInfo &fiaInfo)
{
    uint32_t attrN = fiaInfo.n1Size;
    const gert::StorageShape *queryShape = fiaInfo.opParamInfo.query.shape;
    uint32_t queryShapeHeadNum = attrN;
    int64_t queryH = 0;

    if (fiaInfo.qLayout == FiaLayout::BNSD || fiaInfo.qLayout == FiaLayout::TND) {
        queryShapeHeadNum = queryShape->GetStorageShape().GetDim(1);
    } else if (fiaInfo.qLayout == FiaLayout::BSND) {
        queryShapeHeadNum = queryShape->GetStorageShape().GetDim(2);
    } else if (fiaInfo.qLayout == FiaLayout::BSH) {
        queryH = queryShape->GetStorageShape().GetDim(2);
    } else if (fiaInfo.qLayout == FiaLayout::NTD) {
        queryShapeHeadNum = queryShape->GetStorageShape().GetDim(0);
    }
    if (attrN != queryShapeHeadNum) {
        std::string reason = "N of query must be equal to the attr num_heads: " + std::to_string(attrN);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "query",
            ToString(queryShape->GetStorageShape()).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (fiaInfo.qLayout == FiaLayout::BSH && queryH % attrN != 0) {
        std::string reason = "H of query must be an integer multiple of num_heads: " +
            std::to_string(attrN) + " when layout is BSH";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "query",
            ToString(queryShape->GetStorageShape()).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckKeyNHVaild(const FiaTilingInfo &fiaInfo, const gert::Shape &keyShape)
{
    uint32_t attrKvN = fiaInfo.n2Size;
    uint32_t keyShapeHeadNum = attrKvN;
    int64_t keyH = 0;

    if (fiaInfo.kvLayout == FiaLayout::BNSD || fiaInfo.kvLayout == FiaLayout::TND) {
        keyShapeHeadNum = keyShape.GetDim(1);
    } else if (fiaInfo.kvLayout == FiaLayout::BSND) {
        keyShapeHeadNum = keyShape.GetDim(2);
    } else if (fiaInfo.kvLayout == FiaLayout::NTD) {
        keyShapeHeadNum = keyShape.GetDim(0);
    } else if (fiaInfo.kvLayout == FiaLayout::BSH) {
        keyH = keyShape.GetDim(2);
    }
    if (attrKvN != keyShapeHeadNum) {
        std::string reason = "N of key must be equal to the attr num_key_value_heads: " + std::to_string(attrKvN);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key",
            ToStringRaw(keyShape).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (fiaInfo.kvLayout == FiaLayout::BSH && keyH % attrKvN != 0) {
        std::string reason = "H of key must be a multiple of num_key_value_heads: " +
            std::to_string(attrKvN) + " when layout is BSH";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key",
            ToStringRaw(keyShape).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckKeyDVaild(const FiaTilingInfo &fiaInfo, const gert::Shape &keyShape)
{
    size_t keyDim = keyShape.GetDimNum();
    uint32_t keyHeadDim = 0;
    if (fiaInfo.kvLayout != FiaLayout::BSH) {
        keyHeadDim = keyShape.GetDim(keyDim -1);
    } else {
        keyHeadDim = keyShape.GetDim(keyDim - 1) / fiaInfo.n2Size;
    }

    if (keyHeadDim != fiaInfo.qkHeadDim) {
        std::string shapeStr = ToStringRaw(keyShape);
        std::string reason = "D of query must be equal to axis D of key";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "key", shapeStr.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckKeyShape(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }
    if (fiaInfo.kvStorageMode == KvStorageMode::BATCH_CONTINUOUS) {
        return CheckKeyNHVaild(fiaInfo, fiaInfo.opParamInfo.key.shape->GetStorageShape());
    } else {
        for (uint32_t i = 0; i < fiaInfo.kCache.size(); i++) {
            auto keyShape = fiaInfo.kCache[i]->GetStorageShape();
            if (CheckKeyNHVaild(fiaInfo, keyShape) != ge::GRAPH_SUCCESS) {
                OP_LOGE(fiaInfo.opName, "When kv storage mode is tensorlist, the input %d of key shape is invalid", i);
                return ge::GRAPH_FAILED;
            }
        }
    return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus CommonChecker::CheckQueryKeyTensorlistConsistency(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.bSize != fiaInfo.kCache.size()) {
        std::string valuesStr = std::to_string(fiaInfo.bSize) + " and " + std::to_string(fiaInfo.kCache.size());
        std::string reason = "B of query must be equal to the number of key: " +
            std::to_string(fiaInfo.kCache.size());
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(fiaInfo.opName, "B of query and the number of key",
            valuesStr.c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    for (uint32_t i = 0; i < fiaInfo.kCache.size(); i++) {
        auto keyShape = fiaInfo.kCache[i]->GetStorageShape();
        if (CheckKeyDVaild(fiaInfo, keyShape) != ge::GRAPH_SUCCESS) {
            OP_LOGE(fiaInfo.opName, "When kv storage mode is tensorlist, the input %d of key shape is invalid", i);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckQueryKeyConsistency(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION || fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST) {
        return ge::GRAPH_SUCCESS;
    }
    ge::DataType queryDataType = fiaInfo.opParamInfo.query.desc->GetDataType();
    ge::DataType keyDataType = fiaInfo.opParamInfo.key.desc->GetDataType();
    if (enableNonQuant_) {
        if (queryDataType != keyDataType) {
            std::string dtypeMsg = ToString(queryDataType) + " and " + ToString(keyDataType);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, "query and key",
                dtypeMsg.c_str(), "The dtypes of query and key must be the same");
            return ge::GRAPH_FAILED;
        }
    }
    int64_t keyB = fiaInfo.opParamInfo.key.shape->GetStorageShape().GetDim(0);
    if (fiaInfo.kvStorageMode == KvStorageMode::BATCH_CONTINUOUS) {
        if ((fiaInfo.qLayout != FiaLayout::TND && fiaInfo.qLayout != FiaLayout::NTD) && fiaInfo.bSize != keyB) {
            std::string shapesStr = ToString(fiaInfo.opParamInfo.query.shape->GetStorageShape()) + " and " +
                ToString(fiaInfo.opParamInfo.key.shape->GetStorageShape());
            std::string reason = "B of query must be equal to the same axis of key";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "query and key",
                shapesStr.c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
        return CheckKeyDVaild(fiaInfo, fiaInfo.opParamInfo.key.shape->GetStorageShape());
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMultiAttr(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.npuArch == NpuArch::DAV_3510) {
        OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION && fiaInfo.isQKVDDifferent,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "blockTable",
                "blockTable must be empty when D of query and key is not equal to D of value"),
            return ge::GRAPH_FAILED);
    }

    if (fiaInfo.inputQType != ge::DT_FLOAT16) {
        OP_LOGW(fiaInfo.opName, "When query input is not fp16,innerPrecise will not take effect");
    }

    if (fiaInfo.qLayout == FiaLayout::TND) {
        OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "tensorList",
                        "When layout is TND, tensorlist is not supported"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// enableNonQuant 相关校验函数
ge::graphStatus CommonChecker::CheckNonQuantDataType(const FiaTilingInfo &fiaInfo)
{
    const std::map<std::string, std::vector<ge::DataType>> NO_QUANT_MAP = {
        {"query",                  {ge::DT_FLOAT16, ge::DT_BF16}},
        {"key",                    {ge::DT_FLOAT16, ge::DT_BF16}},
        {"value",                  {ge::DT_FLOAT16, ge::DT_BF16}},
        {"attentionOut",           {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN}},
    };
    if (ge::GRAPH_SUCCESS != CheckDtypeCommon(fiaInfo.opParamInfo.query.desc, "query", NO_QUANT_MAP) ||
        ge::GRAPH_SUCCESS != CheckDtypeCommon(fiaInfo.opParamInfo.key.desc, "key", NO_QUANT_MAP) ||
        ge::GRAPH_SUCCESS != CheckDtypeCommon(fiaInfo.opParamInfo.value.desc, "value", NO_QUANT_MAP) ||
        ge::GRAPH_SUCCESS != CheckDtypeCommon(fiaInfo.opParamInfo.attenOut.desc,
        "attentionOut", NO_QUANT_MAP)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckAttr(const FiaTilingInfo &fiaInfo)
{
    if (CheckHeadNum(fiaInfo) != ge::GRAPH_SUCCESS ||
        CheckInputLayout(fiaInfo) != ge::GRAPH_SUCCESS ||
        CheckInnerPrecise(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckDimNum(const FiaTilingInfo &fiaInfo)
{
    size_t queryDim = fiaInfo.opParamInfo.query.shape->GetStorageShape().GetDimNum();
    size_t keyDim = fiaInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum();
    if ((fiaInfo.qLayout == FiaLayout::BNSD || fiaInfo.qLayout == FiaLayout::BSND) && queryDim != 4U) {
        std::string reason = "The shape dim of query must be 4 when input_layout is " +
            std::string(fiaInfo.opParamInfo.layOut);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "query",
            std::to_string(queryDim).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if ((fiaInfo.qLayout == FiaLayout::BSH || fiaInfo.qLayout == FiaLayout::TND ||
        fiaInfo.qLayout == FiaLayout::NTD) && queryDim != 3U) {
        std::string reason = "The shape dim of query must be 3 when input_layout is " +
            std::string(fiaInfo.opParamInfo.layOut);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "query",
            std::to_string(queryDim).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if ((fiaInfo.kvLayout == FiaLayout::BNSD || fiaInfo.kvLayout == FiaLayout::BSND) && keyDim != 4U) {
        std::string reason = "The shape dim of key must be 4 when input_layout is " +
            std::string(fiaInfo.opParamInfo.layOut);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "key",
            std::to_string(keyDim).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if ((fiaInfo.kvLayout == FiaLayout::BSH || fiaInfo.kvLayout == FiaLayout::TND ||
        fiaInfo.kvLayout == FiaLayout::NTD) && keyDim != 3U) {
        std::string reason = "The shape dim of key must be 3 when input_layout is " +
            std::string(fiaInfo.opParamInfo.layOut);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "key",
            std::to_string(keyDim).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckHeadNum(const FiaTilingInfo &fiaInfo)
{
    if ((fiaInfo.n1Size < 0) || (fiaInfo.n2Size < 0)) {
        std::string valMsg = std::to_string(fiaInfo.n1Size) + " and " + std::to_string(fiaInfo.n2Size);
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(fiaInfo.opName, "num_heads and num_key_value_heads",
            valMsg.c_str(), "The value of num_heads and num_key_value_heads cannot be negative");
        return ge::GRAPH_FAILED;
    }

    if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) { // ifamla
        static const std::set<uint32_t> SUPPORT_NUM_HEAD_IN_IFAMLA = {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U}; // ifa mla场景qN支持范围
        if ((SUPPORT_NUM_HEAD_IN_IFAMLA.find(fiaInfo.n1Size) == SUPPORT_NUM_HEAD_IN_IFAMLA.end())) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "num_heads",
                std::to_string(fiaInfo.n1Size).c_str(),
                "The value of num_heads must be in the range of {1, 2, 4, 8, 16, 32, 64, 128} "
                "in the Decode MLA scenario");
            return ge::GRAPH_FAILED;
        }
        if (fiaInfo.n2Size != 1U) {
            std::string reason = "The value of num_key_value_heads must be 1 in the Decode MLA scenario";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "num_key_value_heads",
                std::to_string(fiaInfo.n2Size).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::ValidateNoRopeLayoutDim(const FiaTilingInfo &fiaInfo, const std::string &inputLayout)
{
    const std::vector<std::string> noRopeLayoutSupportListA = {"BSH", "BSND", "BNSD"};
    const std::vector<std::string> noRopeLayoutSupportListB = {"BNSD_BSND"};
    const std::vector<std::string> noRopeLayoutSupportListC = {"NTD", "BSH_BNSD", "BSND_BNSD", "NTD_TND"};
    const std::vector<std::string> noRopeLayoutSupportListD = {"TND"};

    if (!fiaInfo.isLegacyIfa &&
        fiaInfo.vHeadDim % 16 != 0) { // 16: qkvD need 16 align when qs>1, in specific input layout
        OP_CHECK_IF(std::find(noRopeLayoutSupportListA.begin(), noRopeLayoutSupportListA.end(), inputLayout) !=
                        noRopeLayoutSupportListA.end(),
                    OP_LOGE(fiaInfo.opName,
                            "In %s %s situation, when Qs>1 and input_layout is %s, headDim of query|key|value "
                            "should be align to 16.",
                            QuantModeToSerialString(fiaInfo.quantMode).c_str(),
                            SituationToSerialString(fiaInfo.ropeMode).c_str(), inputLayout.c_str()),
                    return ge::GRAPH_FAILED);

        OP_CHECK_IF(std::find(noRopeLayoutSupportListB.begin(), noRopeLayoutSupportListB.end(), inputLayout) !=
                        noRopeLayoutSupportListB.end(),
                    OP_LOGE(fiaInfo.opName,
                            "In %s %s situation, when Qs>1 and input_layout is %s, headDim of query|key|value "
                            "should be align to 16.",
                            QuantModeToSerialString(fiaInfo.quantMode).c_str(),
                            SituationToSerialString(fiaInfo.ropeMode).c_str(), inputLayout.c_str()),
                    return ge::GRAPH_FAILED);
    }

    if (fiaInfo.isLegacyIfa &&
        (fiaInfo.vHeadDim != 64 &&
         fiaInfo.vHeadDim != 128)) { // 64: qkvD need 64 128: qkvD need 128 in specific input layout
        OP_CHECK_IF(std::find(noRopeLayoutSupportListB.begin(), noRopeLayoutSupportListB.end(), inputLayout) !=
                        noRopeLayoutSupportListB.end(),
                    OP_LOGE(fiaInfo.opName,
                            "In %s %s situation, when Qs=1 and input_layout is BNSD_BSND, only query|key|value "
                            "headDim = 64/128 are supported, but got %u",
                            QuantModeToSerialString(fiaInfo.quantMode).c_str(),
                            SituationToSerialString(fiaInfo.ropeMode).c_str(), fiaInfo.vHeadDim),
                    return ge::GRAPH_FAILED);
    }

    if (std::find(noRopeLayoutSupportListC.begin(), noRopeLayoutSupportListC.end(), inputLayout) !=
        noRopeLayoutSupportListC.end()) {
        OP_CHECK_IF(fiaInfo.vHeadDim != 64 && fiaInfo.vHeadDim != 128,
                    OP_LOGE(fiaInfo.opName,
                            "In %s %s situation, when input_layout is NTD, BSH_BNSD, BSND_BNSD, NTD_TND, only "
                            "query|key|value headDim = 64/128 are supported, but got %u",
                            QuantModeToSerialString(fiaInfo.quantMode).c_str(),
                            SituationToSerialString(fiaInfo.ropeMode).c_str(), fiaInfo.vHeadDim),
                    return ge::GRAPH_FAILED);
    }

    if (std::find(noRopeLayoutSupportListD.begin(), noRopeLayoutSupportListD.end(), inputLayout) !=
        noRopeLayoutSupportListD.end()) {
        OP_CHECK_IF(fiaInfo.vHeadDim != 64 && fiaInfo.vHeadDim != 128 && fiaInfo.vHeadDim != 192,
                    OP_LOGE(fiaInfo.opName,
                            "In %s %s situation, when input_layout is TND, only query|key|value headDim = "
                            "64/128/192 are supported, but got %u",
                            QuantModeToSerialString(fiaInfo.quantMode).c_str(),
                            SituationToSerialString(fiaInfo.ropeMode).c_str(), fiaInfo.vHeadDim),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckInputLayout(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.opParamInfo.layOut == nullptr) {
        return ge::GRAPH_FAILED;
    }
    std::string inputLayout = fiaInfo.opParamInfo.layOut;
    if (fiaInfo.ropeMode == RopeMode::NO_ROPE) {
        if (fiaInfo.socVersion != platform_ascendc::SocVersion::ASCEND910B) {
            const std::vector<std::string> INPUT_LAYOUT_LIST = {
                "BSH", "BSND", "BNSD", "TND", "NTD", "BSND_BNSD", "BSH_BNSD", "NTD_TND", "BNSD_BSND"
            };
            if (std::find(INPUT_LAYOUT_LIST.begin(), INPUT_LAYOUT_LIST.end(), inputLayout) == INPUT_LAYOUT_LIST.end()) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout", inputLayout.c_str(),
                    "The value of input_layout must be in BSH, BSND, BNSD, TND, NTD, BSND_BNSD, "
                    "BSH_BNSD, NTD_TND, BNSD_BSND in the GQA scenario");
                return ge::GRAPH_FAILED;
            }
        } else {
            const std::vector<std::string> INPUT_LAYOUT_LIST = {
                "BSH", "BSND", "BNSD", "TND", "NTD", "NTD_TND", "BSH_BNSD", "BSND_BNSD", "BNSD_BSND"
            };
            OP_CHECK_IF(std::find(INPUT_LAYOUT_LIST.begin(), INPUT_LAYOUT_LIST.end(), inputLayout) == INPUT_LAYOUT_LIST.end(),
                OP_LOGE(fiaInfo.opName, "When gqa noquant scenario is applied, layout only supports BSH, BSND, BNSD, TND, "
                    "NTD, NTD_TND, BSH_BNSD, BSND_BNSD, BNSD_BSND, but got %s",
                    inputLayout.c_str()),
                return ge::GRAPH_FAILED);
            return ValidateNoRopeLayoutDim(fiaInfo, inputLayout);
        }
    } else if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) { // decode mla
        const std::vector<std::string> INPUT_LAYOUT_LIST = {
            "BSH", "BSND", "BNSD", "TND", "BNSD_NBSD", "BSND_NBSD", "BSH_NBSD", "TND_NTD"
        };
        if (std::find(INPUT_LAYOUT_LIST.begin(), INPUT_LAYOUT_LIST.end(), inputLayout) == INPUT_LAYOUT_LIST.end()) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout",
                inputLayout.c_str(), "The value of input_layout must be in BSH, BSND, "
                "BNSD, TND ,BNSD_BSND, BSND_NBSD, BSH_NBSD, TND_NTD in the Decode MLA scenario");
            return ge::GRAPH_FAILED;
        }
    } else { // prefill mla
        const std::vector<std::string> INPUT_LAYOUT_LIST = {
            "BSH", "BSND", "BNSD", "TND", "NTD", "BSND_BNSD", "BSH_BNSD", "NTD_TND", "BNSD_BSND"
        };
        if (std::find(INPUT_LAYOUT_LIST.begin(), INPUT_LAYOUT_LIST.end(), inputLayout) == INPUT_LAYOUT_LIST.end()) {
            std::string reason = "The value of input_layout must be in BSH, BSND, BNSD, "
                "TND, NTD, BSND_BNSD, BSH_BNSD, NTD_TND, BNSD_BSND in the Prefill MLA scenario";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout",
                inputLayout.c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        } else {
            OP_CHECK_IF(fiaInfo.ropeMode == RopeMode::ROPE_COMBINE &&
                (fiaInfo.qkHeadDim != NUM_192 || fiaInfo.vHeadDim != NUM_128),
            OP_LOGE(fiaInfo.opName,
                "In %s %s situation, when input_layout is BSH, BSND, BNSD, BNSD_BSND, TND, NTD, "
                "BSH_BNSD, BSND_BNSD, NTD_TND, "
                "and the headDim shared by query and key is not equal to that of value, "
                "only query|key headDim = 192, value headDim = 128 are supported, "
                "but got query|key headDim: %u, value headDim: %u",
                QuantModeToSerialString(fiaInfo.quantMode).c_str(),
                SituationToSerialString(fiaInfo.ropeMode).c_str(), fiaInfo.qkHeadDim, fiaInfo.vHeadDim),
            return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckInnerPrecise(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.innerPrecise > INNER_PRECISE_LIMIT || fiaInfo.innerPrecise < 0) {
        std::string reason = "The value of inner_precise must be in the range of [0, " +
            std::to_string(INNER_PRECISE_LIMIT) + "]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "inner_precise",
            std::to_string(fiaInfo.innerPrecise).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    
    return ge::GRAPH_SUCCESS;
}

bool CommonChecker::CheckTNDLayoutCrossover(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.inputLayout != TilingKeyLayout::TND) {
        return true;
    }

    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512 && layoutStr == "TND_NTD") { // Decode MLA
        OP_CHECK_IF(fiaInfo.isOutQuantEnable,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "quantScale2",
                "quantScale2 must be empty in Decode MLA scenario, when layout is TND_NTD"),
            return false);
    }
    
    OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "tensorList",
            "When layout is TND, tensorlist is not supported"),
        return false);

    OP_CHECK_IF(fiaInfo.pseShiftFlag,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "pseShift",
            "pseShift must be empty when layout is TND"),
        return false);

    return true;
}

bool CommonChecker::CheckNTDLayoutCrossover(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.inputLayout != TilingKeyLayout::NTD) {
        return true;
    }
    bool isGqa = enableNonQuant_ && !(fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) && !(fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D128) &&
        !(fiaInfo.mlaMode == MlaMode::ROPE_COMBINE_D128);
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    if (isGqa) { // GQA
        if ((fiaInfo.qkHeadDim != 64 && fiaInfo.qkHeadDim != 128)) {
            std::string reason = "D of query must be 64 or 128 in the GQA scenario, when layout is " + layoutStr;
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "query",
                ToStringRaw(fiaInfo.opParamInfo.query.shape->GetStorageShape()).c_str(), reason.c_str());
            return false;
        }
    }
    if (isGqa) { // GQA
        OP_CHECK_IF(fiaInfo.isQKVDDifferent,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout",
                layoutStr.c_str(), "The value of input_layout cannot be " + layoutStr +
                " in the GQA scenario, when D of query and key is not equal to D of value"),
            return false);
    }
    
    OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "tensorList",
            ("When layout is " + layoutStr + ", tensorlist is not supported").c_str()),
        return false);

    OP_CHECK_IF(fiaInfo.pseShiftFlag,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "pseShift",
            ("pseShift must be empty when layout is " + layoutStr + "").c_str()),
        return false);

    return true;
}

bool CommonChecker::CheckTransposeLayoutCrossover(const FiaTilingInfo &fiaInfo)
{
    std::string layoutStr(fiaInfo.opParamInfo.layOut);
    if (layoutStr != "BSH_BNSD" && layoutStr != "BSND_BNSD" && layoutStr != "BNSD_BSND") {
        return true;
    }
    bool isGqa = enableNonQuant_ && !(fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) && !(fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D128) &&
        !(fiaInfo.mlaMode == MlaMode::ROPE_COMBINE_D128);
    if (isGqa) { // GQA
        OP_CHECK_IF(fiaInfo.isQKVDDifferent,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "input_layout",
                layoutStr.c_str(), "The value of input_layout cannot be " + layoutStr +
                " in the GQA scenario, when D of query and key is not equal to D of value"),
            return false);
    }
    if (layoutStr == "BSH_BNSD" || layoutStr == "BSND_BNSD") {
        if (isGqa) { // GQA
            if ((fiaInfo.qkHeadDim != 64 && fiaInfo.qkHeadDim != 128)) {
                std::string reason = "D of query must be 64 or 128 in the GQA scenario, when layout is " +
                    layoutStr;
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "query",
                    ToStringRaw(fiaInfo.opParamInfo.query.shape->GetStorageShape()).c_str(), reason.c_str());
                return false;
            }
        }
        
        OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "tensorList",
                ("When layout is " + layoutStr + ", tensorlist is not supported").c_str()),
            return false);

        OP_CHECK_IF(fiaInfo.pseShiftFlag,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "pseShift",
                ("pseShift must be empty when layout is " + layoutStr + "").c_str()),
            return false);
    } else if (layoutStr == "BNSD_BSND") {
        if (isGqa) { // GQA
            if (fiaInfo.outputType == ge::DT_INT8 && fiaInfo.qkHeadDim % 32 != 0) {
                std::string reason = "D of query must be a multiple of 32 in the GQA scenario when layout is " +
                    layoutStr + " and output dtype is INT8";
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "query",
                    ToStringRaw(fiaInfo.opParamInfo.query.shape->GetStorageShape()).c_str(), reason.c_str());
                return false;
            }
            if (fiaInfo.qkHeadDim % 16 != 0) {
                std::string reason = "D of query must be a multiple of 16 in the GQA scenario, "
                    "when layout is " + layoutStr;
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "query",
                    ToStringRaw(fiaInfo.opParamInfo.query.shape->GetStorageShape()).c_str(), reason.c_str());
                return false;
            }
        }
    }
    return true;
}

ge::graphStatus CommonChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    if (enableNonQuant_) {
        if (CheckNonQuantDataType(fiaInfo) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    if (CheckInputFormat(fiaInfo) != ge::GRAPH_SUCCESS ||
        CheckAttr(fiaInfo) != ge::GRAPH_SUCCESS ||
        CheckDimNum(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    if (CheckParaExistenceImpl(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckKVContiguous(const FiaTilingInfo &fiaInfo) const
{
    if (fiaInfo.isTensorV1) {
        return ge::GRAPH_SUCCESS;
    }
    if (fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::Shape &keyShape = fiaInfo.opParamInfo.key.shape->GetStorageShape();
    const gert::Shape &valueShape = fiaInfo.opParamInfo.value.shape->GetStorageShape();
    uint32_t keyDimNum = keyShape.GetDimNum();
    uint32_t valueDimNum = valueShape.GetDimNum();
    int32_t dimIndex = 0;

    if (!fiaInfo.pageAttentionFlag) {
        OP_CHECK_IF((CheckTensorContiguous(keyDimNum, keyShape,
                        fiaInfo.keyStrides, dimIndex) != ge::GRAPH_SUCCESS) ||
                    (CheckTensorContiguous(valueDimNum, valueShape,
                        fiaInfo.valueStrides, dimIndex) != ge::GRAPH_SUCCESS),
                    OP_LOGE(fiaInfo.opName,
                            "In non-PA scenarios, key/value tensors must be contiguous."),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    if (fiaInfo.kvLayout == FiaLayout::BnBsH) {
        OP_CHECK_IF(((ge::GRAPH_SUCCESS !=
                        CheckTensorContiguous(keyDimNum, keyShape,
                            fiaInfo.keyStrides, dimIndex)) &&
                    (dimIndex != 0)),
                    OP_LOGE(fiaInfo.opName,
                            "In PA BBND scenarios, key only supports non-contiguous "
                            "tensors in dimension 0, "
                            "but the first non-contiguous dimension is index %d.",
                            dimIndex),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(((ge::GRAPH_SUCCESS !=
                        CheckTensorContiguous(valueDimNum, valueShape,
                            fiaInfo.valueStrides, dimIndex)) &&
                    (dimIndex != 0)),
                    OP_LOGE(fiaInfo.opName,
                            "In PA BBND scenarios, value only supports non-contiguous "
                            "tensors in dimension 0, "
                            "but the first non-contiguous dimension is index %d.",
                            dimIndex),
                    return ge::GRAPH_FAILED);
    } else if (fiaInfo.kvLayout == FiaLayout::BnNBsD || fiaInfo.kvLayout == FiaLayout::NZ) {
        OP_CHECK_IF(((ge::GRAPH_SUCCESS !=
                        CheckTensorContiguous(keyDimNum, keyShape,
                            fiaInfo.keyStrides, dimIndex)) &&
                    (dimIndex != 0 && dimIndex != 1)),
                    OP_LOGE(fiaInfo.opName,
                            "In PA BNBD/NZ scenarios, key only supports non-contiguous "
                            "tensors in dimensions 0 or 1, "
                            "but the first non-contiguous dimension is index %d.",
                            dimIndex),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(((ge::GRAPH_SUCCESS !=
                        CheckTensorContiguous(valueDimNum, valueShape,
                            fiaInfo.valueStrides, dimIndex)) &&
                    (dimIndex != 0 && dimIndex != 1)),
                    OP_LOGE(fiaInfo.opName,
                            "In PA BNBD/NZ scenarios, value only supports non-contiguous "
                            "tensors in dimensions 0 or 1, "
                            "but the first non-contiguous dimension is index %d.",
                            dimIndex),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckKVStorageConsistency(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        if (CheckPAKeyValue(fiaInfo) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else if (fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST) {
        OP_CHECK_IF((CheckTensorList(fiaInfo)) != ge::GRAPH_SUCCESS,
            OPS_REPORT_VECTOR_INNER_ERR(fiaInfo.opName,
                "Check Tensorlist failed!"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckShapeConsistency(const FiaTilingInfo &fiaInfo)
{
    if (CheckAxis(fiaInfo) != ge::GRAPH_SUCCESS ||
        CheckQueryOutConsistency(fiaInfo) != ge::GRAPH_SUCCESS ||
        CheckKeyValueConsistency(fiaInfo) != ge::GRAPH_SUCCESS ||
        CheckValueOutDConsistency(fiaInfo) != ge::GRAPH_SUCCESS ||
        CheckQueryShape(fiaInfo) != ge::GRAPH_SUCCESS ||
        CheckKeyShape(fiaInfo) != ge::GRAPH_SUCCESS ||
        CheckQueryKeyConsistency(fiaInfo) != ge::GRAPH_SUCCESS ||
        CheckMultiAttr(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo)
{
    if (enableNonQuant_) {
        if (!CheckTNDLayoutCrossover(fiaInfo) || !CheckNTDLayoutCrossover(fiaInfo) || !CheckTransposeLayoutCrossover(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (CheckMultiDtype(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckKVStorageConsistency(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType queryDataType = fiaInfo.opParamInfo.query.desc->GetDataType();
    ge::DataType attenOutDataType = fiaInfo.opParamInfo.attenOut.desc->GetDataType();
    if (!fiaInfo.isOutQuantEnable && !enableFullQuant_) {
        if (queryDataType != attenOutDataType) {
            std::string dtypeMsg = ToString(queryDataType) + " and " + ToString(attenOutDataType);
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(fiaInfo.opName, "query and attention_out",
                dtypeMsg.c_str(), "The dtypes of query and attention_out must be the same");
            return ge::GRAPH_FAILED;
        }
    }
    
    if (CheckShapeConsistency(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (fiaInfo.socVersion != platform_ascendc::SocVersion::ASCEND910B &&
        enableNonQuant_ && CheckKVContiguous(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling