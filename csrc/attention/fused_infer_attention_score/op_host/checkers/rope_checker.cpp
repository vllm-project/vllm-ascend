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
 * \file rope_checker.cpp
 * \brief
 */

#include <map>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "rope_checker.h"

namespace optiling {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// check rope dtype
ge::graphStatus RopeChecker::CheckRopeDtype(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.ropeMode != RopeMode::ROPE_SPLIT) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF((fiaInfo.inputQRopeType != ge::DT_BF16 && fiaInfo.inputQRopeType != ge::DT_FLOAT16),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    fiaInfo.opName, "query_rope", ToString(fiaInfo.inputQRopeType).c_str(),
                    "The datatype of query_rope must be BF16 or FLOAT16 when query_rope is not empty"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((fiaInfo.inputKRopeType != ge::DT_BF16 && fiaInfo.inputKRopeType != ge::DT_FLOAT16),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    fiaInfo.opName, "key_rope", ToString(fiaInfo.inputKRopeType).c_str(),
                    "The datatype of key_rope must be BF16 or FLOAT16 when key_rope is not empty"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// check rope DSize
ge::graphStatus RopeChecker::CheckRopeDSizeSupport(const FiaTilingInfo &fiaInfo) const
{
    if (fiaInfo.ropeMode != RopeMode::ROPE_SPLIT) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF((fiaInfo.ropeHeadDim != NUM_64),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    fiaInfo.opName, "query_rope",
                    ToStringRaw(fiaInfo.opParamInfo.queryRope.tensor->GetStorageShape()).c_str(),
                    "When rope exists, the d axis of rope must be 64"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckQDsizeSupport(const FiaTilingInfo &fiaInfo) const
{
    if (enableFullQuant_ && fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_MXFP8_FULL_QUANT) {
        OP_CHECK_IF((fiaInfo.qkHeadDim != NUM_64 && fiaInfo.qkHeadDim != NUM_128),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        fiaInfo.opName, "query and key",
                        (ToString(fiaInfo.opParamInfo.query.shape->GetStorageShape()) + " and " +
                         ToString(fiaInfo.opParamInfo.key.shape->GetStorageShape()))
                            .c_str(),
                        "In MXFP8 fullquant scenario with rope, the D axis of query and key must be 64 or 128"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((fiaInfo.qkHeadDim != NUM_128 && fiaInfo.qkHeadDim != NUM_512),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        fiaInfo.opName, "query and key",
                        (ToString(fiaInfo.opParamInfo.query.shape->GetStorageShape()) + " and " +
                         ToString(fiaInfo.opParamInfo.key.shape->GetStorageShape()))
                            .c_str(),
                        "When rope exists, the D axis of query and key must be 128 or 512"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckKRopeContiguous(const FiaTilingInfo &fiaInfo)
{
    // mxfp8全量化场景检查krope连续性
    if (!enableFullQuant_ || fiaInfo.fullQuantMode != FiaFullQuantMode::QKV_MXFP8_FULL_QUANT) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::Shape keyRopeShape = fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape();
    const uint32_t keyRopeDimNum = keyRopeShape.GetDimNum();
    int32_t dimIndex = 0;
    OP_CHECK_IF(((ge::GRAPH_SUCCESS != CheckTensorContiguous(keyRopeDimNum, keyRopeShape, fiaInfo.kRopeStrides, dimIndex)) &&
            !fiaInfo.pageAttentionFlag),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "keyRope",
                "In non-PA scenarios, MXFP8 full quantization does not support non-contiguous tensors"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(((ge::GRAPH_SUCCESS != CheckTensorContiguous(keyRopeDimNum, keyRopeShape, fiaInfo.kRopeStrides, dimIndex)) &&
            (dimIndex != 0 && dimIndex != 1) && fiaInfo.pageAttentionFlag),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "keyRope",
                "In MXFP8 Fullquant BnNBsD/NZ scenario, only 0th and 1st axis of keyRope can be non-contiguous, the "
                + std::to_string(dimIndex) + "th axis of keyRope must be contiguous"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckShapeConsistency(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.ropeMode != RopeMode::ROPE_SPLIT) {
        return ge::GRAPH_SUCCESS;
    }
    // check qk head dim and rope head dim
    if (ge::GRAPH_SUCCESS != CheckQDsizeSupport(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape queryShape = fiaInfo.opParamInfo.query.shape->GetStorageShape();
    const gert::Shape queryRopeShape = fiaInfo.opParamInfo.queryRope.tensor->GetStorageShape();
    const gert::Shape keyShape = fiaInfo.opParamInfo.key.shape->GetStorageShape();
    const gert::Shape keyRopeShape = fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape();

    // check queryRope shape
    if (ge::GRAPH_SUCCESS != CheckQKAndQKRopeShapeConsistency(fiaInfo, queryShape, queryRopeShape, "query")) {
        return ge::GRAPH_FAILED;
    }
    // check keyRope shape
    if (ge::GRAPH_SUCCESS != CheckQKAndQKRopeShapeConsistency(fiaInfo, keyShape, keyRopeShape, "key") ||
        ge::GRAPH_SUCCESS != CheckPAKeyAndKeyRopeShapeConsistency(fiaInfo, keyShape, keyRopeShape) ||
        ge::GRAPH_SUCCESS != CheckTensorlistKeyAndKeyRopeShapeConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// check rope dtype 与 query/key的数据类型一致
ge::graphStatus RopeChecker::CheckRopeDtypeConsistency(const FiaTilingInfo &fiaInfo)
{
    if (enableNonQuant_ && fiaInfo.ropeMode == RopeMode::ROPE_SPLIT) {
        OP_CHECK_IF((fiaInfo.inputQRopeType != fiaInfo.inputQType),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "query_rope",
                                                          ToString(fiaInfo.inputQRopeType).c_str(),
                                                          "The datatype of query_rope and query must be the same"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((fiaInfo.inputKRopeType != fiaInfo.inputKvType),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "key_rope",
                                                          ToString(fiaInfo.inputKRopeType).c_str(),
                                                          "The datatype of key_rope and key must be the same"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// 除D轴外 queryRope/keyRope其余轴需和query/key一致
ge::graphStatus RopeChecker::CheckQKAndQKRopeShapeConsistency(const FiaTilingInfo &fiaInfo,
    const gert::Shape shape, const gert::Shape ropeShape, const std::string &inputName) const
{
    if (inputName == "key" && fiaInfo.kvStorageMode != KvStorageMode::BATCH_CONTINUOUS) {
        return ge::GRAPH_SUCCESS;
    }
    const std::string inputRopeName = inputName + "_rope";
    const uint32_t dimNum = shape.GetDimNum();
    const uint32_t ropeDimNum = ropeShape.GetDimNum();
    if (dimNum != ropeDimNum) {
        std::string paramMsg = inputRopeName + " and " + inputName;
        std::string dimMsg = std::to_string(ropeDimNum) + " and " + std::to_string(dimNum);
        std::string reasonMsg =
            "The shape dims of input " + inputRopeName + " must be equal to that of input " + inputName;
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(fiaInfo.opName, paramMsg.c_str(), dimMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    for (uint32_t i = 0; i < dimNum - 1; i++) {
        if (shape.GetDim(i) != ropeShape.GetDim(i)) {
            std::string paramMsg = inputRopeName + " and " + inputName;
            std::string shapeMsg = ToString(ropeShape) + " and " + ToString(shape);
            std::string reasonMsg = "The " + std::to_string(i) + "th axis of input " + inputRopeName +
                                    " must be equal to the " + std::to_string(i) + "th axis of input " + inputName;
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, paramMsg.c_str(), shapeMsg.c_str(),
                                                   reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    if (fiaInfo.qLayout == FiaLayout::BSH) {
        uint32_t tmpN = static_cast<uint32_t>(shape.GetDim(dimNum - 1) / fiaInfo.qkHeadDim);
        uint32_t ropeN = static_cast<uint32_t>(ropeShape.GetDim(ropeDimNum - 1) / fiaInfo.ropeHeadDim);
        OP_CHECK_IF(tmpN != ropeN,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, inputRopeName.c_str(), ToStringRaw(ropeShape),
                "The N axis of " + inputRopeName + "(" + std::to_string(ropeN) + ") must be equal to " +
                    inputName + "(" + std::to_string(tmpN) + ")"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// PA场景 除D轴外 keyRope其余轴需和key一致
ge::graphStatus RopeChecker::CheckPAKeyAndKeyRopeShapeConsistency(const FiaTilingInfo &fiaInfo,
    const gert::Shape &keyShape, const gert::Shape &keyRopeShape) const
{
    if (fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }

    const uint32_t keyDimNum = keyShape.GetDimNum();
    const uint32_t keyRopeDimNum = keyRopeShape.GetDimNum();
    OP_CHECK_IF(keyDimNum != keyRopeDimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    fiaInfo.opName, "key_rope and key",
                    (std::to_string(keyRopeDimNum) + "D and " + std::to_string(keyDimNum) + "D").c_str(),
                    "The dim num of key_rope and key must be the same"),
                return ge::GRAPH_FAILED);

    uint32_t keyBlockNum = 0;
    uint32_t keyRopeBlockNum = 0;
    uint32_t keyBlockSize = 0;
    uint32_t keyRopeBlockSize = 0;
    uint32_t keyN = 0;
    uint32_t keyRopeN = 0;

    if (keyDimNum == DIM_NUM_3) { // BBH
        keyBlockNum = keyShape.GetDim(DIM_NUM_0);
        keyBlockSize = keyShape.GetDim(DIM_NUM_1);
        keyN = keyShape.GetDim(DIM_NUM_2) / fiaInfo.qkHeadDim;
        keyRopeBlockNum = keyRopeShape.GetDim(DIM_NUM_0);
        keyRopeBlockSize = keyRopeShape.GetDim(DIM_NUM_1);
        keyRopeN = keyRopeShape.GetDim(DIM_NUM_2) / fiaInfo.ropeHeadDim;
    } else if (keyDimNum == DIM_NUM_4) { // BNBD
        keyBlockNum = keyShape.GetDim(DIM_NUM_0);
        keyN = keyShape.GetDim(DIM_NUM_1);
        keyBlockSize = keyShape.GetDim(DIM_NUM_2);
        keyRopeBlockNum = keyRopeShape.GetDim(DIM_NUM_0);
        keyRopeN = keyRopeShape.GetDim(DIM_NUM_1);
        keyRopeBlockSize = keyRopeShape.GetDim(DIM_NUM_2);
    } else { // BND1BD0
        keyBlockNum = keyShape.GetDim(DIM_NUM_0);
        keyN = keyShape.GetDim(DIM_NUM_1);
        keyBlockSize = keyShape.GetDim(DIM_NUM_3);
        keyRopeBlockNum = keyRopeShape.GetDim(DIM_NUM_0);
        keyRopeN = keyRopeShape.GetDim(DIM_NUM_1);
        keyRopeBlockSize = keyRopeShape.GetDim(DIM_NUM_3);
    }

    OP_CHECK_IF(keyBlockNum != keyRopeBlockNum,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    fiaInfo.opName, "key_rope and key",
                    (ToString(keyRopeShape) + " and " + ToString(keyShape)).c_str(),
                    "When page attention is enabled, the blockNum of key_rope and key must be the same"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyBlockSize != keyRopeBlockSize,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    fiaInfo.opName, "key_rope and key",
                    (ToString(keyRopeShape) + " and " + ToString(keyShape)).c_str(),
                    "When page attention is enabled, the blockSize of key_rope and key must be the same"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyN != keyRopeN,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    fiaInfo.opName, "key_rope and key",
                    (ToString(keyRopeShape) + " and " + ToString(keyShape)).c_str(),
                    "When page attention is enabled, the N axis of key_rope and key must be the same"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// tensorlist keyRope和key的一致性校验
ge::graphStatus RopeChecker::CheckTensorlistKeyAndKeyRopeShapeConsistency(const FiaTilingInfo &fiaInfo) const
{
    if (fiaInfo.kvStorageMode != KvStorageMode::TENSOR_LIST) {
        return ge::GRAPH_SUCCESS;
    }

    if (fiaInfo.opParamInfo.keyRope.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    // b = len(kCache)
    const gert::Shape keyRopeShape = fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape();
    if (fiaInfo.kCache.size() != keyRopeShape.GetDim(DIM_NUM_0)) {
        std::string shapeMsg = ToString(keyRopeShape) + " and " + std::to_string(fiaInfo.kCache.size());
        std::string reason = "The axis B of keyRope(" + std::to_string(keyRopeShape.GetDim(DIM_NUM_0))  +
            ") must be equal to the tensor number of key(" + std::to_string(fiaInfo.kCache.size()) + ")";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "keyRope and key", shapeMsg.c_str(), reason);
        return ge::GRAPH_FAILED;
    }

    // k_s = krope_s
    const std::string inputLayout = fiaInfo.opParamInfo.layOut;
    int64_t keyRopeS = 0;
    int64_t keyRopeN = 0;
    if (inputLayout == "BNSD" || inputLayout == "BNSD_BSND") {
        keyRopeS = keyRopeShape.GetDim(DIM_NUM_2);
        keyRopeN = keyRopeShape.GetDim(DIM_NUM_1);
        for (uint32_t i = 0; i < fiaInfo.kCache.size(); i++) {
            const gert::Shape keyShape = fiaInfo.kCache[i]->GetStorageShape();
            if (keyShape.GetDim(DIM_NUM_2) != keyRopeS) {
                std::string shapeMsg = ToString(keyRopeShape) + " and " + ToString(keyShape);
                std::string reason = "The axis S of keyRope(" + std::to_string(keyRopeS)  + ") must be equal to "
                    "the axis S of key(" + std::to_string(keyShape.GetDim(DIM_NUM_2)) + ")";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "keyRope and key", shapeMsg.c_str(), reason);
                return ge::GRAPH_FAILED;
            }
            if (keyShape.GetDim(DIM_NUM_1) != keyRopeN) {
                std::string shapeMsg = ToString(keyRopeShape) + " and " + ToString(keyShape);
                std::string reason = "The axis N of keyRope(" + std::to_string(keyRopeN)  + ") must be equal to "
                    "the axis N of key(" + std::to_string(keyShape.GetDim(DIM_NUM_1)) + ")";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "keyRope and key", shapeMsg.c_str(), reason);
                return ge::GRAPH_FAILED;
            }
        }
    } else if (inputLayout == "BSH") {
        keyRopeS = keyRopeShape.GetDim(DIM_NUM_1);
        keyRopeN = keyRopeShape.GetDim(DIM_NUM_2) / fiaInfo.ropeHeadDim;
        for (uint32_t i = 0; i < fiaInfo.kCache.size(); i++) {
            const gert::Shape keyShape = fiaInfo.kCache[i]->GetStorageShape();
            if (keyShape.GetDim(DIM_NUM_1) != keyRopeS) {
                std::string shapeMsg = ToString(keyRopeShape) + " and " + ToString(keyShape);
                std::string reason = "The axis S of keyRope(" + std::to_string(keyRopeS) + ") must be equal to "
                    "the axis S of key(" + std::to_string(keyShape.GetDim(DIM_NUM_1)) + ")";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "keyRope and key", shapeMsg.c_str(), reason);
                return ge::GRAPH_FAILED;
            }
            if (keyShape.GetDim(DIM_NUM_2) / fiaInfo.qkHeadDim != keyRopeN) {
                std::string shapeMsg = ToString(keyRopeShape) + " and " + ToString(keyShape);
                std::string reason = "The axis N of keyRope(" + std::to_string(keyRopeN) + ") must be equal to "
                    "the axis N of key(" + std::to_string(keyShape.GetDim(DIM_NUM_2) / fiaInfo.qkHeadDim) + ")";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "keyRope and key", shapeMsg.c_str(), reason);
                return ge::GRAPH_FAILED;
            }
        }
    } else if (inputLayout == "BSND") {
        keyRopeS = keyRopeShape.GetDim(DIM_NUM_1);
        keyRopeN = keyRopeShape.GetDim(DIM_NUM_2);
        for (uint32_t i = 0; i < fiaInfo.kCache.size(); i++) {
            const gert::Shape keyShape = fiaInfo.kCache[i]->GetStorageShape();
            if (keyShape.GetDim(DIM_NUM_1) != keyRopeS) {
                std::string shapeMsg = ToString(keyRopeShape) + " and " + ToString(keyShape);
                std::string reason = "The axis S of keyRope(" + std::to_string(keyRopeS) + ") must be equal to "
                    "the axis S of key(" + std::to_string(keyShape.GetDim(DIM_NUM_1)) + ")";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "keyRope and key", shapeMsg.c_str(), reason);
                return ge::GRAPH_FAILED;
            }
            if (keyShape.GetDim(DIM_NUM_2) != keyRopeN) {
                std::string shapeMsg = ToString(keyRopeShape) + " and " + ToString(keyShape);
                std::string reason = "The axis N of keyRope(" + std::to_string(keyRopeN) + ") must be equal to "
                    "the axis N of key(" + std::to_string(keyShape.GetDim(DIM_NUM_2)) + ")";
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "keyRope and key", shapeMsg.c_str(), reason);
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckRopeExistence(const FiaTilingInfo &fiaInfo) const
{
    const gert::Tensor *queryRopeTensor = fiaInfo.opParamInfo.queryRope.tensor;
    const gert::Tensor *keyRopeTensor = fiaInfo.opParamInfo.keyRope.tensor;

    // rope split时 rope必须都存在
    if (fiaInfo.ropeMode == RopeMode::ROPE_SPLIT) {
        OP_LOGI(fiaInfo.opName, "Rope mode is ROPE_SPLIT.");
        OP_CHECK_IF((queryRopeTensor == nullptr || keyRopeTensor == nullptr),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "query_rope and key_rope",
                                                           "When rope exists, queryRope or keyRope cannot be empty"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckRopeConsistency(const FiaTilingInfo &fiaInfo) const
{
    const gert::Tensor *queryRopeTensor = fiaInfo.opParamInfo.queryRope.tensor;
    const gert::Tensor *keyRopeTensor = fiaInfo.opParamInfo.keyRope.tensor;

    // 一个空 一个非空
    OP_CHECK_IF((queryRopeTensor == nullptr && keyRopeTensor != nullptr),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "key_rope", "not empty",
                                                      "When query_rope is empty, key_rope must be empty"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((keyRopeTensor == nullptr && queryRopeTensor != nullptr),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "key_rope", "not empty",
                                                      "When query_rope is not empty, key_rope cannot be empty"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckFeatureSupport(const FiaTilingInfo &fiaInfo) const
{
    if (fiaInfo.ropeMode != RopeMode::ROPE_SPLIT) {
        return ge::GRAPH_SUCCESS;
    }
    // 不支持 prefix
    OP_CHECK_IF(fiaInfo.sysPrefixFlag,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "key_shared_prefix",
                                                      "When query_rope is not empty, key_shared_prefix must be empty"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.pseShiftFlag,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "pse_shift",
                                                      "When query_rope is not empty, pse_shift must be empty"),
                return ge::GRAPH_FAILED);

    // 不支持alibepse
    OP_CHECK_IF(fiaInfo.enableAlibiPse,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "pse_type", "2/3",
                                                      "When query_rope is not empty, pseType cannot be 2 or 3"),
                return ge::GRAPH_FAILED);

    if (fiaInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B) {
        // 不支持左padding
        OP_CHECK_IF(fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "query_padding_size or kv_padding_size",
                "When query_rope is not empty, query_padding_size and kv_padding_size must both be empty"),
            return ge::GRAPH_FAILED);
        // 不支持tensorlist
        OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "key",
                                                          "When query_rope is not empty, key cannot be TensorList"),
                    return ge::GRAPH_FAILED);
        // 不支持后量化
        OP_CHECK_IF(fiaInfo.isOutQuantEnable,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "attention_out",
                                                          ToString(fiaInfo.outputType).c_str(),
                                                          "When query_rope is not empty, postQuant is not supported"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckFeatureDecodeMLA(const FiaTilingInfo &fiaInfo) const
{
    if (fiaInfo.mlaMode != MlaMode::ROPE_SPLIT_D512) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(fiaInfo.qPaddingSizeFlag || fiaInfo.kvPaddingSizeFlag,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "query_padding_size and kv_padding_size",
            "In MLA decode scenario, query_padding_size and kv_padding_size must both be empty"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(fiaInfo.kvStorageMode == KvStorageMode::TENSOR_LIST,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "key",
                                                      "When query_rope is not empty, key cannot be TensorList"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckFeatureAntiQuant(const FiaTilingInfo &fiaInfo) const
{
    // 不支持伪量化
    OP_CHECK_IF(fiaInfo.opParamInfo.queryRope.tensor != nullptr || fiaInfo.opParamInfo.keyRope.tensor != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "query_rope or key_rope",
                    "In antiquant mode, query_rope and key_rope must both be empty"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// check QS size
ge::graphStatus RopeChecker::CheckQSSize(const FiaTilingInfo &fiaInfo) const
{
    if (fiaInfo.mlaMode != MlaMode::ROPE_SPLIT_D512 || fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }

    // 全量化场景下 QS仅支持1-16
    constexpr uint32_t maxQuerySeqLenForMLAFullquant = 16U;

    if (fiaInfo.s1Size < NUM1) {
        std::string shapeStr = ToStringRaw(fiaInfo.opParamInfo.query.shape->GetStorageShape());
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "query", shapeStr.c_str(),
                                              "In Decode MLA scenario, query sequence length must be positive");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(enableFullQuant_ && (fiaInfo.s1Size > maxQuerySeqLenForMLAFullquant),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    fiaInfo.opName, "query",
                    ToStringRaw(fiaInfo.opParamInfo.query.shape->GetStorageShape()).c_str(),
                    "In MLA decode fullquant scenario, the sequence length of query must be within the range [1, 16]"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Decode MLA N1:1/2/4/8/16/32/64/128 N2:1
ge::graphStatus RopeChecker::CheckNSize(const FiaTilingInfo &fiaInfo) const
{
    if (fiaInfo.mlaMode != MlaMode::ROPE_SPLIT_D512) {
        return ge::GRAPH_SUCCESS;
    }
    static const std::set<uint32_t> supportNumHeadForMLA = {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U};
    OP_CHECK_IF((supportNumHeadForMLA.find(fiaInfo.n1Size) == supportNumHeadForMLA.end()),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    fiaInfo.opName, "query",
                    ToStringRaw(fiaInfo.opParamInfo.query.shape->GetStorageShape()).c_str(),
                    "In MLA decode scenario, the N axis of query must be in {1,2,4,8,16,32,64,128}"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((*fiaInfo.opParamInfo.kvHeadNums != NUM1),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    fiaInfo.opName, "key",
                    ToStringRaw(fiaInfo.opParamInfo.key.shape->GetStorageShape()).c_str(),
                    "In MLA decode scenario, the N axis of key must be 1"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckAxisSupport(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckQSSize(fiaInfo) || ge::GRAPH_SUCCESS != CheckNSize(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckRopeDtype(fiaInfo) || ge::GRAPH_SUCCESS != CheckRopeDSizeSupport(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckRopeExistence(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.ropeMode != RopeMode::ROPE_SPLIT) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckFeatureSupport(fiaInfo) || ge::GRAPH_SUCCESS != CheckFeatureDecodeMLA(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckAxisSupport(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckKRopeContiguous(fiaInfo)) {
            return ge::GRAPH_FAILED;
    }

    if (enableAntiQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFeatureAntiQuant(fiaInfo)) {
            return ge::GRAPH_FAILED;
        };
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckRopeConsistency(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckShapeConsistency(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckRopeDtypeConsistency(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
