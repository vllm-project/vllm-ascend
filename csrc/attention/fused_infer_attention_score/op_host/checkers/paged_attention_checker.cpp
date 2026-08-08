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
 * \file paged_attention_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "paged_attention_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// 公共校验函数
// check blocktable dtype
ge::graphStatus PagedAttentionChecker::CheckBlockTableDtype(const FiaTilingInfo &fiaInfo) const
{
    if (fiaInfo.opParamInfo.blockTable.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::CompileTimeTensorDesc *blockTableDesc = fiaInfo.opParamInfo.blockTable.desc;
    OP_CHECK_IF(blockTableDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(fiaInfo.opName,
                "When page attention enable, blockTable datatype only support INT32."),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// check blockTable shape size
ge::graphStatus PagedAttentionChecker::CheckBlockTableShapeSize(const FiaTilingInfo &fiaInfo) const
{
    if (fiaInfo.opParamInfo.blockTable.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::Shape blockTableShape = fiaInfo.opParamInfo.blockTable.tensor->GetStorageShape();
    // check dim num
    if (blockTableShape.GetDimNum() != 2) {
        std::string dimStr = std::to_string(blockTableShape.GetDimNum()) + "D";
        std::string reasonMsg = "When blockTable is not empty, the shape of blockTable must be 2D";
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "blockTable", dimStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    // check blockTable each dim cannot be 0
    if (blockTableShape.GetShapeSize() == 0) {
        std::string shapeStr = ToStringRaw(blockTableShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            fiaInfo.opName, "blockTable", shapeStr.c_str(),
            "When page attention enable, all axes of blockTable must be positive numbers");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// check blocksize
ge::graphStatus PagedAttentionChecker::CheckBlockSize(const FiaTilingInfo &fiaInfo) const
{
    OP_CHECK_IF(fiaInfo.opParamInfo.blockSize == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "block_size", "null",
            "When page attention enable, block_size cannot be null"),
            return ge::GRAPH_FAILED);
    // blockSize 需要大于0
    OP_CHECK_IF(fiaInfo.blockSize <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "block_size", std::to_string(fiaInfo.blockSize).c_str(),
            "When page attention enable, block_size must be positive"),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckBlockTableExistence(const FiaTilingInfo &fiaInfo) const
{
    OP_CHECK_IF(fiaInfo.opParamInfo.blockTable.tensor == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "block_table",
            "When page attention is enabled, block_table cannot be empty"),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckFeatureSupport(const FiaTilingInfo &fiaInfo) const
{
    OP_CHECK_IF((fiaInfo.opParamInfo.queryPaddingSize.tensor != nullptr) ||
        (fiaInfo.opParamInfo.kvPaddingSize.tensor != nullptr),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "query_padding_size or kv_padding_size",
            "When page attention is enabled, query_padding_size and kv_padding_size must both be empty"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((fiaInfo.opParamInfo.keySharedPrefix.tensor != nullptr) ||
        (fiaInfo.opParamInfo.valueSharedPrefix.tensor != nullptr),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(fiaInfo.opName, "key_shared_prefix or value_shared_prefix",
            "When page attention is enabled, key_shared_prefix and value_shared_prefix must both be empty"),
        return ge::GRAPH_FAILED);
    if (fiaInfo.npuArch == NpuArch::DAV_3510) {
        if (fiaInfo.isQKVDDifferent) {
            std::string shapeMsg = ToString(fiaInfo.opParamInfo.query.shape->GetStorageShape()) + ", " +
                ToString(fiaInfo.opParamInfo.key.shape->GetStorageShape()) + " and " +
                ToString(fiaInfo.opParamInfo.value.shape->GetStorageShape());
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(fiaInfo.opName, "query, key and value", shapeMsg.c_str(),
                "When page attention is enabled, the headDim of query, key and value must be the same");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckSeqLengthKVExistence(const FiaTilingInfo &fiaInfo) const
{
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(fiaInfo.opParamInfo.actualSeqLengths.tensor == nullptr ||
        fiaInfo.opParamInfo.actualSeqLengths.tensor->GetData<int64_t>() == nullptr ||
        fiaInfo.opParamInfo.actualSeqLengths.tensor->GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "actual_seq_lengths_kv", "empty",
            "When page attention enable, actual_seq_lengths_kv cannot be empty"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

int64_t PagedAttentionChecker::GetMaxBlockNumPerBatch(const FiaTilingInfo &fiaInfo) const
{
    const int32_t blockSize = fiaInfo.blockSize;
    const gert::Tensor* actSeqLenKV = fiaInfo.opParamInfo.actualSeqLengths.tensor;
    uint32_t actualSeqLengthsKVSize = static_cast<uint32_t>(actSeqLenKV->GetShapeSize());
    int64_t actualSeqKVPerBatch = 0;
    int64_t blockNumPerBatch = 0;
    int64_t maxBlockNumPerBatch = 0;
    uint32_t loop = std::min(actualSeqLengthsKVSize, fiaInfo.bSize);
    if (actSeqLenKV->GetData<int64_t>() == nullptr) {
        return 0;
    }
    for (uint32_t i = 0; i < loop; i++) {
        actualSeqKVPerBatch = actSeqLenKV->GetData<int64_t>()[i];
        blockNumPerBatch = (actualSeqKVPerBatch + blockSize - 1) / blockSize;
        if (blockNumPerBatch > maxBlockNumPerBatch) {
            maxBlockNumPerBatch = blockNumPerBatch;
        }
    }
    return maxBlockNumPerBatch;
}

// check mask shape
ge::graphStatus PagedAttentionChecker::CheckMaskShape(const FiaTilingInfo &fiaInfo)
{
    if ((fiaInfo.sparseMode == SPARSE_MODE_NO_MASK || fiaInfo.sparseMode == SPARSE_MODE_ALL_MASK) &&
        fiaInfo.opParamInfo.attenMask.tensor != nullptr) {
        if (fiaInfo.opParamInfo.actualSeqLengths.tensor == nullptr) {
            return ge::GRAPH_SUCCESS;
        }
        const gert::Shape attenMaskShape = fiaInfo.opParamInfo.attenMask.tensor->GetStorageShape();
        int64_t maxBlockNumPerBatch = GetMaxBlockNumPerBatch(fiaInfo);
        uint32_t attenMaskDimNum = attenMaskShape.GetDimNum();
        if (attenMaskDimNum > 0 &&
            (attenMaskShape.GetDim(attenMaskDimNum - 1) < maxBlockNumPerBatch * fiaInfo.blockSize)) {
            std::string shapeStr = ToStringRaw(attenMaskShape);
            std::string reasonMsg =
                std::string("When page attention enable and attenMask enable, "
                            "the last dimension of input atten_mask must be >= maxBlockNumPerBatch(") +
                std::to_string(maxBlockNumPerBatch) + ") * blockSize(" + std::to_string(fiaInfo.blockSize) + ")";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "atten_mask", shapeStr.c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// check pse shape
ge::graphStatus PagedAttentionChecker::CheckPseShape(const FiaTilingInfo &fiaInfo)
{
    if (!fiaInfo.pseShiftFlag) {
        // 若不使能pse，则放弃后续校验
        return ge::GRAPH_SUCCESS;
    }
    // Page attention使能场景下，传入的PseShift的最后一维需要大于等于maxBlockNumPerSeq * blockSize
    if (*fiaInfo.opParamInfo.pseType != 0) {
        uint32_t pseShiftS2 = fiaInfo.pseShiftS2;
        int32_t blockSize = fiaInfo.blockSize;
        uint32_t maxBlockNumPerBatch = fiaInfo.maxBlockNumPerBatch;
        if (pseShiftS2 < maxBlockNumPerBatch * blockSize) {
            std::string reason = "The last axis of pse_shift must be greater than or equal to maxBlockNumPerBatch(" +
                std::to_string(maxBlockNumPerBatch) + ") * blockSize(" + std::to_string(blockSize) +
                ") when page attention is enabled";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, "pse_shift",
                ToStringRaw(fiaInfo.opParamInfo.pseShift.tensor->GetStorageShape()).c_str(), reason.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// check pa cache shape
ge::graphStatus PagedAttentionChecker::CheckPACacheShape3D(const FiaTilingInfo &fiaInfo, const gert::Shape &tempShape,
    const std::string &inputName, uint32_t compareD, const std::string &shapeStr) const
{
    int64_t tempBlockSize = tempShape.GetDim(DIM_NUM_1);
    int64_t tempH = tempShape.GetDim(DIM_NUM_2);
    if (tempBlockSize != fiaInfo.blockSize) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, inputName.c_str(), shapeStr.c_str(),
            ("When page attention is enabled, blockSize of " + inputName +
                " must be " + std::to_string(fiaInfo.blockSize)).c_str());
        return ge::GRAPH_FAILED;
    }

    if (fiaInfo.inputKvType == ge::DT_INT4) {
        if (tempH != fiaInfo.n2Size * compareD) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, inputName.c_str(), shapeStr.c_str(),
                ("When page attention is enabled, if input kv dataType is INT32, the axis H of " +
                    inputName + " must be " + std::to_string(fiaInfo.n2Size * compareD / NUM8) +
                    "; if input kv dataType is INT4, the axis H of " + inputName +
                    " must be " + std::to_string(fiaInfo.n2Size * compareD)).c_str());
            return ge::GRAPH_FAILED;
        }

        if (tempH > H_LIMIT) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, inputName.c_str(), shapeStr.c_str(),
                ("When page attention is enabled and layout is BSH, if input kv dataType is INT32, "
                    "the axis H of " + inputName + " cannot be greater than " +
                    std::to_string(H_LIMIT / NUM8) + "; if input kv dataType is INT4, "
                    "the axis H of " + inputName + " cannot be greater than " +
                    std::to_string(H_LIMIT)).c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        if (tempH != fiaInfo.n2Size * compareD) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, inputName.c_str(), shapeStr.c_str(),
                ("When page attention is enabled, if input kv dataType is INT32, the axis H of " +
                    inputName + " must be " + std::to_string(fiaInfo.n2Size * compareD / NUM8) +
                    "; if input kv dataType is INT4, the axis H of " + inputName +
                    " must be " + std::to_string(fiaInfo.n2Size * compareD)).c_str());
            return ge::GRAPH_FAILED;
        }

        if (tempH > H_LIMIT) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, inputName.c_str(), shapeStr.c_str(),
                ("When page attention is enabled and layout is BSH, if input kv dataType is INT32, "
                    "the axis H of " + inputName + " cannot be greater than " +
                    std::to_string(H_LIMIT / NUM8) + "; if input kv dataType is INT4, "
                    "the axis H of " + inputName + " cannot be greater than " +
                    std::to_string(H_LIMIT)).c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckPACacheShape4D(const FiaTilingInfo &fiaInfo, const gert::Shape &tempShape,
    const std::string &inputName, uint32_t compareD, const std::string &shapeStr) const
{
    int64_t tempN = fiaInfo.kvLayout == FiaLayout::BnNBsD ?
        tempShape.GetDim(DIM_NUM_1) : tempShape.GetDim(DIM_NUM_2);
    int64_t tempBlockSize = fiaInfo.kvLayout == FiaLayout::BnNBsD ?
        tempShape.GetDim(DIM_NUM_2) : tempShape.GetDim(DIM_NUM_1);
    int64_t tempD = tempShape.GetDim(DIM_NUM_3);

    if (tempN != fiaInfo.n2Size) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, inputName.c_str(), shapeStr.c_str(),
            ("When page attention is enabled, the axis N of kvCache must be " +
                std::to_string(fiaInfo.n2Size)).c_str());
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(tempBlockSize != fiaInfo.blockSize,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, inputName.c_str(), shapeStr.c_str(),
                        ("When page attention is enabled, blockSize of " + inputName +
                         " must be " + std::to_string(fiaInfo.blockSize)).c_str()),
                    return ge::GRAPH_FAILED);

    if (tempD != compareD) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, inputName.c_str(), shapeStr.c_str(),
            ("When page attention is enabled, the D axis of " + inputName +
                " must be " + std::to_string(compareD)).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckPACacheShapeNZAntiquant(const FiaTilingInfo &fiaInfo,
    const std::string &inputName, const std::string &shapeStr, uint32_t compareD,
    int64_t tempD0, int64_t tempD1) const
{
    uint32_t d0Size = NUM_16;
    if (fiaInfo.inputKvType == ge::DT_INT4) {
        if (tempD0 != d0Size) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, inputName.c_str(), shapeStr.c_str(),
                ("When PA_NZ is enabled, if input kv dataType is INT32, the last dim of kvCache "
                    "must be " + std::to_string(d0Size / NUM8) +
                    "; if input kv dataType is INT4, the last dim of " + inputName +
                    " must be " + std::to_string(d0Size)).c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        if (tempD0 != d0Size) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, inputName.c_str(), shapeStr.c_str(),
                ("When PA_NZ is enabled, if input kv dataType is INT32, the last dim of kvCache "
                    "must be " + std::to_string(d0Size / NUM8) +
                    "; if input kv dataType is INT4, the last dim of " + inputName +
                    " must be " + std::to_string(d0Size)).c_str());
            return ge::GRAPH_FAILED;
        }
    }
    if (tempD1 != compareD / d0Size) {
        std::string reasonMsg = "When PA_NZ is enabled, in " +
            std::string(QuantModeToSerialString(fiaInfo.quantMode)) + " " +
            std::string(SituationToSerialString(fiaInfo.ropeMode)) +
            " situation, the third dim of kvCache must be " + std::to_string(compareD / d0Size);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, inputName.c_str(),
            shapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckPACacheShapeNZNonAntiquant(const FiaTilingInfo &fiaInfo,
    const std::string &inputName, const std::string &shapeStr, uint32_t compareD,
    int64_t tempD0, int64_t tempD1) const
{
    std::unordered_map<ge::DataType, float> typeSizeMap = {
        {ge::DT_FLOAT16, static_cast<float>(FLOAT16SIZE)},
        {ge::DT_BF16, static_cast<float>(BFLOAT16SIZE)},
        {ge::DT_INT8, static_cast<float>(INT8SIZE)},
        {ge::DT_HIFLOAT8, static_cast<float>(FLOAT8SIZE)},
        {ge::DT_FLOAT8_E4M3FN, static_cast<float>(FLOAT8SIZE)}};
    float dataTypeSizeValue = static_cast<float>(FLOAT16SIZE);
    auto inputTypeCheck = typeSizeMap.find(fiaInfo.inputKvType);
    if (inputTypeCheck != typeSizeMap.end()) {
        dataTypeSizeValue = inputTypeCheck->second;
    }

    if (enableFullQuant_ && inputName == "keyRope") {
        dataTypeSizeValue = static_cast<float>(BFLOAT16SIZE);
    }

    uint32_t d0Size = BYTE_BLOCK / dataTypeSizeValue;
    if (tempD0 != d0Size) {
        std::string reasonMsg = "When PA_NZ is enabled, in " +
            std::string(QuantModeToSerialString(fiaInfo.quantMode)) + " " +
            std::string(SituationToSerialString(fiaInfo.ropeMode)) +
            " situation, the last dim of kvCache must be " + std::to_string(d0Size);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, inputName.c_str(),
            shapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    if (tempD1 != compareD / d0Size) {
        std::string reasonMsg = "When PA_NZ is enabled, in " +
            std::string(QuantModeToSerialString(fiaInfo.quantMode)) + " " +
            std::string(SituationToSerialString(fiaInfo.ropeMode)) +
            " situation, the third dim of kvCache must be " + std::to_string(compareD / d0Size);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, inputName.c_str(),
            shapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckPACacheShape(const FiaTilingInfo &fiaInfo,
    const gert::Shape tempShape, const std::string& inputName) const
{
    uint32_t shapeDim = tempShape.GetDimNum();

    uint32_t compareD = 0;
    if (inputName == "key") {
        compareD = fiaInfo.qkHeadDim;
    } else if (inputName == "keyRope") {
        compareD = fiaInfo.ropeHeadDim;
    } else {
        compareD = fiaInfo.vHeadDim;
    }

    std::string shapeStr = ToStringRaw(tempShape);

    if (shapeDim == DIM_NUM_3) { // [blockNums, blockSize, H]
        return CheckPACacheShape3D(fiaInfo, tempShape, inputName, compareD, shapeStr);
    } else if (shapeDim == DIM_NUM_4) { // [blockNums, N, blockSize, D] or [blockNums, blockSize, N, D]
        return CheckPACacheShape4D(fiaInfo, tempShape, inputName, compareD, shapeStr);
    } else { // [blockNums, N, D1, blocksize, D0]
        int64_t tempN = tempShape.GetDim(DIM_NUM_1);
        int64_t tempD1 = tempShape.GetDim(DIM_NUM_2);
        int64_t tempBlockSize = tempShape.GetDim(DIM_NUM_3);
        int64_t tempD0 = tempShape.GetDim(DIM_NUM_4);

        if (tempN != fiaInfo.n2Size) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, inputName.c_str(), shapeStr.c_str(),
                ("When page attention is enabled, the axis N of " + inputName +
                    " must be " + std::to_string(fiaInfo.n2Size)).c_str());
            return ge::GRAPH_FAILED;
        }

        if (tempBlockSize != fiaInfo.blockSize) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(fiaInfo.opName, inputName.c_str(), shapeStr.c_str(),
                ("When page attention is enabled, blockSize of " + inputName +
                    " must be " + std::to_string(fiaInfo.blockSize)).c_str());
            return ge::GRAPH_FAILED;
        }

        if (enableAntiQuant_) {
            return CheckPACacheShapeNZAntiquant(fiaInfo, inputName, shapeStr, compareD, tempD0, tempD1);
        } else {
            return CheckPACacheShapeNZNonAntiquant(fiaInfo, inputName, shapeStr, compareD, tempD0, tempD1);
        }
    }
}

// check input query dtype
ge::graphStatus PagedAttentionChecker::CheckQDtypeSupport(const FiaTilingInfo &fiaInfo)
{
    OP_CHECK_IF(fiaInfo.inputQType == ge::DT_INT8 && fiaInfo.ropeMode != RopeMode::ROPE_SPLIT,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(fiaInfo.opName, "query", ToString(fiaInfo.inputQType).c_str(),
            "When the page attention function is enabled, the data type of the query operation cannot be INT8 in"
            " the GQA scenario. INT8 is supported only in the full quantization scenario of MLA"),
            return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckBlockTableShape(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.isMaxWorkspace) {
        return ge::GRAPH_SUCCESS;
    }
    
    if (fiaInfo.opParamInfo.actualSeqLengths.tensor == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    // 每个batch的 blockNum 小于 blocktable dim2
    int64_t maxBlockNumPerBatch = GetMaxBlockNumPerBatch(fiaInfo);
    const gert::Shape blockTableShape = fiaInfo.opParamInfo.blockTable.tensor->GetStorageShape();
    if ((blockTableShape.GetDim(0) != fiaInfo.bSize) || (blockTableShape.GetDim(1) < maxBlockNumPerBatch)) {
        std::string shapeStr = ToStringRaw(blockTableShape);
        std::string reasonMsg = "When page attention enable, block_table shape must be [batch_size(" +
            std::to_string(fiaInfo.bSize) + "), >=max_block_num_per_batch(" +
            std::to_string(maxBlockNumPerBatch) + ")]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            fiaInfo.opName, "block_table", shapeStr.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }

    // check key cache shape
    if (ge::GRAPH_SUCCESS != CheckPACacheShape(fiaInfo, fiaInfo.opParamInfo.key.shape->GetStorageShape(), "key") ||
        ge::GRAPH_SUCCESS != CheckPACacheShape(fiaInfo, fiaInfo.opParamInfo.value.shape->GetStorageShape(), "value")) {
        return ge::GRAPH_FAILED;
    }

    // check rope cache shape
    if (fiaInfo.ropeMode == RopeMode::ROPE_SPLIT) {
        if (ge::GRAPH_SUCCESS != CheckPACacheShape(fiaInfo, fiaInfo.opParamInfo.keyRope.tensor->GetStorageShape(),
            "keyRope")) {
            return ge::GRAPH_FAILED;
        }
    }

    // warning: S2 <= 20M
    if (maxBlockNumPerBatch * fiaInfo.blockSize > S_LIMIT) {
        OP_LOGW(fiaInfo.opName,
            "When page attention enable, sequence length(%ld) of kv should <= 20M.",
            maxBlockNumPerBatch * fiaInfo.blockSize);
    }
    return ge::GRAPH_SUCCESS;
}

// check blocksize
ge::graphStatus PagedAttentionChecker::CheckBlockSizeNonQuant910B(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.ropeMode != RopeMode::NO_ROPE) { // MLA场景 [16, 1024]且16对齐
        if (fiaInfo.blockSize > BLOCK_SIZE_MAX_FOR_NO_QUANT || fiaInfo.blockSize < BLOCK_SIZE_ALIGN_SIZE_16 ||
            fiaInfo.blockSize % BLOCK_SIZE_ALIGN_SIZE_16 != 0) {
            std::string reasonMsg =
                "In no quant GQA (QS > 1) scenario, when page attention enable, blockSize(" +
                std::to_string(fiaInfo.blockSize) + ") must be a multiple of " +
                std::to_string(BLOCK_SIZE_ALIGN_SIZE_16) + ", and must be within the range [" +
                std::to_string(BLOCK_SIZE_ALIGN_SIZE_16) + ", " + std::to_string(BLOCK_SIZE_MAX_FOR_NO_QUANT) +
                "]";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "blockSize",
                                                  std::to_string(fiaInfo.blockSize).c_str(), reasonMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (fiaInfo.qkHeadDim == NUM_64 || fiaInfo.qkHeadDim == NUM_128) { // GQA D =64/128场景 [16, 1024]且16对齐
        if (fiaInfo.blockSize > BLOCK_SIZE_MAX_FOR_NO_QUANT || fiaInfo.blockSize < BLOCK_SIZE_ALIGN_SIZE_16 ||
            fiaInfo.blockSize % BLOCK_SIZE_ALIGN_SIZE_16 != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                fiaInfo.opName, "block_size", std::to_string(fiaInfo.blockSize).c_str(),
                "a multiple of " + std::to_string(BLOCK_SIZE_ALIGN_SIZE_16) + " in range of [" +
                    std::to_string(BLOCK_SIZE_ALIGN_SIZE_16) + ", " +
                    std::to_string(BLOCK_SIZE_MAX_FOR_NO_QUANT) + "]");
            return ge::GRAPH_FAILED;
        }
    } else {
        // GQA D != 64/128, QS > 1 [128, 1024]且128对齐
        if ((fiaInfo.s1Size > NUM1) &&
            (fiaInfo.blockSize > BLOCK_SIZE_MAX_FOR_NO_QUANT || fiaInfo.blockSize < BLOCK_SIZE_ALIGN_SIZE_128 ||
             fiaInfo.blockSize % BLOCK_SIZE_ALIGN_SIZE_128 != 0)) {
            OP_LOGE_FOR_INVALID_VALUE(fiaInfo.opName, "block_size", std::to_string(fiaInfo.blockSize).c_str(),
                                      "a multiple of " + std::to_string(BLOCK_SIZE_ALIGN_SIZE_128) +
                                      " in range of [" + std::to_string(BLOCK_SIZE_ALIGN_SIZE_128) + ", " +
                                      std::to_string(BLOCK_SIZE_MAX_FOR_NO_QUANT) + "]");
            return ge::GRAPH_FAILED;
        }
        // GQA D != 64/128, QS = 1 [16, 512]且16对齐
        if ((fiaInfo.s1Size == NUM1) &&
            (fiaInfo.blockSize > BLOCK_SIZE_MAX || fiaInfo.blockSize < BLOCK_SIZE_ALIGN_SIZE_16 ||
             fiaInfo.blockSize % BLOCK_SIZE_ALIGN_SIZE_16 != 0)) {
            OP_LOGE_FOR_INVALID_VALUE(fiaInfo.opName, "block_size", std::to_string(fiaInfo.blockSize).c_str(),
                                      "a multiple of " + std::to_string(BLOCK_SIZE_ALIGN_SIZE_16) +
                                      " in range of [" + std::to_string(BLOCK_SIZE_ALIGN_SIZE_16) + ", " +
                                      std::to_string(BLOCK_SIZE_MAX) + "]");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckBlockSizeNonQuantOther(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.ropeMode != RopeMode::NO_ROPE) { // MLA场景 [16, 1024]且16对齐
        if (fiaInfo.blockSize > BLOCK_SIZE_MAX_FOR_NO_QUANT || fiaInfo.blockSize < BLOCK_SIZE_ALIGN_SIZE_16 ||
            fiaInfo.blockSize % BLOCK_SIZE_ALIGN_SIZE_16 != 0) {
            OP_LOGE_FOR_INVALID_VALUE(fiaInfo.opName, "block_size", std::to_string(fiaInfo.blockSize).c_str(),
                                      "a multiple of " + std::to_string(BLOCK_SIZE_ALIGN_SIZE_16) +
                                          " in range of [" + std::to_string(BLOCK_SIZE_ALIGN_SIZE_16) + ", " +
                                          std::to_string(BLOCK_SIZE_MAX_FOR_NO_QUANT) + "]");
            return ge::GRAPH_FAILED;
        }
    } else if (fiaInfo.qkHeadDim == NUM_64 || fiaInfo.qkHeadDim == NUM_128) { // GQA D =64/128场景 [16, 1024]且16对齐
        if (fiaInfo.blockSize > BLOCK_SIZE_MAX_FOR_NO_QUANT || fiaInfo.blockSize < BLOCK_SIZE_ALIGN_SIZE_16 ||
            fiaInfo.blockSize % BLOCK_SIZE_ALIGN_SIZE_16 != 0) {
            OP_LOGE_FOR_INVALID_VALUE(fiaInfo.opName, "block_size", std::to_string(fiaInfo.blockSize).c_str(),
                                      "a multiple of " + std::to_string(BLOCK_SIZE_ALIGN_SIZE_16) +
                                          " in range of [" + std::to_string(BLOCK_SIZE_ALIGN_SIZE_16) + ", " +
                                          std::to_string(BLOCK_SIZE_MAX_FOR_NO_QUANT) + "]");
            return ge::GRAPH_FAILED;
        }
    } else {
        // GQA D != 64/128, QS > 1 [128, 1024]且128对齐
        if ((fiaInfo.s1Size > NUM1) &&
            (fiaInfo.blockSize > BLOCK_SIZE_MAX_FOR_NO_QUANT || fiaInfo.blockSize < BLOCK_SIZE_ALIGN_SIZE_128 ||
             fiaInfo.blockSize % BLOCK_SIZE_ALIGN_SIZE_128 != 0)) {
            OP_LOGE_FOR_INVALID_VALUE(fiaInfo.opName, "block_size", std::to_string(fiaInfo.blockSize).c_str(),
                                      "a multiple of " + std::to_string(BLOCK_SIZE_ALIGN_SIZE_128) +
                                          " in range of [" + std::to_string(BLOCK_SIZE_ALIGN_SIZE_128) + ", " +
                                          std::to_string(BLOCK_SIZE_MAX_FOR_NO_QUANT) + "]");
            return ge::GRAPH_FAILED;
        }
        // GQA D != 64/128, QS = 1 [16, 512]且16对齐
        if ((fiaInfo.s1Size == NUM1) &&
            (fiaInfo.blockSize > BLOCK_SIZE_MAX || fiaInfo.blockSize < BLOCK_SIZE_ALIGN_SIZE_16 ||
             fiaInfo.blockSize % BLOCK_SIZE_ALIGN_SIZE_16 != 0)) {
            OP_LOGE_FOR_INVALID_VALUE(fiaInfo.opName, "block_size", std::to_string(fiaInfo.blockSize).c_str(),
                                      "a multiple of " + std::to_string(BLOCK_SIZE_ALIGN_SIZE_16) +
                                          " in range of [" + std::to_string(BLOCK_SIZE_ALIGN_SIZE_16) + ", " +
                                          std::to_string(BLOCK_SIZE_MAX) + "]");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckBlockSizeAntiquant(const FiaTilingInfo &fiaInfo)
{
    std::unordered_map<ge::DataType, float> typeSizeMap = {{ge::DT_FLOAT16, static_cast<float>(FLOAT16SIZE)},
                                                           {ge::DT_BF16, static_cast<float>(BFLOAT16SIZE)},
                                                           {ge::DT_INT8, static_cast<float>(INT8SIZE)},
                                                           {ge::DT_HIFLOAT8, static_cast<float>(FLOAT8SIZE)},
                                                           {ge::DT_FLOAT8_E4M3FN, static_cast<float>(FLOAT8SIZE)},
                                                           {ge::DT_INT4, INT4SIZE},
                                                           {ge::DT_FLOAT4_E2M1, FLOAT4SIZE}};
    float dataTypeSizeValue = static_cast<float>(FLOAT16SIZE);
    auto inputTypeCheck = typeSizeMap.find(fiaInfo.inputKvType);
    if (inputTypeCheck != typeSizeMap.end()) {
        dataTypeSizeValue = inputTypeCheck->second;
    }
    uint32_t blockSizeAlign = static_cast<uint32_t>(BYTE_BLOCK / dataTypeSizeValue);

    // 伪量化, 与Dtype相关
    if (fiaInfo.blockSize > BLOCK_SIZE_MAX ||
        fiaInfo.blockSize < blockSizeAlign || fiaInfo.blockSize % blockSizeAlign != 0) {
        std::string reasonMsg =
            "In antiquant scenario, when page attention is enabled, block_size must be a multiple of " +
            std::to_string(blockSizeAlign) + " and in range of [" +
            std::to_string(blockSizeAlign) + ", " + std::to_string(BLOCK_SIZE_MAX) +
            "] if kvCache datatype is " + DataTypeToSerialString(fiaInfo.inputKvType);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "block_size",
            std::to_string(fiaInfo.blockSize).c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// check blocksize
ge::graphStatus PagedAttentionChecker::CheckBlockSizeSupport(const FiaTilingInfo &fiaInfo)
{
    if (enableNonQuant_) { // 非量化
        if (fiaInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B) {
            return CheckBlockSizeNonQuant910B(fiaInfo);
        } else {
            return CheckBlockSizeNonQuantOther(fiaInfo);
        }
    } else if (enableAntiQuant_) {
        return CheckBlockSizeAntiquant(fiaInfo);
    } else { // 全量化
        if (fiaInfo.mlaMode == MlaMode::ROPE_SPLIT_D512 && fiaInfo.blockSize != NUM_128) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "block_size",
                std::to_string(fiaInfo.blockSize).c_str(),
                "In MLA fullquant scenario, when page attention is enabled, block_size must be 128");
            return ge::GRAPH_FAILED;
        }

        // mxfp8 仅支持blocksize等于512或者1024
        if (fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_MXFP8_FULL_QUANT &&
            (fiaInfo.blockSize != BLOCK_SIZE_FOR_MXFP8 && fiaInfo.blockSize != BLOCK_SIZE_1024_FOR_MXFP8)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "block_size",
                std::to_string(fiaInfo.blockSize).c_str(),
                "In MXFP8 fullquant scenario, when page attention is enabled, block_size must be 512 or 1024");
            return ge::GRAPH_FAILED;
        }

        // fp8 gqa 仅支持blocksize等于128
        OP_CHECK_IF(
            fiaInfo.fullQuantMode == FiaFullQuantMode::QK_PER_TOKEN_HEAD_V_PER_HEAD &&
            fiaInfo.blockSize != NUM_128,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "block_size",
                std::to_string(fiaInfo.blockSize).c_str(),
                "In FP8 GQA fullquant scenario, when page attention is enabled, block_size must be 128"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckKVLayout(const FiaTilingInfo &fiaInfo) const
{
    if (!enableFullQuant_) {
        return ge::GRAPH_SUCCESS;
    }
    const string inputLayout = fiaInfo.opParamInfo.layOut;
    const uint32_t dimNum = fiaInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum();
    if (fiaInfo.fullQuantMode == FiaFullQuantMode::QKV_MXFP8_FULL_QUANT) {
        OP_CHECK_IF(dimNum == 3, OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "key", "3D(BnBsH)",
            "In MXFP8 fullquant scenario, when Page Attention is enabled, the layout of key cannot be BnBsH"),
            return ge::GRAPH_FAILED);
    } else if (fiaInfo.fullQuantMode == FiaFullQuantMode::QK_PER_TOKEN_HEAD_V_PER_HEAD) {
        OP_CHECK_IF(dimNum != DIM_NUM_4,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(fiaInfo.opName, "key", std::to_string(dimNum).c_str(),
                        "In FP8 GQA fullquant scenario, KV cache layout must be BnNBsD, "
                        "PA BnBsH and NZ are not supported"),
                    return ge::GRAPH_FAILED);
    } else if (inputLayout == "BSH" || inputLayout == "BSND" || inputLayout == "BSH_NBSD" || inputLayout == "BSND_NBSD") {
        if (dimNum == 4) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "inputLayout", inputLayout.c_str(),
                "When page attention is enabled, PA BnNBsD format is not supported");
            return ge::GRAPH_FAILED;
        }
    }
    if (fiaInfo.socVersion == platform_ascendc::SocVersion::ASCEND910B) {
        OP_CHECK_IF(fiaInfo.kvLayout == FiaLayout::BnBsH && (fiaInfo.qLayout != FiaLayout::BSH && fiaInfo.qLayout != FiaLayout::BSND &&
                        fiaInfo.qLayout != FiaLayout::BNSD && fiaInfo.qLayout != FiaLayout::TND && fiaInfo.qLayout != FiaLayout::NTD),
            OP_LOGE(fiaInfo.opName, "In %s %s situation, the key/value's layout is BnBsH, query layout must be BSH, BSND, BNSD TND and TND in page attention scene, but got %s",
                QuantModeToSerialString(fiaInfo.quantMode).c_str(), SituationToSerialString(fiaInfo.ropeMode).c_str(), LayoutToSerialString(fiaInfo.qLayout).c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(fiaInfo.kvLayout == FiaLayout::BnNBsD && (fiaInfo.qLayout != FiaLayout::BSH && fiaInfo.qLayout != FiaLayout::BSND &&
                        fiaInfo.qLayout != FiaLayout::BNSD && fiaInfo.qLayout != FiaLayout::TND && fiaInfo.qLayout != FiaLayout::NTD),
            OP_LOGE(fiaInfo.opName, "In %s %s situation, the key/value's layout is BnNBsD, "
            "query layout must be BSH, BSND, BNSD TND and NTD in page attention scene, but got %s",
                QuantModeToSerialString(fiaInfo.quantMode).c_str(), SituationToSerialString(fiaInfo.ropeMode).c_str(), LayoutToSerialString(fiaInfo.qLayout).c_str()),
            return ge::GRAPH_FAILED);

        OP_CHECK_IF(fiaInfo.kvLayout == FiaLayout::NZ && (fiaInfo.qLayout != FiaLayout::BSH && fiaInfo.qLayout != FiaLayout::BSND &&
                        fiaInfo.qLayout != FiaLayout::BNSD && fiaInfo.qLayout != FiaLayout::TND && fiaInfo.qLayout != FiaLayout::NTD),
            OP_LOGE(fiaInfo.opName, "In %s %s situation, the key/value's layout is PA_NZ, "
                "query layout must be BSH, BSND, BNSD TND and NTD in page attention scene, but got %s",
                QuantModeToSerialString(fiaInfo.quantMode).c_str(), SituationToSerialString(fiaInfo.ropeMode).c_str(), LayoutToSerialString(fiaInfo.qLayout).c_str()),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckFeatureQueryS(const FiaTilingInfo &fiaInfo) const
{
    // When antiquantMode is 0 or 1 and data type of key/value is int8 scenario, page attention is not supported.
    if (fiaInfo.s1Size > 1) {
        int64_t keyAntiquantMode = 0;
        if (fiaInfo.opParamInfo.keyAntiquantMode != nullptr) {
            keyAntiquantMode = *fiaInfo.opParamInfo.keyAntiquantMode;
        }
        OP_CHECK_IF(
            (keyAntiquantMode == PER_CHANNEL_MODE || keyAntiquantMode == PER_TOKEN_MODE) &&
                fiaInfo.inputKvType == ge::DT_INT8,
            OP_LOGE(fiaInfo.opName,
                "In keyAntiquant/valueAntiquant split mode and data type of key/value is int8 scenario, if "
                "keyAntiquantMode/valueAntiquantMode is 0 or 1, page attention is not supported!"),
                return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckFeatureInputLayoutForAntiquant(const FiaTilingInfo &fiaInfo) const
{
    uint32_t kDimNum = fiaInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum();
    if (kDimNum == DIM_NUM_4 && fiaInfo.inputLayout != TilingKeyLayout::BNSD &&
        fiaInfo.inputLayout != TilingKeyLayout::TND) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(fiaInfo.opName, "inputLayout", fiaInfo.opParamInfo.layOut,
            "When Page Attention is enabled, and KV cache dimensions are 4-dimensional, "
            "inputLayout must be BNSD or TND");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }

    if (ge::GRAPH_SUCCESS != CheckBlockTableDtype(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckBlockTableShapeSize(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckBlockSize(fiaInfo)) {
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }

    if (ge::GRAPH_SUCCESS != CheckBlockTableExistence(fiaInfo)) {
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckSeqLengthKVExistence(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckKVLayout(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckMaskShape(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckPseShape(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckFeatureSupport(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckQDtypeSupport(fiaInfo)) {
            return ge::GRAPH_FAILED;
    }
    if (enableAntiQuant_) {
        if (ge::GRAPH_SUCCESS != CheckFeatureQueryS(fiaInfo) ||
            ge::GRAPH_SUCCESS != CheckFeatureInputLayoutForAntiquant(fiaInfo)) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PagedAttentionChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo)
{
    if (fiaInfo.kvStorageMode != KvStorageMode::PAGE_ATTENTION) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckBlockSizeSupport(fiaInfo) ||
        ge::GRAPH_SUCCESS != CheckBlockTableShape(fiaInfo)) {
            return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling