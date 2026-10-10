/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GENERIC_BLOCK_SPARSE_ATTENTION_METADATA_CHECK_H
#define GENERIC_BLOCK_SPARSE_ATTENTION_METADATA_CHECK_H

#include <cstring>
#include <string>

#include "opdev/data_type_utils.h"
#include "opdev/op_log.h"
#include "../generic_block_sparse_attention_metadata_tiling.h"

namespace {
constexpr int64_t GBSA_METADATA_DIM_NUM = 1;
constexpr int64_t GBSA_SEQ_LENGTH_DIM_NUM = 1;
constexpr int64_t GBSA_TND_SPARSE_BLOCK_IDX_DIM_NUM = 3;
constexpr int64_t GBSA_TND_SPARSE_BLOCK_COUNT_DIM_NUM = 2;
constexpr int64_t GBSA_BATCHED_SPARSE_BLOCK_IDX_DIM_NUM = 4;
constexpr int64_t GBSA_BATCHED_SPARSE_BLOCK_COUNT_DIM_NUM = 3;
constexpr int64_t GBSA_TND_BLOCK_CAPACITY_DIM_INDEX = 2;
constexpr int64_t GBSA_BATCHED_Q_BLOCK_DIM_INDEX = 2;
constexpr int64_t GBSA_BATCHED_BLOCK_CAPACITY_DIM_INDEX = 3;

constexpr int64_t GBSA_CURRENT_HEAD_DIM = 128;
constexpr int64_t GBSA_ARCH35_NONQUANT_MAX_HEAD_DIM = 512;
constexpr int64_t GBSA_CURRENT_MAX_GROUP_SIZE = 128;
constexpr int64_t GBSA_CURRENT_BLOCK_SHAPE_X = 1;
constexpr int64_t GBSA_CURRENT_BLOCK_SHAPE_Y = 128;
constexpr int64_t GBSA_LAYOUT_SPARSE_KVN_TOTALQB_KB = 4;
constexpr int64_t GBSA_RESIDUAL_BLOCK_MODE_NONE = 0;
constexpr int64_t GBSA_RESIDUAL_BLOCK_MODE_FORCE = 1;
constexpr int64_t GBSA_CURRENT_MAX_SPARSE_BLOCK_COUNT = 256;
constexpr int64_t GBSA_QUANT_TYPE_NONE = 0;
constexpr int64_t GBSA_QUANT_TYPE_FLOAT8 = 5;
constexpr int64_t GBSA_SOFTMAX_PRECISION_HIGH = 0;
constexpr int64_t GBSA_SOFTMAX_PRECISION_MIXED = 1;
constexpr int64_t GBSA_CURRENT_MASK_TYPE = 1;
constexpr int64_t GBSA_CURRENT_WINDOW_SIZE = -1;
constexpr int64_t GBSA_DEFAULT_MAX_SEQ_LEN = -1;

bool GbsaTensorValid(const aclTensor *tensor)
{
    return tensor != nullptr && tensor->GetViewShape().GetDimNum() > 0;
}

bool GbsaIsLayout(const char *layout, const char *expected)
{
    return layout != nullptr && std::strcmp(layout, expected) == 0;
}

aclnnStatus CheckGbsaTensorDataType(const aclTensor *tensor, aclDataType expectedDataType, const char *expectedTypeName,
                                    const char *tensorName)
{
    aclDataType dataType = ACL_DT_UNDEFINED;
    aclGetDataType(tensor, &dataType);
    if (dataType != expectedDataType) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s must be %s.", tensorName, expectedTypeName);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaTensorDim(const aclTensor *tensor, int64_t expectedDim, const char *tensorName)
{
    if (tensor->GetViewShape().GetDimNum() != expectedDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s dimension must be %ld.", tensorName, expectedDim);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaRequiredInt32Tensor(const aclTensor *tensor, const char *tensorName)
{
    if (!GbsaTensorValid(tensor)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s must be provided.", tensorName);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return CheckGbsaTensorDataType(tensor, ACL_INT32, "INT32", tensorName);
}

aclnnStatus CheckGbsaRequiredInputs(const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount,
                                    const aclTensor *metadata)
{
    aclnnStatus status = CheckGbsaRequiredInt32Tensor(sparseBlockIdx, "sparseBlockIdx");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaRequiredInt32Tensor(sparseBlockCount, "SparseBlockCount");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckGbsaRequiredInt32Tensor(metadata, "metadata");
}

aclnnStatus CheckGbsaMetadataShape(const aclTensor *metadata)
{
    aclnnStatus status = CheckGbsaTensorDim(metadata, GBSA_METADATA_DIM_NUM, "metadata");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (metadata->GetViewShape().GetDim(0) !=
        static_cast<int64_t>(optiling::generic_block_sparse_attention_metadata::METADATA_TOTAL_SIZE)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "metadata shape must be [%u].",
                optiling::generic_block_sparse_attention_metadata::METADATA_TOTAL_SIZE);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaPositiveScalarAttr(int64_t attrValue, const char *attrName)
{
    if (attrValue <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s must be greater than 0, but got %lld.", attrName, attrValue);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaHeadAttrs(int64_t numQHeads, int64_t numKvHeads)
{
    if (numQHeads < numKvHeads || numQHeads % numKvHeads != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "numQHeads must be greater than or equal to and divisible by numKvHeads, but got numQHeads=%lld "
                "and numKvHeads=%lld.",
                numQHeads, numKvHeads);
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t groupSize = numQHeads / numKvHeads;
    if (groupSize > GBSA_CURRENT_MAX_GROUP_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groupSize must be less than or equal to %lld, but got %lld.",
                GBSA_CURRENT_MAX_GROUP_SIZE, groupSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaBlockAttrs(int64_t blockShapeX, int64_t blockShapeY, int64_t layoutSparsePattern,
                                bool arch35NonQuant = false)
{
    if (blockShapeX != GBSA_CURRENT_BLOCK_SHAPE_X) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeX currently only supports %lld, but got %lld.",
                GBSA_CURRENT_BLOCK_SHAPE_X, blockShapeX);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (arch35NonQuant ? (blockShapeY < 1 || blockShapeY > GBSA_CURRENT_BLOCK_SHAPE_Y) :
                        (blockShapeY != GBSA_CURRENT_BLOCK_SHAPE_Y)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeY=%lld is unsupported (arch35NonQuant=%d).",
                blockShapeY, arch35NonQuant);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (layoutSparsePattern != GBSA_LAYOUT_SPARSE_KVN_TOTALQB_KB) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "layout_sparse_pattern currently only supports %lld (KVN_TotalQB_KB), but got %lld.",
                GBSA_LAYOUT_SPARSE_KVN_TOTALQB_KB, layoutSparsePattern);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaScalarAttrs(int64_t numQHeads, int64_t numKvHeads, int64_t headDim, int64_t blockShapeX,
                                 int64_t blockShapeY, int64_t layoutSparsePattern, bool arch35NonQuant = false)
{
    const struct {
        int64_t value;
        const char *name;
    } positiveAttrs[] = {
        {numQHeads, "numQHeads"}, {numKvHeads, "numKvHeads"}, {headDim, "headDim"}, {blockShapeY, "blockShapeY"}};
    for (const auto &attr : positiveAttrs) {
        const aclnnStatus status = CheckGbsaPositiveScalarAttr(attr.value, attr.name);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    aclnnStatus status = CheckGbsaHeadAttrs(numQHeads, numKvHeads);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (arch35NonQuant ? (headDim > GBSA_ARCH35_NONQUANT_MAX_HEAD_DIM) : (headDim != GBSA_CURRENT_HEAD_DIM)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "headDim=%lld is unsupported (arch35NonQuant=%d).", headDim, arch35NonQuant);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return CheckGbsaBlockAttrs(blockShapeX, blockShapeY, layoutSparsePattern, arch35NonQuant);
}

aclnnStatus CheckGbsaMaxSeqLenAttr(int64_t maxSeqLen, const char *attrName)
{
    if (maxSeqLen < GBSA_DEFAULT_MAX_SEQ_LEN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s must be -1 or greater than or equal to 0, but got %lld.", attrName,
                maxSeqLen);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaMaxSeqLenAttrs(int64_t maxQSeqLen, int64_t maxKvSeqLen)
{
    aclnnStatus status = CheckGbsaMaxSeqLenAttr(maxQSeqLen, "maxQSeqLen");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckGbsaMaxSeqLenAttr(maxKvSeqLen, "maxKvSeqLen");
}

aclnnStatus CheckGbsaLayouts(const char *qInputLayout, const char *kvInputLayout)
{
    if (!GbsaIsLayout(qInputLayout, "TND")) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "qInputLayout currently only supports TND.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!GbsaIsLayout(kvInputLayout, "PA_BBND") && !GbsaIsLayout(kvInputLayout, "TND")) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kvInputLayout currently only supports PA_BBND or TND.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaOptionalSeqLength(const aclTensor *tensor, int64_t batch, aclDataType dataType,
                                       int64_t expectedElements, const char *typeName, const char *tensorName)
{
    if (!GbsaTensorValid(tensor)) {
        return ACLNN_SUCCESS;
    }
    aclnnStatus status = CheckGbsaTensorDataType(tensor, dataType, typeName, tensorName);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaTensorDim(tensor, GBSA_SEQ_LENGTH_DIM_NUM, tensorName);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (tensor->GetViewShape().GetDim(0) != expectedElements) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s shape does not match batch %lld.", tensorName, batch);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus GetGbsaBatch(const aclTensor *sparseBlockIdx, const aclTensor *cuSeqLengthsOptional,
                         const char *qInputLayout, int64_t &batch)
{
    if (!GbsaIsLayout(qInputLayout, "TND")) {
        batch = sparseBlockIdx->GetViewShape().GetDim(0);
        return batch >= 0 ? ACLNN_SUCCESS : ACLNN_ERR_PARAM_INVALID;
    }
    if (!GbsaTensorValid(cuSeqLengthsOptional)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLengths is required for TND query.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (cuSeqLengthsOptional->GetViewShape().GetDimNum() != GBSA_SEQ_LENGTH_DIM_NUM ||
        cuSeqLengthsOptional->GetViewShape().GetDim(0) < GBSA_SEQ_LENGTH_DIM_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLengths shape must be [batch + 1].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    batch = cuSeqLengthsOptional->GetViewShape().GetDim(0) - 1;
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaSparseTensorShapes(const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount,
                                        int64_t maxQSeqLen, int64_t numQHeads, int64_t numKvHeads, int64_t blockShapeX,
                                        int64_t layoutSparsePattern, const char *qInputLayout, int64_t batch)
{
    const int64_t sparseHeads = layoutSparsePattern == GBSA_LAYOUT_SPARSE_KVN_TOTALQB_KB ? numKvHeads : numQHeads;
    if (GbsaIsLayout(qInputLayout, "TND")) {
        if (CheckGbsaTensorDim(sparseBlockIdx, GBSA_TND_SPARSE_BLOCK_IDX_DIM_NUM, "sparseBlockIdx") != ACLNN_SUCCESS ||
            CheckGbsaTensorDim(sparseBlockCount, GBSA_TND_SPARSE_BLOCK_COUNT_DIM_NUM, "SparseBlockCount") !=
                ACLNN_SUCCESS) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (sparseBlockIdx->GetViewShape().GetDim(0) != sparseHeads ||
            sparseBlockCount->GetViewShape().GetDim(0) != sparseHeads ||
            sparseBlockIdx->GetViewShape().GetDim(1) != sparseBlockCount->GetViewShape().GetDim(1) ||
            sparseBlockIdx->GetViewShape().GetDim(GBSA_TND_BLOCK_CAPACITY_DIM_INDEX) <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TND sparse block tensor shapes do not match attrs.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    if (CheckGbsaTensorDim(sparseBlockIdx, GBSA_BATCHED_SPARSE_BLOCK_IDX_DIM_NUM, "sparseBlockIdx") != ACLNN_SUCCESS ||
        CheckGbsaTensorDim(sparseBlockCount, GBSA_BATCHED_SPARSE_BLOCK_COUNT_DIM_NUM, "SparseBlockCount") !=
            ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t qBlocks = maxQSeqLen / blockShapeX + static_cast<int64_t>(maxQSeqLen % blockShapeX != 0);
    if (sparseBlockIdx->GetViewShape().GetDim(0) != batch || sparseBlockCount->GetViewShape().GetDim(0) != batch ||
        sparseBlockIdx->GetViewShape().GetDim(1) != sparseHeads ||
        sparseBlockCount->GetViewShape().GetDim(1) != sparseHeads ||
        sparseBlockIdx->GetViewShape().GetDim(GBSA_BATCHED_Q_BLOCK_DIM_INDEX) != qBlocks ||
        sparseBlockCount->GetViewShape().GetDim(GBSA_BATCHED_Q_BLOCK_DIM_INDEX) != qBlocks ||
        sparseBlockIdx->GetViewShape().GetDim(GBSA_BATCHED_BLOCK_CAPACITY_DIM_INDEX) <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BSND/BNSD sparse block tensor shapes do not match attrs.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaSeqLengthInputs(const aclTensor *cuSeqLengthsOptional, const aclTensor *cuSeqLengthsKvOptional,
                                     const aclTensor *seqUsedQOptional, const aclTensor *seqUsedKvOptional,
                                     int64_t batch, const char *qInputLayout, const char *kvInputLayout)
{
    // Legacy issue: when BSND/BNSD provides both cuSeqLengths and seqUsedQ, the device side does not yet verify that
    // seqUsedQ is no greater than the corresponding prefix-sum difference.
    if (GbsaIsLayout(qInputLayout, "TND") && !GbsaTensorValid(cuSeqLengthsOptional)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLengths is required for TND query.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool isPaKv = GbsaIsLayout(kvInputLayout, "PA_BBND") || GbsaIsLayout(kvInputLayout, "PA_BNBD");
    if (GbsaIsLayout(kvInputLayout, "TND")) {
        if (!GbsaTensorValid(cuSeqLengthsKvOptional)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLengthsKv is required for TND KV.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (isPaKv) {
        // FA-style PA: sequsedKv required; cuSeqLengthsKv only for TND.
        if (!GbsaTensorValid(seqUsedKvOptional)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "seqUsedKv is required for paged KV.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (GbsaTensorValid(cuSeqLengthsKvOptional)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "cuSeqLengthsKv must be empty when layoutKv is PA, only supported in TND layout.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    aclnnStatus status =
        CheckGbsaOptionalSeqLength(cuSeqLengthsOptional, batch, ACL_INT64, batch + 1, "INT64", "cuSeqLengths");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaOptionalSeqLength(cuSeqLengthsKvOptional, batch, ACL_INT64, batch + 1, "INT64", "cuSeqLengthsKv");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaOptionalSeqLength(seqUsedQOptional, batch, ACL_INT32, batch, "INT32", "seqUsedQ");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckGbsaOptionalSeqLength(seqUsedKvOptional, batch, ACL_INT32, batch, "INT32", "seqUsedKv");
}

aclnnStatus CheckGbsaQuantType(int64_t quantType, bool isArch35)
{
    if (quantType != GBSA_QUANT_TYPE_NONE && quantType != GBSA_QUANT_TYPE_FLOAT8) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quantType only supports %lld or %lld, but got %lld.", GBSA_QUANT_TYPE_NONE,
                GBSA_QUANT_TYPE_FLOAT8, quantType);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (quantType == GBSA_QUANT_TYPE_FLOAT8 && !isArch35) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quantType %lld is only supported on Atlas A5.", GBSA_QUANT_TYPE_FLOAT8);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaSoftmaxPrecision(int64_t softmaxPrecision, int64_t quantType)
{
    if (softmaxPrecision != GBSA_SOFTMAX_PRECISION_HIGH && softmaxPrecision != GBSA_SOFTMAX_PRECISION_MIXED) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "softmaxPrecision only supports %lld or %lld, but got %lld.",
                GBSA_SOFTMAX_PRECISION_HIGH, GBSA_SOFTMAX_PRECISION_MIXED, softmaxPrecision);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (quantType == GBSA_QUANT_TYPE_FLOAT8 && softmaxPrecision != GBSA_SOFTMAX_PRECISION_MIXED) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FP8 quantization only supports softmaxPrecision=%lld, but got %lld.",
                GBSA_SOFTMAX_PRECISION_MIXED, softmaxPrecision);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaPlatformAttrs(int64_t quantType, int64_t softmaxPrecision, bool isArch35)
{
    aclnnStatus status = CheckGbsaQuantType(quantType, isArch35);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckGbsaSoftmaxPrecision(softmaxPrecision, quantType);
}

aclnnStatus CheckGbsaResidualBlockMode(int64_t residualBlockMode, bool arch35NonQuant)
{
    if (residualBlockMode != GBSA_RESIDUAL_BLOCK_MODE_NONE &&
        !(arch35NonQuant && residualBlockMode == GBSA_RESIDUAL_BLOCK_MODE_FORCE)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "residual_block_mode=%lld is unsupported; mode 1 only supports A5 non-quant currently.",
                residualBlockMode);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaMaskAttrs(int64_t maskType, int64_t windowSizeLeft, int64_t windowSizeRight)
{
    if (maskType != GBSA_CURRENT_MASK_TYPE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "maskType currently only supports %lld, but got %lld.", GBSA_CURRENT_MASK_TYPE,
                maskType);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (windowSizeLeft != GBSA_CURRENT_WINDOW_SIZE || windowSizeRight != GBSA_CURRENT_WINDOW_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "windowSizeLeft and windowSizeRight currently only support %lld, but got %lld and %lld.",
                GBSA_CURRENT_WINDOW_SIZE, windowSizeLeft, windowSizeRight);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaSparseBlockCapacity(const aclTensor *sparseBlockIdx, bool arch35NonQuant)
{
    const int64_t dimNum = sparseBlockIdx->GetViewShape().GetDimNum();
    const int64_t capacity = sparseBlockIdx->GetViewShape().GetDim(dimNum - 1);
    const int64_t maxCapacity = arch35NonQuant ? std::numeric_limits<int32_t>::max() :
                                                GBSA_CURRENT_MAX_SPARSE_BLOCK_COUNT;
    if (capacity > maxCapacity) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sparseBlockIdx last dimension cannot exceed %lld, but got %lld.",
                maxCapacity, capacity);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckGbsaMetadataParams(const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount,
                                    const aclTensor *cuSeqLengthsOptional, const aclTensor *cuSeqLengthsKvOptional,
                                    const aclTensor *seqUsedQOptional, const aclTensor *seqUsedKvOptional,
                                    int64_t maxQSeqLen, int64_t maxKvSeqLen, int64_t numQHeads, int64_t numKvHeads,
                                    int64_t headDim, int64_t blockShapeX, int64_t blockShapeY, const char *qInputLayout,
                                    const char *kvInputLayout, int64_t layoutSparsePattern, int64_t maskType,
                                    int64_t quantType, int64_t softmaxPrecision, int64_t windowSizeLeft,
                                    int64_t windowSizeRight, int64_t residualBlockMode, bool isArch35,
                                    const aclTensor *metadata)
{
    aclnnStatus status = CheckGbsaRequiredInputs(sparseBlockIdx, sparseBlockCount, metadata);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaLayouts(qInputLayout, kvInputLayout);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (GbsaIsLayout(kvInputLayout, "TND") && (!isArch35 || quantType != GBSA_QUANT_TYPE_NONE)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TND non-paged KV requires arch35 and quantType=0.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    status = CheckGbsaMaxSeqLenAttrs(maxQSeqLen, maxKvSeqLen);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaScalarAttrs(numQHeads, numKvHeads, headDim, blockShapeX, blockShapeY, layoutSparsePattern,
                                 isArch35 && quantType == GBSA_QUANT_TYPE_NONE);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaMetadataShape(metadata);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    int64_t batch = 0;
    status = GetGbsaBatch(sparseBlockIdx, cuSeqLengthsOptional, qInputLayout, batch);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaSparseTensorShapes(sparseBlockIdx, sparseBlockCount, maxQSeqLen, numQHeads, numKvHeads,
                                         blockShapeX, layoutSparsePattern, qInputLayout, batch);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaSparseBlockCapacity(sparseBlockIdx, isArch35 && quantType == GBSA_QUANT_TYPE_NONE);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaSeqLengthInputs(cuSeqLengthsOptional, cuSeqLengthsKvOptional, seqUsedQOptional, seqUsedKvOptional,
                                      batch, qInputLayout, kvInputLayout);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaPlatformAttrs(quantType, softmaxPrecision, isArch35);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckGbsaResidualBlockMode(residualBlockMode,
                                        isArch35 && quantType == GBSA_QUANT_TYPE_NONE);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckGbsaMaskAttrs(maskType, windowSizeLeft, windowSizeRight);
}

} // namespace

#endif // GENERIC_BLOCK_SPARSE_ATTENTION_METADATA_CHECK_H
