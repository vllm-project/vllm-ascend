/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef SPARSE_ATTENTION_SCORE_METADATA_CHECK_H
#define SPARSE_ATTENTION_SCORE_METADATA_CHECK_H

#include <cstring>
#include <string>

#include "opdev/data_type_utils.h"
#include "opdev/op_log.h"
#include "../sparse_attention_score_metadata.h"

namespace {

bool SasTensorValid(const aclTensor *tensor)
{
    return tensor != nullptr && tensor->GetViewShape().GetDimNum() > 0;
}

bool SasIsLayout(const char *layout, const char *expected)
{
    return layout != nullptr && std::strcmp(layout, expected) == 0;
}

aclnnStatus CheckSasTensorDataType(const aclTensor *tensor, aclDataType expectedDataType, const char *expectedTypeName,
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

aclnnStatus CheckSasTensorDim(const aclTensor *tensor, int64_t expectedDim, const char *tensorName)
{
    if (tensor->GetViewShape().GetDimNum() != expectedDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s dimension must be %ld.", tensorName, expectedDim);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckSasRequiredInt32Tensor(const aclTensor *tensor, const char *tensorName)
{
    if (!SasTensorValid(tensor)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s must be provided.", tensorName);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return CheckSasTensorDataType(tensor, ACL_INT32, "INT32", tensorName);
}

aclnnStatus CheckSasRequiredInputs(const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount,
                                   const aclTensor *metadata)
{
    aclnnStatus status = CheckSasRequiredInt32Tensor(sparseBlockIdx, "sparseBlockIdx");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckSasRequiredInt32Tensor(sparseBlockCount, "SparseBlockCount");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckSasRequiredInt32Tensor(metadata, "metadata");
}

aclnnStatus CheckSasMetadataShape(const aclTensor *metadata)
{
    aclnnStatus status = CheckSasTensorDim(metadata, 1, "metadata");
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

aclnnStatus CheckSasPositiveScalarAttr(int64_t attrValue, const char *attrName)
{
    if (attrValue <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s must be greater than 0, but got %lld.", attrName, attrValue);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckSasScalarAttrs(int64_t maxQSeqLen, int64_t maxKvSeqLen, int64_t numQHeads, int64_t numKvHeads,
                                int64_t headDim, int64_t blockShapeX, int64_t blockShapeY, int64_t isPackedGQA)
{
    const struct {
        int64_t value;
        const char *name;
    } positiveAttrs[] = {{maxQSeqLen, "maxQSeqLen"}, {maxKvSeqLen, "maxKvSeqLen"}, {numQHeads, "numQHeads"},
                         {numKvHeads, "numKvHeads"}, {headDim, "headDim"},         {blockShapeY, "blockShapeY"}};
    for (const auto &attr : positiveAttrs) {
        const aclnnStatus status = CheckSasPositiveScalarAttr(attr.value, attr.name);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    if (blockShapeX != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeX currently only supports 1, but got %lld.", blockShapeX);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (blockShapeY % 16 != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeY must be a multiple of 16, but got %lld.", blockShapeY);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (isPackedGQA != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "isPackedGQA currently only supports 1, but got %lld.", isPackedGQA);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckSasLayouts(const char *qInputLayout, const char *kvInputLayout)
{
    const bool qLayoutValid =
        SasIsLayout(qInputLayout, "TND") || SasIsLayout(qInputLayout, "BNSD") || SasIsLayout(qInputLayout, "BSND");
    const bool kvLayoutValid = SasIsLayout(kvInputLayout, "TND") || SasIsLayout(kvInputLayout, "BNSD") ||
                               SasIsLayout(kvInputLayout, "BSND") || SasIsLayout(kvInputLayout, "PAGED_BBND") ||
                               SasIsLayout(kvInputLayout, "PAGED_BNBD");
    if (!qLayoutValid || !kvLayoutValid) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported qInputLayout or kvInputLayout.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckSasOptionalSeqLength(const aclTensor *tensor, int64_t batch, aclDataType dataType,
                                      int64_t expectedElements, const char *typeName, const char *tensorName)
{
    if (!SasTensorValid(tensor)) {
        return ACLNN_SUCCESS;
    }
    aclnnStatus status = CheckSasTensorDataType(tensor, dataType, typeName, tensorName);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckSasTensorDim(tensor, 1, tensorName);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    if (tensor->GetViewShape().GetDim(0) != expectedElements) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "%s shape does not match batch %lld.", tensorName, batch);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus GetSasBatch(const aclTensor *sparseBlockIdx, const aclTensor *cuSeqLengthsOptional,
                        const char *qInputLayout, int64_t &batch)
{
    if (!SasIsLayout(qInputLayout, "TND")) {
        batch = sparseBlockIdx->GetViewShape().GetDim(0);
        return batch >= 0 ? ACLNN_SUCCESS : ACLNN_ERR_PARAM_INVALID;
    }
    if (!SasTensorValid(cuSeqLengthsOptional)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLengths is required for TND query.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (cuSeqLengthsOptional->GetViewShape().GetDimNum() != 1 || cuSeqLengthsOptional->GetViewShape().GetDim(0) < 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLengths shape must be [batch + 1].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    batch = cuSeqLengthsOptional->GetViewShape().GetDim(0) - 1;
    return ACLNN_SUCCESS;
}

aclnnStatus CheckSasSparseTensorShapes(const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount,
                                       int64_t maxQSeqLen, int64_t numQHeads, int64_t numKvHeads, int64_t blockShapeX,
                                       int64_t isPackedGQA, const char *qInputLayout, int64_t batch)
{
    const int64_t sparseHeads = isPackedGQA == 1 ? numKvHeads : numQHeads;
    if (SasIsLayout(qInputLayout, "TND")) {
        if (CheckSasTensorDim(sparseBlockIdx, 3, "sparseBlockIdx") != ACLNN_SUCCESS ||
            CheckSasTensorDim(sparseBlockCount, 2, "SparseBlockCount") != ACLNN_SUCCESS) {
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (sparseBlockIdx->GetViewShape().GetDim(0) != sparseHeads ||
            sparseBlockCount->GetViewShape().GetDim(0) != sparseHeads ||
            sparseBlockIdx->GetViewShape().GetDim(1) != sparseBlockCount->GetViewShape().GetDim(1) ||
            sparseBlockIdx->GetViewShape().GetDim(2) <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TND sparse block tensor shapes do not match attrs.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    if (CheckSasTensorDim(sparseBlockIdx, 4, "sparseBlockIdx") != ACLNN_SUCCESS ||
        CheckSasTensorDim(sparseBlockCount, 3, "SparseBlockCount") != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t qBlocks = maxQSeqLen / blockShapeX + static_cast<int64_t>(maxQSeqLen % blockShapeX != 0);
    if (sparseBlockIdx->GetViewShape().GetDim(0) != batch || sparseBlockCount->GetViewShape().GetDim(0) != batch ||
        sparseBlockIdx->GetViewShape().GetDim(1) != sparseHeads ||
        sparseBlockCount->GetViewShape().GetDim(1) != sparseHeads ||
        sparseBlockIdx->GetViewShape().GetDim(2) != qBlocks || sparseBlockCount->GetViewShape().GetDim(2) != qBlocks ||
        sparseBlockIdx->GetViewShape().GetDim(3) <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BSND/BNSD sparse block tensor shapes do not match attrs.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckSasSeqLengthInputs(const aclTensor *cuSeqLengthsOptional, const aclTensor *cuSeqLengthsKvOptional,
                                    const aclTensor *seqUsedQOptional, const aclTensor *seqUsedKvOptional,
                                    int64_t batch, const char *qInputLayout, const char *kvInputLayout)
{
    if (SasIsLayout(qInputLayout, "TND") && !SasTensorValid(cuSeqLengthsOptional)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLengths is required for TND query.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const bool kvSeqLengthsRequired = SasIsLayout(kvInputLayout, "TND") || SasIsLayout(kvInputLayout, "PAGED_BBND") ||
                                      SasIsLayout(kvInputLayout, "PAGED_BNBD");
    if (kvSeqLengthsRequired && !SasTensorValid(cuSeqLengthsKvOptional)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cuSeqLengthsKv is required for TND or paged KV.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    aclnnStatus status =
        CheckSasOptionalSeqLength(cuSeqLengthsOptional, batch, ACL_INT64, batch + 1, "INT64", "cuSeqLengths");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckSasOptionalSeqLength(cuSeqLengthsKvOptional, batch, ACL_INT64, batch + 1, "INT64", "cuSeqLengthsKv");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckSasOptionalSeqLength(seqUsedQOptional, batch, ACL_INT32, batch, "INT32", "seqUsedQ");
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckSasOptionalSeqLength(seqUsedKvOptional, batch, ACL_INT32, batch, "INT32", "seqUsedKv");
}

aclnnStatus CheckSasQuantType(int64_t quantType, const std::string &socVersion)
{
    if (quantType != 0 && socVersion.find("Ascend950") == std::string::npos) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quantType is only supported on Atlas A5.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckSasMaskAttrs(int64_t maskType, int64_t windowSizeLeft, int64_t windowSizeRight)
{
    if (maskType < 0 || maskType > 2 || (maskType != 2 && (windowSizeLeft != -1 || windowSizeRight != -1)) ||
        (maskType == 2 && (windowSizeLeft < 0 || windowSizeRight < 0))) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "maskType/windowSize attrs are invalid.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckSasMetadataParams(const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount,
                                   const aclTensor *cuSeqLengthsOptional, const aclTensor *cuSeqLengthsKvOptional,
                                   const aclTensor *seqUsedQOptional, const aclTensor *seqUsedKvOptional,
                                   int64_t maxQSeqLen, int64_t maxKvSeqLen, int64_t numQHeads, int64_t numKvHeads,
                                   int64_t headDim, int64_t blockShapeX, int64_t blockShapeY, int64_t isPackedGQA,
                                   const char *qInputLayout, const char *kvInputLayout, int64_t maskType,
                                   int64_t quantType, int64_t windowSizeLeft, int64_t windowSizeRight,
                                   const std::string &socVersion, const aclTensor *metadata)
{
    aclnnStatus status = CheckSasRequiredInputs(sparseBlockIdx, sparseBlockCount, metadata);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckSasScalarAttrs(maxQSeqLen, maxKvSeqLen, numQHeads, numKvHeads, headDim, blockShapeX, blockShapeY,
                                 isPackedGQA);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckSasLayouts(qInputLayout, kvInputLayout);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckSasMetadataShape(metadata);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    int64_t batch = 0;
    status = GetSasBatch(sparseBlockIdx, cuSeqLengthsOptional, qInputLayout, batch);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckSasSparseTensorShapes(sparseBlockIdx, sparseBlockCount, maxQSeqLen, numQHeads, numKvHeads,
                                        blockShapeX, isPackedGQA, qInputLayout, batch);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckSasSeqLengthInputs(cuSeqLengthsOptional, cuSeqLengthsKvOptional, seqUsedQOptional, seqUsedKvOptional,
                                     batch, qInputLayout, kvInputLayout);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    status = CheckSasQuantType(quantType, socVersion);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    return CheckSasMaskAttrs(maskType, windowSizeLeft, windowSizeRight);
}

} // namespace

#endif // SPARSE_ATTENTION_SCORE_METADATA_CHECK_H
