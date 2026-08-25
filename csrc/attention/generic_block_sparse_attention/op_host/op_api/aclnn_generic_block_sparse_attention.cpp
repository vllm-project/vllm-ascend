/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_generic_block_sparse_attention.h"

#include "generic_block_sparse_attention.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/common_types.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include <acl/acl.h>
#include <algorithm>
#include <string>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

// Keep in sync with sparse_attention_score_metadata.h METADATA_TOTAL_SIZE.
constexpr int64_t GSA_METADATA_TOTAL_SIZE = 1024;

// Design doc: layoutQ/layoutKv are String attrs (unlike SAS int). Keep as string end-to-end.
static std::string ConvertLayoutString(char *layoutStr)
{
    return op::ToString(layoutStr).GetString();
}

// Align SMLA / FlashAttn consumer checks: required INT32 1D shell of fixed size.
static aclnnStatus CheckMetadata(const aclTensor *metadata)
{
    if (metadata == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "metadata must be provided.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    aclDataType dataType = ACL_DT_UNDEFINED;
    aclGetDataType(metadata, &dataType);
    if (dataType != ACL_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "metadata must be INT32.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (metadata->GetViewShape().GetDimNum() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "metadata dimension must be 1.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (metadata->GetViewShape().GetDim(0) != GSA_METADATA_TOTAL_SIZE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "metadata shape must be [%ld].", GSA_METADATA_TOTAL_SIZE);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// Contiguous most inputs. Do not touch key/value here so PAGED_BBND dim0-strided
// views (page-axis holes) are preserved; page interior must still be contiguous (tiling checks).
static aclnnStatus MakeContiguous(const aclTensor *&query,
                                  const aclTensor *&sparseBlockIdx,
                                  const aclTensor *&sparseBlockCount,
                                  const aclTensor *&metadataOptional,
                                  const aclTensor *&attenMaskOptional,
                                  const aclTensor *&qDequantScaleOptional,
                                  const aclTensor *&kDequantScaleOptional,
                                  const aclTensor *&vDequantScaleOptional,
                                  const aclTensor *&pQuantScaleOptional,
                                  const aclTensor *&cuSeqLengthsQOptional,
                                  const aclTensor *&cuSeqLengthsKvOptional,
                                  const aclTensor *&sequsedQOptional,
                                  const aclTensor *&sequsedKvOptional,
                                  const aclTensor *&blockTableOptional,
                                  aclOpExecutor *executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    sparseBlockIdx = l0op::Contiguous(sparseBlockIdx, executor);
    CHECK_RET(sparseBlockIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    sparseBlockCount = l0op::Contiguous(sparseBlockCount, executor);
    CHECK_RET(sparseBlockCount != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    metadataOptional = l0op::Contiguous(metadataOptional, executor);
    CHECK_RET(metadataOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (attenMaskOptional != nullptr) {
        attenMaskOptional = l0op::Contiguous(attenMaskOptional, executor);
        CHECK_RET(attenMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (qDequantScaleOptional != nullptr) {
        qDequantScaleOptional = l0op::Contiguous(qDequantScaleOptional, executor);
        CHECK_RET(qDequantScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (kDequantScaleOptional != nullptr) {
        kDequantScaleOptional = l0op::Contiguous(kDequantScaleOptional, executor);
        CHECK_RET(kDequantScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (vDequantScaleOptional != nullptr) {
        vDequantScaleOptional = l0op::Contiguous(vDequantScaleOptional, executor);
        CHECK_RET(vDequantScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (pQuantScaleOptional != nullptr) {
        pQuantScaleOptional = l0op::Contiguous(pQuantScaleOptional, executor);
        CHECK_RET(pQuantScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (cuSeqLengthsQOptional != nullptr) {
        cuSeqLengthsQOptional = l0op::Contiguous(cuSeqLengthsQOptional, executor);
        CHECK_RET(cuSeqLengthsQOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (cuSeqLengthsKvOptional != nullptr) {
        cuSeqLengthsKvOptional = l0op::Contiguous(cuSeqLengthsKvOptional, executor);
        CHECK_RET(cuSeqLengthsKvOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (sequsedQOptional != nullptr) {
        sequsedQOptional = l0op::Contiguous(sequsedQOptional, executor);
        CHECK_RET(sequsedQOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (sequsedKvOptional != nullptr) {
        sequsedKvOptional = l0op::Contiguous(sequsedKvOptional, executor);
        CHECK_RET(sequsedKvOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (blockTableOptional != nullptr) {
        blockTableOptional = l0op::Contiguous(blockTableOptional, executor);
        CHECK_RET(blockTableOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

// If ATK/aclTensor reports storage==view while dim0 stride has holes, GE will
// rewrite storage to a degenerate shape (e.g. dim0=stride0, dim1=0). Expand dim0
// so storage covers the last viewed element and keeps BBND inner dims.
static op::Shape MakeKvStorageShape(const op::Shape &viewShape, const op::Shape &storageShape,
                                    int64_t viewOffset, const int64_t *strides, size_t strideN)
{
    const size_t dimNum = viewShape.GetDimNum();
    op::Shape out = storageShape;
    if (dimNum == 0 || strides == nullptr || strideN < dimNum) {
        return out;
    }

    int64_t minElems = viewOffset;
    int64_t inner = 1;
    for (size_t i = 0; i < dimNum; ++i) {
        const int64_t dim = viewShape.GetDim(i);
        if (dim > 0) {
            minElems += (dim - 1) * strides[i];
        }
        if (i >= 1) {
            inner *= std::max(dim, static_cast<int64_t>(1));
        }
    }
    minElems += 1;
    if (inner <= 0) {
        return out;
    }

    int64_t storageElems = 1;
    bool storageOk = storageShape.GetDimNum() == dimNum;
    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        const int64_t d = storageShape.GetDim(i);
        if (d <= 0) {
            storageOk = false;
            break;
        }
        storageElems *= d;
    }
    if (storageOk && storageElems >= minElems) {
        return out;
    }

    out = viewShape;
    const int64_t storage0 = std::max((minElems + inner - 1) / inner, viewShape.GetDim(0));
    out.SetDim(0, storage0);
    return out;
}

// Expose KV view shape/stride to GE tiling (TensorV2) for PAGED_BBND dim0-strided caches.
static const aclTensor *CreateKvView(const aclTensor *tensor, aclOpExecutor *executor)
{
    const auto &viewShape = tensor->GetViewShape();
    const auto &storageShape = tensor->GetStorageShape();
    const auto &viewStrides = tensor->GetViewStrides();
    const int64_t viewOffset = tensor->GetViewOffset();
    const size_t strideN = viewStrides.size();
    const int64_t *stridePtr = (strideN > 0) ? &viewStrides[0] : nullptr;
    const op::Shape storageForView =
        MakeKvStorageShape(viewShape, storageShape, viewOffset, stridePtr, strideN);
    return executor->CreateView(tensor, viewShape, storageForView, viewStrides, viewOffset);
}

}  // namespace

__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseBlockIdx,
    const aclTensor *sparseBlockCount,
    const aclTensor *metadataOptional,
    const aclTensor *attenMaskOptional,
    const aclTensor *qDequantScaleOptional,
    const aclTensor *kDequantScaleOptional,
    const aclTensor *vDequantScaleOptional,
    const aclTensor *pQuantScaleOptional,
    const aclTensor *cuSeqLengthsQOptional,
    const aclTensor *cuSeqLengthsKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional,
    const aclTensor *blockTableOptional,
    const aclIntArray *blockShape,
    int64_t isPackedGQA,
    char *layoutQ,
    char *layoutKv,
    double scaleValue,
    int64_t maskType,
    int64_t quantType,
    double dstTypeMax,
    int64_t softmaxPrecision,
    int64_t winLeft,
    int64_t winRight,
    int64_t returnSoftmaxlse,
    aclTensor *attentionOut,
    aclTensor *softmaxLseOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(sparseBlockIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(sparseBlockCount != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // Align kernel / AICPU packed path: only isPackedGQA=1 (task = T * Nkv).
    if (isPackedGQA != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Unsupported isPackedGQA=%ld, only 1 (packed GQA) is supported.", isPackedGQA);
        return ACLNN_ERR_PARAM_INVALID;
    }

    aclnnStatus metaStatus = CheckMetadata(metadataOptional);
    if (metaStatus != ACLNN_SUCCESS) {
        return metaStatus;
    }

    L2_DFX_PHASE_1(aclnnGenericBlockSparseAttention,
                   DFX_IN(query, key, value, sparseBlockIdx, sparseBlockCount,
                          metadataOptional, attenMaskOptional, qDequantScaleOptional,
                          kDequantScaleOptional, vDequantScaleOptional, pQuantScaleOptional,
                          cuSeqLengthsQOptional, cuSeqLengthsKvOptional,
                          sequsedQOptional, sequsedKvOptional, blockTableOptional,
                          blockShape, isPackedGQA, layoutQ, layoutKv, scaleValue,
                          maskType, quantType, dstTypeMax, softmaxPrecision,
                          winLeft, winRight, returnSoftmaxlse),
                   DFX_OUT(attentionOut, softmaxLseOptional));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto *executorImpl = uniqueExecutor.get();

    aclnnStatus ret = MakeContiguous(query, sparseBlockIdx, sparseBlockCount,
                                     metadataOptional, attenMaskOptional,
                                     qDequantScaleOptional, kDequantScaleOptional, vDequantScaleOptional,
                                     pQuantScaleOptional, cuSeqLengthsQOptional, cuSeqLengthsKvOptional,
                                     sequsedQOptional, sequsedKvOptional,
                                     blockTableOptional, executorImpl);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    std::string layoutQStr = ConvertLayoutString(layoutQ);
    std::string layoutKvStr = ConvertLayoutString(layoutKv);

    const aclTensor *keyFinal = CreateKvView(key, executorImpl);
    CHECK_RET(keyFinal != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *valueFinal = CreateKvView(value, executorImpl);
    CHECK_RET(valueFinal != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outputs = l0op::GenericBlockSparseAttention(
        query, keyFinal, valueFinal, sparseBlockIdx, sparseBlockCount,
        metadataOptional, attenMaskOptional,
        qDequantScaleOptional, kDequantScaleOptional, vDequantScaleOptional,
        pQuantScaleOptional, cuSeqLengthsQOptional, cuSeqLengthsKvOptional,
        sequsedQOptional, sequsedKvOptional, blockTableOptional,
        blockShape, isPackedGQA, layoutQStr.c_str(), layoutKvStr.c_str(), scaleValue,
        maskType, quantType, dstTypeMax, softmaxPrecision,
        winLeft, winRight, returnSoftmaxlse,
        attentionOut, executorImpl);

    if (outputs[0] == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GenericBlockSparseAttention returned nullptr output.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyResult = l0op::ViewCopy(outputs[0], attentionOut, executorImpl);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    static constexpr int64_t LSE_OUT = 1;
    if (returnSoftmaxlse == LSE_OUT) {
        CHECK_RET(softmaxLseOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        CHECK_RET(outputs[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyLseResult = l0op::ViewCopy(outputs[1], softmaxLseOptional, executorImpl);
        CHECK_RET(viewCopyLseResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = executorImpl->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGenericBlockSparseAttention);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
