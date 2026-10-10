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
#include "opdev/platform.h"
#include <algorithm>
#include <string>
#include <unordered_map>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static constexpr int64_t LSE_OUT = 1;

static bool CheckDataType(const aclTensor *query, const aclTensor *key, const aclTensor *value)
{
    const DataType qDtype = query->GetDataType();
    const DataType kDtype = key->GetDataType();
    const DataType vDtype = value->GetDataType();

    static const std::unordered_map<DataType, std::vector<DataType>> validKvType = {
        {DataType::DT_FLOAT16, {DataType::DT_FLOAT16}},
        {DataType::DT_BF16, {DataType::DT_BF16}},
        {DataType::DT_FLOAT8_E4M3FN, {DataType::DT_FLOAT8_E4M3FN}},
    };

    auto iter = validKvType.find(qDtype);
    if (iter == validKvType.end()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported query datatype %d.", static_cast<int>(qDtype));
        return false;
    }

    if (std::find(iter->second.begin(), iter->second.end(), kDtype) == iter->second.end() ||
        std::find(iter->second.begin(), iter->second.end(), vDtype) == iter->second.end()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Key/Value datatype mismatch with query.");
        return false;
    }

    return true;
}

static aclnnStatus CheckMandatoryTensors(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                         const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount)
{
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(sparseBlockIdx != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(sparseBlockCount != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParseBlockShape(const aclIntArray *blockShape)
{
    if (blockShape == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "blockShape must not be null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    if (blockShape->Size() != 2U) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShape must contain two elements [x, y].");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const int64_t *data = blockShape->GetData();
    if (data == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShape data is null.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (data[0] <= 0 || data[1] <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShape values must be positive, got [%ld, %ld].", data[0], data[1]);
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus ValidateParams(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                  const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount,
                                  const aclIntArray *blockShape, char *layoutQ, char *layoutKv)
{
    CHECK_RET(CheckMandatoryTensors(query, key, value, sparseBlockIdx, sparseBlockCount) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);

    if (!CheckDataType(query, key, value)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (layoutQ == nullptr || layoutKv == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "layoutQ/layoutKv must not be null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }

    const std::string layoutQStr = op::ToString(layoutQ).GetString();
    const std::string layoutKvStr = op::ToString(layoutKv).GetString();
    if (layoutQStr != "TND") {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layoutQ only supports TND, got %s.", layoutQStr.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (layoutKvStr != "PA_BBND" && layoutKvStr != "TND") {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layoutKv only supports PA_BBND or TND, got %s.", layoutKvStr.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    aclnnStatus ret = ParseBlockShape(blockShape);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus ValidateAdditionalParams(int64_t softmaxPrecision, const aclTensor *attentionOut,
                                            uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (softmaxPrecision != 0 && softmaxPrecision != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "softmaxPrecision must be 0 or 1, got %ld.", softmaxPrecision);
        return ACLNN_ERR_PARAM_INVALID;
    }

    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

// layoutQ / layoutKv are string attrs; keep as string end-to-end.
static std::string ConvertLayoutString(char *layoutStr)
{
    return op::ToString(layoutStr).GetString();
}

// Contiguous non-KV inputs for aclnn L0 InferShape (torch_npu often has view=ND, storage=1D;
// GetInputShape may return storage unless Contiguous/CreateView normalizes the descriptor).
// Keep Contiguous even when OpDef has AutoContiguous.
// Do NOT Contiguous key/value here — dim0-strided KV is handled by NormalizeDim0KvTensor
// (CreateView + SetStorageShape(viewShape)).
static aclnnStatus MakeContiguous(const aclTensor *&query, const aclTensor *&sparseBlockIdx,
                                  const aclTensor *&sparseBlockCount, const aclTensor *&metadataOptional,
                                  const aclTensor *&attenMaskOptional, const aclTensor *&qDequantScaleOptional,
                                  const aclTensor *&kDequantScaleOptional, const aclTensor *&vDequantScaleOptional,
                                  const aclTensor *&pQuantScaleOptional, const aclTensor *&cuSeqLengthsQOptional,
                                  const aclTensor *&cuSeqLengthsKvOptional, const aclTensor *&sequsedQOptional,
                                  const aclTensor *&sequsedKvOptional, const aclTensor *&blockTableOptional,
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

// Only dim0 may be non-contiguous for PA_BBND KV.
static bool IsFirstAxisOnlyNonContiguous(const aclTensor *tensor, const char *name)
{
    if (tensor == nullptr || IsContiguous(tensor)) {
        return true;
    }

    const auto &viewShape = tensor->GetViewShape();
    const int64_t dimNum = static_cast<int64_t>(viewShape.GetDimNum());
    if (dimNum <= 1) {
        OP_LOGW("%s is non-contiguous with dim num[%ld] <= 1, it will be forced contiguous.", name, dimNum);
        return false;
    }

    const auto &viewStrides = tensor->GetViewStrides();
    if (viewStrides.size() < static_cast<size_t>(dimNum)) {
        OP_LOGW("%s view strides num[%zu] < dim num[%ld], it will be forced contiguous.", name, viewStrides.size(),
                dimNum);
        return false;
    }

    int64_t expectedStride = 1;
    for (int64_t dim = dimNum - 1; dim >= 1; --dim) {
        if (viewShape.GetDim(dim) != 1 && viewStrides[static_cast<size_t>(dim)] != expectedStride) {
            OP_LOGW("%s is non-contiguous at axis[%ld]: actual stride[%ld], expected[%ld], shape[%s]. "
                    "It will be forced contiguous.",
                    name, dim, viewStrides[static_cast<size_t>(dim)], expectedStride,
                    op::ToString(viewShape).GetString());
            return false;
        }
        expectedStride *= viewShape.GetDim(dim);
    }
    return true;
}

// CreateView + SetStorageShape(viewShape): InferShape/GetInputShape often reads storage;
// forcing storage==view keeps ND dims for tiling while preserving view strides (zero-copy).
static const aclTensor *CreateStrideAwareView(const aclTensor *tensor, const char *name, aclOpExecutor *executor)
{
    auto strideView = executor->CreateView(tensor, tensor->GetViewShape(), tensor->GetStorageShape(),
                                           tensor->GetViewStrides(), tensor->GetViewOffset());
    if (strideView == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "CreateStrideAwareView for %s failed.", name);
        return nullptr;
    }
    const_cast<aclTensor *>(strideView)->SetStorageShape(tensor->GetViewShape());
    return strideView;
}

// Contiguous KV: Contiguous (normalize descriptor). Dim0-only holes: stride-aware view.
// Other non-contig: Contiguous fallback (tiling rejects non-dim0 holes for PA_BBND).
static const aclTensor *NormalizeDim0KvTensor(const aclTensor *tensor, const char *name, aclOpExecutor *executor)
{
    if (tensor == nullptr) {
        return nullptr;
    }
    if (IsContiguous(tensor)) {
        return l0op::Contiguous(tensor, executor);
    }
    if (IsFirstAxisOnlyNonContiguous(tensor, name)) {
        return CreateStrideAwareView(tensor, name, executor);
    }
    return l0op::Contiguous(tensor, executor);
}

} // namespace

__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *sparseBlockIdx,
    const aclTensor *sparseBlockCount, const aclTensor *metadataOptional, const aclTensor *attenMaskOptional,
    const aclTensor *qDequantScaleOptional, const aclTensor *kDequantScaleOptional,
    const aclTensor *vDequantScaleOptional, const aclTensor *pQuantScaleOptional,
    const aclTensor *cuSeqLengthsQOptional, const aclTensor *cuSeqLengthsKvOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional, const aclTensor *blockTableOptional, const aclIntArray *blockShape,
    char *layoutQ, char *layoutKv, int64_t layoutSparsePattern, double scaleValue, int64_t maskType, int64_t quantType,
    double dstTypeMax, int64_t softmaxPrecision, int64_t winLeft, int64_t winRight, int64_t returnSoftmaxlse,
    int64_t residualBlockMode, bool isConsistentTopk, aclTensor *attentionOut, aclTensor *softmaxLseOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    aclnnStatus ret =
        ValidateParams(query, key, value, sparseBlockIdx, sparseBlockCount, blockShape, layoutQ, layoutKv);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    // Storage mode comes from blockTable presence; layout only constrains the tensor axes.
    const bool isPaged = blockTableOptional != nullptr;
    if (isPaged) {
        if (std::string(layoutKv) != "PA_BBND") {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "With blockTable, the supported KV layout is PA_BBND.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (std::string(layoutKv) == "PA_BBND") {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "PA_BBND requires blockTable; it must not be nullptr.");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510 || quantType != 0 ||
            (query->GetDataType() != DataType::DT_FLOAT16 && query->GetDataType() != DataType::DT_BF16)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TND non-paged KV requires A5, FP16/BF16 and quantType=0.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (cuSeqLengthsQOptional == nullptr || cuSeqLengthsKvOptional == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "TND non-paged KV requires Q/KV cuSeqLengths and no blockTable.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    ret = ValidateAdditionalParams(softmaxPrecision, attentionOut, workspaceSize, executor);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    L2_DFX_PHASE_1(
        aclnnGenericBlockSparseAttention,
        DFX_IN(query, key, value, sparseBlockIdx, sparseBlockCount, metadataOptional, attenMaskOptional,
               qDequantScaleOptional, kDequantScaleOptional, vDequantScaleOptional, pQuantScaleOptional,
               cuSeqLengthsQOptional, cuSeqLengthsKvOptional, sequsedQOptional, sequsedKvOptional, blockTableOptional,
               blockShape, layoutQ, layoutKv, layoutSparsePattern, scaleValue, maskType, quantType, dstTypeMax,
               softmaxPrecision, winLeft, winRight, returnSoftmaxlse, residualBlockMode, isConsistentTopk),
        DFX_OUT(attentionOut, softmaxLseOptional));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto *executorImpl = uniqueExecutor.get();

    ret = MakeContiguous(query, sparseBlockIdx, sparseBlockCount, metadataOptional, attenMaskOptional,
                         qDequantScaleOptional, kDequantScaleOptional, vDequantScaleOptional, pQuantScaleOptional,
                         cuSeqLengthsQOptional, cuSeqLengthsKvOptional, sequsedQOptional, sequsedKvOptional,
                         blockTableOptional, executorImpl);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    std::string layoutQStr = ConvertLayoutString(layoutQ);
    std::string layoutKvStr = ConvertLayoutString(layoutKv);

    const aclTensor *keyFinal = isPaged ? NormalizeDim0KvTensor(key, "key", executorImpl) :
                                         l0op::Contiguous(key, executorImpl);
    CHECK_RET(keyFinal != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *valueFinal = isPaged ? NormalizeDim0KvTensor(value, "value", executorImpl) :
                                           l0op::Contiguous(value, executorImpl);
    CHECK_RET(valueFinal != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto outputs = l0op::GenericBlockSparseAttention(
        query, keyFinal, valueFinal, sparseBlockIdx, sparseBlockCount, metadataOptional, attenMaskOptional,
        qDequantScaleOptional, kDequantScaleOptional, vDequantScaleOptional, pQuantScaleOptional, cuSeqLengthsQOptional,
        cuSeqLengthsKvOptional, sequsedQOptional, sequsedKvOptional, blockTableOptional, blockShape, layoutQStr.c_str(),
        layoutKvStr.c_str(), layoutSparsePattern, scaleValue, maskType, quantType, dstTypeMax, softmaxPrecision,
        winLeft, winRight, returnSoftmaxlse, residualBlockMode, isConsistentTopk, attentionOut, executorImpl);

    if (outputs[0] == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GenericBlockSparseAttention returned nullptr output.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyResult = l0op::ViewCopy(outputs[0], attentionOut, executorImpl);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (returnSoftmaxlse == LSE_OUT) {
        CHECK_RET(outputs[1] != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewCopyLseResult = l0op::ViewCopy(outputs[1], softmaxLseOptional, executorImpl);
        CHECK_RET(viewCopyLseResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = executorImpl->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnGenericBlockSparseAttention(void *workspace,
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
