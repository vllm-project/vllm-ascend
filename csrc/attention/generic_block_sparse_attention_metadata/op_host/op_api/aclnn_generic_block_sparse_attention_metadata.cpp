/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_generic_block_sparse_attention_metadata.h"
#include "generic_block_sparse_attention_metadata.h"

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "../generic_block_sparse_attention_metadata_check.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {

aclnnStatus ParseBlockShape(const aclIntArray *blockShape, int64_t &blockShapeX, int64_t &blockShapeY)
{
    if (blockShape == nullptr || blockShape->Size() != 2U || blockShape->GetData() == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShape must contain exactly two elements [x, y].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    blockShapeX = blockShape->GetData()[0];
    blockShapeY = blockShape->GetData()[1];
    return ACLNN_SUCCESS;
}

const aclTensor *MakeContiguousOptional(const aclTensor *tensor, aclOpExecutor *executor)
{
    return tensor == nullptr ? nullptr : l0op::Contiguous(tensor, executor);
}

} // namespace

aclnnStatus aclnnGenericBlockSparseAttentionMetadataGetWorkspaceSize(
    const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount, const aclTensor *cuSeqLengthsQOptional,
    const aclTensor *cuSeqLengthsKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
    int64_t maxQSeqLen, int64_t maxKvSeqLen, int64_t numQHeads, int64_t numKvHeads, int64_t headDim,
    const aclIntArray *blockShape, const char *layoutQ, const char *layoutKv, int64_t layoutSparsePattern,
    int64_t maskType, int64_t quantType, int64_t softmaxPrecision, int64_t winLeft, int64_t winRight,
    int64_t residualBlockMode, bool isConsistentTopk, const aclTensor *metadataOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    if (workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "workspaceSize/executor must not be null.");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    int64_t blockShapeX = 0;
    int64_t blockShapeY = 0;
    aclnnStatus status = ParseBlockShape(blockShape, blockShapeX, blockShapeY);
    CHECK_RET(status == ACLNN_SUCCESS, status);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    const op::PlatformInfo &platformInfo = op::GetCurrentPlatformInfo();
    const uint32_t aicCoreNum = platformInfo.GetCubeCoreNum();
    const bool isArch35 = platformInfo.GetCurNpuArch() == NpuArch::DAV_3510;
    status = CheckGbsaMetadataParams(sparseBlockIdx, sparseBlockCount, cuSeqLengthsQOptional, cuSeqLengthsKvOptional,
                                     sequsedQOptional, sequsedKvOptional, maxQSeqLen, maxKvSeqLen, numQHeads,
                                     numKvHeads, headDim, blockShapeX, blockShapeY, layoutQ, layoutKv,
                                     layoutSparsePattern, maskType, quantType, softmaxPrecision, winLeft, winRight,
                                     residualBlockMode, isArch35, metadataOptional);
    CHECK_RET(status == ACLNN_SUCCESS, status);

    // Map the layout pattern to the AICPU-side packed-GQA flag (1: packed by KV head, 0: unpacked).
    const int64_t isPackedGqaFlag = layoutSparsePattern == GBSA_LAYOUT_SPARSE_KVN_TOTALQB_KB ? 1 : 0;
    // is_consistent_topk crosses the AICPU attr boundary as an int64 flag (0/1).
    const int64_t isConsistentTopkFlag = isConsistentTopk ? 1 : 0;

    const aclTensor *idxContiguous = l0op::Contiguous(sparseBlockIdx, uniqueExecutor.get());
    CHECK_RET(idxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *countContiguous = l0op::Contiguous(sparseBlockCount, uniqueExecutor.get());
    CHECK_RET(countContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *cuQContiguous = MakeContiguousOptional(cuSeqLengthsQOptional, uniqueExecutor.get());
    CHECK_RET(cuSeqLengthsQOptional == nullptr || cuQContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *seqUsedQContiguous = MakeContiguousOptional(sequsedQOptional, uniqueExecutor.get());
    CHECK_RET(sequsedQOptional == nullptr || seqUsedQContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *cuKvContiguous = MakeContiguousOptional(cuSeqLengthsKvOptional, uniqueExecutor.get());
    CHECK_RET(cuSeqLengthsKvOptional == nullptr || cuKvContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *seqUsedKvContiguous = MakeContiguousOptional(sequsedKvOptional, uniqueExecutor.get());
    CHECK_RET(sequsedKvOptional == nullptr || seqUsedKvContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *output = l0op::GenericBlockSparseAttentionMetadata(
        idxContiguous, countContiguous, cuQContiguous, cuKvContiguous, seqUsedQContiguous, seqUsedKvContiguous,
        maxQSeqLen, numQHeads, numKvHeads, headDim, blockShapeX, blockShapeY, isPackedGqaFlag, layoutQ, aicCoreNum,
        residualBlockMode, isConsistentTopkFlag, metadataOptional, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGenericBlockSparseAttentionMetadata(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                     aclrtStream stream)
{
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
