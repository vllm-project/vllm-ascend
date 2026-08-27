/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "aclnn_generic_block_sparse_attention_metadata.h"
#include "sparse_attention_score_metadata.h"

#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "../sparse_attention_score_metadata_check.h"

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
    const aclTensor *sparseBlockIdx, const aclTensor *sparseBlockCount, const aclTensor *cuSeqLengthsOptional,
    const aclTensor *cuSeqLengthsKvOptional, const aclTensor *seqUsedQOptional, const aclTensor *seqUsedKvOptional,
    int64_t maxQSeqLen, int64_t maxKvSeqLen, int64_t numQHeads, int64_t numKvHeads, int64_t headDim,
    const aclIntArray *blockShape, int64_t isPackedGQA, const char *qInputLayout, const char *kvInputLayout,
    int64_t maskType, int64_t quantType, int64_t softmaxPrecision, int64_t windowSizeLeft, int64_t windowSizeRight,
    const aclTensor *metadata, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    (void)softmaxPrecision;
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
    const std::string socVersion = platformInfo.GetSocLongVersion();
    status = CheckSasMetadataParams(sparseBlockIdx, sparseBlockCount, cuSeqLengthsOptional, cuSeqLengthsKvOptional,
                                    seqUsedQOptional, seqUsedKvOptional, maxQSeqLen, maxKvSeqLen, numQHeads, numKvHeads,
                                    headDim, blockShapeX, blockShapeY, isPackedGQA, qInputLayout, kvInputLayout,
                                    maskType, quantType, windowSizeLeft, windowSizeRight, socVersion, metadata);
    CHECK_RET(status == ACLNN_SUCCESS, status);

    const aclTensor *idxContiguous = l0op::Contiguous(sparseBlockIdx, uniqueExecutor.get());
    CHECK_RET(idxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *countContiguous = l0op::Contiguous(sparseBlockCount, uniqueExecutor.get());
    CHECK_RET(countContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *cuQContiguous = MakeContiguousOptional(cuSeqLengthsOptional, uniqueExecutor.get());
    CHECK_RET(cuSeqLengthsOptional == nullptr || cuQContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *seqUsedQContiguous = MakeContiguousOptional(seqUsedQOptional, uniqueExecutor.get());
    CHECK_RET(seqUsedQOptional == nullptr || seqUsedQContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *output = l0op::GenericBlockSparseAttentionMetadata(
        idxContiguous, countContiguous, cuQContiguous, cuSeqLengthsKvOptional, seqUsedQContiguous, seqUsedKvOptional,
        maxQSeqLen, numQHeads, numKvHeads, headDim, blockShapeX, blockShapeY, isPackedGQA, qInputLayout,
        aicCoreNum, metadata, uniqueExecutor.get());
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
