/**
 * Copyright (c) 2026 SeaGaruda
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "aclnn_top_k_per_row.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {
constexpr int64_t K_TOPK_MAX = 2048;

bool IsSupportedDtype(const aclTensor* tensor)
{
    const auto dtype = tensor->GetDataType();
    return dtype == DataType::DT_FLOAT || dtype == DataType::DT_FLOAT16 || dtype == DataType::DT_BF16;
}
}  // namespace

namespace l0op {
OP_TYPE_REGISTER(TopKPerRow);

aclnnStatus TopKPerRow(const aclTensor* logits, const aclTensor* rowStarts, const aclTensor* rowEnds,
                       int64_t topk, aclTensor* indices, aclOpExecutor* executor)
{
    L0_DFX(TopKPerRow, logits, rowStarts, rowEnds, topk, indices);
    return ADD_TO_LAUNCHER_LIST_AICORE(TopKPerRow,
                                       OP_INPUT(logits, rowStarts, rowEnds),
                                       OP_OUTPUT(indices),
                                       OP_ATTR(topk));
}
}  // namespace l0op

extern "C" aclnnStatus aclnnTopKPerRowGetWorkspaceSize(
    const aclTensor* logits, const aclTensor* rowStarts, const aclTensor* rowEnds,
    int64_t topk, aclTensor* indices, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnTopKPerRow, DFX_IN(logits, rowStarts, rowEnds, topk), DFX_OUT(indices));

    if (logits == nullptr || rowStarts == nullptr || rowEnds == nullptr || indices == nullptr ||
        workspaceSize == nullptr || executor == nullptr) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (topk <= 0 || topk > K_TOPK_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "topk must be in (0, 2048], got %ld", topk);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!IsSupportedDtype(logits)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "logits must be float32/float16/bfloat16");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (rowStarts->GetDataType() != DataType::DT_INT32 || rowEnds->GetDataType() != DataType::DT_INT32 ||
        indices->GetDataType() != DataType::DT_INT32) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "row_starts/row_ends/indices must be int32");
        return ACLNN_ERR_PARAM_INVALID;
    }

    const auto& logitsShape = logits->GetViewShape();
    const auto& startsShape = rowStarts->GetViewShape();
    const auto& endsShape = rowEnds->GetViewShape();
    const auto& outShape = indices->GetViewShape();
    if (logitsShape.GetDimNum() != 2 || startsShape.GetDimNum() != 1 || endsShape.GetDimNum() != 1 ||
        outShape.GetDimNum() != 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "unexpected ranks: logits/indices must be 2-D, rows 1-D");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t numRows = logitsShape.GetDim(0);
    if (startsShape.GetDim(0) != numRows || endsShape.GetDim(0) != numRows ||
        outShape.GetDim(0) != numRows || outShape.GetDim(1) != topk) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "shape mismatch: expect row_starts/row_ends of %ld and indices (%ld, %ld)",
                numRows, numRows, topk);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // The kernel reads one window per row; the row layout itself is described
    // by the view shape (strides are irrelevant, we index by row * width).
    // SetOriginalShape is a non-const accessor, so the input views are cast here
    // rather than making the whole launcher take non-const tensors.
    aclTensor* tensors[] = {const_cast<aclTensor*>(logits), const_cast<aclTensor*>(rowStarts),
                            const_cast<aclTensor*>(rowEnds), indices};
    for (auto* tensor : tensors) {
        tensor->SetOriginalShape(tensor->GetViewShape());
    }

    auto exec = CREATE_EXECUTOR();
    if (!exec.get()) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    if (indices->IsEmpty()) {
        *workspaceSize = 0;
        exec.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const auto ret = l0op::TopKPerRow(logits, rowStarts, rowEnds, topk, indices, exec.get());
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    *workspaceSize = exec->GetWorkspaceSize();
    exec.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnTopKPerRow(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnTopKPerRow);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
