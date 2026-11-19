// SPDX-License-Identifier: Apache-2.0
#include "aclnn_attn_res_fwd.h"
#include <limits>
#include "../../attn_res_fwd_validation.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
using namespace op;
namespace {
bool HasContiguousView(const aclTensor *tensor)
{
    const auto &shape = tensor->GetViewShape();
    const auto &strides = tensor->GetViewStrides();
    uint64_t expected = 1;
    for (size_t dim = shape.GetDimNum(); dim > 0; --dim) {
        const int64_t size = shape.GetDim(dim - 1);
        if (size < 0) return false;
        if (size == 0) return true;
        if (size > 1 && (strides[dim - 1] < 0 || static_cast<uint64_t>(strides[dim - 1]) != expected)) {
            return false;
        }
        if (expected > std::numeric_limits<int64_t>::max() / static_cast<uint64_t>(size)) return false;
        expected *= static_cast<uint64_t>(size);
    }
    return true;
}
} // namespace
namespace l0op {
OP_TYPE_REGISTER(AttnResFwd);

aclnnStatus AttnResFwd(
    const aclTensor *prefix, const aclTensor *blocks, const aclTensor *proj,
    const aclTensor *norm, const aclTensor *addend, const aclTensor *outputNorm,
    double eps, int64_t validBlocks, int64_t blockTokenStride, int64_t blockWriteIdx,
    double outputNormEps, bool saveMaterialized, bool mix, bool fuseAdd, bool optimizePrefill,
    aclTensor *output, aclTensor *prefixOut, aclTensor *materialized, aclOpExecutor *executor)
{
    L0_DFX(AttnResFwd, prefix, blocks, proj, norm, addend, outputNorm, eps, validBlocks,
           blockTokenStride, blockWriteIdx, outputNormEps, saveMaterialized, mix,
           fuseAdd, optimizePrefill, output, prefixOut, materialized);
    return ADD_TO_LAUNCHER_LIST_AICORE(AttnResFwd,
        OP_INPUT(prefix, blocks, proj, norm, addend, outputNorm),
        OP_OUTPUT(output, prefixOut, materialized),
        OP_ATTR(static_cast<float>(eps), false, validBlocks, blockTokenStride,
                blockWriteIdx, static_cast<float>(outputNormEps), saveMaterialized,
                mix, fuseAdd, optimizePrefill));
}
} // namespace l0op

extern "C" aclnnStatus aclnnAttnResFwdGetWorkspaceSize(
    const aclTensor *prefix, const aclTensor *blocks, const aclTensor *proj,
    const aclTensor *norm, const aclTensor *addend, const aclTensor *outputNorm,
    double eps, int64_t validBlocks, int64_t blockTokenStride, int64_t blockWriteIdx,
    double outputNormEps, bool saveMaterialized, bool mix, bool fuseAdd, bool optimizePrefill,
    aclTensor *output, aclTensor *prefixOut, aclTensor *materialized,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnAttnResFwd,
        DFX_IN(prefix, blocks, proj, norm, addend, outputNorm, eps, validBlocks,
               blockTokenStride, blockWriteIdx, outputNormEps, saveMaterialized,
               mix, fuseAdd, optimizePrefill),
        DFX_OUT(output, prefixOut, materialized));
    if (!prefix || !blocks || !proj || !norm || !addend || !outputNorm ||
        !output || !prefixOut || !materialized || !workspaceSize || !executor ||
        !AttnResFwd::IsValidNormEpsilon(eps) || !AttnResFwd::IsValidNormEpsilon(outputNormEps, true)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const aclTensor *tensors[] = {prefix, blocks, proj, norm, addend, outputNorm,
                                   output, prefixOut, materialized};
    for (const auto *tensor : tensors) {
        if (tensor->GetDataType() != DataType::DT_BF16) {
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    const auto &prefixShape = prefix->GetViewShape();
    const auto &bankShape = blocks->GetViewShape();
    const auto &projShape = proj->GetViewShape();
    const auto &normShape = norm->GetViewShape();
    if (prefixShape.GetDimNum() != 2 || bankShape.GetDimNum() != 3 ||
        bankShape.GetDim(0) != prefixShape.GetDim(0) || bankShape.GetDim(2) != prefixShape.GetDim(1) ||
        projShape.GetDimNum() != 2 || projShape.GetDim(0) != 1 || projShape.GetDim(1) != prefixShape.GetDim(1) ||
        normShape.GetDimNum() != 1 || normShape.GetDim(0) != prefixShape.GetDim(1)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto samePrefixShape = [&prefixShape](const aclTensor *tensor) {
        const auto &shape = tensor->GetViewShape();
        return shape.GetDimNum() == 2 && shape.GetDim(0) == prefixShape.GetDim(0) &&
            shape.GetDim(1) == prefixShape.GetDim(1);
    };
    const auto &outputNormShape = outputNorm->GetViewShape();
    if (!samePrefixShape(output) || !samePrefixShape(prefixOut) || !samePrefixShape(materialized) ||
        (fuseAdd && !samePrefixShape(addend)) ||
        (outputNormEps > 0 && (outputNormShape.GetDimNum() != 1 ||
                              outputNormShape.GetDim(0) != prefixShape.GetDim(1)))) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto &bankStrides = blocks->GetViewStrides();
    if (!AttnResFwd::IsValidBankExtent(prefixShape.GetDim(0), bankShape.GetDim(1), prefixShape.GetDim(1),
                                      validBlocks, blockTokenStride, blockWriteIdx) ||
        bankStrides[2] != 1 || bankStrides[1] != prefixShape.GetDim(1) || bankStrides[0] != blockTokenStride ||
        !HasContiguousView(prefix) || !HasContiguousView(proj) || !HasContiguousView(norm) ||
        !HasContiguousView(output) || !HasContiguousView(prefixOut) || !HasContiguousView(materialized) ||
        (fuseAdd && !HasContiguousView(addend)) || (outputNormEps > 0 && !HasContiguousView(outputNorm))) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    // Match the existing AttnRes ACLNN entry: external tensors carry their
    // logical dimensions in the view shape. This is host metadata only.
    for (const auto *tensor : tensors) {
        tensor->SetOriginalShape(tensor->GetViewShape());
    }
    auto exec = CREATE_EXECUTOR();
    if (!exec.get()) return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    if (output->IsEmpty()) {
        *workspaceSize = 0;
        exec.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // The torch adapter validates contiguous rows. Pass the original bank to
    // the kernel; packing its valid prefix would add a launch after every RS.
    auto ret = l0op::AttnResFwd(prefix, blocks, proj, norm, addend, outputNorm, eps,
        validBlocks, blockTokenStride, blockWriteIdx, outputNormEps, saveMaterialized,
        mix, fuseAdd, optimizePrefill, output, prefixOut, materialized, exec.get());
    if (ret != ACLNN_SUCCESS) return ret;
    *workspaceSize = exec->GetWorkspaceSize();
    exec.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
extern "C" aclnnStatus aclnnAttnResFwd(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAttnResFwd);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
