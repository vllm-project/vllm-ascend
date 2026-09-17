// SPDX-License-Identifier: Apache-2.0
#include "aclnn_attn_res_fwd_fused.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
using namespace op;
namespace l0op { OP_TYPE_REGISTER(AttnResFwdFused); }
using namespace l0op;
extern "C" aclnnStatus aclnnAttnResFwdFusedGetWorkspaceSize(
    const aclTensor *prefix, const aclTensor *blocks, const aclTensor *proj,
    const aclTensor *norm, const aclTensor *addend, const aclTensor *outputNorm,
    double eps, int64_t validBlocks, int64_t blockTokenStride, int64_t blockWriteIdx,
    double outputNormEps, bool saveMaterialized, bool mix, bool fuseAdd,
    aclTensor *output, aclTensor *prefixOut, aclTensor *materialized,
    uint64_t *workspaceSize, aclOpExecutor **executorOut)
{
    if (!prefix || !blocks || !proj || !norm || !addend || !outputNorm ||
        !output || !prefixOut || !materialized || eps <= 0) return ACLNN_ERR_PARAM_INVALID;
    // Match the existing AttnRes ACLNN entry: external tensors carry their
    // logical dimensions in the view shape. This is host metadata only.
    for (const aclTensor *tensor : {prefix, blocks, proj, norm, addend, outputNorm,
                                   static_cast<const aclTensor *>(output),
                                   static_cast<const aclTensor *>(prefixOut),
                                   static_cast<const aclTensor *>(materialized)}) {
        tensor->SetOriginalShape(tensor->GetViewShape());
    }
    auto exec = CREATE_EXECUTOR();
    if (!exec.get()) return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    if (output->IsEmpty()) {
        *workspaceSize = 0;
        exec.ReleaseTo(executorOut);
        return ACLNN_SUCCESS;
    }
    // The torch adapter validates contiguous rows. Pass the original bank to
    // the kernel; packing its valid prefix would add a launch after every RS.
    auto *executor = exec.get();
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(AttnResFwdFused,
        OP_INPUT(prefix, blocks, proj, norm, addend, outputNorm),
        OP_OUTPUT(output, prefixOut, materialized),
        OP_ATTR(static_cast<float>(eps), false, validBlocks, blockTokenStride,
                blockWriteIdx, static_cast<float>(outputNormEps), saveMaterialized, mix, fuseAdd));
    if (ret != ACLNN_SUCCESS) return ret;
    *workspaceSize = exec->GetWorkspaceSize();
    exec.ReleaseTo(executorOut);
    return ACLNN_SUCCESS;
}
extern "C" aclnnStatus aclnnAttnResFwdFused(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
