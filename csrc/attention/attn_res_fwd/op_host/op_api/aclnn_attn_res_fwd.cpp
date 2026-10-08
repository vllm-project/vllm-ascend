// SPDX-License-Identifier: Apache-2.0
#include "aclnn_attn_res_fwd.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
using namespace op;
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
