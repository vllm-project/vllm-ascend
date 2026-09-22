// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "aclnn_attn_res_fwd_with_add.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
using namespace op;
namespace l0op {
OP_TYPE_REGISTER(AttnResFwdWithAdd);
}
using namespace l0op;
extern "C" aclnnStatus aclnnAttnResFwdWithAddGetWorkspaceSize(
    const aclTensor *prefix, const aclTensor *blocks, const aclTensor *proj,
    const aclTensor *norm, const aclTensor *addend, double eps,
    aclTensor *output, aclTensor *prefixOut, uint64_t *workspaceSize, aclOpExecutor **executorOut)
{
    if (!prefix || !blocks || !proj || !norm || !addend || !output || !prefixOut || eps <= 0) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto exec = CREATE_EXECUTOR();
    if (!exec.get()) return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    if (output->IsEmpty()) {
        *workspaceSize = 0;
        exec.ReleaseTo(executorOut);
        return ACLNN_SUCCESS;
    }
    auto p = l0op::Contiguous(prefix, exec.get());
    auto b = l0op::Contiguous(blocks, exec.get());
    auto w = l0op::Contiguous(proj, exec.get());
    auto n = l0op::Contiguous(norm, exec.get());
    auto a = l0op::Contiguous(addend, exec.get());
    if (!p || !b || !w || !n || !a) return ACLNN_ERR_INNER_NULLPTR;
    auto *executor = exec.get();
    auto y = exec->AllocTensor(prefix->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto z = exec->AllocTensor(prefix->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto ret = INFER_SHAPE(AttnResFwdWithAdd, OP_INPUT(p, b, w, n, a), OP_OUTPUT(y, z),
                          OP_ATTR(static_cast<float>(eps), false));
    if (ret != ACLNN_SUCCESS) return ret;
    ret = ADD_TO_LAUNCHER_LIST_AICORE(AttnResFwdWithAdd, OP_INPUT(p, b, w, n, a), OP_OUTPUT(y, z),
                                     OP_ATTR(static_cast<float>(eps), false));
    if (ret != ACLNN_SUCCESS) return ret;
    if (!l0op::ViewCopy(y, output, exec.get()) || !l0op::ViewCopy(z, prefixOut, exec.get())) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    *workspaceSize = exec->GetWorkspaceSize();
    exec.ReleaseTo(executorOut);
    return ACLNN_SUCCESS;
}
extern "C" aclnnStatus aclnnAttnResFwdWithAdd(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
