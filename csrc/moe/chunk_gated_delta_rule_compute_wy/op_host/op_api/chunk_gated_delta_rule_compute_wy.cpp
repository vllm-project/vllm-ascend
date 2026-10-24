#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/make_op_executor.h"

#include "chunk_gated_delta_rule_compute_wy.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkGatedDeltaRuleComputeWy);

const std::array<const aclTensor *, 5> ChunkGatedDeltaRuleComputeWy(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    int64_t chunkSize,
    const aclTensor *qKernelOut,
    const aclTensor *kKernelOut,
    const aclTensor *wKernelOut,
    const aclTensor *uKernelOut,
    const aclTensor *gKernelOut,
    aclOpExecutor *executor)
{
    L0_DFX(ChunkGatedDeltaRuleComputeWy, q, k, v, g, beta, chunkSize, qKernelOut, kKernelOut, wKernelOut, uKernelOut,
           gKernelOut);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkGatedDeltaRuleComputeWy, OP_INPUT(q, k, v, g, beta), OP_OUTPUT(qKernelOut, kKernelOut, wKernelOut, uKernelOut, gKernelOut),
        OP_ATTR(chunkSize));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    return {qKernelOut, kKernelOut, wKernelOut, uKernelOut, gKernelOut};
}

} // namespace l0op
