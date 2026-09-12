#ifndef OP_API_INC_LEVEL0_OP_CHUNK_GATED_DELTA_RULE_COMPUTE_WY_H
#define OP_API_INC_LEVEL0_OP_CHUNK_GATED_DELTA_RULE_COMPUTE_WY_H

#include "opdev/op_executor.h"

namespace l0op {
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
    aclOpExecutor *executor);
}

#endif
