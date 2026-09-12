#include "aclnn_chunk_gated_delta_rule_compute_wy.h"
#include "chunk_gated_delta_rule_compute_wy.h"

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/make_op_executor.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

struct ChunkGatedDeltaRuleComputeWyParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *beta = nullptr;
    int64_t chunkSize = 64;
    const aclTensor *qKernelOut = nullptr;
    const aclTensor *kKernelOut = nullptr;
    const aclTensor *wKernelOut = nullptr;
    const aclTensor *uKernelOut = nullptr;
    const aclTensor *gKernelOut = nullptr;
};

static aclnnStatus CheckNotNull(const ChunkGatedDeltaRuleComputeWyParams &p)
{
    CHECK_COND(p.q != nullptr, ACLNN_ERR_PARAM_NULLPTR, "q must not be nullptr.");
    CHECK_COND(p.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(p.v != nullptr, ACLNN_ERR_PARAM_NULLPTR, "v must not be nullptr.");
    CHECK_COND(p.g != nullptr, ACLNN_ERR_PARAM_NULLPTR, "g must not be nullptr.");
    CHECK_COND(p.beta != nullptr, ACLNN_ERR_PARAM_NULLPTR, "beta must not be nullptr.");
    CHECK_COND(p.qKernelOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "qKernelOut must not be nullptr.");
    CHECK_COND(p.kKernelOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "kKernelOut must not be nullptr.");
    CHECK_COND(p.wKernelOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "wKernelOut must not be nullptr.");
    CHECK_COND(p.uKernelOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "uKernelOut must not be nullptr.");
    CHECK_COND(p.gKernelOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "gKernelOut must not be nullptr.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const ChunkGatedDeltaRuleComputeWyParams &p)
{
    CHECK_RET(CheckNotNull(p) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(p.chunkSize == 64, ACLNN_ERR_PARAM_INVALID, "chunk_size must be 64.");
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus MakeInputsContiguous(ChunkGatedDeltaRuleComputeWyParams &p, aclOpExecutor *executor)
{
    CHECK_COND(DataContiguous(p.q, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Contiguous q failed.");
    CHECK_COND(DataContiguous(p.k, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Contiguous k failed.");
    CHECK_COND(DataContiguous(p.v, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Contiguous v failed.");
    CHECK_COND(DataContiguous(p.g, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Contiguous g failed.");
    CHECK_COND(DataContiguous(p.beta, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Contiguous beta failed.");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleComputeWyGetWorkspaceSize(
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
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    CHECK_COND(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR, "workspaceSize must not be nullptr.");
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor must not be nullptr.");

    ChunkGatedDeltaRuleComputeWyParams p{q, k, v, g, beta, chunkSize, qKernelOut, kKernelOut, wKernelOut, uKernelOut,
                                         gKernelOut};

    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRuleComputeWy, DFX_IN(q, k, v, g, beta),
                   DFX_OUT(qKernelOut, kKernelOut, wKernelOut, uKernelOut, gKernelOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    CHECK_RET(CheckParams(p) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(MakeInputsContiguous(p, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "MakeInputsContiguous failed.");

    auto result = l0op::ChunkGatedDeltaRuleComputeWy(p.q, p.k, p.v, p.g, p.beta, p.chunkSize, p.qKernelOut, p.kKernelOut,
                                                     p.wKernelOut, p.uKernelOut, p.gKernelOut, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(l0op::ViewCopy(result[0], p.qKernelOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[1], p.kKernelOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[2], p.wKernelOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[3], p.uKernelOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result[4], p.gKernelOut, executorPtr) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleComputeWy(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    CHECK_COND(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR, "executor must not be nullptr.");
    CHECK_COND(stream != nullptr, ACLNN_ERR_PARAM_NULLPTR, "stream must not be nullptr.");
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRuleComputeWy);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "ChunkGatedDeltaRuleComputeWy launch failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
