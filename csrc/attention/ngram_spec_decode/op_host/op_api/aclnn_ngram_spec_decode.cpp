#include <string.h>
#include "graph/types.h"
#include "aclnn_ngram_spec_decode.h"

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerNgramSpecDecodeGetWorkspaceSize(
    const aclTensor *tokenIds,
    const aclTensor *numTokensNoSpec,
    const aclTensor *sampledTokenIds,
    const aclTensor *discardRequestMask,
    int64_t vocabSize,
    int64_t minN,
    int64_t maxN,
    int64_t k,
    const aclTensor *nextTokenIds,
    const aclTensor *draftTokenIds,
    const aclTensor *numValidDraftTokens,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus aclnnInnerNgramSpecDecode(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

aclnnStatus aclnnNgramSpecDecodeGetWorkspaceSize(
    const aclTensor *tokenIds,
    const aclTensor *numTokensNoSpec,
    const aclTensor *sampledTokenIds,
    const aclTensor *discardRequestMask,
    int64_t vocabSize,
    int64_t minN,
    int64_t maxN,
    int64_t k,
    const aclTensor *nextTokenIds,
    const aclTensor *draftTokenIds,
    const aclTensor *numValidDraftTokens,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerNgramSpecDecodeGetWorkspaceSize(
        tokenIds, numTokensNoSpec, sampledTokenIds, discardRequestMask,
        vocabSize, minN, maxN, k,
        nextTokenIds, draftTokenIds, numValidDraftTokens,
        workspaceSize, executor);
}

aclnnStatus aclnnNgramSpecDecode(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerNgramSpecDecode(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
