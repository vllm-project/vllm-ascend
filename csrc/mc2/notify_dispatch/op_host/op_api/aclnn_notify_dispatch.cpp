#include <string.h>
#include "graph/types.h"
#include "aclnn_notify_dispatch.h"

extern void NnopbaseOpLogE(const aclnnStatus code, const char *const expr);

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

extern aclnnStatus aclnnInnerNotifyDispatchGetWorkspaceSize(
    const aclTensor *sendData,
    const aclTensor *tokenPerExpertData,
    int64_t sendCount,
    int64_t numTokens,
    char *commGroup,
    int64_t rankSize,
    int64_t rankId,
    int64_t localRankSize,
    int64_t localRankId,
    const aclTensor *sendDataOffset,
    const aclTensor *recvData,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus aclnnInnerNotifyDispatch(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

aclnnStatus aclnnNotifyDispatchGetWorkspaceSize(
    const aclTensor *sendData,
    const aclTensor *tokenPerExpertData,
    int64_t sendCount,
    int64_t numTokens,
    char *commGroup,
    int64_t rankSize,
    int64_t rankId,
    int64_t localRankSize,
    int64_t localRankId,
    const aclTensor *sendDataOffset,
    const aclTensor *recvData,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerNotifyDispatchGetWorkspaceSize(
        sendData,
        tokenPerExpertData,
        sendCount,
        numTokens,
        commGroup,
        rankSize,
        rankId,
        localRankSize,
        localRankId,
        sendDataOffset,
        recvData,
        workspaceSize,
        executor);
}

aclnnStatus aclnnNotifyDispatch(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerNotifyDispatch(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
