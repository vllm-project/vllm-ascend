#include <string.h>
#include "graph/types.h"
#include "aclnn_cam_moe_dispatch_normal.h"
#include "aclnnInner_cam_moe_dispatch_normal.h"

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnCamMoeDispatchNormalGetWorkspaceSize(const aclTensor *x, const aclTensor *topkIdx,
    const aclTensor *sendOffset, const aclTensor *sendTokenIdx, const aclTensor *recvOffset, const aclTensor *recvCount,
    char *groupEp, int64_t epWorldSize, int64_t epRankId, char *groupTpOptional, int64_t tpWorldSize, int64_t tpRankId,
    int64_t moeExpertNum, int64_t quantMode, int64_t globalBs, const aclTensor *recvX,
    const aclTensor *recvXScales, const aclTensor *assistInfoForCombine, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerCamMoeDispatchNormalGetWorkspaceSize(x,
        topkIdx,
        sendOffset,
        sendTokenIdx,
        recvOffset,
        recvCount,
        groupEp,
        epWorldSize,
        epRankId,
        groupTpOptional,
        tpWorldSize,
        tpRankId,
        moeExpertNum,
        quantMode,
        globalBs,
        recvX,
        recvXScales,
        assistInfoForCombine,
        workspaceSize,
        executor);
}

aclnnStatus aclnnCamMoeDispatchNormal(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerCamMoeDispatchNormal(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif