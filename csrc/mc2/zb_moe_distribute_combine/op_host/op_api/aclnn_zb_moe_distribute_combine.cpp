#include "aclnn_zb_moe_distribute_combine.h"
#include "graph/types.h"

#ifdef __cplusplus
extern "C" {
#endif

static constexpr int32_t DISPATCH_DYNAMIC_QUANT_MODE = 2;
enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerZbMoeDistributeCombineGetWorkspaceSize(
    const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine, const aclTensor *epSendCount,
    const aclTensor *scales, const aclTensor *tpSendCount, const aclTensor *xActiveMask, const aclTensor *activationScale,
    const aclTensor *weightScale, const aclTensor *groupList, const aclTensor *expandScales, const aclTensor *sharedExpertX,
    const aclTensor *elasticInfo, const aclTensor *oriX, const aclTensor *constExpertAlpha1, const aclTensor *constExpertAlpha2,
    const aclTensor *constExpertV, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t globalBs,
    int64_t outDtype, int64_t commQuantMode, int64_t extInfo, int64_t groupListType, char *commAlg, int64_t zeroExpertNum,
    int64_t copyExpertNum, int64_t constExpertNum, const aclTensor *XOut, uint64_t *workspaceSize, aclOpExecutor **executor);
extern aclnnStatus aclnnInnerZbMoeDistributeCombine(void *workspace, uint64_t workspaceSize,
                                                                  aclOpExecutor *executor, aclrtStream stream);

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnZbMoeDistributeCombineGetWorkspaceSize(
    const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine, const aclTensor *epSendCount,
    const aclTensor *scales, const aclTensor *tpSendCount, const aclTensor *xActiveMask, const aclTensor *activationScale,
    const aclTensor *weightScale, const aclTensor *groupList, const aclTensor *expandScales, const aclTensor *sharedExpertX,
    const aclTensor *elasticInfo, const aclTensor *oriX, const aclTensor *constExpertAlpha1, const aclTensor *constExpertAlpha2,
    const aclTensor *constExpertV, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t globalBs,
    int64_t outDtype, int64_t commQuantMode, int64_t extInfo, int64_t groupListType, char *commAlg, int64_t zeroExpertNum,
    int64_t copyExpertNum, int64_t constExpertNum, const aclTensor *XOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerZbMoeDistributeCombineGetWorkspaceSize(
        expandX, expertIds, assistInfoForCombine, epSendCount, scales, tpSendCount, xActiveMask, activationScale,
        weightScale, groupList, expandScales, sharedExpertX, elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2,
        constExpertV, epWorldSize, epRankId, moeExpertNum, tpWorldSize, tpRankId, expertShardType, sharedExpertNum,
        sharedExpertRankNum, globalBs, outDtype, commQuantMode, extInfo, groupListType, commAlg, zeroExpertNum,
        copyExpertNum, constExpertNum, XOut, workspaceSize, executor);
}

aclnnStatus aclnnZbMoeDistributeCombine(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                     aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerZbMoeDistributeCombine(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif