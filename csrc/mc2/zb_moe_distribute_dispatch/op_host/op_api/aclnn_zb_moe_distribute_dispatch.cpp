#include "aclnn_zb_moe_distribute_dispatch.h"
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

extern aclnnStatus aclnnInnerZbMoeDistributeDispatchGetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIds, const aclTensor *scales, const aclTensor *xActiveMask, const aclTensor *elasticInfo,
    int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType,
    int64_t sharedExpertNum, int64_t shareExpertRankNum, int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, int64_t extInfo,
    char *commAlg, int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum,
    const aclTensor *expandX, const aclTensor *dynamicScales, const aclTensor *assist_info_for_combine, const aclTensor *expertTokensNums,
    const aclTensor *epRecvCounts, const aclTensor *tpRecvCounts, uint64_t *workspaceSize, aclOpExecutor **executor);
extern aclnnStatus aclnnInnerZbMoeDistributeDispatch(void *workspace, uint64_t workspaceSize,
                                                                   aclOpExecutor *executor, aclrtStream stream);

extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void *executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnZbMoeDistributeDispatchGetWorkspaceSize(
    const aclTensor *x, const aclTensor *expertIds, const aclTensor *scalesOptional,
    const aclTensor *xActiveMaskOptional, const aclTensor *elasticInfoOptional, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum,
    int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum,
    int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, int64_t extInfo,
    char *commAlgOptional, int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum,
    const aclTensor *expandXOut, const aclTensor *dynamicScalesOut, const aclTensor *assistInfoForCombineOut,
    const aclTensor *expertTokenNumsOut, const aclTensor *epRecvCountOut, const aclTensor *tpRecvCountOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    return aclnnInnerZbMoeDistributeDispatchGetWorkspaceSize(
        x, expertIds, scalesOptional, xActiveMaskOptional, elasticInfoOptional, epWorldSize, epRankId, moeExpertNum, tpWorldSize,
        tpRankId, expertShardType, sharedExpertNum, sharedExpertRankNum, quantMode, globalBs, expertTokenNumsType, extInfo,
        commAlgOptional, zeroExpertNum, copyExpertNum, constExpertNum, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut,
        epRecvCountOut, tpRecvCountOut, workspaceSize, executor);
}

aclnnStatus aclnnZbMoeDistributeDispatch(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                      aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerZbMoeDistributeDispatch(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif