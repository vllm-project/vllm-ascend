#ifndef ACLNN_ZB_MOE_DISTRIBUTE_COMBINE_H_
#define ACLNN_ZB_MOE_DISTRIBUTE_COMBINE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnZbMoeDistributeCombineGetWorkspaceSize
 * parameters :
 * expandX : required
 * expertIds : required
 * assistInfoForCombine : required
 * epSendCount : required
 * scales : required
 * tpSendCount : optional
 * xActiveMask : optional
 * activationScale : optional
 * weightScale : optional
 * groupList : optional
 * expandScales : optional
 * sharedExpertX : optional
 * elasticInfo : optional
 * oriX : optional
 * constExpertAlpha1 : optional
 * constExpertAlpha2 : optional
 * constExpertV : optional
 * epWorldSize : required
 * epRankId : required
 * moeExpertNum : required
 * tpWorldSize : optional
 * tpRankId : optional
 * expertShardType : optional
 * sharedExpertNum : optional
 * sharedExpertRankNum : optional
 * globalBs : optional
 * outDtype : optional
 * commQuantMode : optional
 * extInfo : required
 * groupListType : optional
 * commAlg : optional
 * zeroExpertNum : optional
 * copyExpertNum : optional
 * constExpertNum : optional
 * XOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default"))) aclnnStatus aclnnZbMoeDistributeCombineGetWorkspaceSize(
    const aclTensor *expandX, const aclTensor *expertIds, const aclTensor *assistInfoForCombine, const aclTensor *epSendCount,
    const aclTensor *scales, const aclTensor *tpSendCount, const aclTensor *xActiveMask, const aclTensor *activationScale,
    const aclTensor *weightScale, const aclTensor *groupList, const aclTensor *expandScales, const aclTensor *sharedExpertX,
    const aclTensor *elasticInfo, const aclTensor *oriX, const aclTensor *constExpertAlpha1, const aclTensor *constExpertAlpha2,
    const aclTensor *constExpertV, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, int64_t tpWorldSize,
    int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t globalBs,
    int64_t outDtype, int64_t commQuantMode, int64_t extInfo, int64_t groupListType, char *commAlg, int64_t zeroExpertNum,
    int64_t copyExpertNum, int64_t constExpertNum, const aclTensor *XOut, uint64_t *workspaceSize, aclOpExecutor **executor);

/* function: aclnnZbMoeDistributeCombine
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnZbMoeDistributeCombine(void *workspace,
                                                                                            uint64_t workspaceSize,
                                                                                            aclOpExecutor *executor,
                                                                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif