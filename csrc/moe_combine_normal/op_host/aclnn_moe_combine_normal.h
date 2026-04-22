#ifndef ACLNN_MOE_COMBINE_NORMAL_H_
#define ACLNN_MOE_COMBINE_NORMAL_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnMoeCombineGetWorkspaceSize
 * recvX : required
 * tokenSrcInfo : required
 * epRecvCounts : required
 * recvTopkWeights : required
 * tpRecvCountsOptional : required
 * epGroupName : optional
 * epWorldSize : required
 * epRankId : required
 * tpGroupNameOptional : required
 * tpWorldSize : optional
 * tpRankId : optional
 * moeExpertNum : optional
 * globalBs : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMoeCombineNormalGetWorkspaceSize(
                                            const aclTensor *recvX,
                                            const aclTensor *tokenSrcInfo,
                                            const aclTensor *epRecvCounts,
                                            const aclTensor *recvTopkWeights,
                                            const aclTensor *tpRecvCountsOptional,
                                            char *epGroupName,
                                            int64_t epWorldSize,
                                            int64_t epRankId,
                                            char *tpGroupNameOptional,
                                            int64_t tpWorldSize,
                                            int64_t tpRankId,
                                            int64_t moeExpertNum,
                                            int64_t globalBs,
                                            const aclTensor *out,
                                            uint64_t *workspaceSize,
                                            aclOpExecutor **executor);

/* function: aclnnMoeCombine
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMoeCombineNormal(
                                            void *workspace,
                                            uint64_t workspaceSize,
                                            aclOpExecutor *executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif