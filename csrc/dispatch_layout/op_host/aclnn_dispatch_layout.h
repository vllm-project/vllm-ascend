#ifndef ACLNN_DISPATCH_LAYOUT_H_
#define ACLNN_DISPATCH_LAYOUT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnDispatchLayoutGetWorkspaceSize
 * topkIdx : required
 * numTokens : required
 * numRanks : required
 * numExperts : required
 * numTopk : required
 * numTokensPerRank : required
 * numTokensPerExpert : required
 * isTokenInRank : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDispatchLayoutGetWorkspaceSize(
                                            const aclTensor *topkIdx,
                                            int64_t numTokens,
                                            int64_t numRanks,
                                            int64_t numExperts,
                                            int64_t numTopk,
                                            const aclTensor *numTokensPerRank,
                                            const aclTensor *numTokensPerExpert,
                                            const aclTensor *isTokenInRank,
                                            uint64_t *workspaceSize,
                                            aclOpExecutor **executor);

/* function: aclnnDispatchLayout
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnDispatchLayout(
                                            void *workspace,
                                            uint64_t workspaceSize,
                                            aclOpExecutor *executor,
                                            aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
