#ifndef ACLNN_COMBINE_AND_DISPATCH_REBATCH_H_
#define ACLNN_COMBINE_AND_DISPATCH_REBATCH_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnCombineAndDispatchRebatchGetWorkspaceSize
 * batchBeforeDispatch    : required, BF16 [bs, hidden_size]
 * residualsBeforeDispatch: required, BF16 [bs, hidden_size]
 * cutIdx                 : required, INT32 [num_cuts]
 * discardLabel           : required, INT32 [discard_num]
 * discardTokens          : required, BF16 [discard_num, hidden_size]
 * discardMoeReqIdTensor  : required, INT32 [discard_num]
 * reqIdList              : required, INT32 [req_num]
 * discardResiduals       : required, BF16 [discard_num, hidden_size]
 * runningTopkWeight      : required, BF16 [bs]
 * discardTopkWeight      : required, BF16 [discard_num]
 * isCombineRebatch       : required, int64
 * moeCntInput            : required, int64
 * rebatchOutput          : required, BF16 output
 * rebatchResiduals       : required, BF16 output
 * workspaceSize          : size of workspace (output)
 * executor               : executor context (output)
 */
__attribute__((visibility("default"))) aclnnStatus aclnnCombineAndDispatchRebatchGetWorkspaceSize(
    const aclTensor *batchBeforeDispatch,
    const aclTensor *residualsBeforeDispatch,
    const aclTensor *cutIdx,
    const aclTensor *prefixIndex,
    const aclTensor *discardLabel,
    const aclTensor *discardTokens,
    const aclTensor *discardMoeReqIdTensor,
    const aclTensor *reqIdList,
    const aclTensor *discardResiduals,
    const aclTensor *runningTopk,
    const aclTensor *discardTopk,
    const aclTensor *runningTopkWeight,
    const aclTensor *discardTopkWeight,
    const aclTensor *moeCntInput,
    const aclTensor *actualBs,
    const aclTensor *historyBs,
    const aclTensor *runningBs,
    const aclTensor *originalBs,
    int64_t isCombineRebatch,
    int64_t maxBs,
    int64_t k,
    // const aclTensor *rebatchOutput,
    const aclTensor *rebatchResiduals,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnCombineAndDispatchRebatch
 * workspace     : workspace memory addr (input)
 * workspaceSize : size of workspace (input)
 * executor      : executor context (input)
 * stream        : acl stream
 */
__attribute__((visibility("default"))) aclnnStatus aclnnCombineAndDispatchRebatch(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_COMBINE_AND_DISPATCH_REBATCH_H_
