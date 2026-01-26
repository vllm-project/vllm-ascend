
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SPLIT_MROPE_H_
#define ACLNN_SPLIT_MROPE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSplitMropeGetWorkspaceSize
 * parameters :
 * positions : required
 * inQkv : required
 * inCosSinCache : required
 * numQHeads : required
 * numKvHeads : required
 * headSize : required
 * mropeSection : required
 * isNeoxStyle : required
 * outQueryOut : required
 * outKeyOut : required
 * outValueOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSplitMropeGetWorkspaceSize(
    const aclTensor *positions,
    const aclTensor *inQkv,
    const aclTensor *inCosSinCache,
    int64_t numQHeads,
    int64_t numKvHeads,
    int64_t headSize,
    const aclIntArray *mropeSection,
    int64_t isNeoxStyle,
    const aclTensor *outQueryOut,
    const aclTensor *outKeyOut,
    const aclTensor *outValueOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSplitMrope
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSplitMrope(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
