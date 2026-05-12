
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_INNER_GATHER_PA_KV_CACHE_H_
#define ACLNN_INNER_GATHER_PA_KV_CACHE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnGatherPaKvCacheVllmGetWorkspaceSize
 * parameters :
 * keyCache : required
 * valueCache : required
 * blockTables : required
 * seqLens : required
 * keyRef : required
 * valueRef : required
 * seqOffsetOptional : optional
 * cacheModeOptional : optional
 * isSeqLensCumsum : optional
 * kvCacheStridesOptional : optional
 * keyRef : required
 * valueRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnGatherPaKvCacheVllmGetWorkspaceSize(
    const aclTensor *keyCache,
    const aclTensor *valueCache,
    const aclTensor *blockTables,
    const aclTensor *seqLens,
    aclTensor *keyRef,
    aclTensor *valueRef,
    const aclTensor *seqOffsetOptional,
    char *cacheModeOptional,
    bool isSeqLensCumsum,
    const aclIntArray *kvCacheStridesOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnInnerGatherPaKvCacheVllm
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnInnerGatherPaKvCacheVllm(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
