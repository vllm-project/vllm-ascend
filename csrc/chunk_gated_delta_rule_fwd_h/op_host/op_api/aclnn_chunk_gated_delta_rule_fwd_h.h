
/*
 * calution: this file was generated automaticlly donot change it.
*/
#ifndef ACLNN_CHUNK_GATED_DELTA_RULE_FWD_H_H_
#define ACLNN_CHUNK_GATED_DELTA_RULE_FWD_H_H_

#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnChunkFwdOGetWorkspaceSize
 * parameters :
 * q : required
 * k : required
 * v : required
 * h : required
 * g : required
 * cuSeqlensOptional : optional
 * chunkOffsetsOptional : optional
 * scale : required
 * chunkSize : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *g,
    const aclTensor *initialStateOptional,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *chunkIndicesOptional,
    bool outputFinalState,
    int64_t chunkSize,
    const aclTensor *h,
    const aclTensor *vNew,
    const aclTensor *finalState,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
    

/* funtion: aclnnChunkFwdO
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnChunkGatedDeltaRuleFwdH(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
