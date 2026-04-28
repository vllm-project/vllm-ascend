
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_CHUNK_FWD_O_H_
#define ACLNN_CHUNK_FWD_O_H_

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
aclnnStatus aclnnChunkFwdOGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *h,
    const aclTensor *g,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *chunkOffsetsOptional,
    double scale,
    int64_t chunkSize,
    const aclTensor *out,
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
aclnnStatus aclnnChunkFwdO(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
