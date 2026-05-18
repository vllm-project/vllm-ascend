#ifndef ACLNN_NGRAM_SPEC_DECODE_H_
#define ACLNN_NGRAM_SPEC_DECODE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnNgramSpecDecodeGetWorkspaceSize
 * tokenIds : required, [batch_size, max_seq_len], int32
 * numTokensNoSpec : required, [batch_size], int32
 * sampledTokenIds : required, [batch_size, max_new_tokens], int32
 * discardRequestMask : required, [batch_size], int32
 * vocabSize : required, int
 * minN : required, int
 * maxN : required, int
 * k : required, int
 * nextTokenIds : required, [batch_size], int32
 * draftTokenIds : required, [batch_size, k], int32
 * numValidDraftTokens : required, [batch_size], int32
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default"))) aclnnStatus aclnnNgramSpecDecodeGetWorkspaceSize(
    const aclTensor *tokenIds,
    const aclTensor *numTokensNoSpec,
    const aclTensor *sampledTokenIds,
    const aclTensor *discardRequestMask,
    int64_t vocabSize,
    int64_t minN,
    int64_t maxN,
    int64_t k,
    const aclTensor *nextTokenIds,
    const aclTensor *draftTokenIds,
    const aclTensor *numValidDraftTokens,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* function: aclnnNgramSpecDecode
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnNgramSpecDecode(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_NGRAM_SPEC_DECODE_H_
