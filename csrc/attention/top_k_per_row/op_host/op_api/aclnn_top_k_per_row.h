/**
 * Copyright (c) 2026 SeaGaruda
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#ifndef ACLNN_TOP_K_PER_ROW_H_
#define ACLNN_TOP_K_PER_ROW_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Ragged per-row top-k over a column window.
 *
 * @param logits      (num_rows, width), float32 / float16 / bfloat16, ND
 * @param rowStarts   (num_rows,), int32, inclusive window start
 * @param rowEnds     (num_rows,), int32, exclusive window end
 * @param topk        1 <= topk <= 2048
 * @param indices     (num_rows, topk), int32 out, -1 padded
 */
__attribute__((visibility("default"))) aclnnStatus aclnnTopKPerRowGetWorkspaceSize(
    const aclTensor* logits, const aclTensor* rowStarts, const aclTensor* rowEnds,
    int64_t topk, aclTensor* indices, uint64_t* workspaceSize, aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus aclnnTopKPerRow(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif  // ACLNN_TOP_K_PER_ROW_H_
