/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstring>
#include "graph/types.h"
#include "aclnn_sparse_flash_attention.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

extern aclnnStatus aclnnInnerSparseFlashAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *sparse_indices,
    const aclTensor *blockTableOptional, const aclTensor *actualSeqLengthsQueryOptional, const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional, double scaleValue,
    int64_t sparseBlockSizeOptional, char *layoutQueryOptional, char *layoutKvOptional,
    int64_t sparseMode, int64_t preTokens, int64_t nextTokens, int64_t attentionMode,
    bool returnSoftmaxLse, const aclTensor *attentionOut, const aclTensor *softmaxMax,
    const aclTensor *softmaxSum, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerSparseFlashAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         const aclrtStream stream);

class TensorHolder {
public:
    TensorHolder(const aclTensor *&output, aclDataType dataType, std::string varName) {
        inner_ = nullptr;
        name_ = varName;
        if (output == nullptr) {
            std::vector<int64_t> shape = {0};
            int64_t addr = 0xff;
            inner_ = aclCreateTensor(shape.data(), shape.size(),
                dataType, shape.data(), 0, ACL_FORMAT_ND,
                shape.data(), shape.size(), static_cast<void *>(&addr));
            output = inner_;
        }
    }

    ~TensorHolder() {
        if (inner_) {
            aclDestroyTensor(inner_);
            inner_ = nullptr;
        }
    }
    
    void CheckTensorConditionalNotNull(bool conditional) const {
        if (inner_ && conditional) {
            OP_LOGW("Check %s != nullptr failed!", name_.c_str());
        } else if (!inner_ && !conditional) {
            OP_LOGW("Check %s == nullptr failed!", name_.c_str());
        }
    }

    bool IsTensorNotNull() const {
        return inner_ == nullptr;
    }

private:
    const aclTensor *inner_;
    std::string name_;
};

aclnnStatus aclnnSparseFlashAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseIndices,
    const aclTensor *blockTableOptional,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional,
    double           scaleValue,
    int64_t          sparseBlockSizeOptional,
    char             *layoutQueryOptional,
    char             *layoutKvOptional,
    int64_t          sparseMode,
    int64_t          preTokens,
    int64_t          nextTokens,
    int64_t          attentionMode,
    bool             returnSoftmaxLse,
    const aclTensor *attentionOut,
    const aclTensor *softmaxMax,
    const aclTensor *softmaxSum,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    if (returnSoftmaxLse) {
        if (softmaxMax == nullptr || softmaxSum == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "when returnSoftmaxLse is true, softmaxMax and softmaxSum cannot be nullptr.");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (softmaxMax == nullptr && softmaxSum == nullptr) {
            auto softmaxMaxHolder = TensorHolder(softmaxMax, aclDataType::ACL_FLOAT, std::string("softmaxMax"));
            auto softmaxSumHolder = TensorHolder(softmaxSum, aclDataType::ACL_FLOAT, std::string("softmaxSum"));
            if (softmaxMax == nullptr) {
                OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to create the holder of tensor softmaxMax!");
                return ge::GRAPH_FAILED;
            }
            if (softmaxSum == nullptr) {
                OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to create the holder of tensor softmaxSum!");
                return ge::GRAPH_FAILED;
            }
        }
    }
    return aclnnInnerSparseFlashAttentionGetWorkspaceSize(
        query, key, value, sparseIndices, blockTableOptional, actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional, queryRopeOptional, keyRopeOptional,
        scaleValue, sparseBlockSizeOptional, layoutQueryOptional, layoutKvOptional, sparseMode, preTokens,
        nextTokens, attentionMode, returnSoftmaxLse, attentionOut,
        softmaxMax, softmaxSum, workspaceSize, executor);
}

aclnnStatus aclnnSparseFlashAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    return aclnnInnerSparseFlashAttention(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
