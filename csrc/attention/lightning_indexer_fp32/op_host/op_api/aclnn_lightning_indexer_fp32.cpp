/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string.h>
#include "graph/types.h"
#include "aclnn_lightning_indexer_fp32.h"

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

extern aclnnStatus aclnnInnerLightningIndexerFp32GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights,
    const aclTensor *actualSeqLengthsQueryOptional, const aclTensor *actualSeqLengthsKeyOptional,
    const aclTensor *blockTableOptional, char *layoutQueryOptional,
    char *layoutKeyOptional, int64_t sparseCount, int64_t sparseMode,
    int64_t preTokens, int64_t nextTokens, bool returnValues,
    const aclTensor *sparseIndicesOut, const aclTensor *sparseValuesOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerLightningIndexerFp32(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         const aclrtStream stream);

class TensorHolder {
public:
    TensorHolder(const aclTensor *&output, aclDataType dataType, std::string varName) {
        inner_ = nullptr;
        name_ = varName;
        if (output == nullptr) {
            inner_ = aclCreateTensor(shape_.data(), shape_.size(),
                dataType, shape_.data(), 0, ACL_FORMAT_ND,
                shape_.data(), shape_.size(), static_cast<void *>(&addr_));
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
    std::vector<int64_t> shape_ = {0};
    int64_t addr_ = 0xff;
    const aclTensor *inner_;
    std::string name_;
};

aclnnStatus aclnnLightningIndexerFp32GetWorkspaceSize(
        const aclTensor *query,
        const aclTensor *key,
        const aclTensor *weights,
        const aclTensor *actualSeqLengthsQueryOptional,
        const aclTensor *actualSeqLengthsKeyOptional,
        const aclTensor *blockTableOptional,
        char *layoutQueryOptional,
        char *layoutKeyOptional,
        int64_t sparseCount,
        int64_t sparseMode,
        int64_t preTokens,
        int64_t nextTokens,
        bool returnValues,
        const aclTensor *sparseIndicesOut,
        const aclTensor *sparseValuesOut,
        uint64_t *workspaceSize,
        aclOpExecutor **executor)
{
    if (query == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Query pointer is null, cannot get data type!");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (returnValues) {
        if (sparseValuesOut == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "sparseValuesOut cannot be nullptr.");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
    }
    auto sparseValuesOutHolder = TensorHolder(sparseValuesOut, ACL_FLOAT, std::string("sparseValuesOut"));
    if (sparseValuesOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Failed to create the holder of tensor sparseValuesOut!");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    return aclnnInnerLightningIndexerFp32GetWorkspaceSize(
        query, key, weights, actualSeqLengthsQueryOptional, actualSeqLengthsKeyOptional, blockTableOptional,
        layoutQueryOptional, layoutKeyOptional, sparseCount, sparseMode, preTokens, nextTokens, returnValues,
        sparseIndicesOut, sparseValuesOut, workspaceSize, executor);
}

aclnnStatus aclnnLightningIndexerFp32(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    return aclnnInnerLightningIndexerFp32(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
