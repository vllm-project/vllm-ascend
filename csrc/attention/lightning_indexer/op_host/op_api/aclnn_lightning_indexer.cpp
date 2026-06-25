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
#include "aclnn_lightning_indexer.h"

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

extern aclnnStatus aclnnInnerLightningIndexerGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights,
    const aclTensor *actualSeqLengthsQueryOptional, const aclTensor *actualSeqLengthsKeyOptional,
    const aclTensor *blockTableOptional, char *layoutQueryOptional,
    char *layoutKeyOptional, int64_t sparseCount, int64_t sparseMode,
    int64_t preTokens, int64_t nextTokens, bool returnValues,
    const aclTensor *sparseIndicesOut, const aclTensor *sparseValuesOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerLightningIndexer(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
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

aclnnStatus aclnnLightningIndexerGetWorkspaceSize(
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
        return ge::GRAPH_FAILED;
    }
    DataType queryDataType = query->GetDataType();
    aclDataType queryAclDataType = ToAclDataType(queryDataType);
    if (returnValues) {
        if (sparseValuesOut == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "sparseValuesOut cannot be nullptr.");
            return ge::GRAPH_FAILED;
        }
    }
    auto sparseValuesOutHolder = TensorHolder(sparseValuesOut, queryAclDataType, std::string("sparseValuesOut"));
    if (sparseValuesOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Failed to create the holder of tensor sparseValuesOut!");
        return ge::GRAPH_FAILED;
    }

    return aclnnInnerLightningIndexerGetWorkspaceSize(
        query, key, weights, actualSeqLengthsQueryOptional, actualSeqLengthsKeyOptional, blockTableOptional,
        layoutQueryOptional, layoutKeyOptional, sparseCount, sparseMode, preTokens, nextTokens, returnValues,
        sparseIndicesOut, sparseValuesOut, workspaceSize, executor);
}

aclnnStatus aclnnLightningIndexer(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    return aclnnInnerLightningIndexer(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
