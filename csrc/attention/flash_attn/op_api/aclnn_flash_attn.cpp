/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_flash_attn.cpp
 * \brief
 */

#include "aclnn/aclnn_base.h"

#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

// inner 接口（由框架根据 L0 op 注册自动生成）
extern aclnnStatus aclnnInnerFlashAttnGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *blockTableOptional,
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKvOptional, const aclTensor *sinksOptional, const aclTensor *attnMaskOptional,
    const aclTensor *metadataOptional, double softmaxScale, int64_t maskMode, int64_t winLeft, int64_t winRight,
    int64_t maxSeqlenQ, int64_t maxSeqlenKV, const char *layoutQ, const char *layoutKv, const char *layoutOut,
    int64_t returnSoftmaxLse, const aclTensor *attnOut, const aclTensor *softmaxLse, uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus aclnnInnerFlashAttn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       const aclrtStream stream);

namespace {

void FlashAttnProcessSoftmaxLse(int64_t returnSoftmaxLse, const aclTensor *softmaxLse, const aclTensor *&tempTensor,
                                const aclTensor *&placeHolder)
{
    if (returnSoftmaxLse == false) {
        std::vector<int64_t> shape = {0};
        int64_t addr = 0xff;
        tempTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape.data(), 0, ACL_FORMAT_ND,
                                     shape.data(), shape.size(), static_cast<void *>(&addr));
        placeHolder = tempTensor;
    } else {
        placeHolder = softmaxLse;
    }
}

// sinks shape为{0}时置nullptr
void FlashAttnProcessSinks(const aclTensor *&sinksOptional)
{
    if (sinksOptional != nullptr) {
        const auto &shape = sinksOptional->GetViewShape();
        if (shape.GetDimNum() == 1U && shape[0] == 0) {
            OP_LOGD("sinks shape is {0}, treat as nullptr.");
            sinksOptional = nullptr;
        }
    }
}

} // namespace

// 第一段接口：计算workspace大小
aclnnStatus aclnnFlashAttnGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v,
                                           const aclTensor *blockTableOptional, const aclTensor *cuSeqlensQOptional,
                                           const aclTensor *cuSeqlensKvOptional, const aclTensor *sequsedQOptional,
                                           const aclTensor *sequsedKvOptional, const aclTensor *sinksOptional,
                                           const aclTensor *attnMaskOptional, const aclTensor *metadataOptional,
                                           double softmaxScale, int64_t maskMode, int64_t winLeft, int64_t winRight,
                                           int64_t maxSeqlenQ, int64_t maxSeqlenKV, const char *layoutQ,
                                           const char *layoutKv, const char *layoutOut, int64_t returnSoftmaxLse,
                                           const aclTensor *attnOut, const aclTensor *softmaxLseOptional,
                                           uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("start aclnnFlashAttnGetWorkspaceSize");

    // sinks shape为{0}时置nullptr
    FlashAttnProcessSinks(sinksOptional);

    const aclTensor *placeHolder = nullptr;
    const aclTensor *tempTensor = nullptr;

    FlashAttnProcessSoftmaxLse(returnSoftmaxLse, softmaxLseOptional, tempTensor, placeHolder);

    aclnnStatus ret = aclnnInnerFlashAttnGetWorkspaceSize(
        q, k, v, blockTableOptional, cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional,
        sinksOptional, attnMaskOptional, metadataOptional, softmaxScale, maskMode, winLeft, winRight, maxSeqlenQ,
        maxSeqlenKV, layoutQ, layoutKv, layoutOut, returnSoftmaxLse, attnOut, placeHolder, workspaceSize, executor);

    // 销毁占位符
    if (returnSoftmaxLse == 0) {
        aclDestroyTensor(tempTensor);
    }

    return ret;
}

// 第二段接口：执行计算
aclnnStatus aclnnFlashAttn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)
{
    return aclnnInnerFlashAttn(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
