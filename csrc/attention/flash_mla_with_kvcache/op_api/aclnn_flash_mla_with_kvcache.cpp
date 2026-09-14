/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_flash_mla_with_kvcache.cpp
 * \brief
 */

#include "aclnn_flash_mla_with_kvcache.h"

#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

// aclnnInner* 实现由构建工具自动生成（build/autogen/inner/aclnnInner_flash_mla_with_kvcache.cpp）
// 并链接进 libcust_opapi.so，此处仅作前置声明（合并自原 aclnn_flash_mla_with_kvcache_inner.h）。
extern aclnnStatus aclnnInnerFlashMlaWithKvcacheGetWorkspaceSize(
    const aclTensor *q, const aclTensor *kCache, const aclTensor *blockTableOptional,
    const aclTensor *cacheSeqlensOptional, const aclTensor *cuSeqlensQOptional, const aclTensor *sequsedQOptional,
    const aclTensor *attnMaskOptional, const aclTensor *metadataOptional, int64_t headDimV, double softmaxScale,
    int64_t maskMode, int64_t maxSeqlenQ, int64_t maxSeqlenKV, const char *layoutQ, const char *layoutKv,
    const char *layoutOut, int64_t returnSoftmaxLse, const aclTensor *attnOut, const aclTensor *softmaxLse,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerFlashMlaWithKvcache(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                 const aclrtStream stream);

namespace {

void FlashMlaWithKvcacheProcessSoftmaxLse(int64_t returnSoftmaxLse, const aclTensor *softmaxLse,
                                          const aclTensor *&tempTensor, const aclTensor *&placeHolder)
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

} // namespace

// 第一段接口：计算workspace大小
aclnnStatus aclnnFlashMlaWithKvcacheGetWorkspaceSize(
    const aclTensor *q, const aclTensor *kCache, const aclTensor *blockTableOptional,
    const aclTensor *cacheSeqlensOptional, const aclTensor *cuSeqlensQOptional, const aclTensor *sequsedQOptional,
    const aclTensor *attnMaskOptional, const aclTensor *metadataOptional, int64_t headDimV, double softmaxScale,
    int64_t maskMode, int64_t maxSeqlenQ, int64_t maxSeqlenKV, const char *layoutQ, const char *layoutKv,
    const char *layoutOut, int64_t returnSoftmaxLse, const aclTensor *attnOut, const aclTensor *softmaxLseOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("start aclnnFlashMlaWithKvcacheGetWorkspaceSize");

    const aclTensor *placeHolder = nullptr;
    const aclTensor *tempTensor = nullptr;

    FlashMlaWithKvcacheProcessSoftmaxLse(returnSoftmaxLse, softmaxLseOptional, tempTensor, placeHolder);

    aclnnStatus ret = aclnnInnerFlashMlaWithKvcacheGetWorkspaceSize(
        q, kCache, blockTableOptional, cacheSeqlensOptional, cuSeqlensQOptional, sequsedQOptional, attnMaskOptional,
        metadataOptional, headDimV, softmaxScale, maskMode, maxSeqlenQ, maxSeqlenKV, layoutQ, layoutKv, layoutOut,
        returnSoftmaxLse, attnOut, placeHolder, workspaceSize, executor);

    // 销毁占位符
    if (returnSoftmaxLse == 0) {
        aclDestroyTensor(tempTensor);
    }

    return ret;
}

// 第二段接口：执行计算
aclnnStatus aclnnFlashMlaWithKvcache(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    return aclnnInnerFlashMlaWithKvcache(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
