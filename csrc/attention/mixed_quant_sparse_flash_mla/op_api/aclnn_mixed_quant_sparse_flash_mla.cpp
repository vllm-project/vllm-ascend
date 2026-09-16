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
 * \file aclnn_mixed_quant_sparse_flash_mla.cpp
 * \brief
 */

#include "aclnn_mixed_quant_sparse_flash_mla.h"

#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_mixed_quant_sparse_flash_mla_inner.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

// 新版本opbase存在TensorV2的新接口，用弱符号判断当前opbase是新版本还是旧版本，旧版本不支持传入非连续tensor
bool NnopbaseSupportTensorV2() __attribute__((weak));

static aclnnStatus CheckTensorContiguous(const aclTensor *oriKvOptional, const aclTensor *cmpKvOptional)
{
    if ((oriKvOptional != nullptr && !IsContiguous(oriKvOptional)) ||
        (cmpKvOptional != nullptr && !IsContiguous(cmpKvOptional))) {
        return ACLNN_ERR_INNER;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMixedQuantSparseFlashMlaGetWorkspaceSize(
    const aclTensor *q, const aclTensor *oriKvOptional, const aclTensor *cmpKvOptional,
    const aclTensor *oriSparseIndicesOptional, const aclTensor *cmpSparseIndicesOptional,
    const aclTensor *oriBlockTableOptional, const aclTensor *cmpBlockTableOptional, const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensOriKvOptional, const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedOriKvOptional, const aclTensor *sequsedCmpKvOptional,
    const aclTensor *cmpResidualKvOptional, const aclTensor *oriTopkLengthOptional,
    const aclTensor *cmpTopkLengthOptional, const aclTensor *sinksOptional, const aclTensor *metadataOptional,
    int64_t quantMode, int64_t ropeHeadDim, double softmaxScale, int64_t cmpRatio, int64_t oriMaskMode,
    int64_t cmpMaskMode, int64_t oriWinLeft, int64_t oriWinRight, const char *layoutQOptional,
    const char *layoutKvOptional, int64_t topkValueMode, bool returnSoftmaxLse, const aclTensor *attnOut,
    const aclTensor *softmaxLseOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_LOGD("start aclnnMixedQuantSparseFlashMlaGetWorkspaceSize");

    MixedQuantSparseFlashMlaKvTensorPreProcess(oriKvOptional, "ori_kv");
    MixedQuantSparseFlashMlaKvTensorPreProcess(cmpKvOptional, "cmp_kv");
    MixedQuantSparseFlashMlaProcessSinks(sinksOptional);

    aclnnStatus ret = CheckTensorContiguous(oriKvOptional, cmpKvOptional);
    if (ret != ACLNN_SUCCESS && NnopbaseSupportTensorV2 == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER, "When tensor is not contiguous, opbase package version check failed");
        return ret;
    }
    ret = aclnnInnerMixedQuantSparseFlashMlaGetWorkspaceSize(
        q, oriKvOptional, cmpKvOptional, oriSparseIndicesOptional, cmpSparseIndicesOptional, oriBlockTableOptional,
        cmpBlockTableOptional, cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional,
        sequsedOriKvOptional, sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional, cmpTopkLengthOptional,
        sinksOptional, metadataOptional, quantMode, ropeHeadDim, softmaxScale, cmpRatio, oriMaskMode, cmpMaskMode,
        oriWinLeft, oriWinRight, layoutQOptional, layoutKvOptional, topkValueMode, returnSoftmaxLse, attnOut,
        softmaxLseOptional, workspaceSize, executor);
    return ret;
}

aclnnStatus aclnnMixedQuantSparseFlashMla(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          const aclrtStream stream)
{
    return aclnnInnerMixedQuantSparseFlashMla(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
