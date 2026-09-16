/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_MIXED_QUANT_SPARSE_FLASH_MLA_INNER_H_
#define ACLNN_MIXED_QUANT_SPARSE_FLASH_MLA_INNER_H_
#define ACLNN_API __attribute__((visibility("default")))

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerMixedQuantSparseFlashMlaGetWorkspaceSize(
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
    const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerMixedQuantSparseFlashMla(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                      const aclrtStream stream);

void MixedQuantSparseFlashMlaKvTensorPreProcess(const aclTensor *&kvTensor, const char *tensorName);

void MixedQuantSparseFlashMlaProcessSinks(const aclTensor *&sinksOptional);

#ifdef __cplusplus
}
#endif

#endif
