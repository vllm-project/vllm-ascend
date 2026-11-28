// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef ACLNN_GMM_SWIGLU_OAI_MX_QUANT_H
#define ACLNN_GMM_SWIGLU_OAI_MX_QUANT_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnGmmSwigluOaiMxQuantGetWorkspaceSize(
    const aclTensor* x, const aclTensor* xScale, const aclTensor* groupList, const aclTensorList* weight,
    const aclTensorList* weightScale, int64_t groupListType, double clampLimit, double gluAlpha, double gluBias,
    double dstTypeMax, aclTensor* y, aclTensor* yScale, uint64_t* workspaceSize, aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus aclnnGmmSwigluOaiMxQuant(void* workspace, uint64_t workspaceSize,
                                                                               aclOpExecutor* executor,
                                                                               aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif
