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
 * \file aclnn_mhc_pre_clamp_sinkhorn.h
 * \brief MhcPreClampSinkhorn ACLNN API
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_MHC_PRE_CLAMP_SINKHORN_H_
#define OP_API_INC_LEVEL2_ACLNN_MHC_PRE_CLAMP_SINKHORN_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMhcPreClampSinkhornGetWorkspaceSize的第一段接口，计算workspace大小。
 * 功能描述：该算子实现Manifold-Constraint Hyper-Connection的Pre-Sinkhorn部分，
 *          通过Sinkhorn迭代算法生成正交化的连接矩阵，用于Attention/MLP层的输入预处理。
 * 计算公式：H = Sinkhorn(x @ phi + bias) * alpha
 *          其中：x @ phi 表示输入与权重矩阵的矩阵乘法
 *                Sinkhorn迭代实现矩阵的正交化归一化
 *                alpha为缩放系数，用于调整输出范围
 * @domain aclnn_ops_infer
 */
aclnnStatus aclnnMhcPreClampSinkhornGetWorkspaceSize(
    const aclTensor *x, const aclTensor *phi, const aclTensor *alpha, const aclTensor *bias,
    int64_t hcMult, int64_t numIters, double hcEps, double normEps, bool needBackward, double clamp_min, double clamp_max,
    aclTensor *hin, aclTensor *hPost, aclTensor *hRes,
    aclTensor *hPre, aclTensor *hcBeforeNorm, aclTensor *invRms,
    aclTensor *sumOut, aclTensor *normOut, aclTensor *hResLogits,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnMhcPreClampSinkhorn的第二段接口，用于执行计算。
 */
aclnnStatus aclnnMhcPreClampSinkhorn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_MHC_PRE_CLAMP_SINKHORN_H_
