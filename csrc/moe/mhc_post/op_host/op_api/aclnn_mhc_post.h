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
 * \file aclnn_mhc_post.h
 * \brief MhcPost ACLNN API
 */

#ifndef OP_API_INC_MHC_POST_H
#define OP_API_INC_MHC_POST_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMhcPostGetWorkspaceSize的第一段接口，计算workspace大小。
 * 功能描述：该算子实现Manifold-Constraint Hyper-Connection的Post部分，处理残差连接的后处理操作。
 * 计算公式：x_{l+1} = (H_{l}^{res})^{T} * x_l + h_{l}^{out} * H_{t}^{post}
 *          其中：(H_{l}^{res})^{T} * x_l 表示对x_l使用转置后的h_res矩阵进行矩阵乘法变换
 *                h_{l}^{out} * H_{t}^{post} 表示逐元素相乘后广播到所有维度
 * @domain aclnn_ops_infer
 */
aclnnStatus aclnnMhcPostGetWorkspaceSize(const aclTensor *x, const aclTensor *hRes, const aclTensor *hOut,
                                         const aclTensor *hPost, aclTensor *out, uint64_t *workspaceSize,
                                         aclOpExecutor **executor);

/**
 * @brief aclnnMhcPost的第二段接口，用于执行计算。
 */
aclnnStatus aclnnMhcPost(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MHC_POST_H