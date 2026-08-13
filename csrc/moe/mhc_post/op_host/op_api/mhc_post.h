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
 * \file mhc_post.h
 * \brief MhcPost L0 kernel interface
 */

#ifndef OP_API_INC_LEVEL0_OP_MHC_POST_OP_H
#define OP_API_INC_LEVEL0_OP_MHC_POST_OP_H

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief MhcPost L0 kernel interface
 * Function: Implements the MHC (Manifold-Constraint Hyper-Connection) Post component
 * Formula: x_{l+1} = (H_{l}^{res})^{T} * x_l + h_{l}^{out} * H_{t}^{post}
 *
 * @param [in] x: Input tensor x_l, shape is [T, n, D] or [B, S, n, D]
 * @param [in] h_res: Residual connection matrix, shape is [T, n, n] or [B, S, n, n], dtype is FP32
 * @param [in] h_out: Output state from previous layer, shape is [T, D] or [B, S, D]
 * @param [in] h_post: Post processing matrix, shape is [T, n] or [B, S, n], dtype is FP32
 * @param [in] executor: Op executor
 * @return aclTensor*: Output tensor x_{l+1}
 */
const aclTensor *MhcPost(const aclTensor *x, const aclTensor *h_res,
                         const aclTensor *h_out, const aclTensor *h_post,
                         aclOpExecutor *executor);

}

#endif