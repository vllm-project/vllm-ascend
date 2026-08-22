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
 * \file mhc_pre_clamp_sinkhorn.h
 * \brief MhcPreClampSinkhorn L0 kernel interface
 */

#ifndef PTA_NPU_OP_API_INC_LEVEL0_OP_MHC_PRE_SINKHORN_OP_H_
#define PTA_NPU_OP_API_INC_LEVEL0_OP_MHC_PRE_SINKHORN_OP_H_

#include "opdev/op_executor.h"

namespace l0op {

/**
 * @brief MhcPreClampSinkhorn L0 kernel interface
 * Function: Implements the MHC (Manifold-Constraint Hyper-Connection) Pre-Sinkhorn component
 * Formula: H = Sinkhorn(x @ phi + bias) * alpha
 *
 * @param [in] x: Input tensor, shape is [T, D] or [B, S, D], dtype is BF16
 * @param [in] phi: Weight matrix, shape is [D, hcMix], dtype is FP32
 * @param [in] alpha: Scaling coefficients, shape is [hcMix], dtype is FP32
 * @param [in] bias: Bias vector, shape is [hcMix], dtype is FP32
 * @param [in] hcMult: Number of heads (n)
 * @param [in] numIters: Number of Sinkhorn iterations
 * @param [in] hcEps: Epsilon value for Sinkhorn iteration
 * @param [in] normEps: Epsilon value for normalization
 * @param [in] needBackward: Whether backward outputs are needed
 * @param [out] hin: Output tensor for Attention/MLP input, shape is [T, D] or [B, S, D], dtype is BF16
 * @param [out] hPost: Post-processing transformation matrix, shape is [T, n] or [B, S, n], dtype is FP32
 * @param [out] hRes: Combination matrix before Sinkhorn, shape is [T, n, n] or [B, S, n, n], dtype is FP32
 * @param [out] hPre: H_pre after sigmoid (optional), shape is [T, n] or [B, S, n], dtype is FP32
 * @param [out] hcBeforeNorm: Result of x @ phi (optional), shape is [T, n, D] or [B, S, n, D], dtype is FP32
 * @param [out] invRms: 1/r result of RmsNorm (optional), shape is [T, n, D] or [B, S, n, D], dtype is FP32
 * @param [out] sumOut: Intermediate sum results from Sinkhorn (optional), shape is [T, n, D] or [B, S, n, D], dtype is FP32
 * @param [out] normOut: Intermediate norm results from Sinkhorn (optional), shape is [T, n, D] or [B, S, n, D], dtype is FP32
 * @param [in] executor: Op executor
 * @return aclTensor*: Output tensor hin
 */
const aclTensor *MhcPreClampSinkhorn(const aclTensor *x, const aclTensor *phi, const aclTensor *alpha,
                                const aclTensor *bias,
                                int64_t hcMult, int64_t numIters, double hcEps,
                                double normEps, bool needBackward, double clamp_min, double clamp_max,
                                aclTensor *hin, aclTensor *hPost, aclTensor *hRes,
                                aclTensor *hPre, aclTensor *hcBeforeNorm, aclTensor *invRms,
                                aclTensor *sumOut, aclTensor *normOut, aclTensor *hResLogits,
                                aclOpExecutor *executor);

}

#endif // PTA_NPU_OP_API_INC_LEVEL0_OP_MHC_PRE_SINKHORN_OP_H_
