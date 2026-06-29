/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MEGA_MOE_
#define OP_API_INC_MEGA_MOE_

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：该算子将 MoE（Mixture of Experts）以及 FFN（Feed-Forward Network）的完整计算流程融合为单个算子，
 *         实现 Dispatch + GroupMatmul1 + SwiGLUQuant + GroupMatmul2 + Combine 的端到端融合计算。
 * @brief aclnnMegaMoe的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] context: 计算输入，Tensor，数据类型int32，必须为1维，数据格式支持ND。本卡通信域信息数据。
 * @param [in] x: 计算输入，Tensor，数据类型bfloat16，必须为2维，数据格式支持ND。本卡发送的token数据。
 * @param [in] topkIds: 计算输入，Tensor，数据类型int32，必须为2维，数据格式支持ND。每个token的topK个专家索引。
 * @param [in] topkWeights: 计算输入，Tensor，数据类型float32或bfloat16，必须为2维，数据格式支持ND。每个token的topK个专家权重。
 * @param [in] weight1: 计算输入，TensorList，数据类型float8_e5m2或float8_e4m3fn，数据格式支持ND。
 *                      GroupMatmul1计算的右矩阵，用于计算SwiGLU激活前的线性变换。
 * @param [in] weight2: 计算输入，TensorList，数据类型float8_e5m2或float8_e4m3fn，数据格式支持ND。
 *                      GroupMatmul2计算的右矩阵，用于SwiGLU激活后的线性变换。
 * @param [in] weightScales1Optional: 计算可选输入，TensorList，数据类型float8_e8m0，数据格式支持ND。
 *                                    量化场景需要，GroupMatmul1右矩阵反量化参数。
 * @param [in] weightScales2Optional: 计算可选输入，TensorList，数据类型float8_e8m0，数据格式支持ND。
 *                                    量化场景需要，GroupMatmul2右矩阵反量化参数。
 * @param [in] bias1Optional: 计算可选输入，TensorList，数据类型float32，数据格式支持ND。
 *                             GroupMatmul1的偏置参数。
 * @param [in] bias2Optional: 计算可选输入，TensorList，数据类型float32，数据格式支持ND。
 *                             GroupMatmul2的偏置参数。
 * @param [in] xActiveMaskOptional: 计算可选输入，Tensor，数据类型int8，数据格式支持ND。预留参数，暂不支持。
 * @param [in] moeExpertNum: 计算输入，int。MoE模型的总专家数量。
 * @param [in] epWorldSize: 计算输入，int。专家并行通信域大小。
 * @param [in] cclBufferSize: 计算输入，int。CCL通信缓冲区大小。
 * @param [in] maxRecvTokenNum: 计算可选输入，int。每个Rank最大可接收Token数，默认值为0表示自动计算。
 * @param [in] dispatchQuantMode: 计算可选输入，int。dispatch通信时量化模式，目前仅支持4（MXFP模式）。默认值为0。
 * @param [in] dispatchQuantOutDtype: 计算可选输入，int。dispatch量化后输出的数据类型。支持23（FP8_E5M2）或24（FP8_E4M3）。
 * @param [in] combineQuantMode: 计算可选输入，int。预留参数，暂不支持。默认值为0。
 * @param [in] commAlg: 计算可选输入，str。预留参数，暂不支持。默认值为""。
 * @param [in] numMaxTokensPerRank: 计算可选输入，int。每个Rank最大可接收Token数，默认值为0表示所有卡的token数相同且为当前bs（x的dim0）。
 * @param [in] activation: 计算可选输入，str。激活函数类型，支持"swiglu"、"gelu"、"silu"、"relu"等，默认值为"swiglu"。
 * @param [in] activationClamp: 计算可选输入，float。激活函数截断值，默认值为float最大值。
 * @param [out] yOut: 计算输出，Tensor，必选输出，数据类型bfloat16，仅支持2维，数据格式支持ND。
 *                    计算输出结果，与输入x shape相同。
 * @param [out] expertTokenNumsOut: 计算输出，Tensor，必选输出，数据类型int32，仅支持1维，数据格式支持ND。
 *                                  每个专家收到的token数量。
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码
 *
 */
ACLNN_API aclnnStatus aclnnMegaMoeGetWorkspaceSize(
    const aclTensor *context, const aclTensor *x, const aclTensor *topkIds, const aclTensor *topkWeights,
    const aclTensorList *weight1, const aclTensorList *weight2, const aclTensorList *weightScales1Optional,
    const aclTensorList *weightScales2Optional, const aclTensorList *bias1Optional, const aclTensorList *bias2Optional,
    const aclTensor *xActiveMaskOptional, int64_t moeExpertNum, int64_t epWorldSize, int64_t cclBufferSize,
    int64_t maxRecvTokenNum, int64_t dispatchQuantMode, int64_t dispatchQuantOutDtype, int64_t combineQuantMode,
    const char *commAlg, int64_t numMaxTokensPerRank, const char *activation, float activationClamp, aclTensor *yOut,
    aclTensor *expertTokenNumsOut, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnMegaMoe的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口aclnnMegaMoeGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMegaMoe(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                   aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MEGA_MOE_