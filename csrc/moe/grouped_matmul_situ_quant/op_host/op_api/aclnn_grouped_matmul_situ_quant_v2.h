/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_OP_API_ACLNN_GROUPED_MATMUL_SITU_QUANT_V2_H
#define OP_HOST_OP_API_ACLNN_GROUPED_MATMUL_SITU_QUANT_V2_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupedMatmulSituQuantV2 的第一段接口，计算 workspace 大小。
 *
 * 入参与 op 原型顺序一致（与 aclnnGroupedMatmulSituQuant 完全相同），唯一的区别是
 * weight / weightScale 的 int32 / int64 打包形态：
 *   - A2/A3 的 A8W4 MSD 权重在 torch 侧的自然表示是 int32（1 个 int32 承载 8 个 int4，
 *     与 grouped_matmul_swiglu_quant_v2 等算子一致）。op 原型按 INT4 声明 weight，
 *     因此本入口先把它改写成逻辑 INT4 描述符（view 末维 x8、dtype=INT4，NZ 排布同时
 *     改写 storage 末维），weightScale 的 int64 同理改写为 uint64，再交给 op 原型
 *     自动生成的入口完成 tiling 与下发。
 *   - Ascend950（arch35，x=FP8、weight=FP4）不经过本入口，仍走
 *     aclnnGroupedMatmulSituQuant；本入口对非 INT32 / INT64 的入参原样透传。
 *
 * @param [in] x: 左矩阵，A2/A3 为 INT8。
 * @param [in] xScale: 激活量化参数，A2/A3 为 per-token FLOAT32。
 * @param [in] weight: 右矩阵，A2/A3 为 INT4（允许传 int32 打包形态）。
 * @param [in] weightScale: 权重量化参数，A2/A3 为 UINT64（允许传 int64 打包形态）。
 * @param [in] groupList: 分组信息，INT64。
 * @param [in] weightAssistMatrix: A8W4 MSD 的离线补偿矩阵，FLOAT32。
 * @param [in] bias: 偏移，暂不支持，必须为空。
 * @param [in] smoothScale: 平滑因子，暂不支持，必须为空。
 * @param [in] groupListType: 分组的解释方式，0 为累加和，1 为每组行数。
 * @param [in] beta: SiTU 的 beta 参数，需大于 0。
 * @param [in] linearBeta: SiTU 的 up 分支线性缩放参数，<= 0 表示不做变换。
 * @param [in] dequantMode: 反量化计算类型，A2/A3 为 0（per-channel）。
 * @param [in] dequantDtype: 中间 GEMM 结果数据类型，0 为 BF16。
 * @param [in] quantMode: 量化计算类型，A2/A3 为 0（per-token）。
 * @param [in] quantDtype: 量化输出数据类型，0 为默认（跟 y 一致）。
 * @param [in] transposeWeight: 权重是否转置，A2/A3 只支持 false。
 * @param [in] tuningConfigOptional: 用于算子预估 m/e 大小，可为空。
 * @param [in] activateLeft: true 表示 gate 取前半区、up 取后半区。
 * @param [out] output: 量化输出，A2/A3 为 INT8，shape 为 [M, N / 2]。
 * @param [out] outputScale: 量化 scale，A2/A3 为 FLOAT32，shape 为 [M]。
 * @param [out] workspaceSize: 返回用户需要在 npu device 侧申请的 workspace 大小。
 * @param [out] executor: 返回 op 执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulSituQuantV2GetWorkspaceSize(
    const aclTensor *x, const aclTensor *xScale, const aclTensorList *weight, const aclTensorList *weightScale,
    const aclTensor *groupList, const aclTensorList *weightAssistMatrix, const aclTensor *bias,
    const aclTensor *smoothScale, int64_t groupListType, double beta, double linearBeta, int64_t dequantMode,
    int64_t dequantDtype, int64_t quantMode, int64_t quantDtype, bool transposeWeight,
    const aclIntArray *tuningConfigOptional, bool activateLeft, aclTensor *output, aclTensor *outputScale,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnGroupedMatmulSituQuantV2 的第二段接口，用于执行计算。
 *
 * @param [in] workspace: 在 npu device 侧申请的 workspace 内存起址。
 * @param [in] workspaceSize: 在 npu device 侧申请的 workspace 大小，由第一段接口获取。
 * @param [in] executor: op 执行器，包含了算子计算流程。
 * @param [in] stream: acl stream 流。
 * @return aclnnStatus: 返回状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnGroupedMatmulSituQuantV2(void *workspace, uint64_t workspaceSize,
                                                                                 aclOpExecutor *executor,
                                                                                 aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_HOST_OP_API_ACLNN_GROUPED_MATMUL_SITU_QUANT_V2_H
