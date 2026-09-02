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
 * \file aclnn_ffn.h
 * \brief
 */

#ifndef OP_API_INC_FFN_H
#define OP_API_INC_FFN_H
#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFFN的第一段接口，根据具体的计算流程，计算workspace大小。
 * 功能描述：该FFN算子提供MoeFFN和FFN的计算功能
 * 计算公式：y=activation(xW1+b1)W2+b2
 * @domain aclnn_ops_infer
 * @param [in]
 * x：必选参数，Device侧的aclTensor，公式中的输入x，数据类型支持FLOAT16、BFLOAT16、INT8，数据格式支持ND，支持输入的维度最少是2维[M,
 * K1]，最多是8维。
 * @param [in]
 * weight1：必选参数，Device侧的aclTensor，专家的权重数据，公式中的W1，数据类型支持FLOAT16、BFLOAT16、INT8、INT4，数据格式支持ND，输入在有/无专家时分别为[E,
 * K1, N1]/[K1, N1]。
 * @param [in]
 * weight2：必选参数，Device侧的aclTensor，专家的权重数据，公式中的W2，数据类型支持FLOAT16、BFLOAT16、INT8、INT4，数据格式支持ND，输入在有/无专家时分别为[E,
 * K2, N2]/[K2, N2]。
 * @param [in]
 * expertTokens：可选参数，Host侧的aclIntArray类型，代表各专家的token数，数据类型支持INT64，数据格式支持ND，若不为空时可支持的最大长度为256个。
 * @param [in]
 * bias1：可选参数，Device侧的aclTensor，权重数据修正值，公式中的b1，数据类型支持FLOAT16、FLOAT32、INT32，数据格式支持ND，输入在有/无专家时分别为[E,
 * N1]/[N1]。
 * @param [in]
 * bias2：可选参数，Device侧的aclTensor，权重数据修正值，公式中的b2，数据类型支持FLOAT16、FLOAT32、INT32，数据格式支持ND，输入在有/无专家时分别为[E,
 * N2]/[N2]。
 * @param [in]
 * scale：可选参数，Device侧的aclTensor，量化参数，量化缩放系数，数据类型支持FLOAT32，数据格式支持ND，per-tensor下输入在有/无专家时均为一维向量，输入元素个数在有/无专家时分别为[E]/[1]；per-channel下输入在有/无专家时为二维向量/一维向量，输入元素个数在有/无专家时分别为[E,
 * N1]/[N1]。
 * @param [in]
 * offset：可选参数，Device侧的aclTensor，量化参数，量化偏移量，数据类型支持FLOAT32，数据格式支持ND，一维向量，输入元素个数在有/无专家时分别为[E]/[1]。
 * @param [in]
 * deqScale1：可选参数，Device侧的aclTensor，量化参数，第一个matmul的反量化缩放系数，数据类型支持UINT64、INT64、FLOAT32、BFLOAT16，数据格式支持ND，输入在有/无专家时分别为[E,
 * N1]/[N1]。
 * @param [in]
 * deqScale2：可选参数，Device侧的aclTensor，量化参数，第二个matmul的反量化缩放系数，数据类型支持UINT64、INT64、FLOAT32、BFLOAT16，数据格式支持ND，输入在有/无专家时分别为[E,
 * N2]/[N2]。
 * @param [in]
 * antiquantScale1：可选参数，Device侧的aclTensor，伪量化参数，第一个matmul的缩放系数，数据类型支持FLOAT16、BFLOAT16，数据格式支持ND，per-channel下输入在有/无专家时分别为[E,
 * N1]/[N1]，per-in-group下输入在有/无专家时分别为[E, G, N1]/[G, N1]。
 * @param [in]
 * antiquantScale2：可选参数，Device侧的aclTensor，伪量化参数，第二个matmul的缩放系数，数据类型支持FLOAT16、BFLOAT16，数据格式支持ND，per-channel下输入在有/无专家时分别为[E,
 * N2]/[N2]，per-in-group下输入在有/无专家时分别为[E, G, N2]/[G, N2]。
 * @param [in]
 * antiquantOffset1：可选参数，Device侧的aclTensor，伪量化参数，第一个matmul的偏移量，数据类型支持FLOAT16、BFLOAT16，数据格式支持ND，per-channel下输入在有/无专家时分别为[E,
 * N1]/[N1]，per-in-group下输入在有/无专家时分别为[E, G, N1]/[G, N1]。
 * @param [in]
 * antiquantOffset2：可选参数，Device侧的aclTensor，伪量化参数，第二个matmul的偏移量，数据类型支持FLOAT16、BFLOAT16，数据格式支持ND，per-channel下输入在有/无专家时分别为[E,
 * N2]/[N2]，per-in-group下输入在有/无专家时分别为[E, G, N2]/[G, N2]。
 * @param [in]
 * activation：必选参数，Host侧的属性值，代表使用的激活函数，公式中的activation，当前支持fastgelu/gelu/relu/silu以及geglu/swiglu/reglu。
 * @param [in]
 * innerPrecise：可选参数，Host侧的int，表示高精度或者高性能选择。数据类型支持：INT64。该参数仅对FLOAT16生效，BFLOAT16和INT8不区分高精度和高性能。
 * @param [out] y：输出Tensor，公式中的输出y，数据类型支持FLOAT16、BFLOAT16，数据格式支持ND，输出维度与x一致。
 * @param [out] workspaceSize：返回用户需要在Device侧申请的workspace大小。
 * @param [out] executor：返回op执行器，包含了算子计算流程。
 * @return      aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus
aclnnFFNGetWorkspaceSize(const aclTensor *x, const aclTensor *weight1, const aclTensor *weight2,
                         const aclIntArray *expertTokens, const aclTensor *bias1, const aclTensor *bias2,
                         const aclTensor *scale, const aclTensor *offset, const aclTensor *deqScale1,
                         const aclTensor *deqScale2, const aclTensor *antiquantScale1, const aclTensor *antiquantScale2,
                         const aclTensor *antiquantOffset1, const aclTensor *antiquantOffset2, const char *activation,
                         int64_t innerPrecise, const aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnFFN的第二段接口，用于执行计算。
 * @param [in] workspace: 在Device侧申请的workspace内存起址。
 * @param [in] workspaceSize: 在Device侧申请的workspace大小，由第一段接口aclnnFFNGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: 指定执行任务的AscendCL stream流。
 * @return     aclnnStatus: 返回状态码
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFFN(void *workspace, uint64_t workspaceSize,
                                                            aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_FFN_H