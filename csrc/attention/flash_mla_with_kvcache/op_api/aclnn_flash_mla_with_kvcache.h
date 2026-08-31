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
 * \file aclnn_flash_mla_with_kvcache.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_FLASH_MLA_WITH_KVCACHE_H_
#define OP_API_INC_LEVEL2_ACLNN_FLASH_MLA_WITH_KVCACHE_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFlashMlaWithKvcache的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_train_infer
 *
 * @param q                   [IN]  query tensor。数据类型FLOAT16/BFLOAT16，ND格式。
 *                                  layout由layoutQ决定，支持TND。
 *                                  TND: shape=(T_q, N_q, D)，T_q为总token数，
 *                                  D=576=nope(512)+rope(64)，rope已合并在q中。
 * @param kCache              [IN]  key-cache tensor。数据类型与q一致，layout由layoutKv决定。
 *                                  PA_NZ: shape=(NumBlocks, N_kv, BlockSize/16, D, 16)，
 *                                  分页注意力PA_NZ格式，D=576=nope(512)+rope(64)，rope已合并在k_cache中。
 *                                  MLA场景固定N_kv=1（单latent KV头）。
 * @param blockTableOptional  [IN]  可选。分页KV缓存的块映射表，数据类型INT32。
 *                                  shape=(B, MaxBlocksPerSeq)，与layoutKv=PA_Nz配合使用。
 * @param cacheSeqlensOptional [IN] 可选。各batch的KV cache实际长度，数据类型INT32。
 *                                  shape=(B,)，表征每个batch的KV序列长度（seqused_kv语义）。
 * @param cuSeqlensQOptional  [IN]  可选。query的累积序列长度（前缀和），数据类型INT32。
 *                                  shape=(B+1,)，cu_seqlens_q[i+1]-cu_seqlens_q[i]为第i个sample的q序列长度。
 *                                  TND layout变长场景时有效。与seqused_q互斥。
 * @param sequsedQOptional    [IN]  可选。各batch中query的实际序列长度，数据类型INT32。
 *                                  shape=(B,)，padded batch模式下有效。与cuSeqlensQOptional互斥。
 * @param attnMaskOptional    [IN]  可选。attnmask的参数，数据类型INT8。
 * @param metadataOptional    [IN]  可选。预计算的tiling切分方案，数据类型INT32，由上游算子传入。
 *                                  当该输入不为nullptr时，tiling侧跳过切分方案计算，直接使用该元数据。
 * @param headDimV            [IN]  ATTR可选。k_cache中nope（value）段的宽度。数据类型INT，默认值512。
 *                                  q/k_cache最后维=headDimV+64（rope宽）。
 * @param softmaxScale        [IN]  ATTR可选。softmax缩放系数，对应老接口的scaleValue。
 *                                  数据类型DOUBLE。默认值0.0表示使用1/sqrt(D)。
 * @param maskMode            [IN]  ATTR可选。掩码模式，对应老接口的sparseMode。数据类型INT。
 *                                  0: 不使用掩码；1: 因果掩码（下三角）。
 * @param maxSeqlenQ          [IN]  ATTR可选。query的最大序列长度。数据类型INT，默认值-1。
 * @param maxSeqlenKV         [IN]  ATTR可选。kv的最大序列长度。数据类型INT，默认值-1。
 * @param layoutQ             [IN]  ATTR可选。query的数据布局，支持"TND"（大小写不敏感）。
 * @param layoutKv            [IN]  ATTR可选。key/value的数据布局，支持"PA_NZ"。
 * @param layoutOut           [IN]  ATTR可选。输出的数据布局，支持"NTD"（大小写不敏感）。
 * @param returnSoftmaxLse    [IN]  ATTR可选。是否输出softmax_lse。INT类型，1表示输出，0表示不输出。
 *                                  训练正向传播时置1，推理时置0。
 * @param attnOut        [OUT] 必选输出。attention计算结果，数据类型与q一致，layout由layoutOut决定。
 * @param softmaxLseOptional  [OUT] 可选输出。softmax的log-sum-exp值，FLOAT32类型。
 *                                  returnSoftmaxLse=1时有效。
 * @param workspaceSize       [OUT] workspace大小（字节数）。
 * @param executor            [OUT] op执行器句柄，供第二段接口使用。
 * @return aclnnStatus 执行状态。ACLNN_SUCCESS表示成功。
 */
aclnnStatus aclnnFlashMlaWithKvcacheGetWorkspaceSize(
    const aclTensor *q, const aclTensor *kCache, const aclTensor *blockTableOptional,
    const aclTensor *cacheSeqlensOptional, const aclTensor *cuSeqlensQOptional, const aclTensor *sequsedQOptional,
    const aclTensor *attnMaskOptional, const aclTensor *metadataOptional, int64_t headDimV, double softmaxScale,
    int64_t maskMode, int64_t maxSeqlenQ, int64_t maxSeqlenKV, const char *layoutQ, const char *layoutKv,
    const char *layoutOut, int64_t returnSoftmaxLse, const aclTensor *attnOut, const aclTensor *softmaxLseOptional,
    uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief aclnnFlashMlaWithKvcache的第二段接口，用于执行计算。
 * @param workspace       [IN] 由第一段接口计算得到的workspace设备内存指针。
 * @param workspaceSize   [IN] workspace大小（字节数）。
 * @param executor        [IN] 第一段接口输出的op执行器句柄。
 * @param stream          [IN] 用于执行计算的acl stream。
 * @return aclnnStatus 执行状态。
 */
aclnnStatus aclnnFlashMlaWithKvcache(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_FLASH_MLA_WITH_KVCACHE_H_
