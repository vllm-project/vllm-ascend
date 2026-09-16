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
 * \file aclnn_mixed_quant_sparse_flash_mla.h
 * \brief MixedQuantSparseFlashMla aclnn level-2 interface
 */

#ifndef OP_API_INC_LEVEL2_ACLNN_MIXED_QUANT_SPARSE_FLASH_MLA_H_
#define OP_API_INC_LEVEL2_ACLNN_MIXED_QUANT_SPARSE_FLASH_MLA_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief First phase of aclnnMixedQuantSparseFlashMla: calculate workspace size.
 * @domain aclnn_ops_train_infer
 *
 * @param q                         [IN]  query tensor, BF16.
 * @param oriKvOptional             [IN]  optional original KV tensor, FLOAT8_E4M3FN.
 * @param cmpKvOptional             [IN]  optional compressed KV tensor, FLOAT8_E4M3FN.
 * @param oriSparseIndicesOptional  [IN]  optional original KV sparse indices, INT32.
 * @param cmpSparseIndicesOptional  [IN]  optional compressed KV sparse indices, INT32.
 * @param oriBlockTableOptional     [IN]  optional original KV block table, INT32.
 * @param cmpBlockTableOptional     [IN]  optional compressed KV block table, INT32.
 * @param cuSeqlensQOptional        [IN]  optional query cumulative sequence lengths, INT32.
 * @param cuSeqlensOriKvOptional    [IN]  optional original KV cumulative sequence lengths, INT32.
 * @param cuSeqlensCmpKvOptional    [IN]  optional compressed KV cumulative sequence lengths, INT32.
 * @param sequsedQOptional          [IN]  optional actual query sequence lengths, INT32.
 * @param sequsedOriKvOptional      [IN]  optional actual original KV sequence lengths, INT32.
 * @param sequsedCmpKvOptional      [IN]  optional actual compressed KV sequence lengths, INT32.
 * @param cmpResidualKvOptional     [IN]  optional compressed KV residual lengths, INT32.
 * @param oriTopkLengthOptional     [IN]  optional original KV top-k lengths, INT32.
 * @param cmpTopkLengthOptional     [IN]  optional compressed KV top-k lengths, INT32.
 * @param sinksOptional             [IN]  optional attention sink weights, FLOAT32.
 * @param metadataOptional          [IN]  optional pre-computed tiling metadata, INT32.
 * @param quantMode                 [IN]  ATTR. Quantization mode.
 * @param ropeHeadDim               [IN]  ATTR. RoPE head dimension.
 * @param softmaxScale              [IN]  ATTR. Softmax scaling factor.
 * @param cmpRatio                  [IN]  ATTR. Compressed KV ratio.
 * @param oriMaskMode               [IN]  ATTR. Original KV mask mode.
 * @param cmpMaskMode               [IN]  ATTR. Compressed KV mask mode.
 * @param oriWinLeft                [IN]  ATTR. Original KV left window size.
 * @param oriWinRight               [IN]  ATTR. Original KV right window size.
 * @param layoutQOptional           [IN]  ATTR. Query layout.
 * @param layoutKvOptional          [IN]  ATTR. KV layout.
 * @param topkValueMode             [IN]  ATTR. Top-k value mode.
 * @param returnSoftmaxLse          [IN]  ATTR. Whether to output softmax LSE.
 * @param attnOut                   [OUT] Required attention output, BF16.
 * @param softmaxLseOptional  [OUT] Optional output. Softmax log-sum-exp, FLOAT32.
 *                                  Valid when returnSoftmaxLse=True.
 * @param workspaceSize       [OUT] Workspace size in bytes.
 * @param executor            [OUT] Op executor handle for second phase.
 * @return aclnnStatus. ACLNN_SUCCESS on success.
 */
aclnnStatus aclnnMixedQuantSparseFlashMlaGetWorkspaceSize(
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
    const aclTensor *softmaxLseOptional, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief Second phase of aclnnMixedQuantSparseFlashMla: execute computation.
 * @param workspace       [IN] Workspace device memory pointer from first phase.
 * @param workspaceSize   [IN] Workspace size in bytes.
 * @param executor        [IN] Op executor handle from first phase.
 * @param stream          [IN] ACL stream for execution.
 * @return aclnnStatus.
 */
aclnnStatus aclnnMixedQuantSparseFlashMla(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_LEVEL2_ACLNN_MIXED_QUANT_SPARSE_FLASH_MLA_H_
