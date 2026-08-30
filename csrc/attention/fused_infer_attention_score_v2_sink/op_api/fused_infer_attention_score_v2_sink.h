/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_OP_FUSED_INFER_ATTENTION_SCORE_V2_SINK_OP_H_
#define OP_API_INC_LEVEL0_OP_FUSED_INFER_ATTENTION_SCORE_V2_SINK_OP_H_

#include "opdev/op_executor.h"

namespace FusedInferAttentionScoreV2Sinkl0op {
struct FusedInferAttentionScoreV2SinkOutputs {
    const aclTensor *AttentionOutOut;
    const aclTensor *SoftmaxOutOut;
    const aclTensor *SoftmaxMaxOut;
    const aclTensor *SoftmaxSumOut;
};

const FusedInferAttentionScoreV2SinkOutputs FusedInferAttentionScoreV2Sink(
    const aclTensor *query,
    const aclTensorList *key,
    const aclTensorList *value,
    const aclTensor *pseShift,
    const aclTensor *attenMask,
    const aclTensor *actualSeqLengths,
    const aclTensor *actualSeqLengthsKv,
    const aclTensor *deqScale1,
    const aclTensor *quantScale1,
    const aclTensor *deqScale2,
    const aclTensor *quantScale2,
    const aclTensor *quantOffset2,
    const aclTensor *antiquantScale,
    const aclTensor *antiquantOffset,
    const aclTensor *blockTable,
    const aclTensor *queryPaddingSize,
    const aclTensor *kvPaddingSize,
    const aclTensor *keyAntiquantScale,
    const aclTensor *keyAntiquantOffset,
    const aclTensor *valueAntiquantScale,
    const aclTensor *valueAntiquantOffset,
    const aclTensor *queryRope,
    const aclTensor *keyRope,
    const aclTensor *keyRopeAntiquantScale,
    const aclTensor *dequantScaleQuery,
    const aclTensor *metaData,
    const aclTensor *learnableSink,
    const aclTensor *keySinkOptional,
    const aclTensor *keyRopeSinkOptional,
    const aclTensor *valueSinkOptional,
    int64_t numHeads,
    double scaleValue,
    int64_t preTokens,
    int64_t nextTokens,
    const char *inputLayout,
    int64_t numKeyValueHeads,
    int64_t sparseMode,
    int64_t innerPrecise,
    int64_t blockSize,
    int64_t antiquantMode,
    bool softmaxLseFlag,
    int64_t keyAntiquantMode,
    int64_t valueAntiquantMode,
    int64_t queryQuantMode,
    int64_t pseType,
    int64_t sinkNumber,
    bool batchInvariant,
    bool softmaxMaxSumFlag,
    int64_t outType,
    const aclTensor *attentionOut,
    const aclTensor *softmaxLse,
    const aclTensor *softmaxMax,
    const aclTensor *softmaxSum,
    aclOpExecutor *executor);
}
#endif /* OP_API_INC_LEVEL0_OP_FUSED_INFER_ATTENTION_SCORE_V2_SINK_OP_H_ */