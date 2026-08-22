/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL0_OP_PROMPT_FLASH_ATTENTION_OP_H_
#define OP_API_INC_LEVEL0_OP_PROMPT_FLASH_ATTENTION_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *PromptFlashAttention(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *pseShift,
    const aclTensor *attenMask, // attenMask of pfa
    const aclIntArray *actualSeqLengths,
    const aclIntArray *actualSeqLengthsKv,
    const aclTensor *deqScale1,
    const aclTensor *quantScale1,
    const aclTensor *deqScale2,
    const aclTensor *quantScale2,
    const aclTensor *quantOffset2,
    int64_t numHeads, // q_n
    double scaleValue,
    int64_t preTokens,
    int64_t nextTokens,
    const char *inputLayout,
    int64_t numKeyValueHeads, // kv_n
    int64_t sparseMode,
    int64_t innerPrecise,
    const aclTensor *attentionOut,
    aclOpExecutor *executor);
}

#endif /* OP_API_INC_LEVEL0_OP_PROMPT_FLASH_ATTENTION_OP_H_ */