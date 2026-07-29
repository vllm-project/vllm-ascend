/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_RECURRENT_KDA
#define PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_RECURRENT_KDA

#include "opdev/make_op_executor.h"
#include "opdev/op_executor.h"
#include <array>

namespace l0op {
const std::array<const aclTensor *, 2> RecurrentKda(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *gate,
    const aclTensor *beta,
    aclTensor *stateRef,
    const aclTensor *cuSeqlens,
    const aclTensor *ssmStateIndicesOptional,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *numAcceptedTokensOptional,
    const char *layout,
    double scale,
    bool useQkL2normInKernel,
    bool useGateInKernel,
    bool useBetaSigmoidInKernel,
    bool allowNegEigval,
    bool safeGate,
    double lowerBound,
    bool stateVFirst,
    const aclTensor *out,
    aclOpExecutor *executor);
}

#endif // PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_RECURRENT_KDA
