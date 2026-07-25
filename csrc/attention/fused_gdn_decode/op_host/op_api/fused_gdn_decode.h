/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef FUSED_GDN_DECODE_L0_OP_API_H
#define FUSED_GDN_DECODE_L0_OP_API_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *FusedGdnDecode(const aclTensor *mixedQkv, const aclTensor *a, const aclTensor *b,
                                const aclTensor *aLog, const aclTensor *dtBias, aclTensor *stateRef,
                                const aclTensor *ssmStateIndices, float scale, float softplusThreshold,
                                aclOpExecutor *executor);
}

#endif // FUSED_GDN_DECODE_L0_OP_API_H
