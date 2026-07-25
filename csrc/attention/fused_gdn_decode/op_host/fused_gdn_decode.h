/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_FUSED_GDN_DECODE
#define PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_FUSED_GDN_DECODE

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {
const aclTensor *FusedGdnDecode(const aclTensor *mixedQkv, const aclTensor *a, const aclTensor *b,
                                const aclTensor *aLog, const aclTensor *dtBias, aclTensor *stateRef,
                                const aclTensor *ssmStateIndices, float scale, float softplusThreshold,
                                aclOpExecutor *executor);
}

#endif // PTA_NPU_OP_API_COMMON_INC_LEVEL0_OP_FUSED_GDN_DECODE
