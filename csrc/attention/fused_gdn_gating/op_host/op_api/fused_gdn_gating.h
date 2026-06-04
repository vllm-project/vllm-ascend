/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

#ifndef PTA_NPU_OP_API_FUSED_GDN_GATING_H
#define PTA_NPU_OP_API_FUSED_GDN_GATING_H

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {

struct FusedGdnGatingOutput {
    const aclTensor *g;
    const aclTensor *beta_output;
};

FusedGdnGatingOutput FusedGdnGating(const aclTensor *aLog, const aclTensor *a,
                                    const aclTensor *b, const aclTensor *dtBias,
                                    float beta,
                                    aclOpExecutor *executor);

} // namespace l0op

#endif // PTA_NPU_OP_API_FUSED_GDN_GATING_H
