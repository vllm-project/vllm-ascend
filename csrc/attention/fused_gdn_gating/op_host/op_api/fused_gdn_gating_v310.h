/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PTA_NPU_OP_API_FUSED_GDN_GATING_V310_H
#define PTA_NPU_OP_API_FUSED_GDN_GATING_V310_H

#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

namespace l0op {

struct FusedGdnGatingV310Output {
    const aclTensor *g;
    const aclTensor *beta_output;
};

// 声明底层发射器函数签名
FusedGdnGatingV310Output FusedGdnGatingV310(const aclTensor *aLog, const aclTensor *a,
                                            const aclTensor *b, const aclTensor *dtBias,
                                            float beta, float threshold,
                                            aclOpExecutor *executor);

} // namespace l0op

#endif // PTA_NPU_OP_API_FUSED_GDN_GATING_V310_H
