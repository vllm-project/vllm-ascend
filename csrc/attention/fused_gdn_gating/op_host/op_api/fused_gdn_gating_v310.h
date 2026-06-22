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
