#ifndef DCUT_RECURRENT_GATED_DELTA_RULE_HOST_H
#define DCUT_RECURRENT_GATED_DELTA_RULE_HOST_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor* DcutRecurrentGatedDeltaRule(const aclTensor* query, const aclTensor* key, const aclTensor* value,
                                             const aclTensor* beta, aclTensor* stateRef,
                                             const aclTensor* actualSeqLengths, const aclTensor* ssmStateIndices,
                                             const aclTensor* g, const aclTensor* gk,
                                             const aclTensor* numAcceptedTokens, float scaleValue,
                                             aclOpExecutor* executor);
}

#endif
