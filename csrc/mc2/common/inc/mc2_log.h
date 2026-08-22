/*
 * The open CANN 9.1 package provides the MC2 logging macros through
 * op_common/log/log.h, but does not ship the source-tree-only mc2_log.h.
 * Keep the fallback host sources buildable without importing the unrelated
 * MC2 implementation.
 */
#ifndef VLLM_ASCEND_MC2_LOG_H_
#define VLLM_ASCEND_MC2_LOG_H_

#include "log/log.h"

#ifndef OPS_ERR_IF
#define OPS_ERR_IF(condition, log_func, expression) \
    if (condition) {                              \
        log_func;                                 \
        expression;                               \
    }
#endif

#endif  // VLLM_ASCEND_MC2_LOG_H_
