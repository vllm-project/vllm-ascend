/*
 * Build-only compatibility for importing MegaMoe into vLLM Ascend.
 *
 * Upstream mc2_log.h also declares MatMul/QBMM tiling dump helpers. MegaMoe
 * only uses the common logging/checking macros, so including that header would
 * unnecessarily pull the whole mc2/3rd MatMul source tree into this build.
 */
#ifndef VLLM_ASCEND_MEGA_MOE_MC2_LOG_COMPAT_H
#define VLLM_ASCEND_MEGA_MOE_MC2_LOG_COMPAT_H

/* Also satisfy upstream mc2_log.h's guard for this MegaMoe translation unit. */
#ifndef MC2_LOG_H
#define MC2_LOG_H
#endif

#include "mc2_common_log.h"

#ifndef OP_TILING_CHECK
#define OP_TILING_CHECK(cond, log_func, expr)                                                                          \
    do {                                                                                                               \
        if (cond) {                                                                                                    \
            log_func;                                                                                                  \
            expr;                                                                                                      \
        }                                                                                                              \
    } while (0)
#endif

#ifndef VECTOR_INNER_ERR_REPORT_TILING
#define VECTOR_INNER_ERR_REPORT_TILING(op_name, err_msg, ...)                                                          \
    do {                                                                                                               \
        OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__);                                                       \
        REPORT_INNER_ERR_MSG("E89999", "op[%s], " err_msg, op_name, ##__VA_ARGS__);                                  \
    } while (0)
#endif

#endif
