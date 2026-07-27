#ifndef OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_
#define OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_

#include <cstdio>
#include <string>

#include "tiling_base/tiling_util.h"
#include "toolchain/slog.h"

#define OP_LOGI(opname, ...)
#define OP_LOGD(opname, ...)

#define OP_LOGW(opname, ...)       \
    do {                           \
        (void)(opname);            \
        std::printf("[WARN] ");    \
        std::printf(__VA_ARGS__);  \
        std::printf("\n");        \
    } while (0)

#define OP_LOGE_WITHOUT_REPORT(opname, ...) \
    do {                                    \
        (void)(opname);                     \
        std::printf("[ERRORx] ");           \
        std::printf(__VA_ARGS__);           \
        std::printf("\n");                 \
    } while (0)

#define OP_LOGE(opname, ...)       \
    do {                           \
        (void)(opname);            \
        std::printf("[ERROR] ");   \
        std::printf(__VA_ARGS__);  \
        std::printf("\n");        \
    } while (0)

namespace optiling {

#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...)   \
    do {                                                         \
        OP_LOGE_WITHOUT_REPORT(op_name, err_msg, ##__VA_ARGS__); \
    } while (0)

#define OP_CHECK_IF(cond, log_func, expr) \
    do {                                  \
        if (cond) {                       \
            log_func;                     \
            expr;                         \
        }                                 \
    } while (0)

#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)

#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr)                 \
    do {                                                         \
        if ((ptr) == nullptr) {                                  \
            OP_LOGE(context->GetNodeType(), "%s is null", #ptr); \
            return ge::GRAPH_FAILED;                             \
        }                                                        \
    } while (0)

} // namespace optiling

using Ops::Transformer::CeilAlign;
using Ops::Transformer::CeilDiv;

#endif // OPS_BUILT_IN_OP_TILING_ERROR_LOG_H_
