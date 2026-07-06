#pragma once

#include <stdint.h>

// B080 op_common/log/log.h stopped exposing the unqualified OP module id used
// by inherited ops-transformer tiling/error headers. Include the CANN log type
// header early so OP still comes from the active CANN version.
#if defined(__has_include)
#if __has_include("base/log_types.h")
#include "base/log_types.h"
#elif __has_include("toolchain/log_types.h")
#include "toolchain/log_types.h"
#endif
#endif

#if !defined(LOG_TYPES_H_) && !defined(OP)
#define OP 63
#endif

#if defined(LOG_CPP) && !defined(DLOG_PUB_H_)
#ifdef __cplusplus
extern "C" {
#endif
int32_t CheckLogLevel(int32_t moduleId, int32_t logLevel);
void DlogRecord(int32_t moduleId, int32_t level, const char *fmt, ...);
#ifdef __cplusplus
}
#endif
#define DLOG_PUB_H_
#endif
