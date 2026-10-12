#ifndef COMPAT_310P_H
#define COMPAT_310P_H

#ifndef __CCE_KT_TEST__
#include "kernel_operator.h"
#endif

// Bisheng CCE provides bfloat16_t on 310P; legacy compilers need a stub.
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200)
#define __COMPAT_310P_ACTIVE__
#if !defined(__BISHENG_CCEC__) && !defined(__bfloat16_t_defined)
#define __bfloat16_t_defined
struct bfloat16_t {
    uint16_t val;
    bfloat16_t() = default;
    bfloat16_t(float v) : val(0) { (void)v; }
    operator float() const { return 0.f; }
};
#endif
#endif

// 310P has no fixpipe unit; post-matmul stores go through MTE3
#ifndef PIPE_FIX
#define PIPE_FIX PIPE_MTE3
#endif

// 310P renames LoadDataWithSparse → LoadDataWithSparseCal
#ifdef __COMPAT_310P_ACTIVE__
#define LoadDataWithSparse LoadDataWithSparseCal
#endif

// Keep the 310P scalar conversion shim for both native and legacy bf16 types.
#ifdef __COMPAT_310P_ACTIVE__
namespace AscendC {
    inline float ToFloat(bfloat16_t v) { return (float)v; }
}
#endif

#endif
