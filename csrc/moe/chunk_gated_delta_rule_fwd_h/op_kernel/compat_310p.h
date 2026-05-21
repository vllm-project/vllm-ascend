#ifndef COMPAT_310P_H
#define COMPAT_310P_H

#ifndef __CCE_KT_TEST__
#include "kernel_operator.h"
#endif

#if !defined(__bfloat16_t_defined)
#define __bfloat16_t_defined
#define __COMPAT_310P_ACTIVE__
struct bfloat16_t {
    uint16_t val;
    bfloat16_t() = default;
    bfloat16_t(float v) : val(0) { (void)v; }
    operator float() const { return 0.f; }
};
#endif

// 310P has no fixpipe unit; post-matmul stores go through MTE3
#ifndef PIPE_FIX
#define PIPE_FIX PIPE_MTE3
#endif

// 310P renames LoadDataWithSparse → LoadDataWithSparseCal
#ifdef __COMPAT_310P_ACTIVE__
#define LoadDataWithSparse LoadDataWithSparseCal
#endif

#endif
