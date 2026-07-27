/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */

#ifndef SOLVE_TRI_ASCEND950_COMMON_H
#define SOLVE_TRI_ASCEND950_COMMON_H

#include "kernel_operator.h"

// 8x16 ND 块对角 mask
// [0] = ODD  (奇数条带, 对角在 col 8..15)
// [1] = EVEN (偶数条带, 对角在 col 0..7)
constexpr uint64_t DIAG_MASK_8X16[2][2] = {
    { 0x0800040002000100ULL, 0x8000400020001000ULL },  // ODD
    { 0x0008000400020001ULL, 0x0080004000200010ULL }   // EVEN
};

#endif  // SOLVE_TRI_ASCEND950_COMMON_H
