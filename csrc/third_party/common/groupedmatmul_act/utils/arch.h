/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file arch.h
 * \brief
 */

#ifndef UTILS_ARCH_H
#define UTILS_ARCH_H

namespace Act {
namespace Gemm {
namespace Arch {
struct Ascend910B {};
struct Ascend910_95 {};
} // namespace Arch

// buffer size
constexpr static int64_t L0A_SIZE = 64 * 1024;
constexpr static int64_t L0B_SIZE = 64 * 1024;

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101) // for Ascend910_95
constexpr static int64_t L0C_SIZE = 256 * 1024;
#else
constexpr static int64_t L0C_SIZE = 128 * 1024;
#endif

constexpr static int64_t L1_SIZE = 512 * 1024;
} // namespace Gemm
} // namespace Act
#endif