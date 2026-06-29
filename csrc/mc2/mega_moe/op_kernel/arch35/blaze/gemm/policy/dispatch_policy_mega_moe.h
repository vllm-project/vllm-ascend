/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dispatch_policy_mega_moe.h
 * \brief Dispatch policy for the MX FP8/FP4 dynamic KL1 tail-resplit path.
 *        Extracted from blaze/gemm/policy/dispatch_policy.h for mega_moe.
 */
#pragma once

#include "../utils/common_utils_mega_moe.h"

namespace Blaze {
namespace Gemm {

/**
 * @struct MatmulMxFp8Fp4DynamicKL1TailResplit
 * @brief Dispatch policy for the MX FP8/FP4 dynamic KL1 tail-resplit path.
 */
struct MatmulMxFp8Fp4DynamicKL1TailResplit {};

} // namespace Gemm
} // namespace Blaze
