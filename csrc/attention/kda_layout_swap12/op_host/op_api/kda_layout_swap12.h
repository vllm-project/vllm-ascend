/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#pragma once

#include "aclnn/aclnn_base.h"
#include <array>

namespace l0op {
const std::array<const aclTensor *, 1> KdaLayoutSwap12(
    const aclTensor *x,
    const aclTensor *dependency,
    const aclTensor *y,
    aclOpExecutor *executor);

const std::array<const aclTensor *, 1> KdaLayoutSwap12(
    const aclTensor *x,
    const aclTensor *y,
    aclOpExecutor *executor);
} // namespace l0op
