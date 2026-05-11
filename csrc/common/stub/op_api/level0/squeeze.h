/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_LEVEL0_SQUEEZE_ND_H
#define OP_API_INC_LEVEL0_SQUEEZE_ND_H

# include "opdev/op_def.h"

namespace l0op {

const aclTensor *SqueezeNd(const aclTensor *x, const aclIntArray* dim, aclOpExecutor *executor);

const aclTensor *SqueezeNd(const aclTensor *x, int64_t dim, aclOpExecutor *executor);

} // l0op

#endif  // OP_API_INC_LEVEL0_SQUEEZE_ND_H