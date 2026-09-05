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
 * \file ffn.h
 * \brief
 */

#ifndef OP_API_INC_LEVEL0_OP_FFN_OP_H
#define OP_API_INC_LEVEL0_OP_FFN_OP_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *FFN(const aclTensor *x, const aclTensor *weight1, const aclTensor *weight2,
                     const aclTensor *expertTokensOptional, const aclTensor *bias1Optional,
                     const aclTensor *bias2Optional, const aclTensor *scaleOptional, const aclTensor *offsetOptional,
                     const aclTensor *deqScale1Optional, const aclTensor *deqScale2Optional,
                     const aclTensor *antiquantScale1Optional, const aclTensor *antiquantScale2Optional,
                     const aclTensor *antiquantOffset1Optional, const aclTensor *antiquantOffset2Optional,
                     const char *activation, int64_t innerPrecise, const op::DataType yDtype, bool tokensIndexFlag,
                     aclOpExecutor *executor);
}

#endif