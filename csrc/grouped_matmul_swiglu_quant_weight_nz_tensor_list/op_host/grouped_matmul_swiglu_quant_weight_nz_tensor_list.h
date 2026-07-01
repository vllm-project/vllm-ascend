/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_LEVEL0_OP_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_TENSOR_LIST_OP_H
#define OP_API_INC_LEVEL0_OP_GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_TENSOR_LIST_OP_H

#include "opdev/op_executor.h"

namespace l0op {
const std::tuple<aclTensor*, aclTensor*> GroupedMatmulSwigluQuantWeightNzTensorList(const aclTensor *x,
                                                                  const aclTensorList *weight,
                                                                  const aclTensorList *perChannelScale,
                                                                  const aclTensor *perTokenScale,
                                                                  const aclTensor *groupList,
                                                                  aclOpExecutor *executor);
}

#endif
