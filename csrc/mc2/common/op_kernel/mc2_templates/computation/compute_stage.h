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
 * \file compute_stage.h
 * \brief
 */

#ifndef MC2_COMPUTE_STAGE_H
#define MC2_COMPUTE_STAGE_H

#if ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_FLOAT16) || (ORIG_DTYPE_X1 == DT_BF16)))
#include "./matmul/mc2_fp_matmul.h"
#else
#include "./matmul/mc2_kc_quant_matmul.h"
#include "./matmul/mc2_mx_quant_matmul.h"
#endif
#include "./math/mc2_vec_transpose.h"

#endif // MC2_COMPUTE_STAGE_H
