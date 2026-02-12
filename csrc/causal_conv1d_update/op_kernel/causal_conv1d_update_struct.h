/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_update_tiling_key.h
 * \brief causal_conv1d_update tiling key declare
 */

#ifndef __CAUSAL_CONV1D_UPDATE_TILING_KEY_H__
#define __CAUSAL_CONV1D_UPDATE_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_BF16 0
#define TPL_FP16 1

ASCENDC_TPL_ARGS_DECL(CausalConv1dUpdate,
    ASCENDC_TPL_DTYPE_DECL(
        dType, TPL_BF16, TPL_FP16)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DTYPE_SEL(
            dType, TPL_BF16, TPL_FP16)));

#endif // __CAUSAL_CONV1D_UPDATE_TILING_KEY_H__

