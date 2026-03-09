/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_tiling_key.h
 * \brief causal_conv1d tiling key declare
 */

#ifndef __CAUSAL_CONV1D_TILING_KEY_H__
#define __CAUSAL_CONV1D_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define CAUSAL_CONV1D_TPL_SCH_MODE_DEFAULT 0

ASCENDC_TPL_ARGS_DECL(CausalConv1d,
    ASCENDC_TPL_UINT_DECL(
        schMode, 1, ASCENDC_TPL_UI_LIST, CAUSAL_CONV1D_TPL_SCH_MODE_DEFAULT)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(
            schMode, ASCENDC_TPL_UI_LIST, CAUSAL_CONV1D_TPL_SCH_MODE_DEFAULT)));

#endif // __CAUSAL_CONV1D_TILING_KEY_H__