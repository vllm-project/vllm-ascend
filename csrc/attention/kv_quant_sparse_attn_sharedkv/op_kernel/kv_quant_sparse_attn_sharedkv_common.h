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
 * \file kv_quant_sparse_attn_sharedkv_common.h
 * \brief
 */

#ifndef KV_QUANT_SPARSE_FLASH_ATTENTION_COMMON_H
#define KV_QUANT_SPARSE_FLASH_ATTENTION_COMMON_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "kv_quant_sparse_attn_sharedkv_metadata.h"

using namespace AscendC;

enum class SAS_LAYOUT {
    BSND = 0,
    TND = 1,
    PA_ND = 2
};

enum class SASTemplateMode {
    SWA_TEMPLATE_MODE = 0,
    CFA_TEMPLATE_MODE = 1,
    SCFA_TEMPLATE_MODE = 2
};
#endif // KV_QUANT_SPARSE_FLASH_ATTENTION_COMMON_H