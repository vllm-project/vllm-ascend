/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
// 将isCheckTiling设置为false, 输入输出的max&sum&exp的shape为(m, 1)
constexpr SoftmaxConfig SAS_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

enum class SAS_RUN_MODE {
    SWA_MODE = 0,
    SCFA_MODE = 1,
    CFA_MODE = 2,
};

enum class SAS_LAYOUT {
    BSND = 0,
    TND = 1
};

enum class SAS_KV_LAYOUT {
    TND = 0,
    PA_ND = 1
};

enum class SASTemplateMode {
    SWA_TEMPLATE_MODE = 0,
    CFA_TEMPLATE_MODE = 1,
    SCFA_TEMPLATE_MODE = 2
};

template <typename Q_T, typename KV_T, typename OUT_T, const bool FLASH_DECODE = false,
	  SAS_LAYOUT LAYOUT_T = SAS_LAYOUT::BSND, SAS_KV_LAYOUT KV_LAYOUT_T = SAS_KV_LAYOUT::PA_ND, 
      typename... Args>
struct SASType {
    using queryType = Q_T;
    using kvType = KV_T;
    using outputType = OUT_T;
    static constexpr bool flashDecode = FLASH_DECODE;
    static constexpr SAS_LAYOUT layout = LAYOUT_T;
    static constexpr SAS_KV_LAYOUT kvLayout = KV_LAYOUT_T;
    static constexpr bool pageAttention = (KV_LAYOUT_T == SAS_KV_LAYOUT::PA_ND);
};

#endif // KVQUANT_SPARSE_FLASH_ATTENTION_COMMON_H