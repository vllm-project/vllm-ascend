/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef TEMPLATE_TILING_KEY_LI_MANAGE_MTP_C8_H_
#define TEMPLATE_TILING_KEY_LI_MANAGE_MTP_C8_H_

#include "ascendc/host_api/tiling/template_argument.h"

// C8 唯一量化形态: int8 query/key + bf16 weights + fp16 scales。
// key 值沿用 host 侧 GET_TPL_TILING_KEY(inputQType) 的 ge::DT_INT8 枚举值。
#define LI_MTP_C8_TPL_INT8 2

ASCENDC_TPL_ARGS_DECL(FusedLiManageMtpC8,
                      ASCENDC_TPL_DTYPE_DECL(DT, LI_MTP_C8_TPL_INT8));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DT, LI_MTP_C8_TPL_INT8)), );

#endif
