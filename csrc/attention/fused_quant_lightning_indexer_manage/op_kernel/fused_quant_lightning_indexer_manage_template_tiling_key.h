/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef TEMPLATE_TILING_KEY_LI_MANAGE_MTP_C8_H_
#define TEMPLATE_TILING_KEY_LI_MANAGE_MTP_C8_H_

#include "ascendc/host_api/tiling/template_argument.h"

// arch22: INT8 + FP16 scales; arch35: FP8 E4M3FN + FP32 scales.
// key 值沿用 host 侧 GET_TPL_TILING_KEY(inputQType) 的 ge::DT_INT8 枚举值。
#define LI_MTP_C8_TPL_INT8 2
#define LI_MTP_C8_TPL_FLOAT8_E4M3FN 36

ASCENDC_TPL_ARGS_DECL(FusedQuantLightningIndexerManage,
                      ASCENDC_TPL_DTYPE_DECL(DT, LI_MTP_C8_TPL_INT8, LI_MTP_C8_TPL_FLOAT8_E4M3FN));

#if (__CCE_AICORE__ == 310)
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DT, LI_MTP_C8_TPL_FLOAT8_E4M3FN)), );
#elif defined(__CCE_AICORE__)
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DT, LI_MTP_C8_TPL_INT8)), );
#else
// Host dispatch generates keys for both architectures.
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DT, LI_MTP_C8_TPL_INT8)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DT, LI_MTP_C8_TPL_FLOAT8_E4M3FN)), );
#endif

#endif
