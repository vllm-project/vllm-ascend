/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "rms_norm_cast.h"

using namespace AscendC;

#define RUN_RMS_NORM_CAST(TYPE)                     \
    do {                                            \
        KernelRmsNormCast<TYPE> op(&pipe);          \
        op.Init(x, gamma, y, y_fp32, &tiling_data); \
        op.Process();                               \
    } while (0)

extern "C" __global__ __aicore__ void rms_norm_cast(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR y_fp32,
    GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(1)) {
        RUN_RMS_NORM_CAST(half);
    } else if (TILING_KEY_IS(3)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        RUN_RMS_NORM_CAST(bfloat16_t);
#endif
    }
}
