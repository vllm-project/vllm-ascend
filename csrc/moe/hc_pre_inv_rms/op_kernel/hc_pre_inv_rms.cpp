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
 * \file hc_pre_inv_rms_apt.cpp
 * \brief hc_pre_inv_rms kernel file
 */
#include "hc_pre_inv_rms_full_load.h"
#if defined(__DAV_C310__) 
  #include "hc_pre_inv_rms_full_load_regbase.h"
  using namespace HcPreInvRmsRegbase;
#endif
#include "kernel_operator.h"
using namespace AscendC;
using namespace HcPreInvRms;

#define FULL_LOAD_TILING_KEY 1000
#define REGBASE_FULL_LOAD_TILING_KEY 2000

extern "C" __global__ __aicore__ void hc_pre_inv_rms(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(FULL_LOAD_TILING_KEY)) {
        HcPreInvRmsFullLoad<DTYPE_X> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    }
    #if defined(__DAV_C310__) 
      else if (TILING_KEY_IS(REGBASE_FULL_LOAD_TILING_KEY)) {
        HcPreInvRmsFullLoadRegbase<DTYPE_X> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
      }
    #endif
}
