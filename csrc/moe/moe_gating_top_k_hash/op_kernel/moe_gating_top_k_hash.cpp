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
 * \file moe_gating_top_k_hash.cpp
 * \brief
 */

#include "moe_gating_top_k_hash_e_k_fullload.h"
#include "moe_gating_top_k_hash_without_group.h"
#include "moe_gating_top_k_hash_generalized.h"
#if defined(__DAV_C310__) 
  #include "moe_gating_top_k_hash_regbase.h"
  using namespace MoeGatingTopKHashRegbaseNS;
#endif
#define TILING_KEY_PER_GROUP_COUNT_32 0
#define TILING_KEY_WITHOUT_GROUP 1
#define TILING_KEY_GENERALIZED 2
#define TILING_KEY_WITHOUT_GROUP_1 3
#define TILING_KEY_WITHOUT_GROUP_2 4
#define TILING_KEY_WITHOUT_GROUP_3 5
#define TILING_KEY_WITHOUT_GROUP_4 6
#define TILING_KEY_REGBASE 10000
#define TILING_KEY_REGBASE_1 10001
#define TILING_KEY_REGBASE_2 10002
#define TILING_KEY_REGBASE_3 10003
#define TILING_KEY_REGBASE_4 10004

using namespace AscendC;
using namespace MoeGatingTopKHash;
extern "C" __global__ __aicore__ void moe_gating_top_k_hash(GM_ADDR x, GM_ADDR bias, GM_ADDR inputIds, GM_ADDR tid2eid, GM_ADDR y, GM_ADDR expertIdx,
                                                            GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (g_coreType == AIC) {
        return;
    }
    
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    TPipe tPipe;
    if (TILING_KEY_IS(TILING_KEY_PER_GROUP_COUNT_32)) {
        GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKHashTilingData, tilingData, tiling);
        const MoeGatingTopKHashTilingData *__restrict t = &tilingData;
        MoeGatingTopKHashEKFullload<DTYPE_X> op;
        op.Init(x, bias, y, expertIdx, out, userWS, t, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_WITHOUT_GROUP)) {
        GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKHashTilingData, tilingData, tiling);
        const MoeGatingTopKHashTilingData *__restrict t = &tilingData;
        MoeGatingTopKHashWithoutGroup<DTYPE_X, int32_t, int32_t> op;
        op.Init(x, bias, inputIds, tid2eid, y, expertIdx, out, userWS, t, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_WITHOUT_GROUP_1)) {
        GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKHashTilingData, tilingData, tiling);
        const MoeGatingTopKHashTilingData *__restrict t = &tilingData;
        MoeGatingTopKHashWithoutGroup<DTYPE_X, int32_t, int64_t> op;
        op.Init(x, bias, inputIds, tid2eid, y, expertIdx, out, userWS, t, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_WITHOUT_GROUP_2)) {
        GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKHashTilingData, tilingData, tiling);
        const MoeGatingTopKHashTilingData *__restrict t = &tilingData;
        MoeGatingTopKHashWithoutGroup<DTYPE_X, int32_t, int32_t> op;
        op.Init(x, bias, inputIds, tid2eid, y, expertIdx, out, userWS, t, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_WITHOUT_GROUP_3)) {
        GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKHashTilingData, tilingData, tiling);
        const MoeGatingTopKHashTilingData *__restrict t = &tilingData;
        MoeGatingTopKHashWithoutGroup<DTYPE_X, int64_t, int64_t> op;
        op.Init(x, bias, inputIds, tid2eid, y, expertIdx, out, userWS, t, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_WITHOUT_GROUP_4)) {
        GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKHashTilingData, tilingData, tiling);
        const MoeGatingTopKHashTilingData *__restrict t = &tilingData;
        MoeGatingTopKHashWithoutGroup<DTYPE_X, int64_t, int32_t> op;
        op.Init(x, bias, inputIds, tid2eid, y, expertIdx, out, userWS, t, &tPipe);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_GENERALIZED)) {
        GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKHashTilingData, tilingData, tiling);
        const MoeGatingTopKHashTilingData *__restrict t = &tilingData;
        MoeGatingTopKHashGenerlized<DTYPE_X> op;
        op.Init(x, bias, y, expertIdx, out, userWS, t, &tPipe);
        op.Process();
    }
    #if defined(__DAV_C310__) 
      else if (TILING_KEY_IS(TILING_KEY_REGBASE)) {
          GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKHashRegbaseTilingData, tiling_data_in, tiling);
          const MoeGatingTopKHashRegbaseTilingData *__restrict tilingData = &tiling_data_in;
          MoeGatingTopKHashRegbase<DTYPE_X, int32_t, int32_t> op;
          op.Init(x, bias, inputIds, tid2eid, y, expertIdx, out, userWS, tilingData, &tPipe);
          op.Process();
      } else if (TILING_KEY_IS(TILING_KEY_REGBASE_1)) {
          GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKHashRegbaseTilingData, tiling_data_in, tiling);
          const MoeGatingTopKHashRegbaseTilingData *__restrict tilingData = &tiling_data_in;
          MoeGatingTopKHashRegbase<DTYPE_X, int32_t, int64_t>  op;
          op.Init(x, bias, inputIds, tid2eid, y, expertIdx, out, userWS, tilingData, &tPipe);
          op.Process();
      } else if (TILING_KEY_IS(TILING_KEY_REGBASE_2)) {
          GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKHashRegbaseTilingData, tiling_data_in, tiling);
          const MoeGatingTopKHashRegbaseTilingData *__restrict tilingData = &tiling_data_in;
          MoeGatingTopKHashRegbase<DTYPE_X, int32_t, int32_t>  op;
          op.Init(x, bias, inputIds, tid2eid, y, expertIdx, out, userWS, tilingData, &tPipe);
          op.Process();
      } else if (TILING_KEY_IS(TILING_KEY_REGBASE_3)) {
          GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKHashRegbaseTilingData, tiling_data_in, tiling);
          const MoeGatingTopKHashRegbaseTilingData *__restrict tilingData = &tiling_data_in;
          MoeGatingTopKHashRegbase<DTYPE_X, int64_t, int64_t>  op;
          op.Init(x, bias, inputIds, tid2eid, y, expertIdx, out, userWS, tilingData, &tPipe);
          op.Process();
      } else if (TILING_KEY_IS(TILING_KEY_REGBASE_4)) {
          GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKHashRegbaseTilingData, tiling_data_in, tiling);
          const MoeGatingTopKHashRegbaseTilingData *__restrict tilingData = &tiling_data_in;
          MoeGatingTopKHashRegbase<DTYPE_X, int64_t, int32_t>  op;
          op.Init(x, bias, inputIds, tid2eid, y, expertIdx, out, userWS, tilingData, &tPipe);
          op.Process();
      }
    #endif
}
