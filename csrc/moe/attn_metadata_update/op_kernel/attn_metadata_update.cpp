/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "attn_metadata_update.h"
#include "attn_metadata_update_tiling.h"

using namespace AscendC;
using namespace AttnMetadataUpdateImpl;

extern "C" __global__ __aicore__ void attn_metadata_update(
    GM_ADDR reqIds,
    GM_ADDR discardLabel,
    GM_ADDR positions,
    GM_ADDR slotMapping,
    GM_ADDR blockTable,
    GM_ADDR seqLens,
    GM_ADDR discardPositions,
    GM_ADDR discardSlotMapping,
    GM_ADDR discardContextLens,
    GM_ADDR discardBlockTables,
    GM_ADDR discardAttnCnt,
    GM_ADDR discardReqIds,
    GM_ADDR originalBs,
    GM_ADDR outPositions,
    GM_ADDR outSlots,
    GM_ADDR outSeqLens,
    GM_ADDR outBlockTable,
    GM_ADDR workspace,
    GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(AttnMetadataUpdateTilingData);
    GET_TILING_DATA_WITH_STRUCT(AttnMetadataUpdateTilingData, tilingData, tilingGM);
    TPipe pipe;
    AttnMetadataUpdate op;
    op.Init(reqIds, discardLabel, positions, slotMapping, blockTable, seqLens,
            discardPositions, discardSlotMapping, discardContextLens, discardBlockTables,
            discardAttnCnt, discardReqIds,originalBs,
            outPositions, outSlots, outSeqLens, outBlockTable,
            &pipe, &tilingData);
    op.Process();
}
