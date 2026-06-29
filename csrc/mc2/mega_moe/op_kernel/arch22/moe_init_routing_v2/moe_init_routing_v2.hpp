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
 * \file moe_init_routing_v2.cpp
 * \brief
 */

#ifndef MC2_MOE_INIT_ROUTING_V2_HPP
#define MC2_MOE_INIT_ROUTING_V2_HPP

#ifdef __DAV_C310__
#include "arch35/moe_v2_mrgsort_out.h"
#include "arch35/moe_v2_mrgsort.h"
#include "arch35/moe_v2_sort_multi_core.h"
#include "arch35/moe_v2_sort_one_core.h"
#include "arch35/moe_v2_expert_token_out_regbase.h"
#include "arch35/moe_v2_expert_token_out_simt.h"
#include "arch35/moe_v2_src_to_dst_op_simt.h"
#include "arch35/moe_v2_src_to_dst_with_capacity_simt.h"
#include "arch35/moe_v2_gather_out_for_simt.h"
#else
#include "moe_v2_mrgsort_out.h"
#include "moe_v2_mrgsort.h"
#include "moe_v2_sort_multi_core.h"
#include "moe_v2_sort_one_core.h"
#include "moe_v2_expert_token_out.h"
#include "moe_v2_src_to_dst_op.h"
#include "moe_v2_src_to_dst_with_capacity.h"
#include "moe_v2_gather_out.h"
#include "moe_v2_init_routing_fullload.h"
#endif

using namespace AscendC;
using namespace MoeInitRoutingV2;
using namespace Mc2Tiling;


template <class DTYPE_X = bfloat16_t>
__aicore__ inline void moe_init_routing_v2(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                           GM_ADDR expertTokensCountOrCumsum, GM_ADDR expertTokensBeforeCapacity,
                                           GM_ADDR workspace, const MoeInitRoutingV2TilingData *tilingData,
                                           uint64_t tilingKey)
{
    if (g_coreType == AIC) {
        return;
    }

    // GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = workspace;
    if (userWS == nullptr) {
        return;
    }
    // auto t = tilingData;
    if (tilingKey == 20000) {
        TPipe sortPipe;
        MoeV2FullLoad<DTYPE_X> op;
        op.Init(x, expertIdx, expandedX, expandedRowIdx, expertTokensCountOrCumsum, userWS, tilingData, &sortPipe);
        op.Process();
        sortPipe.Destroy();
        // trap();
        return;
    }

    if (tilingKey == 10001 || tilingKey == 10011) {
        TPipe sortPipe;
        MoeInitRoutingV2::MoeV2SortOneCore op;
        op.Init<MoeInitRoutingV2TilingData>(expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity, userWS,
                                            tilingData, &sortPipe);
        op.Process();
        sortPipe.Destroy();
    } else if (tilingKey == 10002 || tilingKey == 10012) {
        TPipe sortPipe;
        MoeInitRoutingV2::MoeV2SortMultiCore op;
        op.Init<MoeInitRoutingV2TilingData>(expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity, userWS,
                                            tilingData, &sortPipe);
        op.Process();
        sortPipe.Destroy();
    }

    if (tilingKey == 10001 || tilingKey == 10002) {
        if (tilingData->expertTokensCountOrCumsumFlag != MoeInitRoutingV2::EXERPT_TOKENS_NONE) {
            TPipe expertTokenOutPipe;
            MoeInitRoutingV2::MoeV2ExpertTokenOut expertTokenOutOp;
            expertTokenOutOp.Init<MoeInitRoutingV2TilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                                              expandedRowIdx, userWS, tilingData, &expertTokenOutPipe);
            expertTokenOutOp.Process();
            expertTokenOutPipe.Destroy();
        }
        TPipe srcToDstPipe;
        MoeInitRoutingV2::MoeV2SrcToDstOp srcToDstOp;
        srcToDstOp.Init<MoeInitRoutingV2TilingData>(expandedRowIdx, userWS, tilingData, &srcToDstPipe);
        srcToDstOp.Process();
        srcToDstPipe.Destroy();
    } else if (tilingKey == 10011 || tilingKey == 10012) {
        TPipe expertTokenOutPipe;
        MoeInitRoutingV2::MoeV2ExpertTokenOut expertTokenOutOp;
        expertTokenOutOp.Init<MoeInitRoutingV2TilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                                          expandedRowIdx, userWS, tilingData, &expertTokenOutPipe);
        expertTokenOutOp.Process();
        expertTokenOutPipe.Destroy();

        TPipe srcToDstPipe;
        MoeInitRoutingV2::MoeV2SrcToDstWithCapacity<DTYPE_X, MoeInitRoutingV2TilingData> srcToDstWithCapacityOp;
        srcToDstWithCapacityOp.Init(expandedRowIdx, expandedX, userWS, tilingData, &srcToDstPipe);
        srcToDstWithCapacityOp.Process();
        srcToDstPipe.Destroy();
    }

    TPipe gatherPipe;
    MoeInitRoutingV2::MoeV2GatherOut<DTYPE_X> gatherOp;
    gatherOp.Init(x, expandedRowIdx, expandedX, userWS, tilingData, &gatherPipe);
    gatherOp.Process();
    gatherPipe.Destroy();
}

#endif // MC2_MOE_INIT_ROUTING_V2_HPP