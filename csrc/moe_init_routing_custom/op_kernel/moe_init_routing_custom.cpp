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
 * \file moe_init_routing_custom.cpp
 * \brief
 */
#include "moe_custom_mrgsort_out.h"
#include "moe_custom_mrgsort.h"
#include "moe_custom_sort_one_core.h"
#include "moe_custom_sort_multi_core.h"
#include "moe_custom_gather_sort_multi_core.h"
#include "moe_custom_expert_tokens_count.h"
#include "moe_custom_row_idx_gather.h"
#include "moe_custom_gather_out.h"
#include "moe_custom_gather_dynamic_quant.h"
#include "moe_custom_gather_static_quant.h"
#include "moe_custom_full_load.h"
#include "moe_custom_full_load_dynamic_quant.h"
#include "moe_custom_full_load_static_quant.h"
#include "moe_custom_full_load_unquantized.h"
#include "moe_custom_sort_actual_expert.h"
#include "moe_custom_sort_multi_core_performance.h"
#include "moe_custom_row_idx_gather_droppad_dynamic.h"
#include "moe_custom_row_idx_gather_droppad.h"
#include "moe_custom_gather_out_droppad.h"
#include "moe_custom_gather_droppad_static_quant.h"

#define MOE_INIT_ROUTING_CUSTOM_PERFORMANCE 2000000
#define UNQUANTIZED_FULLLOAD 2100000
#define STATIC_QUANT_FULLLOAD 2200000
#define DYNAMIC_QUANT_GATHER_NO_SCALE_FULLLOAD 2300000
#define DYNAMIC_QUANT_GATHER_1H_DIM_SCALE_FULLLOAD 2301000
#define DYNAMIC_QUANT_GATHER_EH_SCALE_FULLLOAD 2302000
#define DYNAMIC_QUANT_SCATTER_NO_SCALE_FULLLOAD 2310000
#define DYNAMIC_QUANT_SCATTER_1H_SCALE_FULLLOAD 2311000
#define DYNAMIC_QUANT_SCATTER_EH_SCALE_FULLLOAD 2312000

#define MOE_INIT_ROUTING_CUSTOM_SORTONECORE_GATHER_NODROP 1000000
#define MOE_INIT_ROUTING_CUSTOM_SORTONECORE_SCATTER_NODROP 1001000
#define MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_GATHER_NODROP 1100000
#define MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_SCATTER_NODROP 1101000

#define MOE_INIT_ROUTING_CUSTOM_SORTONECORE_DYNAMICQUANT_GATHER_NODROP 1020000
#define MOE_INIT_ROUTING_CUSTOM_SORTONECORE_DYNAMICQUANT_SCATTER_NODROP 1021000
#define MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_DYNAMICQUANT_GATHER_NODROP 1120000
#define MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_DYNAMICQUANT_SCATTER_NODROP 1121000

#define MOE_INIT_ROUTING_CUSTOM_SORTONECORE_QUANT_GATHER_NODROP 1010000
#define MOE_INIT_ROUTING_CUSTOM_SORTONECORE_QUANT_SCATTER_NODROP 1011000
#define MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_QUANT_GATHER_NODROP 1110000
#define MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_QUANT_SCATTER_NODROP 1111000

#define MOE_INIT_ROUTING_CUSTOM_SORTONECORE_GATHER_DROP 1000100
#define MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_GATHER_DROP 1100100
#define MOE_INIT_ROUTING_CUSTOM_SORTONECORE_DYNAMICQUANT_GATHER_DROP 1020100
#define MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_DYNAMICQUANT_GATHER_DROP 1120100
#define MOE_INIT_ROUTING_CUSTOM_SORTONECORE_QUANT_GATHER_DROP 1010100
#define MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_QUANT_GATHER_DROP 1110100

#define MOE_INIT_ROUTING_CUSTOM_GATHER_SORTONECORE_GATHER 1200000
#define MOE_INIT_ROUTING_CUSTOM_GATHER_SORTONECORE_SCATTER 1201000
#define MOE_INIT_ROUTING_CUSTOM_GATHER_SORTMULTICORE_GATHER 1300000
#define MOE_INIT_ROUTING_CUSTOM_GATHER_SORTMULTICORE_SCATTER 1301000


using namespace AscendC;
using namespace MoeInitRoutingCustom;
extern "C" __global__ __aicore__ void moe_init_routing_custom(GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset,
                                                          GM_ADDR expandedX, GM_ADDR expandedRowIdx,
                                                          GM_ADDR expertTokensCountOrCumsum, GM_ADDR expandedScale,
                                                          GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (g_coreType == AIC) {
        return;
    }

    GET_TILING_DATA(tilingData, tiling);
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    auto t = &tilingData;

    if (TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_PERFORMANCE)) {
        TPipe fullLoadPipe;
        MoeCustomFullLoad op;
        op.Init(x, expertIdx, scale, offset, expandedX, expandedRowIdx, expertTokensCountOrCumsum, expandedScale, t,
                &fullLoadPipe);
        op.Process();
        fullLoadPipe.Destroy();
        return;
    }

    if (TILING_KEY_IS(DYNAMIC_QUANT_GATHER_NO_SCALE_FULLLOAD)) {
        if constexpr (!IsSameType<DTYPE_X, int8_t>::value) {
            TPipe fullLoadPipe;
            MoeCustomFullLoadDynamicQuant<DTYPE_X, GATHER, NO_SCALE> op;
            op.Init(x, expertIdx, scale, expandedX, expandedRowIdx, expertTokensCountOrCumsum, expandedScale, userWS, t,
                    &fullLoadPipe);
            op.Process();
            fullLoadPipe.Destroy();
        }
        return;
    }

    if (TILING_KEY_IS(DYNAMIC_QUANT_GATHER_1H_DIM_SCALE_FULLLOAD)) {
        if constexpr (!IsSameType<DTYPE_X, int8_t>::value) {
            TPipe fullLoadPipe;
            MoeCustomFullLoadDynamicQuant<DTYPE_X, GATHER, SCALE_1H> op;
            op.Init(x, expertIdx, scale, expandedX, expandedRowIdx, expertTokensCountOrCumsum, expandedScale, userWS, t,
                    &fullLoadPipe);
            op.Process();
            fullLoadPipe.Destroy();
        }
        return;
    }

    if (TILING_KEY_IS(DYNAMIC_QUANT_GATHER_EH_SCALE_FULLLOAD)) {
        if constexpr (!IsSameType<DTYPE_X, int8_t>::value) {
            TPipe fullLoadPipe;
            MoeCustomFullLoadDynamicQuant<DTYPE_X, GATHER, SCALE_EH> op;
            op.Init(x, expertIdx, scale, expandedX, expandedRowIdx, expertTokensCountOrCumsum, expandedScale, userWS, t,
                    &fullLoadPipe);
            op.Process();
            fullLoadPipe.Destroy();
        }
        return;
    }

    if (TILING_KEY_IS(DYNAMIC_QUANT_SCATTER_NO_SCALE_FULLLOAD)) {
        if constexpr (!IsSameType<DTYPE_X, int8_t>::value) {
            TPipe fullLoadPipe;
            MoeCustomFullLoadDynamicQuant<DTYPE_X, SCATTER, NO_SCALE> op;
            op.Init(x, expertIdx, scale, expandedX, expandedRowIdx, expertTokensCountOrCumsum, expandedScale, userWS, t,
                    &fullLoadPipe);
            op.Process();
            fullLoadPipe.Destroy();
        }
        return;
    }

    if (TILING_KEY_IS(DYNAMIC_QUANT_SCATTER_1H_SCALE_FULLLOAD)) {
        if constexpr (!IsSameType<DTYPE_X, int8_t>::value) {
            TPipe fullLoadPipe;
            MoeCustomFullLoadDynamicQuant<DTYPE_X, SCATTER, SCALE_1H> op;
            op.Init(x, expertIdx, scale, expandedX, expandedRowIdx, expertTokensCountOrCumsum, expandedScale, userWS, t,
                    &fullLoadPipe);
            op.Process();
            fullLoadPipe.Destroy();
        }
        return;
    }

    if (TILING_KEY_IS(DYNAMIC_QUANT_SCATTER_EH_SCALE_FULLLOAD)) {
        if constexpr (!IsSameType<DTYPE_X, int8_t>::value) {
            TPipe fullLoadPipe;
            MoeCustomFullLoadDynamicQuant<DTYPE_X, SCATTER, SCALE_EH> op;
            op.Init(x, expertIdx, scale, expandedX, expandedRowIdx, expertTokensCountOrCumsum, expandedScale, userWS, t,
                    &fullLoadPipe);
            op.Process();
            fullLoadPipe.Destroy();
        }
        return;
    }

    if (TILING_KEY_IS(UNQUANTIZED_FULLLOAD)) {
        TPipe fullLoadPipe;
        MoeCustomFullLoadUnquantized<DTYPE_X> op;
        op.Init(x, expertIdx, scale, expandedX, expandedRowIdx, expertTokensCountOrCumsum, expandedScale, userWS, t,
                &fullLoadPipe);
        op.Process();
        fullLoadPipe.Destroy();
        return;
    }

    if (TILING_KEY_IS(STATIC_QUANT_FULLLOAD)) {
        if constexpr (!IsSameType<DTYPE_X, int8_t>::value) {
            TPipe fullLoadPipe;
            MoeCustomFullLoadStaticQuant<DTYPE_X> op;
            op.Init(x, expertIdx, scale, offset, expandedX, expandedRowIdx, expertTokensCountOrCumsum, userWS, t,
                    &fullLoadPipe);
            op.Process();
            fullLoadPipe.Destroy();
        }
        return;
    }

    if (TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_GATHER_SORTONECORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_GATHER_SORTONECORE_SCATTER)) {
        TPipe sortActualExpertPipe;
        MoeSortActualExpert<DTYPE_X> op;
        bool isFinished = false;
        op.Init(x, expertIdx, scale, expandedX, expandedRowIdx, expertTokensCountOrCumsum, expandedScale, userWS, t,
                &sortActualExpertPipe);
        isFinished = op.Process();
        sortActualExpertPipe.Destroy();
        if (isFinished) {
            return;
        }
    }

    if (TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_GATHER_SORTMULTICORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_GATHER_SORTMULTICORE_SCATTER)) {
        TPipe gatherSortMultiCorePipe;
        MoeGatherSortMultiCore op;
        op.Init(expertIdx, expandedRowIdx, userWS, t, &gatherSortMultiCorePipe);
        op.Process();
        gatherSortMultiCorePipe.Destroy();
    }

    if (TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_GATHER_SORTONECORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_GATHER_SORTONECORE_SCATTER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_GATHER_SORTMULTICORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_GATHER_SORTMULTICORE_SCATTER)) {
        TPipe mergeSortMultiCorePipe;
        MoeSortMultiCorePerformance op;
        op.Init(expandedRowIdx, userWS, t, &mergeSortMultiCorePipe);
        op.Process();
        mergeSortMultiCorePipe.Destroy();
    }

    TPipe sortPipe;
    if (TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTONECORE_GATHER_NODROP) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTONECORE_SCATTER_NODROP) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTONECORE_QUANT_GATHER_NODROP) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTONECORE_QUANT_SCATTER_NODROP) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTONECORE_DYNAMICQUANT_GATHER_NODROP) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTONECORE_DYNAMICQUANT_SCATTER_NODROP) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTONECORE_QUANT_GATHER_DROP) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTONECORE_DYNAMICQUANT_GATHER_DROP) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTONECORE_GATHER_DROP)) {
        MoeSortOneCore op;
        op.Init(expertIdx, expandedRowIdx, userWS, t, &sortPipe);
        op.Process();
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_DYNAMICQUANT_GATHER_NODROP) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_DYNAMICQUANT_SCATTER_NODROP) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_QUANT_SCATTER_NODROP) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_QUANT_GATHER_NODROP) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_SCATTER_NODROP) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_GATHER_NODROP) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_QUANT_GATHER_DROP) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_DYNAMICQUANT_GATHER_DROP) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_GATHER_DROP)) {
        MoeSortMultiCore op;
        op.Init(expertIdx, expandedRowIdx, userWS, t, &sortPipe);
        op.Process();
    }
    sortPipe.Destroy();

    if (TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_GATHER_SORTONECORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_GATHER_SORTONECORE_SCATTER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_GATHER_SORTMULTICORE_GATHER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_GATHER_SORTMULTICORE_SCATTER)) {
        TPipe histogramPipe;
        if (t->expertTokensNumType == CUMSUM_MODE) {
            ExpertTokensCount<CUMSUM_MODE> countOp;
            countOp.Init<true>(expandedRowIdx, expertTokensCountOrCumsum, userWS, t, &histogramPipe);
            countOp.Process();
            histogramPipe.Destroy();
        } else if (t->expertTokensNumType == COUNT_MODE) {
            ExpertTokensCount<COUNT_MODE> countOp;
            countOp.Init<true>(expandedRowIdx, expertTokensCountOrCumsum, userWS, t, &histogramPipe);
            countOp.Process();
            histogramPipe.Destroy();
        } else {
            ExpertTokensCount<KEY_VALUE_MODE> countOp;
            countOp.Init<true>(expandedRowIdx, expertTokensCountOrCumsum, userWS, t, &histogramPipe);
            countOp.Process();
            histogramPipe.Destroy();
        }

    } else {
        if (t->dropPadMode == 1 || t->ep == 1 || t->expertTokensNumFlag != EXERPT_TOKENS_NONE) {
            TPipe histogramPipe;
            if (t->expertTokensNumType == CUMSUM_MODE) {
                ExpertTokensCount<CUMSUM_MODE> countOp;
                countOp.Init<false>(expandedRowIdx, expertTokensCountOrCumsum, userWS, t, &histogramPipe);
                countOp.Process();
                histogramPipe.Destroy();
            } else if (t->expertTokensNumType == COUNT_MODE) {
                ExpertTokensCount<COUNT_MODE> countOp;
                countOp.Init<false>(expandedRowIdx, expertTokensCountOrCumsum, userWS, t, &histogramPipe);
                countOp.Process();
                histogramPipe.Destroy();
            } else {
                ExpertTokensCount<KEY_VALUE_MODE> countOp;
                countOp.Init<false>(expandedRowIdx, expertTokensCountOrCumsum, userWS, t, &histogramPipe);
                countOp.Process();
                histogramPipe.Destroy();
            }
        }
    }

    if (TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTONECORE_GATHER_DROP) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_GATHER_DROP)) {
        TPipe rowIdxGatherDropPadPipe;
        MoeCustomSrcToDstWithCapacity<DTYPE_X, MoeInitRoutingCustomTilingData> rowIdxGatherDropPadOp;
        rowIdxGatherDropPadOp.Init(expandedRowIdx, expandedX, expandedScale, userWS, t, &rowIdxGatherDropPadPipe);
        rowIdxGatherDropPadOp.Process();
        rowIdxGatherDropPadPipe.Destroy();
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTONECORE_QUANT_GATHER_DROP) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_QUANT_GATHER_DROP)) {
        TPipe rowIdxGatherDropPadPipe;
        MoeCustomSrcToDstWithCapacity<int8_t, MoeInitRoutingCustomTilingData> rowIdxGatherDropPadOp;
        rowIdxGatherDropPadOp.Init(expandedRowIdx, expandedX, expandedScale, userWS, t, &rowIdxGatherDropPadPipe);
        rowIdxGatherDropPadOp.Process();
        rowIdxGatherDropPadPipe.Destroy();
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTONECORE_DYNAMICQUANT_GATHER_DROP) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_DYNAMICQUANT_GATHER_DROP)) {
        if constexpr (!IsSameType<DTYPE_X, int8_t>::value) {
            TPipe gatherPipe;
            MoeCustomSrcToDstAndGather<DTYPE_X, MoeInitRoutingCustomTilingData> gatherDroppadDynamicQuantOp;
            gatherDroppadDynamicQuantOp.Init(x, scale, expandedRowIdx, expandedX, expandedScale, userWS, t,
                                             &gatherPipe);
            gatherDroppadDynamicQuantOp.Process();
            gatherPipe.Destroy();
        }
    } else {
        TPipe rowIdxPipe;
        RowIdxGather rowIdxGatherOp;
        rowIdxGatherOp.Init(expandedRowIdx, userWS, t, &rowIdxPipe);
        rowIdxGatherOp.Process();
        rowIdxPipe.Destroy();
    }

    if (TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTONECORE_GATHER_NODROP) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTONECORE_SCATTER_NODROP) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_GATHER_NODROP) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_SCATTER_NODROP) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_GATHER_SORTONECORE_SCATTER) ||
        TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_GATHER_SORTMULTICORE_SCATTER)) {
        TPipe gatherPipe;
        if (t->ep == 1) {
            MoeGatherOut<DTYPE_X, 1> gatherOp;
            gatherOp.Init(x, scale, userWS, expandedRowIdx, expandedX, expandedScale, t, &gatherPipe);
            gatherOp.Process();
            gatherPipe.Destroy();
        } else {
            MoeGatherOut<DTYPE_X, 0> gatherOp;
            gatherOp.Init(x, scale, userWS, expandedRowIdx, expandedX, expandedScale, t, &gatherPipe);
            gatherOp.Process();
            gatherPipe.Destroy();
        }

    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_DYNAMICQUANT_SCATTER_NODROP) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_DYNAMICQUANT_GATHER_NODROP) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTONECORE_DYNAMICQUANT_SCATTER_NODROP) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTONECORE_DYNAMICQUANT_GATHER_NODROP)) {
        if constexpr (!IsSameType<DTYPE_X, int8_t>::value) {
            TPipe gatherPipe;
            if (t->ep == 0 and t->smoothType != SCALE_EH) {
                MoeGatherOutDynamicQuant<DTYPE_X, GATHER> gatherDynamicQuantOp;
                gatherDynamicQuantOp.Init(x, scale, userWS, expandedRowIdx, expandedX, expandedScale, t, &gatherPipe);
                gatherDynamicQuantOp.Process();
                gatherPipe.Destroy();
            } else {
                MoeGatherOutDynamicQuant<DTYPE_X, SCATTER> gatherDynamicQuantOp;
                gatherDynamicQuantOp.Init(x, scale, userWS, expandedRowIdx, expandedX, expandedScale, t, &gatherPipe);
                gatherDynamicQuantOp.Process();
                gatherPipe.Destroy();
            }
        }
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_QUANT_SCATTER_NODROP) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_QUANT_GATHER_NODROP) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTONECORE_QUANT_SCATTER_NODROP) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTONECORE_QUANT_GATHER_NODROP)) {
        if constexpr (!IsSameType<DTYPE_X, int8_t>::value) {
            TPipe gatherPipe;
            if (t->ep == 1) {
                MoeGatherOutQuant<DTYPE_X, 1> gatherStaticQuantOp;
                gatherStaticQuantOp.Init(x, scale, offset, expandedRowIdx, expandedX, userWS, t, &gatherPipe);
                gatherStaticQuantOp.Process();
                gatherPipe.Destroy();
            } else {
                MoeGatherOutQuant<DTYPE_X, 0> gatherStaticQuantOp;
                gatherStaticQuantOp.Init(x, scale, offset, expandedRowIdx, expandedX, userWS, t, &gatherPipe);
                gatherStaticQuantOp.Process();
                gatherPipe.Destroy();
            }
        }
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTONECORE_GATHER_DROP) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_GATHER_DROP)) {
        TPipe gatherPipe;
        MoeGatherOutDroppad<DTYPE_X> gatherDroppadOp;
        gatherDroppadOp.Init(x, scale, expandedRowIdx, expandedX, expandedScale, userWS, t, &gatherPipe);
        gatherDroppadOp.Process();
        gatherPipe.Destroy();
    } else if (TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTONECORE_QUANT_GATHER_DROP) ||
               TILING_KEY_IS(MOE_INIT_ROUTING_CUSTOM_SORTMULTICORE_QUANT_GATHER_DROP)) {
        if constexpr (!IsSameType<DTYPE_X, int8_t>::value) {
            TPipe gatherPipe;
            MoeGatherDroppadQuant<DTYPE_X> gatherDroppadStaticQuantOp;
            gatherDroppadStaticQuantOp.Init(x, scale, offset, expandedRowIdx, expandedX, userWS, t, &gatherPipe);
            gatherDroppadStaticQuantOp.Process();
            gatherPipe.Destroy();
        }
    }
}