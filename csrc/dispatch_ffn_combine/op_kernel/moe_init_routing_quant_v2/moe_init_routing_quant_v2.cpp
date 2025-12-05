/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_init_routing_quant_v2.cpp
 * \brief
 */
#include "moe_v2_sort_one_core.h"
#include "moe_v2_sort_multi_core.h"
#include "moe_v2_mrgsort_out.h"
#include "moe_v2_mrgsort.h"
#include "moe_v2_expert_token_out.h"
#include "moe_v2_src_to_dst_op.h"
#include "moe_v2_src_to_dst_with_capacity.h"
#include "moe_v2_fullload_quant.h"
#include "moe_v2_fullload_dynamic_quant.h"
#include "moe_v2_gather_quant.h"
#include "moe_v2_gather_dynamic_quant.h"
#include "moe_v2_src_to_dst_and_gather.h"

using namespace AscendC;
using namespace MoeInitRoutingQuantV2;
using namespace optiling;

template <class DTYPE_X = bfloat16_t>
__aicore__ inline  void moe_init_routing_quant_v2(
    GM_ADDR x, GM_ADDR expertIdx, GM_ADDR scale, GM_ADDR offset, GM_ADDR expandedX, GM_ADDR expandedRowIdx,
    GM_ADDR expertTokensCountOrCumsum, GM_ADDR expertTokensBeforeCapacity, GM_ADDR dynamicQuantScale, GM_ADDR workspace,
    const MoeInitRoutingQuantV2TilingData* tilingData, uint64_t tilingKey) {

  if (g_coreType == AIC) {
    return;
  }

  if (workspace == nullptr) {
    return;
  }

  if (tilingKey == 20000) {  // quant full load
    TPipe sortPipe;
    MoeV2FullLoadQuant<DTYPE_X> op;
    op.Init(x, expertIdx, scale, offset, expandedX, expandedRowIdx, expertTokensCountOrCumsum, workspace, tilingData, &sortPipe);
    op.Process();
    sortPipe.Destroy();
    return;
  } 
  
  
  else if (tilingKey == 21000) {  // dynamic quant full load
    TPipe sortPipe;
    MoeV2FullLoadDynamicQuant<DTYPE_X> op;
    op.Init(x, expertIdx, expandedX, expandedRowIdx, expertTokensCountOrCumsum, scale, dynamicQuantScale, workspace, tilingData,
            &sortPipe);
    op.Process();
    sortPipe.Destroy();
    return;
  }

  // sort
  if (tilingKey == 10000 || tilingKey == 10100 || tilingKey == 11000 || tilingKey == 11100) {
    TPipe sortPipe;
    MoeV2SortOneCore op;
    op.Init<MoeInitRoutingQuantV2TilingData>(expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity, workspace,
                                             tilingData, &sortPipe);
    op.Process();
    sortPipe.Destroy();
  } else if (tilingKey == 10010 || tilingKey == 10110 || tilingKey == 11010  || tilingKey== 11110) {
    TPipe sortPipe;
    MoeV2SortMultiCore op;
    op.Init<MoeInitRoutingQuantV2TilingData>(expertIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity, workspace,
                                             tilingData, &sortPipe);
    op.Process();
    sortPipe.Destroy();
  }

  if (tilingKey == 10000 || tilingKey == 10010 || tilingKey ==11000 || tilingKey ==11010) { // No drop scenario
    if (tilingData->expertTokensCountOrCumsumFlag != EXERPT_TOKENS_NONE) {
      TPipe expertTokenOutPipe;
      MoeV2ExpertTokenOut expertTokenOutOp;
      expertTokenOutOp.Init<MoeInitRoutingQuantV2TilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                                             expandedRowIdx, workspace, tilingData, &expertTokenOutPipe);
      expertTokenOutOp.Process();
      expertTokenOutPipe.Destroy();
    }
    TPipe srcToDstPipe;
    MoeV2SrcToDstOp srcToDstOp;
    srcToDstOp.Init<MoeInitRoutingQuantV2TilingData>(expandedRowIdx, workspace, tilingData, &srcToDstPipe);
    srcToDstOp.Process();
    srcToDstPipe.Destroy();
  } else if (tilingKey ==10100 || tilingKey ==10110 || tilingKey ==11100 || tilingKey ==11110) { // Drop scenario
    TPipe expertTokenOutPipe;
    MoeV2ExpertTokenOut expertTokenOutOp;
    expertTokenOutOp.Init<MoeInitRoutingQuantV2TilingData>(expertTokensCountOrCumsum, expertTokensBeforeCapacity,
                                                           expandedRowIdx, workspace, tilingData, &expertTokenOutPipe);
    expertTokenOutOp.Process();
    expertTokenOutPipe.Destroy();

    if (tilingKey == 10100 || tilingKey == 10110) {
      TPipe srcToDstPipe;
      MoeV2SrcToDstWithCapacity<int8_t, MoeInitRoutingQuantV2TilingData> srcToDstWithCapacityOp;
      srcToDstWithCapacityOp.Init(expandedRowIdx, expandedX, workspace, tilingData, &srcToDstPipe);
      srcToDstWithCapacityOp.Process();
      srcToDstPipe.Destroy();
    } else {
      TPipe srcToDstGatherPipe;
      MoeV2SrcToDstAndGather<DTYPE_X, MoeInitRoutingQuantV2TilingData> srcToDstAndGatherOp;
      srcToDstAndGatherOp.Init(x, scale, expandedRowIdx, expandedX, dynamicQuantScale, workspace, tilingData, &srcToDstGatherPipe);
      srcToDstAndGatherOp.Process();
      srcToDstGatherPipe.Destroy();
      return;
    }
  }

  if (tilingKey == 10000 || tilingKey == 10010 || tilingKey == 10100 || tilingKey == 10110) {
    TPipe gatherPipe;
    MoeV2GatherQuant<DTYPE_X> gatherQuantOp;
    gatherQuantOp.Init(x, scale, offset, expandedRowIdx, expandedX, workspace, tilingData, &gatherPipe);
    gatherQuantOp.Process();
    gatherPipe.Destroy();
  } else if (tilingKey  == 11000 || tilingKey == 11010) {
    TPipe gatherPipe;
    MoeV2GatherDynamicQuant<DTYPE_X> gatherDynamicQuantOp;
    gatherDynamicQuantOp.Init(x, scale, expandedRowIdx, expandedX, dynamicQuantScale, workspace, tilingData, &gatherPipe);
    gatherDynamicQuantOp.Process();
    gatherPipe.Destroy();
  }
}
