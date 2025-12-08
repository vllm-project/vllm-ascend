/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_init_routing_v3_tiling.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INIT_ROUTING_V3_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INIT_ROUTING_V3_H
#include <cstdint>
#include <algorithm>
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
BEGIN_TILING_DATA_DEF(MoeV3VBSComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, perCoreElements);
TILING_DATA_FIELD_DEF(int64_t, perCoreLoops);
TILING_DATA_FIELD_DEF(int64_t, perCorePerLoopElements);
TILING_DATA_FIELD_DEF(int64_t, perCoreLastLoopElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLoops);
TILING_DATA_FIELD_DEF(int64_t, lastCorePerLoopElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLastLoopElements);
TILING_DATA_FIELD_DEF(int64_t, oneLoopMaxElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeV3VBSComputeTilingDataOp, MoeV3VBSComputeTilingData)

BEGIN_TILING_DATA_DEF(MoeV3VMSMiddleComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeV3VMSMiddleComputeTilingDataOp, MoeV3VMSMiddleComputeTilingData)

BEGIN_TILING_DATA_DEF(MoeV3SortOutComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, oneLoopMaxElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeV3SortOutComputeTilingDataOp, MoeV3SortOutComputeTilingData)

BEGIN_TILING_DATA_DEF(MoeV3ExpertTokensCountTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, perCoreElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreElements);
TILING_DATA_FIELD_DEF(int64_t, perCoreLoops);
TILING_DATA_FIELD_DEF(int64_t, perCorePerLoopElements);
TILING_DATA_FIELD_DEF(int64_t, perCoreLastLoopElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLoops);
TILING_DATA_FIELD_DEF(int64_t, lastCorePerLoopElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLastLoopElements);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeV3ExpertTokensCountTilingDataOp, MoeV3ExpertTokensCountTilingData)

BEGIN_TILING_DATA_DEF(MoeV3GatherOutComputeTilingData)
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, perCoreIndicesElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreIndicesElements);
TILING_DATA_FIELD_DEF(int64_t, perCoreIndicesLoops);
TILING_DATA_FIELD_DEF(int64_t, perCorePerLoopIndicesElements);
TILING_DATA_FIELD_DEF(int64_t, perCoreLastLoopIndicesElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreIndicesLoops);
TILING_DATA_FIELD_DEF(int64_t, lastCorePerLoopIndicesElements);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLastLoopIndicesElements);
TILING_DATA_FIELD_DEF(int64_t, colsLoops);
TILING_DATA_FIELD_DEF(int64_t, perLoopCols);
TILING_DATA_FIELD_DEF(int64_t, lastLoopCols);
TILING_DATA_FIELD_DEF(int64_t, activeNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeV3GatherOutComputeTilingDataOp, MoeV3GatherOutComputeTilingData)

BEGIN_TILING_DATA_DEF(MoeV3SrcToDstCapacityComputeTilingData) // 注意：to do activeNum已经移走了
TILING_DATA_FIELD_DEF(int64_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, perCoreRows);
TILING_DATA_FIELD_DEF(int64_t, perCorePerLoopRows);
TILING_DATA_FIELD_DEF(int64_t, perCoreLastLoopRows);
TILING_DATA_FIELD_DEF(int64_t, lastCoreRows);
TILING_DATA_FIELD_DEF(int64_t, lastCorePerLoopRows);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLastLoopRows);
TILING_DATA_FIELD_DEF(int64_t, perCoreLoops);
TILING_DATA_FIELD_DEF(int64_t, lastCoreLoops);
TILING_DATA_FIELD_DEF(int64_t, perLoopCols);
TILING_DATA_FIELD_DEF(int64_t, lastLoopCols);
TILING_DATA_FIELD_DEF(int64_t, colLoops);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeV3SrcToDstCapacityComputeTilingDataOp, MoeV3SrcToDstCapacityComputeTilingData)

BEGIN_TILING_DATA_DEF(MoeInitRoutingV3TilingData)
TILING_DATA_FIELD_DEF(int64_t, coreNum);
TILING_DATA_FIELD_DEF(int64_t, n);
TILING_DATA_FIELD_DEF(int64_t, cols);
TILING_DATA_FIELD_DEF(int64_t, k);
TILING_DATA_FIELD_DEF(int64_t, expertStart);
TILING_DATA_FIELD_DEF(int64_t, expertEnd);
TILING_DATA_FIELD_DEF(int64_t, actualExpertNum);
TILING_DATA_FIELD_DEF(int64_t, quantMode);
TILING_DATA_FIELD_DEF(int64_t, rowIdxType);
TILING_DATA_FIELD_DEF(int64_t, isInputScale);
TILING_DATA_FIELD_DEF(int64_t, isInputOffset);
TILING_DATA_FIELD_DEF(int64_t, expertNum);
TILING_DATA_FIELD_DEF(int64_t, expertTokensNumType);
TILING_DATA_FIELD_DEF(int64_t, expertTokensNumFlag);
TILING_DATA_FIELD_DEF(int64_t, gatherFirstFullload);
TILING_DATA_FIELD_DEF(int64_t, ep);
TILING_DATA_FIELD_DEF(int64_t, activeNum);
TILING_DATA_FIELD_DEF(int64_t, dropPadMode);
TILING_DATA_FIELD_DEF(int64_t, smoothType);
TILING_DATA_FIELD_DEF(int64_t, expertCountElements);
TILING_DATA_FIELD_DEF(int64_t, expertCapacity);
TILING_DATA_FIELD_DEF_STRUCT(MoeV3VBSComputeTilingData, vbsComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(MoeV3VMSMiddleComputeTilingData, vmsMiddleComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(MoeV3SortOutComputeTilingData, sortOutComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(MoeV3ExpertTokensCountTilingData, expertTokensCountTilingDataOp);
TILING_DATA_FIELD_DEF_STRUCT(MoeV3GatherOutComputeTilingData, gatherOutComputeParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(MoeV3SrcToDstCapacityComputeTilingData, srcToDstDropPadParamsOp);
TILING_DATA_FIELD_DEF_STRUCT(MoeV3SrcToDstCapacityComputeTilingData, srcToDstDropPadDynamicParamsOp);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MoeInitRoutingV3, MoeInitRoutingV3TilingData)
struct MoeInitRoutingV3CompileInfo {
        int32_t aivNum = 0;
        uint64_t ubSize = 0;
        platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
  };
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_MOE_INIT_ROUTING_V3_H