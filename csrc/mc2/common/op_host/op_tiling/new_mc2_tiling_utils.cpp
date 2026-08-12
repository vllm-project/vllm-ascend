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
 * \file mc2_tiling_utils.cpp
 * \brief
 */

#include <cstdlib>

#include "graph/utils/type_utils.h"
#include "mc2_hcom_topo_info.h"
#include "mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_tiling_strategy.h"
#include "mc2_log.h"
#include "mc2_tiling_utils.h"


namespace mc2tiling {

uint64_t NewGetDataTypeSize(const std::string &opName, ge::DataType type)
{
    static const std::map<ge::DataType, int64_t> DATA_TYPE_SIZE_MAP = {
        {ge::DT_BF16, 2},     {ge::DT_FLOAT16, 2},       {ge::DT_FLOAT, 4},
        {ge::DT_HIFLOAT8, 1}, {ge::DT_FLOAT8_E4M3FN, 1}, {ge::DT_FLOAT8_E5M2, 1},
    };

    auto iterator = DATA_TYPE_SIZE_MAP.find(type);
    if (iterator != DATA_TYPE_SIZE_MAP.end()) {
        return iterator->second;
    }

    OP_LOGI(opName, "cannot find datatype size according to ge datatype: %d", static_cast<int32_t>(type));
    return 0;
}
void NewUpdateMatmulV3Args(optiling::mc2_matmul_v3_advanced::Mc2MatMulV3Args &mmV3Args, const TilingArgs &args,
                           const char *opName)
{
    mmV3Args.opName = opName;
    mmV3Args.isATrans = args.isATrans;
    mmV3Args.isBTrans = args.isBTrans;
    mmV3Args.isHf32 = false;
    mmV3Args.hasBias = args.isBias;
    mmV3Args.aType = args.geAType;
    mmV3Args.bType = args.geBType;
    mmV3Args.cType = args.geCType;
    mmV3Args.biasType = args.geBiasType;
    mmV3Args.aFormat = ge::FORMAT_ND;
    mmV3Args.bFormat = ge::FORMAT_ND;
    mmV3Args.outFormat = ge::FORMAT_ND;
    mmV3Args.mValue = args.mValue;
    mmV3Args.nValue = args.nValue;
    mmV3Args.kValue = args.kValue;
    mmV3Args.aDtypeSize = NewGetDataTypeSize(opName, mmV3Args.aType);
    mmV3Args.bDtypeSize = NewGetDataTypeSize(opName, mmV3Args.bType);
}

ge::graphStatus NewGetMatmulV3PriorityPolicy(const NpuArch npuArch, std::vector<int32_t> &priorities,
                                             const char *opName)
{
    const static std::map<NpuArch, std::vector<int32_t>> MATMUL_V3_PRIOR_MAP = {
        {NpuArch::DAV_3510, {optiling::mc2_matmul_v3_advanced::strategy::BASE}},
    };

    if (MATMUL_V3_PRIOR_MAP.find(npuArch) != MATMUL_V3_PRIOR_MAP.end()) {
        priorities = MATMUL_V3_PRIOR_MAP.at(npuArch);
    }
    if (priorities.empty()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "npuArch", std::to_string(static_cast<uint32_t>(npuArch)).c_str(),
                                              "The value of npuArch must be within the supported range.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace mc2tiling
