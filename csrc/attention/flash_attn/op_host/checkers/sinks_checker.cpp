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
 * \file sinks_checker.cpp
 * \brief Checker for sinks parameter (文档参数名: sinks, 文档约束: LearnableSink参数组)
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fa_tiling_info.h"
#include "sinks_checker_flash_attn.h"

namespace optiling {
namespace flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

ge::graphStatus SinksChecker::CheckSinglePara(const FaTilingInfo &faInfo)
{
    if (faInfo.sinksFlag) {
        if (faInfo.opParamInfo.sinks.tensor == nullptr || faInfo.opParamInfo.sinks.desc == nullptr) {
            OP_LOGE(faInfo.opName, "sinks tensor or desc is null.");
            return ge::GRAPH_FAILED;
        }
        auto dtype = faInfo.opParamInfo.sinks.desc->GetDataType();
        if (dtype != ge::DT_FLOAT) {
            OP_LOGE(faInfo.opName, "sinks dtype must be FLOAT32, but got %s.",
                    ge::TypeUtils::DataTypeToSerialString(dtype).c_str());
            return ge::GRAPH_FAILED;
        }
        auto &shape = faInfo.opParamInfo.sinks.tensor->GetStorageShape();
        if (shape.GetDimNum() != 1) {
            OP_LOGE(faInfo.opName, "sinks shape must be 1D (Q_N,), but got %lu dims.", shape.GetDimNum());
            return ge::GRAPH_FAILED;
        }
        if (static_cast<uint64_t>(shape.GetDim(0)) != static_cast<uint64_t>(faInfo.n1Size)) {
            OP_LOGE(faInfo.opName, "sinks shape[0] must equal Q_N (%ld), but got %ld.", faInfo.n1Size, shape.GetDim(0));
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace flash_attn
} // namespace optiling
