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
 * \file base_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "../flash_mla_with_kvcache_tiling_info.h"
#include "base_checker.h"
#include "log/log.h"
namespace optiling {
namespace flash_mla_with_kvcache {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35MLA;

ge::graphStatus FlashMlaWithKvcacheBaseChecker::CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc, const std::string &name) const
{
    if (desc != nullptr) {
        const auto &it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(
            it == DTYPE_SUPPORT_MAP.end(),
            OP_LOGE("FlashMlaWithKvcache", "%s datatype support list should be specify in DTYPE_SUPPORT_MAP", name.c_str()),
            return ge::GRAPH_FAILED);
        auto &expectDtypeList = it->second;
        if (std::find(expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end()) {
            std::string dtypeStr = DataTypeToSerialString(desc->GetDataType());
            std::string expectedDtypes;
            for (size_t i = 0; i < expectDtypeList.size(); i++) {
                if (i > 0) expectedDtypes += ", ";
                expectedDtypes += DataTypeToSerialString(expectDtypeList[i]);
            }
            OP_LOGE_FOR_INVALID_DTYPE("FlashMlaWithKvcache", name.c_str(), dtypeStr.c_str(), expectedDtypes.c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheBaseChecker::CheckFormatSupport(const gert::CompileTimeTensorDesc *desc,
                                                  const std::string &name) const
{
    if (desc != nullptr) {
        auto format = desc->GetOriginFormat();
        OP_CHECK_IF((FORMAT_SUPPORT_SET.find(format) == FORMAT_SUPPORT_SET.end()),
            OP_LOGE_FOR_INVALID_FORMAT("FlashMlaWithKvcache", name.c_str(), Ops::Base::ToString(format).c_str(), "ND"),
                return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

std::string FlashMlaWithKvcacheBaseChecker::DataTypeToSerialString(ge::DataType type) const
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("FlashMlaWithKvcache", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

uint32_t FlashMlaWithKvcacheBaseChecker::GetTypeSize(ge::DataType dtype)
{
    constexpr uint32_t NUM_BYTES_FLOAT = 4;
    constexpr uint32_t NUM_BYTES_FLOAT16 = 2;
    constexpr uint32_t NUM_BYTES_INT8 = 1;

    switch (dtype) {
        case ge::DT_FLOAT:
            return NUM_BYTES_FLOAT;
        case ge::DT_FLOAT16:
            return NUM_BYTES_FLOAT16;
        case ge::DT_INT8:
        case ge::DT_UINT8:
            return NUM_BYTES_INT8;
        default:
            return NUM_BYTES_FLOAT16;
    }
}
} // namespace flash_mla_with_kvcache
} // namespace optiling
