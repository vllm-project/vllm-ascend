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
 * \file base_checker.h
 * \brief Base checker class for npu_flash_mla_with_kvcache parameters
 */

#ifndef FLASH_MLA_WITH_KVCACHE_BASE_CHECKER_H
#define FLASH_MLA_WITH_KVCACHE_BASE_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"

#include "../flash_mla_with_kvcache_tiling_info.h"
#include "../flash_mla_with_kvcache_tiling_utils.h"

namespace optiling {
namespace flash_mla_with_kvcache {
class FlashMlaWithKvcacheBaseChecker {
public:
    FlashMlaWithKvcacheBaseChecker() = default;
    virtual ~FlashMlaWithKvcacheBaseChecker() = default;

    virtual ge::graphStatus CheckSinglePara(const FlashMlaWithKvcacheTilingInfo &faInfo)
    {
        (void)faInfo;
        return ge::GRAPH_SUCCESS;
    }
    virtual ge::graphStatus CheckParaExistence(const FlashMlaWithKvcacheTilingInfo &faInfo)
    {
        (void)faInfo;
        return ge::GRAPH_SUCCESS;
    }
    virtual ge::graphStatus CheckFeature(const FlashMlaWithKvcacheTilingInfo &faInfo)
    {
        (void)faInfo;
        return ge::GRAPH_SUCCESS;
    }
    virtual ge::graphStatus CheckMultiPara(const FlashMlaWithKvcacheTilingInfo &faInfo)
    {
        (void)faInfo;
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc, const std::string &name) const;
    ge::graphStatus CheckFormatSupport(const gert::CompileTimeTensorDesc *desc, const std::string &name) const;
    template <typename T>
    ge::graphStatus CheckValueSupport(const T value, const std::vector<T> &expectValList) const
    {
        if (std::find(expectValList.begin(), expectValList.end(), value) == expectValList.end()) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    std::string DataTypeToSerialString(ge::DataType type) const;
    static uint32_t GetTypeSize(ge::DataType dtype);
};
} // namespace flash_mla_with_kvcache
} // namespace optiling
#endif // FLASH_MLA_WITH_KVCACHE_BASE_CHECKER_H
