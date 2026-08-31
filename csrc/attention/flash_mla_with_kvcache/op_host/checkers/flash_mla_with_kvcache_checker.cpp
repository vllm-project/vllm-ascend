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
 * \file flash_mla_with_kvcache_checker.cpp
 * \brief FlashMlaWithKvcacheChecker implementation using Composite Pattern
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_mla_with_kvcache_tiling_info.h"
#include "flash_mla_with_kvcache_checker.h"
#include "common_checker.h"

namespace optiling {
namespace flash_mla_with_kvcache {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35MLA;

// RegisterCheckers - 注册所有 checker
// 新增 checker 时在此方法中添加: checkers_.push_back(std::make_unique<XxxChecker>());
void FlashMlaWithKvcacheChecker::RegisterCheckers()
{
    checkers_.push_back(std::make_unique<CommonChecker>());
    checkers_.push_back(std::make_unique<ActualSeqLenChecker>());
    checkers_.push_back(std::make_unique<PagedAttentionChecker>());
    checkers_.push_back(std::make_unique<MaskChecker>());
    checkers_.push_back(std::make_unique<MetadataChecker>());
    checkers_.push_back(std::make_unique<SoftmaxLSEChecker>());
}

ge::graphStatus FlashMlaWithKvcacheChecker::Init(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    (void)faInfo;
    RegisterCheckers();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheChecker::RunCheck(const CheckMethod &method, const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    for (const auto &checker : checkers_) {
        if (ge::GRAPH_SUCCESS != method(checker.get(), faInfo)) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashMlaWithKvcacheChecker::CheckSinglePara(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    return RunCheck([](FlashMlaWithKvcacheBaseChecker *c, const FlashMlaWithKvcacheTilingInfo &info) { return c->CheckSinglePara(info); }, faInfo);
}

ge::graphStatus FlashMlaWithKvcacheChecker::CheckParaExistence(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    return RunCheck([](FlashMlaWithKvcacheBaseChecker *c, const FlashMlaWithKvcacheTilingInfo &info) { return c->CheckParaExistence(info); }, faInfo);
}

ge::graphStatus FlashMlaWithKvcacheChecker::CheckFeature(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    return RunCheck([](FlashMlaWithKvcacheBaseChecker *c, const FlashMlaWithKvcacheTilingInfo &info) { return c->CheckFeature(info); }, faInfo);
}

ge::graphStatus FlashMlaWithKvcacheChecker::CheckMultiPara(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    return RunCheck([](FlashMlaWithKvcacheBaseChecker *c, const FlashMlaWithKvcacheTilingInfo &info) { return c->CheckMultiPara(info); }, faInfo);
}

ge::graphStatus FlashMlaWithKvcacheChecker::Process(const FlashMlaWithKvcacheTilingInfo &faInfo)
{
    if (faInfo.emptyTensorFlag) {
        OP_LOGE(faInfo.opName, "Empty tensor detected: q, k_cache or attn_out has dimension with size 0");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckSinglePara(faInfo)) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckParaExistence(faInfo)) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckFeature(faInfo)) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckMultiPara(faInfo)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace flash_mla_with_kvcache
} // namespace optiling