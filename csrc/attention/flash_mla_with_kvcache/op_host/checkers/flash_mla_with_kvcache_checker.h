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
 * \file flash_mla_with_kvcache_checker.h
 * \brief FlashMlaWithKvcacheChecker using Composite Pattern - manages all checkers uniformly
 */

#ifndef FLASH_MLA_WITH_KVCACHE_FA_CHECKER_H
#define FLASH_MLA_WITH_KVCACHE_FA_CHECKER_H

#include <memory>
#include <vector>
#include <functional>
#include "tiling/tiling_api.h"
#include "base_checker.h"

#include "./seq_len_checker.h"
#include "./common_checker.h"
#include "./mask_checker.h"
#include "./metadata_checker.h"
#include "./paged_attention_checker.h"
#include "./softmax_lse_checker.h"

namespace optiling {
namespace flash_mla_with_kvcache {

class FlashMlaWithKvcacheChecker {
public:
    FlashMlaWithKvcacheChecker() = default;
    ~FlashMlaWithKvcacheChecker() = default;

    ge::graphStatus Init(const FlashMlaWithKvcacheTilingInfo &faInfo);
    ge::graphStatus Process(const FlashMlaWithKvcacheTilingInfo &faInfo);

    ge::graphStatus CheckSinglePara(const FlashMlaWithKvcacheTilingInfo &faInfo);
    ge::graphStatus CheckParaExistence(const FlashMlaWithKvcacheTilingInfo &faInfo);
    ge::graphStatus CheckFeature(const FlashMlaWithKvcacheTilingInfo &faInfo);
    ge::graphStatus CheckMultiPara(const FlashMlaWithKvcacheTilingInfo &faInfo);

private:
    using CheckMethod = std::function<ge::graphStatus(FlashMlaWithKvcacheBaseChecker *, const FlashMlaWithKvcacheTilingInfo &)>;

    ge::graphStatus RunCheck(const CheckMethod &method, const FlashMlaWithKvcacheTilingInfo &faInfo);

    void RegisterCheckers();

    std::vector<std::unique_ptr<FlashMlaWithKvcacheBaseChecker>> checkers_;
};

} // namespace flash_mla_with_kvcache
} // namespace optiling
#endif // FLASH_MLA_WITH_KVCACHE_FA_CHECKER_H