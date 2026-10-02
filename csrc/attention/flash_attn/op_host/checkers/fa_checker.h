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
 * \file fa_checker.h
 * \brief FAChecker using Composite Pattern - manages all checkers uniformly
 */

#ifndef FLASH_ATTN_FA_CHECKER_H
#define FLASH_ATTN_FA_CHECKER_H

#include <memory>
#include <vector>
#include <functional>
#include "tiling/tiling_api.h"
#include "base_checker_flash_attn.h"

#include "seq_len_checker_flash_attn.h"
#include "common_checker_flash_attn.h"
#include "mask_checker_flash_attn.h"
#include "metadata_checker_flash_attn.h"
#include "paged_attention_checker_flash_attn.h"
#include "sinks_checker_flash_attn.h"
#include "softmax_lse_checker_flash_attn.h"

namespace optiling {
namespace flash_attn {

class FAChecker {
public:
    FAChecker() = default;
    ~FAChecker() = default;

    ge::graphStatus Init(const FaTilingInfo &faInfo);
    ge::graphStatus Process(const FaTilingInfo &faInfo);

    ge::graphStatus CheckSinglePara(const FaTilingInfo &faInfo);
    ge::graphStatus CheckParaExistence(const FaTilingInfo &faInfo);
    ge::graphStatus CheckFeature(const FaTilingInfo &faInfo);
    ge::graphStatus CheckMultiPara(const FaTilingInfo &faInfo);

private:
    using CheckMethod = std::function<ge::graphStatus(FABaseChecker *, const FaTilingInfo &)>;

    ge::graphStatus RunCheck(const CheckMethod &method, const FaTilingInfo &faInfo);

    void RegisterCheckers();

    std::vector<std::unique_ptr<FABaseChecker>> checkers_;
};

} // namespace flash_attn
} // namespace optiling
#endif // FLASH_ATTN_FA_CHECKER_H
