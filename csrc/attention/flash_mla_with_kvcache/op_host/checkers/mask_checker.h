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
 * \file mask_checker.h
 * \brief Checker for mask_mode and attn_mask parameters (文档约束: Mask参数组)
 */

#ifndef FLASH_MLA_WITH_KVCACHE_MASK_CHECKER_H
#define FLASH_MLA_WITH_KVCACHE_MASK_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
namespace flash_mla_with_kvcache {
class MaskChecker : public FlashMlaWithKvcacheBaseChecker {
public:
    MaskChecker() = default;
    ~MaskChecker() override = default;

    ge::graphStatus CheckSinglePara(const FlashMlaWithKvcacheTilingInfo &faInfo) override;
    ge::graphStatus CheckParaExistence(const FlashMlaWithKvcacheTilingInfo &faInfo) override;

private:
    ge::graphStatus CheckSingleParaMaskMode(const FlashMlaWithKvcacheTilingInfo &faInfo);
    ge::graphStatus CheckSingleParaAttnMask(const FlashMlaWithKvcacheTilingInfo &faInfo);
};

} // namespace flash_mla_with_kvcache
} // namespace optiling
#endif // FLASH_MLA_WITH_KVCACHE_MASK_CHECKER_H