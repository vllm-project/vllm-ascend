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
 * \file common_checker.h
 * \brief Common checker for layout, shape, dtype, and scalar attr parameters
 */

#ifndef FLASH_MLA_WITH_KVCACHE_COMMON_CHECKER_H
#define FLASH_MLA_WITH_KVCACHE_COMMON_CHECKER_H

#include <map>
#include <memory>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
namespace flash_mla_with_kvcache {

class CommonChecker : public FlashMlaWithKvcacheBaseChecker {
public:
    CommonChecker() = default;
    ~CommonChecker() override = default;

    ge::graphStatus CheckSinglePara(const FlashMlaWithKvcacheTilingInfo &faInfo) override;

    ge::graphStatus CheckParaExistence(const FlashMlaWithKvcacheTilingInfo &faInfo) override;

    ge::graphStatus CheckMultiPara(const FlashMlaWithKvcacheTilingInfo &faInfo) override;

    void SetFaShapeCompare(const FlashMlaWithKvcacheTilingInfo &faInfo);
    ge::graphStatus CheckQueryShape(const FlashMlaWithKvcacheTilingInfo &faInfo) const;
    ge::graphStatus CheckKVShape(const FlashMlaWithKvcacheTilingInfo &faInfo) const;
    ge::graphStatus CheckAttnOutShape(const FlashMlaWithKvcacheTilingInfo &faInfo) const;

private:
    // --- Layout checks ---
    ge::graphStatus CheckSingleParaLayout(const FlashMlaWithKvcacheTilingInfo &faInfo);
    ge::graphStatus CheckMultiParaLayout(const FlashMlaWithKvcacheTilingInfo &faInfo);

    // --- Shape/dt checks ---
    ge::graphStatus CheckNonQuantDataType(const FlashMlaWithKvcacheTilingInfo &faInfo);
    ge::graphStatus CheckNonQuantHeadNum(const FlashMlaWithKvcacheTilingInfo &faInfo);
    ge::graphStatus CheckDtypeConsistency(const FlashMlaWithKvcacheTilingInfo &faInfo);
    ge::graphStatus CheckAxis(const FlashMlaWithKvcacheTilingInfo &faInfo);
    ge::graphStatus CheckMlaGeometry(const FlashMlaWithKvcacheTilingInfo &faInfo);
    ge::graphStatus CheckShapeConsistency(const FlashMlaWithKvcacheTilingInfo &faInfo);

    ge::graphStatus CheckKVShapeForPageAttention(const FlashMlaWithKvcacheTilingInfo &faInfo) const;

    // --- Attr checks ---

    std::shared_ptr<FlashMlaWithKvcacheTilingShapeCompare> queryShapeCmp_;
    std::shared_ptr<FlashMlaWithKvcacheTilingShapeCompare> keyShapeCmp_;
    std::shared_ptr<FlashMlaWithKvcacheTilingShapeCompare> attnOutShapeCmp_;
};

} // namespace flash_mla_with_kvcache
} // namespace optiling
#endif // FLASH_MLA_WITH_KVCACHE_COMMON_CHECKER_H