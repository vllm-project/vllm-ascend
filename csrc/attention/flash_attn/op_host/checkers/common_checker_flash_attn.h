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
 * \file common_checker_flash_attn.h
 * \brief Common checker for layout, shape, dtype, and scalar attr parameters
 */

#ifndef FLASH_ATTN_COMMON_CHECKER_FLASH_ATTN_H
#define FLASH_ATTN_COMMON_CHECKER_FLASH_ATTN_H

#include <map>
#include <memory>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker_flash_attn.h"

namespace optiling {
namespace flash_attn {

class CommonChecker : public FABaseChecker {
public:
    CommonChecker() = default;
    ~CommonChecker() override = default;

    ge::graphStatus CheckSinglePara(const FaTilingInfo &faInfo) override;

    ge::graphStatus CheckParaExistence(const FaTilingInfo &faInfo) override;

    ge::graphStatus CheckMultiPara(const FaTilingInfo &faInfo) override;

    void SetFaShapeCompare(const FaTilingInfo &faInfo);
    ge::graphStatus CheckQueryShape(const FaTilingInfo &faInfo) const;
    ge::graphStatus CheckKVShape(const FaTilingInfo &faInfo) const;
    ge::graphStatus CheckAttnOutShape(const FaTilingInfo &faInfo) const;

private:
    // --- Layout checks ---
    ge::graphStatus CheckSingleParaLayout(const FaTilingInfo &faInfo);
    ge::graphStatus CheckMultiParaLayout(const FaTilingInfo &faInfo);

    // --- Shape/dt checks ---
    ge::graphStatus CheckNonQuantDataType(const FaTilingInfo &faInfo);
    ge::graphStatus CheckNonQuantHeadNum(const FaTilingInfo &faInfo);
    ge::graphStatus CheckDtypeConsistency(const FaTilingInfo &faInfo);
    ge::graphStatus CheckAxis(const FaTilingInfo &faInfo);
    ge::graphStatus CheckShapeConsistency(const FaTilingInfo &faInfo);

    ge::graphStatus CheckKVShapeForContinuous(const FaTilingInfo &faInfo) const;
    ge::graphStatus CheckKVShapeForPageAttention(const FaTilingInfo &faInfo) const;

    // --- Attr checks ---

    std::shared_ptr<FaTilingShapeCompare> queryShapeCmp_;
    std::shared_ptr<FaTilingShapeCompare> keyShapeCmp_;
    std::shared_ptr<FaTilingShapeCompare> valueShapeCmp_;
    std::shared_ptr<FaTilingShapeCompare> attnOutShapeCmp_;
};

} // namespace flash_attn
} // namespace optiling
#endif // FLASH_ATTN_COMMON_CHECKER_FLASH_ATTN_H
