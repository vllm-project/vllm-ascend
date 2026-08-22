/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file paged_attention_checker.h
 * \brief
 */

#ifndef PAGED_ATTENTION_CHECKER_H
#define PAGED_ATTENTION_CHECKER_H

#include <map>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {

class PagedAttentionChecker : public BaseChecker {
public:
    PagedAttentionChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~PagedAttentionChecker() override = default;

    ge::graphStatus CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckCrossFeature(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo) override;

private:
    // 公共校验函数

    ge::graphStatus CheckBlockTableDtype(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckBlockTableShapeSize(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckBlockSize(const FiaTilingInfo &fiaInfo) const;


    ge::graphStatus CheckBlockTableExistence(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureSupport(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckSeqLengthKVExistence(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckQDtypeSupport(const FiaTilingInfo &fiaInfo);

    int64_t GetMaxBlockNumPerBatch(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckMaskShape(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckPseShape(const FiaTilingInfo &fiaInfo);

    ge::graphStatus CheckBlockTableShape(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckBlockSizeSupport(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckKVLayout(const FiaTilingInfo &fiaInfo) const;

    ge::graphStatus CheckPACacheShape3D(const FiaTilingInfo &fiaInfo, const gert::Shape &tempShape,
        const std::string &inputName, uint32_t compareD, const std::string &shapeStr) const;
    ge::graphStatus CheckPACacheShape4D(const FiaTilingInfo &fiaInfo, const gert::Shape &tempShape,
        const std::string &inputName, uint32_t compareD, const std::string &shapeStr) const;
    ge::graphStatus CheckPACacheShapeNZAntiquant(const FiaTilingInfo &fiaInfo,
        const std::string &inputName, const std::string &shapeStr, uint32_t compareD,
        int64_t tempD0, int64_t tempD1) const;
    ge::graphStatus CheckPACacheShapeNZNonAntiquant(const FiaTilingInfo &fiaInfo,
        const std::string &inputName, const std::string &shapeStr, uint32_t compareD,
        int64_t tempD0, int64_t tempD1) const;
    ge::graphStatus CheckPACacheShape(const FiaTilingInfo &fiaInfo, const gert::Shape tempShape,
        const std::string& inputName) const;
    ge::graphStatus CheckBlockSizeNonQuant910B(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckBlockSizeNonQuantOther(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckBlockSizeAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureQueryS(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureInputLayoutForAntiquant(const FiaTilingInfo &fiaInfo) const;
};

} // namespace optiling
#endif // PAGED_ATTENTION_CHECKER_H