/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rope_checker.h
 * \brief
 */

#ifndef ROPE_CHECKER_H
#define ROPE_CHECKER_H

#include <map>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
class RopeChecker : public BaseChecker {
public:
    RopeChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~RopeChecker() override = default;

    ge::graphStatus CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckCrossFeature(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo) override;

private:
    // 公共校验函数
    ge::graphStatus CheckQDsizeSupport(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckRopeDSizeSupport(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckRopeDtype(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckRopeDtypeConsistency(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckKRopeContiguous(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckQKAndQKRopeShapeConsistency(const FiaTilingInfo &fiaInfo,
        const gert::Shape shape, const gert::Shape ropeShape, const std::string &inputName) const;
    ge::graphStatus CheckPAKeyAndKeyRopeShapeConsistency(const FiaTilingInfo &fiaInfo,
        const gert::Shape &keyShape, const gert::Shape &keyRopeShape) const;
    ge::graphStatus CheckTensorlistKeyAndKeyRopeShapeConsistency(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckRopeExistence(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckRopeConsistency(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureDecodeMLA(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureSupport(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureAntiQuant(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckShapeConsistency(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckQSSize(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckNSize(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckAxisSupport(const FiaTilingInfo &fiaInfo);

private:
};

} // namespace optiling
#endif // ROPE_CHECKER_H