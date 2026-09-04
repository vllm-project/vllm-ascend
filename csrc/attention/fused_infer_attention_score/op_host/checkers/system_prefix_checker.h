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
 * \file system_prefix_checker.h
 * \brief
 */

#ifndef SYSTEM_REPFIX_CHECKER_H
#define SYSTEM_REPFIX_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {

class SystemPrefixChecker : public BaseChecker {
public:
    SystemPrefixChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~SystemPrefixChecker() override = default;

    ge::graphStatus CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckCrossFeature(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo) override;

private:
    // singlepara
    ge::graphStatus CheckSharedPrefixDim(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckSharedPrefixDataType(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckSharedPrefixShape(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckActualSharedPrefixLenData(const FiaTilingInfo &fiaInfo);

    // existence
    ge::graphStatus CheckSharedPrefixExistence(const FiaTilingInfo &fiaInfo);

    // feature
    ge::graphStatus CheckUnSupportFeature(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureAntiquant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureAntiquantS1Gt1(const FiaTilingInfo &fiaInfo,
        int64_t keyAntiquantMode, int64_t valueAntiquantMode);
    ge::graphStatus CheckFeatureAntiquantS1Eq1(const FiaTilingInfo &fiaInfo,
        int64_t keyAntiquantMode, int64_t valueAntiquantMode);

    // multipara
private:
};

} // namespace optiling
#endif // SYSTEM_REPFIX_CHECKER_H