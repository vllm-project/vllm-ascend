/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file mask_checker.h
 * \brief
 */

#ifndef MASK_CHECKER_H
#define MASK_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {
class MaskChecker : public BaseChecker {
public:
    MaskChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~MaskChecker() override = default;

    ge::graphStatus CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckCrossFeature(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo) override;

private:
    // 公共校验函数
    struct MaskInfo {
        int64_t attenMaskN = 1U;
        uint32_t attenMaskBatch = 1;
        uint32_t attenMaskQSize = 0;
        uint32_t attenMaskSize = 0;
        std::string strMaskShape;
    };
    ge::graphStatus CheckDtypeAndFormat(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckSparseMode(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckNoQuantIFAMLA(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFullQuantIFAMLA(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckMXFP8FullQuant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFP8GQAFullQuant(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckQKVDDifferent(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckFeatureSparseMode(const FiaTilingInfo &fiaInfo) const;
    ge::graphStatus CheckPretokenAndNexttoken(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckIFADimAndShape(const FiaTilingInfo &fiaInfo) const;
        ge::graphStatus CheckIFAMask2DShape(const FiaTilingInfo &fiaInfo, const std::string &layoutStr) const;
    ge::graphStatus CheckIFAMask3DShape(const FiaTilingInfo &fiaInfo, const gert::Tensor *maskShape,
        uint32_t minAttenMaskSize) const;
    ge::graphStatus CheckIFAMask4DShape(const FiaTilingInfo &fiaInfo, const gert::Tensor *maskShape,
        uint32_t minAttenMaskSize) const;
    ge::graphStatus GetMaskInfo(const FiaTilingInfo &fiaInfo, MaskInfo &maskInfo) const;
    ge::graphStatus CheckAndProcessTreeMask(const FiaTilingInfo &fiaInfo, MaskInfo &maskInfo) const;
    ge::graphStatus CheckAndProcess2DMask(const FiaTilingInfo &fiaInfo, const gert::Tensor *maskShape,
                                          MaskInfo &maskInfo) const;
    ge::graphStatus CheckDimAndShape(const FiaTilingInfo &fiaInfo);
    ge::graphStatus ValidateMaskDimAndShape(const FiaTilingInfo &fiaInfo, const MaskInfo &maskInfo) const;
    ge::graphStatus CheckAntiquantSparseMode(const FiaTilingInfo &fiaInfo) const;
private:
    bool enableIFAMLA = false;
    bool isIFAFlag = false;
    bool enableMXFP8 = false;
    bool enableFP8GQAFullQuant = false;
};

}  // namespace optiling
#endif  // MASK_CHECKER_H