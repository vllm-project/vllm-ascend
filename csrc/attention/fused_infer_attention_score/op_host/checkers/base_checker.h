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
 * \file base_checker.h
 * \brief
 */

#ifndef BASE_CHECKER_H
#define BASE_CHECKER_H

#include <numeric>
#include "tiling/tiling_api.h"

#include "../../../common/op_host/fia_tiling_info.h"
#include "../../../common/op_host/fia_tiling_shape.h"
#include "../fused_infer_attention_score_tiling_utils.h"

namespace optiling {
class BaseChecker {
public:
    BaseChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
                enableNonQuant_(enableNonQuant),
                enableFullQuant_(enableFullQuant),
                enableAntiQuant_(enableAntiQuant) {}
    virtual ~BaseChecker() = default;

protected:
    virtual ge::graphStatus CheckSinglePara(const FiaTilingInfo &fiaInfo) = 0;
    virtual ge::graphStatus CheckParaExistence(const FiaTilingInfo &fiaInfo) = 0;
    virtual ge::graphStatus CheckCrossFeature(const FiaTilingInfo &fiaInfo) = 0;
    virtual ge::graphStatus CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo) = 0;

    // public check funcs
    ge::graphStatus CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc, const std::string &name) const;
    ge::graphStatus CheckFormatSupport(const gert::CompileTimeTensorDesc *desc, const std::string &name) const;
    template <typename T>
    ge::graphStatus CheckValueSupport(const T value, const std::vector<T> &expectValList) const;
    ge::graphStatus CheckTensorContiguous(const uint32_t &tensorDimNum, const gert::Shape &inputShape,
        const gert::Stride *Strides, int32_t &index) const;

    // public funcs
    std::string DataTypeToSerialString(ge::DataType type) const;

protected:
    bool enableNonQuant_ = false;
    bool enableFullQuant_ = false;
    bool enableAntiQuant_ = false;
};
} // namespace optiling
#endif // BASE_CHECKER_H

