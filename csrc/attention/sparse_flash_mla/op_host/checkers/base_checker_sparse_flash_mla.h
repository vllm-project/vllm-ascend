/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPARSE_MLA_BASE_CHECKER_SPARSE_FLASH_MLA_H
#define SPARSE_MLA_BASE_CHECKER_SPARSE_FLASH_MLA_H

#include <initializer_list>
#include <string>
#include "checker_context.h"

namespace optiling {
namespace sparse_mla_checker {

class BaseChecker {
public:
    BaseChecker() = default;
    virtual ~BaseChecker() = default;

    virtual ge::graphStatus CheckSinglePara(const CheckContext &context) const;
    virtual ge::graphStatus CheckParaExistence(const CheckContext &context) const;
    virtual ge::graphStatus CheckFeature(const CheckContext &context) const;
    virtual ge::graphStatus CheckMultiPara(const CheckContext &context) const;

protected:
    ge::graphStatus CheckTensorDesc(const CheckContext &context, const TensorParam &param, const char *name,
                                    std::initializer_list<ge::DataType> dtypes) const;
    ge::graphStatus CheckDimNum(const CheckContext &context, const TensorParam &param, const char *name,
                                std::initializer_list<size_t> dimNums) const;
    ge::graphStatus CheckNoEmptyDim(const CheckContext &context, const TensorParam &param, const char *name) const;
    ge::graphStatus CheckShape(const CheckContext &context, const TensorParam &param, const char *name,
                               std::initializer_list<int64_t> expected) const;
    ge::graphStatus CheckSameShape(const CheckContext &context, const TensorParam &left, const char *leftName,
                                   const TensorParam &right, const char *rightName) const;
    int64_t GetDim(const TensorParam &param, size_t index) const;
    bool CanOmitSequsedOriKv(const CheckContext &context) const;
    bool CanOmitSequsedCmpKv(const CheckContext &context) const;
};

} // namespace sparse_mla_checker
} // namespace optiling

#endif // SPARSE_MLA_BASE_CHECKER_SPARSE_FLASH_MLA_H
