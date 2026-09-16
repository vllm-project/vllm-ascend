/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MQSMLA_C624_PRIVATE_SPARSE_MLA_SPARSE_COMPRESSION_CHECKER_H
#define MQSMLA_C624_PRIVATE_SPARSE_MLA_SPARSE_COMPRESSION_CHECKER_H

#include "base_checker_sparse_flash_mla.h"

namespace optiling {
namespace mixed_quant_smla_checker {

class SparseCompressionChecker : public BaseChecker {
public:
    ge::graphStatus CheckSinglePara(const CheckContext &context) const override;
    ge::graphStatus CheckParaExistence(const CheckContext &context) const override;
    ge::graphStatus CheckFeature(const CheckContext &context) const override;
    ge::graphStatus CheckMultiPara(const CheckContext &context) const override;

private:
    ge::graphStatus CheckIndex(const CheckContext &context, const TensorParam &param, const char *name) const;
    ge::graphStatus CheckTopkLength(const CheckContext &context, const TensorParam &param, const char *name) const;
    ge::graphStatus CheckIndexShape(const CheckContext &context, const TensorParam &param, const char *name) const;
    ge::graphStatus CheckTopkLengthShape(const CheckContext &context, const TensorParam &param, const char *name) const;
};

} // namespace mixed_quant_smla_checker
} // namespace optiling

#endif // MQSMLA_C624_PRIVATE_SPARSE_MLA_SPARSE_COMPRESSION_CHECKER_H
