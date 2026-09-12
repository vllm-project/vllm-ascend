/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPARSE_MLA_SEQ_LEN_CHECKER_SPARSE_FLASH_MLA_H
#define SPARSE_MLA_SEQ_LEN_CHECKER_SPARSE_FLASH_MLA_H

#include "base_checker_sparse_flash_mla.h"

namespace optiling {
namespace sparse_mla_checker {

class SeqLenChecker : public BaseChecker {
public:
    ge::graphStatus CheckSinglePara(const CheckContext &context) const override;
    ge::graphStatus CheckParaExistence(const CheckContext &context) const override;
    ge::graphStatus CheckMultiPara(const CheckContext &context) const override;

private:
    ge::graphStatus CheckLengthTensor(const CheckContext &context, const TensorParam &param, const char *name) const;
    ge::graphStatus CheckLength(const CheckContext &context, const TensorParam &param, const char *name,
                                int64_t expected) const;
};

} // namespace sparse_mla_checker
} // namespace optiling

#endif // SPARSE_MLA_SEQ_LEN_CHECKER_SPARSE_FLASH_MLA_H
