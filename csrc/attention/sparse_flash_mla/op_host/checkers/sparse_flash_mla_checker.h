/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPARSE_FLASH_MLA_CHECKER_H
#define SPARSE_FLASH_MLA_CHECKER_H

#include "../sparse_flash_mla_tiling.h"
#include "log/error_code.h"

namespace optiling {

class SparseFlashMlaChecker {
public:
    explicit SparseFlashMlaChecker(const SMLATilingInfo &info)
        : info_(info)
    {}
    ge::graphStatus Process() const;

private:
    const SMLATilingInfo &info_;
};

} // namespace optiling

#endif // SPARSE_FLASH_MLA_CHECKER_H
