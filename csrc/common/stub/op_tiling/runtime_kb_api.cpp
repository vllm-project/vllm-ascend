/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime_kb_api.h"

namespace RuntimeKb {
uint32_t QueryBank(
    const void* /*src*/, size_t /*src_len*/, const std::string& /*op_type*/, const std::string& /*soc_version*/,
    uint32_t /*core_num*/, tuningtiling::TuningTilingDefPtr& /*tiling*/)
{
    return 0;
}
} // namespace RuntimeKb