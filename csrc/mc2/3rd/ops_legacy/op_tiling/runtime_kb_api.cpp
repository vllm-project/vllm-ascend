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
#ifdef BUILD_OPEN_PROJECT
#include "legacy_common_manager.h"
#include "mc2_log_compat.h"
#endif

namespace RuntimeKb {
#ifdef BUILD_OPEN_PROJECT
uint32_t QueryBank(const void *src, size_t src_len, const std::string &op_type, const std::string &soc_version,
                   uint32_t core_num, tuningtiling::TuningTilingDefPtr &tiling)
{
    using FuncType = uint32_t (*)(const void *, size_t, const std::string &, const std::string &, uint32_t,
                                  tuningtiling::TuningTilingDefPtr &);
    const char *symbolName = "LegacyQueryBank";
    static FuncType func = Ops::MC2::LegacyCommonMgr::GetInstance().GetFunc<FuncType>(symbolName);
    if (func == nullptr) {
        OP_LOGW("LegacyCommonMgr", "dest func %s pointer is null.", symbolName);
        return 0xFFU; // 0: succ, 1: kye not exists, 0xFFU: fail
    } else {
        return func(src, src_len, op_type, soc_version, core_num, tiling);
    }
}
#else
uint32_t QueryBank(const void * /*src*/, size_t /*src_len*/, const std::string & /*op_type*/,
                   const std::string & /*soc_version*/, uint32_t /*core_num*/,
                   tuningtiling::TuningTilingDefPtr & /*tiling*/)
{
    return 0;
}
#endif
} // namespace RuntimeKb
