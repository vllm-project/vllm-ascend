/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file legacy_common_manager.h
 * \brief
 */
#ifndef LEGACY_COMMON_MANAGER_H
#define LEGACY_COMMON_MANAGER_H

#include <string>
#include <dlfcn.h>
#include "mc2_log_compat.h"

namespace Ops {
namespace MC2 {
class LegacyCommonMgr {
public:
    // 删除复制构造函数和赋值运算符
    LegacyCommonMgr(const LegacyCommonMgr &) = delete;
    LegacyCommonMgr &operator=(const LegacyCommonMgr &) = delete;

    // 获取单例实例
    static LegacyCommonMgr &GetInstance()
    {
        static LegacyCommonMgr instance;
        return instance;
    }

    /**
     * @brief 获取函数指针
     * @tparam FuncType 函数指针类型
     * @param symbolName 符号名称
     * @return 函数指针
     */
    template <typename FuncType>
    FuncType GetFunc(const char *symbolName) const
    {
        if (symbolName == nullptr || strlen(symbolName) == 0) {
            OP_LOGW("LegacyCommonMgr", "Invalid symbol name (null or empty).");
            return nullptr;
        }
        OP_LOGI("LegacyCommonMgr", "try to find symbol %s.", symbolName);
        if (handle_ == nullptr) {
            return nullptr;
        }

        dlerror(); // 清除之前的错误

        void *symbol = dlsym(handle_, symbolName);
        if (symbol == nullptr) {
            OP_LOGW("LegacyCommonMgr", "Fail to find symbol %s.", symbolName);
            return nullptr;
        }
        OP_LOGI("LegacyCommonMgr", "Find symbol %s success.", symbolName);
        return reinterpret_cast<FuncType>(symbol);
    }

private:
    // 私有构造函数
    LegacyCommonMgr();
    ~LegacyCommonMgr();

    bool GetLegacyCommonSoPath(std::string &soPath) const;
    std::string GetCpuArch() const;

    void *handle_{nullptr};
};
} // namespace MC2
} // namespace Ops

#endif // LEGACY_COMMON_MANAGER_H
