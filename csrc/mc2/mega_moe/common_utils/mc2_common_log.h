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
 * \file mc2_common_log.h
 * \brief
 */

#ifndef MC2_COMMON_LOG_H
#define MC2_COMMON_LOG_H
#include <stdarg.h>
#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>

#include "base/err_msg.h"
#include "securec.h"
#if __has_include("error_manager/error_manager.h")
#include "error_manager/error_manager.h"
#else
#include "err_mgr.h"
#endif

// Forward compatibility: includes log/log.h and provides OP_LOGE_LIBOPAPI_REPORT and
// OP_LOGE_FOR_* macros (EZ0008-EZ0034) when using an older log.h.
#include "mc2_log_compat.h"

struct ErrorResult {
    operator bool() const
    {
        return false;
    }
    operator ge::graphStatus() const
    {
        return ge::GRAPH_PARAM_INVALID;
    }
    template <typename T>
    operator std::unique_ptr<T>() const
    {
        return nullptr;
    }
    template <typename T>
    operator std::shared_ptr<T>() const
    {
        return nullptr;
    }
    template <typename T>
    operator T *() const
    {
        return nullptr;
    }
    template <typename T>
    operator std::vector<std::shared_ptr<T>>() const
    {
        return {};
    }
    template <typename T>
    operator std::vector<T>() const
    {
        return {};
    }
    operator std::string() const
    {
        return "";
    }
    template <typename T>
    operator T() const
    {
        return T();
    }
};

inline std::vector<char> CreateErrorMsg(const char *format, ...) __attribute__((format(printf, 1, 2)));

inline std::vector<char> CreateErrorMsg();

inline std::vector<char> CreateErrorMsg(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    va_list args_copy;
    va_copy(args_copy, args);
    const size_t len = 4095;
    va_end(args_copy);
    std::vector<char> msg(len + 1U, '\0');
    const auto ret = vsnprintf_s(msg.data(), len + 1U, len, format, args);
    va_end(args);
    return (ret > 0) ? msg : std::vector<char>{};
}

inline std::vector<char> CreateErrorMsg()
{
    return {};
}

#if !defined(__ANDROID__) && !defined(ANDROID)
#define OPS_ERR_IF(COND, LOG_FUNC, EXPR)                                                                               \
    if (__builtin_expect(COND, 0)) {                                                                                   \
        LOG_FUNC;                                                                                                      \
        EXPR;                                                                                                          \
    }

#define OPS_CHECK(COND, LOG_FUNC, EXPR)                                                                                \
    if (COND) {                                                                                                        \
        LOG_FUNC;                                                                                                      \
        EXPR;                                                                                                          \
    }

#define OPS_LOG_FULL(LEVEL, OPS_DESC, ...)

#define OPS_LOG_E(opName, ...) D_OP_LOGE(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OPS_LOG_W(opName, ...) D_OP_LOGW(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OPS_LOG_I(opName, ...) D_OP_LOGI(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OPS_LOG_D(opName, ...) D_OP_LOGD(Ops::Base::GetOpInfo(opName), __VA_ARGS__)
#define OP_LOGE_IF(condition, returnValue, opName, fmt, ...)                                                           \
    static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool");       \
    do {                                                                                                               \
        if (unlikely(condition)) {                                                                                     \
            OP_LOGE(Ops::Base::GetOpInfo(opName), fmt, ##__VA_ARGS__);                                                 \
            return returnValue;                                                                                        \
        }                                                                                                              \
    } while (0)
#define OP_LOGI_IF_RETURN(condition, returnValue, opName, fmt, ...)                                                    \
    static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool");       \
    do {                                                                                                               \
        if (unlikely(condition)) {                                                                                     \
            OP_LOGI(Ops::Base::GetOpInfo(opName), fmt, ##__VA_ARGS__);                                                 \
            return returnValue;                                                                                        \
        }                                                                                                              \
    } while (0)
// MC2_CHECK_GRAPH_STATUS — 检查 graphStatus 返回值为 0，上报 EZ9999
#define MC2_CHECK_LOG_RET(opName, callExpr)                                              \
    do {                                                                                       \
        const ge::graphStatus _mc2_gs_ = (callExpr);                                           \
        if (_mc2_gs_ != 0) {                                                                   \
            OP_LOGE(opName, "call %s failed, status=%d", #callExpr, (int)_mc2_gs_);            \
            return ge::FAILED;                                                                 \
        }                                                                                      \
    } while (0)

// MC2_CHECK_NOTNULL — 空指针检查，上报 EZ0004
#define MC2_CHECK_NOTNULL_RET(opName, ptr)                                                         \
    do {                                                                                       \
        if (unlikely((ptr) == nullptr)) {                                                      \
            OP_LOGE_WITH_INVALID_INPUT(opName, #ptr);                                          \
            return ge::FAILED;                                                                 \
        }                                                                                      \
    } while (0)

// MC2_CHECK_TRUE — 布尔条件断言，上报 EZ9999
#define MC2_CHECK_TRUE_RET(opName, condition)                                                      \
    do {                                                                                       \
        if (unlikely(!(condition))) {                                                          \
            OP_LOGE(opName, "assert %s failed", #condition);                                   \
            return ge::FAILED;                                                                 \
        }                                                                                      \
    } while (0)

// MC2_CHECK_SUCCESS — 检查返回值 == ge::SUCCESS，上报 EZ9999
#define MC2_CHECK_SUCCESS_RET(opName, callExpr)                                                    \
    do {                                                                                       \
        const auto _mc2_ret_ = (callExpr);                                                     \
        if (_mc2_ret_ != ge::SUCCESS) {                                                        \
            OP_LOGE(opName, "call %s failed, ret=%d", #callExpr, (int)_mc2_ret_);              \
            return ge::FAILED;                                                                 \
        }                                                                                      \
    } while (0)

// MC2_CHECK_EQ_ERR — 等值断言，返回 ::ErrorResult()，上报 EZ9999
#define MC2_CHECK_EQ_RET(opName, x, y)                                                         \
    do {                                                                                       \
        const auto &_mc2_x_ = (x);                                                             \
        const auto &_mc2_y_ = (y);                                                             \
        if (_mc2_x_ != _mc2_y_) {                                                              \
            std::stringstream _mc2_ss_;                                                        \
            _mc2_ss_ << "Assert (" << #x << " == " << #y << ")failed, expect "                \
                     << _mc2_y_ << " actual " << _mc2_x_;                                      \
            OP_LOGE(opName, "%s", _mc2_ss_.str().c_str());                                     \
            return ::ErrorResult();                                                            \
        }                                                                                      \
    } while (0)

namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                                      \
    if ((ptr) == nullptr) {                                                                                            \
        const char *name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName();                   \
        OP_LOGE_WITHOUT_REPORT(name, "%s is nullptr!", #ptr);                                                          \
        REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                                          \
        return ge::GRAPH_FAILED;                                                                                       \
    }
#define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret)                                                             \
    if ((ptr) == nullptr) {                                                                                            \
        const char *name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName();                   \
        OP_LOGE_WITHOUT_REPORT(name, "%s is nullptr!", #ptr);                                                          \
        REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                                          \
        return ret;                                                                                                    \
    }
} // namespace ops

#else
#define OPS_ERR_IF(COND, LOG_FUNC, EXPR)
#define OPS_CHECK(COND, LOG_FUNC, EXPR)
#define OPS_LOG_FULL(LEVEL, OPS_DESC, ...)
#define OPS_LOG_E(opName, ...)
#define OPS_LOG_W(opName, ...)
#define OPS_LOG_I(opName, ...)
#define OPS_LOG_D(opName, ...)
#define MC2_CHECK_LOG_RET(opName, callExpr)
#define MC2_CHECK_NOTNULL_RET(opName, ptr)
#define MC2_CHECK_TRUE_RET(opName, condition)
#define MC2_CHECK_SUCCESS_RET(opName, callExpr)
#define MC2_CHECK_EQ_RET(opName, x, y)
#endif
#endif