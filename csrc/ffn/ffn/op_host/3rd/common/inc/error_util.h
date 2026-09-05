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
 * \file error_util.h
 * \brief
 */
#ifndef COMMON_INC_ERROR_UTIL_H_
#define COMMON_INC_ERROR_UTIL_H_

#include "log/log.h"
#include "graph/operator.h"

#define CUBE_INNER_ERR_REPORT(opName, errMsg, ...)                                                             \
    do {                                                                                                       \
        OP_LOGE_WITHOUT_REPORT(opName, errMsg, ##__VA_ARGS__);                                                 \
        REPORT_INNER_ERR_MSG("E69999", "op[%s], " errMsg, Ops::Base::GetSafeStr(Ops::Base::GetOpInfo(opName)), \
                             ##__VA_ARGS__);                                                                   \
    } while (0)

#define VECTOR_INNER_ERR_REPORT_TILIING(opName, errMsg, ...)                                                   \
    do {                                                                                                       \
        OP_LOGE_WITHOUT_REPORT(opName, errMsg, ##__VA_ARGS__);                                                 \
        REPORT_INNER_ERR_MSG("E89999", "op[%s], " errMsg, Ops::Base::GetSafeStr(Ops::Base::GetOpInfo(opName)), \
                             ##__VA_ARGS__);                                                                   \
    } while (0)

#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)

#define OPS_LOG_D(OPS_DESC, fmt, ...) \
    D_OP_LOGD(Ops::Base::GetSafeStr(Ops::Base::GetOpInfo(OPS_DESC)), fmt, ##__VA_ARGS__)
#define OPS_LOG_I(OPS_DESC, fmt, ...) \
    D_OP_LOGI(Ops::Base::GetSafeStr(Ops::Base::GetOpInfo(OPS_DESC)), fmt, ##__VA_ARGS__)
#define OPS_LOG_W(OPS_DESC, fmt, ...) \
    D_OP_LOGW(Ops::Base::GetSafeStr(Ops::Base::GetOpInfo(OPS_DESC)), fmt, ##__VA_ARGS__)
#define OPS_LOG_E(OPS_DESC, fmt, ...) \
    D_OP_LOGE(Ops::Base::GetSafeStr(Ops::Base::GetOpInfo(OPS_DESC)), fmt, ##__VA_ARGS__)

#define OPS_ERR_IF(COND, LOG_FUNC, EXPR) \
    if (__builtin_expect(COND, 0)) {     \
        LOG_FUNC;                        \
        EXPR;                            \
    }

template <typename T>
std::string Shape2String(const T& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

#define OPS_CHECK_NULL_WITH_CONTEXT OP_CHECK_NULL_WITH_CONTEXT
#define OPS_REPORT_VECTOR_INNER_ERR(OPS_DESC, ...) OPS_LOG_E(OPS_DESC, ##__VA_ARGS__)
#define OPS_REPORT_CUBE_INNER_ERR(OPS_DESC, ...) OPS_LOG_E(OPS_DESC, ##__VA_ARGS__)

#define OP_LOGE_IF(condition, returnValue, opName, fmt, ...)                                                     \
    static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
    do {                                                                                                         \
        if (unlikely(condition)) {                                                                               \
            OP_LOGE(opName, fmt, ##__VA_ARGS__);                                                                 \
            return returnValue;                                                                                  \
        }                                                                                                        \
    } while (0)

#define OP_LOGW_IF(condition, opName, fmt, ...)                                                                  \
    static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
    do {                                                                                                         \
        if (unlikely(condition)) {                                                                               \
            OP_LOGW(opName, fmt, ##__VA_ARGS__);                                                                 \
        }                                                                                                        \
    } while (0)

#define OP_LOGI_IF_RETURN(condition, returnValue, opName, fmt, ...)                                              \
    static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
    do {                                                                                                         \
        if (unlikely(condition)) {                                                                               \
            OP_LOGI(opName, fmt, ##__VA_ARGS__);                                                                 \
            return returnValue;                                                                                  \
        }                                                                                                        \
    } while (0)

#endif // COMMON_INC_ERROR_UTIL_H_
