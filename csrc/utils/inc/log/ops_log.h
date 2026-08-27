/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ops_log.h
 * \brief
 */

#pragma once

#include "log/inner/dfx_base.h"

/* 基础日志 */
#define OPS_LOG_D(OPS_DESC, ...) OPS_LOG_STUB_D(OPS_DESC, __VA_ARGS__)
#define OPS_LOG_I(OPS_DESC, ...) OPS_LOG_STUB_I(OPS_DESC, __VA_ARGS__)
#define OPS_LOG_W(OPS_DESC, ...) OPS_LOG_STUB_W(OPS_DESC, __VA_ARGS__)
#define OPS_LOG_E(OPS_DESC, ...) OPS_INNER_ERR_STUB("EZ9999", OPS_DESC, __VA_ARGS__)
#define OPS_LOG_E_WITHOUT_REPORT(OPS_DESC, ...) OPS_LOG_STUB_E(OPS_DESC, __VA_ARGS__)
#define OPS_LOG_EVENT(OPS_DESC, ...) OPS_LOG_STUB_EVENT(OPS_DESC, __VA_ARGS__)

/* 全量日志
 * 输出超长日志, 若日志超长, 则会被分为多行输出 */
#define OPS_LOG_FULL(LEVEL, OPS_DESC, ...) OPS_LOG_STUB_FULL(LEVEL, OPS_DESC, __VA_ARGS__)
#define OPS_LOG_D_FULL(OPS_DESC, ...) OPS_LOG_STUB_FULL(DLOG_DEBUG, OPS_DESC, __VA_ARGS__)
#define OPS_LOG_I_FULL(OPS_DESC, ...) OPS_LOG_STUB_FULL(DLOG_INFO, OPS_DESC, __VA_ARGS__)
#define OPS_LOG_W_FULL(OPS_DESC, ...) OPS_LOG_STUB_FULL(DLOG_WARN, OPS_DESC, __VA_ARGS__)

/* 条件日志 */
#define OPS_LOG_D_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_STUB_IF(COND, OPS_LOG_D(OP_DESC, __VA_ARGS__), EXPR)
#define OPS_LOG_I_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_STUB_IF(COND, OPS_LOG_I(OP_DESC, __VA_ARGS__), EXPR)
#define OPS_LOG_W_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_STUB_IF(COND, OPS_LOG_W(OP_DESC, __VA_ARGS__), EXPR)
#define OPS_LOG_E_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_STUB_IF(COND, OPS_LOG_E(OP_DESC, __VA_ARGS__), EXPR)
#define OPS_LOG_EVENT_IF(COND, OP_DESC, EXPR, ...) OPS_LOG_STUB_IF(COND, OPS_LOG_EVENT(OP_DESC, __VA_ARGS__), EXPR)

#define OPS_LOG_E_IF_NULL(OPS_DESC, PTR, EXPR)                                                                         \
    if (__builtin_expect((PTR) == nullptr, 0)) {                                                                       \
        OPS_LOG_STUB_E(OPS_DESC, "%s is nullptr!", #PTR);                                                              \
        OPS_CALL_ERR_STUB("EZ9999", OPS_DESC, "%s is nullptr!", #PTR);                                                 \
        EXPR;                                                                                                          \
    }

#define OPS_CHECK(COND, LOG_FUNC, EXPR)                                                                                \
    if (COND) {                                                                                                        \
        LOG_FUNC;                                                                                                      \
        EXPR;                                                                                                          \
    }

#define OP_CHECK(COND, LOG_FUNC, EXPR)                                                                                 \
    if (COND) {                                                                                                        \
        LOG_FUNC;                                                                                                      \
        EXPR;                                                                                                          \
    }
