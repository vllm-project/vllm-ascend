/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ops_err.h
 * \brief
 */

#ifndef TRANSFORMER_COMMON_OPS_ERR_H
#define TRANSFORMER_COMMON_OPS_ERR_H

#include "log/log.h"


/* 基础报错 */
#define OPS_REPORT_VECTOR_INNER_ERR(OPS_DESC, ...) OPS_INNER_ERR_STUB("E89999", OPS_DESC, __VA_ARGS__)
#define OPS_REPORT_CUBE_INNER_ERR(OPS_DESC, ...) OPS_INNER_ERR_STUB("E69999", OPS_DESC, __VA_ARGS__)

/* 条件报错 */
#define OPS_ERR_IF(COND, LOG_FUNC, EXPR) OPS_LOG_STUB_IF(COND, LOG_FUNC, EXPR)

#endif // TRANSFORMER_COMMON_OPS_ERR_H