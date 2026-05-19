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
 * \file static_register_symbol.h
 * \brief
 */

#pragma once

#include <string>

#define GLOBAL_REGISTER_SYMBOL_REAL(op_type, class_name, priority, counter, line)   \
[[maybe_unused]] std::string op_impl_register_template_##op_type##_##class_name##priority##counter##line =  \
    std::string("op_impl_register_template_" #op_type) \


#define GLOBAL_REGISTER_SYMBOL(op_type, class_name, priority, counter, line)   \
GLOBAL_REGISTER_SYMBOL_REAL(op_type, class_name, priority, counter, line)


#define GLOBAL_REGISTER_STR_SYMBOL_REAL(op_type, class_name, priority, counter, line)   \
[[maybe_unused]] std::string op_impl_register_template_##class_name##priority##counter##line =  \
    std::string("op_impl_register_template_" op_type)  \


#define GLOBAL_REGISTER_STR_SYMBOL(op_type, class_name, priority, counter, line)   \
GLOBAL_REGISTER_STR_SYMBOL_REAL(op_type, class_name, priority, counter, line)
