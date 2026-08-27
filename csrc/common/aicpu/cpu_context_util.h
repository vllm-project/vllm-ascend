/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cou_context_util.h
 * \brief
 */

#ifndef CPU_CONTEXT_UTIL_H
#define CPU_CONTEXT_UTIL_H

#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"
#include "log.h"
#include <string>
#include <type_traits>

#define KERNEL_STATUS_OK      0
#define KERNEL_STATUS_PARAM_INVALID 1

namespace aicpu {
template <typename T>
inline typename std::enable_if<std::is_integral_v<T>, bool>::type
GetAttrValue(CpuKernelContext &ctx, const std::string &name, T &value) {
  auto attr = ctx.GetAttr(name);
  if (!attr) {
    KERNEL_LOG_ERROR("attr is null: %s", name.c_str());
    return false;
  }
  value = static_cast<T>(attr->GetInt());
  return true;
}

inline bool GetAttrValue(CpuKernelContext &ctx, const std::string &name,
                         std::string &value) {
  auto attr = ctx.GetAttr(name);
  if (!attr) {
    KERNEL_LOG_ERROR("attr is null: %s", name.c_str());
    return false;
  }
  value = attr->GetString();
  return true;
}

inline bool GetAttrValue(CpuKernelContext &ctx, const std::string &name,
                         bool &value) {
  auto attr = ctx.GetAttr(name);
  if (!attr) {
    KERNEL_LOG_ERROR("attr is null: %s", name.c_str());
    return false;
  }
  value = attr->GetBool();
  return true;
}

template <typename T>
inline typename std::enable_if<std::is_integral_v<T>, void>::type
GetAttrValueOpt(CpuKernelContext &ctx, const std::string &name, T &value) {
  auto attr = ctx.GetAttr(name);
  if (attr != nullptr) {
    value = static_cast<T>(attr->GetInt());
  }
}

inline void GetAttrValueOpt(CpuKernelContext &ctx, const std::string &name,
                            std::string &value) {
  auto attr = ctx.GetAttr(name);
  if (attr != nullptr) {
    value = attr->GetString();
  }
}

inline void GetAttrValueOpt(CpuKernelContext &ctx, const std::string &name,
                            bool &value) {
  auto attr = ctx.GetAttr(name);
  if (attr != nullptr) {
    value = attr->GetBool();
  }
}

} // namespace aicpu

#endif