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
 * \file device_op_impl_registry.cpp
 * \brief
 */
#include "register/device_op_impl_registry.h"
#include <memory>
#include "device_op_impl_registry_impl.h"
#include "err/ops_err.h"

namespace AiInfraOps {
namespace Transformer {
namespace optiling {
DeviceOpImplRegistry &DeviceOpImplRegistry::GetSingleton()
{
  static DeviceOpImplRegistry g_deviceOpImplRegistry;
  return g_deviceOpImplRegistry;
}

void DeviceOpImplRegistry::RegisterSinkTiling(std::string &opType, SinkTilingFunc &func)
{
  std::string opTypeString = opType;
  sinkTilingFuncsMap_[opTypeString] = func;
}

SinkTilingFunc DeviceOpImplRegistry::GetSinkTilingFunc(std::string &opType)
{
  std::string opTypeString = opType;
  auto func = sinkTilingFuncsMap_.find(opTypeString);
  if (func == sinkTilingFuncsMap_.end()) {
    return nullptr;
  }
  return func->second;
}

std::string& DeviceOpImplRegisterImpl::GetOpType()
{
  return opType_;
}

DeviceOpImplRegisterImpl::~DeviceOpImplRegisterImpl()
{
}

DeviceOpImplRegister::DeviceOpImplRegister(const char *opType)
{
  impl_ = std::make_unique<DeviceOpImplRegisterImpl>();
  impl_->GetOpType() = opType;
}

DeviceOpImplRegister &DeviceOpImplRegister::Tiling(SinkTilingFunc func)
{
  DeviceOpImplRegistry::GetSingleton().RegisterSinkTiling(impl_->GetOpType(), func);
  return *this;
}

DeviceOpImplRegister::DeviceOpImplRegister(DeviceOpImplRegister &&other) noexcept
{
  impl_ = std::make_unique<DeviceOpImplRegisterImpl>();
  impl_->GetOpType() = other.impl_->GetOpType();
}

DeviceOpImplRegister::DeviceOpImplRegister(const DeviceOpImplRegister &other)
{
  impl_ = std::make_unique<DeviceOpImplRegisterImpl>();
  impl_->GetOpType() = other.impl_->GetOpType();
}

DeviceOpImplRegister::~DeviceOpImplRegister()
{
}
} // namespace OpTiling
} // namespace Transformer
} // namespace AiInfraOps