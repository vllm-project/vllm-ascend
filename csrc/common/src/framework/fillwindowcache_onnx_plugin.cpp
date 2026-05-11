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
 * \file fillwindowcache_onnx_plugin.cpp
 * \brief
 */

#include "onnx_common.h"

namespace domi {
static Status parseParamsFillWindowCache(const Message* op_src, ge::Operator& op_dest)
{
  const ge::onnx::NodeProto *node = reinterpret_cast<const ge::onnx::NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest), "Dynamic fillwindowcache op_src to NodeProto failed.");
    return FAILED;
  }
  
  int axis = 0;
  int cache_depth = 0;
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "axis") {
      axis = attr.i();
    }
    if (attr.name() == "cache_depth") {
      cache_depth = attr.i();
    }
  }
  op_dest.SetAttr("axis", axis);
  op_dest.SetAttr("cache_depth", cache_depth);
  return SUCCESS;
}

// register FillWindowCache op info to GE
REGISTER_CUSTOM_OP("FillWindowCache")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::FillWindowCache"),
                   ge::AscendString("ai.onnx::9::FillWindowCache"),
                   ge::AscendString("ai.onnx::10::FillWindowCache"),
                   ge::AscendString("ai.onnx::11::FillWindowCache"),
                   ge::AscendString("ai.onnx::12::FillWindowCache"),
                   ge::AscendString("ai.onnx::13::FillWindowCache"),
                   ge::AscendString("ai.onnx::14::FillWindowCache"),
                   ge::AscendString("ai.onnx::15::FillWindowCache"),
                   ge::AscendString("ai.onnx::16::FillWindowCache"),
                   ge::AscendString("ai.onnx::17::FillWindowCache"),
                   ge::AscendString("ai.onnx::18::FillWindowCache")})
    .ParseParamsFn(parseParamsFillWindowCache)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
