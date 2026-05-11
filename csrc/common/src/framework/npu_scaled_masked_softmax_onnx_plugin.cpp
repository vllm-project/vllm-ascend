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
 * \file npu_scaled_masked_softmax_onnx_plugin.cpp
 * \brief onnx plugin for npu custom operator npu_scaled_masked_softmax
 */

#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsNPUScaledMaskedSoftmax(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Dynamic cast op_src to NodeProto failed!");
    return FAILED;
  }

  float scale = 1.0;
  bool fixed_triu_mask = false;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "scale" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      scale = attr.f();
    } else if (attr.name() == "fixed_triu_mask" && attr.type() == ge::onnx::AttributeProto::INT && attr.i() == 1) {
      fixed_triu_mask = true;
    }
  }
  op_dest.SetAttr("scale", scale);
  op_dest.SetAttr("fixed_triu_mask", fixed_triu_mask);

  return SUCCESS;
}

// register npu_scaled_masked_softmax op info to GE
REGISTER_CUSTOM_OP("ScaledMaskedSoftmax")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("npu::1::NPUScaledMaskedSoftmax"),
                   ge::AscendString("ai.onnx::11::NPUScaledMaskedSoftmax"),
                   ge::AscendString("ai.onnx::12::NPUScaledMaskedSoftmax"),
                   ge::AscendString("ai.onnx::13::NPUScaledMaskedSoftmax"),
                   ge::AscendString("ai.onnx::14::NPUScaledMaskedSoftmax"),
                   ge::AscendString("ai.onnx::15::NPUScaledMaskedSoftmax"),
                   ge::AscendString("ai.onnx::16::NPUScaledMaskedSoftmax"),
                   ge::AscendString("ai.onnx::17::NPUScaledMaskedSoftmax"),
                   ge::AscendString("ai.onnx::18::NPUScaledMaskedSoftmax")})
    .ParseParamsFn(ParseParamsNPUScaledMaskedSoftmax)
    .ImplyType(ImplyType::TVM);
}  // namespace domi