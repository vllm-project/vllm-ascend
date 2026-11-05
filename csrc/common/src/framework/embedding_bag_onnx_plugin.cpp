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
 * \file embedding_bag_onnx_plugin.cpp
 * \brief
 */

#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;

static Status ParseParamsEmbeddingBag(const Message *op_src, ge::Operator &op_dest) {
  const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  // set attr mode_value
  std::string mode_value;
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "mode" && attr.type() == ge::onnx::AttributeProto::STRING) {
      mode_value = attr.s();
      op_dest.SetAttr("mode", mode_value);
    }
  }
  // set attr scale_grad_by_freq
  bool scale_grad_by_freq = false;
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "scale_grad_by_freq" && attr.i() != 0) {
      scale_grad_by_freq = true;
      break;
    }
  }
  op_dest.SetAttr("scale_grad_by_freq", scale_grad_by_freq);
  // set attr sparse
  bool sparse = false;
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "sparse" && attr.i() != 0) {
      sparse = true;
      break;
    }
  }
  op_dest.SetAttr("sparse", sparse);
  // set attr include_last_offset
  bool include_last_offset = false;
  for (const auto &attr : node->attribute()) {
    if (attr.name() == "include_last_offset" && attr.i() != 0) {
      include_last_offset = true;
      break;
    }
  }
  op_dest.SetAttr("include_last_offset", include_last_offset);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("EmbeddingBag")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::8::EmbeddingBag"),
                    ge::AscendString("ai.onnx::9::EmbeddingBag"),
                    ge::AscendString("ai.onnx::10::EmbeddingBag"),
                    ge::AscendString("ai.onnx::11::EmbeddingBag"),
                    ge::AscendString("ai.onnx::12::EmbeddingBag"),
                    ge::AscendString("ai.onnx::13::EmbeddingBag"),
                    ge::AscendString("ai.onnx::14::EmbeddingBag"),
                    ge::AscendString("ai.onnx::15::EmbeddingBag"),
                    ge::AscendString("ai.onnx::16::EmbeddingBag")})
    .ParseParamsFn(ParseParamsEmbeddingBag)
    .ImplyType(ImplyType::TVM);
} // namespace domi
