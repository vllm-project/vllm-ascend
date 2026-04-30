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
 * \file multi_head_attention_onnx_plugin.cpp
 * \brief
 */

#include "onnx_common.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
static const int REQUIRED_ATTRS_NUM = 6;
static Status ParseParamsMultiHeadAttention(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("MultiHeadAttention", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }
  int attn_head_num = 0;
  int attn_dim_per_head = 0;
  int src_len = 0;
  int tgt_len = 0;
  float dropout_prob = 0.0f;
  int softmax_use_float = 0;
  int attr_num = 0;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "attn_head_num" && attr.type() == ge::onnx::AttributeProto::INT) {
      attn_head_num = attr.i();
      ++attr_num;
    } else if (attr.name() == "attn_dim_per_head" && attr.type() == ge::onnx::AttributeProto::INT) {
      attn_dim_per_head = attr.i();
      ++attr_num;
    } else if (attr.name() == "src_len" && attr.type() == ge::onnx::AttributeProto::INT) {
      src_len = attr.i();
      ++attr_num;
    } else if (attr.name() == "tgt_len" && attr.type() == ge::onnx::AttributeProto::INT) {
      tgt_len = attr.i();
      ++attr_num;
    } else if (attr.name() == "dropout_prob" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      dropout_prob = attr.f();
      ++attr_num;
    } else if (attr.name() == "softmax_use_float" && attr.type() == ge::onnx::AttributeProto::INT) {
      softmax_use_float = attr.i();
      ++attr_num;
    }
  }

  if (attr_num != REQUIRED_ATTRS_NUM) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Node must have attrs attn_head_num/attn_dim_per_head/"
                                                "src_len/tgt_len/dropout_prob/softmax_use_float");
    return FAILED;
  }
  op_dest.SetAttr("attn_head_num", attn_head_num);
  op_dest.SetAttr("attn_dim_per_head", attn_dim_per_head);
  op_dest.SetAttr("src_len", src_len);
  op_dest.SetAttr("tgt_len", tgt_len);
  op_dest.SetAttr("keep_prob", static_cast<float>(1 - dropout_prob));
  op_dest.SetAttr("softmax_use_float", static_cast<bool>(softmax_use_float));
  return SUCCESS;
}

// register Yolo op info to GE
REGISTER_CUSTOM_OP("MultiHeadAttention")
    .FrameworkType(ONNX)
    .OriginOpType({ge::AscendString("ai.onnx::11::NPUMultiHeadAttention"),
                   ge::AscendString("ai.onnx::12::NPUMultiHeadAttention"),
                   ge::AscendString("ai.onnx::13::NPUMultiHeadAttention"),
                   ge::AscendString("ai.onnx::14::NPUMultiHeadAttention"),
                   ge::AscendString("ai.onnx::15::NPUMultiHeadAttention"),
                   ge::AscendString("ai.onnx::16::NPUMultiHeadAttention"),
                   ge::AscendString("ai.onnx::17::NPUMultiHeadAttention"),
                   ge::AscendString("ai.onnx::18::NPUMultiHeadAttention"),
                   ge::AscendString("npu::1::NPUMultiHeadAttention")})
    .ParseParamsFn(ParseParamsMultiHeadAttention)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
