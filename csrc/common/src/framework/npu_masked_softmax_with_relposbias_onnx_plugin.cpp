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
 * \file npu_masked_softmax_with_relposbias_onnx_plugin.cpp
 * \brief
 */

#include "onnx_common.h"
#include "op_transformer_proto_extend.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
constexpr int OUTPUT_INDEX = 1;
// x, atten_mask, relative_pos_bias, scale_value, inner_precision_mode float scale_value=1.0, int inner_precision_mode=0
static Status ParseParamsNpuMaskedSoftmaxWithRelPosBias(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("MaskedSoftmaxWithRelPosBias", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int input_size = node->input_size();
  int output_size = node->output_size();
  op_dest.DynamicInputRegister("x", input_size);
  op_dest.DynamicOutputRegister("y", output_size);

  float scale_value = 0;
  int inner_precision_mode = 1.0;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "inner_precision_mode" && attr.type() == ge::onnx::AttributeProto::INT) {
      inner_precision_mode = attr.i();
    } else if (attr.name() == "scale_value" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      scale_value = attr.f();
    }
  }
  op_dest.SetAttr("name", node->name());
  op_dest.SetAttr("inner_precision_mode", inner_precision_mode);
  op_dest.SetAttr("scale_value", scale_value);
  op_dest.SetAttr("original_type", "npu::1::NPUMaskedSoftmaxWithRelPosBias");
  return SUCCESS;
}

static Status ParseOpToGraphNpuMaskedSoftmaxWithRelPosBias(const ge::Operator& op, ge::Graph& graph) {
  std::string ori_name;
  if (op.GetAttr("name", ori_name) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
    return FAILED;
  }

  auto data0 = ge::op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
  auto data1 = ge::op::Data((ori_name + "_data1").c_str()).set_attr_index(1);
  auto data2 = ge::op::Data((ori_name + "_data2").c_str()).set_attr_index(2);

  int inner_precision_mode = 0;
  if (op.GetAttr("inner_precision_mode", inner_precision_mode) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get inner_precision_mode from op failed");
    return FAILED;
  }
  float scale_value = 1.0f;
  if (op.GetAttr("scale_value", scale_value) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get scale_value from op failed");
    return FAILED;
  }
  auto masked_softmax_with_relposbias = ge::op::MaskedSoftmaxWithRelPosBias((ori_name + "_MaskedSoftmaxWithRelPosBias").c_str())
                                        .set_input_x(data0)
                                        .set_input_atten_mask(data1)
                                        .set_input_relative_pos_bias(data2)
                                        .set_attr_scale_value(scale_value)
                                        .set_attr_inner_precision_mode(inner_precision_mode);

  std::vector<ge::Operator> inputs{ data0, data1, data2 };
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
  outputs.emplace_back(masked_softmax_with_relposbias, std::vector<std::size_t>{OUTPUT_INDEX});
  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}

// register npu_masked_softmax_with_rel_pos_bias op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPUMaskedSoftmaxWithRelPosBias"),
                 ge::AscendString("ai.onnx::11::NPUMaskedSoftmaxWithRelPosBias"),
                 ge::AscendString("ai.onnx::12::NPUMaskedSoftmaxWithRelPosBias"),
                 ge::AscendString("ai.onnx::13::NPUMaskedSoftmaxWithRelPosBias"),
                 ge::AscendString("ai.onnx::14::NPUMaskedSoftmaxWithRelPosBias"),
                 ge::AscendString("ai.onnx::15::NPUMaskedSoftmaxWithRelPosBias"),
                 ge::AscendString("ai.onnx::16::NPUMaskedSoftmaxWithRelPosBias"),
                 ge::AscendString("ai.onnx::17::NPUMaskedSoftmaxWithRelPosBias"),
                 ge::AscendString("ai.onnx::18::NPUMaskedSoftmaxWithRelPosBias"),
                 ge::AscendString("ai.onnx::19::NPUMaskedSoftmaxWithRelPosBias")})
  .ParseParamsFn(ParseParamsNpuMaskedSoftmaxWithRelPosBias)
  .ParseOpToGraphFn(ParseOpToGraphNpuMaskedSoftmaxWithRelPosBias)
  .ImplyType(ImplyType::TVM);
}  // namespace domi