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
 * \file npu_incre_flash_attention_onnx_plugin.cpp
 * \brief
 */
#include "onnx_common.h"
#include "attention/incre_flash_attention/op_graph/incre_flash_attention_proto.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
constexpr int REQUIRED_ATTR = 2;
constexpr int OUTPUT_INDEX = 0;
constexpr int POS_ACTUAL_SEQ_LENGTHS = 1;
constexpr int POS_ATTEN_MASK = 2;
constexpr int POS_PSE_SHIFT = 3;
constexpr int REQUIRED_INPUT_SIZE = 4; // query/pse_shift/atten_mask/actual_seq_lengths
constexpr int GROUP_SIZE = 2;
static Status ParseParamsIncreFlashAttention(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("IncreFlashAttention", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int input_size = node->input_size();
  int dynamicSize = input_size - REQUIRED_INPUT_SIZE;
  int output_size = node->output_size();
  op_dest.DynamicInputRegister("x", input_size);
  op_dest.DynamicOutputRegister("y", output_size);

  if (dynamicSize % GROUP_SIZE != 0) {
    OP_LOGE(GetOpName(op_dest).c_str(), "dynamic size error.");
    return FAILED;
  }
  dynamicSize /= GROUP_SIZE;

  int num_heads = 0;
  std::string input_layout = "";
  float scale_value = 1.0f;
  int required_attr_num = 0;
  for (const auto& attr : node->attribute()) {
    if (attr.name() == "num_heads" && attr.type() == ge::onnx::AttributeProto::INT) {
      num_heads = attr.i();
      ++required_attr_num;
    } else if (attr.name() == "input_layout" && attr.type() == ge::onnx::AttributeProto::STRING) {
      input_layout = attr.s();
      ++required_attr_num;
    } else if (attr.name() == "scale_value" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      scale_value = attr.f();
    }
  }

  if (required_attr_num != REQUIRED_ATTR) {
    OP_LOGE(GetOpName(op_dest).c_str(), "attr num two is required.");
    return FAILED;
  }

  op_dest.SetAttr("name", node->name());
  op_dest.SetAttr("num_heads", num_heads);
  op_dest.SetAttr("input_layout", input_layout);
  op_dest.SetAttr("scale_value", scale_value);
  op_dest.SetAttr("dynamic_size", dynamicSize);
  op_dest.SetAttr("original_type", "npu::1::NPUIncreFlashAttention");
  return SUCCESS;
}

static Status GetNpuIncreFlashAttentionAttrByName(
  const ge::Operator& op, int& num_heads, std::string& input_layout, float& scale_value) {
  if (op.GetAttr("num_heads", num_heads) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get num_heads from op failed");
    return FAILED;
  }
  if (op.GetAttr("input_layout", input_layout) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get input_layout from op failed");
    return FAILED;
  }
  if (op.GetAttr("scale_value", scale_value) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get scale_value from op failed");
    return FAILED;
  }
  return SUCCESS;
}

static Status ParseOpToGraphNpuIncreFlashAttention(const ge::Operator& op, ge::Graph& graph) {
  std::string ori_name;
  if (op.GetAttr("name", ori_name) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
    return FAILED;
  }

  int dynamicSize = 0;
  op.GetAttr("dynamic_size", dynamicSize);
  auto data0 = ge::op::Data((ori_name + "_data0").c_str()).set_attr_index(0);

  int num_heads = 0;
  string input_layout = "BSH";
  float scale_value = 1.0f;
  Status ret = GetNpuIncreFlashAttentionAttrByName(op, num_heads, input_layout, scale_value);
  if (ret != SUCCESS) {
    return FAILED;
  }

  auto incre_flash_attention = ge::op::IncreFlashAttention((ori_name + "_IncreFlashAttention").c_str())
                              .set_input_query(data0)
                              .create_dynamic_input_key(dynamicSize)
                              .create_dynamic_input_value(dynamicSize);

  std::vector<ge::Operator> inputs{data0};

  int index = 1;
  for (int i = 0; i < dynamicSize; ++i) {
    auto datak = ge::op::Data((ori_name + "_datak" + to_string(i)).c_str()).set_attr_index(index);
    incre_flash_attention.set_dynamic_input_key(index++, datak);
    inputs.emplace_back(datak);
  }
  for (int i = 0; i < dynamicSize; ++i) {
    auto datav = ge::op::Data((ori_name + "_datav" + to_string(i)).c_str()).set_attr_index(index);
    incre_flash_attention.set_dynamic_input_value(index++, datav);
    inputs.emplace_back(datav);
  }

  auto data_pd = ge::op::Data((ori_name + "_data_pd").c_str()).set_attr_index(index++);
  auto data_am = ge::op::Data((ori_name + "_data_am").c_str()).set_attr_index(index++);
  auto data_acq = ge::op::Data((ori_name + "_data_acq").c_str()).set_attr_index(index);

  incre_flash_attention.set_input_pse_shift(data_pd)
                       .set_input_atten_mask(data_am)
                       .set_input_actual_seq_lengths(data_acq)
                       .set_attr_num_heads(num_heads)
                       .set_attr_scale_value(scale_value)
                       .set_attr_input_layout(input_layout);

  inputs.emplace_back(data_pd);
  inputs.emplace_back(data_am);
  inputs.emplace_back(data_acq);

  std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
  outputs.emplace_back(incre_flash_attention, std::vector<std::size_t>{OUTPUT_INDEX});
  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}

// register npu_incre_flash_attention op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPUIncreFlashAttention"), 
                 ge::AscendString("ai.onnx::11::NPUIncreFlashAttention"),
                 ge::AscendString("ai.onnx::12::NPUIncreFlashAttention"),
                 ge::AscendString("ai.onnx::13::NPUIncreFlashAttention"),
                 ge::AscendString("ai.onnx::14::NPUIncreFlashAttention"),
                 ge::AscendString("ai.onnx::15::NPUIncreFlashAttention"),
                 ge::AscendString("ai.onnx::16::NPUIncreFlashAttention"),
                 ge::AscendString("ai.onnx::19::NPUIncreFlashAttention")})
  .ParseParamsFn(ParseParamsIncreFlashAttention)
  .ParseOpToGraphFn(ParseOpToGraphNpuIncreFlashAttention)
  .ImplyType(ImplyType::TVM);
}  // namespace domi