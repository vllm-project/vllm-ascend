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
 * \file npu_prompt_flash_attention_onnx_plugin.cpp
 * \brief
 */
#include "onnx_common.h"
#include "attention/prompt_flash_attention/op_graph/prompt_flash_attention_proto.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
constexpr int REQUIRED_ATTR = 2;
constexpr int OUTPUT_INDEX = 0;
constexpr int POS_ACTUAL_SEQ_LENGTHS = 1;
constexpr int POS_ATTEN_MASK = 2;
constexpr int POS_PSE_SHIFT = 3;
constexpr int TOTAL_INPUT_SIZE = 6;
constexpr int HIGH_PRECISION = 0;
constexpr int HIGH_PERFORMANCE = 1;

constexpr int value_index = 3;
constexpr int atten_mask_index = 5;
constexpr int actual_seq_lengths_kv_index = 7;
constexpr int quant_scale1_index = 9;

namespace {
static void UpdatePromptFlashAttentionByNode(ge::Operator& op_dest, const NodeProto* node) {
  int input_size = node->input_size();
  int output_size = node->output_size();
  op_dest.DynamicInputRegister("x", input_size);
  op_dest.DynamicOutputRegister("y", output_size);
  op_dest.SetAttr("input_size", input_size);
  op_dest.SetAttr("name", node->name());
  op_dest.SetAttr("original_type", "npu::1::NPUPromptFlashAttention");
}

static Status GetAttrFromPre4(
  const ge::Operator& op, int& num_heads, string& input_layout, float& scale_value, int& pre_tokens) {
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
  if (op.GetAttr("pre_tokens", pre_tokens) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get pre_tokens from op failed");
    return FAILED;
  }
  return SUCCESS;
}

static Status GetAttrFromLast4(
  const ge::Operator& op, int& next_tokens, int& num_key_value_heads, int& sparse_mode, int& inner_precise) {
  if (op.GetAttr("next_tokens", next_tokens) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get next_tokens from op failed");
    return FAILED;
  }
  if (op.GetAttr("num_key_value_heads", num_key_value_heads) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get num_key_value_heads from op failed");
    return FAILED;
  }
  if (op.GetAttr("sparse_mode", sparse_mode) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get sparse_mode from op failed");
    return FAILED;
  }
  if (op.GetAttr("inner_precise", inner_precise) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get inner_precise from op failed");
    return FAILED;
  }
  return SUCCESS;
}

static Status GetAttrFromPromptFlashAttention(const ge::Operator& op, std::string& ori_name, int& input_size) {
  if (op.GetAttr("name", ori_name) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get name from op failed.");
    return FAILED;
  }
  if (op.GetAttr("input_size", input_size) != SUCCESS) {
    OP_LOGE(GetOpName(op).c_str(), "get input_size from op failed.");
    return FAILED;
  }
  return SUCCESS;
}
}

static Status ParseParamsPromptFlashAttention(const Message* op_src, ge::Operator& op_dest) {
  const NodeProto* node = dynamic_cast<const NodeProto*>(op_src);
  if (node == nullptr) {
    OP_LOGE("PromptFlashAttention", "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int required_attr_num = 0;
  int next_tokens = 0;
  int pre_tokens = 0;
  float scale_value = 1.0f;
  std::string input_layout = "";
  int num_heads = 0;
  int num_key_value_heads = 0;
  int sparse_mode = 0;
  int inner_precise = HIGH_PERFORMANCE;

  for (const auto& attr : node->attribute()) {
    if ((attr.name() == "num_heads") && (attr.type() == ge::onnx::AttributeProto::INT)) {
      num_heads = attr.i();
      ++required_attr_num;
    } else if ((attr.name() == "input_layout") && (attr.type() == ge::onnx::AttributeProto::STRING)) {
      input_layout = attr.s();
      ++required_attr_num;
    } else if ((attr.name() == "scale_value") && (attr.type() == ge::onnx::AttributeProto::FLOAT)) {
      scale_value = attr.f();
    } else if ((attr.name() == "pre_tokens") && (attr.type() == ge::onnx::AttributeProto::INT)) {
      pre_tokens = attr.i();
    } else if ((attr.name() == "next_tokens") && (attr.type() == ge::onnx::AttributeProto::INT)) {
      next_tokens = attr.i();
    } else if ((attr.name() == "num_key_value_heads") && (attr.type() == ge::onnx::AttributeProto::INT)) {
      num_key_value_heads = attr.i();
    } else if ((attr.name() == "sparse_mode") && (attr.type() == ge::onnx::AttributeProto::INT)) {
      sparse_mode = attr.i();
    } else if ((attr.name() == "inner_precise") && (attr.type() == ge::onnx::AttributeProto::INT)) {
      inner_precise = attr.i();
    }
  }

  if (required_attr_num != REQUIRED_ATTR) {
    OP_LOGE(GetOpName(op_dest).c_str(), "Attr num_heads and input_layout are required, please check.");
    return FAILED;
  }

  UpdatePromptFlashAttentionByNode(op_dest, node);
  op_dest.SetAttr("num_heads", num_heads);
  op_dest.SetAttr("input_layout", input_layout);
  op_dest.SetAttr("scale_value", scale_value);
  op_dest.SetAttr("pre_tokens", pre_tokens);
  op_dest.SetAttr("next_tokens", next_tokens);
  op_dest.SetAttr("num_key_value_heads", num_key_value_heads);
  op_dest.SetAttr("sparse_mode", sparse_mode);
  op_dest.SetAttr("inner_precise", inner_precise);
  return SUCCESS;
}

static Status ParseOpToGraphNpuPromptFlashAttention(const ge::Operator& op, ge::Graph& graph) {
  std::string ori_name;
  int input_size = 0;
  if (GetAttrFromPromptFlashAttention(op, ori_name, input_size) != SUCCESS) {
    return FAILED;
  }

  auto prompt_flash_attention = ge::op::PromptFlashAttention((ori_name + "_PromptFlashAttention").c_str());

  auto data0 = ge::op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
  auto data1 = ge::op::Data((ori_name + "_data1").c_str()).set_attr_index(1);
  auto data2 = ge::op::Data((ori_name + "_data2").c_str()).set_attr_index(2);
  prompt_flash_attention.set_input_query(data0).set_input_key(data1).set_input_value(data2);
  std::vector<ge::Operator> inputs{ data0, data1, data2 };
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
  if (input_size > value_index) {
    auto data3 = ge::op::Data((ori_name + "_data3").c_str()).set_attr_index(3);
    auto data4 = ge::op::Data((ori_name + "_data4").c_str()).set_attr_index(4);
    inputs.push_back(data3);
    inputs.push_back(data4);
    prompt_flash_attention.set_input_pse_shift(data3).set_input_atten_mask(data4);
  }
  if (input_size > atten_mask_index) {
    auto data5 = ge::op::Data((ori_name + "_data5").c_str()).set_attr_index(5);
    auto data6 = ge::op::Data((ori_name + "_data6").c_str()).set_attr_index(6);
    inputs.push_back(data5);
    inputs.push_back(data6);
    prompt_flash_attention.set_input_actual_seq_lengths(data5).set_input_actual_seq_lengths_kv(data6);
  }
  if (input_size > actual_seq_lengths_kv_index) {
    auto data7 = ge::op::Data((ori_name + "_data7").c_str()).set_attr_index(7);
    auto data8 = ge::op::Data((ori_name + "_data8").c_str()).set_attr_index(8);
    inputs.push_back(data7);
    inputs.push_back(data8);
    prompt_flash_attention.set_input_deq_scale1(data7).set_input_quant_scale1(data8);
  }
  if (input_size > quant_scale1_index) {
    auto data9 = ge::op::Data((ori_name + "_data9").c_str()).set_attr_index(9);
    auto data10 = ge::op::Data((ori_name + "_data10").c_str()).set_attr_index(10);
    auto data11 = ge::op::Data((ori_name + "_data11").c_str()).set_attr_index(11);
    inputs.push_back(data9);
    inputs.push_back(data10);
    inputs.push_back(data11);
    prompt_flash_attention.set_input_deq_scale2(data9).set_input_quant_scale2(data10).set_input_quant_offset2(data11);
  }

  int num_heads = 0;
  string input_layout = "BSH";
  float scale_value = 1.0f;
  int pre_tokens = 0;
  int next_tokens = 0;
  int num_key_value_heads = 0;
  int sparse_mode = 0;
  int inner_precise = 1;
  Status pre4Ret = GetAttrFromPre4(op, num_heads, input_layout, scale_value, pre_tokens);
  Status last4Ret = GetAttrFromLast4(op, next_tokens, num_key_value_heads, sparse_mode, inner_precise);
  if ((pre4Ret != SUCCESS) || (last4Ret != SUCCESS)) {
    return FAILED;
  }

  prompt_flash_attention.set_attr_num_heads(num_heads).set_attr_scale_value(scale_value).set_attr_pre_tokens(pre_tokens)
                        .set_attr_next_tokens(next_tokens).set_attr_input_layout(input_layout)
                        .set_attr_num_key_value_heads(num_key_value_heads).set_attr_sparse_mode(sparse_mode)
                        .set_attr_inner_precise(inner_precise);

  outputs.emplace_back(prompt_flash_attention, std::vector<std::size_t>{OUTPUT_INDEX});
  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}

// register npu_prompt_flash_attention op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPUPromptFlashAttention"), 
                 ge::AscendString("ai.onnx::11::NPUPromptFlashAttention"),
                 ge::AscendString("ai.onnx::12::NPUPromptFlashAttention"),
                 ge::AscendString("ai.onnx::13::NPUPromptFlashAttention"),
                 ge::AscendString("ai.onnx::14::NPUPromptFlashAttention"),
                 ge::AscendString("ai.onnx::15::NPUPromptFlashAttention"),
                 ge::AscendString("ai.onnx::16::NPUPromptFlashAttention"),
                 ge::AscendString("ai.onnx::19::NPUPromptFlashAttention")})
  .ParseParamsFn(ParseParamsPromptFlashAttention)
  .ParseOpToGraphFn(ParseOpToGraphNpuPromptFlashAttention)
  .ImplyType(ImplyType::TVM);
}  // namespace domi