/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "onnx_common.h"
#include "op_transformer_proto_extend.h"

namespace domi {
using NodeProto = ge::onnx::NodeProto;
constexpr int REQUIRED_ATTR = 1;
constexpr int ONE = 1;
constexpr int INDEX_TWO = 2;
constexpr int INDEX_THREE = 3;
constexpr int ACL_FLOAT16 = 1;

static Status ParseParamsNpuFusedAttentionScoreFwd(const Message *op_src, ge::Operator &op_dest) {
  const NodeProto *node = dynamic_cast<const NodeProto *>(op_src);
  if (node == nullptr) {
    OP_LOGE(GetOpName(op_dest), "Dynamic cast op_src to NodeProto failed.");
    return FAILED;
  }

  int input_size = node->input_size();
  int output_size = node->output_size();
  op_dest.DynamicInputRegister("x", input_size);
  op_dest.DynamicOutputRegister("y", output_size);

  int required_attr_num = 0;
  float scale = 0;
  float keep_prob = 1.;
  bool query_transpose = false;
  bool key_transpose = false;
  bool bmm_score_transpose_a = false;
  bool bmm_score_transpose_b = false;

  for (const auto &attr : node->attribute()) {
    if (attr.name() == "scale" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      scale = attr.f();
      required_attr_num++;
    } else if (attr.name() == "keep_prob" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
      keep_prob = attr.f();
    } else if (attr.name() == "query_transpose" && attr.type() == ge::onnx::AttributeProto::INT) {
      query_transpose = (attr.i() == 1);
    } else if (attr.name() == "key_transpose" && attr.type() == ge::onnx::AttributeProto::INT) {
      key_transpose = (attr.i() == 1);
    } else if (attr.name() == "bmm_score_transpose_a" && attr.type() == ge::onnx::AttributeProto::INT) {
      bmm_score_transpose_a = (attr.i() == 1);
    } else if (attr.name() == "bmm_score_transpose_b" && attr.type() == ge::onnx::AttributeProto::INT) {
      bmm_score_transpose_b = (attr.i() == 1);
    } 
  }

  if (required_attr_num != REQUIRED_ATTR) {
    OP_LOGE(GetOpName(op_dest), "attr scale is required.");
    return FAILED;
  }

  op_dest.SetAttr("name", node->name());
  op_dest.SetAttr("scale", scale);
  op_dest.SetAttr("keep_prob", keep_prob);
  op_dest.SetAttr("query_transpose", query_transpose);
  op_dest.SetAttr("key_transpose", key_transpose);
  op_dest.SetAttr("bmm_score_transpose_a", bmm_score_transpose_a);
  op_dest.SetAttr("bmm_score_transpose_b", bmm_score_transpose_b);
  op_dest.SetAttr("original_type", "npu::1::NPUFusedAttentionScoreFwd");
  return SUCCESS;
}

namespace{
static Status GetAttrFromOperator(const ge::Operator& op, float& scale, float& keep_prob, bool& query_transpose, 
  bool& key_transpose, bool& bmm_score_transpose_a, bool& bmm_score_transpose_b) {
  if (op.GetAttr("scale", scale) != SUCCESS) {
    OP_LOGE(GetOpName(op), "get scale from op failed");
    return FAILED;
  }
  if (op.GetAttr("keep_prob", keep_prob) != SUCCESS) {
    OP_LOGE(GetOpName(op), "get keep_prob from op failed");
    return FAILED;
  }
  if (op.GetAttr("query_transpose", query_transpose) != SUCCESS) {
    OP_LOGE(GetOpName(op), "get query_transpose from op failed");
    return FAILED;
  }
  if (op.GetAttr("key_transpose", key_transpose) != SUCCESS) {
    OP_LOGE(GetOpName(op), "get key_transpose from op failed");
    return FAILED;
  }
  if (op.GetAttr("bmm_score_transpose_a", bmm_score_transpose_a) != SUCCESS) {
    OP_LOGE(GetOpName(op), "get bmm_score_transpose_a from op failed");
    return FAILED;
  }
  if (op.GetAttr("bmm_score_transpose_b", bmm_score_transpose_b) != SUCCESS) {
    OP_LOGE(GetOpName(op), "get bmm_score_transpose_b from op failed");
    return FAILED;
  }
  return SUCCESS;
  }
}

static Status ParseOpToGraphNpuFusedAttentionScoreFwd(const ge::Operator& op, ge::Graph& graph) {
  std::string ori_name;
  if (op.GetAttr("name", ori_name) != SUCCESS) {
    OP_LOGE(GetOpName(op), "get name from op failed.");
    return FAILED;
  }

  auto data0 = ge::op::Data((ori_name + "_data0").c_str()).set_attr_index(0);
  auto data1 = ge::op::Data((ori_name + "_data1").c_str()).set_attr_index(1);
  auto data2 = ge::op::Data((ori_name + "_data2").c_str()).set_attr_index(2);
  auto data3 = ge::op::Data((ori_name + "_data3").c_str()).set_attr_index(3);
  
  float scale = 0;
  float keep_prob = 0;
  bool query_transpose = false;
  bool key_transpose = false;
  bool bmm_score_transpose_a = false;
  bool bmm_score_transpose_b = false;
  Status ret = GetAttrFromOperator(
    op, scale, keep_prob, query_transpose, key_transpose, bmm_score_transpose_a, bmm_score_transpose_b);
  if (ret != SUCCESS) {
    return FAILED;
  }
  // create const input tensor "drop_mask" which is filled with the scalar value 1 for inferencing
  // deop_mask.size = {query_size[0], query_size[1], query_size[2], query_size[2]}
  ge::Tensor saclar_one = CreateScalar(ONE, ge::DT_UINT8);
  auto const_one = ge::op::Const((ori_name + "_Const_one").c_str()).set_attr_value(saclar_one);
  std::vector<int64_t> dims = op.GetInputDesc(0).GetShape().GetDims();
  dims[INDEX_THREE] = dims[INDEX_TWO];
  auto tensor_dims = Vec2Tensor(dims, {4}, ge::DT_INT64);
  auto const_dims = ge::op::Const((ori_name + "_Const_dims").c_str()).set_attr_value(tensor_dims);
  auto drop_mask = ge::op::Fill((ori_name + "_Fill_ones").c_str()).set_input_dims(const_dims)
                                                        .set_input_value(const_one);

  ge::Tensor tensor_scale = CreateScalar(scale, ge::DT_FLOAT);
  auto const_scale = ge::op::Const((ori_name + "_Const_scale").c_str()).set_attr_value(tensor_scale);
  auto cast_const_scale = ge::op::Cast((ori_name + "_Cast_const_scale").c_str()).set_input_x(const_scale)
                                                                      .set_attr_dst_type(ACL_FLOAT16);
  
  auto AttentionScore = ge::op::AttentionScore((ori_name + "_AttentionScore").c_str()).set_input_query(data0)
                            .set_input_key(data1).set_input_value(data2).set_input_padding_mask(data3)
                            .set_input_scale(cast_const_scale).set_input_drop_mask(drop_mask)
                            .set_attr_keep_prob(keep_prob).set_attr_query_transpose(query_transpose)
                            .set_attr_key_transpose(key_transpose).set_attr_bmm_score_transpose_a(bmm_score_transpose_a)
                            .set_attr_bmm_score_transpose_b(bmm_score_transpose_b).set_attr_softmax_axes({-1});

  std::vector<ge::Operator> inputs{data0, data1, data2, data3};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> outputs;
  outputs.emplace_back(AttentionScore, std::vector<std::size_t>{0});
  outputs.emplace_back(AttentionScore, std::vector<std::size_t>{1});
  outputs.emplace_back(drop_mask, std::vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(outputs);
  return SUCCESS;
}

// register npu_fused_attention_score_fwd op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
  .FrameworkType(ONNX)
  .OriginOpType({ge::AscendString("npu::1::NPUFusedAttentionScoreFwd"), 
                 ge::AscendString("ai.onnx::11::NPUFusedAttentionScoreFwd"),
                 ge::AscendString("ai.onnx::12::NPUFusedAttentionScoreFwd"),
                 ge::AscendString("ai.onnx::13::NPUFusedAttentionScoreFwd"),
                 ge::AscendString("ai.onnx::14::NPUFusedAttentionScoreFwd"),
                 ge::AscendString("ai.onnx::15::NPUFusedAttentionScoreFwd"),
                 ge::AscendString("ai.onnx::16::NPUFusedAttentionScoreFwd"),
                 ge::AscendString("ai.onnx::17::NPUFusedAttentionScoreFwd"),
                 ge::AscendString("ai.onnx::18::NPUFusedAttentionScoreFwd")})
  .ParseParamsFn(ParseParamsNpuFusedAttentionScoreFwd)
  .ParseOpToGraphFn(ParseOpToGraphNpuFusedAttentionScoreFwd)
  .ImplyType(ImplyType::TVM);
} // namespace domi
