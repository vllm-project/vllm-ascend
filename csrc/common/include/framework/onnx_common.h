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
 * \file onnx_common.h
 * \brief
 */

#ifndef MATH_COMMON_ONNX_COMMON_H
#define MATH_COMMON_ONNX_COMMON_H

#include <string>
#include <vector>
#include <map>

#include "stub_ops.h"
#include "register/register.h"
#include "graph/operator.h"
#include "graph/graph.h"
#include "base/err_msg.h"
#include "log/log.h"
#include "onnx/proto/ge_onnx.pb.h"

namespace domi {
template <typename T>
inline std::string GetOpName(const T& op)
{
  ge::AscendString op_ascend_name;
  ge::graphStatus ret = op.GetName(op_ascend_name);
  if (ret != ge::GRAPH_SUCCESS) {
    std::string op_name = "None";
    return op_name;
  }
  return op_ascend_name.GetString();
}

template<typename T>
inline ge::Tensor Vec2Tensor(vector<T>& vals, const vector<int64_t>& dims, ge::DataType dtype, ge::Format format = ge::FORMAT_ND) {
  ge::Shape shape(dims);
  ge::TensorDesc desc(shape, format, dtype);
  ge::Tensor tensor(desc, reinterpret_cast<uint8_t*>(vals.data()), vals.size() * sizeof(T));
  return tensor;
}

template<typename T>
inline ge::Tensor CreateScalar(T val, ge::DataType dtype, ge::Format format = ge::FORMAT_ND) {
  vector<int64_t> dims_scalar = {};
  ge::Shape shape(dims_scalar);
  ge::TensorDesc desc(shape, format, dtype);
  ge::Tensor tensor(desc, reinterpret_cast<uint8_t*>(&val), sizeof(T));
  return tensor;
}

inline Status ChangeFormatFromOnnx(ge::Operator& op, const int idx, ge::Format format, bool is_input) {
  if (is_input) {
    ge::TensorDesc org_tensor = op.GetInputDesc(idx);
    org_tensor.SetOriginFormat(format);
    org_tensor.SetFormat(format);
    auto ret = op.UpdateInputDesc(idx, org_tensor);
    if (ret != ge::GRAPH_SUCCESS) {
      OP_LOGE(GetOpName(op).c_str(), "change input format failed.");
      return FAILED;
    }
  } else {
    ge::TensorDesc org_tensor_y = op.GetOutputDesc(idx);
    org_tensor_y.SetOriginFormat(format);
    org_tensor_y.SetFormat(format);
    auto ret_y = op.UpdateOutputDesc(idx, org_tensor_y);
    if (ret_y != ge::GRAPH_SUCCESS) {
      OP_LOGE(GetOpName(op).c_str(), "change output format failed.");
      return FAILED;
    }
  }
  return SUCCESS;
}
}  // namespace domi

#endif  //  MATH_COMMON_ONNX_COMMON_H