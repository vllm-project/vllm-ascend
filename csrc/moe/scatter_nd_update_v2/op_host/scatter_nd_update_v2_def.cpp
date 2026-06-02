/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file scatter_nd_update_v2.cpp
 * \brief ScatterNdUpdateV2 ophost
 */
#include "register/op_def_registry.h"

namespace ops {
class ScatterNdUpdateV2 : public OpDef {
 public:
  explicit ScatterNdUpdateV2(const char* name) : OpDef(name) {
    this->Input("var")
      .ParamType(REQUIRED)
      .DataType(
          {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BOOL, ge::DT_INT64, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8,
            ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BOOL, ge::DT_INT64, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8})
      .Format(
          {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat(
          {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("indices")
      .ParamType(REQUIRED)
      .DataType(
          {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
      .Format(
          {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat(
          {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("updates")
      .ParamType(REQUIRED)
      .DataType(
          {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BOOL, ge::DT_INT64, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8,
            ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BOOL, ge::DT_INT64, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8})
      .Format(
          {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat(
          {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("var")
      .ParamType(REQUIRED)
      .DataType(
          {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BOOL, ge::DT_INT64, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8,
            ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_BOOL, ge::DT_INT64, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8})
      .Format(
          {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat(
          {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("strides").AttrType(REQUIRED).ListInt();
    this->Attr("use_locking").AttrType(OPTIONAL).Bool(false);
    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
      .DynamicFormatFlag(true)
      .DynamicRankSupportFlag(true)
      .DynamicShapeSupportFlag(true);
    this->AICore().AddConfig("ascend910b", aicore_config);
    this->AICore().AddConfig("ascend910_93", aicore_config);
  }
};

OP_ADD(ScatterNdUpdateV2);
}  // namespace ops