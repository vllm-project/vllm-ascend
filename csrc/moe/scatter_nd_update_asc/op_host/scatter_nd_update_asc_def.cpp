/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_update_asc_def.cpp
 * \brief ScatterNdUpdateAsc ophost
 */
#include "register/op_def_registry.h"

namespace ops {
class ScatterNdUpdateAsc : public OpDef {
 public:
  explicit ScatterNdUpdateAsc(const char* name) : OpDef(name) {
    this->Input("var")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("indices")
      .ParamType(REQUIRED)
      .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("updates")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("var")
      .ParamType(REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

    // 显式传入 var 各轴 stride（首轴可能非连续），与 scatter_nd_update_v2 保持一致
    this->Attr("strides").AttrType(REQUIRED).ListInt();

    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
  }
};

OP_ADD(ScatterNdUpdateAsc);
}  // namespace ops