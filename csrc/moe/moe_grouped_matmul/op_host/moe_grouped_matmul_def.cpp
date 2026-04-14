/**
* This program is free software, you can redistribute it and/or modify.
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_grouped_matmul_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "moe_grouped_matmul_infershape.cpp"

namespace ops {
class MoeGroupedMatmul : public OpDef {
public:
  explicit MoeGroupedMatmul(const char *name) : OpDef(name) {
    this->Input("x")
        .ParamType(DYNAMIC)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("weight")
        .ParamType(DYNAMIC)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ});
    this->Input("group_list")
        .ParamType(REQUIRED)
        .DataTypeList({ge::DT_INT64, ge::DT_INT32})
        .FormatList({ge::FORMAT_ND});
    this->Output("y")
        .ParamType(DYNAMIC)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("transpose_weight")
        .AttrType(OPTIONAL)
        .Bool(false);
    this->SetInferShape(ge::InferShape);
    this->SetInferDataType(ge::InferDataType);
    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
  }


};

OP_ADD(MoeGroupedMatmul);

} // namespace ops