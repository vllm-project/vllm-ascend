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
 * \file swiglustep_def.cpp
 * \brief SwigluStep op definition: out = silu(gate).clamp(max=limit) * up.clamp(-limit, limit)
 */
#include "register/op_def_registry.h"

namespace ops {
class Swiglustep : public OpDef {
public:
    explicit Swiglustep(const char* name) : OpDef(name)
    {
        // x: row-major [M, 2N], gate = first N cols, up = last N cols (bf16 / fp16)
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        // limit: clamp threshold scalar (Step-3.7 = 7.0)
        this->Attr("limit").AttrType(REQUIRED).Float();
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(Swiglustep);
}  // namespace ops
