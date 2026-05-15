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
 * \file moe_gating_top_k_hash_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class MoeGatingTopKHash : public OpDef {
public:
    explicit MoeGatingTopKHash(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, 
                      ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                      ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                      ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, 
                      ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                      ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                      ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("input_ids")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, 
                      ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, 
                      ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, 
                      ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("tid2eid")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, 
                      ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                      ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                      ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("expert_idx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                      ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, 
                      ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, 
                      ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, 
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("k").Int();
        this->Attr("k_group").AttrType(OPTIONAL).Int(1);
        this->Attr("group_count").AttrType(OPTIONAL).Int(1);
        this->Attr("group_select_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("renorm").AttrType(OPTIONAL).Int(0);
        this->Attr("norm_type").AttrType(OPTIONAL).Int(0);
        this->Attr("out_flag").AttrType(OPTIONAL).Bool(false);
        this->Attr("routed_scaling_factor").AttrType(OPTIONAL).Float(1.0);
        this->Attr("eps").AttrType(OPTIONAL).Float(1e-20f);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

        OpAICoreConfig regbaseCfg;
        regbaseCfg.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "moe_gating_top_k_hash");
        this->AICore().AddConfig("ascend910_95", regbaseCfg);
    }
};

OP_ADD(MoeGatingTopKHash);
} // namespace ops