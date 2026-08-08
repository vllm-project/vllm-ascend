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
 * \file custom_fused_infer_attention_v310.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class CustomFusedInferAttentionV310 : public OpDef {
public:
    CustomFusedInferAttentionV310(const char *name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("key")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ});
        this->Input("value")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ});
        this->Input("attn_mask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("actual_seq_lengths_q")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("actual_seq_lengths_kv")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("num_heads").AttrType(REQUIRED).Int(1);
        this->Attr("scale_value").AttrType(OPTIONAL).Float(1.0);
        this->Attr("input_layout").AttrType(OPTIONAL).String("BSH");
        this->Attr("num_key_value_heads").AttrType(OPTIONAL).Int(0);
        this->Attr("block_size").AttrType(OPTIONAL).Int(0);
        this->Attr("inner_precise").AttrType(OPTIONAL).Int(1);

        OpAICoreConfig config_310p;
        config_310p.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend310p", config_310p);
    }
};
OP_ADD(CustomFusedInferAttentionV310);
} // namespace ops
