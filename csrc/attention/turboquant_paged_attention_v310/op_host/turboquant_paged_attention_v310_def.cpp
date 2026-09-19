/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file turboquant_paged_attention_v310_def.cpp
 * \brief TurboQuant fused dequant + paged attention (decode).
 */
#include "register/op_def_registry.h"

namespace ops {
class TurboquantPagedAttentionV310 : public OpDef {
public:
    explicit TurboquantPagedAttentionV310(const char *name) : OpDef(name)
    {
        // [batch, num_heads, head_dim] -- one query token per sequence (decode)
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // Packed FRACTAL_NZ caches, fp16-typed (packed bytes ride through the
        // fp16 view; an int8-typed cache silently writes nothing on this SoC).
        this->Input("key_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
        this->Input("value_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
        // [num_slots, num_kv_heads] -- produced by the write op
        this->Input("key_norms")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value_norms")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("block_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("seq_lens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // D vector of Pi = D*H*D, +-1.0f, length head_dim. MUST be the same
        // vector the write op used, or the rotated bases will not match.
        this->Input("signs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("centroids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("attn_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("bits").AttrType(REQUIRED).Int(3);
        this->Attr("scale").AttrType(REQUIRED).Float(0.0625f);
        this->Attr("variant").AttrType(OPTIONAL).Int(0);
        this->Attr("codebook_mode").AttrType(OPTIONAL).Int(0);

        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(TurboquantPagedAttentionV310);
}  // namespace ops
