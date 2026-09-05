/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file turboquant_reshape_and_cache_v310_def.cpp
 * \brief TurboQuant KV-cache write path (rotate -> quantize -> pack -> scatter).
 */
#include "register/op_def_registry.h"

namespace ops {
class TurboquantReshapeAndCacheV310 : public OpDef {
public:
    explicit TurboquantReshapeAndCacheV310(const char *name) : OpDef(name)
    {
        // K/V for the tokens being written: [num_tokens, num_kv_heads, head_dim]
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // Paged caches, FRACTAL_NZ: (num_blocks, C1, block_size, 16).
        // fp16-typed on purpose -- an int8-typed cache reports success on this
        // SoC while writing nothing. Packed bytes are carried through the fp16
        // view; the op is a pure scatter so the bytes move verbatim.
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
        this->Input("slot_mapping")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // D vector of the rotation Pi = D*H*D, as +-1.0f, length head_dim.
        this->Input("signs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        // Lloyd-Max table: [LEVELS centroids, LEVELS-1 midpoints]. Ignored when
        // codebook_mode == uniform, but kept REQUIRED so the graph shape is
        // stable across A/B scenarios.
        this->Input("centroids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // Norm planes: [num_slots, num_kv_heads]. Stored outside the packed
        // plane so the packed plane keeps exact NZ tile alignment.
        this->Output("key_norms")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("value_norms")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // Scenario selectors. bits drives the tiling key (compile-time);
        // variant and codebook_mode are runtime tiling-data fields.
        this->Attr("bits").AttrType(REQUIRED).Int(3);
        this->Attr("variant").AttrType(OPTIONAL).Int(0);        // 0 = MSE, 1 = MSE+QJL
        this->Attr("codebook_mode").AttrType(OPTIONAL).Int(0);  // 0 = uniform, 1 = Lloyd-Max

        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(TurboquantReshapeAndCacheV310);
}  // namespace ops
