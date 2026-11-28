/**
 * Copyright (c) 2026 SeaGaruda
 * Licensed under CANN Open Software License Agreement Version 2.0.
 *
 * TopKPerRow - ragged per-row top-k selection over a column window.
 *
 *   logits     : (num_rows, width)  float32 | float16 | bfloat16
 *   row_starts : (num_rows,)        int32, inclusive window start
 *   row_ends   : (num_rows,)        int32, exclusive window end
 *   attr topk  : int, 1 <= topk <= 2048
 *   indices    : (num_rows, topk)   int32, -1 padded
 */
#include "register/op_def_registry.h"

namespace ops {
constexpr int64_t TOPK_ATTR_INDEX = 0;
constexpr int64_t K_TOPK_MAX = 2048;

class TopKPerRow : public OpDef {
public:
    explicit TopKPerRow(const char* name) : OpDef(name)
    {
        this->Input("logits")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("row_starts")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("row_ends")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("topk").AttrType(REQUIRED).Int(K_TOPK_MAX);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(TopKPerRow);
}  // namespace ops
