/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "register/op_def_registry.h"

namespace ops {
class SparseKvGather : public OpDef {
public:
    explicit SparseKvGather(const char *name) : OpDef(name)
    {
        // Paged KV cache: [num_blocks, 128, 1, feature_dim].
        this->Input("paged_ctkv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("paged_kpe")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // block_table[q, logical_block] -> physical block.
        this->Input("block_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // Logical token positions, shape [num_actual, topk_n].
        this->Input("topk_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // Current logical token position for every query row.
        this->Input("cur_pos")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("out_ctkv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("out_kpe")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // SFA paged cache currently fixes one page to 128 tokens.
        this->Attr("block_size").AttrType(OPTIONAL).Int(128);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false);
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(SparseKvGather);
}  // namespace ops
