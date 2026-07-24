/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "register/op_def_registry.h"
namespace ops {
class SparseKvGatherGroup : public OpDef {
public:
    explicit SparseKvGatherGroup(const char *name) : OpDef(name)
    {
        this->Input("paged_ctkv_0").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        this->Input("paged_kpe_0").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        this->Input("paged_ctkv_1").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        this->Input("paged_kpe_1").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        this->Input("paged_ctkv_2").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        this->Input("paged_kpe_2").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        this->Input("block_table").ParamType(REQUIRED).DataType({ge::DT_INT32, ge::DT_INT64}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        this->Input("topk_indices").ParamType(REQUIRED).DataType({ge::DT_INT32, ge::DT_INT64}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        this->Input("cur_pos").ParamType(REQUIRED).DataType({ge::DT_INT32, ge::DT_INT64}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        this->Output("out_ctkv_0").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out_kpe_0").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out_ctkv_1").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out_kpe_1").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out_ctkv_2").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("out_kpe_2").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16}).Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("block_size").AttrType(OPTIONAL).Int(128);
        this->Attr("num_cache_layers").AttrType(OPTIONAL).Int(3);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false).PrecisionReduceFlag(false);
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(SparseKvGatherGroup);
}  // namespace ops
