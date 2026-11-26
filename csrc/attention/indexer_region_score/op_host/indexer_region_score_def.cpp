// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "register/op_def_registry.h"
namespace ops {
class IndexerRegionScore : public OpDef {
public:
    explicit IndexerRegionScore(const char *name) : OpDef(name)
    {
        this->Input("query").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("weights").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
                ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("key_summary").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        for (const char *input : {"actual_kv_lengths", "block_table", "cu_seq_lens"}) {
            this->Input(input).ParamType(REQUIRED).DataTypeList({ge::DT_INT32})
                .FormatList({ge::FORMAT_ND}).AutoContiguous();
        }
        this->Output("output").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        this->Output("selected_region_ids").ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Attr("max_region_len").AttrType(REQUIRED).Int();
        this->Attr("top_k").AttrType(OPTIONAL).Int(512);
        this->Attr("region_size").AttrType(OPTIONAL).Int(8);
        this->Attr("head_scale").AttrType(OPTIONAL).Float(1.0F);
        this->Attr("cache_mode").AttrType(OPTIONAL).Int(1);
        this->Attr("mask_mode").AttrType(OPTIONAL).Int(3);
        this->Attr("sort_mode").AttrType(OPTIONAL).Int(1);
        this->Attr("return_score").AttrType(OPTIONAL).Bool(true);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(false).DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false).PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "indexer_region_score")
            .ExtendCfgInfo("opInterface.value", "indexer_region_score")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(IndexerRegionScore);
} // namespace ops
