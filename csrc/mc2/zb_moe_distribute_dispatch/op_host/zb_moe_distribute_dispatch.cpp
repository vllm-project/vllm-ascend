#include "register/op_def_registry.h"

namespace ops {
class ZbMoeDistributeDispatch : public OpDef
{
public:
    explicit ZbMoeDistributeDispatch(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("expert_ids")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("scales")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x_active_mask")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_BOOL})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("elastic_info")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("expand_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_INT8})
            .FormatList({ge::FORMAT_ND});

        this->Output("dynamic_scales").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});

        this->Output("assist_info_for_combine")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Output("expert_token_nums").ParamType(REQUIRED).DataTypeList({ge::DT_INT64}).FormatList({ge::FORMAT_ND});
        this->Output("ep_recv_count").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Output("tp_recv_count").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});

        this->Attr("ep_world_size").AttrType(REQUIRED).Int();
        this->Attr("ep_rank_id").AttrType(REQUIRED).Int();
        this->Attr("moe_expert_num").AttrType(REQUIRED).Int();
        this->Attr("tp_world_size").AttrType(OPTIONAL).Int(0);
        this->Attr("tp_rank_id").AttrType(OPTIONAL).Int(0);
        this->Attr("expert_shard_type").AttrType(OPTIONAL).Int(0);
        this->Attr("shared_expert_num").AttrType(OPTIONAL).Int(1);
        this->Attr("shared_expert_rank_num").AttrType(OPTIONAL).Int(0);
        this->Attr("quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("global_bs").AttrType(OPTIONAL).Int(0);
        this->Attr("expert_token_nums_type").AttrType(OPTIONAL).Int(1);
        this->Attr("extInfo").AttrType(REQUIRED).Int();
        this->Attr("comm_alg").AttrType(OPTIONAL).String("");
        this->Attr("zero_expert_num").AttrType(OPTIONAL).Int(0);
        this->Attr("copy_expert_num").AttrType(OPTIONAL).Int(0);
        this->Attr("const_expert_num").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("jitCompile.flag", "static_true")
            .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");

        this->AICore().AddConfig("ascend910_93", aicore_config);
        // this->MC2().HcclGroup({"group_ep", "group_tp"});
    }
};

OP_ADD(ZbMoeDistributeDispatch);

}  // namespace ops
