// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"

namespace ops {
class FlashAttnC8 : public OpDef {
public:
    explicit FlashAttnC8(const char *name) : OpDef(name)
    {
        this->Input("q").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("k").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("v").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("query_rope").ParamType(REQUIRED).DataTypeList({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("key_rope").ParamType(REQUIRED).DataTypeList({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("dequant_scale_query").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("dequant_scale_key").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("dequant_scale_value").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("cu_seqlens_q").ParamType(REQUIRED).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("cu_seqlens_kv").ParamType(REQUIRED).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("seqused_q").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("attn_mask").ParamType(OPTIONAL).DataTypeList({ge::DT_INT8})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("metadata").ParamType(REQUIRED).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Output("attn_out").ParamType(REQUIRED).DataTypeList({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND});
        this->Output("softmax_lse").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Attr("softmax_scale").AttrType(OPTIONAL).Float(1.0f);
        this->Attr("mask_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("max_seqlen_q").AttrType(OPTIONAL).Int(-1);
        this->Attr("max_seqlen_kv").AttrType(OPTIONAL).Int(-1);
        this->Attr("return_softmax_lse").AttrType(OPTIONAL).Bool(true);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false).PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(FlashAttnC8);
}
