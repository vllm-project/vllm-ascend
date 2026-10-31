// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "register/op_def_registry.h"

namespace ops {
class EngramHashIds : public OpDef {
public:
    explicit EngramHashIds(const char *name) : OpDef(name)
    {
        this->Input("input_ids").ParamType(REQUIRED).DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("token_map").ParamType(REQUIRED).DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("dead_mask").ParamType(REQUIRED).DataTypeList({ge::DT_BOOL})
            .FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("positions").ParamType(REQUIRED).DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("block_table").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("query_start_loc").ParamType(REQUIRED).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("req_ids").ParamType(REQUIRED).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("multipliers").ParamType(REQUIRED).DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("primes").ParamType(REQUIRED).DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("offsets").ParamType(REQUIRED).DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("cache").ParamType(OPTIONAL).DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("lookback_token_ids").ParamType(REQUIRED).DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("lookback_dead_mask").ParamType(REQUIRED).DataTypeList({ge::DT_BOOL})
            .FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Output("output").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});

        this->Attr("parameters").AttrType(REQUIRED).ListInt();

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(EngramHashIds);
} // namespace ops
