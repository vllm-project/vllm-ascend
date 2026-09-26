// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class KdaStateCopy : public OpDef {
public:
    explicit KdaStateCopy(const char *name) : OpDef(name)
    {
        Input("source").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        Input("indices").ParamType(REQUIRED).DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        Input("has_initial_state").ParamType(OPTIONAL).DataTypeList({ge::DT_BOOL})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        Output("destination").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        for (const char *attr : {"cache_rows", "selected_rows", "cache_stride_bytes", "payload_bytes"}) {
            Attr(attr).AttrType(REQUIRED).Int();
        }
        Attr("to_cache").AttrType(OPTIONAL).Bool(false);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(KdaStateCopy);
}
