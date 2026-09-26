// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class KdaRmsNormGated : public OpDef {
public:
    explicit KdaRmsNormGated(const char *name) : OpDef(name)
    {
        for (const char *input : {"x", "gate"}) {
            Input(input).ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}).IgnoreContiguous();
        }
        Input("weight").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        Output("output").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        for (const char *attr : {"tokens", "heads", "x_token_stride", "gate_token_stride"}) {
            Attr(attr).AttrType(REQUIRED).Int();
        }
        Attr("epsilon").AttrType(REQUIRED).Float();
        Attr("sigmoid_only").AttrType(REQUIRED).Bool();
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false).PrecisionReduceFlag(false)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(KdaRmsNormGated);
}
