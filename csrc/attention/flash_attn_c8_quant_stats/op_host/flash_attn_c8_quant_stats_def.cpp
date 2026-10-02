// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class FlashAttnC8QuantStats : public OpDef {
public:
    explicit FlashAttnC8QuantStats(const char *name) : OpDef(name)
    {
        for (const char *input : {"key", "value"}) {
            Input(input).ParamType(REQUIRED).DataType({ge::DT_BF16})
                .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        }
        Output("partial").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        for (const char *attr : {"tokens", "heads", "key_token_stride", "key_head_stride",
                                "value_token_stride", "value_head_stride"}) {
            Attr(attr).AttrType(REQUIRED).Int();
        }
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false).PrecisionReduceFlag(false)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(FlashAttnC8QuantStats);
}
