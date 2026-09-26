// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class FlashAttnC8Prepare : public OpDef {
public:
    explicit FlashAttnC8Prepare(const char *name) : OpDef(name)
    {
        for (const char *input : {"query", "key", "value", "key_rope"}) {
            Input(input).ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        }
        Input("partial").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        for (const char *output : {"query_out", "key_out", "value_out"}) {
            Output(output).ParamType(REQUIRED).DataType({ge::DT_FLOAT8_E4M3FN, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        for (const char *output : {"query_rope_out", "key_rope_out"}) {
            Output(output).ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        for (const char *output : {"query_scale", "key_scale", "value_scale"}) {
            Output(output).ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        for (const char *attr : {"query_tokens", "key_tokens", "heads", "query_token_stride", "query_head_stride",
                                "key_token_stride", "key_head_stride", "value_token_stride", "value_head_stride",
                                "rope_token_stride", "rope_head_stride"}) {
            Attr(attr).AttrType(REQUIRED).Int();
        }
        Attr("fake_quant").AttrType(OPTIONAL).Bool(false);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false).PrecisionReduceFlag(false)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(FlashAttnC8Prepare);
}
