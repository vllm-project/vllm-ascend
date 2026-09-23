// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"

namespace ops {
class FlashMlaBf16Prepare : public OpDef {
public:
    explicit FlashMlaBf16Prepare(const char *name) : OpDef(name)
    {
        for (const char *input : {"key_nope", "value", "key_rope"}) {
            // Read projection split views directly; an implicit contiguous
            // conversion would undo the purpose of this operator.
            this->Input(input).ParamType(REQUIRED).DataTypeList({ge::DT_BF16})
                .FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        }
        for (const char *output : {"key", "packed_value"}) {
            this->Output(output).ParamType(REQUIRED).DataTypeList({ge::DT_BF16})
                .FormatList({ge::FORMAT_ND});
        }
        for (const char *attr : {"key_token_stride", "key_head_stride", "value_token_stride",
                                "value_head_stride", "rope_token_stride", "rope_head_stride"}) {
            this->Attr(attr).AttrType(REQUIRED).Int();
        }
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false).PrecisionReduceFlag(false)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(FlashMlaBf16Prepare);
} // namespace ops
