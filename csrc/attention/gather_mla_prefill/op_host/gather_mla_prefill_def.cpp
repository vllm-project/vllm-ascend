// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"

namespace ops {
class GatherMlaPrefill : public OpDef {
public:
    explicit GatherMlaPrefill(const char *name) : OpDef(name)
    {
        // Layer-interleaved caches are views. Materializing them would copy the
        // entire persistent cache; physical page/token strides are explicit attrs.
        this->Input("latent_cache").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("rope_cache").ParamType(REQUIRED).DataTypeList({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        for (const char *name : {"block_table", "cumulative_lengths", "lengths", "starts"}) {
            this->Input(name).ParamType(REQUIRED).DataTypeList({ge::DT_INT32})
                .FormatList({ge::FORMAT_ND}).AutoContiguous();
        }
        this->Input("scale").ParamType(REQUIRED).DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Output("latent").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Output("rope").ParamType(REQUIRED).DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Attr("num_tokens").AttrType(REQUIRED).Int();
        this->Attr("max_seq_len").AttrType(REQUIRED).Int();
        this->Attr("latent_page_stride").AttrType(REQUIRED).Int();
        this->Attr("latent_row_stride").AttrType(REQUIRED).Int();
        this->Attr("rope_page_stride").AttrType(REQUIRED).Int();
        this->Attr("rope_row_stride").AttrType(REQUIRED).Int();
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false).PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(GatherMlaPrefill);
} // namespace ops
