// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class AttnResFwdPrefill : public OpDef {
public:
    explicit AttnResFwdPrefill(const char *name) : OpDef(name) {
        for (const char *input : {"prefix_sum", "block_residual", "proj_weight", "norm_weight", "addend", "output_norm"}) {
            this->Input(input).ParamType(REQUIRED).DataType({ge::DT_BF16})
                .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        }
        for (const char *output : {"hidden_states", "prefix_out", "materialized"}) {
            this->Output(output).ParamType(REQUIRED).DataType({ge::DT_BF16})
                .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND}).AutoContiguous();
        }
        this->Attr("norm_eps").AttrType(OPTIONAL).Float(1e-5f);
        this->Attr("need_backward").AttrType(OPTIONAL).Bool(false);
        this->Attr("valid_blocks").AttrType(REQUIRED).Int();
        this->Attr("block_token_stride").AttrType(REQUIRED).Int();
        this->Attr("block_write_idx").AttrType(OPTIONAL).Int(-1);
        this->Attr("output_norm_eps").AttrType(OPTIONAL).Float(0.0f);
        this->Attr("save_materialized").AttrType(OPTIONAL).Bool(false);
        this->Attr("mix").AttrType(OPTIONAL).Bool(true);
        this->Attr("fuse_add").AttrType(OPTIONAL).Bool(false);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false)
            .ExtendCfgInfo("softsync.flag", "true").ExtendCfgInfo("opFile.value", "attn_res_fwd_prefill_apt");
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(AttnResFwdPrefill);
}
