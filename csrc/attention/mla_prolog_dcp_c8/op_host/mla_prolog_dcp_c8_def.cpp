// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "register/op_def_registry.h"

namespace ops {
class MlaPrologDcpC8 : public OpDef {
public:
    explicit MlaPrologDcpC8(const char *name) : OpDef(name)
    {
        this->Input("token_x").ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("weight_dq").ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT8_E4M3FN}).FormatList({ge::FORMAT_FRACTAL_NZ}).AutoContiguous();
        this->Input("weight_uq_qr").ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT8_E4M3FN}).FormatList({ge::FORMAT_FRACTAL_NZ}).AutoContiguous();
        this->Input("weight_uk").ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("weight_dkv_kr").ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT8_E4M3FN}).FormatList({ge::FORMAT_FRACTAL_NZ}).AutoContiguous();
        this->Input("rmsnorm_gamma_cq").ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("rmsnorm_gamma_ckv").ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("rope_sin").ParamType(OPTIONAL)
            .DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("rope_cos").ParamType(OPTIONAL)
            .DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("kv_cache").ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("kr_cache").ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).IgnoreContiguous();
        this->Input("cache_index").ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("dequant_scale_x").ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT8_E8M0}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("dequant_scale_w_dq").ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT8_E8M0}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("dequant_scale_w_uq_qr").ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT8_E8M0}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("dequant_scale_w_dkv_kr").ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT8_E8M0}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("quant_scale_ckv").ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("quant_scale_ckr").ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("smooth_scales_cq").ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("actual_seq_len").ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("k_nope_clip_alpha").ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("kv_descale").ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Output("query").ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Output("query_rope").ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Output("kv_cache").ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Output("kr_cache").ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Output("dequant_scale_q_nope").ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        this->Output("query_norm").ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT8_E4M3FN}).FormatList({ge::FORMAT_ND});
        this->Output("dequant_scale_q_norm").ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT8_E8M0}).FormatList({ge::FORMAT_ND});
        this->Output("query_c8").ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT8_E4M3FN}).FormatList({ge::FORMAT_ND});
        this->Output("query_rope_c8").ParamType(REQUIRED)
            .DataTypeList({ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        this->Output("query_scale_c8").ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        this->Attr("rmsnorm_epsilon_cq").AttrType(OPTIONAL).Float(1e-05f);
        this->Attr("rmsnorm_epsilon_ckv").AttrType(OPTIONAL).Float(1e-05f);
        this->Attr("cache_mode").AttrType(OPTIONAL).String("PA_BSND");
        this->Attr("query_norm_flag").AttrType(OPTIONAL).Bool(false);
        this->Attr("weight_quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("kv_cache_quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("query_quant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("ckvkr_repo_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("quant_scale_repo_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("tile_size").AttrType(OPTIONAL).Int(128); // 128 : set value of tile size
        this->Attr("qc_qr_scale").AttrType(OPTIONAL).Float(1.0f);
        this->Attr("kc_scale").AttrType(OPTIONAL).Float(1.0f);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false).PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(MlaPrologDcpC8);
}  // namespace ops
