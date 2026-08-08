/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*!
 * \file recurrent_kda_def.cpp
 * \brief Recurrent KDA operator definition.
 */
#include "register/op_def_registry.h"

namespace ops {
class RecurrentKda : public OpDef {
public:
    explicit RecurrentKda(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> qkvTypes = {ge::DT_BF16, ge::DT_BF16};
        const std::initializer_list<ge::DataType> floatTypes = {ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> stateTypes = {ge::DT_BF16, ge::DT_FLOAT};

        this->Input("query").ParamType(REQUIRED).DataType(qkvTypes).FormatList({ge::FORMAT_ND});
        this->Input("key").ParamType(REQUIRED).DataType(qkvTypes).FormatList({ge::FORMAT_ND});
        this->Input("value").ParamType(REQUIRED).DataType(qkvTypes).FormatList({ge::FORMAT_ND});
        this->Input("gate").ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND});
        this->Input("beta").ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16}).FormatList({ge::FORMAT_ND});
        this->Input("initial_state")
            .ParamType(REQUIRED)
            .DataType(stateTypes)
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("cu_seqlens")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("ssm_state_indices")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Input("A_log").ParamType(OPTIONAL).DataType(floatTypes).FormatList({ge::FORMAT_ND});
        this->Input("dt_bias").ParamType(OPTIONAL).DataType(floatTypes).FormatList({ge::FORMAT_ND});
        this->Input("num_accepted_tokens")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32, ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});
        this->Output("attn_out").ParamType(REQUIRED).DataType(qkvTypes).FormatList({ge::FORMAT_ND});
        this->Output("initial_state")
            .ParamType(REQUIRED)
            .DataType(stateTypes)
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Output("final_state")
            .ParamType(REQUIRED)
            .DataType(stateTypes)
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();

        this->Attr("layout").AttrType(OPTIONAL).String("BSND");
        this->Attr("scale").AttrType(OPTIONAL).Float(1.0);
        this->Attr("output_final_state").AttrType(OPTIONAL).Bool(false);
        this->Attr("inplace_final_state").AttrType(OPTIONAL).Bool(true);
        this->Attr("use_qk_l2norm_in_kernel").AttrType(OPTIONAL).Bool(false);
        this->Attr("use_gate_in_kernel").AttrType(OPTIONAL).Bool(false);
        this->Attr("use_beta_sigmoid_in_kernel").AttrType(OPTIONAL).Bool(false);
        this->Attr("allow_neg_eigval").AttrType(OPTIONAL).Bool(false);
        this->Attr("safe_gate").AttrType(OPTIONAL).Bool(false);
        this->Attr("lower_bound").AttrType(OPTIONAL).Float(-5.0);
        this->Attr("state_v_first").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicConfig;
        aicConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("softsync.flag", "true");
        this->AICore().AddConfig("ascend910b", aicConfig);
        this->AICore().AddConfig("ascend910_93", aicConfig);
        this->AICore().AddConfig("ascend950", aicConfig);
    }
};

OP_ADD(RecurrentKda);

} // namespace ops
