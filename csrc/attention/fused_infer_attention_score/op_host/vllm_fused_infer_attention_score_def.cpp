/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_infer_attention_score_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
class VllmFusedInferAttentionScore : public OpDef {
public:
    explicit VllmFusedInferAttentionScore(const char *name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("pse_shift")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("atten_mask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths_kv")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("blocktable")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("attention_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("softmax_lse")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("sparse_stats")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("num_heads").AttrType(REQUIRED).Int(1);
        this->Attr("scale").AttrType(REQUIRED).Float(1.0);
        this->Attr("pre_tokens").AttrType(OPTIONAL).Int(2147483647);
        this->Attr("next_tokens").AttrType(OPTIONAL).Int(2147483647);
        this->Attr("input_layout").AttrType(OPTIONAL).String("TND");
        this->Attr("num_key_value_heads").AttrType(OPTIONAL).Int(0);
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("inner_precise").AttrType(OPTIONAL).Int(0);
        this->Attr("block_size").AttrType(OPTIONAL).Int(0);
        this->Attr("antiquant_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("sparse_lambda").AttrType(OPTIONAL).Float(-99.0);
        this->Attr("softmax_lse_flag").AttrType(OPTIONAL).Bool(false);
        this->Attr("actual_seq_lengths_q_host").AttrType(OPTIONAL).ListInt({});
        this->Attr("actual_seq_lengths_kv_host").AttrType(OPTIONAL).ListInt({});
        this->Attr("sparse_stats_flag").AttrType(OPTIONAL).Bool(false);
        // Behavior switch fed from vllm_ascend CustomFIAConfig.flash_decode.
        // (Former getenv kill-switch VLLM_FIA_FD; an attr keeps it reviewable,
        // per-call, and graph-capture deterministic.)
        this->Attr("flash_decode").AttrType(OPTIONAL).Bool(true);
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("opFile.value", "fused_infer_attention_score")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};
OP_ADD(VllmFusedInferAttentionScore);
} // namespace ops
