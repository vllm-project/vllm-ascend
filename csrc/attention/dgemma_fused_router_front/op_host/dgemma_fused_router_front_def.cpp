/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
 */

/*! \file dgemma_fused_router_front_def.cpp
 *  \brief OpDef for DgemmaFusedRouterFront (MIX: RMSNorm+scale (vector) -> GateLinear GEMM (cube)). */
#include "register/op_def_registry.h"

namespace ops {
class DgemmaFusedRouterFront : public OpDef {
public:
    explicit DgemmaFusedRouterFront(const char *name) : OpDef(name)
    {
        // x: [num_tokens, hidden_size] bf16/fp16
        this->Input("x").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // scale: [hidden_size] per-dim RMSNorm scale bf16/fp16
        this->Input("scale").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // proj_weight: [num_experts, hidden_size] row-major (applied as x @ proj_weight.T)
        this->Input("proj_weight").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // norm_scratch: [num_tokens, hidden_size] caller-provided persistent intermediate (graph-safe)
        this->Input("norm_scratch").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // logits_scratch: [num_tokens, num_experts] persistent intermediate consumed inside the op
        this->Input("logits_scratch").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // per_expert_scale: [num_experts] fp32
        this->Input("per_expert_scale").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // sync_scratch: [2] int32 persistent AIV<->AIC GM handoff flags
        this->Input("sync_scratch").ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // topk_weights: [num_tokens, top_k] fp32
        this->Output("topk_weights").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        // topk_ids: [num_tokens, top_k] int32
        this->Output("topk_ids").ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6f);
        this->Attr("hidden_size").AttrType(REQUIRED).Int();
        this->Attr("num_experts").AttrType(REQUIRED).Int();
        this->Attr("top_k").AttrType(OPTIONAL).Int(8);
        this->Attr("sync_base").AttrType(OPTIONAL).Int(1);

        OpAICoreConfig aicConfig;
        aicConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("softsync.flag", "true");
        this->AICore().AddConfig("ascend910b", aicConfig);
        this->AICore().AddConfig("ascend910_93", aicConfig);
    }
};
OP_ADD(DgemmaFusedRouterFront);
} // namespace ops
