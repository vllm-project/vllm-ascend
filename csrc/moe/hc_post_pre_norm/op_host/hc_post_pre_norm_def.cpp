/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#include "../op_kernel/hc_post_pre_norm_config.h"
#include "register/op_def_registry.h"

namespace ops {

class HcPostPreNorm : public OpDef {
 public:
  explicit HcPostPreNorm(const char* name) : OpDef(name) {
    const auto addInput = [this](const char* tensorName, ge::DataType dtype) {
      this->Input(tensorName)
          .ParamType(REQUIRED)
          .DataType({dtype})
          .Format({ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND})
          .AutoContiguous();
    };

    const auto addOutput = [this](const char* tensorName, ge::DataType dtype) {
      this->Output(tensorName)
          .ParamType(REQUIRED)
          .DataType({dtype})
          .Format({ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND});
    };

    addInput("residual", ge::DT_BF16);
    addInput("ffn_hc_fn", ge::DT_FLOAT);
    addInput("ffn_hc_scale", ge::DT_FLOAT);
    addInput("ffn_hc_base", ge::DT_FLOAT);
    addInput("attn_pre", ge::DT_FLOAT);

    addInput("attn_out", ge::DT_BF16);
    addInput("attn_post", ge::DT_FLOAT);
    addInput("attn_comb", ge::DT_FLOAT);
    addInput("ffn_norm_weight", ge::DT_BF16);

    addOutput("y", ge::DT_BF16);
    addOutput("ffn_post", ge::DT_FLOAT);
    addOutput("ffn_comb", ge::DT_FLOAT);
    addOutput("ffn_pre", ge::DT_FLOAT);
    addOutput("new_residual", ge::DT_BF16);
    addOutput("y_fp32", ge::DT_FLOAT);

    this->Attr("hc_mult").AttrType(OPTIONAL).Int(::HcPostPreNorm::kHcMult);
    this->Attr("hc_sinkhorn_iters").AttrType(OPTIONAL).Int(20);
    this->Attr("hc_eps").AttrType(OPTIONAL).Float(1e-6f);
    this->Attr("hc_norm_eps").AttrType(OPTIONAL).Float(1e-6f);
    this->Attr("ffn_norm_eps").AttrType(OPTIONAL).Float(1e-6f);
    this->Attr("return_fp32").AttrType(OPTIONAL).Bool(true);

    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
  }
};

OP_ADD(HcPostPreNorm);

}  // namespace ops
