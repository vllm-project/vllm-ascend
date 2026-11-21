// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "register/op_def_registry.h"

namespace ops {
class GmmSwigluOaiMxQuant : public OpDef {
 public:
  explicit GmmSwigluOaiMxQuant(const char* name) : OpDef(name) {
    const std::initializer_list<ge::DataType> mxTypes{ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E4M3FN};
    const std::initializer_list<ge::DataType> scaleTypes{ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0};
    const std::initializer_list<ge::Format> nd{ge::FORMAT_ND, ge::FORMAT_ND};
    const std::initializer_list<ge::Format> nz{ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ};

    Input("x").ParamType(REQUIRED).DataType(mxTypes).Format(nd).UnknownShapeFormat(nd);
    Input("x_scale").ParamType(REQUIRED).DataType(scaleTypes).Format(nd).UnknownShapeFormat(nd);
    Input("group_list").ParamType(REQUIRED).DataType({ge::DT_INT64, ge::DT_INT64}).Format(nd).UnknownShapeFormat(nd);
    Input("weight").ParamType(DYNAMIC).DataType(mxTypes).Format(nz).UnknownShapeFormat(nz);
    Input("weight_scale").ParamType(DYNAMIC).DataType(scaleTypes).Format(nd).UnknownShapeFormat(nd);
    Output("y").ParamType(REQUIRED).DataType(mxTypes).Format(nd).UnknownShapeFormat(nd);
    Output("y_scale").ParamType(REQUIRED).DataType(scaleTypes).Format(nd).UnknownShapeFormat(nd);

    Attr("group_list_type").AttrType(OPTIONAL).Int(0);
    Attr("clamp_limit").AttrType(OPTIONAL).Float(7.0F);
    Attr("glu_alpha").AttrType(OPTIONAL).Float(1.702F);
    Attr("glu_bias").AttrType(OPTIONAL).Float(1.0F);
    Attr("dst_type_max").AttrType(OPTIONAL).Float(0.0F);

    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true).DynamicShapeSupportFlag(true).ExtendCfgInfo(
        "opFile.value", "gmm_swiglu_oai_mx_quant_apt");
    AICore().AddConfig("ascend950", config);
  }
};

OP_ADD(GmmSwigluOaiMxQuant);
}  // namespace ops
