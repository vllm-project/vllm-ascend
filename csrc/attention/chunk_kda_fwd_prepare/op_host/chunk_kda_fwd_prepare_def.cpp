/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 */

#include "register/op_def_registry.h"

#include <initializer_list>

namespace ops {

class ChunkKdaFwdPrepare : public OpDef {
public:
    explicit ChunkKdaFwdPrepare(const char *name) : OpDef(name)
    {
        // gate 与 beta 可以独立选择 BF16/FP32，因此显式登记四种 dtype 组合。
        const std::initializer_list<ge::DataType> dataTypes = {
            ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16};
        const std::initializer_list<ge::DataType> gateTypes = {
            ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> betaTypes = {
            ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> fp32Types = {
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> indexTypes = {
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("q").ParamType(REQUIRED).DataType(dataTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("k").ParamType(REQUIRED).DataType(dataTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("v").ParamType(REQUIRED).DataType(dataTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("g").ParamType(REQUIRED).DataType(gateTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("beta").ParamType(REQUIRED).DataType(betaTypes).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("a_log").ParamType(OPTIONAL).DataType(fp32Types).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("dt_bias").ParamType(OPTIONAL).DataType(fp32Types).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(indexTypes).Format(formats).UnknownShapeFormat(formats).AutoContiguous();
        this->Input("chunk_indices").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(indexTypes).Format(formats).UnknownShapeFormat(formats).AutoContiguous();

        this->Output("gk").ParamType(REQUIRED).DataType(fp32Types).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("aqk").ParamType(REQUIRED).DataType(dataTypes).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("akk").ParamType(REQUIRED).DataType(dataTypes).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("w").ParamType(REQUIRED).DataType(dataTypes).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("u").ParamType(REQUIRED).DataType(dataTypes).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("qg").ParamType(REQUIRED).DataType(dataTypes).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("kg").ParamType(REQUIRED).DataType(dataTypes).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("qg_scaled").ParamType(REQUIRED).DataType(dataTypes).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("q_hat").ParamType(REQUIRED).DataType(dataTypes).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("k_hat").ParamType(REQUIRED).DataType(dataTypes).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("q_rstd").ParamType(REQUIRED).DataType(fp32Types).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("k_rstd").ParamType(REQUIRED).DataType(fp32Types).Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("beta_eff").ParamType(REQUIRED).DataType(fp32Types).Format(formats)
            .UnknownShapeFormat(formats);

        this->Attr("layout").AttrType(OPTIONAL).String("BNSD");
        this->Attr("scale").AttrType(OPTIONAL).Float(1.0F);
        this->Attr("chunk_size").AttrType(OPTIONAL).Int(64);
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1.0e-6F);
        this->Attr("use_qk_l2norm_in_kernel").AttrType(OPTIONAL).Bool(false);
        this->Attr("use_gate_in_kernel").AttrType(OPTIONAL).Bool(false);
        this->Attr("use_beta_sigmoid_in_kernel").AttrType(OPTIONAL).Bool(false);
        this->Attr("allow_neg_eigval").AttrType(OPTIONAL).Bool(false);
        this->Attr("safe_gate").AttrType(OPTIONAL).Bool(false);
        this->Attr("lower_bound").AttrType(OPTIONAL).Float(-5.0F);
        this->Attr("use_exp2").AttrType(OPTIONAL).Bool(false);
        // L2 根据 nullptr 输出组合设置；仅用于选择编译期搬出策略。
        this->Attr("output_mode").AttrType(OPTIONAL).Int(2);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(ChunkKdaFwdPrepare);

} // namespace ops
