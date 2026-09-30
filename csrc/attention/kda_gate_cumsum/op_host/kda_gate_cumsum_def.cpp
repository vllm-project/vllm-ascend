/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#include "register/op_def_registry.h"

namespace ops {
class KdaGateCumsum : public OpDef {
public:
    explicit KdaGateCumsum(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> gateTypes = {
            ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16
        };
        const std::initializer_list<ge::DataType> floatTypes = {
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT
        };
        const std::initializer_list<ge::DataType> intTypes = {
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64
        };
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
        };

        this->Input("g").ParamType(REQUIRED).DataType(gateTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("a_log").ParamType(OPTIONAL).DataType(floatTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("dt_bias").ParamType(OPTIONAL).DataType(floatTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(intTypes).Format(formats).UnknownShapeFormat(formats);

        this->Output("gk").ParamType(REQUIRED).DataType(floatTypes).Format(formats).UnknownShapeFormat(formats);

        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);
        this->Attr("use_gate_in_kernel").AttrType(REQUIRED).Bool(false);
        this->Attr("safe_gate").AttrType(REQUIRED).Bool(false);
        this->Attr("lower_bound").AttrType(REQUIRED).Float(-5.0);
        this->Attr("layout").AttrType(OPTIONAL).String("BSND");

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");

        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(KdaGateCumsum);
} // namespace ops
