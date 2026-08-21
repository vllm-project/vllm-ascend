/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */

#include "register/op_def_registry.h"

namespace ops {
class KdaLayoutSwap12 : public OpDef {
public:
    explicit KdaLayoutSwap12(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> dataTypes = {
            ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16
        };
        const std::initializer_list<ge::DataType> dependencyTypes = {
            ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT
        };
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
        };

        this->Input("x").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("dependency").ParamType(OPTIONAL)
            .DataType(dependencyTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("y").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);

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

OP_ADD(KdaLayoutSwap12);
} // namespace ops
