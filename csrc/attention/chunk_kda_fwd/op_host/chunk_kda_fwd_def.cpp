/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "register/op_def_registry.h"

namespace ops {
class ChunkKdaFwd : public OpDef {
public:
    explicit ChunkKdaFwd(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> dataTypes = {
            ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
            ge::DT_FLOAT16, ge::DT_BF16
        };
        const std::initializer_list<ge::DataType> stateTypes = {
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT, ge::DT_FLOAT
        };
        const std::initializer_list<ge::DataType> akkTypes = {
            ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT16, ge::DT_BF16
        };
        const std::initializer_list<ge::DataType> outputDataTypes = {
            ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT,
            ge::DT_FLOAT16, ge::DT_BF16
        };
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND
        };

        this->Input("q").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("k").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("v").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("gk").ParamType(REQUIRED)
            .DataType(stateTypes)
            .Format(formats).UnknownShapeFormat(formats);
        this->Input("beta").ParamType(REQUIRED)
            .DataType(stateTypes)
            .Format(formats).UnknownShapeFormat(formats);
        this->Input("initial_state").ParamType(OPTIONAL).DataType(stateTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64})
            .Format(formats).UnknownShapeFormat(formats);
        this->Input("chunk_indices").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                       ge::DT_INT64, ge::DT_INT64})
            .Format(formats).UnknownShapeFormat(formats);
        this->Input("stage_qg").ParamType(OPTIONAL).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("stage_aqk").ParamType(OPTIONAL).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("stage_v_new").ParamType(OPTIONAL).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Input("stage_h").ParamType(OPTIONAL).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);

        this->Output("o").ParamType(REQUIRED).DataType(outputDataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("final_state").ParamType(REQUIRED).DataType(stateTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("Aqk").ParamType(REQUIRED).DataType(stateTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("Akk").ParamType(REQUIRED).DataType(akkTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("w").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("u").ParamType(REQUIRED).DataType(outputDataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("qg").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("kg").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("v_new").ParamType(REQUIRED).DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("h").ParamType(REQUIRED).DataType(stateTypes).Format(formats).UnknownShapeFormat(formats);

        this->Attr("scale").AttrType(REQUIRED).Float(1.0);
        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);
        this->Attr("output_final_state").AttrType(REQUIRED).Bool(false);
        this->Attr("total_chunks").AttrType(REQUIRED).Int(1);
        this->Attr("stage").AttrType(OPTIONAL).Int(0);

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

OP_ADD(ChunkKdaFwd);
} // namespace ops
