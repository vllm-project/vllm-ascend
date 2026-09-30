/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file chunk_fwd_h_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {

class ChunkFwdH : public OpDef {
public:
    explicit ChunkFwdH(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> dataTypes = {ge::DT_BF16, ge::DT_BF16,
                                                               ge::DT_BF16, ge::DT_BF16};
        const std::initializer_list<ge::DataType> gateTypes = {ge::DT_BF16, ge::DT_BF16,
                                                               ge::DT_FLOAT, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> stateTypes = {ge::DT_BF16, ge::DT_FLOAT,
                                                                ge::DT_BF16, ge::DT_FLOAT};
        const std::initializer_list<ge::DataType> indexTypes = {ge::DT_INT64, ge::DT_INT64,
                                                                ge::DT_INT64, ge::DT_INT64};
        const std::initializer_list<ge::Format> formats = {ge::FORMAT_ND, ge::FORMAT_ND,
                                                           ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("k")
            .ParamType(REQUIRED)
            .DataType(dataTypes).Format(formats).UnknownShapeFormat(formats)
            .AutoContiguous();

        this->Input("w")
            .ParamType(REQUIRED)
            .DataType(dataTypes).Format(formats).UnknownShapeFormat(formats)
            .AutoContiguous();

        this->Input("u")
            .ParamType(REQUIRED)
            .DataType(dataTypes).Format(formats).UnknownShapeFormat(formats)
            .AutoContiguous();

        this->Input("g")
            .ParamType(OPTIONAL)
            .DataType(gateTypes).Format(formats).UnknownShapeFormat(formats)
            .AutoContiguous();

        this->Input("gk")
            .ParamType(OPTIONAL)
            .DataType(gateTypes).Format(formats).UnknownShapeFormat(formats)
            .AutoContiguous();

        this->Input("initial_state")
            .ParamType(OPTIONAL)
            .DataType(stateTypes).Format(formats).UnknownShapeFormat(formats)
            .AutoContiguous();

        this->Input("cu_seqlens")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataType(indexTypes).Format(formats).UnknownShapeFormat(formats)
            .AutoContiguous();

        this->Input("chunk_indices")
            .ParamType(OPTIONAL)
            .ValueDepend(OPTIONAL)
            .DataType(indexTypes).Format(formats).UnknownShapeFormat(formats)
            .AutoContiguous();

        this->Output("h")
            .ParamType(REQUIRED)
            .DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);

        this->Output("v_new")
            .ParamType(REQUIRED)
            .DataType(dataTypes).Format(formats).UnknownShapeFormat(formats);

        this->Output("final_state")
            .ParamType(OPTIONAL)
            .DataType(stateTypes).Format(formats).UnknownShapeFormat(formats);

        this->Attr("output_final_state").AttrType(REQUIRED).Bool(false);
        this->Attr("chunk_size").AttrType(REQUIRED).Int(64);
        this->Attr("save_new_value").AttrType(REQUIRED).Bool(true);
        this->Attr("use_exp2").AttrType(REQUIRED).Bool(false);
        this->Attr("state_v_first").AttrType(REQUIRED).Bool(false);
        this->Attr("logical_batch").AttrType(REQUIRED).Int(1);
        this->Attr("logical_seqlen").AttrType(REQUIRED).Int(1);
        this->Attr("logical_k_heads").AttrType(REQUIRED).Int(1);
        this->Attr("logical_v_heads").AttrType(REQUIRED).Int(1);
        this->Attr("logical_k_dim").AttrType(REQUIRED).Int(1);
        this->Attr("logical_v_dim").AttrType(REQUIRED).Int(1);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");

        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
        this->AICore().AddConfig("ascend950", aicore_config);

    }
};

OP_ADD(ChunkFwdH);

} // namespace ops
