/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_pa_kv_cache_vllm_def.cpp
 * \brief scatter_pa_kv_cache_vllm op host
 */

#include "register/op_def_registry.h"


namespace ops {
static const std::vector<ge::DataType> keyDataType = {
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
};

static const std::vector<ge::DataType> DataType_950 = {
    ge::DT_FLOAT,  ge::DT_FLOAT16, ge::DT_BF16,   ge::DT_INT8,     ge::DT_UINT8,       ge::DT_INT16,
    ge::DT_UINT16, ge::DT_INT32,   ge::DT_UINT32, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN,
    ge::DT_FLOAT,  ge::DT_FLOAT16, ge::DT_BF16,   ge::DT_INT8,     ge::DT_UINT8,       ge::DT_INT16,
    ge::DT_UINT16, ge::DT_INT32,   ge::DT_UINT32, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN,
};

static const std::vector<ge::DataType> valueDataType = {
    ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
    ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
};

static const std::vector<ge::DataType> indexDataType = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
};

static const std::vector<ge::DataType> indexDataType_950 = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
};

static const std::vector<ge::Format> DataFormat = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
};

static const std::vector<ge::Format> DataFormat_950 = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
};

class ScatterPaKvCacheVllm : public OpDef {
public:
    explicit ScatterPaKvCacheVllm(const char *name) : OpDef(name)
    {
        this->Input("key").ParamType(REQUIRED).DataType(keyDataType).Format(DataFormat).UnknownShapeFormat(DataFormat);
        this->Input("key_cache")
            .ParamType(REQUIRED)
            .DataType(keyDataType)
            .Format(DataFormat)
            .UnknownShapeFormat(DataFormat)
            .IgnoreContiguous();
        this->Input("slot_mapping")
            .ParamType(REQUIRED)
            .DataType(indexDataType)
            .Format(DataFormat)
            .UnknownShapeFormat(DataFormat);
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType(valueDataType)
            .Format(DataFormat)
            .UnknownShapeFormat(DataFormat);
        this->Input("value_cache")
            .ParamType(REQUIRED)
            .DataType(valueDataType)
            .Format(DataFormat)
            .UnknownShapeFormat(DataFormat)
            .IgnoreContiguous();
        this->Input("compress_lens")
            .ParamType(OPTIONAL)
            .DataType(indexDataType)
            .Format(DataFormat)
            .UnknownShapeFormat(DataFormat);
        this->Input("compress_seq_offset")
            .ParamType(OPTIONAL)
            .DataType(indexDataType)
            .Format(DataFormat)
            .UnknownShapeFormat(DataFormat);
        this->Input("seq_lens")
            .ParamType(OPTIONAL)
            .DataType(indexDataType)
            .Format(DataFormat)
            .UnknownShapeFormat(DataFormat);
        this->Output("key_cache")
            .ParamType(REQUIRED)
            .DataType(keyDataType)
            .Format(DataFormat)
            .UnknownShapeFormat(DataFormat);
        this->Output("value_cache")
            .ParamType(REQUIRED)
            .DataType(valueDataType)
            .Format(DataFormat)
            .UnknownShapeFormat(DataFormat);
        this->Attr("cache_mode").AttrType(OPTIONAL).String("Norm");
        this->Attr("scatter_mode").AttrType(OPTIONAL).String("None");
        this->Attr("strides").AttrType(OPTIONAL).ListInt({1, 1});
        this->Attr("offsets").AttrType(OPTIONAL).ListInt({0, 0});
        this->Attr("kv_cache_strides").AttrType(OPTIONAL).ListInt({0, 0});

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

        // OpAICoreConfig config_950;
        // config_950.Input("key")
        //     .ParamType(REQUIRED)
        //     .DataType(DataType_950)
        //     .Format(DataFormat_950)
        //     .UnknownShapeFormat(DataFormat_950);
        // config_950.Input("key_cache")
        //     .ParamType(REQUIRED)
        //     .DataType(DataType_950)
        //     .Format(DataFormat_950)
        //     .UnknownShapeFormat(DataFormat_950)
        //     .IgnoreContiguous();
        // config_950.Input("slot_mapping")
        //     .ParamType(REQUIRED)
        //     .DataType(indexDataType_950)
        //     .Format(DataFormat_950)
        //     .UnknownShapeFormat(DataFormat_950);
        // config_950.Input("value")
        //     .ParamType(REQUIRED)
        //     .DataType(DataType_950)
        //     .Format(DataFormat_950)
        //     .UnknownShapeFormat(DataFormat_950);
        // config_950.Input("value_cache")
        //     .ParamType(REQUIRED)
        //     .DataType(DataType_950)
        //     .Format(DataFormat_950)
        //     .UnknownShapeFormat(DataFormat_950)
        //     .IgnoreContiguous();
        // config_950.Input("compress_lens")
        //     .ParamType(OPTIONAL)
        //     .DataType(indexDataType_950)
        //     .Format(DataFormat_950)
        //     .UnknownShapeFormat(DataFormat_950);
        // config_950.Input("compress_seq_offset")
        //     .ParamType(OPTIONAL)
        //     .DataType(indexDataType_950)
        //     .Format(DataFormat_950)
        //     .UnknownShapeFormat(DataFormat_950);
        // config_950.Input("seq_lens")
        //     .ParamType(OPTIONAL)
        //     .DataType(indexDataType_950)
        //     .Format(DataFormat_950)
        //     .UnknownShapeFormat(DataFormat_950);
        // config_950.Output("key_cache")
        //     .ParamType(REQUIRED)
        //     .DataType(DataType_950)
        //     .Format(DataFormat_950)
        //     .UnknownShapeFormat(DataFormat_950);
        // config_950.Output("value_cache")
        //     .ParamType(REQUIRED)
        //     .DataType(DataType_950)
        //     .Format(DataFormat_950)
        //     .UnknownShapeFormat(DataFormat_950);
        // config_950.DynamicCompileStaticFlag(true)
        //     .DynamicFormatFlag(false)
        //     .DynamicRankSupportFlag(true)
        //     .DynamicShapeSupportFlag(true)
        //     .NeedCheckSupportFlag(false)
        //     .PrecisionReduceFlag(true)
        //     .ExtendCfgInfo("opFile.value", "scatter_pa_kv_cache_vllm_apt");
        // this->AICore().AddConfig("ascend950", config_950);
    }
};
OP_ADD(ScatterPaKvCacheVllm);
} // namespace ops