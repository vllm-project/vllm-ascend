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
 * \file lightning_indexer_def.cpp
 * \brief
 */
#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
class LightningIndexer : public OpDef {
public:
    explicit LightningIndexer(const char *name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths_query")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths_key")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("sparse_indices").ParamType(REQUIRED).DataTypeList({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        this->Attr("layout_query").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_key").AttrType(OPTIONAL).String("PA_BSND");
        this->Attr("sparse_count").AttrType(OPTIONAL).Int(2048); // 2048: Default value, filter the top 2048
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3); // 3: Default value, only calculate the lower triangular matrix
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};
OP_ADD(LightningIndexer);
} // namespace ops