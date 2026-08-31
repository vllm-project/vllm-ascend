/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_mla_with_kvcache_def.cpp
 * \brief FlashMlaWithKvcache算子定义（训练推理归一，仅非量化）
 *        输入数据类型仅支持FLOAT16和BFLOAT16。
 *        无独立 v / q_rope / k_rope 输入：rope 已合并进 q 与 k_cache（最后维 576 = nope 512 + rope 64，
 *        head_dim_v 为 k_cache 内 nope（value）段宽度，head_dim_v + 64 == 576 由 checker/infershape 保证）。
 *        cache_seqlens 携带逐 batch 的 KV cache 长度（替代 flash_attn 的 cu_seqlens_kv/seqused_kv，KV 走分页）。
 */

#include "register/op_def_registry.h"

namespace ops {

class FlashMlaWithKvcache : public OpDef {
public:
    explicit FlashMlaWithKvcache(const char *name)
        : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("k_cache")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cache_seqlens")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("cu_seqlens_q")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("seqused_q")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("attn_mask")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("metadata")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("attn_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("softmax_lse")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("head_dim_v").AttrType(OPTIONAL).Int(512);
        this->Attr("softmax_scale").AttrType(OPTIONAL).Float(0.0f);
        this->Attr("mask_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("max_seqlen_q").AttrType(OPTIONAL).Int(-1);
        this->Attr("max_seqlen_kv").AttrType(OPTIONAL).Int(-1);
        this->Attr("layout_q").AttrType(OPTIONAL).String("TND");
        this->Attr("layout_kv").AttrType(OPTIONAL).String("PA_NZ");
        this->Attr("layout_out").AttrType(OPTIONAL).String("NTD");
        this->Attr("return_softmax_lse").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicore_config_95;
        aicore_config_95.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("opFile.value", "flash_mla_with_kvcache");

        this->AICore().AddConfig("ascend950", aicore_config_95);
    }
};

OP_ADD(FlashMlaWithKvcache);

} // namespace ops
