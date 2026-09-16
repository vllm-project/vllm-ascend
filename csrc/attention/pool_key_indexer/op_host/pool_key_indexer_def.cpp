/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
class PoolKeyIndexer : public OpDef {
public:
    explicit PoolKeyIndexer(const char *name)
        : OpDef(name)
    {
        // ---- Base config: A2/A3 (ascend910b, ascend910_93) ----
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("pool_key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Input("weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("pool_tail_k")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();
        this->Input("actual_seq_q")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();
        this->Input("actual_seq_k")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("q_descale")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("k_descale")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Output("sparse_indices")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND});
        this->Output("sparse_values")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        this->Attr("topk").AttrType(OPTIONAL).Int(2048);
        this->Attr("pool_size").AttrType(OPTIONAL).Int(16);
        this->Attr("layout_q").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_k").AttrType(OPTIONAL).String("BSND");
        this->Attr("mask_mode").AttrType(OPTIONAL).Int(3);
        this->Attr("quant_mode").AttrType(OPTIONAL).Int(-1);
        this->Attr("return_value").AttrType(OPTIONAL).Bool(false);
        // pool_key 0轴非连续 stride(元素单位), 由 torch_extension 层从
        // at::Tensor::stride(0) 直读传入(参考 compressor 方案); -1 表示未指定
        // (连续输入), tiling 回退到 shape 推导。
        this->Attr("key_stride0").AttrType(OPTIONAL).Int(-1);
        // k_descale 0轴非连续 stride(元素单位, 量化场景), 与 key_stride0 同机制:
        // eager/aclnn 路径 GetDynamicInputStride 返回 nullptr, 属性是唯一来源;
        // -1 表示未指定(连续输入), tiling 回退到 shape 推导。
        this->Attr("k_descale_stride0").AttrType(OPTIONAL).Int(-1);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);

        // ---- 950 config: override only the inputs that differ from base (FP8 combos) ----
        //   6 combinations:
        //   1) FP16  quant_mode=-1 (no quant)     : q/k/weights=FP16,    descale=FLOAT
        //   2) BF16  quant_mode=-1 (no quant)     : q/k/weights=BF16,    descale=FLOAT
        //   3) FP8   quant_mode=0  (per-token FP8) : q/k=FP8_E4M3FN, weights=FP16, descale=FLOAT
        //   4) FP8   quant_mode=0  (per-token FP8) : q/k=FP8_E4M3FN, weights=BF16, descale=FLOAT
        //   5) FP8   quant_mode=1  (mxFP8)         : q/k=FP8_E4M3FN, weights=FP16, descale=FLOAT8_E8M0
        //   6) FP8   quant_mode=1  (mxFP8)         : q/k=FP8_E4M3FN, weights=BF16, descale=FLOAT8_E8M0
        // Note: pool_tail_k / actual_seq_q / actual_seq_k (DT_INT64 + ValueDepend) and
        //       block_table / outputs inherit from the top-level declaration; not
        //       redeclared here. INT64 is the canonical host-value dtype required by
        //       the ValueDepend convention (see fused_infer_attention_score et al.).
        // Note: 量化场景 weights 规格(文档+tiling)允许 FP16/BF16, kernel entry 按
        //       ORIG_DTYPE_WEIGHTS 分发 W_T(half/bfloat16_t), 与 quant_lightning_indexer
        //       的 weights 双 dtype 模式一致。
        OpAICoreConfig aicoreConfig950;
        aicoreConfig950.Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicoreConfig950.Input("pool_key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                       ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        aicoreConfig950.Input("weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicoreConfig950.Input("q_descale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        aicoreConfig950.Input("k_descale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0})
            .FormatList({ge::FORMAT_ND})
            .IgnoreContiguous();
        aicoreConfig950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", aicoreConfig950);
    }
};
OP_ADD(PoolKeyIndexer);
} // namespace ops
