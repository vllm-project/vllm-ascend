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
 * \file turbo_quant_sparse_flash_attention_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
class TurboQuantSparseFlashAttention : public OpDef {
public:
    explicit TurboQuantSparseFlashAttention(const char *name)
        : OpDef(name)
    {
        this->Input("query").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("key").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND}).AutoContiguous();
        // MLA 场景下 K 与 V 为同一份 latent，本算子不单独读取 value；
        // 保留该输入是为与 c8 稀疏注意力路径的接口保持一致，调用方传入与 key
        // 相同的张量即可。host 侧校验其 dtype 与 shape 必须与 key 一致。
        this->Input("value").ParamType(REQUIRED).Follow("key").AutoContiguous();
        this->Input("sparse_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        // quant_scale_repo_mode 仅支持 1（COMBINE）：反量化系数与 Nope / Rope 合并存放
        // 在每个 KV slot 的尾部 2 字节内，由 kernel 就地读取。下列两个可选输入在该模式下
        // 不被消费，保留是为与 c8 稀疏注意力路径的接口保持一致，调用方可不传。
        this->Input("key_dequant_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("value_dequant_scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        // 下列三个输入在当前唯一支持的 TND + PA_BSND 路径下均为必需，故声明为 REQUIRED：
        // 缺 actual_seq_lengths_query 无法切分 TND 的变长 query，缺 block_table 或
        // actual_seq_lengths_kv 无法定位 PageAttention 的 KV。
        this->Input("block_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths_query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths_kv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("attention_out").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        this->Output("softmax_max").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("softmax_sum").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Attr("scale_value").AttrType(REQUIRED).Float(1.0);
        // 下列默认值须与 host 侧 tiling 校验的唯一合法取值保持一致，否则绕过
        // PyTorch 封装、直接按声明默认值调用 aclnn 接口时会被 tiling 拒绝。
        this->Attr("key_quant_mode").AttrType(REQUIRED).Int(3);   // 3: TQ4 码本量化
        this->Attr("value_quant_mode").AttrType(REQUIRED).Int(3); // 3: TQ4 码本量化
        this->Attr("sparse_block_size").AttrType(OPTIONAL).Int(1);
        this->Attr("layout_query").AttrType(OPTIONAL).String("TND");  // 当前仅支持 TND
        this->Attr("layout_kv").AttrType(OPTIONAL).String("PA_BSND"); // 当前仅支持 PA_BSND
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);          // 3:默认值，只计算下三角
        this->Attr("pre_tokens").AttrType(OPTIONAL).Int(INT64_MAX);
        this->Attr("next_tokens").AttrType(OPTIONAL).Int(INT64_MAX);
        this->Attr("attention_mode").AttrType(OPTIONAL).Int(2);        // 2: MLA-absorb
        this->Attr("quant_scale_repo_mode").AttrType(OPTIONAL).Int(1); // 1: COMBINE，scale 合并存放于 KV slot 内
        this->Attr("tile_size").AttrType(OPTIONAL).Int(128);           // 128:默认值
        this->Attr("rope_head_dim").AttrType(OPTIONAL).Int(64);        // 64:默认值
        this->Attr("return_softmax_lse").AttrType(OPTIONAL).Bool(false);
        OpAICoreConfig aicore_config;
        // DynamicShapeSupportFlag 保留：推理场景下 query 的 T 与各 batch 的 KV 长度逐步
        // 变化，属于典型动态 shape。DynamicRankSupportFlag 关闭：InferShape 已要求
        // query 为 3 维、key 为 4 维，未知 rank 的输入必然被拒绝。
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};
OP_ADD(TurboQuantSparseFlashAttention);
} // namespace ops
