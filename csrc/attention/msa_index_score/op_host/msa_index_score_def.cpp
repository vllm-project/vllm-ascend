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
 * \file msa_index_score_def.cpp
 * \brief MsaIndexScore 算子原型注册（对齐 docs/aclnnMsaIndexScore.md）。
 *
 * dtype 组合（按列表下标对齐）：
 *   0: query=bf16, key=bf16  （非量化）
 *   1: query=fp16, key=fp16  （非量化）
 *   2: query=fp16, key=int8  （前融合 int8，scale 为反量化系数）
 *
 * A2/A3：PageAttention BBND/BNBD key 与 TND packed key；sparse_mode∈{0,3}。不支持 FP8 / Ascend 950。
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
class MsaIndexScore : public OpDef {
public:
    explicit MsaIndexScore(const char *name)
        : OpDef(name)
    {
        // Q_idx, TND: [T1, N1, D]
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // K_idx：BBND [block_num, block_size, N2, D]、BNBD [block_num, N2, block_size, D]、TND [T2, N2, D]
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // PA 逻辑 block → 物理 page；PA 场景必须传入
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // 非量化：必须为空。int8：反量化系数 [block_num, N2, block_size]
        this->Input("scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // sparse_mode=3 时必选，压缩下三角模板 [2048, 2048]；内核按 rightDownCausal 解析，不逐元素加载
        this->Input("atten_mask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT8, ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // query 长度前缀和 (cumsum), [B+1]；TND 必选
        this->Input("actual_seq_qlen")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // PA：各请求可见 S2，[B]
        this->Input("actual_seq_klen")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // 当前 query 所在逻辑 block 索引，用于生成 local_mask，[B]
        this->Input("start_loc")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // [N1, T1, RoundUp(maxBlockNumPerSeq, 16)]
        this->Output("score")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // key 布局：TND / BBND / BNBD。aclnn 参数名为 layoutKeyOptional。
        this->Attr("layout_key").AttrType(OPTIONAL).String("BBND");
        // 0: defaultMask；3: rightDownCausal
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);
        // local_mask：强制选中 block 数（对齐 Triton prepare / HF；对比 raw score 时可置 0）
        this->Attr("init_blocks").AttrType(OPTIONAL).Int(0);
        this->Attr("local_blocks").AttrType(OPTIONAL).Int(1);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        // Atlas A2（ascend910b）/ A3（ascend910_93）；不注册 Ascend 950。
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
    }
};
OP_ADD(MsaIndexScore);
} // namespace ops
