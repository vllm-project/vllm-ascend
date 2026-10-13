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
 * \file fused_quant_scatter_copy_sparse_flash_attention_def.cpp
 * \brief
 */

#include <vector>
#include "register/op_def_registry.h"

namespace ops {
class FusedQuantScatterCopySparseFlashAttention : public OpDef {
public:
    explicit FusedQuantScatterCopySparseFlashAttention(const char *name) : OpDef(name)
    {
        const std::vector<ge::DataType> queryTypes = {ge::DT_BF16, ge::DT_FLOAT16};
        const std::vector<ge::DataType> byteTypes = {ge::DT_INT8, ge::DT_INT8};
        const std::vector<ge::DataType> indexTypes = {ge::DT_INT32, ge::DT_INT32};
        const std::vector<ge::Format> formats = {ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("query").ParamType(REQUIRED).DataType(queryTypes).Format(formats).AutoContiguous();
        this->Input("hbm_kv_bytes").ParamType(REQUIRED).DataType(byteTypes).Format(formats).IgnoreContiguous();
        this->Input("dram_kv_bytes").ParamType(REQUIRED).DataType(byteTypes).Format(formats).IgnoreContiguous();
        for (const char *name : {"hbm_block_table", "dram_block_table", "topk_dst_slots", "topk_src_ids",
             "topk_miss_counts", "actual_seq_lengths_query", "actual_seq_lengths_kv", "num_cache_tokens",
             "miss_src_ids", "miss_dst_slots", "miss_counts"}) {
            this->Input(name).ParamType(REQUIRED).DataType(indexTypes).Format(formats).AutoContiguous();
        }
        this->Output("hbm_kv_bytes").ParamType(REQUIRED).DataType(byteTypes).Format(formats);
        this->Output("attention_out").ParamType(REQUIRED).DataType(queryTypes).Format(formats);
        this->Attr("scale_value").AttrType(REQUIRED).Float();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(FusedQuantScatterCopySparseFlashAttention);
} // namespace ops
