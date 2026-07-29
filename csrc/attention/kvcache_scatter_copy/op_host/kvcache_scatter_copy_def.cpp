/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#include "register/op_def_registry.h"

namespace ops {
class KvcacheScatterCopy : public OpDef {
public:
    explicit KvcacheScatterCopy(const char* name) : OpDef(name)
    {
        this->Input("hbm_k_rope").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("hbm_kv_cache").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("dram_k_rope").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("dram_kv_cache").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("hbm_block_table").ParamType(REQUIRED).DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("dram_block_table").ParamType(REQUIRED).DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("src_token_ids").ParamType(REQUIRED).DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("dst_slots").ParamType(REQUIRED).DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("copy_counts").ParamType(REQUIRED).DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("hbm_k_rope").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("hbm_kv_cache").ParamType(REQUIRED).DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->AICore().AddConfig("ascend910_93").AddConfig("ascend910b");
    }
};
OP_ADD(KvcacheScatterCopy);
} // namespace ops
