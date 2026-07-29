/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include <vector>

#include "register/op_def_registry.h"

namespace ops {
class KvCacheFullBlockDump : public OpDef {
public:
    explicit KvCacheFullBlockDump(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> payloadTypes = {
            ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8};
        const std::vector<ge::Format> payloadFormats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
        const std::vector<ge::DataType> indexTypes = {
            ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
        const std::vector<ge::Format> indexFormats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("src_cache_0")
            .ParamType(REQUIRED)
            .DataType(payloadTypes)
            .Format(payloadFormats)
            .UnknownShapeFormat(payloadFormats);
        this->Input("src_cache_1")
            .ParamType(REQUIRED)
            .DataType(payloadTypes)
            .Format(payloadFormats)
            .UnknownShapeFormat(payloadFormats);
        this->Input("dst_cache_0")
            .ParamType(REQUIRED)
            .DataType(payloadTypes)
            .Format(payloadFormats)
            .UnknownShapeFormat(payloadFormats);
        this->Input("dst_cache_1")
            .ParamType(REQUIRED)
            .DataType(payloadTypes)
            .Format(payloadFormats)
            .UnknownShapeFormat(payloadFormats);
        this->Input("src_block_ids")
            .ParamType(REQUIRED)
            .DataType(indexTypes)
            .Format(indexFormats)
            .UnknownShapeFormat(indexFormats);
        this->Input("dst_block_ids")
            .ParamType(REQUIRED)
            .DataType(indexTypes)
            .Format(indexFormats)
            .UnknownShapeFormat(indexFormats);

        // Outputs share names with their mutable inputs so aclnn treats the
        // destination cache planes as in-place tensors. The kernel writes the
        // input GM pointers directly.
        this->Output("dst_cache_0")
            .ParamType(REQUIRED)
            .DataType(payloadTypes)
            .Format(payloadFormats)
            .UnknownShapeFormat(payloadFormats);
        this->Output("dst_cache_1")
            .ParamType(REQUIRED)
            .DataType(payloadTypes)
            .Format(payloadFormats)
            .UnknownShapeFormat(payloadFormats);

        this->AICore()
            .AddConfig("ascend910_93")
            .AddConfig("ascend910b");
    }
};

OP_ADD(KvCacheFullBlockDump);
}  // namespace ops
