/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#include "register/op_def_registry.h"

namespace ops {
class BuildDsparkSwaIndices : public OpDef {
public:
    explicit BuildDsparkSwaIndices(const char* name) : OpDef(name)
    {
        // 3 inputs — all int32
        this->Input("kvBlockTable")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("queryStartLoc")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("seqLens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        // 1 output — int32 [numDecodeTokens, 1, indexWidth]
        this->Output("perTokenSlots")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // 3 scalar attrs — the rest (indexWidth, numDecodeTokens, numReqs)
        // are derived from shapes inside the tiling function.
        this->Attr("numSpeculativeTokens").Int();
        this->Attr("windowSize").Int();
        this->Attr("blockSize").Int();

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(BuildDsparkSwaIndices);
}  // namespace ops
