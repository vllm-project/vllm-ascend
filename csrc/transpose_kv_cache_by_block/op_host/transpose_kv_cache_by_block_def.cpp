#include "register/op_def_registry.h"

namespace ops {
class TransposeKvCacheByBlock : public OpDef {
public:
    explicit TransposeKvCacheByBlock(const char* name) : OpDef(name)
    {
        this->Input("KCache")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("VCache")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("blockIDs")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("blockSize").Int();
        this->Attr("headNum").Int();
        this->Attr("headDim").Int();
        this->Attr("splitNum").Int();
        this->Attr("layerNum").Int();

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

    }
};

OP_ADD(TransposeKvCacheByBlock);
}
