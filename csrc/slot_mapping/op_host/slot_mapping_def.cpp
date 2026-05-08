#include "register/op_def_registry.h"

namespace ops {

class SlotMapping : public OpDef {
public:
    explicit SlotMapping(const char* name) : OpDef(name)
    {
        // -------------------- Inputs (真·Tensor) --------------------
        this->Input("queryStartLoc")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("positions")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("blockTable")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // -------------------- Outputs --------------------
        // Output dtype is int32 to match BlockTable.slot_mapping in vllm-ascend
        // (the downstream reshape_and_cache_bnsd kernel also reads it as int32).
        this->Output("slotMapping")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // -------------------- Attributes --------------------
        this->Attr("numTokens").AttrType(REQUIRED).Int();
        this->Attr("maxNumTokens").AttrType(REQUIRED).Int();
        this->Attr("blockSize").AttrType(REQUIRED).Int();
        // CP（Context Parallelism）
        this->Attr("totalCpWorldSize").AttrType(OPTIONAL).Int(1);
        this->Attr("totalCpRank").AttrType(OPTIONAL).Int(0);
        this->Attr("cpKvCacheInterleaveSize").AttrType(OPTIONAL).Int(1);
        this->Attr("padId").AttrType(OPTIONAL).Int(-1);

        // -------------------- Platform --------------------
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SlotMapping);

}  // namespace ops
