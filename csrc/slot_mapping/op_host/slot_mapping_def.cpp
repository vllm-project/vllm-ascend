/**
 * @file slot_mapping_def.cpp
 * @brief SlotMapping OpDef 注册（vllm-ascend 风格：Tiling / InferShape 外挂）
 *
 * 接口精简：原先 `num_tokens / max_num_tokens / block_table_stride / block_size` 四个
 * 标量被硬塞成 int32 tensor（其中 `num_tokens_t = full((max_num_tokens,), val)` 浪费
 * 16 KB GM），此版本改为 Attr int64 一次性矫正。输入只保留真正的 tensor：
 * queryStartLoc / positions / blockTable。
 */

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

        // -------------------- Attributes (标量) --------------------
        // 必填：描述输入/输出的规模参数，取代原来的 4 个 tensor 输入
        this->Attr("numTokens").AttrType(REQUIRED).Int();
        this->Attr("maxNumTokens").AttrType(REQUIRED).Int();
        this->Attr("blockSize").AttrType(REQUIRED).Int();
        // 可选：CP（Context Parallelism）相关
        this->Attr("totalCpWorldSize").AttrType(OPTIONAL).Int(1);
        this->Attr("totalCpRank").AttrType(OPTIONAL).Int(0);
        this->Attr("cpKvCacheInterleaveSize").AttrType(OPTIONAL).Int(1);
        this->Attr("padId").AttrType(OPTIONAL).Int(-1);

        // -------------------- Platform --------------------
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SlotMapping);

}  // namespace ops
