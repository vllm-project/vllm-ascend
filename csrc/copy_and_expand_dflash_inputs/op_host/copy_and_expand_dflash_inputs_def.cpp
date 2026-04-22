/**
 * @file copy_and_expand_dflash_inputs_def.cpp
 * @brief CopyAndExpandDflashInputs OpDef registration
 */

#include "register/op_def_registry.h"

namespace ops {

class CopyAndExpandDflashInputs : public OpDef {
public:
    explicit CopyAndExpandDflashInputs(const char* name) : OpDef(name)
    {
        // -------------------- Inputs --------------------
        this->Input("next_token_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("target_positions")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("query_start_loc")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("num_rejected_tokens")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("block_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // -------------------- In-place Outputs (as Inputs) --------------------
        this->Input("out_input_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("out_context_positions")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("out_query_positions")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("out_context_slot_mapping")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("out_query_slot_mapping")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("out_token_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // -------------------- Outputs (empty for in-place) --------------------

        // -------------------- Attributes --------------------
        this->Attr("parallel_drafting_token_id").Int();
        this->Attr("num_query_per_req").Int();
        this->Attr("num_speculative_tokens").Int();
        this->Attr("block_size").Int();
        this->Attr("total_input_tokens").Int();
        this->Attr("has_num_rejected").Bool();

        // -------------------- Platform --------------------
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(CopyAndExpandDflashInputs);

}  // namespace ops