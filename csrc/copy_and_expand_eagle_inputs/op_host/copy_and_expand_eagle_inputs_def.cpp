/**
 * @file copy_and_expand_eagle_inputs_def.cpp
 * @brief CopyAndExpandEagleInputs OpDef registration
 */

#include "register/op_def_registry.h"

namespace ops {

class CopyAndExpandEagleInputs : public OpDef {
public:
    explicit CopyAndExpandEagleInputs(const char* name) : OpDef(name)
    {
        // -------------------- Inputs --------------------
        this->Input("target_token_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("target_positions")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("next_token_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("query_start_loc")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("query_end_loc")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // -------------------- Outputs --------------------
        this->Output("out_input_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out_positions")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out_is_rejected_token_mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out_is_masked_token_mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out_new_token_indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out_hidden_state_mapping")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // -------------------- Attributes --------------------
        this->Attr("padding_token_id").Int();
        this->Attr("parallel_drafting_token_id").Int();
        this->Attr("num_padding_slots_per_request").Int();
        this->Attr("shift_input_ids").Bool();
        this->Attr("total_input_tokens").Int();

        // -------------------- Platform --------------------
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(CopyAndExpandEagleInputs);

}  // namespace ops
