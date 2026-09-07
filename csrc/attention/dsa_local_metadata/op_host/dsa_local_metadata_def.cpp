/**
 * @file dsa_local_metadata_def.cpp
 * @brief DsaLocalMetadata OpDef registration
 *
 * Fused kernel that computes DSA context-parallel local token metadata:
 *   lqs/lqe = clamp(query_start_loc[i] / [i+1], local_start, local_end)
 *   local_query_start_loc = [0, cumsum(lqe - lqs)]
 *   local_seq_lens = where((lql > 0) & (seq_lens > 0), max(seq_lens - offset, 0), 0)
 *   start_pos = seq_lens - (q_end - q_start)   [optional]
 */

#include "register/op_def_registry.h"

namespace ops {

class DsaLocalMetadata : public OpDef {
public:
    explicit DsaLocalMetadata(const char* name) : OpDef(name)
    {
        // -------------------- Inputs --------------------
        this->Input("query_start_loc")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("seq_lens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // -------------------- Outputs --------------------
        this->Output("local_query_start_loc")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("local_seq_lens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("start_pos_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        // -------------------- Attributes --------------------
        this->Attr("local_start").Int();
        this->Attr("local_end").Int();
        this->Attr("num_reqs").Int();
        this->Attr("compute_start_pos").Bool();

        // -------------------- Platform --------------------
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(DsaLocalMetadata);

} // namespace ops
