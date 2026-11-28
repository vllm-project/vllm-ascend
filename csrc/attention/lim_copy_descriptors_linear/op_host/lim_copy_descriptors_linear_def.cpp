/**
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file lim_copy_descriptors_linear_def.cpp
 * \brief Operator definition for LimCopyDescriptorsLinear
 *
 * Moves missed KV rows from the registered host DRAM pool into the
 * fused-overlap selection buffer on NPU: inputs are the plan-owner layer's
 * miss plan (source logical tokens, LINEAR destination slots - the Python
 * side folds the logical->physical row map into the slots -, per-row
 * counts), the source block table for address resolution, the host/device
 * base pointers, an int64 params tensor carrying the deployment constants
 * (block size, capacity, miss stride, kv dim, rope dim, row capacity), and
 * an int64 active_geometry tensor carrying the per-step geometry (active
 * rows, block-table width). The operator has no outputs: its effect is the
 * copy itself.
 */
#include "register/op_def_registry.h"

namespace ops {
class LimCopyDescriptorsLinear : public OpDef {
public:
    explicit LimCopyDescriptorsLinear(const char* name) : OpDef(name) {
        // -------------------- Inputs --------------------
        this->Input("miss_src_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("miss_dst_slots")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("miss_counts")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("source_block_table")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("host_ptrs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("dev_ptrs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("params")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("active_geometry")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // -------------------- Platform --------------------
        // Sparse KV offload (the fused-overlap pipeline and this onboarding
        // op) is built for Atlas A3 only; keep the platform list in sync
        // with the "lim_copy_descriptors_linear" entry in csrc/build_aclnn.sh.
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(LimCopyDescriptorsLinear);

}  // namespace ops
