/**
 * SPDX-License-Identifier: Apache-2.0
 */

/*!
 * \file lim_copy_descriptors_def.cpp
 * \brief Operator definition for LimCopyDescriptors
 *
 * Moves missed KV rows from the registered host DRAM pool into per-request
 * hot rows: inputs are the owner LIM's miss plan (source logical tokens,
 * destination hot-row slots, per-request counts), the source block table
 * and pool entries for address resolution, the host/device base pointers,
 * an int64 params tensor carrying the deployment constants
 * (block size, hot tokens, miss stride, kv dim, rope dim), and an int64
 * active_geometry tensor carrying the per-step geometry (active rows,
 * block-table width). The operator has no outputs: its effect is the copy
 * itself.
 */
#include "register/op_def_registry.h"

namespace ops {
class LimCopyDescriptors : public OpDef {
public:
    explicit LimCopyDescriptors(const char* name) : OpDef(name) {
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
        this->Input("pool_entries")
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
        // Sparse KV offload (LIM, Copy-SFA, this onboarding op) is built for
        // Atlas A3 only; keep the platform list in sync with the
        // "lim_copy_descriptors" entry in csrc/build_aclnn.sh.
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(LimCopyDescriptors);

}  // namespace ops
