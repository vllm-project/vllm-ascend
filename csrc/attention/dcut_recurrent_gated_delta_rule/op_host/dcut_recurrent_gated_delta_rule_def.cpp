#include "register/op_def_registry.h"

namespace ops {
class DcutRecurrentGatedDeltaRule : public OpDef {
 public:
  explicit DcutRecurrentGatedDeltaRule(const char* name) : OpDef(name) {
    this->Input("query")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("key")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("value")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("beta")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("state")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("query_start_loc")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("ssm_state_indices")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("g")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("gk")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    // Previous-step accepted counts select a column in the fixed
    // ssm_state_indices[B, S] rows, independent of this step's length.
    this->Input("num_accepted_tokens")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("out")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_BF16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("state")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("scale_value").AttrType(OPTIONAL).Float(1.0);

    OpAICoreConfig aicConfig;
    aicConfig.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .ExtendCfgInfo("softsync.flag", "true");
    this->AICore().AddConfig("ascend910b", aicConfig);
    this->AICore().AddConfig("ascend910_93", aicConfig);
    this->AICore().AddConfig("ascend950", aicConfig);
  }
};

OP_ADD(DcutRecurrentGatedDeltaRule);
}  // namespace ops
