#include "register/op_def_registry.h"

namespace ops {
class DcutCausalConv1d : public OpDef {
 public:
  explicit DcutCausalConv1d(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("weight")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("bias")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("convStates")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("queryStartLoc")
        .ParamType(OPTIONAL)
        .DataTypeList({ge::DT_INT32, ge::DT_INT64})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("cacheIndices")
        .ParamType(OPTIONAL)
        .DataTypeList({ge::DT_INT32, ge::DT_INT64})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("initialStateMode")
        .ParamType(OPTIONAL)
        .DataTypeList({ge::DT_BOOL, ge::DT_INT32, ge::DT_INT64})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    // Previous-step accepted counts; the kernel derives zero-based offsets.
    this->Input("numAcceptedTokens")
        .ParamType(OPTIONAL)
        .DataTypeList({ge::DT_INT32, ge::DT_INT64})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();

    this->Attr("activationMode").AttrType(OPTIONAL).Int(0);
    this->Attr("padSlotId").AttrType(OPTIONAL).Int(-1);
    this->Attr("runMode").AttrType(OPTIONAL).Int(1);

    OpAICoreConfig aicoreConfig;
    aicoreConfig.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(false)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("coreType.value", "AiCore");
    this->AICore().AddConfig("ascend910b", aicoreConfig);
    this->AICore().AddConfig("ascend910_93", aicoreConfig);
    this->AICore().AddConfig("ascend950", aicoreConfig);
  }
};

OP_ADD(DcutCausalConv1d);
}  // namespace ops
