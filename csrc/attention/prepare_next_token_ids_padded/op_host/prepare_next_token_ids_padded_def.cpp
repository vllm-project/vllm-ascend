#include "register/op_def_registry.h"

namespace ops {
    
class PrepareNextTokenIdsPadded : public OpDef {
public:
    explicit PrepareNextTokenIdsPadded(const char* name) : OpDef(name)
    {
        this->Input("sampledTokenIds")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("discardRequestMask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("backupNextTokenIds")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        
        
        this->Attr("vocabSize").AttrType(REQUIRED).Int();
        
        this->Output("nextTokenIds")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("validSampledTokensCount")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo(
                "aclnnSupport.value",
                "support_aclnn")
            .ExtendCfgInfo(
                "jitCompile.flag",
                "static_true");
        
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);

    }
};

OP_ADD(PrepareNextTokenIdsPadded);

}  // namespace ops
