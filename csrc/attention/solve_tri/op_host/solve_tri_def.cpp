/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * BSD 3-Clause License.
 */
 #include "register/op_def_registry.h"

 namespace ops {
 class SolveTri : public OpDef {
 public:
     explicit SolveTri(const char* name) : OpDef(name)
     {
         this->Input("x")
             .ParamType(REQUIRED)
             .DataType({ge::DT_FLOAT16, ge::DT_BF16})
             .FormatList({ge::FORMAT_ND})
             .AutoContiguous();
 
         this->Input("cu_seqlens")
             .ParamType(OPTIONAL)
             .DataTypeList({ge::DT_INT64})
             .FormatList({ge::FORMAT_ND})
             .ValueDepend(OPTIONAL);
 
         this->Input("chunk_indices")
             .ParamType(OPTIONAL)
             .DataTypeList({ge::DT_INT64})
             .FormatList({ge::FORMAT_ND})
             .ValueDepend(OPTIONAL);
 
         this->Output("x_out")
             .ParamType(REQUIRED)
             .DataType({ge::DT_FLOAT16, ge::DT_BF16})
             .FormatList({ge::FORMAT_ND})
             .AutoContiguous();
 
         this->Attr("layout")
             .AttrType(OPTIONAL)
             .String("bsnd");
 
         OpAICoreConfig aicore_config;
         aicore_config.DynamicCompileStaticFlag(true)
             .DynamicFormatFlag(true)
             .DynamicRankSupportFlag(false)
             .DynamicShapeSupportFlag(true)
             .NeedCheckSupportFlag(false)
             .PrecisionReduceFlag(true)
             .ExtendCfgInfo("opFile.value", "solve_tri")
             .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
 
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};
 OP_ADD(SolveTri);
 }  // namespace ops
 