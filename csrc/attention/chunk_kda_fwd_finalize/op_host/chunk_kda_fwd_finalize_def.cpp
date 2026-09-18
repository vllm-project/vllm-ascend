#include <initializer_list>

#include "register/op_def_registry.h"

namespace ops {

class ChunkKdaFwdFinalize : public OpDef {
public:
    explicit ChunkKdaFwdFinalize(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> bf16Types = {
            ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16};
        const std::initializer_list<ge::DataType> indexTypes = {
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("qg_scaled").ParamType(REQUIRED).DataType(bf16Types)
            .Format(formats).UnknownShapeFormat(formats).AutoContiguous();
        this->Input("aqk").ParamType(REQUIRED).DataType(bf16Types)
            .Format(formats).UnknownShapeFormat(formats).AutoContiguous();
        this->Input("v_new").ParamType(REQUIRED).DataType(bf16Types)
            .Format(formats).UnknownShapeFormat(formats).AutoContiguous();
        this->Input("h").ParamType(REQUIRED).DataType(bf16Types)
            .Format(formats).UnknownShapeFormat(formats).AutoContiguous();
        this->Input("cu_seqlens").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(indexTypes).Format(formats).UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Input("chunk_indices").ParamType(OPTIONAL).ValueDepend(OPTIONAL)
            .DataType(indexTypes).Format(formats).UnknownShapeFormat(formats)
            .AutoContiguous();
        this->Output("attn_out").ParamType(REQUIRED).DataType(bf16Types)
            .Format(formats).UnknownShapeFormat(formats);

        this->Attr("output_layout").AttrType(OPTIONAL).String("BSND");
        this->Attr("state_v_first").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("prebuildPattern.value", "Opaque")
            .ExtendCfgInfo("coreType.value", "AiCore")
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn");
        this->AICore().AddConfig("ascend950", config);
    }
};

OP_ADD(ChunkKdaFwdFinalize);

} // namespace ops
