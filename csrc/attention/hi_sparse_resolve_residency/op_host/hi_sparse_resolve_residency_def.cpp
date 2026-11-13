// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"

namespace ops {
class HiSparseResolveResidency : public OpDef {
public:
    explicit HiSparseResolveResidency(const char* name) : OpDef(name) {
        const std::initializer_list<ge::Format> formats = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
        Input("logical").ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}).Format(formats).UnknownShapeFormat(formats).IgnoreContiguous();
        Input("requests").ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}).Format(formats).UnknownShapeFormat(formats).IgnoreContiguous();
        Input("states").ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}).Format(formats).UnknownShapeFormat(formats).IgnoreContiguous();
        Input("source").ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}).Format(formats).UnknownShapeFormat(formats).IgnoreContiguous();
        Input("resident").ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}).Format(formats).UnknownShapeFormat(formats).IgnoreContiguous();
        Input("hot_table").ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}).Format(formats).UnknownShapeFormat(formats).IgnoreContiguous();
        Input("valid_rows").ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}).Format(formats).UnknownShapeFormat(formats).IgnoreContiguous();
        Output("tags").ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}).Format(formats).UnknownShapeFormat(formats);
        Output("lru").ParamType(REQUIRED)
            .DataType({ge::DT_INT16, ge::DT_INT16, ge::DT_INT16}).Format(formats).UnknownShapeFormat(formats);
        Output("physical").ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}).Format(formats).UnknownShapeFormat(formats);
        Output("attention").ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}).Format(formats).UnknownShapeFormat(formats);
        Output("resolved").ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}).Format(formats).UnknownShapeFormat(formats);
        Output("counts").ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}).Format(formats).UnknownShapeFormat(formats);
        Output("swap_host").ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}).Format(formats).UnknownShapeFormat(formats);
        Output("swap_device").ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}).Format(formats).UnknownShapeFormat(formats);
        Output("swap_counts").ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}).Format(formats).UnknownShapeFormat(formats);
        Output("miss_mask").ParamType(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_UINT8, ge::DT_BOOL}).Format(formats).UnknownShapeFormat(formats);
        Output("stats").ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64}).Format(formats).UnknownShapeFormat(formats);
        Output("errors").ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}).Format(formats).UnknownShapeFormat(formats);
        Attr("geometry").AttrType(REQUIRED).ListInt();
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicShapeSupportFlag(true)
            .DynamicRankSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("prebuildPattern.value", "Opaque");
        AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(HiSparseResolveResidency);
}  // namespace ops
