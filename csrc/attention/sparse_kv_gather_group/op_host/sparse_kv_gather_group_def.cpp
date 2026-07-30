/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */
#include "register/op_def_registry.h"
namespace ops {
class SparseKvGatherGroup : public OpDef {
public:
    explicit SparseKvGatherGroup(const char *name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> cacheTypes = {
            ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16,
            ge::DT_BF16};
        const std::initializer_list<ge::DataType> tableAndTopkTypes = {
            ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
            ge::DT_INT64};
        const std::initializer_list<ge::DataType> positionTypes = {
            ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,
            ge::DT_INT64};
        const std::initializer_list<ge::DataType> slotTypes = {
            ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
            ge::DT_INT32};
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND};
        this->Input("paged_ctkv_0").ParamType(REQUIRED).DataType(cacheTypes).Format(formats).UnknownShapeFormat(formats).AutoContiguous();
        this->Input("paged_kpe_0").ParamType(REQUIRED).DataType(cacheTypes).Format(formats).UnknownShapeFormat(formats).AutoContiguous();
        this->Input("paged_ctkv_1").ParamType(REQUIRED).DataType(cacheTypes).Format(formats).UnknownShapeFormat(formats).AutoContiguous();
        this->Input("paged_kpe_1").ParamType(REQUIRED).DataType(cacheTypes).Format(formats).UnknownShapeFormat(formats).AutoContiguous();
        this->Input("paged_ctkv_2").ParamType(REQUIRED).DataType(cacheTypes).Format(formats).UnknownShapeFormat(formats).AutoContiguous();
        this->Input("paged_kpe_2").ParamType(REQUIRED).DataType(cacheTypes).Format(formats).UnknownShapeFormat(formats).AutoContiguous();
        this->Input("block_table").ParamType(REQUIRED).DataType(tableAndTopkTypes).Format(formats).UnknownShapeFormat(formats).AutoContiguous();
        this->Input("topk_indices").ParamType(REQUIRED).DataType(tableAndTopkTypes).Format(formats).UnknownShapeFormat(formats).AutoContiguous();
        this->Input("cur_pos").ParamType(REQUIRED).DataType(positionTypes).Format(formats).UnknownShapeFormat(formats).AutoContiguous();
        this->Output("out_ctkv_0").ParamType(REQUIRED).DataType(cacheTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("out_kpe_0").ParamType(REQUIRED).DataType(cacheTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("out_ctkv_1").ParamType(REQUIRED).DataType(cacheTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("out_kpe_1").ParamType(REQUIRED).DataType(cacheTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("out_ctkv_2").ParamType(REQUIRED).DataType(cacheTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("out_kpe_2").ParamType(REQUIRED).DataType(cacheTypes).Format(formats).UnknownShapeFormat(formats);
        this->Output("current_topk_slots").ParamType(REQUIRED).DataType(slotTypes).Format(formats).UnknownShapeFormat(formats);
        this->Attr("block_size").AttrType(OPTIONAL).Int(128);
        this->Attr("num_cache_layers").AttrType(OPTIONAL).Int(3);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(true).DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false).PrecisionReduceFlag(false);
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(SparseKvGatherGroup);
}  // namespace ops
