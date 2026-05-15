/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_gating_top_k_hash_tiling.cpp
 * \brief
 */
#include "moe_gating_top_k_hash_tiling.h"
#include "moe_gating_top_k_hash_tiling_arch35.h"
#include <graph/utils/type_utils.h>

namespace optiling {
const static int64_t GROUP_SELECT_MODE_MAX = 0;
const static int64_t GROUP_SELECT_MODE_SUM = 1;
const static int64_t RENORM_NO = 0;
const static int64_t RENORM_L1 = 1;
const static int64_t NORM_TYPE_SOFTMAX = 0;
const static int64_t NORM_TYPE_SIGMOID = 1;
const static int64_t NORM_TYPE_SOFTPLUS = 2;
const static int64_t OUT_FLAG_FALSE = 0;
const static int64_t OUT_FLAG_TRUE = 1;
const static size_t X_INPUT_DIMS = 2;
const static size_t BIAS_INPUT_DIMS = 1;
const static size_t Y_OUTPUT_DIMS = 2;
const static size_t EXPERT_IDX_OUTPUY_DIMS = 2;
const static size_t OUT_OUTPUT_DIMS = 2;
const static int64_t MAX_EXPERT_COUNT = 2048;

const static int64_t X_INPUT_INDEX = 0;
const static int64_t BIAS_INPUT_INDEX = 1;
const static int64_t INPUT_IDS_INPUT_INDEX = 2;
const static int64_t TID_TO_EID_INPUT_INDEX = 3;
const static int64_t Y_OUTPUT_INDEX = 0;
const static int64_t EXPERT_IDX_OUTPUT_INDEX = 1;
const static int64_t OUT_OUTPUT_INDEX = 2;
const static int64_t K_ATTR_INDEX = 0;
const static int64_t K_GROUP_ATTR_INDEX = 1;
const static int64_t GROUP_COUNT_ATTR_INDEX = 2;
const static int64_t GROUP_SELECT_MODE_ATTR_INDEX = 3;
const static int64_t RENORM_ATTR_INDEX = 4;
const static int64_t NORM_TYPE_ATTR_INDEX = 5;
const static int64_t OUT_FLAG_ATTR_INDEX = 6;
const static int64_t ROUTED_SCALING_FACTOR_ATTR_INDEX = 7;
const static int64_t EPS_ATTR_INDEX = 8;
const static int64_t DEFAULT_WORKSPACE_SIZE = 16777216; // 预留16M空间
const static uint32_t DATATYPESIZE_FLOAT = 4;
const static bool IS_LARGEST = true;
const static bool IS_INITINDEX = false;
const static bool IS_REUSESOURCE = false;
const static uint64_t WITH_GROUP_CONDITION = 1;
const static uint64_t WITHOUT_GROUP_CONDITION = 2;
const static uint64_t MAX_IN_GROUP_CONDITION = 3;
constexpr int32_t ROW_COUNT_PER_TASK = 1;

const static uint64_t TILING_KEY_EXPERTNUM_GROUPNUM_ALIGN_HIGH_PERF = 0;
const static uint64_t TILING_KEY_WITHOUT_GROUP = 1;
const static uint64_t TILING_KEY_GENERALIZED = 2;
const static uint64_t TILING_KEY_WITHOUT_GROUP_0 = 3;
const static uint64_t TILING_KEY_WITHOUT_GROUP_1 = 4;
const static uint64_t TILING_KEY_WITHOUT_GROUP_2 = 5;
const static uint64_t TILING_KEY_WITHOUT_GROUP_3 = 6;

template <typename T>
static inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd)));
}


template <typename T>
static inline T CeilAlign(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd)) * (rnd));
}

inline static int64_t CeilLog4(int64_t x)
{
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(4))); // 4 for four
}

class MoeGatingTopKHashTilingBase {
public:
    explicit MoeGatingTopKHashTilingBase(gert::TilingContext *context) : context_(context)
    {
        Reset();
    }
    ~MoeGatingTopKHashTilingBase()  = default;

    bool IsCapable()
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() ;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() ;
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() ;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() ;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const ;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() ;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() ;
    void Reset();

private:
    ge::graphStatus CheckInputShape();
    ge::graphStatus CheckAttr();
    ge::graphStatus CheckOutShape();
    void SplitRows();
    void CalTmpBufUbSize();

    const gert::Shape *xShape_ = nullptr;
    const gert::Shape *biasShape_ = nullptr;
    const gert::Shape *inputIdsShape_ = nullptr;
    const gert::Shape *tid2eidShape_ = nullptr;
    const gert::Shape *yShape_ = nullptr;
    const gert::Shape *expertIdxShape_ = nullptr;
    const gert::Shape *outShape_ = nullptr;
    ge::DataType inputIdsDtype;
    ge::DataType tid2eidDtype;

    uint64_t coreNum_ = 0;
    int64_t rows_ = 0;
    int64_t expertCount_ = 0;
    int64_t addBias_ = 0;

    int64_t k_ = 0;
    int64_t kGroup_ = 0;
    int64_t groupCount_ = 0;
    int64_t perGroupExpertCount_ = 0;
    int64_t groupSelectMode_ = GROUP_SELECT_MODE_MAX;
    int64_t renorm_ = RENORM_NO;
    int64_t normType_ = NORM_TYPE_SOFTMAX;
    int64_t outFlag_ = OUT_FLAG_FALSE;
    int64_t hashFlag_ = 0;
    float routedScalingFactor_ = 1.0;
    float eps_ = 1e-20f;

    int64_t inputDtypeSize_;
    const char *opName_ = "";
    MoeGatingTopKHashTilingData moeGatingTopKTilingData_;
    gert::TilingContext *context_ = nullptr;
    uint64_t workspaceSize_ = 0;
};

ge::graphStatus MoeGatingTopKHashTilingBase::CheckInputShape()
{
    size_t xDimNum = xShape_->GetDimNum();
    OPS_ERR_IF(xDimNum != X_INPUT_DIMS,
                OPS_LOG_E(context_, "The dim number of x is: %zu, but should be %zu.", xDimNum, X_INPUT_DIMS),
                return ge::GRAPH_FAILED);

    // 通过输入获取rows 和 expertCount
    rows_ = xShape_->GetDim(0);
    expertCount_ = xShape_->GetDim(1);
    moeGatingTopKTilingData_.set_rowCount(rows_);
    moeGatingTopKTilingData_.set_expertCount(expertCount_);
    if (biasShape_ != nullptr) {
        addBias_ = 1;
        size_t biasDimNum = biasShape_->GetDimNum();
        OPS_ERR_IF(biasDimNum != BIAS_INPUT_DIMS,
                    OPS_LOG_E(context_, "The dim number of bias is: %zu, but should be %zu.", biasDimNum, BIAS_INPUT_DIMS),
                    return ge::GRAPH_FAILED);
        OPS_ERR_IF(
            biasShape_->GetDim(0) != expertCount_,
            OPS_LOG_E(context_, "The first dim of bias is: %ld, but should be %ld.", biasShape_->GetDim(0), expertCount_),
            return ge::GRAPH_FAILED);
    }
    moeGatingTopKTilingData_.set_addBias(addBias_);

    if (inputIdsShape_ != nullptr) {
        OPS_ERR_IF(
            tid2eidShape_ == nullptr,
            OPS_LOG_E(context_, "The tid2eid should not be empty when inputIds has value."),
            return ge::GRAPH_FAILED);
    }
    if (tid2eidShape_ != nullptr) {
        OPS_ERR_IF(
            inputIdsShape_ == nullptr,
            OPS_LOG_E(context_, "The inputIds should not be empty when tid2eid has value."),
            return ge::GRAPH_FAILED);
    }
    if (inputIdsShape_ != nullptr && tid2eidShape_ != nullptr) {
      hashFlag_ = 1;
      OPS_LOG_I(context_, "hashFlag_ is 1.");
    }
    moeGatingTopKTilingData_.set_hashFlag(hashFlag_);

    OPS_ERR_IF(k_ > expertCount_,
                OPS_LOG_E(context_, "k is: %ld, expert num is: %ld, k cannot be greater than expert num.", k_, expertCount_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKHashTilingBase::CheckAttr()
{
    OPS_ERR_IF(
        expertCount_ > MAX_EXPERT_COUNT,
        OPS_LOG_E(context_, "expert count is: %ld, but should not greater than %ld.", expertCount_, MAX_EXPERT_COUNT),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(k_ <= 0, OPS_LOG_E(context_, "k is: %ld, but should be greater than 0.", k_), return ge::GRAPH_FAILED);

    OPS_ERR_IF(kGroup_ <= 0, OPS_LOG_E(context_, "k_group is: %ld, but should be greater than 0.", kGroup_),
                return ge::GRAPH_FAILED);

    OPS_ERR_IF(kGroup_ > groupCount_,
                OPS_LOG_E(context_, "k_group is: %ld, but should not greater than %ld.", kGroup_, groupCount_),
                return ge::GRAPH_FAILED);

    OPS_ERR_IF(groupCount_ <= 0, OPS_LOG_E(context_, "group_count is: %ld, but should be greater than 0.", groupCount_),
                return ge::GRAPH_FAILED);

    OPS_ERR_IF(normType_ != NORM_TYPE_SOFTMAX && normType_ != NORM_TYPE_SIGMOID && normType_ != NORM_TYPE_SOFTPLUS,
                OPS_LOG_E(context_, "norm type is: %ld, but currently only support %ld, %ld and %ld.", normType_,
                     NORM_TYPE_SOFTMAX, NORM_TYPE_SIGMOID, NORM_TYPE_SOFTPLUS),
                return ge::GRAPH_FAILED);

    OPS_ERR_IF(normType_ == NORM_TYPE_SOFTPLUS && groupCount_ != 1,
                OPS_LOG_E(context_, "norm type softplus only supported when groupCount equals 1, but got %ld.", groupCount_),
                return ge::GRAPH_FAILED);

    OPS_ERR_IF(groupSelectMode_ != GROUP_SELECT_MODE_SUM && groupSelectMode_ != GROUP_SELECT_MODE_MAX,
                OPS_LOG_E(context_, "group select mode is: %ld, but currently only support %ld and %ld.", groupSelectMode_,
                     GROUP_SELECT_MODE_SUM, GROUP_SELECT_MODE_MAX),
                return ge::GRAPH_FAILED);

    OPS_ERR_IF(renorm_ != RENORM_NO,
                OPS_LOG_E(context_, "renorm is: %ld, but currently only support %ld.", renorm_, RENORM_NO),
                return ge::GRAPH_FAILED);

    OPS_ERR_IF(expertCount_ % groupCount_ != 0,
                OPS_LOG_E(context_, "Expert count : %ld is not divisible by k_group: %ld", expertCount_, groupCount_),
                return ge::GRAPH_FAILED);
    perGroupExpertCount_ = expertCount_ / groupCount_;

    OPS_ERR_IF(perGroupExpertCount_ < 1,
                OPS_LOG_E(context_, "group expert count is: %ld, but should be greater than 1.", perGroupExpertCount_),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(
        groupSelectMode_ == GROUP_SELECT_MODE_SUM && perGroupExpertCount_ < 2,
        OPS_LOG_E(context_,
             "group expert count is: %ld, if group select mode is: %ld, group expert count should be greater than 1.",
             perGroupExpertCount_, groupSelectMode_),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(k_ > kGroup_ * perGroupExpertCount_,
                OPS_LOG_E(context_, "k is: %ld, but should be smaller than %ld.", k_, kGroup_ * perGroupExpertCount_),
                return ge::GRAPH_FAILED);
    int64_t groupExpertCountAlign = CeilAlign(perGroupExpertCount_, 32L);
    if (groupCount_ != 1 && groupCount_ != expertCount_ && kGroup_ != groupCount_) {
        // 分组场景下才需要校验对齐后的数量
        OPS_ERR_IF(groupCount_ * groupExpertCountAlign > MAX_EXPERT_COUNT,
                    OPS_LOG_E(context_, "group count * group expert count align is: %ld, but should not greater than %ld.",
                         groupCount_ * groupExpertCountAlign, MAX_EXPERT_COUNT),
                    return ge::GRAPH_FAILED);
    }

    moeGatingTopKTilingData_.set_perGroupExpertCount(perGroupExpertCount_);
    moeGatingTopKTilingData_.set_perGroupExpertCountAlign(groupExpertCountAlign);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKHashTilingBase::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    // 获取输入shape信息
    auto xShapePtr = context_->GetInputShape(X_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context_, xShapePtr, return ge::GRAPH_FAILED);
    xShape_ = &xShapePtr->GetStorageShape();

    auto biasShapePtr = context_->GetOptionalInputShape(BIAS_INPUT_INDEX);
    biasShape_ = biasShapePtr == nullptr ? nullptr : &biasShapePtr->GetStorageShape();

    auto inputIdsShapePtr = context_->GetOptionalInputShape(INPUT_IDS_INPUT_INDEX);
    inputIdsShape_ = inputIdsShapePtr == nullptr ? nullptr : &inputIdsShapePtr->GetStorageShape();

    auto tid2eidShapePtr = context_->GetOptionalInputShape(TID_TO_EID_INPUT_INDEX);
    tid2eidShape_ = tid2eidShapePtr == nullptr ? nullptr : &tid2eidShapePtr->GetStorageShape();

    // 获取输出shape
    auto yShapePtr = context_->GetOutputShape(Y_OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL(context_, yShapePtr, return ge::GRAPH_FAILED);
    yShape_ = &yShapePtr->GetStorageShape();
    auto expertIdxPtr = context_->GetOutputShape(EXPERT_IDX_OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL(context_, expertIdxPtr, return ge::GRAPH_FAILED);
    expertIdxShape_ = &expertIdxPtr->GetStorageShape();
    auto outPtr = context_->GetOutputShape(OUT_OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL(context_, outPtr, return ge::GRAPH_FAILED);
    outShape_ = &outPtr->GetStorageShape();

    auto x = context_->GetInputDesc(X_INPUT_INDEX);
    OPS_LOG_E_IF_NULL(context_, x, return ge::GRAPH_FAILED);
    auto xDtype = x->GetDataType();
    OPS_ERR_IF(
        (xDtype != ge::DataType::DT_FLOAT && xDtype != ge::DataType::DT_FLOAT16 && xDtype != ge::DataType::DT_BF16),
        OPS_LOG_E(context_, "x dtype %s error, only supports float32, half, bf16. please check.",
             ge::TypeUtils::DataTypeToSerialString(xDtype).c_str()),
        return ge::GRAPH_FAILED);

    if (biasShapePtr != nullptr) {
        auto biasDtype = context_->GetOptionalInputDesc(BIAS_INPUT_INDEX)->GetDataType();
        OPS_ERR_IF((biasDtype != xDtype),
                    OPS_LOG_E(context_, "bias dtype %s not equal x dtype %s, please check.",
                         ge::TypeUtils::DataTypeToSerialString(biasDtype).c_str(),
                         ge::TypeUtils::DataTypeToSerialString(xDtype).c_str()),
                    return ge::GRAPH_FAILED);
    }
    if (inputIdsShapePtr != nullptr) {
        inputIdsDtype = context_->GetOptionalInputDesc(INPUT_IDS_INPUT_INDEX)->GetDataType();
        OPS_ERR_IF((inputIdsDtype != ge::DataType::DT_INT32 && inputIdsDtype != ge::DataType::DT_INT64),
                    OPS_LOG_E(context_, "inputIds dtype %s error, only supports int32 and int64. please check.",
                      ge::TypeUtils::DataTypeToSerialString(inputIdsDtype).c_str()),
                return ge::GRAPH_FAILED);
    }
    if (tid2eidShapePtr != nullptr) {
        tid2eidDtype = context_->GetOptionalInputDesc(TID_TO_EID_INPUT_INDEX)->GetDataType();
        OPS_ERR_IF((tid2eidDtype != ge::DataType::DT_INT32 && tid2eidDtype != ge::DataType::DT_INT64),
                    OPS_LOG_E(context_, "tid2eid dtype %s error, only supports int32 and int64. please check.",
                      ge::TypeUtils::DataTypeToSerialString(tid2eidDtype).c_str()),
                return ge::GRAPH_FAILED);
    }

    auto yDesc = context_->GetOutputDesc(Y_OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL(context_, yDesc, return ge::GRAPH_FAILED);
    auto yDtype = yDesc->GetDataType();
    OPS_ERR_IF((yDtype != xDtype),
                OPS_LOG_E(context_, "y out dtype %s must be the same with x dtype %s.",
                     ge::TypeUtils::DataTypeToSerialString(yDtype).c_str(),
                     ge::TypeUtils::DataTypeToSerialString(xDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto expertIdDesc = context_->GetOutputDesc(EXPERT_IDX_OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL(context_, expertIdDesc, return ge::GRAPH_FAILED);
    auto expertIdDtype = expertIdDesc->GetDataType();
    OPS_ERR_IF((expertIdDtype != ge::DataType::DT_INT32),
                OPS_LOG_E(context_, "expertId out dtype %s error, only supports int32. please check.",
                     ge::TypeUtils::DataTypeToSerialString(expertIdDtype).c_str()),
                return ge::GRAPH_FAILED);

    auto normOutDesc = context_->GetOutputDesc(OUT_OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL(context_, normOutDesc, return ge::GRAPH_FAILED);
    auto normOutDtype = normOutDesc->GetDataType();
    OPS_ERR_IF((normOutDtype != ge::DataType::DT_FLOAT),
                OPS_LOG_E(context_, "norm out dtype %s error, only supports float. please check.",
                     ge::TypeUtils::DataTypeToSerialString(normOutDtype).c_str()),
                return ge::GRAPH_FAILED);

    // 获取属性
    auto attrs = context_->GetAttrs();
    OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED);

    const int64_t *kPtr = attrs->GetAttrPointer<int64_t>(K_ATTR_INDEX);
    OPS_LOG_E_IF_NULL(context_, kPtr, return ge::GRAPH_FAILED);
    k_ = *kPtr;
    moeGatingTopKTilingData_.set_k(k_);
    OPS_LOG_I(context_, "Attr k is: %ld ", k_);

    const int64_t *kGroupPtr = attrs->GetAttrPointer<int64_t>(K_GROUP_ATTR_INDEX);
    if (kGroupPtr != nullptr) {
        kGroup_ = *kGroupPtr;
        moeGatingTopKTilingData_.set_kGroup(kGroup_);
    }
    OPS_LOG_I(context_, "Attr k_group is: %ld ", kGroup_);

    const int64_t *groupCountPtr = attrs->GetAttrPointer<int64_t>(GROUP_COUNT_ATTR_INDEX);
    if (groupCountPtr != nullptr) {
        groupCount_ = *groupCountPtr;
        moeGatingTopKTilingData_.set_groupCount(groupCount_);
    }
    OPS_LOG_I(context_, "Attr group_count is: %ld ", groupCount_);

    const int64_t *groupSelectModePtr = attrs->GetAttrPointer<int64_t>(GROUP_SELECT_MODE_ATTR_INDEX);
    if (groupSelectModePtr != nullptr) {
        groupSelectMode_ = *groupSelectModePtr;
        moeGatingTopKTilingData_.set_groupSelectMode(groupSelectMode_);
    }
    OPS_LOG_I(context_, "Attr group_select_mode is: %ld ", groupSelectMode_);

    const int64_t *renormPtr = attrs->GetAttrPointer<int64_t>(RENORM_ATTR_INDEX);
    if (renormPtr != nullptr) {
        renorm_ = *renormPtr;
        moeGatingTopKTilingData_.set_renorm(renorm_);
    }
    OPS_LOG_I(context_, "Attr renorm is: %ld ", renorm_);

    const int64_t *normTypePtr = attrs->GetAttrPointer<int64_t>(NORM_TYPE_ATTR_INDEX);
    if (normTypePtr != nullptr) {
        normType_ = *normTypePtr;
        moeGatingTopKTilingData_.set_normType(normType_);
    }
    OPS_LOG_I(context_, "Attr norm_type is: %ld ", normType_);

    const bool *outFlagPtr = attrs->GetAttrPointer<bool>(OUT_FLAG_ATTR_INDEX);
    if (outFlagPtr != nullptr) {
        outFlag_ = (*outFlagPtr) ? 1 : 0;
        moeGatingTopKTilingData_.set_outFlag(outFlag_);
    }
    OPS_LOG_I(context_, "Attr out_flag is: %ld ", outFlag_);

    const float *routedScalingFactorPtr = attrs->GetAttrPointer<float>(ROUTED_SCALING_FACTOR_ATTR_INDEX);
    if (routedScalingFactorPtr != nullptr) {
        routedScalingFactor_ = *routedScalingFactorPtr;
        moeGatingTopKTilingData_.set_routedScalingFactor(routedScalingFactor_);
    }
    OPS_LOG_I(context_, "Attr routed_scaling_factor is: %f ", routedScalingFactor_);

    const float *epsPtr = attrs->GetAttrPointer<float>(EPS_ATTR_INDEX);
    if (epsPtr != nullptr) {
        eps_ = *epsPtr;
        moeGatingTopKTilingData_.set_eps(eps_);
    }
    OPS_LOG_I(context_, "Attr eps is: %f ", eps_);

    inputDtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(context_->GetInputDesc(0)->GetDataType()));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKHashTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_LOG_E(context_, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKHashTilingBase::CheckOutShape()
{
    OPS_ERR_IF((yShape_->GetDimNum() != xShape_->GetDimNum()),
                OPS_LOG_E(context_, "y out shape num %zu and x shape num %zu not equal, please check.", yShape_->GetDimNum(),
                     xShape_->GetDimNum()),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF((expertIdxShape_->GetDimNum() != xShape_->GetDimNum()),
                OPS_LOG_E(context_, "expertId out shape num %zu and x shape num %zu not equal, please check.",
                     expertIdxShape_->GetDimNum(), xShape_->GetDimNum()),
                return ge::GRAPH_FAILED);
    if (outShape_ != nullptr) {
        OPS_ERR_IF((outShape_->GetDimNum() != xShape_->GetDimNum()),
                    OPS_LOG_E(context_, "norm out shape num %zu and x shape num %zu not equal, please check.",
                         outShape_->GetDimNum(), xShape_->GetDimNum()),
                    return ge::GRAPH_FAILED);
    }

    OPS_ERR_IF((yShape_->GetDim(0) != xShape_->GetDim(0)),
                OPS_LOG_E(context_, "y out dim[0] %ld not euqal x dim[0] %ld, please check.", yShape_->GetDim(0),
                     xShape_->GetDim(0)),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF((expertIdxShape_->GetDim(0) != xShape_->GetDim(0)),
                OPS_LOG_E(context_, "expertId out dim[0] %ld not euqal x dim[0] %ld, please check.",
                     expertIdxShape_->GetDim(0), xShape_->GetDim(0)),
                return ge::GRAPH_FAILED);
    if (outFlag_ && outShape_ != nullptr) {
        OPS_ERR_IF((outShape_->GetDim(0) != xShape_->GetDim(0)),
                    OPS_LOG_E(context_, "norm out dim[0] %ld and x dim[0] %ld not equal, please check.",
                         outShape_->GetDim(0), outShape_->GetDim(0)),
                    return ge::GRAPH_FAILED);
    }

    OPS_ERR_IF((yShape_->GetDim(1) != k_),
                OPS_LOG_E(context_, "y dim[1] %ld not euqal k %ld, please check.", yShape_->GetDim(1), k_),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF((expertIdxShape_->GetDim(1) != k_),
                OPS_LOG_E(context_, "expertId dim[1] %ld not euqal k %ld, please check.", expertIdxShape_->GetDim(1), k_),
                return ge::GRAPH_FAILED);
    if (outFlag_ && outShape_ != nullptr) {
        OPS_ERR_IF((outShape_->GetDim(1) != xShape_->GetDim(1)),
                    OPS_LOG_E(context_, "normOut dim[1] %ld and x dim[1] %ld not equal, please check.", outShape_->GetDim(1),
                         xShape_->GetDim(1)),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void MoeGatingTopKHashTilingBase::SplitRows()
{
    int64_t perCoreRows = CeilDiv(rows_, static_cast<int64_t>(coreNum_));
    int64_t needCoreNum = CeilDiv(rows_, perCoreRows);
    // perCoreRows cannot be 0
    int64_t lastCoreRows = rows_ % perCoreRows == 0 ? perCoreRows : rows_ % perCoreRows;
    moeGatingTopKTilingData_.set_needCoreNum(needCoreNum);
    moeGatingTopKTilingData_.set_perCoreRowCount(perCoreRows);
    moeGatingTopKTilingData_.set_lastCoreRowCount(lastCoreRows);
    int64_t vmsCount = CeilLog4(CeilDiv(kGroup_, 4L));
    OPS_LOG_I(context_, "vms count is: %ld", vmsCount);
    moeGatingTopKTilingData_.set_vmsCount(vmsCount); // 需要归并的轮数
}

void MoeGatingTopKHashTilingBase::CalTmpBufUbSize()
{
    std::vector<int64_t> shape_vec = {expertCount_};
    ge::Shape shape(shape_vec);
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSigmoidMaxMinTmpSize(shape, sizeof(float), false, maxValue, minValue);

    int64_t indexTmpBuf = (expertCount_ + 31) / 32 * 32 * static_cast<int64_t>(sizeof(float));
    moeGatingTopKTilingData_.set_calTmpBufUbSize(std::max(indexTmpBuf, static_cast<int64_t>(minValue)));
}

ge::graphStatus MoeGatingTopKHashTilingBase::DoOpTiling()
{
    auto ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = GetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckInputShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckOutShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckAttr();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    CalTmpBufUbSize();
    SplitRows();

    ret = PostTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKHashTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKHashTilingBase::GetWorkspaceSize()
{
    // 计算workspace大小
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKHashTilingBase::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(moeGatingTopKTilingData_.get_needCoreNum());
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    moeGatingTopKTilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                          context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(moeGatingTopKTilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeGatingTopKHashTilingBase::GetTilingKey() const
{
    // DeepSeekV3排序对齐高性能场景
    if (expertCount_ == 256 && groupCount_ == 8 && kGroup_ == 4 && k_ <= 32 && addBias_ &&
        groupSelectMode_ == GROUP_SELECT_MODE_SUM && renorm_ == RENORM_NO && normType_ == NORM_TYPE_SIGMOID &&
        !outFlag_) {
        // DeepSeekV3排序对齐高性能场景
        return TILING_KEY_EXPERTNUM_GROUPNUM_ALIGN_HIGH_PERF;
    } else if (groupCount_ == 1 || groupCount_ == expertCount_ || kGroup_ == groupCount_) {
        /**
         * 不分组场景：
         * 1. 分组数为 1
         * 2. 分组数等于专家数（每个组只有一个专家）
         * 3. 选择所有组
         */
        if (inputIdsShape_ == nullptr) {
          return TILING_KEY_WITHOUT_GROUP;
        } else if (inputIdsDtype == ge::DataType::DT_INT32 && tid2eidDtype == ge::DataType::DT_INT64) {
          return TILING_KEY_WITHOUT_GROUP_0;
        } else if (inputIdsDtype == ge::DataType::DT_INT32 && tid2eidDtype == ge::DataType::DT_INT32) {
          return TILING_KEY_WITHOUT_GROUP_1;
        } else if (inputIdsDtype == ge::DataType::DT_INT64 && tid2eidDtype == ge::DataType::DT_INT64) {
          return TILING_KEY_WITHOUT_GROUP_2;
        } else if (inputIdsDtype == ge::DataType::DT_INT64 && tid2eidDtype == ge::DataType::DT_INT32) {
          return TILING_KEY_WITHOUT_GROUP_3;
        }
    } else {
        return TILING_KEY_GENERALIZED;
    }
}

void MoeGatingTopKHashTilingBase::Reset()
{
    opName_ = nullptr;
    return;
}

ge::graphStatus TilingForMoeGatingTopKHash(gert::TilingContext *context)
{
    OPS_LOG_I(context, "TilingForMoeGatingTopKHash start");
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("TilingForMoeGatingTopKHash", "Tiling context is null"),
               return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_REPORT_VECTOR_INNER_ERR("TilingForMoeGatingTopKHash", "Tiling platformInfo is null"),
               return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion == platform_ascendc::SocVersion::ASCEND910_95) {
        OPS_LOG_I(context, "Using arch35 tiling for ASCEND910_95");
          MoeGatingTopKHashRegBase::MoeGatingTopKHashTilingRegbase moeGatingTopKTilingRegbase(context);
          return moeGatingTopKTilingRegbase.DoOpTiling();
    }

    MoeGatingTopKHashTilingBase moeGatingTopKTilingBase(context);
    return moeGatingTopKTilingBase.DoOpTiling();
}

static ge::graphStatus TilingPrepareForMoeGatingTopKHash(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeGatingTopKHash)
    .Tiling(TilingForMoeGatingTopKHash)
    .TilingParse<MoeGatingTopKHashCompileInfo>(TilingPrepareForMoeGatingTopKHash);
} // namespace optiling