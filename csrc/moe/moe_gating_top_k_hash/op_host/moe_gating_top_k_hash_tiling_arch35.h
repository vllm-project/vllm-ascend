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
 * \file moe_gating_top_k_hash_tiling_arch35.cpp
 * \brief
 */
#include "moe_gating_top_k_hash_tiling.h"
#include <graph/utils/type_utils.h>

namespace optiling {
namespace MoeGatingTopKHashRegBase {

const static uint64_t MOE_GATING_TOP_K_REGBASE_TILING_KEY = 10000;
const static uint64_t MOE_GATING_TOP_K_REGBASE_TILING_KEY_1 = 10001;
const static uint64_t MOE_GATING_TOP_K_REGBASE_TILING_KEY_2 = 10002;
const static uint64_t MOE_GATING_TOP_K_REGBASE_TILING_KEY_3 = 10003;
const static uint64_t MOE_GATING_TOP_K_REGBASE_TILING_KEY_4 = 10004;

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
const static int64_t MRGSORT_SIZE = 4;
const static int64_t NORM_TYPE_ATTR_INDEX = 5;
const static int64_t OUT_FLAG_ATTR_INDEX = 6;
const static int64_t ROUTED_SCALING_FACTOR_ATTR_INDEX = 7;
const static int64_t EPS_ATTR_INDEX = 8;
const static int64_t DEFAULT_WORKSPACE_SIZE = static_cast<int64_t>(16 * 1024 * 1024); // 预留16M空间

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

class MoeGatingTopKHashTilingRegbase {
public:
    explicit MoeGatingTopKHashTilingRegbase(gert::TilingContext *context) : context_(context)
    {
        Reset();
    }
    ~MoeGatingTopKHashTilingRegbase() = default;

    bool IsCapable()
    {
        if (socVersion_ != platform_ascendc::SocVersion::ASCEND910_95) {
            return false;
        }
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo();
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo();
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling();
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling();
    // 5、计算TilingKey
    uint64_t GetTilingKey() const;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize();
    // 7、保存Tiling数据
    ge::graphStatus PostTiling();
    void Reset();

private:
    ge::graphStatus CheckInputShape();
    ge::graphStatus CheckAttr();
    ge::graphStatus CheckOutShape();
    void CalTmpBufUbSize();
    void SplitRows();
    void Tiling4GatherOutComputeSplitK();

    const gert::Shape *xShape_ = nullptr;
    const gert::Shape *biasShape_ = nullptr;
    const gert::Shape *yShape_ = nullptr;
    const gert::Shape *expertIdxShape_ = nullptr;
    const gert::Shape *outShape_ = nullptr;
    const gert::Shape *inputIdsShape_ = nullptr;
    const gert::Shape *tid2eidShape_ = nullptr;

    ge::DataType inputIdsDtype;
    ge::DataType tid2eidDtype;

    uint64_t coreNum_ = 0;
    int64_t rows_ = 0;
    int64_t expertCount_ = 0;
    int64_t addBias_ = 0;

    int64_t k_ = 0;
    int64_t kGroup_ = 1;
    int64_t groupCount_ = 1;
    int64_t groupSelectMode_ = GROUP_SELECT_MODE_MAX;
    int64_t renorm_ = RENORM_NO;
    int64_t normType_ = NORM_TYPE_SOFTMAX;
    int64_t outFlag_ = OUT_FLAG_FALSE;
    int64_t hashFlag_ = 0;
    float routedScalingFactor_ = 1.0;
    float eps_ = 1e-20f;

    int64_t inputDtypeSize_;
    const char *opName_ = "";
    MoeGatingTopKHashRegbaseTilingData moeGatingTopKTilingData_;
    gert::TilingContext *context_ = nullptr;
    uint64_t workspaceSize_ = 0;
    platform_ascendc::SocVersion socVersion_;
};

ge::graphStatus MoeGatingTopKHashTilingRegbase::CheckInputShape()
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
    OPS_ERR_IF(
        expertCount_ > MAX_EXPERT_COUNT,
        OPS_LOG_E(context_, "expert count is: %ld, but should not greater than %ld.", expertCount_, MAX_EXPERT_COUNT),
        return ge::GRAPH_FAILED);

    if (biasShape_ != nullptr) {
        addBias_ = 1;
        size_t biasDimNum = biasShape_->GetDimNum();
        OPS_ERR_IF(biasDimNum != BIAS_INPUT_DIMS,
                    OPS_LOG_E(context_, "The number of bias dim is: %zu, but should be %zu.", biasDimNum, BIAS_INPUT_DIMS),
                    return ge::GRAPH_FAILED);
        OPS_ERR_IF(biasShape_->GetDim(0) != expertCount_,
                    OPS_LOG_E(context_, "The first dim of bias is: %ld, but should be expert num: %ld.",
                         biasShape_->GetDim(0), expertCount_),
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

ge::graphStatus MoeGatingTopKHashTilingRegbase::CheckAttr()
{
    OPS_ERR_IF(k_ <= 0, OPS_LOG_E(context_, "k is: %ld, but should be greater than 0.", k_), return ge::GRAPH_FAILED);
    OPS_ERR_IF(kGroup_ <= 0, OPS_LOG_E(context_, "k_group is: %ld, but should be greater than 0.", kGroup_),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(groupCount_ <= 0, OPS_LOG_E(context_, "group_count is: %ld, but should be greater than 0.", groupCount_),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(expertCount_ % groupCount_ != 0,
                OPS_LOG_E(context_, "expert num : %ld is not divisible by group_count: %ld", expertCount_, groupCount_),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(kGroup_ > groupCount_,
                OPS_LOG_E(context_, "k_group is: %ld, but should not greater than group_count: %ld", kGroup_, groupCount_),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(groupCount_ == expertCount_ && kGroup_ < k_,
                OPS_LOG_E(context_, "k_group * group expert count is: %ld, but it must be greater than or equal to k: %ld.",
                     kGroup_, k_),
                return ge::GRAPH_FAILED);

    if (kGroup_ == groupCount_ || groupCount_ == expertCount_) {
        kGroup_ = 1;
        groupCount_ = 1;
    }
    moeGatingTopKTilingData_.set_kGroup(kGroup_);
    moeGatingTopKTilingData_.set_groupCount(groupCount_);
    int64_t groupExpertCount = expertCount_ / groupCount_;
    int64_t groupExpertCountAlign = CeilAlign(groupExpertCount, 32L);
    moeGatingTopKTilingData_.set_perGroupExpertCount(expertCount_ / groupCount_);
    moeGatingTopKTilingData_.set_perGroupExpertCountAlign(groupExpertCountAlign);

    OPS_ERR_IF(groupCount_ * groupExpertCountAlign > MAX_EXPERT_COUNT,
                OPS_LOG_E(context_, "group count * group expert count align is: %ld, but should not greater than %ld.",
                     groupCount_ * groupExpertCountAlign, MAX_EXPERT_COUNT),
                return ge::GRAPH_FAILED);

    OPS_ERR_IF(kGroup_ * groupExpertCount < k_,
                OPS_LOG_E(context_, "k_group * group expert count is: %ld, but it must be greater than or equal to k: %ld.",
                     kGroup_ * groupExpertCount, k_),
                return ge::GRAPH_FAILED);

    OPS_ERR_IF(groupExpertCount < 1,
                OPS_LOG_E(context_, "per group expert count is: %ld, but should be greater than 0.", groupExpertCount),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(
        groupSelectMode_ != GROUP_SELECT_MODE_SUM && groupSelectMode_ != GROUP_SELECT_MODE_MAX,
        OPS_LOG_E(context_, "group select mode is: %ld, but currently only support %ld and %ld.", groupSelectMode_,
            GROUP_SELECT_MODE_SUM, GROUP_SELECT_MODE_MAX),
            return ge::GRAPH_FAILED);
    OPS_ERR_IF(groupSelectMode_ == GROUP_SELECT_MODE_SUM && groupExpertCount < 2,
        OPS_LOG_E(context_,
             "group expert count is: %ld, if group select mode is: %ld, group expert count should be greater than 1.",
             groupExpertCount, groupSelectMode_),
        return ge::GRAPH_FAILED);

    OPS_ERR_IF(renorm_ != RENORM_NO,
                OPS_LOG_E(context_, "renorm is: %ld, but currently only support %ld.", renorm_, RENORM_NO),
                return ge::GRAPH_FAILED);

    OPS_ERR_IF(normType_ != NORM_TYPE_SOFTMAX && normType_ != NORM_TYPE_SIGMOID && normType_ != NORM_TYPE_SOFTPLUS,
                OPS_LOG_E(context_, "norm type is: %ld, but currently only support %ld, %ld and %ld.", normType_,
                     NORM_TYPE_SOFTMAX, NORM_TYPE_SIGMOID, NORM_TYPE_SOFTPLUS),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKHashTilingRegbase::GetShapeAttrsInfo()
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
    if (outPtr != nullptr) {
        outShape_ = &outPtr->GetStorageShape();
    }

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
    }
    OPS_LOG_I(context_, "Attr k_group is: %ld ", kGroup_);

    const int64_t *groupCountPtr = attrs->GetAttrPointer<int64_t>(GROUP_COUNT_ATTR_INDEX);
    if (groupCountPtr != nullptr) {
        groupCount_ = *groupCountPtr;
    }
    OPS_LOG_I(context_, "Attr group_count is: %ld ", groupCount_);

    const int64_t *groupSelectModePtr = attrs->GetAttrPointer<int64_t>(GROUP_SELECT_MODE_ATTR_INDEX);
    if (groupSelectModePtr != nullptr) {
        groupSelectMode_ = *groupSelectModePtr;
    }
    moeGatingTopKTilingData_.set_groupSelectMode(groupSelectMode_);
    OPS_LOG_I(context_, "Attr group_select_mode is: %ld ", groupSelectMode_);

    const int64_t *renormPtr = attrs->GetAttrPointer<int64_t>(RENORM_ATTR_INDEX);
    if (renormPtr != nullptr) {
        renorm_ = *renormPtr;
    }
    moeGatingTopKTilingData_.set_renorm(renorm_);
    OPS_LOG_I(context_, "Attr renorm is: %ld ", renorm_);

    const int64_t *normTypePtr = attrs->GetAttrPointer<int64_t>(NORM_TYPE_ATTR_INDEX);
    if (normTypePtr != nullptr) {
        normType_ = *normTypePtr;
    }
    moeGatingTopKTilingData_.set_normType(normType_);
    OPS_LOG_I(context_, "Attr norm_type is: %ld ", normType_);

    const bool *outFlagPtr = attrs->GetAttrPointer<bool>(OUT_FLAG_ATTR_INDEX);
    if (outFlagPtr != nullptr) {
        outFlag_ = (*outFlagPtr) ? 1 : 0;
    }
    moeGatingTopKTilingData_.set_outFlag(outFlag_);
    OPS_LOG_I(context_, "Attr out_flag is: %ld ", outFlag_);

    const float *routedScalingFactorPtr = attrs->GetAttrPointer<float>(ROUTED_SCALING_FACTOR_ATTR_INDEX);
    if (routedScalingFactorPtr != nullptr) {
        routedScalingFactor_ = *routedScalingFactorPtr;
    }
    moeGatingTopKTilingData_.set_routedScalingFactor(routedScalingFactor_);
    OPS_LOG_I(context_, "Attr routed_scaling_factor is: %f ", routedScalingFactor_);

    const float *epsPtr = attrs->GetAttrPointer<float>(EPS_ATTR_INDEX);
    if (epsPtr != nullptr) {
        eps_ = *epsPtr;
    }
    moeGatingTopKTilingData_.set_eps(eps_);
    OPS_LOG_I(context_, "Attr eps is: %f ", eps_);

    auto outDesc = context_->GetOutputDesc(OUT_OUTPUT_INDEX);
    if (outFlag_ && outDesc != nullptr) {
        auto outDtype = outDesc->GetDataType();
        OPS_ERR_IF((outDtype != ge::DataType::DT_FLOAT),
                    OPS_LOG_E(context_, "norm out dtype %s error, only supports float32. please check.",
                         ge::TypeUtils::DataTypeToSerialString(outDtype).c_str()),
                    return ge::GRAPH_FAILED);
    }

    inputDtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(context_->GetInputDesc(0)->GetDataType()));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKHashTilingRegbase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_LOG_E(context_, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    socVersion_ = ascendcPlatform.GetSocVersion();
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKHashTilingRegbase::CheckOutShape()
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

void MoeGatingTopKHashTilingRegbase::CalTmpBufUbSize()
{
    std::vector<int64_t> shape_vec = {groupCount_ * moeGatingTopKTilingData_.get_perGroupExpertCountAlign()};
    ge::Shape softmaxShape(shape_vec);

    uint32_t softmaxTmpSize = AscendC::GetSoftMaxMaxTmpSize(softmaxShape, sizeof(float), true);
    AscendC::SoftMaxTilingFunc(softmaxShape, sizeof(float), softmaxTmpSize, moeGatingTopKTilingData_.softmaxTilingData);
}

void MoeGatingTopKHashTilingRegbase::SplitRows()
{
    int64_t perCoreRows = CeilDiv(rows_, static_cast<int64_t>(coreNum_));
    int64_t needCoreNum = CeilDiv(rows_, perCoreRows);
    if (perCoreRows == 0) {
        OPS_LOG_E(context_, "perCoreRows can't be 0.");
        return;
    }
    int64_t lastCoreRows = rows_ % perCoreRows == 0 ? perCoreRows : rows_ % perCoreRows;
    moeGatingTopKTilingData_.set_needCoreNum(needCoreNum);
    moeGatingTopKTilingData_.set_perCoreRowCount(perCoreRows);
    moeGatingTopKTilingData_.set_lastCoreRowCount(lastCoreRows);

    int64_t vmsCount = 0;
    if (kGroup_ > MRGSORT_SIZE) {
        int64_t index = MRGSORT_SIZE;
        while (index < kGroup_) {
            index = index * MRGSORT_SIZE;
            vmsCount++;
        }
    }
    moeGatingTopKTilingData_.set_vmsCount(vmsCount);
}

ge::graphStatus MoeGatingTopKHashTilingRegbase::DoOpTiling()
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

    ret = CheckAttr();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckOutShape();
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

ge::graphStatus MoeGatingTopKHashTilingRegbase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKHashTilingRegbase::GetWorkspaceSize()
{
    // 计算workspace大小
    workspaceSize_ = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeGatingTopKHashTilingRegbase::PostTiling()
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

uint64_t MoeGatingTopKHashTilingRegbase::GetTilingKey() const
{
    if (inputIdsShape_ == nullptr) {
      return MOE_GATING_TOP_K_REGBASE_TILING_KEY;
    } else if (inputIdsDtype == ge::DataType::DT_INT32 && tid2eidDtype == ge::DataType::DT_INT64) {
      return MOE_GATING_TOP_K_REGBASE_TILING_KEY_1;
    } else if (inputIdsDtype == ge::DataType::DT_INT32 && tid2eidDtype == ge::DataType::DT_INT32) {
      return MOE_GATING_TOP_K_REGBASE_TILING_KEY_2;
    } else if (inputIdsDtype == ge::DataType::DT_INT64 && tid2eidDtype == ge::DataType::DT_INT64) {
      return MOE_GATING_TOP_K_REGBASE_TILING_KEY_3;
    } else if (inputIdsDtype == ge::DataType::DT_INT64 && tid2eidDtype == ge::DataType::DT_INT32) {
      return MOE_GATING_TOP_K_REGBASE_TILING_KEY_4;
    }
}

void MoeGatingTopKHashTilingRegbase::Reset()
{
    opName_ = nullptr;
    return;
}

} // namespace MoeGatingTopKHashRegBase
} // namespace optiling
