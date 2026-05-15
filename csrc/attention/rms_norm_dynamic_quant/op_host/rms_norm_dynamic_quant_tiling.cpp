/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_dynamic_quant_tiling.cpp
 * \brief
 */
#include "rms_norm_dynamic_quant_tiling.h"

namespace optiling {

constexpr int X_IDX = 0;
constexpr int GAMMA_IDX = 1;
constexpr int SMOOTH1_IDX = 2;
constexpr int SMOOTH2_IDX = 3;
constexpr int BETA_IDX = 4;

constexpr int Y1_IDX = 0;
constexpr int Y2_IDX = 1;
constexpr int SCALE1_IDX = 2;
constexpr int SCALE2_IDX = 3;

constexpr int NUM_WITH_BETA = 4;
constexpr int NUM_WITHOUT_BETA = 3;

constexpr int EPS_IDX = 0;
constexpr int OUT_QUANT_1_IDX = 1;
constexpr int OUT_QUANT_2_IDX = 2;
constexpr int DST_TYPE_IDX = 2;

constexpr uint64_t USR_WORKSPACE_SIZE_910B = 1;

constexpr uint32_t SIZEOF_B16 = 2;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint64_t ROW_FACTOR = 128;
constexpr uint64_t UB_RESERVED_BYTE = 768;
constexpr uint32_t MAX_ROW_STEP = 16;
constexpr uint32_t INT4_ALIGN_SIZE = 64;

constexpr uint32_t UB_TILING_POLICY_NORMAL = 1;
constexpr uint32_t UB_TILING_POLICY_SINGLE_ROW = 2;
constexpr uint32_t UB_TILING_POLICY_SLICE_D = 3;

constexpr uint32_t SLICE_COL_LEN = 8864;
constexpr uint32_t SLICE_COL_LEN_INT4 = 8832;

constexpr int32_t INT_NEGATIVE_ONE = -1;
constexpr int32_t INT_ZERO = 0;
constexpr int32_t INT_ONE = 1;
constexpr int32_t INT_TWO = 2;

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

bool CheckOptionalShapeExisting(const gert::StorageShape* smoothShape)
{
    OPS_CHECK(nullptr == smoothShape, OPS_LOG_D("CheckOptionalShapeExisting", "Get nullptr smoothShape"), return false);
    int64_t smoothShapeSize = smoothShape->GetOriginShape().GetShapeSize();
    OPS_CHECK((smoothShapeSize <= 0), OPS_LOG_D("CheckOptionalShapeExisting", "Get empty smoothShape"), return false);
    return true;
}

bool CheckOptionalBetaExisting(const gert::StorageShape* betaShape)
{
    OPS_CHECK(nullptr == betaShape, OPS_LOG_D("CheckOptionalBetaExisting", "Get nullptr betaShape"), return false);
    int64_t betaShapeSize = betaShape->GetOriginShape().GetShapeSize();
    OPS_CHECK((betaShapeSize <= 0), OPS_LOG_D("CheckOptionalBetaExisting", "Get empty betaShape"), return false);
    return true;
}

size_t GetworkspaceRowsNum(int32_t outQuant1Flag, int32_t outQuant2Flag, uint32_t smoothNum1_, uint32_t smoothNum2_)
{
    size_t workspaceRowsNum = INT_ZERO;
    if ((outQuant1Flag == INT_NEGATIVE_ONE && outQuant2Flag == INT_NEGATIVE_ONE)) {
        workspaceRowsNum = (smoothNum1_ == INT_ZERO && smoothNum2_ == INT_ZERO) ? INT_ONE : INT_TWO;
    } else {
        workspaceRowsNum = (outQuant1Flag == INT_ONE || outQuant2Flag == INT_ONE) ? INT_TWO : INT_ONE;
    }
    return workspaceRowsNum;
}

void RmsNormDynamicQuantTilingHelper::SetTilingDataAndTilingKeyAndWorkSpace(RmsNormDynamicQuantTilingData* tiling)
{
    context_->SetBlockDim(this->useCore_);
    tiling->set_useCore(this->useCore_);
    tiling->set_numFirstDim(this->numFirstDim_);
    tiling->set_numLastDim(this->numLastDim_);
    tiling->set_numLastDimAligned(this->numLastDimAligned_);
    tiling->set_firstDimPerCore(this->firstDimPerCore_);
    tiling->set_firstDimPerCoreTail(this->firstDimPerCoreTail_);
    tiling->set_firstDimPerLoop(this->firstDimPerLoop_);
    tiling->set_lastDimSliceLen(this->lastDimSliceLen_);
    tiling->set_lastDimLoopNum(this->lastDimLoopNum_);
    tiling->set_lastDimSliceLenTail(this->lastDimSliceLenTail_);
    tiling->set_smoothNum1(this->smoothNum1_);
    tiling->set_smoothNum2(this->smoothNum2_);
    tiling->set_epsilon(this->eps_);
    tiling->set_outQuant1Flag(this->outQuant1Flag);
    tiling->set_outQuant2Flag(this->outQuant2Flag);
    tiling->set_avgFactor(this->avgFactor_);
    tiling->set_betaFlag(this->betaFlag_);
    uint32_t tilingKey = 0;
    size_t usrSize = USR_WORKSPACE_SIZE_910B;

    if (this->ubTilingPolicy_ == UB_TILING_POLICY::NORMAL) {
        tilingKey += UB_TILING_POLICY_NORMAL;
    } else if (this->ubTilingPolicy_ == UB_TILING_POLICY::SINGLE_ROW) {
        tilingKey += UB_TILING_POLICY_SINGLE_ROW;
    } else {
        tilingKey += UB_TILING_POLICY_SLICE_D;
        size_t workspaceRowsNum =
            GetworkspaceRowsNum(this->outQuant1Flag, this->outQuant2Flag, this->smoothNum1_, this->smoothNum2_);
        usrSize = this->useCore_ * this->numLastDim_ * sizeof(float) * workspaceRowsNum;
    }

    context_->SetTilingKey(tilingKey);

    tiling->SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tiling->GetDataSize());

    // set workspace
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = this->sysWorkspaceSize_ + usrSize;

    OPS_LOG_I(
        "SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata useCore_: %lu, smoothNum1_: %u, smoothNum2_: %u",
        this->useCore_, this->smoothNum1_, this->smoothNum2_);
    OPS_LOG_I(
        "SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata N: %lu, D:%lu, DAligned: %lu", numFirstDim_, numLastDim_,
        numLastDimAligned_);
    OPS_LOG_I(
        "SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata firstDimPerCore_: %lu, firstDimPerCoreTail_: %lu",
        firstDimPerCore_, firstDimPerCoreTail_);
    OPS_LOG_I("SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata firstDimPerLoop_: %lu", firstDimPerLoop_);
    OPS_LOG_I(
        "SetTilingDataAndTilingKeyAndWorkSpace",
        "Tilingdata lastDimSliceLen_: %lu, lastDimLoopNum_: %lu, lastDimSliceLenTail_: %lu", lastDimSliceLen_,
        lastDimLoopNum_, lastDimSliceLenTail_);
    OPS_LOG_I("SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata eps_: %f, avgFactor_: %f", eps_, avgFactor_);
    OPS_LOG_I(
        "SetTilingDataAndTilingKeyAndWorkSpace", "Tilingdata tilingKey = %u, usr Workspace: %zu", tilingKey, usrSize);
}

bool RmsNormDynamicQuantTilingHelper::DoTiling()
{
    OPS_CHECK(
        (nullptr == context_), OPS_LOG_E("AddRmsNormDynamicQuantTiling", "Helper context_ get nullptr, return failed."),
        return false);
    OPS_CHECK(!GetBaseInfo(), OPS_LOG_E(context_->GetNodeName(), "GetBaseInfo falied, return false"), return false);
    OPS_CHECK(
        !GetShapeInfo(), OPS_LOG_E(context_->GetNodeName(), "GetShapeInfo falied, return false"), return false);
    OPS_CHECK(
        !DoBlockTiling(), OPS_LOG_E(context_->GetNodeName(), "DoBlockTiling falied, return false"), return false);
    OPS_CHECK(!DoUbTiling(), OPS_LOG_E(context_->GetNodeName(), "DoUbTiling falied, return false"), return false);
    return true;
}

bool RmsNormDynamicQuantTilingHelper::DoBlockTiling()
{
    // Block Tiling, Cut N
    this->firstDimPerCore_ = CeilDiv(this->numFirstDim_, this->socCoreNums_);
    this->useCore_ = CeilDiv(this->numFirstDim_, this->firstDimPerCore_);
    this->firstDimPerCore_ = CeilDiv(this->numFirstDim_, this->useCore_);
    this->firstDimPerCoreTail_ = this->numFirstDim_ - this->firstDimPerCore_ * (this->useCore_ - 1);
    OPS_LOG_I(
        "DoBlockTiling", "BlockTiling Factor: useCore_: %lu, firstDimPerCore_: %lu, firstDimPerCoreTail_: %lu",
        this->useCore_, this->firstDimPerCore_, this->firstDimPerCoreTail_);
    return true;
}

bool RmsNormDynamicQuantTilingHelper::InitializePlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    // OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    this->socCoreNums_ = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, this->ubSize_);
    this->sysWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return true;
}

bool RmsNormDynamicQuantTilingHelper::GetBaseInfo()
{
    if (!InitializePlatformInfo()) {
        return false;
    }

    auto attrs = context_->GetAttrs();
    OPS_CHECK(
        nullptr == attrs, OPS_LOG_E(context_->GetNodeName(), "Get attrs nullptr, return false."), return false);

    const float* epsPtr = attrs->GetFloat(EPS_IDX);
    if (epsPtr != nullptr) {
        this->eps_ = *epsPtr;
    }

    const gert::ContinuousVector* outputMaskAttr = attrs->GetAttrPointer<gert::ContinuousVector>(OUT_QUANT_1_IDX);
    if (outputMaskAttr != nullptr && outputMaskAttr->GetSize() == INT_TWO) {
        const bool* scalesArray = static_cast<const bool*>(outputMaskAttr->GetData());
        this->outQuant1Flag = (scalesArray[0] == true) ? 1 : 0;
        this->outQuant2Flag = (scalesArray[1] == true) ? 1 : 0;
    } else {
        this->outQuant1Flag = -1;
        this->outQuant2Flag = -1;
    }
    OPS_LOG_I("outputMask", "outQuant1Flag: %u, outQuant2Flag: %u", this->outQuant1Flag, this->outQuant2Flag);
    if (!ValidateBaseParameters()) {
        return false;
    }
    OPS_LOG_I(
        "GetBaseInfo", "socCoreNum: %lu, ubSize: %lu, sysWorkspaceSize: %lu, epsilon: %f", this->socCoreNums_,
        this->ubSize_, this->sysWorkspaceSize_, this->eps_);

    return true;
}

bool RmsNormDynamicQuantTilingHelper::ValidateBaseParameters()
{
    OPS_CHECK(
        this->eps_ <= 0,
        OPS_LOG_E(context_->GetNodeName(), "Epsilon less or equal than precision threshold, please check."),
        return false);
    OPS_CHECK(
        (this->ubSize_ <= 0), OPS_LOG_E(context_->GetNodeName(), "ubSize less or equal than zero, please check."),
        return false);
    OPS_CHECK(
        (this->socCoreNums_ <= 0),
        OPS_LOG_E(context_->GetNodeName(), "socCoreNums_ less or equal than zero, please check."), return false);

    return true;
}

ge::graphStatus CheckDtypeVaild(ge::DataType& srcDtype, std::vector<ge::DataType>& supportDtypeList)
{
    for (const auto& supportedDtype : supportDtypeList) {
        if (supportedDtype == srcDtype) {
            return ge::GRAPH_SUCCESS;
        }
    }
    return ge::GRAPH_FAILED;
}

bool RmsNormDynamicQuantTilingHelper::ValidateInputOutput()
{
    // 检查输入输出形状
    OPS_CHECK(
        CheckInputOutputShape() == false, OPS_LOG_E(context_->GetNodeName(), "Check tensor shape failed."), return false);

    // 验证输出数据类型
    auto y1DataType = context_->GetOutputDesc(Y1_IDX)->GetDataType();
    auto y2DataType = context_->GetOutputDesc(Y2_IDX)->GetDataType();
    std::vector<ge::DataType> supportedYDtypes = {ge::DataType::DT_INT8, ge::DataType::DT_INT4};
    if ((ge::GRAPH_SUCCESS != CheckDtypeVaild(y1DataType, supportedYDtypes)) ||
        (ge::GRAPH_SUCCESS != CheckDtypeVaild(y2DataType, supportedYDtypes)) || (y1DataType != y2DataType)) {
        OPS_LOG_E(context_->GetNodeName(), "Output dtype should be int8 int4 hifp8 and y1DataType y2DataType need same.");
        return false;
    }

    return true;
}

bool RmsNormDynamicQuantTilingHelper::CalculateShapeParameters()
{
    // 设置数据类型大小
    this->dtSize_ = SIZEOF_B16;

    // 获取输入形状
    auto xShape = context_->GetInputShape(X_IDX)->GetStorageShape();
    auto gammaShape = context_->GetInputShape(GAMMA_IDX)->GetStorageShape();
    size_t xDimNum = xShape.GetDimNum();
    size_t gammaDimNum = gammaShape.GetDimNum();

    // 计算numRow和numCol
    uint64_t numRow = 1;
    uint64_t numCol = 1;
    for (size_t i = 0; i < xDimNum - gammaDimNum; i++) {
        numRow *= xShape.GetDim(i);
    }
    for (size_t i = 0; i < gammaDimNum; i++) {
        numCol *= gammaShape.GetDim(i);
    }

    // 设置对齐大小和目标类型
    this->numFirstDim_ = numRow;
    this->numLastDim_ = numCol;
    auto y1DataType = context_->GetOutputDesc(Y1_IDX)->GetDataType();
    uint32_t alignSize = y1DataType == ge::DT_INT4 ? INT4_ALIGN_SIZE : BLOCK_SIZE;
    this->dstType_ = static_cast<uint32_t>(y1DataType);
    this->numLastDimAligned_ =
        CeilDiv(numCol, static_cast<uint64_t>(alignSize)) * static_cast<uint64_t>(alignSize);

    // 计算平均因子
    this->avgFactor_ = 1.0 / ((float)this->numLastDim_);

    return true;
}

bool RmsNormDynamicQuantTilingHelper::SetFlagsAndCheckConsistency()
{
    // 检查可选输入是否存在
    const gert::StorageShape* smooth1Shape = this->context_->GetOptionalInputShape(SMOOTH1_IDX);
    const gert::StorageShape* smooth2Shape = this->context_->GetOptionalInputShape(SMOOTH2_IDX);
    const gert::StorageShape* betaShape = this->context_->GetOptionalInputShape(BETA_IDX);
    bool smooth1Exist = CheckOptionalShapeExisting(smooth1Shape);
    bool smooth2Exist = CheckOptionalShapeExisting(smooth2Shape);
    bool betaExist = CheckOptionalBetaExisting(betaShape);

    // 设置标志位
    this->smoothNum1_ = (smooth1Exist) ? 1 : 0;
    this->smoothNum2_ = (smooth2Exist) ? 1 : 0;
    this->betaFlag_ = (betaExist) ? 1 : 0;

    // 检查形状匹配性
    auto gammaShape = context_->GetInputShape(GAMMA_IDX)->GetStorageShape();
    OPS_CHECK(
        (smooth1Exist && smooth1Shape->GetStorageShape() != gammaShape),
        OPS_LOG_E(context_->GetNodeName(), "GammaShape is not same to smooth1Shape."), return false);
    OPS_CHECK(
        (smooth2Exist && smooth2Shape->GetStorageShape() != gammaShape),
        OPS_LOG_E(context_->GetNodeName(), "GammaShape is not same to smooth2Shape."), return false);

    // 检查量化标志和可选输入的一致性
    if (this->outQuant1Flag == INT_NEGATIVE_ONE && this->outQuant2Flag == INT_NEGATIVE_ONE) {
        OPS_CHECK(
            (!smooth1Exist) && (smooth2Exist),
            OPS_LOG_E(context_->GetNodeName(), "Smooth2 exist but smooth1 not exist, bad input."), return false);
    }

    return true;
}

bool RmsNormDynamicQuantTilingHelper::GetShapeInfo()
{
    // 验证输入输出
    if (!ValidateInputOutput()) {
        return false;
    }

    // 计算形状参数
    if (!CalculateShapeParameters()) {
        return false;
    }

    // 设置标志和检查一致性
    if (!SetFlagsAndCheckConsistency()) {
        return false;
    }

    // 打印日志
    OPS_LOG_I("GetShapeInfo", "[N, D] = [%lu, %lu]", this->numFirstDim_, this->numLastDim_);
    OPS_LOG_I("GetShapeInfo", "dtSize_=%lu, avgFactor_=%f", this->dtSize_, this->avgFactor_);
    return true;
}

bool RmsNormDynamicQuantTilingHelper::DoUbTiling()
{
    OPS_CHECK(CheckUbNormalTiling(), OPS_LOG_I(context_->GetNodeName(), "Ub Tiling: Normal."), return true);
    OPS_CHECK(CheckUbSingleRowTiling(), OPS_LOG_I(context_->GetNodeName(), "Ub Tiling: SingleRow."), return true);
    OPS_CHECK(CheckUbSliceDTiling(), OPS_LOG_I(context_->GetNodeName(), "Ub Tiling: SliceD."), return true);
    return false;
}

bool RmsNormDynamicQuantTilingHelper::CheckUbNormalTiling()
{
    // 3 weights tensor required.
    int64_t ubConst = 0;
    if (this->betaFlag_ == 1) {
        ubConst = this->numLastDimAligned_ * this->dtSize_ * NUM_WITH_BETA + UB_RESERVED_BYTE;
    } else {
        ubConst = this->numLastDimAligned_ * this->dtSize_ * NUM_WITHOUT_BETA + UB_RESERVED_BYTE;
    }
    int64_t ubAvaliable = this->ubSize_ - ubConst;
    // 2 rows for tmpBuffer.
    int64_t coexistingRowsNum = 2 * (this->dtSize_) + 2 * (this->dtSize_) + 1 * sizeof(float) + 1 * sizeof(float);
    // 2 buffers for out_scale.
    int64_t rowCommons = coexistingRowsNum * this->numLastDimAligned_ + 2 * sizeof(float);
    int64_t rowStep = ubAvaliable / rowCommons;
    bool ret = (rowStep >= 1);
    OPS_LOG_I(
        this->context_->GetNodeName(),
        "CheckUbNormalTiling, ret:%d, ubConst: %ld, ubAvaliable=%ld, coexistingRowsNum: %ld, rowStep: %ld, "
        "rowCommons: %ld",
        ret, ubConst, ubAvaliable, coexistingRowsNum, rowStep, rowCommons);
    if (ret) {
        // No mutilN now. max RowStep = 16
        this->firstDimPerLoop_ = (rowStep <= MAX_ROW_STEP) ? rowStep : MAX_ROW_STEP;
        this->lastDimSliceLen_ = this->numLastDimAligned_;
        this->lastDimLoopNum_ = 1;
        this->lastDimSliceLenTail_ = 0;
        this->ubTilingPolicy_ = UB_TILING_POLICY::NORMAL;
    }
    return ret;
}

bool RmsNormDynamicQuantTilingHelper::CheckUbSingleRowTiling()
{
    // 2 tmp buffer, 2 rows copy in and 1 rows copy out
    int64_t ubRequired = ((2 + 1 + 1) * this->dtSize_ + 2 * sizeof(float)) * this->numLastDimAligned_;
    ubRequired = ubRequired + 2L * ROW_FACTOR * sizeof(float);
    bool ret = (((int64_t)this->ubSize_) >= ubRequired);
    OPS_LOG_I(this->context_->GetNodeName(), "CheckUbSingleRowTiling, ret:%d, ubRequired: %ld", ret, ubRequired);
    if (ret) {
        this->firstDimPerLoop_ = 1;
        this->lastDimSliceLen_ = this->numLastDimAligned_;
        this->lastDimLoopNum_ = 1;
        this->lastDimSliceLenTail_ = 0;
        this->ubTilingPolicy_ = UB_TILING_POLICY::SINGLE_ROW;
    }
    return ret;
}

bool RmsNormDynamicQuantTilingHelper::CheckUbSliceDTiling()
{
    OPS_LOG_I(this->context_->GetNodeName(), "CheckUbSliceDTiling success. Compute tiling by yourself.");
    this->ubTilingPolicy_ = UB_TILING_POLICY::SLICE_D;
    this->firstDimPerLoop_ = 1;
    if (this->dstType_ == 29) {
        this->lastDimSliceLen_ = SLICE_COL_LEN_INT4;
    } else {
        this->lastDimSliceLen_ = SLICE_COL_LEN;
    }
    this->lastDimSliceLenTail_ = (this->numLastDim_ % this->lastDimSliceLen_ == 0) ?
                                     this->lastDimSliceLen_ :
                                     this->numLastDim_ % this->lastDimSliceLen_;
    this->lastDimLoopNum_ = (this->numLastDim_ - this->lastDimSliceLenTail_) / this->lastDimSliceLen_;
    return true;
}

ge::graphStatus Tiling4AddRmsNormDynamicQuant(gert::TilingContext* context)
{
    OPS_CHECK(nullptr == context, OPS_LOG_E("AddRmsNormDynamicQuant", "Context is null"), return ge::GRAPH_FAILED);
    OPS_LOG_I(context->GetNodeName(), "Enter Tiling4AddRmsNormDynamicQuant");
    auto colShape = context->GetInputShape(GAMMA_IDX);
    // OP_CHECK_NULL_WITH_CONTEXT(context, colShape);
    auto colStorageShape = optiling::EnsureNotScalar(colShape->GetStorageShape());
    uint32_t col_val = colStorageShape.GetDim(0);
    bool isEmptyTensor = (col_val == 0);
    auto ptrCompileInfo = reinterpret_cast<const RmsNormDynamicQuantCompileInfo*>(context->GetCompileInfo());
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    platform_ascendc::SocVersion curSocVersion =
        (ptrCompileInfo) == nullptr ? ascendcPlatform.GetSocVersion() : ptrCompileInfo->curSocVersion;
    RmsNormDynamicQuantTilingData tiling;
    RmsNormDynamicQuantTilingHelper instanceNormV3TilingHelper(context);
    bool status = instanceNormV3TilingHelper.DoTiling();
    OPS_CHECK(
        !status, OPS_LOG_E(context->GetNodeName(), "DoTiling Failed, return Failed."), return ge::GRAPH_FAILED);
    instanceNormV3TilingHelper.SetTilingDataAndTilingKeyAndWorkSpace(&tiling);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepare4AddRmsNormDynamicQuant(gert::TilingParseContext* context)
{
    OPS_CHECK(nullptr == context, OPS_LOG_E("AddRmsNormDynamicQuant", "Context is null"), return ge::GRAPH_FAILED);
    OPS_LOG_D(context->GetNodeName(), "Enter TilingPrepare4AddRmsNormDynamicQuant.");
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    // OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto compileInfoPtr = context->GetCompiledInfo<RmsNormDynamicQuantCompileInfo>();
    // OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->curSocVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->maxUbSize);
    return ge::GRAPH_SUCCESS;
}

bool RmsNormDynamicQuantTilingHelper::CheckInputOutputShape()
{
    // Check Shape Not NULL
    const gert::StorageShape* xShape = this->context_->GetInputShape(X_IDX);
    const gert::StorageShape* gammaShape = this->context_->GetInputShape(GAMMA_IDX);

    const gert::StorageShape* y1Shape = this->context_->GetOutputShape(Y1_IDX);
    const gert::StorageShape* y2Shape = this->context_->GetOutputShape(Y2_IDX);
    const gert::StorageShape* scale1Shape = this->context_->GetOutputShape(SCALE1_IDX);
    const gert::StorageShape* scale2Shape = this->context_->GetOutputShape(SCALE2_IDX);

    // OP_CHECK_NULL_WITH_CONTEXT(this->context_, xShape);
    // OP_CHECK_NULL_WITH_CONTEXT(this->context_, gammaShape);
    // OP_CHECK_NULL_WITH_CONTEXT(this->context_, y1Shape);
    // OP_CHECK_NULL_WITH_CONTEXT(this->context_, y2Shape);
    // OP_CHECK_NULL_WITH_CONTEXT(this->context_, scale1Shape);
    // OP_CHECK_NULL_WITH_CONTEXT(this->context_, scale2Shape);

    // Check Shape relations
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gammaShape->GetStorageShape().GetDimNum();
    size_t y1DimNum = y1Shape->GetStorageShape().GetDimNum();
    size_t y2DimNum = y2Shape->GetStorageShape().GetDimNum();
    size_t scale1DimNum = scale1Shape->GetStorageShape().GetDimNum();
    size_t scale2DimNum = scale2Shape->GetStorageShape().GetDimNum();

    OPS_LOG_I(
        this->context_->GetNodeName(),
        "ShapeDim info: x.dim=%zu, gamma.dim=%zu, y1.dim=%zu, y2.dim=%zu, scale1.dim=%zu, "
        "scale2.dim=%zu",
        xDimNum, gammaDimNum, y1DimNum, y2DimNum, scale1DimNum, scale2DimNum);

    bool hasZeroDimTensor = xDimNum <= 0 || gammaDimNum <= 0;
    OPS_CHECK(
        (hasZeroDimTensor),
        OPS_LOG_E(
            this->context_->GetNodeName(),
            "Input x/y1/scale1DimNum shape invaild, dim num should not be smaller or equal to zero."),
        return false);
    OPS_CHECK(
        ((gammaDimNum != 1)), OPS_LOG_E(this->context_->GetNodeName(), "gamma shape dims not equal to 1. Tiling failed."),
        return false);
    gert::Shape shapeOfX = xShape->GetStorageShape();
    gert::Shape shapeOfGamma = gammaShape->GetStorageShape();
    OPS_CHECK(
        (shapeOfX[xDimNum - 1] != shapeOfGamma[gammaDimNum - 1]),
        OPS_LOG_E(context_->GetNodeName(), "gammaShape isn't consistent with the last dimension of x."), return false);
    return true;
}

IMPL_OP_OPTILING(RmsNormDynamicQuant)
    .Tiling(Tiling4AddRmsNormDynamicQuant)
    .TilingParse<RmsNormDynamicQuantCompileInfo>(TilingPrepare4AddRmsNormDynamicQuant);

} // namespace optiling