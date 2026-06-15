/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "inplace_partial_rotary_mul_tiling.h"

namespace optiling {
namespace {
constexpr int64_t FP16_BF16_DTYPE_SIZE = 2;
constexpr int64_t FP32_DTYPE_SIZE = 4;
constexpr int64_t REPEAT_FP32 = 64;
constexpr int64_t ALIGN_32 = 8;
constexpr int64_t ALIGN_16 = 16;
constexpr int64_t CONST_4 = 4;
constexpr int64_t DSA_BY_CACHE_BRC_KEY = 3001;
constexpr int64_t DSA_BY_CACHE_NO_BRC_KEY = 3002;
constexpr int64_t DSA_BY_CACHE_FP32_ROPE_OFFSET = 10;
constexpr uint32_t MODE_ATTR_IDX = 0;
constexpr uint32_t PARTIAL_SLICE_ATTR_IDX = 1;
constexpr uint32_t ROPE_DIM_ATTR_IDX = 2;
constexpr uint32_t INVERSE_ATTR_IDX = 3;

int64_t CeilDivInt(int64_t value, int64_t factor)
{
    if (factor == 0) {
        return 0;
    }
    return (value + factor - 1) / factor;
}

int64_t RemInt(int64_t value, int64_t factor)
{
    if (factor == 0) {
        return 0;
    }
    return value % factor;
}
} // namespace

class InplacePartialRotaryMulDsaByCacheTiling {
public:
    explicit InplacePartialRotaryMulDsaByCacheTiling(gert::TilingContext* context) : context_(context) {};

    ge::graphStatus Init();
    ge::graphStatus DoTiling();

private:
    ge::graphStatus CheckInput();
    ge::graphStatus CalTilingData();
    void FillTilingData();

    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t dtypeSize_ = FP16_BF16_DTYPE_SIZE;
    int64_t oneBlockSize_ = ALIGN_16;
    int64_t dim0_ = 0;
    int64_t dim1_ = 0;
    int64_t dim2_ = 0;
    int64_t allHeadDim_ = 0;
    int64_t headDim_ = 0;
    int64_t start_ = 0;
    int64_t end_ = 0;
    int64_t cacheStride_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t coreTUbLoopTime_ = 0;
    int64_t coreBUbLoopTime_ = 0;
    int64_t coreTUbLoopTail_ = 0;
    int64_t coreBUbLoopTail_ = 0;
    int64_t ubFactor_ = 0;
    int64_t blockFactor_ = 0;
    int64_t tilingKey_ = DSA_BY_CACHE_BRC_KEY;
    int64_t inverse_ = 0;
    bool isBrc_ = true;
    bool isFp32Rope_ = false;
    gert::TilingContext* context_ = nullptr;
    RopeRegbaseTilingData tilingData_;
};

ge::graphStatus InplacePartialRotaryMulDsaByCacheTiling::Init()
{
    OPS_ERR_IF(context_ == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("Tiling4InplacePartialRotaryMulDsaByCache", "Tiling context is null"),
        return ge::GRAPH_FAILED);
    auto platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("Tiling4InplacePartialRotaryMulDsaByCache", "Tiling platformInfo is null"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OPS_ERR_IF(coreNum_ <= 0, OPS_LOG_E(context_->GetNodeName(), "coreNum must be greater than 0."),
        return ge::GRAPH_FAILED);
    uint64_t ubSizePlatform = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    ubSize_ = static_cast<int64_t>(ubSizePlatform);
    OPS_ERR_IF(ubSize_ <= 0, OPS_LOG_E(context_->GetNodeName(), "ubSize must be greater than 0."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulDsaByCacheTiling::CheckInput()
{
    auto xInput = context_->GetInputShape(0);
    auto positionsInput = context_->GetInputShape(1);
    auto cosSinInput = context_->GetInputShape(2);
    auto xDesc = context_->GetInputDesc(0);
    auto positionsDesc = context_->GetInputDesc(1);
    auto cosSinDesc = context_->GetInputDesc(2);
    OPS_ERR_IF(xInput == nullptr || positionsInput == nullptr || cosSinInput == nullptr,
        OPS_LOG_E(context_->GetNodeName(), "input shape is nullptr."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(xDesc == nullptr || positionsDesc == nullptr || cosSinDesc == nullptr,
        OPS_LOG_E(context_->GetNodeName(), "input desc is nullptr."), return ge::GRAPH_FAILED);

    auto dataDtype = xDesc->GetDataType();
    auto positionsDtype = positionsDesc->GetDataType();
    auto cosSinDtype = cosSinDesc->GetDataType();
    OPS_ERR_IF(positionsDtype != ge::DT_INT64,
        OPS_LOG_E(context_->GetNodeName(), "positions only supports int64."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(cosSinDtype != dataDtype && cosSinDtype != ge::DT_FLOAT,
        OPS_LOG_E(context_->GetNodeName(), "cos_sin_cache dtype must be same as x or float32."),
        return ge::GRAPH_FAILED);
    isFp32Rope_ = (dataDtype != ge::DT_FLOAT && cosSinDtype == ge::DT_FLOAT);
    if (dataDtype == ge::DT_FLOAT) {
        dtypeSize_ = FP32_DTYPE_SIZE;
        oneBlockSize_ = ALIGN_32;
    }

    gert::Shape xShape = xInput->GetStorageShape();
    gert::Shape positionsShape = positionsInput->GetStorageShape();
    gert::Shape cosSinShape = cosSinInput->GetStorageShape();
    OPS_ERR_IF(xShape.GetDimNum() != CONST_4,
        OPS_LOG_E(context_->GetNodeName(), "x must be 4D, got %zu.", xShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(positionsShape.GetDimNum() != 1,
        OPS_LOG_E(context_->GetNodeName(), "positions must be 1D, got %zu.", positionsShape.GetDimNum()),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(cosSinShape.GetDimNum() < 2,
        OPS_LOG_E(context_->GetNodeName(), "cos_sin_cache dim is invalid."), return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(context_->GetNodeName(), "attrs is nullptr"),
        return ge::GRAPH_FAILED);
    int64_t mode = *(attrs->GetAttrPointer<int64_t>(MODE_ATTR_IDX));
    OPS_ERR_IF(mode != 1, OPS_LOG_E(context_->GetNodeName(), "mode only supports interleave."),
        return ge::GRAPH_FAILED);
    auto sliceListAttr = attrs->GetAttrPointer<gert::ContinuousVector>(PARTIAL_SLICE_ATTR_IDX);
    OPS_ERR_IF(sliceListAttr == nullptr, OPS_LOG_E(context_->GetNodeName(), "partial_slice is nullptr"),
        return ge::GRAPH_FAILED);
    auto sliceData = static_cast<const int64_t *>(sliceListAttr->GetData());
    start_ = sliceData[0];
    end_ = sliceData[1];
    headDim_ = end_ - start_;
    OPS_ERR_IF(start_ < 0 || headDim_ <= 0,
        OPS_LOG_E(context_->GetNodeName(), "partial_slice is invalid."), return ge::GRAPH_FAILED);
    int64_t ropeDim = *(attrs->GetAttrPointer<int64_t>(ROPE_DIM_ATTR_IDX));
    OPS_ERR_IF(ropeDim > 0 && ropeDim != headDim_,
        OPS_LOG_E(context_->GetNodeName(), "rope_dim must match partial_slice length."), return ge::GRAPH_FAILED);
    inverse_ = *(attrs->GetAttrPointer<bool>(INVERSE_ATTR_IDX)) ? 1 : 0;

    dim0_ = xShape.GetDim(0);
    dim1_ = xShape.GetDim(1) * xShape.GetDim(2);
    dim2_ = headDim_;
    allHeadDim_ = xShape.GetDim(3);
    OPS_ERR_IF(positionsShape.GetDim(0) != dim0_,
        OPS_LOG_E(context_->GetNodeName(), "positions length must match x dim0."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(end_ > allHeadDim_,
        OPS_LOG_E(context_->GetNodeName(), "partial_slice end exceeds x last dim."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(headDim_ > REPEAT_FP32,
        OPS_LOG_E(context_->GetNodeName(), "headDim greater than one fp32 repeat is not supported."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(headDim_ % oneBlockSize_ != 0,
        OPS_LOG_E(context_->GetNodeName(), "headDim must be 32B aligned."), return ge::GRAPH_FAILED);
    int64_t cacheLastDim = cosSinShape.GetDim(cosSinShape.GetDimNum() - 1);
    OPS_ERR_IF(cacheLastDim != headDim_ * 2,
        OPS_LOG_E(context_->GetNodeName(), "cos_sin_cache last dim must be twice partial_slice length."),
        return ge::GRAPH_FAILED);
    cacheStride_ = 1;
    for (int64_t i = 1; i < cosSinShape.GetDimNum(); ++i) {
        cacheStride_ *= cosSinShape.GetDim(i);
    }
    isBrc_ = dim1_ != 1;
    tilingKey_ = isBrc_ ? DSA_BY_CACHE_BRC_KEY : DSA_BY_CACHE_NO_BRC_KEY;
    if (isFp32Rope_) {
        tilingKey_ += DSA_BY_CACHE_FP32_ROPE_OFFSET;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplacePartialRotaryMulDsaByCacheTiling::CalTilingData()
{
    int64_t ubNum = ubSize_ / sizeof(float);
    int64_t reserve = dim1_ * dim2_;
    int64_t perToken = CONST_4 * dim1_ * dim2_ + CONST_4 * dim2_ + 2;
    int64_t preCoreNumFactor = CeilDivInt(dim0_, coreNum_);
    usedCoreNum_ = CeilDivInt(dim0_, preCoreNumFactor);
    int64_t tailCoreNum = dim0_ - preCoreNumFactor * (usedCoreNum_ - 1);
    blockFactor_ = preCoreNumFactor;
    ubFactor_ = (ubNum - reserve) / perToken;
    if (ubFactor_ > preCoreNumFactor) {
        ubFactor_ = preCoreNumFactor;
    }
    OPS_ERR_IF(ubFactor_ <= 0, OPS_LOG_E(context_->GetNodeName(), "ubFactor must be greater than 0."),
        return ge::GRAPH_FAILED);

    coreBUbLoopTime_ = CeilDivInt(preCoreNumFactor, ubFactor_);
    coreBUbLoopTail_ = RemInt(preCoreNumFactor, ubFactor_);
    if (coreBUbLoopTail_ == 0) {
        coreBUbLoopTail_ = ubFactor_;
    }
    coreTUbLoopTime_ = CeilDivInt(tailCoreNum, ubFactor_);
    coreTUbLoopTail_ = RemInt(tailCoreNum, ubFactor_);
    if (coreTUbLoopTail_ == 0) {
        coreTUbLoopTail_ = ubFactor_;
    }
    return ge::GRAPH_SUCCESS;
}

void InplacePartialRotaryMulDsaByCacheTiling::FillTilingData()
{
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_numHead(dim1_);
    tilingData_.set_headDim(headDim_);
    tilingData_.set_allHeadDim(allHeadDim_);
    tilingData_.set_coreTUbLoopTime(coreTUbLoopTime_);
    tilingData_.set_coreBUbLoopTime(coreBUbLoopTime_);
    tilingData_.set_coreTUbLoopTail(coreTUbLoopTail_);
    tilingData_.set_coreBUbLoopTail(coreBUbLoopTail_);
    tilingData_.set_ubFactor(ubFactor_);
    tilingData_.set_start(start_);
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_cacheStride(cacheStride_);
    tilingData_.set_cacheOffset(0);
    tilingData_.set_inverse(inverse_);
}

ge::graphStatus InplacePartialRotaryMulDsaByCacheTiling::DoTiling()
{
    OPS_ERR_IF(CheckInput() != ge::GRAPH_SUCCESS,
        OPS_LOG_E(context_->GetNodeName(), "CheckInput failed."), return ge::GRAPH_FAILED);
    OPS_ERR_IF(CalTilingData() != ge::GRAPH_SUCCESS,
        OPS_LOG_E(context_->GetNodeName(), "CalTilingData failed."), return ge::GRAPH_FAILED);

    FillTilingData();
    context_->SetBlockDim(usedCoreNum_);
    context_->SetTilingKey(tilingKey_);
    size_t *workspace = context_->GetWorkspaceSizes(1);
    workspace[0] = static_cast<size_t>(16 * 1024 * 1024);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4InplacePartialRotaryMulDsaByCache(gert::TilingContext* context)
{
    InplacePartialRotaryMulDsaByCacheTiling tilingImpl(context);
    if (tilingImpl.Init() != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context, "Tiling4InplacePartialRotaryMulDsaByCache init failed.");
        return ge::GRAPH_FAILED;
    }
    if (tilingImpl.DoTiling() != ge::GRAPH_SUCCESS) {
        OPS_LOG_E(context, "Tiling4InplacePartialRotaryMulDsaByCache do tiling failed.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(InplacePartialRotaryMulDsaByCache)
    .Tiling(Tiling4InplacePartialRotaryMulDsaByCache)
    .TilingParse<RotaryPositionEmbeddingCompileInfo>(TilingPrepareForRotaryPositionEmbedding);
} // namespace optiling
