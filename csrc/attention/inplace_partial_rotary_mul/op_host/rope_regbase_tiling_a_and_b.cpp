/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rope_regbase_tiling_a_and_b.cc
 * \brief
 */
#include "inplace_partial_rotary_mul_tiling.h"

using namespace AscendC;

namespace optiling {
constexpr uint64_t ROPE_A_AND_B_TILING_PRIORITY = 40000;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t UB_FACTOR = 4;
constexpr int64_t UB_X_FACTOR = 4;
constexpr int64_t UB_COS_SIN_FACTOR = 2;
constexpr int64_t MAX_COPY_BLOCK_COUNT = 4095;
constexpr int32_t WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint64_t TILING_KEY_A = 20040;
constexpr uint64_t TILING_KEY_B = 20041;
constexpr uint64_t TILING_KEY_A_BF16_FP32 = 20140;
constexpr uint64_t TILING_KEY_A_FP16_FP32 = 20240;
constexpr uint64_t TILING_KEY_B_BF16_FP32 = 20141;
constexpr uint64_t TILING_KEY_B_FP16_FP32 = 20241;

bool RopeRegBaseTilingClassAAndB::IsCapable()
{
    // 处理全boardcast和不boardcast的情况
    return (IsRegbaseSocVersion()) && (layout_ == RopeLayout::NO_BROADCAST || layout_ == RopeLayout::BROADCAST_BSN);
}

ge::graphStatus RopeRegBaseTilingClassAAndB::MergeDim()
{
    b_ = b_ * n_ * s_;
    n_ = 1;
    s_ = 1;
    if (layout_ == RopeLayout::NO_BROADCAST) {
        cosb_ = b_;
    } else {
        cosb_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClassAAndB::SplitCore()
{
    blockFactorB_ = CeilDiv(static_cast<uint64_t>(b_), aicoreParams_.blockDim);
    blockNumB_ = CeilDiv(static_cast<int64_t>(b_), blockFactorB_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClassAAndB::ComputeUbFactor()
{
    ubFactorB_ = 1;

    auto cosDtype = context_->GetInputDesc(1)->GetDataType();
    bool isMixedPrecision = (dtype_ == ge::DT_BF16 || dtype_ == ge::DT_FLOAT16) && cosDtype == ge::DT_FLOAT;

    int64_t dSizeX = CeilAlign(sliceLength_ * GetSizeByDataType(dtype_) / dSplitCoef_, this->blockSize_) * dSplitCoef_;
    int64_t dSizeCosSin =
        CeilAlign(sliceLength_ * GetSizeByDataType(cosDtype) / dSplitCoef_, this->blockSize_) * dSplitCoef_;

    if (layout_ == RopeLayout::NO_BROADCAST) {
        int64_t totalPerBUnit;
        if (isMixedPrecision) {
            // UB: 4 queues x double-buffer = 8 total buffers
            // 4 * ubFactorB * dSizeX + 4 * ubFactorB * dSizeCosSin
            // Per-B unit: 4 * dSizeX + 4 * dSizeCosSin
            totalPerBUnit = (dSizeX + dSizeCosSin) * UB_FACTOR;
        } else {
            // UB: 4 queues x double-buffer = 8 * ubFactorB * dSizeX
            // Per-B unit: 8 * dSizeX
            totalPerBUnit = dSizeX * UB_FACTOR * DOUBLE_BUFFER;
        }
        int64_t numOfDAvailable = FloorDiv(static_cast<int64_t>(aicoreParams_.ubSize), totalPerBUnit);
        OPS_ERR_IF(numOfDAvailable < 1,
            OPS_LOG_E(context_,
                "D is too big to load in ub, ubSize is %ld bytes, loading requires %ld bytes.",
                static_cast<int64_t>(aicoreParams_.ubSize),
                totalPerBUnit),
            return ge::GRAPH_FAILED);
        ubFactorB_ = std::min(blockFactorB_, numOfDAvailable);
    } else {
        if (isMixedPrecision) {
            // UB: UB_X_FACTOR * ubFactorB * dSizeX + UB_COS_SIN_FACTOR * dSizeCosSin
            int64_t availableForX = static_cast<int64_t>(aicoreParams_.ubSize) - UB_COS_SIN_FACTOR * dSizeCosSin;
            if (availableForX <= 0) {
                OPS_LOG_E(context_,
                    "D is too big to load in ub, ubSize is %ld bytes, cos/sin requires %ld bytes.",
                    static_cast<int64_t>(aicoreParams_.ubSize),
                    UB_COS_SIN_FACTOR * dSizeCosSin);
                return ge::GRAPH_FAILED;
            }
            int64_t numOfDAvailable = FloorDiv(availableForX, UB_X_FACTOR * dSizeX);
            OPS_ERR_IF(numOfDAvailable < 1,
                OPS_LOG_E(context_,
                    "D is too big to load in ub, ubSize is %ld bytes, loading requires %ld bytes.",
                    static_cast<int64_t>(aicoreParams_.ubSize),
                    UB_X_FACTOR * dSizeX + UB_COS_SIN_FACTOR * dSizeCosSin),
                return ge::GRAPH_FAILED);
            ubFactorB_ = std::min(blockFactorB_, numOfDAvailable);
        } else {
            int64_t numOfDAvailable = FloorDiv(static_cast<int64_t>(aicoreParams_.ubSize), DOUBLE_BUFFER * dSizeX);
            OPS_ERR_IF(numOfDAvailable < UB_FACTOR,
                OPS_LOG_E(context_,
                    "D is too big to load in ub, ubSize is %ld bytes, loading requires %ld bytes.",
                    static_cast<int64_t>(aicoreParams_.ubSize),
                    UB_FACTOR * dSizeX * (ubFactorB_ + 1)),
                return ge::GRAPH_FAILED);
            numOfDAvailable -= 1;
            numOfDAvailable /= DOUBLE_BUFFER;
            ubFactorB_ = std::min(blockFactorB_, numOfDAvailable);
        }
    }

    ubFactorB_ = std::min(ubFactorB_, MAX_COPY_BLOCK_COUNT / dSplitCoef_);

    return ge::GRAPH_SUCCESS;
}

void RopeRegBaseTilingClassAAndB::SetTilingData()
{
    tilingData_.set_B(b_);
    tilingData_.set_CosB(cosb_);
    tilingData_.set_S(s_);
    tilingData_.set_D(d_);
    tilingData_.set_N(n_);
    tilingData_.set_blockNumB(blockNumB_);
    tilingData_.set_blockFactorB(blockFactorB_);
    tilingData_.set_blockNumS(blockNumS_);
    tilingData_.set_blockFactorS(blockFactorS_);
    tilingData_.set_ubLoopNumS(ubLoopNumS_);
    tilingData_.set_ubFactorS(ubFactorS_);
    tilingData_.set_ubTailFactorS(ubTailFactorS_);
    tilingData_.set_ubLoopNumB(ubLoopNumB_);
    tilingData_.set_ubFactorB(ubFactorB_);
    tilingData_.set_ubTailFactorB(ubTailFactorB_);
    tilingData_.set_ubLoopNumN(ubLoopNumN_);
    tilingData_.set_ubFactorN(ubFactorN_);
    tilingData_.set_ubTailFactorN(ubTailFactorN_);
    tilingData_.set_rotaryMode(static_cast<int64_t>(rotaryMode_));
    tilingData_.set_sliceStart(static_cast<int64_t>(sliceStart_));
    tilingData_.set_sliceEnd(static_cast<int64_t>(sliceEnd_));
    tilingData_.set_sliceLength(static_cast<int64_t>(sliceLength_));

    OPS_LOG_I(context_->GetNodeName(),
        "RopeRegBaseTilingClassAAndB tilingData: "
        "B is %ld, CosB is %ld, S is %ld, D is %ld, N is %ld, blockNumB %ld,"
        "blockFactorB_ is %ld, blockNumS %ld, blockFactorS is %ld, ubLoopNumS is %ld,"
        "ubFactorS is %ld, ubTailFactorS %ld, ubLoopNumB is %ld, ubFactorB is %ld,"
        "ubTailFactorB is %ld, ubLoopNumN is %ld, ubFactorN is %ld, ubTailFactorN is %ld,"
        "rotaryMode is %ld, tilingKey is %ld, sliceStart is %ld, sliceEnd is %ld, sliceLength is %ld",
        tilingData_.get_B(),
        tilingData_.get_CosB(),
        tilingData_.get_S(),
        tilingData_.get_D(),
        tilingData_.get_N(),
        tilingData_.get_blockNumB(),
        tilingData_.get_blockFactorB(),
        tilingData_.get_blockNumS(),
        tilingData_.get_blockFactorS(),
        tilingData_.get_ubLoopNumS(),
        tilingData_.get_ubFactorS(),
        tilingData_.get_ubTailFactorS(),
        tilingData_.get_ubLoopNumB(),
        tilingData_.get_ubFactorB(),
        tilingData_.get_ubTailFactorB(),
        tilingData_.get_ubLoopNumN(),
        tilingData_.get_ubFactorN(),
        tilingData_.get_ubTailFactorN(),
        tilingData_.get_rotaryMode(),
        GetTilingKey(),
        tilingData_.get_sliceStart(),
        tilingData_.get_sliceEnd(),
        tilingData_.get_sliceLength());
}

ge::graphStatus RopeRegBaseTilingClassAAndB::DoOpTiling()
{
    OPS_ERR_IF(MergeDim() != ge::GRAPH_SUCCESS,
        OPS_LOG_E(context_->GetNodeName(), "failed to merge dim."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(SplitCore() != ge::GRAPH_SUCCESS,
        OPS_LOG_E(context_->GetNodeName(), "failed to split core."),
        return ge::GRAPH_FAILED);
    OPS_ERR_IF(ComputeUbFactor() != ge::GRAPH_SUCCESS,
        OPS_LOG_E(context_->GetNodeName(), "failed to compute ub factor."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClassAAndB::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t RopeRegBaseTilingClassAAndB::GetTilingKey() const
{
    auto xDtype = context_->GetInputDesc(0)->GetDataType();
    auto cosDtype = context_->GetInputDesc(1)->GetDataType();

    bool isNoBroadcast = (layout_ == RopeLayout::NO_BROADCAST);

    if (xDtype == ge::DT_BF16 && cosDtype == ge::DT_FLOAT) {
        return isNoBroadcast ? TILING_KEY_A_BF16_FP32 : TILING_KEY_B_BF16_FP32;
    } else if (xDtype == ge::DT_FLOAT16 && cosDtype == ge::DT_FLOAT) {
        return isNoBroadcast ? TILING_KEY_A_FP16_FP32 : TILING_KEY_B_FP16_FP32;
    }

    return isNoBroadcast ? TILING_KEY_A : TILING_KEY_B;
}

ge::graphStatus RopeRegBaseTilingClassAAndB::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClassAAndB::PostTiling()
{
    SetTilingData();
    uint64_t tilingKey = GetTilingKey();
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(blockNumB_);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL(context_, workspaces, return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

// REGISTER_OPS_TILING_TEMPLATE(InplacePartialRotaryMul, RopeRegBaseTilingClassAAndB, ROPE_A_AND_B_TILING_PRIORITY);
}  // namespace optiling
