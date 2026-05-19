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
 * \file rope_regbase_tiling_aba_and_ba.cc
 * \brief
 */
#include "inplace_partial_rotary_mul_tiling.h"

using namespace AscendC;

namespace optiling {

constexpr uint64_t ROPE_ABA_AND_BA_TILING_PRIORITY = 10000;
constexpr uint64_t TILING_KEY_ABA = 20010;
constexpr uint64_t TILING_KEY_BA = 20011;
constexpr int64_t UB_FACTOR = 4;
constexpr int64_t MAX_COPY_BLOCK_COUNT = 4095;
constexpr int32_t WORKSPACE_SIZE = 16 * 1024 * 1024;

bool RopeRegBaseTilingClassABAAndBA::IsCapable()
{
    // BNSD对应11SD和B1SD两种brc模式
    return (IsRegbaseSocVersion()) && (layout_ == RopeLayout::BNSD);
}

ge::graphStatus RopeRegBaseTilingClassABAAndBA::SplitCore()
{
    // B大于等于核数，且能被核数整除，则仅在B轴分核
    if (b_ % aicoreParams_.blockDim == 0) {
        blockNumB_ = aicoreParams_.blockDim;
        blockFactorB_ = b_ / aicoreParams_.blockDim;
        blockNumS_ = 1;
        blockFactorS_ = s_;
        return ge::GRAPH_SUCCESS;
    }

    // S大于等于核数，且能被核数整除，则仅在S轴分核
    if (s_ % aicoreParams_.blockDim == 0) {
        blockNumS_ = aicoreParams_.blockDim;
        blockFactorS_ = s_ / aicoreParams_.blockDim;
        blockNumB_ = 1;
        blockFactorB_ = b_;
        return ge::GRAPH_SUCCESS;
    }

    // 尝试优先对B分核，再尝试优先对S分核，比较二者切分后的总核数
    auto blockFactorB1 = CeilDiv(static_cast<uint64_t>(b_), aicoreParams_.blockDim);
    auto blockNumB1 = CeilDiv(static_cast<uint64_t>(b_), blockFactorB1);
    if (blockNumB1 == 0) {
        OPS_LOG_I("RopeRegBaseTilingClassABAAndBA SplitCore error, blockNumB1 == 0");
        return ge::GRAPH_FAILED;
    }
    auto blockNumS1 = std::min(static_cast<uint64_t>(s_), aicoreParams_.blockDim / blockNumB1);
    auto blockFactorS1 = CeilDiv(static_cast<uint64_t>(s_), blockNumS1);
    blockNumS1 = CeilDiv(static_cast<uint64_t>(s_), blockFactorS1);
    auto usedCoreNum1 = blockNumB1 * blockNumS1;

    auto blockFactorS2 = CeilDiv(static_cast<uint64_t>(s_), aicoreParams_.blockDim);
    auto blockNumS2 = CeilDiv(static_cast<uint64_t>(s_), blockFactorS2);
    if (blockNumS2 == 0) {
        OPS_LOG_I("RopeRegBaseTilingClassABAAndBA SplitCore error, blockNumS2 == 0");
        return ge::GRAPH_FAILED;
    }
    auto blockNumB2 = std::min(static_cast<uint64_t>(b_), aicoreParams_.blockDim / blockNumS2);
    auto blockFactorB2 = CeilDiv(static_cast<uint64_t>(b_), blockNumB2);
    blockNumB2 = CeilDiv(static_cast<uint64_t>(b_), blockFactorB2);
    auto usedCoreNum2 = blockNumB2 * blockNumS2;

    if (usedCoreNum1 >= usedCoreNum2) {
        blockNumB_ = blockNumB1;
        blockFactorB_ = blockFactorB1;
        blockNumS_ = blockNumS1;
        blockFactorS_ = blockFactorS1;
    } else {
        blockNumB_ = blockNumB2;
        blockFactorB_ = blockFactorB2;
        blockNumS_ = blockNumS2;
        blockFactorS_ = blockFactorS2;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClassABAAndBA::ComputeUbFactor()
{
    ubFactorS_ = 1;
    ubFactorB_ = 1;
    ubFactorN_ = 1;
    // UB占用大小：
    // 11sd：4 * (bn + 1) * s * dAlign
    // b1sd：4 * b * (n + 1) * s * dAlign
    int64_t dSize = CeilAlign(sliceLength_ * GetSizeByDataType(dtype_) / dSplitCoef_, this->blockSize_) * dSplitCoef_;
    int64_t numOfDAvailable = FloorDiv(static_cast<int64_t>(aicoreParams_.ubSize), UB_FACTOR * dSize);
    OPS_ERR_IF(numOfDAvailable < ubFactorB_ + 1,
                OPS_LOG_E(context_, "D is too big to load in ub, ubSize is %ld bytes, loading requires %ld bytes.",
                        static_cast<int64_t>(aicoreParams_.ubSize), UB_FACTOR * dSize * (ubFactorB_ + 1)),
                return ge::GRAPH_FAILED);

    ubFactorS_ = std::min(blockFactorS_, FloorDiv(numOfDAvailable, ubFactorB_ + 1));
    ubFactorS_ = std::min(ubFactorS_, MAX_COPY_BLOCK_COUNT / dSplitCoef_);
    if (ubFactorS_ == 0) {
        OPS_LOG_I("RopeRegBaseTilingClassABAAndBA ComputeUbFactor error, ubFactorS_ == 0");
        return ge::GRAPH_FAILED;
    }
    numOfDAvailable /= ubFactorS_;
    if (numOfDAvailable <= ubFactorB_ + 1) {
        return ge::GRAPH_SUCCESS;
    }

    if (cosb_ == 1) {
        numOfDAvailable -= 1;
        ubFactorN_ = std::min(n_, numOfDAvailable);
        if (ubFactorN_ == 0) {
            OPS_LOG_I("RopeRegBaseTilingClassABAAndBA ComputeUbFactor error, ubFactorN_ == 0");
            return ge::GRAPH_FAILED;
        }
        numOfDAvailable /= ubFactorN_;
    } else {
        ubFactorN_ = std::min(n_, numOfDAvailable - 1);
        numOfDAvailable /= (ubFactorN_ + 1);
    }

    if (numOfDAvailable <= 1) {
        return ge::GRAPH_SUCCESS;
    }
    ubFactorB_ = std::min(blockFactorB_, numOfDAvailable);

    return ge::GRAPH_SUCCESS;
}

void RopeRegBaseTilingClassABAAndBA::SetTilingData()
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
            "RopeRegBaseTilingClassABAAndBA tilingData: "
            "B is %ld, CosB is %ld, S is %ld, D is %ld, N is %ld, blockNumB %ld,"
            "blockFactorB_ is %ld, blockNumS %ld, blockFactorS is %ld, ubLoopNumS is %ld,"
            "ubFactorS is %ld, ubTailFactorS %ld, ubLoopNumB is %ld, ubFactorB is %ld,"
            "ubTailFactorB is %ld, ubLoopNumN is %ld, ubFactorN is %ld, ubTailFactorN is %ld,"
            "rotaryMode is %ld, tilingKey is %ld, sliceStart is %ld, sliceEnd is %ld, sliceLength is %ld",
            tilingData_.get_B(), tilingData_.get_CosB(), tilingData_.get_S(), tilingData_.get_D(), tilingData_.get_N(),
            tilingData_.get_blockNumB(), tilingData_.get_blockFactorB(), tilingData_.get_blockNumS(),
            tilingData_.get_blockFactorS(), tilingData_.get_ubLoopNumS(), tilingData_.get_ubFactorS(),
            tilingData_.get_ubTailFactorS(), tilingData_.get_ubLoopNumB(), tilingData_.get_ubFactorB(),
            tilingData_.get_ubTailFactorB(), tilingData_.get_ubLoopNumN(), tilingData_.get_ubFactorN(),
            tilingData_.get_ubTailFactorN(), tilingData_.get_rotaryMode(), GetTilingKey(), tilingData_.get_sliceStart(), tilingData_.get_sliceEnd(), tilingData_.get_sliceLength());
}

ge::graphStatus RopeRegBaseTilingClassABAAndBA::DoOpTiling()
{
    OPS_ERR_IF(SplitCore() != ge::GRAPH_SUCCESS, OPS_LOG_E(context_->GetNodeName(), "failed to split core."),
                return ge::GRAPH_FAILED);
    OPS_ERR_IF(ComputeUbFactor() != ge::GRAPH_SUCCESS,
                OPS_LOG_E(context_->GetNodeName(), "failed to compute ub factor."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClassABAAndBA::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t RopeRegBaseTilingClassABAAndBA::GetTilingKey() const
{
    return cosb_ == 1 ? TILING_KEY_BA : TILING_KEY_ABA;
}

ge::graphStatus RopeRegBaseTilingClassABAAndBA::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClassABAAndBA::PostTiling()
{
    SetTilingData();
    uint64_t tilingKey = GetTilingKey();
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(blockNumB_ * blockNumS_);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL(context_, workspaces, return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

// REGISTER_OPS_TILING_TEMPLATE(InplacePartialRotaryMul, RopeRegBaseTilingClassABAAndBA, ROPE_ABA_AND_BA_TILING_PRIORITY);
} // namespace optiling