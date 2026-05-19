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
 * \file rope_regbase_tiling_ab.cc
 * \brief
 */

#include "inplace_partial_rotary_mul_tiling.h"

namespace optiling {

constexpr size_t RESERVERD_WORKSPACE_SIZE = static_cast<size_t>(16 * 1024 * 1024);
constexpr int64_t MAX_COPY_BLOCK_COUNT = 4095;
constexpr int64_t CONST_TWO = 2;
constexpr int64_t CONST_FOUR = 4;
constexpr int64_t DB_FLAG = 2;
constexpr int64_t TILING_KEY_AB = 20030;

ge::graphStatus RopeRegBaseTilingClassAB::DoOpTiling()
{
    int64_t bs = b_ * s_;
    if (cosb_ == 1) {
        bs = s_;
        n_ = b_ * n_;
    }
    int64_t typeSize = ge::GetSizeByDataType(dtype_);
    if (typeSize == 0) {
        OPS_LOG_I("RopeRegBaseTilingClassAB DoOpTiling error, typeSize == 0");
        return ge::GRAPH_FAILED;
    }
    dAlign_ = CeilAlign(sliceLength_ / dSplitCoef_, blockSize_ / typeSize) * dSplitCoef_;

    blockFactorBS_ = CeilDiv(bs, int64_t(aicoreParams_.blockDim));
    blockNumBS_ = CeilDiv(bs, blockFactorBS_);
    blockTailBS_ = bs - (blockNumBS_ - 1) * blockFactorBS_;

    if (bs <= int64_t(aicoreParams_.blockDim) / CONST_TWO) {
        if (blockNumBS_ == 0) {
            OPS_LOG_I("RopeRegBaseTilingClassAB ComputeUbFactor error, blockNumBS_ == 0");
            return ge::GRAPH_FAILED;
        }
        blockNumN_ = aicoreParams_.blockDim / blockNumBS_;
        blockFactorN_ = CeilDiv(n_, blockNumN_);
        blockNumN_ = CeilDiv(n_, blockFactorN_);
        blockTailN_ = n_ - (blockNumN_ - 1) * blockFactorN_;
    } else {
        blockNumN_ = 1;
        blockFactorN_ = n_;
        blockTailN_ = n_;
    }

    int64_t baseBufferSize = dAlign_ * typeSize;
    int64_t baseBlockInUb =
        FloorAlign(static_cast<int64_t>(aicoreParams_.ubSize / CONST_TWO / DB_FLAG), blockSize_) / baseBufferSize;
    OPS_ERR_IF(baseBlockInUb < 1, OPS_LOG_I(context_->GetNodeName(), "ubSize can't load 8 d size, d = %ld.", d_),
                return ge::GRAPH_FAILED);

    ubFactorN_ = std::min(blockFactorN_, baseBlockInUb - 1);
    ubFactorN_ = std::min(ubFactorN_, MAX_COPY_BLOCK_COUNT / dSplitCoef_);

    ubFactorBS_ = std::min(FloorDiv(baseBlockInUb, ubFactorN_ + 1), blockFactorBS_);
    ubFactorBS_ = (ubFactorBS_ == 0) ? 1 : ubFactorBS_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus RopeRegBaseTilingClassAB::PostTiling()
{
    tilingData_.set_B(b_);
    tilingData_.set_CosB(cosb_);
    tilingData_.set_S(s_);
    tilingData_.set_D(d_);
    tilingData_.set_N(n_);
    tilingData_.set_dAlign(dAlign_);
    tilingData_.set_dSplitCoef(dSplitCoef_);
    tilingData_.set_blockNumBS(blockNumBS_);
    tilingData_.set_blockFactorBS(blockFactorBS_);
    tilingData_.set_blockTailBS(blockTailBS_);
    tilingData_.set_blockNumN(blockNumN_);
    tilingData_.set_blockFactorN(blockFactorN_);
    tilingData_.set_blockTailN(blockTailN_);
    tilingData_.set_ubFactorBS(ubFactorBS_);
    tilingData_.set_ubFactorN(ubFactorN_);
    tilingData_.set_rotaryMode(static_cast<int64_t>(rotaryMode_));
    tilingData_.set_sliceStart(static_cast<int64_t>(sliceStart_));
    tilingData_.set_sliceEnd(static_cast<int64_t>(sliceEnd_));
    tilingData_.set_sliceLength(static_cast<int64_t>(sliceLength_));

    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(blockNumBS_ * blockNumN_);
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = RESERVERD_WORKSPACE_SIZE;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    OPS_LOG_I(context_->GetNodeName(),
            "RopeRegBaseTilingClassAB tilingData is B: %ld, CosB: %ld, S: %ld, D: %ld, N: %ld, kAlign: %ld, "
            "dSplitCoef: %ld, BlockNumBS: %ld, BlockFactorBS: %ld, BlockTailBS: %ld, BlockNumN: %ld, "
            "BlockFactorN: %ld, BlockTailN: %ld, UBFactorBS: %ld, UBFactorN: %ld, RotaryMode: %ld, TilingKey: %ld, sliceStart is %ld, sliceEnd is %ld, sliceLength is %ld",
            tilingData_.get_B(), tilingData_.get_CosB(), tilingData_.get_S(), tilingData_.get_D(), tilingData_.get_N(),
            tilingData_.get_dAlign(), tilingData_.get_dSplitCoef(), tilingData_.get_blockNumBS(),
            tilingData_.get_blockFactorBS(), tilingData_.get_blockTailBS(), tilingData_.get_blockNumN(),
            tilingData_.get_blockFactorN(), tilingData_.get_blockTailN(), tilingData_.get_ubFactorBS(),
            tilingData_.get_ubFactorN(), tilingData_.get_rotaryMode(), GetTilingKey(), tilingData_.get_sliceStart(), tilingData_.get_sliceEnd(), tilingData_.get_sliceLength());

    return ge::GRAPH_SUCCESS;
}

uint64_t RopeRegBaseTilingClassAB::GetTilingKey() const
{
    return TILING_KEY_AB;
}

bool RopeRegBaseTilingClassAB::IsCapable()
{
    if (!IsRegbaseSocVersion()) {
        return false;
    }

    OPS_LOG_I(context_->GetNodeName(), "layout: %ld", static_cast<int64_t>(layout_));
    // 1. qk:bsnd, cos:bs1d 2. qk:sbnd, cos:sb1d 3. qk:sbnd, cos:s11d
    return layout_ == RopeLayout::SBND;
}

// REGISTER_OPS_TILING_TEMPLATE(InplacePartialRotaryMul, RopeRegBaseTilingClassAB, 25000);

} // namespace optiling