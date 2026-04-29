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
 * \file paged_select_attention_kernel_bridge.h
 * \brief Specialized paged_select_attention kernel bridge.
 */

#include "kernel_operator.h"
#include "paged_select_attention_kernel.h"

using namespace NpuArch;

namespace PagedSelectAttentionKernel {
template <typename InputDtype = half, typename IntermCalcPrec = float>
__global__ __aicore__ void Run(
    GM_ADDR q,
    GM_ADDR k,
    GM_ADDR v,
    GM_ADDR blockTables,
    GM_ADDR selectedKvIndices,
    GM_ADDR o,
    GM_ADDR lse,
    GM_ADDR actualQseqlen,
    GM_ADDR actualKvseqlen,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    using ElementQ = InputDtype;
    using LayoutQ = layout::RowMajor;
    using ElementK = InputDtype;
    using LayoutK = layout::ColumnMajor;
    using ElementV = InputDtype;
    using LayoutV = layout::RowMajor;
    using ElementS = IntermCalcPrec;
    using LayoutS = layout::RowMajor;
    using ElementP = InputDtype;
    using LayoutP = layout::RowMajor;
    using ElementO = InputDtype;
    using LayoutO = layout::RowMajor;
    using ElementLse = float;
    using LayoutLse = layout::RowMajor;
    using ElementMask = int8_t;
    using LayoutMask = layout::RowMajor;
    using ElementOTmp = IntermCalcPrec;
    using LayoutOTmp = layout::RowMajor;
    using ElementUpdate = IntermCalcPrec;
    using LayoutUpdate = layout::RowMajor;

    using L1TileShapeQK = GemmShape<Q_TILE_CEIL, 128, 128>;
    using L0TileShapeQK = GemmShape<128, 128, 128>;
    using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQK<true, false>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK =
        Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, QType, KType, SType>;

    using DispatchPolicyOnlineSoftmax =
        Epilogue::EpilogueAtlasA2OnlineSoftmax<Epilogue::LseMode::NONE, Epilogue::SinkMode::DISABLE, IntermCalcPrec>;
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using MaskTensorType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax =
        Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, MaskTensorType>;

    using L1TileShapePV = GemmShape<128, 128, 256>;
    using L0TileShapePV = GemmShape<128, 128, 128>;
    using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPV<true, false>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV =
        Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, PType, VType, OTmpType>;

    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleO<Epilogue::LseMode::NONE, IntermCalcPrec>;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using LseType = Gemm::GemmType<ElementLse, LayoutLse>;
    using EpilogueRescaleO =
        Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType, LseType>;

    using DispatchPolicyInitOutWhenZero = Epilogue::EpilogueAtlasA2InitOutWhenZero<Epilogue::LseMode::NONE>;
    using EpilogueInitOut = Epilogue::Block::BlockEpilogue<DispatchPolicyInitOutWhenZero, OType, LseType>;

    using PagedSelectAttentionInferKernel =
        SplitFuse::FAInferKernel<BlockMmadQK,
                                 BlockMmadPV,
                                 EpilogueOnlineSoftmax,
                                 EpilogueRescaleO,
                                 EpilogueInitOut,
                                 true,
                                 FaiKernel::MaskType::NO_MASK,
                                 FaiKernel::inputLayout::TND>;

    FAIKernelParams params{q,
                           k,
                           v,
                           nullptr,
                           blockTables,
                           selectedKvIndices,
                           actualQseqlen,
                           actualKvseqlen,
                           o,
                           lse,
                           workspace,
                           tiling,
                           nullptr};
    PagedSelectAttentionInferKernel pagedSelectAttention;
    pagedSelectAttention(params);
}
} // namespace PagedSelectAttentionKernel
