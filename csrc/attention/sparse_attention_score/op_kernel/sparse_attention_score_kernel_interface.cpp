/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_attention_score_kernel_interface.cpp
 * \brief Sparse Attention Score Kernel Interface
 */
#include "kernel_operator.h"
#if (__CCE_AICORE__ == 220)
#include "arch22/sparse_attention_score_kernel_arch22.h"
#endif
#if (__CCE_AICORE__ == 310)
#include "arch35/sparse_attention_score_kernel_arch35.h"
#include "arch35/sparse_attention_score_kernel_arch35_full_quant.h"
#endif

using namespace NpuArch;

#if (__CCE_AICORE__ == 220)

using namespace SasaKernelArch22;

template <class InDtype, class SMDtype>
__global__ __aicore__ void SasaInferIntfRegularArch22(
    GM_ADDR q, GM_ADDR k, GM_ADDR v,
    GM_ADDR selectIdx, GM_ADDR blockTable, GM_ADDR selectNumIdx,
    GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
    GM_ADDR o, GM_ADDR softmaxLse,
    GM_ADDR workspace, GM_ADDR tiling)
{
    using ArchTag = Arch::AtlasA2;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    using ElementP = InDtype;
    using ElementO = InDtype;
    using ElementOTmp = SMDtype;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;

    // QK matmul
    using L1TileShapeQK = GemmShape<128, 128, 128>;
    using L0TileShapeQK = GemmShape<128, 128, 128>;
    using DispatchPolicyQK = Gemm::MmadAtlasA2SFAIQK<false, false>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<
        DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK,
        QType, KType, SType>;

    // Online softmax
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax<Epilogue::LseMode::NONE, SMDtype>;
    using MaskType = Gemm::GemmType<int8_t, layout::RowMajor>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<
        DispatchPolicyOnlineSoftmax, PType, SType, MaskType>;

    // PV matmul
    using L1TileShapePV = GemmShape<128, 128, 256>;
    using L0TileShapePV = GemmShape<128, 128, 128>;
    using DispatchPolicyPV = Gemm::MmadAtlasA2SFAIPV<false, false>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<
        DispatchPolicyPV, L1TileShapePV, L0TileShapePV,
        PType, VType, OTmpType>;

    // Rescale O
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleO<Epilogue::LseMode::NONE, SMDtype>;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OTmpUpdateType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using LseType = Gemm::GemmType<float, layout::RowMajor>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<
        DispatchPolicyRescaleO, OType, OTmpType, OTmpUpdateType, LseType>;

    using SasaKernel = SasaRegularKernelArch22<
        BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO>;

    SasaKernelParamsArch22 params{q, k, v, selectIdx, blockTable, selectNumIdx,
        actualQseqlen, actualKvseqlen, o, softmaxLse, workspace, tiling};
    SasaKernel sasaKernel;
    sasaKernel(params);
}

#endif
#if (__CCE_AICORE__ == 310)

using namespace SasaKernelArch35;

template <class InDtype, class SMDtype, class REDtype, Format qFormat>
__global__ __aicore__ void SasaInferIntfRegular(
    GM_ADDR q, GM_ADDR k, GM_ADDR v,
    GM_ADDR selectIdx, GM_ADDR blockTable, GM_ADDR selectNumIdx,
    GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
    GM_ADDR o, GM_ADDR softmaxLse,
    GM_ADDR workspace, GM_ADDR tiling)
{
    using ArchTag = Arch::AtlasA5;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    using ElementP = InDtype;
    using ElementO = InDtype;
    using ElementOTmp = REDtype;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutPDummy = layout::zN;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;

    // QK matmul
    using L1TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK = Gemm::MmadAtlasA5BsaQK;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS, LayoutS,
        void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<
        DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK,
        ElementQ, ElementK, ElementS, void, TileCopyQK>;

    // Online softmax
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueOnlineSoftmaxBsa;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<
        DispatchPolicyOnlineSoftmax, PType, SType>;

    // PV matmul
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyPV = Gemm::MmadAtlasA5BsaPV;
    using TileCopyPV = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
        void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<
        DispatchPolicyPV, L1TileShapePV, L0TileShapePV,
        ElementP, ElementV, ElementOTmp, void, TileCopyPV>;

    // Rescale O
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA5BsaRescaleO;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<
        ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<
        DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS, TileCopyRescaleO, Arch::PositionL0C>;

    using SasaKernel = SasaRegularKernelArch35<
        BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO, qFormat, qFormat>;

    SasaKernelParamsArch35 params{q, k, v, selectIdx, blockTable, selectNumIdx,
        actualQseqlen, actualKvseqlen, o, softmaxLse, workspace, tiling};
    SasaKernel sasaKernel;
    sasaKernel(params);
}

template <class InDtype, class SMDtype, class REDtype, Format qFormat>
__global__ __aicore__ void SasaInferInterfaceFullQuant(
    GM_ADDR q, GM_ADDR k, GM_ADDR v,
    GM_ADDR selectIdx, GM_ADDR blockTable, GM_ADDR selectNumIdx,
    GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
    GM_ADDR qDequantScale, GM_ADDR kDequantScale, GM_ADDR vDequantScale,
    GM_ADDR o, GM_ADDR softmaxLse,
    GM_ADDR workspace, GM_ADDR tiling)
{
    using ArchTag = Arch::AtlasA5;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    using ElementP = InDtype;
    using ElementO = SMDtype;
    using ElementOTmp = REDtype;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutPDummy = layout::zN;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;

    using L1TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK = Gemm::MmadAtlasA5BsaQK;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS, LayoutS,
        void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT, false,
        Gemm::Tile::ScaleGranularity::PER_TENSOR>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<
        DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK,
        ElementQ, ElementK, ElementS, void, TileCopyQK>;

    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueOnlineSoftmaxBsa;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<
        DispatchPolicyOnlineSoftmax, PType, SType>;

    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyPV = Gemm::MmadAtlasA5BsaPV;
    using TileCopyPV = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
        void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT, false,
        Gemm::Tile::ScaleGranularity::PER_TENSOR>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<
        DispatchPolicyPV, L1TileShapePV, L0TileShapePV,
        ElementP, ElementV, ElementOTmp, void, TileCopyPV>;

    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA5BsaRescaleO;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<
        ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<
        DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS, TileCopyRescaleO, Arch::PositionL0C>;

    using SasaKernel = SasaFullQuantKernelArch35<
        BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO, qFormat, qFormat>;

    SasaFullQuantKernelParamsArch35 params{
        q, k, v, selectIdx, blockTable, selectNumIdx,
        actualQseqlen, actualKvseqlen,
        qDequantScale, kDequantScale, vDequantScale,
        o, softmaxLse, workspace, tiling};
    SasaKernel sasaKernel;
    sasaKernel(params);
}
#endif
