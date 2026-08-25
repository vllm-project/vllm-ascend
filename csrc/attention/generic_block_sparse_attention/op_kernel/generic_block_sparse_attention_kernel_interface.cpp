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
 * \file generic_block_sparse_attention_kernel_interface.cpp
 * \brief Generic Sparse Attention Kernel Interface
 */
#include "kernel_operator.h"
#if (__CCE_AICORE__ == 220)
#include "arch22/generic_block_sparse_attention_kernel_arch22.h"
#endif
#if (__CCE_AICORE__ == 310)
#include "arch35/generic_block_sparse_attention_kernel_arch35.h"
#include "arch35/generic_block_sparse_attention_kernel_arch35_full_quant.h"
#endif

using namespace NpuArch;

#if (__CCE_AICORE__ == 220)

using namespace GsaKernelArch22;

template <class InDtype, class SMDtype, class REDtype = SMDtype,
    Epilogue::LseMode lseMode = Epilogue::LseMode::NONE>
__global__ __aicore__ void GsaInferIntfRegularArch22(
    GM_ADDR q, GM_ADDR k, GM_ADDR v,
    GM_ADDR sparseBlockIdx, GM_ADDR sparseBlockCount,
    GM_ADDR metaData,
    GM_ADDR cuSeqLengths, GM_ADDR cuSeqLengthsKv,
    GM_ADDR sequsedQ, GM_ADDR sequsedKv,
    GM_ADDR blockTable,
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
    using ElementOTmp = REDtype;

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
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax<lseMode, SMDtype>;
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

    // Rescale O (REDtype=float → high-prec rescale; Softmax online gm/dm/gl are float)
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleO<lseMode, REDtype>;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OTmpUpdateType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using LseType = Gemm::GemmType<float, layout::RowMajor>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<
        DispatchPolicyRescaleO, OType, OTmpType, OTmpUpdateType, LseType>;

    using GsaKernel = GsaRegularKernelArch22<
        BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO>;

    GsaKernelParamsArch22 params{q, k, v, sparseBlockIdx, sparseBlockCount,
        metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ, sequsedKv, blockTable,
        o, softmaxLse, workspace, tiling};
    GsaKernel gsaKernel;
    gsaKernel(params);
}

#endif
#if (__CCE_AICORE__ == 310)

using namespace GsaKernelArch35;

template <class InDtype, class SMDtype, class REDtype, Format qFormat,
    Epilogue::LseMode lseMode = Epilogue::LseMode::NONE,
    Epilogue::LseFormat lseFormat = Epilogue::LseFormat::TN1>
__global__ __aicore__ void GsaInferIntfRegular(
    GM_ADDR q, GM_ADDR k, GM_ADDR v,
    GM_ADDR sparseBlockIdx, GM_ADDR sparseBlockCount,
    GM_ADDR metaData,
    GM_ADDR cuSeqLengths, GM_ADDR cuSeqLengthsKv,
    GM_ADDR sequsedQ, GM_ADDR sequsedKv,
    GM_ADDR blockTable,
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
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA5BsaRescaleO<lseMode, lseFormat>;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<
        ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<
        DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS, ElementK, TileCopyRescaleO, Arch::PositionL0C>;

    using GsaKernel = GsaRegularKernelArch35<
        BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO, qFormat, qFormat>;

    GsaKernelParamsArch35 params{q, k, v, sparseBlockIdx, sparseBlockCount,
        metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ, sequsedKv, blockTable,
        o, softmaxLse, workspace, tiling};
    GsaKernel gsaKernel;
    gsaKernel(params);
}

template <class InDtype, class SMDtype, class REDtype, Format qFormat,
    Epilogue::LseMode lseMode = Epilogue::LseMode::NONE,
    Epilogue::LseFormat lseFormat = Epilogue::LseFormat::TN1>
__global__ __aicore__ void GsaInferInterfaceFullQuant(
    GM_ADDR q, GM_ADDR k, GM_ADDR v,
    GM_ADDR sparseBlockIdx, GM_ADDR sparseBlockCount,
    GM_ADDR metaData,
    GM_ADDR cuSeqLengths, GM_ADDR cuSeqLengthsKv,
    GM_ADDR sequsedQ, GM_ADDR sequsedKv,
    GM_ADDR blockTable,
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
        void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
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
        void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<
        DispatchPolicyPV, L1TileShapePV, L0TileShapePV,
        ElementP, ElementV, ElementOTmp, void, TileCopyPV>;

    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA5BsaRescaleO<lseMode, lseFormat>;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<
        ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<
        DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS, ElementK, TileCopyRescaleO, Arch::PositionL0C>;

    using GsaKernel = GsaFullQuantKernelArch35<
        BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO, qFormat, qFormat>;

    GsaFullQuantKernelParamsArch35 params{
        q, k, v, sparseBlockIdx, sparseBlockCount,
        metaData, cuSeqLengths, cuSeqLengthsKv, sequsedQ, sequsedKv, blockTable,
        qDequantScale, kDequantScale, vDequantScale,
        o, softmaxLse, workspace, tiling};
    GsaKernel gsaKernel;
    gsaKernel(params);
}
#endif
