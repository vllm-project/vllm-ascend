/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef DISPATCH_GMM_COMBINE_DECODE_BF16_FP16_H
#define DISPATCH_GMM_COMBINE_DECODE_BF16_FP16_H

#include "lib/matmul_intf.h"
#include <kernel_operator.h>

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "dispatch_gmm_combine_decode/gemm/kernel/grouped_matmul_slice_m_multistage_workspace_bf16_fp16.h"
#include "catlass/gemm/gemm_type.hpp"
#include "dispatch_gmm_combine_decode/epilogue/dispatch_policy.h"
#include "dispatch_gmm_combine_decode/gemm/dispatch_policy.h"
#include "dispatch_gmm_combine_decode/epilogue/block/block_epilogue.h"
#include "dispatch_gmm_combine_decode/gemm/block/block_mmad.h"
#include "dispatch_gmm_combine_decode/gemm/kernel/grouped_matmul_slice_m_swiglu_multistage_workspace_bf16_fp16.h"

#include "dispatch_gmm_combine_decode/raw_distributed/cam_moe_distribute_dispatch.h"

#include "dispatch_gmm_combine_decode_tiling.h"
#include "dispatch_gmm_combine_decode_base.h"

using namespace Catlass;

namespace DispatchGmmCombineDecodeBf16Fp16Impl {

using MmadAtlasA2Custom =
    Gemm::MmadAtlasA2PreloadAsyncWithCallback<CUSTOM_PRELOAD_STAGES, CUSTOM_L1_STAGES, CUSTOM_L0A_STAGES,
                                              CUSTOM_L0B_STAGES, CUSTOM_L0C_STAGES, CUSTOM_ENABLE_UNIT_FLAG,
                                              CUSTOM_ENABLE_SHUFFLE_K>;

using Gmm1L1TileShape = GemmShape<FP16_BF16_L1M, FP16_BF16_L1N, GMM1_L1K>;
using Gmm1L0TileShape = GemmShape<Gmm1L1TileShape::M, Gmm1L1TileShape::N, GMM1_L0K>;
using Gmm1EpilogueTileShape = MatrixShape<GMM1_EPIM, Gmm1L1TileShape::N>;
using Gmm1BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<GMM1_SWIZZLE_OFFSET, GMM1_SWIZZLE_DIRECTION>;

using Gmm2L1TileShape = GemmShape<FP16_BF16_GMM2_L1M, FP16_BF16_GMM2_L1N, GMM2_L1K>;
using Gmm2L0TileShape = GemmShape<Gmm2L1TileShape::M, Gmm2L1TileShape::N, GMM2_L0K>;
using Gmm2EpilogueTileShape = MatrixShape<GMM2_EPIM, Gmm2L1TileShape::N>;
using Gmm2BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<GMM2_SWIZZLE_OFFSET, GMM2_SWIZZLE_DIRECTION>;
using Gmm2DispatchPolicy =
    Gemm::MmadAtlasA2PreloadAsyncWithCallbackResidentA<CUSTOM_PRELOAD_STAGES, GMM2_L1A_STAGES, GMM2_L1B_STAGES,
                                                       GMM2_L0A_STAGES, GMM2_L0B_STAGES, CUSTOM_L0C_STAGES,
                                                       CUSTOM_ENABLE_UNIT_FLAG, CUSTOM_ENABLE_SHUFFLE_K>;

template <TemplateMC2TypeClass, class L1TileShape_, class L0TileShape_, class EpilogueTileShape_,
          class BlockScheduler_, class DispatchPolicy_ = MmadAtlasA2Custom>
CATLASS_DEVICE void GmmDeqSwigluQuant(GemmCoord problemShape, uint32_t groupCount, GM_ADDR gmGroupList, GM_ADDR gmA,
                                  layout::RowMajor layoutA, GM_ADDR gmB,
                                  typename std::conditional<(EXEC_FLAG & EXEC_FLAG_ND_FORMAT) != 0, layout::RowMajor, layout::zN>::type layoutB,
                                  GM_ADDR gmScale,
                                  layout::VectorLayout layoutScale, GM_ADDR gmPerTokenScale,
                                  layout::VectorLayout layoutPerTokenScale, GM_ADDR gmD, layout::RowMajor layoutD,
                                  GM_ADDR gmDequantScale, layout::VectorLayout layoutDequantScale, GM_ADDR gmWorkspace,
                                  GM_ADDR gmX, GM_ADDR debugGm, GM_ADDR gmexpertIds, GM_ADDR gmExpandIdx,
                                  GM_ADDR gmEpSendCount, GM_ADDR xActiveMask, GM_ADDR gmResvered, GM_ADDR gmExpertTokenNums,
                                  uint32_t epRankSize, uint32_t epRankId, uint32_t moeExpertNum,
                                  uint32_t moeExpertNumPerRank, uint32_t sharedExpertNum, uint32_t sharedExpertRankNum,
                                  uint32_t quantMode, uint32_t globalBs, uint32_t bs, uint32_t topK, uint32_t tokenLen)
{
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = DispatchPolicy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;

    using AType = Gemm::GemmType<ExpandXType, layout::RowMajor>;
    using LayoutB = typename std::conditional<(EXEC_FLAG & EXEC_FLAG_ND_FORMAT) != 0, layout::RowMajor, layout::zN>::type;
    using BType = Gemm::GemmType<WType, LayoutB>;
    using CType = Gemm::GemmType<float, layout::RowMajor>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    constexpr uint32_t ubStages = 1;
    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2Swiglu<ubStages, 0>;
    using ScaleType = Gemm::GemmType<W1ScaleType, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using DType = Gemm::GemmType<float, layout::RowMajor>;

    using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

    using EpilogueTileShape = EpilogueTileShape_;
    using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
    using TileBroadcastOneBlk =
        Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul =
        Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    using TileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
    using TileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType, ScaleType, PerTokenScaleType,
                                                         DType, TileRowBroadcastMul, TileBroadcastOneBlk,
                                                         TileOneBlkColumnBroadcastMul, TileCopy, TileScheduler>;

    using BlockScheduler = BlockScheduler_;

    // kernel level
    using ElementGroupList = int64_t;

    using GemmKernel = typename std::conditional<
        (EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) != 0,
        Gemm::Kernel::GroupedMatmulSliceMSwigluMultiStageWorkspace<
            TemplateMC2TypeFunc, BlockMmad, BlockEpilogue, BlockScheduler, WORKSPACE_STAGES, ElementGroupList>,
        Gemm::Kernel::GroupedMatmulSliceMSwigluMultiStageWorkspaceWithShallowDispatch<
            TemplateMC2TypeFunc, BlockMmad, BlockEpilogue, BlockScheduler, WORKSPACE_STAGES, ElementGroupList>>::type;

    if constexpr ((EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) != 0) {
        typename GemmKernel::Params params{problemShape,
                                           groupCount,
                                           gmGroupList,
                                           gmA,
                                           layoutA,
                                           gmB,
                                           layoutB,
                                           gmScale,
                                           layoutScale,
                                           gmPerTokenScale,
                                           layoutPerTokenScale,
                                           gmD,
                                           layoutD,
                                           gmDequantScale,
                                           layoutDequantScale,
                                           gmWorkspace,
                                           gmX,
                                           debugGm,
                                           gmexpertIds,
                                           gmExpandIdx,
                                           gmEpSendCount,
                                           xActiveMask,
                                           gmResvered,
                                           gmExpertTokenNums,
                                           epRankSize,
                                           epRankId,
                                           moeExpertNum,
                                           moeExpertNumPerRank,
                                           sharedExpertNum,
                                           sharedExpertRankNum,
                                           quantMode,
                                           globalBs,
                                           bs,
                                           topK,
                                           tokenLen};
        // call a kernel
        GemmKernel gemm;
        gemm(params);
    } else {
        typename GemmKernel::Params params{problemShape,
                                           groupCount,
                                           gmGroupList,
                                           gmA,
                                           layoutA,
                                           gmB,
                                           layoutB,
                                           gmScale,
                                           layoutScale,
                                           gmPerTokenScale,
                                           layoutPerTokenScale,
                                           gmD,
                                           layoutD,
                                           gmDequantScale,
                                           layoutDequantScale,
                                           gmWorkspace};
        // call a kernel
        GemmKernel gemm;
        gemm(params);
    }
}

template <TemplateMC2TypeClass, class L1TileShape_, class L0TileShape_, class EpilogueTileShape_, class BlockScheduler_,
          class DispatchPolicy_ = MmadAtlasA2Custom>
CATLASS_DEVICE void GmmDeq(GemmCoord problemShape, uint32_t groupCount, GM_ADDR gmGroupList, GM_ADDR gmA,
                       layout::RowMajor layoutA, GM_ADDR gmB,
                       typename std::conditional<(EXEC_FLAG & EXEC_FLAG_ND_FORMAT) != 0, layout::RowMajor, layout::zN>::type layoutB,
                       GM_ADDR gmScale,
                       layout::VectorLayout layoutScale, GM_ADDR gmPerTokenScale,
                       layout::VectorLayout layoutPerTokenScale, GM_ADDR gmD, layout::RowMajor layoutD,
                       GM_ADDR gmWorkspace, void *combiner)
{
    using ArchTag = Arch::AtlasA2;
    using DispatchPolicy = DispatchPolicy_;
    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;

    using AType = Gemm::GemmType<ExpandXType, layout::RowMajor>;
    using LayoutB = typename std::conditional<(EXEC_FLAG & EXEC_FLAG_ND_FORMAT) != 0, layout::RowMajor, layout::zN>::type;
    using BType = Gemm::GemmType<WType, LayoutB>;
    using CType = Gemm::GemmType<float, layout::RowMajor>;

    using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    constexpr uint32_t ubStages = 1;
    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2Combine<ubStages, EXEC_FLAG>;
    using ScaleType = Gemm::GemmType<W2ScaleType, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using DType = Gemm::GemmType<ExpandXType, layout::RowMajor>;

    using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

    using EpilogueTileShape = EpilogueTileShape_;
    using TileRowBroadcastMul = Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
    using TileBroadcastOneBlk =
        Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul =
        Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    using TileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
    using TileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType, ScaleType, PerTokenScaleType,
                                                         DType, TileRowBroadcastMul, TileBroadcastOneBlk,
                                                         TileOneBlkColumnBroadcastMul, TileCopy, TileScheduler>;

    using BlockScheduler = BlockScheduler_;

    // kernel level
    using ElementGroupList = int64_t;
    using GemmKernel = Gemm::Kernel::GroupedMatmulSliceMMultiStageWorkspace<
        TemplateMC2TypeFunc, BlockMmad, BlockEpilogue, BlockScheduler, WORKSPACE_STAGES, ElementGroupList>;

    typename GemmKernel::Params params{
        problemShape, groupCount,      gmGroupList,         gmA, layoutA, gmB,         layoutB, gmScale,
        layoutScale,  gmPerTokenScale, layoutPerTokenScale, gmD, layoutD, gmWorkspace, combiner};

    // call a kernel
    GemmKernel gemm;
    gemm(params);
}

template <TemplateMC2TypeClass>
class DispatchGmmCombineDecodeBf16Fp16
{
public:
    __aicore__ inline DispatchGmmCombineDecodeBf16Fp16(){};
    __aicore__ inline void Init(
        // input
        GM_ADDR x, GM_ADDR expert_ids, GM_ADDR gmm1_permuted_weight, GM_ADDR gmm1_permuted_weight_scale,
        GM_ADDR gmm2_weight, GM_ADDR gmm2_weight_scale, GM_ADDR expert_scales, GM_ADDR expert_smooth_scales, GM_ADDR x_active_mask,
        // output
        GM_ADDR output, GM_ADDR expertTokenNums,
        // system
        GM_ADDR workspaceGM, AscendC::TPipe *pipe, const DispatchGmmCombineDecodeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    GM_ADDR gmX_;
    GM_ADDR gmexpertIds_;
    GM_ADDR gmPermuteWeight1_;
    GM_ADDR gmPermuteScale1_;
    GM_ADDR gmWeight2_;
    GM_ADDR gmScale2_;
    GM_ADDR gmOutput_;
    GM_ADDR gmExpertTokenNums_;
    GM_ADDR workspaceGM_;
    GM_ADDR gmSmoothScales_;
    GM_ADDR gmexpertScales_;
    GM_ADDR xActiveMask_;

    uint32_t maxTokenNum_{0};
    uint32_t gmm1OutputDim_{0};
    uint32_t tokenHiddenSize_{0};
    uint32_t groupCount_{0};
    uint32_t gmm2OutputDim_{0};
    uint32_t gmm2InputDim_{0};
    uint32_t globalRankId_{0};
    uint32_t winSizePerRank_{0};
    uint32_t blockDim_{0};
    uint32_t epRankSize_{0};
    uint32_t epRankId_{0};
    uint32_t moeExpertNum_{0};
    uint32_t moeExpertNumPerRank_{0};
    uint32_t sharedExpertNum_{0};
    uint32_t sharedExpertRankNum_{0};
    uint32_t quantMode_{0};
    uint32_t globalBs_{0};
    uint32_t bs_{0};
    uint32_t maxBs_{0};
    uint32_t topK_{0};

    AscendC::TPipe *tpipe_{nullptr};
    __gm__ HcclOpResParam *winContext_{nullptr};
    const DispatchGmmCombineDecodeTilingData *tilingData_;
};

template <TemplateMC2TypeClass>
__aicore__ inline void DispatchGmmCombineDecodeBf16Fp16<TemplateMC2TypeFunc>::Init(
    // input
    GM_ADDR x, GM_ADDR expert_ids, GM_ADDR gmm1_permuted_weight, GM_ADDR gmm1_permuted_weight_scale,
    GM_ADDR gmm2_weight, GM_ADDR gmm2_weight_scale, GM_ADDR expert_scales, GM_ADDR expert_smooth_scales,
    GM_ADDR x_active_mask,
    // output
    GM_ADDR output, GM_ADDR expertTokenNums,
    // system
    GM_ADDR workspaceGM, AscendC::TPipe *pipe, const DispatchGmmCombineDecodeTilingData *tilingData)
{
    tpipe_ = pipe;
    blockDim_ = AscendC::GetBlockNum();
    winContext_ = (__gm__ HcclOpResParam *)AscendC::GetHcclContext<AscendC::HCCL_GROUP_ID_0>();

    gmSmoothScales_ = expert_smooth_scales;  // not used now
    gmX_ = x;                                // input token
    gmexpertIds_ = expert_ids;
    gmPermuteWeight1_ = gmm1_permuted_weight;
    gmPermuteScale1_ = nullptr;
    gmWeight2_ = gmm2_weight;
    gmScale2_ = nullptr;
    gmOutput_ = output;
    gmExpertTokenNums_ = expertTokenNums;
    workspaceGM_ = workspaceGM;
    gmexpertScales_ = expert_scales;
    xActiveMask_ = x_active_mask;
    tilingData_ = tilingData;
    epRankSize_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.epRankSize;
    epRankId_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.epRankId;
    moeExpertNum_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNum;
    moeExpertNumPerRank_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNumPerRank;
    sharedExpertNum_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.sharedExpertNum;
    sharedExpertRankNum_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.sharedExpertRankNum;
    quantMode_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.quantMode;
    globalBs_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.globalBs;
    bs_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.bs;
    topK_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.k;
    maxBs_ = globalBs_ / epRankSize_;

    bool isShareExpert = (epRankId_ < sharedExpertRankNum_);
    if (isShareExpert) {
        maxTokenNum_ = maxBs_ * epRankSize_ / sharedExpertRankNum_;
    } else {
        maxTokenNum_ = maxBs_ * epRankSize_ * (topK_ < moeExpertNumPerRank_ ? topK_ : moeExpertNumPerRank_);
    }

    gmm1OutputDim_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.gmm1HLen;
    tokenHiddenSize_ = tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.h;
    groupCount_ = isShareExpert ? 1 : tilingData->disGmmDeqSwigluQuantGmmDeqComInfo.moeExpertNumPerRank;
    gmm2OutputDim_ = tokenHiddenSize_;
    gmm2InputDim_ = gmm1OutputDim_ / 2;
}

template<uint32_t EXEC_FLAG, typename WType>
__aicore__ inline auto CreateWeightLayout(uint32_t k, uint32_t n) {
    if constexpr ((EXEC_FLAG & EXEC_FLAG_ND_FORMAT) != 0) {
        MatrixCoord mc{k, n};
        return layout::RowMajor::template MakeLayoutInUb<WType>(mc);
    } else {
        return layout::zN::template MakeLayout<WType>(k, n);
    }
}

template <TemplateMC2TypeClass>
__aicore__ inline void DispatchGmmCombineDecodeBf16Fp16<TemplateMC2TypeFunc>::Process()
{
    using LayoutB = typename std::conditional<(EXEC_FLAG & EXEC_FLAG_ND_FORMAT) != 0, layout::RowMajor, layout::zN>::type;
    GemmCoord gmm1ProblemShape{maxTokenNum_, gmm1OutputDim_, tokenHiddenSize_};
    GemmCoord gmm2ProblemShape{maxTokenNum_, gmm2OutputDim_, gmm2InputDim_};

    layout::RowMajor layoutX1{maxTokenNum_, tokenHiddenSize_};
    auto layoutWeight1 = CreateWeightLayout<EXEC_FLAG, WType>(tokenHiddenSize_, gmm1OutputDim_);
    layout::VectorLayout layoutW1Scale{gmm1OutputDim_};
    layout::VectorLayout layoutX1Scale{maxTokenNum_};
    layout::RowMajor layoutX2{maxTokenNum_, gmm2InputDim_};
    auto layoutWeight2 = CreateWeightLayout<EXEC_FLAG, WType>(gmm2InputDim_, gmm2OutputDim_);
    layout::VectorLayout layoutW2Scale{gmm2OutputDim_};
    layout::VectorLayout layoutX2Scale{maxTokenNum_};
    layout::RowMajor layoutOutput{maxTokenNum_, gmm2OutputDim_};

    size_t workspaceOffset = 0;
    constexpr int32_t resveredWorkSpaceSize = 256 * 1024;
    int64_t x1TokenSize = maxTokenNum_ * tokenHiddenSize_ * sizeof(ExpandXType);
    int64_t x2TokenSize = maxTokenNum_ * gmm2InputDim_ * sizeof(ExpandXType);
    int64_t maxTokenSize = x1TokenSize < x2TokenSize ? x2TokenSize : x1TokenSize;
    GM_ADDR gmX1 = workspaceGM_ + workspaceOffset;
    GM_ADDR gmX2 = workspaceGM_ + workspaceOffset;
    workspaceOffset += RoundUp<GM_ALIGN_BYTE>(maxTokenSize);
    GM_ADDR gmX1Scale = nullptr;
    GM_ADDR gmX2Scale = nullptr;
    GM_ADDR gmWorkspace = workspaceGM_ + workspaceOffset;
    GM_ADDR gmCVSwap = workspaceGM_ + workspaceOffset;
    workspaceOffset += RoundUp<GM_ALIGN_BYTE>(static_cast<size_t>(blockDim_) * (FP16_BF16_L1M * FP16_BF16_L1N) *
                                              WORKSPACE_STAGES * sizeof(float));
    int64_t swigluOutSize = maxTokenNum_ * gmm1OutputDim_ * sizeof(float);
    int64_t gmm2OutSize = maxTokenNum_ * tokenHiddenSize_ * sizeof(ExpandXType);
    int64_t maxSwigluGmm2Size = swigluOutSize < gmm2OutSize ? gmm2OutSize : swigluOutSize;
    GM_ADDR gmSwigluOut = workspaceGM_ + workspaceOffset;
    GM_ADDR gmGmm2DepOut = workspaceGM_ + workspaceOffset;
    workspaceOffset += RoundUp<GM_ALIGN_BYTE>(maxSwigluGmm2Size);
    GM_ADDR gmGroupList = workspaceGM_ + workspaceOffset;
    workspaceOffset += RoundUp<GM_ALIGN_BYTE>(static_cast<size_t>(groupCount_) * sizeof(int64_t));
    GM_ADDR gmExpandIdx = workspaceGM_ + workspaceOffset;
    workspaceOffset += RoundUp<GM_ALIGN_BYTE>(static_cast<size_t>(bs_) * topK_ * sizeof(int32_t));
    GM_ADDR gmEpSendCount = workspaceGM_ + workspaceOffset;
    workspaceOffset += RoundUp<GM_ALIGN_BYTE>(static_cast<size_t>(epRankSize_) * groupCount_ * sizeof(int32_t));
    GM_ADDR gmResvered = workspaceGM_ + workspaceOffset;
    workspaceOffset += RoundUp<GM_ALIGN_BYTE>(resveredWorkSpaceSize);

    if constexpr ((EXEC_FLAG & EXEC_FLAG_DEEP_FUSE) == 0) {
        if constexpr (g_coreType == AscendC::AIV) {
            AscendC::TPipe tpipe;
            MoeDistributeDispatchImpl::CamMoeDistributeDispatch<ExpandXType, ExpandXType, false, false, false, false, EXEC_FLAG>
                dispatcher;
            dispatcher.Init(gmX_, gmexpertIds_, gmSmoothScales_, xActiveMask_, gmX1, gmX1Scale, gmExpandIdx, gmGroupList,
                            gmEpSendCount, gmExpertTokenNums_, nullptr, gmWorkspace, &tpipe, tilingData_);
            dispatcher.Process();
            tpipe.Destroy();
            icache_preload(8);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        Arch::CrossCoreFlag gmm1AivFinished{0};
        if constexpr (g_coreType == AscendC::AIV) {
            Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(gmm1AivFinished);
        } else {
            Arch::CrossCoreWaitFlag(gmm1AivFinished);
        }
    }
    GmmDeqSwigluQuant<TemplateMC2TypeFunc, Gmm1L1TileShape, Gmm1L0TileShape, Gmm1EpilogueTileShape,
                      Gmm1BlockScheduler>(
        gmm1ProblemShape, groupCount_, gmGroupList, gmX1, layoutX1, gmPermuteWeight1_, layoutWeight1,
        gmPermuteScale1_, layoutW1Scale, gmX1Scale, layoutX1Scale, gmX2, layoutX2, gmX2Scale,
        layoutX2Scale, gmWorkspace, gmX_, gmSmoothScales_, gmexpertIds_, gmExpandIdx, gmEpSendCount, xActiveMask_, gmResvered,
        gmExpertTokenNums_, epRankSize_, epRankId_, moeExpertNum_, moeExpertNumPerRank_, sharedExpertNum_,
        sharedExpertRankNum_, quantMode_, globalBs_, bs_, topK_, tokenHiddenSize_);
    AscendC::PipeBarrier<PIPE_ALL>();
    Arch::CrossCoreFlag gmm1AivFinished{0};
    if constexpr (g_coreType == AscendC::AIV) {
        Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(gmm1AivFinished);
    } else {
        Arch::CrossCoreWaitFlag(gmm1AivFinished);
    }

    MoeDistributeCombineImpl::CamMoeDistributeCombine<TemplateMC2TypeFunc> combiner;
    if (g_coreType == AscendC::AIV) {
        combiner.Init(gmGmm2DepOut, gmexpertIds_, gmExpandIdx, gmEpSendCount, nullptr, gmexpertScales_, xActiveMask_, gmOutput_,
                      workspaceGM_, nullptr, tilingData_);
    }
    GmmDeq<TemplateMC2TypeFunc, Gmm2L1TileShape, Gmm2L0TileShape, Gmm2EpilogueTileShape, Gmm2BlockScheduler,
           Gmm2DispatchPolicy>(gmm2ProblemShape, groupCount_, gmGroupList, gmX2, layoutX2, gmWeight2_, layoutWeight2,
                               gmScale2_, layoutW2Scale, gmX2Scale, layoutX2Scale, gmGmm2DepOut,
                               layoutOutput, gmWorkspace, &combiner);
}
} // namespace DispatchGmmCombineDecodeBf16Fp16Impl
#endif  // DISPATCH_GMM_COMBINE_DECODE_BF16_FP16_H
