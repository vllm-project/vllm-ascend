/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe.h
 * \brief Unified MegaMoe implementation that merges the three legacy paths:
 *          - mega_moe          (INT8 weight)
 *          - mega_moe_bf16     (FP16 / BF16 weight)
 *          - mega_moe_w4_a8    (INT4 weight, W4A8)
 *        Differences from the legacy single-dtype implementations:
 *          - Inputs use the new unified naming (x / topk_ids / topk_weights / weight1
 *            / weight_scales1 / ... ) following the unified op def.
 *          - The first input is `context` (replaces the removed `group` attribute).
 *          - The single legacy `transB` attribute is split into `transpose_weight1`
 *            and `transpose_weight2`, surfaced via two template flags.
 *          - The legacy `M` attribute is renamed to `max_recv_token_num`.
 *          - Tiling fields are read in the order defined in the unified op def.
 */

#ifndef MEGA_MOE_H
#define MEGA_MOE_H

using namespace AscendC;

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "../../common_utils/moe_distribute_base.h"

#include "mega_moe_tiling_a2a3.h"
#include "moe_init_routing_v2/moe_init_routing_v2_tiling.h"

#include "template_linear_algebra_v2/catlass.hpp"
#include "template_linear_algebra_v2/arch/arch.hpp"
#include "template_linear_algebra_v2/epilogue/dispatch_policy.hpp"
#include "template_linear_algebra_v2/epilogue/block/block_epilogue.hpp"
#include "template_linear_algebra_v2/epilogue/tile/tile_copy.hpp"
#include "template_linear_algebra_v2/epilogue/tile/tile_elemwise_add.hpp"
#include "template_linear_algebra_v2/epilogue/tile/tile_elemwise_muls.hpp"
#include "template_linear_algebra_v2/gemm/block/block_mmad.hpp"
#include "template_linear_algebra_v2/gemm/block/block_swizzle.hpp"
#include "template_linear_algebra_v2/gemm/dispatch_policy.hpp"
#include "template_linear_algebra_v2/gemm/kernel/matmul_epilogue.hpp"
#include "template_linear_algebra_v2/gemm/gemm_type.hpp"
#include "template_linear_algebra_v2/layout/layout.hpp"

#include "utils/select_helper.hpp"
#include "utils/const_args.hpp"
#include "utils/dispatch_policy_custom.hpp"
#include "../../common_utils/mc2_moe_context.h"
#include "utils/block_epilogue_pertoken_v2.hpp"
#include "mega_moe_kernel_a3.hpp"
#include "moe_init_routing_quant_v2/moe_init_routing_quant_v2_tiling.h"

namespace MegaMoeImpl {
using namespace AscendC;
using namespace AscendC::HcclContextDef;
using namespace Catlass;
using namespace Mc2Tiling;

// Template parameter pack used by the unified MegaMoe class.
//   AType_ : input X element type (DTYPE_X)
//   BType_ : weight element type   (DTYPE_WEIGHT1)
//   CType_ : output  element type  (DTYPE_Y)
//   TB1_   : transpose_weight1 attribute
//   TB2_   : transpose_weight2 attribute
//   Nz_    : whether weight1/2 is in FRACTAL_NZ format (both are the same)
//   IS_A2_ : whether running on Atlas A2 (910B); false means A3 (910_93)
#define MegaMoeClass typename AType_, typename BType_, typename CType_, bool TB1_, bool TB2_, bool Nz_, bool IS_A2_
#define MegaMoeFunc AType_, BType_, CType_, TB1_, TB2_, Nz_, IS_A2_

template <MegaMoeClass>
class MegaMoe {
public:
    __aicore__ inline MegaMoe(){};
    __aicore__ inline void Init(GM_ADDR contextGM, GM_ADDR xGM, GM_ADDR topkIdsGM, GM_ADDR topkWeightsGM,
                                GM_ADDR weight1GM, GM_ADDR weight2GM, GM_ADDR weightScales1GM, GM_ADDR weightScales2GM,
                                GM_ADDR bias1GM, GM_ADDR bias2GM, GM_ADDR xActiveMaskGM, GM_ADDR scalesGM, GM_ADDR yGM,
                                GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM, GM_ADDR tilingGM);
    __aicore__ inline void Process();

private:
    GM_ADDR contextGM_;
    GM_ADDR xGM_;
    GM_ADDR topkIdsGM_;
    GM_ADDR topkWeightsGM_;
    GM_ADDR weight1GM_;
    GM_ADDR weight2GM_;
    GM_ADDR weightScales1GM_;
    GM_ADDR weightScales2GM_;
    GM_ADDR bias1GM_;
    GM_ADDR bias2GM_;
    GM_ADDR xActiveMaskGM_;
    GM_ADDR scalesGM_;
    GM_ADDR yGM_;
    GM_ADDR gmExpertTokenNums_;
    GM_ADDR workspaceGM_;

    GM_ADDR moeInitRoutingQuantV2Scale = nullptr;
    GM_ADDR moeInitRoutingQuantV2Offset = nullptr;
    GM_ADDR expertTokensBeforeCapacity = nullptr;

    TBuf<AscendC::TPosition::VECCALC> uBuf_;

    int32_t aivNum;

    static constexpr int32_t UB_MOVE_NUM = 16 * 1024;

    int32_t m;
    int32_t k;
    int32_t n;
    int32_t topK;
    int32_t expertPerRank;
    int32_t maxOutputSize;
    int32_t EP;
    int32_t listLen;
    float activationClamp;

    MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData;
    MoeInitRoutingV2TilingData moeInitRoutingV2TilingData;
    uint64_t initRoutingQuantTilingKey;
};


template <MegaMoeClass>
__aicore__ inline void MegaMoe<MegaMoeFunc>::Init(GM_ADDR contextGM, GM_ADDR xGM, GM_ADDR topkIdsGM,
                                                  GM_ADDR topkWeightsGM, GM_ADDR weight1GM, GM_ADDR weight2GM,
                                                  GM_ADDR weightScales1GM, GM_ADDR weightScales2GM, GM_ADDR bias1GM,
                                                  GM_ADDR bias2GM, GM_ADDR xActiveMaskGM, GM_ADDR scalesGM, GM_ADDR yGM,
                                                  GM_ADDR expertTokenNumsGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    constexpr bool kRoutingIsQuant =
        std::is_same_v<BType_, AscendC::int4b_t> || std::is_same_v<BType_, int32_t> || std::is_same_v<BType_, int8_t>;

    contextGM_ = contextGM;
    xGM_ = xGM;
    topkIdsGM_ = topkIdsGM;
    topkWeightsGM_ = topkWeightsGM;
    weight1GM_ = weight1GM;
    weight2GM_ = weight2GM;
    weightScales1GM_ = weightScales1GM;
    weightScales2GM_ = weightScales2GM;
    bias1GM_ = bias1GM;
    bias2GM_ = bias2GM;
    xActiveMaskGM_ = xActiveMaskGM;
    scalesGM_ = scalesGM;
    yGM_ = yGM;
    gmExpertTokenNums_ = expertTokenNumsGM;
    workspaceGM_ = workspaceGM;

    if constexpr (kRoutingIsQuant) {
        GET_TILING_DATA_WITH_STRUCT(MegaMoeTilingDataQuant, tilingData, tilingGM);

        aivNum = tilingData.common.aivNum;
        m = tilingData.common.M;
        k = tilingData.common.K;
        n = tilingData.common.N;
        EP = tilingData.common.worldSize;
        topK = tilingData.common.topK;
        expertPerRank = tilingData.common.expertPerRank;
        maxOutputSize = tilingData.common.maxRecvTokenNum;
        listLen = tilingData.common.listLen;

        activationClamp = tilingData.common.activationClamp;

        moeInitRoutingQuantV2TilingData = tilingData.moeInitRoutingQuantV2TilingData;
        initRoutingQuantTilingKey = tilingData.common.initRoutingQuantTilingKey;
    } else {
        GET_TILING_DATA_WITH_STRUCT(MegaMoeTilingDataNonQuant, tilingData, tilingGM);

        aivNum = tilingData.common.aivNum;
        m = tilingData.common.M;
        k = tilingData.common.K;
        n = tilingData.common.N;
        EP = tilingData.common.worldSize;
        topK = tilingData.common.topK;
        expertPerRank = tilingData.common.expertPerRank;
        maxOutputSize = tilingData.common.maxRecvTokenNum;
        listLen = tilingData.common.listLen;

        activationClamp = tilingData.common.activationClamp;

        moeInitRoutingV2TilingData = tilingData.moeInitRoutingV2TilingData;
        initRoutingQuantTilingKey = tilingData.common.initRoutingQuantTilingKey;
    }
}

template <MegaMoeClass>
__aicore__ inline void MegaMoe<MegaMoeFunc>::Process()
{
    using ArchTag = Arch::AtlasA2;
    constexpr bool enableUnitFlag = false;
    constexpr bool enableShuffleK = true;

    constexpr bool isW4A8 = std::is_same_v<BType_, AscendC::int4b_t> || std::is_same_v<BType_, int32_t>;
    constexpr bool isInt8 = std::is_same_v<BType_, int8_t>;

    using AElement = std::conditional_t<isW4A8, AscendC::int4b_t, std::conditional_t<isInt8, int8_t, AType_>>;
    using BElement = std::conditional_t<isW4A8, AscendC::int4b_t, BType_>;
    using CElement = std::conditional_t<isW4A8 || isInt8, float16_t, CType_>;
    using D1Element = std::conditional_t<isW4A8 || isInt8, int8_t, CType_>;

    uint32_t k2 = n / 2;
    uint32_t n2 = k;

    using LayoutA = layout::RowMajor;
    using LayoutB =
        typename std::conditional<Nz_, layout::zN,
                                  typename std::conditional<TB1_, layout::ColumnMajor, layout::RowMajor>::type>::type;

    LayoutB layoutB1 = LayoutBInitializer<LayoutB, BElement>::create(k, n);
    LayoutB layoutB2 = LayoutBInitializer<LayoutB, BElement>::create(k2, n2);
    using LayoutC = layout::RowMajor;

    using L1TileShape =
        std::conditional_t<isW4A8, GemmShape<128, 256, 1024>,
                           std::conditional_t<isInt8, GemmShape<128, 256, 512>, GemmShape<128, 256, 256>>>;
    using L0TileShape =
        std::conditional_t<isW4A8, GemmShape<128, 256, 256>,
                           std::conditional_t<isInt8, GemmShape<128, 256, 128>, GemmShape<128, 256, 64>>>;

    constexpr uint32_t preloadStages = 1;
    constexpr uint32_t l1Stages = 2;
    constexpr uint32_t l0AStages = 2;
    constexpr uint32_t l0BStages = 2;
    constexpr uint32_t l0CStages = 1;

    using DispatchPolicy = Gemm::MmadDispatchPolicyFor_t<BElement, preloadStages, l1Stages, l0AStages, l0BStages,
                                                         l0CStages, enableUnitFlag, enableShuffleK>;

    using AType = Gemm::GemmType<AElement, layout::RowMajor>;
    using BType = Gemm::GemmType<BElement, LayoutB>;
    using CType = Gemm::GemmType<CElement, layout::RowMajor>;
    using D1Type = Gemm::GemmType<D1Element, layout::RowMajor>;
    using D2Type = std::conditional_t<std::is_same_v<CType_, bfloat16_t>, Gemm::GemmType<bfloat16_t, layout::RowMajor>,
                                      Gemm::GemmType<CType_, layout::RowMajor>>;

    using ScaleGranularity = Catlass::Gemm::Tile::ScaleGranularity;
    using TileCopyMmad = Gemm::Tile::QuantTileCopy<ArchTag, AType, BType, CType, void, ScaleGranularity::PER_CHANNEL>;

    using BlockMmad = std::conditional_t<
        isW4A8,
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, void, TileCopyMmad>,
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>>;

    constexpr uint32_t ubStages = 2;

    using EpilogueDispatchPolicy1 =
        std::conditional_t<isW4A8, Epilogue::EpilogueAtlasA2W4A8PostPerTokenDequantSwigluQuant<ubStages>,
                           Epilogue::EpilogueAtlasA2PerTokenDequantSwigluQuant<ubStages>>;
    using EpilogueDispatchPolicy2 = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
    using EpilogueDispatchPolicy3 =
        std::conditional_t<isW4A8, Epilogue::EpilogueAtlasA2W4A8PostPerTokenDequantV2<ubStages>,
                           Epilogue::EpilogueAtlasA2PerTokenDequantV2<ubStages>>;

    using ScaleType = Gemm::GemmType<uint64_t, layout::VectorLayout>;
    using PerTokenScaleType = Gemm::GemmType<float, layout::VectorLayout>;
    using ElementMulType = Gemm::GemmType<float, layout::RowMajor>;
    using TileElemWiseMuls = Epilogue::Tile::TileElemWiseMuls<ArchTag, ElementMulType, 0>;

    using TileCopy1 = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D1Type>;
    using BlockEpilogue1 = Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy1, CType, PerTokenScaleType, D1Type,
                                                          TileElemWiseMuls, TileCopy1>;

    using TileCopy2 = Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, D2Type>;
    using IS_A2 = Epilogue::Block::IS_A2_T<IS_A2_>;
    using BlockEpilogue2 =
        Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy2, CType, PerTokenScaleType, D2Type, TileCopy2>;
    using BlockEpilogue3 = std::conditional_t<
        isW4A8,
        Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy3, CType, PerTokenScaleType, D2Type, TileCopy2, IS_A2>,
        Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy3, CType, PerTokenScaleType, D2Type, TileCopy2, IS_A2>>;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;
    using ElementGroupList = int64_t;

    using MatmulKernel = Gemm::Kernel::MegaMoeKernel<BlockMmad, BlockScheduler, ElementGroupList, BlockEpilogue1,
                                                     BlockEpilogue2, BlockEpilogue3>;

    LayoutA layoutA1{static_cast<uint32_t>(m), static_cast<uint32_t>(k)};
    LayoutA layoutA2{static_cast<uint32_t>(m), static_cast<uint32_t>(k2)};
    layout::VectorLayout layoutScale1{static_cast<uint32_t>(n)};
    layout::VectorLayout layoutScale2{static_cast<uint32_t>(n2)};
    layout::RowMajor layoutD1{static_cast<uint32_t>(maxOutputSize), static_cast<uint32_t>(k2)};
    layout::RowMajor layoutD2{static_cast<uint32_t>(m * topK), static_cast<uint32_t>(n2)};

    GemmCoord problemShape{static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k)};

    uint32_t epilogueCoreNum;
    uint32_t epilogueGranularity;
    if constexpr (isW4A8) {
        epilogueCoreNum = static_cast<uint32_t>(aivNum);
        epilogueGranularity = 0U;
    } else if constexpr (isInt8) {
        epilogueCoreNum = static_cast<uint32_t>(aivNum);
        epilogueGranularity = (expertPerRank > 4) ? static_cast<uint32_t>(expertPerRank - 3) :
                              (expertPerRank > 0) ? static_cast<uint32_t>(expertPerRank - 1) :
                                                    0U;
    } else {
        epilogueCoreNum = static_cast<uint32_t>(aivNum);
        epilogueGranularity = (expertPerRank > 1) ? static_cast<uint32_t>(expertPerRank - 2) : 0U;
    }

    constexpr bool kRoutingIsQuant =
        std::is_same_v<BType_, AscendC::int4b_t> || std::is_same_v<BType_, int32_t> || std::is_same_v<BType_, int8_t>;

    typename MatmulKernel::Params params;
    if constexpr (kRoutingIsQuant) {
        params = typename MatmulKernel::Params{problemShape,
                                               static_cast<uint32_t>(EP),
                                               static_cast<uint32_t>(listLen),
                                               static_cast<uint32_t>(expertPerRank),
                                               static_cast<uint32_t>(maxOutputSize),
                                               static_cast<uint32_t>(topK),
                                               initRoutingQuantTilingKey,
                                               epilogueCoreNum,
                                               contextGM_,
                                               xGM_,
                                               layoutA1,
                                               layoutA2,
                                               weight1GM_,
                                               layoutB1,
                                               bias1GM_,
                                               weight2GM_,
                                               layoutB2,
                                               bias2GM_,
                                               weightScales1GM_,
                                               layoutScale1,
                                               weightScales2GM_,
                                               layoutScale2,
                                               yGM_,
                                               layoutD1,
                                               layoutD2,
                                               topkIdsGM_,
                                               moeInitRoutingQuantV2Scale,
                                               moeInitRoutingQuantV2Offset,
                                               expertTokensBeforeCapacity,
                                               topkWeightsGM_,
                                               workspaceGM_,
                                               gmExpertTokenNums_,
                                               xActiveMaskGM_,
                                               scalesGM_,
                                               moeInitRoutingQuantV2TilingData,
                                               epilogueGranularity,
                                               activationClamp};
    } else {
        params = typename MatmulKernel::Params{problemShape,
                                               static_cast<uint32_t>(EP),
                                               static_cast<uint32_t>(listLen),
                                               static_cast<uint32_t>(expertPerRank),
                                               static_cast<uint32_t>(maxOutputSize),
                                               static_cast<uint32_t>(topK),
                                               initRoutingQuantTilingKey,
                                               epilogueCoreNum,
                                               contextGM_,
                                               xGM_,
                                               layoutA1,
                                               layoutA2,
                                               weight1GM_,
                                               layoutB1,
                                               bias1GM_,
                                               weight2GM_,
                                               layoutB2,
                                               bias2GM_,
                                               weightScales1GM_,
                                               layoutScale1,
                                               weightScales2GM_,
                                               layoutScale2,
                                               yGM_,
                                               layoutD1,
                                               layoutD2,
                                               topkIdsGM_,
                                               moeInitRoutingQuantV2Scale,
                                               moeInitRoutingQuantV2Offset,
                                               expertTokensBeforeCapacity,
                                               topkWeightsGM_,
                                               workspaceGM_,
                                               gmExpertTokenNums_,
                                               xActiveMaskGM_,
                                               scalesGM_,
                                               moeInitRoutingV2TilingData,
                                               epilogueGranularity,
                                               activationClamp};
    }

    MatmulKernel kernel(params);
    kernel(params);
}

} // namespace MegaMoeImpl
#endif // MEGA_MOE_H
