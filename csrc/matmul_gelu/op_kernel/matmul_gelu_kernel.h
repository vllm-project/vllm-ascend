/*
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

#ifndef _CUSTOM_MATMUL_GELU_KERNEL_H
#define _CUSTOM_MATMUL_GELU_KERNEL_H

#include <acl/acl.h>

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_gelu.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"

#include "./kernel_matmul_activation.hpp"
#include "matmul_gelu_tiling.h"

namespace MatmulGelu_Kernel {
using namespace Catlass;
template <class LayoutWeight, class InDType, uint32_t m, uint32_t n, uint32_t k1, uint32_t k0>
CATLASS_DEVICE void MatmulGeluImpl(MatmulGeluTilingData tiling_data, GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR output, GM_ADDR workspace)
{
        Catlass::GemmCoord problemShape{tiling_data.m, tiling_data.n, tiling_data.k};

        // Define ArchTag
        using ArchTag = Arch::AtlasA2;

        using LayoutA = layout::RowMajor;
        using LayoutB = LayoutWeight;
        using LayoutD = layout::RowMajor;
        using LayoutBias = layout::VectorLayout;
        using LayoutC = layout::RowMajor;

        // Block level, define BlockMmad
        constexpr bool enableUnitFlag = true;
        using MmadDispatchPolicy = Gemm::MmadAtlasA2PingpongBias<enableUnitFlag>;
        using L1TileShape = GemmShape<m, n, k1>;
        using L0TileShape = GemmShape<m, n, k0>;
        using AType = Gemm::GemmType<half, LayoutA>;
        using BType = Gemm::GemmType<half, LayoutB>;
        using CType = Gemm::GemmType<float, LayoutC>;
        using BiasType = Gemm::GemmType<InDType, LayoutBias>;
        using DType = Gemm::GemmType<half, LayoutC>;
        using BlockMmad = Gemm::Block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType, BiasType>;
        using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2ElemWiseNoSource;

        // Define epilogue
        constexpr uint32_t computeLength = (m * n) / 2;
        using TileElemWiseEpilogue = Epilogue::Tile::TileElemWiseGelu<ArchTag, CType, computeLength>;
        using EpilogueTileCopy = Epilogue::Tile::TileCopy<
                ArchTag,
                CType, // CopyGmtoUbC
                DType  // CopyUbtoGmD
                >;
        using BlockEpilogue =
        Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType, DType, TileElemWiseEpilogue, EpilogueTileCopy>;
        using EpilogueParams = typename BlockEpilogue::Params;

        // 内部模板函数：处理具体的BlockScheduler类型
        auto runMatmulKernel = [&]<typename BlockSchedulerType>() {
                // Kernel level
                using MatmulKernel = Gemm::Kernel::MatmulActivation<BlockMmad, BlockEpilogue, BlockSchedulerType>;

                // Prepare params
                LayoutA layoutA{problemShape.m(), problemShape.k()};
                LayoutB layoutB{problemShape.k(), problemShape.n()};
                LayoutC layoutC{problemShape.m(), problemShape.n()};

                EpilogueParams epilogueParams(workspace, layoutC, output, layoutC);
                typename MatmulKernel::Params params(
                                problemShape,
                                x, layoutA,
                                weight, layoutB,
                                bias,
                                workspace,
                                epilogueParams
                                );
                MatmulKernel matmulKernel;
                matmulKernel(params);
        };

        // 根据条件选择不同的BlockScheduler并执行
        if (tiling_data.m > tiling_data.n) {
                // Swizzle offset is 3 and direction is 0.
                using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
                runMatmulKernel.template operator()<BlockScheduler>();
        } else {
                // Swizzle offset is 3 and direction is 1.
                using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
                runMatmulKernel.template operator()<BlockScheduler>();
        }
}
}  // namespace CustomMatmulGelu_Kernel

#endif