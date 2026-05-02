/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_CONV_KERNEL_BASIC_CONV3D_HPP
#define CATLASS_CONV_KERNEL_BASIC_CONV3D_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/conv_coord.hpp"
#include "catlass/coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catlass::Conv::Kernel {

// Template for conv3d with bias kernel.
template <
    class BlockConv_,
    class BlockEpilogue_,
    class BlockScheduler_
>
class ConvBias {
public:
    using BlockConv = BlockConv_;
    using ArchTag = typename BlockConv::ArchTag;
    using CoreTileShape = typename BlockConv::CoreTileShape;
    using ElementFmap = typename BlockConv::ElementFmap;
    using LayoutFmap = typename BlockConv::LayoutFmap;
    using ElementFilter = typename BlockConv::ElementFilter;
    using LayoutFilter = typename BlockConv::LayoutFilter;
    using ElementOut = typename BlockConv::ElementOut;
    using LayoutOut = typename BlockConv::LayoutOut;
    using ElementBias = typename BlockConv::ElementBias;
    using ElementAccumulator = typename BlockConv::ElementAccumulator;

    // using Conv3dParams = typename Catlass::Conv3dParams;

    using BlockScheduler = BlockScheduler_;

    struct Params {
        //Data members
        Conv3dParams problemShape;
        GM_ADDR ptrFmap;
        LayoutFmap layoutFmap;
        GM_ADDR ptrFilter;
        LayoutFilter layoutFilter;
        GM_ADDR ptrOut;
        LayoutOut layoutOut;
        GM_ADDR ptrBias;

        //Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(
            Conv3dParams const &problemShape_,
            GM_ADDR ptrFmap_, LayoutFmap const &layoutFmap_,
            GM_ADDR ptrFilter_, LayoutFilter const &layoutFilter_,
            GM_ADDR ptrOut_, LayoutOut const &layoutOut_,
            GM_ADDR ptrBias_
        ) : problemShape(problemShape_), ptrFmap(ptrFmap_), layoutFmap(layoutFmap_), ptrFilter(ptrFilter_), layoutFilter(layoutFilter_),
            ptrOut(ptrOut_), layoutOut(layoutOut_), ptrBias(ptrBias_) {}
    };

    struct Arguments {
        Conv3dParams problemShape;
        GM_ADDR ptrFmap;
        GM_ADDR ptrFilter;
        GM_ADDR ptrOut;
        GM_ADDR ptrBias;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        LayoutFmap layoutFmap = LayoutFmap::MakeLayout(args.problemShape.batch(), args.problemShape.di(), args.problemShape.cin1(), args.problemShape.hi(), args.problemShape.wi(), args.problemShape.cin0());
        LayoutFilter layoutFilter = LayoutFilter::MakeLayout(args.problemShape.kdc1khkw(), args.problemShape.n1(), args.problemShape.n0(), args.problemShape.cin0());
        LayoutOut layoutOut = LayoutOut::MakeLayout(args.problemShape.batch(), args.problemShape.dout(), args.problemShape.cout1(), args.problemShape.ho(), args.problemShape.wo(), args.problemShape.cout0());
        Params params{
            args.problemShape,
            args.ptrFmap,
            layoutFmap,
            args.ptrFilter,
            layoutFilter,
            args.ptrOut,
            layoutOut,
            args.ptrBias};
        return params;
    }

    // Methods
    CATLASS_DEVICE
    ConvBias() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    ConvBias() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params const &params);

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params const &params)
    {
        BlockScheduler convBlockScheduler(Conv3d6HdCoord{params.problemShape.batch(), params.problemShape.dout(),
                                          params.problemShape.cout1(), params.problemShape.howo()},
                                          Conv3d6HdCoord{CoreTileShape::noCnt, CoreTileShape::doCnt,
                                                         CoreTileShape::co1Cnt, CoreTileShape::howoCnt});
        uint32_t coreLoops = convBlockScheduler.GetCoreLoops();

        Arch::Resource<ArchTag> resource;
        BlockConv blockConv(resource, params.problemShape);

        // represent the full gm
        AscendC::GlobalTensor<ElementFmap> fmapGm;
        fmapGm.SetGlobalBuffer((__gm__ ElementFmap *)params.ptrFmap);
        AscendC::GlobalTensor<ElementFilter> filterGm;
        filterGm.SetGlobalBuffer((__gm__ ElementFilter *)params.ptrFilter);
        AscendC::GlobalTensor<ElementOut> outGm;
        outGm.SetGlobalBuffer((__gm__ ElementOut *)params.ptrOut);
        AscendC::GlobalTensor<ElementBias> biasGm;
        biasGm.SetGlobalBuffer((__gm__ ElementBias *)params.ptrBias);

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < coreLoops; loopIdx += AscendC::GetBlockNum()) {
            Conv3d6HdCoord blockCoord = convBlockScheduler.GetBlockCoord(loopIdx);
            Conv3d6HdCoord dimStartCoord = convBlockScheduler.GetDimStartIdx(blockCoord);
            Conv3d6HdCoord actualBlockShape = convBlockScheduler.GetActualBlockShape(blockCoord, dimStartCoord);

            uint32_t diIdxStart = Max(dimStartCoord.d() * params.problemShape.sD(), params.problemShape.padhead(), 0);
            uint32_t hiwiIdxStart = Max((dimStartCoord.hw() / params.problemShape.wo()) * params.problemShape.sH(),
                                         params.problemShape.padtop(), 0) * params.problemShape.wi();
            
            // Compute initial location in logical coordinates
            Conv3d6HdCoord offsetFmap{dimStartCoord.n(), diIdxStart, 0, hiwiIdxStart};
            Conv3dFracZ3dCoord offsetFilter{0, dimStartCoord.c1() * params.problemShape.cout0()};
            Conv3d6HdCoord offsetOut{dimStartCoord.n(), dimStartCoord.d(), dimStartCoord.c1(), dimStartCoord.hw()};
            Conv3d6HdCoord actualIdxStartFmap{0, dimStartCoord.d() * params.problemShape.sD(), 0, dimStartCoord.hw()};

            int64_t gmOffsetFmap = params.layoutFmap.GetOffset(offsetFmap);
            int64_t gmOffsetFilter = params.layoutFilter.GetOffset(offsetFilter);
            int64_t gmOffsetOut = params.layoutOut.GetOffset(offsetOut);
            int64_t gmOffsetBias = dimStartCoord.c1() * params.problemShape.cout0();

            blockConv(
                fmapGm[gmOffsetFmap], params.layoutFmap,
                filterGm[gmOffsetFilter], params.layoutFilter,
                outGm[gmOffsetOut], params.layoutOut,
                biasGm[gmOffsetBias],
                actualBlockShape,
                actualIdxStartFmap);
        }
    }

    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params const &params) {}

    __aicore__ inline uint32_t Max(uint32_t a, uint32_t b, uint32_t c)
    {
        if (a > b) {
            return a - b;
        } else {
            return c;
        }
    }
};

}  // namespace Catlass::Conv::Kernel
#endif // CATLASS_CONV_KERNEL_BASIC_CONV3D_HPP