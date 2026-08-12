/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_CATLASS_BLOCK_DAVID_UB_ANTIQUANT_SCMC_LOAD_IN_ADVANCE_AIC_TAIL_RESPLIT_H
#define ARCH35_CATLASS_BLOCK_DAVID_UB_ANTIQUANT_SCMC_LOAD_IN_ADVANCE_AIC_TAIL_RESPLIT_H

#include "../iterator/continuous_iterator.h"
#include "../pipeline/pipeline_stage_mixcore.h"
#include "../pipeline/pipeline_stage_singlecore_copy_in_advance.h"
#include "../pipeline/pipeline_state.h"
#include "../tile/copy_gm_to_l1.h"
#include "../utils/constant.h"
#include "../utils/device_utils.h"
#include "../utils/math_utils.h"
#include "block_decl.h"
#include "block_utils.h"
#include "david_ub_antiquant_scmc.h"

using AscendC::AIC;
using AscendC::BLOCK_CUBE;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::TBuf;
using AscendC::TPosition;
using AscendC::Std::get;

namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass {

template <TPosition POSITION, CubeFormat FORMAT, typename TYPE, bool ISTRANS = false,
          LayoutMode LAYOUT = LayoutMode::NONE, bool IBSHARE = false>
struct MatmulL1GmType : matmul::MatmulType<POSITION, FORMAT, TYPE, ISTRANS, LAYOUT, IBSHARE> {
    constexpr static TPosition srcPos = TPosition::GM;
};

template <int Stages, typename TileShapeUb, typename TileShapeReg, int32_t SubBlockNum, uint8_t StageWeightIn,
          uint8_t StageVfOut, int32_t Kub_, typename KernelSchedule, typename TileShapeL1, typename TileShapeL0,
          typename DtypeA, typename StrideA, typename DtypeB, typename StrideBOptionalTuple, typename DtypeBias,
          typename StrideBiasGm, typename DtypeC, typename StrideC>
struct BlockMmad<MainloopDavidWqbmmUbAntiquantScmc<Stages, TileShapeUb, TileShapeReg, AscendC::AIC, SubBlockNum,
                                                   StageWeightIn, StageVfOut, Kub_, KernelSchedule>,
                 TileShapeL1, TileShapeL0, DtypeA, StrideA, DtypeB, StrideBOptionalTuple, DtypeBias, StrideBiasGm,
                 DtypeC, StrideC>
    : public DavidUbAntiquantScmc<Stages, TileShapeUb, TileShapeReg, AscendC::AIC, SubBlockNum, StageWeightIn,
                                  StageVfOut, Kub_, KernelSchedule, TileShapeL1, TileShapeL0, DtypeA, StrideA, DtypeB,
                                  StrideBOptionalTuple, DtypeBias, StrideBiasGm, DtypeC, StrideC> {
public:
    using Base = DavidUbAntiquantScmc<Stages, TileShapeUb, TileShapeReg, AscendC::AIC, SubBlockNum, StageWeightIn,
                                      StageVfOut, Kub_, KernelSchedule, TileShapeL1, TileShapeL0, DtypeA, StrideA,
                                      DtypeB, StrideBOptionalTuple, DtypeBias, StrideBiasGm, DtypeC, StrideC>;
    using Arguments = typename Base::Arguments;
    using NonVoidStrideBias = typename Base::NonVoidStrideBias;

private:
    // gm global tensor
    AscendC::GlobalTensor<DtypeA> tensorXGm_;
    AscendC::GlobalTensor<DtypeBias> tensorBiasGm_;
    AscendC::GlobalTensor<uint64_t> quantScaleGlobal_;
    AscendC::GlobalTensor<DtypeC> tensorCGm_;

    // matmul api
    using inputXType = MatmulL1GmType<TPosition::TSCM, CubeFormat::NZ, DtypeA, Base::isTransA>;
    using inputWType = MatmulL1GmType<TPosition::TSCM, CubeFormat::NZ, DtypeA, Base::isTransB>;
    using outputDtypeY = matmul::MatmulType<TPosition::GM, CubeFormat::ND, DtypeC>;
    using inputBiasType = matmul::MatmulType<TPosition::TSCM, CubeFormat::ND, DtypeBias>;
    matmul::MatmulImpl<inputXType, inputWType, outputDtypeY, inputBiasType, CFG_MDL> mmObj_;

    using StrideXL1 = AscendC::Std::tuple<uint32_t, _256, _16, _1, uint32_t>;
    using StrideBiasL1 = AscendC::Std::tuple<_1, uint32_t>;
    using StrideWL1 = AscendC::Std::tuple<uint32_t>;

    LocalTensor<DtypeA> tensorXL1_;
    LocalTensor<DtypeA> tensorWL1_;
    LocalTensor<DtypeBias> tensorBiasL1_;

    TBuf<TPosition::TSCM> l1TBuf_;

    ContinuousIterator<uint64_t> iterKaLoad_;

public:
    struct Params {
        __gm__ DtypeA *aGmAddr = nullptr;
        __gm__ DtypeC *cGmAddr = nullptr;
        __gm__ DtypeBias *biasGmAddr = nullptr;

        TileShapeL1 tileShapeL1;

        StrideA strideXGm;
        StrideXL1 strideXL1;

        NonVoidStrideBias strideBiasGm;
        StrideBiasL1 strideBiasL1;

        StrideWL1 strideWL1;

        StrideC strideC;

        TCubeTiling matmulTiling;

        uint64_t mainBlockCount;
        uint32_t firstTailBlockCount;
        uint32_t secondTailBlockCount;
        uint32_t mainBlockSize;
        uint32_t firstTailBlockSize;
        uint32_t secondTailBlockSize;

        uint32_t cubeBlockDimM;
        uint32_t cubeBlockDimN;

        uint32_t hasBias;

        bool fullloadA;
        bool fullloadBias;
    };

    using PipelineTuple =
        AscendC::Std::tuple<PipelineStageSingleCoreCopyInAdvance<Hardware::GM, Hardware::L1, Stages>, /*x*/
                            PipelineStageSingleCoreCopyInAdvance<Hardware::GM, Hardware::L1, Stages> /*bias*/>;
    using PipelineStateTuple = AscendC::Std::tuple<PipelineState<Stages>, /*x gm2l1 Load*/
                                                   PipelineState<Stages>, /*x gm2l1 compute*/
                                                   PipelineState<Stages>, /*w aiv2aic*/
                                                   PipelineState<Stages>, /*bias gm2l1 Load*/
                                                   PipelineState<Stages> /*bias gm2l1 compute*/>;
    using PipelineStateAiv2Aic = typename AscendC::Std::tuple_element<Stages, PipelineStateTuple>::type;
    using PipelineAiv2Aic = PipelineStageMixCore<PIPE_MTE3, PIPE_MTE1, SubBlockNum>;

public:
    DEVICE BlockMmad()
    {
    }

    template <class ProblemShape, typename TilingData>
    DEVICE static Params GetParams(ProblemShape const &problemShape, Arguments const &args, TilingData const &tiling)
    {
        decltype(auto) sizeM = get<0>(problemShape);
        decltype(auto) sizeN = get<1>(problemShape);
        decltype(auto) sizeK = get<2>(problemShape);
        // tiling中保证stepM,stepN等于1，所以用baseM,baseN代表L1大小
        uint32_t mL1 = tiling.matmulTiling.baseM;
        // NOTE 由于尾块重切分的nL1是变化的，stride得按照最大的计算
        uint32_t nL1 = Max(Max(tiling.mainBlockL1Size, tiling.firstTailBlockL1Size), tiling.secondTailBlockL1Size);
        uint32_t kaL1 = tiling.matmulTiling.baseK * tiling.matmulTiling.stepKa;
        uint32_t kbL1 = tiling.matmulTiling.baseK * tiling.matmulTiling.stepKb;
        return {
            .aGmAddr = args.aGmAddr,
            .cGmAddr = args.cGmAddr,
            .biasGmAddr = args.biasGmAddr,

            .tileShapeL1 = AscendC::Std::make_tuple(mL1, nL1, kaL1, kbL1),

            .strideXGm = args.strideA,
            .strideXL1 =
                Base::isTransA ? /*m1,k1,k0,m0*/ AscendC::Std::make_tuple(
                                     kaL1 * 16U, _256{}, _16{}, _1{},
                                     static_cast<uint32_t>(256 * 1024 / sizeof(DtypeA) - nL1 * kbL1 -
                                                           (tiling.hasBias ? 4 * 1024 / sizeof(DtypeA) : 0))) :
                                 /*k1,m1,m0,k0*/
                                 AscendC::Std::make_tuple(mL1 * 16U, _256{}, _16{}, _1{},
                                                          static_cast<uint32_t>(256 * 1024 / sizeof(DtypeA) - nL1 * kbL1 -
                                                                                (tiling.hasBias ? 4 * 1024 / sizeof(DtypeA) : 0))),
            .strideBiasGm = args.strideBias,
            // w0 bias0(4KB) x0 | x1 bias1(4KB) w1 quantScale(8KB,目前不支持)
            .strideBiasL1 = tiling.hasBias ?
                                AscendC::Std::make_tuple(
                                    _1{}, static_cast<uint32_t>((508 * 1024 - 2 * nL1 * kbL1 * sizeof(DtypeA)) /
                                                                sizeof(DtypeBias))) :
                                AscendC::Std::make_tuple(_1{}, 0U),
            .strideWL1 = AscendC::Std::make_tuple(static_cast<uint32_t>(512 * 1024U / 2 - nL1 * kbL1)),
            .strideC = args.strideC,
            .matmulTiling = tiling.matmulTiling,

            .mainBlockCount = tiling.mainBlockCount,
            .firstTailBlockCount = tiling.firstTailBlockCount,
            .secondTailBlockCount = tiling.secondTailBlockCount,
            .mainBlockSize = tiling.mainBlockL1Size,
            .firstTailBlockSize = tiling.firstTailBlockL1Size,
            .secondTailBlockSize = tiling.secondTailBlockL1Size,
            .cubeBlockDimM = tiling.cubeBlockDimM,
            .cubeBlockDimN = tiling.cubeBlockDimN,
            // NOTE non-constant-expression cannot be narrowed from uint32_t to bool in initializaer list
            .hasBias = tiling.hasBias,
            .fullloadA = sizeK <= kaL1 && CeilDiv(sizeM, static_cast<uint64_t>(tiling.cubeBlockDimM)) <= mL1,
            .fullloadBias = (AscendC::GetSubBlockIdx() % tiling.cubeBlockDimN + tiling.cubeBlockDimN) >=
                            (tiling.mainBlockCount + tiling.firstTailBlockCount + tiling.secondTailBlockCount),
        };
    }

    template <class ProblemShape>
    DEVICE static Params GetParams(ProblemShape const &problemShape, Arguments const &args)
    {
    }

    template <class ProblemShape>
    DEVICE auto Init(ProblemShape const &problemShape, Params const &params)
    {
        tensorXGm_.SetGlobalBuffer(params.aGmAddr);
        tensorCGm_.SetGlobalBuffer(params.cGmAddr);
        if (params.hasBias) {
            tensorBiasGm_.SetGlobalBuffer(params.biasGmAddr);
        }
        mmObj_.SetSubBlockIdx(0);
        mmObj_.Init(&params.matmulTiling, GetTPipePtr());

        // b0 a1 a0 b1
        GetTPipePtr()->InitBuffer(l1TBuf_, 512 * 1024);

        auto l1n = get<1>(params.tileShapeL1);
        auto l1Kb = get<3>(params.tileShapeL1);
        int32_t weightL1Space = l1Kb * l1n;
        tensorWL1_ = l1TBuf_.Get<DtypeA>();
        // W0 bias0 X0 | X1 bias1 W1
        if (params.hasBias) {
            if constexpr (AscendC::IsSameType<DtypeBias, float>::value) {
                tensorBiasL1_ = l1TBuf_.Get<DtypeBias>()[weightL1Space >> 1];
            } else {
                tensorBiasL1_ = l1TBuf_.Get<DtypeBias>()[weightL1Space];
            }

            // bias单块分配4K空间
            tensorXL1_ = l1TBuf_.Get<DtypeA>()[weightL1Space + 4 * 1024 / sizeof(DtypeA)];
        } else {
            tensorXL1_ = l1TBuf_.Get<DtypeA>()[weightL1Space];
        }

        auto sizeK = get<2>(problemShape);
        iterKaLoad_.Update(sizeK, get<2>(params.tileShapeL1));
    }

    // 辅助函数：处理X数据加载
    template <typename ProblemShape, typename Iterators, typename Params>
    DEVICE void HandleXLoad(ProblemShape const &problemShape, PipelineTuple const &pipelines,
                            PipelineStateTuple &states, Params const &params, Iterators &iters)
    {
        decltype(auto) sizeM = get<0>(problemShape);
        decltype(auto) sizeK = get<2>(problemShape);

        decltype(auto) pipelineXGm2L1 = get<0>(pipelines);
        decltype(auto) iterMLoad = get<0>(iters);
        decltype(auto) pipelineStateXLoad = get<0>(states);

        pipelineXGm2L1.ProducerWait(pipelineStateXLoad);
        CopyXGm2L1(crd2idx(params.strideXGm, *iterMLoad, *iterKaLoad_),
                   crd2idx(params.strideXL1, _0{}, _0{}, _0{}, _0{}, pipelineStateXLoad.index()), sizeM, sizeK,
                   iterMLoad.Size(), iterKaLoad_.Size(), get<0>(params.tileShapeL1), get<2>(params.tileShapeL1));
        pipelineXGm2L1.ProducerRelease(pipelineStateXLoad);
        ++pipelineStateXLoad;
    }

    // 辅助函数：处理Bias数据加载
    template <typename Params, typename Iterators, typename PipelineTuple, typename PipelineStateTuple>
    DEVICE void HandleBiasLoad(Params const &params, Iterators &iters, PipelineTuple const &pipelines,
                               PipelineStateTuple &states)
    {
        decltype(auto) iterNBiasLoad = get<3>(iters);
        decltype(auto) pipelineStateBiasLoad = get<3>(states);

        CopyBiasGm2L1(crd2idx(params.strideBiasGm, *iterNBiasLoad),
                      crd2idx(params.strideBiasL1, _0{}, pipelineStateBiasLoad.index()), iterNBiasLoad.Size());
    }

    // 重构后的LoadInAdvance函数
    template <typename ProblemShape, typename Iterators>
    DEVICE void LoadInAdvance(ProblemShape const &problemShape, PipelineTuple const &pipelines,
                              PipelineStateTuple &states, Params const &params, Iterators &iters)
    {
        HandleXLoad(problemShape, pipelines, states, params, iters);
        decltype(auto) pipelineBiasGm2L1 = get<1>(pipelines);
        decltype(auto) pipelineStateBiasLoad = get<3>(states);
        pipelineBiasGm2L1.ProducerWait(pipelineStateBiasLoad);
        if (params.hasBias) {
            HandleBiasLoad(params, iters, pipelines, states);
        }
        pipelineBiasGm2L1.ProducerRelease(pipelineStateBiasLoad);
        ++pipelineStateBiasLoad;
        // 处理Bias
        if (params.hasBias) {
            // 2是iterMBiasLoad, 3是iterNBiasLoad
            UpdateNMIterators(get<2>(iters), get<3>(iters));
        }

        decltype(auto) iterMLoad = get<0>(iters);
        if (params.fullloadA) {
            ++iterKaLoad_;
            ++iterMLoad;
        } else {
            UpdateKNMIterators(iterMLoad, get<1>(iters));
        }
    }

    // 辅助函数：处理计算阶段的Bias加载
    template <typename Params, typename Iterators, typename PipelineTuple, typename PipelineStateTuple>
    DEVICE void ProcessBiasLoad(Params const &params, Iterators &itersLoad, PipelineTuple const &pipelines,
                                PipelineStateTuple &states)
    {
        decltype(auto) iterMBiasLoad = get<2>(itersLoad);
        if (iterMBiasLoad.End())
            return;

        decltype(auto) pipelineBiasGm2L1 = get<1>(pipelines);
        decltype(auto) iterNBiasLoad = get<3>(itersLoad);
        decltype(auto) pipelineStateBiasLoad = get<3>(states);

        CopyBiasGm2L1(crd2idx(params.strideBiasGm, *iterNBiasLoad),
                      crd2idx(params.strideBiasL1, _0{}, pipelineStateBiasLoad.index()), iterNBiasLoad.Size());
        UpdateNMIterators(iterMBiasLoad, iterNBiasLoad);
    }

    // 辅助函数：处理计算阶段的X加载
    template <typename ProblemShape, typename Iterators, typename PipelineTuple, typename PipelineStateTuple,
              typename Params>
    DEVICE void ProcessXLoad(ProblemShape const &problemShape, Iterators &itersLoad, PipelineTuple const &pipelines,
                             PipelineStateTuple &states, Params const &params)
    {
        decltype(auto) iterMLoad = get<0>(itersLoad);
        decltype(auto) iterNLoad = get<1>(itersLoad);
        if (iterMLoad.End())
            return;

        decltype(auto) sizeM = get<0>(problemShape);
        decltype(auto) sizeK = get<2>(problemShape);
        decltype(auto) pipelineXGm2L1 = get<0>(pipelines);
        decltype(auto) pipelineStateXLoad = get<0>(states);

        CopyXGm2L1(crd2idx(params.strideXGm, *iterMLoad, *iterKaLoad_),
                   crd2idx(params.strideXL1, _0{}, _0{}, _0{}, _0{}, pipelineStateXLoad.index()), sizeM, sizeK,
                   iterMLoad.Size(), iterKaLoad_.Size(), get<0>(params.tileShapeL1), get<2>(params.tileShapeL1));
        UpdateKNMIterators(iterMLoad, iterNLoad);
    }

    // 重构后的Process函数
    template <typename ProblemShape, typename Iterators>
    DEVICE void Process(ProblemShape const &problemShape, PipelineTuple const &pipelines, PipelineStateTuple &states,
                        Params const &params, Iterators &itersLoad, Iterators const &itersCompute)
    {
        decltype(auto) sizeM = get<0>(problemShape);
        decltype(auto) sizeN = get<1>(problemShape);
        decltype(auto) sizeK = get<2>(problemShape);
        decltype(auto) kaL1 = get<2>(params.tileShapeL1);

        decltype(auto) pipelineXGm2L1 = get<0>(pipelines);
        decltype(auto) pipelineBiasGm2L1 = get<1>(pipelines);

        decltype(auto) pipelineStateXLoad = get<0>(states);
        decltype(auto) pipelineStateBiasLoad = get<3>(states);

        pipelineBiasGm2L1.ProducerWait(pipelineStateBiasLoad);
        if (params.hasBias) {
            ProcessBiasLoad(params, itersLoad, pipelines, states);
        }
        pipelineBiasGm2L1.ProducerRelease(pipelineStateBiasLoad);
        ++pipelineStateBiasLoad;

        pipelineBiasGm2L1.ConsumerWait(get<4>(states));
        for (ContinuousIterator<uint64_t> iterKaCompute(sizeK, get<2>(params.tileShapeL1)); !iterKaCompute.End();
             ++iterKaCompute) {
            pipelineXGm2L1.ProducerWait(pipelineStateXLoad);
            ProcessXLoad(problemShape, itersLoad, pipelines, states, params);
            pipelineXGm2L1.ProducerRelease(pipelineStateXLoad);
            ++pipelineStateXLoad;

            pipelineXGm2L1.ConsumerWait(get<1>(states));
            for (ContinuousIterator<uint32_t> iterKb(0, iterKaCompute.Size(), get<3>(params.tileShapeL1),
                                                     *iterKaCompute);
                 !iterKb.End(); ++iterKb) {
                Mmad(crd2idx(params.strideXL1, CeilDiv(*iterKb - *iterKaCompute, 16UL), _0{}, _0{}, _0{},
                             params.fullloadA ? 0 : get<1>(states).index()),
                     crd2idx(params.strideWL1, get<2>(states).index()),
                     crd2idx(params.strideBiasL1, _0{}, params.fullloadBias ? 0 : get<4>(states).index()),
                     get<0>(itersCompute).Size(), get<1>(itersCompute).Size(), kaL1, iterKaCompute.Size(),
                     iterKb.Size(), *iterKb != 0, sizeN, get<2>(states), params.hasBias, get<0>(params.tileShapeL1),
                     get<2>(params.tileShapeL1));
            }
            pipelineXGm2L1.ConsumerRelease(get<1>(states));
            ++get<1>(states);
        }
        pipelineBiasGm2L1.ConsumerRelease(get<4>(states));
        ++get<4>(states);
        CopyL0c2GM(crd2idx(params.strideC, *get<0>(itersCompute), *get<1>(itersCompute)));
    }

    DEVICE void ClearPipeline(PipelineTuple const &pipelines, PipelineStateTuple &states)
    {
        decltype(auto) pipelineXGm2L1 = get<0>(pipelines);
        decltype(auto) pipelineBiasGm2L1 = get<1>(pipelines);

        decltype(auto) pipelineStateXLoad = get<0>(states);
        decltype(auto) pipelineStateAiv2Aic = get<2>(states);
        decltype(auto) pipelineStateBiasLoad = get<3>(states);

        PipelineAiv2Aic::Clear(pipelineStateAiv2Aic);
        pipelineXGm2L1.Clear(pipelineStateXLoad);
        pipelineBiasGm2L1.Clear(pipelineStateBiasLoad);
    }

private:
    DEVICE void CopyXGm2L1(uint64_t offsetGm, uint64_t offsetL1, uint64_t m, uint64_t k, uint64_t tileSizeM,
                           uint64_t tileSizeK, uint32_t dstSizeM, uint32_t dstSizeK)
    {
        if constexpr (Base::isTransB) {
            CopyGmToL1IntervalDataCopy(tensorXL1_[offsetL1], tensorXGm_[offsetGm], tileSizeK, tileSizeM, dstSizeK, m);
        } else {
            CopyGmToL1IntervalDataCopy(tensorXL1_[offsetL1], tensorXGm_[offsetGm], tileSizeM, tileSizeK, dstSizeM, k);
        }
    }

    DEVICE void CopyBiasGm2L1(uint64_t offsetGm, uint64_t offsetL1, uint64_t tileSizeN)
    {
        CopyGmToL1(tensorBiasL1_[offsetL1], tensorBiasGm_[offsetGm],
                   CeilAlign(tileSizeN, static_cast<uint64_t>(BLOCK_CUBE)));
    }

    DEVICE void Mmad(uint64_t offsetXL1, uint64_t offsetWL1, uint64_t offsetBiasL1, uint64_t tileSizeM,
                     uint64_t tileSizeN, uint64_t tileSizeKa, uint64_t realTileSizeKa, uint64_t realTileSizeKb,
                     bool enPartialSum, uint64_t sizeN, PipelineStateAiv2Aic &pipelineStateAiv2Aic, bool hasBias,
                     uint64_t strideM, uint64_t strideKa)
    {
        mmObj_.SetOrgShape(strideM, CeilAlign(tileSizeN, static_cast<uint64_t>(BLOCK_CUBE)), strideKa,
                           CeilAlign(realTileSizeKb, static_cast<uint64_t>(BLOCK_CUBE)), sizeN);

        mmObj_.SetTensorA(tensorXL1_[offsetXL1], Base::isTransA);
        mmObj_.SetTensorB(tensorWL1_[offsetWL1], Base::isTransB);

        if (hasBias) {
            mmObj_.SetBias(tensorBiasL1_[offsetBiasL1]);
        }

        mmObj_.SetTail(tileSizeM, tileSizeN, realTileSizeKb);

        PipelineAiv2Aic::ConsumerWait();
        mmObj_.Iterate(enPartialSum); // false 清零 true 保留
        PipelineAiv2Aic::ConsumerRelease();
        ++pipelineStateAiv2Aic;
    }

    DEVICE void CopyL0c2GM(uint64_t offsetYGm)
    {
        mmObj_.GetTensorC(tensorCGm_[offsetYGm]);
    }

    template <typename IterM, typename IterN>
    DEVICE void UpdateKNMIterators(IterM &iterMLoad, IterN &iterNLoad)
    {
        ++iterKaLoad_;
        if (iterKaLoad_.End()) {
            iterKaLoad_.Reset();
            ++iterNLoad;
            if (iterNLoad.End()) {
                iterNLoad.Reset();
                ++iterMLoad;
            }
        }
    }

    template <typename IterM, typename IterN>
    DEVICE void UpdateNMIterators(IterM &iterMLoad, IterN &iterNLoad)
    {
        ++iterNLoad;
        if (iterNLoad.End()) {
            iterNLoad.Reset();
            ++iterMLoad;
        }
    }
};
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass
#endif
