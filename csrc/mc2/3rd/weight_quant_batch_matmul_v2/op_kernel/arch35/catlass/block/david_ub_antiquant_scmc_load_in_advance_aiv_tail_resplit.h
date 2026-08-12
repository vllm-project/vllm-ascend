/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_CATLASS_BLOCK_DAVID_UB_ANTIQUANT_SCMC_LOAD_IN_ADVANCE_AIV_TAIL_RESPLIT_H
#define ARCH35_CATLASS_BLOCK_DAVID_UB_ANTIQUANT_SCMC_LOAD_IN_ADVANCE_AIV_TAIL_RESPLIT_H

#include "../constant.h"
#include "../iterator/continuous_iterator.h"
#include "../pipeline/pipeline_stage_mixcore.h"
#include "../pipeline/pipeline_stage_singlecore.h"
#include "../pipeline/pipeline_stage_singlecore_copy_in_advance.h"
#include "../pipeline/pipeline_state.h"
#include "../simd/a16w4_pergroup_kn_nz.h"
#include "../tile/copy_gm_to_ub.h"
#include "../tile/copy_ub_to_l1.h"
#include "../utils/constant.h"
#include "../utils/device_utils.h"
#include "../utils/math_utils.h"
#include "block_decl.h"
#include "constant.h"
#include "david_ub_antiquant_scmc.h"

using AscendC::DataCopyParams;
using AscendC::GlobalTensor;
using AscendC::int4b_t;
using AscendC::IsSameType;
using AscendC::LocalTensor;
using AscendC::ONE_BLK_SIZE;
using AscendC::TBuf;
using AscendC::VECTOR_REG_WIDTH;

namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass {

template <typename StrideAntiquantScale>
DEVICE static constexpr Mc2QuantType GetQuantType() // StrideAntiquantScale /*strideAntiquantScale*/)
{
    if constexpr (AscendC::Std::tuple_size<StrideAntiquantScale>::value == 1) {
        return Mc2QuantType::PER_TENSOR;
    } else if constexpr (AscendC::Std::tuple_size<StrideAntiquantScale>::value == 2 &&
                         AscendC::Std::is_same_v<typename AscendC::Std::tuple_element<0, StrideAntiquantScale>::type,
                                                 _1> &&
                         AscendC::Std::is_same_v<typename AscendC::Std::tuple_element<1, StrideAntiquantScale>::type,
                                                 _1>) {
        return Mc2QuantType::PER_CHANNEL;
    } else {
        return Mc2QuantType::PER_GROUP;
    }
}

template <int Stages, typename TileShapeUb, typename TileShapeReg, int32_t SubBlockNum, uint8_t StageWeightIn,
          uint8_t StageVfOut, int32_t Kub_, typename KernelSchedule, typename TileShapeL1, typename TileShapeL0,
          typename DtypeA, typename StrideA, typename DtypeB, typename StrideBOptionalTuple, typename DtypeBias,
          typename StrideBiasGm, typename DtypeC, typename StrideC>
struct BlockMmad<MainloopDavidWqbmmUbAntiquantScmc<Stages, TileShapeUb, TileShapeReg, AscendC::AIV, SubBlockNum,
                                                   StageWeightIn, StageVfOut, Kub_, KernelSchedule>,
                 TileShapeL1, TileShapeL0, DtypeA, StrideA, DtypeB, StrideBOptionalTuple, DtypeBias, StrideBiasGm,
                 DtypeC, StrideC>
    : public DavidUbAntiquantScmc<Stages, TileShapeUb, TileShapeReg, AscendC::AIV, SubBlockNum, StageWeightIn,
                                  StageVfOut, Kub_, KernelSchedule, TileShapeL1, TileShapeL0, DtypeA, StrideA, DtypeB,
                                  StrideBOptionalTuple, DtypeBias, StrideBiasGm, DtypeC, StrideC> {
public:
    using Base = DavidUbAntiquantScmc<Stages, TileShapeUb, TileShapeReg, AscendC::AIV, SubBlockNum, StageWeightIn,
                                      StageVfOut, Kub_, KernelSchedule, TileShapeL1, TileShapeL0, DtypeA, StrideA,
                                      DtypeB, StrideBOptionalTuple, DtypeBias, StrideBiasGm, DtypeC, StrideC>;
    using TileShape = TileShapeL1;
    using Arguments = typename Base::Arguments;
    using StrideAntiquantScale = typename Base::StrideAntiquantScale;
    using StrideAntiquantOffset = typename Base::StrideAntiquantOffset;
    using StrideB = typename Base::StrideB;

    using StrideWeightInUb = AscendC::Std::tuple<uint32_t, _256, _16, _1, uint32_t>;
    using StrideWeightOutUb = AscendC::Std::tuple<_0, _0, _0, _0, uint32_t>;
    using StrideWeightL1 = AscendC::Std::tuple<uint32_t, _256, _16, _1, _0, uint32_t>;
    using StrideScaleUb = AscendC::Std::tuple<uint32_t, uint32_t, uint32_t>;

    static constexpr Mc2QuantType antiQuantType = GetQuantType<StrideAntiquantScale>();

private:
    // gm global tensor
    AscendC::GlobalTensor<DtypeB> tensorWGm_;
    AscendC::GlobalTensor<DtypeA> tensorScaleGm_;
    AscendC::GlobalTensor<DtypeA> tensorOffsetGm_;

    // ub local tensor
    AscendC::LocalTensor<int8_t> tensorWeightInUb;
    AscendC::LocalTensor<DtypeA> tensorWeightOutUb;
    AscendC::LocalTensor<DtypeA> tensorScaleUb;
    AscendC::LocalTensor<DtypeA> tensorOffsetUb;
    DtypeA scaleValue_;
    DtypeA offsetValue_;

    // l1 local tensor
    AscendC::LocalTensor<DtypeA> weightF16L1_;

    // TBuf
    TBuf<> ubBuffer_;
    TBuf<TPosition::TSCM> l1TBuf_;

public:
    DEVICE BlockMmad()
    {
    }

    struct Params {
        __gm__ DtypeB *bGmAddr = nullptr;
        __gm__ DtypeA *antiquantScaleGmAddr = nullptr;
        __gm__ DtypeA *antiquantOffsetGmAddr = nullptr;
        uint64_t groupSize;

        TileShapeL1 tileShapeL1;
        TileShapeUb tileShapeUb;
        TileShapeReg tileShapeReg;

        StrideB strideWGm;
        StrideWeightInUb strideWeightInUb;
        StrideWeightOutUb strideWeightOutUb;
        StrideWeightL1 strideWeightL1;

        StrideAntiquantScale strideScaleGm;
        StrideScaleUb strideScaleUb;

        TCubeTiling matmulTiling;

        uint64_t mainBlockCount;
        uint32_t firstTailBlockCount;
        uint32_t secondTailBlockCount;
        uint32_t mainBlockSize;
        uint32_t firstTailBlockSize;
        uint32_t secondTailBlockSize;

        uint32_t cubeBlockDimM;
        uint32_t cubeBlockDimN;
    };

public:
    using PipelineTuple =
        AscendC::Std::tuple<PipelineStageSingleCoreCopyInAdvance<Hardware::GM, Hardware::UB, StageWeightIn>,
                            PipelineStageSingleCore<Hardware::UB, Hardware::L1, StageVfOut>>;
    using PipelineStateTuple = AscendC::Std::tuple<PipelineState<StageWeightIn>, // gm->ub weight Load
                                                   PipelineState<StageWeightIn>, // gm->ub weight Compute
                                                   PipelineState<StageVfOut>,    // reg->ub weight(ub buffer)
                                                   PipelineState<2>              // aiv2aic
                                                   >;

private:
    constexpr static uint8_t STAGE_VF_OUT = (StageVfOut + 1) / 2 * 2;
    using PipelineAiv2Aic = PipelineStageMixCore<PIPE_MTE3, PIPE_MTE1, SubBlockNum>;
    using PipelineUb2L1 = typename AscendC::Std::tuple_element<1, PipelineTuple>::type;

    using PipelineStateWeightInLoad = typename AscendC::Std::tuple_element<0, PipelineStateTuple>::type;
    using PipelineStateWeightInCompute = typename AscendC::Std::tuple_element<1, PipelineStateTuple>::type;
    // element index is 2
    using PipelineStateUb2L1 = typename AscendC::Std::tuple_element<2, PipelineStateTuple>::type;
    // element index is 3
    using PipelineStateAiv2Aic = typename AscendC::Std::tuple_element<3, PipelineStateTuple>::type;

    ContinuousIterator<uint64_t> iterKLoad_;

public:
    template <class ProblemShape, typename TilingData>
    DEVICE static Params GetParams(ProblemShape &problemShape, Arguments const &args, TilingData const &tiling)
    {
        decltype(auto) sizeK = get<2>(problemShape);

        // 切分
        // NOTE 由于尾块重切分的nL1是变化的，stride得按照最大的计算
        uint32_t nL1 = Max(Max(tiling.mainBlockL1Size, tiling.firstTailBlockL1Size), tiling.secondTailBlockL1Size);
        uint32_t kaL1 = tiling.matmulTiling.baseK * tiling.matmulTiling.stepKa;
        uint32_t kbL1 = tiling.matmulTiling.baseK * tiling.matmulTiling.stepKb;

        // kBL1到kUb不能切分，如果切分ub->L1有两层dst stride
        // (n1, k1, k0, n0) n1切分成AIV0，AIV1的好处是vec写到cube的数据都是连续的
        //                  k1切分成AIV0, AIV1的坏处是ub->L1时有两层stride(k stride和n1 stride)
        // nUb=1/2kUb

        // n_first模板现状
        // ND n,k k1,n1,n0,k0 N方向到两个vector核计算
        // ND k,n n1,k1,k0,n0 K方向到两个vector核计算
        // Nz                 n1方向到两个vector核计算
        // NOTE 按照aiv0的最大大小计算stride
        uint32_t nUb = nL1 > 16 ? CeilAlign(nL1 >> 1, 16U) : nL1;
        constexpr uint32_t kUb = Kub_;
        return {
            .bGmAddr = args.bGmAddr,
            .antiquantScaleGmAddr = args.antiquantScaleGmAddr,
            .antiquantOffsetGmAddr = args.antiquantOffsetGmAddr,
            .groupSize = args.groupSize,

            // ub空间(232KB)
            //   weightIn  144KB = 48KB * 3 (128, 768)
            //   weightOut  64KB = 32KB * 2 (128, 128)
            //   scaleIn     12KB = 4KB * 3  (128, 4)
            //   offsetIn    12KB = 4KB * 3
            // L1空间(512KB)
            .tileShapeL1 = AscendC::Std::make_tuple(/*mL1*/ 0U, nL1, kaL1, kbL1),
            .tileShapeUb = AscendC::Std::make_tuple(nUb, kUb),
            .tileShapeReg = AscendC::Std::make_tuple(get<0>(TileShapeReg{}), get<1>(TileShapeReg{})),

            .strideWGm = args.strideB,
            .strideWeightInUb = Base::isTransB ? /*(k1,n1,n0,k0)*/ AscendC::Std::make_tuple(
                                                     CeilAlign(nUb, 16U) * 16, _256{}, _16{}, _1{},
                                                     static_cast<uint32_t>(nUb * kUb)) : /*(n1,k1,k0,n0)*/
                                    AscendC::Std::make_tuple(CeilAlign(kUb, 16U) * 16, _256{}, _16{}, _1{},
                                                             static_cast<uint32_t>(nUb * kUb)),
            // k,n (n1,k1,k0,n0) NOTE buffer按照bank大小交织
            // .strideWeightOutUb = AscendC::Std::make_tuple(kReg * 16, _256{}, _16{}, _1{}, static_cast<uint32_t>(_256
            // / sizeof(DtypeA))),
            .strideWeightOutUb =
                AscendC::Std::make_tuple(_0{}, _0{}, _0{}, _0{}, static_cast<uint32_t>(256 / sizeof(DtypeA))),

            // aic (256, 256) (n1, k1, k0, n0) (16, 16, 16, 16)
            // aiv (128, 256)                  ( 8, 16, 16, 16)
            //   vf (128, 64)                  ( 8,  4, 16, 16)
            // aiv (128, 256)
            .strideWeightL1 =
                AscendC::Std::make_tuple(Min(kbL1, static_cast<uint32_t>(sizeK)) * 16, _256{}, _16{}, _1{}, _0{},
                                         static_cast<uint32_t>(512 * 1024 / 2 /*b16*/ - nL1 * kbL1)),

            .strideScaleGm = args.strideAntiquantScale,
            // NOTE 只支持Pergroup
            // NOTE tiling需要保证kUb是args.groupSize的倍数，尾块没关系
            .strideScaleUb =
                Base::isTransB ?
                    /*(n,gn)*/ AscendC::Std::make_tuple(static_cast<uint32_t>(kUb / args.groupSize), 1U,
                                                        static_cast<uint32_t>(nUb * kUb / args.groupSize)) : /*(gn,n)*/
                    AscendC::Std::make_tuple(1U, nUb,
                                             static_cast<uint32_t>(nUb * kUb / args.groupSize)), // n, k, buffer

            .matmulTiling = tiling.matmulTiling,

            .mainBlockCount = tiling.mainBlockCount,
            .firstTailBlockCount = tiling.firstTailBlockCount,
            .secondTailBlockCount = tiling.secondTailBlockCount,
            .mainBlockSize = tiling.mainBlockL1Size,
            .firstTailBlockSize = tiling.firstTailBlockL1Size,
            .secondTailBlockSize = tiling.secondTailBlockL1Size,
            .cubeBlockDimM = tiling.cubeBlockDimM,
            .cubeBlockDimN = tiling.cubeBlockDimN,
        };
    }

    template <class ProblemShape>
    DEVICE static Params GetParams(ProblemShape &problemShape, Arguments const &args)
    {
    }

    template <class ProblemShape>
    DEVICE auto Init(ProblemShape const &problemShape, Params const &params)
    {
#if defined(__CCE_KT_TEST__)
        if ASCEND_IS_AIC {
            return;
        }
#endif
        tensorWGm_.SetGlobalBuffer(params.bGmAddr);
        tensorScaleGm_.SetGlobalBuffer(params.antiquantScaleGmAddr);
        if constexpr (Base::hasAntiQuantOffset) {
            tensorOffsetGm_.SetGlobalBuffer(params.antiquantOffsetGmAddr);
        }

        GetTPipePtr()->InitBuffer(ubBuffer_, 248 * 1024);
        GetTPipePtr()->InitBuffer(l1TBuf_, 512 * 1024);

        // UB
        // 128KB
        tensorWeightInUb = ubBuffer_.Get<int8_t>();
        // 32KB * 2
        tensorWeightOutUb =
            ubBuffer_.GetWithOffset<DtypeA>(/*size*/ 32 * 1024 * STAGE_VF_OUT / sizeof(DtypeA) /*elems*/,
                                            /*offset*/ 128 * 1024 /*B*/);
        // nub 128 k 1024/512 -> scale/offset 128*32
        // 8KB * 2
        tensorScaleUb = ubBuffer_.GetWithOffset<DtypeA>(/*size*/ 16 * 1024 / sizeof(DtypeA) /*elems*/,
                                                        /*offset*/ 128 * 1024 + 32 * 1024 * STAGE_VF_OUT /*B*/);
        tensorOffsetUb =
            ubBuffer_.GetWithOffset<DtypeA>(/*size*/ 16 * 1024 / sizeof(DtypeA) /*elems*/,
                                            /*offset*/ 128 * 1024 + 32 * 1024 * STAGE_VF_OUT + 16 * 1024 /*B*/);

#if defined(__CCE_KT_TEST__)
        uint32_t sizeWeightIn = 128 * 1024 / StageWeightIn;
        ASCENDC_ASSERT(sizeWeightIn <= (64 * 1024), { X_LOG("size of weight(%d) in should <= 64KB", sizeWeightIn); });
        uint32_t sizeWeightOut = CeilAlign(64 * 256 * 2, 32);
        uint32_t sizeScaleIn = 8 * 1024 / sizeof(DtypeA);

        uint32_t ubSize = (sizeWeightIn + 2 * sizeScaleIn) * StageWeightIn + sizeWeightOut * STAGE_VF_OUT;
        ASCENDC_ASSERT(ubSize <= (248 * 1024), { X_LOG("size of ub(%d) in should <= 248KB", ubSize); });
#endif
        // L1
        weightF16L1_ = l1TBuf_.Get<DtypeA>();
        // NOTE 由于存在尾块场景，并且L1上要求数据是连续的，所以aiv1的地址偏移需要动态修改

        decltype(auto) sizeK = get<2>(problemShape);
        iterKLoad_.Update(sizeK, get<1>(params.tileShapeUb));
    }

    // 提取的辅助函数：执行权重和反量化数据的拷贝
    template <typename ProblemShape, typename Iterators, typename Params>
    DEVICE void CopyWeightAndAntiquant(ProblemShape const &problemShape, Iterators const &iters, Params const &params,
                                       uint64_t pipelineIndex)
    {
        decltype(auto) sizeN = get<1>(problemShape);
        decltype(auto) sizeK = get<2>(problemShape);
        decltype(auto) iterNLoad = get<1>(iters);

        CopyWeightGmToUb(crd2idx(params.strideWGm, CeilDiv(*iterNLoad, 16UL), CeilDiv(*iterKLoad_, 16UL), _0{}, _0{}),
                         crd2idx(params.strideWeightInUb, _0{}, _0{}, _0{}, _0{}, pipelineIndex), iterNLoad.Size(),
                         iterKLoad_.Size(), sizeN, sizeK, get<0>(params.tileShapeUb), get<1>(params.tileShapeUb));

        CopyAntiquantGmToUb(crd2idx(params.strideScaleGm, *iterNLoad, *iterKLoad_ / params.groupSize),
                            crd2idx(params.strideScaleUb, _0{}, _0{}, pipelineIndex), iterNLoad.Size(),
                            CeilDiv(iterKLoad_.Size(), params.groupSize), sizeN, sizeK, get<0>(params.tileShapeUb),
                            CeilDiv(static_cast<uint64_t>(get<1>(params.tileShapeUb)), params.groupSize));
    }

    // 提取的辅助函数：更新迭代器状态
    template <typename Iterators>
    DEVICE void UpdateIteratorState(Iterators &iters)
    {
        decltype(auto) iterMLoad = get<0>(iters);
        decltype(auto) iterNLoad = get<1>(iters);

        ++iterKLoad_;
        if (iterKLoad_.End()) {
            iterKLoad_.Reset();
            ++iterNLoad;
            if (iterNLoad.End()) {
                iterNLoad.Reset();
                ++iterMLoad;
            }
        }
    }

    // 优化后的LoadInAdvance函数
    template <typename ProblemShape, typename Iterators>
    DEVICE void LoadInAdvance(ProblemShape const &problemShape, PipelineTuple const &pipelines,
                              PipelineStateTuple &states, Params const &params, Iterators &iters)
    {
#if defined(__CCE_KT_TEST__)
        if ASCEND_IS_AIC {
            return;
        }
#endif
        decltype(auto) pipelineGm2Ub = get<0>(pipelines);
        decltype(auto) pipelineStateWeightInLoad = get<0>(states);

        pipelineGm2Ub.ProducerWait(pipelineStateWeightInLoad);
        CopyWeightAndAntiquant(problemShape, iters, params, pipelineStateWeightInLoad.index());
        pipelineGm2Ub.ProducerRelease(pipelineStateWeightInLoad);
        ++pipelineStateWeightInLoad;

        UpdateIteratorState(iters);
    }

    // 优化后的Process函数
    template <typename ProblemShape, typename Iterators>
    DEVICE void Process(ProblemShape const &problemShape, PipelineTuple const &pipelines, PipelineStateTuple &states,
                        Params const &params, Iterators &itersLoad, Iterators const &itersCompute)
    {
        decltype(auto) sizeK = get<2>(problemShape);
        decltype(auto) iterNCompute = get<1>(itersCompute);

        // 提取处理循环体
        for (ContinuousIterator<uint64_t> iterKCompute(sizeK, get<1>(params.tileShapeUb)); !iterKCompute.End();
             ++iterKCompute) {
            ProcessSingleKIteration(problemShape, pipelines, states, params, itersLoad, iterKCompute, iterNCompute);
        }
    }

    // 提取的处理单个K迭代的函数
    template <typename ProblemShape, typename Iterators>
    DEVICE void ProcessSingleKIteration(ProblemShape const &problemShape, PipelineTuple const &pipelines,
                                        PipelineStateTuple &states, Params const &params, Iterators &itersLoad,
                                        ContinuousIterator<uint64_t> const &iterKCompute,
                                        decltype(get<1>(Iterators())) const &iterNCompute)
    {
        decltype(auto) pipelineGm2Ub = get<0>(pipelines);
        decltype(auto) pipelineUb2L1 = get<1>(pipelines);
        decltype(auto) pipelineStateWeightInLoad = get<0>(states);
        decltype(auto) pipelineStateWeightInCompute = get<1>(states);
        decltype(auto) pipelineStateUb2L1 = get<2>(states);
        decltype(auto) pipelineStateAiv2Aic = get<3>(states);
        decltype(auto) iterMLoad = get<0>(itersLoad);
        decltype(auto) iterNLoad = get<1>(itersLoad);

        pipelineGm2Ub.ProducerWait(pipelineStateWeightInLoad);
        if (!iterNLoad.End() && !iterMLoad.End()) {
            CopyWeightAndAntiquant(problemShape, itersLoad, params, pipelineStateWeightInLoad.index());
            UpdateIteratorState(itersLoad);
        }
        pipelineGm2Ub.ProducerRelease(pipelineStateWeightInLoad);
        ++pipelineStateWeightInLoad;

        pipelineGm2Ub.ConsumerWait(pipelineStateWeightInCompute);
        ConsumeUbInL1(iterKCompute, iterNCompute, params, pipelineStateWeightInCompute, pipelineStateAiv2Aic,
                      pipelineUb2L1, pipelineStateUb2L1);
        pipelineGm2Ub.ConsumerRelease(pipelineStateWeightInCompute);
        ++pipelineStateWeightInCompute;
    }

    DEVICE void ClearPipeline(PipelineTuple const &pipelines, PipelineStateTuple &states)
    {
        decltype(auto) pipelineGm2Ub = get<0>(pipelines);
        decltype(auto) pipelineUb2L1 = get<1>(pipelines);

        decltype(auto) pipelineStateWeightInLoad = get<0>(states);

        decltype(auto) pipelineStateAiv2Aic = get<3>(states);

        PipelineAiv2Aic::Clear(pipelineStateAiv2Aic);
        pipelineGm2Ub.Clear(pipelineStateWeightInLoad);
        pipelineUb2L1.Clear();
    }

private:
    DEVICE void CopyWeightGmToUb(uint64_t offsetWGm, uint64_t offsetWUb, uint64_t tileSizeN, uint64_t tileSizeK,
                                 uint64_t srcSizeN, uint64_t srcSizeK, uint64_t dstSizeN, uint64_t dstSizeK)
    {
        if (unlikely(tileSizeN == 0)) {
            return;
        }
        // NOTE tensorWeightInUb固定为int8类型，真实类型是int4时，需要换算
        if constexpr (IsSameType<DtypeB, int4b_t>::value) {
            offsetWUb = offsetWUb >> 1;
        }
        if constexpr (!Base::isWeightNz) {
            if constexpr (!Base::isTransB) {
                CopyGmToUbIntervalDataCopy(tensorWeightInUb[offsetWUb].template ReinterpretCast<DtypeB>(),
                                           tensorWGm_[offsetWGm], tileSizeK, tileSizeN, dstSizeK, srcSizeN);
            } else {
                CopyGmToUbIntervalDataCopy(tensorWeightInUb[offsetWUb].template ReinterpretCast<DtypeB>(),
                                           tensorWGm_[offsetWGm], tileSizeN, tileSizeK, dstSizeN, srcSizeK);
            }
        } else {
            if constexpr (!Base::isTransB) {
                // (n1, k1, k0, n0)
                CopyGmToUbIntervalDataCopy(tensorWeightInUb[offsetWUb].template ReinterpretCast<DtypeB>(),
                                           tensorWGm_[offsetWGm], CeilDiv(tileSizeN, static_cast<uint64_t>(BLOCK_CUBE)),
                                           CeilAlign(tileSizeK, static_cast<uint64_t>(BLOCK_CUBE)) * BLOCK_CUBE,
                                           dstSizeK * BLOCK_CUBE, srcSizeK * BLOCK_CUBE);
            } else {
                // (k1, n1, n0, k0)
                CopyGmToUbIntervalDataCopy(tensorWeightInUb[offsetWUb].template ReinterpretCast<DtypeB>(),
                                           tensorWGm_[offsetWGm], CeilDiv(tileSizeN, static_cast<uint64_t>(BLOCK_CUBE)),
                                           CeilAlign(tileSizeK, static_cast<uint64_t>(BLOCK_CUBE)) * BLOCK_CUBE,
                                           dstSizeN * BLOCK_CUBE, srcSizeN * BLOCK_CUBE);
            }
        }
    }

    DEVICE void CopyAntiquantGmToUb(uint64_t offsetScaleGm, uint64_t offsetScaleUb, uint64_t tileSizeN,
                                    uint64_t tileSizeK, uint64_t srcSizeN, uint64_t srcSizeK, uint64_t dstSizeN,
                                    uint64_t dstSizeK)
    {
        if (unlikely(tileSizeN == 0)) {
            return;
        }
        if constexpr (antiQuantType == Mc2QuantType::PER_CHANNEL) {
            DataCopyPad2D(tensorScaleUb[offsetScaleUb], tensorScaleGm_[offsetScaleGm], 1, tileSizeN,
                          CeilAlign(dstSizeN, static_cast<uint64_t>(VECTOR_REG_WIDTH)), srcSizeN);
            if constexpr (Base::hasAntiQuantOffset) {
                DataCopyPad2D(tensorOffsetUb[offsetScaleUb], tensorOffsetGm_[offsetScaleGm], 1, tileSizeN,
                              CeilAlign(dstSizeN, static_cast<uint64_t>(VECTOR_REG_WIDTH)), srcSizeN);
            }
        } else if constexpr (antiQuantType == Mc2QuantType::PER_TENSOR) {
            scaleValue_ = tensorScaleGm_.GetValue(0);
            if constexpr (Base::hasAntiQuantOffset) {
                offsetValue_ = tensorOffsetGm_.GetValue(0);
            }
        } else if constexpr (Base::isTransB) {
            // PERGROUP (n, k)
        } else {
            // PERGROUP (k, n)
            DataCopyPadExtParams<DtypeA> padParams;
            DataCopyExtParams intriParams;
            intriParams.blockCount = tileSizeK;
            intriParams.blockLen = tileSizeN * sizeof(DtypeA);
            intriParams.srcStride = (srcSizeN - tileSizeN) * sizeof(DtypeA);
            intriParams.dstStride = (dstSizeN - tileSizeN) * sizeof(DtypeA) / BLOCK_SIZE;
            X_LOG("DataCopyPad blockCount %d blockLen %d srcStride %d dstStride %d", intriParams.blockCount,
                  intriParams.blockLen, intriParams.srcStride, intriParams.dstStride);
#if defined(__CCE_KT_TEST__)
            ASCENDC_ASSERT(intriParams.blockLen > 0,
                           { X_LOG("blockLen[%d] should be larger than 0.", intriParams.blockLen); });
#endif
            DataCopyPad(tensorScaleUb[offsetScaleUb], tensorScaleGm_[offsetScaleGm], intriParams, padParams);
            if constexpr (Base::hasAntiQuantOffset) {
                DataCopyPad(tensorOffsetUb[offsetScaleUb], tensorOffsetGm_[offsetScaleGm], intriParams, padParams);
            }
        }
    }

    template <typename IterN, typename IterK>
    DEVICE void ConsumeUbInL1(IterK const &iterKUb, IterN const &iterN, Params const &params,
                              PipelineStateWeightInCompute const &pipelineStateWeightIn,
                              PipelineStateAiv2Aic &pipelineStateAiv2Aic, PipelineUb2L1 const &pipelineUb2L1,
                              PipelineStateUb2L1 &pipelineStateUb2L1)
    {
        // ub->l1 k
        for (ContinuousIterator<uint32_t> iterKUb2L1(/*stop*/ iterKUb.Size(), /*step*/ get<3>(params.tileShapeL1));
             !iterKUb2L1.End(); ++iterKUb2L1) {
            PipelineAiv2Aic::ProducerWaitAfterStages(pipelineStateAiv2Aic);
            ProduceL1InUb(iterKUb2L1, iterN, params, pipelineStateWeightIn, pipelineUb2L1, pipelineStateUb2L1,
                          pipelineStateAiv2Aic);
            PipelineAiv2Aic::ProducerRelease();
            ++pipelineStateAiv2Aic;
        }
    }

    template <typename IterN, typename IterK>
    DEVICE void ProduceL1InUb(IterK const &iterKUb2L1, IterN const &iterN, Params const &params,
                              PipelineStateWeightInCompute const &pipelineStateWeightIn,
                              PipelineUb2L1 const &pipelineUb2L1, PipelineStateUb2L1 &pipelineStateUb2L1,
                              PipelineStateAiv2Aic const &pipelineStateAiv2Aic)
    {
        // 两处需要用到
        //   reg2Ub
        //   reg2L1
        for (ContinuousIterator<uint32_t> iterKReg2L1(/*start*/ 0, /*stop*/ iterKUb2L1.Size(),
                                                      /*step*/ get<1>(params.tileShapeReg), *iterKUb2L1);
             !iterKReg2L1.End(); ++iterKReg2L1) {
            for (ContinuousIterator<uint32_t> iterNReg2L1(/*stop*/ iterN.Size(), /*step*/ get<0>(params.tileShapeReg));
                 !iterNReg2L1.End(); ++iterNReg2L1) {
                pipelineUb2L1.ProducerWait(pipelineStateUb2L1);
                AntiQuantComputeKNGroupWeightNz<DtypeA, Base::hasAntiQuantOffset, Kub_, STAGE_VF_OUT>(
                    tensorWeightInUb[crd2idx(params.strideWeightInUb, CeilDiv(*iterNReg2L1, 16U),
                                             CeilDiv(*iterKReg2L1, 16U), _0{}, _0{}, pipelineStateWeightIn.index()) >>
                                     1],
                    tensorScaleUb[crd2idx(params.strideScaleUb, *iterNReg2L1, *iterKReg2L1 / params.groupSize,
                                          pipelineStateWeightIn.index())],
                    tensorOffsetUb[crd2idx(params.strideScaleUb, *iterNReg2L1, *iterKReg2L1 / params.groupSize,
                                           pipelineStateWeightIn.index())],
                    tensorWeightOutUb[crd2idx(params.strideWeightOutUb, CeilDiv(*iterNReg2L1, 16U),
                                              CeilDiv(*iterKReg2L1, 16U), _0{}, _0{}, pipelineStateUb2L1.index())],
                    iterNReg2L1.Size(), iterKReg2L1.Size(), get<0>(params.tileShapeUb), params.groupSize);
                pipelineUb2L1.ProducerRelease(pipelineStateUb2L1);

                pipelineUb2L1.ConsumerWait(pipelineStateUb2L1);
                // NOTE aic:aiv 1:2场景 k方向有尾块时，为了保持ub->l1仍然使用一条指令完成，需要使用动态stride
                auto strideWeightL1 = AscendC::Std::make_tuple(
                    CeilDiv(iterKReg2L1.Size(), 16U) * _256{}, _256{}, _16{}, _1{},
                    /*aiv0aiv1 stride*/ iterN.PrevHalfSize() * CeilAlign(iterKReg2L1.Size(), 16U),
                    /*l1 buf stride*/ get<5>(params.strideWeightL1));
                CopyUbToL1<STAGE_VF_OUT>(
                    crd2idx(params.strideWeightOutUb, _0{}, _0{}, _0{}, _0{}, pipelineStateUb2L1.index()),
                    crd2idx(strideWeightL1, CeilDiv(*iterNReg2L1, 16U), CeilDiv(*iterKReg2L1 - *iterKUb2L1, 16U), _0{},
                            _0{}, AscendC::GetSubBlockIdx(), pipelineStateAiv2Aic.index()),
                    iterNReg2L1.Size(), iterKReg2L1.Size(), iterKUb2L1.Size(), iterN.Size());
                pipelineUb2L1.ConsumerRelease(pipelineStateUb2L1);
                ++pipelineStateUb2L1;
            }
        }
    }

    constexpr static uint64_t WEIGHT_F16_UB_NZ_STRIDE = 65;
    template <uint32_t UB_WEIGHT_OUTPUT_F16BUFFER_NUM>
    DEVICE void CopyUbToL1(uint64_t offsetWeightUb, uint64_t offsetWeightL1, uint64_t tileSizeN, uint64_t tileSizeK,
                           uint64_t nSize, uint64_t kSize)
    {
        static constexpr uint64_t ONE_REG_ELEM = VECTOR_REG_WIDTH / sizeof(DtypeA);
        static constexpr uint64_t ONE_BLOCK_ELEM = BLOCK_CUBE / sizeof(uint8_t);

        DataCopyParams params;
        if constexpr (!Base::isWeightNz) {
            if constexpr (Base::isTransB) {
                CopyUbToL1IntervalDataCopy(
                    weightF16L1_[offsetWeightL1], tensorWeightOutUb[offsetWeightUb], CeilDiv(tileSizeK, ONE_BLOCK_ELEM),
                    tileSizeN, CeilAlign(nSize, ONE_BLOCK_ELEM) - tileSizeN, WEIGHT_F16_UB_NZ_STRIDE - tileSizeN);
            } else {
                CopyUbToL1IntervalDataCopy(
                    weightF16L1_[offsetWeightL1], tensorWeightOutUb[offsetWeightUb], CeilDiv(tileSizeN, ONE_BLOCK_ELEM),
                    tileSizeK, CeilAlign(kSize, ONE_BLOCK_ELEM) - tileSizeK, WEIGHT_F16_UB_NZ_STRIDE - tileSizeK);
            }
        } else {
            if constexpr (Base::isTransB) {
            } else {
                // (n1, k1, k0, n0)
                CopyUbToL1IntervalDataCopy(weightF16L1_[offsetWeightL1], tensorWeightOutUb[offsetWeightUb],
                                           CeilAlign(tileSizeK, ONE_BLOCK_ELEM) * CeilAlign(tileSizeN, ONE_BLOCK_ELEM) /
                                               ONE_REG_ELEM,
                                           ONE_REG_ELEM / ONE_BLOCK_ELEM, 0,
                                           (UB_WEIGHT_OUTPUT_F16BUFFER_NUM - 1) * ONE_REG_ELEM / ONE_BLOCK_ELEM);
            }
        }
    }
};
} // namespace Mc2WeightQuantBatchMatmulV2::Arch35::Catlass
#endif
