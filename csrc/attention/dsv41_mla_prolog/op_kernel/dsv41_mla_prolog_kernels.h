/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef DSV41_MLA_PROLOG_KERNELS_H_
#define DSV41_MLA_PROLOG_KERNELS_H_

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/quant_matmul_multistage_workspace.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "kernel_operator.h"
#include "tiling/platform/platform_ascendc.h"

#include "dsv41_mla_prolog_tiling_data.h"

#include <cstdint>

using namespace Catlass;


namespace Dsv41 {

using Bf16MatmulDispatch = Gemm::MmadAtlasA2Pingpong<true>;
using Bf16L1Tile = GemmShape<128, 256, 256>;
using Bf16L0Tile = GemmShape<128, 256, 64>;
using Bf16A = Gemm::GemmType<bfloat16_t, layout::RowMajor>;
using Bf16B = Gemm::GemmType<bfloat16_t, layout::ColumnMajor>;
using Bf16C = Gemm::GemmType<float, layout::RowMajor>;
using Bf16BlockMmad = Gemm::Block::BlockMmad<
    Bf16MatmulDispatch, Bf16L1Tile, Bf16L0Tile, Bf16A, Bf16B, Bf16C>;
using Bf16Scheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

struct DualProjectionKernel {
    using ElementA = typename Bf16BlockMmad::ElementA;
    using LayoutA = typename Bf16BlockMmad::LayoutA;
    using ElementB = typename Bf16BlockMmad::ElementB;
    using LayoutB = typename Bf16BlockMmad::LayoutB;
    using ElementC = typename Bf16BlockMmad::ElementC;
    using LayoutC = typename Bf16BlockMmad::LayoutC;

    struct Params {
        GemmCoord qaShape;
        GemmCoord kvShape;
        GM_ADDR x;
        LayoutA layoutA;
        GM_ADDR wQa;
        LayoutB layoutQaB;
        GM_ADDR wKv;
        LayoutB layoutKvB;
        GM_ADDR qa;
        LayoutC layoutQaC;
        GM_ADDR kv;
        LayoutC layoutKvC;
        uint32_t projectionSplits;
    };

    struct Arguments {
        GemmCoord qaShape;
        GemmCoord kvShape;
        GM_ADDR x;
        GM_ADDR wQa;
        GM_ADDR wKv;
        GM_ADDR qa;
        GM_ADDR kv;
        uint32_t projectionSplits;
    };

    static bool CanImplement(const Arguments &args)
    {
        return args.x != nullptr && args.wQa != nullptr && args.wKv != nullptr &&
            args.qa != nullptr && args.kv != nullptr &&
            args.projectionSplits > 0 &&
            args.projectionSplits <= kMaxProjectionSplits &&
            kHidden % args.projectionSplits == 0;
    }

    static size_t GetWorkspaceSize(const Arguments &)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *)
    {
        return Params{
            args.qaShape,
            args.kvShape,
            args.x,
            LayoutA::template MakeLayout<ElementA>(args.qaShape.m(), args.qaShape.k()),
            args.wQa,
            LayoutB::template MakeLayout<ElementB>(args.qaShape.k(), args.qaShape.n()),
            args.wKv,
            LayoutB::template MakeLayout<ElementB>(args.kvShape.k(), args.kvShape.n()),
            args.qa,
            LayoutC::template MakeLayout<ElementC>(args.qaShape.m(), args.qaShape.n()),
            args.kv,
            LayoutC::template MakeLayout<ElementC>(args.kvShape.m(), args.kvShape.n()),
            args.projectionSplits};
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(const Params &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(const Params &params)
    {
        const GemmCoord qaSplitShape{
            params.qaShape.m(), params.qaShape.n(),
            params.qaShape.k() / params.projectionSplits};
        const GemmCoord kvSplitShape{
            params.kvShape.m(), params.kvShape.n(),
            params.kvShape.k() / params.projectionSplits};
        Bf16Scheduler qaScheduler(
            qaSplitShape, MakeCoord(Bf16L1Tile::M, Bf16L1Tile::N));
        Bf16Scheduler kvScheduler(
            kvSplitShape, MakeCoord(Bf16L1Tile::M, Bf16L1Tile::N));
        const uint32_t qaLoops = qaScheduler.GetCoreLoops();
        const uint32_t kvLoops = kvScheduler.GetCoreLoops();
        const uint32_t totalLoops =
            (qaLoops + kvLoops) * params.projectionSplits;

        Arch::Resource<Arch::AtlasA2> resource;
        Bf16BlockMmad blockMmad(resource);
        AscendC::GlobalTensor<ElementA> gmX;
        AscendC::GlobalTensor<ElementB> gmWQa;
        AscendC::GlobalTensor<ElementB> gmWKv;
        AscendC::GlobalTensor<ElementC> gmQa;
        AscendC::GlobalTensor<ElementC> gmKv;
        gmX.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.x));
        gmWQa.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.wQa));
        gmWKv.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.wKv));
        gmQa.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.qa));
        gmKv.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.kv));

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < totalLoops;
             loopIdx += AscendC::GetBlockNum()) {
            const uint32_t splitId = loopIdx % params.projectionSplits;
            const uint32_t combinedLoop = loopIdx / params.projectionSplits;
            const bool isQa = combinedLoop < qaLoops;
            const uint32_t localLoop = isQa ? combinedLoop : combinedLoop - qaLoops;
            const GemmCoord problemShape = isQa ? qaSplitShape : kvSplitShape;
            Bf16Scheduler scheduler(
                problemShape, MakeCoord(Bf16L1Tile::M, Bf16L1Tile::N));
            const GemmCoord blockCoord = scheduler.GetBlockCoord(localLoop);
            const GemmCoord actualBlockShape =
                scheduler.GetActualBlockShape(blockCoord);
            const MatrixCoord offsetA{
                blockCoord.m() * Bf16L1Tile::M,
                splitId * problemShape.k() + blockCoord.k() * Bf16L1Tile::K};
            const MatrixCoord offsetB{
                splitId * problemShape.k() + blockCoord.k() * Bf16L1Tile::K,
                blockCoord.n() * Bf16L1Tile::N};
            const MatrixCoord offsetC{
                blockCoord.m() * Bf16L1Tile::M,
                blockCoord.n() * Bf16L1Tile::N};
            if (isQa) {
                const uint64_t splitOffset =
                    static_cast<uint64_t>(splitId) * params.qaShape.m() * params.qaShape.n();
                blockMmad(
                    gmX[params.layoutA.GetOffset(offsetA)], params.layoutA,
                    gmWQa[params.layoutQaB.GetOffset(offsetB)], params.layoutQaB,
                    gmQa[splitOffset + params.layoutQaC.GetOffset(offsetC)], params.layoutQaC,
                    actualBlockShape);
            } else {
                const uint64_t splitOffset =
                    static_cast<uint64_t>(splitId) * params.kvShape.m() * params.kvShape.n();
                blockMmad(
                    gmX[params.layoutA.GetOffset(offsetA)], params.layoutA,
                    gmWKv[params.layoutKvB.GetOffset(offsetB)], params.layoutKvB,
                    gmKv[splitOffset + params.layoutKvC.GetOffset(offsetC)], params.layoutKvC,
                    actualBlockShape);
            }
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(const Params &)
    {}
};

struct NormQuantKernel {
    struct Params {
        GM_ADDR qa;
        GM_ADDR kv;
        GM_ADDR kvNorm;
        GM_ADDR gammaQa;
        GM_ADDR gammaKv;
        GM_ADDR qInt8;
        GM_ADDR qScale;
        GM_ADDR wScale;
        GM_ADDR wScaleFp;
        GM_ADDR qr;
        uint32_t t;
        uint32_t projectionSplits;
        float eps;
    };
    using Arguments = Params;

    static bool CanImplement(const Arguments &args)
    {
        return args.qa != nullptr && args.kv != nullptr && args.kvNorm != nullptr &&
            args.gammaQa != nullptr &&
            args.gammaKv != nullptr && args.qInt8 != nullptr && args.qScale != nullptr &&
            args.wScale != nullptr && args.wScaleFp != nullptr && args.qr != nullptr &&
            args.t > 0 && args.t <= kMaxT &&
            args.projectionSplits > 0 &&
            args.projectionSplits <= kMaxProjectionSplits;
    }

    static size_t GetWorkspaceSize(const Arguments &)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *)
    {
        return args;
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(const Params &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(const Params &)
    {}

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(const Params &params)
    {
        constexpr uint32_t kGammaQaBf16 = 0;
        constexpr uint32_t kGammaKvBf16 = 2560;
        constexpr uint32_t kGammaQaFp = 3584;
        constexpr uint32_t kGammaKvFp = 8704;
        constexpr uint32_t kRowBf16 = 10752;
        constexpr uint32_t kNormBf16 = 13312;
        constexpr uint32_t kFp = 15872;
        constexpr uint32_t kTmp = 20992;
        constexpr uint32_t kScratch = 26112;
        constexpr uint32_t kQuantInt32 = 42496;
        constexpr uint32_t kQuantHalf = 47616;
        constexpr uint32_t kQuantInt8 = 50176;
        constexpr uint32_t kScalar = 51456;
        constexpr uint32_t kScaleOut = 51712;

        Arch::Resource<Arch::AtlasA2> resource;
        auto gammaQaBf16 = resource.ubBuf.GetBufferByByte<bfloat16_t>(kGammaQaBf16);
        auto gammaKvBf16 = resource.ubBuf.GetBufferByByte<bfloat16_t>(kGammaKvBf16);
        auto gammaQaFp = resource.ubBuf.GetBufferByByte<float>(kGammaQaFp);
        auto gammaKvFp = resource.ubBuf.GetBufferByByte<float>(kGammaKvFp);
        auto rowBf16 = resource.ubBuf.GetBufferByByte<bfloat16_t>(kRowBf16);
        auto normBf16 = resource.ubBuf.GetBufferByByte<bfloat16_t>(kNormBf16);
        auto fp = resource.ubBuf.GetBufferByByte<float>(kFp);
        auto tmp = resource.ubBuf.GetBufferByByte<float>(kTmp);
        auto scratch = resource.ubBuf.GetBufferByByte<float>(kScratch);
        auto quantInt32 = resource.ubBuf.GetBufferByByte<int32_t>(kQuantInt32);
        auto quantHalf = resource.ubBuf.GetBufferByByte<half>(kQuantHalf);
        auto quantInt8 = resource.ubBuf.GetBufferByByte<int8_t>(kQuantInt8);
        auto scalar = resource.ubBuf.GetBufferByByte<float>(kScalar);
        auto scaleOut = resource.ubBuf.GetBufferByByte<float>(kScaleOut);

        AscendC::GlobalTensor<float> gmQa;
        AscendC::GlobalTensor<float> gmKv;
        AscendC::GlobalTensor<bfloat16_t> gmKvNorm;
        AscendC::GlobalTensor<bfloat16_t> gmGammaQa;
        AscendC::GlobalTensor<bfloat16_t> gmGammaKv;
        AscendC::GlobalTensor<int8_t> gmQInt8;
        AscendC::GlobalTensor<float> gmQScale;
        AscendC::GlobalTensor<bfloat16_t> gmWScale;
        AscendC::GlobalTensor<float> gmWScaleFp;
        AscendC::GlobalTensor<bfloat16_t> gmQr;
        gmQa.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.qa));
        gmKv.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.kv));
        gmKvNorm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.kvNorm));
        gmGammaQa.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.gammaQa));
        gmGammaKv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.gammaKv));
        gmQInt8.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(params.qInt8));
        gmQScale.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.qScale));
        gmWScale.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.wScale));
        gmWScaleFp.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.wScaleFp));
        gmQr.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.qr));

        auto *pipe = GetTPipePtr();
        const int32_t mte2V = pipe->AllocEventID<AscendC::HardEvent::MTE2_V>();
        const int32_t vMte3 = pipe->AllocEventID<AscendC::HardEvent::V_MTE3>();
        const int32_t mte3V = pipe->AllocEventID<AscendC::HardEvent::MTE3_V>();
        const int32_t vMte2 = pipe->AllocEventID<AscendC::HardEvent::V_MTE2>();
        const int32_t vS = pipe->AllocEventID<AscendC::HardEvent::V_S>();
        const int32_t sV = pipe->AllocEventID<AscendC::HardEvent::S_V>();
        const int32_t sMte3 = pipe->AllocEventID<AscendC::HardEvent::S_MTE3>();

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum =
            AscendC::GetBlockNum() * AscendC::GetSubBlockNum();

        // This conversion is independent of the projection output. Run it while
        // the AIC side is still producing the projection to hide its latency.
        constexpr uint32_t kScaleChunk = 256;
        for (uint32_t chunk = blockIdx; chunk < kQbOut / kScaleChunk;
             chunk += blockNum) {
            const uint32_t start = chunk * kScaleChunk;
            AscendC::DataCopy(gammaQaBf16, gmWScale[start], kScaleChunk);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2V);
            AscendC::Cast(
                gammaQaFp, gammaQaBf16,
                AscendC::RoundMode::CAST_NONE, kScaleChunk);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vMte3);
            AscendC::DataCopy(gmWScaleFp[start], gammaQaFp, kScaleChunk);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3V);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        AscendC::DataCopy(gammaQaBf16, gmGammaQa, 512);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2V);
        AscendC::DataCopy(gammaQaBf16[512], gmGammaQa[512], 512);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2V);
        AscendC::DataCopy(gammaQaBf16[1024], gmGammaQa[1024], 256);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2V);
        AscendC::DataCopy(gammaKvBf16, gmGammaKv, kKv);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2V);
        AscendC::Cast(gammaQaFp, gammaQaBf16, AscendC::RoundMode::CAST_NONE, 512);
        AscendC::Cast(gammaQaFp[512], gammaQaBf16[512], AscendC::RoundMode::CAST_NONE, 512);
        AscendC::Cast(gammaQaFp[1024], gammaQaBf16[1024], AscendC::RoundMode::CAST_NONE, 256);
        AscendC::Cast(gammaKvFp, gammaKvBf16, AscendC::RoundMode::CAST_NONE, kKv);
        AscendC::PipeBarrier<PIPE_V>();

        Arch::CrossCoreFlag projectionReady{4};
        Arch::CrossCoreWaitFlag(projectionReady);
        Arch::CrossCoreBarrier<0x0, PIPE_MTE2>();

        const uint32_t workCount = params.t * 2U;
        for (uint32_t work = blockIdx; work < workCount; work += blockNum) {
            const bool isQ = work < params.t;
            const uint32_t row = isQ ? work : work - params.t;
            const uint32_t dim = isQ ? kQa : kKv;
            const uint64_t offset = static_cast<uint64_t>(row) * dim;
            const uint64_t splitStride = static_cast<uint64_t>(params.t) * dim;
            auto gmProjection = isQ ? gmQa : gmKv;
            const uint32_t vectorChunks = (dim + 511U) / 512U;
            for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                const uint32_t start = chunkId * 512U;
                const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                AscendC::DataCopy(fp[start], gmProjection[offset + start], count);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2V);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2V);
                for (uint32_t splitId = 1;
                     splitId < params.projectionSplits; ++splitId) {
                    AscendC::DataCopy(
                        tmp[start],
                        gmProjection[splitId * splitStride + offset + start], count);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2V);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2V);
                    AscendC::Add(fp[start], fp[start], tmp[start], count);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vMte2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vMte2);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (params.t == 1U) {
                for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                    const uint32_t start = chunkId * 512U;
                    const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                    AscendC::Muls(
                        fp[start], fp[start], kProjectionUlpGuard, count);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
            for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                const uint32_t start = chunkId * 512U;
                const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                AscendC::Cast(
                    rowBf16[start], fp[start], AscendC::RoundMode::CAST_RINT, count);
            }
            AscendC::PipeBarrier<PIPE_V>();

            if (isQ && params.t > 1U) {
                for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                    const uint32_t start = chunkId * 512U;
                    const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                    AscendC::Muls(
                        fp[start], fp[start], kProjectionGeneralGuard, count);
                    AscendC::Cast(
                        normBf16[start], fp[start], AscendC::RoundMode::CAST_RINT, count);
                }
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                    const uint32_t start = chunkId * 512U;
                    const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                    AscendC::Cast(
                        fp[start], normBf16[start], AscendC::RoundMode::CAST_NONE, count);
                    AscendC::Mul(tmp[start], fp[start], fp[start], count);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
            for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                const uint32_t start = chunkId * 512U;
                const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                AscendC::Cast(
                    fp[start], rowBf16[start], AscendC::RoundMode::CAST_NONE, count);
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (!isQ || params.t == 1U) {
                for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                    const uint32_t start = chunkId * 512U;
                    const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                    AscendC::Mul(tmp[start], fp[start], fp[start], count);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::ReduceSum(scalar, tmp, scratch, static_cast<int32_t>(dim));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(vS);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(vS);
            const float invDim = isQ ? (1.0F / 1280.0F) : (1.0F / 512.0F);
            AscendC::Muls(scalar, scalar, invDim, 1);
            AscendC::Adds(scalar, scalar, params.eps, 1);
            AscendC::Sqrt(scalar[8], scalar, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(scalar[16], 1.0F, 1);
            AscendC::Div(scalar, scalar[16], scalar[8], 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(vS);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(vS);
            const float rstd = scalar.GetValue(0) *
                (params.t == 1U ? kRmsNormUlpGuard : 1.0F);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(sV);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(sV);
            for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                const uint32_t start = chunkId * 512U;
                const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                AscendC::Muls(fp[start], fp[start], rstd, count);
                AscendC::Mul(
                    fp[start], fp[start],
                    (isQ ? gammaQaFp : gammaKvFp)[start], count);
            }
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                const uint32_t start = chunkId * 512U;
                const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                AscendC::Cast(
                    normBf16[start], fp[start], AscendC::RoundMode::CAST_RINT, count);
            }
            AscendC::PipeBarrier<PIPE_V>();

            if (isQ) {
                for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                    const uint32_t start = chunkId * 512U;
                    const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                    AscendC::Cast(
                        fp[start], normBf16[start], AscendC::RoundMode::CAST_NONE, count);
                }
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                    const uint32_t start = chunkId * 512U;
                    const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                    AscendC::Abs(tmp[start], fp[start], count);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ReduceMax(
                    scalar, tmp, scratch, static_cast<int32_t>(dim), false);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_S>(vS);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(vS);
                const float rowMax = scalar.GetValue(0);
                float outputScale = 0.0F;
                float quantMultiplier = 0.0F;
                if (rowMax != 0.0F) {
                    AscendC::Muls(scaleOut, scalar, 1.0F / 127.0F, 1);
                    AscendC::Duplicate(scalar[8], 127.0F, 1);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Div(scalar[16], scalar[8], scalar, 1);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_S>(vS);
                    AscendC::WaitFlag<AscendC::HardEvent::V_S>(vS);
                    outputScale = scaleOut.GetValue(0);
                    quantMultiplier = scalar.GetValue(16);
                }
                AscendC::SetFlag<AscendC::HardEvent::S_V>(sV);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(sV);
                if (rowMax == 0.0F) {
                    for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                        const uint32_t start = chunkId * 512U;
                        const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                        AscendC::Duplicate(fp[start], 0.0F, count);
                    }
                } else {
                    for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                        const uint32_t start = chunkId * 512U;
                        const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                        AscendC::Muls(fp[start], fp[start], quantMultiplier, count);
                    }
                }
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                    const uint32_t start = chunkId * 512U;
                    const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                    AscendC::Cast(
                        quantInt32[start], fp[start],
                        AscendC::RoundMode::CAST_RINT, count);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetDeqScale(half(1.0F));
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                    const uint32_t start = chunkId * 512U;
                    const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                    AscendC::Cast(
                        quantHalf[start], quantInt32[start],
                        AscendC::RoundMode::CAST_ROUND, count);
                }
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                    const uint32_t start = chunkId * 512U;
                    const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                    AscendC::Cast(
                        quantInt8[start], quantHalf[start],
                        AscendC::RoundMode::CAST_TRUNC, count);
                }
                scaleOut.SetValue(0, outputScale);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(sMte3);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(sMte3);
                AscendC::DataCopyParams scaleCopy{1, sizeof(float), 0, 0};
                AscendC::DataCopyPad(gmQScale[row], scaleOut, scaleCopy);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vMte3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vMte3);
                AscendC::DataCopy(gmQInt8[offset], quantInt8, 512);
                AscendC::DataCopy(gmQInt8[offset + 512], quantInt8[512], 512);
                AscendC::DataCopy(gmQInt8[offset + 1024], quantInt8[1024], 256);
                AscendC::DataCopy(gmQr[offset], normBf16, 512);
                AscendC::DataCopy(gmQr[offset + 512], normBf16[512], 512);
                AscendC::DataCopy(gmQr[offset + 1024], normBf16[1024], 256);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3V);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3V);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vMte3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vMte3);
                AscendC::DataCopy(gmKvNorm[offset], normBf16, dim);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3V);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3V);
            }
        }

        pipe->ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2V);
        pipe->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vMte3);
        pipe->ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3V);
        pipe->ReleaseEventID<AscendC::HardEvent::V_MTE2>(vMte2);
        pipe->ReleaseEventID<AscendC::HardEvent::V_S>(vS);
        pipe->ReleaseEventID<AscendC::HardEvent::S_V>(sV);
        pipe->ReleaseEventID<AscendC::HardEvent::S_MTE3>(sMte3);
    }
};

using QuantDispatch = Gemm::MmadAtlasA2PreloadAsyncWithCallback<1, 2, 2, 2, 1, true, true>;
using QuantL1Tile = GemmShape<128, 256, 512>;
using QuantL0Tile = GemmShape<128, 256, 128>;
using QuantA = Gemm::GemmType<int8_t, layout::RowMajor>;
using QuantB = Gemm::GemmType<int8_t, layout::zN>;
using QuantC = Gemm::GemmType<int32_t, layout::RowMajor>;
using QuantBlockMmad = Gemm::Block::BlockMmad<
    QuantDispatch, QuantL1Tile, QuantL0Tile, QuantA, QuantB, QuantC>;
using QuantEpilogueDispatch = Epilogue::EpilogueAtlasA2PerTokenDequant<2>;
using QuantScale = Gemm::GemmType<float, layout::VectorLayout>;
using QuantTokenScale = Gemm::GemmType<float, layout::VectorLayout>;
using QuantD = Gemm::GemmType<bfloat16_t, layout::RowMajor>;
using QuantRowMul = Gemm::GemmType<float, layout::RowMajor>;
using QuantOneBlock = Gemm::GemmType<float, layout::RowMajor>;
using QuantColumnMul = Gemm::GemmType<float, layout::RowMajor>;
using QuantEpilogueTile = MatrixShape<32, 256>;
using QuantTileRowMul = Epilogue::Tile::TileRowBroadcastMul<
    Arch::AtlasA2, QuantRowMul, QuantEpilogueTile>;
using QuantTileOneBlock = Epilogue::Tile::TileBroadcastOneBlk<
    Arch::AtlasA2, QuantOneBlock, QuantEpilogueTile::ROW>;
using QuantTileColumnMul = Epilogue::Tile::TileOneBlkColumnBroadcastMul<
    Arch::AtlasA2, QuantColumnMul, QuantEpilogueTile>;
using QuantEpilogueCopy = Epilogue::Tile::TileCopy<
    Arch::AtlasA2, QuantC, QuantScale, QuantTokenScale, QuantD>;
using QuantTileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;
using QuantBlockEpilogue = Epilogue::Block::BlockEpilogue<
    QuantEpilogueDispatch, QuantC, QuantScale, QuantTokenScale, QuantD,
    QuantTileRowMul, QuantTileOneBlock, QuantTileColumnMul, QuantEpilogueCopy,
    QuantTileScheduler>;
using QuantScheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
using QuantMatmul = Gemm::Kernel::QuantMatmulMultiStageWorkspace<
    QuantBlockMmad, QuantBlockEpilogue, QuantScheduler, kWorkspaceStages>;

struct RopeScatterKernel {
    struct Params {
        GM_ADDR qOutput;
        GM_ADDR kv;
        GM_ADDR ropeCos;
        GM_ADDR ropeSin;
        GM_ADDR slots;
        GM_ADDR cache;
        uint64_t cacheStride0;
        uint64_t cacheStride1;
        uint32_t cacheBlockSize;
        uint32_t t;
    };
    using Arguments = Params;

    static bool CanImplement(const Arguments &args)
    {
        return args.qOutput != nullptr && args.kv != nullptr && args.ropeCos != nullptr &&
            args.ropeSin != nullptr && args.slots != nullptr && args.cache != nullptr &&
            args.t > 0 && args.t <= kMaxT;
    }

    static size_t GetWorkspaceSize(const Arguments &)
    {
        return 0;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *)
    {
        return args;
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(const Params &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(const Params &)
    {}

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(const Params &params)
    {
        constexpr uint32_t kHeadBatch = 4;
        constexpr uint32_t kBatchRope = kHeadBatch * kRopeDim;
        constexpr uint32_t kRow0 = 0;
        constexpr uint32_t kRow1 = 1024;
        constexpr uint32_t kX0 = 2048;
        constexpr uint32_t kX1 = 3072;
        constexpr uint32_t kRotated0 = 4096;
        constexpr uint32_t kRotated1 = 5120;
        constexpr uint32_t kOut0 = 6144;
        constexpr uint32_t kOut1 = 7168;
        constexpr uint32_t kCos = 8192;
        constexpr uint32_t kSin = 9216;
        constexpr uint32_t kSign = 10240;
        constexpr uint32_t kOffsets = 11264;

        Arch::Resource<Arch::AtlasA2> resource;
        AscendC::LocalTensor<bfloat16_t> rowLocal[2] = {
            resource.ubBuf.GetBufferByByte<bfloat16_t>(kRow0),
            resource.ubBuf.GetBufferByByte<bfloat16_t>(kRow1)};
        AscendC::LocalTensor<float> x[2] = {
            resource.ubBuf.GetBufferByByte<float>(kX0),
            resource.ubBuf.GetBufferByByte<float>(kX1)};
        AscendC::LocalTensor<float> rotated[2] = {
            resource.ubBuf.GetBufferByByte<float>(kRotated0),
            resource.ubBuf.GetBufferByByte<float>(kRotated1)};
        AscendC::LocalTensor<float> output[2] = {
            resource.ubBuf.GetBufferByByte<float>(kOut0),
            resource.ubBuf.GetBufferByByte<float>(kOut1)};
        auto cos = resource.ubBuf.GetBufferByByte<float>(kCos);
        auto sin = resource.ubBuf.GetBufferByByte<float>(kSin);
        auto sign = resource.ubBuf.GetBufferByByte<float>(kSign);
        auto offsets = resource.ubBuf.GetBufferByByte<uint32_t>(kOffsets);

        AscendC::GlobalTensor<bfloat16_t> gmQ;
        AscendC::GlobalTensor<bfloat16_t> gmKv;
        AscendC::GlobalTensor<float> gmCos;
        AscendC::GlobalTensor<float> gmSin;
        AscendC::GlobalTensor<int32_t> gmSlots;
        AscendC::GlobalTensor<bfloat16_t> gmCache;
        gmQ.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.qOutput));
        gmKv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.kv));
        gmCos.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.ropeCos));
        gmSin.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.ropeSin));
        gmSlots.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.slots));
        gmCache.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.cache));

        auto *pipe = GetTPipePtr();
        const int32_t inputReady[2] = {
            pipe->AllocEventID<AscendC::HardEvent::MTE2_V>(),
            pipe->AllocEventID<AscendC::HardEvent::MTE2_V>()};
        const int32_t vMte3 = pipe->AllocEventID<AscendC::HardEvent::V_MTE3>();
        const int32_t vMte2 = pipe->AllocEventID<AscendC::HardEvent::V_MTE2>();
        const int32_t slotFree[2] = {
            pipe->AllocEventID<AscendC::HardEvent::MTE3_MTE2>(),
            pipe->AllocEventID<AscendC::HardEvent::MTE3_MTE2>()};
        bool pending[2] = {false, false};

        for (uint32_t batch = 0; batch < kHeadBatch; ++batch) {
            for (uint32_t i = 0; i < kRopeDim / 2; ++i) {
                const uint32_t base = batch * kRopeDim;
                offsets.SetValue(base + 2U * i,
                    (base + 2U * i + 1U) * sizeof(float));
                offsets.SetValue(base + 2U * i + 1U,
                    (base + 2U * i) * sizeof(float));
                sign.SetValue(base + 2U * i, -1.0F);
                sign.SetValue(base + 2U * i + 1U, 1.0F);
            }
        }

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum =
            AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        for (uint32_t token = blockIdx; token < params.t; token += blockNum) {
            const uint64_t ropeOffset = static_cast<uint64_t>(token) * kRopeDim;
            for (uint32_t batch = 0; batch < kHeadBatch; ++batch) {
                AscendC::DataCopy(
                    cos[batch * kRopeDim], gmCos[ropeOffset], kRopeDim);
                AscendC::DataCopy(
                    sin[batch * kRopeDim], gmSin[ropeOffset], kRopeDim);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady[0]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady[0]);
            AscendC::Mul(sin, sin, sign, kBatchRope);
            AscendC::PipeBarrier<PIPE_V>();

            for (uint32_t head = 0; head < kHeads; head += kHeadBatch) {
                const uint32_t slot = (head / kHeadBatch) & 1U;
                if (pending[slot]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slotFree[slot]);
                    pending[slot] = false;
                }
                const uint64_t rowOffset =
                    (static_cast<uint64_t>(token) * kHeads + head) * kHeadDim;
                AscendC::DataCopy(
                    rowLocal[slot], gmQ[rowOffset + kNopeDim],
                    AscendC::DataCopyParams(kHeadBatch, 4U, 28U, 0U));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady[slot]);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady[slot]);

                AscendC::Cast(
                    x[slot], rowLocal[slot],
                    AscendC::RoundMode::CAST_NONE, kBatchRope);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(output[slot], x[slot], cos, kBatchRope);
                AscendC::Gather(rotated[slot], x[slot], offsets, 0U, kBatchRope);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::FusedMulAdd(
                    rotated[slot], sin, output[slot], kBatchRope);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(
                    rowLocal[slot], rotated[slot],
                    AscendC::RoundMode::CAST_RINT, kBatchRope);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vMte3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vMte3);

                AscendC::DataCopy(
                    gmQ[rowOffset + kNopeDim], rowLocal[slot],
                    AscendC::DataCopyParams(kHeadBatch, 4U, 0U, 28U));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slotFree[slot]);
                pending[slot] = true;
            }

            constexpr uint32_t kvSlot = 0;
            if (pending[kvSlot]) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slotFree[kvSlot]);
                pending[kvSlot] = false;
            }
            const uint64_t kvOffset = static_cast<uint64_t>(token) * kHeadDim;
            AscendC::DataCopy(rowLocal[kvSlot], gmKv[kvOffset], kHeadDim);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady[kvSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady[kvSlot]);
            AscendC::Cast(
                x[kvSlot], rowLocal[kvSlot][kNopeDim],
                AscendC::RoundMode::CAST_NONE, kRopeDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(output[kvSlot], x[kvSlot], cos, kRopeDim);
            AscendC::Gather(rotated[kvSlot], x[kvSlot], offsets, 0U, kRopeDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::FusedMulAdd(
                rotated[kvSlot], sin, output[kvSlot], kRopeDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(
                rowLocal[kvSlot][kNopeDim], rotated[kvSlot],
                AscendC::RoundMode::CAST_RINT, kRopeDim);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vMte3);
            const int32_t slot =
                gmSlots.GetValue(static_cast<uint64_t>(token));
            if (slot >= 0) {
                const uint32_t page =
                    static_cast<uint32_t>(slot) / params.cacheBlockSize;
                const uint32_t row =
                    static_cast<uint32_t>(slot) % params.cacheBlockSize;
                const uint64_t cacheOffset =
                    static_cast<uint64_t>(page) * params.cacheStride0 +
                    static_cast<uint64_t>(row) * params.cacheStride1;
                AscendC::DataCopy(
                    gmCache[cacheOffset], rowLocal[kvSlot], kHeadDim);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slotFree[kvSlot]);
                pending[kvSlot] = true;
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vMte2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vMte2);
        }

        for (uint32_t slot = 0; slot < 2; ++slot) {
            if (pending[slot]) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slotFree[slot]);
            }
            pipe->ReleaseEventID<AscendC::HardEvent::MTE2_V>(inputReady[slot]);
            pipe->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(slotFree[slot]);
        }
        pipe->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vMte3);
        pipe->ReleaseEventID<AscendC::HardEvent::V_MTE2>(vMte2);
    }
};

struct UnifiedMlaPrologKernel {
    struct Arguments {
        DualProjectionKernel::Arguments projection;
        NormQuantKernel::Arguments normQuant;
        QuantMatmul::Arguments quantMatmul;
        RopeScatterKernel::Arguments ropeScatter;
    };

    struct Params {
        DualProjectionKernel::Params projection;
        NormQuantKernel::Params normQuant;
        QuantMatmul::Params quantMatmul;
        RopeScatterKernel::Params ropeScatter;
    };

    static bool CanImplement(const Arguments &args)
    {
        return DualProjectionKernel::CanImplement(args.projection) &&
            NormQuantKernel::CanImplement(args.normQuant) &&
            QuantMatmul::CanImplement(args.quantMatmul) &&
            RopeScatterKernel::CanImplement(args.ropeScatter);
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return QuantMatmul::GetWorkspaceSize(args.quantMatmul);
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        return Params{
            DualProjectionKernel::ToUnderlyingArguments(args.projection, nullptr),
            NormQuantKernel::ToUnderlyingArguments(args.normQuant, nullptr),
            QuantMatmul::ToUnderlyingArguments(args.quantMatmul, workspace),
            RopeScatterKernel::ToUnderlyingArguments(args.ropeScatter, nullptr)};
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(const Params &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(const Params &params)
    {
        DualProjectionKernel projection;
        projection.template operator()<AscendC::AIC>(params.projection);
        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(projectionReady);

        Arch::CrossCoreWaitFlag(normReady);
        quantMatmul.template operator()<AscendC::AIC>(params.quantMatmul);
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(const Params &params)
    {
        NormQuantKernel normQuant;
        normQuant.template operator()<AscendC::AIV>(params.normQuant);
        Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(normReady);

        quantMatmul.template operator()<AscendC::AIV>(params.quantMatmul);
        if constexpr (kEnableRopeScatter) {
            Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            RopeScatterKernel ropeScatter;
            ropeScatter.template operator()<AscendC::AIV>(params.ropeScatter);
        }
    }

private:
    static constexpr Arch::FlagID kProjectionReadyFlag = 4;
    static constexpr Arch::FlagID kNormReadyFlag = 5;
    Arch::CrossCoreFlag projectionReady{kProjectionReadyFlag};
    Arch::CrossCoreFlag normReady{kNormReadyFlag};
    QuantMatmul quantMatmul;
};

} // namespace Dsv41

#endif  // DSV41_MLA_PROLOG_KERNELS_H_
