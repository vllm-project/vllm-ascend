// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include "kda_rms_norm_gated_tiling_data.h"

namespace KdaNormGate {
using namespace AscendC;
constexpr MicroAPI::CastTrait TO_FLOAT = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_NONE};
constexpr MicroAPI::CastTrait TO_BF16 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::DivSpecificMode DIV_MODE = {MicroAPI::MaskMergeMode::ZEROING, false};

template <typename Weight, bool SigmoidOnly>
__aicore__ inline void ComputeTile(__local_mem__ bfloat16_t *x, __local_mem__ bfloat16_t *gate,
    __local_mem__ Weight *weight, __local_mem__ bfloat16_t *output, uint16_t rows, float epsilon)
{
    __VEC_SCOPE__ {
        MicroAPI::RegTensor<bfloat16_t> rawX0, rawX1, rawG0, rawG1, result;
        MicroAPI::RegTensor<Weight> rawW0, rawW1;
        MicroAPI::RegTensor<float> x0, x1, g0, g1, w0, w1, square0, square1;
        MicroAPI::RegTensor<float> sum, inverse, exp0, exp1, one;
        auto mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(one, 1.0f, mask);
        if constexpr (AscendC::IsSameType<Weight, float>::value) {
            MicroAPI::DataCopy(w0, weight);
            MicroAPI::DataCopy(w1, weight + 64);
        } else {
            MicroAPI::DataCopy<Weight, MicroAPI::LoadDist::DIST_UNPACK_B16>(rawW0, weight);
            MicroAPI::DataCopy<Weight, MicroAPI::LoadDist::DIST_UNPACK_B16>(rawW1, weight + 64);
            MicroAPI::Cast<float, Weight, TO_FLOAT>(w0, rawW0, mask);
            MicroAPI::Cast<float, Weight, TO_FLOAT>(w1, rawW1, mask);
        }
        for (uint16_t row = 0; row < rows; ++row) {
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(rawX0, x + row * 128);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(rawX1, x + row * 128 + 64);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(rawG0, gate + row * 128);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::LoadDist::DIST_UNPACK_B16>(rawG1, gate + row * 128 + 64);
            MicroAPI::Cast<float, bfloat16_t, TO_FLOAT>(x0, rawX0, mask);
            MicroAPI::Cast<float, bfloat16_t, TO_FLOAT>(x1, rawX1, mask);
            MicroAPI::Cast<float, bfloat16_t, TO_FLOAT>(g0, rawG0, mask);
            MicroAPI::Cast<float, bfloat16_t, TO_FLOAT>(g1, rawG1, mask);
            MicroAPI::Mul(square0, x0, x0, mask);
            MicroAPI::Mul(square1, x1, x1, mask);
            MicroAPI::Add(sum, square0, square1, mask);
            MicroAPI::ReduceSum(sum, sum, mask);
            MicroAPI::Duplicate(inverse, sum, mask);
            MicroAPI::Muls(inverse, inverse, 1.0f / 128.0f, mask);
            MicroAPI::Adds(inverse, inverse, epsilon, mask);
            MicroAPI::Sqrt(inverse, inverse, mask);
            MicroAPI::Div<float, &DIV_MODE>(inverse, one, inverse, mask);
            MicroAPI::Mul(x0, x0, inverse, mask);
            MicroAPI::Mul(x1, x1, inverse, mask);
            MicroAPI::Mul(x0, x0, w0, mask);
            MicroAPI::Mul(x1, x1, w1, mask);
            // Kimi K3 uses sigmoid; preserve SiLU for the existing general op.
            if constexpr (!SigmoidOnly) {
                MicroAPI::Mul(x0, x0, g0, mask);
                MicroAPI::Mul(x1, x1, g1, mask);
            }
            MicroAPI::Muls(exp0, g0, -1.0f, mask);
            MicroAPI::Muls(exp1, g1, -1.0f, mask);
            MicroAPI::Exp(exp0, exp0, mask);
            MicroAPI::Exp(exp1, exp1, mask);
            MicroAPI::Adds(exp0, exp0, 1.0f, mask);
            MicroAPI::Adds(exp1, exp1, 1.0f, mask);
            MicroAPI::Div<float, &DIV_MODE>(exp0, one, exp0, mask);
            MicroAPI::Div<float, &DIV_MODE>(exp1, one, exp1, mask);
            MicroAPI::Mul(x0, x0, exp0, mask);
            MicroAPI::Mul(x1, x1, exp1, mask);
            MicroAPI::Cast<bfloat16_t, float, TO_BF16>(result, x0, mask);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(output + row * 128, result, mask);
            MicroAPI::Cast<bfloat16_t, float, TO_BF16>(result, x1, mask);
            MicroAPI::DataCopy<bfloat16_t, MicroAPI::StoreDist::DIST_PACK_B32>(output + row * 128 + 64, result, mask);
        }
    }
}

template <typename Weight, bool SigmoidOnly> class Kernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gate, GM_ADDR weight, GM_ADDR output,
        const optiling::KdaRmsNormGatedTilingData &tiling)
    {
        data = tiling;
        xGm.SetGlobalBuffer((__gm__ bfloat16_t *)x);
        gateGm.SetGlobalBuffer((__gm__ bfloat16_t *)gate);
        weightGm.SetGlobalBuffer((__gm__ Weight *)weight);
        outputGm.SetGlobalBuffer((__gm__ bfloat16_t *)output);
        const uint32_t bytes = data.tileTokens * data.heads * 128 * sizeof(bfloat16_t);
        pipe.InitBuffer(xQueue, 2, bytes);
        pipe.InitBuffer(gateQueue, 2, bytes);
        pipe.InitBuffer(outputQueue, 2, bytes);
        pipe.InitBuffer(weightBuffer, 512);
        DataCopy(weightBuffer.Get<Weight>(), weightGm, 128);
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
    }

    __aicore__ inline uint32_t Tokens(int64_t tile)
    {
        const int64_t remaining = data.tokens - tile * data.tileTokens;
        return static_cast<uint32_t>(remaining < data.tileTokens ? remaining : data.tileTokens);
    }

    __aicore__ inline void CopyIn(int64_t tile)
    {
        const uint16_t tokens = Tokens(tile);
        const uint32_t rowBytes = data.heads * 128 * sizeof(bfloat16_t);
        auto x = xQueue.AllocTensor<bfloat16_t>();
        auto gate = gateQueue.AllocTensor<bfloat16_t>();
        DataCopyPad(x, xGm[tile * data.tileTokens * data.xTokenStride],
            DataCopyExtParams{tokens, rowBytes,
                static_cast<uint32_t>((data.xTokenStride - data.heads * 128) * 2), 0, 0},
            DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
        DataCopyPad(gate, gateGm[tile * data.tileTokens * data.gateTokenStride],
            DataCopyExtParams{tokens, rowBytes,
                static_cast<uint32_t>((data.gateTokenStride - data.heads * 128) * 2), 0, 0},
            DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
        xQueue.EnQue(x);
        gateQueue.EnQue(gate);
    }

    __aicore__ inline void Compute(int64_t tile)
    {
        auto x = xQueue.DeQue<bfloat16_t>();
        auto gate = gateQueue.DeQue<bfloat16_t>();
        auto output = outputQueue.AllocTensor<bfloat16_t>();
        ComputeTile<Weight, SigmoidOnly>((__local_mem__ bfloat16_t *)x.GetPhyAddr(),
            (__local_mem__ bfloat16_t *)gate.GetPhyAddr(),
            (__local_mem__ Weight *)weightBuffer.Get<Weight>().GetPhyAddr(),
            (__local_mem__ bfloat16_t *)output.GetPhyAddr(),
            static_cast<uint16_t>(Tokens(tile) * data.heads), data.epsilon);
        xQueue.FreeTensor(x);
        gateQueue.FreeTensor(gate);
        outputQueue.EnQue(output);
    }

    __aicore__ inline void CopyOut(int64_t tile)
    {
        auto output = outputQueue.DeQue<bfloat16_t>();
        DataCopy(outputGm[tile * data.tileTokens * data.heads * 128], output,
            Tokens(tile) * data.heads * 128);
        outputQueue.FreeTensor(output);
    }

    __aicore__ inline void Process()
    {
        const int64_t tiles = (data.tokens + data.tileTokens - 1) / data.tileTokens;
        const int64_t first = GetBlockIdx();
        if (first >= tiles) return;
        CopyIn(first);
        for (int64_t tile = first; tile < tiles; tile += GetBlockNum()) {
            const int64_t next = tile + GetBlockNum();
            if (next < tiles) CopyIn(next);
            Compute(tile);
            CopyOut(tile);
        }
    }
private:
    TPipe pipe;
    TQue<TPosition::VECIN, 2> xQueue, gateQueue;
    TQue<TPosition::VECOUT, 2> outputQueue;
    TBuf<TPosition::VECCALC> weightBuffer;
    GlobalTensor<bfloat16_t> xGm, gateGm, outputGm;
    GlobalTensor<Weight> weightGm;
    optiling::KdaRmsNormGatedTilingData data;
};
}

extern "C" __global__ __aicore__ void kda_rms_norm_gated(GM_ADDR x, GM_ADDR gate, GM_ADDR weight,
    GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::KdaRmsNormGatedTilingData);
    GET_TILING_DATA(data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(0)) {
        KdaNormGate::Kernel<bfloat16_t, false> kernel;
        kernel.Init(x, gate, weight, output, data);
        kernel.Process();
    } else if (TILING_KEY_IS(1)) {
        KdaNormGate::Kernel<float, false> kernel;
        kernel.Init(x, gate, weight, output, data);
        kernel.Process();
    } else if (TILING_KEY_IS(2)) {
        KdaNormGate::Kernel<bfloat16_t, true> kernel;
        kernel.Init(x, gate, weight, output, data);
        kernel.Process();
    } else if (TILING_KEY_IS(3)) {
        KdaNormGate::Kernel<float, true> kernel;
        kernel.Init(x, gate, weight, output, data);
        kernel.Process();
    }
}
