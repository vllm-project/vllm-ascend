// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"
#include "compressor_serial_params.h"

namespace CompressorSerial {
using namespace AscendC;
template<typename Output> class Finalize {
public:
    __aicore__ inline void Run(GM_ADDR pooled, GM_ADDR gamma, GM_ADDR sin, GM_ADDR cos,
                              GM_ADDR output, __gm__ const CompressorSerialParams* p)
    {
        TPipe pipe;
        TBuf<TPosition::VECCALC> storage;
        pipe.InitBuffer(storage, 16 * 512 * sizeof(float));
        auto x = storage.Get<float>();
        auto weight = x[512], square = x[1024], scratch = x[1536];
        auto scaled = x[2048], rotated = x[2560], sine = x[3072], cosine = x[3584];
        auto offsets = storage.Get<uint32_t>()[4096];
        auto signs = x[4160];
        auto result = storage.Get<Output>()[12288];
        GlobalTensor<uint32_t> gmOffsets;
        GlobalTensor<float> gmSigns;
        gmOffsets.SetGlobalBuffer((__gm__ uint32_t*)p->ropeOffsets);
        gmSigns.SetGlobalBuffer((__gm__ float*)p->ropeSigns);
        DataCopy(offsets, gmOffsets, 64);
        DataCopy(signs, gmSigns, 64);
        GlobalTensor<float> src, gmGamma, gmSin, gmCos;
        GlobalTensor<Output> dst;
        src.SetGlobalBuffer((__gm__ float*)pooled);
        gmGamma.SetGlobalBuffer((__gm__ float*)gamma);
        gmSin.SetGlobalBuffer((__gm__ float*)sin);
        gmCos.SetGlobalBuffer((__gm__ float*)cos);
        dst.SetGlobalBuffer((__gm__ Output*)output);
        uint32_t d = p->dim;
        union { uint32_t bits; float value; } eps;
        eps.bits = static_cast<uint32_t>(p->epsBits);
        DataCopy(weight, gmGamma, d);
        PipeBarrier<PIPE_ALL>();
        for (int64_t row = GetBlockIdx(); row < p->outputRows; row += GetBlockNum()) {
            DataCopy(x, src[row * d], d);
            DataCopy(sine, gmSin[row * 64], 64);
            DataCopy(cosine, gmCos[row * 64], 64);
            PipeBarrier<PIPE_ALL>();
            Mul(square, x, x, d);
            PipeBarrier<PIPE_V>();
            ReduceSum(scratch, square, rotated, d);
            PipeBarrier<PIPE_ALL>();
            float mean = scratch.GetValue(0) / d + eps.value;
            Duplicate(square, mean, 8);
            PipeBarrier<PIPE_V>();
            Rsqrt(square, square, 8);
            PipeBarrier<PIPE_ALL>();
            Muls(scaled, x, square.GetValue(0), d);
            PipeBarrier<PIPE_V>();
            Mul(scaled, scaled, weight, d);
            PipeBarrier<PIPE_ALL>();
            Gather(rotated, scaled[d - 64], offsets, 0, 64);
            PipeBarrier<PIPE_V>();
            Mul(rotated, rotated, signs, 64);
            PipeBarrier<PIPE_V>();
            Mul(square, scaled[d - 64], cosine, 64);
            Mul(rotated, rotated, sine, 64);
            PipeBarrier<PIPE_V>();
            Add(scaled[d - 64], square, rotated, 64);
            PipeBarrier<PIPE_V>();
            Cast(result, scaled, RoundMode::CAST_RINT, d);
            PipeBarrier<PIPE_ALL>();
            DataCopy(dst[row * d], result, d);
            PipeBarrier<PIPE_ALL>();
        }
    }
};
} // namespace CompressorSerial
