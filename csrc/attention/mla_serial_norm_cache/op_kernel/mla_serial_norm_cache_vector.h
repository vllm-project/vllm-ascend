// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"
using namespace AscendC;
constexpr uint32_t ROPE_DIM = 64;
constexpr uint32_t MAX_NORM_WIDTH = 4096;
constexpr float NORM_EPSILON = 1.0e-5f;
class MlaVectorStage {
public:
    __aicore__ inline MlaVectorStage(TPipe &pipe) : pipe_(pipe) {}

    __aicore__ inline void Init()
    {
        pipe_.InitBuffer(input_, 1, MAX_NORM_WIDTH * sizeof(bfloat16_t));
        pipe_.InitBuffer(gamma_, 1, MAX_NORM_WIDTH * sizeof(bfloat16_t));
        pipe_.InitBuffer(output_, 1, MAX_NORM_WIDTH * sizeof(bfloat16_t));
        pipe_.InitBuffer(work_, 5 * MAX_NORM_WIDTH * sizeof(float));
    }

    __aicore__ inline LocalTensor<bfloat16_t> Read(GM_ADDR src, uint64_t offset, uint32_t n,
                                                uint32_t rows = 1, uint32_t stride = 0)
    {
        GlobalTensor<bfloat16_t> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(src));
        auto x = input_.AllocTensor<bfloat16_t>();
        const uint32_t rowStride = stride == 0 ? n : stride;
        DataCopyPad(x, gm[offset], DataCopyExtParams{static_cast<uint16_t>(rows), n * 2,
                    (rowStride - n) * 2, 0, 0},
                    DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
        input_.EnQue(x);
        return input_.DeQue<bfloat16_t>();
    }

    __aicore__ inline void Write(GM_ADDR dst, uint64_t offset, LocalTensor<bfloat16_t> x,
                                uint32_t n, uint32_t rows = 1, uint32_t stride = 0)
    {
        GlobalTensor<bfloat16_t> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dst));
        output_.EnQue(x);
        auto ready = output_.DeQue<bfloat16_t>();
        const uint32_t rowStride = stride == 0 ? n : stride;
        DataCopyPad(gm[offset], ready, DataCopyExtParams{static_cast<uint16_t>(rows), n * 2,
                    0, (rowStride - n) * 2, 0});
        output_.FreeTensor(ready);
    }

    __aicore__ inline void Norm(GM_ADDR src, uint64_t from, GM_ADDR scale,
                               GM_ADDR dst, uint64_t to, uint32_t n)
    {
        auto x = Read(src, from, n);
        auto f = work_.Get<float>();
        auto square = f[MAX_NORM_WIDTH];
        auto gf = f[MAX_NORM_WIDTH * 2];
        auto scratch = f[MAX_NORM_WIDTH * 3];
        auto sum = f[MAX_NORM_WIDTH * 4];
        if (scale != cachedScale_ || n != cachedWidth_) {
            GlobalTensor<bfloat16_t> gg;
            gg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(scale));
            auto g = gamma_.AllocTensor<bfloat16_t>();
            DataCopy(g, gg, n);
            gamma_.EnQue(g);
            g = gamma_.DeQue<bfloat16_t>();
            Cast(gf, g, RoundMode::CAST_NONE, n);
            gamma_.FreeTensor(g);
            cachedScale_ = scale;
            cachedWidth_ = n;
        }
        Cast(f, x, RoundMode::CAST_NONE, n);
        Mul(square, f, f, n);
        ReduceSum(sum, square, scratch, n);
        Muls(sum, sum, 1.0f / n, 1);
        Adds(sum, sum, NORM_EPSILON, 1);
        Rsqrt(sum, sum, 1);
        PipeBarrier<PIPE_V>();
        const auto event = pipe_.FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(event);
        WaitFlag<HardEvent::V_S>(event);
        Muls(f, f, sum.GetValue(0), n);
        Mul(f, f, gf, n);
        auto y = output_.AllocTensor<bfloat16_t>();
        Cast(y, f, RoundMode::CAST_RINT, n);
        Write(dst, to, y, n);
        input_.FreeTensor(x);
    }

    __aicore__ inline void Rope(GM_ADDR src, uint64_t from, GM_ADDR cos, GM_ADDR sin,
                               uint32_t token, GM_ADDR dst, uint64_t to, uint32_t rows = 1,
                               uint32_t srcStride = ROPE_DIM, uint32_t dstStride = ROPE_DIM)
    {
        auto x = Read(src, from, ROPE_DIM, rows, srcStride);
        auto f = work_.Get<float>();
        auto cf = f[MAX_NORM_WIDTH];
        // Work slot 2 preserves normalized gamma across rows within this invocation.
        auto sf = f[MAX_NORM_WIDTH * 3];
        auto rotated = f[MAX_NORM_WIDTH * 4];
        Cast(f, x, RoundMode::CAST_NONE, ROPE_DIM * rows);
        if (cos != nullptr) {
            GlobalTensor<bfloat16_t> cg, sg;
            cg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(cos));
            sg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(sin));
            auto cs = gamma_.AllocTensor<bfloat16_t>();
            DataCopy(cs, cg[token * ROPE_DIM], ROPE_DIM);
            DataCopy(cs[ROPE_DIM], sg[token * ROPE_DIM], ROPE_DIM);
            gamma_.EnQue(cs);
            cs = gamma_.DeQue<bfloat16_t>();
            Cast(cf, cs, RoundMode::CAST_NONE, ROPE_DIM);
            Cast(sf, cs[ROPE_DIM], RoundMode::CAST_NONE, ROPE_DIM);
            constexpr uint8_t ROW_BLOCKS = ROPE_DIM * sizeof(float) / 32;
            const uint8_t repeats = static_cast<uint8_t>(rows);
            const UnaryRepeatParams rowRepeat{1, 1, ROW_BLOCKS, ROW_BLOCKS};
            const BinaryRepeatParams broadcast{1, 1, 1, ROW_BLOCKS, ROW_BLOCKS, 0};
            Muls(rotated, f[ROPE_DIM / 2], -1.0f, uint64_t(ROPE_DIM / 2), repeats, rowRepeat);
            Muls(rotated[ROPE_DIM / 2], f, 1.0f, uint64_t(ROPE_DIM / 2), repeats, rowRepeat);
            Mul(rotated, rotated, sf, uint64_t(ROPE_DIM), repeats, broadcast);
            Mul(f, f, cf, uint64_t(ROPE_DIM), repeats, broadcast);
            Add(f, f, rotated, ROPE_DIM * rows);
            gamma_.FreeTensor(cs);
        }
        auto y = output_.AllocTensor<bfloat16_t>();
        Cast(y, f, RoundMode::CAST_RINT, ROPE_DIM * rows);
        Write(dst, to, y, ROPE_DIM, rows, dstStride);
        input_.FreeTensor(x);
    }

private:
    TPipe &pipe_;
    TQue<TPosition::VECIN, 1> input_, gamma_;
    TQue<TPosition::VECOUT, 1> output_;
    TBuf<TPosition::VECCALC> work_;
    GM_ADDR cachedScale_ = nullptr;
    uint32_t cachedWidth_ = 0;
};
