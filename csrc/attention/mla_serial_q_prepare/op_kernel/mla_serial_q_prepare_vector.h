// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t ROPE_DIM = 64;
constexpr uint32_t MAX_ROPE_ROWS = 128;
constexpr uint32_t MAX_ROPE_ELEMENTS = MAX_ROPE_ROWS * ROPE_DIM;

class MlaPrepareStage {
public:
    __aicore__ inline MlaPrepareStage(TPipe &pipe) : pipe_(pipe) {}

    __aicore__ inline void Init()
    {
        pipe_.InitBuffer(input_, 1, MAX_ROPE_ELEMENTS * sizeof(bfloat16_t));
        pipe_.InitBuffer(trig_, 1, 2 * ROPE_DIM * sizeof(bfloat16_t));
        pipe_.InitBuffer(output_, 1, MAX_ROPE_ELEMENTS * sizeof(bfloat16_t));
        pipe_.InitBuffer(work_, (2 * MAX_ROPE_ELEMENTS + 2 * ROPE_DIM) * sizeof(float));
    }

    // Reuse the RoPE queues; the packed BF16 values are copied without arithmetic.
    __aicore__ inline void Pack(GM_ADDR src, uint64_t from, GM_ADDR dst, uint64_t to,
                               uint32_t width, uint32_t rows)
    {
        GlobalTensor<bfloat16_t> source, target;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(src));
        target.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dst));
        auto x = input_.AllocTensor<bfloat16_t>();
        DataCopyPad(x, source[from], DataCopyExtParams{static_cast<uint16_t>(rows),
                    width * 2, ROPE_DIM * 2, 0, 0},
                    DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
        input_.EnQue(x);
        x = input_.DeQue<bfloat16_t>();
        auto y = output_.AllocTensor<bfloat16_t>();
        DataCopy(y, x, width * rows);
        output_.EnQue(y);
        y = output_.DeQue<bfloat16_t>();
        DataCopyPad(target[to], y, DataCopyExtParams{1, width * rows * 2, 0, 0, 0});
        output_.FreeTensor(y);
        input_.FreeTensor(x);
    }

    __aicore__ inline void Rope(GM_ADDR src, uint64_t from, GM_ADDR cos, GM_ADDR sin,
                               uint32_t token, GM_ADDR dst, uint64_t to, uint32_t rows,
                               uint32_t srcStride, uint32_t dstStride)
    {
        GlobalTensor<bfloat16_t> source, target;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(src));
        target.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(dst));
        auto x = input_.AllocTensor<bfloat16_t>();
        DataCopyPad(x, source[from], DataCopyExtParams{static_cast<uint16_t>(rows),
                    ROPE_DIM * 2, (srcStride - ROPE_DIM) * 2, 0, 0},
                    DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
        input_.EnQue(x);
        x = input_.DeQue<bfloat16_t>();
        auto f = work_.Get<float>();
        auto rotated = f[MAX_ROPE_ELEMENTS];
        auto cf = f[2 * MAX_ROPE_ELEMENTS];
        auto sf = f[2 * MAX_ROPE_ELEMENTS + ROPE_DIM];
        Cast(f, x, RoundMode::CAST_NONE, ROPE_DIM * rows);
        if (cos != nullptr) {
            GlobalTensor<bfloat16_t> cg, sg;
            cg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(cos));
            sg.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(sin));
            auto cs = trig_.AllocTensor<bfloat16_t>();
            DataCopy(cs, cg[token * ROPE_DIM], ROPE_DIM);
            DataCopy(cs[ROPE_DIM], sg[token * ROPE_DIM], ROPE_DIM);
            trig_.EnQue(cs);
            cs = trig_.DeQue<bfloat16_t>();
            Cast(cf, cs, RoundMode::CAST_NONE, ROPE_DIM);
            Cast(sf, cs[ROPE_DIM], RoundMode::CAST_NONE, ROPE_DIM);
            constexpr uint8_t ROW_BLOCKS = ROPE_DIM * sizeof(float) / 32;
            const auto repeats = static_cast<uint8_t>(rows);
            const UnaryRepeatParams rowRepeat{1, 1, ROW_BLOCKS, ROW_BLOCKS};
            const BinaryRepeatParams broadcast{1, 1, 1, ROW_BLOCKS, ROW_BLOCKS, 0};
            Muls(rotated, f[ROPE_DIM / 2], -1.0f, uint64_t(ROPE_DIM / 2), repeats, rowRepeat);
            Muls(rotated[ROPE_DIM / 2], f, 1.0f, uint64_t(ROPE_DIM / 2), repeats, rowRepeat);
            Mul(rotated, rotated, sf, uint64_t(ROPE_DIM), repeats, broadcast);
            Mul(f, f, cf, uint64_t(ROPE_DIM), repeats, broadcast);
            Add(f, f, rotated, ROPE_DIM * rows);
            trig_.FreeTensor(cs);
        }
        auto y = output_.AllocTensor<bfloat16_t>();
        Cast(y, f, RoundMode::CAST_RINT, ROPE_DIM * rows);
        output_.EnQue(y);
        y = output_.DeQue<bfloat16_t>();
        DataCopyPad(target[to], y, DataCopyExtParams{static_cast<uint16_t>(rows),
                    ROPE_DIM * 2, 0, (dstStride - ROPE_DIM) * 2, 0});
        output_.FreeTensor(y);
        input_.FreeTensor(x);
    }

private:
    TPipe &pipe_;
    TQue<TPosition::VECIN, 1> input_, trig_;
    TQue<TPosition::VECOUT, 1> output_;
    TBuf<TPosition::VECCALC> work_;
};
