#ifndef FSA_LRU_PREFIX_A3_H
#define FSA_LRU_PREFIX_A3_H

#include "kernel_operator.h"

namespace fsa_lru_prefix {
constexpr uint32_t kAlignWords = 8;

// Both scan buffers have an aligned eight-word pad before count active words.
// The masks contain zero or one, so all inclusive counts are exact in float32.
__aicore__ inline AscendC::LocalTensor<float> InclusiveScan(
    AscendC::LocalTensor<float> in, AscendC::LocalTensor<float> out,
    const AscendC::LocalTensor<float>& shiftValues,
    const AscendC::LocalTensor<int32_t>& byteOffsets, uint32_t count)
{
    using namespace AscendC;
    Duplicate(in, 0.0f, kAlignWords);
    Duplicate(out, 0.0f, kAlignWords);
    PipeBarrier<PIPE_V>();
    for (uint32_t shift = 1; shift < count; shift *= 2) {
        if (shift < kAlignWords) {
            ArithProgression<int32_t>(byteOffsets, (kAlignWords - shift) * sizeof(float), sizeof(float), count);
            PipeBarrier<PIPE_V>();
            Gather(shiftValues, in, byteOffsets.ReinterpretCast<uint32_t>(), 0, count);
            PipeBarrier<PIPE_V>();
            Add(out[kAlignWords], in[kAlignWords], shiftValues, count);
        } else {
            DataCopy(out[kAlignWords], in[kAlignWords], shift);
            Add(out[kAlignWords + shift], in[kAlignWords + shift], in[kAlignWords], count - shift);
        }
        PipeBarrier<PIPE_V>();
        auto previous = in;
        in = out;
        out = previous;
    }
    return in[kAlignWords];
}
} // namespace fsa_lru_prefix
#endif
