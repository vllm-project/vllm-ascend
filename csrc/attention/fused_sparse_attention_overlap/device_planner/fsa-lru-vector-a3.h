// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "fsa-lru-geometry-a3.h"
#include "fsa-lru-prefix-a3.h"

namespace FsaLruVector {
using namespace AscendC;
using F32 = LocalTensor<float>;
using I32 = LocalTensor<int32_t>;
using Mask = LocalTensor<uint8_t>;

// Counts/slot indices here are <= 32768 and exactly representable in FP32.
// BiSheng AICore scalar conversion supports signed integer intermediates only.
__aicore__ inline float CountAsFloat(uint32_t count) {
    return static_cast<float>(static_cast<int32_t>(count));
}

__aicore__ inline uint32_t CountFromFloat(float count) {
    return static_cast<uint32_t>(static_cast<int32_t>(count));
}

__aicore__ inline void Intersect(Mask a, Mask b, uint32_t n) {
    And(a.ReinterpretCast<uint16_t>(), a.ReinterpretCast<uint16_t>(),
        b.ReinterpretCast<uint16_t>(), n / 16);
    PipeBarrier<PIPE_V>();
}

// Numeric FP32 preserves all observable keys when the legal domain ends at
// 2^24. Out-of-domain positives stay >= that exact endpoint; negatives stay
// negative. Larger domains use exact ordered bit encoding; its gap at 2^30
// skips signed zero and subnormals (which hardware may flush).
__aicore__ inline float TokenKey(int32_t token, int32_t maxToken=INT32_MAX) {
    if (maxToken <= (1 << 24)) return static_cast<float>(token);
    union { uint32_t bits; float value; } key;
    key.bits = token < 0 ? 0xc1000000U :
        (token < (1 << 30) ? 0xc0800000U - uint32_t(token) :
                            0x00800000U + uint32_t(token) - (1U << 30));
    return key.value;
}

__aicore__ inline void TokenKeys(F32 out, I32 tokens, I32 a, I32 b, uint32_t n,
                                  int32_t maxToken=INT32_MAX) {
    if (maxToken <= (1 << 24)) {
        Cast(out,tokens,RoundMode::CAST_ROUND,n);
        PipeBarrier<PIPE_V>();
        return;
    }
    auto bits = out.ReinterpretCast<int32_t>();
    // Low / high positive halves, selected using the original high bit pair.
    Muls(bits, tokens, int32_t{-1}, n);
    Adds(a, tokens, int32_t{-0x3f800000}, n);
    ShiftRight(b.ReinterpretCast<uint32_t>(), tokens.ReinterpretCast<uint32_t>(), 30U, n);
    PipeBarrier<PIPE_V>();
    Adds(bits, bits, static_cast<int32_t>(0xc0800000U), n);
    Muls(b, b, int32_t{-1}, n);
    PipeBarrier<PIPE_V>();
    And(a.ReinterpretCast<uint16_t>(), a.ReinterpretCast<uint16_t>(), b.ReinterpretCast<uint16_t>(), 2*n);
    Not(b.ReinterpretCast<uint16_t>(), b.ReinterpretCast<uint16_t>(), 2*n);
    PipeBarrier<PIPE_V>();
    And(bits.ReinterpretCast<uint16_t>(), bits.ReinterpretCast<uint16_t>(), b.ReinterpretCast<uint16_t>(), 2*n);
    PipeBarrier<PIPE_V>();
    Or(bits.ReinterpretCast<uint16_t>(), bits.ReinterpretCast<uint16_t>(), a.ReinterpretCast<uint16_t>(), 2*n);
    ShiftRight(b.ReinterpretCast<uint32_t>(), tokens.ReinterpretCast<uint32_t>(), 31U, n);
    PipeBarrier<PIPE_V>();
    // The sign is 0/1: form either the sentinel or zero, and retain the
    // positive encoding only where sign-1 is all ones. No numeric FP cast.
    Muls(a, b, static_cast<int32_t>(0xc1000000U), n);
    Adds(b, b, int32_t{-1}, n);
    PipeBarrier<PIPE_V>();
    And(bits.ReinterpretCast<uint16_t>(), bits.ReinterpretCast<uint16_t>(), b.ReinterpretCast<uint16_t>(), 2*n);
    PipeBarrier<PIPE_V>();
    Or(bits.ReinterpretCast<uint16_t>(), bits.ReinterpretCast<uint16_t>(), a.ReinterpretCast<uint16_t>(), 2*n);
    PipeBarrier<PIPE_V>();
}

// The input key/value vectors may occupy work[0:2*n]. Once Sort32 has read
// them, the two interleaved buffers alternate; extraction uses the dead buffer.
__aicore__ inline F32 SortPacked(F32 keys, I32 values, F32 work, uint32_t n) {
    auto from = work[2*n], to = work;
    // The repeat field is uint8. Chunk by the ISA limit, independently of the
    // selected input tile, so a larger UB cannot silently wrap repeat=256.
    constexpr uint32_t maxRepeat=255;
    for(uint32_t block=0;block<n/32;block+=maxRepeat) {
        const uint32_t repeats=n/32-block<maxRepeat?n/32-block:maxRepeat;
        Sort32(from[block*64],keys[block*32],values[block*32].ReinterpretCast<uint32_t>(),repeats);
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t width=32; width<n; width*=4) {
        const uint32_t lists = n/width < 4 ? n/width : 4;
        MrgSort4Info info;
        for (int i=0; i<4; ++i) info.elementLengths[i] = i < lists ? width : 0;
        info.ifExhaustedSuspension = false;
        info.validBit = (1U << lists)-1;
        const uint32_t groups=n/(lists*width);
        for(uint32_t group=0;group<groups;group+=maxRepeat) {
            info.repeatTimes = groups-group<maxRepeat?groups-group:maxRepeat;
            auto chunk=from[2*group*lists*width];
            MrgSortSrcList<float> src;
            src.src1=chunk; src.src2=chunk[2*width];
            src.src3=chunk[lists>2 ? 4*width : 0]; src.src4=chunk[lists>3 ? 6*width : 0];
            MrgSort<float>(to[2*group*lists*width],src,info);
        }
        PipeBarrier<PIPE_V>();
        auto old=from; from=to; to=old;
    }
    for(uint32_t block=0;block<n/32;block+=maxRepeat) {
        const uint32_t repeats=n/32-block<maxRepeat?n/32-block:maxRepeat;
        Extract(to[block*32],to[n+block*32].ReinterpretCast<uint32_t>(),from[block*64],repeats);
    }
    PipeBarrier<PIPE_V>();
    return to;
}

__aicore__ inline void GatherAt(F32 dst, F32 src, I32 offsets, F32 positions, uint32_t n) {
    Muls(dst, positions, 4.0f, n);
    PipeBarrier<PIPE_V>();
    Cast(offsets, dst, RoundMode::CAST_ROUND, n);
    PipeBarrier<PIPE_V>();
    Gather(dst, src, offsets.ReinterpretCast<uint32_t>(), 0, n);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline uint32_t Compact(LocalTensor<T> dst, LocalTensor<T> src, Mask mask, uint32_t n) {
    const GatherMaskParams params{1,1,8,0};
    uint64_t kept=0;
    GatherMask(dst, src, mask.ReinterpretCast<uint32_t>(), true, n, params, kept);
    PipeBarrier<PIPE_V>();
    return static_cast<uint32_t>(kept);
}

// Sort once. Equal valid token keys resolve to the first ORIGINAL TopK
// position. Lookup discards negative keys, including padding.
__aicore__ inline void SortTopk(I32 topk, F32 keys, F32 first, F32 work, Mask mask, uint32_t k, int32_t maxToken=INT32_MAX) {
    TokenKeys(keys, topk, work.ReinterpretCast<int32_t>(), work[k].ReinterpretCast<int32_t>(), k, maxToken);
    ArithProgression<int32_t>(first.ReinterpretCast<int32_t>(),0,1,k);
    PipeBarrier<PIPE_V>();
    auto sorted=SortPacked(keys,first.ReinterpretCast<int32_t>(),work,k);
    DataCopy(keys,sorted,k);
    DataCopy(first.ReinterpretCast<int32_t>(),sorted[k].ReinterpretCast<int32_t>(),k);
    PipeBarrier<PIPE_V>();
    Cast(first, first.ReinterpretCast<int32_t>(), RoundMode::CAST_ROUND, k);
    PipeBarrier<PIPE_V>();
    auto positions=work, gathered=work[k], candidate=work[2*k];
    auto offsets=work[3*k].ReinterpretCast<int32_t>();
    // Check adjacent keys once. Negative/padded entries cannot be hits and
    // must not force log2(k) tie-propagation passes on otherwise unique TopK.
    ArithProgression<int32_t>(offsets,4,4,k);
    SetFlag<HardEvent::V_S>(0); WaitFlag<HardEvent::V_S>(0);
    offsets.SetValue(k-1,0);
    SetFlag<HardEvent::S_V>(0); WaitFlag<HardEvent::S_V>(0);
    Gather(gathered,keys,offsets.ReinterpretCast<uint32_t>(),0,k);
    PipeBarrier<PIPE_V>();
    Compare(mask,keys,gathered,CMPMODE::EQ,k);
    PipeBarrier<PIPE_V>();
    Select(candidate,mask,keys,TokenKey(-1,maxToken),SELMODE::VSEL_TENSOR_SCALAR_MODE,k);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask,candidate,TokenKey(0,maxToken),CMPMODE::GE,k);
    PipeBarrier<PIPE_V>();
    const uint32_t duplicateCount=Compact(positions,first,mask,k);
    if (duplicateCount == 0) {
        return;
    }
    for (uint32_t shift=1; shift<k; shift*=2) {
        ArithProgression<float>(positions,CountAsFloat(shift*4),4.0f,k);
        PipeBarrier<PIPE_V>();
        Mins(positions,positions,CountAsFloat((k-1)*4),k);
        PipeBarrier<PIPE_V>();
        Cast(offsets,positions,RoundMode::CAST_ROUND,k);
        PipeBarrier<PIPE_V>();
        Gather(gathered,keys,offsets.ReinterpretCast<uint32_t>(),0,k);
        PipeBarrier<PIPE_V>();
        Compare(mask,keys,gathered,CMPMODE::EQ,k);
        PipeBarrier<PIPE_V>();
        Gather(gathered,first,offsets.ReinterpretCast<uint32_t>(),0,k);
        PipeBarrier<PIPE_V>();
        Min(candidate,first,gathered,k);
        PipeBarrier<PIPE_V>();
        Select(first,mask,candidate,first,SELMODE::VSEL_TENSOR_TENSOR_MODE,k);
        PipeBarrier<PIPE_V>();
    }
}

// Descending lower bound. Return both the gathered key and its byte offset
// so the caller can read the matching position without converting it again.
__aicore__ inline void LowerBound(F32 query, F32 sorted, uint32_t width,
                                 F32 lo, F32 candidate, F32 gathered, I32 offsets,
                                 Mask mask, uint32_t n) {
    // Keep the search position in bytes: Gather consumes byte offsets, so
    // each binary-search round needs only Cast instead of Muls + Cast.
    Duplicate(lo,-4.0f,n);
    PipeBarrier<PIPE_V>();
    for (uint32_t step=width/2; step; step/=2) {
        Adds(candidate,lo,CountAsFloat(step*4),n);
        PipeBarrier<PIPE_V>();
        Cast(offsets,candidate,RoundMode::CAST_ROUND,n);
        PipeBarrier<PIPE_V>();
        Gather(gathered,sorted,offsets.ReinterpretCast<uint32_t>(),0,n);
        PipeBarrier<PIPE_V>();
        Compare(mask,gathered,query,CMPMODE::GT,n);
        PipeBarrier<PIPE_V>();
        Select(lo,mask,candidate,lo,SELMODE::VSEL_TENSOR_TENSOR_MODE,n);
        PipeBarrier<PIPE_V>();
    }
    Adds(candidate,lo,4.0f,n);
    PipeBarrier<PIPE_V>();
    Cast(offsets,candidate,RoundMode::CAST_ROUND,n);
    PipeBarrier<PIPE_V>();
    Gather(gathered,sorted,offsets.ReinterpretCast<uint32_t>(),0,n);
    PipeBarrier<PIPE_V>();
}

template <bool RewriteTokens>
__aicore__ inline void Lookup(F32 keys, F32 first, I32 tokens, F32 hitPos,
                             F32 work, Mask mask, Mask valid, uint32_t k, uint32_t t,
                             uint32_t active, int32_t stable, int32_t visible, int32_t maxToken,
                             Mask savedInvalid, bool saveInvalid) {
    auto query=work, lo=work[t], candidate=work[2*t], gathered=work[3*t];
    auto offsets=work[4*t].ReinterpretCast<int32_t>();
    TokenKeys(query,tokens,lo.ReinterpretCast<int32_t>(),candidate.ReinterpretCast<int32_t>(),t,maxToken);
    if (saveInvalid || RewriteTokens) {
        auto invalid=saveInvalid?savedInvalid:valid;
        CompareScalar(invalid,query,TokenKey(stable,maxToken),CMPMODE::GE,t);
        CompareScalar(mask,query,TokenKey(maxToken,maxToken),CMPMODE::LT,t);
        Intersect(invalid,mask,t);
        if constexpr (RewriteTokens) {
            // Retain negative and out-of-domain stale IDs in recurrent state.
            Duplicate(lo.ReinterpretCast<int32_t>(),int32_t{-1},t);
            PipeBarrier<PIPE_V>();
            Select(tokens.ReinterpretCast<float>(),invalid,lo,tokens.ReinterpretCast<float>(),
                   SELMODE::VSEL_TENSOR_TENSOR_MODE,t);
            PipeBarrier<PIPE_V>();
        }
    }
    // A hit must be both stable and visible. Only the repartitioning pass
    // needs mutated raw tokens, so lookup can compare the original keys.
    const int32_t upper=stable<visible?stable:visible;
    CompareScalar(mask,query,TokenKey(upper,maxToken),CMPMODE::LT,t);
    CompareScalar(valid,query,TokenKey(0,maxToken),CMPMODE::GE,t);
    Intersect(valid,mask,t);
    LowerBound(query,keys,k,lo,candidate,gathered,offsets,mask,t);
    Compare(mask,gathered,query,CMPMODE::EQ,t);
    Intersect(valid,mask,t);
    Gather(gathered,first,offsets.ReinterpretCast<uint32_t>(),0,t);
    PipeBarrier<PIPE_V>();
    ArithProgression<float>(candidate,0.0f,1.0f,t);
    PipeBarrier<PIPE_V>();
    CompareScalar(mask,candidate,CountAsFloat(active),CMPMODE::LT,t);
    Intersect(valid,mask,t);
    Select(hitPos,valid,gathered,-1.0f,SELMODE::VSEL_TENSOR_SCALAR_MODE,t);
    PipeBarrier<PIPE_V>();
}
} // namespace FsaLruVector
