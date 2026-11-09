// SPDX-License-Identifier: Apache-2.0
#ifndef ATTN_RES_FWD_COOPERATIVE_REGBASE_HELPERS_H
#define ATTN_RES_FWD_COOPERATIVE_REGBASE_HELPERS_H

#include "attn_res_fwd_regbase_common.h"

namespace AttnResFwd::CooperativeRegBase {
using namespace AscendC;
using namespace AscendC::MicroAPI;

// Linear register loads keep every row/tile offset aligned, including odd H.
// The caller allocates row buffers rounded to kVlFp32 elements.
template <typename T>
__aicore__ inline void CastToFloat(const LocalTensor<float>& dst, const LocalTensor<T>& src, uint32_t count) {
  auto* input = (__local_mem__ T*)src.GetPhyAddr();
  auto* output = (__local_mem__ float*)dst.GetPhyAddr();
  uint32_t remaining = count;
  const uint16_t repeats = static_cast<uint16_t>((count + RegBase::kVlFp32 - 1) / RegBase::kVlFp32);
  __VEC_SCOPE__ {
    RegTensor<T> inputReg;
    RegTensor<float> value;
    for (uint16_t i = 0; i < repeats; ++i) {
      MaskReg mask = UpdateMask<float>(remaining);
      DataCopy<T, LoadDist::DIST_UNPACK_B16>(inputReg, input + i * RegBase::kVlFp32);
      Cast<float, T, RegBase::kCastB16ToB32>(value, inputReg, mask);
      DataCopy(output + i * RegBase::kVlFp32, value, mask);
    }
  }
}

template <typename T>
__aicore__ inline void CastToBf16(const LocalTensor<T>& dst, const LocalTensor<float>& src, uint32_t count) {
  auto* input = (__local_mem__ float*)src.GetPhyAddr();
  auto* output = (__local_mem__ T*)dst.GetPhyAddr();
  uint32_t remaining = count;
  const uint16_t repeats = static_cast<uint16_t>((count + RegBase::kVlFp32 - 1) / RegBase::kVlFp32);
  __VEC_SCOPE__ {
    RegTensor<float> value;
    RegTensor<T> outputReg;
    for (uint16_t i = 0; i < repeats; ++i) {
      MaskReg mask = UpdateMask<float>(remaining);
      DataCopy(value, input + i * RegBase::kVlFp32);
      Cast<T, float, RegBase::kCastB32ToB16Rint>(outputReg, value, mask);
      DataCopy<T, StoreDist::DIST_PACK_B32>(output + i * RegBase::kVlFp32, outputReg, mask);
    }
  }
}

__aicore__ inline void Multiply(const LocalTensor<float>& dst, const LocalTensor<float>& lhs,
                                const LocalTensor<float>& rhs, uint32_t count) {
  auto* a = (__local_mem__ float*)lhs.GetPhyAddr();
  auto* b = (__local_mem__ float*)rhs.GetPhyAddr();
  auto* output = (__local_mem__ float*)dst.GetPhyAddr();
  uint32_t remaining = count;
  const uint16_t repeats = static_cast<uint16_t>((count + RegBase::kVlFp32 - 1) / RegBase::kVlFp32);
  __VEC_SCOPE__ {
    RegTensor<float> x;
    RegTensor<float> y;
    for (uint16_t i = 0; i < repeats; ++i) {
      MaskReg mask = UpdateMask<float>(remaining);
      DataCopy(x, a + i * RegBase::kVlFp32);
      DataCopy(y, b + i * RegBase::kVlFp32);
      Mul(x, x, y, mask);
      DataCopy(output + i * RegBase::kVlFp32, x, mask);
    }
  }
}

template <typename T>
__aicore__ inline void WeightedAdd(const LocalTensor<float>& dst, const LocalTensor<T>& src,
                                   const LocalTensor<float>& weight, uint32_t count) {
  auto* input = (__local_mem__ T*)src.GetPhyAddr();
  auto* output = (__local_mem__ float*)dst.GetPhyAddr();
  auto* w = (__local_mem__ float*)weight.GetPhyAddr();
  uint32_t remaining = count;
  const uint16_t repeats = static_cast<uint16_t>((count + RegBase::kVlFp32 - 1) / RegBase::kVlFp32);
  __VEC_SCOPE__ {
    RegTensor<T> inputReg;
    RegTensor<float> value;
    RegTensor<float> acc;
    RegTensor<float> scale;
    DataCopy<float, LoadDist::DIST_BRC_B32>(scale, w);
    for (uint16_t i = 0; i < repeats; ++i) {
      MaskReg mask = UpdateMask<float>(remaining);
      DataCopy<T, LoadDist::DIST_UNPACK_B16>(inputReg, input + i * RegBase::kVlFp32);
      Cast<float, T, RegBase::kCastB16ToB32>(value, inputReg, mask);
      DataCopy(acc, output + i * RegBase::kVlFp32);
      // Keep separate multiply/add rounding, as in the original A5 kernel.
      Mul(value, value, scale, mask);
      Add(acc, acc, value, mask);
      DataCopy(output + i * RegBase::kVlFp32, acc, mask);
    }
  }
}

// Each register load stays within a scores buffer padded to kVlFp32.
// Active-lane reductions exclude padding for every R, including 9 and 65.
__aicore__ inline void Softmax(const LocalTensor<float>& scores, const LocalTensor<float>& scalar, uint32_t count) {
  auto* values = (__local_mem__ float*)scores.GetPhyAddr();
  auto* stats = (__local_mem__ float*)scalar.GetPhyAddr();
  const uint16_t repeats = static_cast<uint16_t>((count + RegBase::kVlFp32 - 1) / RegBase::kVlFp32);
  uint32_t remaining = count;
  __VEC_SCOPE__ {
    RegTensor<float> x;
    RegTensor<float> part;
    RegTensor<float> maximum;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    Duplicate(maximum, -3.402823466e38f, all);
    for (uint16_t i = 0; i < repeats; ++i) {
      MaskReg mask = UpdateMask<float>(remaining);
      DataCopy(x, values + i * RegBase::kVlFp32);
      ReduceMax(part, x, mask);
      Duplicate(part, part, all);
      Max(maximum, maximum, part, all);
    }
    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(stats, maximum, all);
  }
  PipeBarrier<PIPE_V>();
  remaining = count;
  __VEC_SCOPE__ {
    RegTensor<float> x;
    RegTensor<float> maximum;
    RegTensor<float> part;
    RegTensor<float> total;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    DataCopy<float, LoadDist::DIST_BRC_B32>(maximum, stats);
    Duplicate(total, 0.0f, all);
    for (uint16_t i = 0; i < repeats; ++i) {
      MaskReg mask = UpdateMask<float>(remaining);
      DataCopy(x, values + i * RegBase::kVlFp32);
      Sub(x, x, maximum, mask);
      Exp<float, &RegBase::kAccurateExp>(x, x, mask);
      DataCopy(values + i * RegBase::kVlFp32, x, mask);
      ReduceSum(part, x, mask);
      Add(total, total, part, all);
    }
    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(stats, total, all);
  }
  PipeBarrier<PIPE_V>();
  remaining = count;
  __VEC_SCOPE__ {
    RegTensor<float> x;
    RegTensor<float> total;
    RegTensor<float> one;
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    DataCopy<float, LoadDist::DIST_BRC_B32>(total, stats);
    Duplicate(one, 1.0f, all);
    Div<float, &RegBase::kAccurateDiv>(total, one, total, all);
    for (uint16_t i = 0; i < repeats; ++i) {
      MaskReg mask = UpdateMask<float>(remaining);
      DataCopy(x, values + i * RegBase::kVlFp32);
      Mul(x, x, total, mask);
      DataCopy(values + i * RegBase::kVlFp32, x, mask);
    }
  }
  PipeBarrier<PIPE_V>();
}
// Brcb reads eight compact probabilities and writes eight broadcast blocks.
// Do not narrow the complete repeat count before splitting it into ISA batches.
__aicore__ inline void BroadcastScores(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                      uint32_t alignedCount) {
  constexpr uint32_t elementsPerBlock = 8;
  constexpr uint32_t elementsPerRepeat = elementsPerBlock * elementsPerBlock;
  constexpr uint32_t maxRepeats = 255;
  const uint32_t repeats = alignedCount / elementsPerBlock;
  for (uint32_t done = 0; done < repeats;) {
    const uint32_t batch = repeats - done > maxRepeats ? maxRepeats : repeats - done;
    Brcb(dst[done * elementsPerRepeat], src[done * elementsPerBlock], static_cast<uint8_t>(batch), {1, MOV_8});
    done += batch;
  }
  PipeBarrier<PIPE_V>();
}
}  // namespace AttnResFwd::CooperativeRegBase
#endif
