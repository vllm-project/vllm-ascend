// SPDX-License-Identifier: Apache-2.0
// Math substitutes for testing A5's production task/buffer/control flow.
// These do not emulate register instructions, rounding trees or device pipes.
#pragma once
#define ATTN_RES_FWD_REGBASE_COMMON_H
#define ATTN_RES_FWD_COOPERATIVE_REGBASE_HELPERS_H

namespace AttnResFwd::CooperativeRegBase {
template <typename T>
void CastToFloat(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<T> src, uint32_t count) {
  AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
}
template <typename T>
void CastToBf16(AscendC::LocalTensor<T> dst, AscendC::LocalTensor<float> src, uint32_t count) {
  AscendC::Cast(dst, src, AscendC::RoundMode::CAST_RINT, count);
}
inline void Multiply(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> a, AscendC::LocalTensor<float> b,
                     uint32_t count) {
  AscendC::Mul(dst, a, b, count);
}
template <typename T>
void WeightedAdd(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<T> src, AscendC::LocalTensor<float> weight,
                 uint32_t count) {
  for (uint32_t i = 0; i < count; ++i) {
    const float product = static_cast<float>(src.At(i)) * weight.At();
    dst.At(i) += product;
  }
}
inline void Softmax(AscendC::LocalTensor<float> scores, AscendC::LocalTensor<float> scalar, uint32_t count) {
  float maximum = -INFINITY;
  for (uint32_t i = 0; i < count; ++i) maximum = std::max(maximum, scores.At(i));
  float sum = 0;
  for (uint32_t i = 0; i < count; ++i) {
    scores.At(i) = std::exp(scores.At(i) - maximum);
    sum += scores.At(i);
  }
  scalar.At() = sum;
  for (uint32_t i = 0; i < count; ++i) scores.At(i) /= sum;
}
}  // namespace AttnResFwd::CooperativeRegBase

namespace AttnResFwd::RegBase {
constexpr uint32_t kVlFp32 = 64;
inline void ReduceSquareSum(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<float> row, uint32_t count) {
  float sum = 0;
  for (uint32_t i = 0; i < count; ++i) sum += row.At(i) * row.At(i);
  dst.At() = sum;
}
inline void InvRmsScalar(AscendC::LocalTensor<float> dst, uint32_t hidden, float epsilon) {
  dst.At() = 1.0f / std::sqrt(dst.At() / static_cast<float>(hidden) + epsilon);
}
template <typename T>
void AddB16(AscendC::LocalTensor<T> dst, AscendC::LocalTensor<T> rhs, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i) dst.At(i) = static_cast<float>(dst.At(i)) + static_cast<float>(rhs.At(i));
}
template <typename T>
void CastAndInvRms(AscendC::LocalTensor<float> row, AscendC::LocalTensor<float> inv, AscendC::LocalTensor<T> input,
                   uint32_t hidden, float, float epsilon) {
  CooperativeRegBase::CastToFloat(row, input, hidden);
  ReduceSquareSum(inv, row, hidden);
  InvRmsScalar(inv, hidden, epsilon);
}
inline void NormalizeAndReduceScore(AscendC::LocalTensor<float> score, AscendC::LocalTensor<float> row,
                                    AscendC::LocalTensor<float> weight, AscendC::LocalTensor<float> inv,
                                    uint32_t count) {
  float sum = 0;
  const float scale = inv.At();
  for (uint32_t i = 0; i < count; ++i) sum += (row.At(i) * scale) * weight.At(i);
  score.At() = sum;
}
template <typename T>
void NormalizeAndCast(AscendC::LocalTensor<T> output, AscendC::LocalTensor<float> row, AscendC::LocalTensor<T> weight,
                      AscendC::LocalTensor<float> inv, uint32_t count) {
  for (uint32_t i = 0; i < count; ++i) output.At(i) = (row.At(i) * inv.At()) * static_cast<float>(weight.At(i));
}
}  // namespace AttnResFwd::RegBase
