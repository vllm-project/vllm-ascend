// SPDX-License-Identifier: Apache-2.0
// CPU-only API shim for executing the real cooperative kernel's control flow.
// Transfers and pipes are synchronous here; this does not simulate NPU timing,
// compiler constraints, or the hardware reduction instruction's addition tree.
#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#define __aicore__
#define __gm__
#define REDUCE_COMMON_H_ATTN_RES_FWD
using GM_ADDR = unsigned char*;
using event_t = int;

struct bfloat16_t {
  uint16_t bits;
  bfloat16_t(float value = 0.0f) {
    uint32_t raw;
    std::memcpy(&raw, &value, sizeof(raw));
    bits = static_cast<uint16_t>((raw + 0x7fffU + ((raw >> 16) & 1U)) >> 16);
  }
  operator float() const {
    const uint32_t raw = static_cast<uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
  }
};

namespace CpuAttnTest {
class Barrier {
 public:
  explicit Barrier(unsigned cores) : cores_(cores) {}
  unsigned Cores() const { return cores_; }
  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    const unsigned generation = generation_;
    if (++arrived_ == cores_) {
      arrived_ = 0;
      ++generation_;
      cv_.notify_all();
    } else if (!cv_.wait_for(lock, std::chrono::seconds(15), [&] { return generation_ != generation || broken_; })) {
      throw std::runtime_error("Cooperative kernel barrier timed out");
    }
    if (broken_) throw std::runtime_error("Another CPU kernel worker failed");
  }
  void Cancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    broken_ = true;
    cv_.notify_all();
  }

 private:
  const unsigned cores_;
  unsigned arrived_{0}, generation_{0};
  bool broken_{false};
  std::mutex mutex_;
  std::condition_variable cv_;
};
struct Region {
  const void* ptr;
  size_t bytes;
};
struct Context {
  unsigned core;
  Barrier& barrier;
  const std::vector<Region>& regions;
  uint32_t maskCount{0};
  uint32_t syncCount{0};
};
// Thread-local fixture state exists only for one worker's kernel invocation.
inline thread_local Context* context = nullptr;
struct Storage {
  explicit Storage(size_t size) : bytes(size, 0xff) {}  // Poison UB padding.
  std::vector<unsigned char> bytes;
};
}  // namespace CpuAttnTest

namespace AscendC {
enum class QuePosition { VECIN, VECOUT };
enum class TPosition { VECCALC };
enum class RoundMode { CAST_NONE, CAST_RINT };
enum class HardEvent { MTE3_V };
enum class MaskMode { COUNTER };
enum { PIPE_V };
struct CopyParams {
  int a, b, c, d;
};
struct BrcbParams {
  int a, b;
};
struct DataCopyExtParams {
  uint16_t blocks;
  uint32_t bytes, srcStride, dstStride;
  uint8_t reserved;
};
template <typename T>
struct DataCopyPadExtParams {
  bool pad;
  uint8_t left, right;
  T value;
};

template <typename T>
struct LocalTensor {
  std::shared_ptr<CpuAttnTest::Storage> storage;
  size_t offset{0};
  LocalTensor operator[](uint64_t index) const { return {storage, offset + index * sizeof(T)}; }
  T& At(size_t index = 0) const {
    assert(storage && offset + (index + 1) * sizeof(T) <= storage->bytes.size());
    return *reinterpret_cast<T*>(storage->bytes.data() + offset + index * sizeof(T));
  }
};
template <typename T>
struct GlobalTensor {
  T* ptr{nullptr};
  size_t count{0};
  void SetGlobalBuffer(T* address) {
    for (const auto& region : CpuAttnTest::context->regions) {
      if (address == region.ptr) {
        ptr = address;
        count = region.bytes / sizeof(T);
        return;
      }
    }
    throw std::runtime_error("Unregistered CPU GM region");
  }
  GlobalTensor operator[](uint64_t index) const {
    assert(index <= count);
    return {ptr + index, count - index};
  }
  T& At(size_t index) const {
    assert(index < count);
    return ptr[index];
  }
};
template <QuePosition P, int N>
struct TQue {
  std::vector<std::shared_ptr<CpuAttnTest::Storage>> buffers;
  std::vector<bool> allocated;
  std::deque<std::shared_ptr<CpuAttnTest::Storage>> queued;
  template <typename T>
  LocalTensor<T> AllocTensor() {
    for (size_t i = 0; i < buffers.size(); ++i)
      if (!allocated[i]) {
        allocated[i] = true;
        return {buffers[i], 0};
      }
    throw std::runtime_error("CPU queue exhausted");
  }
  template <typename T>
  void EnQue(LocalTensor<T> tensor) {
    assert(std::find(queued.begin(), queued.end(), tensor.storage) == queued.end());
    queued.push_back(tensor.storage);
  }
  template <typename T>
  LocalTensor<T> DeQue() {
    assert(!queued.empty());
    auto storage = queued.front();
    queued.pop_front();
    return {storage, 0};
  }
  template <typename T>
  void FreeTensor(LocalTensor<T> tensor) {
    assert(std::find(queued.begin(), queued.end(), tensor.storage) == queued.end());
    const auto found = std::find(buffers.begin(), buffers.end(), tensor.storage);
    assert(found != buffers.end() && allocated[found - buffers.begin()]);
    allocated[found - buffers.begin()] = false;
  }
};
template <TPosition P>
struct TBuf {
  std::shared_ptr<CpuAttnTest::Storage> storage;
  template <typename T>
  LocalTensor<T> Get() {
    return {storage, 0};
  }
};
struct TPipe {
  size_t ubBytes{0};
  template <typename B>
  void InitBuffer(B& buffer, uint32_t bytes) {
    assert(bytes % 32 == 0);
    buffer.storage = std::make_shared<CpuAttnTest::Storage>(bytes);
    ubBytes += bytes;
  }
  template <QuePosition P, int N>
  void InitBuffer(TQue<P, N>& queue, uint32_t depth, uint32_t bytes) {
    assert(depth > 0 && bytes % 32 == 0);
    for (uint32_t i = 0; i < depth; ++i) queue.buffers.push_back(std::make_shared<CpuAttnTest::Storage>(bytes));
    queue.allocated.resize(depth, false);
    ubBytes += depth * bytes;
  }
  int FetchEventID(HardEvent) { return 0; }
};
template <int P>
void PipeBarrier() {}
template <HardEvent P>
void SetFlag(event_t) {}
template <HardEvent P>
void WaitFlag(event_t) {}
template <bool B>
void SyncAll() {
  static_assert(B, "CPU shim covers AIV-only sync");
  ++CpuAttnTest::context->syncCount;
  CpuAttnTest::context->barrier.Wait();
}
inline uint32_t GetBlockIdx() { return CpuAttnTest::context->core; }
inline uint32_t GetBlockNum() { return CpuAttnTest::context->barrier.Cores(); }
inline void SetMaskCount() {}
inline void SetMaskNorm() {}
inline void ResetMask() {}
template <typename T, MaskMode M>
void SetVectorMask(uint32_t count) {
  CpuAttnTest::context->maskCount = count;
}
template <typename T>
void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyExtParams copy, DataCopyPadExtParams<T>) {
  assert(copy.blocks == 1 && copy.bytes % sizeof(T) == 0);
  for (size_t i = 0; i < copy.bytes / sizeof(T); ++i) dst.At(i) = src.At(i);
}
template <typename T>
void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams copy) {
  assert(copy.blocks == 1 && copy.bytes % sizeof(T) == 0);
  for (size_t i = 0; i < copy.bytes / sizeof(T); ++i) dst.At(i) = src.At(i);
}
template <typename T>
void Copy(LocalTensor<T> dst, LocalTensor<T> src, uint64_t mask, uint8_t repeat, CopyParams params) {
  constexpr size_t blockElements = 32 / sizeof(T);
  assert(repeat > 0 && mask <= 256 / sizeof(T));
  for (size_t r = 0; r < repeat; ++r)
    for (size_t i = 0; i < mask; ++i) {
      const size_t dstIndex = (r * params.c + (i / blockElements) * params.a) * blockElements + i % blockElements;
      const size_t srcIndex = (r * params.d + (i / blockElements) * params.b) * blockElements + i % blockElements;
      dst.At(dstIndex) = src.At(srcIndex);
    }
}
template <typename T, typename U>
void Cast(LocalTensor<T> dst, LocalTensor<U> src, RoundMode, uint32_t count) {
  for (size_t i = 0; i < count; ++i) dst.At(i) = static_cast<T>(src.At(i));
}
template <typename T, typename U>
void Duplicate(LocalTensor<T> dst, U value, uint32_t count) {
  for (size_t i = 0; i < count; ++i) dst.At(i) = static_cast<T>(value);
}
inline void Mul(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t count) {
  for (size_t i = 0; i < count; ++i) dst.At(i) = a.At(i) * b.At(i);
}
inline void Add(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t count) {
  for (size_t i = 0; i < count; ++i) dst.At(i) = a.At(i) + b.At(i);
}
inline void Exp(LocalTensor<float> dst, LocalTensor<float> src, uint32_t count) {
  for (size_t i = 0; i < count; ++i) dst.At(i) = std::exp(src.At(i));
}
inline void Div(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t count) {
  for (size_t i = 0; i < count; ++i) dst.At(i) = a.At(i) / b.At(i);
}
inline void Brcb(LocalTensor<float> dst, LocalTensor<float> src, uint8_t repeats, BrcbParams) {
  for (size_t i = 0; i < repeats * 8U; ++i)
    for (size_t j = 0; j < 8; ++j) dst.At(i * 8 + j) = src.At(i);
}
}  // namespace AscendC

namespace AttnResFwd {
using namespace AscendC;
constexpr float SOFTMAX_PAD = -1e20f;
constexpr uint32_t MOV_8 = 8;
constexpr uint32_t ELEM_PER_REP_FP32 = 64, ELEM_PER_BLK_FP32 = 8, SCALAR_LOCAL_ELEMS = 8;
inline void CopyMetaScalarToLocal(LocalTensor<float> dst, LocalTensor<float> src) { dst.At() = src.At(); }
inline void CopyCompactFloatsUb(LocalTensor<float> dst, LocalTensor<float> src, uint32_t count) {
  for (size_t i = 0; i < count; ++i) dst.At(i) = src.At(i);
}
inline void ClearFp32TailBlock(LocalTensor<float> row, uint32_t count) {
  if (count % 8 != 0) Duplicate(row[count / 8 * 8], 0.0f, 8);
}
template <typename T>
void CastRowToFp32CounterNoSetMask(LocalTensor<float> dst, LocalTensor<T> src) {
  Cast(dst, src, RoundMode::CAST_NONE, CpuAttnTest::context->maskCount);
}
template <bool MAXIMUM>
void ReduceHalf(LocalTensor<float> dst, LocalTensor<float> src, int32_t count) {
  auto combine = [](float a, float b) { return MAXIMUM ? std::max(a, b) : a + b; };
  uint32_t body = static_cast<uint32_t>(count);
  if (body > 64) {
    uint32_t power = 1;
    while (power * 2 <= body) power *= 2;
    const uint32_t tail = body - power;
    for (uint32_t i = 0; i < (tail + 7) / 8 * 8; ++i) src.At(i) = combine(src.At(i), src.At(power + i));
    body = power;
    while (body > 64) {
      body /= 2;
      for (uint32_t i = 0; i < body; ++i) src.At(i) = combine(src.At(i), src.At(body + i));
    }
  }
  float value = MAXIMUM ? -INFINITY : 0.0f;
  for (uint32_t i = 0; i < body; ++i) value = combine(value, src.At(i));
  dst.At() = value;
}
inline void ReduceSumHalfInterval(LocalTensor<float> dst, LocalTensor<float> src, int32_t count) {
  ReduceHalf<false>(dst, src, count);
}
inline void ReduceMaxHalfInterval(LocalTensor<float> dst, LocalTensor<float> src, int32_t count) {
  ReduceHalf<true>(dst, src, count);
}
inline void InvRmsInPlace(LocalTensor<float> dst, float invHidden, float eps, LocalTensor<float>) {
  dst.At() = 1.0f / std::sqrt(dst.At() * invHidden + eps);
}
inline void BroadcastScalarMulTensor(LocalTensor<float> dst, LocalTensor<float> src, LocalTensor<float> scalar,
                                     LocalTensor<float> scratch, LocalTensor<float> duplicate, uint32_t count,
                                     uint32_t) {
  Duplicate(duplicate, 0.0f, 8);
  duplicate.At() = scalar.At();
  Brcb(scratch, duplicate, 1, {1, MOV_8});
  for (size_t i = 0; i < count; ++i) dst.At(i) = src.At(i) * scratch.At();
}
inline void BroadcastCompactScores(LocalTensor<float> dst, LocalTensor<float> src, uint32_t alignedCount) {
  for (uint32_t done = 0; done < alignedCount / 8;) {
    const uint32_t count = std::min(255U, alignedCount / 8 - done);
    Brcb(dst[done * 64], src[done * 8], static_cast<uint8_t>(count), {1, MOV_8});
    done += count;
  }
}
inline void BrcbScalarRow1(LocalTensor<float> dst, LocalTensor<float> scalar) { Brcb(dst, scalar, 1, {1, MOV_8}); }
inline void SubLastDimRow1NoBrc(LocalTensor<float> dst, LocalTensor<float> src, LocalTensor<float> brc, int32_t count) {
  for (int32_t i = 0; i < count; ++i) dst.At(i) = src.At(i) - brc.At();
}
inline void MulLastDimRow1NoBrc(LocalTensor<float> dst, LocalTensor<float> src, LocalTensor<float> brc, int32_t count) {
  for (int32_t i = 0; i < count; ++i) dst.At(i) = src.At(i) * brc.At();
}
inline void MulAddRowByBrcBlock(LocalTensor<float> dst, LocalTensor<float> src, LocalTensor<float> brc, uint32_t count,
                                uint32_t, bool = true) {
  for (size_t i = 0; i < count; ++i) dst.At(i) += src.At(i) * brc.At();
}
inline void SoftmaxSmallVec(LocalTensor<float> scores, uint32_t rows, uint32_t metaAlign, LocalTensor<float> scalar,
                            LocalTensor<float> scratch, LocalTensor<float> brc) {
  float maximum = -INFINITY;
  for (uint32_t n = 0; n < rows; ++n) maximum = std::max(maximum, scores.At(n));
  scalar.At() = maximum;
  BrcbScalarRow1(brc, scalar);
  float sum = 0;
  for (uint32_t n = 0; n < rows; ++n) {
    scratch.At(n) = std::exp(scores.At(n) - maximum);
    sum += scratch.At(n);
  }
  for (uint32_t n = rows; n < metaAlign; ++n) scratch.At(n) = 0;
  for (uint32_t n = 0; n < rows; ++n) scores.At(n) = scratch.At(n) / sum;
}
}  // namespace AttnResFwd
