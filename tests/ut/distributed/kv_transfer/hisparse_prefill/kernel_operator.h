// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#define __aicore__
#define __gm__
#define __global__
#define KERNEL_TASK_TYPE_DEFAULT(...)
#define KERNEL_TYPE_AIV_ONLY 0
#define SIM_EXPORT extern "C" inline __attribute__((used, visibility("default")))

using GM_ADDR = uint8_t*;

namespace AscendC {
using namespace std;
using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using sz = size_t;
using event_t = int;
using LE = logic_error;
using RE = runtime_error;
using OE = out_of_range;
using Buf = shared_ptr<vector<u64>>;

constexpr int PIPE_V = 0;
constexpr int PIPE_ALL = 1;
enum class CMPMODE { GE, LT };
enum class RoundMode { CAST_NONE };
enum class HardEvent { V_MTE2, MTE2_V, V_S, S_MTE3, MTE3_S, MTE2_S, V_MTE3, S_V, S_MTE2, MTE3_V };
enum class QuePosition { VECCALC, VECIN, VECOUT };
using TPosition = QuePosition;
using QP = QuePosition;

struct Reg {
  uintptr_t a, b;
};

inline sz tCnt = 0, qPeak = 0;
inline u32 bIdx = 0, bCnt = 32;
inline const char* tMsg = "Trap";
inline vector<Reg> pReg, cReg;

inline void SetSimTrapReason(const char* m) { tMsg = m ? m : "Trap"; }
[[noreturn]] inline void Trap() {
  ++tCnt;
  throw RE(tMsg);
}
inline u32 GetBlockIdx() { return bIdx; }
inline u32 GetBlockNum() { return bCnt; }

template <class F>
void RunHostKernel(u32 n, F&& f) {
  for (bCnt = n, bIdx = 0; bIdx < n; ++bIdx) f();
}

template <class A, class B>
inline A rc(B p) {
  return reinterpret_cast<A>(p);
}
inline uintptr_t Addr(const void* d) { return rc<uintptr_t>(d); }

inline void RegisterPersistentRegion(void* d, sz b) {
  const Reg r{Addr(d), Addr(d) + b};
  if (d && b && none_of(pReg.begin(), pReg.end(), [&](Reg e) { return e.a == r.a && e.b == r.b; })) {
    pReg.push_back(r);
  }
}

struct ScopedCallRegions {
  sz m = cReg.size();
  ~ScopedCallRegions() { cReg.resize(m); }
  void Add(void* d, sz b) {
    if (d) cReg.push_back({Addr(d), Addr(d) + b});
  }
};

inline void CheckGM(const void* d, sz b) {
  const auto f = Addr(d);
  auto ok = [&](const auto& t) {
    return any_of(t.begin(), t.end(), [&](Reg r) { return f >= r.a && f <= r.b && b <= r.b - f; });
  };
  if (!ok(cReg) && !ok(pReg)) throw OE("GM");
}

template <class T, bool G = false>
struct Tensor {
  u8* d_ = nullptr;
  sz n_ = 0;
  Buf o_;

  void SetGlobalBuffer(T* d) { d_ = rc<u8*>(d); }
  u8* Acc(sz b) const {
    if constexpr (G)
      CheckGM(d_, b);
    else if (b > n_)
      throw OE("UB");
    return d_;
  }
  T GetValue(sz i) const {
    T v;
    memcpy(&v, (*this)[i].Acc(sizeof(T)), sizeof(T));
    return v;
  }
  void SetValue(sz i, T v) const { memcpy((*this)[i].Acc(sizeof(T)), &v, sizeof(T)); }
  Tensor operator[](sz i) const {
    if constexpr (!G) {
      if (i * sizeof(T) > n_) throw OE("UB");
    }
    return {d_ + i * sizeof(T), G ? 0 : n_ - i * sizeof(T), o_};
  }
  template <class U>
  Tensor<U, G> ReinterpretCast() const {
    return {d_, n_, o_};
  }
};

template <class T>
using LocalTensor = Tensor<T>;
template <class T>
using GlobalTensor = Tensor<T, true>;

template <QP>
struct TBuf {
  Buf d;
  sz n = 0;
  u8* P() const { return rc<u8*>(d->data()); }
  void Init(sz b) {
    n = b;
    d = make_shared<vector<u64>>((b + 7) / 8, UINT64_MAX);
  }
  template <class T>
  Tensor<T> Get() const {
    return {P(), n, d};
  }
};

template <QP, QP, u32 D>
struct TQueBind {
  enum { F, A, Q, U };
  sz q_[D]{}, h_ = 0, t_ = 0, l_ = 0;
  int st_[D]{};
  TBuf<QP::VECCALC> sl_[D];

  template <class T>
  sz Idx(Tensor<T> t) const {
    for (sz i = 0; i < D; ++i) {
      if (t.Acc(0) == sl_[i].P()) return i;
    }
    throw LE("F");
  }
  void Init(sz c, sz b) {
    if (c != D) throw LE("D");
    for (sz i = 0; i < D; ++i) {
      sl_[i].Init(b);
      st_[i] = F;
    }
  }
  template <class T>
  Tensor<T> AllocTensor() {
    for (sz i = 0; i < D; ++i) {
      if (st_[i] == F) {
        st_[i] = A;
        return sl_[i].template Get<T>();
      }
    }
    throw LE("O");
  }
  template <class T>
  void EnQue(Tensor<T> t) {
    sz i = Idx(t);
    if (st_[i] != A) throw LE("Q");
    st_[i] = Q;
    q_[t_] = i;
    t_ = (t_ + 1) % D;
    qPeak = max(qPeak, ++l_);
  }
  template <class T>
  Tensor<T> DeQue() {
    if (!l_) throw LE("E");
    sz i = q_[h_];
    h_ = (h_ + 1) % D;
    --l_;
    if (st_[i] != Q) throw LE("B");
    st_[i] = U;
    return sl_[i].template Get<T>();
  }
  template <class T>
  void FreeTensor(Tensor<T> t) {
    sz i = Idx(t);
    if (st_[i] != U) throw LE("R");
    st_[i] = F;
  }
  ~TQueBind() noexcept(false) {
    if (!uncaught_exceptions() && any_of(st_, st_ + D, [](int s) { return s != F; })) throw LE("U");
  }
};

struct TPipe {
  sz ub_ = 0;
  void AddUb(sz b) {
    if (b > 196608 || (ub_ += ((b + 31) / 32) * 32) > 196608) throw OE("UB");
  }
  template <QP P>
  void InitBuffer(TBuf<P>& b, sz n) {
    AddUb(n);
    b.Init(n);
  }
  template <class Q>
  void InitBuffer(Q& q, sz c, sz n) {
    for (sz i = 0; i < c; ++i) AddUb(n);
    q.Init(c, n);
  }
  int FetchEventID(HardEvent) const { return 0; }
};

inline TPipe* GetTPipePtr() {
  static TPipe p;
  return &p;
}
template <HardEvent>
void SetFlag(event_t) {}
template <HardEvent>
void WaitFlag(event_t) {}
template <int>
void PipeBarrier() {}

struct DataCopyParams {
  u32 blocks, bytes, sourceStride, destinationStride, reserved = 0;
};
struct PaddingParams {
  bool enabled;
  u32 left, right;
  float value;
};
template <class T>
using DataCopyPadExtParams = PaddingParams;

template <class T, bool D, bool S>
void DataCopy(Tensor<T, D> d, Tensor<T, S> s, sz n) {
  if ((n * sizeof(T)) % 32 != 0) throw LE("DataCopy 32B align");
  memcpy(d.Acc(n * sizeof(T)), s.Acc(n * sizeof(T)), n * sizeof(T));
}

template <class T, bool D, bool S>
void DataCopyPad(Tensor<T, D> d, Tensor<T, S> s, DataCopyParams p, PaddingParams x = {false, 0, 0, 0}) {
  if (x.enabled || p.blocks != 1 || p.sourceStride || p.destinationStride || p.bytes % sizeof(T)) throw RE("Pad");
  memcpy(d.Acc(p.bytes), s.Acc(p.bytes), p.bytes);
}

template <class T>
void Duplicate(Tensor<T> d, T v, sz n) {
  for (sz i = 0; i < n; ++i) d.SetValue(i, v);
}
template <class T>
void ArithProgression(Tensor<T> d, T a, T s, sz n) {
  for (sz i = 0; i < n; ++i) d.SetValue(i, a + static_cast<T>(i) * s);
}
template <class T>
void And(Tensor<T> d, Tensor<T> a, Tensor<T> b, sz n) {
  for (sz i = 0; i < n; ++i) d.SetValue(i, a.GetValue(i) & b.GetValue(i));
}
template <class T>
void Or(Tensor<T> d, Tensor<T> a, Tensor<T> b, sz n) {
  for (sz i = 0; i < n; ++i) d.SetValue(i, a.GetValue(i) | b.GetValue(i));
}

inline void CompareScalar(Tensor<u8> d, Tensor<float> s, float v, CMPMODE m, sz n) {
  if ((n * sizeof(float)) % 256 != 0) throw LE("CompareScalar 256B align");
  Duplicate(d, u8{0}, (n + 7) / 8);
  for (sz i = 0; i < n; ++i) {
    if (m == CMPMODE::GE ? s.GetValue(i) >= v : s.GetValue(i) < v) {
      d.SetValue(i / 8, d.GetValue(i / 8) | (1U << (i % 8)));
    }
  }
}

template <class D, class S>
void Cast(Tensor<D> d, Tensor<S> s, RoundMode, sz n) {
  for (sz i = 0; i < n; ++i) d.SetValue(i, static_cast<D>(s.GetValue(i)));
}

struct GatherMaskParams {
  u32 repeatTimes, src0BlockStride, src0RepeatStride, src1RepeatStride;
};

inline void GatherMask(Tensor<u32> d, Tensor<u32> s, Tensor<u32> b, bool, u32 n, GatherMaskParams, u64& sel) {
  sel = 0;
  for (sz i = 0; i < n; ++i) {
    if (b.GetValue(i / 32) & (1U << (i % 32))) d.SetValue(sel++, s.GetValue(i));
  }
}
}  // namespace AscendC

SIM_EXPORT void selected_kv_sim_reset() {
  AscendC::pReg.clear();
  AscendC::cReg.clear();
  AscendC::qPeak = AscendC::tCnt = 0;
}
SIM_EXPORT size_t selected_kv_sim_peak_queue() { return AscendC::qPeak; }
SIM_EXPORT size_t selected_kv_sim_trap_count() { return AscendC::tCnt; }
