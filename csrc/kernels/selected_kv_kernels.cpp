// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include <cstdint>

// Invalid device metadata or union status triggers AscendC::Trap() and returns immediately
// without draining faulted UB queues, preventing downstream SFA from reading stale two-slot KV.
#ifdef SELECTED_KV_HOST_SIM
  #define SELECTED_KV_CHECK(c, m)     \
    do {                              \
      if (!(c)) {                     \
        AscendC::SetSimTrapReason(m); \
        AscendC::Trap();              \
      }                               \
    } while (0)
  #define SELECTED_KV_LAUNCH(b, s, f, ...)                  \
    do {                                                    \
      (void)(s);                                            \
      AscendC::RunHostKernel((b), [&]() { f(__VA_ARGS__); }); \
    } while (0)
#else
  #define SELECTED_KV_CHECK(c, m) \
    do {                          \
      if (!(c)) AscendC::Trap();  \
    } while (0)
  #define SELECTED_KV_LAUNCH(b, s, f, ...) f<<<(b), nullptr, (s)>>>(__VA_ARGS__)
#endif

namespace {
using namespace AscendC;
constexpr uint32_t CORES = 32U, MAX_LI_CORES = 16U, TOPK = 2048U, CHUNK = 2048U, STRIDE = 8U, PAGE_ROWS = 128U;
constexpr uint32_t ROW_BYTES = 656U, MAX_HISTORY = 1048576U, MAX_WORDS_PER_CORE = (MAX_HISTORY / 32U) / CORES;
constexpr uint32_t MASK_STRIDE = 256U, DATA_PAGE_BYTES = PAGE_ROWS * 128U, SCALE_PAGE_BYTES = 256U;
constexpr uint32_t LI_SLOT_BYTES = DATA_PAGE_BYTES + SCALE_PAGE_BYTES, MAX_LOCAL_IDS = 8192U, MAX_COLUMNS = 8192U;
constexpr uint32_t MAX_COPY_BYTES = 32768U, REDUCE_ROWS = 4096U, GROUP_SCALE_IDS = 16384U;
constexpr int32_t MAX_BATCHED_TOKENS = 8192;

#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
using CopyQueue = TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2>;
enum class RowState : uint8_t { Invalid, Full, Selected };
struct RowAddress {
  uint64_t source = 0, destination = 0;
};
struct PageAddress {
  uint32_t column = UINT32_MAX;
  uint64_t dstBase = 0, srcBase = 0;
};

template <HardEvent E>
__aicore__ inline void Fence() {
  event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
  SetFlag<E>(id);
  WaitFlag<E>(id);
}
template <typename T>
__aicore__ inline GlobalTensor<T> BindGm(GM_ADDR a) {
  GlobalTensor<T> t;
  t.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a));
  return t;
}
template <typename T>
__aicore__ inline LocalTensor<T> AllocUb(TPipe& p, TBuf<TPosition::VECCALC>& b, uint32_t n) {
  p.InitBuffer(b, n * sizeof(T));
  return b.Get<T>();
}
template <typename U, typename T>
__aicore__ inline LocalTensor<U> As(LocalTensor<T> t) {
  return t.template ReinterpretCast<U>();
}
__aicore__ inline bool Ok(bool c, const __gm__ char* m) {
  SELECTED_KV_CHECK(c, m);
  return c;
}
__aicore__ inline GatherMaskParams CompactParams() { return {1U, 1U, 8U, 8U}; }
__aicore__ inline uint32_t HistoryPages(uint32_t h) { return (h + PAGE_ROWS - 1U) / PAGE_ROWS; }

__aicore__ inline void WriteWords(LocalTensor<int32_t> b, GlobalTensor<int32_t> d, int32_t w0, int32_t w1 = 0,
                                  int32_t w2 = 0, int32_t w3 = 0, int32_t w4 = 0) {
  const int32_t v[STRIDE] = {w0, w1, w2, w3, w4, 0, 0, 0};
  for (uint32_t i = 0U; i < STRIDE; ++i) {
    b.SetValue(i, v[i]);
  }
  Fence<HardEvent::S_MTE3>();
  DataCopy(d, b, STRIDE);
  Fence<HardEvent::MTE3_S>();
}

__aicore__ inline RowState ResolveRow(int32_t id, int32_t h, LocalTensor<int32_t>& tbl, GlobalTensor<uint64_t>& bases,
                                      uint64_t mainOff, uint32_t srcPgs, uint32_t dstPgs, PageAddress& cached,
                                      RowAddress& addr) {
  if (!Ok(id >= 0 && id < h, "Selected copy token outside history")) {
    return RowState::Invalid;
  }
  const uint32_t uid = static_cast<uint32_t>(id), col = uid / PAGE_ROWS;
  if (col != cached.column) {
    const int32_t pg = tbl.GetValue(col);
    const auto upg = static_cast<uint32_t>(pg);
    if (!Ok(pg >= 0 && upg < srcPgs && upg < dstPgs, "Selected copy page outside source or destination domain")) {
      return RowState::Invalid;
    }
    const uint64_t blob = bases.GetValue(pg);
    if (!Ok((blob == 0U) || (blob <= UINT64_MAX - mainOff), "Selected copy main offset overflows")) {
      return RowState::Invalid;
    }
    cached = {col, static_cast<uint64_t>(pg) * PAGE_ROWS * ROW_BYTES, (blob == 0U) ? 0U : (blob + mainOff)};
  }
  const uint64_t rOff = static_cast<uint64_t>(uid % PAGE_ROWS) * ROW_BYTES;
  addr.destination = cached.dstBase + rOff;
  if (cached.srcBase == 0U) {
    return RowState::Full;
  }
  if (!Ok(cached.srcBase <= UINT64_MAX - rOff - (ROW_BYTES - 1U), "Invalid selected row address")) {
    return RowState::Invalid;
  }
  addr.source = cached.srcBase + rOff;
  return RowState::Selected;
}

__aicore__ inline bool Adjacent(const RowAddress& cur, const RowAddress& nxt, uint32_t run) {
  const uint64_t b = static_cast<uint64_t>(run) * ROW_BYTES;
  return nxt.source >= cur.source && nxt.source - cur.source == b && nxt.destination >= cur.destination &&
         nxt.destination - cur.destination == b;
}

__aicore__ inline void CopyOut(CopyQueue& q, GlobalTensor<uint8_t>& dst, uint64_t off, uint32_t bytes) {
  auto loc = q.DeQue<uint8_t>();
  DataCopyPad(dst[off], loc, {1, bytes, 0, 0, 0});
  q.FreeTensor(loc);
}

__aicore__ inline void FlushLiPage(CopyQueue& q, GlobalTensor<uint8_t>& k, GlobalTensor<uint8_t>& s, uint64_t pg) {
  auto loc = q.DeQue<uint8_t>();
  DataCopyPad(k[pg * DATA_PAGE_BYTES], loc, {1, DATA_PAGE_BYTES, 0, 0, 0});
  DataCopyPad(s[pg * SCALE_PAGE_BYTES], loc[DATA_PAGE_BYTES], {1, SCALE_PAGE_BYTES, 0, 0, 0});
  q.FreeTensor(loc);
}
#endif
}  // namespace

// Three-stage 32-core UB bitset union (map -> reduce -> finish) separated by GM launch barriers.
extern "C" __global__ __aicore__ void selected_kv_map_queries_kernel(
    GM_ADDR topk, GM_ADDR partBits, GM_ADDR partErr, uint32_t history, uint32_t query, uint32_t wordsPerCore) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
  auto topkGm = BindGm<int32_t>(topk), errGm = BindGm<int32_t>(partErr);
  auto partBitsGm = BindGm<uint32_t>(partBits);
  const uint32_t totalWords = CORES * wordsPerCore, perCore = query * (TOPK / CORES);
  TPipe pipe;
  TBuf<TPosition::VECCALC> b0, b1, b2, b3, b4, b5;
  auto bits = AllocUb<uint32_t>(pipe, b0, totalWords), valid = AllocUb<uint32_t>(pipe, b3, CHUNK);
  auto raw = AllocUb<int32_t>(pipe, b1, CHUNK), errOut = AllocUb<int32_t>(pipe, b5, STRIDE);
  auto flt = AllocUb<float>(pipe, b2, CHUNK);
  auto live = AllocUb<uint8_t>(pipe, b4, MASK_STRIDE * 4U);
  auto hi = live[MASK_STRIDE], bad = live[MASK_STRIDE * 2U], extra = live[MASK_STRIDE * 3U];
  for (uint32_t off = 0U; off < totalWords; off += 8192U) {
    Duplicate(As<int32_t>(bits[off]), 0, totalWords - off < 8192U ? totalWords - off : 8192U);
  }
  Duplicate(As<uint16_t>(live), static_cast<uint16_t>(0U), MASK_STRIDE * 2U);
  Fence<HardEvent::V_S>();
  const uint64_t baseOff = static_cast<uint64_t>(GetBlockIdx()) * perCore;
  uint32_t error = 0U;
  const float hF = static_cast<float>(static_cast<int32_t>(history));
  const float limF = static_cast<float>(static_cast<int32_t>(history + query));
  for (uint32_t off = 0U; off < perCore; off += CHUNK) {
    if (off != 0U) {
      Fence<HardEvent::V_MTE2>();
      Fence<HardEvent::S_V>();
    }
    const uint32_t n = perCore - off < CHUNK ? perCore - off : CHUNK;
    DataCopy(raw, topkGm[baseOff + off], n);
    Fence<HardEvent::MTE2_V>();
    Cast(flt, raw, RoundMode::CAST_NONE, n);
    PipeBarrier<PIPE_V>();
    CompareScalar(live, flt, 0.0F, CMPMODE::GE, n);
    CompareScalar(hi, flt, hF, CMPMODE::LT, n);
    CompareScalar(bad, flt, -1.0F, CMPMODE::LT, n);
    CompareScalar(extra, flt, limF, CMPMODE::GE, n);
    PipeBarrier<PIPE_V>();
    And(As<uint16_t>(live), As<uint16_t>(live), As<uint16_t>(hi), n / 16U);
    Or(As<uint16_t>(bad), As<uint16_t>(bad), As<uint16_t>(extra), n / 16U);
    PipeBarrier<PIPE_V>();
    uint64_t vc = 0U;
    GatherMask(valid, As<uint32_t>(raw), As<uint32_t>(live), true, n, CompactParams(), vc);
    Fence<HardEvent::V_S>();
    auto badW = As<uint32_t>(bad);
    for (uint32_t w = 0U; w < n / 32U; ++w) {
      error |= badW.GetValue(w);
    }
    uint32_t curW = UINT32_MAX, curM = 0U;
    for (uint32_t i = 0U; i < static_cast<uint32_t>(vc); ++i) {
      const uint32_t id = valid.GetValue(i);
      if ((id >> 5U) != curW) {
        if (curM != 0U) {
          bits.SetValue(curW, bits.GetValue(curW) | curM);
        }
        curW = id >> 5U;
        curM = 0U;
      }
      curM |= (1U << (id & 31U));
    }
    if (curM != 0U) {
      bits.SetValue(curW, bits.GetValue(curW) | curM);
    }
  }
  Fence<HardEvent::S_MTE3>();
  DataCopy(partBitsGm[static_cast<uint64_t>(GetBlockIdx()) * totalWords], bits, totalWords);
  Fence<HardEvent::MTE3_S>();
  WriteWords(errOut, errGm[GetBlockIdx() * STRIDE], error != 0U ? 1 : 0);
#endif
}

extern "C" __global__ __aicore__ void selected_kv_reduce_bitmap_kernel(
    GM_ADDR partBits, GM_ADDR partErr, GM_ADDR groups, GM_ADDR bitmap, uint32_t wordsPerCore) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
  auto partBitsGm = BindGm<uint32_t>(partBits), bitmapGm = BindGm<uint32_t>(bitmap);
  auto errGm = BindGm<int32_t>(partErr), groupsGm = BindGm<int32_t>(groups);
  const uint32_t totalWords = CORES * wordsPerCore, firstWord = GetBlockIdx() * wordsPerCore;
  TPipe pipe;
  TBuf<TPosition::VECCALC> b0, b1, b2, b3, b4;
  auto mat = AllocUb<uint32_t>(pipe, b0, totalWords), dSrc = AllocUb<uint32_t>(pipe, b1, REDUCE_ROWS);
  auto dDst = AllocUb<uint32_t>(pipe, b2, REDUCE_ROWS);
  auto allErr = AllocUb<int32_t>(pipe, b3, CORES * STRIDE), stats = AllocUb<int32_t>(pipe, b4, STRIDE);
  for (uint32_t p = 0U; p < CORES; ++p) {
    DataCopy(mat[p * wordsPerCore], partBitsGm[static_cast<uint64_t>(p) * totalWords + firstWord], wordsPerCore);
  }
  DataCopy(allErr, errGm, CORES * STRIDE);
  Fence<HardEvent::MTE2_V>();
  Fence<HardEvent::MTE2_S>();
  for (uint32_t half = CORES / 2U; half > 0U; half >>= 1U) {
    const uint32_t words = half * wordsPerCore;
    for (uint32_t off = 0U; off < words; off += 8192U) {
      const uint32_t w = words - off < 8192U ? words - off : 8192U;
      Or(As<uint16_t>(mat[off]), As<uint16_t>(mat[off]), As<uint16_t>(mat[words + off]), w * 2U);
    }
    PipeBarrier<PIPE_V>();
  }
  Duplicate(As<int32_t>(dSrc), 0, REDUCE_ROWS);
  PipeBarrier<PIPE_V>();
  uint32_t count = 0U, error = 0U;
  for (uint32_t off = 0U; off < wordsPerCore * 32U; off += REDUCE_ROWS) {
    if (off != 0U) {
      PipeBarrier<PIPE_V>();
    }
    const uint32_t rows = wordsPerCore * 32U - off < REDUCE_ROWS ? wordsPerCore * 32U - off : REDUCE_ROWS;
    uint64_t sub = 0U;
    GatherMask(dDst, dSrc, mat[off / 32U], true, rows, CompactParams(), sub);
    Fence<HardEvent::V_S>();
    count += static_cast<uint32_t>(sub);
  }
  for (uint32_t p = 0U; p < CORES; ++p) {
    error |= static_cast<uint32_t>(allErr.GetValue(p * STRIDE));
  }
  Fence<HardEvent::V_MTE3>();
  DataCopy(bitmapGm[firstWord], mat, wordsPerCore);
  Fence<HardEvent::MTE3_S>();
  WriteWords(stats, groupsGm[GetBlockIdx() * STRIDE], static_cast<int32_t>(count), static_cast<int32_t>(error));
#endif
}

extern "C" __global__ __aicore__ void selected_kv_finish_bitmap_kernel(
    GM_ADDR groups, GM_ADDR bitmap, GM_ADDR sources, GM_ADDR lengths, uint32_t query, uint32_t capacity,
    uint32_t wordsPerCore) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
  if (!Ok(wordsPerCore >= 64U && wordsPerCore <= MAX_WORDS_PER_CORE && wordsPerCore % 64U == 0U,
          "Invalid compact bitmap extent")) {
    return;
  }
  auto groupsGm = BindGm<int32_t>(groups), sourcesGm = BindGm<int32_t>(sources), lengthsGm = BindGm<int32_t>(lengths);
  auto bitmapGm = BindGm<uint32_t>(bitmap);
  TPipe pipe;
  TBuf<TPosition::VECCALC> b0, b1, b2, b3, b4;
  auto bits = AllocUb<uint32_t>(pipe, b0, wordsPerCore);
  auto in = AllocUb<int32_t>(pipe, b1, 8192U);
  auto out = AllocUb<int32_t>(pipe, b2, 8192U), outLen = AllocUb<int32_t>(pipe, b3, STRIDE);
  auto allGroups = AllocUb<int32_t>(pipe, b4, CORES * STRIDE);
  DataCopy(allGroups, groupsGm, CORES * STRIDE);
  Fence<HardEvent::MTE2_S>();
  uint32_t before = 0U, total = 0U, error = 0U;
  for (uint32_t g = 0U; g < CORES; ++g) {
    const uint32_t c = static_cast<uint32_t>(allGroups.GetValue(g * STRIDE));
    if (g < GetBlockIdx()) {
      before += c;
    }
    total += c;
    error |= static_cast<uint32_t>(allGroups.GetValue(g * STRIDE + 1U));
  }
  if (GetBlockIdx() == 0U) {
    WriteWords(outLen, lengthsGm, static_cast<int32_t>(total), static_cast<int32_t>(query),
               static_cast<int32_t>(total + query), static_cast<int32_t>(error), total > capacity ? 1 : 0);
  }
  if ((total > capacity) || (error != 0U)) {
    return;
  }
  const uint32_t firstWord = GetBlockIdx() * wordsPerCore;
  uint32_t packed = 0U;
  DataCopy(bits, bitmapGm[firstWord], wordsPerCore);
  Fence<HardEvent::MTE2_V>();
  Fence<HardEvent::S_V>();
  for (uint32_t off = 0U; off < wordsPerCore * 32U; off += 8192U) {
    const uint32_t rows = wordsPerCore * 32U - off < 8192U ? wordsPerCore * 32U - off : 8192U;
    uint64_t cnt = 0U;
    ArithProgression(in, static_cast<int32_t>(firstWord * 32U + off), 1, rows);
    PipeBarrier<PIPE_V>();
    GatherMask(As<uint32_t>(out), As<uint32_t>(in), bits[off / 32U], true, rows, CompactParams(), cnt);
    Fence<HardEvent::V_S>();
    const uint32_t sel = static_cast<uint32_t>(cnt);
    if (sel != 0U) {
      Fence<HardEvent::V_MTE3>();
      DataCopyPad(sourcesGm[before + packed], out, {1, static_cast<uint16_t>(sel * 4U), 0, 0});
      Fence<HardEvent::MTE3_S>();
    }
    packed += sel;
    Fence<HardEvent::S_V>();
  }
#endif
}

// Partitions 32 AIV cores into groups of `groupCores` lanes; each group streams contiguous Main-KV row runs.
extern "C" __global__ __aicore__ void selected_kv_inplace_copy_kernel(
    GM_ADDR ids, GM_ADDR status, GM_ADDR objectBases, GM_ADDR metadata, GM_ADDR table, GM_ADDR cache,
    uint64_t mainOffset, uint32_t requests, uint32_t capacity, uint32_t sourcePages, uint32_t tableRows,
    uint32_t tableColumns, uint32_t tableStride, uint32_t destinationPages, uint32_t groupCores) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
  auto sel = BindGm<int32_t>(ids), state = BindGm<int32_t>(status), meta = BindGm<int32_t>(metadata);
  auto blockTable = BindGm<int32_t>(table);
  auto pages = BindGm<uint64_t>(objectBases);
  auto dst = BindGm<uint8_t>(cache);
  GlobalTensor<uint8_t> src;
  TPipe pipe;
  CopyQueue copyQ;
  TBuf<TPosition::VECCALC> idsBuf, tblBuf;
  pipe.InitBuffer(copyQ, 2, MAX_COPY_BYTES);
  auto localIds = AllocUb<int32_t>(pipe, idsBuf, MAX_LOCAL_IDS), localTbl = AllocUb<int32_t>(pipe, tblBuf, MAX_COLUMNS);
  if (!Ok(groupCores > 0U && groupCores <= CORES && (CORES % groupCores == 0U), "Invalid copy group geometry")) {
    return;
  }
  const uint32_t group = GetBlockIdx() / groupCores, lane = GetBlockIdx() % groupCores;
  const uint32_t groups = GetBlockNum() / groupCores, maxRun = MAX_COPY_BYTES / ROW_BYTES;
  bool usedUb = false;
  for (uint32_t r = group; r < requests; r += groups) {
    const int32_t row = meta.GetValue(r * 4U), h = meta.GetValue(r * 4U + 1U);
    const int32_t qs = meta.GetValue(r * 4U + 2U), q = meta.GetValue(r * 4U + 3U), cv = state.GetValue(r * 8U);
    const bool valid = row >= 0 && static_cast<uint32_t>(row) < tableRows && h > 0 &&
                       h <= static_cast<int32_t>(MAX_HISTORY) &&
                       HistoryPages(static_cast<uint32_t>(h)) <= tableColumns &&
                       q > 0 && q <= MAX_BATCHED_TOKENS && qs >= 0 && qs <= MAX_BATCHED_TOKENS - q && cv >= 0 &&
                       static_cast<uint32_t>(cv) <= capacity && cv <= h && state.GetValue(r * 8U + 1U) == q &&
                       state.GetValue(r * 8U + 3U) == 0 && state.GetValue(r * 8U + 4U) == 0;
    if (!Ok(valid, "Invalid original-page copy metadata or union status")) {
      return;
    }
    const uint32_t cnt = static_cast<uint32_t>(cv), perLane = (cnt + groupCores - 1U) / groupCores;
    const uint32_t begin = lane * perLane, end = begin + perLane < cnt ? begin + perLane : cnt;
    if (begin >= end) {
      continue;
    }
    if (usedUb) {
      Fence<HardEvent::S_MTE2>();
    }
    usedUb = true;
    DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    DataCopyPad(localTbl, blockTable[static_cast<uint64_t>(row) * tableStride],
                {1, HistoryPages(static_cast<uint32_t>(h)) * 4U, 0, 0, 0}, pad);
    uint32_t pendBytes = 0U;
    uint64_t pendDst = 0U;
    PageAddress cached;
    for (uint32_t span = begin; span < end; span += MAX_LOCAL_IDS) {
      if (span != begin) {
        Fence<HardEvent::S_MTE2>();
      }
      const uint32_t spanEnd = span + MAX_LOCAL_IDS < end ? span + MAX_LOCAL_IDS : end;
      DataCopyPad(localIds, sel[static_cast<uint64_t>(r) * capacity + span], {1, (spanEnd - span) * 4U, 0, 0, 0}, pad);
      Fence<HardEvent::MTE2_S>();
      uint32_t curIdx = span;
      while (curIdx < spanEnd) {
        RowAddress cur, nxt;
        const auto st = ResolveRow(localIds.GetValue(curIdx - span), h, localTbl, pages, mainOffset, sourcePages,
                                   destinationPages, cached, cur);
        if (st == RowState::Invalid) {
          return;
        }
        if (st == RowState::Full) {
          ++curIdx;
          continue;
        }
        uint32_t run = 1U;
        while (run < maxRun && curIdx + run < spanEnd) {
          const auto nxtSt = ResolveRow(localIds.GetValue(curIdx + run - span), h, localTbl, pages, mainOffset,
                                        sourcePages, destinationPages, cached, nxt);
          if (nxtSt == RowState::Invalid) {
            return;
          }
          if (nxtSt == RowState::Full || !Adjacent(cur, nxt, run)) {
            break;
          }
          ++run;
        }
        src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(cur.source));
        auto loc = copyQ.AllocTensor<uint8_t>();
        DataCopyPad(loc, src, {1, run * ROW_BYTES, 0, 0, 0}, DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
        copyQ.EnQue(loc);
        if (pendBytes != 0U) {
          CopyOut(copyQ, dst, pendDst, pendBytes);
        }
        pendDst = cur.destination;
        pendBytes = run * ROW_BYTES;
        curIdx += run;
      }
    }
    if (pendBytes != 0U) {
      CopyOut(copyQ, dst, pendDst, pendBytes);
    }
  }
  PipeBarrier<PIPE_ALL>();
#endif
}

extern "C" __global__ __aicore__ void selected_kv_li_gather_kernel(
    GM_ADDR objectBases, GM_ADDR targetK, GM_ADDR targetScale, uint32_t domain, uint64_t dataOffset,
    uint64_t scaleOffset) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
  auto bases = BindGm<uint64_t>(objectBases);
  auto dstK = BindGm<uint8_t>(targetK), dstScale = BindGm<uint8_t>(targetScale);
  GlobalTensor<uint8_t> src;
  TPipe pipe;
  CopyQueue copyQ;
  pipe.InitBuffer(copyQ, 2, LI_SLOT_BYTES);
  DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
  uint64_t pendPage = 0U;
  bool hasPend = false;
  for (uint64_t page = GetBlockIdx(); page < domain; page += GetBlockNum()) {
    const uint64_t blob = bases.GetValue(page);
    if (blob == 0U) {
      continue;
    }
    if (!Ok(blob <= UINT64_MAX - dataOffset - DATA_PAGE_BYTES && blob <= UINT64_MAX - scaleOffset - SCALE_PAGE_BYTES,
            "LI component address overflows")) {
      return;
    }
    auto loc = copyQ.AllocTensor<uint8_t>();
    src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(blob + dataOffset));
    DataCopyPad(loc, src, {1, DATA_PAGE_BYTES, 0, 0, 0}, pad);
    src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(blob + scaleOffset));
    DataCopyPad(loc[DATA_PAGE_BYTES], src, {1, SCALE_PAGE_BYTES, 0, 0, 0}, pad);
    copyQ.EnQue(loc);
    if (hasPend) {
      FlushLiPage(copyQ, dstK, dstScale, pendPage);
    }
    pendPage = page;
    hasPend = true;
  }
  if (hasPend) {
    FlushLiPage(copyQ, dstK, dstScale, pendPage);
  }
  PipeBarrier<PIPE_ALL>();
#endif
}

namespace vllm_ascend {
void launch_selected_kv_union_plan(void* s, void* topk, void* pBits, void* bmap, void* stWords, void* srcs, void* lens,
                                   uint32_t h, uint32_t q, uint32_t cap) {
  const uint32_t g = (h + CORES * 2048U - 1U) / (CORES * 2048U), wpc = (g == 0U ? 1U : g) * 64U;
  auto* pErr = static_cast<uint8_t*>(stWords);
  auto* gStats = pErr + CORES * STRIDE * sizeof(int32_t);
  auto* tk = static_cast<uint8_t*>(topk);
  auto* pb = static_cast<uint8_t*>(pBits);
  auto* bm = static_cast<uint8_t*>(bmap);
  SELECTED_KV_LAUNCH(CORES, s, selected_kv_map_queries_kernel, tk, pb, pErr, h, q, wpc);
  SELECTED_KV_LAUNCH(CORES, s, selected_kv_reduce_bitmap_kernel, pb, pErr, gStats, bm, wpc);
  SELECTED_KV_LAUNCH(CORES, s, selected_kv_finish_bitmap_kernel, gStats, bm, static_cast<uint8_t*>(srcs),
                     static_cast<uint8_t*>(lens), q, cap, wpc);
}

void launch_selected_kv_inplace_copy(void* s, void* ids, void* st, void* bases, void* meta, void* tbl, void* cache,
                                     uint64_t mOff, uint32_t reqs, uint32_t cap, uint32_t sPgs, uint32_t tRows,
                                     uint32_t tCols, uint32_t tStride, uint32_t dPgs) {
  uint32_t gCores = 8U;
  while (gCores < CORES && gCores * GROUP_SCALE_IDS < cap) {
    gCores *= 2U;
  }
  SELECTED_KV_LAUNCH(CORES, s, selected_kv_inplace_copy_kernel, static_cast<uint8_t*>(ids), static_cast<uint8_t*>(st),
                     static_cast<uint8_t*>(bases), static_cast<uint8_t*>(meta), static_cast<uint8_t*>(tbl),
                     static_cast<uint8_t*>(cache), mOff, reqs, cap, sPgs, tRows, tCols, tStride, dPgs, gCores);
}

void launch_selected_kv_li_gather(void* s, void* bases, void* dstK, void* dstS, uint32_t dom, uint64_t dOff,
                                  uint64_t sOff) {
  SELECTED_KV_LAUNCH(dom < MAX_LI_CORES ? dom : MAX_LI_CORES, s, selected_kv_li_gather_kernel,
                     static_cast<uint8_t*>(bases), static_cast<uint8_t*>(dstK), static_cast<uint8_t*>(dstS), dom, dOff,
                     sOff);
}
}  // namespace vllm_ascend
