// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#include <cstdint>

#ifdef HISPARSE_PREFILL_HOST_SIM
  #define HISPARSE_PREFILL_CHECK(c, m) \
    do {                               \
      if (!(c)) {                      \
        AscendC::SetSimTrapReason(m);  \
        AscendC::Trap();               \
      }                                \
    } while (0)
  #define HISPARSE_PREFILL_LAUNCH(b, s, f, ...)             \
    do {                                                    \
      (void)(s);                                            \
      AscendC::RunHostKernel((b), [&]() { f(__VA_ARGS__); }); \
    } while (0)
#else
  #define HISPARSE_PREFILL_CHECK(c, m) \
    do {                               \
      if (!(c)) AscendC::Trap();       \
    } while (0)
  #define HISPARSE_PREFILL_LAUNCH(b, s, f, ...) f<<<(b), nullptr, (s)>>>(__VA_ARGS__)
#endif

namespace {
using namespace AscendC;

constexpr uint32_t CORES = 32U;
constexpr uint32_t TOPK = 2048U;
constexpr uint32_t CHUNK = 2048U;
constexpr uint32_t STRIDE = 8U;
constexpr uint32_t PAGE_ROWS = 128U;
constexpr uint32_t ROW_BYTES = 656U;
constexpr uint32_t MAX_HISTORY = 1048576U;
constexpr uint32_t MAX_BITMAP_WORDS = MAX_HISTORY / 32U;
constexpr uint32_t SUPER_WORDS = 8U;
constexpr uint32_t MAX_SUPER_BLOCKS = MAX_BITMAP_WORDS / SUPER_WORDS;
constexpr uint32_t MAX_WORDS_PER_CORE = MAX_BITMAP_WORDS / CORES;
constexpr uint32_t MASK_STRIDE = 256U;
constexpr uint32_t MAX_LOCAL_IDS = 8192U;
constexpr uint32_t MAX_COLUMNS = MAX_HISTORY / PAGE_ROWS;
constexpr uint32_t MAX_FULL_PAGES = MAX_BITMAP_WORDS;
constexpr uint32_t MAX_COPY_BYTES = 32768U;
constexpr uint32_t REDUCE_ROWS = 4096U;
constexpr uint32_t MAX_REQS = 256U;
constexpr int32_t MAX_BATCHED_TOKENS = 65536;

#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
using CopyQueue = TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2>;

struct RowAddress {
  uint64_t source = 0U;
  uint64_t destination = 0U;
};

struct PageCache {
  uint32_t column = UINT32_MAX;
  uint64_t srcPageBase = 0U;
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
  HISPARSE_PREFILL_CHECK(c, m);
  return c;
}

__aicore__ inline GatherMaskParams CompactParams() { return {1U, 1U, 8U, 8U}; }
__aicore__ inline uint32_t HistoryPages(uint32_t h) { return (h + PAGE_ROWS - 1U) / PAGE_ROWS; }

__aicore__ inline uint32_t Popcount32(uint32_t x) {
  x = x - ((x >> 1U) & 0x55555555U);
  x = (x & 0x33333333U) + ((x >> 2U) & 0x33333333U);
  return (((x + (x >> 4U)) & 0x0F0F0F0FU) * 0x01010101U) >> 24U;
}

__aicore__ inline void WriteWords(LocalTensor<int32_t> b, GlobalTensor<int32_t> d, int32_t w0, int32_t w1 = 0,
                                  int32_t w2 = 0, int32_t w3 = 0, int32_t w4 = 0, int32_t w5 = 0) {
  const int32_t v[STRIDE] = {w0, w1, w2, w3, w4, w5, 0, 0};
  for (uint32_t i = 0U; i < STRIDE; ++i) {
    b.SetValue(i, v[i]);
  }
  Fence<HardEvent::S_MTE3>();
  DataCopy(d, b, STRIDE);
  Fence<HardEvent::MTE3_S>();
}

__aicore__ inline void CopyOut(CopyQueue& q, GlobalTensor<uint8_t>& dst, uint64_t off, uint32_t bytes) {
  auto loc = q.DeQue<uint8_t>();
  DataCopyPad(dst[off], loc, {1, bytes, 0, 0, 0});
  q.FreeTensor(loc);
}

__aicore__ inline bool ResolveHistoricalRow(
    int32_t tok,
    int32_t history,
    uint32_t dstRowIndex,
    LocalTensor<int32_t>& srcTbl,
    LocalTensor<int32_t>& resTbl,
    uint64_t hostDvaBase,
    uint64_t hostTotalBytes,
    uint64_t resBase,
    uint64_t resBlockStrideBytes,
    uint32_t resNumBlocks,
    uint32_t stagingRows,
    PageCache& cached,
    RowAddress& out) {
  if (!Ok(tok >= 0 && tok < history && dstRowIndex < stagingRows, "Union token or staging row out of range")) {
    return false;
  }
  const uint32_t utok = static_cast<uint32_t>(tok);
  const uint32_t col = utok / PAGE_ROWS;
  const uint64_t offBytes = static_cast<uint64_t>(utok % PAGE_ROWS) * ROW_BYTES;
  if (col != cached.column) {
    const int32_t resBlk = resTbl.GetValue(col);
    if (resBlk > 0) {
      const uint32_t uRes = static_cast<uint32_t>(resBlk);
      if (!Ok(uRes < resNumBlocks, "Resident block ID out of range")) {
        return false;
      }
      cached = {col, resBase + static_cast<uint64_t>(uRes) * resBlockStrideBytes};
    } else {
      if (!Ok(resBlk == 0, "Negative resident block ID")) {
        return false;
      }
      const int32_t hostBlk = srcTbl.GetValue(col);
      if (!Ok(hostBlk > 0, "Non-positive host source block ID on non-resident page")) {
        return false;
      }
      const uint64_t pageOff = static_cast<uint64_t>(static_cast<uint32_t>(hostBlk)) * PAGE_ROWS * ROW_BYTES;
      if (!Ok(pageOff + PAGE_ROWS * ROW_BYTES <= hostTotalBytes && hostDvaBase <= UINT64_MAX - pageOff,
              "Host source page exceeds mapped host pool")) {
        return false;
      }
      cached = {col, hostDvaBase + pageOff};
    }
  }
  out.source = cached.srcPageBase + offBytes;
  out.destination = static_cast<uint64_t>(dstRowIndex) * ROW_BYTES;
  return true;
}

__aicore__ inline bool SourceAdjacent(const RowAddress& cur, const RowAddress& nxt, uint32_t run) {
  const uint64_t b = static_cast<uint64_t>(run) * ROW_BYTES;
  return nxt.source >= cur.source && (nxt.source - cur.source) == b &&
         nxt.destination >= cur.destination && (nxt.destination - cur.destination) == b;
}
#endif
}  // namespace

// Stage 1: 32-AIV query TopK scan into per-core UB bitmaps.
extern "C" __global__ __aicore__ void hisparse_prefill_map_queries_kernel(
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

// Stage 2: 32-AIV tree OR-reduction across core partitions.
extern "C" __global__ __aicore__ void hisparse_prefill_reduce_bitmap_kernel(
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

// Stage 3: Emit sorted union_ids, status words, AND 256-bit superblock popcount prefixes for O(1) UB remap.
extern "C" __global__ __aicore__ void hisparse_prefill_finish_bitmap_kernel(
    GM_ADDR groups, GM_ADDR bitmap, GM_ADDR superPrefix, GM_ADDR sources, GM_ADDR lengths, uint32_t query,
    uint32_t capacity, uint32_t wordsPerCore) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
  if (!Ok(wordsPerCore >= 64U && wordsPerCore <= MAX_WORDS_PER_CORE && wordsPerCore % 64U == 0U,
          "Invalid compact bitmap extent")) {
    return;
  }
  auto groupsGm = BindGm<int32_t>(groups), sourcesGm = BindGm<int32_t>(sources), lengthsGm = BindGm<int32_t>(lengths);
  auto superPrefixGm = BindGm<int32_t>(superPrefix);
  auto bitmapGm = BindGm<uint32_t>(bitmap);
  TPipe pipe;
  TBuf<TPosition::VECCALC> b0, b1, b2, b3, b4, b5;
  auto bits = AllocUb<uint32_t>(pipe, b0, wordsPerCore);
  auto in = AllocUb<int32_t>(pipe, b1, 8192U);
  auto out = AllocUb<int32_t>(pipe, b2, 8192U), outLen = AllocUb<int32_t>(pipe, b3, STRIDE);
  auto allGroups = AllocUb<int32_t>(pipe, b4, CORES * STRIDE);
  const uint32_t superPerCore = wordsPerCore / SUPER_WORDS;
  auto superUb = AllocUb<int32_t>(pipe, b5, superPerCore);

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
               static_cast<int32_t>(total + query), static_cast<int32_t>(error), total > capacity ? 1 : 0,
               static_cast<int32_t>(wordsPerCore * CORES));
  }
  if ((total > capacity) || (error != 0U)) {
    return;
  }
  const uint32_t firstWord = GetBlockIdx() * wordsPerCore;
  DataCopy(bits, bitmapGm[firstWord], wordsPerCore);
  Fence<HardEvent::MTE2_S>();
  Fence<HardEvent::MTE2_V>();

  uint32_t running = before;
  for (uint32_t sb = 0U; sb < superPerCore; ++sb) {
    superUb.SetValue(sb, static_cast<int32_t>(running));
    for (uint32_t k = 0U; k < SUPER_WORDS; ++k) {
      running += Popcount32(bits.GetValue(sb * SUPER_WORDS + k));
    }
  }
  Fence<HardEvent::S_MTE3>();
  DataCopy(superPrefixGm[firstWord / SUPER_WORDS], superUb, superPerCore);
  Fence<HardEvent::MTE3_S>();

  uint32_t packed = 0U;
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

// Unified heterogeneous [FULL/ON_CARD + ON_DEMAND] physical index remap & order-preserving left compaction.
// reqDesc layout per row (8 x int32):
//   [0] mode (0 = FULL/ON_CARD page-table remap, 1 = ON_DEMAND bitmap+super_prefix remap)
//   [1] H_r (historical token count)
//   [2] Q_r (query token count)
//   [3] q_in_start (start row in topk relative to topkRowOffset)
//   [4] q_out_start (start row in mappedOut)
//   [5] slot_idx (row in pageTable for mode=0; slot in bitmapBuf/superPrefixBuf for mode=1)
//   [6] dst_hist_base (union base row in staging_pool for mode=1; unused for mode=0)
//   [7] aux (num_pages for mode=0; dst_query_base in staging_pool for mode=1)
extern "C" __global__ __aicore__ void hisparse_prefill_remap_topk_kernel(
    GM_ADDR topk, GM_ADDR reqDesc, GM_ADDR bitmapBuf, GM_ADDR superPrefixBuf, GM_ADDR pageTable, GM_ADDR mappedOut,
    uint32_t numReqs, uint32_t topkRowOffset, uint32_t pageTableStride, uint32_t stagingRows, uint32_t bitmapRows,
    uint32_t pageTableRows) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
  if (!Ok(numReqs > 0U && numReqs <= MAX_REQS, "Invalid remap request count")) {
    return;
  }
  auto topkGm = BindGm<int32_t>(topk), descGm = BindGm<int32_t>(reqDesc);
  auto bitmapGm = BindGm<uint32_t>(bitmapBuf);
  auto superGm = BindGm<int32_t>(superPrefixBuf), pageTblGm = BindGm<int32_t>(pageTable);
  auto outGm = BindGm<int32_t>(mappedOut);

  TPipe pipe;
  TBuf<TPosition::VECCALC> b0, b1, b2, b3, b4, b5, b6;
  auto tblOrBits = AllocUb<uint32_t>(pipe, b0, MAX_BITMAP_WORDS);
  auto superPref = AllocUb<int32_t>(pipe, b1, MAX_SUPER_BLOCKS);
  auto rawTopk = AllocUb<int32_t>(pipe, b2, TOPK);
  auto mappedTopk = AllocUb<int32_t>(pipe, b3, TOPK);
  auto compactedTopk = AllocUb<int32_t>(pipe, b4, TOPK);
  auto validBits = AllocUb<uint32_t>(pipe, b5, TOPK / 32U);
  auto descUb = AllocUb<int32_t>(pipe, b6, MAX_REQS * STRIDE);

  DataCopy(descUb, descGm, numReqs * STRIDE);
  Fence<HardEvent::MTE2_S>();

  bool usedLookupUb = false;
  for (uint32_t r = 0U; r < numReqs; ++r) {
    const int32_t mode = descUb.GetValue(r * STRIDE + 0U);
    const int32_t h = descUb.GetValue(r * STRIDE + 1U);
    const int32_t q = descUb.GetValue(r * STRIDE + 2U);
    const int32_t qInStart = descUb.GetValue(r * STRIDE + 3U);
    const int32_t qOutStart = descUb.GetValue(r * STRIDE + 4U);
    const int32_t slotIdx = descUb.GetValue(r * STRIDE + 5U);
    const int32_t dstHistBase = descUb.GetValue(r * STRIDE + 6U);
    const int32_t aux = descUb.GetValue(r * STRIDE + 7U);

    if (!Ok((mode == 0 || mode == 1) && h >= 0 && q > 0 && q <= MAX_BATCHED_TOKENS && qInStart >= 0 && qOutStart >= 0 &&
                slotIdx >= 0,
            "Invalid remap request descriptor")) {
      return;
    }
    if (GetBlockIdx() >= static_cast<uint32_t>(q)) {
      continue;
    }
    if (usedLookupUb) {
      Fence<HardEvent::S_MTE2>();
    }
    usedLookupUb = true;

    uint32_t totalWords = 0U;
    uint32_t numPages = 0U;
    if (mode == 1) {
      if (!Ok(static_cast<uint32_t>(slotIdx) < bitmapRows && static_cast<uint32_t>(h) <= MAX_HISTORY &&
                  dstHistBase >= 0 && aux >= 0,
              "Invalid on-demand remap geometry")) {
        return;
      }
      const uint32_t g = (static_cast<uint32_t>(h) + CORES * 2048U - 1U) / (CORES * 2048U);
      const uint32_t wpc = (g == 0U ? 1U : g) * 64U;
      totalWords = wpc * CORES;
      DataCopy(tblOrBits, bitmapGm[static_cast<uint64_t>(slotIdx) * MAX_BITMAP_WORDS], totalWords);
      DataCopy(superPref, superGm[static_cast<uint64_t>(slotIdx) * MAX_SUPER_BLOCKS], totalWords / SUPER_WORDS);
      Fence<HardEvent::MTE2_S>();
    } else {
      numPages = static_cast<uint32_t>(aux);
      if (!Ok(static_cast<uint32_t>(slotIdx) < pageTableRows && numPages > 0U && numPages <= MAX_FULL_PAGES &&
                  numPages <= pageTableStride,
              "Invalid full-load page table geometry")) {
        return;
      }
      DataCopyPad(As<int32_t>(tblOrBits), pageTblGm[static_cast<uint64_t>(slotIdx) * pageTableStride],
                  {1, numPages * 4U, 0, 0, 0}, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
      Fence<HardEvent::MTE2_S>();
    }

    auto pageTblUb = As<int32_t>(tblOrBits);
    const int32_t seqLimit = h + q;
    for (uint32_t ql = GetBlockIdx(); ql < static_cast<uint32_t>(q); ql += CORES) {
      Fence<HardEvent::S_MTE2>();
      Fence<HardEvent::V_MTE2>();
      const uint64_t inRow = static_cast<uint64_t>(topkRowOffset) + static_cast<uint64_t>(qInStart) + ql;
      DataCopy(rawTopk, topkGm[inRow * TOPK], TOPK);
      Duplicate(compactedTopk, -1, TOPK);
      Fence<HardEvent::MTE2_S>();
      Fence<HardEvent::V_S>();

      const int32_t causalLimit = h + static_cast<int32_t>(ql);
      for (uint32_t w = 0U; w < TOPK / 32U; ++w) {
        uint32_t mWord = 0U;
        for (uint32_t b = 0U; b < 32U; ++b) {
          const uint32_t idx = w * 32U + b;
          const int32_t tok = rawTopk.GetValue(idx);
          int32_t mapped = -1;
          if (tok >= 0 && tok <= causalLimit && tok < seqLimit) {
            if (mode == 0) {
              const uint32_t utok = static_cast<uint32_t>(tok);
              const uint32_t pgCol = utok / PAGE_ROWS;
              if (!Ok(pgCol < numPages, "Full-load token exceeds staged page table")) {
                return;
              }
              const int32_t dstPg = pageTblUb.GetValue(pgCol);
              if (!Ok(dstPg >= 0, "Negative staged page index")) {
                return;
              }
              const uint32_t phys = static_cast<uint32_t>(dstPg) * PAGE_ROWS + (utok % PAGE_ROWS);
              if (!Ok(phys < stagingRows, "Staged full-load physical index exceeds staging_pool")) {
                return;
              }
              mapped = static_cast<int32_t>(phys);
            } else if (tok < h) {
              const uint32_t utok = static_cast<uint32_t>(tok);
              const uint32_t wordIdx = utok >> 5U;
              const uint32_t bitIdx = utok & 31U;
              const uint32_t bWord = tblOrBits.GetValue(wordIdx);
              if (!Ok(wordIdx < totalWords && ((bWord & (1U << bitIdx)) != 0U),
                      "Selected historical token missing from union bitmap")) {
                return;
              }
              const uint32_t sb = wordIdx / SUPER_WORDS;
              uint32_t rank = static_cast<uint32_t>(superPref.GetValue(sb));
              for (uint32_t k = sb * SUPER_WORDS; k < wordIdx; ++k) {
                rank += Popcount32(tblOrBits.GetValue(k));
              }
              rank += Popcount32(bWord & ((1U << bitIdx) - 1U));
              const uint32_t phys = static_cast<uint32_t>(dstHistBase) + rank;
              if (!Ok(phys < stagingRows, "On-demand union row exceeds staging_pool")) {
                return;
              }
              mapped = static_cast<int32_t>(phys);
            } else {
              const uint32_t phys = static_cast<uint32_t>(aux) + static_cast<uint32_t>(tok - h);
              if (!Ok(phys < stagingRows, "On-demand query row exceeds staging_pool")) {
                return;
              }
              mapped = static_cast<int32_t>(phys);
            }
            mWord |= (1U << b);
          }
          mappedTopk.SetValue(idx, mapped);
        }
        validBits.SetValue(w, mWord);
      }

      Fence<HardEvent::S_V>();
      uint64_t vc = 0U;
      GatherMask(As<uint32_t>(compactedTopk), As<uint32_t>(mappedTopk), validBits, true, TOPK, CompactParams(), vc);
      Fence<HardEvent::V_S>();
      const uint32_t uvc = static_cast<uint32_t>(vc);
      const uint32_t alignedVc = (uvc + 7U) & ~7U;
      for (uint32_t i = uvc; i < alignedVc && i < TOPK; ++i) {
        compactedTopk.SetValue(i, -1);
      }
      Fence<HardEvent::S_V>();
      if (alignedVc < TOPK) {
        Duplicate(compactedTopk[alignedVc], -1, TOPK - alignedVc);
      }
      Fence<HardEvent::V_MTE3>();
      const uint64_t outRow = static_cast<uint64_t>(qOutStart) + ql;
      DataCopy(outGm[outRow * TOPK], compactedTopk, TOPK);
      Fence<HardEvent::MTE3_V>();
      Fence<HardEvent::MTE3_S>();
    }
  }
  PipeBarrier<PIPE_ALL>();
#endif
}

// 32-AIV CopyQueue coalesced historical row gather for ON_DEMAND requests into contiguous staging_pool slots.
// reqMeta layout per row (4 x int32):
//   [0] batch_req_idx (index into stateIndices & sourceBlockTable)
//   [1] H_r (historical token count)
//   [2] dst_union_base (destination row offset in staging_pool)
//   [3] union_slot (row index in unionIds & unionStatus)
extern "C" __global__ __aicore__ void hisparse_prefill_gather_union_kernel(
    GM_ADDR unionIds, GM_ADDR unionStatus, GM_ADDR reqMeta, GM_ADDR stateIndices, GM_ADDR sourceBlockTable,
    GM_ADDR residentBlockTable, GM_ADDR residentCache, GM_ADDR stagingPool, uint64_t hostDvaBase,
    uint64_t hostTotalBytes, uint64_t residentBlockStrideBytes, uint32_t numDemandReqs, uint32_t unionStride,
    uint32_t stateCount, uint32_t sourceRows, uint32_t sourceStride, uint32_t residentRows, uint32_t residentStride,
    uint32_t residentNumBlocks, uint32_t stagingRows, uint32_t unionRows, uint32_t groupCores) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
  if (!Ok(groupCores > 0U && groupCores <= CORES && (CORES % groupCores == 0U), "Invalid gather groupCores")) {
    return;
  }
  auto idsGm = BindGm<int32_t>(unionIds), statusGm = BindGm<int32_t>(unionStatus), metaGm = BindGm<int32_t>(reqMeta);
  auto statesGm = BindGm<int32_t>(stateIndices);
  auto srcTblGm = BindGm<int32_t>(sourceBlockTable), resTblGm = BindGm<int32_t>(residentBlockTable);
  auto dstGm = BindGm<uint8_t>(stagingPool);
  const uint64_t resBase = reinterpret_cast<uint64_t>(residentCache);
  GlobalTensor<uint8_t> srcGm;

  TPipe pipe;
  CopyQueue copyQ;
  TBuf<TPosition::VECCALC> idsBuf, srcTblBuf, resTblBuf, hdrBuf;
  pipe.InitBuffer(copyQ, 2, MAX_COPY_BYTES);
  auto localIds = AllocUb<int32_t>(pipe, idsBuf, MAX_LOCAL_IDS);
  auto localSrcTbl = AllocUb<int32_t>(pipe, srcTblBuf, MAX_COLUMNS);
  auto localResTbl = AllocUb<int32_t>(pipe, resTblBuf, MAX_COLUMNS);
  auto hdrUb = AllocUb<int32_t>(pipe, hdrBuf, 32U);

  const uint32_t group = GetBlockIdx() / groupCores, lane = GetBlockIdx() % groupCores;
  const uint32_t groups = GetBlockNum() / groupCores, maxRun = MAX_COPY_BYTES / ROW_BYTES;
  DataCopyPadExtParams<int32_t> padInt{false, 0, 0, 0};
  DataCopyPadExtParams<uint8_t> padByte{false, 0, 0, 0};
  bool usedUb = false;
  bool usedHdrUb = false;

  for (uint32_t m = group; m < numDemandReqs; m += groups) {
    if (usedHdrUb) {
      Fence<HardEvent::S_MTE2>();
    }
    usedHdrUb = true;
    DataCopyPad(hdrUb, metaGm[m * 4U], {1, 16U, 0, 0, 0}, padInt);
    Fence<HardEvent::MTE2_S>();

    const int32_t batchReq = hdrUb.GetValue(0U);
    const int32_t h = hdrUb.GetValue(1U);
    const int32_t dstBase = hdrUb.GetValue(2U);
    const int32_t uSlot = hdrUb.GetValue(3U);

    if (!Ok(batchReq >= 0 && static_cast<uint32_t>(batchReq) < stateCount &&
                static_cast<uint32_t>(batchReq) < sourceRows && h >= 0 && static_cast<uint32_t>(h) <= MAX_HISTORY &&
                dstBase >= 0 && uSlot >= 0 && static_cast<uint32_t>(uSlot) < unionRows,
            "Invalid gather request metadata")) {
      return;
    }
    Fence<HardEvent::S_MTE2>();
    DataCopy(hdrUb[8], statusGm[static_cast<uint64_t>(uSlot) * STRIDE], STRIDE);
    DataCopyPad(hdrUb[16], statesGm[batchReq], {1, 4U, 0, 0, 0}, padInt);
    Fence<HardEvent::MTE2_S>();

    const int32_t cntVal = hdrUb.GetValue(8U);
    const int32_t errVal = hdrUb.GetValue(11U);
    const int32_t ovfVal = hdrUb.GetValue(12U);
    const int32_t stateRow = hdrUb.GetValue(16U);
    if (!Ok(stateRow >= 0 && static_cast<uint32_t>(stateRow) < residentRows && cntVal >= 0 &&
                static_cast<uint32_t>(cntVal) <= unionStride && cntVal <= h && errVal == 0 && ovfVal == 0 &&
                static_cast<uint32_t>(dstBase) + static_cast<uint32_t>(cntVal) <= stagingRows,
            "Invalid union status or state row in gather")) {
      return;
    }
    const uint32_t cnt = static_cast<uint32_t>(cntVal);
    if (cnt == 0U) {
      continue;
    }
    const uint32_t pages = HistoryPages(static_cast<uint32_t>(h));
    if (!Ok(pages <= MAX_COLUMNS && pages <= sourceStride && pages <= residentStride,
            "History pages exceed block table columns")) {
      return;
    }
    const uint32_t perLane = (cnt + groupCores - 1U) / groupCores;
    const uint32_t begin = lane * perLane;
    const uint32_t end = begin + perLane < cnt ? begin + perLane : cnt;
    if (begin >= end) {
      continue;
    }
    if (usedUb) {
      Fence<HardEvent::S_MTE2>();
    }
    usedUb = true;
    DataCopyPad(localSrcTbl, srcTblGm[static_cast<uint64_t>(batchReq) * sourceStride], {1, pages * 4U, 0, 0, 0},
                padInt);
    DataCopyPad(localResTbl, resTblGm[static_cast<uint64_t>(stateRow) * residentStride], {1, pages * 4U, 0, 0, 0},
                padInt);

    uint32_t pendBytes = 0U;
    uint64_t pendDst = 0U;
    PageCache cached;
    for (uint32_t span = begin; span < end; span += MAX_LOCAL_IDS) {
      if (span != begin) {
        Fence<HardEvent::S_MTE2>();
      }
      const uint32_t spanEnd = span + MAX_LOCAL_IDS < end ? span + MAX_LOCAL_IDS : end;
      DataCopyPad(localIds, idsGm[static_cast<uint64_t>(uSlot) * unionStride + span],
                  {1, (spanEnd - span) * 4U, 0, 0, 0}, padInt);
      Fence<HardEvent::MTE2_S>();

      uint32_t curIdx = span;
      while (curIdx < spanEnd) {
        RowAddress cur, nxt;
        if (!ResolveHistoricalRow(localIds.GetValue(curIdx - span), h, static_cast<uint32_t>(dstBase) + curIdx,
                                  localSrcTbl, localResTbl, hostDvaBase, hostTotalBytes, resBase,
                                  residentBlockStrideBytes, residentNumBlocks, stagingRows, cached, cur)) {
          return;
        }
        uint32_t run = 1U;
        while (run < maxRun && curIdx + run < spanEnd) {
          if (!ResolveHistoricalRow(localIds.GetValue(curIdx + run - span), h,
                                    static_cast<uint32_t>(dstBase) + curIdx + run, localSrcTbl, localResTbl,
                                    hostDvaBase, hostTotalBytes, resBase, residentBlockStrideBytes, residentNumBlocks,
                                    stagingRows, cached, nxt)) {
            return;
          }
          if (!SourceAdjacent(cur, nxt, run)) {
            break;
          }
          ++run;
        }
        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(cur.source));
        auto loc = copyQ.AllocTensor<uint8_t>();
        DataCopyPad(loc, srcGm, {1, run * ROW_BYTES, 0, 0, 0}, padByte);
        copyQ.EnQue(loc);
        if (pendBytes != 0U) {
          CopyOut(copyQ, dstGm, pendDst, pendBytes);
        }
        pendDst = cur.destination;
        pendBytes = run * ROW_BYTES;
        curIdx += run;
      }
    }
    if (pendBytes != 0U) {
      CopyOut(copyQ, dstGm, pendDst, pendBytes);
    }
  }
  PipeBarrier<PIPE_ALL>();
#endif
}

// Fast 32-AIV HBM D2D span copier for Resident pages, active tail pages, and ON_DEMAND new Q rows.
// spans layout per row (8 x int32):
//   [0] batch_req_idx
//   [1] logical_page_col
//   [2] row_offset_in_page (0..127)
//   [3] num_rows (1..128)
//   [4] dst_row_in_staging
extern "C" __global__ __aicore__ void hisparse_prefill_copy_spans_kernel(
    GM_ADDR spans, GM_ADDR stateIndices, GM_ADDR residentBlockTable, GM_ADDR residentCache, GM_ADDR stagingPool,
    uint64_t residentBlockStrideBytes, uint32_t numSpans, uint32_t stateCount, uint32_t residentRows,
    uint32_t residentStride, uint32_t residentNumBlocks, uint32_t stagingRows) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if !defined(__CCE_AICORE__) || (__CCE_AICORE__ >= 220)
  if (numSpans == 0U) {
    return;
  }
  auto spansGm = BindGm<int32_t>(spans), statesGm = BindGm<int32_t>(stateIndices);
  auto resTblGm = BindGm<int32_t>(residentBlockTable);
  auto dstGm = BindGm<uint8_t>(stagingPool);
  const uint64_t resBase = reinterpret_cast<uint64_t>(residentCache);
  GlobalTensor<uint8_t> srcGm;

  TPipe pipe;
  CopyQueue copyQ;
  TBuf<TPosition::VECCALC> hdrBuf;
  pipe.InitBuffer(copyQ, 2, MAX_COPY_BYTES);
  auto hdrUb = AllocUb<int32_t>(pipe, hdrBuf, 24U);
  const uint32_t maxRun = MAX_COPY_BYTES / ROW_BYTES;
  DataCopyPadExtParams<int32_t> padInt{false, 0, 0, 0};
  DataCopyPadExtParams<uint8_t> padByte{false, 0, 0, 0};

  uint32_t pendBytes = 0U;
  uint64_t pendDst = 0U;
  bool usedHdrUb = false;
  for (uint32_t s = GetBlockIdx(); s < numSpans; s += GetBlockNum()) {
    if (usedHdrUb) {
      Fence<HardEvent::S_MTE2>();
    }
    usedHdrUb = true;
    DataCopy(hdrUb, spansGm[s * STRIDE], STRIDE);
    Fence<HardEvent::MTE2_S>();

    const int32_t batchReq = hdrUb.GetValue(0U);
    const int32_t pageCol = hdrUb.GetValue(1U);
    const int32_t rowOff = hdrUb.GetValue(2U);
    const int32_t numRows = hdrUb.GetValue(3U);
    const int32_t dstRow = hdrUb.GetValue(4U);
    if (!Ok(batchReq >= 0 && static_cast<uint32_t>(batchReq) < stateCount && pageCol >= 0 &&
                static_cast<uint32_t>(pageCol) < residentStride && rowOff >= 0 && numRows > 0 &&
                static_cast<uint32_t>(rowOff + numRows) <= PAGE_ROWS && dstRow >= 0 &&
                static_cast<uint32_t>(dstRow + numRows) <= stagingRows,
            "Invalid D2D resident span descriptor")) {
      return;
    }
    Fence<HardEvent::S_MTE2>();
    DataCopyPad(hdrUb[8], statesGm[batchReq], {1, 4U, 0, 0, 0}, padInt);
    Fence<HardEvent::MTE2_S>();
    const int32_t stateRow = hdrUb.GetValue(8U);
    if (!Ok(stateRow >= 0 && static_cast<uint32_t>(stateRow) < residentRows, "Invalid state row in D2D span")) {
      return;
    }
    Fence<HardEvent::S_MTE2>();
    DataCopyPad(hdrUb[16], resTblGm[static_cast<uint64_t>(stateRow) * residentStride + static_cast<uint32_t>(pageCol)],
                {1, 4U, 0, 0, 0}, padInt);
    Fence<HardEvent::MTE2_S>();
    const int32_t resBlk = hdrUb.GetValue(16U);
    if (!Ok(resBlk > 0 && static_cast<uint32_t>(resBlk) < residentNumBlocks,
            "Expected positive Resident block ID in D2D span")) {
      return;
    }
    const uint64_t srcPageBase = resBase + static_cast<uint64_t>(static_cast<uint32_t>(resBlk)) * residentBlockStrideBytes;
    uint32_t copied = 0U;
    const uint32_t total = static_cast<uint32_t>(numRows);
    while (copied < total) {
      const uint32_t chunk = (total - copied) < maxRun ? (total - copied) : maxRun;
      const uint64_t curSrc = srcPageBase + static_cast<uint64_t>(static_cast<uint32_t>(rowOff) + copied) * ROW_BYTES;
      const uint64_t curDst = static_cast<uint64_t>(static_cast<uint32_t>(dstRow) + copied) * ROW_BYTES;
      srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(curSrc));
      auto loc = copyQ.AllocTensor<uint8_t>();
      DataCopyPad(loc, srcGm, {1, chunk * ROW_BYTES, 0, 0, 0}, padByte);
      copyQ.EnQue(loc);
      if (pendBytes != 0U) {
        CopyOut(copyQ, dstGm, pendDst, pendBytes);
      }
      pendDst = curDst;
      pendBytes = chunk * ROW_BYTES;
      copied += chunk;
    }
  }
  if (pendBytes != 0U) {
    CopyOut(copyQ, dstGm, pendDst, pendBytes);
  }
  PipeBarrier<PIPE_ALL>();
#endif
}

namespace vllm_ascend {
void launch_hisparse_prefill_union_plan(void* s, void* topk, void* pBits, void* bmap, void* sPref, void* stWords,
                                        void* srcs, void* lens, uint32_t h, uint32_t q, uint32_t cap) {
  const uint32_t g = (h + CORES * 2048U - 1U) / (CORES * 2048U);
  const uint32_t wpc = (g == 0U ? 1U : g) * 64U;
  auto* pErr = static_cast<uint8_t*>(stWords);
  auto* gStats = pErr + CORES * STRIDE * sizeof(int32_t);
  auto* tk = static_cast<uint8_t*>(topk);
  auto* pb = static_cast<uint8_t*>(pBits);
  auto* bm = static_cast<uint8_t*>(bmap);
  auto* sp = static_cast<uint8_t*>(sPref);
  HISPARSE_PREFILL_LAUNCH(CORES, s, hisparse_prefill_map_queries_kernel, tk, pb, pErr, h, q, wpc);
  HISPARSE_PREFILL_LAUNCH(CORES, s, hisparse_prefill_reduce_bitmap_kernel, pb, pErr, gStats, bm, wpc);
  HISPARSE_PREFILL_LAUNCH(CORES, s, hisparse_prefill_finish_bitmap_kernel, gStats, bm, sp, static_cast<uint8_t*>(srcs),
                          static_cast<uint8_t*>(lens), q, cap, wpc);
}

void launch_hisparse_prefill_remap_topk(void* s, void* topk, void* reqDesc, void* bmapBuf, void* sPrefBuf,
                                        void* pageTbl, void* mappedOut, uint32_t numReqs, uint32_t topkRowOffset,
                                        uint32_t pageTblStride, uint32_t stagingRows, uint32_t bitmapRows,
                                        uint32_t pageTableRows) {
  HISPARSE_PREFILL_LAUNCH(CORES, s, hisparse_prefill_remap_topk_kernel, static_cast<uint8_t*>(topk),
                          static_cast<uint8_t*>(reqDesc), static_cast<uint8_t*>(bmapBuf),
                          static_cast<uint8_t*>(sPrefBuf), static_cast<uint8_t*>(pageTbl),
                          static_cast<uint8_t*>(mappedOut), numReqs, topkRowOffset, pageTblStride, stagingRows,
                          bitmapRows, pageTableRows);
}

void launch_hisparse_prefill_gather_union(void* s, void* unionIds, void* unionStatus, void* reqMeta, void* stateIndices,
                                          void* srcTbl, void* resTbl, void* resCache, void* stagingPool,
                                          uint64_t hostDvaBase, uint64_t hostTotalBytes, uint64_t resBlockStrideBytes,
                                          uint32_t numDemandReqs, uint32_t unionStride, uint32_t stateCount,
                                          uint32_t srcRows, uint32_t srcStride, uint32_t resRows, uint32_t resStride,
                                          uint32_t resNumBlocks, uint32_t stagingRows, uint32_t unionRows) {
  uint32_t groupCores = CORES;
  if (numDemandReqs >= 4U) {
    groupCores = 8U;
  } else if (numDemandReqs >= 2U) {
    groupCores = 16U;
  }
  HISPARSE_PREFILL_LAUNCH(CORES, s, hisparse_prefill_gather_union_kernel, static_cast<uint8_t*>(unionIds),
                          static_cast<uint8_t*>(unionStatus), static_cast<uint8_t*>(reqMeta),
                          static_cast<uint8_t*>(stateIndices), static_cast<uint8_t*>(srcTbl),
                          static_cast<uint8_t*>(resTbl), static_cast<uint8_t*>(resCache),
                          static_cast<uint8_t*>(stagingPool), hostDvaBase, hostTotalBytes, resBlockStrideBytes,
                          numDemandReqs, unionStride, stateCount, srcRows, srcStride, resRows, resStride, resNumBlocks,
                          stagingRows, unionRows, groupCores);
}

void launch_hisparse_prefill_copy_spans(void* s, void* spans, void* stateIndices, void* resTbl, void* resCache,
                                        void* stagingPool, uint64_t resBlockStrideBytes, uint32_t numSpans,
                                        uint32_t stateCount, uint32_t resRows, uint32_t resStride,
                                        uint32_t resNumBlocks, uint32_t stagingRows) {
  const uint32_t blocks = numSpans < CORES ? numSpans : CORES;
  if (blocks == 0U) {
    return;
  }
  HISPARSE_PREFILL_LAUNCH(blocks, s, hisparse_prefill_copy_spans_kernel, static_cast<uint8_t*>(spans),
                          static_cast<uint8_t*>(stateIndices), static_cast<uint8_t*>(resTbl),
                          static_cast<uint8_t*>(resCache), static_cast<uint8_t*>(stagingPool), resBlockStrideBytes,
                          numSpans, stateCount, resRows, resStride, resNumBlocks, stagingRows);
}
}  // namespace vllm_ascend
