// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <torch/library.h>
#ifdef HISPARSE_PREFILL_HOST_SIM
  #include "kernel_operator.h"
  #define HISPARSE_PREFILL_BACKEND CPU
  #define HISPARSE_PREFILL_NS _C_ascend_prefill_sim
#else
  #include <acl/acl.h>
  #include <torch_npu/csrc/core/npu/NPUGuard.h>
  #include <torch_npu/csrc/core/npu/NPUStream.h>
  #include <torch_npu/csrc/framework/OpCommand.h>
  #define HISPARSE_PREFILL_BACKEND PrivateUse1
  #define HISPARSE_PREFILL_NS _C_ascend
#endif
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace vllm_ascend {
#if defined(HISPARSE_PREFILL_HOST_SIM) || defined(VLLM_ENABLE_ATB_AND_DIRECT_KERNELS)
void launch_hisparse_prefill_union_plan(void*, void*, void*, void*, void*, void*, void*, void*, uint32_t, uint32_t,
                                        uint32_t);
void launch_hisparse_prefill_remap_topk(void*, void*, void*, void*, void*, void*, void*, uint32_t, uint32_t, uint32_t,
                                        uint32_t, uint32_t, uint32_t);
void launch_hisparse_prefill_gather_union(void*, void*, void*, void*, void*, void*, void*, void*, void*, uint64_t,
                                          uint64_t, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                          uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
void launch_hisparse_prefill_copy_spans(void*, void*, void*, void*, void*, void*, uint64_t, uint32_t, uint32_t,
                                        uint32_t, uint32_t, uint32_t, uint32_t);
#else
template <typename... A>
inline void launch_hisparse_prefill_union_plan(A...) {}
template <typename... A>
inline void launch_hisparse_prefill_remap_topk(A...) {}
template <typename... A>
inline void launch_hisparse_prefill_gather_union(A...) {}
template <typename... A>
inline void launch_hisparse_prefill_copy_spans(A...) {}
#endif
}  // namespace vllm_ascend

namespace {
using TList = c10::List<at::Tensor>;
using IList = c10::List<int64_t>;

constexpr int64_t MAX_HISTORY = 1048576;
constexpr int64_t MAX_BITMAP_WORDS = MAX_HISTORY / 32;
constexpr int64_t MAX_SUPER_BLOCKS = MAX_HISTORY / 256;
constexpr int64_t MAX_TOKENS = 65536;
constexpr int64_t MAX_REQUESTS = 256;
constexpr int64_t TOPK = 2048;
constexpr int64_t ROW_BYTES = 656;
constexpr int64_t PAGE_ROWS = 128;

inline bool IsExpectedDevice(const at::Tensor& t) {
#ifdef HISPARSE_PREFILL_HOST_SIM
  return t.device().is_cpu();
#else
  return t.device().is_privateuseone();
#endif
}

inline bool IsC8PayloadType(at::ScalarType dt) {
  return dt == at::kChar || dt == at::kByte || dt == at::ScalarType::Float8_e4m3fn;
}

inline void CheckContigTensor(const at::Tensor& t, const at::Tensor& ref, at::ScalarType dt,
                              std::initializer_list<int64_t> shape, const char* msg) {
  TORCH_CHECK(t.defined() && t.device() == ref.device() && t.is_contiguous() && t.scalar_type() == dt &&
                  (shape.size() == 0 || static_cast<size_t>(t.dim()) == shape.size()),
              msg);
  size_t ax = 0;
  for (int64_t e : shape) {
    if (e >= 0) {
      TORCH_CHECK(t.size(ax) == e, msg);
    }
    ++ax;
  }
}

inline void CheckRowContigTable(const at::Tensor& t, const at::Tensor& ref, const char* msg) {
  TORCH_CHECK(t.defined() && t.device() == ref.device() && t.scalar_type() == at::kInt && t.dim() == 2 &&
                  t.size(0) > 0 && t.size(1) > 0 && t.stride(1) == 1 && t.stride(0) >= t.size(1) &&
                  t.stride(0) <= UINT32_MAX,
              msg);
}

inline void CheckPackedCache(const at::Tensor& t, const at::Tensor& ref, bool requireContig, const char* msg) {
  TORCH_CHECK(t.defined() && t.device() == ref.device() && IsC8PayloadType(t.scalar_type()) && t.dim() == 3 &&
                  t.size(0) > 0 && t.size(0) <= UINT32_MAX && t.size(1) == PAGE_ROWS &&
                  t.size(2) * t.element_size() == ROW_BYTES && t.stride(2) == 1 && t.stride(1) == t.size(2) &&
                  t.stride(0) >= t.size(1) * t.size(2),
              msg);
  if (requireContig) {
    TORCH_CHECK(t.is_contiguous(), msg);
  }
  TORCH_CHECK(t.size(0) * PAGE_ROWS < (1LL << 31), msg);
}

inline size_t TensorSpanBytes(const at::Tensor& t) {
  if (!t.defined() || t.numel() == 0) {
    return 0;
  }
  int64_t maxElem = 0;
  for (int64_t d = 0; d < t.dim(); ++d) {
    if (t.size(d) > 1) {
      maxElem += (t.size(d) - 1) * t.stride(d);
    }
  }
  return static_cast<size_t>(maxElem + 1) * static_cast<size_t>(t.element_size());
}

template <typename Fn>
void Dispatch([[maybe_unused]] const char* name, [[maybe_unused]] const at::Tensor& ref,
              [[maybe_unused]] std::vector<at::Tensor> keep, [[maybe_unused]] uint64_t extraPtr,
              [[maybe_unused]] uint64_t extraBytes, [[maybe_unused]] Fn launch) {
#ifdef HISPARSE_PREFILL_HOST_SIM
  AscendC::ScopedCallRegions guard;
  for (const auto& t : keep) {
    if (t.defined() && t.numel() > 0) {
      guard.Add(t.data_ptr(), std::max(t.nbytes(), TensorSpanBytes(t)));
    }
  }
  if (extraPtr != 0U && extraBytes != 0U) {
    guard.Add(reinterpret_cast<void*>(static_cast<uintptr_t>(extraPtr)), static_cast<size_t>(extraBytes));
  }
  launch(reinterpret_cast<void*>(1), keep);
#elif defined(VLLM_ENABLE_ATB_AND_DIRECT_KERNELS)
  c10_npu::NPUGuard guard(ref.device());
  const auto s = c10_npu::getCurrentNPUStream().stream(false);
  at_npu::native::OpCommand::RunOpApi(name, [=]() -> int {
    launch(s, keep);
    return 0;
  });
#else
  TORCH_CHECK(false, "HiSparse prefill kernels are not compiled for this target");
#endif
}

// scratch: [part_bits (MAX_HISTORY int32), status_words (64*8 int32)]
// outputs: [union_ids [M, cap], union_status [M, 8], bitmap_buf [M, MAX_HISTORY/32], super_prefix_buf [M, MAX_HISTORY/256]]
void HisparsePrefillUnionPlan(const at::Tensor& topk, const TList& scratch, const TList& outputs,
                              const IList& histories, const IList& starts, const IList& queries, int64_t cap) {
  TORCH_CHECK(IsExpectedDevice(topk), "Invalid TopK device for HiSparse prefill union");
  CheckContigTensor(topk, topk, at::kInt, {-1, TOPK}, "Expected contiguous int32 [T, 2048] TopK");
  TORCH_CHECK(topk.size(0) > 0 && topk.size(0) <= MAX_TOKENS, "TopK token rows out of range");

  const auto h = histories.vec(), st = starts.vec(), q = queries.vec();
  const auto cnt = static_cast<int64_t>(h.size());
  TORCH_CHECK(cnt > 0 && cnt <= MAX_REQUESTS && st.size() == h.size() && q.size() == h.size() && cap > 0 &&
                  cap <= MAX_HISTORY,
              "Invalid HiSparse prefill union geometry");
  for (int64_t i = 0; i < cnt; ++i) {
    TORCH_CHECK(h[i] >= 0 && h[i] <= MAX_HISTORY && q[i] > 0 && q[i] <= topk.size(0) && st[i] >= 0 &&
                    st[i] <= topk.size(0) - q[i],
                "Invalid request H/Q span in HiSparse prefill union");
  }

  TORCH_CHECK(scratch.size() == 2 && outputs.size() == 4, "Wrong scratch/output list length in union plan");
  auto sc = scratch.vec();
  auto out = outputs.vec();
  CheckContigTensor(sc[0], topk, at::kInt, {}, "part_bits must be contiguous int32");
  CheckContigTensor(sc[1], topk, at::kInt, {}, "status_words must be contiguous int32");
  TORCH_CHECK(sc[0].numel() >= MAX_HISTORY && sc[1].numel() >= 64 * 8, "Union scratch buffer too small");

  CheckContigTensor(out[0], topk, at::kInt, {-1, cap}, "union_ids shape mismatch");
  CheckContigTensor(out[1], topk, at::kInt, {-1, 8}, "union_status shape mismatch");
  CheckContigTensor(out[2], topk, at::kInt, {-1, MAX_BITMAP_WORDS}, "bitmap_buf shape mismatch");
  CheckContigTensor(out[3], topk, at::kInt, {-1, MAX_SUPER_BLOCKS}, "super_prefix_buf shape mismatch");
  TORCH_CHECK(out[0].size(0) >= cnt && out[1].size(0) >= cnt && out[2].size(0) >= cnt && out[3].size(0) >= cnt,
              "Union output rows smaller than request count");

  Dispatch("HisparsePrefillUnionPlan", topk, {topk, sc[0], sc[1], out[0], out[1], out[2], out[3]}, 0U, 0U,
           [=](void* s, auto& b) {
             for (int64_t i = 0; i < cnt; ++i) {
               vllm_ascend::launch_hisparse_prefill_union_plan(
                   s, b[0].template data_ptr<int32_t>() + st[i] * TOPK, b[1].data_ptr(),
                   b[5].template data_ptr<int32_t>() + i * MAX_BITMAP_WORDS,
                   b[6].template data_ptr<int32_t>() + i * MAX_SUPER_BLOCKS, b[2].data_ptr(),
                   b[3].template data_ptr<int32_t>() + i * cap, b[4].template data_ptr<int32_t>() + i * 8,
                   static_cast<uint32_t>(h[i]), static_cast<uint32_t>(q[i]), static_cast<uint32_t>(cap));
             }
           });
}

void HisparsePrefillRemapTopk(const at::Tensor& topk, const at::Tensor& reqDesc, const at::Tensor& bitmapBuf,
                              const at::Tensor& superPrefixBuf, const at::Tensor& pageTable,
                              const at::Tensor& mappedOut, int64_t topkRowOffset, int64_t stagingRows) {
  TORCH_CHECK(IsExpectedDevice(topk), "Invalid TopK device for HiSparse remap");
  CheckContigTensor(topk, topk, at::kInt, {-1, TOPK}, "Expected contiguous int32 [T, 2048] TopK");
  CheckContigTensor(reqDesc, topk, at::kInt, {-1, 8}, "Expected contiguous int32 [R, 8] req_desc");
  CheckContigTensor(bitmapBuf, topk, at::kInt, {-1, MAX_BITMAP_WORDS}, "Invalid bitmap_buf shape");
  CheckContigTensor(superPrefixBuf, topk, at::kInt, {-1, MAX_SUPER_BLOCKS}, "Invalid super_prefix_buf shape");
  CheckRowContigTable(pageTable, topk, "Invalid staged_page_table");
  CheckContigTensor(mappedOut, topk, at::kInt, {-1, TOPK}, "Expected contiguous int32 [Q, 2048] mapped_out");
  TORCH_CHECK(reqDesc.size(0) > 0 && reqDesc.size(0) <= MAX_REQUESTS && topkRowOffset >= 0 &&
                  topkRowOffset < topk.size(0) && stagingRows > 0 && stagingRows < (1LL << 31),
              "Invalid remap bounds");

  const uint32_t bitmapRows = static_cast<uint32_t>(std::min(bitmapBuf.size(0), superPrefixBuf.size(0)));
  const uint32_t pageTableRows = static_cast<uint32_t>(pageTable.size(0));
  Dispatch("HisparsePrefillRemapTopk", topk, {topk, reqDesc, bitmapBuf, superPrefixBuf, pageTable, mappedOut}, 0U, 0U,
           [=](void* s, auto& b) {
             vllm_ascend::launch_hisparse_prefill_remap_topk(
                 s, b[0].data_ptr(), b[1].data_ptr(), b[2].data_ptr(), b[3].data_ptr(), b[4].data_ptr(),
                 b[5].data_ptr(), static_cast<uint32_t>(b[1].size(0)), static_cast<uint32_t>(topkRowOffset),
                 static_cast<uint32_t>(b[4].stride(0)), static_cast<uint32_t>(stagingRows), bitmapRows, pageTableRows);
           });
}

void HisparsePrefillGatherUnion(const at::Tensor& unionIds, const at::Tensor& unionStatus, const at::Tensor& reqMeta,
                                const at::Tensor& stateIndices, const at::Tensor& sourceBlockTable,
                                const at::Tensor& residentBlockTable, const at::Tensor& residentCache,
                                const at::Tensor& stagingPool, int64_t hostDvaBase, int64_t hostTotalBytes) {
  TORCH_CHECK(IsExpectedDevice(unionIds), "Invalid device for HiSparse prefill gather");
  CheckContigTensor(unionIds, unionIds, at::kInt, {-1, -1}, "Invalid union_ids");
  CheckContigTensor(unionStatus, unionIds, at::kInt, {-1, 8}, "Invalid union_status");
  CheckContigTensor(reqMeta, unionIds, at::kInt, {-1, 4}, "Invalid req_meta");
  CheckContigTensor(stateIndices, unionIds, at::kInt, {-1}, "Invalid request_state_indices");
  CheckRowContigTable(sourceBlockTable, unionIds, "Invalid source_block_table");
  CheckRowContigTable(residentBlockTable, unionIds, "Invalid resident_block_table");
  CheckPackedCache(residentCache, unionIds, false, "Invalid resident_cache");
  CheckPackedCache(stagingPool, unionIds, true, "Invalid staging_pool");
  TORCH_CHECK(reqMeta.size(0) > 0 && reqMeta.size(0) <= MAX_REQUESTS && unionIds.size(1) > 0 &&
                  unionIds.size(1) <= MAX_HISTORY && hostTotalBytes >= 0,
              "Invalid gather geometry");

  const uint64_t uHostDva = static_cast<uint64_t>(hostDvaBase);
  const uint64_t uHostBytes = static_cast<uint64_t>(hostTotalBytes);
  const uint64_t resBlockStrideBytes =
      static_cast<uint64_t>(residentCache.stride(0)) * static_cast<uint64_t>(residentCache.element_size());
  const uint32_t stagingRows = static_cast<uint32_t>(stagingPool.size(0) * PAGE_ROWS);
  const uint32_t unionRows = static_cast<uint32_t>(std::min(unionIds.size(0), unionStatus.size(0)));

  Dispatch("HisparsePrefillGatherUnion", unionIds,
           {unionIds, unionStatus, reqMeta, stateIndices, sourceBlockTable, residentBlockTable, residentCache,
            stagingPool},
           uHostDva, uHostBytes, [=](void* s, auto& b) {
             vllm_ascend::launch_hisparse_prefill_gather_union(
                 s, b[0].data_ptr(), b[1].data_ptr(), b[2].data_ptr(), b[3].data_ptr(), b[4].data_ptr(),
                 b[5].data_ptr(), b[6].data_ptr(), b[7].data_ptr(), uHostDva, uHostBytes, resBlockStrideBytes,
                 static_cast<uint32_t>(b[2].size(0)), static_cast<uint32_t>(b[0].size(1)),
                 static_cast<uint32_t>(b[3].size(0)), static_cast<uint32_t>(b[4].size(0)),
                 static_cast<uint32_t>(b[4].stride(0)), static_cast<uint32_t>(b[5].size(0)),
                 static_cast<uint32_t>(b[5].stride(0)), static_cast<uint32_t>(b[6].size(0)), stagingRows, unionRows);
           });
}

void HisparsePrefillCopySpans(const at::Tensor& spans, int64_t numSpans, const at::Tensor& stateIndices,
                              const at::Tensor& residentBlockTable, const at::Tensor& residentCache,
                              const at::Tensor& stagingPool) {
  TORCH_CHECK(IsExpectedDevice(spans), "Invalid device for HiSparse prefill D2D spans");
  if (numSpans <= 0) {
    return;
  }
  CheckContigTensor(spans, spans, at::kInt, {-1, 8}, "Invalid spans tensor");
  TORCH_CHECK(numSpans <= spans.size(0) && numSpans <= UINT32_MAX, "num_spans exceeds spans tensor");
  CheckContigTensor(stateIndices, spans, at::kInt, {-1}, "Invalid request_state_indices");
  CheckRowContigTable(residentBlockTable, spans, "Invalid resident_block_table");
  CheckPackedCache(residentCache, spans, false, "Invalid resident_cache");
  CheckPackedCache(stagingPool, spans, true, "Invalid staging_pool");

  const uint64_t resBlockStrideBytes =
      static_cast<uint64_t>(residentCache.stride(0)) * static_cast<uint64_t>(residentCache.element_size());
  const uint32_t stagingRows = static_cast<uint32_t>(stagingPool.size(0) * PAGE_ROWS);

  Dispatch("HisparsePrefillCopySpans", spans, {spans, stateIndices, residentBlockTable, residentCache, stagingPool},
           0U, 0U, [=](void* s, auto& b) {
             vllm_ascend::launch_hisparse_prefill_copy_spans(
                 s, b[0].data_ptr(), b[1].data_ptr(), b[2].data_ptr(), b[3].data_ptr(), b[4].data_ptr(),
                 resBlockStrideBytes, static_cast<uint32_t>(numSpans), static_cast<uint32_t>(b[1].size(0)),
                 static_cast<uint32_t>(b[2].size(0)), static_cast<uint32_t>(b[2].stride(0)),
                 static_cast<uint32_t>(b[3].size(0)), stagingRows);
           });
}
bool HisparsePrefillKernelsCompiled() {
#if defined(HISPARSE_PREFILL_HOST_SIM) || defined(VLLM_ENABLE_ATB_AND_DIRECT_KERNELS)
  return true;
#else
  return false;
#endif
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(HISPARSE_PREFILL_NS, m) {
  m.def("hisparse_prefill_kernels_compiled() -> bool", HisparsePrefillKernelsCompiled);
  m.def("hisparse_prefill_union_plan(Tensor topk, Tensor(a!)[] scratch, Tensor(b!)[] outputs, int[] histories, "
        "int[] starts, int[] queries, int capacity) -> ()");
  m.def("hisparse_prefill_remap_topk(Tensor topk, Tensor req_desc, Tensor bitmap_buf, Tensor super_prefix_buf, "
        "Tensor page_table, Tensor(a!) mapped_out, int topk_row_offset, int staging_rows) -> ()");
  m.def("hisparse_prefill_gather_union(Tensor union_ids, Tensor union_status, Tensor req_meta, Tensor state_indices, "
        "Tensor source_block_table, Tensor resident_block_table, Tensor resident_cache, Tensor(a!) staging_pool, "
        "int host_dva_base, int host_total_bytes) -> ()");
  m.def("hisparse_prefill_copy_spans(Tensor spans, int num_spans, Tensor state_indices, Tensor resident_block_table, "
        "Tensor resident_cache, Tensor(a!) staging_pool) -> ()");
}

TORCH_LIBRARY_IMPL(HISPARSE_PREFILL_NS, HISPARSE_PREFILL_BACKEND, m) {
  m.impl("hisparse_prefill_union_plan", HisparsePrefillUnionPlan);
  m.impl("hisparse_prefill_remap_topk", HisparsePrefillRemapTopk);
  m.impl("hisparse_prefill_gather_union", HisparsePrefillGatherUnion);
  m.impl("hisparse_prefill_copy_spans", HisparsePrefillCopySpans);
}

TORCH_LIBRARY_IMPL(HISPARSE_PREFILL_NS, Meta, m) {
  using T = const at::Tensor&;
  m.impl("hisparse_prefill_union_plan",
         +[](T, const TList&, const TList&, const IList&, const IList&, const IList&, int64_t) {});
  m.impl("hisparse_prefill_remap_topk", +[](T, T, T, T, T, T, int64_t, int64_t) {});
  m.impl("hisparse_prefill_gather_union", +[](T, T, T, T, T, T, T, T, int64_t, int64_t) {});
  m.impl("hisparse_prefill_copy_spans", +[](T, int64_t, T, T, T, T) {});
}
