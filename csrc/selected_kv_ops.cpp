// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <dlfcn.h>
#include <torch/library.h>
#ifdef SELECTED_KV_HOST_SIM
  #include "kernel_operator.h"
  #define SELECTED_KV_BACKEND CPU
  #define SELECTED_KV_NS _C_ascend_host_sim
#else
  #include <acl/acl.h>
  #include <torch_npu/csrc/core/npu/NPUGuard.h>
  #include <torch_npu/csrc/core/npu/NPUStream.h>
  #include <torch_npu/csrc/framework/OpCommand.h>
  #define SELECTED_KV_BACKEND PrivateUse1
  #define SELECTED_KV_NS _C_ascend
#endif
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vllm_ascend {
#if defined(SELECTED_KV_HOST_SIM) || defined(VLLM_ENABLE_ATB_AND_DIRECT_KERNELS)
void launch_selected_kv_union_plan(void*, void*, void*, void*, void*, void*, void*, uint32_t, uint32_t, uint32_t);
void launch_selected_kv_inplace_copy(void*, void*, void*, void*, void*, void*, void*, uint64_t, uint32_t, uint32_t,
                                     uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
void launch_selected_kv_li_gather(void*, void*, void*, void*, uint32_t, uint64_t, uint64_t);
#else
template <typename... A>
inline void launch_selected_kv_union_plan(A...) {}
template <typename... A>
inline void launch_selected_kv_inplace_copy(A...) {}
template <typename... A>
inline void launch_selected_kv_li_gather(A...) {}
#endif
}  // namespace vllm_ascend

namespace {
using TList = c10::List<at::Tensor>;
using IList = c10::List<int64_t>;
using SList = c10::List<std::string>;
using HybmMapperFn = int32_t (*)(uint64_t, uint32_t, uint64_t*);

struct BoundHybm {
  std::mutex mu;
  void* handle = nullptr;
  HybmMapperFn mapper = nullptr;
};

BoundHybm& HybmState() {
  static BoundHybm state;
  return state;
}

constexpr uint32_t HYBM_DEVICE_MEMORY = 1U;
constexpr uint64_t HUGEPAGE_PROBE_STEP = 2ULL << 20;
constexpr int64_t MAX_HISTORY = 1048576, MAX_TOKENS = 8192, MAX_REQUESTS = 256, TOPK = 2048;
constexpr int64_t ROW_BYTES = 656, PAGE_ROWS = 128, LI_DATA_BYTES = PAGE_ROWS * 128, LI_SCALE_BYTES = 256;

inline bool IsExpectedDevice(const at::Tensor& t) {
#ifdef SELECTED_KV_HOST_SIM
  return t.device().is_cpu();
#else
  return t.device().is_privateuseone();
#endif
}

inline void CheckTensor(const at::Tensor& t, const at::Tensor& ref, at::ScalarType dt,
                        std::initializer_list<int64_t> shape, const char* msg) {
  TORCH_CHECK(t.device() == ref.device() && t.is_contiguous() && t.scalar_type() == dt &&
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

std::vector<at::Tensor> CheckIntBuffers(const TList& ts, size_t cnt, const at::Tensor& ref) {
  TORCH_CHECK(ts.size() == cnt, "Wrong selected union buffer count");
  auto bufs = ts.vec();
  for (const auto& t : bufs) {
    CheckTensor(t, ref, at::kInt, {}, "Union buffers must be contiguous int32 on one device");
  }
  return bufs;
}

template <typename Fn>
void Dispatch([[maybe_unused]] const char* name, [[maybe_unused]] const at::Tensor& ref,
              [[maybe_unused]] std::vector<at::Tensor> keep, [[maybe_unused]] Fn launch) {
#ifdef SELECTED_KV_HOST_SIM
  AscendC::ScopedCallRegions guard;
  for (const auto& t : keep) {
    const size_t el = static_cast<size_t>(t.element_size());
    const size_t sp = t.numel() == 0 ? 0 : static_cast<size_t>((t.size(0) - 1) * t.stride(0) + t.size(-1)) * el;
    guard.Add(t.data_ptr(), std::max(t.nbytes(), sp));
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
  TORCH_CHECK(false, "Selected KV kernels are not compiled for this target");
#endif
}

void SelectedKvUnionPlan(const at::Tensor& topk, const TList& scratch, const TList& outputs, const IList& histories,
                         const IList& starts, const IList& queries, int64_t cap) {
  TORCH_CHECK(IsExpectedDevice(topk), "Invalid selected TopK");
  CheckTensor(topk, topk, at::kInt, {-1, 1, TOPK}, "Invalid selected TopK");
  TORCH_CHECK(topk.size(0) > 0 && topk.size(0) <= MAX_TOKENS, "Expected original [T,1,2048] TopK");
  const auto h = histories.vec(), st = starts.vec(), q = queries.vec();
  const auto cnt = static_cast<int64_t>(h.size());
  TORCH_CHECK(cnt > 0 && cnt <= MAX_REQUESTS && st.size() == h.size() && q.size() == h.size() && cap > 0 &&
                  cap <= MAX_HISTORY && cap % PAGE_ROWS == 0,
              "Invalid union geometry");
  for (int64_t i = 0, prev = 0; i < cnt; prev = st[i] + q[i], ++i) {
    TORCH_CHECK(h[i] >= 0 && h[i] <= MAX_HISTORY && q[i] > 0 && q[i] <= topk.size(0) && st[i] >= prev &&
                    st[i] <= topk.size(0) - q[i],
                "Invalid selected H/Q span");
  }
  const auto a = CheckIntBuffers(scratch, 3, topk), o = CheckIntBuffers(outputs, 2, topk);
  const int64_t needs[] = {MAX_HISTORY, MAX_HISTORY / 32, 64 * 8};
  for (size_t i = 0; i < 3; ++i) {
    TORCH_CHECK(a[i].numel() >= needs[i], "Selected union scratch too small");
  }
  const char* outErr = "Invalid union output shape";
  CheckTensor(o[0], topk, at::kInt, {-1, cap}, outErr);
  CheckTensor(o[1], topk, at::kInt, {-1, 8}, outErr);
  TORCH_CHECK(o[0].size(0) >= cnt && o[1].size(0) >= cnt, outErr);
  Dispatch("SelectedKvUnionPlan", topk, {topk, a[0], a[1], a[2], o[0], o[1]}, [=](void* s, auto& b) {
    for (int64_t i = 0; i < cnt; ++i) {
      vllm_ascend::launch_selected_kv_union_plan(
          s, b[0].template data_ptr<int32_t>() + st[i] * TOPK, b[1].data_ptr(), b[2].data_ptr(), b[3].data_ptr(),
          b[4].template data_ptr<int32_t>() + i * cap, b[5].template data_ptr<int32_t>() + i * 8,
          static_cast<uint32_t>(h[i]), static_cast<uint32_t>(q[i]), static_cast<uint32_t>(cap));
    }
  });
}

void SelectedKvInplaceCopy(const at::Tensor& ids, const at::Tensor& status, const at::Tensor& bases,
                           const at::Tensor& meta, const at::Tensor& tbl, const at::Tensor& cache, int64_t mOff) {
  TORCH_CHECK(IsExpectedDevice(ids) && mOff >= 0, "Selected copy requires a valid device and offset");
  CheckTensor(ids, ids, at::kInt, {-1, -1}, "Invalid selected IDs shape");
  const auto reqs = ids.size(0), cap = ids.size(1);
  TORCH_CHECK(reqs > 0 && reqs <= MAX_REQUESTS && cap > 0 && cap <= MAX_HISTORY && cap % PAGE_ROWS == 0,
              "Invalid selected IDs shape");
  const char* metaErr = "Invalid selected count or request metadata shape";
  const char* dstErr = "Selected destination must be native packed C8 pages";
  CheckTensor(status, ids, at::kInt, {reqs, 8}, metaErr);
  CheckTensor(meta, ids, at::kInt, {reqs, 4}, metaErr);
  CheckTensor(cache, ids, at::kChar, {-1, PAGE_ROWS, 1, ROW_BYTES}, dstErr);
  TORCH_CHECK(cache.size(0) > 0 && cache.size(0) <= UINT32_MAX, dstErr);
  CheckTensor(bases, ids, at::kUInt64, {-1}, "Invalid selected source page domain");
  TORCH_CHECK(bases.numel() > 0 && bases.numel() <= cache.size(0), "Invalid selected source page domain");
  TORCH_CHECK(tbl.device() == ids.device() && tbl.scalar_type() == at::kInt && tbl.dim() == 2 && tbl.size(0) > 0 &&
                  tbl.size(0) <= MAX_REQUESTS && tbl.size(1) > 0 && tbl.size(1) <= UINT32_MAX && tbl.stride(1) == 1 &&
                  tbl.stride(0) >= tbl.size(1) && tbl.stride(0) <= UINT32_MAX,
              "Original page table needs unit column stride");
  Dispatch("SelectedKvInplaceCopy", ids, {ids, status, bases, meta, tbl, cache}, [=](void* s, auto& b) {
    vllm_ascend::launch_selected_kv_inplace_copy(
        s, b[0].data_ptr(), b[1].data_ptr(), b[2].data_ptr(), b[3].data_ptr(), b[4].data_ptr(), b[5].data_ptr(),
        static_cast<uint64_t>(mOff), static_cast<uint32_t>(reqs), static_cast<uint32_t>(cap),
        static_cast<uint32_t>(b[2].numel()), static_cast<uint32_t>(b[4].size(0)), static_cast<uint32_t>(b[4].size(1)),
        static_cast<uint32_t>(b[4].stride(0)), static_cast<uint32_t>(b[5].size(0)));
  });
}

void SelectedKvLiGather(const at::Tensor& bases, const at::Tensor& dstK, const at::Tensor& dstS, int64_t dOff,
                        int64_t sOff) {
  TORCH_CHECK(IsExpectedDevice(bases), "Selected LI gather requires matching device tensors");
  const char* bufErr = "Selected LI gather requires contiguous uint64/int8/fp16 buffers";
  CheckTensor(bases, bases, at::kUInt64, {-1}, bufErr);
  CheckTensor(dstK, bases, at::kChar, {}, bufErr);
  CheckTensor(dstS, bases, at::kHalf, {}, bufErr);
  const auto pgs = dstK.dim() >= 1 ? dstK.size(0) : 0;
  TORCH_CHECK(bases.numel() > 0 && bases.numel() <= pgs && dstS.dim() >= 1 && pgs > 0 && pgs <= UINT32_MAX &&
                  dstS.size(0) == pgs && dstK.nbytes() == static_cast<size_t>(pgs * LI_DATA_BYTES) &&
                  dstS.nbytes() == static_cast<size_t>(pgs * LI_SCALE_BYTES),
              "Selected LI gather page geometry mismatch");
  TORCH_CHECK(dOff >= 0 && sOff >= 0 && static_cast<uint64_t>(dOff) <= UINT64_MAX - LI_DATA_BYTES &&
                  static_cast<uint64_t>(sOff) <= UINT64_MAX - LI_SCALE_BYTES,
              "Invalid selected LI component offset");
  Dispatch("SelectedKvLiGather", bases, {bases, dstK, dstS}, [=](void* s, auto& b) {
    vllm_ascend::launch_selected_kv_li_gather(s, b[0].data_ptr(), b[1].data_ptr(), b[2].data_ptr(),
                                              static_cast<uint32_t>(b[0].numel()), static_cast<uint64_t>(dOff),
                                              static_cast<uint64_t>(sOff));
  });
}

void SelectedKvBindHybm(const SList& paths) {
  TORCH_CHECK(!paths.empty(), "Selected KV requires candidate HyBM library paths from the active store");
  void* foundHandle = nullptr;
  HybmMapperFn foundMapper = nullptr;
  for (const std::string& p : paths) {
    if (p.empty()) {
      continue;
    }
    void* h = dlopen(p.c_str(), RTLD_NOW | RTLD_NOLOAD);
    if (h == nullptr) {
      continue;
    }
    if (auto* sym = dlsym(h, "hybm_gva_to_va")) {
      foundHandle = h;
      foundMapper = reinterpret_cast<HybmMapperFn>(sym);
      break;
    }
    dlclose(h);
  }
  TORCH_CHECK(foundMapper != nullptr,
              "Selected KV requires the already initialized native HyBM SDK (libmf_hybm_core.so)");
  void* toClose = nullptr;
  bool conflict = false;
  {
    auto& st = HybmState();
    std::lock_guard<std::mutex> lock(st.mu);
    if (st.mapper != nullptr && st.mapper != foundMapper) {
      toClose = foundHandle;
      conflict = true;
    } else if (st.handle == nullptr) {
      st.handle = foundHandle;
      st.mapper = foundMapper;
    } else {
      toClose = foundHandle;
      st.mapper = foundMapper;
    }
  }
  if (toClose != nullptr) {
    dlclose(toClose);
  }
  TORCH_CHECK(!conflict, "Selected KV HyBM mapper conflicts with an already bound SDK instance");
}

at::Tensor SelectedKvExportBases(const at::Tensor& blobGvas, const at::Tensor& blobSizes, int64_t spanB) {
  const char* vecErr = "Source address buffers must be equal-length contiguous CPU uint64 vectors";
  TORCH_CHECK(blobGvas.device().is_cpu(), vecErr);
  CheckTensor(blobGvas, blobGvas, at::kUInt64, {-1}, vecErr);
  CheckTensor(blobSizes, blobGvas, at::kUInt64, {blobGvas.size(0)}, vecErr);
  TORCH_CHECK(spanB > 0, "span_bytes must be a positive integer");
  auto out = at::empty_like(blobGvas);
  if (blobGvas.size(0) == 0) {
    return out;
  }
  HybmMapperFn mapper = nullptr;
  {
    auto& st = HybmState();
    std::lock_guard<std::mutex> lock(st.mu);
    mapper = st.mapper;
  }
  TORCH_CHECK(mapper != nullptr, "Selected KV source requires an initialized HyBM mapper (hybm_gva_to_va)");
  const auto* gvas = blobGvas.data_ptr<uint64_t>();
  const auto* sizes = blobSizes.data_ptr<uint64_t>();
  auto* dst = out.data_ptr<uint64_t>();
  const uint64_t span = static_cast<uint64_t>(spanB);
  // Deduplicate repeated GVAs within one step's leased call only; never cache mappings across steps.
  std::unordered_map<uint64_t, uint64_t> mapped;
  mapped.reserve(static_cast<size_t>(blobGvas.size(0)));
  for (int64_t i = 0; i < blobGvas.size(0); ++i) {
    const uint64_t gva = gvas[i];
    TORCH_CHECK(span <= sizes[i], "Component at index ", i, " exceeds its leased allocation");
    TORCH_CHECK(gva != 0U && gva <= UINT64_MAX - (span - 1U), "Component at index ", i,
                " has no contiguous device byte span");
    if (const auto it = mapped.find(gva); it != mapped.end()) {
      dst[i] = it->second;
      continue;
    }
    auto mapAt = [&](uint64_t addr) {
      uint64_t va = 0;
      const int32_t rc = mapper(addr, HYBM_DEVICE_MEMORY, &va);
      TORCH_CHECK(rc == 0 && va != 0U, "No live device mapping for object ", i, " in HyBM domain (status ", rc, ")");
      return va;
    };
    const uint64_t va0 = mapAt(gva);
    auto check = [&](uint64_t d, const char* err) {
      const uint64_t va = mapAt(gva + d);
      TORCH_CHECK(va >= va0 && va - va0 == d, "Component at index ", i, err);
    };
    // Verify linear device-VA contiguity across end-of-span and every 2 MiB HyBM slab boundary.
    check(span - 1U, " has no contiguous device byte span");
    for (uint64_t step = HUGEPAGE_PROBE_STEP; step < span; step += HUGEPAGE_PROBE_STEP) {
      check(step, " crosses a non-contiguous HyBM slab boundary");
    }
    mapped.emplace(gva, va0);
    dst[i] = va0;
  }
#ifdef SELECTED_KV_HOST_SIM
  for (const auto& kv : mapped) {
    AscendC::RegisterPersistentRegion(reinterpret_cast<void*>(static_cast<uintptr_t>(kv.second)), span);
  }
#endif
  return out;
}
}  // namespace

#ifdef SELECTED_KV_HOST_SIM
SIM_EXPORT void selected_kv_sim_set_mapper(HybmMapperFn fn) {
  void* toClose = nullptr;
  {
    auto& st = HybmState();
    std::lock_guard<std::mutex> lock(st.mu);
    toClose = st.handle;
    st.handle = nullptr;
    st.mapper = fn;
  }
  if (toClose != nullptr) {
    dlclose(toClose);
  }
}
#endif

TORCH_LIBRARY_FRAGMENT(SELECTED_KV_NS, m) {
  m.def("selected_kv_union_plan(Tensor topk, Tensor(a!)[] scratch, Tensor(b!)[] outputs, int[] histories, int[] starts,"
        " int[] queries, int capacity) -> ()");
  m.def("selected_kv_inplace_copy(Tensor ids, Tensor status, Tensor object_bases, Tensor metadata, Tensor table, "
        "Tensor(a!) cache, int main_offset) -> ()");
  m.def("selected_kv_li_gather(Tensor object_bases, Tensor(a!) target_k, Tensor(b!) target_scale, int data_offset, "
        "int scale_offset) -> ()");
  m.def("selected_kv_bind_hybm(str[] candidate_paths) -> ()");
  m.impl("selected_kv_bind_hybm", c10::DispatchKey::CompositeExplicitAutograd, SelectedKvBindHybm);
  m.def("selected_kv_export_bases(Tensor blob_gvas, Tensor blob_sizes, int span_bytes) -> Tensor");
}
TORCH_LIBRARY_IMPL(SELECTED_KV_NS, SELECTED_KV_BACKEND, m) {
  m.impl("selected_kv_union_plan", SelectedKvUnionPlan);
  m.impl("selected_kv_inplace_copy", SelectedKvInplaceCopy);
  m.impl("selected_kv_li_gather", SelectedKvLiGather);
#ifdef SELECTED_KV_HOST_SIM
  m.impl("selected_kv_export_bases", SelectedKvExportBases);
#endif
}
#ifndef SELECTED_KV_HOST_SIM
TORCH_LIBRARY_IMPL(SELECTED_KV_NS, CPU, m) { m.impl("selected_kv_export_bases", SelectedKvExportBases); }
#endif
TORCH_LIBRARY_IMPL(SELECTED_KV_NS, Meta, m) {
  using T = const at::Tensor&;
  m.impl("selected_kv_union_plan",
         +[](T, const TList&, const TList&, const IList&, const IList&, const IList&, int64_t) {});
  m.impl("selected_kv_inplace_copy", +[](T, T, T, T, T, T, int64_t) {});
  m.impl("selected_kv_li_gather", +[](T, T, T, int64_t, int64_t) {});
  m.impl("selected_kv_export_bases", +[](T g, T, int64_t) { return at::empty_like(g); });
}
