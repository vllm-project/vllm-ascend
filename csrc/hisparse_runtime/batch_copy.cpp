// SPDX-License-Identifier: Apache-2.0
// Installed PTA2.10.post7 has no exported batch-copy wrapper. Keep only this
// CANN submit boundary inside PTA's guard/stream/task queue. Allocation and
// single-run copies use the public tensor API in acl_stage.py.
#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <acl/acl.h>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {
struct CopyState {
  int64_t token, owner, device, caller_tid;
  std::atomic<int64_t> submit_tid{0}, result{-1}, batch_used{0}, finished{0};
  std::vector<void*> sources, destinations;
  std::vector<size_t> lengths;
  std::vector<at::Tensor> owners;
};
std::atomic<int64_t> next_token{1};
std::mutex states_mutex;
std::unordered_map<int64_t, std::shared_ptr<CopyState>> states;

bool covers(const std::vector<at::Tensor>& owners, int64_t address, int64_t length,
            bool npu, int64_t device) {
  for (const auto& tensor : owners) {
    if (!tensor.is_contiguous() || tensor.numel() == 0) continue;
    if (npu ? (tensor.device().type() != c10::DeviceType::PrivateUse1 ||
               tensor.device().index() != device) : !tensor.device().is_cpu()) continue;
    const auto start = reinterpret_cast<uintptr_t>(tensor.const_data_ptr());
    const auto size = tensor.nbytes();
    const auto ptr = static_cast<uintptr_t>(address);
    if (ptr >= start && static_cast<uint64_t>(length) <= size &&
        ptr - start <= size - static_cast<uint64_t>(length)) return true;
  }
  return false;
}

int64_t batch_copy(const at::Tensor& source, const at::Tensor& destination,
                   const at::Tensor& sizes, int64_t direction,
                   const std::vector<at::Tensor>& payload_owners,
                   int64_t device, int64_t owner) {
  for (const auto& tensor : {source, destination, sizes})
    TORCH_CHECK(tensor.device().is_cpu() && tensor.scalar_type() == at::kLong &&
                tensor.dim() == 1 && tensor.is_contiguous(),
                "Batch descriptors require contiguous CPU int64 vectors");
  const auto count = source.numel();
  TORCH_CHECK(count > 0 && count <= 256 && destination.numel() == count && sizes.numel() == count,
              "Batch requires1..256 equally sized descriptor vectors");
  TORCH_CHECK(direction >= 0 && direction <= 2 && device >= 0 &&
              device <= std::numeric_limits<c10::DeviceIndex>::max() && owner > 0,
              "Invalid direction, device or owner");
  auto state = std::make_shared<CopyState>();
  state->token = next_token.fetch_add(1);
  state->owner = owner;
  state->device = device;
  state->caller_tid = syscall(SYS_gettid);
  for (const auto& tensor : payload_owners) {
    state->owners.push_back(tensor.detach());
    TORCH_CHECK(state->owners.back().unsafeGetTensorImpl()->pyobj_slot()->load_pyobj() == nullptr,
                "PTA queue owners must not carry a Python object");
  }
  const auto* src = source.const_data_ptr<int64_t>();
  const auto* dst = destination.const_data_ptr<int64_t>();
  const auto* len = sizes.const_data_ptr<int64_t>();
  for (int64_t i = 0; i < count; ++i) {
    TORCH_CHECK(src[i] > 0 && dst[i] > 0 && len[i] > 0 &&
                src[i] <= std::numeric_limits<int64_t>::max() - len[i] &&
                dst[i] <= std::numeric_limits<int64_t>::max() - len[i],
                "Invalid batch address or length");
    TORCH_CHECK(covers(state->owners, src[i], len[i], direction != 0, device) &&
                covers(state->owners, dst[i], len[i], direction != 1, device),
                "Batch extent escapes its retained CPU/NPU owner");
    for (int64_t j = 0; j < i; ++j)
      TORCH_CHECK(dst[i] >= dst[j] + len[j] || dst[j] >= dst[i] + len[i],
                  "Overlapping batch destinations need the serial tensor path");
    state->sources.push_back(reinterpret_cast<void*>(src[i]));
    state->destinations.push_back(reinterpret_cast<void*>(dst[i]));
    state->lengths.push_back(static_cast<size_t>(len[i]));
  }
  const c10_npu::NPUGuard guard(c10::Device(c10::DeviceType::PrivateUse1,
                                         static_cast<c10::DeviceIndex>(device)));
  // Do not empty the task queue to obtain the current stream handle.
  const auto stream = c10_npu::getCurrentNPUStream(device).stream(false);
  {
    std::lock_guard<std::mutex> lock(states_mutex);
    states.emplace(state->token, state);
  }
  at_npu::native::OpCommand command;
  command.Name("HiSparsePTASegmentedCopy2830");
  command.SetCustomHandler([state, stream, direction, count]() -> int {
    state->submit_tid = syscall(SYS_gettid);
    const c10_npu::NPUGuard worker_guard(c10::Device(c10::DeviceType::PrivateUse1,
                                  static_cast<c10::DeviceIndex>(state->device)));
    aclError status = ACL_SUCCESS;
    // BatchAsync has stalled inside the driver in full C32 Decode. Keep the
    // same PTA task, retained descriptors/payloads and existing stream, but
    // submit each validated extent through the regular asynchronous copy API.
    const auto kind = direction == 0 ? ACL_MEMCPY_HOST_TO_DEVICE :
                      direction == 1 ? ACL_MEMCPY_DEVICE_TO_HOST : ACL_MEMCPY_DEVICE_TO_DEVICE;
    for (int64_t i = 0; i < count && status == ACL_SUCCESS; ++i)
      status = aclrtMemcpyAsync(state->destinations[i], state->lengths[i],
                               state->sources[i], state->lengths[i], kind, stream);
    state->result = status;
    state->finished = 1;
    TORCH_CHECK(status == ACL_SUCCESS, "PTA queued DMA failed: ", status);
    return 0;
  });
  // A throwing submission leaves the state discoverable by this stage's
  // owner cookie. Only a subsequent proven device drain may release it.
  command.Run();
  return state->token;
}

std::vector<int64_t> copy_status(int64_t token) {
  std::lock_guard<std::mutex> lock(states_mutex);
  auto it = states.find(token);
  TORCH_CHECK(it != states.end(), "Unknown copy ticket");
  auto& s = *it->second;
  return {s.token, s.device, s.caller_tid, s.submit_tid.load(), s.result.load(),
          s.batch_used.load(), s.finished.load(), static_cast<int64_t>(s.lengths.size())};
}
std::vector<int64_t> pending_copies(int64_t owner) {
  std::lock_guard<std::mutex> lock(states_mutex);
  std::vector<int64_t> result;
  for (const auto& pair : states) if (pair.second->owner == owner) result.push_back(pair.first);
  return result;
}
void release_copy(int64_t token, bool dma_drained, bool global_device_drained) {
  TORCH_CHECK(dma_drained, "Retain copy payloads and descriptors until DMA completion");
  std::lock_guard<std::mutex> lock(states_mutex);
  auto it = states.find(token);
  TORCH_CHECK(it != states.end(), "Unknown copy ticket");
  TORCH_CHECK(it->second->finished || global_device_drained,
              "A possibly queued submission requires a successful global drain");
  states.erase(it);
}
bool has_native_batch() { return false; }
}  // namespace

TORCH_LIBRARY_FRAGMENT(hisparse_pta_host2511, m) {
  m.def("batch_copy(Tensor source, Tensor destination, Tensor sizes, int direction, Tensor[] owners, int device, int owner) -> int");
  m.def("copy_status(int ticket) -> int[]", &copy_status);
  m.def("pending_copies(int owner) -> int[]", &pending_copies);
  m.def("release_copy(int ticket, bool dma_drained, bool global_device_drained) -> ()", &release_copy);
  m.def("has_native_batch() -> bool", &has_native_batch);
}
TORCH_LIBRARY_IMPL(hisparse_pta_host2511, CPU, m) { m.impl("batch_copy", &batch_copy); }
// The payload owner list contains NPU tensors. Without this registration PTA
// invokes its CPU fallback and may copy owners: it is not an async-DMA proof.
TORCH_LIBRARY_IMPL(hisparse_pta_host2511, PrivateUse1, m) { m.impl("batch_copy", &batch_copy); }
