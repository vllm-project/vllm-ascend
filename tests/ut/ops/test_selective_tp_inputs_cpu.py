# SPDX-License-Identifier: Apache-2.0
"""Exercise the real TP input checker without per-token Tensor pointer calls."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_tp_input_checks_keep_semantics_without_per_token_tensor_access(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the Host checker")
    source = (Path(__file__).resolve().parents[3] / "csrc/torch_binding.cpp").read_text()
    methods = source[source.index("  int32_t tp_selected_page(") : source.index("  void tp_publish_plans(")]
    program = (
        r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>
static size_t pointer_calls = 0;
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("check"); } while (0)
namespace at {
struct Tensor {
  void* ptr = nullptr;
  int64_t elements = 0;
  size_t item_size = 1;
  template<class T> T* data_ptr() const { ++pointer_calls; return static_cast<T*>(ptr); }
  void* data_ptr() const { ++pointer_calls; return ptr; }
  int64_t numel() const { return elements; }
};
}
size_t selective_tensor_bytes(const at::Tensor& tensor) { return tensor.elements * tensor.item_size; }
constexpr int64_t SELECTIVE_TP_ACK_BASE = 28;
constexpr int64_t SELECTIVE_TP_COMMIT_EPOCH = 27;
struct Shared {
  std::vector<int64_t> owner = std::vector<int64_t>(1 << 19);
  std::array<int64_t, 128> control{};
  std::array<at::Tensor, 8> tensors{};
  at::Tensor k, v;
  size_t offset = 0;
  at::Tensor allocate(int64_t n, size_t bytes) {
    auto* base = reinterpret_cast<uint8_t*>(owner.data());
    at::Tensor result{base + offset, n, bytes};
    offset += n * bytes;
    assert(offset <= owner.size() * 8);
    return result;
  }
  Shared() {
    tensors[0] = allocate(128, 8);
    tensors[1] = allocate(16 * 6400, 4);
    tensors[2] = allocate(16 * 12, 8);
    tensors[3] = allocate(16 * 2048, 4);
    tensors[4] = allocate(3 * 16 * 2048, 4);
    tensors[5] = {owner.data(), static_cast<int64_t>(owner.size()), 8};
    tensors[7] = allocate(4, 8);
    k = allocate(16 * 2048 * 16, 1);
    v = allocate(16 * 2048 * 8, 1);
  }
};
struct State {
  Shared& shared;
  int64_t rows_ = 16, width_ = 2048, copy_width_ = 2048, page_columns_ = 20, tp_capacity_rows_ = 16;
  int64_t tp_world_ = 4, tp_rank_;
  size_t layer_count_ = 3;
  std::vector<at::Tensor> b_ = std::vector<at::Tensor>(30);
  std::array<at::Tensor, 8> tp_;
  at::Tensor normalized_topk_;
  std::array<int64_t, 16 * 12> meta{};
  std::array<int32_t, 16 * 2048> topk{};
  std::array<int64_t, 4> ids{2, 3, 4, 5};
  std::array<std::array<int32_t, 16 * 20>, 3> pages{};
  struct Layer { at::Tensor pages; };
  Layer layer_binding(size_t layer) { return {{pages[layer].data(), 16 * 20, 4}}; }
  int64_t tp_load(int64_t index) { return shared.control[index]; }
  void tp_store(int64_t index, int64_t value) { shared.control[index] = value; }
  void tp_wait(int64_t index, int64_t value) { TORCH_CHECK(tp_load(index) == value); }
  State(Shared& owner, int64_t rank) : shared(owner), tp_rank_(rank), tp_(owner.tensors) {
    b_[5] = {meta.data(), 16 * 12, 8};
    b_[10] = owner.k; b_[11] = owner.v;
    b_[20] = {nullptr, 16 * 161, 4};
    tp_[6] = {ids.data(), 4, 8};
    normalized_topk_ = {topk.data(), 16 * 2048, 4};
    for (int row = 0; row < 16; ++row) {
      meta[row * 12] = row < 14;
      meta[row * 12 + 4] = 8192;
      meta[row * 12 + 7] = row < 14 ? 128 : 0;
      for (int i = 0; i < 2048; ++i) topk[row * 2048 + i] = i;
      for (int layer = 0; layer < 3; ++layer)
        for (int i = 0; i < 20; ++i) pages[layer][row * 20 + i] = 11 + i + row * 20;
    }
    topk[3] = -1; topk[4] = 8192; topk[5] = 5000;
    topk[2048] = topk[2049];
  }
"""
        + methods
        + r"""
};
template<class Function> void rejected(Function fn) {
  bool failed = false;
  try { fn(); } catch (const std::runtime_error&) { failed = true; }
  assert(failed);
}
int main() {
  Shared shared;
  State owner(shared, 0), peer(shared, 1);
  pointer_calls = 0; owner.tp_check_or_publish_inputs(1);
  const auto owner_calls = pointer_calls;
  shared.control[1] = 1;
  pointer_calls = 0; peer.tp_check_or_publish_inputs(1);
  const auto peer_calls = pointer_calls;
  const auto* snapshot = shared.tensors[4].data_ptr<int32_t>();
  assert(snapshot[0] == 11 && snapshot[3] == -1 && snapshot[4] == -1 && snapshot[5] == -1);
  for (int layer = 0; layer < 3; ++layer)
    for (int row = 14; row < 16; ++row)
      for (int i = 0; i < 2048; ++i) assert(snapshot[(layer * 16 + row) * 2048 + i] == -1);
  peer.topk[0]++; rejected([&] { peer.tp_check_or_publish_inputs(1); }); peer.topk[0]--;
  peer.meta[4]++; rejected([&] { peer.tp_check_or_publish_inputs(1); }); peer.meta[4]--;
  peer.ids[0]++; rejected([&] { peer.tp_check_or_publish_inputs(1); }); peer.ids[0]--;
  peer.pages[1][0]++; rejected([&] { peer.tp_check_or_publish_inputs(1); }); peer.pages[1][0]--;
  peer.pages[1][19]++; peer.tp_check_or_publish_inputs(1); peer.pages[1][19]--;
  peer.b_[10].elements--; rejected([&] { peer.tp_check_or_publish_inputs(1); });
  std::cout << "OWNER_POINTER_CALLS " << owner_calls << " PEER_POINTER_CALLS " << peer_calls << std::endl;
  assert(owner_calls < 1000 && peer_calls < 1000);
  std::cout << "TP_INPUT_CHECKS_PASS" << std::endl;
}
"""
    )
    cpp = tmp_path / "inputs.cpp"
    cpp.write_text(program)
    binary = tmp_path / "inputs"
    subprocess.run([compiler, "-std=c++17", "-O2", str(cpp), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
