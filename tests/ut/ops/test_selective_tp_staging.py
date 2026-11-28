# SPDX-License-Identifier: Apache-2.0
"""Check the native per-layer publication protocol without an NPU runtime."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_layer_publication_does_not_wait_for_peer_dma(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the native Host helper")
    source = (Path(__file__).resolve().parents[3] / "csrc/torch_binding.cpp").read_text()
    start = source.index("  int64_t tp_begin_stage(")
    end = source.index("  SelectiveGroupLayerTransfer layer_binding(", start)
    program = (
        r"""
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("check"); } while (0)
constexpr int64_t SELECTIVE_TP_ACK_BASE = 28;
struct Probe {
  std::array<int64_t, 64>& state;
  int64_t tp_rank_, tp_world_ = 2;
  int64_t tp_load(int64_t i) { return state.at(i); }
  void tp_store(int64_t i, int64_t v) { state.at(i) = v; }
  void tp_wait(int64_t i, int64_t v, bool = false) { TORCH_CHECK(state.at(i) == v); }
"""
        + source[start:end]
        + r"""
};
int main() {
  std::array<int64_t, 64> state{};
  Probe owner{state, 0}, peer{state, 1};
  state[1] = 1;
  // No rank has acknowledged any H2D, including the first layer's payload.
  for (size_t layer = 0; layer < 4; ++layer) {
    const auto epoch = owner.tp_begin_stage(layer, 1, layer + 1);
    owner.tp_publish_stage(epoch, layer);
  }
  assert(state[SELECTIVE_TP_ACK_BASE] == 0 && state[SELECTIVE_TP_ACK_BASE + 1] == 0);
  for (size_t layer = 0; layer < 4; ++layer)
    assert(peer.tp_begin_stage(layer, 1, layer + 1) == int64_t(layer + 1));
  bool failed = false;
  try { peer.tp_begin_stage(0, 2, 1); } catch (const std::runtime_error&) { failed = true; }
  assert(failed);
  // Replay retirement protects reuse across steps; each layer gets a new epoch.
  state[1] = 2;
  for (size_t layer = 0; layer < 4; ++layer) {
    const auto epoch = owner.tp_begin_stage(layer, 2, 5 + layer);
    owner.tp_publish_stage(epoch, layer);
    assert(peer.tp_begin_stage(layer, 2, 5 + layer) == int64_t(5 + layer));
  }
}
"""
    )
    path = tmp_path / "tp_stage.cpp"
    path.write_text(program)
    binary = tmp_path / "tp_stage"
    subprocess.run([compiler, "-std=c++17", "-O2", str(path), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=10)
