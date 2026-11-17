# SPDX-License-Identifier: Apache-2.0
"""Host input packing preserves device masks without NPU libraries."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_host_input_packing(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the Host input regression")
    source = (Path(__file__).resolve().parents[3] / "csrc/torch_binding.cpp").read_text()
    start = source.index("template <typename Token>\nvoid selective_normalize_topk_cpu(")
    end = source.index("class SelectiveNativeGraph", start)
    program = (
        r"""
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <limits>
#include <stdexcept>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("check"); } while (0)
"""
        + source[start:end]
        + r"""
int main() {
  int64_t raw[] = {-1, 0, 99, 100, INT64_MAX, INT64_MIN, 1LL << 32, 9};
  int32_t output[8];
  selective_normalize_topk_cpu(raw, output, 8, 100);
  const int32_t expected[] = {-1, 0, 99, -1, -1, -1, -1, 9};
  assert(std::equal(output, output+8, expected));
  int32_t small[] = {-7, 0, 1, 100};
  selective_normalize_topk_cpu(small, output, 4, 100);
  assert(output[0] == -1 && output[1] == 0 && output[2] == 1 && output[3] == -1);
  int64_t requests[] = {73, 0}, metadata[24] = {};
  int32_t lengths[] = {129, 0}, tails[] = {5, 6, -1, -1};
  for (int row=0;row<2;++row) {
    metadata[row*12+7] = 128;
    metadata[row*12+8] = 99;
    metadata[row*12+11] = 4096;
  }
  selective_pack_native_metadata_cpu(requests, lengths, tails, 2, metadata);
  assert(metadata[0]==73 && metadata[4]==129 && metadata[5]==5 && metadata[6]==6);
  assert(metadata[9]==128 && metadata[10]==0 && metadata[12]==0 && metadata[21]==0 && metadata[22]==1);
  assert(metadata[7]==128 && metadata[8]==99 && metadata[11]==4096);
  int64_t lengths64[] = {256, 1}, tails64[] = {7, 8, 9, 10};
  requests[0] = 74;
  selective_pack_native_metadata_cpu(requests, lengths64, tails64, 2, metadata);
  assert(metadata[0]==74 && metadata[4]==256 && metadata[5]==7 && metadata[9]==255);
  assert(metadata[16]==1 && metadata[17]==9 && metadata[21]==0);
  assert(metadata[19]==128 && metadata[20]==99 && metadata[23]==4096);
}
"""
    )
    cpp = tmp_path / "test.cpp"
    executable = tmp_path / "test"
    cpp.write_text(program)
    subprocess.run([compiler, "-std=c++17", "-O2", str(cpp), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
