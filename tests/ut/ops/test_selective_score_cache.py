# SPDX-License-Identifier: Apache-2.0
"""Exercise score eviction and pinning against the actual native cache state."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_score_eviction_keeps_endpoints_current_topk_and_latest_score(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required")
    source = (Path(__file__).resolve().parents[3] / "csrc/torch_binding.cpp").read_text(encoding="utf-8")
    start = source.index("struct SelectivePersistentLRU {")
    end = source.index("template <typename Function>", start)
    code = (
        r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("check"); } while(0)
"""
        + source[start:end]
        + r"""
int main() {
  constexpr int cap=256, seq=1000;
  SelectivePersistentLRU pool;
  std::vector<int32_t> resident(cap,-1), lru(cap), cur(cap), mt(cap), ms(cap);
  std::vector<float> scores(2000,10.f);
  for(int i=0;i<32;++i) scores[i]=-1000;
  for(int i=872;i<1001;++i) scores[i]=-2000;
  scores[40]=100; scores[41]=1; scores[42]=2; scores[43]=3;
  std::vector<int32_t> tokens;
  for(int i=0;i<32;++i) tokens.push_back(i);
  for(int i=872;i<1000;++i) tokens.push_back(i);
  for(int i=40;i<136;++i) tokens.push_back(i);
  auto run=[&](int visible=1000) {
    return pool.plan(tokens.data(), tokens.size(), cap, resident.data(), lru.data(),
      visible,cur.data(),mt.data(),ms.data(),scores.data(),visible);
  };
  assert(run()==cap && pool.free_slots.empty());
  tokens={500,501,502,503}; assert(run()==4);
  assert(pool.slot_of_token.count(40));
  for(int i : {41,42,43}) assert(!pool.slot_of_token.count(i));
  for(int i=0;i<32;++i) assert(pool.slot_of_token.count(i));
  for(int i=872;i<1000;++i) assert(pool.slot_of_token.count(i));
  scores[40]=0.25f; tokens={40}; assert(run()==0);
  tokens={600,601,602,603}; assert(run()==4);
  assert(!pool.slot_of_token.count(40));
  scores[90]=-5000; tokens={90,700,701,702,703}; assert(run()==4);
  assert(pool.slot_of_token.count(90));
  tokens={1000,801}; assert(run(1001)==2);
  assert(!pool.slot_of_token.count(872));
  assert(pool.slot_of_token.count(1000));
  pool.reset(cap,resident.data());
  assert(pool.slot_of_token.empty() && pool.free_slots.size()==cap);
  tokens={55}; assert(run()==1 && pool.slot_of_token.count(55));
}
"""
    )
    cpp = tmp_path / "score.cpp"
    exe = tmp_path / "score"
    cpp.write_text(code, encoding="utf-8")
    subprocess.run([compiler, "-std=c++17", "-O2", str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=30)
