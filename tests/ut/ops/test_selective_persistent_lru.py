# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual native planner state without an NPU runtime."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_persistent_free_slots_and_timestamp_lru(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the native Host helper")
    source = (Path(__file__).resolve().parents[3] / "csrc/torch_binding.cpp").read_text()
    start = source.index("struct SelectivePersistentLRU {")
    end = source.index("std::vector<at::Tensor> selective_group_lru_plan_impl(", start)
    program = (
        r"""
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("check"); } while (0)
namespace c10 { struct InferenceMode { explicit InferenceMode(bool) {} }; }
"""
        + source[start:end]
        + r"""
int main() {
  const int cap=20480, topk=2048;
  SelectivePersistentLRU p;
  std::vector<int32_t> resident(cap,-1),lru(cap),selected(topk),current(topk),mt(topk),ms(topk);
  std::iota(lru.begin(),lru.end(),0); std::iota(selected.begin(),selected.end(),0);
  auto run=[&] { return p.plan(selected.data(),topk,cap,resident.data(),lru.data(),100000,
                              current.data(),mt.data(),ms.data()); };
  assert(run()==topk && p.free_slots.size()==cap-topk);
  const auto* node=&p.slot_of_token.at(0);
  assert(run()==0 && p.free_slots.size()==cap-topk && node==&p.slot_of_token.at(0));
  for(int i=0;i<cap;++i) assert(lru[i]==i); // Native path does not rebuild a permutation.
  selected.back()=topk;
  assert(run()==1 && p.free_slots.size()==cap-topk-1);
  assert(p.slot_of_token.count(topk-1)); // An unselected row stays resident while space remains.
  p.invalidate(0,resident.data());
  assert(p.free_slots.size()==cap-topk && run()==1);
  for(int i=0;i<topk;++i) assert(resident[current[i]]==selected[i]);
  p.reset(cap,resident.data());
  assert(p.slot_of_token.empty() && p.free_slots.size()==cap);
  assert(std::all_of(resident.begin(),resident.end(),[](auto v){return v==-1;}));
  SelectivePersistentLRU q;
  int32_t r[4]={-1,-1,-1,-1},l[4]={0,1,2,3},t[4]={0,1,2,3},c[4],a[4],b[4];
  assert(q.plan(t,4,4,r,l,100,c,a,b)==4);
  t[3]=4;
  assert(q.plan(t,4,4,r,l,100,c,a,b)==1 && a[0]==4);
  assert(!q.slot_of_token.count(3));
  for(int i=0;i<4;++i) assert(r[c[i]]==t[i]);
  assert(q.plan(t,4,4,r,l,2,c,a,b)==4); // Prefix rollback resets the request cache.
  assert(q.plan(t,4,4,r,l,2,c,a,b)==2); // Unstable suffix must reload even without another rollback.
  q.step=INT64_MAX;
  bool threw=false;try {q.plan(t,4,4,r,l,100,c,a,b);}catch(const std::exception&){threw=true;}
  assert(threw);

  SelectivePersistentLRU::PageChecks checks;
  checks.prepare(2,2,4);
  int32_t pages[4]={1,2,3,4}; // Two logical blocks, two layers.
  int32_t tags[8]={1,1,2,0,3,3,4,0};
  checks.validate_block(pages,2,2,0,8,1);
  assert(checks.matches(0,0,tags,4,2,2));
  assert(checks.matches(0,1,tags,4,2,2));
  assert(checks.blocks[0].checked==1 && checks.blocks[1].checked==0);
  checks.validate_block(pages,2,2,1,8,1);
  assert(checks.matches(1,2,tags,4,2,2));
  pages[2]=5; // Remap only layer 1 of logical block 0.
  checks.validate_block(pages,2,2,0,8,2);
  assert(!checks.matches(0,0,tags,4,2,2) && !checks.matches(0,1,tags,4,2,2));
  tags[4]=5; // Reload one row, leaving the other row on its prior mapping.
  assert(checks.matches(0,0,tags,4,2,2) && !checks.matches(0,1,tags,4,2,2));
  pages[2]=3;
  checks.validate_block(pages,2,2,0,8,3);
  assert(!checks.matches(0,0,tags,4,2,2) && checks.matches(0,1,tags,4,2,2));
  pages[2]=-1;
  threw=false;try {checks.validate_block(pages,2,2,0,8,4);}
  catch(const std::exception&){threw=true;} assert(threw);
  std::vector<int> values(20,0);
  selective_parallel_requests(20,[&](int64_t i){values[i]=int(i+1);});
  for(int i=0;i<20;++i) assert(values[i]==i+1);
  threw=false;try {selective_parallel_requests(7,[&](int64_t i){if(i==3)throw std::runtime_error("row");});}
  catch(const std::exception&){threw=true;} assert(threw);
}
"""
    )
    cpp, exe = tmp_path / "planner.cpp", tmp_path / "planner"
    cpp.write_text(program)
    subprocess.run([compiler, "-std=c++17", "-O2", "-fopenmp", str(cpp), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)
