# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual peer plan reader and bound its steady-state allocations."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_peer_plan_reads_published_views_without_topk_hash_allocations(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the native Host helper")
    source = (Path(__file__).resolve().parents[3] / "csrc/torch_binding.cpp").read_text()
    start = source.index("  std::vector<at::Tensor> tp_read_plan(")
    end = source.index("  int64_t tp_begin_stage(", start)
    program = (
        r"""
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <vector>
static bool counting=false;
static size_t allocations=0;
void* operator new(size_t n) {
  if (counting) ++allocations;
  if (auto p=std::malloc(n ? n : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("check"); } while (0)
namespace at {
struct Tensor {
  std::shared_ptr<std::vector<uint64_t>> storage;
  int64_t elements=0, item=4, offset=0, width=0;
  Tensor(int64_t n=0,int64_t bytes=4)
    :storage(std::make_shared<std::vector<uint64_t>>((n*bytes+7)/8)),elements(n),item(bytes) {}
  int64_t numel() const {return elements;}
  bool defined() const {return elements != 0;}
  template<class T> T* data_ptr() const {
    return reinterpret_cast<T*>(reinterpret_cast<char*>(storage->data())+offset*item);
  }
  Tensor view(std::initializer_list<int64_t> shape) const {auto t=*this;t.width=*(shape.begin()+1);return t;}
  Tensor operator[](int64_t row) const {auto t=*this;t.offset+=row*width;t.elements=width;return t;}
  Tensor narrow(int64_t dim,int64_t begin,int64_t count) const {
    assert(dim==0 && begin>=0 && count>=0 && begin+count<=elements);
    auto t=*this;t.offset+=begin;t.elements=count;return t;
  }
};
}
size_t selective_tensor_bytes(const at::Tensor& t) {return t.numel()*t.item;}
struct Layer {at::Tensor stage_host_k,stage_host_v,hot_k,hot_v;};
struct Reader {
  int64_t rows_=2,width_=2048,copy_width_=2048,capacity_=20480,tp_capacity_rows_=2;
  size_t layer_count_=2;
  std::vector<at::Tensor> b_=std::vector<at::Tensor>(30),tp_=std::vector<at::Tensor>(8);
  at::Tensor normalized_topk_{rows_*width_}, statistics_{0};
  Layer layer;
  Layer layer_binding(size_t) {return layer;}
  Reader() {
    const int64_t q=capacity_/128+1,stride=3*width_+q+1;
    b_[20]=at::Tensor(rows_*q);b_[5]=at::Tensor(rows_*12,8);tp_[1]=at::Tensor(rows_*stride);
    layer.stage_host_k=at::Tensor(rows_*width_,2);layer.stage_host_v=at::Tensor(rows_*width_,2);
    layer.hot_k=at::Tensor(rows_*(capacity_+128),2);layer.hot_v=at::Tensor(rows_*(capacity_+128),2);
    auto* v=tp_[1].data_ptr<int32_t>();
    auto* meta=b_[5].data_ptr<int64_t>();auto* top=normalized_topk_.data_ptr<int32_t>();
    meta[4]=102400;meta[7]=128;
    for(int64_t i=0;i<width_;++i) {v[i]=i;top[i]=i;v[stride+i]=-1;}
    v[width_+q]=102;
    for(int i=0;i<102;++i) {v[width_+q+1+i]=i;v[2*width_+q+1+i]=i;}
  }
"""
        + source[start:end]
        + r"""
};
int main() {
  Reader r;
  const int q=r.capacity_/128+1;
  auto* v=r.tp_[1].data_ptr<int32_t>();
  allocations=0;counting=true;
  auto result=r.tp_read_plan(0,false);
  counting=false;
  std::cout<<"peer allocations="<<allocations<<std::endl;
  assert(allocations<=4); // Views plus one return vector, independent of TopK.
  assert(result.size()==4 && result[0].numel()==2048 && result[1].numel()==102);
  assert(result[0].data_ptr<int32_t>()==v && result[2].data_ptr<int32_t>()[101]==101);
  auto rejected=[&](int32_t& value,int32_t invalid) {
    const auto saved=value;value=invalid;bool failed=false;
    try {r.tp_read_plan(0,false);} catch(const std::runtime_error&) {failed=true;}
    value=saved;assert(failed);
  };
  rejected(v[0],20480+128);rejected(v[0],-1);
  rejected(v[2048+q],-1);rejected(v[2048+q],2049);
  rejected(v[2048+q+1],102400);rejected(v[2*2048+q+1],20480);
  rejected(v[2048],322);
  auto* top=r.normalized_topk_.data_ptr<int32_t>();top[0]=-1;
  rejected(v[0],0);top[0]=0;
  v[2048+q]=0;top[1]=top[0];v[1]=v[0]; // Producer-approved duplicate TopK maps to one slot.
  assert(r.tp_read_plan(0,false)[1].numel()==0);
  assert(r.tp_read_plan(1,true).empty());
  std::cout<<"PEER_PLAN_BOUNDS_PASS"<<std::endl;
}
"""
    )
    path = tmp_path / "peer_plan.cpp"
    path.write_text(program)
    binary = tmp_path / "peer_plan"
    subprocess.run([compiler, "-std=c++17", "-O2", str(path), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=10)
