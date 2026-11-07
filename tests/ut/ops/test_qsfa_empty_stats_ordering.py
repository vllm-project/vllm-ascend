# SPDX-License-Identifier: Apache-2.0
"""Regression for asynchronous stack-UB reuse in legacy empty-row InitOutput."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HEADER = ROOT / (
    "csrc/attention/kv_quant_sparse_flash_attention_vllm/op_kernel/kv_quant_sparse_flash_attention_vllm_kernel_mla.h"
)
PREFIX = r"""
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
using T=float; using OUT_T=float;
enum class QSFA_LAYOUT {TND,BSND};
constexpr int PIPE_ALL=0;
struct GM {std::vector<float>* data;uint64_t offset=0;
    GM operator[](uint64_t x) {return {data,offset+x};}};
struct Write {GM gm;uint32_t size;};
std::vector<Write> pending;float sharedUbValue=0;
template<int> void PipeBarrier() {
    for(auto w:pending)std::fill_n(w.gm.data->begin()+w.gm.offset,w.size,sharedUbValue);
    pending.clear();
}
namespace matmul {
template<class V> void InitOutput(GM gm,uint32_t size,V value) {
    sharedUbValue=value;pending.push_back({gm,size});
}}
struct {uint64_t emptySparseLength=0,actS1Size=1;} tempLoopInfo;
struct {QSFA_LAYOUT outputLayout=QSFA_LAYOUT::TND;uint32_t gSize=128,
    batchSize=2,qSeqSize=2;bool returnSoftmaxLse=true;} constInfo;
struct {uint32_t GetValue(uint32_t i) {return i==0?1:3;}} actualSeqLengthsQGm;
uint32_t kvHeadNum=1,headDim=512;
std::vector<float> out(4*128*512),maxima(4*128),sums(4*128);
GM attentionOutGm{&out},softmaxMaxGm{&maxima},softmaxSumGm{&sums};
void production(uint32_t bIdx,uint32_t s1Idx,uint32_t n2Idx)
"""
SUFFIX = r"""
int main() {
 uint32_t cases=0;
 for(auto layout:{QSFA_LAYOUT::TND,QSFA_LAYOUT::BSND})
 for(uint64_t length:{0ULL,1ULL,128ULL,513ULL,2048ULL})
 for(uint32_t batch:{0U,1U}) {
  constInfo.outputLayout=layout;tempLoopInfo.emptySparseLength=length;
  std::fill(out.begin(),out.end(),13);std::fill(maxima.begin(),maxima.end(),13);
  std::fill(sums.begin(),sums.end(),13);pending.clear();
  production(batch,0,0);PipeBarrier<PIPE_ALL>();
  uint32_t row=layout==QSFA_LAYOUT::TND?batch:batch*2;
  for(uint32_t i=0;i<128*512;++i)if(out[row*128*512+i]!=0) {
   std::cerr<<"output reused stats UB before DMA completion";return 1;
  }
  for(uint32_t i=0;i<128;++i)if(sums[row*128+i]!=float(length) ||
      maxima[row*128+i]!=(length==0?0:-2e38f)) {
   std::cerr<<"raw MAX/SUM reused stack UB before DMA completion";return 2;
  }
  ++cases;
 }
 std::cout<<cases;
}
"""


class EmptyStatsOrderingTest(unittest.TestCase):
    def test_init_output_drains_before_stack_ub_reuse(self):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            self.skipTest("C++17 host compiler unavailable")
        source = HEADER.read_text(encoding="utf-8")
        start = source.index("{", source.index("::InitAllZeroOutput("))
        depth, end = 1, start + 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        with tempfile.TemporaryDirectory(prefix="qsfa-stats-") as directory:
            cpp, executable = Path(directory) / "test.cpp", Path(directory) / "test"
            cpp.write_text(PREFIX + source[start:end] + SUFFIX, encoding="utf-8")
            build = subprocess.run(
                [compiler, "-std=c++17", "-O2", str(cpp), "-o", str(executable)], capture_output=True, text=True
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(int(run.stdout), 20)


if __name__ == "__main__":
    unittest.main()
