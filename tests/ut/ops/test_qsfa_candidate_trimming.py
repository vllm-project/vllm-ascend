# SPDX-License-Identifier: Apache-2.0
"""Execute production candidate-length selection with bounded CPU GM reads.

Standalone unittest, independent of torch/NPU imports. This does not model
Ascend event timing; graph/NPU numerical checks are still required.
"""

import json
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
#include <stdexcept>
#include <vector>
enum class QSFA_LAYOUT {TND, BSND, BNSD};
constexpr int V_TEMPLATE=0, TEMPLATE_MODE=0;
constexpr auto LAYOUT_T=QSFA_LAYOUT::TND;
struct GM {
    std::vector<int32_t> data;
    uint64_t reads=0;
    int32_t GetValue(uint64_t i) {
        ++reads;
        if(i>=data.size()) throw std::runtime_error("out-of-bounds GM read");
        return data[i];
    }
} topKGm, actualSeqLengthsQGm;
struct {
    uint64_t sparseBlockCount=2048,sparseBlockSize=1,qSeqSize=1;
    uint32_t sparseMode=3,mBaseSize=128,gSize=128,kvHeadNum=1;
    uint32_t s2Start=0,coreStartKVSplitPos=0;
} constInfo;
struct {
    int32_t nextTokensPerBatch=9215;
    uint64_t curActualSeqLenOri=9216,curActualSeqLen=0,actS1Size=1;
    uint64_t emptySparseLength=0;
    uint32_t gS1Idx=0;
} tempLoopInfo;
uint32_t kvHeadNum=1;
void production(uint32_t bIdx,uint32_t s1Idx,uint32_t n2Idx)
"""

SUFFIX = r"""
uint64_t cases=0;
void check(const std::vector<int32_t>& row,uint64_t want,
           uint32_t batch=0,uint32_t query=0) {
    constInfo.qSeqSize=std::max(1U,batch);
    topKGm.data.assign((batch+query+1)*2048,-1);
    actualSeqLengthsQGm.data={static_cast<int32_t>(batch)};
    std::copy(row.begin(),row.end(),topKGm.data.begin()+(batch+query)*2048);
    topKGm.reads=0;
    production(batch?1:0,query,0);
    if(tempLoopInfo.curActualSeqLen!=want) {
        std::cerr<<"case "<<cases<<" expected "<<want<<" got "
                 <<tempLoopInfo.curActualSeqLen;std::exit(1);
    }
    if(want>0 && tempLoopInfo.emptySparseLength!=0) {
        std::cerr<<"stale empty-row metadata";std::exit(3);
    }
    ++cases;
}
int main() {
    for(int n: {0,1,2,15,16,17,31,32,33,127,128,129,255,256,257,
                511,512,513,1023,1024,1025,2047,2048}) {
        std::vector<int32_t> row(2048,-1);
        for(int i=0;i<n;++i) row[i]=n-i-1; // IDs need not be ascending.
        for(uint32_t b: {0U,1U,3U}) for(uint32_t q: {0U,1U,2U})
            check(row,n>2016 && n<2048?2048:n,b,q);
        for(int hole=0;hole<n-1;++hole) {
            auto broken=row;broken[hole]=-1;check(broken,2048);
        }
    }
    std::vector<int32_t> counterexample(2048,-1);
    for(int i=0;i<128;++i)counterexample[i]=i;
    for(int i=256;i<320;++i)counterexample[i]=i;
    check(counterexample,2048);
    std::vector<int32_t> row(2048,-1);std::fill_n(row.begin(),128,7);
    constInfo.mBaseSize=64;check(row,2048);constInfo.mBaseSize=128;
    constInfo.sparseBlockSize=2;check(row,4096);constInfo.sparseBlockSize=1;
    constInfo.s2Start=1;check(row,2048);constInfo.s2Start=0;
    constInfo.coreStartKVSplitPos=1;check(row,2048);constInfo.coreStartKVSplitPos=0;
    constInfo.kvHeadNum=2;check(row,2048);constInfo.kvHeadNum=1;
    constInfo.sparseMode=0;constInfo.sparseBlockCount=1ULL<<33;
    tempLoopInfo.curActualSeqLenOri=1ULL<<33;check(row,1ULL<<33);
    constInfo.sparseMode=3;constInfo.sparseBlockCount=2048;
    tempLoopInfo.curActualSeqLenOri=9216;
    tempLoopInfo.nextTokensPerBatch=63;
    check(row,64); // Same existing causal bound; never expand it.
    std::vector<int32_t> future(2048,-1);
    for(int i=0;i<32;++i)future[i]=64+i;
    check(future,64); // Nonnegative IDs can still be entirely causally masked.
    tempLoopInfo.nextTokensPerBatch=-2;check(row,0);
    tempLoopInfo.nextTokensPerBatch=9215;
    std::vector<int32_t> full(2048,7);check(full,2048);
    if(topKGm.reads>1) {std::cerr<<"full-valid fast path reads="<<topKGm.reads;return 2;}
    const auto fullValidReads=topKGm.reads;
    full[2047]=-1;check(full,2048);
    if(topKGm.reads>2) {std::cerr<<"no aligned tile savings but scanned row";return 4;}
    std::cout<<"{\"cases\":"<<cases<<",\"full_valid_reads\":"<<fullValidReads<<"}";
}
"""


def production_body():
    source = HEADER.read_text(encoding="utf-8")
    start = source.index("::GetSparseActualSeqLen(")
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[opening:end]


class CandidateTrimmingTest(unittest.TestCase):
    def test_prefix_holes_bounds_and_fallback(self):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            self.skipTest("C++17 host compiler unavailable")
        with tempfile.TemporaryDirectory(prefix="qsfa-candidates-") as directory:
            cpp, executable = Path(directory) / "test.cpp", Path(directory) / "test"
            for layout in ["TND", "BSND"]:
                with self.subTest(layout=layout):
                    prefix = PREFIX.replace("LAYOUT_T=QSFA_LAYOUT::TND", f"LAYOUT_T=QSFA_LAYOUT::{layout}")
                    cpp.write_text(prefix + production_body() + SUFFIX, encoding="utf-8")
                    build = subprocess.run(
                        [compiler, "-std=c++17", "-O2", str(cpp), "-o", str(executable)], capture_output=True, text=True
                    )
                    self.assertEqual(build.returncode, 0, build.stderr)
                    run = subprocess.run([str(executable)], capture_output=True, text=True)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    result = json.loads(run.stdout)
                    self.assertGreater(result["cases"], 5000)
                    self.assertLessEqual(result["full_valid_reads"], 1)


if __name__ == "__main__":
    unittest.main()
