# SPDX-License-Identifier: Apache-2.0
"""CPU model of the three-slot QSFA tail fill and source-level event order.

This models ND/NZ DMA addresses and checks the production source contract. It
cannot execute Ascend instructions or establish device pipeline correctness.
"""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HEADER = (
    ROOT
    / "csrc/attention/kv_quant_sparse_flash_attention_vllm/op_kernel"
    / "kv_quant_sparse_flash_attention_vllm_service_vector_mla.h"
)

HOST_PREFIX = r"""
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
using KV_T = uint8_t;
using K_ROPE_T = uint16_t;
constexpr uint64_t FP32_BLOCK_ELEMENT_NUM = 8;
constexpr uint64_t MERGE_CACHE_GM_BUF_NUM = 4;
constexpr uint32_t MTE2_SLOT_NUM = 3;
namespace AscendC { enum class HardEvent { MTE3_V, MTE3_MTE2, V_MTE3 }; }
bool slotReady[3] = {true, true, true};
bool mte3v = false, vmte3 = false, acquired = false;
int activeSlot = 0;
uint32_t mte2SlotElems_ = 0;
template<AscendC::HardEvent E> void SetFlag(uint64_t id) {
    if constexpr (E == AscendC::HardEvent::MTE3_MTE2) {
        if (id != static_cast<uint64_t>(activeSlot) || slotReady[id] || !acquired)
            throw std::runtime_error("invalid slot release");
        slotReady[id] = true;
    } else if constexpr (E == AscendC::HardEvent::MTE3_V) {
        if (mte3v) throw std::runtime_error("MTE3_V already set");
        mte3v = true;
    } else {
        if (vmte3 || !acquired) throw std::runtime_error("invalid V_MTE3 set");
        vmte3 = true;
    }
}
template<AscendC::HardEvent E> void WaitFlag(uint64_t id) {
    if constexpr (E == AscendC::HardEvent::MTE3_MTE2) {
        if (id != static_cast<uint64_t>(activeSlot) || !slotReady[id])
            throw std::runtime_error("invalid slot acquire");
        slotReady[id] = false;
        acquired = true;
    } else if constexpr (E == AscendC::HardEvent::MTE3_V) {
        if (!mte3v) throw std::runtime_error("missing MTE3_V set");
        mte3v = false;
    } else {
        if (!vmte3) throw std::runtime_error("missing V_MTE3 set");
        vmte3 = false;
    }
}
bool unlikely(bool value) { return value; }
struct DataCopyExtParams { uint32_t blockCount=0, blockLen=0, srcStride=0, dstStride=0, unused=0; };
template<class T> struct LocalTensor {
    std::vector<uint8_t>* data;
    size_t offset = 0;
    LocalTensor operator[](size_t n) const { return {data, offset + n * sizeof(T)}; }
    template<class U> LocalTensor<U> ReinterpretCast() const { return {data, offset}; }
};
template<class T> struct GlobalTensor {
    std::vector<uint8_t>* data;
    size_t offset = 0;
    GlobalTensor operator[](size_t n) const { return {data, offset + n * sizeof(T)}; }
};
template<class T> void Duplicate(LocalTensor<T> out, T value, uint32_t count) {
    if (!acquired || out.offset != activeSlot * mte2SlotElems_ || count * sizeof(T) > mte2SlotElems_)
        throw std::runtime_error("zero fill exceeds acquired slot");
    if (value != 0) throw std::runtime_error("expected zero fill");
    std::fill_n(out.data->begin() + out.offset, count * sizeof(T), 0);
}
uint32_t dmaCalls = 0;
template<class T> void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams p) {
    if (!acquired || vmte3) throw std::runtime_error("DMA before V_MTE3 wait");
    ++dmaCalls;
    for (uint32_t b = 0; b < p.blockCount; ++b) {
        size_t so = src.offset + b * (p.blockLen + p.srcStride * 32ULL);
        size_t d = dst.offset + b * (p.blockLen + p.dstStride);
        if (so < activeSlot * mte2SlotElems_ || so + p.blockLen > (activeSlot + 1) * mte2SlotElems_ ||
            d + p.blockLen > dst.data->size()) throw std::runtime_error("DMA out of bounds");
        std::memcpy(dst.data->data() + d, src.data->data() + so, p.blockLen);
    }
}
struct { uint32_t headDim=512, s2BaseSize=512; } constInfo;
struct { uint64_t loop=0; } kvVecRunInfo;
template<bool N> struct QsfaType { static constexpr bool noRope = N; };
template<bool N> void productionTail(int64_t start, int64_t valid, int64_t limit,
                                     std::vector<uint8_t>& ub, std::vector<uint8_t>& gm) {
    using QSFAT = QsfaType<N>;
    int64_t qsfaS2GmStartOffset = start, qsfaMte2Size = valid, qsfaS2GmLimit = limit;
    int64_t qsfaMergeMte3Idx = activeSlot;
    LocalTensor<KV_T> kvMergUb_{&ub};
    GlobalTensor<K_ROPE_T> kvMergeGm_{&gm};
"""

HOST_SUFFIX = r"""
    if (!slotReady[activeSlot] || mte3v || vmte3) throw std::runtime_error("flag lifecycle incomplete");
}
template<bool N> void runCase(int slot, int start, int limit) {
    activeSlot = slot; mte2SlotElems_ = 32 * (N ? 544 : 672);
    acquired = false; mte3v = false; vmte3 = false; dmaCalls = 0;
    std::fill_n(slotReady, 3, true);
    std::vector<uint8_t> ub(3 * mte2SlotElems_, 0x5A);
    std::vector<uint8_t> gm(512 * 576 * 2, 0x5A), expected = gm;
    productionTail<N>(start, 0, limit, ub, gm);
    for (int row = start; row < limit; ++row) {
        if constexpr (N) {
            std::fill_n(expected.begin() + row * 512 * 2, 512 * 2, 0);
        } else {
            for (int col = 0; col < 36; ++col) {
                size_t offset = (col * 512 * 16 + row * 16) * 2;
                std::fill_n(expected.begin() + offset, 16 * 2, 0);
            }
        }
    }
    if (gm != expected) throw std::runtime_error("tail output mismatch");
    for (int other = 0; other < 3; ++other) {
        if (other == slot) continue;
        if (!std::all_of(ub.begin() + other * mte2SlotElems_, ub.begin() + (other + 1) * mte2SlotElems_,
                         [](uint8_t v) { return v == 0x5A; })) throw std::runtime_error("other slot modified");
    }
    int capacity = mte2SlotElems_ / 1024;
    int batches = (limit - start + capacity - 1) / capacity;
    if (dmaCalls != static_cast<uint32_t>(batches * (N ? 1 : 2)))
        throw std::runtime_error("unexpected DMA count");
}
int main() {
    for (int slot = 0; slot < 3; ++slot) for (int start : {0, 15, 128, 255, 511}) {
        int limit = std::min(512, start + 127);
        runCase<true>(slot, start, limit);
        runCase<false>(slot, start, limit);
    }
}
"""


def production_tail():
    source = HEADER.read_text(encoding="utf-8")
    start = source.index(
        "if (unlikely(qsfaS2GmStartOffset + qsfaMte2Size < qsfaS2GmLimit))", source.index("::MergeKvOptimized(")
    )
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source, source[start:end], end


class SparseTailZeroingTest(unittest.TestCase):
    def test_production_tail_with_host_stub(self):
        compiler = shutil.which("g++") or shutil.which("clang++")
        if compiler is None:
            self.skipTest("C++17 host compiler unavailable")
        _, tail, _ = production_tail()
        with tempfile.TemporaryDirectory(prefix="qsfa-tail-") as directory:
            cpp = Path(directory) / "tail.cpp"
            executable = Path(directory) / "tail.exe"
            cpp.write_text(HOST_PREFIX + tail + HOST_SUFFIX, encoding="utf-8")
            build = subprocess.run(
                [compiler, "-std=c++17", "-O2", str(cpp), "-o", str(executable)], capture_output=True, text=True
            )
            self.assertEqual(build.returncode, 0, build.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_production_slot_bounds_and_event_order(self):
        source, tail, end = production_tail()
        for anchor in (
            "static constexpr uint32_t MTE2_SLOT_NUM = 3;",
            "static constexpr uint32_t combineDimAlign_ = QSFAT::noRope ? 544U : 672U;",
            "static constexpr uint32_t mte2SlotElems_ = 32 * combineDimAlign_;",
        ):
            self.assertIn(anchor, source)
        anchors = (
            "const int64_t slot = qsfaMergeMte3Idx % MTE2_SLOT_NUM;",
            "WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);",
            "kvMergUb_[slot * mte2SlotElems_]",
            "mte2SlotElems_ * sizeof(KV_T) / (constInfo.headDim * sizeof(K_ROPE_T))",
            "Duplicate(mergeUb, static_cast<K_ROPE_T>(0.0), zeroRows * constInfo.headDim);",
            "SetFlag<AscendC::HardEvent::V_MTE3>(0);",
            "WaitFlag<AscendC::HardEvent::V_MTE3>(0);",
            "SetFlag<AscendC::HardEvent::MTE3_MTE2>(slot);",
        )
        positions = [tail.index(anchor) for anchor in anchors]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("if constexpr (QSFAT::noRope)", tail)
        self.assertIn("s2GmOffset * constInfo.headDim", tail)
        self.assertIn("s2GmOffset * blockElementNum", tail)
        self.assertIn("s2GmOffset += zeroRows", tail)
        self.assertIn("rows * constInfo.headDim * sizeof(K_ROPE_T)", tail)
        self.assertIn("dataCopyParams.blockCount = constInfo.headDim / blockElementNum", tail)
        self.assertIn("dataCopyParams.blockCount = 64U / blockElementNum", tail)
        self.assertIn("dataCopyParams.blockLen = rows * blockElementNum * sizeof(K_ROPE_T)", tail)
        self.assertIn("512 * constInfo.headDim + s2GmOffset * blockElementNum", tail)
        self.assertIn("(constInfo.s2BaseSize - rows) * blockElementNum * sizeof(K_ROPE_T)", tail)
        for slot in range(3):
            self.assertIn(f"WaitFlag<AscendC::HardEvent::MTE3_MTE2>({slot});", source[end:])

    def test_single_slot_and_nd_nz_tail_addresses(self):
        # Tiling uses headDim=512, ropeDim=64 or 0, int8 KV and 16-bit output.
        head_dim, rope_dim, block, base = 512, 64, 16, 512
        cases = 0
        for no_rope in (False, True):
            slot_bytes = 32 * (544 if no_rope else 672)
            capacity = slot_bytes // (head_dim * 2)
            self.assertEqual(capacity, 17 if no_rope else 21)
            for slot in range(3):
                for start in (0, 1, 15, 128, 255, 256, 511):
                    for end in sorted({min(base, start + 1), min(base, start + 17), min(base, start + 127), base}):
                        if end <= start:
                            continue
                        batch = min(end - start, capacity)
                        ub = bytearray([0x5A]) * (3 * slot_bytes)
                        ub[slot * slot_bytes : slot * slot_bytes + batch * head_dim * 2] = bytes(batch * head_dim * 2)
                        self.assertEqual(ub[: slot * slot_bytes], bytes([0x5A]) * (slot * slot_bytes))
                        self.assertEqual(ub[(slot + 1) * slot_bytes :], bytes([0x5A]) * ((2 - slot) * slot_bytes))
                        output = bytearray([0x5A]) * (base * (head_dim + rope_dim) * 2)
                        expected = output.copy()
                        for offset in range(start, end, batch):
                            rows = min(end - offset, batch)
                            if no_rope:
                                a, b = offset * head_dim * 2, (offset + rows) * head_dim * 2
                                output[a:b] = bytes(b - a)
                            else:
                                # DataCopyPad blockLen is rows*16 elements and dstStride skips remaining rows.
                                for col in range(head_dim // block):
                                    a = (col * base * block + offset * block) * 2
                                    output[a : a + rows * block * 2] = bytes(rows * block * 2)
                                for col in range(rope_dim // block):
                                    a = (base * head_dim + col * base * block + offset * block) * 2
                                    output[a : a + rows * block * 2] = bytes(rows * block * 2)
                        if no_rope:
                            a, b = start * head_dim * 2, end * head_dim * 2
                            expected[a:b] = bytes(b - a)
                        else:
                            for col in range((head_dim + rope_dim) // block):
                                a = (col * base * block + start * block) * 2
                                expected[a : a + (end - start) * block * 2] = bytes((end - start) * block * 2)
                        self.assertEqual(output, expected)
                        cases += 1
        self.assertGreater(cases, 100)


if __name__ == "__main__":
    unittest.main()
