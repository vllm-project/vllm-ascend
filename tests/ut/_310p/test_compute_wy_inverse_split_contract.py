# SPDX-License-Identifier: Apache-2.0
"""CPU-only source-contract checks, not a device synchronization proof."""

import unittest
from pathlib import Path

import regex as re

ROOT = Path(__file__).resolve().parents[3]
KERNEL = ROOT / "csrc/moe/chunk_gated_delta_rule_compute_wy/op_kernel/arch20/compute_wy_kernel.h"


def function_body(source: str, name: str) -> str:
    # Only match declarations, never earlier call sites. The four-argument
    # substitution overload is the last declaration of that name.
    definitions = list(re.finditer(rf"__aicore__\s+inline\s+\w+\s+{re.escape(name)}\(", source))
    start = definitions[-1].start()
    opening = source.index("{", start)
    depth = 1
    for position in range(opening + 1, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : position]
    raise ValueError(f"Unclosed function: {name}")


class TestComputeWyInverseSplitContract(unittest.TestCase):
    def test_dispatch_is_gated_and_128_only(self):
        body = function_body(KERNEL.read_text(encoding="utf-8"), "ProcessOneTaskAttempt")
        self.assertIn(
            "if (allowCompDoubling && useFp32ForwardSubstitution && kHeadDim_ == 128 && vHeadDim_ == 128)",
            body,
        )
        self.assertLess(body.index("NeedsFp32ForwardSubstitution("), body.index("ProcessRefinedTask("))
        self.assertIn("const bool useScalarFallback", body)

    def test_one_inverse_then_two_compensated_applications(self):
        body = function_body(KERNEL.read_text(encoding="utf-8"), "ProcessRefinedTask")
        self.assertEqual(body.count("Fp32ForwardSubstitution(a, cScratch, 64, 64)"), 1)
        self.assertEqual(body.count("IrCompApply(a, rhs)"), 2)
        self.assertNotIn("IrSolve(", body)
        self.assertNotIn("IrResidual(", body)
        self.assertNotIn("irSnapshotGm_", body)
        self.assertLess(body.index("Fp32ForwardSubstitution("), body.index("Adds(a, scratch"))

    def test_finite_range_checks_and_retry(self):
        source = KERNEL.read_text(encoding="utf-8")
        inverse = function_body(source, "ProcessRefinedTask")
        apply = function_body(source, "IrCompApply")
        retry = function_body(source, "ProcessOneTask")
        self.assertIn("if (!(irInverseNorm_ < 60000.0f)) return false;", inverse)
        self.assertIn("if (!(IrMaxAbs(operand) < 60000.0f)) return false;", apply)
        self.assertIn("/*allowCompDoubling=*/false", retry)
        self.assertLess(retry.index("PipeBarrier<PIPE_ALL>()"), retry.index("/*allowCompDoubling=*/false"))

    def test_signed_result_saved_before_destructive_abs(self):
        body = function_body(KERNEL.read_text(encoding="utf-8"), "IrCompApply")
        self.assertLess(body.index("Muls(rhs[column], operand"), body.index("IrMaxAbs(operand)"))
        self.assertIn("microMm_.MmNz2(resultNz, inverseNz, operand, splitHalf, splitScratch)", body)
        self.assertIn("column < 128; column += 64", body)

    def test_w_store_drained_before_v_load(self):
        body = function_body(KERNEL.read_text(encoding="utf-8"), "ProcessRefinedTask")
        store = body.index("StoreBhtdChunk(wKernelGm_")
        drain = body.index("SyncEvent<HardEvent::MTE3_MTE2>", store)
        load = body.index("LoadBthdChunk(vGm_")
        wait = body.index("SyncEvent<HardEvent::MTE2_V>", load)
        self.assertLess(store, drain)
        self.assertLess(drain, load)
        self.assertLess(wait, body.index("Cast(rhs, halfBuf_", load))

    def test_column_updates_keep_pipe_ordering(self):
        body = function_body(KERNEL.read_text(encoding="utf-8"), "Fp32ForwardSubstitution")
        gather = body.index("Gather(")
        broadcast = body.index("Brcb(")
        update = body.index("MulAddDst(")
        self.assertIn("PipeBarrier<PIPE_V>()", body[gather:broadcast])
        self.assertIn("PipeBarrier<PIPE_V>()", body[broadcast:update])
        self.assertIn("PipeBarrier<PIPE_V>()", body[update:])
        self.assertIn("const uint32_t first = col + 1;", body)

    def test_active_helpers_reuse_existing_buffers(self):
        source = KERNEL.read_text(encoding="utf-8")
        for name in ("ProcessRefinedTask", "IrCompApply", "Fp32ForwardSubstitution"):
            with self.subTest(function=name):
                body = function_body(source, name)
                self.assertNotIn("InitBuffer(", body)
                self.assertNotIn("AllocTensor", body)

    def test_frobenius_admission_preserves_gate_and_retry(self):
        source = KERNEL.read_text(encoding="utf-8")
        body = function_body(source, "ProcessOneTaskAttempt")
        self.assertIn("FAST_DOUBLING_FROB_SQUARED_THRESHOLD = 7.502183356513852f", source)
        row_gate = body.index("NeedsFp32ForwardSubstitution(")
        extra_gate = body.index("FrobeniusSquared(")
        inverse = body.index("ProcessRefinedTask(")
        self.assertLess(row_gate, extra_gate)
        self.assertLess(extra_gate, inverse)
        self.assertIn("allowCompDoubling && useFp32ForwardSubstitution &&", body[row_gate:extra_gate])
        self.assertIn("gateMaxRowSum_ >= FP32_FS_ROW_SUM_THRESHOLD", body[row_gate:extra_gate])
        self.assertIn("gateMaxRowSum_ < 1.0e30f", body[row_gate:extra_gate])
        self.assertIn("frobSquared >= 0.0f && frobSquared < FAST_DOUBLING_FROB_SQUARED_THRESHOLD", body)

    def test_frobenius_reuses_dead_scratch_without_matmul(self):
        body = function_body(KERNEL.read_text(encoding="utf-8"), "FrobeniusSquared")
        self.assertIn("qHalfBuf_.Get<half>().ReinterpretCast<float>()", body)
        self.assertIn("Mul(scratch, a, a, ATTEN_ELEMS)", body)
        self.assertIn("ReduceSum(reduceScratch, scratch, work, ATTEN_ELEMS)", body)
        self.assertLess(body.index("Mul("), body.index("PipeBarrier<PIPE_V>()"))
        self.assertLess(body.index("ReduceSum("), body.index("SyncEvent<HardEvent::V_S>"))
        self.assertLess(body.index("SyncEvent<HardEvent::V_S>"), body.index("GetValue(0)"))
        for forbidden in ("InitBuffer(", "AllocTensor", "microMm_", "cubeGemm_", "irSnapshotGm_"):
            self.assertNotIn(forbidden, body)


if __name__ == "__main__":
    unittest.main()
