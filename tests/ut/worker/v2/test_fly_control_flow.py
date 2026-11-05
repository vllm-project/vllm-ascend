# SPDX-License-Identifier: Apache-2.0
"""Execute the actual kernel control flow with scalar memory on CPU.

This checks window rollback without Triton or torch. NPU tests separately
exercise compilation, reductions, entropy and resampling.
"""

import ast
import inspect
import math
import random
import unittest
from pathlib import Path
from types import SimpleNamespace


class Scalar(int):
    def to(self, dtype):
        return self


class Floating(float):
    def to(self, dtype):
        return self


class Vector(list):
    def to(self, dtype):
        return self

    def __lt__(self, other):
        return Vector(x < other for x in self)


class Pointer:
    def __init__(self, values, offset=0):
        self.values, self.offset = values, offset

    def __add__(self, offset):
        offset = Vector(self.offset + x for x in offset) if isinstance(offset, Vector) else self.offset + offset
        return Pointer(self.values, offset)


def load(ptr, mask=True, other=0):
    if isinstance(ptr.offset, Vector):
        return Vector(ptr.values[i] if valid else other for i, valid in zip(ptr.offset, mask))
    value = ptr.values[ptr.offset] if mask else other
    return Scalar(value) if isinstance(value, int) else Floating(value)


def store(ptr, value):
    ptr.values[ptr.offset] = value


def kernel():
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/v2/spec_decode/rejection_sampler_utils.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_probabilistic_rejection_kernel")
    node.decorator_list = []
    tl = SimpleNamespace(
        constexpr=int,
        int64=int,
        float32=float,
        program_id=lambda axis: 0,
        full=lambda shape, value, dtype: Scalar(value),
        arange=lambda start, end: Vector(range(start, end)),
        load=load,
        store=store,
        argmax=lambda values, axis: Scalar(values.index(max(values))),
    )
    scope = {"tl": tl}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[node.name]


class FLyControlFlowTest(unittest.TestCase):
    def test_random_window_success_and_rollback_keep_original_normalizers(self):
        for has_draft in (False, True):
            for native, expected in [
                ([False, True, True, True], 4),
                ([True, False, True, False], 1),
                ([False, True, True, False], 3),
            ]:
                with self.subTest(has_draft=has_draft, native=native):
                    fn = kernel()
                    tl = fn.__globals__["tl"]
                    tl.int32 = int
                    tl.log, tl.exp = math.log, math.exp
                    tl.minimum, tl.maximum = min, max
                    tl.randint = lambda seed, pos: pos
                    tl.rand = lambda seed, offsets: Vector([0.5])
                    tl.max = lambda values, axis: max(values)
                    fn.__globals__["_TL_RAND_MIN"] = 1e-10
                    fn.__globals__["_compute_global_lse"] = lambda maxima, ms, sums, ss, row, *unused: (
                        maxima.values[row * ms] + math.log(sums.values[row * ss])
                    )
                    temperature = 0.7
                    probabilities = [0.9 if accepted else 0.1 for accepted in native] + [0.5]
                    target = [
                        10 + i + math.log(p)
                        for i, prob in enumerate(probabilities)
                        for p in ((1 - prob) / 2, prob, (1 - prob) / 2)
                    ]
                    draft = [20 + i + math.log(p) for i in range(5) for p in (0.25, 0.5, 0.25)]

                    def stats(values):
                        rows = [values[i : i + 3] for i in range(0, len(values), 3)]
                        maxima = [max(row) for row in rows]
                        sums = [sum(math.exp(x - maximum) for x in row) for row, maximum in zip(rows, maxima)]
                        return Pointer(maxima), Pointer(sums)

                    target_max, target_sum = stats(target)
                    draft_max, draft_sum = stats(draft)
                    args = {name: Pointer([0] * 15) for name in inspect.signature(fn).parameters}
                    for name in args:
                        if "stride" in name:
                            args[name] = 1
                    args.update(
                        sampled_stride=5,
                        target_logits_stride=3,
                        draft_logits_stride_0=15,
                        draft_logits_stride_1=3,
                        vocab_num_blocks=1,
                        vocab_size=3,
                        fly_entropy_threshold=0.0,
                        FLY_WINDOW_SIZE=2,
                        PADDED_VOCAB_NUM_BLOCKS=1,
                        HAS_DRAFT_LOGITS=has_draft,
                        SYNTHETIC_MODE=False,
                        USE_BLOCK_VERIFICATION=False,
                        USE_ENTROPY_VERIFY=False,
                        POSTERIOR_THRESHOLD=1.0,
                        POSTERIOR_ALPHA=0.0,
                        cu_num_logits_ptr=Pointer([0, 5]),
                        temp_ptr=Pointer([temperature]),
                        pos_ptr=Pointer(list(range(5))),
                        draft_sampled_ptr=Pointer([0, 1, 1, 1, 1]),
                        target_logits_ptr=Pointer(target),
                        draft_logits_ptr=Pointer([x * temperature for x in draft]),
                        target_local_max_ptr=target_max,
                        target_local_sumexp_ptr=target_sum,
                        draft_local_max_ptr=draft_max,
                        draft_local_sumexp_ptr=draft_sum,
                        fly_entropy_ptr=Pointer([1.0] * 5),
                    )
                    fn(**args)
                    self.assertEqual(args["rejected_steps_ptr"].values[0], expected)
                    if expected < 4:
                        self.assertAlmostEqual(args["target_rejected_logsumexp_ptr"].values[0], 10 + expected)
                        self.assertAlmostEqual(
                            args["draft_rejected_logsumexp_ptr"].values[0], 20 + expected if has_draft else 0.0
                        )

    def run_case(self, target, window=2, entropy=1, draft=None):
        fn = kernel()
        steps = len(target)
        args = {name: Pointer([0] * (steps * 3)) for name in inspect.signature(fn).parameters}
        for name in args:
            if "stride" in name:
                args[name] = 1
        args.update(
            sampled_stride=steps,
            target_logits_stride=3,
            vocab_num_blocks=1,
            vocab_size=3,
            fly_entropy_threshold=0.3,
            FLY_WINDOW_SIZE=window,
            PADDED_VOCAB_NUM_BLOCKS=1,
            HAS_DRAFT_LOGITS=False,
            SYNTHETIC_MODE=False,
            USE_BLOCK_VERIFICATION=False,
            USE_ENTROPY_VERIFY=False,
            POSTERIOR_THRESHOLD=1.0,
            POSTERIOR_ALPHA=0.0,
            cu_num_logits_ptr=Pointer([0, steps]),
            target_logits_ptr=Pointer([2.0 if col == token else 0.0 for token in target for col in range(3)]),
            target_local_max_ptr=Pointer([2.0] * steps),
            target_local_argmax_ptr=Pointer(target),
            fly_entropy_ptr=Pointer([entropy] * steps),
            draft_sampled_ptr=Pointer(draft or [0] + [1] * (steps - 1)),
        )
        fn(**args)
        return args["rejected_steps_ptr"].values[0], args["sampled_ptr"].values

    def test_isolated_rejection_is_committed_after_full_window(self):
        count, output = self.run_case([0, 1, 1, 2])
        self.assertEqual(count, 3)
        self.assertEqual(output[:3], [1, 1, 1])

    def test_second_rejection_rolls_back_to_original_argmax(self):
        count, output = self.run_case([0, 0, 1, 2])
        self.assertEqual(count, 0)
        self.assertEqual(output[0], 0)

    def test_short_tail_low_entropy_and_disabled_fly(self):
        for target, window, entropy, expected in [
            ([1, 0, 1, 2], 2, 1, 1),
            ([0, 1, 1, 2], 2, 0, 0),
            ([0, 1, 1, 2], 0, 1, 0),
        ]:
            count, output = self.run_case(target, window, entropy)
            self.assertEqual(count, expected)
            self.assertEqual(output[expected], target[expected])

    def test_masked_or_invalid_draft_never_defers(self):
        count, output = self.run_case([0, 1, 1, 2], draft=[0, -1, 1, 1])
        self.assertEqual(count, 0)
        self.assertEqual(output[0], 0)

    def test_random_windows_match_independent_acceptance_specification(self):
        rng = random.Random(17)
        for _ in range(250):
            target = [rng.randrange(3) for _ in range(rng.randrange(4, 10))]
            window = rng.randrange(1, 4)
            native = [token == 1 for token in target[:-1]]
            decisions = [
                accepted or (i + window < len(native) and all(native[i + 1 : i + window + 1]))
                for i, accepted in enumerate(native)
            ]
            expected = next((i for i, accepted in enumerate(decisions) if not accepted), len(native))
            count, output = self.run_case(target, window)
            self.assertEqual(count, expected)
            self.assertEqual(output[:count], [1] * count)
            if count < len(native):
                self.assertEqual(output[count], target[count])


if __name__ == "__main__":
    unittest.main()
