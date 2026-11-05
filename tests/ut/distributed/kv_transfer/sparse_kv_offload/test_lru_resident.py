# SPDX-License-Identifier: Apache-2.0
"""Compile and exercise the production CPU planner without torch, ACL or an NPU.

Standalone: python tests/ut/distributed/kv_transfer/sparse_kv_offload/test_lru_resident.py -v
The scan comparison measures misses, not NPU throughput. The contiguous CPU
planner definitions are compiled verbatim from sparse_kv_offload.cpp; no planner
logic is reimplemented here. Torch/ACL address generation and bindings are
excluded so the regression tests can also run on a CPU-only development host.
"""

import ctypes
import os
import random
import shlex
import shutil
import statistics
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

MAX_AGE = (1 << 31) - 2
MAX_TOKEN = 128


class PlannerState:
    def __init__(
        self, library, capacity, topk, rows=1, threads=1, reuse=True, probation_age=4, max_token=MAX_TOKEN,
    ):
        self.library = library
        self.capacity, self.topk, self.rows = capacity, topk, rows
        self.threads, self.reuse, self.probation_age = threads, reuse, probation_age
        self.req_ids = (ctypes.c_int64 * rows)(*([1] * rows))
        self.last_req_ids = (ctypes.c_int64 * rows)(*([-1] * rows))
        self.selected = (ctypes.c_int32 * (rows * topk))()
        self.stable = (ctypes.c_int32 * rows)(*([max_token] * rows))
        self.tokens = (ctypes.c_int32 * (rows * capacity))(*([-1] * (rows * capacity)))
        self.slots = (ctypes.c_int32 * (rows * capacity))(*(list(range(capacity)) * rows))
        self.current = (ctypes.c_int32 * (rows * topk))()
        self.counts = (ctypes.c_int32 * rows)()
        self.miss_tokens = (ctypes.c_int32 * (rows * topk))()
        self.miss_slots = (ctypes.c_int32 * (rows * topk))()
        self.marks = (ctypes.c_int32 * (threads * max_token))()
        self.positions = (ctypes.c_int32 * (threads * max_token))()
        self.scratch = (ctypes.c_int32 * (threads * capacity * 3))()
        self.miss_positions = (ctypes.c_int32 * (threads * topk))()
        self.epochs = (ctypes.c_int32 * threads)()
        self.ages = (ctypes.c_int32 * (rows * capacity))()
        self.age_scratch = (ctypes.c_int32 * (threads * capacity))()
        buffers = (
            self.req_ids, self.last_req_ids, self.selected, self.stable, self.tokens, self.slots,
            self.current, self.counts, self.miss_tokens, self.miss_slots, self.marks, self.positions,
            self.scratch, self.miss_positions, self.epochs, self.ages, self.age_scratch,
        )
        self.pointers = (ctypes.c_size_t * len(buffers))(*(ctypes.addressof(b) for b in buffers))
        self.dimensions = (ctypes.c_int64 * 6)(rows, topk, capacity, max_token, threads, threads)

    def step(self, selections, req_ids=None, stable=None):
        self.selected[:] = [t for row in selections for t in row]
        if req_ids is not None:
            self.req_ids[:] = req_ids
        if stable is not None:
            self.stable[:] = stable
        result = self.library.run_planner(self.reuse, self.pointers, self.dimensions, self.probation_age)
        if result:
            raise ValueError("planner rejected dimensions or probation_age")
        return [
            (
                list(self.current[r * self.topk : (r + 1) * self.topk]),
                list(self.miss_tokens[r * self.topk : r * self.topk + self.counts[r]]),
            )
            for r in range(self.rows)
        ]

    def snapshot(self):
        return tuple(list(a) for a in (self.tokens, self.slots, self.ages, self.last_req_ids))


def reference_step(tokens, order, ages, selected, reset, stable, probation_age):
    """Independent sorting oracle; production instead performs a linear merge."""
    if reset:
        tokens, order, ages = [-1] * len(tokens), list(range(len(tokens))), [0] * len(tokens)
    tokens = list(tokens)
    wanted = set(t for t in selected if 0 <= t < MAX_TOKEN)
    free, survivors, hits = [], [], []
    lookup = {}
    for slot, age in zip(order, ages):
        token = tokens[slot]
        if not 0 <= token < min(max(stable, 0), MAX_TOKEN):
            tokens[slot] = -1
            free.append(slot)
        elif token in wanted:
            hits.append((slot, 0))
            lookup[token] = slot
        else:
            survivors.append((slot, min(age + 1, MAX_AGE)))
    missing = list(dict.fromkeys(t for t in selected if 0 <= t < MAX_TOKEN and t not in lookup))
    targets = free + [slot for slot, _ in survivors]
    for token, slot in zip(missing, targets):
        tokens[slot] = token
        lookup[token] = slot
    assigned = set(targets[: len(missing)])
    remaining = [pair for pair in survivors if pair[0] not in assigned]
    refills = [(s, probation_age) for s in targets[: len(missing)]]
    ordered = sorted(remaining + refills + hits, key=lambda pair: -pair[1])
    pairs = [(slot, 0) for slot in free if slot not in assigned] + ordered
    current = [lookup[t] if 0 <= t < MAX_TOKEN else -1 for t in selected]
    return tokens, [s for s, _ in pairs], [a for _, a in pairs], current, missing


class TestLRUResident(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not compiler or shutil.which(compiler[0]) is None:
            raise unittest.SkipTest("A C++17 compiler is required to test the CPU planners")
        cls.build_dir = tempfile.TemporaryDirectory(prefix="sfa-lru-test-")
        cls.addClassCleanup(cls.build_dir.cleanup)
        root = Path(__file__).resolve().parents[5]
        include = root / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload"
        # Compile the exact CPU function bodies without extracting a production
        # header or requiring torch/ACL. Markers deliberately fail loudly if the
        # source layout changes; tests must then update the extraction boundary.
        native = (include / "sparse_kv_offload.cpp").read_text()
        begin = native.index("#if defined(__GNUC__)")
        end = native.index("int32_t compute_lru_resident_addrs(")
        cpu_header = Path(cls.build_dir.name) / "lru_resident.h"
        cpu_header.write_text(
            "#pragma once\n#include <algorithm>\n#include <cstdint>\n"
            "#include <limits>\n#include <stdexcept>\n"
            "#ifdef _OPENMP\n#include <omp.h>\n#else\n"
            "inline int omp_get_thread_num() { return 0; }\n#endif\n"
            + native[begin:end]
        )
        source = Path(cls.build_dir.name) / "wrapper.cpp"
        source.write_text('''
#include "lru_resident.h"
extern "C" int run_planner(bool reuse, const uintptr_t* p, const int64_t* d, int32_t age) {
  try {
    if (reuse) {
      reuse_prioritized_lru_resident_compact(
          p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],
          d[0],d[1],d[2],d[3],d[4],d[5],p[15],p[16],age);
    } else {
      lru_resident_compact(
          p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],p[12],p[13],p[14],
          d[0],d[1],d[2],d[3],d[4],d[5]);
    }
    return 0;
  } catch (const std::invalid_argument&) { return -1; }
}
extern "C" bool has_openmp() {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}
''')
        output = Path(cls.build_dir.name) / "planner.so"
        command = compiler + [
            "-std=c++17", "-O2", "-shared", "-fPIC", "-Wno-unknown-pragmas",
            "-I", cls.build_dir.name, str(source), "-o", str(output),
        ]
        # Probe the toolchain separately: an actual planner compilation error
        # must fail, rather than silently disabling the OpenMP test path.
        probe = Path(cls.build_dir.name) / "openmp-probe"
        parallel_build = subprocess.run(
            compiler + ["-fopenmp", "-x", "c++", "-", "-o", str(probe)],
            input="#include <omp.h>\nint main() { return omp_get_max_threads() < 1; }\n",
            capture_output=True,
            text=True,
        )
        flags = ["-fopenmp"] if parallel_build.returncode == 0 else []
        subprocess.run(command + flags, capture_output=True, text=True, check=True)
        cls.library = ctypes.CDLL(str(output))
        cls.library.run_planner.argtypes = [
            ctypes.c_bool, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_int64), ctypes.c_int32,
        ]
        cls.library.run_planner.restype = ctypes.c_int
        cls.library.has_openmp.restype = ctypes.c_bool
        print(f"C++ planner tests: OpenMP={'enabled' if cls.library.has_openmp() else 'unavailable (serial fallback)'}")

    def state(self, capacity=3, topk=1, **kwargs):
        return PlannerState(self.library, capacity, topk, **kwargs)

    def test_scan_resistance_against_existing_lru(self):
        sequence = [10, 11, 12, 12, 13, 14, 15, 12]
        totals = {}
        last_misses = {}
        for reuse in (False, True):
            state = self.state(reuse=reuse)
            total = 0
            for token in sequence:
                _, misses = state.step([[token]])[0]
                total += len(misses)
            totals[reuse] = total
            last_misses[reuse] = misses
        self.assertEqual(totals, {False: 7, True: 6})
        self.assertEqual(last_misses, {False: [12], True: []})
        print(f"Scan comparison ({len(sequence)} accesses): LRU misses=7, reuse-prioritized misses=6")

    def test_scan_comparison_at_4096_capacity_and_2048_topk(self):
        topk = 2048
        hot = list(range(topk))
        sequence = [hot, hot] + [list(range(i * topk, (i + 1) * topk)) for i in (1, 2, 3)] + [hot]
        totals = {}
        for reuse in (False, True):
            state = self.state(capacity=4096, topk=topk, reuse=reuse, max_token=16384)
            totals[reuse] = sum(len(state.step([selected])[0][1]) for selected in sequence)
        self.assertEqual(totals, {False: 10240, True: 8192})
        print("C=4096, K=2048 scan comparison: LRU misses=10240, reuse-prioritized misses=8192")

    def test_cpu_planner_benchmark(self):
        # Time only the planner call (including ctypes overhead), not Python
        # input/output conversion. No timing assertion: CI hosts differ.
        topk = 2048
        hot = list(range(topk))
        trace = [hot, hot] + [list(range(i * topk, (i + 1) * topk)) for i in (1, 2, 3)] + [hot]
        for reuse, expected_misses in ((False, 10240), (True, 8192)):
            state = self.state(capacity=4096, topk=topk, reuse=reuse, max_token=16384)
            durations = []
            misses = 0
            warmup_groups, measured_groups = 2, 16
            for group in range(warmup_groups + measured_groups):
                state.req_ids[0] = group + 1
                for selected in trace:
                    state.selected[:] = selected
                    start = time.perf_counter()
                    result = self.library.run_planner(reuse, state.pointers, state.dimensions, 4)
                    duration = time.perf_counter() - start
                    self.assertEqual(result, 0)
                    if group >= warmup_groups:
                        durations.append(duration)
                        misses += state.counts[0]
            self.assertEqual(misses, measured_groups * expected_misses)
            policy = "reuse_prioritized_lru" if reuse else "lru"
            print(f"CPU benchmark {policy}: median={statistics.median(durations) * 1e6:.2f} us/update")

    def test_original_lru_batch_order_is_unchanged(self):
        state = self.state(capacity=4, topk=2, reuse=False)
        expected = [([2, 5], [2, 3, 0, 1]), ([5, 7], [3, 0, 2, 1]),
                    ([2, 8], [2, 1, 3, 0]), ([9, 10], [3, 0, 2, 1])]
        for selected, slots in expected:
            state.step([selected])
            self.assertEqual(list(state.slots), slots)

    def test_randomized_differential(self):
        rng = random.Random(42)
        for capacity in (1, 3, 8, 16):
            for probation_age in (0, 1, 4, MAX_AGE):
                topk = capacity
                state = self.state(capacity, topk, probation_age=probation_age)
                tokens, order, ages = [-1] * capacity, list(range(capacity)), [0] * capacity
                for turn in range(100):
                    req_id = 1 + turn // 17
                    stable = rng.choice([MAX_TOKEN, 64, 8, 0])
                    selected = rng.sample(range(MAX_TOKEN), topk)
                    selected = [t if rng.random() > 0.2 else -1 for t in selected]
                    reset = state.last_req_ids[0] != req_id
                    tokens, order, ages, current, misses = reference_step(
                        tokens, order, ages, selected, reset, stable, probation_age,
                    )
                    actual_current, actual_misses = state.step([selected], [req_id], [stable])[0]
                    self.assertEqual(state.snapshot(), (tokens, order, ages, [req_id]))
                    self.assertEqual((actual_current, actual_misses), (current, misses))

    def test_free_slots_precede_younger_or_older_valid_entries(self):
        state = self.state(capacity=4, topk=2)
        state.last_req_ids[0] = 1
        state.tokens[:] = [10, 11, -1, -1]
        state.slots[:] = [0, 1, 2, 3]
        state.ages[:] = [8, 1, 0, 0]
        current, misses = state.step([[12, 13]])[0]
        self.assertEqual(current, [2, 3])
        self.assertEqual(misses, [12, 13])
        self.assertEqual(list(state.slots), [0, 2, 3, 1])
        self.assertEqual(list(state.ages), [9, 4, 4, 2])

    def test_hit_is_protected_and_suffix_must_reload(self):
        state = self.state(capacity=3, topk=2)
        state.step([[10, 11]])
        old_slot = list(state.current)[1]
        current, misses = state.step([[12, 11]])[0]
        self.assertEqual(current[1], old_slot)
        self.assertEqual(misses, [12])
        _, misses = state.step([[12, 11]], stable=[12])[0]
        self.assertEqual(misses, [12])
        _, misses = state.step([[12, 11]], req_ids=[2])[0]
        self.assertEqual(misses, [12, 11])

    def test_repeated_positions_share_slot_and_do_not_promote_a_miss(self):
        state = self.state(capacity=3, topk=3)
        current, misses = state.step([[10, 10, -1]])[0]
        self.assertEqual(misses, [10])
        self.assertEqual(current, [0, 0, -1])
        self.assertEqual(list(state.ages), [0, 0, 4])
        current, misses = state.step([[10, 10, MAX_TOKEN]])[0]
        self.assertEqual(current, [0, 0, -1])
        self.assertEqual(misses, [])
        self.assertEqual(list(state.ages), [0, 0, 0])

    def test_saturated_age_and_epoch_reset(self):
        state = self.state(capacity=3, topk=2, probation_age=MAX_AGE)
        state.step([[10, 11]])
        state.epochs[0] = (1 << 30) - 1
        current, misses = state.step([[-1, MAX_TOKEN]])[0]
        self.assertEqual(current, [-1, -1])
        self.assertEqual(misses, [])
        self.assertEqual(max(state.ages), MAX_AGE)
        self.assertEqual(state.epochs[0], 1)
        self.assertEqual(state.step([[10, 11]])[0][1], [])

    def test_row_and_thread_isolation(self):
        if not self.library.has_openmp():
            self.skipTest("OpenMP is unavailable; parallel planner coverage requires OpenMP")
        serial = self.state(capacity=8, topk=4, rows=7)
        parallel = self.state(capacity=8, topk=4, rows=7, threads=3)
        rng = random.Random(19)
        for turn in range(100):
            selected = [rng.sample(range(MAX_TOKEN), 4) for _ in range(7)]
            reqs = [r + 1 + (turn // 13 if r == 2 else 0) for r in range(7)]
            stable = [MAX_TOKEN if r % 2 else 32 for r in range(7)]
            self.assertEqual(serial.step(selected, reqs, stable), parallel.step(selected, reqs, stable))
            self.assertEqual(serial.snapshot(), parallel.snapshot())

    def test_inactive_rows_and_other_layer_state_are_preserved(self):
        state = self.state(capacity=3, topk=1, rows=3)
        other_layer = self.state(capacity=3, topk=1, rows=3)
        state.step([[10], [20], [30]])
        other_before = other_layer.snapshot()
        before = state.snapshot()
        state.dimensions[0] = 1
        state.step([[11], [21], [31]])
        after = state.snapshot()
        for old, new in zip(before[:3], after[:3]):
            self.assertEqual(old[3:], new[3:])
        self.assertEqual(other_layer.snapshot(), other_before)

    def test_age_zero_speculative_suffix_is_reloaded_at_equal_boundary(self):
        # Same request, same physical slot, but the old KV version is invalid.
        # Assert the miss plan, not just the final mapping (which is unchanged).
        for reuse in (False, True):
            with self.subTest(reuse=reuse):
                state = self.state(capacity=1, topk=1, reuse=reuse)
                state.last_req_ids[0] = state.req_ids[0]
                state.tokens[0] = 7
                state.ages[0] = 0
                current, misses = state.step([[7]], stable=[7])[0]
                self.assertEqual(current, [0])
                self.assertEqual(misses, [7])
                self.assertEqual(list(state.counts), [1])
                self.assertEqual(list(state.miss_slots), [0])
                self.assertEqual(list(state.tokens), [7])
                if reuse:
                    self.assertEqual(list(state.ages), [state.probation_age])

    def test_invalidated_slot_behind_valid_non_hit_is_used_first(self):
        # Queue head is a valid non-hit. Only the last slot is invalidated.
        for reuse, expected_slot in ((False, 0), (True, 2)):
            with self.subTest(reuse=reuse):
                state = self.state(capacity=3, topk=1, reuse=reuse)
                state.last_req_ids[0] = state.req_ids[0]
                state.tokens[:] = [2, 4, 9]
                state.slots[:] = [0, 1, 2]
                state.ages[:] = [8, 5, 0]
                current, misses = state.step([[7]], stable=[9])[0]
                self.assertEqual((current, misses), ([expected_slot], [7]))
                self.assertEqual(list(state.counts), [1])
                self.assertEqual(list(state.miss_slots), [expected_slot])
                self.assertEqual(list(state.tokens), [2, 4, 7] if reuse else [7, 4, -1])
                self.assertEqual(sorted(state.slots), [0, 1, 2])

    def test_duplicate_selection_checks_slots_and_single_miss(self):
        state = self.state(capacity=4, topk=3)
        state.last_req_ids[0] = state.req_ids[0]
        state.tokens[:] = [-1, -1, 12, -1]
        state.slots[:] = [0, 1, 3, 2]
        state.ages[:] = [0, 0, 0, 2]
        current, misses = state.step([[7, 7, 12]])[0]
        self.assertEqual(current, [0, 0, 2])
        self.assertEqual(misses, [7])
        self.assertEqual(list(state.counts), [1])
        self.assertEqual(list(state.miss_tokens), [7, -1, -1])
        self.assertEqual(list(state.miss_slots), [0, -1, -1])
        self.assertEqual(list(state.tokens).count(7), 1)
        self.assertEqual(state.ages[list(state.slots).index(current[0])], state.probation_age)
        # Only the next planner update is evidence of resident reuse.
        current_again, misses_again = state.step([[7, 7, 12]])[0]
        self.assertEqual(current_again, current)
        self.assertEqual(misses_again, [])
        self.assertEqual(list(state.counts), [0])
        self.assertEqual(state.ages[list(state.slots).index(current[0])], 0)

    def test_invalid_parameters_rejected_before_mutation(self):
        state = self.state(probation_age=-1)
        before = state.snapshot()
        with self.assertRaises(ValueError):
            state.step([[10]])
        self.assertEqual(state.snapshot(), before)
        state.probation_age = 4
        state.dimensions[2] = 0
        with self.assertRaises(ValueError):
            state.step([[10]])
        self.assertEqual(state.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
