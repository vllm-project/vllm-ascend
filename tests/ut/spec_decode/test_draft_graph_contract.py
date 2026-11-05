# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU structural tests for the DSpark draft-graph capture contract.

The tests execute slices of the real source through AST with numpy-backed
tensor stubs, so they run without vllm_ascend imports or an NPU. They assert
shape/ownership/ordering contracts, not numerical accuracy.
"""

import ast
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_DIR = REPO_ROOT / "vllm_ascend" / "spec_decode"
ATTN_DIR = REPO_ROOT / "vllm_ascend" / "attention"


class Mode(Enum):
    NONE = 0
    FULL = 1


class Tensor:
    def __init__(self, values, dtype=None, device=None):
        self.data = np.asarray(values)
        self.dtype = dtype if dtype is not None else self.data.dtype
        self.device = device

    @property
    def shape(self):
        return self.data.shape

    def dim(self):
        return self.data.ndim

    def __getitem__(self, key):
        return Tensor(self.data[key], self.dtype, self.device)

    def copy_(self, other):
        self.data[...] = other.data if isinstance(other, Tensor) else other
        return self

    def fill_(self, value):
        self.data.fill(value)
        return self

    def zero_(self):
        return self.fill_(0)

    def new_zeros(self, shape):
        return Tensor(np.zeros(shape, dtype=self.data.dtype), self.dtype, self.device)


class Paired:
    def __init__(self, size):
        self.cpu = Tensor(np.zeros(size, dtype=np.int32))
        self.gpu = Tensor(np.zeros(size, dtype=np.int32))
        self.np = self.cpu.data

    def copy_to_gpu(self):
        self.gpu.copy_(self.cpu)


def _torch_ns():
    return NS(
        Tensor=Tensor,
        zeros=lambda n, dtype=None, device=None: Tensor(np.zeros(n, dtype=np.int64), dtype, device),
    )


def load_method(filename, name):
    tree = ast.parse((filename).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {
        "torch": _torch_ns(),
        "logger": NS(info=lambda *args: None, warning=lambda *args: None),
    }
    exec(compile(ast.fix_missing_locations(module), str(filename), "exec"), namespace)
    return namespace[name]


def load_dispatch():
    filename = SPEC_DIR / "llm_base_proposer.py"
    tree = ast.parse(filename.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_propose")
    body = method.body
    start = next(
        i
        for i, n in enumerate(body)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "has_lora" for t in n.targets)
    )
    stop = None
    for i in range(start, len(body)):
        stmt = body[i]
        if isinstance(stmt, ast.If):
            test_src = ast.unparse(stmt.test)
            if "graph_contract" in test_src and "CUDAGraphMode.FULL" in test_src:
                stop = i
                break
    assert stop is not None
    source = "def dispatch(self, num_tokens, common_attn_metadata, target_model_batch_desc, num_prefill_reqs):\n"
    source += "\n".join("    " + line for n in body[start:stop] for line in ast.unparse(n).splitlines())
    source += "\n    return num_input_tokens, aclgraph_runtime_mode, batch_descriptor\n"
    namespace = {"CUDAGraphMode": Mode}
    exec(source, namespace)
    return namespace["dispatch"]


def make_self(q=5, context=192, dp=1):
    def dispatch(num_tokens, uniform_decode, has_lora):
        size = next((b for b in (1, 2, 4, 8, 16, 24, 32) if b * 6 >= num_tokens), 32)
        return Mode.FULL, NS(num_tokens=size * 6, num_reqs=size, uniform=uniform_decode)

    return NS(
        method="dspark",
        use_cuda_graph=True,
        _draft_metadata_current_stream=True,
        runner=NS(
            input_batch=NS(lora_id_to_lora_request={}),
            dp_size=dp,
            dp_rank=0,
            cudagraph_dispatcher=NS(uniform_decode_query_len=6, dispatch=dispatch),
            _sync_metadata_across_dp=lambda n, is_draft_model: (n, None, Mode.NONE),
        ),
        dcp_size=1,
        vllm_config=NS(parallel_config=NS(prefill_context_parallel_size=1)),
        num_query_per_req=q,
        num_speculative_tokens=5,
        max_batch_size=32,
        max_query_tokens=32 * q,
        max_num_tokens=8192,
        _dflash_num_context=context,
    )


class DispatchContractTests(unittest.TestCase):
    def test_dispatch_uses_draft_rows_and_target_key(self):
        dispatch = load_dispatch()
        for q in (5, 6):
            for r in (1, 7, 8, 15, 16, 20, 24, 28, 31, 32):
                with self.subTest(q=q, requests=r):
                    n, mode, key = dispatch(make_self(q, context=r * 6), r * q, NS(num_reqs=r), NS(uniform=True), 0)
                    self.assertEqual(mode, Mode.FULL)
                    self.assertEqual(n, key.num_reqs * q)
                    self.assertEqual(key.num_tokens, key.num_reqs * 6)
                    self.assertGreaterEqual(n, r * q)
                    self.assertLessEqual(n, 32 * q)

    def test_mixed_prefill_falls_back_before_padding(self):
        dispatch = load_dispatch()
        for uniform, prefills in ((False, 0), (True, 1)):
            n, mode, key = dispatch(make_self(), 160, NS(num_reqs=32), NS(uniform=uniform), prefills)
            self.assertEqual((n, mode, key), (160, Mode.NONE, None))

    def test_oversized_context_falls_back(self):
        n, mode, key = load_dispatch()(make_self(context=320), 160, NS(num_reqs=32), NS(uniform=True), 0)
        self.assertEqual((n, mode, key), (160, Mode.NONE, None))


class PaddingTests(unittest.TestCase):
    def test_full_batch_context_tail_is_cleared(self):
        self.check_pad(32, 32, 5, 150)

    def test_query_and_context_padding(self):
        for q in (5, 6):
            for actual, padded in ((1, 2), (7, 8), (15, 16), (20, 32), (31, 32)):
                with self.subTest(actual=actual, padded=padded, q=q):
                    self.check_pad(actual, padded, q, actual * 6)

    def check_pad(self, actual, padded, q, context):
        owner = make_self(q, context=context)
        owner.query_start_loc = Paired(35)
        owner.token_arange_np = np.arange(8193)
        owner._adjust_tensor = lambda tensor, n: Tensor(
            np.pad(
                tensor.data[:n],
                [(0, max(0, n - tensor.shape[0]))] + [(0, 0)] * (tensor.data.ndim - 1),
            )
        )
        owner._per_group_block_table_buffers = {0: Tensor(np.full((34, 10), 99))}
        owner._block_table_buffer_map = lambda: owner._per_group_block_table_buffers
        owner._dflash_hidden_states = Tensor(np.full((8192, 4), 3))
        owner._context_positions_buffer = Tensor(np.full(8192, 77))
        owner._per_group_context_slot_mapping_buffers = {0: Tensor(np.full(8192, 42))}
        cad = NS(
            num_reqs=actual,
            query_start_loc_cpu=Tensor(np.arange(actual + 1) * q),
            seq_lens=Tensor(np.full(actual, 1024)),
            seq_lens_cpu=Tensor(np.full(actual, 1024)),
            _seq_lens_cpu=Tensor(np.full(actual, 1024)),
            num_computed_tokens_cpu=None,
            block_table_tensor=Tensor(np.full((actual, 10), 99)),
        )
        result = load_method(SPEC_DIR / "dspark_proposer.py", "_pad_uniform_draft")(owner, cad, padded)
        self.assertEqual(result, padded)
        np.testing.assert_array_equal(cad.query_start_loc.data, np.arange(padded + 1) * q)
        np.testing.assert_array_equal(cad.seq_lens.data[:actual], 1024)
        np.testing.assert_array_equal(cad.seq_lens.data[actual:], 0)
        slots = owner._per_group_context_slot_mapping_buffers[0].data
        np.testing.assert_array_equal(slots[:context], 42)
        np.testing.assert_array_equal(slots[context:], -1)
        np.testing.assert_array_equal(owner._dflash_hidden_states.data[context : padded * 6], 0)
        np.testing.assert_array_equal(owner._per_group_block_table_buffers[0].data[actual:padded], 0)


class ReplayBufferTests(unittest.TestCase):
    def test_cp_lengths_stay_at_a_fixed_address(self):
        stage = load_method(ATTN_DIR / "dsa_v41.py", "_stage_cp_lengths")
        owner = NS(_max_tokens=20, _cp_lengths_buffer=None, _cp_length_high_water=0)
        captured = stage(owner, Tensor(np.full((20, 1), 133, dtype=np.int32)))
        pointer = captured.data.__array_interface__["data"][0]
        for rows, value in ((20, 37), (17, 93), (0, 0), (1, 5), (20, 133)):
            fresh = Tensor(np.full((rows, 1), value, dtype=np.int32))
            current = stage(owner, fresh)
            expected = np.zeros((20, 1), dtype=np.int32)
            expected[:rows] = value
            np.testing.assert_array_equal(captured.data, expected)
            self.assertTrue(np.shares_memory(current.data, captured.data) or rows == 0)
            self.assertEqual(captured.data.__array_interface__["data"][0], pointer)
        with self.assertRaises(AssertionError):
            stage(owner, Tensor(np.zeros((21, 1), dtype=np.int32)))

    def test_target_positions_stay_at_a_fixed_address(self):
        stage = load_method(SPEC_DIR / "llm_base_proposer.py", "_stage_target_positions")
        owner = NS(
            _draft_target_positions=None,
            max_num_tokens=8,
            logger=NS(info=lambda *args: None),
        )
        first = stage(owner, Tensor([1, 2, 3], dtype="int64", device="npu"))
        pointer = first.data.__array_interface__["data"][0]
        second = stage(owner, Tensor([4, 5], dtype="int64", device="npu"))
        self.assertTrue(np.shares_memory(first.data, second.data))
        self.assertEqual(second.data.__array_interface__["data"][0], pointer)
        np.testing.assert_array_equal(second.data, [4, 5])
        with self.assertRaises(ValueError):
            stage(owner, Tensor(np.zeros(9, dtype=np.int64), dtype="int64", device="npu"))


class MetadataOrderingTests(unittest.TestCase):
    def test_current_stream_does_not_inherit_an_executor(self):
        tree = ast.parse((SPEC_DIR / "llm_base_proposer.py").read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_propose")
        node = next(
            n
            for n in method.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "active_device_metadata_executor" for t in n.targets)
        )
        code = compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            "executor",
            "exec",
        )
        executor = object()
        namespace = {
            "self": NS(
                method="dspark",
                use_cuda_graph=True,
                _draft_metadata_current_stream=True,
                runner=NS(device_metadata_executor=executor),
            )
        }
        exec(code, namespace)
        self.assertIsNone(namespace["active_device_metadata_executor"])

    def test_metadata_tasks_run_on_the_consumer_stream(self):
        tree = ast.parse((SPEC_DIR / "llm_base_proposer.py").read_text())
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "build_draft_attn_metadata")
        node = next(
            n
            for n in fn.body
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "device_metadata_executor is not None and device_metadata_tasks"
        )
        code = compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            "tasks",
            "exec",
        )
        events: list[object] = []
        tasks = [
            NS(stage=2, run=lambda: events.append(2)),
            NS(stage=1, run=lambda: events.append(1)),
        ]
        executor = NS(submit=lambda tasks: events.append("submit"))
        exec(
            code,
            {
                "self": NS(_draft_metadata_current_stream=True, use_cuda_graph=True),
                "device_metadata_executor": executor,
                "device_metadata_tasks": tasks,
            },
        )
        self.assertEqual(events, [1, 2])

    def test_capture_metadata_failure_is_fatal(self):
        tree = ast.parse((SPEC_DIR / "dspark_proposer.py").read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "dummy_run")
        handler = next(n for n in ast.walk(method) if isinstance(n, ast.ExceptHandler) and n.name == "_e")
        code = compile(
            ast.fix_missing_locations(ast.Module(body=handler.body, type_ignores=[])),
            "capture",
            "exec",
        )
        namespace = {
            "self": NS(_draft_graph_contract=True),
            "_e": ValueError("broken metadata"),
            "logger": NS(warning=lambda *args: None),
        }
        with self.assertRaisesRegex(RuntimeError, "Draft capture metadata contract failed"):
            exec(code, namespace)


class SourceHygieneTests(unittest.TestCase):
    def test_no_internal_experiment_switches(self):
        for path in (
            SPEC_DIR / "dspark_proposer.py",
            SPEC_DIR / "llm_base_proposer.py",
            ATTN_DIR / "dsa_v41.py",
        ):
            text = path.read_text()
            self.assertNotIn("RC21_", text, f"{path} still reads an internal switch")
            self.assertNotIn("_rc21_", text, f"{path} still uses an internal prefix")


if __name__ == "__main__":
    unittest.main(verbosity=2)
