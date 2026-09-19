# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU instruction-level checks for both DFlash kernel compatibility variants.

The production bodies execute with a small NumPy implementation of their Triton
operations. This checks indexing and masked tails, not Triton compilation or NPU
performance; the companion nightly test runs the real compiled kernel.
"""

import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class _Value(np.ndarray):
    def to(self, dtype):
        return self.astype(dtype)


class _Pointer:
    def __init__(self, array, offset=0):
        self.array = array
        self.offset = offset

    def __add__(self, offset):
        return _Pointer(self.array, self.offset + offset)

    def __sub__(self, offset):
        return self + (-offset)


class _Language:
    int32 = np.int32
    int64 = np.int64
    minimum = staticmethod(np.minimum)
    where = staticmethod(np.where)

    def __init__(self, num_reqs):
        self.num_reqs = num_reqs
        self.program = (0, 0)
        self.store_calls = 0

    def program_id(self, axis):
        return self.program[axis]

    def num_programs(self, axis):
        assert axis == 0
        return self.num_reqs

    @staticmethod
    def arange(start, end):
        return np.arange(start, end, dtype=np.int32).view(_Value)

    @staticmethod
    def _indices(pointer, mask):
        indices, active = np.broadcast_arrays(pointer.offset, mask)
        selected = indices[active.astype(bool)]
        assert np.all(selected >= 0) and np.all(selected < pointer.array.size), "Unmasked out-of-bounds access"
        return indices, active.astype(bool)

    def load(self, pointer, mask=True, other=0):
        indices, active = self._indices(pointer, mask)
        result = np.full(indices.shape, other, dtype=pointer.array.dtype)
        result[active] = pointer.array[indices[active]]
        return result.view(_Value)

    def store(self, pointer, value, mask=True):
        indices, active = self._indices(pointer, mask)
        values = np.broadcast_to(value, indices.shape)
        pointer.array[indices[active]] = values[active]
        self.store_calls += 1


def _load_kernels(language):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/spec_decode/dflash/speculator.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    kernels = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "_prepare_dflash_inputs_kernel_ascend":
            continue
        node.decorator_list = []
        for argument in node.args.args:
            argument.annotation = None
        namespace = {"tl": language}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        kernels.append((namespace[node.name], [argument.arg for argument in node.args.args]))
    assert len(kernels) == 2, "Test both v0.28 and main compatibility variants"
    return kernels


def make_case(lengths, max_num_tokens=4096, max_num_reqs=32, sample_from_anchor=False, position_base=0):
    """Shared deterministic fixture for CPU parity, NPU parity, and benchmarks."""
    num_reqs = len(lengths)
    num_query = 8
    num_steps = num_query if sample_from_anchor else num_query - 1
    assert num_reqs <= max_num_reqs
    assert max(sum(lengths), num_reqs * num_query) <= max_num_tokens
    max_model_len = 122880
    block_size = 128
    stride = max_model_len // block_size
    query_starts = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
    positions = np.concatenate(
        [np.arange(length, dtype=np.int64) + position_base + i * 127 for i, length in enumerate(lengths)]
    )
    arrays = {
        "out_input_ids_ptr": np.full(max_num_tokens, 777, dtype=np.int32),
        "out_query_positions_ptr": np.full(max_num_tokens, 777, dtype=np.int64),
        "out_query_start_loc_ptr": np.full(max_num_reqs + 1, 777, dtype=np.int32),
        "out_seq_lens_ptr": np.full(max_num_reqs, 777, dtype=np.int32),
        "out_query_slot_mapping_ptr": np.full(max_num_tokens, 777, dtype=np.int64),
        "out_context_positions_ptr": np.full(max_num_tokens, 777, dtype=np.int64),
        "out_context_slot_mapping_ptr": np.full(max_num_tokens, 777, dtype=np.int32),
        "out_sample_indices_ptr": np.full(max_num_reqs * num_steps, 777, dtype=np.int32),
        "out_sample_pos_ptr": np.full(max_num_reqs * num_steps, 777, dtype=np.int64),
        "out_sample_idx_mapping_ptr": np.full(max_num_reqs * num_steps, 777, dtype=np.int32),
        "out_temperature_ptr": np.full(max_num_reqs, -99, dtype=np.float32),
        "out_seeds_ptr": np.full(max_num_reqs, 777, dtype=np.int64),
        "target_positions_ptr": positions,
        "target_query_start_loc_ptr": query_starts,
        "idx_mapping_ptr": np.arange(max_num_reqs - 1, max_num_reqs - num_reqs - 1, -1, dtype=np.int32),
        "last_sampled_ptr": np.arange(max_num_reqs, dtype=np.int32) + 100,
        "next_prefill_tokens_ptr": np.arange(max_num_reqs, dtype=np.int32) + 200,
        "num_sampled_ptr": np.arange(num_reqs, dtype=np.int32) % 2,
        "num_rejected_ptr": np.array([min(i % 3, length - 1) for i, length in enumerate(lengths)], dtype=np.int32),
        "temperature_ptr": np.linspace(0, 1, max_num_reqs, dtype=np.float32),
        "seeds_ptr": np.arange(max_num_reqs, dtype=np.int64) + (1 << 40),
        "block_table_ptr": (np.arange(num_reqs * stride, dtype=np.int32) * 13 % 5400 + 1),
    }
    max_tokens_per_req = max(lengths) + num_query
    tile = min(256, 1 << (max_tokens_per_req - 1).bit_length())
    scalars = dict(
        block_table_stride=stride,
        parallel_drafting_token_id=248070,
        block_size=block_size,
        num_query_per_req=num_query,
        num_speculative_steps=num_steps,
        max_num_reqs=max_num_reqs,
        max_num_tokens=max_num_tokens,
        max_model_len=max_model_len,
        cp_rank=0,
        SAMPLE_FROM_ANCHOR=sample_from_anchor,
        PAD_SLOT_ID=-1,
        CP_SIZE=1,
        CP_INTERLEAVE=1,
        BLOCK_SIZE=tile,
    )
    return arrays, scalars, (num_reqs, (max_tokens_per_req + tile - 1) // tile)


def reference_outputs(arrays, scalars, grid):
    """Independent scalar oracle, including untouched storage and graph tails."""
    output = {name: array.copy() for name, array in arrays.items() if name.startswith("out_")}
    num_reqs = grid[0]
    query_count = scalars["num_query_per_req"]
    steps = scalars["num_speculative_steps"]
    block_size = scalars["block_size"]
    stride = scalars["block_table_stride"]
    sample_offset = 0 if scalars["SAMPLE_FROM_ANCHOR"] else 1
    table = arrays["block_table_ptr"].reshape(num_reqs, stride)

    def slot(req, position):
        return int(table[req, min(position // block_size, stride - 1)]) * block_size + position % block_size

    for req in range(num_reqs):
        start, end = arrays["target_query_start_loc_ptr"][req : req + 2]
        state = arrays["idx_mapping_ptr"][req]
        valid_end = end - arrays["num_rejected_ptr"][req]
        last_position = int(arrays["target_positions_ptr"][valid_end - 1])
        for index in range(start, end):
            if index < valid_end:
                position = int(arrays["target_positions_ptr"][index])
                output["out_context_positions_ptr"][index] = position
                output["out_context_slot_mapping_ptr"][index] = slot(req, position)
            else:
                # Rejected-tail entries are masked: position 0, PAD slot.
                output["out_context_positions_ptr"][index] = 0
                output["out_context_slot_mapping_ptr"][index] = scalars["PAD_SLOT_ID"]
        bonus_source = "last_sampled_ptr" if arrays["num_sampled_ptr"][req] > 0 else "next_prefill_tokens_ptr"
        for offset in range(query_count):
            index = req * query_count + offset
            position = last_position + 1 + offset
            output["out_input_ids_ptr"][index] = arrays[bonus_source][state] if offset == 0 else 248070
            output["out_query_positions_ptr"][index] = min(position, scalars["max_model_len"] - 1)
            output["out_query_slot_mapping_ptr"][index] = slot(req, position)
            if offset >= sample_offset:
                sample_index = req * steps + offset - sample_offset
                output["out_sample_indices_ptr"][sample_index] = index
                output["out_sample_pos_ptr"][sample_index] = position + (sample_offset == 0)
                output["out_sample_idx_mapping_ptr"][sample_index] = state
        output["out_query_start_loc_ptr"][req] = req * query_count
        output["out_seq_lens_ptr"][req] = min(last_position + 1 + query_count, scalars["max_model_len"])
        output["out_temperature_ptr"][state] = arrays["temperature_ptr"][state]
        output["out_seeds_ptr"][state] = arrays["seeds_ptr"][state]
    output["out_query_start_loc_ptr"][num_reqs:] = num_reqs * query_count
    output["out_seq_lens_ptr"][num_reqs:] = 0
    output["out_sample_indices_ptr"][num_reqs * steps :] = 0
    output["out_sample_pos_ptr"][num_reqs * steps :] = 0
    output["out_sample_idx_mapping_ptr"][num_reqs * steps :] = -1
    output["out_query_slot_mapping_ptr"][num_reqs * query_count :] = scalars["PAD_SLOT_ID"]
    return output


class TestDFlashPrepareInputs(unittest.TestCase):
    def _check_case(self, **case_kwargs):
        arrays, scalars, grid = make_case(**case_kwargs)
        expected = reference_outputs(arrays, scalars, grid)
        for variant in range(2):
            with self.subTest(variant=variant, **case_kwargs):
                actual = {name: array.copy() for name, array in arrays.items()}
                language = _Language(grid[0])
                kernel, names = _load_kernels(language)[variant]
                arguments = {name: _Pointer(array) for name, array in actual.items()} | scalars
                for req in reversed(range(grid[0])):
                    for block in reversed(range(grid[1])):
                        language.program = (req, block)
                        kernel(**{name: arguments[name] for name in names})
                for name, array in actual.items():
                    np.testing.assert_array_equal(array, expected.get(name, arrays[name]), err_msg=name)
                # Decode with 16K graph slots used to issue >16K scalar stores.
                if case_kwargs.get("lengths") == (8,) and case_kwargs.get("max_num_tokens") == 16384:
                    self.assertLess(language.store_calls, 1200)

    def test_decode_and_graph_padding(self):
        for lengths in ((8,), (8,) * 32):
            for tokens in (4096, 16384):
                self._check_case(lengths=lengths, max_num_tokens=tokens)

    def test_ragged_context_and_partial_tiles(self):
        self._check_case(lengths=(1, 15, 16, 17, 255, 256, 257), max_num_tokens=4099, max_num_reqs=41)

    def test_chunked_prefill_and_anchor_sampling(self):
        for anchor in (False, True):
            self._check_case(lengths=(4095, 9, 7), max_num_tokens=16384, sample_from_anchor=anchor)

    def test_position_and_block_table_clamping(self):
        self._check_case(lengths=(8,), max_num_tokens=8, max_num_reqs=1, position_base=122878)


# --- Metadata reuse tests (merged from the former test_dflash_metadata_reuse.py) ---


def _load_class(relative_path, class_name, namespace):
    source = Path(__file__).resolve().parents[3] / relative_path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[class_name]


class _UpstreamSpeculator:
    def __init__(self, *args):
        self.build_calls = []
        self.fail_build = False
        self.return_no_metadata = False

    def _build_draft_attn_metadata(self, num_reqs, num_reqs_padded, num_tokens_padded, *args, **kwargs):
        if self.fail_build:
            raise RuntimeError("metadata build failed")
        self.build_calls.append((num_reqs, num_reqs_padded, num_tokens_padded, args, kwargs))
        if self.return_no_metadata:
            return None
        self.last_built = {
            "swa": SimpleNamespace(causal=True, sliding_window=2048, block_tables=object()),
            "full": SimpleNamespace(causal=False, sliding_window=None, block_tables=object()),
        }
        for metadata in self.last_built.values():
            metadata.actual_seq_lengths_q = [min(index + 1, num_reqs) * 8 for index in range(num_reqs_padded)]
        return self.last_built

    def propose(self, *args, **kwargs):
        return self.scenario(self)  # type: ignore[attr-defined]


def _make_speculator():
    cls = _load_class(
        "vllm_ascend/worker/v2/spec_decode/dflash/speculator.py",
        "AscendDFlashSpeculator",
        {
            "DFlashSpeculator": _UpstreamSpeculator,
            "build_attn_metadata_wrapper": nullcontext,
            "vllm_version_is": lambda version: False,
        },
    )
    speculator = cls(None, None)
    speculator.input_batch = SimpleNamespace(num_reqs=1, seq_lens_cpu_upper_bound=object())
    speculator.num_query_per_req = 8
    speculator._group_causal = {2: True, 3: False}
    return speculator


def _propose(speculator, **kwargs):
    return speculator.propose(
        speculator.input_batch,
        {},
        {},
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        **kwargs,
    )


def _build(speculator, num_reqs=1, num_reqs_padded=2, num_tokens_padded=16, **kwargs):
    return speculator._build_draft_attn_metadata(
        num_reqs,
        num_reqs_padded,
        num_tokens_padded,
        seq_lens_cpu_upper_bound=speculator.input_batch.seq_lens_cpu_upper_bound,
        step=8,
        causal=speculator._group_causal,
        **kwargs,
    )


def _replay(speculator):
    return speculator.get_draft_attn_metadatas_for_replay(2, 16, speculator.input_batch.seq_lens_cpu_upper_bound)[0]


class TestDFlashMetadataReuse(unittest.TestCase):
    def assert_cleared(self, speculator):
        self.assertFalse(speculator._reuse_draft_metadata_within_propose)
        self.assertIsNone(speculator._current_propose_draft_metadata)

    def test_same_propose_reuses_all_groups_and_corrects_padded_queries(self):
        speculator = _make_speculator()

        def scenario(model):
            first = _build(model)
            tables = [metadata.block_tables for metadata in first.values()]
            self.assertEqual(first["swa"].actual_seq_lengths_q, [8, 8])
            replay = _replay(model)
            self.assertIs(replay, first)
            self.assertEqual(len(model.build_calls), 1)
            self.assertEqual([metadata.block_tables for metadata in replay.values()], tables)
            self.assertEqual([metadata.causal for metadata in replay.values()], [True, False])
            self.assertEqual([metadata.sliding_window for metadata in replay.values()], [2048, None])
            for metadata in replay.values():
                self.assertEqual(metadata.actual_seq_lengths_q, [8, 16])
            return replay

        speculator.scenario = scenario
        _propose(speculator)
        self.assert_cleared(speculator)

    def test_eager_result_cannot_leak_into_next_propose(self):
        speculator = _make_speculator()
        speculator.scenario = _build
        first = _propose(speculator)
        self.assertEqual(first["swa"].actual_seq_lengths_q, [8, 8])
        self.assert_cleared(speculator)
        speculator.scenario = lambda model: (_build(model), _replay(model))[1]
        second = _propose(speculator)
        self.assertIsNot(second, first)
        self.assertEqual(len(speculator.build_calls), 2)
        self.assert_cleared(speculator)

    def test_exception_clears_metadata(self):
        speculator = _make_speculator()

        def scenario(model):
            _build(model)
            raise RuntimeError("draft failed")

        speculator.scenario = scenario
        with self.assertRaisesRegex(RuntimeError, "draft failed"):
            _propose(speculator)
        self.assert_cleared(speculator)

    def test_failed_or_empty_build_discards_previous_result(self):
        for failure in (True, False):
            with self.subTest(failure=failure):
                speculator = _make_speculator()
                speculator._reuse_draft_metadata_within_propose = True
                _build(speculator)
                speculator.fail_build = failure
                speculator.return_no_metadata = not failure
                if failure:
                    with self.assertRaisesRegex(RuntimeError, "metadata build failed"):
                        _build(speculator)
                else:
                    self.assertIsNone(_build(speculator))
                self.assertIsNone(speculator._current_propose_draft_metadata)

    def test_dummy_and_profile_keep_rebuild_path(self):
        for options in ({"dummy_run": True}, {"is_profile": True}):
            with self.subTest(options=options):
                speculator = _make_speculator()

                def scenario(model):
                    first = _build(model)
                    self.assertIsNone(model._current_propose_draft_metadata)
                    replay = _replay(model)
                    self.assertIsNot(replay, first)
                    return replay

                speculator.scenario = scenario
                _propose(speculator, **options)
                self.assertEqual(len(speculator.build_calls), 2)
                self.assert_cleared(speculator)

    def test_capture_outside_propose_does_not_reuse(self):
        speculator = _make_speculator()
        first = _build(speculator)
        self.assertIsNot(_replay(speculator), first)
        self.assertEqual(len(speculator.build_calls), 2)
        self.assert_cleared(speculator)

    def test_descriptor_or_request_count_mismatch_rebuilds(self):
        for counts in ((2, 2, 16), (1, 1, 8), (1, 2, 32)):
            with self.subTest(counts=counts):
                speculator = _make_speculator()

                def scenario(model, counts=counts):
                    first = _build(model, *counts)
                    replay = _replay(model)
                    self.assertIsNot(replay, first)
                    self.assertEqual(model.build_calls[-1][:3], (1, 2, 16))

                speculator.scenario = scenario
                _propose(speculator)
                self.assertEqual(len(speculator.build_calls), 2)
                self.assert_cleared(speculator)

    def test_reuse_is_consumed_once(self):
        speculator = _make_speculator()

        def scenario(model):
            first = _build(model)
            self.assertIs(_replay(model), first)
            self.assertIsNot(_replay(model), first)
            self.assertIsNone(model._current_propose_draft_metadata)

        speculator.scenario = scenario
        _propose(speculator)
        self.assertEqual(len(speculator.build_calls), 2)

    def test_new_upstream_metadata_kwargs_are_forwarded(self):
        speculator = _make_speculator()
        local_lengths = object()
        _build(speculator, dcp_local_seq_lens=local_lengths)
        self.assertIs(speculator.build_calls[0][4]["dcp_local_seq_lens"], local_lengths)
        self.assertIs(speculator.build_calls[0][4]["causal"], speculator._group_causal)

    def test_full_graph_replay_uses_upstream_build_once(self):
        speculator = _make_speculator()
        updates = []
        replayed = []
        result = object()

        class UpstreamGraph:
            def run_fullgraph(self, desc):
                replayed.append(desc)
                return result

        graph_cls = _load_class(
            "vllm_ascend/worker/v2/spec_decode/dflash/aclgraph.py",
            "DFlashAclGraphManager",
            {
                "DFlashCudaGraphManager": UpstreamGraph,
                "torch": SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: None), full=lambda *args: None),
                "set_forward_context": lambda *args, **kwargs: nullcontext(),
                "get_forward_context": lambda: None,
                "_EXTRA_CTX": SimpleNamespace(),
                "update_full_graph_params": lambda *args, **kwargs: updates.append(kwargs),
            },
        )
        manager = object.__new__(graph_cls)
        manager.speculator = speculator
        manager.update_stream = SimpleNamespace(wait_stream=lambda stream: None)
        manager.vllm_config = None
        speculator.dp_size = 1
        speculator.model_state = SimpleNamespace(attn_metadata={})
        speculator.attn_backends = {"swa": object(), "full": object()}
        speculator.speculative_config = None
        descriptor = SimpleNamespace(num_reqs=2, num_tokens=16, cg_mode="FULL")

        def scenario(model):
            metadata = _build(model)
            self.assertIs(manager.run_fullgraph(descriptor), result)
            self.assertIs(updates[0]["draft_attn_metadatas"][0], metadata)

        speculator.scenario = scenario
        _propose(speculator)
        self.assertEqual(replayed, [descriptor])
        self.assertEqual(len(speculator.build_calls), 1)
        self.assert_cleared(speculator)


if __name__ == "__main__":
    unittest.main()


if __name__ == "__main__":
    unittest.main()
