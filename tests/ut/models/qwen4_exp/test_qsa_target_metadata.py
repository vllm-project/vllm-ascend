# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU control-flow tests; device numeric parity is a separate NPU test."""

import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "vllm_ascend"
HELPER = SOURCE / "models/qwen4_exp/qsa_target_metadata.py"


class Tensor:
    next_ptr = 1000

    def __init__(self, values=None, shape=None, device="npu:0", dtype="int32", ptr=None):
        self.values = values
        self.shape = shape or (len(values),)
        self.device, self.dtype = device, dtype
        self.ptr = ptr or Tensor.next_ptr
        Tensor.next_ptr += 10000

    def data_ptr(self):
        return self.ptr

    def stride(self):
        return (self.shape[1], 1) if len(self.shape) == 2 else (1,)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.values[index]
        start, stop, step = index.indices(self.shape[0])
        assert start == 0 and step == 1
        return Tensor(
            None if self.values is None else self.values[:stop],
            (stop, *self.shape[1:]),
            self.device,
            self.dtype,
            self.ptr,
        )


@dataclass
class Common:
    query_start_loc: Tensor
    query_start_loc_cpu: Tensor
    seq_lens: Tensor
    num_actual_tokens: int
    block_table_tensor: Tensor
    slot_mapping: Tensor
    _token_to_req_indices_cache: Tensor | None = None


class TestTargetMetadata(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.capturing = False

        def original(common, *buffers, **kwargs):
            self.calls.append(dict(common=common, capturing=self.capturing))
            common._token_to_req_indices_cache = buffers[0]
            n = common.num_actual_tokens
            slot = (
                common.slot_mapping
                if kwargs["compress_ratio"] == 1 and kwargs["circular_buffer_size"] == 0
                else buffers[3][:n]
            )
            return buffers[0][:n], buffers[1][:n], buffers[2][:n], slot

        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = Tensor
        fake_torch.npu = types.SimpleNamespace(is_current_stream_capturing=lambda: self.capturing)
        cache = types.ModuleType("vllm.models.qwen4_exp.common.qsa_cache")
        cache._build_qsa_metadata_torch = original
        common_module = types.ModuleType("vllm.models.qwen4_exp.common")
        common_module.qsa_cache = cache
        compilation = types.ModuleType("vllm.config.compilation")
        compilation.CUDAGraphMode = types.SimpleNamespace(FULL="FULL", FULL_DECODE_ONLY="FULL_DECODE_ONLY")
        modules = {
            "torch": fake_torch,
            "vllm.config.compilation": compilation,
            "vllm.models.qwen4_exp.common": common_module,
        }
        self.modules_patch = patch.dict(sys.modules, modules)
        self.modules_patch.start()
        spec = importlib.util.spec_from_file_location("qsa_target_metadata_test", HELPER)
        self.m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.m
        spec.loader.exec_module(self.m)
        config = types.SimpleNamespace(
            compilation_config=types.SimpleNamespace(cudagraph_mode="FULL_DECODE_ONLY"),
            parallel_config=types.SimpleNamespace(
                tensor_parallel_size=8,
                data_parallel_size=1,
                pipeline_parallel_size=1,
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
            speculative_config=types.SimpleNamespace(num_speculative_tokens=3),
            model_config=types.SimpleNamespace(enforce_eager=False),
        )
        self.model_state = types.SimpleNamespace(vllm_config=config)
        self.batch = types.SimpleNamespace(num_tokens_after_padding=8, num_reqs_after_padding=2)
        self.common = Common(
            Tensor([0, 4, 8]),
            Tensor([0, 4, 8], device="cpu"),
            Tensor([10000, 10000]),
            8,
            Tensor(shape=(2, 184)),
            Tensor(shape=(4096,)),
        )
        self.buffers = tuple(Tensor(shape=(4096,)) for _ in range(4))
        self.kwargs = dict(storage_block_size=64, compress_ratio=4, k_work_metadata_buffer=Tensor(shape=(4, 2)))
        self.desc = types.SimpleNamespace(num_tokens=8, num_reqs=2)

    def tearDown(self):
        self.modules_patch.stop()
        sys.modules.pop("qsa_target_metadata_test", None)

    def build(self, common=None, **kwargs):
        return self.m.build_qsa_metadata_original_target_graph(
            common or self.common, *self.buffers, **(self.kwargs | kwargs)
        )

    def preparation(self, capture=False, mode="FULL"):
        return self.m.target_metadata_preparation(self.model_state, self.batch, mode, capture)

    def capture(self):
        with self.m.target_metadata_capture(self.model_state):
            with self.preparation(capture=True):
                outputs = self.build()
            self.capturing = True
            self.m.execute_target_metadata_at_model_entry(self.model_state)
            self.capturing = False
        return outputs

    def test_outside_target_keeps_original_eager(self):
        self.build()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.m.target_metadata_snapshot()["eager_fallbacks_by_reason"], {"outside_target_scope": 1})

    def test_prefill_non_full_is_eager(self):
        with self.preparation(mode="NONE"):
            self.build()
        self.assertEqual(len(self.calls), 1)
        self.assertIsNone(self.model_state._qsa_target_metadata.pending)

    def test_none_dummy_capture_flag_outside_manager_keeps_eager(self):
        with self.preparation(capture=True, mode="NONE"):
            self.build()
        self.assertEqual(len(self.calls), 1)
        self.assertIsNone(self.model_state._qsa_target_metadata.pending)

    def test_dummy_full_replay_capture_flag_uses_runtime_plan(self):
        self.capture()
        with self.preparation(capture=True, mode="FULL"):
            self.build()
        self.m.validate_target_metadata_replay(self.model_state, self.desc)
        self.assertEqual(self.m.target_metadata_snapshot()["validated_replay_batches"], 1)

    def test_full_builder_defers_all_device_work(self):
        with self.preparation():
            self.build()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.m.target_metadata_snapshot()["deferred_calls"], 1)

    def test_original_producer_executes_inside_capture(self):
        self.capture()
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.calls[0]["capturing"])
        self.assertEqual(self.m.target_metadata_snapshot()["capture_batches"], 1)

    def test_warmup_cache_does_not_remove_capture_mapping_work(self):
        with self.m.target_metadata_capture(self.model_state):
            with self.preparation(capture=True):
                self.build()
            self.m.execute_target_metadata_at_model_entry(self.model_state)
            self.capturing = True
            self.m.execute_target_metadata_at_model_entry(self.model_state)
        self.assertEqual(len(self.calls), 2)
        self.assertIsNot(self.calls[0]["common"], self.calls[1]["common"])
        self.assertEqual(self.m.target_metadata_snapshot()["capture_calls"], 1)

    def test_runtime_uses_captured_device_work_without_python_materialization(self):
        self.capture()
        self.common.seq_lens.values[:] = [10033, 10031]
        self.common.query_start_loc.values[:] = [0, 3, 8]
        with self.preparation():
            self.build()
        self.m.validate_target_metadata_replay(self.model_state, self.desc)
        self.m.mark_target_metadata_replay_success(self.model_state)
        self.assertEqual(len(self.calls), 1)
        stats = self.m.target_metadata_snapshot()
        self.assertEqual(stats["validated_replay_batches"], 1)
        self.assertEqual(stats["successful_replay_batches"], 1)

    def test_pointer_mismatch_fails_before_replay(self):
        self.capture()
        self.common.seq_lens = Tensor([10000, 10000])
        with self.preparation():
            self.build()
        with self.assertRaisesRegex(RuntimeError, "inputs or static geometry changed"):
            self.m.validate_target_metadata_replay(self.model_state, self.desc)
        self.assertEqual(self.m.target_metadata_snapshot()["successful_replay_batches"], 0)

    def test_mapped_count_mismatch_fails_before_replay(self):
        self.capture()
        self.common.query_start_loc_cpu.values[:] = [0, 4, 4]
        with self.preparation():
            self.build()
        with self.assertRaises(RuntimeError):
            self.m.validate_target_metadata_replay(self.model_state, self.desc)

    def test_external_cached_mapping_is_rejected(self):
        self.common._token_to_req_indices_cache = Tensor(shape=(8,))
        with self.assertRaisesRegex(RuntimeError, "Cached external"), self.preparation():
            self.build()
        self.assertIsNone(self.m._SCOPE.get())

    def test_ratio_one_preserves_common_slot_alias(self):
        with self.preparation():
            outputs = self.build(compress_ratio=1, k_work_metadata_buffer=None)
        self.assertEqual(outputs[3].data_ptr(), self.common.slot_mapping.data_ptr())

    def test_all_eight_capture_buckets_bind_their_own_geometry(self):
        with self.m.target_metadata_capture(self.model_state):
            for n in (4, 8, 12, 16, 20, 24, 28, 32):
                r = n // 4
                self.batch.num_tokens_after_padding = n
                self.batch.num_reqs_after_padding = r
                common = Common(
                    Tensor(list(range(0, n + 1, 4))),
                    Tensor(list(range(0, n + 1, 4)), device="cpu"),
                    Tensor([10000] * r),
                    n,
                    Tensor(shape=(r, 184)),
                    Tensor(shape=(4096,)),
                )
                with self.preparation(capture=True):
                    self.build(common)
                self.capturing = True
                self.m.execute_target_metadata_at_model_entry(self.model_state)
                self.capturing = False
        snapshot = self.m.target_metadata_snapshot()
        self.assertEqual({row["num_tokens"] for row in snapshot["captured_plans"]}, {4, 8, 12, 16, 20, 24, 28, 32})

    def test_dcp_pcp_unsupported_config_stays_eager(self):
        for field in ("decode_context_parallel_size", "prefill_context_parallel_size"):
            with self.subTest(field=field):
                setattr(self.model_state.vllm_config.parallel_config, field, 2)
                self.assertFalse(self.m.target_metadata_config_supported(self.model_state))
                with self.preparation():
                    self.build()
                setattr(self.model_state.vllm_config.parallel_config, field, 1)
        self.assertEqual(len(self.calls), 2)

    def test_materialization_locations_are_counted(self):
        self.build()
        self.common._token_to_req_indices_cache = None
        self.capture()
        snapshot = self.m.target_metadata_snapshot()
        self.assertEqual(
            snapshot["materializations_by_location"], {"unscoped_original_eager": 1, "target_native_capture": 1}
        )
        self.assertEqual(snapshot["eligible_outside_model_calls"], 0)

    def test_misplaced_eligible_materialization_is_counted_and_rejected(self):
        state = self.m._state(self.model_state)
        with self.assertRaisesRegex(RuntimeError, "escaped"):
            self.m._invoke_original(
                self.common, self.buffers, self.kwargs, state=state, eligible=True, at_model_entry=False
            )
        self.assertEqual(self.m.target_metadata_snapshot()["eligible_outside_model_calls"], 1)
        self.assertEqual(self.calls, [])

    def test_shared_common_token_alias_pattern_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "sharing one Common"), self.preparation():
            self.build()
            self.build()

    def test_output_alias_violation_aborts_capture(self):
        def wrong(common, *buffers, **kwargs):
            return tuple(Tensor(shape=(common.num_actual_tokens,)) for _ in range(4))

        self.m._ORIGINAL_PRODUCER = wrong
        with self.assertRaisesRegex(RuntimeError, "persistent output ABI"):
            self.capture()
        self.assertEqual(self.m.target_metadata_snapshot()["capture_failures"], 1)

    def test_unsupported_query_layout_is_rejected(self):
        self.common.query_start_loc_cpu.values[:] = [0, 3, 8]
        with self.assertRaisesRegex(RuntimeError, "uniform K3"), self.preparation():
            self.build()

    def test_no_nested_context_leak_on_failure(self):
        with self.assertRaises(ValueError), self.preparation():
            raise ValueError("builder failed")
        self.assertIsNone(self.m._SCOPE.get())
        self.assertIsNone(self.model_state._qsa_target_metadata.pending)

    def test_uncaptured_model_forward_is_rejected(self):
        with self.preparation():
            self.build()
        with self.assertRaisesRegex(RuntimeError, "uncaptured model forward"):
            self.m.execute_target_metadata_at_model_entry(self.model_state)

    def test_wrong_descriptor_is_rejected(self):
        self.capture()
        with self.preparation():
            self.build()
        with self.assertRaises(RuntimeError):
            self.m.validate_target_metadata_replay(self.model_state, types.SimpleNamespace(num_tokens=4, num_reqs=1))

    def test_capture_scope_cleans_pending_and_depth(self):
        self.capture()
        state = self.model_state._qsa_target_metadata
        self.assertIsNone(state.pending)
        self.assertEqual(state.capture_depth, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
