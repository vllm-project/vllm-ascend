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

    def __init__(self, values=None, shape=None, device="npu:0", dtype="int32", ptr=None, strides=None):
        self.values = values
        self.shape = shape or (len(values),)
        self.device, self.dtype = device, dtype
        self.ptr = ptr or Tensor.next_ptr
        self.strides = strides
        Tensor.next_ptr += 10000

    def data_ptr(self):
        return self.ptr

    def stride(self):
        return self.strides or ((self.shape[1], 1) if len(self.shape) == 2 else (1,))

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
            self.strides,
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
            if common._token_to_req_indices_cache is None:
                common._token_to_req_indices_cache = buffers[0]
            n = common.num_actual_tokens
            slot = (
                common.slot_mapping
                if kwargs["compress_ratio"] == 1 and kwargs["circular_buffer_size"] == 0
                else buffers[3][:n]
            )
            return common._token_to_req_indices_cache[:n], buffers[1][:n], buffers[2][:n], slot

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
            compilation_config=types.SimpleNamespace(
                cudagraph_mode="FULL_DECODE_ONLY",
                cudagraph_capture_sizes=[1, 2, 4, 8, 12, 16, 20, 24, 28, 32],
            ),
            parallel_config=types.SimpleNamespace(
                tensor_parallel_size=8,
                data_parallel_size=1,
                pipeline_parallel_size=1,
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
            speculative_config=types.SimpleNamespace(
                method="mtp",
                num_speculative_tokens=3,
                uses_dynamic_speculative_decoding=lambda: False,
                enable_adaptive_verification=False,
            ),
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

    def test_capture_and_replay_guards_do_not_require_a_fixed_tp_size(self):
        config = self.model_state.vllm_config
        for tp_size in (1, 2, 4, 8, 16, 32):
            with self.subTest(tp_size=tp_size):
                config.parallel_config.tensor_parallel_size = tp_size
                self.model_state = types.SimpleNamespace(vllm_config=config)
                self.common._token_to_req_indices_cache = None
                before = len(self.calls)
                outputs = self.capture()
                self.assertEqual(len(self.calls), before + 1)
                self.assertTrue(self.calls[-1]["capturing"])
                with self.preparation():
                    replay_outputs = self.build()
                self.m.validate_target_metadata_replay(self.model_state, self.desc)
                self.m.mark_target_metadata_replay_success(self.model_state)
                self.assertEqual(len(self.calls), before + 1)
                self.assertTrue(all(self.m._same_view(a, b) for a, b in zip(outputs, replay_outputs)))
                self.assertEqual(self.model_state._qsa_target_metadata.stats["successful_replay_batches"], 1)
                # Removing the TP gate must not weaken the captured-address proof.
                self.common.seq_lens = Tensor([10000, 10000])
                with self.preparation():
                    self.build()
                with self.assertRaisesRegex(RuntimeError, "inputs or static geometry changed"):
                    self.m.validate_target_metadata_replay(self.model_state, self.desc)

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

    def test_external_cached_mapping_selects_original_bucket_before_capture(self):
        self.common._token_to_req_indices_cache = Tensor(shape=(8,))
        with self.preparation():
            self.build()
        self.assertIsNone(self.m._SCOPE.get())
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.model_state._qsa_target_metadata.baseline[(8, 2)], "external_cached_mapping")
        self.assertFalse(self.m.validate_target_metadata_replay(self.model_state, self.desc))

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

    def test_configured_mtp_lengths_and_large_owner_buffers_capture_and_replay(self):
        config = self.model_state.vllm_config
        for k in (1, 2, 3, 4, 5, 8):
            for requests in (2, 16, 32):
                with self.subTest(k=k, requests=requests):
                    q, n = k + 1, (k + 1) * requests
                    config.speculative_config.num_speculative_tokens = k
                    config.compilation_config.cudagraph_capture_sizes = list(range(q, n + 1, q))
                    self.model_state = types.SimpleNamespace(vllm_config=config)
                    self.batch = types.SimpleNamespace(num_tokens_after_padding=n, num_reqs_after_padding=requests)
                    self.common = Common(
                        Tensor(list(range(0, n + 1, q))),
                        Tensor(list(range(0, n + 1, q)), device="cpu"),
                        Tensor([10000] * requests),
                        n,
                        Tensor(shape=(requests, 184)),
                        Tensor(shape=(4096,)),
                    )
                    self.buffers = tuple(Tensor(shape=(4096,)) for _ in range(4))
                    before = len(self.calls)
                    self.capture()
                    with self.preparation():
                        self.build()
                    # Exercise the descriptor without an explicit request count.
                    self.m.validate_target_metadata_replay(
                        self.model_state, types.SimpleNamespace(num_tokens=n, num_reqs=None)
                    )
                    self.m.mark_target_metadata_replay_success(self.model_state)
                    self.assertEqual(len(self.calls), before + 1)
                    self.assertEqual(self.model_state._qsa_target_metadata.stats["successful_replay_batches"], 1)

    def test_owner_buffer_capacity_is_checked_before_deferred_capture(self):
        for index in range(4):
            with self.subTest(index=index):
                self.buffers = tuple(Tensor(shape=(7 if i == index else 4096,)) for i in range(4))
                with self.preparation():
                    self.build()
                self.assertIsNone(self.model_state._qsa_target_metadata.pending)
                self.assertEqual(self.model_state._qsa_target_metadata.baseline[(8, 2)], "persistent_buffer_capacity")

    def test_input_capacity_is_checked_before_deferred_capture(self):
        for name, value in (
            ("slot_mapping", Tensor(shape=(7,))),
            ("seq_lens", Tensor(shape=(1,))),
            ("block_table_tensor", Tensor(shape=(1, 184))),
        ):
            with self.subTest(name=name), patch.object(self.common, name, value):
                with self.preparation():
                    self.build()
                self.assertIsNone(self.model_state._qsa_target_metadata.pending)
                self.assertEqual(self.model_state._qsa_target_metadata.baseline[(8, 2)], "persistent_buffer_capacity")

    def test_dcp_pcp_unsupported_config_stays_eager(self):
        for field in ("decode_context_parallel_size", "prefill_context_parallel_size"):
            with self.subTest(field=field):
                setattr(self.model_state.vllm_config.parallel_config, field, 2)
                self.assertFalse(self.m.target_metadata_config_supported(self.model_state))
                with self.preparation():
                    self.build()
                setattr(self.model_state.vllm_config.parallel_config, field, 1)
        self.assertEqual(len(self.calls), 2)

    def test_sparse_capture_grid_keeps_original_producer_outside_graph(self):
        self.model_state.vllm_config.compilation_config.cudagraph_capture_sizes = [8, 16, 32]
        with self.preparation():
            self.build()
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(self.calls[0]["capturing"])
        self.assertIsNone(self.model_state._qsa_target_metadata.pending)
        self.assertEqual(self.m.target_metadata_snapshot()["deferred_calls"], 0)

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

    def test_shared_common_token_alias_pattern_uses_baseline_capture(self):
        with self.m.target_metadata_capture(self.model_state):
            with self.preparation(capture=True):
                first = self.build()
                second = self.build()
            self.capturing = True
            self.m.execute_target_metadata_at_model_entry(self.model_state)
            self.capturing = False
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(all(not call["capturing"] for call in self.calls))
        self.assertTrue(all(self.m._same_view(a, b) for a, b in zip(first, second)))
        with self.preparation():
            self.build()
            self.build()
        self.assertFalse(self.m.validate_target_metadata_replay(self.model_state, self.desc))
        self.assertFalse(self.model_state._qsa_target_metadata.captured)

    def test_output_alias_violation_aborts_capture(self):
        def wrong(common, *buffers, **kwargs):
            return tuple(Tensor(shape=(common.num_actual_tokens,)) for _ in range(4))

        self.m._ORIGINAL_PRODUCER = wrong
        with self.assertRaisesRegex(RuntimeError, "persistent output ABI"):
            self.capture()
        self.assertEqual(self.m.target_metadata_snapshot()["capture_failures"], 1)

    def test_original_output_count_must_match_all_four_owner_views(self):
        original = self.m._ORIGINAL_PRODUCER
        for index, missing in enumerate((True, False)):
            with self.subTest(missing=missing):

                def changed(common, *buffers, missing=missing, **kwargs):
                    result = original(common, *buffers, **kwargs)
                    return result[:-1] if missing else (*result, result[0])

                self.m._ORIGINAL_PRODUCER = changed
                with self.assertRaisesRegex(RuntimeError, "persistent output ABI"):
                    self.capture()
                self.assertEqual(len(self.calls), index + 1)
                self.assertEqual(self.m.target_metadata_snapshot()["capture_failures"], index + 1)
                self.assertIsNone(self.common._token_to_req_indices_cache)
                self.assertIsNone(self.model_state._qsa_target_metadata.pending)
                self.assertEqual(self.model_state._qsa_target_metadata.capture_depth, 0)

    def test_padded_device_boundaries_keep_trimmed_cpu_tail_and_owner_views(self):
        self.batch.num_tokens_after_padding = 16
        self.batch.num_reqs_after_padding = 4
        self.desc = types.SimpleNamespace(num_tokens=16, num_reqs=4)
        self.common = Common(
            Tensor([0, 4, 8, 8, 8]),
            Tensor([0, 4, 8], device="cpu"),
            Tensor([10000, 10000, 0, 0]),
            16,
            Tensor(shape=(4, 184), strides=(512, 2)),
            Tensor(shape=(4096,), strides=(2,)),
        )
        outputs = self.capture()
        actual = self.calls[0]["common"]
        self.assertEqual(actual.query_start_loc.values, [0, 4, 8, 8, 8])
        self.assertEqual(actual.query_start_loc_cpu.values, [0, 4, 8])
        self.assertEqual(actual.block_table_tensor.stride(), (512, 2))
        self.assertEqual(actual.slot_mapping.stride(), (2,))
        self.assertEqual([value.shape for value in outputs], [(16,)] * 4)
        self.assertIsNone(self.common._token_to_req_indices_cache)
        with self.preparation():
            self.build()
        self.m.validate_target_metadata_replay(self.model_state, self.desc)

    def test_separate_raw_and_compressed_groups_keep_distinct_owner_buffers(self):
        # Main attention can share the compressed group's Common object, but
        # only its side-cache QSAMetadataBuilder calls this producer. The raw
        # circular group owns a different Common object and different buffers.
        raw_common = Common(
            self.common.query_start_loc,
            self.common.query_start_loc_cpu,
            self.common.seq_lens,
            8,
            Tensor(shape=(2, 1), strides=(8, 1)),
            Tensor(shape=(4096,)),
        )
        raw_buffers = tuple(Tensor(shape=(4096,)) for _ in range(4))
        with self.m.target_metadata_capture(self.model_state):
            with self.preparation(capture=True):
                compressed = self.build()
                raw = self.m.build_qsa_metadata_original_target_graph(
                    raw_common,
                    *raw_buffers,
                    storage_block_size=8,
                    compress_ratio=1,
                    circular_buffer_size=8,
                )
            self.capturing = True
            self.m.execute_target_metadata_at_model_entry(self.model_state)
            self.capturing = False
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.m.target_metadata_snapshot()["captured_plans"][0]["producer_count"], 2)
        self.assertNotEqual(compressed[0].data_ptr(), raw[0].data_ptr())
        self.assertEqual(raw[3].data_ptr(), raw_buffers[3].data_ptr())
        self.assertIsNone(raw_common._token_to_req_indices_cache)
        self.assertIsNone(self.common._token_to_req_indices_cache)

    def test_side_plan_does_not_rewrite_main_owner_common_views(self):
        main_views = (
            self.common.query_start_loc,
            self.common.seq_lens,
            self.common.block_table_tensor,
            self.common.slot_mapping,
        )
        self.capture()
        self.assertEqual(
            (
                self.common.query_start_loc,
                self.common.seq_lens,
                self.common.block_table_tensor,
                self.common.slot_mapping,
            ),
            main_views,
        )
        self.assertIsNone(self.common._token_to_req_indices_cache)

    def test_owner_buffer_stride_and_work_shape_drift_rejected_before_replay(self):
        self.capture()
        token_buffer = self.buffers[0]
        self.buffers = (Tensor(shape=token_buffer.shape, ptr=token_buffer.ptr, strides=(2,)), *self.buffers[1:])
        with self.preparation():
            self.build()
        with self.assertRaisesRegex(RuntimeError, "inputs or static geometry changed"):
            self.m.validate_target_metadata_replay(self.model_state, self.desc)
        self.buffers = (token_buffer, *self.buffers[1:])
        work = self.kwargs["k_work_metadata_buffer"]
        self.kwargs["k_work_metadata_buffer"] = Tensor(shape=(3, 2), ptr=work.ptr)
        with self.preparation():
            self.build()
        with self.assertRaisesRegex(RuntimeError, "inputs or static geometry changed"):
            self.m.validate_target_metadata_replay(self.model_state, self.desc)

    def test_unsupported_query_layout_selects_baseline_bucket(self):
        self.common.query_start_loc_cpu.values[:] = [0, 3, 8]
        with self.preparation():
            self.build()
        self.assertEqual(self.model_state._qsa_target_metadata.baseline[(8, 2)], "query_layout")
        self.assertFalse(self.m.validate_target_metadata_replay(self.model_state, self.desc))

    def test_pipeline_and_ple_offload_do_not_disable_rank_local_producer(self):
        config = self.model_state.vllm_config
        config.parallel_config.pipeline_parallel_size = 4
        config.model_config.hf_text_config = types.SimpleNamespace(ple_cpu_cache_mb=4096)
        self.assertTrue(self.m.target_metadata_config_supported(self.model_state))
        self.capture()
        with self.preparation():
            self.build()
        self.assertTrue(self.m.validate_target_metadata_replay(self.model_state, self.desc))

    def test_sp_modes_select_original_capture_route_without_configuration_error(self):
        for field in ("use_sequence_parallel_moe", "_qsa_target_metadata_sp_padding"):
            with self.subTest(field=field):
                obj = self.model_state.vllm_config.parallel_config if field.startswith("use_") else self.model_state
                setattr(obj, field, True)
                self.assertTrue(self.m.target_metadata_config_supported(self.model_state))
                self.common._token_to_req_indices_cache = None
                before = len(self.calls)
                self.capture()
                self.assertEqual(len(self.calls), before + 1)
                self.assertFalse(self.calls[-1]["capturing"])
                self.common.num_actual_tokens = 4
                self.common.query_start_loc_cpu = Tensor([0, 4], device="cpu")
                self.common.query_start_loc = Tensor([0, 4, 4])
                self.common._token_to_req_indices_cache = None
                with self.preparation():
                    self.build()
                self.assertFalse(self.m.validate_target_metadata_replay(self.model_state, self.desc))
                self.assertEqual(self.model_state._qsa_target_metadata.baseline[(8, 2)], "sequence_parallel_padding")
                delattr(obj, field)

    def test_dp_active_padded_and_idle_ranks_keep_original_producer(self):
        config = self.model_state.vllm_config
        config.parallel_config.data_parallel_size = 4
        self.assertTrue(self.m.target_metadata_config_supported(self.model_state))
        self.capture()
        self.assertFalse(self.calls[0]["capturing"])
        for n, cpu, device in ((8, [0, 4, 8], [0, 4, 8]), (4, [0, 4], [0, 4, 4]), (0, [0], [0, 0, 0])):
            with self.subTest(n=n):
                self.common.num_actual_tokens = n
                self.common.query_start_loc_cpu = Tensor(cpu, device="cpu")
                self.common.query_start_loc = Tensor(device)
                self.common._token_to_req_indices_cache = None
                with self.preparation():
                    self.build()
                self.assertFalse(self.m.validate_target_metadata_replay(self.model_state, self.desc))
                self.assertIsNone(self.model_state._qsa_target_metadata.pending)
        self.assertFalse(self.model_state._qsa_target_metadata.captured)
        self.assertEqual(len(self.calls), 4)

    def test_pipeline_stage_without_qsa_uses_baseline_graph(self):
        self.model_state.vllm_config.parallel_config.pipeline_parallel_size = 4
        with self.m.target_metadata_capture(self.model_state):
            with self.preparation(capture=True):
                pass
            self.m.execute_target_metadata_at_model_entry(self.model_state)
        with self.preparation():
            pass
        self.assertFalse(self.m.validate_target_metadata_replay(self.model_state, self.desc))
        self.assertEqual(self.calls, [])

    def test_new_unsupported_layout_never_reuses_migrated_graph(self):
        self.capture()
        self.common.query_start_loc_cpu.values[:] = [0, 3, 8]
        with self.assertRaisesRegex(RuntimeError, "layout changed after graph capture"), self.preparation():
            self.build()
        # Falling back just the producer would allow captured work to overwrite
        # it. Reject this invariant change before either producer or replay.
        self.assertEqual(len(self.calls), 1)

    def test_baseline_warmup_stays_baseline_when_geometry_later_becomes_eligible(self):
        with self.m.target_metadata_capture(self.model_state):
            self.common.query_start_loc_cpu.values[:] = [0, 3, 8]
            with self.preparation(capture=True):
                self.build()
            self.m.execute_target_metadata_at_model_entry(self.model_state)
            self.common.query_start_loc_cpu.values[:] = [0, 4, 8]
            self.common._token_to_req_indices_cache = None
            with self.preparation(capture=True):
                self.build()
            self.capturing = True
            self.m.execute_target_metadata_at_model_entry(self.model_state)
            self.capturing = False
        self.assertEqual(len(self.calls), 2)
        self.assertFalse(any(call["capturing"] for call in self.calls))
        self.assertFalse(self.model_state._qsa_target_metadata.captured)
        self.common._token_to_req_indices_cache = None
        with self.preparation():
            self.build()
        self.assertFalse(self.m.validate_target_metadata_replay(self.model_state, self.desc))

    def test_eligible_warmup_then_capture_fallback_leaves_no_captured_plan(self):
        with self.m.target_metadata_capture(self.model_state):
            with self.preparation(capture=True):
                self.build()
            self.m.execute_target_metadata_at_model_entry(self.model_state)
            self.assertEqual(len(self.calls), 1)
            self.common.query_start_loc_cpu.values[:] = [0, 3, 8]
            with self.preparation(capture=True):
                self.build()
            self.capturing = True
            self.m.execute_target_metadata_at_model_entry(self.model_state)
            self.capturing = False
        snapshot = self.m.target_metadata_snapshot()
        self.assertEqual(snapshot["capture_batches"], 0)
        self.assertEqual(snapshot["capture_calls"], 0)
        self.assertEqual(snapshot["captured_plans"], [])
        self.assertIsNone(self.model_state._qsa_target_metadata.pending)
        self.assertEqual(snapshot["baseline_plans"][0]["reason"], "query_layout")

    def test_shared_common_fallback_retains_first_builder_token_mapping_alias(self):
        second_buffers = tuple(Tensor(shape=(4096,)) for _ in range(4))
        with self.m.target_metadata_capture(self.model_state):
            with self.preparation(capture=True):
                first = self.build()
                second = self.m.build_qsa_metadata_original_target_graph(self.common, *second_buffers, **self.kwargs)
            self.capturing = True
            self.m.execute_target_metadata_at_model_entry(self.model_state)
            self.capturing = False
        self.assertEqual(len(self.calls), 2)
        self.assertIs(self.calls[0]["common"], self.common)
        self.assertIs(self.calls[1]["common"], self.common)
        self.assertEqual(first[0].data_ptr(), second[0].data_ptr())
        self.assertNotEqual(first[1].data_ptr(), second[1].data_ptr())
        self.assertEqual(second[0].data_ptr(), self.buffers[0].data_ptr())
        self.assertEqual(second[1].data_ptr(), second_buffers[1].data_ptr())
        self.assertEqual(
            self.m.target_metadata_snapshot()["materializations_by_location"], {"target_baseline_original_eager": 2}
        )

    def test_unknown_bucket_does_not_silently_enter_native_replay(self):
        self.capture()
        with self.preparation():
            self.build()
        with self.assertRaises(RuntimeError):
            self.m.validate_target_metadata_replay(self.model_state, types.SimpleNamespace(num_tokens=12, num_reqs=3))

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
