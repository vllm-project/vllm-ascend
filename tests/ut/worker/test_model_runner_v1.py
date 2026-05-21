import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.outputs import LogprobsTensors, SamplerOutput

from vllm_ascend.worker.model_runner_v1 import ExecuteModelState, NPUModelRunner


class TestNPUModelRunnerKVCache(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8_indexer = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.vllm_config = MagicMock()
        runner.vllm_config.kv_transfer_config = None
        runner.model_config = MagicMock()
        runner.model_config.use_mla = True
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            2,
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        return runner

    def test_allocate_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )

        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        k_cache_raw, v_cache_raw = kv_cache_raw_tensors["draft_attn"]

        self.assertEqual(k_cache_raw.numel(), kv_cache_spec.page_size_bytes)
        self.assertEqual(v_cache_raw.numel(), kv_cache_spec.page_size_bytes)

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        kv_cache_spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=64,
            head_size_v=64,
            dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[KVCacheTensor(size=kv_cache_spec.page_size_bytes * 2, shared_by=["draft_attn"])],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["draft_attn"], kv_cache_spec=kv_cache_spec)],
        )
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=runner.attn_backend,
                layer_names=["draft_attn"],
            )
        ]

        kv_caches = runner._reshape_kv_cache_tensors(kv_cache_config, kv_cache_raw_tensors)
        k_cache, v_cache = kv_caches["draft_attn"]

        self.assertEqual(k_cache.shape, (2, 16, 8, 64))
        self.assertEqual(v_cache.shape, (2, 16, 8, 64))


class TestNPUModelRunnerOutputTokenIds(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.vllm_config = MagicMock()
        runner.model_config = MagicMock()
        return runner

    @patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable")
    def test_sample_updates_output_token_ids_before_sampler(self, mock_lmhead_tp_enable):
        """Verify output_token_ids are updated before sampler is called"""
        mock_lmhead_tp_enable.return_value = False

        # Build input batch with historical sampled tokens
        input_batch = MagicMock()
        input_batch.sampling_metadata.output_token_ids = [
            [1, 2, 3, -1],
            [4, 5, -1],
        ]
        input_batch.num_reqs = 2
        input_batch.prev_req_id_to_index = {
            "req0": 0,
            "req1": 1,
        }
        input_batch.sampled_token_ids_cpu = torch.tensor([6, 7])
        input_batch.async_copy_ready_event = MagicMock()
        input_batch.async_copy_ready_event.synchronize = MagicMock()

        # Simulate the real behavior of InputBatch.update_async_output_token_ids
        def mock_update_output_token_ids():
            output_token_ids = input_batch.sampling_metadata.output_token_ids
            sampled_ids = input_batch.sampled_token_ids_cpu.tolist()

            for index, req_id in enumerate(input_batch.prev_req_id_to_index):
                prev_index = input_batch.prev_req_id_to_index[req_id]
                req_output = output_token_ids[index]
                if req_output and req_output[-1] == -1:
                    req_output[-1] = sampled_ids[prev_index]

        input_batch.update_async_output_token_ids.side_effect = mock_update_output_token_ids

        # Build runner and inject dependencies
        runner = self._build_runner()
        runner.input_batch = input_batch
        runner.sampler = MagicMock(return_value=MagicMock())
        runner._gpu_sampler_bridge = None

        # Call sample method
        logits = torch.randn(2, 32000)
        runner._sample(logits=logits, spec_decode_metadata=None)

        # Verify sampler and update_async_output_token_ids were called
        runner.sampler.assert_called_once()
        input_batch.update_async_output_token_ids.assert_called_once()

        # Verify output_token_ids were updated before sampler is called
        call_kwargs = runner.sampler.call_args[1]
        actual_sampling_metadata = call_kwargs["sampling_metadata"]
        actual_output_token_ids = actual_sampling_metadata.output_token_ids
        self.assertEqual(actual_output_token_ids[0], [1, 2, 3, 6])
        self.assertEqual(actual_output_token_ids[1], [4, 5, 7])


class TestNPUModelRunnerGpuSamplerBridge(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        sampling_metadata = SimpleNamespace(
            output_token_ids=[
                [1, 2, -1],
                [3, -1],
            ]
        )
        input_batch = MagicMock()
        input_batch.sampling_metadata = sampling_metadata
        input_batch.num_reqs = 2
        input_batch.update_async_output_token_ids = MagicMock()

        runner.input_batch = input_batch
        runner.sampler = MagicMock(return_value="default-sampler-output")
        runner.rejection_sampler = MagicMock(return_value="rejection-sampler-output")
        runner._gpu_sampler_bridge = None
        runner._positions_at_logits = torch.tensor([10, 11], dtype=torch.int64)
        runner._input_ids_at_logits = torch.tensor([101, 102], dtype=torch.int64)
        runner._req_indices_at_logits = torch.tensor([0, 1], dtype=torch.int32)
        runner._req_ids_at_logits = ("req0", "req1")
        return runner

    def test_probabilistic_rejection_rejects_drafters_without_logits(self):
        runner = self._build_runner()
        runner.speculative_config = SimpleNamespace(
            method="ngram",
            rejection_sample_method="probabilistic",
        )

        with self.assertRaisesRegex(ValueError, "requires a drafter that exposes draft_logits"):
            runner._validate_probabilistic_rejection_config()

    def test_probabilistic_rejection_allows_draft_logits_drafters(self):
        runner = self._build_runner()
        runner.speculative_config = SimpleNamespace(
            method="eagle",
            rejection_sample_method="probabilistic",
        )

        runner._validate_probabilistic_rejection_config()

    @patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable")
    def test_sample_uses_default_sampler_when_sampling_optimization_disabled(self, mock_lmhead_tp_enable):
        mock_lmhead_tp_enable.return_value = False
        runner = self._build_runner()
        logits = torch.randn(2, 8)

        output = runner._sample(logits=logits, spec_decode_metadata=None)

        self.assertEqual(output, "default-sampler-output")
        runner.input_batch.update_async_output_token_ids.assert_called_once_with()
        runner.sampler.assert_called_once_with(
            logits=logits,
            sampling_metadata=runner.input_batch.sampling_metadata,
        )
        runner.rejection_sampler.assert_not_called()

    @patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable")
    def test_sample_uses_gpu_bridge_for_non_spec_decode_when_enabled(self, mock_lmhead_tp_enable):
        mock_lmhead_tp_enable.return_value = False
        runner = self._build_runner()
        runner._gpu_sampler_bridge = SimpleNamespace(
            sample_from_v1=MagicMock(return_value="bridge-output")
        )
        logits = torch.randn(4, 8)

        output = runner._sample(logits=logits, spec_decode_metadata=None)

        self.assertEqual(output, "bridge-output")
        runner.input_batch.update_async_output_token_ids.assert_called_once_with()
        runner._gpu_sampler_bridge.sample_from_v1.assert_called_once_with(
            logits=logits,
            sampling_metadata=runner.input_batch.sampling_metadata,
            num_reqs=runner.input_batch.num_reqs,
            positions=runner._positions_at_logits,
            input_ids=runner._input_ids_at_logits,
            req_indices=runner._req_indices_at_logits,
            req_ids=runner._req_ids_at_logits,
            spec_decode_metadata=None,
            draft_logits=None,
        )
        runner.sampler.assert_not_called()
        runner.rejection_sampler.assert_not_called()

    @patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable")
    def test_sample_truncates_lmhead_tp_logits_before_gpu_bridge(self, mock_lmhead_tp_enable):
        mock_lmhead_tp_enable.return_value = True
        runner = self._build_runner()
        runner._gpu_sampler_bridge = SimpleNamespace(
            sample_from_v1=MagicMock(return_value="bridge-output")
        )
        logits = torch.randn(5, 8)

        output = runner._sample(logits=logits, spec_decode_metadata=None)

        self.assertEqual(output, "bridge-output")
        bridge_logits = runner._gpu_sampler_bridge.sample_from_v1.call_args.kwargs["logits"]
        self.assertEqual(bridge_logits.shape, (runner.input_batch.num_reqs, 8))
        torch.testing.assert_close(bridge_logits, logits[: runner.input_batch.num_reqs])
        runner.sampler.assert_not_called()

    @patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable")
    def test_sample_updates_async_output_token_ids_before_gpu_bridge(self, mock_lmhead_tp_enable):
        mock_lmhead_tp_enable.return_value = False
        runner = self._build_runner()
        runner.input_batch.sampled_token_ids_cpu = torch.tensor([6, 7])
        runner.input_batch.prev_req_id_to_index = {
            "req0": 0,
            "req1": 1,
        }

        def update_output_token_ids():
            output_token_ids = runner.input_batch.sampling_metadata.output_token_ids
            sampled_ids = runner.input_batch.sampled_token_ids_cpu.tolist()
            output_token_ids[0][-1] = sampled_ids[0]
            output_token_ids[1][-1] = sampled_ids[1]

        def bridge_side_effect(**kwargs):
            sampling_metadata = kwargs["sampling_metadata"]
            self.assertEqual(sampling_metadata.output_token_ids[0], [1, 2, 6])
            self.assertEqual(sampling_metadata.output_token_ids[1], [3, 7])
            return "bridge-output"

        runner.input_batch.update_async_output_token_ids.side_effect = update_output_token_ids
        runner._gpu_sampler_bridge = SimpleNamespace(
            sample_from_v1=MagicMock(side_effect=bridge_side_effect)
        )

        output = runner._sample(logits=torch.randn(2, 8), spec_decode_metadata=None)

        self.assertEqual(output, "bridge-output")
        runner.input_batch.update_async_output_token_ids.assert_called_once_with()
        runner._gpu_sampler_bridge.sample_from_v1.assert_called_once()

    @patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable")
    def test_sample_uses_gpu_bridge_for_spec_decode_when_enabled(self, mock_lmhead_tp_enable):
        mock_lmhead_tp_enable.return_value = True
        runner = self._build_runner()
        runner._gpu_sampler_bridge = SimpleNamespace(
            sample_from_v1=MagicMock(return_value="bridge-output")
        )
        runner._positions_at_logits = torch.tensor([10, 11, 20], dtype=torch.int64)
        runner._input_ids_at_logits = torch.tensor([101, 102, 201], dtype=torch.int64)
        runner._req_indices_at_logits = torch.tensor([0, 0, 1], dtype=torch.int32)
        runner.drafter = SimpleNamespace(draft_logits="draft-logits")
        spec_decode_metadata = SimpleNamespace(logits_indices=[0, 2, 4])
        logits = torch.randn(5, 8)

        output = runner._sample(logits=logits, spec_decode_metadata=spec_decode_metadata)

        self.assertEqual(output, "bridge-output")
        bridge_kwargs = runner._gpu_sampler_bridge.sample_from_v1.call_args.kwargs
        torch.testing.assert_close(bridge_kwargs["logits"], logits[:3])
        self.assertIs(bridge_kwargs["sampling_metadata"], runner.input_batch.sampling_metadata)
        self.assertEqual(bridge_kwargs["num_reqs"], runner.input_batch.num_reqs)
        torch.testing.assert_close(bridge_kwargs["positions"], runner._positions_at_logits)
        torch.testing.assert_close(bridge_kwargs["input_ids"], runner._input_ids_at_logits)
        torch.testing.assert_close(bridge_kwargs["req_indices"], runner._req_indices_at_logits)
        self.assertEqual(bridge_kwargs["req_ids"], runner._req_ids_at_logits)
        self.assertIs(bridge_kwargs["spec_decode_metadata"], spec_decode_metadata)
        self.assertEqual(bridge_kwargs["draft_logits"], "draft-logits")
        runner.sampler.assert_not_called()
        runner.rejection_sampler.assert_not_called()

    @patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable")
    def test_sample_tokens_caches_positions_and_input_ids_for_bridge(self, mock_lmhead_tp_enable):
        mock_lmhead_tp_enable.return_value = False
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.kv_connector_output = None
        runner.need_accepted_tokens = False
        runner.speculative_config = None
        runner.model_config = SimpleNamespace(enable_return_routed_experts=False)
        runner.ascend_config = SimpleNamespace(
            profiling_chunk_config=SimpleNamespace(need_timing=False),
        )
        runner.dynamic_eplb = False
        runner.use_async_scheduling = False
        runner.supports_mm_inputs = False
        runner.device = torch.device("cpu")
        runner._gpu_sampler_bridge = MagicMock()
        runner._sample = MagicMock(
            return_value=SamplerOutput(
                sampled_token_ids=torch.tensor([[1], [2]], dtype=torch.int64),
                logprobs_tensors=None,
            )
        )
        runner._bookkeeping_sync = MagicMock(
            return_value=(
                None,
                [[1], [2]],
                {},
                ["req0", "req1"],
                {"req0": 0, "req1": 1},
                [],
            )
        )
        runner._finalize_dump_data = MagicMock()
        runner.input_batch = SimpleNamespace(
            num_reqs=2,
            req_ids=["req0", "req1"],
            input_ids=torch.tensor([101, 102, 999], dtype=torch.int64),
            sampling_metadata=SimpleNamespace(),
        )
        runner.req_indices = SimpleNamespace(gpu=torch.tensor([0, 1, 99], dtype=torch.int32))
        runner.input_ids = SimpleNamespace(gpu=torch.tensor([101, 102, 999], dtype=torch.int64))
        logits = torch.randn(2, 8)
        hidden_states = torch.randn(2, 4)
        positions = torch.tensor([10, 11, 999], dtype=torch.int64)
        logits_indices = torch.tensor([1, 0], dtype=torch.int64)
        runner.execute_model_state = ExecuteModelState(
            scheduler_output=SimpleNamespace(total_num_scheduled_tokens=2),
            logits=logits,
            spec_decode_metadata=None,
            spec_decode_common_attn_metadata=None,
            hidden_states=hidden_states,
            sample_hidden_states=hidden_states,
            aux_hidden_states=None,
            attn_metadata={},
            positions=positions,
            logits_indices=logits_indices,
            ec_connector_output=None,
            cudagraph_stats=None,
            batch_desc=SimpleNamespace(),
        )

        runner.sample_tokens(grammar_output=None)

        runner._sample.assert_called_once_with(logits, None)
        torch.testing.assert_close(runner._positions_at_logits, positions[logits_indices])
        torch.testing.assert_close(
            runner._input_ids_at_logits,
            runner.input_ids.gpu[logits_indices],
        )
        torch.testing.assert_close(
            runner._req_indices_at_logits,
            torch.tensor([1, 0], dtype=torch.int32),
        )
        self.assertEqual(runner._req_ids_at_logits, ("req0", "req1"))

    @patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable")
    def test_sample_tokens_ignores_lmhead_tp_padding_for_bridge_mapping(self, mock_lmhead_tp_enable):
        mock_lmhead_tp_enable.return_value = True
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.kv_connector_output = None
        runner.need_accepted_tokens = False
        runner.speculative_config = None
        runner.model_config = SimpleNamespace(enable_return_routed_experts=False)
        runner.ascend_config = SimpleNamespace(
            profiling_chunk_config=SimpleNamespace(need_timing=False),
        )
        runner.dynamic_eplb = False
        runner.use_async_scheduling = False
        runner.supports_mm_inputs = False
        runner.device = torch.device("cpu")
        runner._gpu_sampler_bridge = MagicMock()
        runner._sample = MagicMock(
            return_value=SamplerOutput(
                sampled_token_ids=torch.tensor([[1], [2]], dtype=torch.int64),
                logprobs_tensors=None,
            )
        )
        runner._bookkeeping_sync = MagicMock(
            return_value=(
                None,
                [[1], [2]],
                {},
                ["req0", "req1"],
                {"req0": 0, "req1": 1},
                [],
            )
        )
        runner._finalize_dump_data = MagicMock()
        runner.input_batch = SimpleNamespace(
            num_reqs=2,
            req_ids=["req0", "req1"],
            input_ids=torch.tensor([101, 102, 999], dtype=torch.int64),
            sampling_metadata=SimpleNamespace(),
        )
        runner.req_indices = SimpleNamespace(
            gpu=torch.tensor([0, 1, 99], dtype=torch.int32),
            np=torch.tensor([0, 1, 99], dtype=torch.int32).numpy(),
        )
        runner.input_ids = SimpleNamespace(
            gpu=torch.tensor([101, 102, 999], dtype=torch.int64),
        )
        runner.logits_indices = torch.tensor([1, 0], dtype=torch.int64)
        runner._logits_indices_np = torch.tensor([1, 0], dtype=torch.int64).numpy()
        logits = torch.randn(4, 8)
        hidden_states = torch.randn(2, 4)
        positions = torch.tensor([10, 11, 999], dtype=torch.int64)
        padded_logits_indices = torch.tensor([1, 0, 0, 0], dtype=torch.int64)
        runner.execute_model_state = ExecuteModelState(
            scheduler_output=SimpleNamespace(total_num_scheduled_tokens=2),
            logits=logits,
            spec_decode_metadata=None,
            spec_decode_common_attn_metadata=None,
            hidden_states=hidden_states,
            sample_hidden_states=hidden_states,
            aux_hidden_states=None,
            attn_metadata={},
            positions=positions,
            logits_indices=padded_logits_indices,
            ec_connector_output=None,
            cudagraph_stats=None,
            batch_desc=SimpleNamespace(),
        )

        runner.sample_tokens(grammar_output=None)

        runner._sample.assert_called_once_with(logits, None)
        torch.testing.assert_close(runner._positions_at_logits, positions[runner.logits_indices])
        torch.testing.assert_close(
            runner._req_indices_at_logits,
            torch.tensor([1, 0], dtype=torch.int32),
        )
        self.assertEqual(runner._req_indices_at_logits_np.tolist(), [1, 0])

    @patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable")
    def test_sample_tokens_builds_spec_decode_request_mapping_for_bridge(self, mock_lmhead_tp_enable):
        mock_lmhead_tp_enable.return_value = False
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.kv_connector_output = None
        runner.need_accepted_tokens = False
        runner.speculative_config = None
        runner.model_config = SimpleNamespace(enable_return_routed_experts=False)
        runner.ascend_config = SimpleNamespace(
            profiling_chunk_config=SimpleNamespace(need_timing=False),
        )
        runner.dynamic_eplb = False
        runner.use_async_scheduling = False
        runner.supports_mm_inputs = False
        runner.device = torch.device("cpu")
        runner._gpu_sampler_bridge = MagicMock()
        runner._sample = MagicMock(
            return_value=SamplerOutput(
                sampled_token_ids=torch.tensor(
                    [
                        [1, 2, 3],
                        [4, -1, -1],
                    ],
                    dtype=torch.int64,
                ),
                logprobs_tensors=None,
            )
        )
        runner._bookkeeping_sync = MagicMock(
            return_value=(
                None,
                [[1, 2, 3], [4]],
                {},
                ["req0", "req1"],
                {"req0": 0, "req1": 1},
                [],
            )
        )
        runner._finalize_dump_data = MagicMock()
        runner.input_batch = SimpleNamespace(
            num_reqs=2,
            req_ids=["req0", "req1"],
            input_ids=torch.tensor([101, 102, 103, 201, 999], dtype=torch.int64),
            sampling_metadata=SimpleNamespace(),
        )
        runner.req_indices = SimpleNamespace(gpu=torch.tensor([0, 0, 0, 1, 99], dtype=torch.int32))
        runner.input_ids = SimpleNamespace(gpu=torch.tensor([101, 102, 103, 201, 999], dtype=torch.int64))
        logits = torch.randn(4, 8)
        hidden_states = torch.randn(5, 4)
        positions = torch.tensor([10, 11, 12, 20, 999], dtype=torch.int64)
        logits_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        spec_decode_metadata = SimpleNamespace(
            num_draft_tokens=[2, 0],
        )
        runner.execute_model_state = ExecuteModelState(
            scheduler_output=SimpleNamespace(total_num_scheduled_tokens=4),
            logits=logits,
            spec_decode_metadata=spec_decode_metadata,
            spec_decode_common_attn_metadata=SimpleNamespace(),
            hidden_states=hidden_states,
            sample_hidden_states=hidden_states[logits_indices],
            aux_hidden_states=None,
            attn_metadata={},
            positions=positions,
            logits_indices=logits_indices,
            ec_connector_output=None,
            cudagraph_stats=None,
            batch_desc=SimpleNamespace(),
        )

        runner.sample_tokens(grammar_output=None)

        runner._sample.assert_called_once_with(logits, spec_decode_metadata)
        torch.testing.assert_close(runner._positions_at_logits, positions[logits_indices])
        torch.testing.assert_close(
            runner._input_ids_at_logits,
            runner.input_ids.gpu[logits_indices],
        )
        torch.testing.assert_close(
            runner._req_indices_at_logits,
            torch.tensor([0, 0, 0, 1], dtype=torch.int32),
        )
        self.assertEqual(runner._req_ids_at_logits, ("req0", "req1"))

    @patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable")
    def test_sample_tokens_accepts_non_spec_expanded_rows_with_explicit_mapping(self, mock_lmhead_tp_enable):
        mock_lmhead_tp_enable.return_value = False
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.kv_connector_output = None
        runner.need_accepted_tokens = False
        runner.speculative_config = None
        runner.model_config = SimpleNamespace(enable_return_routed_experts=False)
        runner.ascend_config = SimpleNamespace(
            profiling_chunk_config=SimpleNamespace(need_timing=False),
        )
        runner.dynamic_eplb = False
        runner.use_async_scheduling = False
        runner.supports_mm_inputs = False
        runner.device = torch.device("cpu")
        runner._gpu_sampler_bridge = MagicMock()
        runner._sample = MagicMock(
            return_value=SamplerOutput(
                sampled_token_ids=torch.tensor(
                    [
                        [1, 2],
                        [3, -1],
                    ],
                    dtype=torch.int64,
                ),
                logprobs_tensors=None,
            )
        )
        runner._bookkeeping_sync = MagicMock(
            return_value=(
                None,
                [[1, 2], [3]],
                {},
                ["req0", "req1"],
                {"req0": 0, "req1": 1},
                [],
            )
        )
        runner._finalize_dump_data = MagicMock()
        runner.input_batch = SimpleNamespace(
            num_reqs=2,
            req_ids=["req0", "req1"],
            input_ids=torch.tensor([101, 102, 201], dtype=torch.int64),
            sampling_metadata=SimpleNamespace(),
        )
        runner.req_indices = SimpleNamespace(gpu=torch.tensor([0, 0, 1], dtype=torch.int32))
        runner.input_ids = SimpleNamespace(gpu=torch.tensor([101, 102, 201], dtype=torch.int64))
        logits = torch.randn(3, 8)
        hidden_states = torch.randn(3, 4)
        positions = torch.tensor([10, 11, 20], dtype=torch.int64)
        logits_indices = torch.tensor([0, 1, 2], dtype=torch.int64)
        runner.execute_model_state = ExecuteModelState(
            scheduler_output=SimpleNamespace(total_num_scheduled_tokens=3),
            logits=logits,
            spec_decode_metadata=None,
            spec_decode_common_attn_metadata=None,
            hidden_states=hidden_states,
            sample_hidden_states=hidden_states[logits_indices],
            aux_hidden_states=None,
            attn_metadata={},
            positions=positions,
            logits_indices=logits_indices,
            ec_connector_output=None,
            cudagraph_stats=None,
            batch_desc=SimpleNamespace(),
        )

        runner.sample_tokens(grammar_output=None)

        runner._sample.assert_called_once_with(logits, None)
        torch.testing.assert_close(
            runner._req_indices_at_logits,
            torch.tensor([0, 0, 1], dtype=torch.int32),
        )

    @patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable")
    def test_sample_tokens_rejects_mismatched_spec_decode_mapping(self, mock_lmhead_tp_enable):
        mock_lmhead_tp_enable.return_value = False
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.kv_connector_output = None
        runner.need_accepted_tokens = False
        runner.speculative_config = None
        runner.model_config = SimpleNamespace(enable_return_routed_experts=False)
        runner.ascend_config = SimpleNamespace(
            profiling_chunk_config=SimpleNamespace(need_timing=False),
        )
        runner.dynamic_eplb = False
        runner.use_async_scheduling = False
        runner.supports_mm_inputs = False
        runner.device = torch.device("cpu")
        runner._gpu_sampler_bridge = MagicMock()
        runner.input_batch = SimpleNamespace(
            num_reqs=2,
            req_ids=["req0", "req1"],
            input_ids=torch.tensor([101, 102, 201], dtype=torch.int64),
            sampling_metadata=SimpleNamespace(),
        )
        runner.req_indices = SimpleNamespace(gpu=torch.tensor([0, 0, 1], dtype=torch.int32))
        runner.input_ids = SimpleNamespace(gpu=torch.tensor([101, 102, 201], dtype=torch.int64))
        runner.execute_model_state = ExecuteModelState(
            scheduler_output=SimpleNamespace(total_num_scheduled_tokens=3),
            logits=torch.randn(3, 8),
            spec_decode_metadata=SimpleNamespace(num_draft_tokens=[2, 1]),
            spec_decode_common_attn_metadata=SimpleNamespace(),
            hidden_states=torch.randn(3, 4),
            sample_hidden_states=torch.randn(3, 4),
            aux_hidden_states=None,
            attn_metadata={},
            positions=torch.tensor([10, 11, 20], dtype=torch.int64),
            logits_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
            ec_connector_output=None,
            cudagraph_stats=None,
            batch_desc=SimpleNamespace(),
        )

        with self.assertRaisesRegex(ValueError, "speculative logits-row mapping"):
            runner.sample_tokens(grammar_output=None)

    def test_bookkeeping_accepts_spec_decode_logprobs_without_cu_tokens(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.discard_request_indices = SimpleNamespace(np=torch.empty(0, dtype=torch.int64).numpy())
        runner.num_discarded_requests = 0
        runner.use_async_scheduling = False
        runner.input_batch = SimpleNamespace(
            req_ids=["req0", "req1"],
            req_id_to_index={"req0": 0, "req1": 1},
            vocab_size=8,
            num_tokens_no_spec=[0, 0],
            token_ids_cpu=torch.zeros((2, 4), dtype=torch.int64).numpy(),
            is_token_ids=torch.zeros((2, 4), dtype=torch.bool).numpy(),
            num_tokens=[0, 0],
            generators={},
        )
        runner.requests = {
            "req0": SimpleNamespace(output_token_ids=[]),
            "req1": SimpleNamespace(output_token_ids=[]),
        }
        runner.max_model_len = 4
        runner._get_prompt_logprobs_dict = MagicMock(return_value={})
        logprobs_tensors = LogprobsTensors(
            logprob_token_ids=torch.tensor([[1], [2], [3]], dtype=torch.int32),
            logprobs=torch.tensor([[-0.1], [-0.2], [-0.3]], dtype=torch.float32),
            selected_token_ranks=torch.tensor([1, 1, 1], dtype=torch.int32),
            cu_num_generated_tokens=[0, 2, 3],
        )
        sampler_output = SamplerOutput(
            sampled_token_ids=torch.tensor([[1, 2], [3, -1]], dtype=torch.int32),
            logprobs_tensors=logprobs_tensors,
        )

        logprobs_lists, valid_sampled_token_ids, *_ = runner._bookkeeping_sync(
            scheduler_output=SimpleNamespace(num_scheduled_tokens={}),
            sampler_output=sampler_output,
            logits=torch.randn(3, 8),
            hidden_states=torch.randn(3, 4),
            num_scheduled_tokens=0,
            spec_decode_metadata=SimpleNamespace(),
        )

        self.assertEqual(valid_sampled_token_ids, [[1, 2], [3]])
        self.assertIsNotNone(logprobs_lists)
        self.assertEqual(runner.requests["req0"].output_token_ids, [1, 2])
        self.assertEqual(runner.requests["req1"].output_token_ids, [3])


class TestNPUModelRunnerDebugger(unittest.TestCase):
    def _build_runner(self, debugger=None):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.debugger = debugger or MagicMock()
        runner.model = MagicMock()
        runner.model_config = MagicMock()
        runner.model_config.enforce_eager = False
        runner._debugger_started = True
        runner._debugger_step_dummy_data_before_execute = False
        return runner

    def test_finalize_dump_data_stops_stop_capable_debugger(self):
        runner = self._build_runner()

        runner._finalize_dump_data()

        runner.debugger.stop.assert_called_once_with()
        runner.debugger.step.assert_called_once_with()
        self.assertFalse(runner._debugger_started)

    def test_finalize_dump_data_steps_graph_debugger_without_stop(self):
        debugger = MagicMock(spec=["start", "step"])
        runner = self._build_runner(debugger)

        runner._finalize_dump_data()

        debugger.step.assert_called_once_with()
        self.assertTrue(runner._debugger_started)

    def test_start_dump_data_noop_when_already_started(self):
        runner = self._build_runner(MagicMock(spec=["start", "step"]))

        runner._start_dump_data()

        runner.debugger.start.assert_not_called()
        runner.debugger.step.assert_not_called()
        self.assertTrue(runner._debugger_started)


if __name__ == "__main__":
    unittest.main()
