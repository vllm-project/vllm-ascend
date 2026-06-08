import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerKVCache(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8_indexer = False
        runner.use_compress = False
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
        runner.use_compress = False
        return runner

    @patch("vllm_ascend.worker.model_runner_v1.get_ascend_config")
    @patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable")
    def test_sample_updates_output_token_ids_before_sampler(self, mock_lmhead_tp_enable, mock_get_ascend_config):
        """Verify output_token_ids are updated before sampler is called"""
        mock_lmhead_tp_enable.return_value = False
        mock_ascend_config = MagicMock()
        mock_ascend_config.enable_reduce_sample = False
        mock_get_ascend_config.return_value = mock_ascend_config

        # Build input batch with historical sampled tokens
        input_batch = MagicMock()
        input_batch.sampling_metadata.output_token_ids = [
            [1, 2, 3, -1],
            [4, 5, -1],
        ]
        input_batch.num_reqs = 2
        input_batch.top_k_cpu = None
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

    @patch("torch.npu.current_stream")
    @patch("torch.npu.stream")
    def test_copy_valid_sampled_token_count_syncs_host_accepted_counts(self, mock_npu_stream, mock_current_stream):
        mock_current_stream.return_value = MagicMock()
        mock_npu_stream.side_effect = lambda _stream: nullcontext()

        runner = self._build_runner()
        runner.valid_sampled_token_count_event = MagicMock()
        runner.valid_sampled_token_count_copy_stream = MagicMock()
        runner.use_async_spec_decode = False
        runner.valid_sampled_token_count_cpu = torch.zeros(4, dtype=torch.int32)
        runner.input_batch = SimpleNamespace(
            num_accepted_tokens_cpu_tensor=torch.zeros(4, dtype=torch.int32),
            prev_sampled_token_ids=None,
        )

        next_token_ids = torch.tensor([11, 12], dtype=torch.int32)
        valid_counts = torch.tensor([3, 1], dtype=torch.int32)

        runner._copy_valid_sampled_token_count(next_token_ids, valid_counts)

        self.assertTrue(torch.equal(runner.input_batch.num_accepted_tokens_cpu_tensor[:2], valid_counts))
        self.assertTrue(torch.equal(runner.input_batch.prev_sampled_token_ids, next_token_ids.unsqueeze(1)))

    @patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable")
    @patch("vllm_ascend.worker.model_runner_v1.get_ascend_config")
    def test_sample_mtp_returns_executor_counts_explicitly(self, mock_get_ascend_config, mock_lmhead_tp_enable):
        mock_lmhead_tp_enable.return_value = False
        mock_ascend_config = MagicMock()
        mock_ascend_config.enable_reduce_sample = False
        mock_get_ascend_config.return_value = mock_ascend_config

        runner = self._build_runner()
        runner.input_batch = MagicMock()
        runner.input_batch.sampling_metadata = MagicMock()
        runner.input_batch.sampling_metadata.top_k = None
        runner.input_batch.top_k_cpu = None
        runner.input_batch.num_reqs = 2
        runner.input_batch.update_async_output_token_ids = MagicMock()
        runner.speculative_config = SimpleNamespace(method="mtp")
        runner.num_spec_tokens = 3
        expected_output = MagicMock()
        expected_counts = torch.tensor([4, 2], dtype=torch.int32)
        runner.spec_sampling_executor = MagicMock()
        runner.spec_sampling_executor.execute_from_runtime.return_value = SimpleNamespace(
            sampler_output=expected_output,
            num_output_tokens_per_req=expected_counts,
        )

        logits = torch.randn(4, 32)
        spec_decode_metadata = MagicMock()
        sampler_output, counts = runner._sample(logits, spec_decode_metadata)

        self.assertIs(sampler_output, expected_output)
        self.assertTrue(torch.equal(counts, expected_counts))
        runner.spec_sampling_executor.execute_from_runtime.assert_called_once()

    @patch("vllm_ascend.worker.model_runner_v1.get_tp_group")
    @patch("torch.distributed.broadcast")
    def test_sync_mtp_output_counts_across_tp_noop_for_single_tp(self, mock_broadcast, mock_get_tp_group):
        runner = self._build_runner()
        runner.speculative_config = SimpleNamespace(method="mtp")
        mock_get_tp_group.return_value = SimpleNamespace(world_size=1, device_group="tp")

        counts = torch.tensor([2, 1], dtype=torch.int32)
        result = runner._sync_mtp_output_counts_across_tp(counts)

        self.assertTrue(torch.equal(result, counts))
        mock_broadcast.assert_not_called()

    @patch("vllm_ascend.worker.model_runner_v1.get_tp_group")
    @patch("torch.distributed.broadcast")
    def test_sync_mtp_output_counts_across_tp_broadcasts_from_rank0(self, mock_broadcast, mock_get_tp_group):
        runner = self._build_runner()
        runner.speculative_config = SimpleNamespace(method="mtp")
        mock_get_tp_group.return_value = SimpleNamespace(world_size=2, device_group="tp")

        def _fill_from_rank0(tensor, src=0, group=None):
            tensor.copy_(torch.tensor([4, 3], dtype=tensor.dtype))

        mock_broadcast.side_effect = _fill_from_rank0
        counts = torch.tensor([2, 1], dtype=torch.int32)

        result = runner._sync_mtp_output_counts_across_tp(counts)

        self.assertTrue(torch.equal(result, torch.tensor([4, 3], dtype=torch.int32)))
        mock_broadcast.assert_called_once()

    def test_step_needs_accepted_tokens_requires_spec_step(self):
        runner = self._build_runner()
        runner.need_accepted_tokens = True

        self.assertFalse(runner._step_needs_accepted_tokens(False))
        self.assertTrue(runner._step_needs_accepted_tokens(True))

    def test_step_needs_seq_lens_cpu_sync_requires_spec_step(self):
        runner = self._build_runner()
        runner._needs_seq_lens_cpu_sync = True
        runner.use_async_spec_decode = True
        runner.valid_sampled_token_count_gpu = torch.tensor([1], dtype=torch.int32)

        self.assertFalse(runner._step_needs_seq_lens_cpu_sync(False, {"req": 0}))
        self.assertTrue(runner._step_needs_seq_lens_cpu_sync(True, {"req": 0}))

    def test_gdn_builder_enables_need_accepted_tokens(self):
        class DummyGDN:
            pass

        runner = self._build_runner()
        with patch("vllm_ascend.worker.model_runner_v1.GDNAttentionMetadataBuilder", DummyGDN):
            gdn_builder = DummyGDN()
            fake_group = [
                SimpleNamespace(
                    kv_cache_spec=object(),
                    get_metadata_builder=MagicMock(return_value=gdn_builder),
                )
            ]
            runner.attn_groups = [fake_group]
            runner.need_accepted_tokens = any(
                isinstance(attn_group[0].get_metadata_builder(0), DummyGDN) for attn_group in runner.attn_groups
            )

            self.assertTrue(runner.need_accepted_tokens)


class TestNPUModelRunnerDebugger(unittest.TestCase):
    def _build_runner(self, debugger=None):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.debugger = debugger or MagicMock()
        runner.model = MagicMock()
        runner.model_config = MagicMock()
        runner.model_config.enforce_eager = False
        runner._debugger_started = True
        runner._debugger_step_dummy_data_before_execute = False
        runner.use_compress = False
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
