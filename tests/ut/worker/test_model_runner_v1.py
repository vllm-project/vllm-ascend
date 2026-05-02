import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


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


def _make_attn_state_runner(num_computed_tokens, spec_method, enable_chunked_prefill):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array(num_computed_tokens, dtype=np.int32),
    )
    runner.scheduler_config = SimpleNamespace(enable_chunked_prefill=enable_chunked_prefill)
    if spec_method is None:
        runner.speculative_config = None
    else:
        runner.speculative_config = SimpleNamespace(method=spec_method, num_speculative_tokens=3)
    return runner


class TestNPUModelRunnerResolveBatchAttnState(unittest.TestCase):
    """Tests for _resolve_batch_attn_state — returns (attn_state, with_prefill).

    Covers both the underlying branch table (delegated to _get_attn_state)
    and the PCP+eagle3 override layered on top. Pure function — no device
    required and no instance state mutation.
    """

    def _resolve(self, num_computed_tokens, num_scheduled, num_valid, spec_method=None, enable_chunked_prefill=False):
        runner = _make_attn_state_runner(num_computed_tokens, spec_method, enable_chunked_prefill)
        return runner._resolve_batch_attn_state(
            len(num_computed_tokens),
            np.array(num_scheduled, dtype=np.int32),
            np.array(num_valid, dtype=np.int32),
        )

    def test_prefill_no_cache_wins_over_chunked(self):
        # All num_computed_tokens == 0 wins, even with chunked_prefill enabled
        for enable_chunked in (False, True):
            state, with_prefill = self._resolve(
                # （num_computed_tokens, num_scheduled_tokens, num_valid_tokens）
                [0, 0, 0],
                [10, 10, 10],
                [10, 10, 10],
                enable_chunked_prefill=enable_chunked,
            )
            self.assertEqual(state, AscendAttentionState.PrefillNoCache)
            self.assertTrue(with_prefill)

    def test_decode_only(self):
        # All 1 wins over chunked_prefill and mtp spec, even with mtp spec_method
        # （num_computed_tokens, num_scheduled_tokens, num_valid_tokens）
        state, with_prefill = self._resolve([5, 10, 15], [1, 1, 1], [1, 1, 1])
        self.assertEqual(state, AscendAttentionState.DecodeOnly)
        self.assertFalse(with_prefill)

    def test_decode_one_token_mtp_upgrades_to_spec(self):
        state, with_prefill = self._resolve(
            [5, 10, 15],
            [1, 1, 1],
            [1, 1, 1],
            spec_method="mtp",
        )
        self.assertEqual(state, AscendAttentionState.SpecDecoding)
        self.assertFalse(with_prefill)

    def test_decode_one_token_non_mtp_stays_decode(self):
        state, with_prefill = self._resolve(
            [5, 10, 15],
            [1, 1, 1],
            [1, 1, 1],
            spec_method="eagle",
        )
        self.assertEqual(state, AscendAttentionState.DecodeOnly)
        self.assertFalse(with_prefill)

    def test_valid_one_no_spec_is_chunked(self):
        state, with_prefill = self._resolve([5, 10, 15], [4, 4, 4], [1, 1, 1])
        self.assertEqual(state, AscendAttentionState.ChunkedPrefill)
        self.assertTrue(with_prefill)

    def test_chunked_prefill_enabled(self):
        state, with_prefill = self._resolve(
            [5, 10, 15],
            [10, 5, 1],
            [10, 5, 1],
            enable_chunked_prefill=True,
        )
        self.assertEqual(state, AscendAttentionState.ChunkedPrefill)
        self.assertTrue(with_prefill)

    def test_prefill_cache_hit_fallback(self):
        state, with_prefill = self._resolve([5, 10, 15], [10, 5, 1], [10, 5, 1])
        self.assertEqual(state, AscendAttentionState.PrefillCacheHit)
        self.assertTrue(with_prefill)

    def test_spec_decoding_mtp_no_override(self):
        # mtp method does NOT trigger PCP override: state stays SpecDecoding
        state, with_prefill = self._resolve(
            [5, 10, 15],
            [4, 4, 4],
            [1, 1, 1],
            spec_method="mtp",
        )
        self.assertEqual(state, AscendAttentionState.SpecDecoding)
        self.assertFalse(with_prefill)

    def test_spec_decoding_non_mtp_pcp_override(self):
        state, with_prefill = self._resolve(
            [5, 10, 15],
            [4, 4, 4],
            [1, 1, 1],
            spec_method="eagle",
        )
        self.assertEqual(state, AscendAttentionState.ChunkedPrefill)
        self.assertFalse(with_prefill)

    def test_does_not_mutate_self_attn_state(self):
        runner = _make_attn_state_runner([0, 0, 0], None, False)
        runner.attn_state = "sentinel"
        runner._resolve_batch_attn_state(3, np.array([10, 10, 10]), np.array([10, 10, 10]))
        self.assertEqual(runner.attn_state, "sentinel")


if __name__ == "__main__":
    unittest.main()
