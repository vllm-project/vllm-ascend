import unittest
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


class TestNPUModelRunnerGetSpecDecodeDraftProbs(unittest.TestCase):
    """Cover ``_get_spec_decode_draft_probs`` introduced for vllm PR #40269.

    The method has no NPU dependencies: it just re-aligns the cached
    ``[num_reqs, num_spec_tokens, vocab]`` draft probabilities to the
    rejection-sampler-expected flat ``[sum(num_draft_tokens), vocab]`` layout
    via a ``req_id -> row`` lookup. Easy to unit-test in pure python.
    """

    VOCAB = 7
    NUM_SPEC = 3

    def _build_runner(
        self,
        draft_probs,
        draft_prob_req_ids,
        current_req_ids,
    ):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner._draft_probs = draft_probs
        runner._draft_prob_req_ids = draft_prob_req_ids
        runner.input_batch = MagicMock()
        runner.input_batch.req_ids = current_req_ids
        return runner

    def _make_spec_metadata(self, num_draft_tokens):
        return SimpleNamespace(num_draft_tokens=num_draft_tokens)

    def test_returns_none_when_probs_never_cached(self):
        """Default state (probabilistic sampling disabled) → no probs."""
        runner = self._build_runner(draft_probs=None, draft_prob_req_ids=None, current_req_ids=["a", "b"])
        spec_md = self._make_spec_metadata([self.NUM_SPEC, self.NUM_SPEC])
        self.assertIsNone(runner._get_spec_decode_draft_probs(spec_md))

    def test_returns_none_when_only_one_side_cached(self):
        """Defensive: half-populated state must not throw, returns None."""
        runner = self._build_runner(
            draft_probs=torch.rand(2, self.NUM_SPEC, self.VOCAB),
            draft_prob_req_ids=None,
            current_req_ids=["a", "b"],
        )
        self.assertIsNone(runner._get_spec_decode_draft_probs(self._make_spec_metadata([1, 1])))

    def test_concats_and_slices_per_request(self):
        """Per-req num_draft_tokens slicing then dim-0 concat."""
        probs = torch.arange(2 * self.NUM_SPEC * self.VOCAB, dtype=torch.float32).reshape(2, self.NUM_SPEC, self.VOCAB)
        runner = self._build_runner(
            draft_probs=probs,
            draft_prob_req_ids=["a", "b"],
            current_req_ids=["a", "b"],
        )
        # req "a" has 2 draft tokens, req "b" has 3 — total 5 rows
        spec_md = self._make_spec_metadata([2, 3])

        out = runner._get_spec_decode_draft_probs(spec_md)

        self.assertIsNotNone(out)
        self.assertEqual(out.shape, (5, self.VOCAB))
        # First 2 rows == probs[0, :2], next 3 rows == probs[1, :3]
        self.assertTrue(torch.equal(out[:2], probs[0, :2]))
        self.assertTrue(torch.equal(out[2:], probs[1, :3]))

    def test_skips_requests_with_zero_draft_tokens(self):
        """Requests with num_draft_tokens=0 (e.g. just finished prefill)
        must contribute zero rows but not crash."""
        probs = torch.rand(3, self.NUM_SPEC, self.VOCAB)
        runner = self._build_runner(
            draft_probs=probs,
            draft_prob_req_ids=["a", "b", "c"],
            current_req_ids=["a", "b", "c"],
        )
        # Middle req has zero draft tokens
        spec_md = self._make_spec_metadata([2, 0, 1])

        out = runner._get_spec_decode_draft_probs(spec_md)

        self.assertIsNotNone(out)
        # Only req "a" (2 rows) and req "c" (1 row) contribute.
        self.assertEqual(out.shape, (3, self.VOCAB))
        self.assertTrue(torch.equal(out[:2], probs[0, :2]))
        self.assertTrue(torch.equal(out[2:3], probs[2, :1]))

    def test_returns_none_when_all_requests_have_zero_draft_tokens(self):
        """Whole batch has no draft tokens (full-prefill step) → None."""
        probs = torch.rand(2, self.NUM_SPEC, self.VOCAB)
        runner = self._build_runner(
            draft_probs=probs,
            draft_prob_req_ids=["a", "b"],
            current_req_ids=["a", "b"],
        )
        self.assertIsNone(runner._get_spec_decode_draft_probs(self._make_spec_metadata([0, 0])))

    @patch("vllm_ascend.worker.model_runner_v1.logger")
    def test_returns_none_when_req_id_unknown(self, mock_logger):
        """If the batch was reshuffled between drafting and sampling and a
        req_id is missing from the snapshot, we must fall back to None (not
        index by mistake) and emit a warning."""
        probs = torch.rand(1, self.NUM_SPEC, self.VOCAB)
        runner = self._build_runner(
            draft_probs=probs,
            draft_prob_req_ids=["a"],
            # Current batch references a request that wasn't in the snapshot.
            current_req_ids=["a", "ghost"],
        )
        spec_md = self._make_spec_metadata([1, 2])

        out = runner._get_spec_decode_draft_probs(spec_md)

        self.assertIsNone(out)
        mock_logger.warning.assert_called_once()

    def test_reordered_request_ids_use_snapshot_index(self):
        """If the runner's current batch reorders the requests w.r.t. the
        draft-time snapshot, we must look up by req_id (NOT by position)."""
        # Snapshot order: ["a", "b"]; probs[0] = a's probs, probs[1] = b's probs.
        probs = torch.arange(2 * self.NUM_SPEC * self.VOCAB, dtype=torch.float32).reshape(2, self.NUM_SPEC, self.VOCAB)
        runner = self._build_runner(
            draft_probs=probs,
            draft_prob_req_ids=["a", "b"],
            # Current batch reverses the order.
            current_req_ids=["b", "a"],
        )
        spec_md = self._make_spec_metadata([1, 1])

        out = runner._get_spec_decode_draft_probs(spec_md)

        self.assertIsNotNone(out)
        self.assertEqual(out.shape, (2, self.VOCAB))
        # First output row corresponds to current_req_ids[0]="b" → probs[1, :1].
        self.assertTrue(torch.equal(out[:1], probs[1, :1]))
        # Second output row corresponds to current_req_ids[1]="a" → probs[0, :1].
        self.assertTrue(torch.equal(out[1:], probs[0, :1]))


if __name__ == "__main__":
    unittest.main()
