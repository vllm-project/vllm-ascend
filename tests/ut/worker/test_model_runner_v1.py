import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerKVCache(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8_indexer = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.use_compress = False
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


class _DummyRequest:
    def __init__(self, token_id: int):
        self.token_id = token_id

    def get_token_id(self, _seq_len: int) -> int:
        return self.token_id


class TestNPUModelRunnerFusedMTP(unittest.TestCase):
    class _QueryStartLoc:
        def __init__(self, size: int):
            self.np = np.zeros(size, dtype=np.int32)
            self.cpu = torch.from_numpy(self.np)
            self.gpu = torch.zeros(size, dtype=torch.int32)

        def copy_to_gpu(self):
            self.gpu.copy_(self.cpu)

    def _build_min_tokens_processor(self) -> MinTokensLogitsProcessor:
        return MinTokensLogitsProcessor(
            SimpleNamespace(
                scheduler_config=SimpleNamespace(max_num_seqs=4),
            ),
            torch.device("cpu"),
            False,
        )

    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.broadcast_pp_output = False
        runner._fused_mtp_wrapper = SimpleNamespace(
            num_reqs_buf=torch.zeros(1, dtype=torch.int32),
            logits_indices_buf=torch.zeros(16, dtype=torch.int64),
            sample_logits_indices_buf=torch.zeros(16, dtype=torch.int64),
            bonus_row_indices_buf=torch.zeros(4, dtype=torch.int64),
            sample_idx_mapping_buf=torch.zeros(16, dtype=torch.int32),
            target_row_indices_buf=torch.zeros(16, dtype=torch.int64),
            num_sample_rows_buf=torch.zeros(1, dtype=torch.int32),
            num_target_rows_buf=torch.zeros(1, dtype=torch.int32),
            num_actual_tokens_buf=torch.zeros(1, dtype=torch.int32),
            target_logits_indices_buf=torch.full((4, 4), -1, dtype=torch.int64),
            spec_decode_token_ids_buf=torch.zeros((4, 4), dtype=torch.int64),
            draft_token_ids_flat_buf=torch.zeros(16, dtype=torch.int64),
            sampled_token_ids_buf=torch.full((4, 5), -1, dtype=torch.int32),
            draft_token_ids_buf=torch.zeros((4, 4), dtype=torch.int64),
            next_token_ids_buf=torch.zeros(4, dtype=torch.int64),
            valid_sampled_tokens_count_buf=torch.zeros(4, dtype=torch.int32),
            cu_num_draft_tokens_buf=torch.zeros(4, dtype=torch.int32),
            backup_next_token_ids_buf=torch.zeros(4, dtype=torch.int64),
            discarded_req_mask_buf=torch.zeros(4, dtype=torch.bool),
        )
        runner.drafter = MagicMock()
        runner.drafter.attn_layer_names = ["draft"]
        runner.drafter.slot_mapping_group = [torch.full((16,), -2, dtype=torch.int32)]
        runner._fused_mtp_graph_outputs_ready = True
        runner.sampler = SimpleNamespace(uses_seeded_gumbel=True)
        runner.speculative_config = SimpleNamespace(method="mtp", num_speculative_tokens=4)
        runner.input_batch = SimpleNamespace(
            num_reqs=3,
            req_ids=["r0", "r1", "r2"],
            sampling_metadata=SimpleNamespace(
                all_greedy=True,
                all_random=False,
                no_penalties=True,
                max_num_logprobs=None,
                allowed_token_ids_mask=None,
                bad_words_token_ids={},
                seeds=torch.tensor([3, 5, 7], dtype=torch.int64),
                logitsprocs=SimpleNamespace(
                    argmax_invariant=[],
                    non_argmax_invariant=[self._build_min_tokens_processor()],
                ),
            ),
        )
        runner.requests = {
            "r0": _DummyRequest(11),
            "r1": _DummyRequest(22),
            "r2": _DummyRequest(33),
        }
        runner.discard_request_indices = SimpleNamespace(gpu=torch.tensor([1], dtype=torch.int64))
        runner.num_discarded_requests = 1
        runner.seq_lens = SimpleNamespace(cpu=torch.tensor([5, 6, 7], dtype=torch.int32))
        return runner

    def _build_spec_decode_metadata(self):
        return SpecDecodeMetadata(
            draft_token_ids=torch.tensor([101, 202], dtype=torch.int32),
            num_draft_tokens=[1, 0, 1],
            cu_num_draft_tokens=torch.tensor([1, 1, 2], dtype=torch.int32),
            cu_num_sampled_tokens=torch.tensor([2, 3, 5], dtype=torch.int32),
            target_logits_indices=torch.tensor([0, 3], dtype=torch.int32),
            bonus_logits_indices=torch.tensor([1, 2, 4], dtype=torch.int32),
            logits_indices=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
        )

    def test_can_use_fused_mtp_draft_path_for_greedy_and_seeded_random(self):
        runner = self._build_runner()
        metadata = self._build_spec_decode_metadata()
        self.assertTrue(runner._can_use_fused_mtp_draft_path(None, metadata))
        self.assertFalse(runner._can_use_fused_mtp_draft_path(None, None))
        self.assertFalse(runner._can_use_fused_mtp_draft_path(object(), metadata))

        runner.input_batch.sampling_metadata.all_greedy = False
        runner.input_batch.sampling_metadata.all_random = True
        self.assertTrue(runner._can_use_fused_mtp_draft_path(None, metadata))
        runner.input_batch.sampling_metadata.all_greedy = True
        runner.input_batch.sampling_metadata.all_random = False
        runner.input_batch.sampling_metadata.no_penalties = False
        self.assertFalse(runner._can_use_fused_mtp_draft_path(None, metadata))

        runner.input_batch.sampling_metadata.no_penalties = True
        runner.input_batch.sampling_metadata.all_greedy = False
        runner.input_batch.sampling_metadata.all_random = True
        runner.input_batch.sampling_metadata.seeds = None
        self.assertFalse(runner._can_use_fused_mtp_draft_path(None, metadata))
        runner.input_batch.sampling_metadata.seeds = torch.tensor([3, 5, 7], dtype=torch.int64)
        runner.sampler.uses_seeded_gumbel = False
        self.assertTrue(runner._can_use_fused_mtp_draft_path(None, metadata))
        runner.sampler.uses_seeded_gumbel = True
        runner.input_batch.sampling_metadata.all_greedy = True
        runner.input_batch.sampling_metadata.all_random = False
        runner._fused_mtp_graph_outputs_ready = False
        self.assertFalse(runner._can_use_fused_mtp_draft_path(None, metadata))

    def test_update_fused_mtp_draft_graph_params_marks_draft_context(self):
        runner = self._build_runner()
        forward_context = SimpleNamespace(is_draft_model=False)
        draft_attn_metadatas = [{"draft": object()}]
        seen = []

        def _record_update(ctx, num_tokens, metadatas):
            seen.append((ctx.is_draft_model, num_tokens, metadatas))

        runner.drafter._update_full_graph_params.side_effect = _record_update

        runner._update_fused_mtp_draft_graph_params(
            forward_context,
            32,
            draft_attn_metadatas,
        )

        self.assertEqual(seen, [(True, 32, draft_attn_metadatas)])
        self.assertFalse(forward_context.is_draft_model)

    def test_update_fused_mtp_graph_params_for_replay_updates_before_replay(
        self,
    ):
        runner = self._build_runner()
        runner.use_sparse = True
        forward_context = SimpleNamespace(
            attn_metadata={},
            draft_attn_metadatas=None,
        )
        draft_attn_metadatas = [{"draft": object()}]
        calls = []

        def _inject(ctx, num_tokens):
            calls.append(("inject", num_tokens))
            ctx.draft_attn_metadatas = draft_attn_metadatas

        runner._inject_mtp_metadata_stubs = _inject
        runner._update_mtp_graph_params_sparse = MagicMock(side_effect=lambda *args: calls.append(("sparse", args[1])))
        runner._update_fused_mtp_draft_graph_params = MagicMock(
            side_effect=lambda *args: calls.append(("draft", args[1]))
        )

        runner._update_fused_mtp_graph_params_for_replay(
            forward_context,
            32,
            ["draft"],
        )

        self.assertEqual(
            calls,
            [
                ("inject", 32),
                ("sparse", 32),
                ("draft", 32),
            ],
        )
        runner._update_mtp_graph_params_sparse.assert_called_once_with(
            forward_context, 32, draft_attn_metadatas, ["draft"]
        )
        runner._update_fused_mtp_draft_graph_params.assert_called_once_with(forward_context, 32, draft_attn_metadatas)

    def test_can_use_fused_mtp_draft_path_blocks_active_min_tokens(self):
        runner = self._build_runner()
        metadata = SpecDecodeMetadata(
            draft_token_ids=torch.tensor([101], dtype=torch.int32),
            num_draft_tokens=[1, 0, 0],
            cu_num_draft_tokens=torch.tensor([1, 1, 1], dtype=torch.int32),
            cu_num_sampled_tokens=torch.tensor([2, 3, 4], dtype=torch.int32),
            target_logits_indices=torch.tensor([0], dtype=torch.int32),
            bonus_logits_indices=torch.tensor([1, 2, 3], dtype=torch.int32),
            logits_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )

        min_tokens = self._build_min_tokens_processor()
        min_tokens.min_toks = {
            0: (2, [], {42}),
        }
        runner.input_batch.sampling_metadata.logitsprocs = SimpleNamespace(
            non_argmax_invariant=[min_tokens],
        )

        self.assertFalse(runner._can_use_fused_mtp_draft_path(None, metadata))

    def test_can_use_fused_mtp_draft_path_blocks_logprobs_request(self):
        runner = self._build_runner()
        metadata = SpecDecodeMetadata(
            draft_token_ids=torch.tensor([101], dtype=torch.int32),
            num_draft_tokens=[1, 0, 0],
            cu_num_draft_tokens=torch.tensor([1, 1, 1], dtype=torch.int32),
            cu_num_sampled_tokens=torch.tensor([2, 3, 4], dtype=torch.int32),
            target_logits_indices=torch.tensor([0], dtype=torch.int32),
            bonus_logits_indices=torch.tensor([1, 2, 3], dtype=torch.int32),
            logits_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        )
        runner.input_batch.sampling_metadata.max_num_logprobs = 4
        self.assertFalse(runner._can_use_fused_mtp_draft_path(None, metadata))

    def test_build_fused_mtp_sampler_output_uses_graph_tokens(self):
        runner = self._build_runner()
        runner._fused_mtp_wrapper.sampled_token_ids_buf[:3, :2] = torch.tensor(
            [[11, -1], [22, 23], [33, -1]],
            dtype=torch.int32,
        )

        sampler_output = runner._build_fused_mtp_sampler_output()

        self.assertTrue(
            torch.equal(
                sampler_output.sampled_token_ids[:, :2],
                torch.tensor([[11, -1], [-1, -1], [33, -1]], dtype=torch.int32),
            )
        )
        self.assertIsNone(sampler_output.logprobs_tensors)

    def test_should_defer_logits_for_fused_mtp_graph_path(self):
        runner = self._build_runner()
        metadata = self._build_spec_decode_metadata()

        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(runner._should_defer_fused_mtp_logits(metadata))
            runner.input_batch.sampling_metadata.max_num_logprobs = 1
            self.assertFalse(runner._should_defer_fused_mtp_logits(metadata))
            runner.input_batch.sampling_metadata.max_num_logprobs = None
            runner._fused_mtp_graph_outputs_ready = False
            self.assertFalse(runner._should_defer_fused_mtp_logits(metadata))

    def test_compute_deferred_logits_only_when_missing(self):
        runner = self._build_runner()
        existing_logits = torch.ones(2, 3)
        sample_hidden_states = torch.ones(2, 4)
        model = SimpleNamespace(compute_logits=MagicMock(return_value=torch.zeros(2, 3)))
        runner.model = model

        self.assertIs(
            runner._compute_deferred_logits(existing_logits, sample_hidden_states),
            existing_logits,
        )
        model.compute_logits.assert_not_called()

        computed = runner._compute_deferred_logits(None, sample_hidden_states)

        self.assertTrue(torch.equal(computed, torch.zeros(2, 3)))
        model.compute_logits.assert_called_once_with(sample_hidden_states)

    def test_prepare_fused_mtp_runtime_buffers_maps_spec_decode_inputs(self):
        runner = self._build_runner()
        metadata = SpecDecodeMetadata(
            draft_token_ids=torch.tensor([101, 202], dtype=torch.int32),
            num_draft_tokens=[1, 0, 1],
            cu_num_draft_tokens=torch.tensor([1, 1, 2], dtype=torch.int32),
            cu_num_sampled_tokens=torch.tensor([2, 3, 5], dtype=torch.int32),
            target_logits_indices=torch.tensor([0, 3], dtype=torch.int32),
            bonus_logits_indices=torch.tensor([1, 2, 4], dtype=torch.int32),
            logits_indices=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
        )

        runner._prepare_fused_mtp_runtime_buffers(
            logits_indices=torch.tensor([10, 11, 20, 30, 31], dtype=torch.int64),
            spec_decode_metadata=metadata,
            spec_decode_common_attn_metadata=SimpleNamespace(
                seq_lens_cpu=torch.tensor([5, 6, 7], dtype=torch.int32),
            ),
        )

        wrapper = runner._fused_mtp_wrapper
        self.assertEqual(int(wrapper.num_reqs_buf[0].item()), 3)
        self.assertTrue(
            torch.equal(
                wrapper.logits_indices_buf[:3],
                torch.tensor([11, 20, 31], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                wrapper.target_logits_indices_buf[:3, :2],
                torch.tensor([[10, -1], [-1, -1], [30, -1]], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                wrapper.bonus_row_indices_buf[:3],
                torch.tensor([1, 2, 4], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                wrapper.spec_decode_token_ids_buf[:3, :2],
                torch.tensor([[101, 0], [0, 0], [202, 0]], dtype=torch.int64),
            )
        )
        self.assertEqual(int(wrapper.num_sample_rows_buf[0].item()), 5)
        self.assertEqual(int(wrapper.num_target_rows_buf[0].item()), 2)
        self.assertTrue(
            torch.equal(
                wrapper.backup_next_token_ids_buf[:3],
                torch.tensor([11, 22, 33], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                wrapper.discarded_req_mask_buf[:3],
                torch.tensor([False, True, False]),
            )
        )

    def test_prepare_fused_mtp_runtime_buffers_updates_draft_slot_mapping(self):
        runner = self._build_runner()
        metadata = self._build_spec_decode_metadata()

        runner._prepare_fused_mtp_runtime_buffers(
            logits_indices=torch.tensor([10, 11, 20, 30, 31], dtype=torch.int64),
            spec_decode_metadata=metadata,
            spec_decode_common_attn_metadata=SimpleNamespace(
                seq_lens_cpu=torch.tensor([5, 6, 7], dtype=torch.int32),
                slot_mapping=torch.tensor([100, 101, 200, 300, 301], dtype=torch.int64),
            ),
        )

        self.assertTrue(
            torch.equal(
                runner.drafter.slot_mapping_group[0][:8],
                torch.tensor([100, 101, 200, 300, 301, 0, 0, 0], dtype=torch.int32),
            )
        )

    def test_prepare_fused_mtp_runtime_buffers_uses_real_token_count(self):
        runner = self._build_runner()
        runner.num_spec_tokens = 1
        metadata = self._build_spec_decode_metadata()

        runner._prepare_fused_mtp_runtime_buffers(
            logits_indices=torch.tensor([10, 11, 20, 30, 31], dtype=torch.int64),
            spec_decode_metadata=metadata,
            spec_decode_common_attn_metadata=SimpleNamespace(
                seq_lens_cpu=torch.tensor([5, 6, 7], dtype=torch.int32),
                num_actual_tokens=8,
                slot_mapping=torch.tensor(
                    [100, 101, 200, 300, 301, -1, -1, -1],
                    dtype=torch.int64,
                ),
            ),
            num_tokens_padded=8,
        )

        wrapper = runner._fused_mtp_wrapper
        self.assertEqual(int(wrapper.num_actual_tokens_buf[0].item()), 5)
        self.assertTrue(
            torch.equal(
                runner.drafter.slot_mapping_group[0][:8],
                torch.tensor([100, 101, 200, 300, 301, 0, 0, 0], dtype=torch.int32),
            )
        )

    def test_prepare_fused_mtp_runtime_buffers_fills_padded_logits_indices(self):
        runner = self._build_runner()
        runner.num_spec_tokens = 1
        runner.input_batch.num_reqs = 1
        runner.input_batch.req_ids = ["r0"]
        runner.requests = {"r0": _DummyRequest(11)}
        runner.num_discarded_requests = 0
        metadata = SpecDecodeMetadata(
            draft_token_ids=torch.tensor([101], dtype=torch.int32),
            num_draft_tokens=[1],
            cu_num_draft_tokens=torch.tensor([1], dtype=torch.int32),
            cu_num_sampled_tokens=torch.tensor([2], dtype=torch.int32),
            target_logits_indices=torch.tensor([0], dtype=torch.int32),
            bonus_logits_indices=torch.tensor([1], dtype=torch.int32),
            logits_indices=torch.tensor([0, 1], dtype=torch.int32),
        )

        runner._prepare_fused_mtp_runtime_buffers(
            logits_indices=torch.tensor([0, 1], dtype=torch.int64),
            spec_decode_metadata=metadata,
            spec_decode_common_attn_metadata=SimpleNamespace(
                seq_lens_cpu=torch.tensor([5], dtype=torch.int32),
                num_actual_tokens=2,
            ),
            num_tokens_padded=8,
        )

        wrapper = runner._fused_mtp_wrapper
        self.assertTrue(
            torch.equal(
                wrapper.logits_indices_buf[:4],
                torch.tensor([1, 0, 0, 0], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                wrapper.sample_logits_indices_buf[:8],
                torch.tensor([0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                wrapper.sample_idx_mapping_buf[:8],
                torch.tensor([0, 0, 15, 15, 15, 15, 15, 15], dtype=torch.int32),
            )
        )
        self.assertEqual(int(wrapper.num_actual_tokens_buf[0].item()), 2)

    def test_prepare_fused_mtp_runtime_buffers_skips_oversized_padding(self):
        runner = self._build_runner()
        runner.num_spec_tokens = 1
        runner.input_batch.num_reqs = 1
        runner.input_batch.req_ids = ["r0"]
        runner.requests = {"r0": _DummyRequest(11)}
        runner.num_discarded_requests = 0
        metadata = SpecDecodeMetadata(
            draft_token_ids=torch.tensor([101], dtype=torch.int32),
            num_draft_tokens=[1],
            cu_num_draft_tokens=torch.tensor([1], dtype=torch.int32),
            cu_num_sampled_tokens=torch.tensor([2], dtype=torch.int32),
            target_logits_indices=torch.tensor([0], dtype=torch.int32),
            bonus_logits_indices=torch.tensor([1], dtype=torch.int32),
            logits_indices=torch.tensor([0, 1], dtype=torch.int32),
        )

        runner._prepare_fused_mtp_runtime_buffers(
            logits_indices=torch.tensor([0, 1], dtype=torch.int64),
            spec_decode_metadata=metadata,
            spec_decode_common_attn_metadata=SimpleNamespace(
                seq_lens_cpu=torch.tensor([5], dtype=torch.int32),
                num_actual_tokens=2,
            ),
            num_tokens_padded=1032,
        )

        wrapper = runner._fused_mtp_wrapper
        self.assertTrue(
            torch.equal(
                wrapper.logits_indices_buf[:4],
                torch.tensor([1, 0, 0, 0], dtype=torch.int64),
            )
        )

    def test_prepare_fused_mtp_capture_buffers_seeds_full_batch(self):
        runner = self._build_runner()
        runner.num_spec_tokens = 1

        runner._prepare_fused_mtp_capture_buffers(num_tokens_padded=8)

        wrapper = runner._fused_mtp_wrapper
        self.assertEqual(int(wrapper.num_reqs_buf[0].item()), 4)
        self.assertEqual(int(wrapper.num_actual_tokens_buf[0].item()), 8)
        self.assertTrue(
            torch.equal(
                wrapper.logits_indices_buf[:4],
                torch.tensor([1, 3, 5, 7], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                wrapper.sample_logits_indices_buf[:8],
                torch.arange(8, dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                wrapper.sample_idx_mapping_buf[:8],
                torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                wrapper.target_row_indices_buf[:4],
                torch.tensor([0, 2, 4, 6], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                wrapper.cu_num_draft_tokens_buf[:4],
                torch.tensor([1, 2, 3, 4], dtype=torch.int32),
            )
        )

    def test_prepare_fused_mtp_capture_buffers_clears_stale_tail(self):
        runner = self._build_runner()
        runner.num_spec_tokens = 1
        wrapper = runner._fused_mtp_wrapper
        wrapper.sample_logits_indices_buf.fill_(7)
        wrapper.sample_idx_mapping_buf.zero_()
        wrapper.logits_indices_buf.fill_(7)
        wrapper.bonus_row_indices_buf.fill_(7)
        wrapper.target_row_indices_buf.fill_(7)

        runner._prepare_fused_mtp_capture_buffers(num_tokens_padded=4)

        self.assertTrue(
            torch.equal(
                wrapper.sample_logits_indices_buf[:8],
                torch.tensor([0, 1, 2, 3, 0, 0, 0, 0], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                wrapper.sample_idx_mapping_buf[:8],
                torch.tensor([0, 0, 1, 1, 15, 15, 15, 15], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                wrapper.logits_indices_buf[:4],
                torch.tensor([1, 3, 0, 0], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                wrapper.bonus_row_indices_buf,
                torch.tensor([1, 3, 0, 0], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                wrapper.target_row_indices_buf[:4],
                torch.tensor([0, 2, 0, 0], dtype=torch.int64),
            )
        )

    def test_fused_mtp_graph_batch_accepts_graph_padded_partial_decode(self):
        runner = self._build_runner()
        runner.num_spec_tokens = 1

        self.assertTrue(
            runner._is_fused_mtp_uniform_decode_graph_batch(BatchDescriptor(num_tokens=8, num_reqs=4, uniform=True))
        )
        runner.input_batch.num_reqs = 1
        self.assertTrue(
            runner._is_fused_mtp_uniform_decode_graph_batch(BatchDescriptor(num_tokens=8, num_reqs=1, uniform=False))
        )
        runner.input_batch.num_reqs = 5
        self.assertFalse(
            runner._is_fused_mtp_uniform_decode_graph_batch(BatchDescriptor(num_tokens=8, num_reqs=4, uniform=False))
        )
        self.assertFalse(
            runner._is_fused_mtp_uniform_decode_graph_batch(BatchDescriptor(num_tokens=1032, num_reqs=4, uniform=True))
        )

    def test_fused_mtp_partial_decode_uses_full_graph_token_shape(self):
        runner = self._build_runner()
        runner.num_spec_tokens = 1
        runner.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=8))
        runner.compilation_config = SimpleNamespace(max_cudagraph_capture_size=32)

        with patch.dict(
            os.environ,
            {"VLLM_ASCEND_ENABLE_FUSED_MTP_FULL_GRAPH": "1"},
            clear=False,
        ):
            self.assertEqual(
                runner._get_fused_mtp_partial_decode_graph_tokens(
                    num_tokens=31,
                    num_reqs=16,
                    max_num_scheduled_tokens=2,
                    is_all_decode=True,
                    force_uniform_decode=None,
                ),
                32,
            )
            self.assertIsNone(
                runner._get_fused_mtp_partial_decode_graph_tokens(
                    num_tokens=32,
                    num_reqs=16,
                    max_num_scheduled_tokens=2,
                    is_all_decode=True,
                    force_uniform_decode=None,
                )
            )
            self.assertIsNone(
                runner._get_fused_mtp_partial_decode_graph_tokens(
                    num_tokens=31,
                    num_reqs=16,
                    max_num_scheduled_tokens=2,
                    is_all_decode=False,
                    force_uniform_decode=None,
                )
            )

    def test_determine_padding_promotes_partial_decode_to_full_graph(self):
        runner = self._build_runner()
        runner.num_spec_tokens = 1
        runner.input_batch.num_reqs = 15
        runner.input_batch.num_computed_tokens_cpu = np.ones(16, dtype=np.int32)
        runner.input_batch.lora_id_to_lora_request = {}
        runner.uniform_decode_query_len = 2
        runner.model_config = SimpleNamespace(is_encoder_decoder=False)
        runner.compilation_config = SimpleNamespace(max_cudagraph_capture_size=32)
        runner.vllm_config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                data_parallel_size=1,
                tensor_parallel_size=8,
            ),
            observability_config=SimpleNamespace(cudagraph_metrics=False),
        )
        runner._pad_for_sequence_parallelism = lambda num_tokens: num_tokens
        calls = []

        class _Dispatcher:
            def dispatch(self, **kwargs):
                calls.append(kwargs)
                return (
                    CUDAGraphMode.FULL,
                    BatchDescriptor(num_tokens=32, num_reqs=16, uniform=True),
                )

        runner.cudagraph_dispatcher = _Dispatcher()

        with patch.dict(
            os.environ,
            {"VLLM_ASCEND_ENABLE_FUSED_MTP_FULL_GRAPH": "1"},
            clear=False,
        ):
            cudagraph_mode, batch_descriptor, *_ = runner._determine_batch_execution_and_padding(
                num_tokens=29,
                num_reqs=15,
                num_scheduled_tokens_np=np.full(15, 2, dtype=np.int32),
                max_num_scheduled_tokens=2,
                use_cascade_attn=False,
            )

        self.assertEqual(cudagraph_mode, CUDAGraphMode.FULL)
        self.assertEqual(batch_descriptor.num_tokens, 32)
        self.assertEqual(batch_descriptor.num_reqs, 16)
        self.assertTrue(batch_descriptor.uniform)
        self.assertEqual(calls[0]["num_tokens"], 30)
        self.assertTrue(calls[0]["uniform_decode"])

    def test_pad_query_start_loc_extends_partial_full_graph_tail(self):
        runner = self._build_runner()
        runner.query_start_loc = self._QueryStartLoc(18)
        runner.query_start_loc.np[:17] = np.arange(17, dtype=np.int32) * 2
        runner.query_start_loc.np[16] = 31
        runner.arange_np = np.arange(18, dtype=np.int32)
        runner.uniform_decode_query_len = 2
        runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)

        num_reqs_padded = runner._pad_query_start_loc_for_fia(
            num_tokens_padded=32,
            num_reqs_padded=16,
            num_reqs=16,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            batch_desc_num_reqs=16,
        )

        self.assertEqual(num_reqs_padded, 16)
        self.assertEqual(int(runner.query_start_loc.np[16]), 32)
        self.assertEqual(int(runner.query_start_loc.gpu[16].item()), 32)

    def test_pad_query_start_loc_extends_partial_padded_req_tail(self):
        runner = self._build_runner()
        runner.query_start_loc = self._QueryStartLoc(18)
        runner.query_start_loc.np[:16] = np.arange(16, dtype=np.int32) * 2
        runner.query_start_loc.np[15] = 29
        runner.arange_np = np.arange(18, dtype=np.int32)
        runner.uniform_decode_query_len = 2
        runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)

        num_reqs_padded = runner._pad_query_start_loc_for_fia(
            num_tokens_padded=32,
            num_reqs_padded=16,
            num_reqs=15,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            batch_desc_num_reqs=16,
        )

        self.assertEqual(num_reqs_padded, 16)
        self.assertEqual(int(runner.query_start_loc.np[15]), 30)
        self.assertEqual(int(runner.query_start_loc.np[16]), 32)
        self.assertEqual(int(runner.query_start_loc.gpu[15].item()), 30)
        self.assertEqual(int(runner.query_start_loc.gpu[16].item()), 32)

    def test_fused_mtp_full_graph_defaults_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(NPUModelRunner._fused_mtp_full_graph_enabled())

    def test_fused_mtp_full_graph_can_be_enabled(self):
        for value in ("1", "true", "YES", "on"):
            with (
                self.subTest(value=value),
                patch.dict(
                    os.environ,
                    {"VLLM_ASCEND_ENABLE_FUSED_MTP_FULL_GRAPH": value},
                    clear=True,
                ),
            ):
                self.assertTrue(NPUModelRunner._fused_mtp_full_graph_enabled())

    def test_fused_mtp_graph_batch_default_allows_batch16(self):
        runner = self._build_runner()
        runner.num_spec_tokens = 1
        wrapper = runner._fused_mtp_wrapper
        wrapper.bonus_row_indices_buf = torch.zeros(16, dtype=torch.int64)
        wrapper.next_token_ids_buf = torch.zeros(16, dtype=torch.int64)
        wrapper.cu_num_draft_tokens_buf = torch.zeros(16, dtype=torch.int32)
        wrapper.sampled_token_ids_buf = torch.full((16, 5), -1, dtype=torch.int32)

        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(
                runner._is_fused_mtp_uniform_decode_graph_batch(
                    BatchDescriptor(num_tokens=16, num_reqs=8, uniform=True)
                )
            )
            self.assertTrue(
                runner._is_fused_mtp_uniform_decode_graph_batch(
                    BatchDescriptor(num_tokens=32, num_reqs=16, uniform=True)
                )
            )

    def test_pad_fused_mtp_graph_inputs_uses_stable_dummy_values(self):
        runner = self._build_runner()
        runner.enable_prompt_embeds = False
        runner.uses_mrope = False
        runner.uses_xdrope_dim = 0
        runner.use_compress = True
        runner.input_ids = SimpleNamespace(
            cpu=torch.arange(8, dtype=torch.int64),
            gpu=torch.arange(8, dtype=torch.int64),
        )
        positions_cpu = torch.arange(8, dtype=torch.int32)
        runner.positions = SimpleNamespace(
            cpu=positions_cpu,
            gpu=positions_cpu.clone(),
            np=positions_cpu.numpy(),
        )

        runner._pad_fused_mtp_graph_inputs(
            num_actual_tokens=6,
            num_tokens_padded=8,
        )

        self.assertTrue(
            torch.equal(
                runner.input_ids.cpu,
                torch.tensor([0, 1, 2, 3, 4, 5, 0, 0], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                runner.input_ids.gpu,
                torch.tensor([0, 1, 2, 3, 4, 5, 0, 0], dtype=torch.int64),
            )
        )
        self.assertTrue(
            torch.equal(
                runner.positions.cpu,
                torch.tensor([0, 1, 2, 3, 4, 5, 127, 127], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                runner.positions.gpu,
                torch.tensor([0, 1, 2, 3, 4, 5, 127, 127], dtype=torch.int32),
            )
        )

    def test_prepare_fused_mtp_runtime_buffers_maps_multi_step_spec_decode_inputs(self):
        runner = self._build_runner()
        metadata = SpecDecodeMetadata(
            draft_token_ids=torch.tensor([101, 102, 201, 301, 302, 303], dtype=torch.int32),
            num_draft_tokens=[2, 1, 3],
            cu_num_draft_tokens=torch.tensor([2, 3, 6], dtype=torch.int32),
            cu_num_sampled_tokens=torch.tensor([3, 5, 9], dtype=torch.int32),
            target_logits_indices=torch.tensor([0, 1, 3, 5, 6, 7], dtype=torch.int32),
            bonus_logits_indices=torch.tensor([2, 4, 8], dtype=torch.int32),
            logits_indices=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int32),
        )

        runner._prepare_fused_mtp_runtime_buffers(
            logits_indices=torch.tensor([10, 11, 12, 20, 21, 30, 31, 32, 33], dtype=torch.int64),
            spec_decode_metadata=metadata,
            spec_decode_common_attn_metadata=SimpleNamespace(
                seq_lens_cpu=torch.tensor([5, 6, 7], dtype=torch.int32),
            ),
        )

        wrapper = runner._fused_mtp_wrapper
        self.assertTrue(
            torch.equal(
                wrapper.target_logits_indices_buf[:3, :4],
                torch.tensor(
                    [[10, 11, -1, -1], [20, -1, -1, -1], [30, 31, 32, -1]],
                    dtype=torch.int64,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                wrapper.spec_decode_token_ids_buf[:3, :4],
                torch.tensor(
                    [[101, 102, 0, 0], [201, 0, 0, 0], [301, 302, 303, 0]],
                    dtype=torch.int64,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                wrapper.bonus_row_indices_buf[:3],
                torch.tensor([2, 4, 8], dtype=torch.int64),
            )
        )
        self.assertEqual(int(wrapper.num_sample_rows_buf[0].item()), 9)
        self.assertEqual(int(wrapper.num_target_rows_buf[0].item()), 6)

    def test_build_and_inject_mtp_attn_metadata_uses_real_draft_metadata(self):
        runner = self._build_runner()
        draft_step0 = {"draft": SimpleNamespace(decode=SimpleNamespace(marker="step0"))}
        draft_step1 = {"draft": SimpleNamespace(decode=SimpleNamespace(marker="step1"))}
        runner.drafter.build_graph_capture_attn_metadata.return_value = [draft_step0, draft_step1]

        forward_context = SimpleNamespace(attn_metadata={"main": SimpleNamespace()}, draft_attn_metadatas=None)

        runner._build_and_inject_mtp_attn_metadata(
            forward_context,
            num_tokens=8,
            num_reqs=4,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
        )

        runner.drafter.build_graph_capture_attn_metadata.assert_called_once_with(
            num_tokens=8,
            num_reqs=4,
            aclgraph_runtime_mode=CUDAGraphMode.FULL,
        )
        self.assertIs(forward_context.attn_metadata["draft"], draft_step0["draft"])
        self.assertEqual(forward_context.draft_attn_metadatas, [draft_step0, draft_step1])

    def test_inject_mtp_metadata_stubs_uses_all_spec_steps(self):
        runner = self._build_runner()
        runner.drafter.num_speculative_tokens = 3
        runner.drafter.attn_layer_names = ["draft"]
        runner.query_start_loc = SimpleNamespace(
            cpu=torch.tensor([0, 2, 4, 7], dtype=torch.int32),
        )
        runner.input_batch.block_table = [
            SimpleNamespace(
                get_device_tensor=MagicMock(return_value=torch.zeros((3, 1), dtype=torch.int32)),
            )
        ]
        forward_context = SimpleNamespace(
            attn_metadata={"main": SimpleNamespace()},
            draft_attn_metadatas=None,
        )

        runner._inject_mtp_metadata_stubs(forward_context)

        self.assertEqual(len(forward_context.draft_attn_metadatas), 3)
        self.assertIn("draft", forward_context.attn_metadata)
        for per_step_metadata in forward_context.draft_attn_metadatas:
            decode_metadata = per_step_metadata["draft"].decode
            self.assertEqual(decode_metadata.actual_seq_lengths_q, [2, 4, 7])
            self.assertEqual(decode_metadata.seq_lens_list, [5, 6, 7])

    def test_inject_mtp_metadata_prefers_builder_metadata_with_graph_padding(self):
        runner = self._build_runner()
        runner.input_batch.num_reqs = 1
        runner.drafter.decode_threshold = 2
        runner.drafter.attn_layer_names = ["draft"]
        draft_metadata = {
            "draft": SimpleNamespace(
                decode=SimpleNamespace(
                    seq_lens_list=[1030, 0, 0, 0],
                )
            ),
        }
        runner.drafter.build_graph_capture_attn_metadata.return_value = [draft_metadata]
        runner.query_start_loc = SimpleNamespace(
            cpu=torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32),
        )
        forward_context = SimpleNamespace(
            attn_metadata={"main": SimpleNamespace()},
            draft_attn_metadatas=None,
        )

        runner._inject_mtp_metadata_stubs(forward_context, num_tokens=8)

        runner.drafter.build_graph_capture_attn_metadata.assert_called_once_with(
            num_tokens=8,
            num_reqs=1,
            aclgraph_runtime_mode=CUDAGraphMode.FULL,
        )
        decode_metadata = forward_context.draft_attn_metadatas[0]["draft"].decode
        self.assertEqual(decode_metadata.actual_seq_lengths_q, [2])
        self.assertIs(forward_context.attn_metadata["draft"], draft_metadata["draft"])

    def test_build_fused_mtp_metadata_uses_graph_padded_tokens(self):
        runner = self._build_runner()
        runner.drafter.num_speculative_tokens = 1
        runner.query_start_loc = self._QueryStartLoc(8)
        runner.arange_np = np.arange(8, dtype=np.int32)
        runner.uniform_decode_query_len = 2
        runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL)
        runner.seq_lens = SimpleNamespace(
            cpu=torch.tensor([5, 6, 7, 0], dtype=torch.int32),
            gpu=torch.tensor([5, 6, 7, 0], dtype=torch.int32),
        )
        runner.get_model = MagicMock(return_value=object())

        def pad_tensor(tensor, desired_size):
            pad_size = desired_size - tensor.shape[0]
            if pad_size <= 0:
                return tensor[:desired_size]
            padding = torch.zeros(
                (pad_size,) + tensor.shape[1:],
                dtype=tensor.dtype,
                device=tensor.device,
            )
            return torch.cat([tensor, padding], dim=0)

        captured = {}

        def build_metadata(_prefix_len, common_attn_metadata, _model, **_kwargs):
            captured["common"] = common_attn_metadata
            return SimpleNamespace(decode=SimpleNamespace(marker="runtime"))

        builder = SimpleNamespace(build=MagicMock(side_effect=build_metadata))
        runner.drafter.draft_attn_groups = [
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(block_size=128),
                get_metadata_builder=MagicMock(return_value=builder),
            )
        ]
        runner.drafter._pad_tensor.side_effect = pad_tensor
        runner.drafter._freeze_draft_step_attn_metadata.side_effect = lambda metadata: metadata
        runner._fused_mtp_runtime_common_attn_metadata = SimpleNamespace(
            num_reqs=3,
            num_actual_tokens=6,
            num_computed_tokens_cpu=torch.tensor([3, 4, 5], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 2, 4, 6], dtype=torch.int32),
            block_table_tensor=torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int32),
            slot_mapping=torch.tensor([100, 101, 200, 201, 300, 301, -1, -1], dtype=torch.int64),
        )

        draft_metadata = runner._build_fused_mtp_builder_metadata(
            num_tokens=8,
            actual_num_reqs=3,
        )

        self.assertEqual(len(draft_metadata), 1)
        self.assertEqual(draft_metadata[0]["draft"].decode.marker, "runtime")
        common = captured["common"]
        self.assertEqual(common.num_actual_tokens, 8)
        self.assertEqual(common.num_input_tokens, 8)
        self.assertEqual(common.num_reqs, 4)
        self.assertTrue(
            torch.equal(
                common.query_start_loc_cpu,
                torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                common.block_table_tensor,
                torch.tensor(
                    [[10, 11], [20, 21], [30, 31], [10, 11]],
                    dtype=torch.int32,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                common.num_computed_tokens_cpu,
                torch.tensor([3, 4, 5, 3], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                runner.drafter.slot_mapping_group[0][:8],
                torch.tensor([100, 101, 200, 201, 300, 301, 0, 0], dtype=torch.int32),
            )
        )

    def test_build_fused_mtp_metadata_uses_safe_padding_slot(self):
        runner = self._build_runner()
        runner.input_batch.num_reqs = 1
        runner.drafter.num_speculative_tokens = 1
        runner.query_start_loc = self._QueryStartLoc(8)
        runner.arange_np = np.arange(8, dtype=np.int32)
        runner.uniform_decode_query_len = 2
        runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL)
        runner.seq_lens = SimpleNamespace(
            cpu=torch.tensor([5, 99, 98, 97], dtype=torch.int32),
            gpu=torch.tensor([5, 99, 98, 97], dtype=torch.int32),
        )
        runner.get_model = MagicMock(return_value=object())

        builder = SimpleNamespace(build=MagicMock(return_value=SimpleNamespace(decode=SimpleNamespace())))
        runner.drafter.draft_attn_groups = [
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(block_size=128),
                get_metadata_builder=MagicMock(return_value=builder),
            )
        ]
        runner.drafter._pad_tensor.side_effect = lambda tensor, desired_size: (
            torch.nn.functional.pad(tensor, (0, 0, 0, max(0, desired_size - tensor.shape[0])))
            if desired_size > tensor.shape[0]
            else tensor[:desired_size]
        )
        runner.drafter._freeze_draft_step_attn_metadata.side_effect = lambda metadata: metadata
        runner._fused_mtp_runtime_common_attn_metadata = SimpleNamespace(
            num_reqs=1,
            num_actual_tokens=2,
            query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
            block_table_tensor=torch.tensor([[10, 11]], dtype=torch.int32),
            slot_mapping=torch.tensor([100, 101], dtype=torch.int64),
        )

        runner._build_fused_mtp_builder_metadata(
            num_tokens=8,
            actual_num_reqs=1,
        )

        self.assertTrue(
            torch.equal(
                runner.drafter.slot_mapping_group[0][:8],
                torch.tensor([100, 101, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
            )
        )

    def test_build_fused_mtp_metadata_pads_to_graph_request_shape(self):
        runner = self._build_runner()
        runner.input_batch.num_reqs = 1
        runner.drafter.num_speculative_tokens = 1
        runner.query_start_loc = self._QueryStartLoc(8)
        runner.arange_np = np.arange(8, dtype=np.int32)
        runner.uniform_decode_query_len = 2
        runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL)
        runner.seq_lens = SimpleNamespace(
            cpu=torch.tensor([5, 0, 0, 0], dtype=torch.int32),
            gpu=torch.tensor([5, 0, 0, 0], dtype=torch.int32),
        )
        runner.get_model = MagicMock(return_value=object())

        captured = {}

        def build_metadata(_prefix_len, common_attn_metadata, _model, **_kwargs):
            captured["common"] = common_attn_metadata
            return SimpleNamespace(decode=SimpleNamespace(marker="runtime"))

        builder = SimpleNamespace(build=MagicMock(side_effect=build_metadata))
        runner.drafter.draft_attn_groups = [
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(block_size=128),
                get_metadata_builder=MagicMock(return_value=builder),
            )
        ]
        runner.drafter._pad_tensor.side_effect = lambda tensor, desired_size: (
            torch.nn.functional.pad(tensor, (0, 0, 0, max(0, desired_size - tensor.shape[0])))
            if desired_size > tensor.shape[0]
            else tensor[:desired_size]
        )
        runner.drafter._freeze_draft_step_attn_metadata.side_effect = lambda metadata: metadata
        runner._fused_mtp_runtime_common_attn_metadata = SimpleNamespace(
            num_reqs=1,
            num_actual_tokens=2,
            query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
            block_table_tensor=torch.tensor(
                [[10, 11], [99, 99], [98, 98], [97, 97]],
                dtype=torch.int32,
            ),
            slot_mapping=torch.tensor([100, 101, -1, -1, -1, -1, -1, -1], dtype=torch.int64),
        )

        draft_metadata = runner._build_fused_mtp_builder_metadata(
            num_tokens=8,
            actual_num_reqs=1,
        )

        self.assertEqual(len(draft_metadata), 1)
        common = captured["common"]
        self.assertEqual(common.num_reqs, 4)
        self.assertEqual(common.num_actual_tokens, 8)
        self.assertTrue(
            torch.equal(
                common.query_start_loc_cpu,
                torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                common.block_table_tensor,
                torch.tensor(
                    [[10, 11], [10, 11], [10, 11], [10, 11]],
                    dtype=torch.int32,
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                common.seq_lens,
                torch.tensor([5, 5, 5, 5], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                common.seq_lens_cpu,
                torch.tensor([5, 5, 5, 5], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                runner.drafter.slot_mapping_group[0][:8],
                torch.tensor([100, 101, 0, 0, 0, 0, 0, 0], dtype=torch.int32),
            )
        )

    def test_build_fused_mtp_metadata_handles_full_batch_token_padding(self):
        runner = self._build_runner()
        runner.input_batch.num_reqs = 4
        runner.drafter.num_speculative_tokens = 1
        runner.query_start_loc = self._QueryStartLoc(8)
        runner.arange_np = np.arange(8, dtype=np.int32)
        runner.uniform_decode_query_len = 2
        runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL)
        runner.seq_lens = SimpleNamespace(
            cpu=torch.tensor([5, 6, 7, 8], dtype=torch.int32),
            gpu=torch.tensor([5, 6, 7, 8], dtype=torch.int32),
        )
        runner.get_model = MagicMock(return_value=object())

        captured = {}

        def build_metadata(_prefix_len, common_attn_metadata, _model, **_kwargs):
            captured["common"] = common_attn_metadata
            return SimpleNamespace(decode=SimpleNamespace(marker="runtime"))

        builder = SimpleNamespace(build=MagicMock(side_effect=build_metadata))
        runner.drafter.draft_attn_groups = [
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(block_size=128),
                get_metadata_builder=MagicMock(return_value=builder),
            )
        ]
        runner.drafter._freeze_draft_step_attn_metadata.side_effect = lambda metadata: metadata
        runner._fused_mtp_runtime_common_attn_metadata = SimpleNamespace(
            num_reqs=4,
            num_actual_tokens=6,
            query_start_loc_cpu=torch.tensor([0, 2, 4, 5, 6], dtype=torch.int32),
            block_table_tensor=torch.tensor(
                [[10, 11], [20, 21], [30, 31], [40, 41]],
                dtype=torch.int32,
            ),
            slot_mapping=torch.tensor(
                [100, 101, 200, 201, 300, 400, -1, -1],
                dtype=torch.int64,
            ),
        )

        draft_metadata = runner._build_fused_mtp_builder_metadata(
            num_tokens=8,
            actual_num_reqs=4,
        )

        self.assertEqual(len(draft_metadata), 1)
        common = captured["common"]
        self.assertEqual(common.num_reqs, 4)
        self.assertEqual(common.num_actual_tokens, 8)
        self.assertTrue(
            torch.equal(
                common.query_start_loc_cpu,
                torch.tensor([0, 2, 4, 5, 8], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                runner.drafter.slot_mapping_group[0][:8],
                torch.tensor([100, 101, 200, 201, 300, 400, 0, 0], dtype=torch.int32),
            )
        )

    def test_pad_mtp_query_start_loc_uses_graph_request_shape(self):
        runner = self._build_runner()
        runner.num_spec_tokens = 1
        runner.query_start_loc = self._QueryStartLoc(8)
        runner.query_start_loc.cpu[:2] = torch.tensor([0, 2], dtype=torch.int32)
        runner.arange_np = np.arange(8, dtype=np.int32)
        runner.uniform_decode_query_len = 2

        num_reqs_padded = runner._pad_mtp_query_start_loc_for_graph(
            num_tokens_padded=8,
            actual_num_reqs=1,
        )

        self.assertEqual(num_reqs_padded, 4)
        self.assertTrue(
            torch.equal(
                runner.query_start_loc.cpu[:5],
                torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32),
            )
        )

        runner.query_start_loc = self._QueryStartLoc(8)
        runner.query_start_loc.cpu[:3] = torch.tensor([0, 2, 4], dtype=torch.int32)
        num_reqs_padded = runner._pad_mtp_query_start_loc_for_graph(
            num_tokens_padded=8,
            actual_num_reqs=2,
        )

        self.assertEqual(num_reqs_padded, 4)
        self.assertTrue(
            torch.equal(
                runner.query_start_loc.cpu[:5],
                torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32),
            )
        )

        runner.query_start_loc = self._QueryStartLoc(8)
        runner.query_start_loc.cpu[:2] = torch.tensor([0, 2], dtype=torch.int32)
        num_reqs_padded = runner._pad_mtp_query_start_loc_for_graph(
            num_tokens_padded=2,
            actual_num_reqs=1,
        )

        self.assertEqual(num_reqs_padded, 1)
        self.assertTrue(
            torch.equal(
                runner.query_start_loc.cpu[:2],
                torch.tensor([0, 2], dtype=torch.int32),
            )
        )


if __name__ == "__main__":
    unittest.main()
