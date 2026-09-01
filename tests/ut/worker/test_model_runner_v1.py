import unittest
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.sampling_params import SamplingParams
from vllm.v1.attention.backends.utils import reorder_batch_to_split_decodes_and_prefills
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_ascend.attention.mla_v1 import AscendMLABackend
from vllm_ascend.attention.utils import get_sfa_qsfa_packed_head_dim
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.utils import AscendDeviceType
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestDummyRunSlotInvalidation(unittest.TestCase):
    def test_backend_metadata_sees_invalidated_dummy_slots(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.uniform_decode_query_len = 1
        runner.scheduler_config = SimpleNamespace(max_num_batched_tokens=8, max_num_seqs=8)
        runner.dynamic_eplb = False
        # use_dcp is a read-only property derived from dcp_size.
        runner.dcp_size = 1
        runner.speculative_config = None
        runner.use_compress = True
        runner._has_gdn = False

        runner._determine_batch_execution_and_padding = MagicMock(
            return_value=(CUDAGraphMode.NONE, SimpleNamespace(num_tokens=1, num_reqs=1), None, None, None)
        )
        runner._should_build_dummy_attn_metadata = MagicMock(return_value=True)
        runner.synchronize_input_prep = MagicMock(return_value=nullcontext())
        runner._get_cumsum_and_arange = MagicMock(return_value=np.array([1], dtype=np.int32))
        runner._pad_query_start_loc_for_fia = MagicMock(return_value=1)

        runner.optimistic_seq_lens_cpu = torch.zeros(8, dtype=torch.int32)
        runner.seq_lens = MagicMock()
        runner.query_pos = SimpleNamespace(np=np.zeros(8, dtype=np.int32))
        runner.query_start_loc = SimpleNamespace(np=np.zeros(9, dtype=np.int32), copy_to_gpu=MagicMock())
        runner.positions = MagicMock()
        runner._dsa_positions_cpu_buf = MagicMock()

        slot_mappings = [torch.tensor([3]), torch.tensor([7])]
        block_tables = MagicMock()
        block_tables.__getitem__.side_effect = lambda index: SimpleNamespace(
            slot_mapping=SimpleNamespace(gpu=slot_mappings[index])
        )
        runner.input_batch = SimpleNamespace(block_table=block_tables)
        runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[object(), object()])

        def check_slots_before_build(**_kwargs):
            for slot_mapping in slot_mappings:
                torch.testing.assert_close(slot_mapping, torch.full_like(slot_mapping, -1))
            raise RuntimeError("metadata checked")

        runner._build_attention_metadata = check_slots_before_build

        with self.assertRaisesRegex(RuntimeError, "metadata checked"):
            runner._dummy_run(1)


class TestDeviceMetadataFullGraphWait(unittest.TestCase):
    def test_waits_all_only_for_effective_full_mode(self):
        for mode, should_wait in (
            (CUDAGraphMode.FULL, True),
            (CUDAGraphMode.PIECEWISE, False),
            (CUDAGraphMode.NONE, False),
        ):
            with self.subTest(mode=mode):
                runner = NPUModelRunner.__new__(NPUModelRunner)
                executor = MagicMock(submission_in_flight=True)
                runner.device_metadata_executor = executor

                active_executor = runner._prepare_device_metadata_for_forward(mode)

                self.assertIs(active_executor, executor)
                self.assertEqual(executor.wait_all.call_count, int(should_wait))

    def test_ignores_executor_without_active_submission(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        executor = MagicMock(submission_in_flight=False)
        runner.device_metadata_executor = executor

        active_executor = runner._prepare_device_metadata_for_forward(CUDAGraphMode.FULL)

        self.assertIsNone(active_executor)
        executor.wait_all.assert_not_called()

    def test_dummy_full_waits_before_graph_forward(self):
        from contextlib import contextmanager, nullcontext

        events = []
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.uniform_decode_query_len = 1
        runner.scheduler_config = SimpleNamespace(max_num_batched_tokens=4, max_num_seqs=4)
        runner.dynamic_eplb = False
        runner.dcp_size = 1
        runner._determine_batch_execution_and_padding = MagicMock(
            return_value=(
                CUDAGraphMode.FULL,
                SimpleNamespace(num_tokens=4, num_reqs=4),
                None,
                None,
                None,
            )
        )
        runner.synchronize_input_prep = nullcontext
        runner._should_build_dummy_attn_metadata = MagicMock(return_value=False)
        runner.maybe_dummy_run_with_lora = MagicMock(return_value=nullcontext())
        runner.lora_config = None
        runner.max_num_tokens = 4
        runner.supports_mm_inputs = False
        runner.model_config = SimpleNamespace(is_encoder_decoder=False)
        runner.enable_prompt_embeds = False
        runner.input_ids = SimpleNamespace(gpu=torch.zeros(4, dtype=torch.int64))
        runner.uses_mrope = False
        runner.uses_xdrope_dim = 0
        runner.positions = torch.zeros(4, dtype=torch.int64)
        runner.drafter = None
        runner.vllm_config = MagicMock()
        runner.model = MagicMock()
        runner._has_sinks = False
        runner.use_aux_hidden_state_outputs = False
        runner.use_compress = False
        runner._finalize_dump_data = MagicMock()

        def model_forward(*args):
            events.append("forward")
            return torch.zeros((4, 1))

        runner._model_forward = MagicMock(side_effect=model_forward)
        executor = MagicMock(submission_in_flight=True)
        executor.wait_all.side_effect = lambda: events.append("wait_all")
        executor.release.side_effect = lambda: events.append("release")
        runner.device_metadata_executor = executor

        @contextmanager
        def forward_context(*args, **kwargs):
            events.append("context_enter")
            yield
            events.append("context_exit")

        with (
            patch("vllm_ascend.worker.model_runner_v1.get_pp_group", return_value=SimpleNamespace(is_first_rank=True)),
            patch("vllm_ascend.worker.model_runner_v1.lmhead_tp_enable", return_value=False),
            patch("vllm_ascend.worker.model_runner_v1.set_ascend_forward_context", forward_context),
            patch("vllm_ascend.worker.model_runner_v1.update_cos_sin"),
        ):
            runner._dummy_run(4, cudagraph_runtime_mode=CUDAGraphMode.FULL, is_graph_capturing=True)

        self.assertEqual(events, ["wait_all", "context_enter", "forward", "context_exit", "release"])


class TestDSparkAuxCaptureMode(unittest.TestCase):
    def _build_runner(
        self,
        *,
        model_type: str,
        architecture: str,
        use_dspark: bool = True,
    ):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.speculative_config = SimpleNamespace(
            use_dspark=MagicMock(return_value=use_dspark),
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    model_type=model_type,
                    architectures=[architecture],
                )
            ),
        )
        return runner

    def test_qwen3_gqa_dspark_uses_materialized_stream(self):
        runner = self._build_runner(
            model_type="qwen3",
            architecture="Qwen3DSparkModel",
        )

        self.assertTrue(runner._draft_uses_qwen3_gqa_dspark())

    def test_mla_dspark_keeps_raw_stream(self):
        runner = self._build_runner(
            model_type="kimi_k3_dspark",
            architecture="KimiK3DSparkForCausalLM",
        )

        self.assertFalse(runner._draft_uses_qwen3_gqa_dspark())

    def test_non_dspark_keeps_raw_stream(self):
        runner = self._build_runner(
            model_type="qwen3",
            architecture="Qwen3DSparkModel",
            use_dspark=False,
        )

        self.assertFalse(runner._draft_uses_qwen3_gqa_dspark())


class TestAcceptedTokenSnapshot(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.use_async_scheduling = True
        runner.speculative_config = object()
        runner.model_config = SimpleNamespace(is_hybrid=True)
        runner.cache_config = SimpleNamespace(mamba_cache_mode="align")
        runner.num_accepted_tokens = CpuGpuBuffer(12, dtype=torch.int32, device=torch.device("cpu"), pin_memory=False)
        runner.prev_positions = CpuGpuBuffer(12, dtype=torch.int32, device=torch.device("cpu"), pin_memory=False)
        batch_counts = torch.ones(12, dtype=torch.int32)
        runner.input_batch = SimpleNamespace(
            num_accepted_tokens_cpu=batch_counts.numpy(),
            num_accepted_tokens_cpu_tensor=batch_counts,
        )
        runner.num_accepted_tokens_event = MagicMock()
        return runner

    def test_snapshot_survives_request_replacement_and_backend_reorder(self):
        runner = self._build_runner()
        with patch("vllm.v1.worker.gpu_input_batch.PIN_MEMORY", False):
            batch = InputBatch(
                max_num_reqs=12,
                max_model_len=32,
                max_num_batched_tokens=32,
                device=torch.device("cpu"),
                vocab_size=128,
                block_sizes=[4],
                kernel_block_sizes=[4],
                max_num_blocks_per_req=[8],
                num_spec_tokens=3,
            )
        runner.input_batch = batch
        for req_id in ("A", "B", "C"):
            batch.add_request(
                CachedRequestState(
                    req_id=req_id,
                    prompt_token_ids=[10, 11],
                    mm_features=[],
                    sampling_params=SamplingParams(temperature=0),
                    generator=None,
                    block_ids=([1],),
                    num_computed_tokens=2,
                    output_token_ids=[],
                )
            )
        batch.refresh_metadata()
        batch.prev_req_id_to_index = dict(batch.req_id_to_index)
        runner._get_mamba_bufs = MagicMock()
        runner.kv_cache_config = object()
        runner.compilation_config = SimpleNamespace(static_forward_context={})
        runner.model = MagicMock()

        # Only replace the device kernel boundary; use real InputBatch mutations.
        def postprocess(**kwargs):
            kwargs["num_accepted_tokens_cpu_tensor"][:3].copy_(kwargs["num_accepted_tokens_gpu"][:3])

        with patch(
            "vllm_ascend.worker.model_runner_v1.mamba_utils.postprocess_mamba_align_gpu",
            side_effect=postprocess,
        ):
            runner._update_states_after_model_execute(
                torch.tensor([[10, 11, -1, -1], [10, 11, 12, -1], [10, 11, 12, 13]]),
                SimpleNamespace(),
            )
        runner.num_accepted_tokens_event.record.assert_called_once()
        np.testing.assert_array_equal(runner.num_accepted_tokens.np[:3], [2, 3, 4])

        batch.remove_request("A")
        batch.add_request(
            CachedRequestState(
                req_id="D",
                prompt_token_ids=[10] * 16,
                mm_features=[],
                sampling_params=SamplingParams(temperature=0),
                generator=None,
                block_ids=([2],),
                num_computed_tokens=0,
                output_token_ids=[],
            )
        )
        batch.condense()
        self.assertTrue(
            reorder_batch_to_split_decodes_and_prefills(
                batch,
                SimpleNamespace(num_scheduled_tokens={"B": 4, "C": 4, "D": 16}),
                decode_threshold=4,
            )
        )
        batch.refresh_metadata()
        runner._compute_prev_positions(batch.num_reqs)
        runner._sync_num_accepted_tokens(batch.num_reqs, has_prev_mapping=True)

        expected = [{"B": 3, "C": 4, "D": 1}[req_id] for req_id in batch.req_ids]
        np.testing.assert_array_equal(runner.num_accepted_tokens.np[:3], expected)
        np.testing.assert_array_equal(batch.num_accepted_tokens_cpu[:3], expected)

    def test_sync_respects_snapshot_and_current_batch_ownership(self):
        for async_scheduling, has_prev_mapping, expected in (
            (True, True, [4, 1, 3]),
            (True, False, [1, 1, 1]),
            (False, True, [5, 6, 7]),
        ):
            with self.subTest(async_scheduling=async_scheduling, has_prev_mapping=has_prev_mapping):
                runner = self._build_runner()
                runner.use_async_scheduling = async_scheduling
                runner.num_accepted_tokens.np[:] = np.arange(12)
                runner.num_accepted_tokens.np[[11, 4]] = [4, 3]
                runner.prev_positions.np[:3] = [11, -1, 4]
                runner.input_batch.num_accepted_tokens_cpu[:3] = [5, 6, 7]

                runner._sync_num_accepted_tokens(3, has_prev_mapping=has_prev_mapping)

                np.testing.assert_array_equal(runner.num_accepted_tokens.np[:3], expected)
                np.testing.assert_array_equal(runner.input_batch.num_accepted_tokens_cpu[:3], expected)

    def test_non_align_postprocess_keeps_an_independent_snapshot(self):
        for mode in ("none", "all"):
            with self.subTest(mode=mode):
                runner = self._build_runner()
                runner.cache_config.mamba_cache_mode = mode
                runner.kv_cache_config = object()
                runner.requests = {}
                runner.mamba_state_idx = {}
                runner.num_spec_tokens = 3
                with patch("vllm_ascend.worker.model_runner_v1.mamba_utils.postprocess_mamba_all") as postprocess_all:
                    runner._update_states_after_model_execute(torch.tensor([[10, -1], [11, 12]]), SimpleNamespace())
                np.testing.assert_array_equal(runner.num_accepted_tokens.np[:2], [1, 2])
                np.testing.assert_array_equal(runner.input_batch.num_accepted_tokens_cpu[:2], [1, 1])
                self.assertEqual(postprocess_all.call_count, int(mode == "all"))
                runner.num_accepted_tokens_event.record.assert_called_once()


class TestNPUModelRunnerKVCache(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.enable_sparse_sfa_c8 = False
        runner.enable_sparse_li_c8 = False
        runner.use_compress = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.sfa_dcp_replicated_indexer_size = 1
        runner.runner_only_attn_layers = set()
        runner.is_kv_consumer = False
        runner.sparse_kv_offload_enabled = False
        runner.sparse_kv_offload_config = MagicMock()
        runner.tp_rank = 0
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
        runner.sparse_kv_offload_enabled = False
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

    @patch("vllm_ascend.worker.model_runner_v1.get_layers_from_vllm_config")
    def test_mla_rope_modes_and_cache_layers_use_separate_metadata_groups(self, mock_get_layers):
        class FakeBuilder:
            def __init__(self, _spec, layer_names, _config, _device):
                self.layer_names = layer_names

        class FakeMLABackend(AscendMLABackend):
            @classmethod
            def full_cls_name(cls):
                return "test.FakeMLABackend"

            @classmethod
            def get_builder_cls(cls):
                return FakeBuilder

        class FakeCacheBackend:
            @classmethod
            def full_cls_name(cls):
                return "test.FakeCacheBackend"

            @classmethod
            def get_builder_cls(cls):
                return FakeBuilder

        runner = self._build_runner()
        runner.attn_groups = []
        runner._check_and_update_cudagraph_mode = MagicMock()
        runner.calculate_reorder_batch_threshold = MagicMock()

        target_layer = "language_model.model.layers.0.self_attn.attn"
        draft_layer = "model.layers.0.self_attn.attn"
        cache_layer = "language_model.model.layers.0.self_attn.indexer.k_cache"
        target_attn = MagicMock(spec=MLAAttention)
        target_attn.impl = SimpleNamespace(use_mla_rope=False)
        target_attn.get_attn_backend.return_value = FakeMLABackend
        draft_attn = MagicMock(spec=MLAAttention)
        draft_attn.impl = SimpleNamespace(use_mla_rope=True)
        draft_attn.get_attn_backend.return_value = FakeMLABackend
        mock_get_layers.return_value = {
            target_layer: target_attn,
            draft_layer: draft_attn,
            cache_layer: SimpleNamespace(get_attn_backend=lambda: FakeCacheBackend),
        }
        specs = {
            target_layer: AscendMLAAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=576,
                dtype=torch.bfloat16,
            ),
            draft_layer: AscendMLAAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=576,
                dtype=torch.bfloat16,
            ),
            cache_layer: AscendMLAAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=576,
                dtype=torch.bfloat16,
            ),
        }
        group_spec = UniformTypeKVCacheSpecs.from_specs(specs)
        self.assertIsNotNone(group_spec)
        assert group_spec is not None
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=[target_layer, draft_layer, cache_layer],
                    kv_cache_spec=group_spec,
                )
            ],
        )

        runner.initialize_attn_backend(kv_cache_config)

        self.assertEqual(len(runner.attn_groups), 1)
        self.assertEqual(
            {tuple(group.layer_names) for group in runner.attn_groups[0]},
            {(target_layer,), (draft_layer,), (cache_layer,)},
        )

    def test_explicit_capture_sizes_must_align_spec_decode_and_sp(self):
        for capture_sizes, expected_tp_size in (([48, 96], 1), ([16, 32], 16)):
            with self.subTest(capture_sizes=capture_sizes):
                runner = self._build_runner()
                compilation_config = SimpleNamespace(
                    pass_config=SimpleNamespace(enable_sp=True),
                    cudagraph_capture_sizes=capture_sizes,
                    resolve_cudagraph_mode_and_sizes=MagicMock(return_value=CUDAGraphMode.FULL_DECODE_ONLY),
                )
                runner.compilation_config = compilation_config
                runner.vllm_config.compilation_config = compilation_config
                runner.parallel_config = SimpleNamespace(tensor_parallel_size=16)
                runner.uniform_decode_query_len = 6
                runner.kv_cache_config = SimpleNamespace()
                runner.max_num_reqs = 16
                runner.cudagraph_dispatcher = MagicMock()
                runner.cudagraph_dispatcher.get_capture_descs.return_value = []
                runner.speculative_config = None
                runner.drafter = None
                runner.use_aclgraph = False

                runner._check_and_update_cudagraph_mode([], [])

                call_kwargs = compilation_config.resolve_cudagraph_mode_and_sizes.call_args.kwargs
                self.assertEqual(
                    call_kwargs["tensor_parallel_size"],
                    expected_tp_size,
                )

    def test_sparse_c8_indexer_reuses_raw_cache_from_shared_descriptor(self):
        runner = self._build_runner()
        layer_names = [
            "model.layers.1.self_attn.indexer.k_cache",
            "model.layers.3.self_attn.indexer.k_cache",
        ]
        indexer_spec = AscendSFAIndexerCacheSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.int8,
            scale_dim=1,
            scale_dtype=torch.float16,
            cache_sparse_li_c8=True,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=indexer_spec.page_size_bytes * 2,
                    shared_by=layer_names,
                )
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=layer_names,
                    kv_cache_spec=indexer_spec,
                )
            ],
        )

        raw_caches = runner._allocate_kv_cache_tensors(kv_cache_config)

        assert raw_caches[layer_names[0]][0] is raw_caches[layer_names[1]][0]
        assert raw_caches[layer_names[0]][1] is raw_caches[layer_names[1]][1]

    def test_reshape_kv_cache_uses_layer_spec_for_draft_gqa(self):
        runner = self._build_runner()
        runner.sparse_kv_offload_enabled = False
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

    @patch("vllm_ascend.worker.model_runner_v1.get_layers_from_vllm_config")
    def test_hybrid_mla_cache_uses_logical_kernel_block_shape(
        self,
        mock_get_layers,
    ):
        """A 384-token scheduler page is exposed as three 128-token blocks."""
        runner = self._build_runner()
        runner.use_hybrid_blocks = True
        runner.hybrid_with_attn_and_mamba = False
        runner.model_config.hf_text_config = SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )

        layer_name = "draft_attn"
        attn_module = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(attn_module)
        attn_module.kv_lora_rank = 512
        attn_module.qk_rope_head_dim = 64
        mock_get_layers.return_value = {layer_name: attn_module}

        physical_block_size = 384
        kernel_block_size = 128
        num_physical_blocks = 2
        kv_cache_spec = AscendMLAAttentionSpec(
            block_size=physical_block_size,
            num_kv_heads=1,
            head_size=512 + 64,
            dtype=torch.bfloat16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=num_physical_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=kv_cache_spec.page_size_bytes * num_physical_blocks,
                    shared_by=[layer_name],
                )
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=[layer_name],
                    kv_cache_spec=kv_cache_spec,
                )
            ],
        )

        # Raw cache tensors are byte buffers. Together they contain two
        # physical pages: 512 latent dimensions plus 64 RoPE dimensions.
        raw_k_cache = torch.empty(
            num_physical_blocks * physical_block_size * 512 * 2,
            dtype=torch.uint8,
        )
        raw_v_cache = torch.empty(
            num_physical_blocks * physical_block_size * 64 * 2,
            dtype=torch.uint8,
        )
        backend = MagicMock()
        backend.get_supported_kernel_block_sizes.return_value = [kernel_block_size]
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=kv_cache_spec,
                backend=backend,
                layer_names=[layer_name],
            )
        ]

        k_cache, v_cache = runner._reshape_kv_cache_tensors(
            kv_cache_config,
            {layer_name: (raw_k_cache, raw_v_cache)},
        )[layer_name]

        num_kernel_blocks = num_physical_blocks * physical_block_size // kernel_block_size
        self.assertEqual(k_cache.shape, (num_kernel_blocks, 128, 1, 512))
        self.assertEqual(v_cache.shape, (num_kernel_blocks, 128, 1, 64))
        self.assertEqual(
            backend.get_kv_cache_shape.call_args.args[:2],
            (num_kernel_blocks, kernel_block_size),
        )

    @patch("vllm_ascend.worker.model_runner_v1.has_ec_transfer", return_value=False)
    @patch("vllm_ascend.worker.model_runner_v1.get_layers_from_vllm_config")
    def test_sparse_layer_without_indexer_allocates_only_mla_kv_cache(
        self,
        mock_get_layers,
        _mock_has_ec_transfer,
    ):
        runner = self._build_runner()
        runner.use_sparse = True
        runner.block_size = 16
        runner.kv_cache_dtype = torch.bfloat16
        runner.shared_kv_cache_layers = {}
        runner.ascend_config = MagicMock()
        runner.model_config.hf_text_config = SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
        runner.vllm_config.cache_config.cache_dtype = "auto"
        runner.sparse_kv_offload_enabled = False

        attn_module = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(attn_module)
        attn_module.impl = SimpleNamespace(
            has_indexer=False,
            enable_sparse_sfa_c8=False,
            enable_sparse_li_c8=False,
        )
        attn_module.kv_lora_rank = 512
        attn_module.qk_rope_head_dim = 64
        layer_name = "model.layers.1.self_attn.attn"
        mock_get_layers.return_value = {layer_name: attn_module}

        specs = runner.get_kv_cache_spec()
        self.assertEqual(set(specs), {layer_name})
        spec = specs[layer_name]
        self.assertEqual(spec.head_size, 512 + 64)
        self.assertFalse(hasattr(spec, "separate_sfa_indexer_cache"))

        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=spec.page_size_bytes * 2,
                    shared_by=[layer_name],
                )
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=[layer_name],
                    kv_cache_spec=spec,
                )
            ],
        )

        raw_caches = runner._allocate_kv_cache_tensors(kv_cache_config)
        raw_k_cache, raw_v_cache = raw_caches[layer_name]

        self.assertEqual(raw_k_cache.numel(), 2 * 16 * 512 * 2)
        self.assertEqual(raw_v_cache.numel(), 2 * 16 * 64 * 2)

    @patch("vllm_ascend.worker.model_runner_v1.has_ec_transfer", return_value=False)
    @patch("vllm_ascend.worker.model_runner_v1.get_layers_from_vllm_config")
    def test_sparse_indexer_allocates_separate_replicated_cache_tensor(
        self,
        mock_get_layers,
        _mock_has_ec_transfer,
    ):
        runner = self._build_runner()
        runner.use_sparse = True
        runner.block_size = 16
        runner.sfa_dcp_replicated_indexer_size = 2
        runner.kv_cache_dtype = torch.bfloat16
        runner.shared_kv_cache_layers = {}
        runner.ascend_config = MagicMock()
        runner.ascend_config.is_sparse_li_c8_layer.return_value = False
        runner.model_config.hf_text_config = SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
        )
        runner.vllm_config.cache_config.cache_dtype = "auto"
        runner.sparse_kv_offload_enabled = False

        attn_module = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(attn_module)
        attn_module.impl = SimpleNamespace(
            has_indexer=True,
            enable_sparse_sfa_c8=False,
            enable_sparse_li_c8=False,
        )
        attn_module.kv_lora_rank = 512
        attn_module.qk_rope_head_dim = 64
        indexer_module = DeepseekV32IndexerCache.__new__(DeepseekV32IndexerCache)
        torch.nn.Module.__init__(indexer_module)
        attn_layer_name = "model.layers.1.self_attn.attn"
        indexer_layer_name = "model.layers.1.self_attn.indexer.k_cache"
        mock_get_layers.return_value = {
            attn_layer_name: attn_module,
            indexer_layer_name: indexer_module,
        }

        specs = runner.get_kv_cache_spec()
        main_spec = specs[attn_layer_name]
        indexer_spec = specs[indexer_layer_name]
        self.assertIsInstance(indexer_spec, AscendSFAIndexerCacheSpec)
        self.assertEqual(main_spec.page_size_bytes, 16 * (512 + 64) * 2)
        self.assertEqual(indexer_spec.page_size_bytes, 2 * 16 * 128 * 2)

        group_spec = UniformTypeKVCacheSpecs.from_specs(specs)
        self.assertIsNotNone(group_spec)
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=main_spec.page_size_bytes * 2,
                    shared_by=[attn_layer_name],
                ),
                KVCacheTensor(
                    size=indexer_spec.page_size_bytes * 2,
                    shared_by=[indexer_layer_name],
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=[attn_layer_name, indexer_layer_name],
                    kv_cache_spec=group_spec,
                )
            ],
        )

        raw_caches = runner._allocate_kv_cache_tensors(kv_cache_config)
        raw_k_cache, raw_v_cache = raw_caches[attn_layer_name]
        (raw_indexer_cache,) = raw_caches[indexer_layer_name]

        self.assertEqual(raw_k_cache.numel(), 2 * 16 * 512 * 2)
        self.assertEqual(raw_v_cache.numel(), 2 * 16 * 64 * 2)
        self.assertEqual(raw_indexer_cache.numel(), 2 * 2 * 16 * 128 * 2)

        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=main_spec,
                backend=backend,
                layer_names=[attn_layer_name],
            ),
            SimpleNamespace(
                kv_cache_spec=indexer_spec,
                backend=backend,
                layer_names=[indexer_layer_name],
            ),
        ]

        caches = runner._reshape_kv_cache_tensors(kv_cache_config, raw_caches)
        k_cache, v_cache = caches[attn_layer_name]
        (indexer_cache,) = caches[indexer_layer_name]

        self.assertEqual(k_cache.shape, (2, 16, 1, 512))
        self.assertEqual(v_cache.shape, (2, 16, 1, 64))
        self.assertEqual(indexer_cache.shape, (4, 16, 1, 128))

    def test_sparse_c8_indexer_owns_quantized_cache_accounting(self):
        main_spec = AscendMLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=512 + 64,
            dtype=torch.bfloat16,
            cache_sparse_sfa_c8=False,
        )
        indexer_spec = AscendSFAIndexerCacheSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.int8,
            scale_dim=1,
            scale_dtype=torch.float16,
            cache_sparse_li_c8=True,
            sfa_dcp_replicated_indexer_size=2,
        )

        self.assertEqual(main_spec.page_size_bytes, 16 * (512 + 64) * 2)
        self.assertEqual(indexer_spec.page_size_bytes, 2 * 16 * (128 + 2))
        self.assertFalse(hasattr(main_spec, "sfa_dcp_replicated_indexer_size"))

    def test_sparse_sfa_and_li_c8_allocate_and_reshape_independently(self):
        runner = self._build_runner()
        runner.use_sparse = True
        runner.block_size = 16
        runner.c8_k_cache_dtype = torch.int8
        runner.c8_k_scale_cache_dtype = torch.float16
        runner._get_attention_kv_cache_dims = lambda _layer_name, _spec: (512, 64)
        runner.sparse_kv_offload_enabled = False

        attn_layer_name = "model.layers.1.self_attn.attn"
        indexer_layer_name = "model.layers.1.self_attn.indexer.k_cache"
        packed_head_dim = get_sfa_qsfa_packed_head_dim(512, 64)

        for enable_sfa_c8, enable_li_c8 in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(
                enable_sfa_c8=enable_sfa_c8,
                enable_li_c8=enable_li_c8,
            ):
                main_spec = AscendMLAAttentionSpec(
                    block_size=runner.block_size,
                    num_kv_heads=1,
                    head_size=packed_head_dim if enable_sfa_c8 else 512 + 64,
                    dtype=torch.int8 if enable_sfa_c8 else torch.bfloat16,
                    cache_sparse_sfa_c8=enable_sfa_c8,
                )
                indexer_spec = AscendSFAIndexerCacheSpec(
                    block_size=runner.block_size,
                    num_kv_heads=1,
                    head_size=128,
                    dtype=torch.int8 if enable_li_c8 else torch.bfloat16,
                    scale_dim=1 if enable_li_c8 else 0,
                    scale_dtype=torch.float16 if enable_li_c8 else torch.int8,
                    cache_sparse_li_c8=enable_li_c8,
                )
                group_spec = UniformTypeKVCacheSpecs.from_specs(
                    {
                        attn_layer_name: main_spec,
                        indexer_layer_name: indexer_spec,
                    }
                )
                self.assertIsNotNone(group_spec)
                assert group_spec is not None

                kv_cache_config = KVCacheConfig(
                    num_blocks=2,
                    kv_cache_tensors=[
                        KVCacheTensor(
                            size=main_spec.page_size_bytes * 2,
                            shared_by=[attn_layer_name],
                        ),
                        KVCacheTensor(
                            size=indexer_spec.page_size_bytes * 2,
                            shared_by=[indexer_layer_name],
                        ),
                    ],
                    kv_cache_groups=[
                        KVCacheGroupSpec(
                            layer_names=[attn_layer_name, indexer_layer_name],
                            kv_cache_spec=group_spec,
                        )
                    ],
                )
                backend = MagicMock()
                backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
                    num_blocks,
                    block_size,
                    num_kv_heads,
                    head_size,
                )
                runner._kv_cache_spec_attn_group_iterator = MagicMock(
                    return_value=[
                        SimpleNamespace(
                            kv_cache_spec=main_spec,
                            backend=backend,
                            layer_names=[attn_layer_name],
                        ),
                        SimpleNamespace(
                            kv_cache_spec=indexer_spec,
                            backend=backend,
                            layer_names=[indexer_layer_name],
                        ),
                    ]
                )

                raw_caches = runner._allocate_kv_cache_tensors(kv_cache_config)
                caches = runner._reshape_kv_cache_tensors(kv_cache_config, raw_caches)

                main_cache = caches[attn_layer_name]
                self.assertEqual(len(main_cache), 1 if enable_sfa_c8 else 2)
                if enable_sfa_c8:
                    self.assertEqual(main_cache[0].shape, (2, 16, 1, packed_head_dim))
                    self.assertEqual(main_cache[0].dtype, torch.int8)
                else:
                    self.assertEqual(main_cache[0].shape, (2, 16, 1, 512))
                    self.assertEqual(main_cache[1].shape, (2, 16, 1, 64))
                    self.assertEqual(main_cache[0].dtype, torch.bfloat16)
                    self.assertEqual(main_cache[1].dtype, torch.bfloat16)

                indexer_cache = caches[indexer_layer_name]
                self.assertEqual(len(indexer_cache), 2 if enable_li_c8 else 1)
                self.assertEqual(indexer_cache[0].shape, (2, 16, 1, 128))
                self.assertEqual(
                    indexer_cache[0].dtype,
                    torch.int8 if enable_li_c8 else torch.bfloat16,
                )
                if enable_li_c8:
                    self.assertEqual(indexer_cache[1].shape, (2, 16, 1, 1))
                    self.assertEqual(indexer_cache[1].dtype, torch.float16)

    @patch(
        "vllm_ascend.worker.model_runner_v1.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A5),
    )
    @patch("vllm_ascend.worker.model_runner_v1.has_ec_transfer", return_value=False)
    @patch("vllm_ascend.worker.model_runner_v1.get_layers_from_vllm_config")
    def test_a5_sparse_c8_specs_keep_main_and_indexer_layouts_separate(
        self,
        mock_get_layers,
        _mock_has_ec_transfer,
        _mock_get_device_type,
    ):
        runner = self._build_runner()
        runner.use_sparse = True
        runner.block_size = 16
        runner.kv_cache_dtype = torch.bfloat16
        runner.c8_k_cache_dtype = torch.float8_e4m3fn
        runner.c8_k_scale_cache_dtype = torch.float32
        runner.shared_kv_cache_layers = {}
        runner.ascend_config = MagicMock()
        runner.model_config.hf_text_config = SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
        )
        runner.vllm_config.cache_config.cache_dtype = "auto"
        runner.sparse_kv_offload_enabled = False

        attn_module = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(attn_module)
        attn_module.kv_lora_rank = 512
        attn_module.qk_rope_head_dim = 64
        indexer_module = DeepseekV32IndexerCache.__new__(DeepseekV32IndexerCache)
        torch.nn.Module.__init__(indexer_module)
        attn_layer_name = "model.layers.1.self_attn.attn"
        indexer_layer_name = "model.layers.1.self_attn.indexer.k_cache"
        mock_get_layers.return_value = {
            attn_layer_name: attn_module,
            indexer_layer_name: indexer_module,
        }

        packed_head_dim = get_sfa_qsfa_packed_head_dim(512, 64)
        for enable_sfa_c8, enable_li_c8 in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(
                enable_sfa_c8=enable_sfa_c8,
                enable_li_c8=enable_li_c8,
            ):
                attn_module.impl = SimpleNamespace(
                    has_indexer=True,
                    enable_sparse_sfa_c8=enable_sfa_c8,
                    enable_sparse_li_c8=enable_li_c8,
                )
                runner.ascend_config.is_sparse_li_c8_layer.return_value = enable_li_c8
                runner.ascend_config.is_sparse_li_c8_layer.reset_mock()

                specs = runner.get_kv_cache_spec()
                main_spec = specs[attn_layer_name]
                indexer_spec = specs[indexer_layer_name]

                self.assertEqual(
                    runner.ascend_config.is_sparse_li_c8_layer.call_args_list,
                    [call(indexer_layer_name)],
                )
                self.assertEqual(
                    main_spec.head_size,
                    packed_head_dim if enable_sfa_c8 else 512 + 64,
                )
                self.assertEqual(
                    main_spec.dtype,
                    torch.float8_e4m3fn if enable_sfa_c8 else torch.bfloat16,
                )
                self.assertEqual(main_spec.cache_sparse_sfa_c8, enable_sfa_c8)
                self.assertEqual(
                    indexer_spec.dtype,
                    torch.float8_e4m3fn if enable_li_c8 else torch.bfloat16,
                )
                self.assertEqual(indexer_spec.scale_dim, 1 if enable_li_c8 else 0)
                self.assertEqual(
                    indexer_spec.scale_dtype,
                    torch.float32 if enable_li_c8 else torch.int8,
                )
                self.assertEqual(indexer_spec.cache_sparse_li_c8, enable_li_c8)

    def test_deepseek_v4_indexer_keeps_compressed_mla_layout(self):
        runner = self._build_runner()
        runner.use_compress = True
        layer_name = "model.layers.1.self_attn.indexer.k_cache"
        indexer_spec = AscendMLAAttentionSpec(
            block_size=16 * 4,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.int8,
            model_version="deepseek_v4",
            compress_ratio=4,
            scale_dim=1,
            scale_dtype=torch.float16,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=indexer_spec.page_size_bytes * 2,
                    shared_by=[layer_name],
                )
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=[layer_name],
                    kv_cache_spec=indexer_spec,
                )
            ],
        )

        raw_caches = runner._allocate_kv_cache_tensors(kv_cache_config)
        self.assertEqual(raw_caches[layer_name].numel(), 2 * 16 * (128 + 2))

        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner.attn_backend = backend
        runner._kv_cache_spec_attn_group_iterator = lambda: [
            SimpleNamespace(
                kv_cache_spec=indexer_spec,
                backend=backend,
                layer_names=[layer_name],
            )
        ]

        indexer_k_cache, indexer_scale_cache = runner._reshape_kv_cache_tensors(kv_cache_config, raw_caches)[layer_name]
        self.assertEqual(indexer_k_cache.shape, (2, 16, 1, 128))
        self.assertEqual(indexer_k_cache.dtype, torch.int8)
        self.assertEqual(indexer_scale_cache.shape, (2, 16, 1, 1))
        self.assertEqual(indexer_scale_cache.dtype, torch.float16)


class TestNPUModelRunnerEncoderCacheReset(unittest.TestCase):
    @staticmethod
    def _build_runner():
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.encoder_cache = {"device": object()}
        runner.tmp_encoder_cache = {}
        runner.cpu_encoder_cache = {}
        runner.cached = {}
        runner._pending_encoder_cache_copies = deque()
        runner.late_interaction_runner = MagicMock()
        runner._sync_device = MagicMock()
        return runner

    def test_reset_clears_score_encoder_cache_state(self):
        runner = self._build_runner()
        runner.tmp_encoder_cache["tmp"] = object()
        runner.cpu_encoder_cache["cpu"] = object()
        runner.cached["tmp"] = {"request"}
        runner._pending_encoder_cache_copies.append((object(), MagicMock()))

        runner.reset_encoder_cache()

        runner._sync_device.assert_called_once_with()
        self.assertFalse(runner.encoder_cache)
        self.assertFalse(runner.tmp_encoder_cache)
        self.assertFalse(runner.cpu_encoder_cache)
        self.assertFalse(runner.cached)
        self.assertFalse(runner._pending_encoder_cache_copies)
        runner.late_interaction_runner.clear.assert_called_once_with()


class TestNPUModelRunnerScoreEncoderCache(unittest.TestCase):
    @staticmethod
    def _build_runner(use_score_encoder_cache=True):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.encoder_cache = {}
        runner.cpu_encoder_cache = {}
        runner.tmp_encoder_cache = {}
        runner.cached = {}
        runner._pending_encoder_cache_copies = deque()
        runner.use_score_encoder_cache = use_score_encoder_cache
        runner.maybe_save_ec_to_connector = MagicMock()
        return runner

    def test_processes_score_cache_migrations_and_frees(self):
        runner = self._build_runner()
        runner.encoder_cache = {
            "npu-freed": "npu",
            "cpu-freed": "npu",
        }
        runner.cpu_encoder_cache = {
            "npu-freed": "cpu",
            "cpu-freed": "cpu",
            "promote": "promote",
            "temporary": "temporary",
        }
        runner._copy_cpu_encoder_cache_to_device = MagicMock(side_effect=lambda value: f"device-{value}")
        metadata = SimpleNamespace(
            promoting_mm_hashes=["promote"],
            cpu_get_encoder_mm_hashes=["temporary"],
            npu_freed=["npu-freed"],
            cpu_freed=["cpu-freed"],
        )
        scheduler_output = SimpleNamespace(ec_manager_metadata=metadata)

        runner._process_encoder_cache_scheduler_output(scheduler_output)

        self.assertEqual(set(runner.encoder_cache), {"cpu-freed", "promote"})
        self.assertEqual(
            set(runner.cpu_encoder_cache),
            {"npu-freed", "promote", "temporary"},
        )
        self.assertEqual(runner.encoder_cache["promote"], "device-promote")
        self.assertEqual(runner.tmp_encoder_cache["temporary"], "device-temporary")

    def test_score_cache_disabled_delegates_to_upstream(self):
        runner = self._build_runner(use_score_encoder_cache=False)
        scheduler_output = SimpleNamespace()

        with patch.object(
            GPUModelRunner,
            "_process_encoder_cache_scheduler_output",
            autospec=True,
        ) as upstream_process:
            runner._process_encoder_cache_scheduler_output(scheduler_output)

        upstream_process.assert_called_once_with(runner, scheduler_output)

    def test_async_copy_keeps_pinned_source_until_event_completes(self):
        runner = self._build_runner()
        runner.device = "npu"
        cpu_value = MagicMock()
        pinned_source = MagicMock()
        npu_value = MagicMock()
        copy_done = MagicMock()
        cpu_value.is_pinned.return_value = False
        cpu_value.pin_memory.return_value = pinned_source
        fake_npu = SimpleNamespace(
            Event=MagicMock(return_value=copy_done),
            current_stream=MagicMock(),
        )

        with (
            patch.object(torch, "empty_like", return_value=npu_value),
            patch.object(torch, "npu", fake_npu, create=True),
        ):
            runner._copy_cpu_encoder_cache_to_device(cpu_value)

        npu_value.copy_.assert_called_once_with(pinned_source, non_blocking=True)
        self.assertIs(runner._pending_encoder_cache_copies[0][0], pinned_source)
        self.assertIs(runner._pending_encoder_cache_copies[0][1], copy_done)
        copy_done.record.assert_called_once()

        copy_done.query.side_effect = [False, True]
        runner._clear_finished_encoder_cache_copies()
        self.assertEqual(len(runner._pending_encoder_cache_copies), 1)
        runner._clear_finished_encoder_cache_copies()
        self.assertFalse(runner._pending_encoder_cache_copies)

    def test_new_output_is_staged_on_cpu(self):
        runner = self._build_runner()
        output = MagicMock()
        staging = MagicMock()

        with patch.object(torch, "empty_like", return_value=staging):
            runner._cache_encoder_output(
                "image",
                output,
                SimpleNamespace(promoting_mm_hashes=[], npu_freed=[]),
                [],
            )

        self.assertIs(runner.cpu_encoder_cache["image"], staging)
        self.assertIs(runner.tmp_encoder_cache["image"], output)
        self.assertNotIn("image", runner.encoder_cache)
        staging.copy_.assert_called_once_with(
            output.detach.return_value,
            non_blocking=True,
        )

    def test_tmp_cache_is_released_after_last_request_reference(self):
        runner = self._build_runner()
        tmp_output = object()
        runner.tmp_encoder_cache["image"] = tmp_output
        runner.cached["image"] = {"first", "second"}
        req_state = SimpleNamespace(mm_features=[SimpleNamespace(identifier="image")])

        runner._on_request_state_removed("first", req_state)

        self.assertIn("image", runner.tmp_encoder_cache)

        runner._on_request_state_removed("second", req_state)

        self.assertNotIn("image", runner.cached)
        self.assertNotIn("image", runner.tmp_encoder_cache)


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
        input_batch.sampling_metadata.top_k = None
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

    def test_placeholder_spec_tokens_are_sanitized_only_for_forward(self):
        runner = self._build_runner()
        runner.input_ids = SimpleNamespace(
            cpu=torch.tensor([11, -1, 33, -1], dtype=torch.int32),
            gpu=torch.tensor([11, -1, 33, -1], dtype=torch.int32),
        )
        scheduler_output = SimpleNamespace(
            scheduled_spec_decode_tokens={"req0": [-1]},
        )

        runner._sanitize_placeholder_input_ids_for_forward(
            scheduler_output,
            num_forward_tokens=4,
        )

        self.assertEqual(runner.input_ids.gpu.tolist(), [11, 0, 33, 0])
        self.assertEqual(runner.input_ids.cpu.tolist(), [11, -1, 33, -1])

    def test_placeholder_sanitization_is_scoped_to_current_forward(self):
        runner = self._build_runner()
        runner.input_ids = SimpleNamespace(
            cpu=torch.tensor([11, -1, 33, -1], dtype=torch.int32),
            gpu=torch.tensor([11, -1, 33, -1], dtype=torch.int32),
        )
        scheduler_output = SimpleNamespace(
            scheduled_spec_decode_tokens={"req0": [-1]},
        )

        runner._sanitize_placeholder_input_ids_for_forward(
            scheduler_output,
            num_forward_tokens=2,
        )

        self.assertEqual(runner.input_ids.gpu.tolist(), [11, 0, 33, -1])

    def test_mtp3_placeholder_metadata_is_preserved_before_sanitizing_forward(self):
        runner = self._build_runner()
        runner.arange_np = np.arange(8, dtype=np.int32)
        runner._arange_scratch = np.empty(8, dtype=np.int32)
        runner.input_ids = SimpleNamespace(
            cpu=torch.tensor([11, -1, -1, -1], dtype=torch.int32),
            gpu=torch.tensor([11, -1, -1, -1], dtype=torch.int32),
        )
        scheduler_output = SimpleNamespace(
            scheduled_spec_decode_tokens={"req0": [-1, -1, -1]},
        )

        spec_decode_metadata = runner._calc_spec_decode_metadata(
            num_draft_tokens=np.array([3], dtype=np.int32),
            cu_num_scheduled_tokens=np.array([4], dtype=np.int32),
        )
        runner._sanitize_placeholder_input_ids_for_forward(
            scheduler_output,
            num_forward_tokens=4,
        )

        self.assertEqual(spec_decode_metadata.draft_token_ids.tolist(), [-1, -1, -1])
        self.assertEqual(runner.input_ids.gpu.tolist(), [11, 0, 0, 0])
        self.assertEqual(runner.input_ids.cpu.tolist(), [11, -1, -1, -1])


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

    def test_start_dump_data_forwards_kwargs_to_debugger_start(self):
        debugger = MagicMock(spec=["start", "step"])
        runner = self._build_runner(debugger)
        runner._debugger_started = False

        runner._start_dump_data(scheduled_tokens={"req-0": 42})

        debugger.start.assert_called_once_with(runner.model, scheduled_tokens={"req-0": 42})
        self.assertTrue(runner._debugger_started)

    @patch("vllm_ascend.worker.model_runner_v1.has_kv_transfer_group", return_value=False)
    @patch("vllm_ascend.worker.model_runner_v1.has_ec_transfer", return_value=False)
    @patch("vllm_ascend.worker.model_runner_v1.get_pp_group")
    @patch("vllm_ascend.worker.model_runner_v1.record_function_or_nullcontext")
    def test_execute_model_skips_dump_start_for_dp_dummy_run(
        self, mock_record_function, mock_get_pp_group, _mock_has_ec_transfer, _mock_has_kv_transfer_group
    ):
        from contextlib import nullcontext

        mock_record_function.return_value = nullcontext()
        mock_get_pp_group.return_value = SimpleNamespace(world_size=1, is_first_rank=True, is_last_rank=True)
        runner = self._build_runner(MagicMock(spec=["start", "stop", "step"]))
        runner.vllm_config = MagicMock()
        runner.vllm_config.model_config.enable_return_routed_experts = False
        runner.ascend_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(enabled=False, need_timing=False))
        )
        runner.execute_model_state = None
        runner.speculative_config = None
        runner.use_async_scheduling = False
        runner.num_spec_tokens = 0
        runner._draft_token_ids = None
        runner.supports_mm_inputs = False
        runner.model_config.is_encoder_decoder = False
        runner.synchronize_input_prep = nullcontext
        runner._update_states = MagicMock(return_value=None)
        runner.parallel_config = SimpleNamespace(distributed_executor_backend="external_launcher", data_parallel_size=2)
        runner._dummy_run = MagicMock()
        runner._start_dump_data = MagicMock()
        runner.requests = {}
        scheduler_output = SimpleNamespace(total_num_scheduled_tokens=0)

        runner.execute_model(scheduler_output)

        runner._dummy_run.assert_called_once_with(1)
        runner._start_dump_data.assert_not_called()

    @patch("vllm_ascend.worker.model_runner_v1.has_kv_transfer_group", return_value=False)
    @patch("vllm_ascend.worker.model_runner_v1.has_ec_transfer", return_value=False)
    @patch("vllm_ascend.worker.model_runner_v1.get_pp_group")
    @patch("vllm_ascend.worker.model_runner_v1.record_function_or_nullcontext")
    def test_execute_model_starts_dump_for_real_batch(
        self, mock_record_function, mock_get_pp_group, _mock_has_ec_transfer, _mock_has_kv_transfer_group
    ):
        from contextlib import nullcontext

        mock_record_function.return_value = nullcontext()
        mock_get_pp_group.return_value = SimpleNamespace(world_size=1, is_first_rank=True, is_last_rank=True)
        runner = self._build_runner(MagicMock(spec=["start", "stop", "step"]))
        runner.vllm_config = MagicMock()
        runner.vllm_config.model_config.enable_return_routed_experts = False
        runner.ascend_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(enabled=False, need_timing=False))
        )
        runner.execute_model_state = None
        runner.speculative_config = None
        runner.use_async_scheduling = False
        runner.num_spec_tokens = 0
        runner._draft_token_ids = None
        runner.supports_mm_inputs = False
        runner.model_config.is_encoder_decoder = False
        runner.synchronize_input_prep = nullcontext
        runner._update_states = MagicMock(return_value=None)
        runner.parallel_config = SimpleNamespace(
            distributed_executor_backend="external_launcher", data_parallel_size=2, enable_dbo=False
        )
        runner.cache_config = SimpleNamespace(kv_sharing_fast_prefill=False)
        runner.input_batch = SimpleNamespace(num_reqs=1, req_ids=["req0"], prev_req_id_to_index=None)
        runner.requests = {}
        runner._start_dump_data = MagicMock()
        runner._prepare_inputs = MagicMock(side_effect=RuntimeError("sentinel"))
        scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1, num_scheduled_tokens={"req0": 1})

        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            runner.execute_model(scheduler_output)

        runner._start_dump_data.assert_called_once_with(scheduled_tokens={"req0": 1})

    @patch("vllm_ascend.worker.model_runner_v1.has_kv_transfer_group", return_value=False)
    @patch("vllm_ascend.worker.model_runner_v1.has_ec_transfer", return_value=False)
    @patch("vllm_ascend.worker.model_runner_v1.get_pp_group")
    @patch("vllm_ascend.worker.model_runner_v1.record_function_or_nullcontext")
    def test_execute_model_ignores_need_timing_when_profiling_chunk_is_disabled(
        self, mock_record_function, mock_get_pp_group, _mock_has_ec_transfer, _mock_has_kv_transfer_group
    ):
        from contextlib import nullcontext

        mock_record_function.return_value = nullcontext()
        mock_get_pp_group.return_value = SimpleNamespace(world_size=1, is_first_rank=True, is_last_rank=True)
        runner = self._build_runner(MagicMock(spec=["start", "stop", "step"]))
        runner.vllm_config = MagicMock()
        runner.vllm_config.model_config.enable_return_routed_experts = False
        runner.ascend_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(enabled=False, need_timing=True))
        )
        runner.execute_model_state = None
        runner.speculative_config = None
        runner.use_async_scheduling = False
        runner.num_spec_tokens = 0
        runner._draft_token_ids = None
        runner.supports_mm_inputs = False
        runner.model_config.is_encoder_decoder = False
        runner.synchronize_input_prep = nullcontext
        runner._update_states = MagicMock(return_value=None)
        runner._sync_device = MagicMock()
        runner.parallel_config = SimpleNamespace(
            distributed_executor_backend="external_launcher", data_parallel_size=2, enable_dbo=False
        )
        runner.cache_config = SimpleNamespace(kv_sharing_fast_prefill=False)
        runner.input_batch = SimpleNamespace(num_reqs=1, req_ids=["req0"], prev_req_id_to_index=None)
        runner.requests = {}
        runner._prepare_inputs = MagicMock(side_effect=RuntimeError("sentinel"))
        scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1, num_scheduled_tokens={"req0": 1})

        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            runner.execute_model(scheduler_output)

        runner._sync_device.assert_not_called()
        self.assertFalse(hasattr(runner, "_execution_start_time"))


class TestCorrectOptimisticSeqLensCpu(unittest.TestCase):
    """Regression tests for async spec-decode seq_lens correction.

    The helper must synchronize the device->host copy event *before* reading
    ``valid_sampled_token_count_cpu``. Reading it early consumes stale counts
    and corrupts the CPU seq_lens, which surfaced as an accuracy regression on
    DeepSeek-V4 (its compressed-KV slot mapping is built from these seq_lens).
    """

    def _build_runner(self, optimistic, prev_positions, prev_drafts, counts_cpu):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.optimistic_seq_lens_cpu = optimistic
        runner.prev_positions = SimpleNamespace(np=prev_positions)
        runner.prev_num_draft_tokens = SimpleNamespace(np=prev_drafts)
        runner.valid_sampled_token_count_cpu = counts_cpu
        return runner

    def test_synchronizes_before_host_read(self):
        num_reqs = 3
        # Optimistic (all drafts assumed accepted):
        #   prev_computed=[100,200,50], prev_drafts=[2,3,1], sched=[3,4,2]
        #   optimistic = prev_computed + (prev_drafts + 1) + sched
        optimistic = torch.tensor([106, 208, 54], dtype=torch.int64)
        prev_positions = np.array([0, 1, 2], dtype=np.int64)
        prev_drafts = np.array([2, 3, 1], dtype=np.int32)

        # CPU buffer initially holds STALE counts (== drafts + 1, i.e. "all
        # accepted"). If the helper reads before synchronizing, the correction
        # is a no-op and the assertion below fails.
        counts_cpu = torch.tensor([3, 4, 2], dtype=torch.int32)
        # The true counts that the async copy delivers on synchronize().
        true_counts = np.array([2, 1, 2], dtype=np.int32)

        runner = self._build_runner(optimistic, prev_positions, prev_drafts, counts_cpu)
        event = MagicMock()
        event.synchronize.side_effect = lambda: counts_cpu.copy_(torch.from_numpy(true_counts))
        runner.valid_sampled_token_count_event = event

        runner._correct_optimistic_seq_lens_cpu(num_reqs)

        event.synchronize.assert_called_once()
        # correction = (prev_drafts + 1 - true_counts) = [1, 3, 0]
        # corrected  = optimistic - correction          = [105, 205, 54]
        np.testing.assert_array_equal(optimistic.numpy(), np.array([105, 205, 54]))

    def test_asserts_event_present(self):
        runner = self._build_runner(
            torch.tensor([10], dtype=torch.int64),
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int32),
            torch.tensor([1], dtype=torch.int32),
        )
        runner.valid_sampled_token_count_event = None
        with self.assertRaises(AssertionError):
            runner._correct_optimistic_seq_lens_cpu(1)


if __name__ == "__main__":
    unittest.main()
