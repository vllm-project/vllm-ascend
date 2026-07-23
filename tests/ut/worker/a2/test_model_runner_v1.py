import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import torch
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSFAIndexerCacheSpec
from vllm_ascend.utils import AscendDeviceType
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class TestNPUModelRunnerKVCache(unittest.TestCase):
    def _build_runner(self):
        runner = NPUModelRunner.__new__(NPUModelRunner)
        runner.device = torch.device("cpu")
        runner.use_sparse = False
        runner.use_sparse_c8 = False
        runner.use_compress = False
        runner.use_hybrid_blocks = False
        runner.hybrid_with_attn_and_mamba = False
        runner.sfa_dcp_replicated_indexer_size = 1
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

        attn_module = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(attn_module)
        attn_module.impl = SimpleNamespace(
            has_indexer=False,
            use_sparse_c8_sfa=False,
            use_sparse_c8_indexer=False,
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
        runner.ascend_config.is_sparse_c8_layer.return_value = False
        runner.model_config.hf_text_config = SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
        )
        runner.vllm_config.cache_config.cache_dtype = "auto"

        attn_module = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(attn_module)
        attn_module.impl = SimpleNamespace(
            has_indexer=True,
            use_sparse_c8_sfa=False,
            use_sparse_c8_indexer=False,
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
            cache_sparse_c8=True,
        )
        indexer_spec = AscendSFAIndexerCacheSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.int8,
            scale_dim=1,
            scale_dtype=torch.float16,
            cache_sparse_c8=True,
            sfa_dcp_replicated_indexer_size=2,
        )

        self.assertEqual(main_spec.page_size_bytes, 16 * (512 + 64) * 2)
        self.assertEqual(indexer_spec.page_size_bytes, 2 * 16 * (128 + 2))
        self.assertFalse(hasattr(main_spec, "sfa_dcp_replicated_indexer_size"))

    @patch(
        "vllm_ascend.worker.model_runner_v1.get_gva_layerwise_config",
        return_value={"layerwise_num_shared_buffers": 2},
    )
    def test_sparse_c8_layer_reuse_allocates_four_complete_physical_slots(self, _mock_config):
        runner = self._build_runner()
        runner.use_sparse = True
        runner.parallel_config = MagicMock()
        runner.model_config.get_num_layers.return_value = 6

        main_spec = AscendMLAAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.int8,
            cache_sparse_c8=True,
        )
        indexer_spec = AscendSFAIndexerCacheSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.int8,
            scale_dim=1,
            scale_dtype=torch.float16,
            cache_sparse_c8=True,
        )
        main_names = [f"model.layers.{layer}.self_attn.attn" for layer in range(6)]
        indexer_names = [f"model.layers.{layer}.self_attn.indexer.k_cache" for layer in (1, 2, 4)]
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[
                *[KVCacheTensor(size=main_spec.page_size_bytes * 2, shared_by=[name]) for name in main_names],
                *[KVCacheTensor(size=indexer_spec.page_size_bytes * 2, shared_by=[name]) for name in indexer_names],
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=main_names, kv_cache_spec=main_spec),
                KVCacheGroupSpec(layer_names=indexer_names, kv_cache_spec=indexer_spec),
            ],
        )

        runner._merge_kv_cache_tensors_for_layer_reuse(kv_cache_config)

        self.assertEqual(runner._layerwise_reuse_num_slots, 4)
        self.assertEqual(
            runner._layerwise_reuse_slot_by_layer_name[main_names[1]],
            runner._layerwise_reuse_slot_by_layer_name[main_names[3]],
        )
        self.assertEqual(
            runner._layerwise_reuse_slot_by_layer_name[indexer_names[1]],
            runner._layerwise_reuse_slot_by_layer_name[indexer_names[2]],
        )

        raw_caches = runner._allocate_kv_cache_tensors(kv_cache_config)

        self.assertEqual(len(runner._layerwise_c8_indexer_slot_buffers), 4)
        self.assertTrue(all(len(raw_cache) == 2 for raw_cache in runner._layerwise_c8_indexer_slot_buffers))
        self.assertIs(raw_caches[main_names[1]][0], raw_caches[main_names[3]][0])
        self.assertIs(raw_caches[indexer_names[1]], raw_caches[indexer_names[2]])

    @patch(
        "vllm_ascend.worker.model_runner_v1.get_gva_layerwise_config",
        return_value={"layerwise_num_shared_buffers": 2},
    )
    def test_multigroup_layer_reuse_rejects_non_sfa_cache_with_indexer(self, _mock_config):
        runner = self._build_runner()
        runner.parallel_config = MagicMock()
        runner.model_config.get_num_layers.return_value = 5

        main_spec = AscendMLAAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.bfloat16,
        )
        indexer_spec = AscendSFAIndexerCacheSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.bfloat16,
        )
        unsupported_spec = FullAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.bfloat16,
        )
        main_names = [f"model.layers.{layer}.self_attn.attn" for layer in range(4)]
        indexer_name = "model.layers.2.self_attn.indexer.k_cache"
        unsupported_name = "model.layers.4.self_attn.attn"
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[
                *[KVCacheTensor(size=main_spec.page_size_bytes * 2, shared_by=[name]) for name in main_names],
                KVCacheTensor(size=indexer_spec.page_size_bytes * 2, shared_by=[indexer_name]),
                KVCacheTensor(size=unsupported_spec.page_size_bytes * 2, shared_by=[unsupported_name]),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=main_names, kv_cache_spec=main_spec),
                KVCacheGroupSpec(layer_names=[indexer_name], kv_cache_spec=indexer_spec),
                KVCacheGroupSpec(layer_names=[unsupported_name], kv_cache_spec=unsupported_spec),
            ],
        )

        with self.assertRaisesRegex(NotImplementedError, "unsupported cache specs"):
            runner._merge_kv_cache_tensors_for_layer_reuse(kv_cache_config)

    @patch(
        "vllm_ascend.worker.model_runner_v1.get_gva_layerwise_config",
        return_value={"layerwise_num_shared_buffers": 2},
    )
    def test_multigroup_layer_reuse_accepts_non_c8_sfa_main_indexer(self, _mock_config):
        runner = self._build_runner()
        runner.parallel_config = MagicMock()
        runner.model_config.get_num_layers.return_value = 5

        main_spec = AscendMLAAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.bfloat16,
        )
        indexer_spec = AscendSFAIndexerCacheSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.bfloat16,
        )
        main_names = [f"model.layers.{layer}.self_attn.attn" for layer in range(5)]
        indexer_name = "model.layers.2.self_attn.indexer.k_cache"
        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[
                *[KVCacheTensor(size=main_spec.page_size_bytes * 2, shared_by=[name]) for name in main_names],
                KVCacheTensor(size=indexer_spec.page_size_bytes * 2, shared_by=[indexer_name]),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=main_names, kv_cache_spec=main_spec),
                KVCacheGroupSpec(layer_names=[indexer_name], kv_cache_spec=indexer_spec),
            ],
        )

        runner._merge_kv_cache_tensors_for_layer_reuse(kv_cache_config)

        self.assertEqual(runner._layerwise_reuse_num_slots, 4)
        self.assertEqual(
            runner._layerwise_reuse_slot_by_layer_name[main_names[1]],
            runner._layerwise_reuse_slot_by_layer_name[main_names[3]],
        )
        self.assertIn(indexer_name, runner._layerwise_reuse_slot_by_layer_name)

    @patch("vllm_ascend.worker.model_runner_v1.get_ascend_device_type", return_value=AscendDeviceType.A5)
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
        runner.ascend_config.is_sparse_c8_layer.return_value = True
        runner.model_config.hf_text_config = SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
        )
        runner.vllm_config.cache_config.cache_dtype = "auto"

        attn_module = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(attn_module)
        attn_module.impl = SimpleNamespace(
            has_indexer=True,
            use_sparse_c8_sfa=True,
            use_sparse_c8_indexer=True,
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

        self.assertEqual(
            runner.ascend_config.is_sparse_c8_layer.call_args_list,
            [call(indexer_layer_name)],
        )
        self.assertEqual(main_spec.head_size, 512 + 64 * 2 + 4 * 4)
        self.assertEqual(main_spec.dtype, torch.float8_e4m3fn)
        self.assertEqual(main_spec.page_size_bytes, 16 * (512 + 64 * 2 + 4 * 4))
        self.assertEqual(indexer_spec.dtype, torch.float8_e4m3fn)
        self.assertEqual(indexer_spec.scale_dtype, torch.float32)
        self.assertEqual(indexer_spec.page_size_bytes, 16 * (128 + 4))

    def test_deepseek_v4_indexer_keeps_compressed_mla_layout(self):
        runner = self._build_runner()
        runner.use_compress = True
        layer_name = "model.layers.1.self_attn.indexer.k_cache"
        indexer_spec = AscendMLAAttentionSpec(
            block_size=16,
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
        runner.pcp_size = 1
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
            num_pcp_pads=None,
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
            scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(need_timing=False))
        )
        runner.execute_model_state = None
        runner.speculative_config = None
        runner.use_async_scheduling = False
        runner.num_spec_tokens = 0
        runner._draft_token_ids = None
        runner.pcp_size = 1
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
            scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(need_timing=False))
        )
        runner.execute_model_state = None
        runner.speculative_config = None
        runner.use_async_scheduling = False
        runner.num_spec_tokens = 0
        runner._draft_token_ids = None
        runner.pcp_size = 1
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

        runner._start_dump_data.assert_called_once_with()


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
