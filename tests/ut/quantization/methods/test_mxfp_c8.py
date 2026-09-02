import torch

from tests.ut.base import TestBase
from vllm_ascend.device.mxfp_kv_cache import (
    MXFP8_GROUP_SIZE,
    MXFP_KV_SCALE_GROUP_SIZE,
    mxfp_k_scale_cache_shape,
    mxfp_k_scale_page_bytes,
    mxfp_v_scale_cache_shape,
    mxfp_v_scale_page_bytes,
    scatter_mxfp_k_scale_cache,
)
from vllm_ascend.quantization.methods.kv_cache.mxfp_c8 import AscendC8MXFPKVCacheAttentionMethod


class TestMXFPScaleCacheShapes(TestBase):
    """Shape/byte-budget formulas for the C8-MXFP E8M0 scale caches."""

    def test_k_scale_cache_shape_d256_bs512(self):
        # PA_BBND: block axis before the head axis, matching the K/V caches.
        shape = mxfp_k_scale_cache_shape(num_blocks=8, block_size=512, num_kv_heads=4, head_dim=256)
        self.assertEqual(shape, (8, 512, 4, 4, 2))

    def test_v_scale_cache_shape_d256_bs512(self):
        shape = mxfp_v_scale_cache_shape(num_blocks=8, block_size=512, num_kv_heads=4, head_dim=256)
        self.assertEqual(shape, (8, 8, 4, 256, 2))

    def test_scale_page_bytes_are_equal_for_k_and_v(self):
        k_bytes = mxfp_k_scale_page_bytes(num_kv_heads=4, block_size=512, head_dim=256)
        v_bytes = mxfp_v_scale_page_bytes(num_kv_heads=4, block_size=512, head_dim=256)
        self.assertEqual(k_bytes, 4 * 512 * 256 // MXFP8_GROUP_SIZE)
        self.assertEqual(k_bytes, v_bytes)

    def test_head_dim_must_align_to_scale_group(self):
        with self.assertRaises(ValueError):
            mxfp_k_scale_cache_shape(num_blocks=1, block_size=512, num_kv_heads=1, head_dim=100)


class TestScatterMXFPKScaleCache(TestBase):
    """Scatter writes valid slots and turns padded (-1) slots into no-ops."""

    def setUp(self):
        torch.manual_seed(0)
        self.block_size = 512
        self.num_kv_heads = 2
        self.head_dim = MXFP_KV_SCALE_GROUP_SIZE
        self.key_scale_cache = torch.zeros(
            (2, self.block_size, self.num_kv_heads, self.head_dim // MXFP_KV_SCALE_GROUP_SIZE, 2),
            dtype=torch.uint8,
        )

    def test_scatter_valid_and_padded_slots(self):
        key_scale = torch.full((3, self.num_kv_heads, 1, 2), 130, dtype=torch.uint8)
        # slot 0 -> block 0, offset 0; slot 513 -> block 1, offset 1; -1 -> padding.
        slot_mapping = torch.tensor([0, 513, -1], dtype=torch.int64)

        scatter_mxfp_k_scale_cache(key_scale, self.key_scale_cache, slot_mapping, self.block_size)

        self.assertTrue(torch.all(self.key_scale_cache[0, 0] == 130))
        self.assertTrue(torch.all(self.key_scale_cache[1, 1] == 130))
        # Untouched positions stay zero, including the padding remap target.
        self.assertTrue(torch.all(self.key_scale_cache[0, 1] == 0))
        self.assertTrue(torch.all(self.key_scale_cache[1, 0] == 0))

    def test_scatter_padding_does_not_clobber_real_slot(self):
        """A padded (-1) row must never overwrite a real token written to the
        same cache slot in the same batch (regression: the old remap-to-slot-0
        trick wrote the stale value back over the real token's scale)."""
        key_scale = torch.zeros((2, self.num_kv_heads, 1, 2), dtype=torch.uint8)
        key_scale[0] = 200  # real token
        key_scale[1] = 77  # padding row (dropped)
        slot_mapping = torch.tensor([0, -1], dtype=torch.int64)

        scatter_mxfp_k_scale_cache(key_scale, self.key_scale_cache, slot_mapping, self.block_size)

        self.assertTrue(torch.all(self.key_scale_cache[0, 0] == 200))

    def test_scatter_all_padding_rows_are_no_op(self):
        key_scale = torch.full((2, self.num_kv_heads, 1, 2), 130, dtype=torch.uint8)
        slot_mapping = torch.tensor([-1, -1], dtype=torch.int64)

        scatter_mxfp_k_scale_cache(key_scale, self.key_scale_cache, slot_mapping, self.block_size)

        self.assertTrue(torch.all(self.key_scale_cache == 0))


class TestAscendC8MXFPKVCacheAttentionMethod(TestBase):
    """Quant-method wiring: v_cache_scale fallback and backend installation."""

    def _make_layer(self, with_impl: bool = False):
        from vllm_ascend.attention.attention_v1 import AscendC8MXFPAttentionBackendImpl

        layer = torch.nn.Module()
        layer.num_kv_heads = 2
        layer.head_size_v = 4
        if with_impl:
            layer.impl = object.__new__(AscendC8MXFPAttentionBackendImpl.__base__)
        return layer

    def test_missing_v_scale_uses_e8m0_unity_default(self):
        method = AscendC8MXFPKVCacheAttentionMethod({}, prefix="model.layers.3")
        layer = self._make_layer()

        method.create_weights(layer)

        self.assertEqual(layer.v_cache_scale.dtype, torch.uint8)
        self.assertTrue(torch.equal(layer.v_cache_scale, torch.full((8,), 127, dtype=torch.uint8)))

    def test_weight_loader_accepts_column_vector_checkpoint_layout(self):
        """ModelSlim exports v_scale as [hidden, 1]; the parameter is 1-D.
        The loader must squeeze the trailing size-1 dims before comparing."""
        from vllm_ascend.quantization.methods.kv_cache.mxfp_c8 import _quant_weight_loader

        param = torch.full((512,), 127, dtype=torch.uint8)
        column_vector_weight = torch.full((512, 1), 119, dtype=torch.uint8)

        _quant_weight_loader(param, column_vector_weight)

        self.assertTrue(torch.equal(param, torch.full((512,), 119, dtype=torch.uint8)))

    def test_weight_loader_slices_replicated_heads_under_tp(self):
        """The loader delivers the FULL-width scale on every rank (attention
        params bypass ColumnParallelLinear sharding). Under GQA TP with
        num_kv_heads < tp_size, vLLM replicates each KV head across
        tp_size // num_kv_heads ranks; rank r owns the head-dim slice of
        head r // (tp_size // num_kv_heads). TP4 over 2 heads: ranks 0/1 ->
        head 0, ranks 2/3 -> head 1. When num_kv_heads == tp_size it
        degenerates to the plain contiguous narrow (TP2: rank 0 -> [0,256),
        rank 1 -> [256,512))."""
        from unittest.mock import patch

        import torch as _torch

        from vllm_ascend.quantization.methods.kv_cache.mxfp_c8 import _quant_weight_loader

        loader_mod = "vllm_ascend.quantization.methods.kv_cache.mxfp_c8"

        def _head_val(head: int) -> int:
            return 100 + head  # head 0 -> 100s, head 1 -> 101s...

        full = _torch.cat(
            [_torch.full((256,), _head_val(h), dtype=_torch.uint8) for h in range(2)]
        )
        param = _torch.zeros(256, dtype=_torch.uint8)

        # TP4 rank 3 -> head 1
        with (
            patch(f"{loader_mod}.get_tensor_model_parallel_rank", return_value=3),
            patch(f"{loader_mod}.get_tensor_model_parallel_world_size", return_value=4),
        ):
            _quant_weight_loader(param, full)
        self.assertTrue(_torch.equal(param, _torch.full((256,), 101, dtype=_torch.uint8)))

        # TP2 rank 1 -> head 1 (plain narrow case)
        with (
            patch(f"{loader_mod}.get_tensor_model_parallel_rank", return_value=1),
            patch(f"{loader_mod}.get_tensor_model_parallel_world_size", return_value=2),
        ):
            _quant_weight_loader(param, full)
        self.assertTrue(_torch.equal(param, _torch.full((256,), 101, dtype=_torch.uint8)))

        # TP4 rank 0 -> head 0
        with (
            patch(f"{loader_mod}.get_tensor_model_parallel_rank", return_value=0),
            patch(f"{loader_mod}.get_tensor_model_parallel_world_size", return_value=4),
        ):
            _quant_weight_loader(param, full)
        self.assertTrue(_torch.equal(param, _torch.full((256,), 100, dtype=_torch.uint8)))

    def test_installs_c8_backend_with_512_token_blocks(self):
        from vllm_ascend.attention.attention_v1 import (
            AscendAttentionBackend,
            AscendC8MXFPAttentionBackend,
            AscendC8MXFPAttentionBackendImpl,
        )

        method = AscendC8MXFPKVCacheAttentionMethod({}, prefix="model.layers.3")
        layer = self._make_layer(with_impl=True)

        method.create_weights(layer)

        self.assertIs(layer.attn_backend, AscendC8MXFPAttentionBackend)
        self.assertIsInstance(layer.impl, AscendC8MXFPAttentionBackendImpl)
        self.assertFalse(layer.impl.enable_hamming_sparse)
        self.assertEqual(layer.impl._v_scale_filled_caches, set())
        self.assertEqual(AscendAttentionBackend.get_supported_kernel_block_sizes(), [128])
        self.assertEqual(AscendC8MXFPAttentionBackend.get_supported_kernel_block_sizes(), [512])
