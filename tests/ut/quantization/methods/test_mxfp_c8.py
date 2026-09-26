from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import vllm_ascend.attention.attention_c8_mxfp as mxfp_kv_cache
from tests.ut.base import TestBase
from vllm_ascend.attention.attention_c8_mxfp import (
    MXFP8_GROUP_SIZE,
    MXFP_K_SCALE_NZ_TOKEN_FRAG,
    MXFP_KV_SCALE_GROUP_SIZE,
    fill_mxfp_v_scale_cache,
    mxfp_k_scale_cache_shape,
    mxfp_k_scale_page_bytes,
    mxfp_k_scale_slot_index,
    mxfp_v_scale_cache_shape,
    mxfp_v_scale_page_bytes,
    scatter_mxfp_k_scale_cache,
    scatter_mxfp_pa_nz_kv_cache,
)
from vllm_ascend.quantization.methods.kv_cache.mxfp_c8 import (
    AscendC8MXFPKVCacheAttentionMethod,
    _quant_weight_loader,
)


class _RecordOps(TorchDispatchMode):
    """Record the aten ops a block dispatches, views aside.

    Every recorded op is a device kernel on the NPU -- one line in a profile,
    one node in a captured graph -- whereas views launch nothing.
    """

    _VIEWS = frozenset(
        {"view", "_unsafe_view", "reshape", "slice", "select", "expand", "unsqueeze", "squeeze", "alias", "detach"}
    )

    def __init__(self):
        super().__init__()
        self.kernels: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func.overloadpacket.__name__
        if name not in self._VIEWS:
            self.kernels.append(name)
        return func(*args, **(kwargs or {}))


class TestMXFPScaleCacheShapes(TestBase):
    """Shape/byte-budget formulas for the C8-MXFP E8M0 scale caches."""

    def test_k_scale_cache_shape_d256_bs512(self):
        # PA_NZ: [Bn, N, Bs//16, D//64, 16, 2] (golden quant_flash_attn_golden.py).
        shape = mxfp_k_scale_cache_shape(num_blocks=8, block_size=512, num_kv_heads=4, head_dim=256)
        self.assertEqual(shape, (8, 4, 32, 4, 16, 2))

    def test_v_scale_cache_shape_d256_bs512(self):
        # PA_NZ: [Bn, N, D//16, Bs//64, 16, 2].
        shape = mxfp_v_scale_cache_shape(num_blocks=8, block_size=512, num_kv_heads=4, head_dim=256)
        self.assertEqual(shape, (8, 4, 16, 8, 16, 2))

    def test_scale_page_bytes_are_equal_for_k_and_v(self):
        k_bytes = mxfp_k_scale_page_bytes(num_kv_heads=4, block_size=512, head_dim=256)
        v_bytes = mxfp_v_scale_page_bytes(num_kv_heads=4, block_size=512, head_dim=256)
        self.assertEqual(k_bytes, 4 * 512 * 256 // MXFP8_GROUP_SIZE)
        self.assertEqual(k_bytes, v_bytes)

    def test_head_dim_must_align_to_scale_group(self):
        with self.assertRaises(ValueError):
            mxfp_k_scale_cache_shape(num_blocks=1, block_size=512, num_kv_heads=1, head_dim=100)


class TestScatterMXFPPaNzKvCache(TestBase):
    """PA_NZ KV scatter hands npu_scatter_pa_kv_cache the FIA C8 contract.

    torch_npu is a MagicMock here, so what is checkable on CPU is the call
    contract -- the shapes, dtypes and slot handling the operator is given.
    That is also the whole substance of the change: the operator itself is
    validated on-device by the FIA C8 path, which makes this exact call.
    """

    BLOCK_SIZE = 4
    NUM_KV_HEADS = 2
    HEAD_DIM = 64  # D//32 = 2 fragments
    NUM_BLOCKS = 2

    def setUp(self):
        self.key_cache = torch.zeros(
            (self.NUM_BLOCKS, self.BLOCK_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM),
            dtype=torch.uint8,
        )
        self.value_cache = torch.zeros_like(self.key_cache)

    def _scatter(self, num_tokens=3, slots=None, dtype=torch.uint8):
        key = (
            torch.arange(num_tokens * self.NUM_KV_HEADS * self.HEAD_DIM, dtype=torch.int32)
            .remainder(251)
            .to(torch.uint8)
            .reshape(num_tokens, self.NUM_KV_HEADS, self.HEAD_DIM)
        )
        if dtype != torch.uint8:
            key = key.view(dtype)
        if slots is None:
            slots = torch.tensor([2, 5, -1][:num_tokens], dtype=torch.int64)
        with patch.object(mxfp_kv_cache.torch_npu, "npu_scatter_pa_kv_cache") as op:
            scatter_mxfp_pa_nz_kv_cache(key, key.clone(), self.key_cache, self.value_cache, slots, self.BLOCK_SIZE)
        return op, key, slots

    def test_caches_are_passed_in_the_nz_five_d_view(self):
        op, _, _ = self._scatter()
        op.assert_called_once()
        for name in ("key_cache", "value_cache"):
            cache = op.call_args.kwargs[name]
            self.assertEqual(
                tuple(cache.shape),
                (
                    self.NUM_BLOCKS,
                    self.NUM_KV_HEADS,
                    self.HEAD_DIM // 32,
                    self.BLOCK_SIZE,
                    32,
                ),
                f"{name} must reach the operator as (Bn, KV_N, D/32, Bs, 32)",
            )

    def test_pa_nz_cache_mode_is_declared(self):
        # Scenario 1 of the ScatterPaKvCache contract is selected by
        # cache_mode; without it the operator reads the caches as "Norm"
        # ([num_blocks, block_size, num_head, head_size]) and rejects the
        # NZ axis order on dim2.
        op, _, _ = self._scatter()
        self.assertEqual(op.call_args.kwargs["cache_mode"], "PA_NZ")

    def test_payload_keeps_its_token_row_shape(self):
        op, key, _ = self._scatter()
        for name in ("key", "value"):
            self.assertEqual(tuple(op.call_args.kwargs[name].shape), tuple(key.shape))

    def test_everything_reaches_the_operator_as_one_byte_int8(self):
        # The FIA C8 path feeds int8; erasing the dtype here is what keeps the
        # FP8 payload out of the operator's type check.
        op, _, _ = self._scatter(dtype=torch.float8_e4m3fn)
        for name in ("key", "value", "key_cache", "value_cache"):
            self.assertEqual(op.call_args.kwargs[name].dtype, torch.int8, name)

    def test_negative_slots_are_left_for_the_operator(self):
        # No clamp, no filtering: the operator skips PAD_SLOT_ID itself, and
        # keeping the tensor untouched is what keeps shapes static under
        # graph capture.
        op, _, slots = self._scatter()
        passed = op.call_args.kwargs["slot_mapping"]
        self.assertIs(passed, slots)
        self.assertTrue(bool((passed < 0).any()), "fixture should include a padded row")

    def test_empty_batch_does_not_call_the_operator(self):
        with patch.object(mxfp_kv_cache.torch_npu, "npu_scatter_pa_kv_cache") as op:
            scatter_mxfp_pa_nz_kv_cache(
                torch.zeros(0, self.NUM_KV_HEADS, self.HEAD_DIM, dtype=torch.uint8),
                torch.zeros(0, self.NUM_KV_HEADS, self.HEAD_DIM, dtype=torch.uint8),
                self.key_cache,
                self.value_cache,
                torch.zeros(0, dtype=torch.int64),
                self.BLOCK_SIZE,
            )
        op.assert_not_called()

    def test_nz_view_places_a_token_at_its_fragment_coordinates(self):
        # Pure indexing math, independent of the operator: channel c of a
        # token at in-block offset o lives at [block, head, c//32, o, c%32].
        cache = torch.zeros((self.NUM_BLOCKS, self.BLOCK_SIZE, self.NUM_KV_HEADS, self.HEAD_DIM), dtype=torch.uint8)
        nz = cache.view(self.NUM_BLOCKS, self.NUM_KV_HEADS, self.HEAD_DIM // 32, self.BLOCK_SIZE, 32)
        nz[1, 0, 1, 2, 5] = 42
        flat = cache.reshape(-1)
        expected = (((1 * self.NUM_KV_HEADS + 0) * (self.HEAD_DIM // 32) + 1) * self.BLOCK_SIZE + 2) * 32 + 5
        self.assertEqual(flat[expected].item(), 42)
        self.assertEqual(int((flat != 0).sum()), 1)


class TestFillMXFPVScaleCache(TestBase):
    """V's static per-channel scale, spread over its whole paged cache.

    Unlike K's, this scale never changes at inference, so it is written once
    at KV cache setup instead of being scattered per step. Every block, every
    token group and both halves of the even/odd pair get the same
    (kv head, channel) byte.
    """

    NUM_BLOCKS = 2
    NUM_KV_HEADS = 2
    BLOCK_SIZE = 128

    def _fill(self, head_dim):
        cache = torch.zeros(
            mxfp_v_scale_cache_shape(self.NUM_BLOCKS, self.BLOCK_SIZE, self.NUM_KV_HEADS, head_dim),
            dtype=torch.uint8,
        )
        # Distinct byte per (kv head, channel) so a transposed or collapsed
        # axis cannot pass by accident, and none of them zero so an untouched
        # slot stays distinguishable from a written one.
        value_scale = (
            torch.arange(self.NUM_KV_HEADS * head_dim, dtype=torch.int32).remainder(251).add(1).to(torch.uint8)
        )
        fill_mxfp_v_scale_cache(value_scale, cache)
        return value_scale, cache

    def test_every_slot_carries_its_channel_scale(self):
        head_dim = 64
        value_scale, cache = self._fill(head_dim)
        num_token_groups = cache.shape[3]
        for block in range(self.NUM_BLOCKS):
            for token_group in range(num_token_groups):
                for half in range(2):
                    self.assertEqual(
                        cache[block, :, :, token_group, :, half].reshape(-1).tolist(),
                        value_scale.tolist(),
                    )

    def test_the_cache_supplies_the_head_dim(self):
        # A model whose V head dim differs from Q/K's must still land
        # correctly, which is why the shapes are read off the cache.
        value_scale, cache = self._fill(32)
        self.assertEqual(cache.shape[2], 32 // 16)
        self.assertEqual(cache[0, :, :, 0, :, 0].reshape(-1).tolist(), value_scale.tolist())

    def test_no_slot_is_left_untouched(self):
        # Zero is the failure signature of the bug this replaced: a scale
        # cache that stayed at its allocation value dequantizes V to ~0 and
        # attention returns exactly zero. The fixture has no zero scales, so
        # any zero left in the cache is a slot the fill missed.
        _, cache = self._fill(64)
        self.assertTrue(bool((cache != 0).all()))


class TestScatterMXFPKScaleCache(TestBase):
    """Scatter writes valid slots and parks padded (-1) rows in the null block.

    Slot 0 is block 0, which vLLM's BlockPool reserves as the null block and
    never hands to a request, so nothing a real token owns lives there. Parking
    padded rows on it is what lets the scatter be one write per layer instead
    of a read, a select and a write.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.block_size = 512
        self.num_kv_heads = 2
        self.head_dim = MXFP_KV_SCALE_GROUP_SIZE
        self.key_scale_cache = torch.zeros(
            (
                2,
                self.num_kv_heads,
                self.block_size // MXFP_K_SCALE_NZ_TOKEN_FRAG,
                self.head_dim // MXFP_KV_SCALE_GROUP_SIZE,
                MXFP_K_SCALE_NZ_TOKEN_FRAG,
                2,
            ),
            dtype=torch.uint8,
        )

    def _at(self, slot):
        """(block, seg, frag) coordinates of a slot in the PA_NZ cache."""
        block, offset = slot // self.block_size, slot % self.block_size
        return block, offset // MXFP_K_SCALE_NZ_TOKEN_FRAG, offset % MXFP_K_SCALE_NZ_TOKEN_FRAG

    def _scatter(self, key_scale, slot_mapping, cache=None):
        cache = self.key_scale_cache if cache is None else cache
        scatter_mxfp_k_scale_cache(key_scale, cache, mxfp_k_scale_slot_index(slot_mapping, self.block_size))
        return cache

    def _read_modify_write_reference(self, key_scale, slot_mapping, cache):
        """The scatter this replaced: padded rows rewrite what they read."""
        slots = slot_mapping.to(torch.long)
        valid = slots >= 0
        safe = torch.where(valid, slots, torch.zeros_like(slots))
        block, offset = safe // self.block_size, safe % self.block_size
        seg, frag = offset // MXFP_K_SCALE_NZ_TOKEN_FRAG, offset % MXFP_K_SCALE_NZ_TOKEN_FRAG
        cached = cache[block, :, seg, :, frag, :]
        cache[block, :, seg, :, frag, :] = torch.where(valid.view(-1, 1, 1, 1), key_scale, cached)
        return cache

    def _without_null_slot(self, cache):
        cache = cache.clone()
        block, seg, frag = self._at(0)
        cache[block, :, seg, :, frag] = 0
        return cache

    def test_scatter_valid_and_padded_slots(self):
        key_scale = torch.full((3, self.num_kv_heads, 1, 2), 130, dtype=torch.uint8)
        key_scale[2] = 77  # the padded row's own (meaningless) scale
        # slot 2 -> block 0, offset 2; slot 513 -> block 1, offset 1; -1 -> padding.
        self._scatter(key_scale, torch.tensor([2, 513, -1], dtype=torch.int64))

        block, seg, frag = self._at(2)
        self.assertTrue(torch.all(self.key_scale_cache[block, :, seg, :, frag] == 130))
        block, seg, frag = self._at(513)
        self.assertTrue(torch.all(self.key_scale_cache[block, :, seg, :, frag] == 130))
        # The padded row is parked on the null block's slot 0 ...
        block, seg, frag = self._at(0)
        self.assertTrue(torch.all(self.key_scale_cache[block, :, seg, :, frag] == 77))
        # ... and neighbours of every written slot stay untouched.
        for slot in (1, 3, 512, 514):
            block, seg, frag = self._at(slot)
            self.assertTrue(torch.all(self.key_scale_cache[block, :, seg, :, frag] == 0))

    def test_scatter_all_padding_rows_touch_only_the_null_slot(self):
        """A pure-padding batch (all -1, e.g. a dummy run) must leave every
        slot a request can own alone."""
        key_scale = torch.full((2, self.num_kv_heads, 1, 2), 130, dtype=torch.uint8)
        self._scatter(key_scale, torch.tensor([-1, -1], dtype=torch.int64))
        self.assertTrue(torch.all(self._without_null_slot(self.key_scale_cache) == 0))

    def test_scatter_padding_and_real_at_nonzero_slot_coexist(self):
        key_scale = torch.zeros((2, self.num_kv_heads, 1, 2), dtype=torch.uint8)
        key_scale[0] = 200  # real token at slot 3
        key_scale[1] = 77  # padding row
        self._scatter(key_scale, torch.tensor([3, -1], dtype=torch.int64))

        block, seg, frag = self._at(3)
        self.assertTrue(torch.all(self.key_scale_cache[block, :, seg, :, frag] == 200))
        expected = torch.zeros_like(self.key_scale_cache)
        expected[block, :, seg, :, frag] = 200
        self.assertTrue(torch.equal(self._without_null_slot(self.key_scale_cache), expected))

    # The two phases are judged separately on purpose: they do not get the
    # same guarantee, and only one of them ever sees a padded row.

    def test_prefill_shaped_batch_is_bit_identical_to_the_read_modify_write(self):
        # Eager batches are sliced to num_actual_tokens, so they carry no -1
        # rows and the old read + select was an identity there.
        slots = torch.randperm(2 * self.block_size)[:300]
        key_scale = torch.randint(1, 255, (300, self.num_kv_heads, 1, 2), dtype=torch.uint8)
        new = self._scatter(key_scale, slots, torch.zeros_like(self.key_scale_cache))
        old = self._read_modify_write_reference(key_scale, slots, torch.zeros_like(self.key_scale_cache))
        self.assertTrue(torch.equal(new, old))

    def test_padded_decode_batch_differs_only_in_the_null_slot(self):
        # FULL-graph replays pad the batch up to the captured size with -1.
        # Real slots start past block 0: the pool never hands block 0 out.
        real = self.block_size + torch.randperm(self.block_size)[:5]
        slots = torch.cat([real, torch.full((11,), -1)])
        key_scale = torch.randint(1, 255, (16, self.num_kv_heads, 1, 2), dtype=torch.uint8)
        new = self._scatter(key_scale, slots, torch.zeros_like(self.key_scale_cache))
        old = self._read_modify_write_reference(key_scale, slots, torch.zeros_like(self.key_scale_cache))
        self.assertFalse(torch.equal(new, old))
        self.assertTrue(torch.equal(self._without_null_slot(new), self._without_null_slot(old)))

    def test_scatter_is_one_write_and_no_read(self):
        # The point of parking padded rows: this runs once per full-attention
        # layer per step, and used to be a gather, a select and a write.
        slot_index = mxfp_k_scale_slot_index(torch.tensor([2, 513, -1]), self.block_size)
        key_scale = torch.full((3, self.num_kv_heads, 1, 2), 130, dtype=torch.uint8)
        with _RecordOps() as ops:
            scatter_mxfp_k_scale_cache(key_scale, self.key_scale_cache, slot_index)
        self.assertEqual([op for op in ops.kernels if "index_put" in op], ["index_put_"])
        self.assertEqual([op for op in ops.kernels if op in ("index", "where")], [])

    def test_slot_index_needs_no_validity_mask(self):
        slots = torch.tensor([2, 513, -1], dtype=torch.int32)
        with _RecordOps() as ops:
            block_ids, seg_ids, frag_ids = mxfp_k_scale_slot_index(slots, self.block_size)
        self.assertEqual(block_ids.tolist(), [0, 1, 0])
        self.assertEqual(seg_ids.tolist(), [0, 0, 0])
        self.assertEqual(frag_ids.tolist(), [2, 1, 0])
        self.assertEqual([op for op in ops.kernels if op in ("ge", "where", "zeros_like")], [])
        self.assertLessEqual(len(ops.kernels), 6)


class TestAscendC8MXFPKVCacheAttentionMethod(TestBase):
    """Quant-method wiring: v_cache_scale fallback and backend installation."""

    def _make_layer(self, with_impl: bool = False):
        from vllm_ascend.attention.attention_c8_mxfp import AscendC8MXFPAttentionBackendImpl

        layer = torch.nn.Module()
        layer.num_kv_heads = 2
        layer.head_size_v = 4
        if with_impl:
            layer.impl = object.__new__(AscendC8MXFPAttentionBackendImpl.__base__)
        return layer

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

        full = _torch.cat([_torch.full((256,), _head_val(h), dtype=_torch.uint8) for h in range(2)])
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

    def test_missing_v_scale_uses_e8m0_unity_default(self):
        method = AscendC8MXFPKVCacheAttentionMethod.__new__(AscendC8MXFPKVCacheAttentionMethod)
        layer = torch.nn.Module()
        layer.num_kv_heads = 2
        layer.head_size_v = 4
        method.create_weights(layer)
        # The missing-weight fallback: E8M0 127 is the neutral scale of 1.0.
        self.assertEqual(layer.v_cache_scale.dtype, torch.uint8)
        self.assertTrue(torch.equal(layer.v_cache_scale, torch.full((8,), 127, dtype=torch.uint8)))

        vllm_config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))
        with patch(
            "vllm_ascend.quantization.methods.kv_cache.mxfp_c8.get_current_vllm_config",
            return_value=vllm_config,
        ):
            method.process_weights_after_loading(layer)

        # The all-127 fallback is a neutral scale of 1.0, so is its reciprocal.
        self.assertTrue(torch.equal(layer.v_cache_scale_float_reciprocal, torch.ones(8, dtype=torch.bfloat16)))

    def test_installs_c8_backend_with_512_token_blocks(self):
        from vllm_ascend.attention.attention_c8_mxfp import (
            AscendC8MXFPAttentionBackend,
            AscendC8MXFPAttentionBackendImpl,
        )
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackend

        method = AscendC8MXFPKVCacheAttentionMethod({}, prefix="model.layers.3")
        layer = self._make_layer(with_impl=True)

        method.create_weights(layer)

        self.assertIs(layer.attn_backend, AscendC8MXFPAttentionBackend)
        self.assertIsInstance(layer.impl, AscendC8MXFPAttentionBackendImpl)
        self.assertFalse(layer.impl.enable_hamming_sparse)
        self.assertEqual(AscendAttentionBackend.get_supported_kernel_block_sizes(), [128])
        self.assertEqual(AscendC8MXFPAttentionBackend.get_supported_kernel_block_sizes(), [512])


class TestQuantWeightLoader(TestBase):
    """How the static V-cache scale is spread over TP ranks.

    The scale is one E8M0 byte per (kv_head, channel), so the split is counted
    in KV heads. Dividing the tensor by tp_size instead breaks in both
    directions: a checkpoint that shares one set of channel scales across heads
    has nothing to divide, and a model whose total_kv_heads is below tp_size
    replicates a KV head rather than sharding it.
    """

    HEAD_SIZE_V = 4

    def _load(self, *, param_heads, ckpt_heads, tp_rank, tp_size):
        head = self.HEAD_SIZE_V
        param = torch.full((param_heads * head,), 127, dtype=torch.uint8)
        # Byte value encodes the head index, so the result names its source.
        loaded = torch.arange(ckpt_heads * head, dtype=torch.int32).div(head, rounding_mode="floor")
        with (
            patch(
                "vllm_ascend.quantization.methods.kv_cache.mxfp_c8.get_tensor_model_parallel_rank",
                return_value=tp_rank,
            ),
            patch(
                "vllm_ascend.quantization.methods.kv_cache.mxfp_c8.get_tensor_model_parallel_world_size",
                return_value=tp_size,
            ),
        ):
            _quant_weight_loader(param, loaded.to(torch.uint8))
        return param[::head].tolist()

    def test_shared_scale_is_tiled_over_kv_heads(self):
        # One set of channel scales for every KV head: TP=1 has to see it twice.
        self.assertEqual(self._load(param_heads=2, ckpt_heads=1, tp_rank=0, tp_size=1), [0, 0])

    def test_shared_scale_survives_tp_split(self):
        # Each rank holds one head, and both heads use the same set.
        for rank in range(2):
            self.assertEqual(self._load(param_heads=1, ckpt_heads=1, tp_rank=rank, tp_size=2), [0])

    def test_per_head_scale_is_sharded_by_rank(self):
        self.assertEqual(self._load(param_heads=1, ckpt_heads=2, tp_rank=0, tp_size=2), [0])
        self.assertEqual(self._load(param_heads=1, ckpt_heads=2, tp_rank=1, tp_size=2), [1])

    def test_per_head_scale_follows_replicated_kv_heads(self):
        # total_kv_heads=2 under TP=4: ranks 0/1 share head 0, ranks 2/3 head 1.
        self.assertEqual(
            [self._load(param_heads=1, ckpt_heads=2, tp_rank=r, tp_size=4)[0] for r in range(4)], [0, 0, 1, 1]
        )

    def test_trailing_axis_from_modelslim_is_accepted(self):
        param = torch.full((8,), 127, dtype=torch.uint8)
        loaded = torch.full((8, 1), 130, dtype=torch.uint8)
        with (
            patch(
                "vllm_ascend.quantization.methods.kv_cache.mxfp_c8.get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch(
                "vllm_ascend.quantization.methods.kv_cache.mxfp_c8.get_tensor_model_parallel_world_size",
                return_value=1,
            ),
        ):
            _quant_weight_loader(param, loaded)
        self.assertTrue(torch.equal(param, torch.full((8,), 130, dtype=torch.uint8)))
