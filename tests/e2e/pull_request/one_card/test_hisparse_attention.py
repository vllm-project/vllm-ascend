# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the actual SFA bridge against an independent FP32 CPU reference."""

import math
from functools import partial
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch
import torch_npu


@pytest.fixture
def _require_npu_sfa():
    pytest.importorskip("vllm.v1.hisparse.backend")
    from vllm_ascend.utils import bootstrap_custom_op_env

    # Match platform initialization: publish the OPP path before device queries
    # or set_device can initialize the runtime and its tiling registry.
    bootstrap_custom_op_env()
    if not torch.npu.is_available():
        pytest.skip("HiSparse SFA tests require an NPU")
    from vllm_ascend.device.device_config import is_310p

    if is_310p():
        pytest.skip("The native SFA operation is not registered for Ascend 310P")
    torch.npu.set_device(0)
    # Load the required SFA extension without changing custom-op selection.
    import_module("vllm_ascend.vllm_ascend_C")


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("num_queries", [1, 7, 35])
@pytest.mark.usefixtures("_require_npu_sfa")
def test_sparse_attention_reference(num_queries, strided, capture):
    from vllm_ascend.attention.sfa_hisparse import _ATTENTION_CHUNK_ROWS, AscendHiSparseImpl

    torch.manual_seed(47)
    device = torch.device("npu:0")
    topk, latent_dim, rope_dim, heads = 2048, 512, 64, 8
    block_size, blocks = 128, 16
    capacity = _ATTENTION_CHUNK_ROWS
    scale = 1 / math.sqrt(latent_dim + rope_dim)
    workspace = (
        torch.empty((capacity, topk, 1, latent_dim), device=device, dtype=torch.bfloat16),
        torch.empty((capacity, topk, 1, rope_dim), device=device, dtype=torch.bfloat16),
        torch.arange(capacity * topk // block_size, dtype=torch.int32, device=device).view(capacity, -1),
        torch.arange(topk, dtype=torch.int32, device=device).view(1, 1, topk),
        torch.arange(1, capacity + 1, dtype=torch.int32, device=device),
    )
    impl = SimpleNamespace(kv_lora_rank=latent_dim, qk_rope_head_dim=rope_dim, scale=scale, enable_sparse_sfa_c8=False)
    impl._workspace = lambda dtype: workspace
    # Quantize before taking the FP32 reference so both paths see the same inputs.
    source = torch.randn((blocks, block_size, latent_dim + rope_dim)).to(torch.bfloat16)
    query = torch.randn((num_queries, heads, latent_dim)).to(torch.bfloat16)
    query_rope = torch.randn((num_queries, heads, rope_dim)).to(torch.bfloat16)
    physical = torch.randint(0, blocks * block_size, (num_queries, topk), dtype=torch.int32)
    physical[:, 3::7] = -1
    physical[:, 1] = physical[:, 0]
    # A short row makes duplicate multiplicity and invalid-padding mistakes large.
    physical[0] = -1
    physical[0, :3] = torch.tensor([0, 0, 1], dtype=torch.int32)
    if num_queries > 1:
        physical[-1] = -1
    if strided:
        backing = torch.empty((blocks, 2, block_size, latent_dim + rope_dim), device=device, dtype=source.dtype)
        cache = backing[:, 1]
        cache.copy_(source)
    else:
        cache = source.to(device)
    query_npu, rope_npu, physical_npu = query.to(device), query_rope.to(device), physical.to(device)
    forward = partial(AscendHiSparseImpl._attention_physical, impl, query_npu, rope_npu, cache, physical_npu)

    graph = None
    try:
        if capture:
            stream = torch.npu.Stream()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                for _ in range(2):
                    forward()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream):
                captured = forward()
            # Change both payload and selection length after graph capture.
            query.neg_()
            physical[0, :5] = torch.tensor([3, 3, 8, 12, 14], dtype=torch.int32)
            query_npu.copy_(query)
            physical_npu.copy_(physical)
            for _ in range(3):
                graph.replay()
            actual = captured.float().cpu()
        else:
            actual = forward().float().cpu()

        selected = source.flatten(0, 1)[physical.clamp_min(0).long()].float()
        logits = torch.einsum("thd,tkd->thk", query.float(), selected[..., :latent_dim])
        logits += torch.einsum("thd,tkd->thk", query_rope.float(), selected[..., latent_dim:])
        logits *= scale
        logits.masked_fill_(~(physical >= 0)[:, None, :], -torch.inf)
        probabilities = torch.softmax(logits, dim=-1).nan_to_num(0)
        expected = torch.einsum("thk,tkd->thd", probabilities, selected[..., :latent_dim])
        torch.testing.assert_close(actual, expected, atol=0.005, rtol=0.025)
        if num_queries > 1:
            assert torch.count_nonzero(actual[-1]) == 0
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("index_mode", ["virtual", "mapped", "mapped_lengths"])
@pytest.mark.parametrize("empty_batch", [False, True])
@pytest.mark.usefixtures("_require_npu_sfa")
def test_c8_attention_uses_existing_native_operator(strided, capture, index_mode, empty_batch):
    from vllm_ascend.device.device_config import is_950

    if not is_950():
        pytest.skip("This case uses the native A5 E4M3FN C8 payload; A2/A3 use an INT8 format")
    from vllm_ascend.attention.sfa_hisparse import AscendHiSparseImpl
    from vllm_ascend.attention.sfa_v1 import custom_kv_rmsnorm_rope
    from vllm_ascend.hisparse.c8 import c8_storage_view, pack_c8_components

    torch.manual_seed(71)
    rows, heads, blocks, block, topk = 3, 8, 2, 128, 128
    device = "npu:0"
    values = torch.randn(blocks * block, 1, 1, 576, dtype=torch.bfloat16, device=device)
    gamma = torch.ones(512, dtype=torch.bfloat16, device=device)
    cos = torch.ones(blocks * block, 1, 1, 64, dtype=torch.bfloat16, device=device)
    sin = torch.zeros_like(cos)
    rope, nope, scales = custom_kv_rmsnorm_rope(
        values, gamma, cos, sin, 512, 64, epsilon=1e-5, dst_type=torch.float8_e4m3fn, tile_size=128
    )
    packed = pack_c8_components(nope.contiguous(), rope.contiguous(), scales.contiguous())
    backing = torch.empty((blocks, 2 if strided else 1, block, 656), dtype=torch.float8_e4m3fn, device=device)
    cache = backing[:, -1]
    cache.view(torch.uint8).copy_(packed.reshape(blocks, block, 656))
    q = torch.randn(rows, heads, 512, dtype=torch.bfloat16, device=device) * 0.25
    qr = torch.randn(rows, heads, 64, dtype=torch.bfloat16, device=device) * 0.25
    indices = torch.randint(0, blocks * block, (rows, topk), dtype=torch.int32)
    indices[:, 3::5] = -1
    indices[0] = -1
    indices[0, :3] = torch.tensor([3, 3, 7], dtype=torch.int32)
    indices[-1] = -1
    indices_are_mapped = index_mode != "virtual"

    def selection():
        if not indices_are_mapped:
            return indices
        output = torch.full_like(indices, -1)
        for original, target in zip(indices.tolist(), output):
            values = [value // block * (cache.stride(0) // 656) + value % block for value in original if value >= 0]
            target[: len(values)] = torch.tensor(values, dtype=torch.int32)
        return output

    selected = selection().to(device)
    lengths = (
        (
            torch.arange(rows + 1, dtype=torch.int32, device=device)[rows:],
            torch.full((1,), c8_storage_view(cache)[0].shape[0], dtype=torch.int32, device=device),
        )
        if index_mode == "mapped_lengths"
        else None
    )
    query_end, key_length = (torch.empty(1, dtype=torch.int32, device=device) for _ in range(2))
    impl = AscendHiSparseImpl.__new__(AscendHiSparseImpl)
    impl.enable_sparse_sfa_c8 = True
    impl.kv_lora_rank = 512
    impl.scale = 1 / math.sqrt(576)
    impl._c8_length_workspace = lambda: (query_end, key_length)

    def forward():
        return impl._attention_physical(q, qr, cache, selected, indices_are_mapped=indices_are_mapped, lengths=lengths)

    graph = None
    try:
        if capture:
            forward()
            forward()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                output = forward()
            if empty_batch:
                indices.fill_(-1)
            else:
                indices[0, :4] = torch.tensor([9, 2, 9, 0], dtype=torch.int32)
            selected.copy_(selection())
            q.neg_()
            graph.replay()
        else:
            output = forward()
            if empty_batch:
                indices.fill_(-1)
                selected.copy_(selection())
                output = forward()
        torch.npu.synchronize()
        scale = scales.contiguous().view(torch.float32).cpu().reshape(-1, 4).repeat_interleave(128, dim=-1)
        latent = nope.float().cpu().reshape(-1, 512) * scale
        rope_cpu = rope.contiguous().view(torch.bfloat16).cpu().reshape(-1, 64).float()
        keys = latent[indices.clamp_min(0).long()]
        rope_keys = rope_cpu[indices.clamp_min(0).long()]
        logits = torch.einsum("thd,tkd->thk", q.float().cpu(), keys)
        logits += torch.einsum("thd,tkd->thk", qr.float().cpu(), rope_keys)
        logits *= 1 / math.sqrt(576)
        logits.masked_fill_(~(indices >= 0)[:, None, :], -torch.inf)
        probability = logits.softmax(-1).nan_to_num(0)
        expected = torch.einsum("thk,tkd->thd", probability, keys)
        torch.testing.assert_close(output.float().cpu(), expected, atol=0.01, rtol=0.05)
        assert torch.count_nonzero(output[-1].cpu()) == 0
        if empty_batch:
            assert torch.count_nonzero(output.cpu()) == 0
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()


@pytest.fixture
def _require_a5_preprocess():
    pytest.importorskip("vllm.v1.hisparse.backend")
    if not torch.npu.is_available() or not torch.npu.get_device_name(0).startswith("Ascend950"):
        pytest.skip("HiSparse integration currently targets Ascend 950")
    torch.npu.set_device(0)


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.usefixtures("_require_a5_preprocess")
def test_preprocess_writes_hisparse_mapping_once(cache_dtype, capture):
    from vllm_ascend.attention.sfa_hisparse import AscendHiSparseImpl
    from vllm_ascend.attention.sfa_v1 import custom_kv_rmsnorm_rope

    rows, block = 5, 128
    c8 = cache_dtype == torch.float8_e4m3fn
    width = 656 if c8 else 576
    torch.manual_seed(71)
    values = torch.randn(rows, 2624, dtype=torch.bfloat16, device="npu")[:, 2048:]
    gamma = torch.ones(512, dtype=torch.bfloat16, device="npu")
    cos = torch.ones(rows, 1, 1, 64, dtype=torch.bfloat16, device="npu")
    sin = torch.zeros_like(cos)
    raw = torch.full((3, 2, block, width * cache_dtype.itemsize), 90, dtype=torch.uint8, device="npu")
    destination = raw[:, 1].view(cache_dtype)
    logical = [-1, 129, 0, 257, 128]
    slots = torch.tensor(logical, dtype=torch.int64, device="npu")
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32, device="npu"), hisparse_for_capture=True
    )
    scratch = (
        torch.empty(1, block, 1, 512, dtype=values.dtype, device="npu"),
        torch.empty(1, block, 1, 64, dtype=values.dtype, device="npu"),
        torch.arange(block, dtype=torch.int64, device="npu"),
    )
    impl = AscendHiSparseImpl.__new__(AscendHiSparseImpl)
    impl.enable_sparse_sfa_c8 = c8
    impl.pcp_enabled = False
    impl.use_rope = True
    impl.enable_sparse_sfa_turboquant = False
    impl.c8_cache_dtype = torch.float8_e4m3fn
    impl.sfa_qsfa_tile_size = 128
    impl.kv_lora_rank, impl.qk_rope_head_dim, impl.num_kv_heads = 512, 64, 1
    impl._kv_staging_capacity = block
    impl._kv_workspace = lambda dtype: scratch
    impl.kv_a_layernorm = SimpleNamespace(weight=gamma, variance_epsilon=1e-5)
    impl.hisparse_cache = SimpleNamespace(write_target=lambda count, slot_count: (destination, slots, count))
    # The ordinary source cache has no Resident/Hot addressing. It must never
    # receive the parent's second C8 store after exec_kv has used write_target.
    source_cache = (torch.empty(0, device="npu"),)

    def forward():
        rope, latent, *scales = impl.exec_kv(values, cos, sin, source_cache, slots, metadata)
        scales = scales[0] if scales else None
        impl._store_parallel_kv(rope, latent, scales, source_cache, slots, metadata)

    graph = None
    try:
        forward()
        torch.npu.synchronize()
        if capture:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                forward()
        for count in (4, 2):
            values.neg_()
            metadata.query_start_loc[-1:].fill_(count)
            raw.fill_(90)
            if graph is None:
                forward()
            else:
                graph.replay()
            torch.npu.synchronize()
            if c8:
                rope, latent, scales = custom_kv_rmsnorm_rope(
                    values.reshape(rows, 1, 1, 576), gamma, cos, sin, 512, 64, tile_size=128
                )
                components: tuple[torch.Tensor, ...] = (latent, rope, scales)
            else:
                # Reuse native SFA as the oracle; test only the new destination,
                # strides, row mapping and padding, not RMSNorm arithmetic.
                reference_k, reference_r = torch.empty_like(scratch[0]), torch.empty_like(scratch[1])
                torch_npu.npu_kv_rmsnorm_rope_cache(
                    values.view(rows, 1, 1, 576),
                    gamma,
                    cos,
                    sin,
                    scratch[2][:rows],
                    reference_r,
                    reference_k,
                    epsilon=1e-5,
                    cache_mode="PA",
                )
                components = (reference_k.view(-1, 512)[:rows], reference_r.view(-1, 64)[:rows])
            payload = torch.cat(
                [part.contiguous().view(torch.uint8).reshape(rows, -1).cpu() for part in components], -1
            )
            expected = torch.full(raw.shape, 90, dtype=torch.uint8)
            for row, slot in enumerate(logical):
                if row < count and slot >= 0:
                    expected[slot // block, 1, slot % block] = payload[row]
            torch.testing.assert_close(raw.cpu(), expected, rtol=0, atol=0)
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("rows", [4, 16])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
@pytest.mark.usefixtures("_require_a5_preprocess")
def test_c8_prolog_keeps_parent_outputs_and_hisparse_destination(rows, capture, slot_dtype):
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.v1.hisparse.runtime import HiSparseCacheHandle

    from vllm_ascend.attention.sfa_hisparse import AscendHiSparseImpl
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
    from vllm_ascend.quantization.methods import AscendW8A8MXFP8DynamicLinearMethod

    torch.manual_seed(71)
    device, block, width = "npu", 128, 656
    impl = AscendHiSparseImpl.__new__(AscendHiSparseImpl)
    impl._quant_type = AscendW8A8MXFP8DynamicLinearMethod
    impl.use_rope = True
    impl.enable_sparse_sfa_c8 = True
    impl.c8_cache_dtype = torch.float8_e4m3fn
    impl.q_lora_rank, impl.kv_lora_rank, impl.qk_rope_head_dim = 2048, 512, 64
    impl.local_num_heads, impl.has_indexer = 8, True
    impl.pcp_enabled = False
    impl.is_kv_consumer, impl.rl_weight_update_enabled = False, False
    impl.sfa_qsfa_tile_size = block

    def mx_layer(inputs, outputs):
        weight, scale = torch_npu.npu_dynamic_mx_quant(
            torch.randn(outputs, inputs, dtype=torch.bfloat16, device=device) * 0.02,
            dst_type=torch.float8_e4m3fn,
        )
        return SimpleNamespace(weight=weight.T.contiguous(), weight_scale=scale.transpose(0, 1).contiguous())

    impl.fused_qkv_a_proj = mx_layer(6144, 2624)
    impl.q_proj = mx_layer(2048, 2048)
    impl.sfa_qsfa_k_nope_clip_alpha = None
    impl.W_UK_T = torch.randn(8, 192, 512, dtype=torch.bfloat16, device=device) * 0.02
    impl.q_a_layernorm = SimpleNamespace(
        weight=torch.ones(2048, dtype=torch.bfloat16, device=device), variance_epsilon=1e-5
    )
    impl.kv_a_layernorm = SimpleNamespace(
        weight=torch.ones(512, dtype=torch.bfloat16, device=device), variance_epsilon=1e-5
    )
    # Use the parent's production quantized-weight preparation and Prolog as
    # the oracle. This case tests ownership/addressing, not Prolog arithmetic.
    impl._process_weights_for_fused_prolog_v3()
    impl.sfa_qsfa_kr_cache_dummy = torch.empty(0, dtype=torch.bfloat16, device=device)
    reference = torch.empty((1, block, 1, width), dtype=impl.c8_cache_dtype, device=device)
    local_slots = torch.arange(block, dtype=torch.int64, device=device)
    raw = torch.full((3, 2, block, width), 90, dtype=torch.uint8, device=device)
    logical = [-1, 129, 0, 257, 128][:rows] + list(range(16, 16 + max(0, rows - 5)))
    slots = torch.tensor(logical, dtype=slot_dtype, device=device)
    cache = HiSparseCacheHandle(SimpleNamespace(max_num_reqs=rows, kv_dtype=impl.c8_cache_dtype, row_width=width))
    cache.bind_cache(
        raw.view(-1),
        byte_offset=block * width,
        block_stride=2 * block * width,
        num_blocks=3,
        block_size=block,
        block_table=torch.empty(0, dtype=torch.int32, device=device),
        slot_mapping=slots,
    )
    cache.dummy_batch = True  # This component test has no Host-mirror connector.
    cache.num_actual_tokens = rows
    impl.hisparse_cache = cache
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, rows], dtype=torch.int32, device=device), hisparse_for_capture=capture
    )
    x = torch.randn(rows, 6144, dtype=torch.bfloat16, device=device)
    cos = torch.ones(rows, 1, 1, 64, dtype=x.dtype, device=device)
    sin = torch.zeros_like(cos)

    def forward():
        context = ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})
        with override_forward_context(context):
            return impl._sfa_preprocess_prolog_v3(x, (), cos, sin, slots, attn_metadata=metadata)

    def compare(actual, expected):
        if isinstance(actual, tuple):
            assert len(actual) == len(expected)
            for a, e in zip(actual, expected):
                compare(a, e)
        elif actual is not None:
            if actual.dtype == torch.float8_e4m3fn:
                actual, expected = actual.view(torch.uint8), expected.view(torch.uint8)
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
        else:
            assert expected is None

    graph = None
    try:
        forward()
        torch.npu.synchronize()
        if capture:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                captured = forward()
        for valid in (rows - 1, 2, 0, rows):
            x.neg_()
            metadata.query_start_loc[-1:].fill_(valid)
            # The real slot producer marks dummy and padded rows with -1.
            # Replay consumes these changed device values at fixed addresses.
            slots.copy_(
                torch.tensor(
                    [slot if row < valid else -1 for row, slot in enumerate(logical)], dtype=slot_dtype, device=device
                )
            )
            if not capture:
                cache.num_actual_tokens = valid
            raw.fill_(90)
            if graph is None:
                actual = forward()
            else:
                graph.replay()
                actual = captured
            expected = AscendSFAImpl._sfa_preprocess_prolog_v3(
                impl, x, (reference,), cos, sin, local_slots[:rows], attn_metadata=metadata
            )
            torch.npu.synchronize()
            compare(actual, expected)
            payload = reference.view(torch.uint8).reshape(block, width).cpu()
            wanted = torch.full(raw.shape, 90, dtype=torch.uint8)
            for row, slot in enumerate(logical):
                if row < valid and slot >= 0:
                    wanted[slot // block, 1, slot % block] = payload[row]
            torch.testing.assert_close(raw.cpu(), wanted, rtol=0, atol=0)
    finally:
        torch.npu.synchronize()
        if graph is not None:
            graph.reset()
