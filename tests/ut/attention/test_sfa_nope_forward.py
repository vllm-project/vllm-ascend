# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode

import vllm_ascend.attention.sfa_v1 as sparse_mla
import vllm_ascend.device.device_op as device_op
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadata
from vllm_ascend.device.device_op import A5DeviceAdaptor, Ascend310PDeviceAdaptor, BaseDeviceAdaptor


@pytest.fixture(autouse=True, params=[BaseDeviceAdaptor, A5DeviceAdaptor, Ascend310PDeviceAdaptor])
def mock_scatter_nd_update(monkeypatch, request):
    def scatter(cache, indices, updates):
        slots = indices[:, 0]
        assert torch.all((slots >= 0) & (slots < cache.shape[0]))
        cache[slots] = updates
        return cache

    def scatter_pa(values, slots, *, key_cache):
        scatter(key_cache.view(-1, key_cache.shape[-1]), slots.view(-1, 1), values.view(-1, values.shape[-1]))

    monkeypatch.setattr(sparse_mla.torch_npu, "npu_scatter_nd_update_", scatter, raising=False)
    monkeypatch.setattr(sparse_mla.torch_npu, "npu_scatter_pa_cache", scatter_pa, raising=False)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", scatter, raising=False)
    monkeypatch.setattr(sparse_mla, "DeviceOperator", request.param)
    monkeypatch.setattr(
        sparse_mla, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.NONE)
    )
    monkeypatch.setattr(
        device_op,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(supports=lambda capability: request.param is BaseDeviceAdaptor),
    )
    return request.param


@pytest.mark.parametrize("block_size", [128, 640])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_nope_exec_kv_preserves_padding_and_special_values(block_size, dtype, slot_dtype, mock_scatter_nd_update):
    dim = 512
    backing = torch.randn(5, block_size, 1, dim, dtype=dtype)
    cache = backing[1:-1]
    before = backing.clone()
    slots = torch.tensor([0, block_size - 1, block_size, block_size + 1, 3 * block_size - 1, -1, 1], dtype=slot_dtype)
    num_actual_tokens = 5
    values = torch.randn(slots.numel(), dim, dtype=dtype)
    values[0, :4] = torch.tensor([-0.0, float("nan"), float("inf"), -float("inf")], dtype=dtype)
    expected = before.clone()
    for slot, value in zip(slots[:num_actual_tokens].tolist(), values):
        expected[1 + slot // block_size, slot % block_size, 0] = value
    impl = SimpleNamespace(qk_rope_head_dim=0, kv_lora_rank=dim, kv_a_layernorm=lambda x: x)
    native = sparse_mla.torch_npu.npu_scatter_nd_update_
    with (
        patch.object(sparse_mla.torch_npu, "npu_scatter_nd_update_", wraps=native) as native_scatter,
        patch.object(
            torch.ops._C_ascend, "npu_scatter_nd_update_sk", wraps=torch.ops._C_ascend.npu_scatter_nd_update_sk
        ) as sk_scatter,
        patch.object(
            sparse_mla.torch_npu, "npu_scatter_pa_cache", wraps=sparse_mla.torch_npu.npu_scatter_pa_cache
        ) as pa_scatter,
    ):
        result = AscendSFAImpl.exec_kv(
            impl, values, None, None, (cache,), slots, SimpleNamespace(num_actual_tokens=num_actual_tokens)
        )
    assert result == (None, None)
    if mock_scatter_nd_update is A5DeviceAdaptor:
        native_scatter.assert_not_called()
        sk_scatter.assert_not_called()
        pa_scatter.assert_called_once()
        keys, indices = pa_scatter.call_args.args
        assert keys.shape == (num_actual_tokens, 1, dim)
        assert keys.is_contiguous()
        assert indices.is_contiguous()
        assert pa_scatter.call_args.kwargs["key_cache"] is cache
    else:
        pa_scatter.assert_not_called()
        if mock_scatter_nd_update is BaseDeviceAdaptor:
            scatter = sk_scatter
            native_scatter.assert_not_called()
        else:
            scatter = native_scatter
            sk_scatter.assert_not_called()
        scatter.assert_called_once()
        indices = scatter.call_args.args[1]
        assert indices.shape == (num_actual_tokens, 1)
        assert scatter.call_args.args[0].data_ptr() == cache.data_ptr()
    assert indices.dtype == slot_dtype
    assert torch.equal(indices.view(-1), slots[:num_actual_tokens])
    assert torch.equal(backing.view(torch.uint8), expected.view(torch.uint8))


def test_nope_exec_kv_trims_unused_slots_and_converts_values():
    cache = torch.zeros(2, 4, 1, 8, dtype=torch.bfloat16)
    values = torch.randn(2, 8, dtype=torch.float32)
    slots = torch.tensor([1, 4, 7], dtype=torch.int32)
    impl = SimpleNamespace(qk_rope_head_dim=0, kv_lora_rank=8, kv_a_layernorm=lambda x: x)
    with patch.object(
        sparse_mla,
        "get_forward_context",
        return_value=SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE),
    ):
        AscendSFAImpl.exec_kv(impl, values, None, None, (cache,), slots, SimpleNamespace(num_actual_tokens=3))
    expected = torch.zeros_like(cache)
    expected[0, 1, 0] = values[0].to(cache.dtype)
    expected[1, 0, 0] = values[1].to(cache.dtype)
    assert torch.equal(cache, expected)


def test_nope_exec_kv_full_graph_uses_device_scatter_cache():
    backing = torch.zeros(4, 4, 1, 8, dtype=torch.bfloat16)
    cache = backing[1:-1]
    values = torch.arange(24, dtype=torch.bfloat16).view(3, 8)
    slots = torch.tensor([1, 3, 4], dtype=torch.int64)
    impl = SimpleNamespace(qk_rope_head_dim=0, kv_lora_rank=8, kv_a_layernorm=lambda x: x)
    with (
        patch.object(
            sparse_mla,
            "get_forward_context",
            return_value=SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL),
        ),
        patch.object(
            sparse_mla.DeviceOperator, "scatter_cache", wraps=sparse_mla.DeviceOperator.scatter_cache
        ) as device_scatter,
    ):
        AscendSFAImpl.exec_kv(impl, values, None, None, (cache,), slots, SimpleNamespace(num_actual_tokens=3))
    device_scatter.assert_called_once()
    stored_values, stored_cache, stored_slots, tokens = device_scatter.call_args.args
    assert torch.equal(stored_values, values)
    assert stored_cache is cache
    assert stored_slots is slots
    assert tokens == 3
    expected = torch.zeros_like(backing)
    expected[1, 1, 0] = values[0]
    expected[1, 3, 0] = values[1]
    expected[2, 0, 0] = values[2]
    assert torch.equal(backing, expected)


def test_nope_exec_kv_rejects_unmergeable_pages():
    backing = torch.randn(3, 5, 1, 8)
    cache = backing[:, :4]
    before = backing.clone()
    values = torch.randn(1, 8)
    slots = torch.tensor([4], dtype=torch.int32)
    impl = SimpleNamespace(qk_rope_head_dim=0, kv_lora_rank=8, kv_a_layernorm=lambda x: x)
    with (
        patch.object(sparse_mla.torch_npu, "npu_scatter_nd_update_") as scatter,
        patch.object(torch.ops._C_ascend, "npu_scatter_nd_update_sk") as sk_scatter,
        patch.object(sparse_mla.torch_npu, "npu_scatter_pa_cache") as pa_scatter,
        pytest.raises(RuntimeError, match="view size is not compatible"),
    ):
        AscendSFAImpl.exec_kv(impl, values, None, None, (cache,), slots, SimpleNamespace(num_actual_tokens=1))
    scatter.assert_not_called()
    sk_scatter.assert_not_called()
    pa_scatter.assert_not_called()
    assert torch.equal(backing, before)


class _Linear:
    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight
        self.calls: list[tuple[torch.Tensor, dict]] = []

    def __call__(self, x: torch.Tensor, **kwargs):
        self.calls.append((x.clone(), kwargs))
        return (x @ self.weight,)


class _RecordingIndexer:
    head_dim: int
    enable_sparse_li_c8: bool

    def __init__(self, indices: torch.Tensor) -> None:
        self.indices = indices
        self.call: tuple | None = None
        self.k_cache = SimpleNamespace(prefix="model.layers.0.self_attn.indexer.k_cache")

    def __call__(self, hidden, q_c, k_hidden, metadata, compute_topk):
        self.call = (hidden.clone(), q_c.clone(), k_hidden.clone(), metadata, compute_topk)
        return self.indices


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


def _reference_sparse_attention(**kwargs):
    query = kwargs["query"]
    cache = kwargs["key"]
    indices = kwargs["sparse_indices"].squeeze(1)
    block_table = kwargs["block_table"]
    query_ends = kwargs["actual_seq_lengths_query"]

    assert kwargs["value"] is cache
    assert kwargs["sparse_block_size"] == 1
    assert kwargs["layout_query"] == "TND"
    assert kwargs["layout_kv"] == "PA_BSND"
    assert kwargs["sparse_mode"] == 3
    assert kwargs["attention_mode"] == 2
    assert kwargs["return_softmax_lse"] is False
    assert kwargs["actual_seq_lengths_query"].dtype == torch.int32
    assert kwargs["actual_seq_lengths_kv"].dtype == torch.int32
    assert kwargs["query_rope"] is None
    assert kwargs["key_rope"] is None

    outputs = []
    for token_idx, token_indices in enumerate(indices):
        if token_idx >= int(query_ends[-1]):
            outputs.append(torch.zeros_like(query[token_idx]))
            continue
        request_idx = int(torch.searchsorted(query_ends, token_idx, right=True))
        physical_blocks = block_table[request_idx, token_indices.to(torch.int64)]
        selected = cache[physical_blocks, 0, 0]
        scores = torch.einsum("hd,kd->hk", query[token_idx], selected)
        probs = torch.softmax(scores * kwargs["scale_value"], dim=-1)
        outputs.append(torch.einsum("hk,kd->hd", probs, selected))
    latent_output = torch.stack(outputs)
    unused = latent_output.new_empty(0)
    return latent_output, unused, unused


@pytest.mark.parametrize("graph_mode", [False, True])
@pytest.mark.parametrize("empty_rope_handle", [False, True])
def test_sparse_mla_full_forward_uses_real_rows_and_latent_values(graph_mode, empty_rope_handle) -> None:
    torch.manual_seed(7)
    num_tokens = 3
    padded_tokens = 4
    hidden_dim = 5
    num_heads = 2
    q_nope_dim = 4
    latent_dim = 8
    value_dim = 3
    output_dim = 4

    hidden_states = torch.randn(padded_tokens, hidden_dim)
    fused_weight = torch.randn(hidden_dim, latent_dim * 2)
    q_weight = torch.randn(latent_dim, num_heads * q_nope_dim) * 0.25
    uk_weight = torch.randn(num_heads, q_nope_dim, latent_dim) * 0.25
    uv_weight = torch.randn(num_heads, latent_dim, value_dim) * 0.25
    gate_weight = torch.randn(hidden_dim, num_heads * value_dim)
    output_weight = torch.randn(num_heads * value_dim, output_dim)

    block_table = torch.tensor([[0, 1, 2], [3, 4, -1]], dtype=torch.int32)
    slot_mapping = torch.tensor([1, 2, 4, -1], dtype=torch.int64)
    positions = torch.tensor([1, 2, 1, 0], dtype=torch.int64)
    indices = torch.tensor([[[0, 1]], [[0, 2]], [[0, 1]]], dtype=torch.int32)
    if graph_mode:
        indices = torch.cat((indices, torch.full_like(indices[:1], -1)))
    initial_cache = torch.randn(5, 1, 1, latent_dim)
    kv_cache: tuple[torch.Tensor, ...] = (initial_cache.clone(),)
    if empty_rope_handle:
        kv_cache = (*kv_cache, torch.empty(5, 1, 1, 0))
    metadata = AscendSFAMetadata(
        num_input_tokens=padded_tokens,
        cos=None,
        sin=None,
        seq_lens_cpu=torch.tensor([3, 2], dtype=torch.int32),
        cum_query_lens=torch.tensor([2, 3], dtype=torch.int32),
        num_actual_tokens=num_tokens,
        # One two-token prefill and one one-token decode share this TND call.
        num_prefills=2,
        max_query_len=2,
        max_seq_len=3,
        seq_lens=torch.tensor([3, 2], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        block_table=block_table,
        block_size=1,
        slot_mapping=slot_mapping,
        positions=positions,
    )

    fused_qkv = _Linear(fused_weight)
    gate_proj = _Linear(gate_weight)
    output_proj = _Linear(output_weight)
    indexer = _RecordingIndexer(indices)
    indexer_metadata = SimpleNamespace(
        block_table=torch.tensor([[7, 6], [4, 5]], dtype=torch.int32),
        slot_mapping=torch.full_like(slot_mapping, -1),
        seq_lens=torch.zeros(2, dtype=torch.int32),
        seq_lens_cpu=torch.zeros(2, dtype=torch.int32),
        positions=positions,
        block_size=1,
        tokens_per_state=4,
        cum_query_lens=metadata.query_start_loc[1:],
        raw_seq_lens=metadata.seq_lens,
        num_actual_tokens=num_tokens,
    )

    indexer.head_dim = latent_dim
    indexer.enable_sparse_li_c8 = False
    config = SimpleNamespace(
        kv_transfer_config=None,
        weight_transfer_config=None,
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
    )
    ascend_config = SimpleNamespace(
        enable_sparse_sfa_c8=False,
        enable_mlapo=False,
        rl_config=SimpleNamespace(enabled=False),
    )
    with (
        patch.object(sparse_mla, "get_current_vllm_config", return_value=config),
        patch.object(sparse_mla, "get_ascend_config", return_value=ascend_config),
        # `AscendSFAImpl.__init__` asks `vllm_ascend.utils` whether RL pushes
        # weights into the live model, and that module holds its own reference
        # to the config singleton, so patching the `sfa_v1` name alone leaves
        # the predicate reading the real (uninitialised) config.
        patch("vllm_ascend.utils.get_ascend_config", return_value=ascend_config),
        patch.object(sparse_mla, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(sparse_mla, "enable_sp", return_value=False),
    ):
        impl = AscendSFAImpl(
            q_lora_rank=latent_dim,
            kv_lora_rank=latent_dim,
            num_heads=num_heads,
            head_size=latent_dim,
            v_head_dim=value_dim,
            scale=0.25,
            qk_rope_head_dim=0,
            qk_nope_head_dim=q_nope_dim,
            qk_head_dim=q_nope_dim,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=None,
            kv_sharing_target_layer_name=None,
            kv_a_layernorm=_rms_norm,
            layer_name="model.layers.0.self_attn",
            fused_qkv_a_proj=fused_qkv,
            q_a_layernorm=_rms_norm,
            q_b_proj=_Linear(q_weight),
            kv_b_proj=None,
            indexer=indexer,
            g_proj=gate_proj,
            o_proj=output_proj,
        )
    impl.W_UK_T, impl.W_UV = uk_weight, uv_weight
    assert not impl.supports_dense_mha_prefill
    output = torch.full((padded_tokens, output_dim), 123.0)

    with (
        patch.object(
            sparse_mla,
            "get_forward_context",
            return_value=SimpleNamespace(
                cudagraph_runtime_mode=CUDAGraphMode.FULL if graph_mode else CUDAGraphMode.NONE,
                attn_metadata={indexer.k_cache.prefix: indexer_metadata},
            ),
        ),
        patch.object(
            torch.ops._C_ascend,
            "npu_sparse_flash_attention",
            side_effect=_reference_sparse_attention,
            create=True,
        ) as sparse_attention,
        patch.object(sparse_mla, "attention_transfer_window"),
        patch.object(sparse_mla, "wait_for_kv_layer_from_connector"),
        patch.object(sparse_mla, "notify_kv_cache_written"),
        patch.object(sparse_mla, "maybe_save_kv_layer_to_connector"),
        patch.object(
            sparse_mla,
            "torch_npu",
            SimpleNamespace(npu_scatter_nd_update_=sparse_mla.torch_npu.npu_scatter_nd_update_),
        ),
    ):
        actual = AscendSFAImpl.forward(
            impl,
            "model.layers.0.self_attn",
            hidden_states,
            kv_cache,
            metadata,
            output=output,
        )

    real_hidden = hidden_states[:num_tokens]
    expected_qkv = real_hidden @ fused_weight
    expected_q_c = _rms_norm(expected_qkv[:, :latent_dim])
    expected_kv = _rms_norm(expected_qkv[:, latent_dim:])
    expected_cache = initial_cache.clone()
    expected_cache[slot_mapping[:num_tokens], 0, 0] = expected_kv
    expected_q_nope = (expected_q_c @ q_weight).view(-1, num_heads, q_nope_dim)
    expected_query = torch.einsum("thd,hdl->thl", expected_q_nope, uk_weight)
    expected_latent = _reference_sparse_attention(
        query=expected_query,
        key=expected_cache,
        value=expected_cache,
        sparse_indices=indices[:num_tokens],
        scale_value=impl.scale,
        sparse_block_size=1,
        block_table=block_table,
        actual_seq_lengths_query=metadata.query_start_loc[1:].to(torch.int32),
        actual_seq_lengths_kv=metadata.seq_lens.to(torch.int32),
        query_rope=None,
        key_rope=None,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=False,
    )[0]
    expected_values = torch.einsum("thl,hlv->thv", expected_latent, uv_weight).reshape(
        num_tokens, num_heads * value_dim
    )
    expected_values *= torch.sigmoid(real_hidden @ gate_weight)
    expected_o_input = torch.zeros(padded_tokens, num_heads * value_dim)
    expected_o_input[:num_tokens] = expected_values
    expected_output = expected_o_input @ output_weight
    if not graph_mode:
        # Eager writes only the active output view; padding stays untouched.
        expected_output[num_tokens:] = 123.0

    assert actual is output
    torch.testing.assert_close(actual, expected_output)
    torch.testing.assert_close(kv_cache[0], expected_cache)
    assert fused_qkv.calls[0][0].shape[0] == (padded_tokens if graph_mode else num_tokens)
    assert indexer.call is not None
    torch.testing.assert_close(indexer.call[0][:num_tokens], real_hidden)
    torch.testing.assert_close(indexer.call[1][:num_tokens], expected_q_c)
    torch.testing.assert_close(indexer.call[2][:num_tokens], real_hidden)
    assert indexer.call[3] is indexer_metadata
    assert indexer.call[4] is True
    expected_o_rows = padded_tokens if graph_mode else num_tokens
    torch.testing.assert_close(output_proj.calls[0][0], expected_o_input[:expected_o_rows])
    assert output_proj.calls[0][1] == {}

    kwargs = sparse_attention.call_args.kwargs
    torch.testing.assert_close(kwargs["query"][:num_tokens], expected_query)
    torch.testing.assert_close(kwargs["key"], expected_cache)
    torch.testing.assert_close(kwargs["actual_seq_lengths_query"], torch.tensor([2, 3], dtype=torch.int32))
    torch.testing.assert_close(kwargs["actual_seq_lengths_kv"], torch.tensor([3, 2], dtype=torch.int32))
    assert kwargs["query"].shape[-1] == latent_dim
    assert latent_dim != value_dim
    assert expected_values.shape[-1] == num_heads * value_dim
    assert metadata.smla_metadata is None
