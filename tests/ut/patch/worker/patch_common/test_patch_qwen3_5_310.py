from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.patch.worker.patch_qwen3_5_310 as patch_qwen3_5_310
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata


def _make_dummy_layer(prefix: str = "layers.0.linear_attn"):
    def rearrange_mixed_qkv(mixed_qkv):
        if mixed_qkv is None:
            return None, None, None
        num_tokens = mixed_qkv.shape[0]
        shape = (1, num_tokens, 1, mixed_qkv.shape[-1])
        q = torch.zeros(shape, dtype=mixed_qkv.dtype)
        k = torch.zeros_like(q)
        v = torch.zeros_like(q)
        return q, k, v

    return SimpleNamespace(
        prefix=prefix,
        activation=True,
        A_log=torch.zeros(1, dtype=torch.float32),
        dt_bias=torch.zeros(1, dtype=torch.float32),
        conv1d=SimpleNamespace(
            weight=torch.zeros((4, 1, 4), dtype=torch.float32),
            bias=torch.zeros(4, dtype=torch.float32),
        ),
        kv_cache=[(
            torch.zeros((8, 4, 4), dtype=torch.float32),
            torch.zeros((8, 1, 4), dtype=torch.float32),
        )],
        rearrange_mixed_qkv=rearrange_mixed_qkv,
    )


def _make_common_patches(monkeypatch, prefix: str, metadata: GDNAttentionMetadata):
    captured_calls: list[dict] = []

    fake_ascend_namespace = SimpleNamespace()

    def fake_causal_conv1d(*args, **kwargs):
        captured_calls.append(kwargs)
        return args[0]

    fake_ascend_namespace.npu_causal_conv1d_310 = fake_causal_conv1d
    monkeypatch.setattr(
        patch_qwen3_5_310.torch.ops,
        "_C_ascend",
        fake_ascend_namespace,
        raising=False,
    )
    monkeypatch.setattr(
        patch_qwen3_5_310,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={prefix: metadata}),
    )
    monkeypatch.setattr(patch_qwen3_5_310, "enable_sp", lambda: False)
    monkeypatch.setattr(
        patch_qwen3_5_310,
        "maybe_save_kv_layer_to_connector",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        patch_qwen3_5_310,
        "fused_gdn_gating_pytorch",
        lambda _a_log, _a, b, _dt_bias: (
            torch.zeros((1, b.shape[0], 1), dtype=b.dtype),
            torch.zeros((1, b.shape[0], 1), dtype=b.dtype),
        ),
    )
    monkeypatch.setattr(
        patch_qwen3_5_310,
        "npu_recurrent_gated_delta_rule_310",
        lambda q, **_kwargs: torch.zeros_like(q),
    )
    monkeypatch.setattr(
        patch_qwen3_5_310,
        "chunk_gated_delta_rule_pytorch",
        lambda q, initial_state, **_kwargs: (torch.zeros_like(q), initial_state.clone()),
    )
    return captured_calls


def _run_forward_core(dummy_layer, monkeypatch, metadata: GDNAttentionMetadata):
    captured_calls = _make_common_patches(monkeypatch, dummy_layer.prefix, metadata)
    num_actual_tokens = metadata.num_actual_tokens
    mixed_qkv = torch.randn(num_actual_tokens, 4, dtype=torch.float32)
    b = torch.randn(num_actual_tokens, 1, dtype=torch.float32)
    a = torch.randn(num_actual_tokens, 1, dtype=torch.float32)
    core_attn_out = torch.zeros((num_actual_tokens, 1, 4), dtype=torch.float32)

    patch_qwen3_5_310.Ascend310Qwen3_5GatedDeltaNet._forward_core(
        dummy_layer,
        mixed_qkv,
        b,
        a,
        core_attn_out,
    )
    return captured_calls


def _assert_int64_tensor(value: torch.Tensor):
    assert isinstance(value, torch.Tensor)
    assert value.dtype == torch.int64
    assert value.is_contiguous()


def test_decode_path_keeps_causal_conv1d_metadata_on_device(monkeypatch):
    dummy_layer = _make_dummy_layer()
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=2,
        num_decode_tokens=2,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=2,
        non_spec_query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([3, 5], dtype=torch.int32),
    )

    captured_calls = _run_forward_core(dummy_layer, monkeypatch, metadata)

    assert len(captured_calls) == 1
    kwargs = captured_calls[0]
    assert kwargs["query_start_loc"] is None
    _assert_int64_tensor(kwargs["cache_indices"])
    _assert_int64_tensor(kwargs["initial_state_mode"])
    assert kwargs["num_accepted_tokens"] is None
    assert not isinstance(kwargs["cache_indices"], (list, tuple))
    assert not isinstance(kwargs["initial_state_mode"], (list, tuple))


def test_prefill_path_keeps_causal_conv1d_metadata_on_device(monkeypatch):
    dummy_layer = _make_dummy_layer()
    metadata = GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=3,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=3,
        has_initial_state=torch.tensor([True], dtype=torch.bool),
        non_spec_query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([2], dtype=torch.int32),
    )

    captured_calls = _run_forward_core(dummy_layer, monkeypatch, metadata)

    assert len(captured_calls) == 1
    kwargs = captured_calls[0]
    _assert_int64_tensor(kwargs["query_start_loc"])
    _assert_int64_tensor(kwargs["cache_indices"])
    _assert_int64_tensor(kwargs["initial_state_mode"])
    assert kwargs["num_accepted_tokens"] is None
    assert not isinstance(kwargs["query_start_loc"], (list, tuple))
    assert not isinstance(kwargs["cache_indices"], (list, tuple))
    assert not isinstance(kwargs["initial_state_mode"], (list, tuple))


def test_spec_decode_path_keeps_causal_conv1d_metadata_on_device(monkeypatch):
    dummy_layer = _make_dummy_layer()
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=2,
        num_spec_decode_tokens=2,
        num_actual_tokens=2,
        spec_query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        spec_state_indices_tensor=torch.tensor(
            [[1, 11], [3, 13]], dtype=torch.int32
        ),
        spec_sequence_masks=torch.tensor([True, True], dtype=torch.bool),
        spec_token_indx=torch.tensor([0, 1], dtype=torch.int64),
        non_spec_token_indx=torch.empty(0, dtype=torch.int64),
        num_accepted_tokens=torch.tensor([1, 1], dtype=torch.int32),
    )

    captured_calls = _run_forward_core(dummy_layer, monkeypatch, metadata)

    assert len(captured_calls) == 1
    kwargs = captured_calls[0]
    _assert_int64_tensor(kwargs["query_start_loc"])
    _assert_int64_tensor(kwargs["cache_indices"])
    _assert_int64_tensor(kwargs["initial_state_mode"])
    _assert_int64_tensor(kwargs["num_accepted_tokens"])
    assert not isinstance(kwargs["query_start_loc"], (list, tuple))
    assert not isinstance(kwargs["cache_indices"], (list, tuple))
    assert not isinstance(kwargs["initial_state_mode"], (list, tuple))
    assert not isinstance(kwargs["num_accepted_tokens"], (list, tuple))


def test_recurrent_wrapper_uses_full_state_without_python_compaction(monkeypatch):
    captured: dict[str, torch.Tensor] = {}

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("Python-side state compaction should not run.")

    def fake_recurrent_op(**kwargs):
        captured.update(kwargs)
        kwargs["state"][5].fill_(9)
        return torch.zeros_like(kwargs["value"])

    fake_ascend_namespace = SimpleNamespace(
        npu_recurrent_gated_delta_rule_310=fake_recurrent_op
    )
    monkeypatch.setattr(
        patch_qwen3_5_310.torch.ops,
        "_C_ascend",
        fake_ascend_namespace,
        raising=False,
    )
    monkeypatch.setattr(patch_qwen3_5_310.torch, "any", _unexpected)
    monkeypatch.setattr(patch_qwen3_5_310.torch, "unique", _unexpected)

    q = torch.randn((1, 2, 1, 4), dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    beta = torch.randn((1, 2, 1), dtype=torch.float16)
    state = torch.zeros((8, 1, 4, 4), dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 1, 2], dtype=torch.int32)
    ssm_state_indices = torch.tensor([[5, -1], [3, -1]], dtype=torch.int32)

    out = patch_qwen3_5_310.npu_recurrent_gated_delta_rule_310(
        q=q,
        k=k,
        v=v,
        g=None,
        beta=beta,
        state=state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=False,
    )

    assert out.shape == v.shape
    assert captured["state"] is state
    assert captured["state"].dtype == torch.float32
    assert torch.equal(
        captured["ssm_state_indices"],
        torch.tensor([5, 3], dtype=torch.int32),
    )
    assert captured["ssm_state_indices"].is_contiguous()
    assert torch.count_nonzero(state[5]).item() > 0
