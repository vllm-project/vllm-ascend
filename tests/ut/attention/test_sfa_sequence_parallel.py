# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADSACPImpl
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.attention.utils import PreprocessType
from vllm_ascend.ops import mla as mla_module
from vllm_ascend.patch.worker import patch_deepseek_v2


def _impl():
    impl = AscendSFADSACPImpl.__new__(AscendSFADSACPImpl)
    impl.tp_size = 8
    impl.dcp_size = 1
    impl.enable_dsa_cp_full_o_proj = True
    impl.o_proj = SimpleNamespace(reduce_results=False)
    impl._is_mtp_layer = False
    impl.g_proj = None
    impl.qk_rope_head_dim = 64
    impl.preprocess_type = PreprocessType.NATIVE
    return impl


def _metadata(tokens=9, rank=0, world_size=8):
    local = (tokens + world_size - 1) // world_size
    return SimpleNamespace(
        num_input_tokens=tokens,
        attn_state=AscendAttentionState.PrefillNoCache,
        dsa_cp_context=SimpleNamespace(local_start=rank * local, local_end_with_pad=(rank + 1) * local),
    )


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("tp_size", 1),
        ("dcp_size", 2),
        ("enable_dsa_cp_full_o_proj", False),
        ("_is_mtp_layer", True),
        ("g_proj", object()),
        ("qk_rope_head_dim", 0),
        ("preprocess_type", PreprocessType.MLAPO),
    ],
)
def test_sequence_parallel_rejects_unsupported_backends(attribute, value):
    impl = _impl()
    assert impl.supports_sequence_parallel(_metadata())
    setattr(impl, attribute, value)
    assert not impl.supports_sequence_parallel(_metadata())


@pytest.mark.parametrize("state", [AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding])
def test_sequence_parallel_rejects_decode(state):
    metadata = _metadata()
    metadata.attn_state = state
    assert not _impl().supports_sequence_parallel(metadata)


@pytest.mark.parametrize("tokens", [0, 1, 7, 8, 9, 33])
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_sequence_parallel_preserves_padding_without_mutating_input(monkeypatch, tokens, world_size):
    impl = _impl()
    impl.tp_size = world_size

    def forward(self, name, hidden, cache, metadata, output, *, sequence_parallel=False):
        assert sequence_parallel
        start = metadata.dsa_cp_context.local_start
        valid = max(0, min(hidden.shape[0], tokens - start))
        assert torch.count_nonzero(hidden[valid:]) == 0
        output.copy_(hidden + 11)

    monkeypatch.setattr(AscendSFAImpl, "_forward", forward)
    for rank in range(world_size):
        metadata = _metadata(tokens, rank, world_size)
        local = (tokens + world_size - 1) // world_size
        hidden = torch.full((local, 4), 7.0)
        output = torch.empty_like(hidden)
        impl.forward_sequence_parallel("layer", hidden, (), metadata, output)
        valid = max(0, min(local, tokens - rank * local))
        torch.testing.assert_close(output[:valid], hidden[:valid] + 11, rtol=0, atol=0)
        assert torch.count_nonzero(output[valid:]) == 0
        assert torch.all(hidden == 7)


def test_sequence_parallel_checks_local_shape_and_full_projection():
    impl = _impl()
    with pytest.raises(RuntimeError, match="padded local token shard"):
        impl.forward_sequence_parallel("layer", torch.empty(9, 4), (), _metadata(), torch.empty(9, 4))
    impl._use_full_o_proj_weights = nullcontext
    impl._apply_o_proj_full_weight = Mock(side_effect=lambda x: x * 2)
    hidden = torch.arange(12).reshape(3, 4).float()
    output = torch.empty_like(hidden)
    impl._finalize_o_proj(hidden, output, True, output_is_sequence_parallel=True)
    torch.testing.assert_close(output, hidden * 2, atol=0, rtol=0)
    with pytest.raises(RuntimeError, match="complete local O-proj"):
        impl._finalize_o_proj(hidden, output, False, output_is_sequence_parallel=True)


def test_wrapper_restricts_sequence_parallel_to_eager_prefill(monkeypatch):
    wrapper = mla_module.AscendMultiHeadLatentAttention.__new__(mla_module.AscendMultiHeadLatentAttention)
    torch.nn.Module.__init__(wrapper)
    wrapper._eager_sequence_parallel = False
    context = Mock(side_effect=AssertionError("Compiled path must not inspect runtime metadata"))
    monkeypatch.setattr(mla_module, "get_forward_context", context)
    assert not wrapper.supports_sequence_parallel()
    wrapper._eager_sequence_parallel = True
    wrapper.mla_attn = SimpleNamespace(layer_name="attention", impl=_impl())
    monkeypatch.setattr(mla_module, "get_forward_context", lambda: SimpleNamespace(attn_metadata=None))
    assert not wrapper.supports_sequence_parallel()
    metadata = _metadata()
    monkeypatch.setattr(
        mla_module, "get_forward_context", lambda: SimpleNamespace(attn_metadata={"attention": metadata})
    )
    assert wrapper.supports_sequence_parallel()
    metadata.attn_state = AscendAttentionState.DecodeOnly
    assert not wrapper.supports_sequence_parallel()


def test_aux_capture_gathers_single_token_shards(monkeypatch):
    # N == local shard size is possible for N=1, TP>1. Shape comparisons
    # cannot determine the layout; the model passes it explicitly.
    gather = Mock(return_value=torch.tensor([[7.0], [0.0], [0.0], [0.0]]))
    monkeypatch.setattr(patch_deepseek_v2, "tensor_model_parallel_all_gather", gather)
    model = SimpleNamespace(aux_hidden_state_layers=(1,))
    outputs: list[torch.Tensor] = []
    patch_deepseek_v2._capture_aux_hidden_state(
        model, outputs, 1, torch.zeros(1, 1), None, torch.zeros(1), sequence_parallel=True
    )
    assert gather.call_count == 1
    torch.testing.assert_close(outputs[0], torch.tensor([[7.0]]))


@pytest.mark.parametrize("tokens", [1, 7, 8, 9, 33])
@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@pytest.mark.parametrize("layout", ["first", "dense_boundary", "sharded"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_decoder_shards_residual_once(monkeypatch, tokens, world_size, layout, dtype):
    full = torch.arange(tokens * 4).reshape(tokens, 4).to(dtype) / 16
    residual = None if layout == "first" else full + 1
    local_tokens = (tokens + world_size - 1) // world_size

    def norm(hidden, residual=None):
        if residual is None:
            return hidden * 2
        combined = hidden + residual
        return combined * 2, combined

    layer = SimpleNamespace(
        input_layernorm=norm,
        post_attention_layernorm=norm,
        mlp=lambda hidden, already_sequence_parallel: hidden * 3,
        routed_scaling_factor=2.0,
        layer_idx=0 if layout == "first" else 4,
    )
    mla = SimpleNamespace(forward_sequence_parallel=lambda hidden: hidden * 2)
    for rank in range(world_size):

        def chunk(tensor, rank=rank):
            padded = torch.nn.functional.pad(tensor, (0, 0, 0, local_tokens * world_size - tokens))
            return padded[rank * local_tokens : (rank + 1) * local_tokens].clone()

        chunk_mock = Mock(side_effect=chunk)
        monkeypatch.setattr(patch_deepseek_v2, "sequence_parallel_chunk", chunk_mock)
        hidden_in = chunk(full) if layout == "sharded" else full.clone()
        residual_in = chunk(residual) if layout == "sharded" else None if residual is None else residual.clone()
        actual, actual_residual = patch_deepseek_v2._forward_dsa_cp_sequence_parallel(
            layer, mla, hidden_in, residual_in, layout == "sharded"
        )
        expected_residual = chunk(full if residual is None else full + residual)
        expected_attention = expected_residual * 4
        if dtype == torch.float16:
            expected_attention *= 0.5
            if layer.layer_idx == 0:
                expected_residual *= 0.5
        expected_residual = expected_attention + expected_residual
        torch.testing.assert_close(actual_residual, expected_residual, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected_residual * 2 * 3, rtol=0, atol=0)
        assert chunk_mock.call_count == (0 if layout == "sharded" else 2)
