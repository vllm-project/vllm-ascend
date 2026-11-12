# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the FIA output-to-V-projection contract with token/head values."""

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm_ascend.attention.mla_v1 as mla_v1
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.mla_v1 import AscendMLAImpl, DecodeMLAPreprocessResult

BLOCK_SIZE = 128
LATENT_DIM = 512
ROPE_DIM = 64
PADDED_HEADS = 16
VALUE_DIM = 4
TOKEN_COUNTS = (1, 2, 4, 8, 16, 32)


class _FIAOutput:
    """Model native layouts and the observed CANN 9.2 FP8 TND_NTD behavior."""

    def __init__(self, token_major: torch.Tensor, fp8: bool):
        self.token_major = token_major
        self.fp8 = fp8
        self.kwargs: dict[str, object] = {}
        self.output: torch.Tensor | None = None

    def __call__(self, query, key, value, **kwargs):
        self.kwargs = kwargs
        layout = kwargs["input_layout"]
        if layout == "TND":
            output = self.token_major.clone()
        elif layout == "BNSD":
            output = self.token_major.unsqueeze(2).clone()
        elif layout == "TND_NTD":
            if self.fp8:
                # The FP8 operator reports NTD shape but retains TND data order.
                tokens, heads, latent = self.token_major.shape
                output = self.token_major.reshape(heads, tokens, latent).clone()
            else:
                output = self.token_major.transpose(0, 1).contiguous()
        else:
            assert layout in ("BSND_NBSD", "BNSD_NBSD")
            output = self.token_major.transpose(0, 1).unsqueeze(2).contiguous()
        self.output = output
        return output, torch.empty(0, dtype=output.dtype)

    def out(self, query, key, value, **kwargs):
        output, _ = self(query, key, value, **kwargs)
        kwargs["out"][0].copy_(output)
        self.output = kwargs["out"][0]


def _make_decode(monkeypatch, num_tokens, speculative, quantization, num_heads, capturing):
    impl = AscendMLAImpl.__new__(AscendMLAImpl)
    impl.num_heads = num_heads
    impl.num_heads_padded = PADDED_HEADS
    impl.head_padding = PADDED_HEADS - num_heads
    impl.num_kv_heads = 1
    impl.kv_lora_rank = LATENT_DIM
    impl.qk_rope_head_dim = ROPE_DIM
    impl.v_head_dim = VALUE_DIM
    impl.enable_kv_nz = False
    impl.fa_quant_layer = quantization != "bf16"
    impl.support_fp8_attention = quantization != "int8"
    impl.speculative_config = SimpleNamespace() if speculative else None
    impl.scale = 1.0
    impl.fak_descale_float = torch.ones(1)

    # Each token/head has different values, and each head has its own V weight.
    tokens = torch.arange(num_tokens).view(-1, 1, 1) * 32
    heads = torch.arange(PADDED_HEADS).view(1, -1, 1) * 2
    features = torch.arange(LATENT_DIM).view(1, 1, -1) % VALUE_DIM
    token_major = (tokens + heads + features).to(torch.bfloat16)
    head_scale = (2 ** (torch.arange(num_heads) % 3)).to(torch.bfloat16)
    impl.W_UV = torch.zeros(num_heads, LATENT_DIM, VALUE_DIM, dtype=torch.bfloat16)
    for feature in range(VALUE_DIM):
        impl.W_UV[:, feature, feature] = head_scale
    expected = token_major[:, :num_heads, :VALUE_DIM] * head_scale.view(1, -1, 1)

    query_dtype = {
        "fp8": torch.float8_e4m3fn,
        "bf16": torch.bfloat16,
        "int8": torch.int8,
    }[quantization]
    preprocess = DecodeMLAPreprocessResult(
        ql_nope=torch.zeros(num_tokens, num_heads, LATENT_DIM, dtype=query_dtype),
        q_pe=torch.zeros(num_tokens, num_heads, ROPE_DIM, dtype=torch.bfloat16),
        k_nope=torch.zeros(1, BLOCK_SIZE, 1, LATENT_DIM, dtype=query_dtype),
        k_pe=torch.zeros(1, BLOCK_SIZE, 1, ROPE_DIM, dtype=torch.bfloat16),
        dequant_scale_q_nope=torch.ones(num_tokens, num_heads) if impl.fa_quant_layer else None,
    )
    metadata = SimpleNamespace(
        attn_state=AscendAttentionState.DecodeOnly,
        causal=True,
        decode=SimpleNamespace(
            actual_seq_lengths_q=[num_tokens],
            seq_lens_list=[BLOCK_SIZE],
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            attn_mask=torch.zeros(num_tokens, num_tokens, dtype=torch.bool),
        ),
    )
    fia = _FIAOutput(token_major, quantization == "fp8")
    monkeypatch.setattr(mla_v1.torch_npu, "npu_fused_infer_attention_score_v2", fia)

    def transpose_batchmatmul(x, weight, *, perm_y, perm_x1=None):
        assert perm_y == (1, 0, 2)
        assert x.is_contiguous()
        if perm_x1 is not None:
            assert perm_x1 == (1, 0, 2)
            assert x.shape == (num_tokens, num_heads, LATENT_DIM)
            x = x.permute(perm_x1)
        else:
            assert x.shape == (num_heads, num_tokens, LATENT_DIM)
        return torch.bmm(x, weight).transpose(0, 1).contiguous()

    projection_op = MagicMock(side_effect=transpose_batchmatmul)
    monkeypatch.setattr(mla_v1.torch_npu, "npu_transpose_batchmatmul", projection_op)
    monkeypatch.setattr(
        mla_v1,
        "_EXTRA_CTX",
        SimpleNamespace(capturing=capturing, is_draft_model=False, is_draft_model_prefill=False),
    )
    graph_params = SimpleNamespace(
        events=defaultdict(list),
        handles=defaultdict(list),
        attn_params=defaultdict(list),
        workspaces={num_tokens: object()},
    )
    monkeypatch.setattr(mla_v1, "get_graph_params", lambda: graph_params)
    # CPU graph tests inspect capture buffers and metadata, not NPU replay.
    if capturing:
        monkeypatch.setattr(torch.npu, "ExternalEvent", MagicMock())
        monkeypatch.setattr(torch.npu, "graph_task_group_begin", MagicMock())
        monkeypatch.setattr(torch.npu, "graph_task_group_end", MagicMock())
    return impl, preprocess, metadata, expected, fia, graph_params, projection_op


def _assert_projection_input(impl, fia, projection_op, *, fused):
    """Check the production projection's operator route and input ownership."""
    projection_op.assert_called_once()
    projection_input, weight = projection_op.call_args.args
    assert weight is impl.W_UV
    expected_kwargs = {"perm_y": (1, 0, 2)}
    if fused:
        expected_kwargs["perm_x1"] = (1, 0, 2)
        assert projection_input.data_ptr() == fia.output.data_ptr()
        assert projection_input.untyped_storage() is fia.output.untyped_storage()
    assert projection_op.call_args.kwargs == expected_kwargs
    if fia.fp8:
        # Graph task updates must retain the operator's token-major buffer.
        torch.testing.assert_close(fia.output.reshape_as(fia.token_major), fia.token_major, rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
@pytest.mark.parametrize("speculative", [False, True], ids=["ordinary", "speculative"])
@pytest.mark.parametrize("quantization", ["fp8", "bf16", "int8"])
@pytest.mark.parametrize("num_heads", [PADDED_HEADS, 12], ids=["no-padding", "padded-heads"])
def test_decode_preserves_token_head_values(monkeypatch, num_tokens, speculative, quantization, num_heads):
    impl, preprocess, metadata, expected, fia, _, projection_op = _make_decode(
        monkeypatch, num_tokens, speculative, quantization, num_heads, capturing=False
    )
    actual = impl._forward_decode(preprocess, BLOCK_SIZE, metadata)
    torch.testing.assert_close(actual.view_as(expected), expected, rtol=0, atol=0)
    _assert_projection_input(
        impl,
        fia,
        projection_op,
        fused=quantization == "fp8" and num_tokens > 1 and num_heads == PADDED_HEADS,
    )
    if quantization == "fp8":
        assert fia.kwargs["input_layout"] == ("TND" if speculative else "BNSD")
    elif speculative:
        assert fia.kwargs["input_layout"] == "TND_NTD"
    else:
        assert fia.kwargs["input_layout"] == ("BSND_NBSD" if quantization == "int8" else "BNSD_NBSD")
    if impl.fa_quant_layer:
        assert fia.kwargs["query_quant_mode"] == 3
        assert fia.kwargs["dequant_scale_query"].numel() == num_tokens * num_heads


@pytest.mark.parametrize("speculative", [False, True], ids=["ordinary", "speculative"])
@pytest.mark.parametrize(
    "quantization,num_tokens",
    [("fp8", num_tokens) for num_tokens in TOKEN_COUNTS] + [("bf16", 4), ("int8", 4)],
)
@pytest.mark.parametrize("num_heads", [PADDED_HEADS, 12], ids=["no-padding", "padded-heads"])
def test_capture_keeps_raw_fia_output_and_layout(monkeypatch, speculative, quantization, num_tokens, num_heads):
    impl, preprocess, metadata, expected, fia, graph_params, projection_op = _make_decode(
        monkeypatch, num_tokens, speculative, quantization, num_heads=num_heads, capturing=True
    )
    actual = impl._forward_decode(preprocess, BLOCK_SIZE, metadata)
    torch.testing.assert_close(actual.view_as(expected), expected, rtol=0, atol=0)
    _assert_projection_input(
        impl,
        fia,
        projection_op,
        fused=quantization == "fp8" and num_tokens > 1 and num_heads == PADDED_HEADS,
    )
    params = graph_params.attn_params[num_tokens][0]
    assert params[6] == fia.kwargs["input_layout"]
    assert params[14] is fia.output
    if quantization == "fp8":
        assert params[6] == ("TND" if speculative else "BNSD")
        shape = (num_tokens, PADDED_HEADS, LATENT_DIM) if speculative else (num_tokens, PADDED_HEADS, 1, LATENT_DIM)
        assert params[14].shape == shape
    else:
        assert params[14].shape[:2] == (PADDED_HEADS, num_tokens)


@pytest.mark.parametrize("speculative", [False, True], ids=["ordinary", "speculative"])
@pytest.mark.parametrize("capturing", [False, True], ids=["eager", "capture"])
@pytest.mark.parametrize(
    "limit_delta,fused", [(-1, False), (0, False), (1, True)], ids=["over-limit", "at-limit", "below-limit"]
)
def test_fp8_fusion_respects_transpose_bmm_limit(monkeypatch, speculative, capturing, limit_delta, fused):
    num_tokens = 4
    impl, preprocess, metadata, expected, fia, graph_params, projection_op = _make_decode(
        monkeypatch, num_tokens, speculative, "fp8", num_heads=PADDED_HEADS, capturing=capturing
    )
    # Exercise the exclusive 65536 boundary with small, otherwise identical
    # tensors. The existing projection helper and decode route share this limit.
    monkeypatch.setattr(mla_v1, "TRANSPOSE_BMM_MAX_SUPPORTED_DIM", impl.num_heads * impl.kv_lora_rank + limit_delta)

    actual = impl._forward_decode(preprocess, BLOCK_SIZE, metadata)

    torch.testing.assert_close(actual.view_as(expected), expected, rtol=0, atol=0)
    _assert_projection_input(impl, fia, projection_op, fused=fused)
    if capturing:
        params = graph_params.attn_params[num_tokens][0]
        assert params[6] == ("TND" if speculative else "BNSD")
        assert params[14] is fia.output
