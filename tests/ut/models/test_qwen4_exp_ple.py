import pytest
import torch
from torch import nn

from vllm_ascend.models.qwen4_exp.nvidia.ple_layer import (
    Qwen4ExpNGramEmbedding,
)


class _IdentityNGramEmbedding(nn.Module):
    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        return ngram_ids.unsqueeze(-1)


def _build_ngram_layer(decode_workspace_width: int = 4) -> Qwen4ExpNGramEmbedding:
    layer = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(layer)
    layer.embedding_dim = 2
    layer.ngram_size = 3
    layer.heads_per_ngram = 1
    layer.ngram_heads = 2
    layer.head_dim = 1
    layer.eos_token_id = 99
    layer.register_buffer("positions_buffer", torch.arange(32, dtype=torch.int64))
    layer.register_buffer(
        "padded_buffer",
        torch.full((4, 32), layer.eos_token_id, dtype=torch.int64),
    )
    layer.register_buffer(
        "decode_padded_buffer",
        torch.full(
            (4, decode_workspace_width),
            layer.eos_token_id,
            dtype=torch.int64,
        ),
    )
    layer.register_buffer("layer_multipliers", torch.tensor([3, 5, 7], dtype=torch.int64))
    layer.register_buffer("ngram_heads_vocab_sizes", torch.tensor([101, 103], dtype=torch.int64))
    layer.register_buffer("ngram_heads_offsets", torch.tensor([0, 101], dtype=torch.int64))
    layer.ngram_embedding = _IdentityNGramEmbedding()
    return layer


def test_compact_workspace_matches_full_workspace_for_ragged_mtp() -> None:
    layer = _build_ngram_layer()
    # Two active MTP rows with query lengths four and two, followed by two
    # graph-padding rows. The EOS inside the first chunk exercises segment
    # truncation for later draft tokens.
    input_ids = torch.tensor([10, 11, 99, 12, 20, 21], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 4, 6, 6, 6], dtype=torch.int32)
    ngram_context = torch.tensor([[7, 8], [18, 19], [99, 99], [99, 99]], dtype=torch.int32)
    hidden_states = torch.empty(input_ids.numel(), 1)

    expected = layer.forward_impl(
        hidden_states,
        input_ids,
        query_start_loc,
        ngram_context,
    )
    actual = layer.forward_impl(
        hidden_states,
        input_ids,
        query_start_loc,
        ngram_context,
        use_compact_workspace=True,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert layer.decode_padded_buffer.tolist() == [
        [10, 11, 99, 12],
        [20, 21, 99, 99],
        [99, 99, 99, 99],
        [99, 99, 99, 99],
    ]


def test_compact_workspace_rejects_an_oversized_decode_row() -> None:
    layer = _build_ngram_layer()

    with pytest.raises(ValueError, match="cannot hold"):
        layer.forward_impl(
            torch.empty(5, 1),
            torch.arange(5),
            torch.tensor([0, 5], dtype=torch.int32),
            torch.tensor([[7, 8]], dtype=torch.int32),
            use_compact_workspace=True,
        )


def test_compact_workspace_handles_full_graph_padding_dummy_row() -> None:
    layer = _build_ngram_layer(decode_workspace_width=8)
    # FULL graph padding may represent all seven padding tokens as one dummy
    # request. The decode-token-capacity width must preserve that row too.
    input_ids = torch.tensor([10, 99, 99, 99, 99, 99, 99, 99])
    query_start_loc = torch.tensor([0, 1, 8, 8, 8], dtype=torch.int32)
    ngram_context = torch.tensor([[7, 8], [99, 99], [99, 99], [99, 99]], dtype=torch.int32)
    hidden_states = torch.empty(input_ids.numel(), 1)

    expected = layer.forward_impl(
        hidden_states,
        input_ids,
        query_start_loc,
        ngram_context,
    )
    actual = layer.forward_impl(
        hidden_states,
        input_ids,
        query_start_loc,
        ngram_context,
        use_compact_workspace=True,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
