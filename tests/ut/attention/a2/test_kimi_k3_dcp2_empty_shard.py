import torch

from vllm_ascend.attention.context_parallel.common_cp import (
    _mask_empty_kv_shards,
    _update_out_and_lse,
)
from vllm_ascend.attention.context_parallel.fia_mla_heads import (
    pad_fia_mla_query_heads,
    trim_fia_mla_query_heads,
)


def test_mask_empty_dcp_kv_shard() -> None:
    attn_output = torch.tensor([[[7.0, 9.0]], [[2.0, 3.0]]])
    softmax_lse = torch.tensor([[[5.0]], [[4.0]]])

    output, lse = _mask_empty_kv_shards(attn_output, softmax_lse, [0, 8])

    torch.testing.assert_close(output[0], torch.zeros_like(output[0]))
    assert torch.isneginf(lse[0]).all()
    torch.testing.assert_close(output[1], attn_output[1])
    torch.testing.assert_close(lse[1], softmax_lse[1])


def test_dcp_lse_combine_ignores_empty_shard() -> None:
    empty_output = torch.zeros(1, 1, 2)
    valid_output = torch.tensor([[[2.0, 3.0]]])
    outputs = torch.stack([empty_output, valid_output])
    lses = torch.tensor([[[[float("-inf")]]], [[[4.0]]]])

    output, lse = _update_out_and_lse(outputs, lses)

    torch.testing.assert_close(output, valid_output)
    torch.testing.assert_close(lse, torch.tensor([[[4.0]]]))


def test_dcp_lse_combine_all_empty_graph_padding_row() -> None:
    outputs = torch.randn(2, 1, 1, 2)
    lses = torch.full((2, 1, 1, 1), float("-inf"))

    output, lse = _update_out_and_lse(outputs, lses)

    torch.testing.assert_close(output, torch.zeros_like(output))
    assert torch.isneginf(lse).all()


def test_empty_shard_mask_graph_replay_is_dynamic() -> None:
    compiled = torch.compile(_mask_empty_kv_shards, backend="eager", fullgraph=True)
    attn_output = torch.tensor([[[7.0, 9.0]], [[2.0, 3.0]]])
    softmax_lse = torch.tensor([[[5.0]], [[4.0]]])

    first_output, first_lse = compiled(attn_output, softmax_lse, torch.tensor([0, 8]))
    second_output, second_lse = compiled(attn_output, softmax_lse, torch.tensor([8, 0]))

    torch.testing.assert_close(first_output[0], torch.zeros_like(first_output[0]))
    torch.testing.assert_close(first_output[1], attn_output[1])
    assert torch.isneginf(first_lse[0]).all()
    torch.testing.assert_close(second_output[0], attn_output[0])
    torch.testing.assert_close(second_output[1], torch.zeros_like(second_output[1]))
    assert torch.isneginf(second_lse[1]).all()


def test_graph_padding_rows_missing_from_cp_lengths_are_empty() -> None:
    attn_output = torch.arange(12, dtype=torch.float32).view(4, 1, 3)
    softmax_lse = torch.arange(4, dtype=torch.float32).view(4, 1, 1)

    output, lse = _mask_empty_kv_shards(attn_output, softmax_lse, [129, 65])

    torch.testing.assert_close(output[:2], attn_output[:2])
    torch.testing.assert_close(lse[:2], softmax_lse[:2])
    torch.testing.assert_close(output[2:], torch.zeros_like(output[2:]))
    assert torch.isneginf(lse[2:]).all()


def test_empty_shard_preserves_greedy_token_and_logprobs() -> None:
    valid_output = torch.tensor([[[0.5, -0.25, 1.0]]])
    outputs = torch.stack([torch.zeros_like(valid_output), valid_output])
    lses = torch.tensor([[[[float("-inf")]]], [[[3.0]]]])
    projection = torch.tensor(
        [
            [0.1, 0.2, -0.3, 0.4],
            [0.2, -0.1, 0.5, 0.3],
            [-0.4, 0.6, 0.2, -0.2],
        ]
    )

    combined, _ = _update_out_and_lse(outputs, lses)
    reference_logits = valid_output @ projection
    combined_logits = combined @ projection

    assert combined_logits.argmax(dim=-1).item() == reference_logits.argmax(dim=-1).item()
    torch.testing.assert_close(
        combined_logits.log_softmax(dim=-1),
        reference_logits.log_softmax(dim=-1),
        rtol=0,
        atol=0,
    )


def test_fia_mla_query_heads_pad_and_trim_bnsd() -> None:
    q_nope = torch.arange(2 * 12 * 512, dtype=torch.float32).view(2, 12, 1, 512)
    q_pe = torch.arange(2 * 12 * 64, dtype=torch.float32).view(2, 12, 1, 64)
    padded_nope, padded_pe, padded_heads = pad_fia_mla_query_heads(q_nope, q_pe, "BNSD", 12)

    assert padded_heads == 16
    torch.testing.assert_close(padded_nope[:, :12], q_nope)
    torch.testing.assert_close(padded_pe[:, :12], q_pe)
    torch.testing.assert_close(padded_nope[:, 12:], torch.zeros_like(padded_nope[:, 12:]))
    torch.testing.assert_close(padded_pe[:, 12:], torch.zeros_like(padded_pe[:, 12:]))

    lse = torch.arange(2 * 16, dtype=torch.float32).view(2, 16, 1, 1)
    output, trimmed_lse = trim_fia_mla_query_heads(padded_nope, lse, "BNSD", 12)
    torch.testing.assert_close(output, q_nope)
    torch.testing.assert_close(trimmed_lse, lse[:, :12])


def test_fia_mla_query_heads_pad_and_trim_bsnd() -> None:
    q_nope = torch.randn(2, 1, 12, 512)
    q_pe = torch.randn(2, 1, 12, 64)
    padded_nope, padded_pe, padded_heads = pad_fia_mla_query_heads(q_nope, q_pe, "BSND", 12)
    lse = torch.randn(2, 16, 1, 1)
    output, trimmed_lse = trim_fia_mla_query_heads(padded_nope, lse, "BSND", 12)

    assert padded_heads == 16
    torch.testing.assert_close(output, q_nope)
    torch.testing.assert_close(trimmed_lse, lse[:, :12])
    torch.testing.assert_close(padded_pe[:, :, 12:], torch.zeros_like(padded_pe[:, :, 12:]))


def test_fia_mla_query_head_padding_graph_replay() -> None:
    compiled = torch.compile(pad_fia_mla_query_heads, backend="eager", fullgraph=True)
    first = torch.ones(1, 12, 1, 512)
    second = torch.full((1, 12, 1, 512), 2.0)
    rope = torch.ones(1, 12, 1, 64)

    first_padded, _, first_heads = compiled(first, rope, "BNSD", 12)
    second_padded, _, second_heads = compiled(second, rope, "BNSD", 12)

    assert first_heads == second_heads == 16
    torch.testing.assert_close(first_padded[:, :12], first)
    torch.testing.assert_close(second_padded[:, :12], second)
    torch.testing.assert_close(second_padded[:, 12:], torch.zeros_like(second_padded[:, 12:]))
