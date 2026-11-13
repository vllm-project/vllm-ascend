# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

from vllm_ascend.ascend_config import FinegrainedTPConfig
from vllm_ascend.ops.vocab_parallel_embedding import AscendLogitsProcessor, AscendUnquantizedLMHeadMethod


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_unquantized_lmhead_nz_policy(mode, dtype):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.ones(32, 16, dtype=dtype), requires_grad=False)
    with (
        patch("vllm_ascend.utils.get_ascend_config", return_value=SimpleNamespace(weight_nz_mode=mode)),
        patch("vllm_ascend.utils.get_current_hardware_profile", return_value=MagicMock()),
        patch("vllm_ascend.utils.torch_npu.npu_format_cast", side_effect=lambda weight, fmt: weight.clone()) as cast,
    ):
        AscendUnquantizedLMHeadMethod().process_weights_after_loading(layer)
    assert cast.call_count == int(mode == 2 and dtype != torch.float32)
    torch.testing.assert_close(layer.weight, torch.ones(32, 16, dtype=dtype))


def test_tied_lmhead_keeps_embedding_storage():
    embedding = torch.nn.Embedding(32, 16)
    head = torch.nn.Module()
    method = AscendUnquantizedLMHeadMethod()
    method.tie_weights(head, embedding)
    with patch("vllm_ascend.ops.vocab_parallel_embedding.maybe_trans_nz") as convert:
        method.process_weights_after_loading(head)
    convert.assert_not_called()
    assert head.weight is embedding.weight


@pytest.mark.parametrize("head_size", [8, 16])
def test_lmhead_tp_can_span_dp_and_tp(head_size):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=4, tensor_parallel_size=4, prefill_context_parallel_size=1),
        model_config=SimpleNamespace(is_moe=True),
    )
    FinegrainedTPConfig(lmhead_tensor_parallel_size=head_size)._validate_preconditions(config)


@pytest.mark.parametrize("head_size", [6, 12, 32])
def test_lmhead_tp_rejects_incomplete_groups(head_size):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=4, tensor_parallel_size=4, prefill_context_parallel_size=1),
        model_config=SimpleNamespace(is_moe=True),
    )
    with pytest.raises(AssertionError, match="complete TP groups"):
        FinegrainedTPConfig(lmhead_tensor_parallel_size=head_size)._validate_preconditions(config)


@pytest.mark.parametrize("rows", [3, 10, 64])
def test_lmhead_dp_tp_restores_logits_without_duplicate_rows(rows):
    """Exercise the real logits path with distinct DP inputs and uneven rows."""
    dp_size, tp_size, hidden_size, vocab_size = 4, 4, 8, 64
    head_size = dp_size * tp_size
    rows_per_rank = (rows + tp_size - 1) // tp_size
    generator = torch.Generator().manual_seed(7)
    inputs = [torch.randn(rows, hidden_size, generator=generator) for _ in range(dp_size)]
    padded = [F.pad(x, (0, 0, 0, rows_per_rank * tp_size - rows)) for x in inputs]
    weight = torch.randn(vocab_size, hidden_size, generator=generator)
    gathered = torch.cat(padded)
    full_logits = F.linear(gathered, weight)

    for rank in range(head_size):
        dp_rank, tp_rank = divmod(rank, tp_size)
        local_rows = slice(tp_rank * rows_per_rank, (tp_rank + 1) * rows_per_rank)
        local_vocab = weight.chunk(head_size)[rank]
        expected_local = F.linear(padded[dp_rank][local_rows], weight)

        def gather_rows(tensor, dim, dp_rank=dp_rank, local_rows=local_rows):
            assert dim == 0
            torch.testing.assert_close(tensor, padded[dp_rank][local_rows])
            return gathered

        def exchange_logits(logits, group, local_vocab=local_vocab, rank=rank):
            torch.testing.assert_close(logits, F.linear(gathered, local_vocab))
            return full_logits[rank * rows_per_rank : (rank + 1) * rows_per_rank]

        def restore_rows(logits, dim, expected_local=expected_local, dp_rank=dp_rank):
            assert dim == 0
            torch.testing.assert_close(logits, expected_local)
            return F.linear(padded[dp_rank], weight)

        head = SimpleNamespace(
            output_reduce_group=SimpleNamespace(world_size=tp_size, rank_in_group=tp_rank, all_gather=restore_rows),
        )
        processor = SimpleNamespace(
            org_vocab_size=vocab_size, _apply_head=lambda head, x, bias, weight=local_vocab: F.linear(x, weight)
        )
        group = SimpleNamespace(world_size=head_size, all_gather=gather_rows)
        with (
            patch("vllm_ascend.ops.vocab_parallel_embedding.get_lmhead_tp_group", return_value=group),
            patch("vllm_ascend.ops.vocab_parallel_embedding.lmhead_all_to_all", side_effect=exchange_logits),
        ):
            actual = AscendLogitsProcessor._get_logits_lmheadtp(processor, inputs[dp_rank], head, None)
        torch.testing.assert_close(actual, F.linear(inputs[dp_rank], weight))
