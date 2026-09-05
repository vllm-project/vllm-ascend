# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.attention.context_parallel.common_cp import write_cp_output
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADSACPImpl
from vllm_ascend.quantization.tp_weight_switch import TPWeightSwitchMixin


@pytest.mark.parametrize("tp_size", [2, 4, 16])
@pytest.mark.parametrize("num_tokens", [1, 3, 16, 17])
@pytest.mark.parametrize("reduce_results", [False, True])
def test_sfa_cp_full_weight_output_matches_decoder_contract(
    tp_size: int, num_tokens: int, reduce_results: bool
) -> None:
    expected = torch.arange(num_tokens * 4, dtype=torch.float32).reshape(num_tokens, 4) + 1
    padded = torch.nn.functional.pad(expected, (0, 0, 0, -num_tokens % tp_size))
    rank_outputs = []
    for rank, local_output in enumerate(padded.chunk(tp_size)):
        group = SimpleNamespace(rank_in_group=rank, all_gather=MagicMock(return_value=padded))
        method = MagicMock(spec=TPWeightSwitchMixin)
        impl = AscendSFADSACPImpl.__new__(AscendSFADSACPImpl)
        impl.enable_dsa_cp_with_o_proj_tp = True
        impl.o_proj = SimpleNamespace(reduce_results=reduce_results)
        impl.o_proj_tp_weight_state = object()
        impl._get_o_proj_linear_method = MagicMock(return_value=method)
        impl._apply_o_proj_full_weight = MagicMock(return_value=local_output)
        output = torch.full_like(expected, float("nan"))
        with patch("vllm_ascend.attention.context_parallel.common_cp.get_tp_group", return_value=group):
            actual = impl._finalize_o_proj(local_output, output, gather_full_o_proj=True)
        assert actual is output
        assert method.switch_tp_weight.call_args.kwargs["use_full_weight"] is False
        if reduce_results:
            torch.testing.assert_close(output, expected)
            group.all_gather.assert_called_once()
        else:
            group.all_gather.assert_not_called()
        rank_outputs.append(torch.nn.functional.pad(output, (0, 0, 0, -num_tokens % tp_size)))

    if not reduce_results:
        # Model the upstream SUM followed by token scatter, including empty ranks.
        reduced = torch.stack(rank_outputs).sum(0)
        residual = padded * 0.25
        for actual, reference, local_residual in zip(
            reduced.chunk(tp_size), padded.chunk(tp_size), residual.chunk(tp_size)
        ):
            torch.testing.assert_close(actual + local_residual, reference + local_residual)


def test_cp_output_reused_buffer_clears_other_ranks_tokens() -> None:
    group = SimpleNamespace(rank_in_group=1)
    output = torch.full((5, 2), 100.0)
    with patch("vllm_ascend.attention.context_parallel.common_cp.get_tp_group", return_value=group):
        write_cp_output(torch.ones(3, 2), output, reduce_results=False)
        torch.testing.assert_close(output[:3], torch.zeros(3, 2))
        torch.testing.assert_close(output[3:], torch.ones(2, 2))


def test_sfa_cp_restores_tp_weight_when_projection_fails() -> None:
    impl = AscendSFADSACPImpl.__new__(AscendSFADSACPImpl)
    impl.enable_dsa_cp_with_o_proj_tp = True
    impl.o_proj = SimpleNamespace(reduce_results=False)
    impl.o_proj_tp_weight_state = object()
    method = MagicMock(spec=TPWeightSwitchMixin)
    impl._get_o_proj_linear_method = MagicMock(return_value=method)
    impl._apply_o_proj_full_weight = MagicMock(side_effect=RuntimeError("projection failed"))
    with pytest.raises(RuntimeError, match="projection failed"):
        impl._finalize_o_proj(torch.ones(2, 2), torch.empty(4, 2), True)
    assert method.switch_tp_weight.call_args.kwargs["use_full_weight"] is False
