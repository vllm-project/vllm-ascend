# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import (
    AscendPard2Speculator,
)


def test_init_speculator_routes_pard2_to_ascend_implementation() -> None:
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="pard2"),
    )
    expected = object()

    with patch.object(AscendPard2Speculator, "__new__", return_value=expected):
        assert init_speculator(config, torch.device("cpu")) is expected


def test_pard2_projects_aux_hidden_states_before_upstream_propose() -> None:
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.method = "pard2"
    projected = torch.randn(2, 4)
    speculator.model = SimpleNamespace(combine_hidden_states=Mock(return_value=projected))
    speculator.input_batch = None

    target_hidden = torch.randn(2, 6)
    aux_hidden = [torch.randn(2, 3), torch.randn(2, 3)]
    sentinel = object()
    with patch(
        "vllm.v1.worker.gpu.spec_decode.autoregressive.speculator.AutoRegressiveSpeculator.propose",
        return_value=sentinel,
    ) as upstream_propose:
        result = speculator.propose(
            input_batch=object(),
            attn_metadata={},
            slot_mappings={},
            last_hidden_states=target_hidden,
            aux_hidden_states=aux_hidden,
            num_sampled=object(),
            num_rejected=object(),
            last_sampled=object(),
            next_prefill_tokens=object(),
            temperature=object(),
            seeds=object(),
        )

    assert result is sentinel
    speculator.model.combine_hidden_states.assert_called_once()
    torch.testing.assert_close(
        speculator.model.combine_hidden_states.call_args.args[0],
        torch.cat(aux_hidden, dim=-1),
    )
    assert upstream_propose.call_args.args[3] is projected
    assert upstream_propose.call_args.args[4] is None
