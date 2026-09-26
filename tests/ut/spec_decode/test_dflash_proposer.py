# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.spec_decode.dflash2_proposer import AscendDflash2Proposer
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer

_MAX_NUM_TOKENS = 256
_NUM_SPECULATIVE_TOKENS = 3
_HIDDEN_SIZE = 8


@pytest.mark.parametrize("proposer_cls", [AscendDflashProposer, AscendDflash2Proposer])
@pytest.mark.parametrize(
    ("max_num_seqs", "max_capture_size", "num_input_tokens"),
    [
        pytest.param(33, 136, 136, id="padded-query"),
        pytest.param(32, 128, 128, id="capture-aligned-query"),
        pytest.param(33, 132, 132, id="equal-capacity"),
        pytest.param(33, 128, 132, id="query-exceeds-capture"),
        pytest.param(33, None, 132, id="unset-capture-size"),
        pytest.param(33, 0, 132, id="disabled-capture"),
    ],
)
def test_query_buffers_cover_execution_shape(proposer_cls, max_num_seqs, max_capture_size, num_input_tokens):
    """Graph padding must not silently truncate query positions or slots."""
    device = torch.device("cpu")
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=max_capture_size),
        speculative_config=SimpleNamespace(draft_sample_method="greedy"),
    )

    def init_base(self, vllm_config, device, runner=None):
        self.max_batch_size = max_num_seqs
        self.num_speculative_tokens = _NUM_SPECULATIVE_TOKENS
        self.max_num_tokens = _MAX_NUM_TOKENS
        self.hidden_size = _HIDDEN_SIZE
        self.dtype = torch.float32
        self.device = device
        self.input_ids = torch.zeros(_MAX_NUM_TOKENS, dtype=torch.int32, device=device)
        self.uses_mrope = False
        self.uses_xdrope_dim = 0
        self.draft_model_config = SimpleNamespace(hf_config=SimpleNamespace(dflash_config={"selector_top_k": 2}))

    with (
        patch.object(AscendEagleProposer, "__init__", init_base),
        patch(
            "vllm_ascend.spec_decode.dflash_proposer.get_ascend_config",
            return_value=SimpleNamespace(dynamic_spec_config=SimpleNamespace(method=None)),
        ),
    ):
        proposer = proposer_cls(config, device)

    input_ids = proposer.input_ids[:num_input_tokens]
    assert input_ids.shape == (num_input_tokens,)
    assert proposer._get_positions(num_input_tokens).shape == input_ids.shape
    assert proposer._slot_mapping_buffer[:num_input_tokens].shape == input_ids.shape
    for buffer in (proposer.positions, proposer._slot_mapping_buffer):
        assert buffer.dtype == torch.int32
        assert buffer.device == device

    # Padding changes storage capacity, not the logical query or context limits.
    assert proposer.max_query_tokens == max_num_seqs * (1 + _NUM_SPECULATIVE_TOKENS)
    assert proposer.max_positions == _MAX_NUM_TOKENS + num_input_tokens
    assert proposer.arange_dflash.shape == (proposer.max_positions + 1,)
    assert proposer._context_positions_buffer.shape == (_MAX_NUM_TOKENS,)
    assert proposer._context_slot_mapping_buffers.shape == (_MAX_NUM_TOKENS,)
    assert proposer._dflash_hidden_states.shape == (_MAX_NUM_TOKENS, _HIDDEN_SIZE)
