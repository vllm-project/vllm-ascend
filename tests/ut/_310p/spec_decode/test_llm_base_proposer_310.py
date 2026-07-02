#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from unittest.mock import MagicMock

import torch

from vllm_ascend._310p.spec_decode.llm_base_proposer_310 import AscendSpecDecodeBaseProposer310


def test_set_inputs_first_pass_310p_rotates_input_ids():
    """Guard 310P MTP input_ids shift: rotate, preserve tail slot, write next tokens."""
    device = torch.device("cpu")
    num_tokens = 9
    hidden_size = 8

    proposer = AscendSpecDecodeBaseProposer310.__new__(AscendSpecDecodeBaseProposer310)
    proposer.needs_extra_input_slots = False
    proposer.uses_xdrope_dim = 0
    proposer.draft_uses_xdrope_dim = 0
    proposer.runner = MagicMock()
    proposer.dtype = torch.float16
    proposer.hidden_size = hidden_size

    buffer_size = 16
    proposer.input_ids = torch.full((buffer_size,), -1, dtype=torch.int32, device=device)
    proposer.input_ids[num_tokens - 1] = 9999
    proposer.positions = torch.zeros(buffer_size, dtype=torch.int64, device=device)
    proposer.hidden_states = torch.zeros(buffer_size, hidden_size, dtype=torch.float16, device=device)

    def _set_positions(num_tokens_to_set: int, positions: torch.Tensor) -> None:
        proposer.positions[:num_tokens_to_set] = positions

    proposer._set_positions = _set_positions

    cad = MagicMock()
    cad.query_start_loc = torch.tensor([0, 3, 5, 9], dtype=torch.int32, device=device)

    target_token_ids = torch.tensor([10, 11, 12, 20, 21, 30, 31, 32, 33], dtype=torch.int32, device=device)
    target_positions = torch.tensor([7, 8, 9, 6, 7, 8, 9, 10, 11], dtype=torch.int64, device=device)
    target_hidden_states = torch.arange(num_tokens * hidden_size, dtype=torch.float16, device=device).view(
        num_tokens, hidden_size
    )
    next_token_ids = torch.tensor([100, 200, 300], dtype=torch.int32, device=device)

    out_num_tokens, out_token_indices, out_cad, long_seq_args = proposer.set_inputs_first_pass(
        target_token_ids=target_token_ids,
        next_token_ids=next_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,
        token_indices_to_sample=None,
        cad=cad,
        num_rejected_tokens_gpu=None,
    )

    assert out_num_tokens == num_tokens
    assert torch.equal(out_token_indices, torch.tensor([2, 4, 8], dtype=torch.int32, device=device))
    assert out_cad is cad
    assert long_seq_args == (None, None)

    expected_input_ids = torch.tensor([11, 12, 100, 21, 200, 31, 32, 33, 300], dtype=torch.int32, device=device)
    assert torch.equal(proposer.input_ids[:num_tokens], expected_input_ids)
    assert torch.equal(proposer.positions[:num_tokens], target_positions)
    assert torch.equal(proposer.hidden_states[:num_tokens], target_hidden_states)
