# SPDX-License-Identifier: Apache-2.0
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v2.spec_decode.dmtp import speculator as dmtp


class NativeRecorder:
    def __init__(self):
        self.calls = []

    def empty_cache(self, slot):
        return SimpleNamespace(length=0, slot=slot)

    def append(self, cache, ids, hidden, positions):
        self.calls.append((cache.slot, ids.tolist(), hidden[:, 0].tolist(), positions.tolist()))
        return SimpleNamespace(length=cache.length + len(ids), slot=cache.slot), hidden.clone()


def _batch(slots, inputs, positions):
    offsets = np.cumsum([0, *map(len, inputs)], dtype=np.int32)
    return SimpleNamespace(
        num_reqs=len(slots),
        idx_mapping=torch.tensor(slots),
        idx_mapping_np=np.array(slots),
        input_ids=torch.tensor(sum(inputs, [])),
        positions=torch.tensor(sum(positions, [])),
        query_start_loc_np=offsets,
        query_start_loc=torch.from_numpy(offsets),
    )


def test_native_advances_only_the_verified_prefix_and_splices_prompt_successors():
    spec = dmtp.AscendDMTPSpeculator.__new__(dmtp.AscendDMTPSpeculator)
    spec._native_caches = {}
    spec.native_mtp = NativeRecorder()
    spec.model = SimpleNamespace(
        lm_head=None, logits_processor=SimpleNamespace(get_top_tokens=lambda _, x: x[:, 0].long())
    )
    spec._t1 = torch.zeros(4, dtype=torch.long)
    spec._t1_hidden = torch.zeros((4, 1))
    prompt = _batch([1, 0], [[10, 11], [20]], [[0, 1], [0]])
    spec._advance_native(
        prompt,
        torch.tensor([[0.0], [1.0], [2.0]]),
        torch.tensor([0, 1]),
        torch.tensor([0, 0]),
        torch.tensor([[21], [99]]),
        torch.tensor([[99, 12]]),
    )
    assert spec.native_mtp.calls == [(1, [11, 12], [0, 1], [1, 2]), (0, [21], [2], [1])]
    # Slot 0 rejects two suffix rows. Slot 1 completes its chunked prefill.
    # Order changes across batches; the next anchor comes from last_sampled.
    mixed = _batch([0, 1], [[21, 22, 23, 24], [12, 13]], [[1, 2, 3, 4], [2, 3]])
    spec._advance_native(
        mixed,
        torch.arange(6).float()[:, None],
        torch.tensor([2, 1]),
        torch.tensor([2, 0]),
        torch.tensor([[25], [14]]),
        torch.tensor([[99, 99]]),
    )
    assert spec.native_mtp.calls[2:] == [(0, [22, 25], [0, 1], [2, 3]), (1, [13, 14], [4, 5], [3, 4])]
    assert spec._t1[:2].tolist() == [1, 5]
    spec.add_request(0)
    assert 0 not in spec._native_caches
    with pytest.raises(ValueError, match="prefix length"):
        spec._advance_native(
            mixed, torch.zeros((6, 1)), torch.ones(2), torch.zeros(2), torch.ones(2, 1), torch.zeros(1, 2)
        )


@pytest.mark.parametrize("rejected", range(8))
def test_native_commits_exactly_the_accepted_prefix(rejected):
    spec = dmtp.AscendDMTPSpeculator.__new__(dmtp.AscendDMTPSpeculator)
    spec._native_caches = {0: SimpleNamespace(length=3, slot=0)}
    spec.native_mtp = NativeRecorder()
    spec.model = SimpleNamespace(
        lm_head=None, logits_processor=SimpleNamespace(get_top_tokens=lambda _, x: x[:, 0].long())
    )
    spec._t1 = torch.zeros(1, dtype=torch.long)
    spec._t1_hidden = torch.zeros((1, 1))
    width = 8 - rejected
    batch = _batch([0], [list(range(40, 48))], [list(range(3, 11))])
    spec._advance_native(
        batch,
        torch.arange(8)[:, None].float(),
        torch.tensor([width]),
        torch.tensor([rejected]),
        torch.tensor([[77]]),
        torch.tensor([[0]]),
    )
    assert spec.native_mtp.calls == [
        (0, list(range(41, 40 + width)) + [77], list(range(width)), list(range(4, 4 + width)))
    ]
    assert spec._native_caches[0].length == 3 + width


def test_generation_conditions_second_row_and_follows_only_six_edges(monkeypatch):
    spec = dmtp.AscendDMTPSpeculator.__new__(dmtp.AscendDMTPSpeculator)
    spec.num_query_per_req = 8
    spec.hidden_size = 1
    spec._t1 = torch.tensor([55, 66])
    spec._t1_hidden = torch.tensor([[10.0], [20.0]])
    spec.input_buffers = SimpleNamespace(
        input_ids=torch.tensor([12] + [999] * 7 + [34] + [999] * 7),
        positions=torch.arange(16),
        is_padding=torch.zeros(16, dtype=torch.bool),
    )
    spec.draft_tokens = torch.empty(2, 7, dtype=torch.long)
    spec.vllm_config = None
    monkeypatch.setattr(dmtp, "set_forward_context", lambda *args, **kwargs: nullcontext())

    class Draft:
        def embed_dmtp_inputs(self, ids, mtp_hidden):
            assert ids.tolist() == [[12, 55] + [999] * 6, [34, 66] + [999] * 6]
            assert torch.equal(mtp_hidden, spec._t1_hidden)
            return torch.arange(16).reshape(2, 8, 1).float()

        def __call__(self, input_ids, positions, inputs_embeds):
            return inputs_embeds

        def compute_candidates(self, hidden):
            assert hidden[:, 0].tolist() == [2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15]
            ids = hidden.long() * 10 + torch.tensor([0, 1])
            return ids, torch.zeros_like(ids).float()

        def candidate_selector(self, ids, unary, hidden, anchors):
            assert anchors.tolist() == [55, 66]
            # After predecessor rank 0 choose rank 1, and vice versa.
            return torch.tensor([[1.0, 2.0], [2.0, 1.0]]).expand(2, 6, 2, 2)

    spec.model = Draft()
    spec.model.model = spec.model
    spec._generate_draft(2, 16, {}, {}, None, None)
    assert spec.draft_tokens.tolist() == [[55, 21, 30, 41, 50, 61, 70], [66, 101, 110, 121, 130, 141, 150]]
