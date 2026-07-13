#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.spec_decode.vocab_mapping import VocabMapping

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer

# CUDAGraphMode values whose ``has_full_cudagraphs()`` is True: FULL plus the
# two composite modes that mix FULL with NONE / PIECEWISE.
FULL_CUDAGRAPH_MODES = [
    CUDAGraphMode.FULL,
    CUDAGraphMode.FULL_DECODE_ONLY,
    CUDAGraphMode.FULL_AND_PIECEWISE,
]

# Modes without a full cudagraph.
NON_FULL_CUDAGRAPH_MODES = [
    CUDAGraphMode.NONE,
    CUDAGraphMode.PIECEWISE,
]


class TestDisablePaddedDrafterBatchWithFullGraph:
    """Guard: ``disable_padded_drafter_batch=True`` + cuda graph + any full
    cudagraph mode must raise ``NotImplementedError``.
    """

    @staticmethod
    def _make_proposer(
        *,
        disable_padded_drafter_batch: bool,
        use_cuda_graph: bool,
        cudagraph_mode: CUDAGraphMode,
    ) -> AscendSpecDecodeBaseProposer:
        """Bypass ``__init__`` and set only the three attrs the guard reads.

        ``cudagraph_mode`` is a real enum value so ``has_full_cudagraphs()`` is
        exercised, not stubbed.
        """
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        proposer.speculative_config = SimpleNamespace(
            disable_padded_drafter_batch=disable_padded_drafter_batch,
        )
        proposer.use_cuda_graph = use_cuda_graph
        proposer.compilation_config = SimpleNamespace(cudagraph_mode=cudagraph_mode)
        return proposer

    @pytest.mark.parametrize("cudagraph_mode", FULL_CUDAGRAPH_MODES)
    def test_guard_raises_when_padded_drafter_batch_disabled_with_full_cudagraph(self, cudagraph_mode: CUDAGraphMode):
        """The bad combo: disable_padded + cuda graph + any full-cudagraph mode
        is intercepted with ``NotImplementedError``."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        with pytest.raises(NotImplementedError, match="disable_padded_drafter_batch"):
            proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    @pytest.mark.parametrize("cudagraph_mode", NON_FULL_CUDAGRAPH_MODES)
    def test_guard_does_not_raise_without_full_cudagraph(self, cudagraph_mode: CUDAGraphMode):
        """NONE / PIECEWISE never trip the guard, even with disable_padded + cuda graph."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        # Must not raise.
        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    @pytest.mark.parametrize("cudagraph_mode", FULL_CUDAGRAPH_MODES)
    def test_guard_does_not_raise_when_padded_drafter_batch_enabled(self, cudagraph_mode: CUDAGraphMode):
        """Padded drafter batch on (the default) is fine with any full cudagraph."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=False,
            use_cuda_graph=True,
            cudagraph_mode=cudagraph_mode,
        )

        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()

    def test_guard_does_not_raise_when_eager(self):
        """``enforce_eager`` -> ``use_cuda_graph=False`` short-circuits the guard."""
        proposer = self._make_proposer(
            disable_padded_drafter_batch=True,
            use_cuda_graph=False,
            cudagraph_mode=CUDAGraphMode.FULL,
        )

        proposer._raise_if_padded_drafter_batch_disabled_and_full_graph_enabled()


class _FakeTokenizer:
    """Minimal tokenizer for building a real ``VocabMapping`` in tests.

    ``encode`` returns ``[]`` so ``VocabMapping`` falls back to its default
    space-prefix set; the toy tokens carry no prefix, so normalization is the
    identity and ``get_vocab`` defines the intersection directly.
    """

    def __init__(self, vocab: dict[str, int], unk_token_id: int):
        self._vocab = vocab
        self._id_to_token = {i: t for t, i in vocab.items()}
        self.unk_token_id = unk_token_id

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return []

    def convert_ids_to_tokens(self, idx):
        return self._id_to_token.get(idx, "")


class TestHeterogeneousVocabTLI:
    """TLI (vllm#38174) helpers on ``AscendSpecDecodeBaseProposer``.

    Target vocab ``{a, b, c, x, <unk_t>}`` and draft vocab
    ``{a, b, c, y, <unk_d>}`` share the intersection ``{a, b, c}`` (ids 0, 1, 2
    in both). Draft id 3 (``y``) and target id 3 (``x``) sit outside it.
    """

    @staticmethod
    def _make_vocab_mapping() -> VocabMapping:
        target = _FakeTokenizer({"a": 0, "b": 1, "c": 2, "x": 3, "<unk_t>": 4}, unk_token_id=4)
        draft = _FakeTokenizer({"a": 0, "b": 1, "c": 2, "y": 3, "<unk_d>": 4}, unk_token_id=4)
        return VocabMapping(
            target_tokenizer=target,
            draft_tokenizer=draft,
            target_vocab_size=5,
            draft_vocab_size=5,
            device="cpu",
        )

    @staticmethod
    def _make_proposer(*, use_heterogeneous_vocab, vocab_mapping) -> AscendSpecDecodeBaseProposer:
        """Bypass ``__init__`` and set only the attrs the helpers read.

        ``use_heterogeneous_vocab=None`` leaves the attribute unset, exercising
        the ``getattr(self, "use_heterogeneous_vocab", False)`` guard.
        """
        proposer = AscendSpecDecodeBaseProposer.__new__(AscendSpecDecodeBaseProposer)
        if use_heterogeneous_vocab is not None:
            proposer.use_heterogeneous_vocab = use_heterogeneous_vocab
        proposer.vocab_mapping = vocab_mapping
        return proposer

    def test_vocab_mapping_intersection(self):
        vm = self._make_vocab_mapping()
        assert vm.intersection_size == 3
        assert vm.intersection_mask_draft.tolist() == [True, True, True, False, False]

    def test_sample_constrains_to_intersection_and_maps_to_target(self):
        """Raw argmax lands on a non-intersection id (3); after constraining, the
        helper must pick the best *intersection* id and return it in target space."""
        proposer = self._make_proposer(
            use_heterogeneous_vocab=True,
            vocab_mapping=self._make_vocab_mapping(),
        )
        # Draft id 3 ("y", outside the intersection) has the highest logit.
        logits = torch.tensor([[0.1, 0.2, 0.3, 9.0, 8.0]])
        out = proposer._sample_draft_token_ids(logits)
        # id 3 is masked to -inf -> best intersection id is 2 ("c") -> target id 2.
        assert out.tolist() == [2]

    def test_to_draft_vocab_maps_and_falls_back_to_unk(self):
        proposer = self._make_proposer(
            use_heterogeneous_vocab=True,
            vocab_mapping=self._make_vocab_mapping(),
        )
        # target ids 0, 1 are shared; 3 ("x") is out of intersection -> draft unk (4).
        out = proposer._to_draft_vocab(torch.tensor([0, 1, 3]))
        assert out.tolist() == [0, 1, 4]

    def test_helpers_are_noop_when_flag_disabled(self):
        proposer = self._make_proposer(
            use_heterogeneous_vocab=False,
            vocab_mapping=self._make_vocab_mapping(),
        )
        logits = torch.tensor([[0.1, 0.2, 0.3, 9.0, 8.0]])
        # Plain argmax, no constraint -> id 3.
        assert proposer._sample_draft_token_ids(logits).tolist() == [3]
        # Passthrough.
        assert proposer._to_draft_vocab(torch.tensor([0, 1, 3])).tolist() == [0, 1, 3]

    def test_helpers_are_noop_when_flag_attr_missing(self):
        """Non-draft_model proposers never set ``use_heterogeneous_vocab``; the
        ``getattr(..., False)`` guard must keep them on the plain path."""
        proposer = self._make_proposer(use_heterogeneous_vocab=None, vocab_mapping=None)
        logits = torch.tensor([[0.1, 0.2, 0.3, 9.0, 8.0]])
        assert proposer._sample_draft_token_ids(logits).tolist() == [3]
        assert proposer._to_draft_vocab(torch.tensor([0, 1, 3])).tolist() == [0, 1, 3]
