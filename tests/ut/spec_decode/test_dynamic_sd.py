# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the dynamic speculative decoding adaptation on Ascend.

Dynamic SD lets the scheduler choose how many draft tokens (K) to propose per
step based on the runtime batch size. These tests cover the Ascend-side glue:
the proposers must forward the scheduled K to the underlying drafting logic
(instead of silently using the configured maximum), and an n-gram request must
reject a K larger than its configured maximum.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm_ascend.spec_decode.ngram_proposer import AscendNgramProposer
from vllm_ascend.spec_decode.suffix_proposer import AscendSuffixDecodingProposer


def _build_ngram_proposer(k: int) -> AscendNgramProposer:
    proposer = AscendNgramProposer.__new__(AscendNgramProposer)
    proposer.k = k
    input_batch = SimpleNamespace(
        req_ids=["r0", "r1"],
        spec_decode_unsupported_reqs=set(),
        num_tokens_no_spec=np.array([4, 4], dtype=np.int32),
        max_model_len=1024,
        token_ids_cpu=np.zeros((2, 1024), dtype=np.int32),
    )
    proposer.runner = SimpleNamespace(input_batch=input_batch)
    return proposer


def test_ngram_forwards_scheduled_k_to_batch_propose():
    """The scheduled K must reach batch_propose as the trailing ``k`` arg."""
    proposer = _build_ngram_proposer(k=5)
    proposer.batch_propose = MagicMock(return_value=[[1], [2]])

    result = proposer.propose(3, [[10], [11]])

    assert result == [[1], [2]]
    proposer.batch_propose.assert_called_once()
    # Signature: (num_requests, valid_ngram_requests, num_tokens_no_spec,
    #             token_ids_cpu, k)
    forwarded_k = proposer.batch_propose.call_args[0][4]
    assert forwarded_k == 3


def test_ngram_rejects_k_above_configured_max():
    proposer = _build_ngram_proposer(k=2)
    proposer.batch_propose = MagicMock()
    with pytest.raises(AssertionError):
        proposer.propose(3, [[10], [11]])


def test_suffix_forwards_scheduled_k_to_super():
    """The suffix proposer must pass K through to the base implementation."""
    proposer = AscendSuffixDecodingProposer.__new__(AscendSuffixDecodingProposer)
    proposer.runner = SimpleNamespace(input_batch="INPUT_BATCH")

    captured = {}

    def fake_super_propose(num_speculative_tokens, input_batch, sampled):
        captured["k"] = num_speculative_tokens
        captured["input_batch"] = input_batch
        captured["sampled"] = sampled
        return [[7]]

    # Patch the bound super().propose via the base class on the instance's type.
    import vllm_ascend.spec_decode.suffix_proposer as mod

    orig = mod.SuffixDecodingProposer.propose
    mod.SuffixDecodingProposer.propose = lambda self, k, ib, s: fake_super_propose(k, ib, s)
    try:
        result = proposer.propose(4, [[9]])
    finally:
        mod.SuffixDecodingProposer.propose = orig

    assert result == [[7]]
    assert captured == {"k": 4, "input_batch": "INPUT_BATCH", "sampled": [[9]]}
