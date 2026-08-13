# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from unittest.mock import MagicMock, patch

from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.spec_decode.suffix_proposer_npu import AscendSuffixProposerNPU


def test_factory_creates_suffix_gpu_proposer():
    config = MagicMock()
    device = MagicMock()
    runner = MagicMock()
    proposer = MagicMock()

    with patch(
        "vllm_ascend.spec_decode.AscendSuffixProposerNPU",
        return_value=proposer,
    ) as proposer_cls:
        result = get_spec_decode_method("suffix_gpu", config, device, runner)

    assert result is proposer
    proposer_cls.assert_called_once_with(config, device, runner)


def test_suffix_gpu_disables_cuda_graph_and_warmup_on_npu():
    proposer = object.__new__(AscendSuffixProposerNPU)
    proposer._graph_failed = False
    proposer._warmed_up = False
    token_ids = MagicMock()

    proposer.capture_draft_graph(token_ids)
    proposer._warmup(token_ids)

    assert proposer._graph_failed
    assert proposer._warmed_up
