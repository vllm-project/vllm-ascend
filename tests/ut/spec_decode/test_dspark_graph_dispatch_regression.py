# SPDX-License-Identifier: Apache-2.0
"""Regression tests for DSpark target-descriptor graph dispatch."""

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


class _TargetDescriptorProposer(AscendSpecDecodeBaseProposer):
    def __init__(self) -> None:
        pass

    def uses_target_batch_descriptor_for_graph(self) -> bool:
        return True


def test_large_non_uniform_eager_descriptor_is_not_used_for_draft() -> None:
    proposer = _TargetDescriptorProposer()
    target_prefill = BatchDescriptor(
        num_tokens=16288,
        num_reqs=None,
        uniform=False,
    )

    assert not proposer._can_use_target_batch_descriptor_for_graph(
        CUDAGraphMode.NONE,
        target_prefill,
    )


def test_uniform_full_descriptor_is_used_for_draft_graph() -> None:
    proposer = _TargetDescriptorProposer()
    target_decode = BatchDescriptor(
        num_tokens=32,
        num_reqs=4,
        uniform=True,
    )

    assert proposer._can_use_target_batch_descriptor_for_graph(
        CUDAGraphMode.FULL,
        target_decode,
    )
