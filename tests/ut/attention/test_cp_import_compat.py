# SPDX-License-Identifier: Apache-2.0

"""Import-compatibility guards for the context-parallel refactor.

The PCP/DCP decoupling moves shared symbols out of the GQA v1 backend module
while keeping their historical import paths working. These tests pin that
contract; follow-up changes of the same refactor extend this file.
"""

from enum import Enum

from vllm_ascend.attention.attention_v1 import AscendAttentionState as FromV1
from vllm_ascend.attention.state import AscendAttentionState as FromState


def test_attention_v1_reexports_attention_state() -> None:
    # The historical GQA v1 import path must hand out the very same enum object
    # that is now defined in the shared, backend-agnostic state module.
    assert FromV1 is FromState
    assert issubclass(FromState, Enum)
    assert [(member.name, member.value) for member in FromState] == [
        ("PrefillNoCache", 0),
        ("PrefillCacheHit", 1),
        ("DecodeOnly", 2),
        ("ChunkedPrefill", 3),
        ("SpecDecoding", 4),
    ]
    assert FromState.__module__ == "vllm_ascend.attention.state"
