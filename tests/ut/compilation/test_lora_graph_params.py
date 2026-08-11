# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from vllm.config import CUDAGraphMode

from vllm_ascend.compilation.acl_graph import _GraphParamStore


@dataclass(frozen=True)
class _Descriptor:
    num_tokens: int
    has_lora: bool


def _forward_context(descriptor: _Descriptor) -> SimpleNamespace:
    return SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_descriptor=descriptor,
    )


def test_graph_param_store_isolates_base_and_lora_descriptors() -> None:
    store = _GraphParamStore([4], list)
    base_descriptor = _Descriptor(num_tokens=4, has_lora=False)
    lora_descriptor = _Descriptor(num_tokens=4, has_lora=True)

    with patch(
        "vllm_ascend.compilation.acl_graph.get_forward_context",
        return_value=_forward_context(base_descriptor),
    ):
        store[4].append("base")

    with patch(
        "vllm_ascend.compilation.acl_graph.get_forward_context",
        return_value=_forward_context(lora_descriptor),
    ):
        store[4].append("lora")

    assert dict.__getitem__(store, base_descriptor) == ["base"]
    assert dict.__getitem__(store, lora_descriptor) == ["lora"]
    assert dict.__getitem__(store, 4) == []


def test_graph_param_store_keeps_integer_key_outside_full_graph_context() -> None:
    store = _GraphParamStore([8], list)

    with patch(
        "vllm_ascend.compilation.acl_graph.get_forward_context",
        side_effect=AssertionError,
    ):
        store[8].append("eager")

    assert store.get(8) == ["eager"]
    assert 8 in store
