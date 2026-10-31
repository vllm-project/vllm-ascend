# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Regression coverage for empty MRV2 cache components."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.worker.v2 import utils as v2_utils
from vllm_ascend.worker.v2.attn_utils import _reshape_combined_mla_kv_cache


@pytest.fixture
def zeroer_factory():
    with patch.object(
        v2_utils, "KVBlockZeroer", side_effect=lambda *args, **kwargs: MagicMock(init_kwargs=kwargs)
    ) as parent:
        yield v2_utils.AscendV2KVBlockZeroer, parent


@pytest.mark.parametrize("container", [tuple, list])
@pytest.mark.parametrize("empty_position", [0, 1])
def test_empty_rope_component_does_not_create_zeroer(zeroer_factory, container, empty_position):
    cls, parent = zeroer_factory
    num_blocks, latent_width = 4, 8
    dtype = torch.float32
    page_bytes = latent_width * torch.empty((), dtype=dtype).element_size()
    full, empty = _reshape_combined_mla_kv_cache(
        torch.zeros(num_blocks * page_bytes, dtype=torch.uint8),
        (num_blocks, 1, 1, latent_width),
        (num_blocks, 1, 1, 0),
        dtype,
        page_bytes,
    )
    assert empty.numel() == 0
    assert empty.untyped_storage().data_ptr() == full.untyped_storage().data_ptr()
    parts = [full]
    parts.insert(empty_position, empty)
    group = _group(["layer"])
    zeroer = cls(torch.device("cpu"), [group], [1], {"layer": SimpleNamespace(kv_cache=container(parts))}, 4)
    parent.assert_called_once()
    assert parent.call_args.kwargs["static_forward_context"]["layer"].kv_cache is full
    assert group.layer_names == ["layer"]
    zeroer.zero_block_ids([0, 2])
    zeroer._zeroers[0].zero_block_ids.assert_called_once_with([0, 2])
    zeroer.warmup(4)
    zeroer._zeroers[0].warmup.assert_called_once_with(4)


@pytest.mark.parametrize("container", [lambda tensor: tensor, lambda tensor: (tensor,), lambda tensor: [tensor]])
def test_all_empty_components_are_noop(zeroer_factory, container):
    cls, parent = zeroer_factory
    binding = container(torch.empty(4, 1, 1, 0))
    zeroer = cls(torch.device("cpu"), [_group(["empty"])], [1], {"empty": SimpleNamespace(kv_cache=binding)}, 4)
    parent.assert_not_called()
    zeroer.zero_block_ids([0])
    zeroer.warmup(4)


def test_nonempty_components_and_exclusions_are_preserved(zeroer_factory):
    cls, parent = zeroer_factory
    key, value = torch.ones(4, 1, 1, 8), torch.ones(4, 1, 1, 4)
    context = {"live": SimpleNamespace(kv_cache=(key, value)), "excluded": SimpleNamespace(kv_cache=key)}
    zeroer = cls(torch.device("cpu"), [_group(["live", "excluded"])], [1], context, 4, {"excluded"})
    assert len(zeroer._zeroers) == parent.call_count == 2
    for call, tensor in zip(parent.call_args_list, [key, value]):
        groups = call.kwargs["attn_groups_iter"]
        assert groups[0].layer_names == ["live"]
        assert call.kwargs["static_forward_context"]["live"].kv_cache is tensor


def _group(names):
    @dataclass
    class Group:
        kv_cache_spec: object
        layer_names: list[str]
        kv_cache_group_id: int

    return Group(object(), names, 0)
