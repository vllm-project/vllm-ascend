# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.spec_decode.standalone_ar import speculator as module


@pytest.mark.parametrize("step", [0, 1, 2])
@pytest.mark.parametrize("target_bounds", [[7, 31], [63, 64]])
def test_correction_token_is_counted_once_in_standalone_attention(step, target_bounds):
    speculator = object.__new__(module.AscendStandaloneARSpeculator)
    speculator.expanded_positions = torch.arange(8)
    speculator.input_buffers = SimpleNamespace(positions=torch.arange(2))
    speculator.draft_is_prefilling = torch.tensor([True, False])
    speculator.vllm_config = SimpleNamespace(parallel_config=object(), model_config=SimpleNamespace(max_model_len=64))
    target = torch.tensor(target_bounds, dtype=torch.int32)
    # #60989's main propose includes the correction before builder dispatch.
    incoming = target.clone() if vllm_version_is("0.33.0") else target + 1
    original = incoming.clone()
    qsl = np.array([0, 3, 8], dtype=np.int32)
    desc, slots, metadata = object(), torch.arange(8), {"draft": object()}

    with (
        patch.object(module, "build_attn_metadata_wrapper", return_value=nullcontext()),
        patch.object(module, "build_attn_metadata_factory", return_value=nullcontext()) as factory,
        patch.object(module.StandaloneARSpeculator, "_build_attn_metadata", return_value=metadata) as parent,
    ):
        actual = speculator._build_attn_metadata(2, desc, qsl, incoming, step, slot_mappings=slots)

    expected = (target + 1).clamp(max=64) if step == 0 and vllm_version_is("0.33.0") else incoming
    assert actual is metadata
    torch.testing.assert_close(parent.call_args.args[3], expected)
    torch.testing.assert_close(incoming, original)
    torch.testing.assert_close(target, torch.tensor(target_bounds, dtype=torch.int32))
    assert parent.call_args.args[0] == 2
    assert parent.call_args.args[1] is desc
    assert parent.call_args.args[2] is qsl
    assert parent.call_args.args[4] == step
    assert parent.call_args.args[7] is slots
    expected_positions = speculator.expanded_positions if step == 0 else speculator.input_buffers.positions
    assert factory.call_args.args[0] is expected_positions
    assert factory.call_args.args[1] == 8
