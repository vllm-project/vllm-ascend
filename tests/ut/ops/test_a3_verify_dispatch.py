# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops import megakda_full


@pytest.mark.parametrize("tokens", [8, 6])
def test_graph_verify_forwards_device_metadata_and_reuses_output(monkeypatch, tokens):
    """Ragged graph inputs must reach the same entry without a host read or packing kernel."""
    heads, dim, batch = 1, 128, 2
    hidden = torch.empty(tokens, 7168, dtype=torch.bfloat16)
    output = torch.empty_like(hidden)
    conv = torch.empty(5, 6, 384, dtype=torch.bfloat16)
    checkpoint = torch.empty(5, heads, dim, dim)
    starts = torch.tensor([0, 4, tokens], dtype=torch.int32)
    slots = torch.tensor([3, 1], dtype=torch.int32)
    counts = torch.tensor([1, 3], dtype=torch.int32)
    desc = SimpleNamespace(query_start_loc=starts, cache_indices=slots, num_accepted_tokens=counts)
    meta = SimpleNamespace(spec_decode_metadata=SimpleNamespace(spec_causal_conv1d=desc))
    layer = SimpleNamespace(
        num_spec=3,
        local_projection_size=dim,
        kv_cache=(conv, checkpoint),
        model_config=SimpleNamespace(enforce_eager=False),
    )
    native = Mock(return_value=output)
    monkeypatch.setattr(megakda_full, "_native_verify", native)
    assert megakda_full.verify(layer, hidden, meta, output=output) is output
    bound_layer, x, out, physical_conv, state, *metadata = native.call_args.args
    assert bound_layer is layer and x is hidden and out is output
    assert physical_conv is conv and state is checkpoint
    assert [m.data_ptr() for m in metadata] == [m.data_ptr() for m in (starts, slots, counts)]
