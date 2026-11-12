# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import model


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("image_lo", [129257, 100])
def test_prepared_routing_preserves_embedding_and_engram_ids(monkeypatch, dtype, image_lo):
    monkeypatch.setattr(model, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: True))
    ids = torch.tensor([-2, -1, 0, 7, image_lo - 1, image_lo, image_lo + 4, image_lo + 5], dtype=dtype)
    original = ids.clone()
    routed, mask = model.prepare_moe_routing_inputs(ids, SimpleNamespace(image_sentinel_base_id=image_lo))
    torch.testing.assert_close(ids, original)
    assert routed.data_ptr() != ids.data_ptr()
    assert routed.dtype == dtype
    assert routed.tolist() == [-2, 0, 0, 7, image_lo - 1, image_lo, image_lo + 4, image_lo + 5]
    assert mask.tolist() == [False, False, False, False, False, True, True, False]


def test_other_hardware_retains_original_routing(monkeypatch):
    monkeypatch.setattr(model, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda _: False))
    ids = torch.tensor([-1, 0, 129257])
    routed, mask = model.prepare_moe_routing_inputs(ids, SimpleNamespace())
    assert routed is ids
    assert mask is None
