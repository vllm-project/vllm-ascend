# SPDX-License-Identifier: Apache-2.0
"""CPU Meta dispatch checks; no NPU tensors or device execution are needed."""

import pytest
import torch


@pytest.mark.parametrize("tokens", [0, 64, 2048])
@pytest.mark.parametrize("with_add,save_materialized", [(False, False), (False, True), (True, False), (True, True)])
def test_attn_res_prefill_meta_shape_and_aliases(tokens, with_add, save_materialized):
    try:
        native = torch.ops._C_ascend.attn_res_fwd.fused_prefill
    except AttributeError:
        pytest.skip("native extension with the prefill overload is required")
    prefix = torch.empty(tokens, 7168, dtype=torch.bfloat16, device="meta")
    addend = torch.empty_like(prefix) if with_add else None
    bank = torch.empty(tokens, 10, 7168, dtype=torch.bfloat16, device="meta")[:, 1:9]
    projection = torch.empty(1, 7168, dtype=torch.bfloat16, device="meta")
    norm = torch.empty(7168, dtype=torch.bfloat16, device="meta")
    output, raw_prefix, materialized = native(
        prefix, addend, bank, projection, norm, 1e-5, 2, norm, 1e-5, -1, save_materialized, True
    )
    for actual in (output, raw_prefix, materialized):
        assert actual.shape == prefix.shape and actual.dtype == prefix.dtype
        assert actual.device.type == "meta"
    assert torch._C._is_alias_of(raw_prefix, prefix) is (not with_add)
    assert torch._C._is_alias_of(materialized, output) is (not save_materialized)
    assert not torch._C._is_alias_of(output, prefix)
