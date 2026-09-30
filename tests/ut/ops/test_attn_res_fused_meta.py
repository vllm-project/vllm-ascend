# SPDX-License-Identifier: Apache-2.0
"""CPU Meta dispatch checks; no NPU tensors or device execution are needed."""

from contextlib import nullcontext

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


@pytest.mark.parametrize("mode", ["meta", "fake"])
@pytest.mark.parametrize("tokens", [0, 64, 2048])
@pytest.mark.parametrize("optimize_prefill", [False, True])
@pytest.mark.parametrize("with_add,save_materialized", [(False, False), (False, True), (True, False), (True, True)])
def test_attn_res_fused_meta_shape_and_aliases(mode, tokens, optimize_prefill, with_add, save_materialized):
    try:
        native = torch.ops._C_ascend.attn_res_fwd
    except AttributeError:
        pytest.skip("native extension with attn_res_fwd is required")
    device = "meta" if mode == "meta" else "npu"
    with FakeTensorMode() if mode == "fake" else nullcontext():
        prefix = torch.empty(tokens, 7168, dtype=torch.bfloat16, device=device)
        addend = torch.empty_like(prefix) if with_add else None
        bank = torch.empty(tokens, 10, 7168, dtype=torch.bfloat16, device=device)[:, 1:9]
        projection = torch.empty(1, 7168, dtype=torch.bfloat16, device=device)
        norm = torch.empty(7168, dtype=torch.bfloat16, device=device)
        output, raw_prefix, materialized = native(
            prefix,
            addend,
            bank,
            projection,
            norm,
            1e-5,
            2,
            norm,
            1e-5,
            -1,
            save_materialized,
            True,
            optimize_prefill,
        )
        for actual in (output, raw_prefix, materialized):
            assert actual.shape == prefix.shape and actual.dtype == prefix.dtype
            assert actual.device.type == device
        assert torch._C._is_alias_of(raw_prefix, prefix) is (not with_add)
        assert torch._C._is_alias_of(materialized, output) is (not save_materialized)
        assert not torch._C._is_alias_of(output, prefix)
