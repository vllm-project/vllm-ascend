# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention import attention_v1


@pytest.fixture
def c8_prefill():
    impl = attention_v1.AscendC8AttentionBackendImpl.__new__(attention_v1.AscendC8AttentionBackendImpl)
    impl.num_kv_heads = 1
    impl.head_size = 32
    key = torch.arange(2 * 32 * 32).remainder(101).to(torch.int8).view(2, 32, 1, 32)
    impl.key_cache = key
    layer = SimpleNamespace(
        _c8_k_scale=torch.full((1, 1, 32), 0.5, dtype=torch.bfloat16),
        _c8_v_scale=torch.full((1, 1, 32), 0.25, dtype=torch.float32),
    )
    return impl, key, key.flip(0), torch.tensor([[1, 0, -1]], dtype=torch.int32), [33], layer


@pytest.mark.parametrize("accepted", [True, False], ids=["fused", "declined"])
def test_c8_cached_prefill_dispatch(c8_prefill, monkeypatch, accepted):
    impl, _, _, table, lengths, layer = c8_prefill
    key = SimpleNamespace(device=SimpleNamespace(type="npu"))
    value = SimpleNamespace(device=key.device)
    fused = (torch.empty(33, 1, 32), torch.empty(33, 1, 32))
    operator = ModuleType("vllm_ascend.ops.triton.c8_prefill")
    launcher = Mock(return_value=fused if accepted else None)
    monkeypatch.setattr(operator, "try_fused_c8_prefill", launcher, raising=False)
    monkeypatch.setitem(sys.modules, operator.__name__, operator)
    monkeypatch.setattr(attention_v1, "HAS_TRITON", True)

    # Stop before the first NPU-dependent operation on the legacy path.
    legacy = Mock(side_effect=RuntimeError("legacy C8 cache view"))
    monkeypatch.setattr(impl, "_nz_5d_view", legacy)
    if accepted:
        assert impl._dequant_paged_kv_to_dense(key, value, table, lengths, torch.bfloat16, layer) is fused
        legacy.assert_not_called()
    else:
        with pytest.raises(RuntimeError, match="legacy C8 cache view"):
            impl._dequant_paged_kv_to_dense(key, value, table, lengths, torch.bfloat16, layer)
        legacy.assert_called_once_with(key, 32)

    launcher.assert_called_once_with(
        key, value, table, lengths, 32, 1, 32, torch.bfloat16, layer._c8_k_scale, layer._c8_v_scale
    )


def test_c8_cached_prefill_fallback_packs_only_live_tokens(c8_prefill):
    impl, key, value, table, lengths, layer = c8_prefill
    actual = impl._dequant_paged_kv_to_dense(key, value, table, lengths, torch.bfloat16, layer)

    for packed, cache, scale in zip(actual, (key, value), (layer._c8_k_scale, layer._c8_v_scale)):
        expected = torch.cat((cache[1], cache[0, :1])).to(torch.bfloat16) * scale
        torch.testing.assert_close(packed, expected, rtol=0, atol=0)
