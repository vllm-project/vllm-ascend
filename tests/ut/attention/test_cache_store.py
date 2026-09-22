# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch_npu

from vllm_ascend.attention import utils as attention_utils
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADSACPImpl
from vllm_ascend.device import device_op
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile


def metadata(**kwargs):
    values = dict(
        num_actual_tokens=2049,
        fast_cache_store=True,
        num_reqs=1,
        is_prefilling=torch.tensor([True]),
        attn_state=AscendAttentionState.ChunkedPrefill,
    )
    return SimpleNamespace(**(values | kwargs))


@pytest.mark.parametrize("family", list(AscendDeviceType))
@pytest.mark.parametrize("dtype", [torch.int8, torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("layout", ["contiguous", "row_gap", "column_gap", "block_gap", "offset"])
def test_platform_dispatch_keeps_destination_storage(monkeypatch, family, dtype, layout):
    monkeypatch.setattr(device_op, "get_current_hardware_profile", lambda: get_hardware_profile(family))
    backing = torch.full((40, 128, 1, 256), -7, dtype=dtype)
    if layout == "contiguous":
        cache = backing.view(80, 128, 1, 128)
    elif layout == "row_gap":
        cache = backing[..., :128]
    elif layout == "column_gap":
        cache = backing[..., ::2]
    elif layout == "block_gap":
        cache = backing.view(80, 128, 1, 128)[::2]
    else:
        cache = backing.view(80, 128, 1, 128)[1:]
    key = (torch.arange(2056 * 256).reshape(2056, 256) % 251 - 125).to(dtype)[:, ::2]
    slots = torch.arange(2056, dtype=torch.int32) + 128
    slots[2049:] = -1
    before = backing.clone()
    sk, pa = Mock(), Mock()

    def write_sk(target, indices, updates):
        assert target.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
        assert (indices >= 0).all() and len(indices) == 2049
        target[indices.flatten().long()] = updates

    def write_pa(updates, indices, *, key_cache):
        assert key_cache is cache and key_cache.is_contiguous()
        assert updates.is_contiguous() and indices.is_contiguous()
        write_sk(key_cache.view(-1, 128), indices, updates.flatten(1))

    sk.side_effect, pa.side_effect = write_sk, write_pa
    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", sk, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_pa_cache", pa, raising=False)
    expected = (
        family in (AscendDeviceType.A2, AscendDeviceType.A3)
        and dtype != torch.float32
        and layout not in ("column_gap", "block_gap")
    ) or (family == AscendDeviceType.A5 and layout in ("contiguous", "offset"))
    assert device_op.get_device_adaptor().try_scatter_cache(key, cache, slots, 2049) == expected
    if expected:
        reference = torch.as_strided(before, cache.shape, cache.stride(), cache.storage_offset())
        reference.view(-1, 128)[slots[:2049].long()] = key[:2049]
    torch.testing.assert_close(backing, before, rtol=0, atol=0)
    assert sk.call_count == int(expected and family != AscendDeviceType.A5)
    assert pa.call_count == int(expected and family == AscendDeviceType.A5)


@pytest.mark.parametrize("state", list(AscendAttentionState))
@pytest.mark.parametrize("prefilling", [[True], [False], [True, False]])
@pytest.mark.parametrize("tokens", [2047, 2048])
def test_only_large_pure_prefill_can_enable_fast_store(state, prefilling, tokens):
    common = metadata(
        attn_state=state, is_prefilling=torch.tensor(prefilling), num_reqs=len(prefilling), num_actual_tokens=tokens
    )
    assert attention_utils.prefill_cache_write_enabled(common) == (
        tokens >= 2048
        and all(prefilling)
        and state
        in (
            AscendAttentionState.PrefillNoCache,
            AscendAttentionState.PrefillCacheHit,
            AscendAttentionState.ChunkedPrefill,
        )
    )
    common.graph_pad_size = tokens
    assert not attention_utils.prefill_cache_write_enabled(common)


@pytest.mark.parametrize("family", [AscendDeviceType.A3, AscendDeviceType.A5])
def test_missing_operator_falls_back(monkeypatch, family):
    monkeypatch.setattr(device_op, "get_current_hardware_profile", lambda: get_hardware_profile(family))
    key, cache = torch.ones(2049, 128, dtype=torch.int8), torch.zeros(32, 128, 1, 128, dtype=torch.int8)
    slots = torch.arange(2049, dtype=torch.int32)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", None, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_pa_cache", None, raising=False)
    assert not device_op.get_device_adaptor().try_scatter_cache(key, cache, slots, 2049)


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("eligible_batch", [False, True])
@pytest.mark.parametrize("producer,consumer", [(False, False), (True, False), (False, True), (True, True)])
def test_main_cache_write_preserves_fallback_and_own_slots(fast, eligible_batch, producer, consumer):
    impl = AscendSFADSACPImpl.__new__(AscendSFADSACPImpl)
    impl.enable_sparse_sfa_c8 = True
    impl.is_kv_producer, impl.is_kv_consumer = producer, consumer
    key = torch.empty(2056, 656, dtype=torch.int8)
    cache = torch.empty(32, 128, 1, 656, dtype=torch.int8)
    slots = torch.arange(2056, dtype=torch.int32) + 512
    if not eligible_batch:
        slots[2046:2049] = -1
    meta = metadata(fast_cache_store=eligible_batch)
    with (
        patch(
            "vllm_ascend.ascend_config.get_ascend_config",
            return_value=SimpleNamespace(c8_enable_reshape_optim=False, c8_reshape_optim_enabled=False),
        ),
        patch(
            "vllm_ascend.attention.context_parallel.sfa_cp.DeviceOperator.try_scatter_cache", return_value=fast
        ) as store,
        patch("torch_npu.npu_scatter_nd_update_", create=True) as scatter,
    ):
        impl._store_parallel_kv(None, None, None, key, [], (cache,), slots, meta, False)
    use_fast_store = eligible_batch
    assert store.call_count == int(use_fast_store)
    if use_fast_store:
        assert store.call_args.args[1] is cache and store.call_args.args[2] is slots
        assert store.call_args.args[3] == meta.num_actual_tokens
    assert scatter.call_count == int(not (use_fast_store and fast))
    if not (use_fast_store and fast):
        torch.testing.assert_close(scatter.call_args.args[1].flatten(), slots[:2049])
