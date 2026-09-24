# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch_npu

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADSACPImpl
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl
from vllm_ascend.device import device_op
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile


def metadata(**kwargs):
    values = dict(
        num_actual_tokens=2049,
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
    sk, pa, scatter = Mock(), Mock(), Mock()

    def write_sk(target, indices, updates):
        assert target.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
        assert (indices >= 0).all() and len(indices) == 2049
        target[indices.flatten().long()] = updates

    def write_pa(updates, indices, *, key_cache):
        assert key_cache is cache and key_cache.is_contiguous()
        assert updates.is_contiguous() and indices.is_contiguous()
        write_sk(key_cache.view(-1, 128), indices, updates.flatten(1))

    sk.side_effect, pa.side_effect, scatter.side_effect = write_sk, write_pa, write_sk
    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", sk, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_pa_cache", pa, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", scatter, raising=False)
    expected = (
        family in (AscendDeviceType.A2, AscendDeviceType.A3)
        and dtype != torch.float32
        and layout not in ("column_gap", "block_gap")
    ) or (family == AscendDeviceType.A5 and layout in ("contiguous", "offset"))
    if layout == "block_gap":
        with pytest.raises(RuntimeError, match="view size is not compatible"):
            device_op.get_device_adaptor().scatter_cache(key, cache, slots, 2049)
    else:
        assert device_op.get_device_adaptor().scatter_cache(key, cache, slots, 2049) is None
        reference = torch.as_strided(before, cache.shape, cache.stride(), cache.storage_offset())
        reference.view(-1, 128)[slots[:2049].long()] = key[:2049]
    torch.testing.assert_close(backing, before, rtol=0, atol=0)
    assert sk.call_count == int(expected and family != AscendDeviceType.A5)
    assert pa.call_count == int(expected and family == AscendDeviceType.A5)
    assert scatter.call_count == int(not expected and layout != "block_gap")


@pytest.mark.parametrize("family", [AscendDeviceType.A3, AscendDeviceType.A5])
def test_missing_operator_falls_back(monkeypatch, family):
    monkeypatch.setattr(device_op, "get_current_hardware_profile", lambda: get_hardware_profile(family))
    key, cache = torch.ones(2049, 128, dtype=torch.int8), torch.zeros(32, 128, 1, 128, dtype=torch.int8)
    slots = torch.arange(2049, dtype=torch.int32)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", None, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_pa_cache", None, raising=False)
    scatter = Mock(
        side_effect=lambda target, indices, updates: target.index_copy_(0, indices.flatten().long(), updates)
    )
    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", scatter, raising=False)
    assert device_op.get_device_adaptor().scatter_cache(key, cache, slots, 2049) is None
    scatter.assert_called_once()
    torch.testing.assert_close(cache.view(-1, 128)[:2049], key)
    assert not cache.view(-1, 128)[2049:].count_nonzero()


@pytest.mark.parametrize("flat_cache,column_slots", [(True, False), (False, True)])
def test_fast_shape_guard_uses_generic_scatter(monkeypatch, flat_cache, column_slots):
    cache = torch.zeros(2, 4, 1, 16)
    if flat_cache:
        cache = cache.view(-1, 16)
    key = torch.arange(6 * 16, dtype=cache.dtype).view(6, 16)
    slots = torch.tensor([2, 4, 6, -1, -1, -1], dtype=torch.int32)
    if column_slots:
        slots = slots.view(-1, 1)
    fast = Mock()
    scatter = Mock(
        side_effect=lambda target, indices, updates: target.index_copy_(0, indices.flatten().long(), updates)
    )
    monkeypatch.setattr(device_op.BaseDeviceAdaptor, "_scatter_cache", fast)
    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", scatter, raising=False)
    assert device_op.BaseDeviceAdaptor.scatter_cache(key, cache, slots, 3) is None
    fast.assert_not_called()
    scatter.assert_called_once()
    torch.testing.assert_close(cache.view(-1, 16)[[2, 4, 6]], key[:3])
    assert not cache.view(-1, 16)[[0, 1, 3, 5, 7]].count_nonzero()


def test_fast_operator_error_is_not_retried(monkeypatch):
    fast = Mock(side_effect=RuntimeError("operator failed"))
    scatter = Mock()
    monkeypatch.setattr(device_op.BaseDeviceAdaptor, "_scatter_cache", fast)
    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", scatter, raising=False)
    with pytest.raises(RuntimeError, match="operator failed"):
        device_op.BaseDeviceAdaptor.scatter_cache(
            torch.ones(1, 16), torch.zeros(1, 4, 1, 16), torch.zeros(1, dtype=torch.int32), 1
        )
    scatter.assert_not_called()


@pytest.mark.parametrize("tokens", [8, 2049])
@pytest.mark.parametrize("state", list(AscendAttentionState))
@pytest.mark.parametrize("producer,consumer", [(False, False), (True, False), (False, True), (True, True)])
def test_main_cache_write_delegates_with_own_slots(tokens, state, producer, consumer):
    impl = AscendSFADSACPImpl.__new__(AscendSFADSACPImpl)
    impl.enable_sparse_sfa_c8 = True
    impl.is_kv_producer, impl.is_kv_consumer = producer, consumer
    key = torch.empty(2056, 656, dtype=torch.int8)
    cache = torch.empty(32, 128, 1, 656, dtype=torch.int8)
    slots = torch.arange(2056, dtype=torch.int32) + 512
    slots[tokens:] = -1
    meta = metadata(num_actual_tokens=tokens, attn_state=state)
    with (
        patch(
            "vllm_ascend.ascend_config.get_ascend_config",
            return_value=SimpleNamespace(c8_enable_reshape_optim=False, c8_reshape_optim_enabled=False),
        ),
        patch("vllm_ascend.attention.context_parallel.sfa_cp.DeviceOperator.scatter_cache", return_value=None) as store,
        patch("torch_npu.npu_scatter_nd_update_", create=True) as scatter,
    ):
        impl._store_parallel_kv(None, None, None, key, [], (cache,), slots, meta, False)
    store.assert_called_once()
    assert store.call_args.args[1] is cache and store.call_args.args[2] is slots
    assert store.call_args.args[3] == tokens
    scatter.assert_not_called()


@pytest.mark.parametrize("tokens", [1, 3])
@pytest.mark.parametrize("fast_available", [False, True])
def test_native_main_c8_cache_packs_and_writes_actual_tokens(monkeypatch, tokens, fast_available):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.enable_sparse_sfa_c8 = True
    impl.sfa_qsfa_packed_kv_head_dim = 656
    packed = (torch.arange(4 * 656).reshape(4, 656) % 251 - 125).to(torch.int8)
    k_nope, k_pe, scale = packed.split([512, 128, 16], dim=-1)
    cache = torch.full((2, 4, 1, 656), -7, dtype=torch.int8)
    slots = torch.tensor([5, 2, 6, -1], dtype=torch.int32)
    slots[tokens:] = -1

    def write(target, indices, updates):
        assert len(indices) == tokens and (indices >= 0).all()
        target.index_copy_(0, indices.flatten().long(), updates)

    def try_fast(key, target, indices):
        if fast_available:
            write(target.view(-1, 656), indices, key)
        return fast_available

    fast = Mock(side_effect=try_fast)
    generic = Mock(side_effect=write)
    monkeypatch.setattr("vllm_ascend.attention.sfa_v1.DeviceOperator", device_op.BaseDeviceAdaptor)
    monkeypatch.setattr(device_op.BaseDeviceAdaptor, "_scatter_cache", fast)
    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", generic, raising=False)
    result = impl._store_parallel_kv(
        k_pe, k_nope, scale, None, [], (cache,), slots, metadata(num_actual_tokens=tokens), False
    )
    assert result[0] is k_pe and result[1] is k_nope
    reference = torch.full_like(cache, -7)
    reference.view(-1, 656)[slots[:tokens].long()] = packed[:tokens]
    torch.testing.assert_close(cache, reference, rtol=0, atol=0)
    fast.assert_called_once()
    assert generic.call_count == int(not fast_available)
