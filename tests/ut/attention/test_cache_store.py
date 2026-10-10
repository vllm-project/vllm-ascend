# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch_npu

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADSACPImpl
from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
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


def pa_writer(write):
    def wrapped(key, slots, *, key_cache):
        assert key.ndim == 3 and key_cache.ndim == 4 and slots.ndim == 1
        assert key.is_contiguous() and slots.is_contiguous() and key_cache.is_contiguous()
        return write(key_cache.view(-1, key.shape[-1]), slots.view(-1, 1), key.view(-1, key.shape[-1]))

    return wrapped


@pytest.mark.parametrize("family", list(AscendDeviceType))
@pytest.mark.parametrize("dtype", [torch.int8, torch.float16, torch.bfloat16, torch.float32, torch.float64])
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
        assert len(indices) == len(key)
        indices = indices.flatten().long()
        valid = indices >= 0
        target[indices[valid]] = updates[valid]

    sk.side_effect = scatter.side_effect = write_sk
    pa.side_effect = pa_writer(write_sk)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", sk, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_pa_cache", pa, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", scatter, raising=False)
    expected_sk = (
        family in (AscendDeviceType.A2, AscendDeviceType.A3)
        and dtype != torch.float64
        and layout not in ("column_gap", "block_gap")
    )
    expected_pa = family == AscendDeviceType.A5 and dtype != torch.float64 and layout in ("contiguous", "offset")
    if layout == "block_gap":
        with pytest.raises(RuntimeError, match="view size is not compatible"):
            device_op.get_device_adaptor().scatter_cache(cache.view(-1, key.shape[-1]), slots.view(-1, 1), key)
    else:
        assert (
            device_op.get_device_adaptor().scatter_cache(cache.view(-1, key.shape[-1]), slots.view(-1, 1), key) is None
        )
        reference = torch.as_strided(before, cache.shape, cache.stride(), cache.storage_offset())
        reference.view(-1, 128)[slots[:2049].long()] = key[:2049]
    torch.testing.assert_close(backing, before, rtol=0, atol=0)
    assert sk.call_count == int(expected_sk)
    assert pa.call_count == int(expected_pa)
    assert scatter.call_count == int(not (expected_sk or expected_pa) and layout != "block_gap")


@pytest.mark.parametrize("family", [AscendDeviceType.A3, AscendDeviceType.A5])
def test_missing_operator_falls_back(monkeypatch, family):
    monkeypatch.setattr(device_op, "get_current_hardware_profile", lambda: get_hardware_profile(family))
    key, cache = torch.ones(2049, 128, dtype=torch.int8), torch.zeros(32, 128, 1, 128, dtype=torch.int8)
    slots = torch.arange(2049, dtype=torch.int32)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", None, raising=False)
    pa = Mock()
    monkeypatch.setattr(torch_npu, "npu_scatter_pa_cache", None, raising=False)
    scatter = Mock(
        side_effect=lambda target, indices, updates: target.index_copy_(0, indices.flatten().long(), updates)
    )
    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", scatter, raising=False)
    assert device_op.get_device_adaptor().scatter_cache(cache.view(-1, key.shape[-1]), slots.view(-1, 1), key) is None
    scatter.assert_called_once()
    pa.assert_not_called()
    torch.testing.assert_close(cache.view(-1, 128)[:2049], key)
    assert not cache.view(-1, 128)[2049:].count_nonzero()


@pytest.mark.parametrize("unsupported", ["target_rank", "index_rank", "update_rank", "row_count", "dtype"])
@pytest.mark.parametrize("family", [AscendDeviceType.A3, AscendDeviceType.A5])
def test_fast_shape_guard_preserves_generic_arguments(monkeypatch, unsupported, family):
    monkeypatch.setattr(device_op, "get_current_hardware_profile", lambda: get_hardware_profile(family))
    var = torch.zeros(8, 16)
    indices = torch.tensor([[2], [4], [6]], dtype=torch.int32)
    updates = torch.ones(3, 16)
    if unsupported == "target_rank":
        var = var.view(2, 4, 16)
    elif unsupported == "index_rank":
        indices = indices.flatten()
    elif unsupported == "update_rank":
        updates = updates.unsqueeze(1)
    elif unsupported == "row_count":
        updates = updates[:2]
    else:
        updates = updates.to(torch.float16)
    fast, generic = Mock(), Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", fast, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_pa_cache", fast, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", generic, raising=False)
    device_op.get_device_adaptor().scatter_cache(var, indices, updates)
    fast.assert_not_called()
    generic.assert_called_once()
    assert all(actual is original for actual, original in zip(generic.call_args.args, (var, indices, updates)))


@pytest.mark.parametrize("family", [AscendDeviceType.A3, AscendDeviceType.A5])
@pytest.mark.parametrize("fast_available", [False, True])
def test_scatter_passes_exact_tensor_objects_to_selected_operator(monkeypatch, family, fast_available):
    monkeypatch.setattr(device_op, "get_current_hardware_profile", lambda: get_hardware_profile(family))
    var = torch.zeros(9, 32)[1:, :16]
    indices = torch.tensor([2, -1, 4, 6], dtype=torch.int32).view(-1, 1)
    updates = torch.arange(128).float().view(4, 32)[:, ::2]
    fast, generic = Mock(), Mock()
    monkeypatch.setattr(
        torch.ops._C_ascend, "npu_scatter_nd_update_sk", fast if fast_available else None, raising=False
    )
    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", generic, raising=False)
    device_op.BaseDeviceAdaptor.scatter_cache(var, indices, updates)
    selected, unused = (fast, generic) if fast_available and family == AscendDeviceType.A3 else (generic, fast)
    selected.assert_called_once()
    unused.assert_not_called()
    assert all(actual is original for actual, original in zip(selected.call_args.args, (var, indices, updates)))


@pytest.mark.parametrize("family", [AscendDeviceType.A3, AscendDeviceType.A5])
def test_fast_operator_error_is_not_retried(monkeypatch, family):
    monkeypatch.setattr(device_op, "get_current_hardware_profile", lambda: get_hardware_profile(family))
    fast = Mock(side_effect=RuntimeError("operator failed"))
    scatter, pa = Mock(), Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", fast, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_pa_cache", fast if family == AscendDeviceType.A5 else pa, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", scatter, raising=False)
    with pytest.raises(RuntimeError, match="operator failed"):
        device_op.get_device_adaptor().scatter_cache(
            torch.zeros(4, 16, dtype=torch.int8),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.ones(1, 16, dtype=torch.int8),
        )
    fast.assert_called_once()
    scatter.assert_not_called()
    pa.assert_not_called()


@pytest.mark.parametrize("family", list(AscendDeviceType))
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("layout", ["contiguous", "row_gap", "offset"])
def test_fp8_cache_dispatch_preserves_bytes(monkeypatch, family, dtype, layout):
    monkeypatch.setattr(device_op, "get_current_hardware_profile", lambda: get_hardware_profile(family))
    # Include every byte pattern, including NaNs: scatter must copy bits.
    key = torch.arange(4 * 16, dtype=torch.int32).mul(7).to(torch.uint8).view(dtype).reshape(4, 16)
    backing = torch.full((3, 4, 1, 32), 0xA5, dtype=torch.uint8).view(dtype)
    if layout == "row_gap":
        cache = backing[:2, ..., :16]
    elif layout == "offset":
        cache = backing.view(-1, 4, 1, 16)[1:3]
    else:
        cache = backing.view(-1, 4, 1, 16)[:2]
    slots = torch.tensor([2, 4, 6, -1], dtype=torch.int32)
    expected = backing.view(torch.uint8).clone()
    reference = torch.as_strided(expected, cache.shape, cache.stride(), cache.storage_offset())
    reference.view(-1, 16)[slots[:3].long()] = key[:3].view(torch.uint8)
    sk, pa, scatter = Mock(), Mock(), Mock()

    def write(target, indices, updates):
        assert target.dtype == dtype and updates.dtype == dtype
        assert target.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
        indices = indices.flatten().long()
        valid = indices >= 0
        assert len(indices) == len(key)
        target.view(torch.uint8)[indices[valid]] = updates.view(torch.uint8)[valid]

    sk.side_effect = scatter.side_effect = write
    pa.side_effect = pa_writer(write)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", sk, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_pa_cache", pa, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", scatter, raising=False)

    assert device_op.get_device_adaptor().scatter_cache(cache.view(-1, key.shape[-1]), slots.view(-1, 1), key) is None

    expected_pa = family == AscendDeviceType.A5 and layout != "row_gap"
    sk.assert_not_called()
    assert scatter.call_count == int(not expected_pa)
    assert pa.call_count == int(expected_pa)
    torch.testing.assert_close(backing.view(torch.uint8), expected, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [8, 2049])
@pytest.mark.parametrize("state", list(AscendAttentionState))
@pytest.mark.parametrize("producer,consumer", [(False, False), (True, False), (False, True), (True, True)])
def test_main_cache_write_delegates_with_own_slots(tokens, state, producer, consumer):
    impl = AscendSFADSACPImpl.__new__(AscendSFADSACPImpl)
    impl.qk_rope_head_dim = 128
    impl.enable_sparse_sfa_c8 = True
    impl.enable_sparse_sfa_turboquant = False
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
    target, indices, updates = store.call_args.args
    assert target.data_ptr() == cache.data_ptr() and target.shape == (32 * 128, 656)
    assert indices.data_ptr() == slots.data_ptr() and indices.shape == (tokens, 1)
    assert updates.data_ptr() == key.data_ptr() and updates.shape == (tokens, 656)
    scatter.assert_not_called()


@pytest.mark.parametrize("tokens", [1, 3])
@pytest.mark.parametrize("fast_available", [False, True])
def test_native_main_c8_cache_packs_all_rows_and_preserves_padding(monkeypatch, tokens, fast_available):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.enable_sparse_sfa_c8 = True
    impl.enable_sparse_sfa_turboquant = False
    impl.sfa_qsfa_packed_kv_head_dim = 656
    packed = (torch.arange(4 * 656).reshape(4, 656) % 251 - 125).to(torch.int8)
    k_nope, k_pe, scale = packed.split([512, 128, 16], dim=-1)
    cache = torch.full((2, 4, 1, 656), -7, dtype=torch.int8)
    slots = torch.tensor([5, 2, 6, -1], dtype=torch.int32)
    slots[tokens:] = -1

    def write(target, indices, updates):
        assert len(indices) == packed.shape[0]
        indices = indices.flatten().long()
        valid = indices >= 0
        target.index_copy_(0, indices[valid], updates[valid])

    def try_fast(target, indices, updates):
        if fast_available:
            write(target, indices, updates)
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


@pytest.mark.parametrize("family", [AscendDeviceType.A3, AscendDeviceType.A5])
@pytest.mark.parametrize(
    "key_dtype,scale_dtype", [(torch.bfloat16, None), (torch.int8, torch.float16), (torch.float8_e4m3fn, torch.float32)]
)
@pytest.mark.parametrize("row_gap", [False, True])
@pytest.mark.parametrize("fast_available", [False, True])
def test_indexer_cache_writes_all_gathered_rows(monkeypatch, family, key_dtype, scale_dtype, row_gap, fast_available):
    if family == AscendDeviceType.A3 and key_dtype == torch.float8_e4m3fn:
        pytest.skip("FP8 indexer cache requires A5")
    monkeypatch.setattr(device_op, "get_current_hardware_profile", lambda: get_hardware_profile(family))
    monkeypatch.setattr("vllm_ascend.attention.indexer.DeviceOperator", device_op.get_device_adaptor())
    slots = torch.tensor([5, -1, 2, 6], dtype=torch.int64)
    keys, caches, backings, expected = [], [], [], []
    for dtype, width in [(key_dtype, 128)] + ([(scale_dtype, 1)] if scale_dtype else []):
        backing = torch.full((2, 4, 1, width * (2 if row_gap else 1)), -7).to(dtype)
        cache = backing[..., :width]
        key = torch.arange(4 * width).reshape(4, width).remainder(17).to(dtype)
        before = backing.view(torch.uint8).clone()
        reference = torch.as_strided(before.view(dtype), cache.shape, cache.stride(), cache.storage_offset())
        reference.view(-1, width).view(torch.uint8)[slots[[0, 2, 3]]] = key.view(torch.uint8)[[0, 2, 3]]
        keys.append(key)
        caches.append(cache)
        backings.append(backing)
        expected.append(before)

    def write(target, indices, updates):
        assert updates.shape[0] == 4  # Includes rows gathered from other ranks.
        assert target.untyped_storage().data_ptr() in [b.untyped_storage().data_ptr() for b in backings]
        indices = indices.flatten().long()
        valid = indices >= 0
        target.view(torch.uint8)[indices[valid]] = updates.view(torch.uint8)[valid]

    sk, generic, pa = Mock(side_effect=write), Mock(side_effect=write), Mock()
    pa.side_effect = pa_writer(write)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", sk if fast_available else None, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", generic, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_pa_cache", pa if fast_available else None, raising=False)
    indexer = SimpleNamespace(
        enable_sparse_li_c8=scale_dtype is not None,
        enable_sparse_li_c4=False,
        enable_sparse_li_quant=scale_dtype is not None,
        k_cache=SimpleNamespace(kv_cache=tuple(caches)),
        _use_c8_reshape_optim=lambda: False,
    )
    # Non-quantized forward_k produces [tokens, 1, head_dim]. Metadata still
    # describes only the local tokens; write_cache receives the gathered rows.
    k_li = keys[0] if scale_dtype else keys[0].unsqueeze(1)
    AscendSFAIndexerBackend.write_cache(
        indexer, k_li, keys[1] if scale_dtype else None, slots, SimpleNamespace(num_actual_tokens=2)
    )
    expected_sk = fast_available and family == AscendDeviceType.A3
    expected_pa = fast_available and family == AscendDeviceType.A5 and not row_gap
    assert sk.call_count == (len(caches) if expected_sk else 0)
    assert generic.call_count == (0 if expected_sk or expected_pa else len(caches))
    assert pa.call_count == (len(caches) if expected_pa else 0)
    for backing, reference in zip(backings, expected):
        torch.testing.assert_close(backing.view(torch.uint8), reference, rtol=0, atol=0)


def test_indexer_grouped_cache_write_keeps_store_kv_block(monkeypatch):
    key, scale = torch.zeros(4, 128, dtype=torch.int8), torch.ones(4, 1, dtype=torch.float16)
    caches = (torch.empty(2, 4, 1, 128, dtype=key.dtype), torch.empty(2, 4, 1, 1, dtype=scale.dtype))
    indexer = SimpleNamespace(
        enable_sparse_li_c8=True,
        enable_sparse_li_c4=False,
        enable_sparse_li_quant=True,
        k_cache=SimpleNamespace(kv_cache=caches),
        _use_c8_reshape_optim=lambda: True,
    )
    meta = SimpleNamespace(group_len=object(), group_key_idx=object(), group_key_cache_idx=object(), block_size=4)
    grouped, scatter = Mock(), Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "store_kv_block", grouped, raising=False)
    monkeypatch.setattr("vllm_ascend.attention.indexer.DeviceOperator.scatter_cache", scatter)
    AscendSFAIndexerBackend.write_cache(indexer, key, scale, torch.arange(4), meta)
    assert grouped.call_count == 2
    for call, updates, cache in zip(grouped.call_args_list, (key, scale), caches):
        assert call.args[0] is updates and call.args[1] is cache
        assert call.args[2:] == (meta.group_len, meta.group_key_idx, meta.group_key_cache_idx, 4)
    scatter.assert_not_called()
