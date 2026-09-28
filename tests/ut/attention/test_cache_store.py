# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch_npu

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADSACPImpl, AscendSFAPCPImpl
from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, PreprocessType
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
    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", sk, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_pa_cache", pa, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", scatter, raising=False)
    expected_sk = (
        family in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5)
        and dtype != torch.float64
        and layout not in ("column_gap", "block_gap")
    )
    if layout == "block_gap":
        with pytest.raises(RuntimeError, match="view size is not compatible"):
            device_op.get_device_adaptor().scatter_cache(key, cache, slots)
    else:
        assert device_op.get_device_adaptor().scatter_cache(key, cache, slots) is None
        reference = torch.as_strided(before, cache.shape, cache.stride(), cache.storage_offset())
        reference.view(-1, 128)[slots[:2049].long()] = key[:2049]
    torch.testing.assert_close(backing, before, rtol=0, atol=0)
    assert sk.call_count == int(expected_sk)
    pa.assert_not_called()
    assert scatter.call_count == int(not expected_sk and layout != "block_gap")


@pytest.mark.parametrize("family", [AscendDeviceType.A3, AscendDeviceType.A5])
def test_missing_operator_falls_back(monkeypatch, family):
    monkeypatch.setattr(device_op, "get_current_hardware_profile", lambda: get_hardware_profile(family))
    key, cache = torch.ones(2049, 128, dtype=torch.int8), torch.zeros(32, 128, 1, 128, dtype=torch.int8)
    slots = torch.arange(2049, dtype=torch.int32)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", None, raising=False)
    pa = Mock()
    monkeypatch.setattr(torch_npu, "npu_scatter_pa_cache", pa, raising=False)
    scatter = Mock(
        side_effect=lambda target, indices, updates: target.index_copy_(0, indices.flatten().long(), updates)
    )
    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", scatter, raising=False)
    assert device_op.get_device_adaptor().scatter_cache(key, cache, slots) is None
    scatter.assert_called_once()
    pa.assert_not_called()
    torch.testing.assert_close(cache.view(-1, 128)[:2049], key)
    assert not cache.view(-1, 128)[2049:].count_nonzero()


@pytest.mark.parametrize("flat_cache,column_slots", [(True, False), (False, True)])
def test_fast_shape_guard_uses_generic_scatter(monkeypatch, flat_cache, column_slots):
    cache = torch.zeros(2, 4, 1, 16)
    if flat_cache:
        cache = cache.view(-1, 16)
    key = torch.arange(3 * 16, dtype=cache.dtype).view(3, 16)
    slots = torch.tensor([2, 4, 6], dtype=torch.int32)
    if column_slots:
        slots = slots.view(-1, 1)
    fast = Mock()
    scatter = Mock(
        side_effect=lambda target, indices, updates: target.index_copy_(0, indices.flatten().long(), updates)
    )
    monkeypatch.setattr(device_op.BaseDeviceAdaptor, "_scatter_cache", fast)
    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", scatter, raising=False)
    assert device_op.BaseDeviceAdaptor.scatter_cache(key, cache, slots) is None
    fast.assert_not_called()
    scatter.assert_called_once()
    torch.testing.assert_close(cache.view(-1, 16)[[2, 4, 6]], key[:3])
    assert not cache.view(-1, 16)[[0, 1, 3, 5, 7]].count_nonzero()


@pytest.mark.parametrize("family", [AscendDeviceType.A3, AscendDeviceType.A5])
def test_fast_operator_error_is_not_retried(monkeypatch, family):
    monkeypatch.setattr(device_op, "get_current_hardware_profile", lambda: get_hardware_profile(family))
    fast = Mock(side_effect=RuntimeError("operator failed"))
    scatter, pa = Mock(), Mock()
    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", fast, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_pa_cache", pa, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", scatter, raising=False)
    with pytest.raises(RuntimeError, match="operator failed"):
        device_op.get_device_adaptor().scatter_cache(
            torch.ones(1, 16, dtype=torch.int8),
            torch.zeros(1, 4, 1, 16, dtype=torch.int8),
            torch.zeros(1, dtype=torch.int32),
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
    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", sk, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_pa_cache", pa, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", scatter, raising=False)

    assert device_op.get_device_adaptor().scatter_cache(key, cache, slots) is None

    assert sk.call_count == int(family == AscendDeviceType.A5)
    assert scatter.call_count == int(family != AscendDeviceType.A5)
    pa.assert_not_called()
    torch.testing.assert_close(backing.view(torch.uint8), expected, rtol=0, atol=0)


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
    assert len(store.call_args.args) == 3
    assert store.call_args.args[0] is key
    scatter.assert_not_called()


@pytest.mark.parametrize("tokens", [1, 3])
@pytest.mark.parametrize("fast_available", [False, True])
def test_native_main_c8_cache_packs_all_rows_and_preserves_padding(monkeypatch, tokens, fast_available):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.enable_sparse_sfa_c8 = True
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


@pytest.mark.parametrize("fast_available", [False, True])
@pytest.mark.parametrize(
    "num_decode,raw_slots,gathered_slots",
    [
        (0, [0, 1, 2, 3], [0, 1, 2, 3]),
        (1, [0, 1, 0, 2], [0, 1, 2]),
        (1, [0, 1, -1, 0, 2, -1], [0, 1, -1, 2, -1]),
        (2, [0, 1, 0, 1], [0, 1]),
    ],
    ids=["prefill", "mixed", "mixed-padding", "decode"],
)
def test_pcp_c8_forward_writes_gathered_rows_with_their_slots(
    monkeypatch, fast_available, num_decode, raw_slots, gathered_slots
):
    # Exercise forward -> PCP exec_kv -> base exec_kv -> packing -> adaptor.
    # Model the upstream PCP gather boundary, including replicated decode
    # slots removed from the mixed-batch mapping. Only kernels/collectives
    # and unrelated projections/attention are mocked.
    impl = AscendSFAPCPImpl.__new__(AscendSFAPCPImpl)
    impl.has_indexer = False
    impl.skip_topk = True
    impl.layer_name = "model.layers.2.self_attn.attn"
    impl.layerwise_kv_cache_hook = impl.g_proj = None
    impl.enable_sparse_sfa_c8 = True
    impl.preprocess_type = PreprocessType.NATIVE
    impl.q_lora_rank, impl.kv_lora_rank, impl.qk_rope_head_dim = 2, 4, 2
    impl.num_kv_heads = 1
    impl.c8_cache_dtype = torch.int8
    impl.sfa_qsfa_tile_size = 128
    impl.sfa_qsfa_packed_kv_head_dim = 8
    impl.kv_a_layernorm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-5)
    slots = torch.tensor(raw_slots, dtype=torch.int32)
    aligned_slots = torch.tensor(gathered_slots, dtype=torch.int32)
    local_rows = len(raw_slots) // 2
    hidden = torch.zeros(local_rows, 4)
    meta = metadata(
        num_actual_tokens=2,
        num_input_tokens=local_rows,
        num_decode_tokens=num_decode,
        cos=hidden,
        sin=hidden,
        pcp_slot_mapping=slots,
    )
    gathered = torch.arange(len(gathered_slots) * 6).reshape(-1, 6).float()
    packed = torch.arange(len(gathered_slots) * 8).reshape(-1, 8).to(torch.int8)
    k_nope, k_pe, scale = packed.split([4, 2, 2], dim=-1)
    cache = torch.full((2, 4, 1, 8), -7, dtype=torch.int8)
    impl._compose_sfa_kv_cache = Mock(return_value=(cache,))
    impl._get_indexer_attn_metadata = Mock(return_value=None)
    impl._get_parallel_forward_context = Mock(
        return_value=SimpleNamespace(
            actual_seq_lengths_query=torch.tensor([local_rows]),
            actual_seq_lengths_key=torch.tensor([4]),
            kv_slot_mapping=slots,
            gather_full_o_proj=False,
            topk_num_tokens=local_rows,
        )
    )
    impl._prepare_native_hidden_states = Mock(return_value=hidden)
    impl.fused_qkv_a_proj = Mock(return_value=(torch.zeros(local_rows, 8),))
    impl.q_a_layernorm = Mock(side_effect=lambda x: x)
    impl._q_proj_and_k_up_proj = Mock(return_value=(hidden, hidden))
    impl.rope_single = Mock(return_value=hidden)
    impl._record_query_gather_context = Mock()
    impl._get_indexcache_topk_indices = Mock(return_value=torch.zeros(local_rows, 1, dtype=torch.int32))
    impl._execute_sparse_flash_attention_process = Mock(return_value=hidden)
    impl._v_up_proj = Mock(return_value=hidden)
    impl._finalize_o_proj = Mock()

    def write(target, indices, updates):
        torch.testing.assert_close(indices.flatten(), aligned_slots)
        torch.testing.assert_close(updates, packed)
        valid = indices.flatten() >= 0
        target[indices.flatten()[valid].long()] = updates[valid]

    def try_fast(key, target, indices):
        if fast_available:
            write(target.view(-1, 8), indices, key)
        return fast_available

    generic = Mock(side_effect=write)
    monkeypatch.setattr("vllm_ascend.attention.sfa_v1.DeviceOperator", device_op.BaseDeviceAdaptor)
    monkeypatch.setattr(device_op.BaseDeviceAdaptor, "_scatter_cache", Mock(side_effect=try_fast))
    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", generic, raising=False)
    with (
        patch(
            "vllm_ascend.attention.context_parallel.sfa_cp._gather_prefill_cache_inputs",
            return_value=((gathered, hidden, hidden), aligned_slots),
        ) as gather,
        patch("vllm_ascend.attention.sfa_v1.custom_kv_rmsnorm_rope", return_value=(k_pe, k_nope, scale)) as quantize,
        patch("vllm_ascend.attention.sfa_v1.wait_for_kv_layer_from_connector"),
        patch("vllm_ascend.attention.sfa_v1.notify_kv_cache_written"),
        patch("vllm_ascend.attention.sfa_v1.attention_transfer_window", MagicMock()),
        patch("vllm_ascend.attention.sfa_v1.maybe_save_kv_layer_to_connector"),
    ):
        impl.forward(impl.layer_name, hidden, (cache,), meta, output=torch.empty_like(hidden))

    assert gather.call_args.args[1] is slots
    assert gather.call_args.args[2] == num_decode
    torch.testing.assert_close(quantize.call_args.args[0].reshape(-1, 6), gathered)
    reference = torch.full_like(cache, -7)
    valid = aligned_slots >= 0
    reference.view(-1, 8)[aligned_slots[valid].long()] = packed[valid]
    torch.testing.assert_close(cache, reference)
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
    monkeypatch.setattr(torch.ops._C_ascend, "npu_scatter_nd_update_sk", sk if fast_available else None, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_nd_update_", generic, raising=False)
    monkeypatch.setattr(torch_npu, "npu_scatter_pa_cache", pa, raising=False)
    indexer = SimpleNamespace(
        enable_sparse_li_c8=scale_dtype is not None,
        k_cache=SimpleNamespace(kv_cache=tuple(caches)),
        _use_c8_reshape_optim=lambda: False,
    )
    # Non-quantized forward_k produces [tokens, 1, head_dim]. Metadata still
    # describes only the local tokens; write_cache receives the gathered rows.
    k_li = keys[0] if scale_dtype else keys[0].unsqueeze(1)
    AscendSFAIndexerBackend.write_cache(
        indexer, k_li, keys[1] if scale_dtype else None, slots, SimpleNamespace(num_actual_tokens=2)
    )
    assert sk.call_count == (len(caches) if fast_available else 0)
    assert generic.call_count == (0 if fast_available else len(caches))
    pa.assert_not_called()
    for backing, reference in zip(backings, expected):
        torch.testing.assert_close(backing.view(torch.uint8), reference, rtol=0, atol=0)


def test_indexer_grouped_cache_write_keeps_store_kv_block(monkeypatch):
    key, scale = torch.zeros(4, 128, dtype=torch.int8), torch.ones(4, 1, dtype=torch.float16)
    caches = (torch.empty(2, 4, 1, 128, dtype=key.dtype), torch.empty(2, 4, 1, 1, dtype=scale.dtype))
    indexer = SimpleNamespace(
        enable_sparse_li_c8=True,
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
