# SPDX-License-Identifier: Apache-2.0
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.attention.context_parallel.sfa_cp import (
    AscendSFADCPImpl,
    AscendSFADCPMetadata,
    AscendSFAPCPDCPImpl,
    DCPGatherContext,
)
from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor


def _make_impl(rank: int, interleave_size: int = 2) -> AscendSFADCPImpl:
    impl = AscendSFADCPImpl.__new__(AscendSFADCPImpl)
    impl._sharded_indexer = False
    impl.dcp_size = 2
    impl.dcp_rank = rank
    impl._dcp_interleave_size = interleave_size
    impl._dcp_index_topk = 8
    impl.qk_rope_head_dim = 1
    return impl


@pytest.mark.parametrize("restore_perm", [None, (1, 0, 2)])
@pytest.mark.parametrize("force_tmajor", [False, True])
@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_quantized_dcp_passes_packed_query_without_split(restore_perm, force_tmajor, dtype):
    from vllm_ascend.attention.context_parallel import sfa_cp

    impl = _make_impl(0)
    impl.enable_sparse_sfa_turboquant = False
    impl._local_topk = True
    impl.dcp_group = object()
    query = torch.arange(3 * 96 * 6).reshape(3, 96, 6).float()
    storage = query.transpose(0, 1).contiguous() if restore_perm else query
    handle = Mock()
    gather = sfa_cp.DCPGatherContext(storage, handle, restore_perm, (4, 2))
    context = SimpleNamespace(gather_context=gather, kv_gather_block_ids=None, seq_lens=object(), block_table=object())
    metadata = SimpleNamespace(dcp_context=context)
    output = torch.ones(3, 96, 4)
    with (
        patch.object(impl, "_has_prefill", return_value=False),
        patch.object(sfa_cp, "enable_sfa_dcp_force_tmajor_restore", return_value=force_tmajor),
        patch.object(
            sfa_cp.DeviceOperator,
            "execute_sparse_flash_attention_process",
            return_value=(output, torch.zeros(1, 3, 96), torch.ones(1, 3, 96)),
        ) as execute,
        patch.object(impl, "_merge_dcp_outputs", return_value=output),
        patch("torch.split", side_effect=AssertionError("packed DCP query must not be split")),
    ):
        actual = impl._execute_sparse_flash_attention_process(
            None, None, (torch.empty(1, dtype=dtype),), None, metadata, None, None
        )
    assert actual is output
    handle.wait.assert_called_once()
    assert context.gather_context is None
    packed = execute.call_args.kwargs["packed_query"]
    torch.testing.assert_close(packed, query)
    if not force_tmajor:
        assert packed.data_ptr() == storage.data_ptr()
    assert packed.is_contiguous() == (force_tmajor or restore_perm is None)
    assert execute.call_args.kwargs["return_lse"] is True


@pytest.mark.parametrize(
    "heads,force_tmajor,strided_cache,paired_enabled",
    [
        (32, False, True, False),
        (96, False, True, False),
        (96, False, True, True),
        (96, True, True, True),
        (96, False, False, True),
    ],
)
def test_kimi_bf16_dcp_layout_at_native_sfa(heads, force_tmajor, strided_cache, paired_enabled):
    from vllm_ascend.attention.context_parallel import sfa_cp
    from vllm_ascend.device import device_op

    impl = _make_impl(0)
    impl.is_kimi_k31 = impl._local_topk = True
    impl.enable_sparse_sfa_turboquant = False
    impl.qk_rope_head_dim, impl.scale, impl.dcp_group = 64, 1.0, object()
    impl._paired_page_sfa = paired_enabled
    packed = torch.randn(heads, 3, 576, dtype=torch.bfloat16)
    q, qr = packed.transpose(0, 1).split((512, 64), dim=-1)
    handle = Mock()
    lengths = torch.tensor([3], dtype=torch.int32)
    table = torch.tensor([[1, 0]], dtype=torch.int32)
    indices = torch.zeros(3, 1, 8, dtype=torch.int32)
    gather = DCPGatherContext(packed, handle, (1, 0, 2), (512, 64))
    metadata = SimpleNamespace(dcp_context=SimpleNamespace(gather_context=gather, seq_lens=lengths, block_table=table))
    # Give the layer a non-zero offset in the shared allocation.
    pages = torch.randn(3, 128 * 576, dtype=torch.bfloat16)[1:]
    k, kr = (plane.view(2, 128, 1, -1) for plane in pages.split((128 * 512, 128 * 64), dim=-1))
    if not strided_cache:
        k, kr = k.contiguous(), kr.contiguous()

    def reference_q(a, b, output_heads):
        padding = (0, 0, 0, output_heads - heads)
        return tuple(torch.nn.functional.pad(x, padding).contiguous() for x in (a, b))

    with (
        patch.object(impl, "_has_prefill", return_value=False),
        patch.object(sfa_cp, "enable_sfa_dcp_force_tmajor_restore", return_value=force_tmajor),
        patch.object(device_op, "prepare_sfa_query_pair", side_effect=reference_q) as prep_q,
        patch.object(
            device_op, "prepare_sfa_kv_pair", side_effect=lambda a, b: (a.contiguous(), b.contiguous())
        ) as prep_kv,
        patch.object(device_op.logger, "info_once") as log,
        patch.object(torch.ops._C_ascend, "npu_sparse_flash_attention", create=True) as sfa,
        patch.object(impl, "_merge_dcp_outputs", side_effect=lambda output, *args, **kwargs: output) as merge,
    ):
        native_heads = 128 if heads == 96 else heads
        expected_q, expected_qr = reference_q(q, qr, native_heads)
        maximum = torch.randn(1, 3, native_heads)
        total = torch.ones_like(maximum)
        sfa.return_value = (expected_q, maximum, total)
        result = impl._execute_sparse_flash_attention_process(None, None, (k, kr), indices, metadata, lengths, lengths)
    handle.wait.assert_called_once()
    assert prep_q.call_count == int(not force_tmajor)
    copied = not paired_enabled and not force_tmajor and strided_cache
    direct = paired_enabled and not force_tmajor and strided_cache
    assert prep_kv.call_count == int(copied)
    native = sfa.call_args.kwargs
    assert native["sparse_indices"] is indices and native["block_table"] is table
    assert native["return_softmax_lse"] and native["sparse_mode"] == 0
    assert native["value"] is native["key"]
    if direct:
        assert native["layout_kv"] == "PA_KR_BSND"
        assert native["key_rope"] is native["key"] and native["key"].data_ptr() == pages.data_ptr()
        expect_k = expect_r = pages.view(2, 128, 1, 576)
    else:
        assert native["layout_kv"] == "PA_BSND"
        expect_k, expect_r = k, kr
    for name, expected in (
        ("query", expected_q),
        ("query_rope", expected_qr),
        ("key", expect_k),
        ("key_rope", expect_r),
    ):
        torch.testing.assert_close(native[name], expected)
        if not force_tmajor:
            assert native[name].is_contiguous()
    torch.testing.assert_close(result, q)
    torch.testing.assert_close(merge.call_args.args[1], maximum[..., :heads].transpose(0, 1).reshape(3, heads, 1))
    torch.testing.assert_close(
        merge.call_args.kwargs["softmax_sum"], total[..., :heads].transpose(0, 1).reshape(3, heads, 1)
    )
    route = "separate_pad" if force_tmajor else "fused_query_pair"
    cache_route = "paired_page_direct" if direct else "fused_kv_pair" if copied else "native"
    assert any(c.args[1:3] == (route, cache_route) for c in log.call_args_list)


@pytest.mark.parametrize(
    "dtype,force_restore,expect_probe",
    [
        (torch.bfloat16, False, True),
        (torch.float16, False, False),
        (torch.bfloat16, True, False),
    ],
)
def test_dcp_paired_page_probe_at_weight_load(dtype, force_restore, expect_probe):
    from vllm_ascend.attention.context_parallel import sfa_cp

    impl = _make_impl(0)
    impl.is_kimi_k31 = True
    impl.enable_sparse_sfa_c8 = impl.enable_sparse_sfa_turboquant = False
    impl.kv_lora_rank, impl.qk_rope_head_dim = 512, 64
    impl.W_UV = SimpleNamespace(device=SimpleNamespace(type="npu"))
    profile = SimpleNamespace(device_adaptor_family=sfa_cp.DeviceAdaptorFamily.FP8_OPTIMIZED)
    with (
        patch.object(sfa_cp.AscendSFAImpl, "process_weights_after_loading") as base,
        patch.object(sfa_cp, "get_current_hardware_profile", return_value=profile),
        patch.object(sfa_cp, "enable_sfa_dcp_force_tmajor_restore", return_value=force_restore),
        patch("vllm_ascend.ops.sfa_paired_page.paired_page_sfa_available", return_value=True) as probe,
    ):
        impl.process_weights_after_loading(dtype)
    base.assert_called_once_with(dtype)
    assert impl._paired_page_sfa is expect_probe
    if expect_probe:
        probe.assert_called_once_with(impl.W_UV.device)
    else:
        probe.assert_not_called()


def test_sfa_paired_pages_rejects_other_cache_layouts():
    from vllm_ascend.device.device_op import _sfa_paired_pages

    block = 128
    flat = torch.empty(3, block * 576, dtype=torch.bfloat16)

    def split(owner):
        return (
            owner[:, : block * 512].view(2, block, 1, 512),
            owner[:, block * 512 : block * 576].view(2, block, 1, 64),
        )

    k, kr = split(flat[:2])
    assert _sfa_paired_pages(k, split(flat.clone()[:2])[1]) is None
    assert _sfa_paired_pages(k, split(flat[1:])[1]) is None
    assert _sfa_paired_pages(*flat[:2].view(2, block, 1, 576).split((512, 64), -1)) is None
    assert _sfa_paired_pages(*split(torch.empty(2, block * 576 + 64, dtype=k.dtype))) is None
    assert _sfa_paired_pages(*split(flat.flatten()[1 : 1 + 2 * block * 576].view(2, -1))) is None


@pytest.mark.parametrize("force_restore", [True, False])
def test_kimi_mixed_dcp_compacts_gathered_kv_once(force_restore):
    from vllm_ascend.attention.context_parallel import sfa_cp
    from vllm_ascend.device import device_op

    impl = _make_impl(0)
    impl.is_kimi_k31 = True
    impl.scale, impl.dcp_group, impl.enable_sparse_sfa_turboquant = 0.5, object(), False
    query = torch.ones(2, 4, 8, dtype=torch.bfloat16)
    qr = torch.zeros(2, 4, 1, dtype=query.dtype)
    packed = torch.randn(4, 2, 1, 9, dtype=query.dtype)
    handle = Mock()
    gather = DCPGatherContext(packed, handle, None, (8, 1))
    table = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32)
    lengths = torch.tensor([2], dtype=torch.int32)
    indices = torch.zeros(2, 1, 2, dtype=torch.int32)
    metadata = SimpleNamespace(
        num_prefills=1, dcp_context=SimpleNamespace(gather_context=gather, kv_gather_block_table=table)
    )
    with (
        patch.object(sfa_cp, "HAS_TRITON", True),
        patch.object(sfa_cp, "enable_sfa_dcp_force_tmajor_restore", return_value=force_restore),
        patch.object(device_op, "prepare_sfa_query_pair") as prep_q,
        patch.object(
            device_op, "prepare_sfa_kv_pair", side_effect=lambda k, r: (k.contiguous(), r.contiguous())
        ) as prep_kv,
        patch.object(
            torch.ops._C_ascend, "npu_sparse_flash_attention", create=True, return_value=(query, None, None)
        ) as sfa,
    ):
        result = impl._execute_sparse_flash_attention_process(query, qr, (packed,), indices, metadata, lengths, lengths)
    assert result is query
    handle.wait.assert_called_once()
    prep_q.assert_not_called()
    assert prep_kv.call_count == int(not force_restore)
    native = sfa.call_args.kwargs
    assert native["key"] is native["value"] and native["block_table"] is table
    assert native["sparse_mode"] == 3 and not native["return_softmax_lse"]
    for plane, original in zip((native["key"], native["key_rope"]), packed.split((8, 1), -1)):
        assert plane.is_contiguous()
        torch.testing.assert_close(plane, original)


@pytest.fixture(autouse=True)
def sparse_remap(monkeypatch):
    def reference(indices, world, rank, interleave):
        result = torch.full_like(indices, -1)
        for ids, out in zip(indices.reshape(-1, indices.shape[-1]), result.reshape(-1, indices.shape[-1])):
            owned = ids[(ids >= 0) & ((ids // interleave) % world == rank)]
            out[: owned.numel()] = owned // (world * interleave) * interleave + owned % interleave
        return result

    module: Any = ModuleType("vllm_ascend.ops.triton.sparse_index_remap")
    module.remap_sparse_indices_triton = Mock(side_effect=reference)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module.remap_sparse_indices_triton


@pytest.fixture(autouse=True)
def query_prep(monkeypatch):
    module: Any = ModuleType("vllm_ascend.ops.triton.query_gather_prep")
    module.prep_query_head_major = Mock(
        side_effect=lambda q, rope: torch.cat((q, rope), -1).transpose(0, 1).contiguous()
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module.prep_query_head_major


def test_sfa_remap_propagates_triton_failure(sparse_remap):
    sparse_remap.side_effect = RuntimeError("remap failed")
    with pytest.raises(RuntimeError, match="remap failed"):
        _make_impl(0)._remap_sparse_indices(torch.zeros(1, 8, dtype=torch.int32))


def test_sfa_query_prep_propagates_triton_failure(query_prep):
    impl = _make_impl(0)
    impl.dcp_group = object()
    query_prep.side_effect = RuntimeError("query prep failed")
    with pytest.raises(RuntimeError, match="query prep failed"):
        impl._start_dcp_query_gather(torch.zeros(1, 2, 4), torch.zeros(1, 2, 0))


def test_sfa_dcp_sparse_indices_are_compacted_per_owner_rank() -> None:
    replicated_indices = torch.tensor([[0, 2, 1, 3, 4, 6, -1, 5]], dtype=torch.int32)

    rank0 = _make_impl(0)._remap_sparse_indices(replicated_indices)
    rank1 = _make_impl(1)._remap_sparse_indices(replicated_indices)

    torch.testing.assert_close(
        rank0,
        torch.tensor([[0, 1, 2, 3, -1, -1, -1, -1]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        rank1,
        torch.tensor([[0, 1, 2, -1, -1, -1, -1, -1]], dtype=torch.int32),
    )


@pytest.mark.parametrize("local_output", [False, True])
def test_sfa_consumes_shared_local_topk_without_second_remap(local_output):
    from vllm_ascend.attention.context_parallel import sfa_cp

    impl = _make_impl(rank=0)
    impl._local_topk = local_output
    impl._sharded_indexer = True
    impl.enable_sparse_sfa_turboquant = False
    impl.dcp_group = SimpleNamespace()
    impl.topk_indices_buffer = torch.full((1, 8), -1, dtype=torch.int32)
    global_ids = torch.tensor([[[0, 2, 1, 3, 4, 6, -1, 5]]], dtype=torch.int32)
    expected = impl._remap_sparse_indices(global_ids)
    impl._update_indexcache_topk_indices(expected if local_output else global_ids)
    shared = impl._get_indexcache_topk_indices(1)
    metadata = SimpleNamespace(dcp_context=SimpleNamespace(gather_context=object(), seq_lens=None, block_table=None))
    output = torch.ones(1, 2, 4)
    with (
        patch.object(impl, "_has_prefill", return_value=False),
        patch.object(impl, "_finish_dcp_gather", return_value=(None, None)),
        patch.object(impl, "_remap_sparse_indices", wraps=impl._remap_sparse_indices) as remap,
        patch.object(
            sfa_cp.DeviceOperator,
            "execute_sparse_flash_attention_process",
            return_value=(output, torch.zeros(2, 1, 1), torch.ones(2, 1, 1)),
        ) as execute,
        patch.object(sfa_cp, "enable_sfa_dcp_force_tmajor_restore", return_value=False),
        patch.object(impl, "_merge_dcp_outputs", return_value=output),
    ):
        actual = impl._execute_sparse_flash_attention_process(
            None, None, (torch.empty(1, dtype=torch.bfloat16),), shared, metadata, None, None
        )
    assert actual is output
    assert remap.call_count == int(not local_output)
    assert torch.equal(execute.call_args.args[4], expected)


@patch("torch.ops.vllm.dcp_a2a_fused")
def test_sfa_dcp_routes_native_output_merge_to_custom_op(fused_a2a) -> None:
    impl = _make_impl(rank=1)
    impl.dcp_group = SimpleNamespace(unique_name="dcp:0")
    output = torch.empty(3, 4, 8)
    lse = torch.empty(3, 4, 1, dtype=torch.float32)
    expected = torch.empty(3, 2, 8)
    fused_a2a.return_value = expected

    with patch(
        "vllm_ascend.attention.context_parallel.sfa_cp.get_pcp_group", return_value=SimpleNamespace(world_size=1)
    ):
        actual = impl._merge_dcp_outputs(output, lse)

    assert actual is expected
    fused_a2a.assert_called_once_with(output, lse, 2, 1, "dcp:0", softmax_sum=None, use_sfa_fast_exchange=True)


@patch("torch.ops.vllm.dcp_a2a_fused")
def test_sfa_dsa_dcp_routes_token_scatter_to_custom_op(fused_a2a) -> None:
    impl = _make_impl(rank=1)
    impl.dcp_group = SimpleNamespace(unique_name="dcp:0")
    output = torch.empty(4, 2, 8)
    lse = torch.empty(4, 2, 1, dtype=torch.float32)
    expected = torch.empty(2, 2, 8)
    fused_a2a.return_value = expected
    dsa_cp_context = SimpleNamespace(
        num_tokens_pad=4,
        local_start=2,
        local_end_with_pad=4,
    )

    actual = impl._merge_dcp_outputs(output, lse, dsa_cp_context)

    assert actual is expected
    fused_a2a.assert_called_once_with(output, lse, 2, 0, "dcp:0", softmax_sum=None, use_sfa_fast_exchange=True)


@pytest.mark.parametrize("scatter_dim", [0, 1])
@pytest.mark.parametrize(
    "dtype,return_lse",
    [
        (torch.float32, False),
        (torch.float32, True),
        (torch.float16, False),
        (torch.bfloat16, False),
    ],
)
def test_sfa_custom_op_optional_lse_fake_shape(scatter_dim, dtype, return_lse):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        output = torch.empty(48, 96, 512, dtype=dtype)
        lse = torch.empty(48, 96, 1, dtype=torch.float32)
        merged = torch.ops.vllm.dcp_a2a_fused(output, lse, 16, scatter_dim, "fake-dcp", return_lse=return_lse)
        expected = (3, 96, 512 + int(return_lse)) if scatter_dim == 0 else (48, 6, 512 + int(return_lse))
        assert merged.shape == expected
        assert merged.dtype == dtype
        assert merged.device == output.device


def test_sfa_custom_op_passes_optional_lse_to_combine():
    import vllm_ascend.ops.triton.dcp.dcp_a2a as kernels

    output = torch.randn(2, 3, 4)
    lse = torch.randn(2, 3, 1)
    expected = torch.randn(2, 3, 5)
    with patch.object(kernels, "dcp_a2a_fused_combine", return_value=expected) as combine:
        actual = kernels.dcp_a2a_fused(output, lse, 1, 1, "", return_lse=True)
    assert actual is expected
    combine.assert_called_once_with(
        output,
        lse,
        1,
        1,
        scatter_group=None,
        pcp_group=None,
        return_lse=True,
        defer_combine=False,
        softmax_sum=None,
        use_sfa_fast_exchange=False,
    )


@pytest.mark.parametrize("impl_type", [AscendSFADCPImpl, AscendSFAPCPDCPImpl])
@pytest.mark.parametrize("dcp_size", [4, 16])
@pytest.mark.parametrize("total_heads", [64, 128])
def test_replicated_pcp_query_heads_and_output_merge(impl_type, dcp_size, total_heads, query_prep):
    # The base class is also used by the MTP draft with logical PCP=1.
    # Physical PCP=4 must not duplicate its TP-local query heads.
    from vllm_ascend.attention.context_parallel import sfa_cp

    impl = impl_type.__new__(impl_type)
    impl.dcp_size = dcp_size
    impl.dcp_group = SimpleNamespace(world_size=dcp_size, unique_name="dcp:0")
    pcp = SimpleNamespace(world_size=4, unique_name="pcp:0")
    tp = SimpleNamespace(world_size=4, unique_name="tp:0")
    q = torch.arange(2 * total_heads // 4 * 8).reshape(2, total_heads // 4, 8).float()
    rope = q[..., :2] + 1000
    packed = torch.cat((q, rope), dim=-1).transpose(0, 1).contiguous()
    tp_shards = [packed + rank * 10000 for rank in range(4)]
    gathered = torch.cat(tp_shards, dim=0)
    with (
        patch.object(sfa_cp, "get_pcp_group", return_value=pcp),
        patch.object(sfa_cp, "get_tp_group", return_value=tp),
        patch.object(sfa_cp, "all_gather_async", return_value=(gathered, None)) as gather,
        patch("torch.ops.vllm.dcp_a2a_fused") as merge,
    ):
        context = impl._start_dcp_query_gather(q, rope)
        restored = context.gathered.permute(context.restore_perm)
        if dcp_size == 4:
            gather.assert_not_called()
            torch.testing.assert_close(restored, torch.cat((q, rope), dim=-1))
        else:
            assert gather.call_count == 1
            assert gather.call_args.args[1] is tp
            torch.testing.assert_close(gather.call_args.args[0], packed)
            torch.testing.assert_close(restored, gathered.transpose(0, 1))
        assert restored.shape[1] == (total_heads if dcp_size == 16 else total_heads // 4)
        assert context.split_sizes == (8, 2)
        output = restored[..., :8]
        lse = torch.zeros(*output.shape[:-1], 1)
        expected = torch.empty_like(q)
        merge.return_value = expected
        assert impl._merge_dcp_outputs(output, lse) is expected
        merge.assert_called_once_with(
            output,
            lse,
            4 if dcp_size == 16 else 1,
            1,
            "tp:0",
            "pcp:0",
            softmax_sum=None,
            use_sfa_fast_exchange=True,
        )


@pytest.mark.parametrize("gather_dim,pcp_size", [(1, 1), (0, 1), (0, 4)])
def test_dcp_query_gather_preserves_distinct_head_or_token_shards(gather_dim, pcp_size):
    from vllm_ascend.attention.context_parallel import sfa_cp

    impl = _make_impl(rank=0)
    impl.dcp_group = SimpleNamespace(unique_name="dcp:0")
    q = torch.arange(24).reshape(2, 3, 4).float()
    rope = q[..., :2] + 100
    packed = torch.cat((q, rope), dim=-1)
    send = packed.transpose(0, 1).contiguous() if gather_dim == 1 else packed
    gathered = torch.cat((send, send + 1000), dim=0)
    with (
        patch.object(impl, "_parallel_query_gather_dim", return_value=gather_dim),
        patch.object(sfa_cp, "get_pcp_group", return_value=SimpleNamespace(world_size=pcp_size)),
        patch.object(sfa_cp, "all_gather_async", return_value=(gathered, None)) as gather,
    ):
        context = impl._start_dcp_query_gather(q, rope)
        result_q, result_rope = impl._finish_dcp_gather(context)
    assert gather.call_args.args[1] is impl.dcp_group
    torch.testing.assert_close(gather.call_args.args[0], send)
    torch.testing.assert_close(result_q, torch.cat((q, q + 1000), dim=gather_dim))
    torch.testing.assert_close(result_rope, torch.cat((rope, rope + 1000), dim=gather_dim))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int8])
def test_nope_dcp_prefill_gathers_one_latent_cache_at_logical_page_size(dtype):
    impl = _make_impl(rank=0)
    impl.qk_rope_head_dim = 0
    impl.enable_sparse_sfa_c8 = dtype == torch.int8
    impl.enable_sparse_sfa_turboquant = False
    impl.dcp_group = object()
    cache = torch.arange(2 * 256 * 8).reshape(2, 256, 1, 8).to(dtype)
    blocks = torch.tensor([3, 1], dtype=torch.int64)
    metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
    metadata.num_prefills = 1
    metadata.pcp_has_global_prefill = False
    metadata.block_size = 128
    metadata.dcp_context = SimpleNamespace(
        kv_gather_block_ids=blocks, kv_gather_block_table=torch.tensor([[0, 1]]), gather_context=None
    )
    context = object()
    with patch.object(impl, "_start_dcp_gather", return_value=context) as gather:
        impl._record_dcp_kv_gather_context((cache,), metadata)
    torch.testing.assert_close(gather.call_args.args[0], cache.view(4, 128, 1, 8)[blocks])
    assert gather.call_args.kwargs == {"dim": 0, "split_sizes": (8,)}
    assert metadata.dcp_context.gather_context is context


@pytest.mark.parametrize("pcp_size", [1, 4])
@pytest.mark.parametrize("missing_rope", [False, True])
def test_nope_dcp_query_gather_accepts_absent_or_empty_rope(pcp_size, missing_rope):
    from vllm_ascend.attention.context_parallel import sfa_cp

    impl = _make_impl(rank=0)
    impl.qk_rope_head_dim = 0
    impl.dcp_group = SimpleNamespace(unique_name="dcp:0")
    query = torch.arange(24).reshape(2, 3, 4).float()
    rope = None if missing_rope else query[..., :0]
    packed = query.transpose(0, 1).contiguous()
    gathered = torch.cat((packed, packed + 1000), dim=0)
    with (
        patch.object(sfa_cp, "get_pcp_group", return_value=SimpleNamespace(world_size=pcp_size)),
        patch.object(sfa_cp, "get_tp_group", return_value=SimpleNamespace(world_size=2)),
        patch.object(sfa_cp, "all_gather_async", return_value=(gathered, None)) as gather,
    ):
        context = impl._start_dcp_query_gather(query, rope)
        actual, empty_rope = impl._finish_dcp_gather(context)
    torch.testing.assert_close(gather.call_args.args[0], packed)
    torch.testing.assert_close(actual, torch.cat((query, query + 1000), dim=1))
    assert empty_rope.shape == (*actual.shape[:-1], 0)


@pytest.mark.parametrize("impl_type", [AscendSFADCPImpl, AscendSFAPCPDCPImpl])
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int8])
@pytest.mark.parametrize("padded", [False, True], ids=["valid", "poison_padding"])
def test_nope_dcp_executes_single_cache_attention_with_existing_device_helper(impl_type, prefill, dtype, padded):
    impl = impl_type.__new__(impl_type)
    impl.qk_rope_head_dim = 0
    impl.enable_sparse_sfa_turboquant = False
    impl.scale = 0.25
    impl.dcp_size = 2
    impl.dcp_rank = 0
    impl._local_topk = False
    impl._sharded_indexer = False
    impl.is_kimi_k31 = False
    events = []
    query = torch.randn(2, 2, 8, dtype=torch.bfloat16)
    cache = torch.zeros(2, 256, 1, 8, dtype=dtype)
    indices = torch.zeros(2, 1, 4, dtype=torch.int32)
    lengths = torch.tensor([1 if padded else 2], dtype=torch.int32)
    gathered = cache.view(4, 128, 1, 8) if prefill else query
    handle = Mock()
    handle.wait.side_effect = lambda: events.append("wait")
    metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
    metadata.num_prefills = int(prefill)
    metadata.pcp_has_global_prefill = False
    metadata.block_size = 128
    metadata.cum_query_lens = lengths
    metadata.dcp_context = SimpleNamespace(
        gather_context=DCPGatherContext(
            gathered=gathered,
            handle=handle,
            restore_perm=None,
            split_sizes=(8,) if prefill else (8, 0),
        ),
        kv_gather_block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
        seq_lens=lengths,
    )
    impl.dcp_group = SimpleNamespace(all_gather=Mock(return_value=indices))
    expected = torch.ones_like(query)
    kernel_output = torch.ones_like(query)
    softmax_max, softmax_sum = torch.zeros(1, 2, 2), torch.ones(1, 2, 2)
    if padded:
        expected[-1] = 0
        kernel_output[-1] = float("nan")
        softmax_max[:, -1] = softmax_sum[:, -1] = float("nan")

    def execute(**kwargs):
        events.append("attention")
        assert kwargs["key"].shape == (4, 128, 1, 8)
        assert kwargs["sparse_mode"] == (3 if prefill else 0)
        assert kwargs["return_softmax_lse"] is not prefill
        if dtype == torch.int8:
            torch.testing.assert_close(kwargs["query"], query)
            assert kwargs["rope_head_dim"] == 0
        else:
            assert kwargs["key_rope"] is None
            assert kwargs["query_rope"] is None
        return kernel_output, softmax_max, softmax_sum

    def merge_output(output, lse, _dsa_context=None, *, softmax_sum):
        events.append("merge")
        torch.testing.assert_close(output, expected)
        assert torch.isfinite(lse).all()
        assert torch.isfinite(softmax_sum).all()
        return output

    op_name = "npu_kv_quant_sparse_flash_attention_vllm" if dtype == torch.int8 else "npu_sparse_flash_attention"
    with (
        patch.object(torch.ops._C_ascend, op_name, side_effect=execute, create=True) as attention,
        patch("vllm_ascend.attention.context_parallel.sfa_cp.enable_sfa_dcp_force_tmajor_restore", return_value=False),
        patch.object(impl, "_remap_sparse_indices", side_effect=lambda values: values),
        patch.object(impl, "_merge_dcp_outputs", side_effect=merge_output) as merge,
    ):
        actual = impl._execute_sparse_flash_attention_process(
            query, None, (cache,), indices, metadata, lengths, lengths
        )
    torch.testing.assert_close(actual, expected)
    attention.assert_called_once()
    assert metadata.dcp_context.gather_context is None
    assert events == (["wait", "attention"] if prefill else ["wait", "attention", "merge"])
    assert merge.call_count == int(not prefill)
    impl.dcp_group.all_gather.assert_not_called()


@pytest.mark.parametrize("return_lse", [False, True])
def test_a5_nope_quantized_attention_accepts_absent_rope(return_lse):
    impl = SimpleNamespace(qk_rope_head_dim=0)
    query = torch.randn(2, 2, 8)
    cache = torch.zeros(2, 128, 1, 8, dtype=torch.int8)
    result = object()
    with patch.object(
        BaseDeviceAdaptor, "_execute_kv_quant_sparse_flash_attention_vllm", return_value=result
    ) as execute:
        actual = A5DeviceAdaptor._execute_kv_quant_sparse_flash_attention_vllm(
            impl,
            query,
            None,
            cache,
            torch.zeros(1, 1),
            torch.zeros(2, 1, 4),
            torch.tensor([2]),
            torch.tensor([2]),
            return_lse=return_lse,
        )
    assert actual is result
    assert execute.call_args.args[2] is None
    assert execute.call_args.kwargs["return_lse"] is return_lse
