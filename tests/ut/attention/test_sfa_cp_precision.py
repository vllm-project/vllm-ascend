# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend.attention.sfa_v1 import (
    AscendSFADCPMetadata,
    AscendSFAImpl,
    DCPGatherContext,
)
from vllm_ascend.device.device_op import A5DeviceAdaptor, BaseDeviceAdaptor


def _make_impl(rank: int, interleave_size: int = 2) -> AscendSFAImpl:
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.dcp_enabled = True
    impl.dcp_size = 2
    impl.dcp_rank = rank
    impl._dcp_interleave_size = interleave_size
    impl._dcp_index_topk = 8
    impl._remap_order = torch.arange(8, dtype=torch.float32)
    impl._remap_invalid_index = torch.tensor(-1.0)
    return impl


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


@patch("torch.ops.vllm.dcp_a2a_fused")
def test_sfa_dcp_routes_native_output_merge_to_custom_op(fused_a2a) -> None:
    impl = _make_impl(rank=1)
    impl.dcp_group = SimpleNamespace(unique_name="dcp:0")
    output = torch.empty(3, 4, 8)
    lse = torch.empty(3, 4, 1, dtype=torch.float32)
    expected = torch.empty(3, 2, 8)
    fused_a2a.return_value = expected

    with patch("vllm_ascend.attention.sfa_v1.get_pcp_group", return_value=SimpleNamespace(world_size=1)):
        actual = impl._merge_dcp_outputs(output, lse)

    assert actual is expected
    fused_a2a.assert_called_once_with(output, lse, 2, 1, "dcp:0")


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
        output, lse, 1, 1, scatter_group=None, pcp_group=None, return_lse=True, defer_combine=False
    )


@pytest.mark.parametrize("pcp_enabled", [False, True])
@pytest.mark.parametrize("dcp_size", [4, 16])
@pytest.mark.parametrize("total_heads", [64, 128])
def test_replicated_pcp_query_heads_and_output_merge(pcp_enabled, dcp_size, total_heads):
    # The base class is also used by the MTP draft with logical PCP=1.
    # Physical PCP=4 must not duplicate its TP-local query heads.
    from vllm_ascend.attention import sfa_v1

    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.dcp_enabled = True
    impl.pcp_enabled = pcp_enabled
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
        patch.object(sfa_v1, "get_pcp_group", return_value=pcp),
        patch.object(sfa_v1, "get_tp_group", return_value=tp),
        patch.object(sfa_v1, "all_gather_async", return_value=(gathered, None)) as gather,
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
        merge.assert_called_once_with(output, lse, 4 if dcp_size == 16 else 1, 1, "tp:0", "pcp:0")


def test_dcp_query_gather_preserves_distinct_head_shards():
    gather_dim, pcp_size = 1, 1
    from vllm_ascend.attention import sfa_v1

    impl = _make_impl(rank=0)
    impl.dcp_group = SimpleNamespace(unique_name="dcp:0")
    q = torch.arange(24).reshape(2, 3, 4).float()
    rope = q[..., :2] + 100
    packed = torch.cat((q, rope), dim=-1)
    send = packed.transpose(0, 1).contiguous() if gather_dim == 1 else packed
    gathered = torch.cat((send, send + 1000), dim=0)
    with (
        patch.object(sfa_v1, "get_pcp_group", return_value=SimpleNamespace(world_size=pcp_size)),
        patch.object(
            sfa_v1,
            "prepare_query_head_major",
            side_effect=lambda q, rope: torch.cat((q, rope), dim=-1).transpose(0, 1).contiguous(),
        ),
        patch.object(sfa_v1, "all_gather_async", return_value=(gathered, None)) as gather,
    ):
        context = impl._start_dcp_query_gather(q, rope)
        result_q, result_rope = impl._finish_dcp_gather(context, keep_view=True)
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
    from vllm_ascend.attention import sfa_v1

    impl = _make_impl(rank=0)
    impl.qk_rope_head_dim = 0
    impl.dcp_group = SimpleNamespace(unique_name="dcp:0")
    query = torch.arange(24).reshape(2, 3, 4).float()
    rope = None if missing_rope else query[..., :0]
    packed = query.transpose(0, 1).contiguous()
    gathered = torch.cat((packed, packed + 1000), dim=0)
    with (
        patch.object(sfa_v1, "get_pcp_group", return_value=SimpleNamespace(world_size=pcp_size)),
        patch.object(sfa_v1, "get_tp_group", return_value=SimpleNamespace(world_size=2)),
        patch.object(sfa_v1, "all_gather_async", return_value=(gathered, None)) as gather,
    ):
        context = impl._start_dcp_query_gather(query, rope)
        actual, empty_rope = impl._finish_dcp_gather(context)
    torch.testing.assert_close(gather.call_args.args[0], packed)
    torch.testing.assert_close(actual, torch.cat((query, query + 1000), dim=1))
    assert empty_rope.shape == (*actual.shape[:-1], 0)


@pytest.mark.parametrize("pcp_enabled", [False, True])
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int8])
@pytest.mark.parametrize("padded", [False, True], ids=["valid", "poison_padding"])
def test_nope_dcp_executes_single_cache_attention_with_existing_device_helper(pcp_enabled, prefill, dtype, padded):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.dcp_enabled = True
    impl.pcp_enabled = pcp_enabled
    impl.qk_rope_head_dim = 0
    impl.enable_sparse_sfa_turboquant = False
    impl.scale = 0.25
    impl.dcp_size = 2
    impl.dcp_rank = 0
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

    def merge_output(output, lse):
        events.append("merge")
        torch.testing.assert_close(output, expected)
        assert torch.isfinite(lse).all()
        return output

    op_name = "npu_kv_quant_sparse_flash_attention_vllm" if dtype == torch.int8 else "npu_sparse_flash_attention"
    with (
        patch.object(torch.ops._C_ascend, op_name, side_effect=execute, create=True) as attention,
        patch("vllm_ascend.attention.sfa_v1.enable_sfa_dcp_force_tmajor_restore", return_value=False),
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


@pytest.mark.parametrize("turboquant", [False, True])
@pytest.mark.parametrize("prefetched", [False, True])
@pytest.mark.parametrize("wait_fails", [False, True])
def test_dcp_prefill_consumes_gather_once_for_prefetched_and_direct_attention(turboquant, prefetched, wait_fails):
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.dcp_enabled = True
    impl.dcp_group = object()
    impl.qk_rope_head_dim = 1
    impl.enable_sparse_sfa_turboquant = turboquant
    impl._start_dcp_query_gather = Mock()
    impl._merge_dcp_outputs = Mock()
    query = torch.ones(2, 1, 1)
    lengths = torch.tensor([2])
    metadata = AscendSFADCPMetadata.__new__(AscendSFADCPMetadata)
    metadata.num_prefills = 1
    metadata.pcp_has_global_prefill = False
    metadata.dcp_context = SimpleNamespace(gather_context=None, kv_gather_block_table=torch.tensor([[0, 1]]))
    failure = RuntimeError("injected gather wait failure")

    def wait():
        assert metadata.dcp_context.gather_context is None
        if wait_fails:
            raise failure

    handle = SimpleNamespace(wait=Mock(side_effect=wait))
    gathered = torch.arange(8.0).reshape(2, 1, 1, 4)
    context = DCPGatherContext(gathered, handle, None, (4,) if turboquant else (2, 2))

    def start(cache, attn_metadata):
        assert attn_metadata is metadata
        assert attn_metadata.dcp_context.gather_context is None
        attn_metadata.dcp_context.gather_context = context

    impl._record_dcp_kv_gather_context = Mock(side_effect=start)
    if prefetched:
        metadata.dcp_context.gather_context = context

    def attention(*args, **kwargs):
        handle.wait.assert_called_once()
        assert metadata.dcp_context.gather_context is None
        assert kwargs["sparse_mode"] == 3
        return (query, None, None) if turboquant else query

    impl._tq_rotate_query = Mock(return_value=query)
    impl._turboquant_sfa = Mock(side_effect=attention)
    impl.turboquant = SimpleNamespace(inverse=Mock(side_effect=lambda value: value + 1))
    with patch(
        "vllm_ascend.attention.sfa_v1.DeviceOperator.execute_sparse_flash_attention_process", side_effect=attention
    ) as execute:
        if wait_fails:
            with pytest.raises(RuntimeError) as error:
                impl._execute_sparse_flash_attention_process(
                    query, query, (gathered,), None, metadata, lengths, lengths
                )
            assert error.value is failure
            execute.assert_not_called()
            impl._turboquant_sfa.assert_not_called()
        else:
            output = impl._execute_sparse_flash_attention_process(
                query, query, (gathered,), None, metadata, lengths, lengths
            )
            torch.testing.assert_close(output, query + int(turboquant))
            assert execute.call_count == int(not turboquant)
            assert impl._turboquant_sfa.call_count == int(turboquant)
            assert impl.turboquant.inverse.call_count == int(turboquant)
    handle.wait.assert_called_once()
    assert impl._record_dcp_kv_gather_context.call_count == int(not prefetched)
    assert metadata.dcp_context.gather_context is None
    impl._start_dcp_query_gather.assert_not_called()
    impl._merge_dcp_outputs.assert_not_called()
