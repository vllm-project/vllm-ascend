# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPImpl, AscendSFAPCPDCPImpl


def _make_impl(rank: int, interleave_size: int = 2) -> AscendSFADCPImpl:
    impl = AscendSFADCPImpl.__new__(AscendSFADCPImpl)
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

    with patch(
        "vllm_ascend.attention.context_parallel.sfa_cp.get_pcp_group", return_value=SimpleNamespace(world_size=1)
    ):
        actual = impl._merge_dcp_outputs(output, lse)

    assert actual is expected
    fused_a2a.assert_called_once_with(output, lse, 2, 1, "dcp:0")


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
    fused_a2a.assert_called_once_with(output, lse, 2, 0, "dcp:0")


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


@pytest.mark.parametrize("impl_type", [AscendSFADCPImpl, AscendSFAPCPDCPImpl])
@pytest.mark.parametrize("dcp_size", [4, 16])
@pytest.mark.parametrize("total_heads", [64, 128])
def test_replicated_pcp_query_heads_and_output_merge(impl_type, dcp_size, total_heads):
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
        merge.assert_called_once_with(output, lse, 4 if dcp_size == 16 else 1, 1, "tp:0", "pcp:0")


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
