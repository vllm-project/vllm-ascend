# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Single-NPU checks for MoeGatingTopKHash and its fused EPLB map."""

import pytest
import torch
import torch_npu

from vllm_ascend.distributed.eplb.tensor_ops import build_expert_replica_routing_table
from vllm_ascend.ops.fused_moe.eplb import map_to_physical_for_ascend
from vllm_ascend.utils import enable_custom_op

enable_custom_op()

IMAGE_SENTINEL_LO = 129257
IMAGE_SENTINEL_COUNT = 5


@pytest.mark.parametrize("device", ["meta", "npu"])
@pytest.mark.parametrize("norm_type,renorm,error", [(3, 0, "norm_type"), (0, 2, "renorm"), (2, 1, "renorm")])
def test_ordinary_map_rejects_out_of_scope_modes(device, norm_type, renorm, error):
    logits = torch.empty(3, 256, device=device)
    table = torch.zeros(7, 256, device=device, dtype=torch.int32)
    with pytest.raises(RuntimeError, match=error):
        torch.ops._C_ascend.moe_gating_top_k_hash(
            logits,
            k=8,
            norm_type=norm_type,
            renorm=renorm,
            expert_replica_routing_table=table,
        )


@pytest.mark.parametrize("norm_type,renorm", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_ordinary_map_meta(norm_type, renorm):
    logits = torch.empty(3, 256, device="meta", dtype=torch.bfloat16)
    table = torch.empty(7, 256, device="meta", dtype=torch.int32)
    weights, ids, scores = torch.ops._C_ascend.moe_gating_top_k_hash(
        logits,
        k=8,
        group_count=8,
        k_group=4,
        norm_type=norm_type,
        renorm=renorm,
        expert_replica_routing_table=table,
    )
    assert weights.shape == ids.shape == (3, 8)
    assert weights.dtype == logits.dtype
    assert ids.dtype == torch.int32
    assert scores.shape == logits.shape and scores.dtype == torch.float32


@pytest.mark.parametrize("device", ["meta", "npu"])
@pytest.mark.parametrize(
    "norm_type,renorm,with_map,extra_input",
    [(0, 1, False, "input_ids"), (1, 1, False, "tid2eid"), (0, 0, True, "bias_vl"), (1, 0, True, "input_ids")],
)
def test_ordinary_routing_rejects_hash_vision_inputs(device, norm_type, renorm, with_map, extra_input):
    logits = torch.empty(3, 256, device=device)
    table = torch.zeros(7, 256, device=device, dtype=torch.int32) if with_map else None
    inputs = {
        "input_ids": torch.zeros(3, device=device, dtype=torch.int32),
        "tid2eid": torch.zeros(7, 8, device=device, dtype=torch.int32),
        "bias_vl": torch.zeros(256, device=device),
    }
    with pytest.raises(RuntimeError, match="does not support hash/vision inputs"):
        torch.ops._C_ascend.moe_gating_top_k_hash(
            logits,
            k=8,
            norm_type=norm_type,
            renorm=renorm,
            expert_replica_routing_table=table,
            **{extra_input: inputs[extra_input]},
        )


@pytest.mark.parametrize("norm_type,renorm", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("groups,selected_groups", [(1, 1), (8, 4)])
def test_ordinary_map_graph_observes_table_updates(norm_type, renorm, groups, selected_groups):
    torch.manual_seed(179)
    # Cross core boundaries and exercise a final partial core.
    rows, experts = 257, 256
    logits = torch.randn(rows, experts, device="npu")
    # Sigmoid renorm=0 with grouped routing and bias must bypass the A3
    # EKFullLoad kernel, which does not apply the routing table.
    bias = torch.randn(experts, device="npu") * 0.2
    table = torch.arange(3 * experts, device="npu", dtype=torch.int32).reshape(3, experts)
    kwargs = dict(
        k=8,
        k_group=selected_groups,
        group_count=groups,
        group_select_mode=1,
        norm_type=norm_type,
        renorm=renorm,
        out_flag=False,
        bias=bias,
    )
    expected_weights, logical_ids, _ = torch_npu.npu_moe_gating_top_k(logits, **kwargs)

    def run():
        return torch.ops._C_ascend.moe_gating_top_k_hash(logits, expert_replica_routing_table=table, **kwargs)

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        weights, ids, _ = run()
    for _ in range(2):
        table.add_(experts)
        graph.replay()
        expected_ids = table[torch.arange(rows, device="npu")[:, None] % 3, logical_ids.long()]
        torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(weights, expected_weights, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("norm_type,renorm", [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "experts,groups,selected_groups,top_k",
    [(32, 1, 1, 1), (256, 1, 1, 8), (256, 8, 4, 8), (240, 8, 4, 8), (256, 8, 8, 8), (512, 8, 4, 80), (480, 8, 4, 80)],
)
@pytest.mark.parametrize("with_bias", [False, True])
def test_ordinary_gating_map_matches_cann(norm_type, renorm, dtype, experts, groups, selected_groups, top_k, with_bias):
    # Cover grouped padding, all-groups selection, and both A5 gather sizes.
    # 137 rows cross core boundaries and the table period; physical IDs must
    # never be used for gathering the logical expert scores.
    torch.manual_seed(173)
    rows, period = 137, 7
    logits = torch.randn(rows, experts, dtype=dtype, device="npu")
    bias = torch.randn(experts, dtype=dtype, device="npu") * 0.2 if with_bias else None
    table = (torch.arange(period * experts, device="npu").reshape(period, experts) + experts).int()
    kwargs = dict(
        k=top_k,
        k_group=selected_groups,
        group_count=groups,
        group_select_mode=1,
        renorm=renorm,
        norm_type=norm_type,
        out_flag=True,
        routed_scaling_factor=1.7,
        eps=1e-20,
        bias=bias,
    )
    expected_weights, logical_ids, expected_scores = torch_npu.npu_moe_gating_top_k(logits, **kwargs)
    unmapped_weights, unmapped_ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(logits, **kwargs)
    actual_weights, physical_ids, actual_scores = torch.ops._C_ascend.moe_gating_top_k_hash(
        logits, expert_replica_routing_table=table, **kwargs
    )
    separate_ids = map_to_physical_for_ascend(logical_ids, table)
    expected_ids = table[torch.arange(rows, device="npu")[:, None] % period, logical_ids.long()]
    torch.testing.assert_close(unmapped_ids, logical_ids, rtol=0, atol=0)
    torch.testing.assert_close(separate_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(physical_ids, separate_ids, rtol=0, atol=0)
    tolerance = {torch.float32: 1e-6, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    # A5 mapped/unmapped softmax uses different reduction implementations.
    torch.testing.assert_close(actual_weights, unmapped_weights, rtol=tolerance, atol=1e-7)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=tolerance, atol=1e-7)
    torch.testing.assert_close(actual_scores, expected_scores, rtol=1e-5, atol=1e-7)


def _reference(
    logits: torch.Tensor,
    text_bias: torch.Tensor,
    bias_vl: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor | None,
    top_k: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.where(input_ids == -1, 0, input_ids)
    scores = torch.nn.functional.softplus(logits.float()).sqrt()
    image_mask = (input_ids >= IMAGE_SENTINEL_LO) & (input_ids < IMAGE_SENTINEL_LO + IMAGE_SENTINEL_COUNT)
    row_bias = torch.where(
        image_mask[:, None],
        bias_vl.float()[None, :],
        text_bias.float()[None, :],
    )
    dynamic_ids = torch.topk(scores + row_bias, top_k, dim=-1, sorted=True).indices
    if tid2eid is None:
        expert_ids = dynamic_ids
    else:
        text_ids = tid2eid[input_ids.clamp_max(tid2eid.shape[0] - 1)].long()
        expert_ids = torch.where(image_mask[:, None], dynamic_ids, text_ids)
    weights = scores.gather(1, expert_ids)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return (weights * routed_scaling_factor).to(logits.dtype), expert_ids.int()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("with_hash", [False, True])
@pytest.mark.parametrize("execution_mode", ["eager", "graph"])
def test_v41_vision_bias_and_image_sentinel(
    dtype: torch.dtype,
    id_dtype: torch.dtype,
    with_hash: bool,
    execution_mode: str,
):
    torch.manual_seed(20260909)
    rows, experts, top_k = 8, 384, 6
    logits = torch.randn(rows, experts, dtype=dtype)
    text_bias = torch.randn(experts, dtype=dtype) * 0.2
    bias_vl = torch.randn(experts, dtype=dtype) * 0.2
    input_ids = torch.tensor(
        [-1, IMAGE_SENTINEL_LO, 22, IMAGE_SENTINEL_LO + 2, 33, IMAGE_SENTINEL_LO + 4, 44, -1],
        dtype=id_dtype,
    )
    tid2eid = None
    if with_hash:
        tid2eid = torch.empty(64, top_k, dtype=torch.int32)
        for token_id in range(tid2eid.shape[0]):
            tid2eid[token_id] = torch.randperm(experts)[:top_k]

    expected_weights, expected_ids = _reference(
        logits,
        text_bias,
        bias_vl,
        input_ids,
        tid2eid,
        top_k,
        routed_scaling_factor=1.5,
    )
    npu_logits = logits.npu()
    npu_text_bias = text_bias.npu()
    npu_bias_vl = bias_vl.npu()
    npu_input_ids = input_ids.npu()
    npu_tid2eid = tid2eid.npu() if tid2eid is not None else None

    def run_op():
        return torch.ops._C_ascend.moe_gating_top_k_hash(
            x=npu_logits,
            k=top_k,
            bias=npu_text_bias,
            input_ids=npu_input_ids,
            tid2eid=npu_tid2eid,
            k_group=1,
            group_count=1,
            routed_scaling_factor=1.5,
            eps=1e-20,
            group_select_mode=1,
            renorm=0,
            norm_type=2,
            out_flag=False,
            bias_vl=npu_bias_vl,
            image_sentinel_lo=IMAGE_SENTINEL_LO,
            image_sentinel_count=IMAGE_SENTINEL_COUNT,
        )

    if execution_mode == "graph":
        run_op()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual_weights, actual_ids, _ = run_op()
        graph.replay()
        torch.npu.synchronize()
    else:
        actual_weights, actual_ids, _ = run_op()

    torch.testing.assert_close(actual_ids.cpu(), expected_ids, rtol=0, atol=0)
    tolerance = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(
        actual_weights.cpu(),
        expected_weights,
        rtol=tolerance,
        atol=tolerance,
    )


def _map_inputs(rows, experts, dtype, with_hash=False):
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(rows, experts, generator=generator).to(dtype).npu()
    bias = (torch.randn(experts, generator=generator) * 0.1).to(dtype).npu()
    # Mixed replica counts and shuffled placement avoid an identity-map false positive.
    counts = torch.arange(experts).remainder(3).add(1).int()
    placement = torch.full((experts, 3), -1, dtype=torch.int64)
    slots = torch.randperm(int(counts.sum()), generator=generator)
    offset = 0
    for expert, count in enumerate(counts.tolist()):
        placement[expert, :count] = slots[offset : offset + count]
        offset += count
    table = build_expert_replica_routing_table(placement.npu(), counts.npu(), ep_rank=1)
    kwargs = dict(x=logits, k=6, bias=bias, group_count=1, k_group=1, norm_type=2, renorm=0, routed_scaling_factor=1.5)
    if with_hash:
        kwargs.update(
            input_ids=torch.arange(rows, device="npu").remainder(11),
            tid2eid=torch.stack([torch.randperm(experts, generator=generator)[:6] for _ in range(11)]).int().npu(),
        )
    return kwargs, table, offset


@pytest.mark.parametrize("rows,experts,with_hash", [(1, 384, False), (33, 513, True), (1025, 2048, False)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("execution_mode", ["eager", "graph"])
def test_gating_map_matches_separate_ops(rows, experts, with_hash, dtype, execution_mode):
    kwargs, table, physical_experts = _map_inputs(rows, experts, dtype, with_hash)
    op = torch.ops._C_ascend.moe_gating_top_k_hash
    reference_weights, logical_ids, _ = op(**kwargs)
    logical_cpu = logical_ids.cpu().long()
    graph = None
    if execution_mode == "graph":
        for _ in range(3):
            op(**kwargs, expert_replica_routing_table=table)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            weights, ids, _ = op(**kwargs, expert_replica_routing_table=table)
    previous = None
    for update in (False, True):
        if update:
            # Preserve addresses to catch stale tables during graph replay too.
            table.copy_((table + 1).remainder(physical_experts))
        if graph is None:
            weights, ids, _ = op(**kwargs, expert_replica_routing_table=table)
        else:
            graph.replay()
        expected = table.cpu()[torch.arange(rows)[:, None] % table.shape[0], logical_cpu]
        separate = map_to_physical_for_ascend(logical_ids, table)
        torch.testing.assert_close(separate.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(ids.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(weights, reference_weights, rtol=0, atol=0)
        assert (expected.long() != logical_cpu).any()
        if previous is not None:
            assert (expected != previous).all()
        previous = expected
