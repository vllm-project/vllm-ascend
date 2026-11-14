import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

IMAGE_SENTINEL_LO = 129257
IMAGE_SENTINEL_COUNT = 5


def _reference(
    logits: torch.Tensor,
    text_bias: torch.Tensor,
    bias_vl: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor | None,
    top_k: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
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


@pytest.mark.parametrize("prepared_mask", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_hash", [False, True])
@pytest.mark.parametrize("execution_mode", ["eager", "graph"])
def test_v41_vision_bias_and_image_sentinel(
    dtype: torch.dtype,
    with_hash: bool,
    execution_mode: str,
    prepared_mask: bool,
):
    torch.manual_seed(20260909)
    rows, experts, top_k = 8, 384, 6
    logits = torch.randn(rows, experts, dtype=dtype)
    text_bias = torch.randn(experts, dtype=dtype) * 0.2
    bias_vl = torch.randn(experts, dtype=dtype) * 0.2
    input_ids = torch.tensor(
        [11, IMAGE_SENTINEL_LO, 22, IMAGE_SENTINEL_LO + 2, 33, IMAGE_SENTINEL_LO + 4, 44, 55],
        dtype=torch.int64,
    )
    tid2eid = None
    if with_hash:
        tid2eid = torch.empty(64, top_k, dtype=torch.int32)
        for token_id in range(tid2eid.shape[0]):
            tid2eid[token_id] = torch.randperm(experts)[:top_k]

    npu_logits = logits.npu()
    npu_text_bias = text_bias.npu()
    npu_bias_vl = bias_vl.npu()
    npu_input_ids = input_ids.npu()
    npu_tid2eid = tid2eid.npu() if tid2eid is not None else None
    npu_image_mask = torch.empty(rows, dtype=torch.bool, device="npu") if prepared_mask else None
    if npu_image_mask is not None:
        npu_image_mask.copy_((input_ids >= IMAGE_SENTINEL_LO) & (input_ids < IMAGE_SENTINEL_LO + IMAGE_SENTINEL_COUNT))

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
            image_mask=npu_image_mask,
        )

    if execution_mode == "graph":
        run_op()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            actual_weights, actual_ids, _ = run_op()
    for current_ids in (input_ids, input_ids.roll(1)):
        # Hash inputs can be normalized once outside all MoE layers. The
        # prepared mask must retain image routing after sentinel IDs are lost.
        normalized = (
            current_ids.clamp_max(tid2eid.shape[0] - 1) if prepared_mask and tid2eid is not None else current_ids
        )
        npu_input_ids.copy_(normalized)
        if npu_image_mask is not None:
            npu_image_mask.copy_(
                (current_ids >= IMAGE_SENTINEL_LO) & (current_ids < IMAGE_SENTINEL_LO + IMAGE_SENTINEL_COUNT)
            )
        expected_weights, expected_ids = _reference(
            logits, text_bias, bias_vl, current_ids, tid2eid, top_k, routed_scaling_factor=1.5
        )
        if execution_mode == "graph":
            graph.replay()
        else:
            actual_weights, actual_ids, _ = run_op()
        torch.npu.synchronize()
        torch.testing.assert_close(actual_ids.cpu(), expected_ids, rtol=0, atol=0)
        tolerance = 1e-5 if dtype == torch.float32 else 1e-2
        torch.testing.assert_close(actual_weights.cpu(), expected_weights, rtol=tolerance, atol=tolerance)
