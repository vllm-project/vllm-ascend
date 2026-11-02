"""-only routing: retain FP32 softmax then fuse normalization and output casts."""
import torch
import torch_npu

def route(logits, topk=8, normalize=True):
    assert logits.dtype == torch.bfloat16 and logits.ndim == 2
    assert logits.shape[1] == 128 and 1 <= logits.shape[0] <= 32 and topk == 8
    weights, indices, _ = torch_npu.npu_moe_gating_top_k_softmax(
        x=logits.float(), finished=None, k=topk)
    return torch.ops.npu.splitquant_router_finish(
        weights.contiguous(), indices.contiguous(), normalize)
