import torch
from torch import nn

from vllm.model_executor.layers.activation import SiluAndMul

def torch_moe(a, w1, w2, score, topk,
              renormalize, num_expert_group, topk_group,
              scoring_func, e_score_correction_bias):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    
    if scoring_func == "softmax":
        score = torch.softmax(score, dim=-1)
    elif scoring_func == "sigmoid":
        score = score.sigmoid()
    
    if e_score_correction_bias is not None:
        original_scores = score
        score = score + e_score_correction_bias.unsqueeze(0)

    # group_topk
    num_token = score.shape[0]
    group_score = score.view(num_token, num_expert_group, -1).max(dim=-1).values
    group_idx = torch.topk(group_score, k=topk_group, dim=-1,
                           sorted=False)[1]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_score)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        score.shape[-1] // num_expert_group).reshape(num_token, -1)  # [n, e]
    score = score.masked_fill(~score_mask.bool(), 0.0)  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(score, k=topk, dim=-1, sorted=False)[1]
        # Use original unbiased scores for the routing weights
        topk_weight = original_scores.gather(1, topk_ids)
    else:
        topk_weight, topk_ids = torch.topk(score,
                                           k=topk,
                                           dim=-1,
                                           sorted=False)
    
    if renormalize:
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul()(
                a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    return (out.view(B, -1, w2.shape[1]) *
            topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)