import torch
import torch.nn.functional as F


def hc_split_sinkhorn_ref(
        mixes: torch.Tensor,  # [b, s, mix_hc] => [b, s, (2 + hc) * hc]
        hc_scale: torch.Tensor,  # [3]
        hc_base: torch.Tensor,  # [(2 + hc) * hc]
        hc_mult: int = 4,  # hc
        sinkhorn_iters: int = 20,
        eps: float = 1e-6):
    mixes = mixes.unsqueeze(0)
    b, s, _ = mixes.size()
    # get pre
    mixes_pre = mixes[:, :, :hc_mult]
    hc_scale_pre = hc_scale[0]
    hc_base_pre = hc_base[:hc_mult]
    pre = F.sigmoid(hc_scale_pre * mixes_pre + hc_base_pre) + eps
    # get post
    mixes_post = mixes[:, :, hc_mult:2 * hc_mult]
    hc_scale_post = hc_scale[1]
    hc_base_post = hc_base[hc_mult:2 * hc_mult]
    post = 2 * F.sigmoid(hc_scale_post * mixes_post + hc_base_post)
    # get comb
    # step 1 : init comb
    mixes_comb = mixes[:, :, 2 * hc_mult:]
    hc_scale_comb = hc_scale[2]
    hc_base_comb = hc_base[2 * hc_mult:]
    comb = (hc_scale_comb * mixes_comb + hc_base_comb).reshape(
        b, s, hc_mult, hc_mult)  # [b, s, hc, hc]
    comb = F.softmax(comb, dim=-1) + eps
    # step 2: do sinkhorn ops
    for _ in range(sinkhorn_iters):
        comb = comb / (comb.sum(dim=-1).unsqueeze(-1) + eps)
        comb = comb / (comb.sum(dim=-2).unsqueeze(-2) + eps)

    return pre.squeeze(0), post.squeeze(0), comb.squeeze(0)
