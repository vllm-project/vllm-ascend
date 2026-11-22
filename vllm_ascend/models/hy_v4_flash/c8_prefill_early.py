import torch

from vllm_ascend.hisparse.c8 import dequantize_c8_rows


def attend_early(query, packed, selected, seq_len, sink, scale):
    key = dequantize_c8_rows(packed.reshape(-1, packed.size(-1))[:seq_len].view(torch.uint8))
    positions = selected.masked_fill(selected < 0, seq_len)
    visible = torch.zeros((query.size(0), seq_len + 1), dtype=torch.bool, device=query.device)
    visible.scatter_(1, positions, True)
    logits = torch.matmul(query.float().transpose(0, 1), key.transpose(0, 1)) * scale
    logits.masked_fill_(~visible[:, :-1], -torch.inf)
    sink_logits = sink[:, None, None].expand(-1, query.size(0), 1)
    probs = torch.cat((logits, sink_logits), dim=-1).softmax(-1)[..., :-1]
    context = torch.matmul(probs, key[:, :512])
    return context.transpose(0, 1).contiguous().to(query.dtype)
