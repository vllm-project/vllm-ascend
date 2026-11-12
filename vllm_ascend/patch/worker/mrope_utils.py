import torch

from vllm_ascend.ops.rotary_embedding import AscendMRotaryEmbedding


def get_rotary_inv_freq(rotary_emb: AscendMRotaryEmbedding, device: torch.device) -> torch.Tensor:
    """Return the contiguous FP32 inverse-frequency vector on ``device``."""
    inv_freq = getattr(rotary_emb, "inv_freq", None)
    if inv_freq is None or inv_freq.device != device:
        inv_freq = rotary_emb._compute_inv_freq(rotary_emb.base).to(device=device, dtype=torch.float32)
        rotary_emb.register_buffer("inv_freq", inv_freq, persistent=False)
    return inv_freq.contiguous()
