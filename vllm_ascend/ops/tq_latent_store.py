"""Host-side TurboQuant (TQ4) compression of the MLA latent for 4-bit KV storage.

Scheme: signed Hadamard rotation, one L2 norm per slot, and the fixed 16-level
codebook used by the CANN TurboQuant sparse-flash-attention kernel. A base slot is
`packed_bytes` nibble bytes (head_dim x 4 bit) plus 2 bytes of fp16 vector norm,
padded up to SLOT_ALIGN; the fused slot the SFA op reads drops the padding and
carries the rope half instead (see `fused_slot_size`). For head_dim=512 that is
256 + 2 bytes padded to 320, and a 386-byte fused slot.

`compress` is the reference implementation; `compress_kernel` is the fused device
path used in production. Both take an already rmsnorm'd latent [N, head_dim]. The
read side (dequant-in-SFA) is the csrc turbo_quant_sparse_flash_attention op.
"""

import math

import numpy as np
import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

HEAD_DIM = 512
ROPE_HEAD_DIM = 64
SCALE_BYTES = 2
SLOT_ALIGN = 64
ROPE_DTYPE_BYTES = 2
TURBOQUANT_CENTROIDS_512 = np.array(
    [
        -0.12091285,
        -0.09111122,
        -0.07112455,
        -0.05513602,
        -0.04132067,
        -0.02874970,
        -0.01700489,
        -0.00568677,
        0.00547294,
        0.01680406,
        0.02857605,
        0.04108622,
        0.05492980,
        0.07101817,
        0.09115373,
        0.12037795,
    ],
    dtype=np.float32,
)


def _is_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0


def _check_head_dim(head_dim):
    if not _is_power_of_2(int(head_dim)):
        raise ValueError(f"kv_lora_rank/head_dim must be a power of 2 for Sylvester Hadamard, got {head_dim}")


def _align_up(n, align):
    return ((int(n) + align - 1) // align) * align


def packed_bytes(head_dim=HEAD_DIM):
    _check_head_dim(head_dim)
    return int(head_dim) // 2


def base_slot_size(head_dim=HEAD_DIM):
    return _align_up(packed_bytes(head_dim) + SCALE_BYTES, SLOT_ALIGN)


def fused_slot_size(head_dim=HEAD_DIM, rope_head_dim=ROPE_HEAD_DIM):
    return packed_bytes(head_dim) + int(rope_head_dim) * ROPE_DTYPE_BYTES + SCALE_BYTES


_CENT = _PIT = _PI = _LUTSQ = None
_BUILT = None


def _require_npu():
    if torch_npu is None:
        raise RuntimeError("torch_npu is required for this NPU-only path")


def _build(device, head_dim=HEAD_DIM):
    global _CENT, _PIT, _PI, _LUTSQ, _BUILT
    head_dim = int(head_dim)
    _check_head_dim(head_dim)
    key = (str(device), head_dim)
    if _CENT is not None and key == _BUILT:
        return
    centroids = TURBOQUANT_CENTROIDS_512
    if head_dim != HEAD_DIM:
        centroids = centroids * math.sqrt(HEAD_DIM / head_dim)
    _CENT = torch.tensor(centroids, device=device)
    rng = np.random.default_rng(0)
    signs = torch.tensor(rng.choice([-1.0, 1.0], head_dim).astype(np.float32), device=device)
    H = torch.tensor(
        np.array(
            [[(-1) ** (bin(i & j).count("1")) for j in range(head_dim)] for i in range(head_dim)], dtype=np.float32
        )
        / math.sqrt(head_dim),
        device=device,
    )
    _PIT = (signs.unsqueeze(1) * H).contiguous()  # forward transform
    _PI = _PIT.t().contiguous()  # inverse (orthonormal)
    _lc = _CENT.detach().float().cpu().numpy()
    _lt = np.zeros(256, dtype=np.float32)
    for _lb in range(256):
        _lt[_lb] = _lc[_lb & 0xF] ** 2 + _lc[(_lb >> 4) & 0xF] ** 2
    _LUTSQ = torch.tensor(_lt, device=device)  # [256] fp32 device-resident (byte -> sum of its 2 nibbles' c^2)
    _BUILT = key


@torch.no_grad()
def compress(latent, head_dim=None):
    """latent [N,head_dim] (rmsnorm'd) -> uint8 [N,base_slot_size(head_dim)] TQ slot."""
    head_dim = int(latent.shape[-1] if head_dim is None else head_dim)
    _check_head_dim(head_dim)
    packed = packed_bytes(head_dim)
    slot_pad = base_slot_size(head_dim)
    _build(latent.device, head_dim)
    assert _CENT is not None and _PIT is not None
    N = latent.shape[0]
    flat = latent.to(torch.float32)
    norms = flat.norm(dim=1, keepdim=True)  # [N,1]
    y = (flat / (norms + 1e-8)) @ _PIT  # [N,head_dim] Hadamard
    nib = torch.argmin((y.unsqueeze(1) - _CENT.view(1, 16, 1)).abs(), dim=1).to(torch.int32)  # [N,head_dim]
    nib4 = nib.view(N, head_dim // 4, 4)
    int16 = nib4[:, :, 0] | (nib4[:, :, 1] << 4) | (nib4[:, :, 2] << 8) | (nib4[:, :, 3] << 12)  # [N,head_dim/4]
    lo = (int16 & 0xFF).to(torch.uint8)
    hi = ((int16 >> 8) & 0xFF).to(torch.uint8)
    slot = torch.zeros(N, slot_pad, dtype=torch.uint8, device=latent.device)
    slot[:, 0:packed:2] = lo
    slot[:, 1:packed:2] = hi
    norms_fp16 = norms.to(torch.float16).view(N)
    slot[:, packed : packed + SCALE_BYTES] = norms_fp16.view(torch.uint8).view(N, SCALE_BYTES)
    return slot  # [N,base_slot_size(head_dim)]


@torch.no_grad()
def compress_kernel(latent, head_dim=None):
    """Fused compress via torch op turbo_quant_compress_latent. latent [N,head_dim] (rmsnorm'd, fp16/bf16) ->
    (slot uint8 [N,base_slot_size(head_dim)], z). Hadamard (1 matmul) in torch; norm/quantize/pack in the
    csrc kernel (aclnnTurboQuantCompressLatent). Replaces the ~18-op torch compress with: 1 matmul + 1 op call."""
    _require_npu()
    head_dim = int(latent.shape[-1] if head_dim is None else head_dim)
    _check_head_dim(head_dim)
    _build(latent.device, head_dim)
    assert _CENT is not None and _PIT is not None
    dev = latent.device
    z = (latent.float() @ _PIT.to(dev)).contiguous()  # Hadamard (un-normalized), fp32
    cent = _CENT.to(dev).contiguous()
    slot = torch.ops._C_ascend.turbo_quant_compress_latent(z, cent)  # [N,320] uint8, fused norm+quantize+pack
    return slot, (z,)  # hold z alive until caller scatters (matches prior tuple shape)


@torch.no_grad()
def had_fwd(x, head_dim=None):
    """Forward Hadamard on the last dim: query -> Hadamard space (approach B)."""
    head_dim = int(x.shape[-1] if head_dim is None else head_dim)
    _check_head_dim(head_dim)
    _build(x.device, head_dim)
    assert _PIT is not None
    return (x.float().reshape(-1, head_dim) @ _PIT.to(x.device)).to(x.dtype).reshape(x.shape)


@torch.no_grad()
def had_inv(x, head_dim=None):
    """Inverse Hadamard on the last dim: attention output -> orig basis (approach B)."""
    head_dim = int(x.shape[-1] if head_dim is None else head_dim)
    _check_head_dim(head_dim)
    _build(x.device, head_dim)
    assert _PI is not None
    return (x.float().reshape(-1, head_dim) @ _PI.to(x.device)).to(x.dtype).reshape(x.shape)


def lutsq(device, head_dim=HEAD_DIM):
    """[256] fp32 LUT: byte -> _CENT[lo]^2 + _CENT[hi]^2 (its 2 nibbles). Folds 1/sqrt(sum c^2) with no
    per-nibble bitwise unpack -> graph-capture safe (gather+sum only, no RightShift/And aclop)."""
    _build(device, head_dim)
    return _LUTSQ
