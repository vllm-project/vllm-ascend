"""TurboQuant Hadamard-512 latent compress for REAL 4-bit MLA-latent storage.

Scheme: signed Hadamard-512, single L2 norm per slot, 16-level Lloyd-Max codebook
scaled to N(0,1/512). The legacy slot is 320 B; DSV4 uses a compact 258 B slot containing
256 nibble bytes (512x4bit) and a 2 B code-norm-corrected FP16 scale.

compress_kernel defaults to the legacy layout; compact output must be selected explicitly.
The legacy read side is turbo_quant_sparse_flash_attention; DeepSeek V4 compact slots are
dequantized by turbo_quant_sparse_attn_sharedkv.
"""
import math
import os as _os
import time as _time
import numpy as np
import torch

try:
    import torch_npu
except ImportError:
    torch_npu = None

# host-side timing accumulators (env TQ_PROF); discriminates dispatch- vs device-bound
_PROF = {"compress": 0.0, "deq_total": 0.0, "deq_kernel": 0.0, "deq_had": 0.0,
         "deq_layout": 0.0, "nc": 0, "nd": 0}


def _prof_report():
    if _PROF["nd"] and _PROF["nd"] % 156 == 0:   # ~2 decode steps x 78 layers
        n = _PROF["nd"]
        print(f"[TQ_PROF] over {n} dequant / {_PROF['nc']} compress calls (host ms total): "
              f"compress={_PROF['compress']*1e3:.0f} deq_total={_PROF['deq_total']*1e3:.0f} "
              f"[kernel={_PROF['deq_kernel']*1e3:.0f} had={_PROF['deq_had']*1e3:.0f} "
              f"layout={_PROF['deq_layout']*1e3:.0f}]  per-call us: "
              f"compress={_PROF['compress']/max(_PROF['nc'],1)*1e6:.0f} deq={_PROF['deq_total']/n*1e6:.0f}",
              flush=True)

HEAD_DIM = 512
ROPE_HEAD_DIM = 64
SCALE_BYTES = 2
SLOT_ALIGN = 64
ROPE_DTYPE_BYTES = 2
COMPRESS_OUTPUT_LEGACY = 0
COMPRESS_OUTPUT_COMPACT_CORRECTED = 1


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


def compact_slot_size(head_dim=HEAD_DIM):
    return packed_bytes(head_dim) + SCALE_BYTES


def fused_slot_size(head_dim=HEAD_DIM, rope_head_dim=ROPE_HEAD_DIM):
    return packed_bytes(head_dim) + int(rope_head_dim) * ROPE_DTYPE_BYTES + SCALE_BYTES


SLOT_PAD = base_slot_size(HEAD_DIM)
PACKED = packed_bytes(HEAD_DIM)          # nibble bytes

_CENT = _PIT = _PI = _LUTSQ = None
_BUILT = None


def _npu_synchronize():
    if torch_npu is not None and hasattr(torch, "npu"):
        torch.npu.synchronize()


def _require_npu():
    if torch_npu is None:
        raise RuntimeError("torch_npu is required for this NPU-only path")


def _build(device, head_dim=HEAD_DIM):
    global _CENT, _PIT, _PI, _LUTSQ, _BUILT
    head_dim = int(head_dim)
    _check_head_dim(head_dim)
    key = (str(device), head_dim)
    if _CENT is not None and _BUILT == key:
        return
    rng = np.random.default_rng(0)
    x = rng.standard_normal(400000) * (1.0 / math.sqrt(head_dim))
    c = np.linspace(-2 / math.sqrt(head_dim), 2 / math.sqrt(head_dim), 16)
    for _ in range(60):
        a = np.argmin(np.abs(x[:, None] - c[None, :]), axis=1)
        n = np.array([x[a == i].mean() if np.any(a == i) else c[i] for i in range(16)])
        if np.allclose(n, c):
            break
        c = n
    _CENT = torch.tensor(np.sort(c).astype(np.float32), device=device)
    signs = torch.tensor(rng.choice([-1.0, 1.0], head_dim).astype(np.float32), device=device)
    H = torch.tensor(np.array([[(-1) ** (bin(i & j).count("1")) for j in range(head_dim)]
                               for i in range(head_dim)], dtype=np.float32) / math.sqrt(head_dim), device=device)
    _PIT = (signs.unsqueeze(1) * H).contiguous()   # forward transform
    _PI = _PIT.t().contiguous()                    # inverse (orthonormal)
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
    _prof = _os.environ.get("TQ_PROF")
    if _prof:
        _npu_synchronize(); _t0 = _time.perf_counter()
    _build(latent.device, head_dim)
    N = latent.shape[0]
    flat = latent.to(torch.float32)
    norms = flat.norm(dim=1, keepdim=True)                       # [N,1]
    y = (flat / (norms + 1e-8)) @ _PIT                           # [N,head_dim] Hadamard
    nib = torch.argmin((y.unsqueeze(1) - _CENT.view(1, 16, 1)).abs(), dim=1).to(torch.int32)  # [N,head_dim]
    nib4 = nib.view(N, head_dim // 4, 4)
    int16 = (nib4[:, :, 0] | (nib4[:, :, 1] << 4) | (nib4[:, :, 2] << 8) | (nib4[:, :, 3] << 12))  # [N,head_dim/4]
    lo = (int16 & 0xff).to(torch.uint8)
    hi = ((int16 >> 8) & 0xff).to(torch.uint8)
    slot = torch.zeros(N, slot_pad, dtype=torch.uint8, device=latent.device)
    slot[:, 0:packed:2] = lo
    slot[:, 1:packed:2] = hi
    norms_fp16 = norms.to(torch.float16).view(N)
    slot[:, packed:packed + SCALE_BYTES] = norms_fp16.view(torch.uint8).view(N, SCALE_BYTES)
    if _prof:
        _npu_synchronize(); _PROF["compress"] += _time.perf_counter() - _t0; _PROF["nc"] += 1
    return slot                                                  # [N,base_slot_size(head_dim)]


@torch.no_grad()
def compress_kernel(latent, head_dim=None, output_mode=COMPRESS_OUTPUT_LEGACY):
    """Fused compress via turbo_quant_compress_latent.

    Returns a legacy aligned slot by default, or a compact corrected slot when requested explicitly.
    Hadamard remains a torch matmul; norm, quantization, packing, and optional correction are fused in C++.
    """
    _require_npu()
    head_dim = int(latent.shape[-1] if head_dim is None else head_dim)
    _check_head_dim(head_dim)
    _build(latent.device, head_dim)
    dev = latent.device
    z = (latent.float() @ _PIT.to(dev)).contiguous()        # Hadamard (un-normalized), fp32
    cent = _CENT.to(dev).contiguous()
    if output_mode not in {COMPRESS_OUTPUT_LEGACY, COMPRESS_OUTPUT_COMPACT_CORRECTED}:
        raise ValueError(f"Unsupported TurboQuant compression output mode: {output_mode}")
    slot = torch.ops._C_ascend.turbo_quant_compress_latent(
        z, cent, output_mode
    )  # Legacy [N,320] raw scale or compact [N,258] corrected scale.
    return slot, (z,)   # hold z alive until caller scatters (matches prior tuple shape)


@torch.no_grad()
def had_fwd(x, head_dim=None):
    """Forward Hadamard on the last dim: query -> Hadamard space (approach B)."""
    head_dim = int(x.shape[-1] if head_dim is None else head_dim)
    _check_head_dim(head_dim)
    _build(x.device, head_dim)
    return (x.float().reshape(-1, head_dim) @ _PIT.to(x.device)).to(x.dtype).reshape(x.shape)


@torch.no_grad()
def had_inv(x, head_dim=None):
    """Inverse Hadamard on the last dim: attention output -> orig basis (approach B)."""
    head_dim = int(x.shape[-1] if head_dim is None else head_dim)
    _check_head_dim(head_dim)
    _build(x.device, head_dim)
    return (x.float().reshape(-1, head_dim) @ _PI.to(x.device)).to(x.dtype).reshape(x.shape)


def lutsq(device, head_dim=HEAD_DIM):
    """[256] fp32 LUT: byte -> _CENT[lo]^2 + _CENT[hi]^2 (its 2 nibbles). Folds 1/sqrt(sum c^2) with no
    per-nibble bitwise unpack -> graph-capture safe (gather+sum only, no RightShift/And aclop)."""
    _build(device, head_dim)
    return _LUTSQ
