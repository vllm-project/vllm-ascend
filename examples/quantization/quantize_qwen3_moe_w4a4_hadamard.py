#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
"""Offline W4A4 + block-diagonal-Hadamard quantizer for Qwen3.x-MoE.

Produces the checkpoint consumed by the fused W4A4 MoE "mega" kernel
(``--quantization ascend``). Calibration-free RTN: each routed expert's
``gate_proj`` / ``up_proj`` / ``down_proj`` is quantized per-output-channel
to symmetric INT4 (range [-8, 7], ``scale = max(|W|) / 7``), and the matching
``N x N`` block-diagonal Hadamard rotation is baked into ``gate_proj`` /
``up_proj`` along the input (hidden) dim. The kernel applies the same online
Hadamard to the activations (Stage 0), so the two cancel: ``(x H) (W H)^T =
x W^T``. ``down_proj`` is quantized but not rotated. The shared expert, router
gate, attention, embeddings, LM head and norms are kept in their original
dtype.

Output layout matches the msModelSlim ``W4A4_DYNAMIC`` convention so the
existing ascend loader picks it up: per-expert ``{gate,up,down}_proj`` split
into ``.weight`` (int8, one nibble per byte), ``.weight_scale`` (fp32
``[out, 1]``) and ``.weight_offset`` (fp32 zeros, symmetric), plus a
``quant_model_description.json`` tagging each tensor ``W4A4_DYNAMIC`` / ``FLOAT``.

Usage:
    python examples/quantization/quantize_qwen3_moe_w4a4_hadamard.py \
        --src  /path/to/Qwen3.6-35B-A3B \
        --dst  /path/to/Qwen3.6-35B-A3B-w4a4-hadamard64 \
        --hadamard-block-size 64
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Source-model stacked MoE tensor names (Qwen3.x-MoE): experts are stored
# fused+stacked as gate_up_proj [E, 2I, H] and down_proj [E, H, I].
_GATE_UP = "mlp.experts.gate_up_proj"
_DOWN = "mlp.experts.down_proj"
# Standard HF/vLLM MoE layout instead stores each routed expert as its own 2D
# matrix: ``...mlp.experts.{e}.{gate_proj,up_proj,down_proj}[.weight]``. We
# quantize that layout too (else those checkpoints would silently pass through
# as bf16 / FLOAT and never hit the W4A4 mega kernel).
_PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")
SHARD_BYTES = 4 * 1024**3  # ~4 GiB output shards


def hadamard_blockdiag(n: int, device: torch.device) -> torch.Tensor:
    """Normalized n x n Walsh-Hadamard matrix H (H @ H = I). n a power of two."""
    rows = [[(-1) ** bin(i & j).count("1") for j in range(n)] for i in range(n)]
    return torch.tensor(rows, dtype=torch.float32, device=device) / math.sqrt(n)


def rotate_last_dim(w: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Block-diagonal Hadamard along the last (input) dim: reshape into n-blocks,
    matmul by H. w: [out, in], h: [n, n], in % n == 0."""
    out, in_ = w.shape
    n = h.shape[0]
    return (w.reshape(out, in_ // n, n) @ h).reshape(out, in_)


def quant_int4_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric INT4 RTN. w: [out, in] float.
    Returns (q int8 in [-8, 7], scale fp32 [out, 1]). Dead channels -> q = 0."""
    amax = w.abs().amax(dim=1, keepdim=True)
    scale = (amax / 7.0).clamp(min=torch.finfo(torch.float32).tiny)
    q = (w / scale).round().clamp_(-8, 7).to(torch.int8)
    return q, scale.to(torch.float32)


def _expert_stack_kind(key: str) -> tuple[str | None, str]:
    """Match a stacked-expert tensor key, tolerating an optional ``.weight`` tail.
    Qwen3.x-MoE stores raw stacked tensors (``...mlp.experts.gate_up_proj``); some
    HF layouts append ``.weight``. Returns ``(kind, prefix)`` where kind is
    ``"gate_up"`` / ``"down"`` / ``None`` and prefix is everything up to ``...mlp.``."""
    base = key[: -len(".weight")] if key.endswith(".weight") else key
    if base.endswith(_GATE_UP):
        return "gate_up", base[: -len("experts.gate_up_proj")]
    if base.endswith(_DOWN):
        return "down", base[: -len("experts.down_proj")]
    return None, ""


def is_stacked_experts(key: str) -> bool:
    return _expert_stack_kind(key)[0] is not None


def quantize_expert_stack(key: str, t: torch.Tensor, h: torch.Tensor | None) -> dict[str, torch.Tensor]:
    """Un-fuse + quantize one stacked expert tensor into per-expert int4 tensors.

    gate_up_proj [E, 2I, H] -> per-expert gate_proj/up_proj [I, H] (rotated along H).
    down_proj    [E, H, I]  -> per-expert down_proj [H, I] (not rotated).
    Returns a dict of {new_key: tensor} ready to serialize.
    """
    # Resolve kind + the "...mlp." prefix (tolerant of an optional .weight tail) so
    # per-expert keys are "<...>mlp.experts.{e}.{name}".
    kind, prefix = _expert_stack_kind(key)
    E = t.shape[0]
    out: dict[str, torch.Tensor] = {}
    if kind == "gate_up":
        two_i = t.shape[1]
        i_dim = two_i // 2
        for e in range(E):
            we = t[e].to(torch.float32)
            for name, w in (("gate_proj", we[:i_dim]), ("up_proj", we[i_dim:])):
                if h is not None:
                    w = rotate_last_dim(w, h)
                q, s = quant_int4_per_channel(w)
                base = f"{prefix}experts.{e}.{name}"
                out[f"{base}.weight"] = q
                out[f"{base}.weight_scale"] = s
                out[f"{base}.weight_offset"] = torch.zeros_like(s)
    else:  # down_proj, not rotated
        for e in range(E):
            q, s = quant_int4_per_channel(t[e].to(torch.float32))
            base = f"{prefix}experts.{e}.down_proj"
            out[f"{base}.weight"] = q
            out[f"{base}.weight_scale"] = s
            out[f"{base}.weight_offset"] = torch.zeros_like(s)
    return out


def per_expert_kind(key: str) -> tuple[str | None, str]:
    """Match an un-stacked per-expert tensor (``...experts.{e}.{name}[.weight]``).
    Returns ``(name, base)`` where name is ``gate_proj``/``up_proj``/``down_proj``
    (or ``None``) and base is the key without an optional ``.weight`` tail. Requires
    the segment before the proj name to be ``...experts.<int>`` so router/attn
    tensors (e.g. ``mlp.gate.weight``) don't match."""
    base = key[: -len(".weight")] if key.endswith(".weight") else key
    for name in _PROJ_NAMES:
        suffix = "." + name
        if base.endswith(suffix):
            head = base[: -len(suffix)]  # ...mlp.experts.{e}
            parent, sep, idx = head.rpartition(".experts.")
            if sep and idx.isdigit():
                return name, base
    return None, ""


def quantize_per_expert_weight(key: str, t: torch.Tensor, h: torch.Tensor | None) -> dict[str, torch.Tensor]:
    """Quantize one already-un-stacked per-expert tensor. gate_proj/up_proj are
    ``[I, H]`` (rotated along the input dim H); down_proj is ``[H, I]`` (not
    rotated). Output keys mirror the stacked path: ``<base>.weight`` (int8),
    ``.weight_scale`` (fp32 ``[out, 1]``), ``.weight_offset`` (fp32 zeros)."""
    name, base = per_expert_kind(key)
    w = t.to(torch.float32)
    if name in ("gate_proj", "up_proj") and h is not None:
        w = rotate_last_dim(w, h)
    q, s = quant_int4_per_channel(w)
    return {f"{base}.weight": q, f"{base}.weight_scale": s, f"{base}.weight_offset": torch.zeros_like(s)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="Source bf16 HF model dir")
    p.add_argument("--dst", required=True, help="Output W4A4 checkpoint dir")
    p.add_argument(
        "--hadamard-block-size",
        type=int,
        default=64,
        choices=[64],
        help="Block-diag Hadamard size baked into gate_up. Only 64 is supported: the "
        "W4A4_DYNAMIC runtime kernel always applies a fixed 64-wide in-kernel Hadamard, "
        "so any other size (or none) would produce a silently-wrong checkpoint.",
    )
    p.add_argument("--device", default="cpu", help="Device for the quant math")
    args = p.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    n = args.hadamard_block_size
    h = hadamard_blockdiag(n, device) if n > 0 else None
    print(f"[quant] src={src} dst={dst} hadamard_block={n}")

    # Copy non-weight files (config, tokenizer, ...) verbatim.
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and not f.name.endswith(".safetensors.index.json"):
            shutil.copy(f, dst / f.name)

    index_path = src / "model.safetensors.index.json"
    weight_map = json.load(index_path.open())["weight_map"]
    shards = sorted(set(weight_map.values()))

    description: dict[str, str] = {}
    out_index: dict[str, str] = {}
    buf: dict[str, torch.Tensor] = {}
    buf_bytes = 0
    shard_no = 0

    def flush(final: bool = False) -> None:
        nonlocal buf, buf_bytes, shard_no
        if not buf or (not final and buf_bytes < SHARD_BYTES):
            return
        shard_no += 1
        name = f"quant_model_weights-{shard_no:05d}.safetensors"
        save_file(buf, str(dst / name), metadata={"format": "pt"})
        for k in buf:
            out_index[k] = name
        print(f"[quant]   wrote {name} ({buf_bytes / 1e9:.2f} GB, {len(buf)} tensors)")
        buf, buf_bytes = {}, 0

    def add(key: str, t: torch.Tensor, quant: bool) -> None:
        nonlocal buf_bytes
        # safetensors.save_file requires CPU tensors; move off NPU/GPU if --device
        # put the quantized weights on an accelerator.
        t = t.cpu().contiguous()
        buf[key] = t
        buf_bytes += t.numel() * t.element_size()
        description[key] = "W4A4_DYNAMIC" if quant else "FLOAT"
        flush()

    n_expert_src = 0  # source expert tensors quantized (stacked or per-expert)
    for shard in shards:
        with safe_open(str(src / shard), framework="pt", device=args.device) as f:
            handle: Any = f  # safetensors cm; mypy mistypes it as Path
            for key in handle.keys():  # noqa: SIM118 (safetensors handle, not a dict)
                t = handle.get_tensor(key)
                if is_stacked_experts(key):  # [E, ...] fused-stacked layout
                    for nk, nt in quantize_expert_stack(key, t, h).items():
                        add(nk, nt, quant=True)
                    n_expert_src += 1
                elif per_expert_kind(key)[0] is not None:  # per-expert 2D layout
                    for nk, nt in quantize_per_expert_weight(key, t, h).items():
                        add(nk, nt, quant=True)
                    n_expert_src += 1
                else:
                    add(key, t, quant=False)  # kept in original dtype
    flush(final=True)

    # Fail fast rather than silently emit an all-FLOAT (bf16) checkpoint: if no
    # routed-expert tensors matched either layout, the W4A4 mega kernel would have
    # nothing to run, so the conversion did not do what the user asked.
    if n_expert_src == 0:
        raise RuntimeError(
            f"No routed-expert weights found under {src} (looked for stacked "
            f"'{_GATE_UP}'/'{_DOWN}' and per-expert '...experts.N.{{gate,up,down}}_proj'). "
            "This source does not look like a Qwen3.x-MoE checkpoint; refusing to write "
            "an all-FLOAT (non-W4A4) output."
        )

    # Rewrite shard names with the final -of-NNNNN suffix.
    total = shard_no
    rename = {}
    for old in {v for v in out_index.values()}:
        idx = old.split("-")[-1].split(".")[0]
        new = f"quant_model_weights-{idx}-of-{total:05d}.safetensors"
        os.rename(dst / old, dst / new)
        rename[old] = new
    out_index = {k: rename[v] for k, v in out_index.items()}

    total_size = sum((dst / s).stat().st_size for s in set(out_index.values()))
    json.dump(
        {"metadata": {"total_size": total_size}, "weight_map": out_index},
        (dst / "quant_model_weights.safetensors.index.json").open("w"),
        indent=2,
    )
    # Stamp the Hadamard block size so the runtime W4A4_DYNAMIC mega-MoE scheme can verify
    # this checkpoint was produced by THIS converter: the kernel applies a matching in-kernel
    # block-diagonal Hadamard, so a plain (unrotated) W4A4_DYNAMIC checkpoint without this
    # marker is rejected rather than silently given an unmatched activation rotation.
    description["hadamard_block_size"] = n
    json.dump(description, (dst / "quant_model_description.json").open("w"), indent=2)
    n_q = sum(v == "W4A4_DYNAMIC" for v in description.values())
    print(
        f"[quant] done: {total} shards, {total_size / 1e9:.2f} GB, "
        f"{n_q} W4A4_DYNAMIC / {len(description) - n_q} FLOAT tensors"
    )


if __name__ == "__main__":
    main()
