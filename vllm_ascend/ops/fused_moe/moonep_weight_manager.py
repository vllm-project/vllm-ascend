# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
"""Publish local expert weights to Home, prefetch remote weights into Slots, and build zero-copy MLP views.

Conventional owner-only dispatch sends every row to the rank
that owns its expert. Under skewed routing that concentrates rows on the hot
expert's rank. The MoonEP planning kernel can instead keep some rows *local*
and mark their (remote) expert in ``plan.experts_to_copy``; the rank then
prefetches that expert's weights into one of its B slots (SHMEM RMA) and
computes the rows locally against the [epn+B] weight alias. Loads stay
balanced across ranks without vLLM's dynamic EPLB layer — the same trade-off
MindSpeed-MM_MoonEP's ``shmem_ep_dispatcher`` makes ("perfect load balance").

Per layer, per *payload* (bf16: w13 [epn,H,2F] and w2 [epn,F,H]; W4A8MXFP adds
the two e8m0 scales, four payloads in total — a remote expert is only usable if
its scales travel with it):
  home [epn, R, C]  VMM chunk, bf16 *byte carrier* mirroring this rank's local
                    weight rows (published once at init; inference weights are
                    static). ``R*C*2`` == the payload's per-expert byte count.
  slot [B,  R, C]   VMM chunk shared across layers of one shape (B = epn);
                    filled per step by prefetch_weight with the remote rows of
                    plan.experts_to_copy
  full [E+B, R, C]  projection view (home at [rank*epn, ..)) used by the
                    prefetch kernel (via attached source VAs)
  alias [epn+B,..]  zero-copy packed view (home rows 0..epn, slot rows epn..)
                    used as the grouped-matmul weights; shape/stride/dtype/
                    npu_format are read off the weight tensor at runtime

The prefetch chain only reads ``shape[1:]`` plus the attached source VAs and
copies ``H*Hp*2`` bytes per expert as flat bytes
(ascend-moonep/kernels/prefetch.cpp:97-141), so the bf16 carrier is transparent
for quantized payloads — no library, csrc or kernel change is involved.

Scope: bf16 (AscendUnquantizedFusedMoEMethod) plus quantized layouts whose
per-expert block is dense (``stride(0)*element_size == numel/E``) — verified
on-card for W4A8MXFP (uint8 NZ weights + e8m0 ND scales). Anything else raises.
Used whenever vLLM selects MoonEP/SHMEM. ``enable_moonep`` still controls
whether MoonEP is available. Quantized payloads follow the same layout
validation unconditionally; this does not quantize BF16 model weights.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from vllm.logger import logger

from vllm_ascend import envs
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEFusedExpertsInput, MoEWeights
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import build_mlp_compute_input
from vllm_ascend.ops.fused_moe.moonep_mlp import apply_slot_mlp, home_gemm_apply_mlp
from vllm_ascend.quantization.quant_type import QuantType

# prefetch kernel 的 UB 约束（ascend-moonep/kernels/prefetch.cpp:100-105）：
# ``elements_per_row`` 就是载体的 Hp，且要求 ub_bytes / sizeof(bfloat16) / Hp >= 1
# （UB 内部暂存区 < 188 KiB），所以 Hp 最多 32768 个 2 字节元素 = 64 KiB 字节。
_MAX_CARRIER_ROW_ELEMS = 32768

# Optional per-layer cache reuses remote weights across steps. Snapshot the
# configured capacity once so layers do not read the environment on each step.
_CACHED_EXPERT_COUNT = envs.MOONEP_CACHED_REDUNDANT_EXPERT_COUNT


# ---------------------------------------------------------------------------
# Payload layout: bf16 byte carriers + runtime-read weight metadata
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _PayloadSpec:
    """One weight payload's mirror/prefetch/alias geometry.

    Everything layout-ish is read off ``ref`` at runtime (shape, stride, dtype,
    element_size, npu_format) and never hardcoded: on-card measurement shows the
    NZ w13/w2 payloads come back as ``dtype=uint8`` with
    ``get_npu_format() == 50`` (the ``29`` passed to ``npu_format_cast`` is an
    input, not the result), and the two e8m0 scales are 4-D non-contiguous ND
    views (format 2).
    """

    name: str
    ref: torch.Tensor
    is_nz: bool  # NZ transposed view (w13/w2): alias needs cast-shape + transpose
    row_bytes: int  # per-expert bytes == stride(0) * element_size()
    nbytes: int  # per-expert bytes * local expert count
    carrier: tuple[int, int]  # bf16 (R, C) with R*C*2 == row_bytes
    npu_format: int


def _flat_bytes(t: torch.Tensor, nbytes: int) -> torch.Tensor:
    """View the first ``nbytes`` bytes of ``t``'s payload as flat uint8.

    Starts at ``data_ptr()``, so a weight that is a slice of a larger storage
    (``full[rank*epn:(rank+1)*epn]`` in the harnesses) mirrors its own rows and
    not the head of the table. Non-contiguous NZ / 4-D scale views cannot be
    reshaped and ``copy_`` between two NZ tensors is not a path we rely on, so
    mirrors move raw bytes; ``_make_spec`` has already established that every
    expert owns one dense ``row_bytes`` block, which makes the payload a dense
    range starting at ``data_ptr()``.
    """
    return _sym_tensor(t.data_ptr(), (int(nbytes),), torch.uint8, t.device, nbytes=int(nbytes))


def _sym_tensor(
    ptr, shape, dtype, device, *, stride=None, storage_offset=0, nbytes=None, npu_format=None
) -> torch.Tensor:
    """Build a non-owning tensor over an existing device address.

    Counterpart of the library's ``buffer_c.create_sym_tensor_from_ptr`` (pure
    Python, no C++ behind it) with one extra ``npu_format`` metadata key — the
    measured way to hand a VMM alias the NZ format without any transdata
    (shape/stride/storage stay untouched; verified on card). Phase 2 can fold
    this parameter into the library.
    """
    import torch_npu

    shape = tuple(int(d) for d in shape)
    if stride is None:
        stride = _contiguous_strides(shape)
    if nbytes is None:
        nbytes = _numel(shape) * torch.empty((), dtype=dtype).element_size()
    if not isinstance(device, torch.device):
        device = torch.device(device)
    metadata = {
        "data_ptr": int(ptr),
        "device": device,
        "nbytes": int(nbytes),
        "dtype": dtype,
        "size": list(shape),
        "stride": [int(s) for s in stride],
        "storage_offset": int(storage_offset),
    }
    if npu_format is not None:
        metadata["npu_format"] = int(npu_format)
    storage = torch_npu._C._construct_storage_from_data_pointer(
        metadata["data_ptr"], metadata["device"], metadata["nbytes"]
    )
    return torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, storage)


def _contiguous_strides(shape) -> tuple[int, ...]:
    if not shape:
        return ()
    strides = [1]
    for dim in reversed(shape[1:]):
        strides.insert(0, strides[0] * dim)
    return tuple(strides)


def _numel(shape) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _carrier_shape(row_bytes: int, natural: tuple[int, int] | None = None) -> tuple[int, int]:
    """Factor the per-expert byte count into a bf16 carrier (R, C), R*C*2 == row_bytes.

    Any factorization is equivalent for the prefetch chain (it copies flat bytes),
    but bf16 weights keep their own natural row split so the unquantized path
    stays byte-for-byte what it was before. Quantized payloads have no bf16 row
    split to reuse, so they take the widest C within the kernel's UB bound.
    """
    elems = row_bytes // 2
    if elems <= 0 or elems * 2 != row_bytes:
        raise ValueError(f"row_bytes={row_bytes} must be a positive even number.")
    if natural is not None and natural[0] * natural[1] == elems and natural[1] <= _MAX_CARRIER_ROW_ELEMS:
        return (int(natural[0]), int(natural[1]))
    cols = min(elems, _MAX_CARRIER_ROW_ELEMS)
    while cols > 1 and elems % cols:
        cols //= 2
    return (elems // cols, cols)


def _make_spec(name: str, ref: torch.Tensor, *, is_nz: bool, natural: tuple[int, int] | None = None) -> _PayloadSpec:
    import torch_npu

    experts = int(ref.shape[0])
    row_bytes = int(ref.stride(0)) * int(ref.element_size())
    if experts <= 0 or row_bytes <= 0:
        raise NotImplementedError(
            f"MoonEP slot mode: {name} has an empty/unstrided layout "
            f"(shape={tuple(ref.shape)} stride={tuple(ref.stride())})."
        )
    # The packed [epn+B] alias assumes one dense block per expert; NZ padding
    # would silently shift every row after the first. Both sides in bytes --
    # for 1-byte payloads the element count happens to match, for bf16 it does not.
    numel_per_expert = ref.numel() // experts * int(ref.element_size())
    if numel_per_expert != row_bytes:
        raise NotImplementedError(
            f"MoonEP slot mode: {name} has per-expert padding "
            f"(numel/experts={numel_per_expert} != stride(0)*element_size={row_bytes}); "
            "the packed [epn+B] alias needs one contiguous block per expert."
        )
    nbytes = experts * row_bytes
    # The weight may be a slice of a larger table (the harnesses build ranks from
    # `full[rank*epn:(rank+1)*epn]`), so it need not start at the storage head --
    # it only needs `nbytes` to fit from its own offset. `_flat_bytes` starts at
    # `data_ptr()`, which already accounts for the offset.
    offset_bytes = int(ref.storage_offset()) * int(ref.element_size())
    storage = ref.untyped_storage()
    if offset_bytes + nbytes > int(storage.nbytes()):
        raise NotImplementedError(
            f"MoonEP slot mode: {name} payload runs past its storage "
            f"(storage_offset*element_size={offset_bytes} + nbytes={nbytes} > "
            f"storage_nbytes={int(storage.nbytes())})."
        )
    return _PayloadSpec(
        name=name,
        ref=ref,
        is_nz=is_nz,
        row_bytes=row_bytes,
        nbytes=nbytes,
        carrier=_carrier_shape(row_bytes, natural),
        npu_format=int(torch_npu.get_npu_format(ref)),
    )


def _infer_F(layer, H: int) -> int:
    """Derive the per-partition ffn size from the layer's own weight layout.

    Both layouts keep ``2F`` on the last axis, so only the hidden axis differs:
    bf16 leaves w13 as ``[epn, H, 2F]``, while W4A8MXFP packs 2 values per byte
    along hidden and transposes to ``[epn, H/2, 2F]`` (measured on card, see
    docs/moonep_slot_quant.md).
    """
    w13 = layer.w13_weight
    if w13.dim() != 3 or int(w13.shape[2]) % 2:
        raise NotImplementedError(f"MoonEP slot mode: unexpected w13 shape {tuple(w13.shape)}.")
    F = int(w13.shape[2]) // 2
    if w13.dtype == torch.bfloat16:
        if int(w13.shape[1]) != H:
            raise NotImplementedError(f"MoonEP slot mode: bf16 w13 should be [epn, H={H}, 2F], got {tuple(w13.shape)}.")
        if tuple(layer.w2_weight.shape) != (int(w13.shape[0]), F, H):
            raise NotImplementedError(
                f"MoonEP slot mode: bf16 w2 should be [epn, F={F}, H={H}], got {tuple(layer.w2_weight.shape)}."
            )
        return F
    if int(w13.shape[1]) * 2 != H:
        raise NotImplementedError(
            f"MoonEP slot mode: quantized w13 should be [epn, H/2={H // 2}, 2F], "
            f"got {tuple(w13.shape)} (2 values per byte along hidden)."
        )
    scale13 = getattr(layer, "w13_weight_scale", None)
    if scale13 is None:
        raise NotImplementedError("MoonEP slot mode with quantized weights requires w13_weight_scale.")
    # DS leaves the scale 4-D after process_weights_after_loading:
    # [epn, H/(2*group), 2F, 2] — cross-check 2F on its own axis.
    if scale13.dim() == 4 and int(scale13.shape[2]) != 2 * F:
        raise NotImplementedError(
            f"MoonEP slot mode: w13_weight_scale {tuple(scale13.shape)} disagrees with w13's 2F={2 * F}."
        )
    return F


def _payload_specs(layer, H: int, F: int) -> list[_PayloadSpec]:
    """Every payload this layer needs mirrored, prefetched and aliased."""
    w13, w2 = layer.w13_weight, layer.w2_weight
    if w13.dtype == torch.bfloat16:
        return [
            _make_spec("w13", w13, is_nz=False, natural=(H, 2 * F)),
            _make_spec("w2", w2, is_nz=False, natural=(F, H)),
        ]
    scale13 = getattr(layer, "w13_weight_scale", None)
    scale2 = getattr(layer, "w2_weight_scale", None)
    if scale13 is None or scale2 is None:
        raise NotImplementedError(
            "MoonEP slot mode with quantized weights requires both w13_weight_scale and w2_weight_scale on the layer."
        )
    for name, weight in (("w13", w13), ("w2", w2)):
        if int(weight.element_size()) != 1:
            raise NotImplementedError(
                f"MoonEP slot mode: unsupported {name} layout dtype={weight.dtype} "
                f"element_size={weight.element_size()}."
            )
    return [
        _make_spec("w13", w13, is_nz=True),
        _make_spec("w13_scale", scale13, is_nz=False),
        _make_spec("w2", w2, is_nz=True),
        _make_spec("w2_scale", scale2, is_nz=False),
    ]


@dataclass(frozen=True)
class _LayerWeightLayout:
    E: int
    H: int
    F: int
    epn: int
    B: int
    specs: list[_PayloadSpec]


def _layer_weight_layout(layer, E: int, H: int) -> _LayerWeightLayout:
    """Describe loaded payloads without allocating Slot or cache storage."""
    if not hasattr(layer, "w13_weight") or not hasattr(layer, "w2_weight"):
        raise NotImplementedError("MoonEP Slot requires unsplit expert weight tensors after loading")
    w13 = layer.w13_weight
    epn = int(w13.shape[0])
    configured_slot_count = envs.MOONEP_REDUNDANT_EXPERT_SLOT_COUNT
    B = int(epn if configured_slot_count is None else configured_slot_count)
    if B < 1 or B > epn:  # noqa: SIM300 - retain the original Slot-bound check
        raise ValueError(f"MOONEP_REDUNDANT_EXPERT_SLOT_COUNT must be in [1, {epn}], got {B}")
    F = _infer_F(layer, H)
    return _LayerWeightLayout(E, H, F, epn, B, _payload_specs(layer, H, F))


def _slot_pool_key(E: int, H: int, F: int, B: int, rank: int, dev: int, specs: list[_PayloadSpec]) -> tuple:
    return (E, H, F, B, rank, dev, tuple((spec.name, *spec.carrier) for spec in specs))


def _build_slot_states(entries, rank: int, dev: int) -> list:
    """Size every pool from unique registered layers before any pool allocation."""
    layers, counts, seen = [], {}, set()
    for entry in entries:
        layer = entry.get("layer")
        if layer is None or id(layer) in seen:
            continue
        seen.add(id(layer))
        layout = _layer_weight_layout(layer, int(entry["E"]), int(entry["H"]))
        key = _slot_pool_key(layout.E, layout.H, layout.F, layout.B, rank, dev, layout.specs)
        layers.append((layer, layout, key))
        counts[key] = counts.get(key, 0) + 1

    states, indices = [], {}
    for layer, layout, key in layers:
        cache_index = indices.get(key, 0)
        state = _LayerWeightState(
            layer,
            layout,
            rank,
            dev,
            cache_layer_index=cache_index,
            cache_layer_count=counts[key],
        )
        states.append(state)
        indices[key] = cache_index + 1
    if indices != counts:
        raise RuntimeError("MoonEP Slot cache layer counts do not match registered layers")
    return states


# ---------------------------------------------------------------------------
# Per-(E, H, F, payload geometry) shared slot pool + per-layer home mirrors
# ---------------------------------------------------------------------------
class _SharedSlots:
    """B prefetch slot chunks per payload, shared across layers of one shape.

    Safe: one layer's dispatch->prefetch->GEMM->combine fully contains its slot
    use, and MoE layers run serially on the stream."""

    _lock = threading.Lock()
    _pools: dict[tuple, dict] = {}

    @classmethod
    def get(
        cls, E: int, H: int, F: int, B: int, rank: int, dev: int, specs: list[_PayloadSpec], *, cache_layers: int
    ) -> dict:
        if cache_layers < 1:
            raise ValueError("MoonEP Slot cache requires at least one registered layer per pool")
        key = _slot_pool_key(E, H, F, B, rank, dev, specs)
        with cls._lock:
            pool = cls._pools.get(key)
            if pool is not None and pool["cache_layers"] != cache_layers:
                raise RuntimeError("MoonEP Slot cache pool layer count changed after allocation")
            if pool is None:
                from ascend_moonep.buffer_c import create_vmm_physical

                # 每个 payload 单独一块槽：四个 payload 的逐专家字节数互不相同。
                slots = {}
                for spec in specs:
                    rows, cols = spec.carrier
                    slot, _, _ = create_vmm_physical([B, rows, cols], torch.bfloat16, dev)
                    slot.zero_()
                    slots[spec.name] = slot
                # One contiguous cache allocation per payload, sized to the
                # actual unique registered layers using this exact Slot pool.
                cache = {}
                if _CACHED_EXPERT_COUNT > 0:
                    for spec in specs:
                        rows, cols = spec.carrier
                        c, _, _ = create_vmm_physical(
                            [cache_layers, _CACHED_EXPERT_COUNT, rows, cols], torch.bfloat16, dev
                        )
                        c.zero_()
                        cache[spec.name] = c
                # full [E+B] projections need home chunks; built per layer, so
                # keep slot chunks here and let _LayerState build the fulls.
                pool = {"slots": slots, "cache": cache, "B": B, "cache_layers": cache_layers}
                cls._pools[key] = pool
        return pool


class _LayerWeightState:
    """Home mirrors + full projections + packed [epn+B] aliases for ONE MoE layer."""

    def __init__(
        self,
        layer,
        layout: _LayerWeightLayout,
        rank: int,
        dev: int,
        *,
        cache_layer_index: int,
        cache_layer_count: int,
    ):
        from ascend_moonep.buffer_c import (
            attach_local_source_vas,
            create_full_projection_tensor,
            create_vmm_physical,
        )

        if not 0 <= cache_layer_index < cache_layer_count:
            raise ValueError("MoonEP Slot cache layer index is outside its pool capacity")
        E, H, F, epn, B = layout.E, layout.H, layout.F, layout.epn, layout.B
        w13 = layer.w13_weight
        self.E, self.H, self.F, self.epn, self.B = E, H, F, epn, B
        self._specs = layout.specs

        shared = _SharedSlots.get(E, H, F, B, rank, dev, self._specs, cache_layers=cache_layer_count)
        for spec in self._specs:
            cached = shared["cache"].get(spec.name)
            if _CACHED_EXPERT_COUNT > 0 and (
                cached is None or tuple(cached.shape) != (cache_layer_count, _CACHED_EXPERT_COUNT, *spec.carrier)
            ):
                raise RuntimeError("MoonEP Slot cache allocation does not match registered layers")
        self.slots: dict[str, torch.Tensor] = {}
        self.homes: dict[str, torch.Tensor] = {}
        self.fulls: dict[str, torch.Tensor] = {}
        self.aliases: dict[str, torch.Tensor] = {}
        for spec in self._specs:
            rows, cols = spec.carrier
            # per-layer home chunk (mirror of the local vllm weight rows)
            home, _, _ = create_vmm_physical([epn, rows, cols], torch.bfloat16, dev)
            home.zero_()
            slot = shared["slots"][spec.name]
            full = create_full_projection_tensor(home, slot, E, B, rank, dev)
            attach_local_source_vas(full, home.data_ptr(), slot.data_ptr())
            self.slots[spec.name] = slot
            self.homes[spec.name] = home
            self.fulls[spec.name] = full
            self.aliases[spec.name] = _packed_alias(home, slot, dev, spec)

        self._w13_src = w13  # keep the original Parameter identity for layer lookup
        self._w2_src = layer.w2_weight
        self._published = False
        self._layer = layer
        # persistent per-layer expert cache (FreeBalance-style reuse): one cache
        # chunk per payload spec (bf16 carriers, like the slots themselves)
        self.cache_layer_index = cache_layer_index
        self.caches = dict(shared.get("cache") or {})
        self.cache_experts = [-1] * max(_CACHED_EXPERT_COUNT, 1)
        self.cache_next = 0
        self._vmm_tensors = list(self.homes.values())
        # Read-only metadata is stable for the life of this layer's aliases.
        self._prefetch_views = [self.fulls[spec.name] for spec in self._specs]
        self._compute_weights = MoEWeights(
            w1=self.aliases["w13"],
            w2=self.aliases["w2"],
            w1_scale=self.aliases.get("w13_scale"),
            w2_scale=self.aliases.get("w2_scale"),
        )
        self._shared = shared
        self._home_gmm_runner = None

    def publish(self) -> None:
        """One-time mirror of the local vllm weight rows into the home chunks.

        Must run after ShmemRuntime init (copies only; no new descriptors).
        Quantized payloads are NZ / 4-D non-contiguous views, so every mirror is
        a raw byte copy of the payload range (``_flat_bytes``).

        For static BF16 and supported W4A8MXFP weights, transfer the existing
        Parameters to Home-backed views. Parameter identity is the runtime's
        layer lookup key. Weight reload and sleep require separate ownership.
        """
        if self._published:
            return
        with torch.no_grad():
            for spec in self._specs:
                dst = _flat_bytes(self.homes[spec.name], spec.nbytes)
                src = _flat_bytes(spec.ref, spec.nbytes)
                dst.copy_(src)
        self._release_sources()
        self._published = True

    def _release_sources(self) -> None:
        """Point the layer's weight Parameters at the home chunks (drop the copies)."""
        if self._specs[0].ref.dtype != torch.bfloat16:
            if getattr(self._layer, "quant_type", None) == QuantType.W4A8MXFP:
                self._release_mxfp_sources()
            return
        released = False
        for name, src_attr, spec_name in (("w13_weight", "_w13_src", "w13"), ("w2_weight", "_w2_src", "w2")):
            src = getattr(self, src_attr, None)
            home = self.homes.get(spec_name)
            if src is None or home is None:
                continue
            layer_param = getattr(self._layer, name, None)
            if layer_param is None:
                continue
            if layer_param is not src:
                # the layer was re-parameterised after the state was built: the
                # id() lookup would already be broken, so leave it alone
                continue
            if tuple(layer_param.shape) != tuple(home.shape) or layer_param.dtype != home.dtype:
                raise RuntimeError(
                    f"MoonEP release: {name} shape/dtype changed "
                    f"({tuple(layer_param.shape)} {layer_param.dtype} vs "
                    f"{tuple(home.shape)} {home.dtype}); refusing to swap"
                )
            # in-place storage swap: the Parameter object (and therefore the
            # runtime's id()-keyed state lookup) is preserved
            layer_param.data = home
            released = True
        if released:
            self._released = True

    def _release_mxfp_sources(self) -> None:
        """Transfer the four loaded packed-NZ / E8M0 parameters as one layer."""
        import torch_npu

        replacements = []
        for spec in self._specs:
            ref = spec.ref
            field = spec.name.replace("_scale", "_weight_scale") if not spec.is_nz else spec.name + "_weight"
            if getattr(self._layer, field) is not ref:
                raise RuntimeError(f"MoonEP: {field} changed after Home allocation")
            home = self.homes[spec.name]
            ptr = home.data_ptr()
            if spec.is_nz:
                if ref.dtype != torch.uint8 or ref.ndim != 3:
                    return
                experts, packed_in, out = ref.shape
                base_shape = (experts, out, packed_in)
            else:
                if ref.dtype != getattr(torch, "float8_e8m0fnu", None) or ref.ndim != 4 or ref.shape[-1] != 2:
                    return
                experts, pairs, out, pair = ref.shape
                base_shape = (experts, out, pairs * pair)

            # Rebuild the pre-transpose descriptor. Creating the 4-D scale
            # directly gives it NCHW metadata on A5 even with ND requested.
            view = _sym_tensor(ptr, base_shape, ref.dtype, ref.device, nbytes=spec.nbytes, npu_format=spec.npu_format)
            if not spec.is_nz:
                view = view.view(experts, out, pairs, pair)
            view = view.transpose(1, 2)
            if (
                view.shape != ref.shape
                or view.stride() != ref.stride()
                or torch_npu.get_npu_format(view) != spec.npu_format
                or view.data_ptr() != ptr
            ):
                return  # The published mirror remains usable for other layouts.
            replacements.append((ref, view))

        # _flat_bytes wraps raw pointers without retaining allocator ownership.
        # Finish the copies before releasing the last reference to the old storage.
        torch.npu.current_stream().synchronize()
        originals = [ref.detach() for ref, _ in replacements]
        try:
            for ref, view in replacements:
                ref.data = view
        except Exception:
            for (ref, _), original in zip(replacements, originals):
                ref.data = original
            raise
        self._released = True

    def prefetch_views(self) -> list[torch.Tensor]:
        """The [E+B] projection descriptors handed to the prefetch kernel."""
        return self._prefetch_views

    def compute_weights(self) -> MoEWeights:
        """Grouped-matmul weights: [epn+B] aliases, scales included."""
        return self._compute_weights

    def home_gmm_runner(self):
        """Share the native entry points within this existing shape/device pool."""
        if self._home_gmm_runner is None:
            from ascend_moonep.home_gmm import HomeGmmRange

            with _SharedSlots._lock:
                runner = self._shared.get("home_gmm_range")
                if runner is None:
                    runner = HomeGmmRange()
                    self._shared["home_gmm_range"] = runner
                self._home_gmm_runner = runner
        return self._home_gmm_runner

    def destroy(self):
        if getattr(self, "_released", False):
            # after a release the home chunks *are* the layer's weights: freeing
            # them would leave the layer pointing at unmapped memory
            raise RuntimeError(
                "MoonEP: refusing to destroy weight state whose home chunks back "
                "the layer after source-storage handoff."
            )
        for tensor in self._vmm_tensors:
            alloc = getattr(tensor, "_vmm_allocation", None)
            if alloc is not None:
                try:  # noqa: SIM105 - retain the existing best-effort cleanup
                    alloc.destroy()
                except Exception:
                    pass


def _packed_alias(home: torch.Tensor, slot: torch.Tensor, dev: int, spec: _PayloadSpec) -> torch.Tensor:
    """Zero-copy [epn+B] alias: home at rows 0..epn, slot after.

    The alias mirrors ``spec.ref`` field by field — NZ payloads are built with
    the *cast* shape plus their format and then transposed (exactly how
    ``process_weights_after_loading`` produced the reference), ND payloads copy
    shape[1:]/stride/storage_offset/format verbatim. On-card spike M4 shows both
    variants are bit-identical to the real weights on a W4A8MXFP gmm1.
    """
    from ascend_moonep.buffer_c import (
        VmmProjection,
        acl_vmm_granularity,
        acl_vmm_map,
        acl_vmm_release,
        acl_vmm_reserve,
        acl_vmm_set_access,
        acl_vmm_unmap,
    )

    try:
        from ascend_moonep.buffer_c import acl_vmm_access_device_id
    except ImportError:  # 上游 buffer_c（无重映射修复）：逻辑号即物理号
        acl_vmm_access_device_id = int

    ref = spec.ref
    epn = int(home.shape[0])
    B = int(slot.shape[0])
    row = spec.row_bytes
    gran = int(acl_vmm_granularity(int(dev)))
    slot_off = epn * row
    if slot_off % gran != 0:
        raise RuntimeError(
            f"packed [epn+B] alias requires epn*row aligned to VMM gran: "
            f"epn*row={slot_off} gran={gran} (payload={spec.name})."
        )
    home_alloc = getattr(home, "_vmm_allocation", None)
    slot_alloc = getattr(slot, "_vmm_allocation", None)
    if home_alloc is None or slot_alloc is None:
        raise ValueError("home/slot must be VMM chunks (create_vmm_physical)")
    n_rows = epn + B
    total = slot_off + int(slot_alloc.size)
    base = int(acl_vmm_reserve(total))
    segments = []
    try:
        for alloc, off in ((home_alloc, 0), (slot_alloc, slot_off)):
            va = base + off
            acl_vmm_map(va, int(alloc.size), alloc.mem_handle, 0)
            # set_access 的 location.id 要物理号，其余 VMM 接口要逻辑号
            # （docs/cross_repo_fixes.md §3）；buffer_c 里的同类调用都做了这一步。
            acl_vmm_set_access(va, int(alloc.size), acl_vmm_access_device_id(int(dev)))
            segments.append((va, int(alloc.size), alloc.mem_handle))
    except Exception:
        for va, _, _ in segments:
            try:  # noqa: SIM105 - retain the existing best-effort cleanup
                acl_vmm_unmap(va)
            except Exception:
                pass
        try:  # noqa: SIM105 - retain the existing best-effort cleanup
            acl_vmm_release(base)
        except Exception:
            pass
        raise
    device = torch.device(f"npu:{int(dev)}")
    if spec.is_nz:
        cast = _sym_tensor(
            base,
            (n_rows, int(ref.shape[2]), int(ref.shape[1])),
            ref.dtype,
            device,
            nbytes=n_rows * row,
            npu_format=spec.npu_format,
        )
        tensor = cast.transpose(1, 2)
    else:
        # Within-row strides come from the reference, but the row base is the
        # alias head: the reference's own storage_offset (a `full[rank*epn:...]`
        # slice carries one) must not shift the alias into the slot segment.
        stride = (int(ref.stride(0)),) + tuple(int(s) for s in ref.stride()[1:])
        tensor = _sym_tensor(
            base,
            (n_rows,) + tuple(int(d) for d in ref.shape[1:]),
            ref.dtype,
            device,
            stride=stride,
            nbytes=n_rows * row,
            npu_format=spec.npu_format,
        )
    expected_shape = (n_rows,) + tuple(int(d) for d in ref.shape[1:])
    # stride(0) is in elements, row_bytes in bytes: only equal for 1-byte payloads.
    stride0_bytes = int(tensor.stride(0)) * int(ref.element_size())
    if tensor.shape != expected_shape or stride0_bytes != row or int(tensor.data_ptr()) != base:
        raise RuntimeError(
            f"MoonEP slot mode: {spec.name} alias mismatch shape={tuple(tensor.shape)} "
            f"(want {expected_shape}) stride0={stride0_bytes}B (want {row}B) "
            f"ptr={int(tensor.data_ptr())} (want {base})."
        )
    tensor._projection = VmmProjection(base, segments)  # noqa: SLF001
    return tensor


# ---------------------------------------------------------------------------
# Same-layer prefetch uses the real plan after planning and before dispatch.
# SDK prefetch never inserts cross-rank barriers. Event dependencies below
# order side-stream transfers, consumers, and slot reuse.
# none: serial prefetch on the main stream, followed by unified GEMM.
# dispatch (default): side-stream prefetch overlaps dispatch; join before GEMM.
# dispatch|home_gemm1: run Home GEMM1 before joining, then run Slot GEMM1.
# Snapshot the validated setting at module load, not in the per-layer hot path.
_PREFETCH_OVERLAP = envs.MOONEP_PREFETCH_SIDE_STREAM_OVERLAP

_PF: dict = {"stream": None, "done": False, "cur_key": None}


def select_prefetch_layer(layer) -> None:
    """Select the weight owner at the actual MoE call, including wrapped models."""
    _PF["cur_key"] = id(layer.w13_weight)


def _pf_prefetch(buffer, state, plan):
    buffer.prefetch_projections(plan=plan, projections=state.prefetch_views())


def maybe_install_pf_stream(decoder_layers: Iterable[torch.nn.Module]) -> None:
    """Let the real-plan prefetch hook identify the current MoE layer.

    钩子在库的 `_run_dispatch_on_current_stream` 里触发（planning 之后、
    dispatch 之前），需要拿到本层的 `_LayerWeightState` 才能发 prefetch。
    层的识别方式：用 `experts.w13_weight` 的 id 当 key，
    由 decoder 层的 forward pre-hook 在每层开始时记下。
    """
    if _PREFETCH_OVERLAP == "none" or _PF["done"]:
        return
    from vllm.distributed.parallel_state import get_ep_group

    from vllm_ascend.ascend_config import get_ascend_config

    if not get_ascend_config().enable_moonep:
        return
    if get_ep_group().world_size <= 1:
        return
    try:
        layers = list(decoder_layers)
        if not layers:
            return

        def _pre(blk, args, kwargs):
            try:
                experts = getattr(getattr(blk, "mlp", None), "experts", None)
                experts = getattr(experts, "routed_experts", experts)
                if experts is not None:
                    _PF["cur_key"] = id(experts.w13_weight)
            except Exception:  # noqa: BLE001 - 探测逻辑绝不能打断推理
                pass
            return None

        for blk in layers:
            blk.register_forward_pre_hook(_pre, with_kwargs=True)
        _PF["done"] = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to register MoonEP prefetch layer hooks: %s", exc)


def _prepare_slot_cache(state, plan):
    """Mask cache hits before issuing remote GETs; consume once per layer call."""
    if _CACHED_EXPERT_COUNT <= 0 or state.caches.get("w13") is None:
        return plan, None, False
    from vllm.distributed.parallel_state import get_ep_group

    rank = get_ep_group().rank_in_group
    row = [int(v) for v in plan.experts_to_copy[rank].tolist()]
    hits, misses = [], []
    masked = list(row)
    for i, expert in enumerate(row):
        if expert < 0:
            continue
        if expert in state.cache_experts:
            hits.append((state.cache_experts.index(expert), i))
            masked[i] = -1
        else:
            misses.append((expert, i))
    if hits:
        experts_to_copy = plan.experts_to_copy.clone()
        experts_to_copy[rank] = torch.tensor(masked, dtype=experts_to_copy.dtype, device=experts_to_copy.device)
        plan = dataclasses.replace(plan, experts_to_copy=experts_to_copy)
    return plan, (hits, misses), not misses


def udma_prefetch_overlap_requested(use_udma: bool, slot_mode: bool) -> bool:
    """Use the existing side-stream switch; no independent transport toggle."""
    return bool(use_udma and slot_mode and _PREFETCH_OVERLAP != "none")


def _pf_stream_hook(buffer, lookup_slot_state: Callable[[int], _LayerWeightState | None]):
    """装在库的 `after_planning_hook` 上的回调：在 side stream 上发 prefetch。

    为什么必须挂在这里：prefetch 要读 planning 产出的 `experts_to_copy`，又希望
    和随后的 dispatch 并行。如果在 `buffer.dispatch()` 返回之后（`run_slot_forward`
    里）才 record_event，那事件已经把 dispatch 也圈进去了，side stream 会一直等
    dispatch 完成 —— 没有重叠。只有在本钩子这个时间点，主 stream 上排在最前面的
    才是 planning。
    """

    def _after(plan):
        ctx = buffer._require_ctx()
        # Check before any stream/event submission. Never silently run an old
        # UDMA runtime on shared QP0 when overlap is requested.
        if (
            _PREFETCH_OVERLAP != "none"
            and ctx.get("_prefetch_use_udma", False)
            and not ctx.get("udma_prefetch_overlap_ready", False)
        ):
            raise RuntimeError("UDMA prefetch overlap requires dedicated QP initialization")
        try:
            key = _PF.get("cur_key")
            state = lookup_slot_state(key) if key is not None else None
            if state is None:  # 本层第一次跑，还没建权重状态
                return
            import torch_npu

            main = torch_npu.npu.current_stream()
            ctx = buffer._require_ctx()
            if ctx.get("slot_prefetch_ready") is not None:
                raise RuntimeError("Unconsumed Slot prefetch from the preceding layer")
            # Cache lookup must precede the side-stream GET. Doing it after
            # dispatch would mask a plan whose full transfer was already issued.
            plan_use, cache_plan, skip_prefetch = _prepare_slot_cache(state, plan)
            if _PF["stream"] is None:
                _PF["stream"] = torch_npu.npu.Stream()
            side = _PF["stream"]
            side.wait_event(main.record_event())
            with torch_npu.npu.stream(side):
                if not skip_prefetch:
                    _pf_prefetch(buffer, state, plan_use)
            # Keep the prepared decision and completion event together on this
            # Buffer so the forward path can reuse both.
            ctx["slot_prefetch_ready"] = (state, plan_use, cache_plan, skip_prefetch, side.record_event())
        except Exception as exc:  # noqa: BLE001 - keep the MTE fallback
            if ctx.get("_prefetch_use_udma", False):
                # A partial issue cannot safely fall back and submit again.
                raise RuntimeError("UDMA side-stream prefetch failed") from exc
            logger.warning_once("MoonEP side-stream prefetch failed: %s", exc)

    return _after


# ---------------------------------------------------------------------------
# Runtime forward: dispatch -> prefetch -> grouped GEMM on [epn+B] aliases
# ---------------------------------------------------------------------------
def run_slot_forward(
    *,
    state: _LayerWeightState,
    lookup_slot_state: Callable[[int], _LayerWeightState | None],
    buffer,
    plan,
    token_dispatch_output,
    fused_experts_input: MoEFusedExpertsInput,
    use_fusion_ops: bool | None,
    moe_config=None,
    quant_method=None,
    record_before_gmm2: bool = False,
):
    """Prefetch remote slot weights, then run the grouped GEMMs on the aliases.

    Mirrors MindSpeed shmem_ep_forward's middle section (weights publish is done
    once at init for inference). Goes through the standard
    ``build_mlp_compute_input`` + ``apply_slot_mlp`` entry points so bf16 and
    quantized layers share one operator call with the owner path, and only the
    weight payloads are swapped for the aliases. Returns the MLP output rows
    (same order as the dispatched rows) for the caller's combine.
    """
    try:
        capturing = _EXTRA_CTX.capturing
    except Exception:
        capturing = False  # outside an engine forward context (harness use)
    if capturing:
        raise RuntimeError(
            "MoonEP slot mode cannot run inside captured graphs; "
            "run with --enforce-eager (or disable graph capture) when enable_moonep is on."
        )
    home_overlap = _PREFETCH_OVERLAP == "dispatch|home_gemm1"
    if home_overlap and _CACHED_EXPERT_COUNT > 0:
        raise RuntimeError(
            "Home GEMM overlap currently requires cache disabled (MOONEP_CACHED_REDUNDANT_EXPERT_COUNT=0)"
        )
    plan_use = plan
    # Install before the next dispatch; first use safely takes the main-stream
    # path. When the hook ran, reuse its cache decision and completion event.
    prefetched = None
    if _PREFETCH_OVERLAP != "none":
        _pf_ctx = buffer._require_ctx()
        if _pf_ctx.get("after_planning_hook") is None:
            _pf_ctx["after_planning_hook"] = _pf_stream_hook(buffer, lookup_slot_state)
        prefetched = _pf_ctx.pop("slot_prefetch_ready", None)
    pf_ev = None
    if prefetched is not None:
        prefetched_state, plan_use, cache_plan, skip_prefetch, pf_ev = prefetched
        if prefetched_state is not state:
            raise RuntimeError("Slot prefetch belongs to a different MoE layer")
    else:
        plan_use, cache_plan, skip_prefetch = _prepare_slot_cache(state, plan_use)
    if not skip_prefetch and pf_ev is None:
        buffer.prefetch_projections(
            plan=plan_use,
            projections=state.prefetch_views(),
        )
    if pf_ev is not None and not home_overlap:
        # Side-stream prefetch overlaps dispatch; the main stream proceeds until
        # cache refill / unified GEMM needs Slot weights. Join here before
        # consuming them; actual overlap depends on the current workload.
        import torch_npu as _tn_pf

        _tn_pf.npu.current_stream().wait_event(pf_ev)
    if cache_plan is not None:
        # stream-ordered after the prefetch: hits come from this layer's cache
        # (no remote GET), the fetched experts refill the cache for later steps
        hits, misses = cache_plan
        with torch.no_grad():
            for j, i in hits:
                for name, cache in state.caches.items():
                    state.slots[name][i].copy_(cache[state.cache_layer_index, j])
            for e, i in misses:
                j = state.cache_next
                state.cache_next = (j + 1) % max(_CACHED_EXPERT_COUNT, 1)
                for name, cache in state.caches.items():
                    cache[state.cache_layer_index, j].copy_(state.slots[name][i])
                state.cache_experts[j] = e
    mlp_compute_input = build_mlp_compute_input(
        fused_experts_input=fused_experts_input,
        token_dispatch_output=token_dispatch_output,
        use_fusion_ops=use_fusion_ops,
        moe_config=moe_config,
    )
    # Native quant hooks consume the explicit Home/Slot layer view.
    mlp_compute_input = dataclasses.replace(mlp_compute_input, weights=state.compute_weights())
    if home_overlap:
        runner = state.home_gmm_runner()
        rows, before_gmm2_evt = home_gemm_apply_mlp(
            mlp_compute_input=mlp_compute_input,
            home_experts=state.epn,
            intermediate_size=state.F,
            range_kernel=runner,
            prefetch_ready=pf_ev,
            record_before_gmm2=record_before_gmm2,
        )
    else:
        rows, before_gmm2_evt = apply_slot_mlp(mlp_compute_input, quant_method, record_before_gmm2=record_before_gmm2)
    return rows, before_gmm2_evt


__all__ = [
    "_LayerWeightState",
    "_SharedSlots",
    "run_slot_forward",
]
