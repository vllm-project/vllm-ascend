# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

"""Static backend selection and semantic dispatch for delayed-mHC seams."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Protocol

import torch

from .contracts import (
    MhcHardwareCapability,
    MhcSeamInputs,
    MhcSeamOutputs,
    MhcSeamSpec,
    NumericsProfile,
    PreparedMhcWeights,
    WeightFormat,
)
from .packing import C4_COEFFICIENTS, OZAKI_FORMATS, ozaki_shape, pack_ozaki_weights
from .reference import coefficients

logger = logging.getLogger(__name__)
ACTIVATION_DTYPES = (torch.bfloat16, torch.float16)
A2_HIDDEN_SIZE = 5120
A2_MAX_ROWS = 72
A3_MAX_HIDDEN_SIZE = 16320
A3_FUSED_GRAPH_MODES = frozenset(
    ("none", "eager", "aclgraph", "full", "full_decode_only", "piecewise", "full_and_piecewise")
)
A3_GRAPH_MODES = A3_FUSED_GRAPH_MODES
A3_DECOUPLED_GRAPH_MODES = frozenset(("none", "eager", "aclgraph", "full", "full_decode_only"))


class MhcBackend(Protocol):
    def supports(self, spec: MhcSeamSpec, cap: MhcHardwareCapability) -> tuple[bool, str]: ...

    def prepare(self, weights: dict, device: str, cap: MhcHardwareCapability) -> PreparedMhcWeights: ...

    def run(self, inputs: MhcSeamInputs, prepared: PreparedMhcWeights, attrs: dict) -> MhcSeamOutputs: ...


@dataclass(frozen=True, slots=True)
class _SeamAttrs:
    norm_eps: float = 1e-20
    hc_eps: float = 1e-6
    iterations: int = 20
    post_mult: float = 2.0
    return_v_fp32: bool = False
    numerics_profile: NumericsProfile = NumericsProfile.STANDARD_NR1


def _parse_attrs(attrs: dict) -> _SeamAttrs:
    try:
        parsed = _SeamAttrs(**attrs)
    except TypeError as exc:
        raise ValueError(f"Invalid mHC attributes: {exc}") from exc
    if not all(
        isinstance(eps, (int, float)) and math.isfinite(eps) and eps > 0 for eps in (parsed.norm_eps, parsed.hc_eps)
    ):
        raise ValueError("norm_eps and hc_eps must be positive and finite")
    if type(parsed.iterations) is not int or not 0 < parsed.iterations < 2**32:
        raise ValueError("iterations must be a positive uint32")
    if not isinstance(parsed.post_mult, (int, float)) or not math.isfinite(parsed.post_mult):
        raise ValueError("post_mult must be finite")
    if type(parsed.return_v_fp32) is not bool or not isinstance(parsed.numerics_profile, NumericsProfile):
        raise ValueError("Expected a bool return_v_fp32 and a NumericsProfile")
    return parsed


def _check_tensor(name: str, tensor: object, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a Tensor")
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_inputs(inputs: MhcSeamInputs, prepared: PreparedMhcWeights) -> tuple[int, int, int]:
    if not isinstance(inputs.y, torch.Tensor) or inputs.y.ndim != 2:
        raise ValueError("y must have shape [T, D]")
    if not isinstance(inputs.residual, torch.Tensor) or inputs.residual.ndim != 3:
        raise ValueError("residual must have shape [T, C, D]")
    tokens, width = inputs.y.shape
    lanes = inputs.residual.shape[1]
    dtype = inputs.y.dtype
    if min(tokens, width, lanes) <= 0:
        raise ValueError("T, C and D must be positive")
    if dtype not in ACTIVATION_DTYPES:
        raise ValueError(f"y must be bfloat16 or float16, got {dtype}")
    device = prepared.device
    mixes = lanes * (lanes + 2)
    expected = (
        ("y", (tokens, width), dtype),
        ("residual", (tokens, lanes, width), dtype),
        ("post", (tokens, lanes), torch.float32),
        ("comb", (tokens, lanes, lanes), torch.float32),
        ("pre_mix", (tokens, lanes), torch.float32),
        ("gamma", (width,), dtype),
        ("hc_scale", (3,), torch.float32),
        ("hc_base", (mixes,), torch.float32),
    )
    for name, shape, tensor_dtype in expected:
        _check_tensor(name, getattr(inputs, name), shape, tensor_dtype, device)
    weight_shape: tuple[int, ...]
    if prepared.weight_format == WeightFormat.FP32_CANONICAL:
        weight_shape, weight_dtype = (mixes, lanes * width), torch.float32
    else:
        weight_shape, weight_dtype = ozaki_shape(lanes * width, prepared.weight_format), torch.bfloat16
    _check_tensor("prepared hc_fn", prepared.tensor, weight_shape, weight_dtype, device)
    return tokens, lanes, width


def resolve_mhc_device(device: str | torch.device) -> torch.device:
    try:
        target = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Invalid mHC preparation device: {device}") from exc
    if target.type == "npu" and target.index is None:
        npu_mod = getattr(torch, "npu", None)
        current_device = getattr(npu_mod, "current_device", None)
        if callable(current_device):
            return torch.device("npu", int(current_device()))
        raise ValueError("mHC preparation requires an explicit NPU device index")
    return target


def _canonical_source(weights: dict, device: str) -> tuple[torch.Tensor, torch.device, str]:
    target = resolve_mhc_device(device)
    source = weights.get("hc_fn")
    if not isinstance(source, torch.Tensor) or source.ndim != 2 or min(source.shape) <= 0:
        raise ValueError("hc_fn must be a nonempty 2D tensor")
    if source.dtype != torch.float32 or not source.is_contiguous():
        raise ValueError("hc_fn must be contiguous float32")
    if source.device.type != "cpu" and source.device != target:
        raise ValueError(f"hc_fn must be on CPU or {target}, got {source.device}")
    version = weights.get("version_tag")
    if not isinstance(version, str) or not version:
        raise ValueError("mHC preparation requires a nonempty version_tag")
    return source, target, version


def _prepare_canonical(weights: dict, device: str) -> PreparedMhcWeights:
    source, target, version = _canonical_source(weights, device)
    tensor = source.to(target)
    return PreparedMhcWeights(tensor, tensor.device, version, WeightFormat.FP32_CANONICAL)


class A5NativeMhcBackend:
    def supports(self, spec: MhcSeamSpec, cap: MhcHardwareCapability) -> tuple[bool, str]:
        if cap.arch_gen != "arch35" or not cap.has_a5_native_op:
            return False, "requires the qualified arch35 native adapter"
        if spec.dtype not in cap.dtype_support or spec.is_layer0_entry or spec.is_write_kv_path:
            return False, "requires a supported dtype and a full seam"
        return True, "arch35 native adapter qualified by the runner"

    def prepare(self, weights: dict, device: str, cap: MhcHardwareCapability) -> PreparedMhcWeights:
        if cap.arch_gen != "arch35" or not cap.has_a5_native_op:
            raise ValueError("A5 preparation requires a qualified arch35 adapter")
        adapter = weights.get("native_adapter")
        if not callable(adapter):
            raise ValueError("A5 preparation requires the runner's native_adapter callable")
        return replace(_prepare_canonical(weights, device), native_adapter=adapter)

    def run(self, inputs: MhcSeamInputs, prepared: PreparedMhcWeights, attrs: dict) -> MhcSeamOutputs:
        _validate_inputs(inputs, prepared)
        _parse_attrs(attrs)
        if prepared.native_adapter is None or prepared.weight_format != WeightFormat.FP32_CANONICAL:
            raise ValueError("A5 run requires weights prepared by its native adapter")
        return prepared.native_adapter(inputs, prepared.tensor, attrs)


def _is_a3_soc(soc_name: str) -> bool:
    soc = soc_name.lower()
    return soc == "ascend910c" or soc.startswith("ascend910_93")


def _a3_capable(cap: MhcHardwareCapability) -> bool:
    return _is_a3_soc(cap.soc_name) and cap.arch_gen == "arch22" and 8 <= cap.aic_count <= 24 and cap.has_a3_fused_op


def _a3_seam_supported(lanes: int, width: int, dtype: torch.dtype) -> bool:
    return dtype == torch.bfloat16 and lanes == 4 and 0 < width <= A3_MAX_HIDDEN_SIZE and width % 64 == 0


def _validate_coefficient_inputs(
    residual: torch.Tensor,
    prepared: PreparedMhcWeights,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    r_sq: torch.Tensor | None,
) -> tuple[int, int, int]:
    if not isinstance(residual, torch.Tensor) or residual.ndim != 3:
        raise ValueError("residual must have shape [T, C, D]")
    tokens, lanes, width = residual.shape
    dtype = residual.dtype
    if min(tokens, lanes, width) <= 0:
        raise ValueError("T, C and D must be positive")
    if dtype not in ACTIVATION_DTYPES:
        raise ValueError(f"residual must be bfloat16 or float16, got {dtype}")
    device = prepared.device
    _check_tensor("residual", residual, (tokens, lanes, width), dtype, device)
    if not _a3_seam_supported(lanes, width, dtype) or prepared.weight_format not in OZAKI_FORMATS:
        raise ValueError("A3 decoupled run_coefficients requires BF16, C=4, qualified D, and prepared Ozaki weights")
    mixes = lanes * (lanes + 2)
    _check_tensor("hc_scale", hc_scale, (3,), torch.float32, device)
    _check_tensor("hc_base", hc_base, (mixes,), torch.float32, device)
    if r_sq is not None:
        _check_tensor("r_sq", r_sq, (tokens,), torch.float32, device)
    weight_shape = ozaki_shape(lanes * width, prepared.weight_format)
    _check_tensor("prepared hc_fn", prepared.tensor, weight_shape, torch.bfloat16, device)
    return tokens, lanes, width


class A3MegaMhcBackend:
    """Single-kernel fused Atlas A3 (arch22) MegaMHC seam backend."""

    def supports(self, spec: MhcSeamSpec, cap: MhcHardwareCapability) -> tuple[bool, str]:
        if not _a3_capable(cap):
            return False, "requires Ascend 910C / Atlas A3 (arch22), 8 to 24 AICs, and npu_mega_mhc_seam"
        if spec.dtype not in cap.dtype_support or not _a3_seam_supported(spec.C, spec.D, spec.dtype):
            return False, "requires BF16, C=4, and D aligned to 64 up to 16320"
        if spec.T is not None and not (0 < spec.T <= 0xFFFFFFFF):
            return False, "requires 0 < T <= uint32_max"
        if spec.is_layer0_entry or (spec.is_write_kv_path and not spec.write_kv_migrated):
            return False, "entry and unmigrated write_kv require canonical operators"
        if spec.weight_format not in OZAKI_FORMATS:
            return False, "requires a versioned Ozaki NZ or ND layout"
        if cap.superkernel_enabled:
            return False, "SuperKernel capture/replay for MegaMHC requires separate qualification"
        if cap.graph_mode.lower() not in A3_FUSED_GRAPH_MODES:
            return False, "requires an eager or ACLGraph seam"
        return True, "qualified Atlas A3 fused single-kernel seam (eager or ACLGraph)"

    def prepare(self, weights: dict, device: str, cap: MhcHardwareCapability) -> PreparedMhcWeights:
        if not _a3_capable(cap):
            raise ValueError("A3 preparation requires Ascend 910C / Atlas A3 (arch22), 8 to 24 AICs, and MegaMHC op")
        source, target, version = _canonical_source(weights, device)
        if target.type not in ("cpu", "npu"):
            raise ValueError(f"A3 preparation requires CPU staging or an NPU target, got {target}")
        rows, k = source.shape
        if rows != C4_COEFFICIENTS or k % 4 or not _a3_seam_supported(4, k // 4, torch.bfloat16):
            raise ValueError(f"A3 hc_fn must have shape [24, 4*D] for a qualified D, got {tuple(source.shape)}")
        fmt = weights.get("weight_format", WeightFormat.OZAKI_BF16_NZ_V1)
        packed = pack_ozaki_weights(source.cpu(), fmt).to(target)
        if target.type == "npu":
            torch.ops._C_ascend.prepare_device_constants(target.index)
        return PreparedMhcWeights(packed, packed.device, version, fmt, index_map_ready=target.type == "npu")

    def run(self, inputs: MhcSeamInputs, prepared: PreparedMhcWeights, attrs: dict) -> MhcSeamOutputs:
        _, lanes, width = _validate_inputs(inputs, prepared)
        if not _a3_seam_supported(lanes, width, inputs.y.dtype) or prepared.weight_format not in OZAKI_FORMATS:
            raise ValueError("A3 MegaMHC run requires BF16, C=4, qualified D, and prepared Ozaki weights")
        if not prepared.index_map_ready:
            raise ValueError("A3 device constants must be prepared before capture and run")
        args = _parse_attrs(attrs)
        outputs = torch.ops._C_ascend.npu_mega_mhc_seam(
            inputs.y,
            inputs.residual,
            inputs.post,
            inputs.comb,
            inputs.pre_mix,
            inputs.gamma,
            prepared.tensor,
            inputs.hc_scale,
            inputs.hc_base,
            weight_format=int(prepared.weight_format),
            numerics_profile=int(args.numerics_profile),
            norm_eps=args.norm_eps,
            hc_eps=args.hc_eps,
            iterations=args.iterations,
            post_mult=args.post_mult,
            return_v_fp32=args.return_v_fp32,
        )
        return MhcSeamOutputs(*outputs)


class A3DecoupledMhcBackend(A3MegaMhcBackend):
    """Decoupled pure-AIV main-chain + standalone Ozaki side-stream coefficient backend.

    Selected only when explicitly opted in by the internal seam spec
    (spec.prefer_side_stream=True), when decoupled operators are registered, and
    when SuperKernel is disabled.
    """

    def supports(self, spec: MhcSeamSpec, cap: MhcHardwareCapability) -> tuple[bool, str]:
        if not spec.prefer_side_stream:
            return False, "decoupled side-stream backend requires prefer_side_stream=True"
        if not cap.has_a3_decoupled_op:
            return False, "requires npu_mega_mhc_main and npu_mhc_coefficients_ozaki"
        if cap.superkernel_enabled:
            return False, "dual-stream decoupled MegaMHC is not yet qualified under SuperKernel"
        if cap.graph_mode.lower() not in A3_DECOUPLED_GRAPH_MODES:
            return False, "dual-stream decoupled MegaMHC requires eager or full ACLGraph capture"
        ok, reason = super().supports(spec, cap)
        if not ok:
            return False, reason
        return True, "qualified Atlas A3 decoupled main-chain + Ozaki coefficient seam"

    def run_main(
        self,
        inputs: MhcSeamInputs,
        prepared: PreparedMhcWeights,
        attrs: dict,
        *,
        return_r_sq: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Stage 1: execute the pure-AIV main-chain kernel on the current stream."""
        _, lanes, width = _validate_inputs(inputs, prepared)
        if not _a3_seam_supported(lanes, width, inputs.y.dtype) or prepared.weight_format not in OZAKI_FORMATS:
            raise ValueError("A3 decoupled run_main requires BF16, C=4, qualified D, and prepared Ozaki weights")
        if not prepared.index_map_ready:
            raise ValueError("A3 device constants must be prepared before capture and run")
        args = _parse_attrs(attrs)
        return torch.ops._C_ascend.npu_mega_mhc_main(
            inputs.y,
            inputs.residual,
            inputs.post,
            inputs.comb,
            inputs.pre_mix,
            inputs.gamma,
            norm_eps=args.norm_eps,
            return_v_fp32=args.return_v_fp32,
            return_r_sq=return_r_sq,
        )

    def run_coefficients(
        self,
        residual: torch.Tensor,
        prepared: PreparedMhcWeights,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        attrs: dict,
        *,
        r_sq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stage 2 (or standalone entry) Ozaki dual-BF16 coefficient kernel."""
        _validate_coefficient_inputs(residual, prepared, hc_scale, hc_base, r_sq)
        if not prepared.index_map_ready:
            raise ValueError("A3 device constants must be prepared before capture and run")
        args = _parse_attrs(attrs)
        return torch.ops._C_ascend.npu_mhc_coefficients_ozaki(
            residual,
            prepared.tensor,
            hc_scale,
            hc_base,
            r_sq=r_sq,
            weight_format=int(prepared.weight_format),
            numerics_profile=int(args.numerics_profile),
            norm_eps=args.norm_eps,
            hc_eps=args.hc_eps,
            iterations=args.iterations,
            post_mult=args.post_mult,
        )

    def launch_coefficients_async(
        self,
        residual: torch.Tensor,
        prepared: PreparedMhcWeights,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        attrs: dict,
        *,
        side_stream,
        ready_event,
        done_event,
        main_stream=None,
        r_sq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fork Ozaki coefficients onto side_stream after ready_event; caller joins done_event."""
        npu_mod = getattr(torch, "npu", None)
        if npu_mod is None:
            raise RuntimeError("torch.npu is required for launch_coefficients_async")
        is_capturing = getattr(npu_mod, "is_current_stream_capturing", None)
        capturing = bool(is_capturing()) if callable(is_capturing) else False
        resolved_main_stream = main_stream
        if resolved_main_stream is None and npu_mod is not None:
            current_stream = getattr(npu_mod, "current_stream", None)
            if callable(current_stream):
                resolved_main_stream = current_stream()

        if main_stream is None:
            ready_event.record()
        else:
            ready_event.record(main_stream)
        with npu_mod.stream(side_stream):
            side_stream.wait_event(ready_event)
            if not capturing:
                for tensor in (residual, hc_scale, hc_base, r_sq):
                    if tensor is not None:
                        tensor.record_stream(side_stream)
            try:
                outputs = self.run_coefficients(residual, prepared, hc_scale, hc_base, attrs, r_sq=r_sq)
            except Exception:
                done_event.record(side_stream)
                raise
            if not capturing and resolved_main_stream is not None:
                for tensor in outputs:
                    tensor.record_stream(resolved_main_stream)
            done_event.record(side_stream)
        return outputs

    def run(self, inputs: MhcSeamInputs, prepared: PreparedMhcWeights, attrs: dict) -> MhcSeamOutputs:
        """Serial fallback implementing the single-return MhcBackend protocol."""
        residual_out, v, v_fp32, r_sq = self.run_main(inputs, prepared, attrs, return_r_sq=False)
        post_out, comb_out, pre_out = self.run_coefficients(
            residual_out, prepared, inputs.hc_scale, inputs.hc_base, attrs, r_sq=r_sq
        )
        return MhcSeamOutputs(residual_out, v, v_fp32, post_out, comb_out, pre_out)


class A2OverlapMhcBackend:
    def supports(self, spec: MhcSeamSpec, cap: MhcHardwareCapability) -> tuple[bool, str]:
        if (
            cap.arch_gen != "arch20"
            or not cap.soc_name.lower().startswith(("ascend910b", "910b"))
            or not cap.has_a2_overlap_op
        ):
            return False, "requires the 910B overlap operators and runner prerequisites"
        if (
            spec.dtype != torch.bfloat16
            or spec.dtype not in cap.dtype_support
            or spec.C != 4
            or spec.D != A2_HIDDEN_SIZE
            or spec.T is None
            or not 0 < spec.T <= A2_MAX_ROWS
        ):
            return False, "A2 overlap requires BF16, C=4, D=5120, and 1 to 72 local rows"
        if (
            cap.graph_mode.lower() != "full_decode_only"
            or cap.superkernel_enabled
            or spec.sp_enabled
            or spec.is_layer0_entry
            or spec.is_write_kv_path
        ):
            return False, "A2 overlap requires a full-decode-only seam without SP"
        if spec.post_mult != 2.0 or spec.numerics_profile != NumericsProfile.STANDARD_NR1:
            return False, "A2 overlap requires post_mult=2 and the standard numerical profile"
        return True, "qualified 910B overlap seam"

    def prepare(self, weights: dict, device: str, cap: MhcHardwareCapability) -> PreparedMhcWeights:
        return _prepare_canonical(weights, device)

    def run(self, inputs: MhcSeamInputs, prepared: PreparedMhcWeights, attrs: dict) -> MhcSeamOutputs:
        tokens, lanes, width = _validate_inputs(inputs, prepared)
        if (
            prepared.weight_format != WeightFormat.FP32_CANONICAL
            or inputs.y.dtype != torch.bfloat16
            or lanes != 4
            or width != A2_HIDDEN_SIZE
            or tokens > A2_MAX_ROWS
        ):
            raise ValueError("A2 run requires canonical weights and a qualified BF16 overlap shape")
        attrs = dict(attrs)
        wait_for_attention = attrs.pop("wait_for_attention", False)
        if type(wait_for_attention) is not bool:
            raise ValueError("wait_for_attention must be a bool")
        args = _parse_attrs(attrs)
        if args.post_mult != 2.0 or args.numerics_profile != NumericsProfile.STANDARD_NR1:
            raise ValueError("A2 overlap requires post_mult=2 and the standard numerical profile")
        op_args = (
            inputs.y,
            inputs.residual,
            inputs.post,
            inputs.comb,
            inputs.pre_mix,
            inputs.gamma,
            prepared.tensor,
            inputs.hc_scale,
            inputs.hc_base,
            args.iterations,
            args.norm_eps,
            args.hc_eps,
        )
        if args.return_v_fp32:
            return MhcSeamOutputs(*torch.ops.vllm.dsv41_mhc_moe_input(*op_args, wait_for_attention))
        residual, v, post, comb, pre = torch.ops.vllm.dsv41_mhc_attention_input(*op_args)
        return MhcSeamOutputs(residual, v, None, post, comb, pre)


class NativeFallbackMhcBackend:
    def supports(self, spec: MhcSeamSpec, cap: MhcHardwareCapability) -> tuple[bool, str]:
        if spec.dtype not in ACTIVATION_DTYPES or spec.dtype not in cap.dtype_support:
            return False, "device does not support the activation dtype"
        return True, "canonical split seam or native entry path"

    def prepare(self, weights: dict, device: str, cap: MhcHardwareCapability) -> PreparedMhcWeights:
        return _prepare_canonical(weights, device)

    def run(self, inputs: MhcSeamInputs, prepared: PreparedMhcWeights, attrs: dict) -> MhcSeamOutputs:
        _, lanes, _ = _validate_inputs(inputs, prepared)
        args = _parse_attrs(attrs)
        if prepared.weight_format != WeightFormat.FP32_CANONICAL:
            raise ValueError("The split seam requires canonical FP32 hc_fn")
        if inputs.device.type == "npu" and inputs.y.dtype == torch.bfloat16 and lanes == 4:
            ops = torch.ops._C_ascend
            updated = ops.npu_hc_post(
                *(tensor.unsqueeze(0) for tensor in (inputs.y, inputs.residual, inputs.post, inputs.comb))
            ).squeeze(0)
            collapsed, post, comb, pre = ops.npu_hc_pre_v3(
                updated,
                prepared.tensor,
                inputs.hc_scale,
                inputs.hc_base,
                inputs.pre_mix,
                hc_mult=lanes,
                hc_sinkhorn_iters=args.iterations,
                norm_eps=args.norm_eps,
                hc_eps=args.hc_eps,
            )
            if args.post_mult != 2.0:
                post = post * (args.post_mult / 2.0)
            if args.return_v_fp32:
                if hasattr(ops, "npu_rms_norm_cast"):
                    v, v_fp32 = ops.npu_rms_norm_cast(collapsed, inputs.gamma, args.norm_eps)
                else:
                    v, _ = torch.ops.npu.npu_rms_norm(collapsed, inputs.gamma, args.norm_eps)
                    v_fp32 = v.bfloat16().float()
            else:
                v, _ = torch.ops.npu.npu_rms_norm(collapsed, inputs.gamma, args.norm_eps)
                v_fp32 = None
            return MhcSeamOutputs(updated, v, v_fp32, post, comb, pre)
        return _torch_split_seam(inputs, prepared.tensor, args)


def _torch_split_seam(inputs: MhcSeamInputs, hc_fn: torch.Tensor, args: _SeamAttrs) -> MhcSeamOutputs:
    updated = inputs.y.float().unsqueeze(1) * inputs.post.unsqueeze(-1)
    updated = updated + torch.bmm(inputs.comb.transpose(1, 2).float(), inputs.residual.float())
    updated = updated.to(inputs.y.dtype)

    collapsed = (updated.float() * inputs.pre_mix.unsqueeze(-1)).sum(dim=1).to(inputs.y.dtype).float()
    denom = (collapsed.square().mean(-1, keepdim=True) + args.norm_eps).sqrt()
    v = ((collapsed * denom.reciprocal()) * inputs.gamma.float()).to(inputs.y.dtype)
    v_fp32 = v.bfloat16().float() if args.return_v_fp32 else None
    post, comb, pre = coefficients(
        updated,
        hc_fn,
        inputs.hc_scale,
        inputs.hc_base,
        norm_eps=args.norm_eps,
        hc_eps=args.hc_eps,
        iterations=args.iterations,
        post_mult=args.post_mult,
    )
    return MhcSeamOutputs(updated, v, v_fp32, post, comb, pre)


BACKENDS: tuple[MhcBackend, ...] = (
    A5NativeMhcBackend(),
    A3DecoupledMhcBackend(),
    A3MegaMhcBackend(),
    A2OverlapMhcBackend(),
    NativeFallbackMhcBackend(),
)


def select_backend(spec: MhcSeamSpec, cap: MhcHardwareCapability) -> MhcBackend:
    if min(spec.C, spec.D) <= 0 or (spec.T is not None and spec.T <= 0):
        raise ValueError("C, D and specified T must be positive")
    if (
        spec.dtype not in ACTIVATION_DTYPES
        or not isinstance(spec.weight_format, WeightFormat)
        or not isinstance(spec.numerics_profile, NumericsProfile)
    ):
        raise ValueError("Invalid activation dtype, weight format, or numerical profile")
    for backend in BACKENDS:
        supported, reason = backend.supports(spec, cap)
        logger.debug("%s for %s: %s", type(backend).__name__, cap.soc_name, reason)
        if supported:
            return backend
    raise ValueError(f"No mHC backend supports {spec.dtype} on {cap.soc_name}")


def should_prefer_side_stream(
    fused_seam_us: float,
    main_chain_us: float,
    downstream_us: float,
    coeff_us: float,
    sync_overhead_us: float,
    downstream_contention_us: float,
    coeff_contention_us: float = 0.0,
) -> bool:
    """Return whether side-stream overlap shortens the joined critical path.

    Compares (fused_seam + downstream) against
    (main_chain + sync_overhead + max(downstream + downstream_contention, coeff + coeff_contention)).
    """
    joined_side_stream = sync_overhead_us + max(
        downstream_us + downstream_contention_us,
        coeff_us + coeff_contention_us,
    )
    return joined_side_stream - downstream_us < fused_seam_us - main_chain_us
