"""Grouped native KVTC quantization with compact FP16 metadata.

INT4 unpack uses NPU AntiQuant; no tensor shifts or CPU fallback. Wire format
is unchanged: packed payloads plus (q + offset) * scale metadata per token/group.
"""

from __future__ import annotations

import numpy as np
import torch

from .quant_layout import INT_TYPES, QuantLayout

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int4": torch.int32,
}


def require_npu_quantization():
    import torch_npu

    for name in ("npu_dynamic_quant_asymmetric", "npu_anti_quant"):
        if not hasattr(torch_npu, name):
            raise RuntimeError(f"Installed torch_npu is missing {name}")
    if not hasattr(torch, "quint4x2"):
        raise RuntimeError("Installed torch is missing quint4x2")
    return torch_npu


def allocate_quant_buffers(layout: QuantLayout, blocks: int, *, device, pin_memory=False):
    if blocks < 0:
        raise ValueError("blocks must be nonnegative")
    result = {
        dtype: torch.empty((blocks, count), dtype=DTYPES[dtype], device=device, pin_memory=pin_memory)
        for dtype, count in layout.payload_elements
    }
    if layout.metadata_count:
        shape = (blocks, layout.block_size, layout.metadata_count)
        for name in ("scales", "offsets"):
            result[name] = torch.empty(shape, dtype=torch.float16, device=device, pin_memory=pin_memory)
    return result


def buffer_prefix(buffers, blocks):
    return {name: tensor[:blocks] for name, tensor in buffers.items()}


def _validate_buffers(buffers, layout, blocks, device):
    shapes = {t: ((blocks, count), DTYPES[t]) for t, count in layout.payload_elements}
    if layout.metadata_count:
        for name in ("scales", "offsets"):
            shapes[name] = (
                (blocks, layout.block_size, layout.metadata_count),
                torch.float16,
            )
    if set(buffers) != set(shapes):
        raise ValueError(f"Payload fields differ: expected {tuple(shapes)}")
    for name, (shape, dtype) in shapes.items():
        tensor = buffers[name]
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"Invalid {name} payload; expected {shape}, {dtype}, {device}, contiguous")


def _repair_constant_rows(values, q, scale, offset):
    """Repair native zero-range rows before FP16 metadata conversion.

    Native offset uses q=round(x/scale+offset); the wire negates it.
    Exact equality is deliberate: near-constant rows must not be collapsed.
    Device-only operations; no .item(), CPU copy or host branch on tensors.
    Constant magnitudes must fit FP16, as required by scale=1 wire encoding.
    This is not a general fix for FP16 metadata range loss on other rows.
    """
    first = values[:, :1]
    constant = (values == first).all(dim=-1)
    shape = q.shape
    q = q.reshape(values.shape[0], -1).masked_fill(constant[:, None], 0)
    scale = torch.where(constant, 1.0, scale.reshape(-1).float())
    offset = torch.where(constant, -first.reshape(-1).float(), offset.reshape(-1).float())
    return q.reshape(shape), scale, offset


class QuantCodec:
    """One grouped codec per immutable K or V layout, bound to one NPU."""

    def __init__(self, layout: QuantLayout, device):
        self.layout = layout
        self._buckets = tuple(tuple(layout.groups[i] for i in ids) for ids in layout.buckets)
        self.device = torch.device(device)
        if self.device.type != "npu":
            raise ValueError("QuantCodec requires an NPU device")
        if self.device.index is None:
            self.device = torch.device("npu", torch.npu.current_device())
        self.native = require_npu_quantization()
        self.unit_scale = torch.ones(1, dtype=torch.float32, device=self.device)

    @torch.inference_mode()
    def quantize(self, x, *, out=None):
        self._check_input(x)
        n = x.shape[0]
        if out is None:
            out = allocate_quant_buffers(self.layout, n, device=self.device)
        _validate_buffers(out, self.layout, n, self.device)
        if n:
            self._quantize_grouped(x, out)
        return out

    @torch.inference_mode()
    def dequantize(self, buffers, *, out=None):
        n = next(iter(buffers.values())).shape[0]
        layout = self.layout
        _validate_buffers(buffers, layout, n, self.device)
        if out is None:
            out = torch.empty((n, layout.block_size, layout.gemm_dim), device=self.device, dtype=torch.bfloat16)
        self._check_input(out)
        if out.shape[0] != n:
            raise ValueError("Incorrect output batch size")
        if n:
            self._dequantize_grouped(buffers, out)
        return out

    def validate_native_unpack(self):
        """Small startup-only hardware check: all signed nibbles and bit 31.

        Synchronizes only at initialization. Failure must not silently enable
        the previous tensor-shift path, which can fall back to CPU.
        """
        if "int4" not in dict(self.layout.payload_elements):
            return
        codes = np.array([[(row + lane) % 16 - 8 for lane in range(8)] for row in range(16)], dtype=np.int32)
        words = ((codes.astype(np.int64) & 15) << (np.arange(8) * 4)).sum(-1).astype(np.int32)
        inp = torch.from_numpy(words.copy()).reshape(1, -1).to(self.device)
        actual = self._unpack_native(inp).cpu()
        expected = torch.from_numpy(codes).reshape(1, -1).bfloat16()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def _unpack_native(self, packed):
        # Native INT4 has a known nonzero-storage-offset hazard. Full worker
        # staging tensors and their prefix slices have offset 0.
        if packed.storage_offset() != 0 or not packed.is_contiguous():
            packed = packed.clone(memory_format=torch.contiguous_format)
        return self.native.npu_anti_quant(packed, self.unit_scale, dst_dtype=torch.bfloat16, src_dtype=torch.quint4x2)

    def _check_input(self, x):
        layout = self.layout
        if (
            x.ndim != 3
            or tuple(x.shape[1:]) != (layout.block_size, layout.gemm_dim)
            or x.dtype != torch.bfloat16
            or x.device != self.device
            or not x.is_contiguous()
        ):
            raise ValueError("QuantCodec expects contiguous BF16 [S,B,C_padded] on its device")

    def _quantize_grouped(self, x, out):
        """Quantize equal-width/type groups together; repair constant rows."""
        n, b = x.shape[0], self.layout.block_size
        for groups in self._buckets:
            dtype, width = groups[0].dtype, groups[0].size
            if dtype not in INT_TYPES:
                for g in groups:
                    out[dtype][:, g.payload_start : g.payload_end].copy_(x[:, :, g.start : g.end].reshape(n, -1))
                continue
            parts = [x[:, :, g.start : g.end] for g in groups]
            values = (parts[0].contiguous() if len(parts) == 1 else torch.stack(parts)).reshape(-1, width)
            q, scale, offset = self.native.npu_dynamic_quant_asymmetric(
                values, dst_type=torch.quint4x2 if dtype == "int4" else torch.int8
            )
            q, scale, offset = _repair_constant_rows(values, q, scale, offset)
            words = width // 8 if dtype == "int4" else width
            if q.dtype != DTYPES[dtype] or q.numel() != len(groups) * n * b * words:
                raise RuntimeError("Unexpected native packed quantizer output")
            q = q.reshape(len(groups), n, b * words)
            scale = scale.reshape(len(groups), n, b)
            offset = offset.reshape(len(groups), n, b)
            for j, group in enumerate(groups):
                out[dtype][:, group.payload_start : group.payload_end].copy_(q[j])
                out["scales"][:, :, group.metadata_index].copy_(scale[j])
                out["offsets"][:, :, group.metadata_index].copy_(-offset[j])
        return out

    def _dequantize_grouped(self, buffers, out):
        """Unpack on NPU and apply FP32 affine arithmetic with row broadcasting."""
        layout, n, b = self.layout, out.shape[0], self.layout.block_size
        out[:, :, layout.feature_count :].zero_()
        for groups in self._buckets:
            dtype, width = groups[0].dtype, groups[0].size
            if dtype not in INT_TYPES:
                for g in groups:
                    out[:, :, g.start : g.end].copy_(
                        buffers[dtype][:, g.payload_start : g.payload_end].reshape(n, b, width)
                    )
                continue
            words = width // 8 if dtype == "int4" else width
            pieces = [buffers[dtype][:, g.payload_start : g.payload_end].reshape(n, b, words) for g in groups]
            packed = torch.stack(pieces).reshape(1, -1)
            scales = torch.stack([buffers["scales"][:, :, g.metadata_index] for g in groups]).float().reshape(-1, 1)
            offsets = torch.stack([buffers["offsets"][:, :, g.metadata_index] for g in groups]).float().reshape(-1, 1)
            unpacked = self._unpack_native(packed) if dtype == "int4" else packed
            restored = unpacked.reshape(-1, width).float()
            restored.add_(offsets).mul_(scales)
            restored = restored.bfloat16().reshape(len(groups), n, b, width)
            for j, g in enumerate(groups):
                out[:, :, g.start : g.end].copy_(restored[j])
        return out
