# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Runtime allocation and checkpoint staging for unquantized BF16 reload.

Quantized schemes and speculative decoding
are outside the first-stage adaptation and validation scope.
"""

import torch
import torch_npu
from vllm.model_executor.model_loader.reload import direct as vllm_direct
from vllm.model_executor.model_loader.reload.direct import direct_reload_enabled

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, _should_trans_nz

# Keep upstream ND planning and its safety checks unchanged.
_UPSTREAM_CHECKPOINT_VIEWS = vllm_direct._checkpoint_views


def _layout(tensor):
    return tensor.shape, tensor.stride(), tensor.dtype, tensor.device


def refill_expert_list(layer, name, tensors):
    """Own expert slices, retaining their list and allocations across reloads.

    Incompatible layouts must fail instead of replacing
    captured allocations. Validate every element before copying any of them.
    """
    old = getattr(layer, name, None)
    if old is None or not direct_reload_enabled():
        setattr(layer, name, [tensor.clone() for tensor in tensors])
        return
    if len(old) != len(tensors) or any(_layout(a) != _layout(b) for a, b in zip(old, tensors)):
        raise ValueError(f"{name}: expert runtime layout changed; reload the model and recapture graphs")
    for target, source in zip(old, tensors):
        target.copy_(source)


def refill_runtime_parameter(layer, name, tensor):
    """Retain a physical-format runtime tensor across direct reloads.

    Keep the cast result itself: an ND storage_view would lose its physical
    format contract. The checkpoint Parameter and loader metadata stay intact.
    Direct begin writes a separate ND staging tensor: NPU set_(storage) keeps
    the storage's NZ descriptor, so expert-slice loaders cannot treat it as ND.
    PWAL refills the retained runtime allocation before graph replay.
    """
    if not direct_reload_enabled():
        getattr(layer, name).data = tensor
        return
    parameter = getattr(layer, name)
    ckpt = getattr(parameter, "_ckpt", None)
    if ckpt is not None and not hasattr(parameter, "_direct_reload_staging"):
        shape, stride, dtype, offset = ckpt[1]
        span = 0 if 0 in shape else 1 + sum((n - 1) * s for n, s in zip(shape, stride))
        if offset == 0:
            parameter._direct_reload_staging = torch.empty_strided(shape, stride, dtype=dtype, device=parameter.device)
        else:
            storage = torch.empty(offset + span, dtype=dtype, device=parameter.device)
            parameter._direct_reload_staging = storage.as_strided(shape, stride, offset)
    parameter.data = refill_runtime_tensor(layer, name, tensor)


def refill_runtime_tensor(layer, name, tensor):
    """Retain an owned runtime allocation, including non-Module attention impls.

    Store the physical-format tensor independently of public attributes: PWAL
    may temporarily rebind an attribute.
    Checkpoint parameters must use refill_runtime_parameter for ND staging.
    """
    if not direct_reload_enabled():
        return tensor
    buffers = layer.__dict__.setdefault("_reload_runtime_buffers", {})
    old = buffers.get(name)
    if old is None:
        buffers[name] = tensor
    else:
        if _layout(old) != _layout(tensor):
            raise ValueError(f"{name}: runtime layout changed; rebuild and recapture graphs")
        if old.device.type == "npu" and torch_npu.get_npu_format(old) != torch_npu.get_npu_format(tensor):
            raise ValueError(
                f"{name}: physical format changed: "
                f"{torch_npu.get_npu_format(old)} -> {torch_npu.get_npu_format(tensor)}; "
                "rebuild and recapture graphs"
            )
        old.copy_(tensor)
        tensor = old
    return tensor


def refill_nz_weight(layer, name, *, force: bool = True):
    """Refill NZ storage, optionally preserving the configured cast policy.

    Fused kernels require NZ unconditionally. Optional BF16 conversions pass
    force=False so disabled/unsupported casts keep the original ND allocation.
    """
    # A tensor rebuilt with set_/as_strided can retain the storage's old
    # base-size descriptor. Clone before casting so NZ conversion uses the
    # final runtime shape rather than the checkpoint storage description.
    source = getattr(layer, name).data
    if not force and not _should_trans_nz(source):
        return
    if direct_reload_enabled():
        source = source.clone(memory_format=torch.contiguous_format)
    tensor = torch_npu.npu_format_cast(source, ACL_FORMAT_FRACTAL_NZ)
    refill_runtime_parameter(layer, name, tensor)


def checkpoint_views(
    model: torch.nn.Module,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Use Ascend staging for NZ tensors and upstream planning for ND tensors."""
    views: list[tuple[torch.Tensor, torch.Tensor]] = []
    nd_tensors = {}
    for name, t in vllm_direct.loadable_tensors(model):
        staging = getattr(t, "_direct_reload_staging", None)
        if staging is None:
            nd_tensors[name] = t
            continue
        ckpt = t._ckpt
        if ckpt[0] is not model._ckpt_owner:
            raise RuntimeError(f"direct reload: {name} staging belongs to another model")
        if vllm_direct._layout(staging) != ckpt[1] or staging.device != t.device:
            raise RuntimeError(f"direct reload: {name} has incompatible checkpoint staging")
        views.append((t, staging))
    if not views:
        return _UPSTREAM_CHECKPOINT_VIEWS(model)
    # A read-only planning container: keep original names, objects and owner,
    # but exclude NZ tensors whose physical storage cannot be viewed as ND.
    # Persistent buffers are included too; upstream only reads these tensors.
    nd_model = torch.nn.Module()
    nd_model._ckpt_owner = model._ckpt_owner
    nd_model._buffers.update(nd_tensors)
    return _UPSTREAM_CHECKPOINT_VIEWS(nd_model) + views


def install_direct_reload_patch():
    """Register Ascend checkpoint-view support without editing vLLM files."""
    vllm_direct._checkpoint_views = checkpoint_views
