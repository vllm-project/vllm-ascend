# SPDX-License-Identifier: Apache-2.0
"""Ascend runtime storage that is not covered by checkpoint parameter views."""

import torch
import torch_npu
from vllm.model_executor.model_loader.reload import direct as vllm_direct
from vllm.model_executor.model_loader.reload.direct import direct_reload_enabled
from vllm.model_executor.utils import storage_view

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


def get_flatquant_clip_ratio(layer):
    """A captured Kronecker kernel keeps its host scalar, not this attribute.

    The operator has no tensor-valued clip_ratio input. Refuse changes during
    direct reload instead of silently replaying a graph with the old scalar.
    """
    value = layer.clip_ratio.item()
    if direct_reload_enabled() and hasattr(layer, "aclnn_clip_ratio") and value != layer.aclnn_clip_ratio:
        raise ValueError("direct reload cannot change FlatQuant clip_ratio; rebuild the model and recapture graphs")
    return value


def _layout(tensor):
    return tensor.shape, tensor.stride(), tensor.dtype, tensor.device


def refill_expert_list(layer, name, tensors):
    """Keep the list and kernel-visible allocations across post-load processing.

    Unlike refill_derived, incompatible layouts must fail instead of replacing
    captured allocations. Validate every element before copying any of them.
    """
    old = getattr(layer, name, None)
    if old is None or not direct_reload_enabled():
        setattr(layer, name, tensors)
        return
    if len(old) != len(tensors) or any(_layout(a) != _layout(b) for a, b in zip(old, tensors)):
        raise ValueError(f"{name}: expert runtime layout changed; reload the model and recapture graphs")
    for target, source in zip(old, tensors):
        # E8M0 scales are byte encodings, not numeric FP8 values. CANN copy
        # does not support every semantic MX dtype; copy ND scale bytes.
        if target.dtype == torch.float8_e8m0fnu:
            target.view(torch.uint8).copy_(source.view(torch.uint8))
        else:
            target.copy_(source)


def process_quant_weights(scheme, layer):
    """Reuse upstream's per-finish guard; invalidate Ascend's legacy guards.

    direct.finish_direct_reload clears _already_called_process_weights_after_loading
    before each transaction's PWAL. Legacy layerwise restore keeps its own path.
    """
    if direct_reload_enabled():
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        layer.__dict__.pop("_mxfp4_transformed", None)
        layer.__dict__.pop("_mxfp8_transformed", None)
    scheme.process_weights_after_loading(layer)
    if direct_reload_enabled():
        layer._already_called_process_weights_after_loading = True


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
    getattr(layer, name).data = tensor


def refill_nz_weight(layer, name):
    """Cast to a temporary NZ tensor, then refill the captured NZ allocation."""
    # A tensor rebuilt with set_/as_strided can retain the storage's old
    # base-size descriptor. Clone before casting so NZ conversion uses the
    # final runtime shape rather than the checkpoint storage description.
    source = getattr(layer, name).data
    if direct_reload_enabled():
        source = source.clone(memory_format=torch.contiguous_format)
    tensor = torch_npu.npu_format_cast(source, ACL_FORMAT_FRACTAL_NZ)
    refill_runtime_parameter(layer, name, tensor)


# Immutable callable captured once per process; repeated worker/test setup must
# not wrap an already-installed Ascend function or form a recursive chain.
_UPSTREAM_CHECKPOINT_VIEWS = vllm_direct._checkpoint_views


def checkpoint_views(
    model: torch.nn.Module,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(tensor, view of its storage in its checkpoint layout) for every
    tensor post-load processing re-laid out; raises if a storage cannot
    hold its checkpoint layout."""
    if not any(getattr(t, "_direct_reload_staging", None) is not None for _, t in vllm_direct.loadable_tensors(model)):
        return _UPSTREAM_CHECKPOINT_VIEWS(model)
    views: list[tuple[torch.Tensor, torch.Tensor]] = []
    for name, t in vllm_direct.loadable_tensors(model):
        ckpt = getattr(t, "_ckpt", None)
        if ckpt is None:
            continue  # no layout: created by post-load processing, derived
        staging = getattr(t, "_direct_reload_staging", None)
        if staging is not None:
            # A hardware plugin may retain a physical-format graph allocation
            # and provide separate checkpoint-layout storage for its loaders.
            # Never build a logical ND view over an opaque physical format.
            if ckpt[0] is not model._ckpt_owner:
                raise RuntimeError(f"direct reload: {name} staging belongs to another model")
            if vllm_direct._layout(staging) != ckpt[1] or staging.device != t.device:
                raise RuntimeError(f"direct reload: {name} has incompatible checkpoint staging")
            views.append((t, staging))
            continue
        if vllm_direct._layout(t) == ckpt[1]:
            continue
        if ckpt[0] is not model._ckpt_owner:
            # shared from another model (a target lm_head in a draft): this
            # finish would reprocess bytes a draft update does not resend
            raise RuntimeError(
                f"direct reload: {name} is shared from another model and its post-load processing changed its layout"
            )
        shape, stride, dtype, offset = ckpt[1]
        storage = t.untyped_storage()
        if storage.data_ptr() != ckpt[3]:  # moved, e.g. `set_` into a fused buffer
            offset = t.storage_offset()
        span = 0 if 0 in shape else 1 + sum((n - 1) * s for n, s in zip(shape, stride))
        need = (offset + span) * dtype.itemsize
        if need > storage.nbytes():
            raise RuntimeError(
                f"direct reload: {name} storage has {storage.nbytes()} bytes, its checkpoint layout needs {need}"
            )
        views.append((t, storage_view(storage, offset, shape, stride, dtype)))
    return views


def install_direct_reload_patch():
    """Register Ascend checkpoint-view support without editing vLLM files."""
    vllm_direct._checkpoint_views = checkpoint_views


def refill_derived_buffer(module, name, tensor):
    """Refill an owned non-persistent buffer while retaining captured storage.

    Checkpoint parameters and persistent buffers must use their weight loaders.
    Callers provide independent storage on first initialization, including when
    dtype conversion or contiguous() alone would return a checkpoint alias.
    """
    if name not in module._buffers or name not in module._non_persistent_buffers_set:
        raise ValueError(f"{name}: expected a registered non-persistent derived buffer")
    old = getattr(module, name)
    if (
        isinstance(old, torch.Tensor)
        and tensor is not None
        and (old.shape, old.dtype, old.device) == (tensor.shape, tensor.dtype, tensor.device)
    ):
        return old.copy_(tensor)
    if direct_reload_enabled() and old is not None:
        raise ValueError(f"{name}: derived buffer layout changed; rebuild and recapture graphs")
    setattr(module, name, tensor)
    return tensor
