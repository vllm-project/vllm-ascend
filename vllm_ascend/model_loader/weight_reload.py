# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Runtime allocation and checkpoint staging for unquantized BF16 experts."""

from functools import wraps

import torch
import torch_npu
from torch._prims_common import compute_required_storage_length
from vllm.model_executor.model_loader.reload import direct as vllm_direct
from vllm.model_executor.utils import refill_derived

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ, _should_trans_nz

# Keep upstream ND planning and its safety checks unchanged.
_UPSTREAM_CHECKPOINT_VIEWS = vllm_direct._checkpoint_views


def can_borrow_expert_storage(param, runtime):
    ckpt = getattr(param, "_ckpt", None)
    if ckpt is None or runtime.device.type != "npu" or runtime.dtype != torch.bfloat16 or runtime.ndim != 3:
        return False
    if not get_current_hardware_profile().supports(HardwareCapability.EXPERT_RELOAD_STORAGE_ALIAS):
        return False
    if not all(
        hasattr(torch_npu._C, name)
        for name in (
            "_construct_storage_from_data_pointer",
            "_construct_NPU_Tensor_From_Storage_And_Metadata",
        )
    ):
        return False
    shape, stride, dtype, offset = ckpt[1]
    if len(shape) != 3:
        return False
    experts, rows, cols = shape
    # Each aligned expert plane occupies the same byte range in ND and NZ.
    return (
        min(shape) > 0
        and rows % 16 == cols % 16 == 0
        and offset == runtime.storage_offset() == 0
        and dtype == runtime.dtype
        and tuple(stride) == (rows * cols, cols, 1)
        and runtime.is_contiguous()
        and tuple(runtime.shape) == (experts, cols, rows)
        and runtime.untyped_storage().nbytes() == runtime.numel() * runtime.element_size()
        and torch_npu.get_npu_format(runtime) in (ACL_FORMAT_FRACTAL_ND, ACL_FORMAT_FRACTAL_NZ)
    )


def borrow_nd(owner, shape, stride):
    # set_(owner.untyped_storage()) would share its format descriptor. Create a
    # non-owning StorageImpl with its own ND descriptor. The caller holds owner
    # throughout the transaction; an alias must never outlive its allocation.
    storage = owner.untyped_storage()
    desc = torch_npu._C._construct_storage_from_data_pointer(storage.data_ptr(), owner.device, storage.nbytes())
    metadata = dict(
        size=tuple(shape),
        stride=tuple(stride),
        storage_offset=0,
        dtype=owner.dtype,
        npu_format=ACL_FORMAT_FRACTAL_ND,
    )
    return torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(metadata, desc)


def init_expert_views(layer, name):
    param = getattr(layer, name)
    runtime = param.data
    if hasattr(param, "_direct_reload_runtime") or not can_borrow_expert_storage(param, runtime):
        return False
    if torch_npu.get_npu_format(runtime) != ACL_FORMAT_FRACTAL_ND:
        return False
    # Keep a single owning tensor and stable kernel-visible views into it.
    param._direct_reload_runtime = runtime
    param._ascend_borrow_checkpoint = True
    setattr(layer, name + "_list", list(runtime.unbind(0)))
    return True


def refill_expert_planes(param):
    owner = param._direct_reload_runtime
    if param.data_ptr() != owner.data_ptr() or param.untyped_storage()._cdata == owner.untyped_storage()._cdata:
        raise RuntimeError("expert reload requires an active checkpoint view")
    if torch_npu.get_npu_format(owner) == ACL_FORMAT_FRACTAL_ND:
        for target, source in zip(owner.unbind(0), param.data.unbind(0), strict=True):
            target.copy_(source.transpose(0, 1).contiguous())
    else:
        for expert in param.data.unbind(0):
            # The contiguous transpose owns its bytes before this plane is
            # overwritten. TorchNPU performs the format conversion.
            packed = torch_npu.npu_format_cast(expert.T.contiguous().unsqueeze(0), ACL_FORMAT_FRACTAL_NZ)
            n = expert.numel()
            if packed.untyped_storage().nbytes() != n * expert.element_size():
                raise ValueError("unexpected physical size for aligned NZ expert")
            expert.view(-1).copy_(borrow_nd(packed, (n,), (1,)))
            del packed
    param.data = owner


def _checkpoint_buffer(param, *, zero=False):
    """Allocate the recorded ND layout, including any leading storage offset."""
    shape, stride, dtype, offset = param._ckpt[1]
    size = max(offset, compute_required_storage_length(shape, stride, offset))
    allocate = torch.zeros if zero else torch.empty
    return allocate(size, dtype=dtype, device=param.device).as_strided(shape, stride, offset)


class ExpertModuleStaging:
    @staticmethod
    def supports(layer):
        loader = getattr(layer, "weight_loader", None)
        if loader is None or not hasattr(layer, "_map_global_expert_id_to_local_expert_id"):
            return False
        if getattr(layer, "expert_substitution", None) is not None or not layer.moe_config.is_act_and_mul:
            return False
        return all(
            hasattr(p, "_ckpt") and len(p._ckpt[1][0]) == 3 and p.dtype == torch.bfloat16 and p.weight_loader == loader
            for p in (layer.w13_weight, layer.w2_weight)
        )

    def __init__(self, layer, names):
        self.layer = layer
        self.names = tuple(names)
        self.targets = {}
        self.missing = None
        self.writer = None
        self.allocated = set()
        for name in self.names:
            param = getattr(layer, name)
            runtime = getattr(param, "_direct_reload_runtime", None)
            # For list-based kernels the expert list owns the inference data.
            # Keep the registered Parameter and loader, release its old payload.
            self.targets[name] = runtime if runtime is not None else param.new_empty(0)
            param.data = self.targets[name]
            param._ascend_expert_stage = self
            param.weight_loader = self.wrap_loader(name, param.weight_loader)

    def _restore_targets(self):
        for name, target in self.targets.items():
            getattr(self.layer, name).data = target
        self.allocated.clear()

    def begin(self):
        # A fresh full update can repair an interrupted transfer or post-load.
        if self.writer is not None:
            self.writer.synchronize()
        # The loader allocates zeroed checkpoint buffers on demand. Leave only
        # empty placeholders here so upstream padding reset cannot zero the
        # kernel's live NZ allocation before the first shard arrives.
        for name, target in self.targets.items():
            getattr(self.layer, name).data = target.new_empty(0)
        self.allocated.clear()
        self.writer = None
        # Each local expert must receive each of its projections exactly once for
        # completion accounting; resending a projection can overwrite its value.
        self.missing = {
            (name, shard): set(range(getattr(self.layer, name)._ckpt[1][0][0]))
            for name in self.names
            for shard in (("w1", "w3") if name == "w13_weight" else ("w2",))
        }

    def wrap_loader(self, name, loader):
        @wraps(loader)
        def load(param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
            local_id = self.layer._map_global_expert_id_to_local_expert_id(expert_id)
            if local_id == -1:
                return False if return_success else None
            if not self.missing:
                raise RuntimeError("expert checkpoint shard outside an active module reload")
            if (name, shard_id) not in self.missing:
                raise ValueError(f"{name}: unexpected expert shard {shard_id}")
            if param.device.type == "npu":
                writer = torch.npu.current_stream(param.device)
                if self.writer is not None and self.writer != writer:
                    writer.wait_stream(self.writer)
                self.writer = writer
            if name not in self.allocated:
                param.data = _checkpoint_buffer(param, zero=True)
                self.allocated.add(name)
            if self.writer is not None:
                param.data.record_stream(self.writer)
            ok = loader(param, loaded_weight, weight_name, shard_id, expert_id, return_success=True)
            if ok:
                missing = self.missing[name, shard_id]
                if loaded_weight.ndim == 3:
                    missing.clear()
                else:
                    missing.discard(local_id)
                if not any(self.missing.values()):
                    self.flush()
            return ok if return_success else None

        return load

    def flush(self):
        if self.missing is None:
            raise RuntimeError("expert staging is not active")
        if self.writer is not None:
            # Packed transfers may deliver the two parameters on different streams.
            # Their allocations must survive all reads on the final writer stream.
            for name in self.names:
                getattr(self.layer, name).data.record_stream(self.writer)
        self.layer.quant_method._process_expert_weights(self.layer, self.names)
        self._restore_targets()
        self.missing.clear()  # Mark completion only after a successful refill.

    def finish(self):
        if self.missing != {}:
            raise RuntimeError("incomplete expert module reload")
        self.missing = None
        self.writer = None


def _check_runtime_layout(name, old, new):
    if (old.shape, old.stride(), old.dtype, old.device) != (new.shape, new.stride(), new.dtype, new.device):
        raise ValueError(f"{name}: runtime layout changed; reload the model and recapture graphs")
    if old.device.type == "npu" and torch_npu.get_npu_format(old) != torch_npu.get_npu_format(new):
        raise ValueError(f"{name}: physical format changed; reload the model and recapture graphs")


def refill_expert_list(layer, name, tensors):
    """Keep the list and kernel-visible allocations across post-load processing.

    Incompatible layouts must fail instead of replacing
    captured allocations. Validate every element before copying any of them.
    """
    old = getattr(layer, name, None)
    if old is None:
        setattr(layer, name, [tensor.clone() for tensor in tensors])
        return
    if len(old) != len(tensors):
        raise ValueError(f"{name}: expert runtime layout changed; reload the model and recapture graphs")
    for index, (target, source) in enumerate(zip(old, tensors)):
        _check_runtime_layout(f"{name}[{index}]", target, source)
    # Normally the transpose and the kernel's owned expert buffers are disjoint.
    # Snapshot only aliased sources before any write (e.g. a reordered list).
    destinations = {target.untyped_storage().data_ptr() for target in old}
    sources = [t.clone() if t.untyped_storage().data_ptr() in destinations else t for t in tensors]
    for target, source in zip(old, sources):
        target.copy_(source)


def refill_runtime_parameter(layer, name, tensor, *, borrow_checkpoint=False):
    """Keep graph storage stable and choose how the loader receives its ND checkpoint."""
    parameter = getattr(layer, name)
    ckpt = getattr(parameter, "_ckpt", None)
    if borrow_checkpoint and can_borrow_expert_storage(parameter, tensor):
        parameter._ascend_borrow_checkpoint = True
    elif (
        ckpt is not None
        and not hasattr(parameter, "_ascend_expert_stage")
        and not ExpertModuleStaging.supports(layer)
        and not hasattr(parameter, "_direct_reload_staging")
    ):
        parameter._direct_reload_staging = _checkpoint_buffer(parameter)
    # The Parameter survives begin/finish even while its .data points at ND staging.
    old = getattr(parameter, "_direct_reload_runtime", None)
    if old is not None:
        _check_runtime_layout(name, old, tensor)
    parameter.data = refill_derived(parameter, "_direct_reload_runtime", tensor)


def refill_nz_weight(layer, name, *, force: bool = True, borrow_checkpoint: bool = False):
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
    source = source.clone(memory_format=torch.contiguous_format)
    tensor = torch_npu.npu_format_cast(source, ACL_FORMAT_FRACTAL_NZ)
    refill_runtime_parameter(layer, name, tensor, borrow_checkpoint=borrow_checkpoint)


def checkpoint_views(model: torch.nn.Module) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Plan all views before changing the state of any expert module."""
    views = []
    stages = set()
    ordinary = []
    for name, param in vllm_direct.loadable_tensors(model):
        ckpt = getattr(param, "_ckpt", None)
        if ckpt is None:
            continue
        stage = getattr(param, "_ascend_expert_stage", None)
        staging = getattr(param, "_direct_reload_staging", None)
        borrow = getattr(param, "_ascend_borrow_checkpoint", False)
        if stage is None and staging is None and not borrow:
            if vllm_direct._layout(param) != ckpt[1]:
                ordinary.append((name, param))
            continue
        if not vllm_direct._owned(model, param):
            raise RuntimeError(f"direct reload: {name} staging belongs to another model")
        if stage is not None:
            stages.add(stage)
        elif borrow:
            owner = param._direct_reload_runtime
            if param.device != owner.device or not can_borrow_expert_storage(param, owner):
                raise RuntimeError(f"direct reload: {name} cannot borrow runtime storage")
            shape, stride, _, _ = ckpt[1]
            views.append((param, borrow_nd(owner, shape, stride)))
        elif staging is not None:
            if vllm_direct._layout(staging) != ckpt[1] or staging.device != param.device:
                raise RuntimeError(f"direct reload: {name} has incompatible checkpoint staging")
            views.append((param, staging))

    if not views and not stages:
        return _UPSTREAM_CHECKPOINT_VIEWS(model)
    if ordinary:
        # The pinned upstream planner takes a module. Give it only the relocated
        # ordinary tensors, under their original names, without mutating the model.
        subset = torch.nn.Module()
        subset._ckpt_owner = model._ckpt_owner
        for name, param in ordinary:
            parent = subset
            *path, leaf = name.split(".")
            for part in path:
                if not hasattr(parent, part):
                    parent.add_module(part, torch.nn.Module())
                parent = parent.get_submodule(part)
            parent.register_buffer(leaf, param)
        views.extend(_UPSTREAM_CHECKPOINT_VIEWS(subset))
    for stage in stages:
        stage.begin()
    return views


def install_direct_reload_patch():
    """Register Ascend checkpoint-view support without editing vLLM files."""
    vllm_direct._checkpoint_views = checkpoint_views
