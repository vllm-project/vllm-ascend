# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from functools import wraps

import torch

from .transfer import TorchExpertTransfer
from .utils import ExpertTensorSpec, PreparedExpertTensor

ACL_MEMCPY_DEVICE_TO_HOST = 2
EXPERTS_PER_HOST_CHUNK = 32


class HostExpertStore:
    """Owned checkpoint tensors and prepared physical bytes; preserve loader semantics."""

    def __init__(self, layer, names: tuple[str, ...]):
        self.names = names
        self.num_experts = layer.local_num_experts
        self.specs = tuple(
            ExpertTensorSpec(name, tuple(getattr(layer, name).shape), getattr(layer, name).dtype) for name in names
        )
        self.loaded: dict[str, set[tuple[int, str]]] = {name: set() for name in names}
        self.tensors: dict[str, torch.Tensor] = {}
        self.prepared: dict[str, PreparedExpertTensor] = {}
        self.prototypes: dict[str, torch.Tensor] = {}
        self.finalized = False
        original_loader = layer.weight_loader
        parameter_names = {id(getattr(layer, name)): name for name in names}

        @wraps(original_loader)
        def tracked_loader(param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
            if self.finalized:
                raise RuntimeError("expert offload does not support online weight reload")
            # Ask the original loader whether it accepted the tensor, even when
            # the caller does not need that return value.
            success = original_loader(param, loaded_weight, weight_name, shard_id, expert_id, return_success=True)
            if success:
                name = parameter_names[id(param)]
                experts = range(self.num_experts) if loaded_weight.ndim == 3 else (expert_id,)
                shards: tuple[str, ...] = (shard_id,)
                if (
                    name == "w13_weight_scale"
                    and loaded_weight.ndim >= 2
                    and loaded_weight.shape[-2] == param.shape[-2]
                ):
                    shards = ("w1", "w3")
                self.loaded[name].update((eid, shard) for eid in experts for shard in shards)
            return success if return_success else None

        layer.weight_loader = tracked_loader
        for name in names:
            getattr(layer, name).weight_loader = tracked_loader

    def finalize(self, layer, *, pin_memory: bool = True) -> None:
        if self.finalized:
            raise RuntimeError("expert host store can only finalize once")
        for spec in self.specs:
            shards = ("w1", "w3") if spec.name.startswith("w13") else ("w2",)
            expected = {(eid, shard) for eid in range(self.num_experts) for shard in shards}
            if not expected.issubset(self.loaded[spec.name]):
                raise RuntimeError(f"incomplete expert checkpoint: {layer.layer_name}.{spec.name}")
            tensor = getattr(layer, spec.name).detach()
            if tensor.device.type != "cpu" or tuple(tensor.shape) != spec.shape or tensor.dtype != spec.dtype:
                raise ValueError(f"unexpected raw expert tensor layout for {spec.name}")
            # Registered CPU Parameters already own loader copies. Pinning
            # creates a new owned allocation, independent of checkpoint mmap.
            self.tensors[spec.name] = tensor.pin_memory() if pin_memory else tensor.clone()
        self.finalized = True

    def prepare_native(self, adapter, destination, device) -> None:
        """Convert each expert once, retaining its complete physical storage on CPU.

        Only one conversion bundle is live on the NPU. Physical copies bypass
        ND interpretation of NZ padding and preserve packed scale strides.
        Raw checkpoint tensors are retired after the complete layer succeeds.
        """
        import acl  # type: ignore[import-not-found]
        import torch_npu

        if not self.finalized or self.prepared:
            raise RuntimeError("native expert preparation requires a finalized raw host store")
        prepared = {}
        for expert_id in range(self.num_experts):
            bundle = adapter.materialize(self, (expert_id,), device)
            if set(bundle) != set(adapter.output_names) or set(destination) != set(adapter.output_names):
                raise RuntimeError("incomplete expert preparation bundle")
            # Loading only: complete format conversion before reading its storage.
            torch.npu.current_stream(device).synchronize()
            for name, tensor in bundle.items():
                _, _, size = TorchExpertTransfer._physical_copy_parameters(destination[name], tensor, 0, torch_npu)
                if expert_id == 0:
                    # The host allocator rounds large pinned allocations up to a
                    # power of two. Splitting 288 experts into 32-expert chunks
                    # avoids reserving space for 512 experts in every GLM layer.
                    chunks = tuple(
                        torch.empty(
                            (min(EXPERTS_PER_HOST_CHUNK, self.num_experts - start), size),
                            dtype=torch.uint8,
                            device="cpu",
                            pin_memory=True,
                        )
                        for start in range(0, self.num_experts, EXPERTS_PER_HOST_CHUNK)
                    )
                    prepared[name] = PreparedExpertTensor(
                        data=chunks[0],
                        shape=tuple(tensor.shape[1:]),
                        stride=tuple(tensor.stride()),
                        dtype=tensor.dtype,
                        device=tensor.device,
                        npu_format=torch_npu.get_npu_format(tensor),
                        expert_bytes=size,
                        extra_data=chunks[1:],
                    )
                elif size != prepared[name].expert_bytes:
                    raise ValueError("expert physical layout changed during preparation")
                # ACL reads the storage as bytes, never interpreting NZ as ND.
                chunk, row = prepared[name].source_chunk(expert_id)
                error = acl.rt.memcpy(
                    chunk[row].data_ptr(),
                    size,
                    tensor.untyped_storage().data_ptr(),
                    size,
                    ACL_MEMCPY_DEVICE_TO_HOST,
                )
                if error:
                    raise RuntimeError(f"expert preparation device-to-host copy failed: ACL error {error}")
            del tensor, bundle
        # A single raw expert is sufficient to allocate a full pool through the
        # exact quantization scheme, including native semantic dtype metadata.
        self.prototypes = {name: tensor[:1].clone() for name, tensor in self.tensors.items()}
        self.prepared = prepared
        self.tensors.clear()

    def close(self) -> None:
        self.tensors.clear()
        self.prepared.clear()
        self.prototypes.clear()
