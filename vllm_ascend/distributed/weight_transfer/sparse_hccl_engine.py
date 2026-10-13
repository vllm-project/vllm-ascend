# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HCCL transport for vLLM's checkpoint-coordinate sparse updates.

The upstream engines own the sparse loader and public trainer lifecycle. The
default wire order matches sparse_nccl: indices followed by values per patch.
Packed payloads, first-session dense seeding and guarded MoE writes are explicit
extensions; fallback patches always use the upstream checkpoint loader.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from math import prod
from typing import Any, ClassVar, Self

import regex as re
import torch
from vllm.distributed.weight_transfer.base import VLLMWeightSyncClient, WeightSource
from vllm.distributed.weight_transfer.sparse_nccl_engine import (
    SparseNCCLTrainerInitInfo,
    SparseNCCLTrainerWeightTransferEngine,
    SparseNCCLWeightTransferEngine,
    SparseNCCLWeightTransferUpdateInfo,
    SparseWeightPatch,
)
from vllm.model_executor.model_loader.checkpoint_weight_patch import (
    CheckpointWeightPatch,
    load_checkpoint_weight_patches,
)

from vllm_ascend.distributed.weight_transfer.hccl_engine import (
    HCCLWeightTransferEngine,
    HCCLWeightTransferInitInfo,
)
from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import apply_sparse_hf_patches_with_loader

__all__ = [
    "SparseWeightPatch",
    "SparseHCCLTrainerInitInfo",
    "SparseHCCLWeightTransferInitInfo",
    "SparseHCCLWeightTransferUpdateInfo",
    "SparseHCCLWeightTransferEngine",
    "SparseHCCLTrainerWeightTransferEngine",
]


@dataclass
class SparseHCCLWeightTransferInitInfo(HCCLWeightTransferInitInfo):
    """Optional first dense session; subsequent sessions apply sparse patches."""

    dense_seed: bool = False


@dataclass
class SparseHCCLWeightTransferUpdateInfo(SparseNCCLWeightTransferUpdateInfo):
    """Upstream metadata with opt-in Ascend transfer/apply extensions."""

    packed: bool = False
    encoding: str = "indices"
    checksum: int | None = None
    checksum_algorithm: str | None = None
    direct_moe: bool = False
    verify: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.encoding not in {"dense", "indices"}:
            raise ValueError(f"Unsupported sparse HCCL encoding: {self.encoding!r}")
        if self.encoding == "dense" and not self.packed:
            raise ValueError("Dense seed requires packed=True")
        if self.checksum_algorithm not in {None, "torch_hash_v1"}:
            raise ValueError(f"Unsupported checksum algorithm: {self.checksum_algorithm!r}")
        if self.checksum is not None and (not self.packed or len(set(self.dtype_names)) != 1):
            raise ValueError("Checksummed sparse HCCL flushes must be packed and use one dtype")
        if any(not name or any(dimension < 0 for dimension in shape) for name, shape in zip(self.names, self.shapes)):
            raise ValueError("Sparse HCCL names must be non-empty and shapes non-negative")
        if self.encoding == "dense":
            if any(count != prod(shape) for count, shape in zip(self.num_updates_list, self.shapes)):
                raise ValueError("Dense seed size mismatch")


@dataclass
class SparseHCCLTrainerInitInfo(SparseNCCLTrainerInitInfo):
    """Single-sender HCCL rendezvous with the upstream sparse trainer contract."""

    backend: ClassVar[str] = "sparse_hccl"
    packed: bool = False
    direct_moe: bool = False
    dense_seed: bool = False


class SparseHCCLWeightTransferEngine(SparseNCCLWeightTransferEngine):
    """Receive HCCL patches and use the upstream checkpoint loader."""

    init_info_cls = SparseHCCLWeightTransferInitInfo
    update_info_cls = SparseHCCLWeightTransferUpdateInfo

    def update_weights(self, update_info: dict[str, Any]) -> None:
        group = self.model_update_group
        if group is None:
            raise RuntimeError("HCCL weight transfer not initialized. Call init_transfer_engine() first.")
        # The inherited update method synchronizes after receive returns. Keep
        # that synchronization on the communicator, not an RPC thread's device.
        with torch.npu.device(group.device):
            super().update_weights(update_info)

    def init_transfer_engine(self, init_info: HCCLWeightTransferInitInfo) -> None:
        # Reuse the existing HCCL worker-rank calculation and initialization.
        self.shutdown()
        HCCLWeightTransferEngine.init_transfer_engine(self, init_info)
        self._dense_seed_pending = getattr(init_info, "dense_seed", False)
        self._dense_received = False

    def start_weight_update(self) -> None:
        if getattr(self, "_dense_seed_pending", False):
            HCCLWeightTransferEngine.start_weight_update(self)
        else:
            super().start_weight_update()

    def finish_weight_update(self) -> None:
        if getattr(self, "_dense_seed_pending", False):
            HCCLWeightTransferEngine.finish_weight_update(self)
            self._dense_seed_pending = not self._dense_received
        else:
            super().finish_weight_update()

    @torch.no_grad()
    def receive_weights(self, update_info: SparseHCCLWeightTransferUpdateInfo) -> None:
        group = self.model_update_group
        if group is None:
            raise RuntimeError("HCCL weight transfer not initialized. Call init_transfer_engine() first.")
        device = group.device
        with torch.npu.device(device):
            if update_info.encoding == "dense" and not getattr(self, "_dense_seed_pending", False):
                raise ValueError("Dense seed requires dense_seed=True at initialization")
            if update_info.encoding == "indices" and getattr(self, "_dense_seed_pending", False):
                raise ValueError("Finish the initial dense session before sending sparse patches")
            if update_info.packed:
                self._receive_packed_weights(update_info)
                return
            stream = torch.npu.current_stream(device)
            patches = []
            for name, dtype_name, shape, count in zip(
                update_info.names,
                update_info.dtype_names,
                update_info.shapes,
                update_info.num_updates_list,
                strict=True,
            ):
                dtype = getattr(torch, dtype_name)
                indices = torch.empty(count, dtype=torch.int32, device=device)
                values = torch.empty(count, dtype=dtype, device=device)
                group.broadcast(indices, src=0, stream=stream)
                group.broadcast(values, src=0, stream=stream)
                patches.append(CheckpointWeightPatch(name, tuple(shape), dtype, values, indices))
            if update_info.direct_moe or update_info.verify:
                from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import (
                    apply_sparse_hf_patches_with_loader,
                )

                pairs = [(SparseWeightPatch(p.name, p.indices, p.values, p.shape), list(p.shape)) for p in patches]
                apply_sparse_hf_patches_with_loader(
                    self.model, pairs, direct_moe=update_info.direct_moe, verify=update_info.verify
                )
            else:
                self._apply_checkpoint_patches(patches)

    def _receive_packed_weights(
        self,
        update_info: SparseHCCLWeightTransferUpdateInfo,
    ) -> None:
        """Receive sparse flat-index patches and apply them in place."""
        if self.model_update_group is None:
            raise RuntimeError("HCCL weight transfer not initialized. Call init_transfer_engine() first.")

        num_updates_list = update_info.num_updates_list
        total_updates = sum(num_updates_list)
        dense = update_info.encoding == "dense"
        packed_indices = torch.empty(
            0 if dense else total_updates,
            dtype=torch.int32,
            device=self.model_update_group.device,
        )
        if not dense and packed_indices.numel():
            self.model_update_group.broadcast(
                packed_indices, src=0, stream=torch.npu.current_stream(self.model_update_group.device)
            )

        # Values cannot be concatenated across different dtypes.  Keep one
        # packed buffer per dtype instead, preserving first-seen dtype order so
        # trainer and workers issue collectives in exactly the same sequence.
        dtype_order = list(dict.fromkeys(update_info.dtype_names))
        packed_values: dict[str, torch.Tensor] = {}
        for dtype_name in dtype_order:
            dtype_updates = sum(
                num_updates
                for patch_dtype, num_updates in zip(
                    update_info.dtype_names,
                    num_updates_list,
                    strict=True,
                )
                if patch_dtype == dtype_name
            )
            values = torch.empty(
                dtype_updates,
                dtype=getattr(torch, dtype_name),
                device=self.model_update_group.device,
            )
            if values.numel():
                self.model_update_group.broadcast(
                    values, src=0, stream=torch.npu.current_stream(self.model_update_group.device)
                )
            packed_values[dtype_name] = values

        if update_info.checksum is not None:
            if len(dtype_order) != 1:
                raise ValueError("Checksummed sparse HCCL flushes must use one dtype")
            positions = packed_indices.contiguous().view(torch.uint8)
            values = packed_values[dtype_order[0]]
            got = _delta_checksum(
                positions,
                values,
                algorithm=update_info.checksum_algorithm or "torch_hash_v1",
            )
            if got != update_info.checksum:
                raise RuntimeError(f"Sparse HCCL checksum mismatch: got {got}, expected {update_info.checksum}")

        if dense:
            self._dense_received = True
            value_offsets = dict.fromkeys(dtype_order, 0)
            named_tensors = []
            for name, dtype_name, expected_shape, num_updates in zip(
                update_info.names,
                update_info.dtype_names,
                update_info.shapes,
                num_updates_list,
                strict=True,
            ):
                expected_numel = 1
                for dimension in expected_shape:
                    expected_numel *= dimension
                if num_updates != expected_numel:
                    raise ValueError(
                        f"Dense seed size mismatch for {name}: got {num_updates}, expected {expected_numel}"
                    )
                value_offset = value_offsets[dtype_name]
                tensor = packed_values[dtype_name][value_offset : value_offset + num_updates].view(expected_shape)
                value_offsets[dtype_name] = value_offset + num_updates
                named_tensors.append((name, tensor))
            if update_info.verify:
                _verify_dense_load(self.model, named_tensors)
            else:
                load_checkpoint_weight_patches(
                    self.model,
                    [
                        CheckpointWeightPatch(name, tuple(tensor.shape), tensor.dtype, tensor.view(-1))
                        for name, tensor in named_tensors
                    ],
                )
            return

        index_offset = 0
        value_offsets = dict.fromkeys(dtype_order, 0)
        patches: list[tuple[SparseWeightPatch, list[int]]] = []
        for name, dtype_name, expected_shape, num_updates in zip(
            update_info.names,
            update_info.dtype_names,
            update_info.shapes,
            num_updates_list,
            strict=True,
        ):
            indices = packed_indices[index_offset : index_offset + num_updates]
            value_offset = value_offsets[dtype_name]
            values = packed_values[dtype_name][value_offset : value_offset + num_updates]
            index_offset += num_updates
            value_offsets[dtype_name] = value_offset + num_updates
            if _is_pipeline_missing_weight(self.model, name):
                continue
            patches.append(
                (
                    SparseWeightPatch(
                        name=name,
                        indices=indices,
                        values=values,
                        full_shape=tuple(expected_shape),
                    ),
                    expected_shape,
                )
            )
        apply_sparse_hf_patches_with_loader(
            self.model,
            patches,
            direct_moe=update_info.direct_moe,
            verify=update_info.verify,
        )
        if update_info.verify:
            logging.getLogger(__name__).info("SPARSE-VERIFY local_patches=%d mismatch_elems=0", len(patches))

    def _apply_checkpoint_patches(self, patches: list[CheckpointWeightPatch]) -> None:
        """Keep the transport independent of accelerator-specific loader extensions."""
        load_checkpoint_weight_patches(self.model, patches)

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            self.model_update_group.close()
            self.model_update_group = None


class SparseHCCLTrainerWeightTransferEngine(SparseNCCLTrainerWeightTransferEngine):
    """Inherit one-shot/chunk lifecycle and patch validation; transmit on NPU."""

    init_info_cls = SparseHCCLTrainerInitInfo

    @classmethod
    def trainer_init(
        cls,
        init_info: SparseHCCLTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> Self:
        engine = cls(client=client, source=source, is_sender=init_info.is_sender)
        engine._packed = init_info.packed
        engine._direct_moe = init_info.direct_moe
        if not engine.is_sender:
            return engine
        worker_info = SparseHCCLWeightTransferInitInfo(
            master_address=init_info.master_address,
            master_port=init_info.master_port,
            rank_offset=1,
            world_size=init_info.world_size,
            dense_seed=init_info.dense_seed,
        )
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(client.init_weight_transfer_engine, asdict(worker_info))
            if future.done():
                future.result()
            engine.model_update_group = HCCLWeightTransferEngine.trainer_init(worker_info)
            future.result()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return engine

    def _prepare_patches(self, patches) -> list[SparseWeightPatch]:
        prepared = super()._prepare_patches(patches)
        if not prepared:
            return prepared
        group = self.model_update_group
        assert group is not None
        device = group.device
        # HCCL reads tensor storage directly. Reject a mismatched device or
        # strided payload before starting a one-shot update session.
        for patch in prepared:
            for tensor in (patch.indices, patch.values):
                if tensor.device != device or not tensor.is_contiguous():
                    raise ValueError(f"{patch.name}: sparse HCCL buffers must be contiguous on {device}")
        return prepared

    def _broadcast_chunk(self, patches: list[SparseWeightPatch]) -> None:
        if getattr(self, "_packed", False):
            self._broadcast_packed_patches(patches)
            return
        group = self.model_update_group
        assert group is not None
        device = group.device
        with torch.npu.device(device):
            update_info = SparseHCCLWeightTransferUpdateInfo(
                names=[patch.name for patch in patches],
                dtype_names=[str(patch.values.dtype).split(".")[-1] for patch in patches],
                shapes=[list(patch.full_shape) for patch in patches],
                num_updates_list=[patch.indices.numel() for patch in patches],
                direct_moe=getattr(self, "_direct_moe", False),
            )
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(self.client.update_weights, asdict(update_info))
                if future.done():
                    future.result()
                stream = torch.npu.current_stream(device)
                for patch in patches:
                    group.broadcast(patch.indices, src=0, stream=stream)
                    group.broadcast(patch.values, src=0, stream=stream)
                future.result()
                self._post_send_sync()
            finally:
                # A broadcast error can leave the matching worker RPC blocked.
                executor.shutdown(wait=False, cancel_futures=True)

    def _post_send_sync(self) -> None:
        assert self.model_update_group is not None
        torch.npu.current_stream(self.model_update_group.device).synchronize()

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            self.model_update_group.close()
            self.model_update_group = None

    def _broadcast_packed_patches(self, patches: list[SparseWeightPatch]) -> None:
        # Keep inherited send_weights/send_weight_chunk usable with this wire
        # protocol too. Consecutive dtype runs preserve ordered overwrites.
        runs = []
        for patch in patches:
            if not runs or runs[-1][0].values.dtype != patch.values.dtype:
                runs.append([])
            runs[-1].append(patch)
        for run in runs:
            positions = torch.cat([patch.indices for patch in run])
            values = torch.cat([patch.values for patch in run])
            info = SparseHCCLWeightTransferUpdateInfo(
                names=[patch.name for patch in run],
                dtype_names=[str(patch.values.dtype).split(".")[-1] for patch in run],
                shapes=[list(patch.full_shape) for patch in run],
                num_updates_list=[patch.indices.numel() for patch in run],
                packed=True,
                direct_moe=getattr(self, "_direct_moe", False),
            )
            self.send_packed_chunk(info, positions, values)

    def send_packed_chunk(self, update_info, positions, values) -> None:
        if not self.is_sender:
            return
        if not update_info.packed:
            raise ValueError("send_packed_chunk requires packed=True")
        group = self.model_update_group
        if group is None:
            raise RuntimeError("trainer_init() must be called before sending weights")
        if positions.dtype != torch.int32 or positions.ndim != 1 or values.ndim != 1:
            raise ValueError("Packed delta requires flat int32 positions and flat values")
        if positions.device != group.device or values.device != group.device:
            raise ValueError("Packed HCCL buffers must reside on the communicator NPU")
        if not positions.is_contiguous() or not values.is_contiguous():
            raise ValueError("Packed HCCL buffers must be contiguous")
        expected = sum(update_info.num_updates_list)
        if values.numel() != expected or positions.numel() != (0 if update_info.encoding == "dense" else expected):
            raise ValueError("Packed HCCL buffers disagree with update metadata")
        if len(set(update_info.dtype_names)) != 1:
            raise ValueError("Preassembled packed flushes must use one dtype")
        if str(values.dtype).split(".")[-1] != update_info.dtype_names[0]:
            raise ValueError("Packed HCCL value dtype disagrees with metadata")
        with torch.npu.device(group.device):
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(self.client.update_weights, asdict(update_info))
                if future.done():
                    future.result()
                stream = torch.npu.current_stream(group.device)
                if positions.numel():
                    group.broadcast(positions, src=0, stream=stream)
                if values.numel():
                    group.broadcast(values, src=0, stream=stream)
                future.result()
                self._post_send_sync()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)


def _delta_checksum(
    positions: torch.Tensor,
    values: torch.Tensor,
    *,
    algorithm: str = "torch_hash_v1",
) -> int:
    """Compute the negotiated payload checksum."""
    if algorithm != "torch_hash_v1":
        raise ValueError(f"Unsupported checksum algorithm: {algorithm!r}")
    p = int(torch.hash_tensor(positions).item()) if positions.numel() else 0
    v = int(torch.hash_tensor(values).item()) if values.numel() else 0
    return (p ^ (v << 1)) & ((1 << 64) - 1)


def _verify_dense_load(
    model: torch.nn.Module,
    named_tensors: list[tuple[str, torch.Tensor]],
) -> None:
    """Replay dense trainer state and require the live state to be unchanged."""
    original_copy = torch.Tensor.copy_
    mismatches: dict[str, int] = {}
    skipped_nonlocal_experts = 0
    skipped_nonlocal_pipeline = 0
    verified_params = 0
    for name, tensor in named_tensors:
        touched: dict[tuple[int, int, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

        def snapshot_then_copy(destination, source, *args, _touched=touched, **kwargs):
            key = (destination.data_ptr(), destination.numel(), destination.dtype)
            if key not in _touched:
                _touched[key] = (destination, destination.detach().clone())
            return original_copy(destination, source, *args, **kwargs)

        torch.Tensor.copy_ = snapshot_then_copy
        try:
            model.load_weights([(name, tensor)])
        finally:
            torch.Tensor.copy_ = original_copy
        if not touched and _is_pipeline_missing_weight(model, name):
            skipped_nonlocal_pipeline += 1
            continue
        if not touched and ".experts." in name:
            skipped_nonlocal_experts += 1
            continue
        if not touched:
            raise RuntimeError(f"Dense verification did not touch any weight for {name}")
        verified_params += 1
        mismatch_count = 0
        for destination, previous in touched.values():
            if destination.is_floating_point() and destination.element_size() == 2:
                mismatch_count += int((destination.view(torch.int16) != previous.view(torch.int16)).sum())
            else:
                mismatch_count += int((destination != previous).sum())
        if mismatch_count:
            mismatches[name] = mismatch_count
    if mismatches:
        offenders = sorted(mismatches.items(), key=lambda item: -item[1])[:12]
        raise RuntimeError(
            f"Sparse HCCL delta state verification failed: {sum(mismatches.values())} elements differ; "
            f"top offenders: {offenders}"
        )

    logging.getLogger(__name__).warning(
        "DELTA-VERIFY params=%d skipped_nonlocal_experts=%d skipped_nonlocal_pipeline=%d mismatch_elems=0",
        verified_params,
        skipped_nonlocal_experts,
        skipped_nonlocal_pipeline,
    )


_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _is_pipeline_missing_weight(model: torch.nn.Module, name: str) -> bool:
    """Return whether an HF-global weight belongs to another PP stage."""
    core = getattr(model, "model", model)
    match = _LAYER_INDEX_RE.search(name)
    if match is not None:
        start_layer = getattr(core, "start_layer", None)
        end_layer = getattr(core, "end_layer", None)
        if start_layer is not None and end_layer is not None:
            layer_index = int(match.group(1))
            return not (start_layer <= layer_index < end_layer)

    if "embed_tokens" in name:
        return type(getattr(core, "embed_tokens", None)).__name__ == "PPMissingLayer"
    if name.startswith("lm_head.") or ".lm_head." in name:
        return type(getattr(model, "lm_head", None)).__name__ == "PPMissingLayer"
    return False
