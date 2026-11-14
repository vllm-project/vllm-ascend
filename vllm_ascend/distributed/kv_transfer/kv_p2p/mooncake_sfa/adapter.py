# SPDX-License-Identifier: Apache-2.0
"""Adapt Mooncake V2 transport to the existing SFA offload cache owners.

The manager owns all Host and hot KV. This adapter owns only transfer views,
bounded landing descriptors and the PD-to-decode readiness bookkeeping.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.transfer_targets import MooncakeKVRegistry
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots import (
    copy_sfa_prefill_dest_geometry,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    OFFLOAD_KV_CACHE_TUPLE_LEN,
)

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.metadata import MooncakeSFAConnectorMetadata

_SPARSE_COPY_MAX_BYTES = (1 << 31) - 1


class MooncakeSFAHostKV:
    """Expose manager-owned Host KV and rank-private HBM PD destinations."""

    def __init__(self, manager: Any, kv_caches: dict[str, Any]) -> None:
        self.manager = manager
        if not manager.use_fused_copy_sfa:
            raise ValueError("Mooncake V2 SFA requires LIM + Copy-SFA")
        names = getattr(manager, "offload_layer_names", ())
        components = [
            getattr(manager, name, ()) for name in ("k_caches_cpu", "v_caches_cpu", "topk_buffers_k", "topk_buffers_v")
        ]
        if not names or any(len(values) != len(names) for values in components):
            raise RuntimeError("Mooncake V2 SFA requires registered Host and hot KV")
        layer_ids = [manager._get_offload_layer_id(name) for name in names]
        if layer_ids != list(range(len(names))):
            raise ValueError("Mooncake V2 SFA layer names and cache indices must match")
        main_groups = {
            group_id
            for group_id, group in enumerate(manager.kv_cache_config.kv_cache_groups)
            if set(group.layer_names) & set(names)
        }
        if len(main_groups) != 1:
            raise ValueError("Mooncake V2 SFA requires one main KV cache group")
        self.main_group_id = main_groups.pop()
        self.rank = manager.tp_rank
        self.size = manager.tp_group.world_size
        # Reuse transfer identity/range validation without allocating a DSA pool
        # or creating a process-global allocation registry.
        self.registry = MooncakeKVRegistry()
        self.transfer_caches = dict(kv_caches)
        self.hot_caches = {}
        self.block_size = manager.kv_cache_config.kv_cache_groups[self.main_group_id].kv_cache_spec.block_size
        self.hot_tokens = manager.topk_buffer_size
        for name, layer_id in zip(names, layer_ids):
            if len(kv_caches[name]) != OFFLOAD_KV_CACHE_TUPLE_LEN:
                raise ValueError(f"Mooncake V2 SFA layer {name!r} must expose the SFA six-component cache")
            pair = (manager.k_caches_cpu[layer_id], manager.v_caches_cpu[layer_id])
            for tensor in pair:
                if tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
                    raise ValueError("Mooncake V2 SFA Host KV must be contiguous BF16 CPU views")
                storage = tensor.untyped_storage()
                self.registry.register_cpu_buffer(int(storage.data_ptr()), int(storage.nbytes()))
                self.registry.register_offloaded_tensor(tensor)
            # Keep main-component wire identity unchanged; hot/tail READs use
            # separate rank-private destinations, not extra wire components.
            self.transfer_caches[name] = pair
            hot_pair = (manager.topk_buffers_k[layer_id], manager.topk_buffers_v[layer_id])
            for host, hot in zip(pair, hot_pair):
                if hot.dtype != host.dtype or not hot.is_contiguous() or hot.ndim != 4:
                    raise ValueError("Mooncake V2 SFA hot KV must be contiguous BF16 row buffers")
                if hot.shape[1] != self.hot_tokens + 2 * self.block_size or hot.shape[2:] != host.shape[2:]:
                    raise ValueError("Mooncake V2 SFA hot rows must include the existing two-block circular tail")
            self.hot_caches[name] = hot_pair
        targets = self.registry.create_transfer_targets([list(pair) for pair in self.hot_caches.values()])
        if targets is None:
            raise ValueError("Mooncake V2 SFA hot buffers overlap Host KV")
        self.hot_transfer_targets = dict(zip(self.hot_caches, targets))

    def device_register_regions(self) -> list[tuple[int, int]]:
        """Register rank-private HBM hot/tail storage without changing wire metadata."""
        return list(
            dict.fromkeys(
                (tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
                for pair in self.hot_caches.values()
                for tensor in pair
            )
        )

    def append_prefill_reads(
        self,
        layer_name,
        request,
        remote_bases,
        remote_strides,
        remote_block_bytes,
        local_ptrs,
        peer_ptrs,
        lengths,
        transfer_targets=None,
    ) -> None:
        """Match SFA's dense prefix and two-block tail placement on every D rank."""
        slot = request.copy_sfa_slot
        if slot is None:
            raise ValueError("Mooncake V2 SFA PD is missing its reserved hot row")
        pair = self.hot_caches[layer_name]
        if not 0 <= slot < pair[0].shape[0]:
            raise ValueError("Mooncake V2 SFA PD hot row is out of bounds")
        # Match the SFA scheduler's finished-prefill extent. The scheduler's
        # subsequent last-token replay does not shorten the received KV.
        kv_tokens = min(request.num_computed_tokens + request.num_external_tokens, request.remote_num_prompt_tokens)
        if not 0 <= kv_tokens <= min(request.local_num_prompt_tokens, request.remote_num_prompt_tokens):
            raise ValueError("Mooncake V2 SFA PD prefix exceeds the available producer KV")
        dense, tail_tokens, tail_block = copy_sfa_prefill_dest_geometry(kv_tokens, self.block_size, self.hot_tokens)
        if not dense and not tail_tokens:
            return
        blocks = request.remote_block_ids[self.main_group_id]
        required = (kv_tokens + self.block_size - 1) // self.block_size
        if len(blocks) < required:
            raise ValueError("Mooncake V2 SFA PD producer main blocks do not cover the hot/tail prefix")
        if len(remote_bases) != 2 or len(remote_block_bytes) != 2:
            raise ValueError("Mooncake V2 SFA PD requires separate producer main K/V")
        block_range = range(required) if dense else (tail_block,)
        for component, tensor in enumerate(pair):
            token_bytes = tensor.element_size() * tensor.shape[2] * tensor.shape[3]
            if remote_block_bytes[component] != self.block_size * token_bytes:
                raise ValueError("Mooncake V2 SFA PD producer main block bytes differ from the hot layout")
            target = self.hot_transfer_targets[layer_name][component]
            component_start = len(lengths)
            for logical_block in block_range:
                tokens = min(self.block_size, kv_tokens - logical_block * self.block_size) if dense else tail_tokens
                row_offset = (
                    logical_block * self.block_size if dense else self.hot_tokens + (tail_block % 2) * self.block_size
                )
                local = tensor.data_ptr() + (slot * tensor.shape[1] + row_offset) * token_bytes
                peer = remote_bases[component] + blocks[logical_block] * remote_strides[component]
                size = tokens * token_bytes
                # Preserve SFA's dense-run coalescing; never merge across K/V.
                if (
                    len(lengths) > component_start
                    and local_ptrs[-1] + lengths[-1] == local
                    and peer_ptrs[-1] + lengths[-1] == peer
                ):
                    lengths[-1] += size
                else:
                    local_ptrs.append(local)
                    peer_ptrs.append(peer)
                    lengths.append(size)
                    if transfer_targets is not None:
                        transfer_targets.append(target)

    def contains(self, ptr: int, size: int = 1) -> bool:
        return self.registry.resolve_offloaded_tensor(ptr, size) is not None

    def land(self, copies: Any, staging: torch.Tensor, owners: list[Any]) -> None:
        """Batch a bank's landing through the manager, preserving its event lifetime."""
        if not copies:
            return
        for copy in copies:
            if not 0 < copy.length <= _SPARSE_COPY_MAX_BYTES or not self.contains(copy.destination, copy.length):
                raise ValueError("Mooncake V2 SFA landing exceeds a Host KV range or the sparse-copy int32 ABI")
        addresses = torch.tensor(
            [[staging.data_ptr() + copy.staging_offset for copy in copies], [copy.destination for copy in copies]],
            dtype=torch.int64,
            device=staging.device,
        )
        lengths = torch.tensor([copy.length for copy in copies], dtype=torch.int32, device=staging.device)
        count = torch.tensor([len(copies)], dtype=torch.int32, device=staging.device)
        # sparse_copy may enqueue before reporting failure. The existing bank
        # drain keeps these descriptors alive on both success and failure.
        owners.append((addresses, lengths, count))
        self.manager.copy_sfa_kv(*addresses.unbind(0), lengths, count)


class MooncakeSFAPDState:
    """Track received prefixes until MRV2 initializes their existing SFA rows."""

    def __init__(self, main_group_id: int) -> None:
        self.main_group_id = main_group_id
        self.bindings: dict[str, int] = {}
        self.pending: set[str] = set()
        self.received: set[str] = set()
        self.remote_prefill_tokens: dict[str, int] = {}
        self._metadata: MooncakeSFAConnectorMetadata | None = None

    def bind_metadata(self, metadata: MooncakeSFAConnectorMetadata) -> None:
        """Publish transport facts before MRV2 builds existing SFA metadata.

        The early runner hook and ordered load hook see the same object. Only
        its first bind may create pending initialization; repeated binds must
        not reinitialize a row whose first decode has already committed.
        """
        if self._metadata is metadata:
            return
        prefills = {}
        for req_id, req in metadata.requests.items():
            if req.local_num_prompt_tokens != req.remote_num_prompt_tokens:
                raise ValueError(
                    "Mooncake V2 Sparse KV offload requires the same original prompt on P and D; "
                    "appending the producer's first output token is not supported."
                )
            prefills[req_id] = min(req.num_computed_tokens + req.num_external_tokens, req.remote_num_prompt_tokens)
        self.update(metadata.copy_sfa_slots, prefills)
        self._metadata = metadata

    def update(
        self,
        bindings: dict[str, int],
        remote_prefill_tokens: dict[str, int],
    ) -> None:
        if len(set(bindings.values())) != len(bindings):
            raise ValueError("Mooncake V2 SFA requests cannot share a hot row")
        for req_id, slot in self.bindings.items():
            if req_id in bindings and bindings[req_id] != slot:
                raise ValueError("Mooncake V2 SFA cannot rebind a live request")
        if not remote_prefill_tokens.keys() <= bindings.keys():
            raise ValueError("Mooncake V2 SFA transfer is missing its reserved hot row")
        newly_bound = set(bindings) - self.bindings.keys()
        self.bindings = dict(bindings)
        self.pending.intersection_update(bindings)
        self.received.intersection_update(bindings)
        self.remote_prefill_tokens = {
            req: count for req, count in self.remote_prefill_tokens.items() if req in bindings
        }
        self.remote_prefill_tokens.update(remote_prefill_tokens)
        self.pending.update(newly_bound | remote_prefill_tokens.keys())
        # A complete local prefix hit needs no READ, but its new hot row still
        # needs initialization from the retained Host blocks before decoding.
        self.received.update(newly_bound - remote_prefill_tokens.keys())
        self.received.difference_update(remote_prefill_tokens)

    def record_received(self, finished: set[str], failed: set[str]) -> None:
        self.received.update((finished - failed) & self.pending)
        self.received.difference_update(failed)

    def prepare(
        self,
        req_ids: Sequence[str],
        slots: Sequence[int],
        computed_tokens: Sequence[int],
        *,
        block_size: int,
        hot_tokens: int,
    ) -> tuple[bool, dict[int, tuple[int, int]], set[str]]:
        restore_tails = False
        dense_fills: dict[int, tuple[int, int]] = {}
        initialized: set[str] = set()
        for row, req_id in enumerate(req_ids):
            if req_id not in self.pending:
                continue
            if req_id not in self.received:
                raise RuntimeError("Mooncake V2 SFA decode started before Host KV receive completed")
            if int(slots[row]) != self.bindings[req_id]:
                raise RuntimeError("Mooncake V2 SFA decode row differs from its PD reservation")
            computed = int(computed_tokens[row])
            if computed < 0:
                raise ValueError("Mooncake V2 SFA computed-token count must be nonnegative")
            prefilled = self.remote_prefill_tokens.get(req_id)
            if prefilled is not None:
                # P contributes prompt KV only. D replays its last prompt token
                # and generates the complete output, as in the SFA connector.
                if computed != max(prefilled - 1, 0):
                    raise RuntimeError("Mooncake V2 SFA first decode prefix differs from the received hot/tail prefix")
                # SFA prefill placement uses the received extent, independently
                # of V2's first-forward replay prefix. An aligned long prefill
                # has no PD tail, but replaying its last token needs the previous
                # block's history. Reuse the existing tail-restore API once.
                restore_tails |= prefilled > hot_tokens and prefilled % block_size == 0
                initialized.add(req_id)
                continue
            dense, tail_tokens, _ = copy_sfa_prefill_dest_geometry(computed, block_size, hot_tokens)
            if dense and computed:
                dense_fills[int(slots[row])] = (row, computed)
            restore_tails |= tail_tokens > 0 and not dense
            initialized.add(req_id)
        return restore_tails, dense_fills, initialized

    def commit(
        self,
        initialized: set[str],
        *,
        tail_restore_required: bool = False,
        tails_restored: bool = False,
    ) -> None:
        """Retire one-time PD state only after the existing copy APIs succeed."""
        if tail_restore_required and not tails_restored:
            raise RuntimeError("Mooncake V2 SFA first decode is missing its tail-copy metadata")
        if not initialized <= self.pending & self.received:
            raise RuntimeError("Mooncake V2 SFA cannot commit initialization before successful receive")
        self.pending.difference_update(initialized)
        self.received.difference_update(initialized)
        for req_id in initialized:
            self.remote_prefill_tokens.pop(req_id, None)
