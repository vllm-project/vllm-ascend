# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend head-sharded Engram table with the INT8 checkpoint contract.

Subclasses upstream ``ParallelEngramEmbedding`` and preserves its parameter
and loader interface. Ascend supplies initialization, uniform head layout,
storage and lookup; upstream rank selection and hash gathering are reused:

* codes are INT8 with group-32 FP32 scales (the Ascend checkpoint uses these
  instead of upstream FP8/UE8M0);
* the table is either NPU memory or CANN-registered host memory read through a
  chunked device pointer table.
"""

import json
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import logger
from vllm.model_executor.utils import set_weight_attrs

from .._compat import ParallelEngramEmbedding
from .npu import (
    HostUvaBuffer,
    SharedUvaBuffer,
    gather_dequantize_engram_int8,
    gather_dequantize_host_uva,
    quantize_engram_rows,
)
from .parallel import (
    _gather_engram_rows,
    engram_head_shard_rank,
    gather_engram_hashes,
    get_engram_dp_group,
    get_engram_dp_size,
)


class AscendParallelEngramEmbedding(ParallelEngramEmbedding):
    """TP(+EDP) head shard of one Engram table, stored as INT8 + FP32 scales."""

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: tuple[int, ...],
        layer_hash_index: int,
        block_size: int = 32,
        cpu_offload: bool = False,
        dp_shared_memory: bool = False,
    ) -> None:
        self.cpu_offload = cpu_offload
        self.layer_hash_index = layer_hash_index
        self._shared_group = None
        if dp_shared_memory:
            group = get_engram_dp_group()
            if group is None or group.world_size <= 1:
                raise ValueError("dp_shared_memory needs a node-local sharing group with more than one rank")
            self._shared_group = group
            # Sharing replaces the per-step DP lookup collectives: every
            # replica looks up its own tokens over the mapped table.
            self.dp_size = 1
        else:
            self.dp_size = max(get_engram_dp_size(), 1)
        self._codes_uva: HostUvaBuffer | SharedUvaBuffer | None = None
        self._scales_uva: HostUvaBuffer | SharedUvaBuffer | None = None
        # The upstream constructor queries CUDA properties; keep its parameter
        # contract with Ascend INT8 storage.
        nn.Module.__init__(self)
        assert head_sizes and all(size > 0 for size in head_sizes)
        assert sum(head_sizes) <= num_embeddings
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.block_size = block_size
        self.n_hash_cols = len(head_sizes)
        self.tp_size = get_tensor_model_parallel_world_size()
        num_shards, head_rank = self._get_shard_info()
        if self.n_hash_cols % num_shards:
            raise ValueError(
                f"Engram requires uniform head shards: {self.n_hash_cols} heads "
                f"cannot be divided over {num_shards} TP x EDP shards. "
                "Use a divisible topology or enable dp_shared_memory."
            )
        self.part_n_hash_cols = self.n_hash_cols // num_shards
        self.head_start = head_rank * self.part_n_hash_cols
        self.vocab_start_idx = sum(head_sizes[: self.head_start])
        self.vocab_end_idx = sum(head_sizes[: self.head_start + self.part_n_hash_cols])
        self.part_num_embeddings = self.vocab_end_idx - self.vocab_start_idx
        weight, scales = self._allocate_weights()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.weight_scale_inv = nn.Parameter(scales, requires_grad=False)
        for param in (self.weight, self.weight_scale_inv):
            set_weight_attrs(param, {"weight_loader": self._weight_loader, "engram_vocab_start": self.vocab_start_idx})
        if cpu_offload:
            set_weight_attrs(self.weight, {"dummy_weight_value": 0})
            set_weight_attrs(self.weight_scale_inv, {"dummy_weight_value": 1.0})
            logger.info(
                "Engram table offloaded to registered host memory: %d rows x %d, %.2f GiB per rank",
                self.part_num_embeddings,
                self.dim,
                self.part_num_embeddings * (self.dim + (self.dim // self.block_size) * 4) / 1024**3,
            )

    def _get_shard_info(self) -> tuple[int, int]:
        if self.dp_size == 1:
            return self.tp_size, get_tensor_model_parallel_rank()
        return self.tp_size * self.dp_size, engram_head_shard_rank()

    def _allocate_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        codes_shape = (self.part_num_embeddings, self.dim)
        scales_shape = (self.part_num_embeddings, self.dim // self.block_size)
        device = torch.device("npu", torch.npu.current_device())
        if not self.cpu_offload:
            return (
                # Zeroed, not empty: the engine profiles the model (and runs the
                # Engram lookups) before the checkpoint is read, and a table of
                # uninitialised rows poisons the stream enough to break the MoE
                # routing later in that same forward.
                torch.zeros(codes_shape, dtype=torch.int8, device=device),
                torch.zeros(scales_shape, dtype=torch.float32, device=device),
            )
        if self._shared_group is not None:
            # One physical copy of the mapped range, registered by each
            # rank in the sharing group.
            self._codes_uva = SharedUvaBuffer(codes_shape, torch.int8, device, self._shared_group)
            self._scales_uva = SharedUvaBuffer(scales_shape, torch.float32, device, self._shared_group)
            return self._codes_uva.tensor, self._scales_uva.tensor
        self._codes_uva = HostUvaBuffer(codes_shape, torch.int8, device)
        self._scales_uva = HostUvaBuffer(scales_shape, torch.float32, device)
        # Same reason as the device path: aclrtMallocHost hands back whatever
        # was in the pages, and the profiling forward looks up before the
        # checkpoint load fills them.
        self._codes_uva.tensor.zero_()
        self._scales_uva.tensor.zero_()
        return self._codes_uva.tensor, self._scales_uva.tensor

    def close_host_offload(self) -> None:
        """Release the registered host ranges (shutdown / reload path).

        The parameters alias the host range, so a released buffer has to take
        its alias with it: otherwise a later reader follows a live tensor into
        unmapped memory.  Each buffer and its own alias are dropped together,
        and only after that buffer's close succeeded, so a failed unregister
        stays retryable and a half-finished release never leaves a live
        parameter pointing at memory the other buffer has already freed.
        """
        aliases = {
            "_codes_uva": "weight",
            "_scales_uva": "weight_scale_inv",
        }
        for name, alias in aliases.items():
            buffer = getattr(self, name)
            if buffer is None:
                continue
            buffer.close()
            setattr(self, name, None)
            param = getattr(self, alias)
            setattr(
                self,
                alias,
                nn.Parameter(
                    torch.empty(0, dtype=param.dtype, device=param.device),
                    requires_grad=False,
                ),
            )

    def bind_checkpoint(self, model_path, key: str) -> None:
        """Use parameter callbacks while retaining the indexed shard reader.

        Codes and scales are read together from the selected index when the
        weight arrives. The scale callback is deliberately a no-op: iterator
        order and a separate unquantized index must not overwrite that pair.
        """
        self._checkpoint_path = model_path
        self._checkpoint_key = key

    def _weight_loader(self, param, loaded_weight) -> None:
        if param is self.weight:
            self.load_checkpoint(self._checkpoint_path, self._checkpoint_key)

    def load_checkpoint(self, model_path, key, chunk_rows=65536):
        """Stream this rank's assigned rows, quantizing BF16 when needed.

        Shared head slices have one writer per EDP group. The final CPU
        collective synchronizes writes and propagates loading failures.
        """
        if self._shared_group is None:
            self._load_into_storage(model_path, key, chunk_rows)
            return
        error = None
        if self._shared_group.rank_in_group == 0:
            try:
                self._load_into_storage(model_path, key, chunk_rows)
            except Exception as exc:  # noqa: BLE001 - propagated to every rank
                error = f"{type(exc).__name__}: {exc}"
        errors: list[str | None] = [None] * self._shared_group.world_size
        dist.all_gather_object(errors, error, group=self._shared_group.cpu_group)
        failures = "; ".join(f"rank {rank}: {failure}" for rank, failure in enumerate(errors) if failure is not None)
        if failures:
            raise RuntimeError(f"Engram shared load failed: {failures}")

    # The Ascend table is streamed out of these indexed safetensors shards.
    _INDEX_FILES = (
        "quant_model_weights.safetensors.index.json",
        "model.safetensors.index.json",
    )

    @classmethod
    def _checkpoint_index(cls, root: Path, key: str) -> dict[str, str]:
        """Weight map of the shard that holds ``key``.

        A checkpoint without an index, or a single-file/``pt`` one, cannot be
        served by this loader.  Inside the loader this is the last line of
        defence, after the tables are already allocated; the construction path
        runs `preflight_engram_checkpoint()` first so an unreadable checkpoint
        fails before any backing exists.
        """
        named = []
        for name in cls._INDEX_FILES:
            candidate = root / name
            if not candidate.is_file():
                continue
            named.append(name)
            weight_map = json.loads(candidate.read_text())["weight_map"]
            if key in weight_map:
                return weight_map
        raise ValueError(
            f"Engram table {key!r} is not in an indexed safetensors checkpoint "
            f"under {root} (found: {', '.join(named) or 'no index'})."
        )

    def _load_into_storage(self, model_path, key, chunk_rows) -> None:
        root = Path(model_path)
        scale_key = key.removesuffix(".weight") + ".scale"
        index = self._checkpoint_index(root, key)
        start, end = (self.vocab_start_idx, self.vocab_end_idx)
        with safe_open(root / index[key], framework="pt", device="cpu") as file:
            tensor = file.get_slice(key)
            quantized = tensor.get_dtype() in ("I8", "INT8")
            if quantized:
                with safe_open(root / index[scale_key], framework="pt", device="cpu") as sf:
                    scale = sf.get_slice(scale_key)
                    for chunk_start in range(start, end, chunk_rows):
                        stop = min(chunk_start + chunk_rows, end)
                        offset = chunk_start - self.vocab_start_idx
                        target_end = offset + (stop - chunk_start)
                        self.weight.data[offset:target_end].copy_(tensor[chunk_start:stop])
                        self.weight_scale_inv.data[offset:target_end].copy_(scale[chunk_start:stop])
            else:
                for chunk_start in range(start, end, chunk_rows):
                    stop = min(chunk_start + chunk_rows, end)
                    offset = chunk_start - self.vocab_start_idx
                    target_end = offset + (stop - chunk_start)
                    codes, scales = quantize_engram_rows(tensor[chunk_start:stop].to(torch.float32))
                    self.weight.data[offset:target_end].copy_(codes)
                    self.weight_scale_inv.data[offset:target_end].copy_(scales)
        logger.info("Engram rows [%d, %d) loaded from %s", start, end, index[key])

    def embed_gathered(self, gathered: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """Embed ids already gathered across the EDP group.

        ``gathered`` is ``[slot * EDP, n_hash_cols]`` rank-major; each replica
        keeps its own ``num_tokens`` window. Returns ``[num_tokens, n_hash_cols,
        dim]`` bf16 with the heads back in checkpoint order.
        """
        # The kernel walks ids as [token, columns] with a unit inner stride, and
        # `gathered` is a slice of a [tokens, layers, columns] buffer.
        gathered = gathered.contiguous()
        out = torch.zeros(
            (gathered.shape[0], self.part_n_hash_cols, self.dim),
            dtype=torch.bfloat16,
            device=gathered.device,
        )
        self.lookup(gathered, out)
        if self.dp_size > 1:
            out = _gather_engram_rows(out, num_tokens)
        else:
            out = out[:num_tokens]
        if self.tp_size > 1:
            out = tensor_model_parallel_all_gather(out, dim=1)
        return out[:, : self.n_hash_cols]

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [num_tokens, n_hash_cols] -> [num_tokens, n_hash_cols, dim].

        The shared mode has to be passed through: the replicas of one TP slot
        map the same table, so a gathered batch would make each of them embed
        EDP0's ids instead of its own window.
        """
        num_tokens = indices.shape[0]
        gathered = gather_engram_hashes(indices, dp_shared_memory=self._shared_group is not None)
        return self.embed_gathered(gathered, num_tokens)

    def lookup(self, indices: torch.Tensor, out: torch.Tensor, background: bool = False) -> None:
        """Look up this shard's heads into ``[tokens, padded_heads, dim]`` bf16.

        All local head columns are written, including zeroes for invalid IDs.
        """
        tokens = indices.shape[0]
        if tokens == 0 or self.part_n_hash_cols == 0:
            return
        launch = dict(
            head_start=self.head_start,
            local_heads=self.part_n_hash_cols,
            pad_heads=self.part_n_hash_cols,
            vocab_start=self.vocab_start_idx,
            vocab_end=self.vocab_end_idx,
            output=out.view(-1, self.dim),
        )
        if self._codes_uva is not None:
            assert self._scales_uva is not None
            codes_uva = cast(HostUvaBuffer, self._codes_uva)
            scales_uva = cast(HostUvaBuffer, self._scales_uva)
            gather_dequantize_host_uva(codes_uva, scales_uva, indices, **launch)
        elif self.weight.device.type != "cpu":
            gather_dequantize_engram_int8(self.weight, self.weight_scale_inv, indices, self.dim, **launch)
        else:
            _torch_lookup(self, indices, out)


def _torch_lookup(embed: AscendParallelEngramEmbedding, indices, out) -> None:
    """CPU reference used by tests and bring-up without a device table."""
    heads = embed.part_n_hash_cols
    columns = indices[:, embed.head_start : embed.head_start + heads].long()
    owned = (columns >= embed.vocab_start_idx) & (columns < embed.vocab_end_idx)
    local = torch.where(owned, columns - embed.vocab_start_idx, 0).reshape(-1)
    codes = torch.index_select(embed.weight.data, 0, local)
    scales = torch.index_select(embed.weight_scale_inv.data, 0, local)
    decoded = codes.float().unflatten(-1, (-1, embed.block_size)) * scales.unsqueeze(-1)
    decoded = decoded.flatten(-2).bfloat16().view(indices.shape[0], heads, embed.dim)
    out[:, :heads].copy_(torch.where(owned.unsqueeze(-1), decoded, 0.0))


def preflight_engram_checkpoint(root, layer_ids, embed_cls=AscendParallelEngramEmbedding) -> None:
    """Check that this checkpoint can be served *before* any table is allocated.

    The shard loader streams indexed safetensors, so a checkpoint without an
    index (or with a missing Engram key, or an INT8 weight without its scale)
    cannot be loaded here.  Finding that out during weight iteration would be
    after every rank has already allocated and registered its table,
    potentially the complete node table in row shared mode. Resolve the index
    entries and read the safetensors headers first.  The loader
    keeps its own check as the last line of defence.
    """
    root = Path(root)
    for layer_id in layer_ids:
        key = f"layers.{layer_id}.engram.embed.weight"
        index = embed_cls._checkpoint_index(root, key)
        shard = root / index[key]
        if not shard.is_file():
            raise ValueError(f"Engram layer {layer_id}: the checkpoint index points at {shard}, which does not exist.")
        with safe_open(shard, framework="pt", device="cpu") as file:
            quantized = file.get_slice(key).get_dtype() in ("I8", "INT8")
        if not quantized:
            continue
        scale_key = key.removesuffix(".weight") + ".scale"
        # The loader resolves scales from the weight's selected index too.
        if scale_key not in index:
            raise ValueError(
                f"Engram layer {layer_id}: {scale_key!r} is missing from the checkpoint index selected for {key!r}."
            )
        scale_shard = root / index[scale_key]
        if not scale_shard.is_file():
            raise ValueError(
                f"Engram layer {layer_id}: the checkpoint index points at "
                f"{scale_shard} for {scale_key!r}, which does not exist."
            )
        with safe_open(scale_shard, framework="pt", device="cpu") as file:
            if scale_key not in file.keys():  # noqa: SIM118 - safe_open is not a mapping
                raise ValueError(
                    f"Engram layer {layer_id}: {scale_shard} does not contain the scale tensor {scale_key!r}."
                )
