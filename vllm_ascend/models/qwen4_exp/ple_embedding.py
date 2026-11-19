# SPDX-License-Identifier: Apache-2.0
"""BF16, INT8 and MXFP8 PLE tables in CANN-registered host memory."""

import ctypes
import hashlib
import json
from pathlib import Path
from typing import cast

import numpy as np
import regex as re
import torch
import torch.distributed as dist
from safetensors import safe_open
from vllm.config import get_current_vllm_config
from vllm.distributed import get_engram_dp_group, get_etp_group, get_tp_group
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import logger
from vllm.models.qwen4_exp.common.ngram_embedding import Qwen4ExpPLEEmbedding, Qwen4ExpPLEPinnedHostEmbedding

from vllm_ascend.models.deepseek_v41.engram.npu import (
    CHUNK_ROWS,
    HostUvaBuffer,
    HostUvaBufferCleanupError,
    SharedUvaBuffer,
)
from vllm_ascend.models.deepseek_v41.engram.shared_rdma import _reader_operation
from vllm_ascend.models.host_offload import register_host_offload_owner
from vllm_ascend.models.qwen4_exp.ple_quant import AscendPLEQuantizedEmbeddingMethod
from vllm_ascend.ops.triton.qwen4_exp import ple_lookup, ple_lookup_quantized

GIB = 1 << 30


class PLECheckpoint:
    """One immutable BF16 PLE table described by the checkpoint index."""

    def __init__(self, model):
        root = Path(model)
        index_path = root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())["weight_map"]
        pattern = re.compile(r"^(.*\.layers\.\d+\.ple\.ple_embedding\.ngram_embedding)\.shard_(\d+)\.weight$")
        matches = [
            (m.group(1), int(m.group(2)), key, filename)
            for key, filename in index.items()
            if (m := pattern.fullmatch(key))
        ]
        prefixes = {item[0] for item in matches}
        if len(prefixes) != 1:
            raise ValueError("shared PLE requires exactly one checkpoint table")
        self.prefix = prefixes.pop()
        config = json.loads((root / "config.json").read_text())
        config = config.get("text_config", config)
        parts = config["split_ngram_parts"]
        if len(config.get("ple_layer_ids", ())) != 1 or not isinstance(parts, int) or parts <= 0:
            raise ValueError("shared PLE requires exactly one PLE layer and positive shard count")
        matches.sort(key=lambda item: item[1])
        if [item[1] for item in matches] != list(range(parts)):
            raise ValueError("PLE checkpoint shard indexes do not match split_ngram_parts")
        self.shards = []
        self.width = 0
        self.rows = 0
        first_rows = 0
        for _, number, key, filename in matches:
            path = root / filename
            with safe_open(path, framework="pt", device="cpu") as handle:
                tensor = handle.get_slice(key)
                shape = tensor.get_shape()
                if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0 or tensor.get_dtype() != "BF16":
                    raise ValueError(f"Unsupported BF16 PLE shard {key}")
                if not self.width:
                    self.width, first_rows = shape[1], shape[0]
                if shape[1] != self.width or (
                    shape[0] != first_rows and (number != parts - 1 or shape[0] > first_rows)
                ):
                    raise ValueError(f"Inconsistent PLE shard shape {key}")
                self.shards.append((path, key, self.rows, self.rows + shape[0]))
                self.rows += shape[0]
        if self.width * 2 > GIB:
            raise ValueError("PLE row exceeds BM mapping size")
        self.weight_names = frozenset(shard[1] for shard in self.shards)
        digest = hashlib.sha256()
        for path in (root / "config.json", index_path):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        digest.update(self.prefix.encode())
        self.identity = digest.hexdigest()

    def copy_rows(self, destination, start):
        if destination.device.type != "cpu" or destination.dtype != torch.bfloat16 or destination.ndim != 2:
            raise ValueError("PLE staging requires a CPU BF16 matrix")
        end = start + len(destination)
        if destination.shape[1] != self.width or start < 0 or end > self.rows:
            raise ValueError("PLE staging range exceeds checkpoint")
        for path, key, left, right in self.shards:
            begin, stop = max(start, left), min(end, right)
            if begin < stop:
                with safe_open(path, framework="pt", device="cpu") as handle, torch.device("cpu"):
                    destination[begin - start : stop - start].copy_(handle.get_slice(key)[begin - left : stop - left])

    @property
    def rows_per_mapping(self):
        return GIB // (self.width * 2)

    @property
    def allocation_bytes(self):
        return ((self.rows + self.rows_per_mapping - 1) // self.rows_per_mapping) * GIB

    @property
    def backing_rows(self):
        return self.allocation_bytes // (self.width * 2)

    def metadata(self):
        return {
            "identity": self.identity,
            "tables": {"ple": {"rows": self.rows, "row_bytes": self.width * 2, "dtype": "BF16"}},
        }

    def load_rdma(self):
        table = torch.empty((self.rows, self.width), dtype=torch.bfloat16, device="cpu")
        for start in range(0, self.rows, 262144):
            self.copy_rows(table[start : min(start + 262144, self.rows)], start)
        return {"ple": table.view(torch.uint8).reshape(self.rows, self.width * 2)}

    def load_bm(self, owner, stop):
        buffer = owner.add(self.prefix, self.backing_rows, self.width, torch.bfloat16)
        stage = torch.empty((min(262144, self.rows), self.width), dtype=torch.bfloat16, device="cpu")
        for start in range(0, self.rows, len(stage)):
            if stop.is_set():
                raise InterruptedError("Shared PLE loading cancelled")
            rows = stage[: min(len(stage), self.rows - start)]
            self.copy_rows(rows, start)
            cursor = 0
            while cursor < len(rows):
                global_row = start + cursor
                slab, within = divmod(global_row, self.rows_per_mapping)
                count = min(len(rows) - cursor, self.rows_per_mapping - within)
                address = buffer.handle.gva_to_va(
                    buffer.gva + slab * GIB + within * self.width * 2, buffer.runtime.bm.BmMemType.LOCAL_HOST
                )
                if not address:
                    raise RuntimeError("PLE owner mapping is not CPU-addressable")
                ctypes.memmove(address, rows[cursor:].data_ptr(), count * self.width * 2)
                cursor += count


def shared_ple_config(vllm_config):
    from vllm_ascend.ascend_config import get_ascend_config

    if not (getattr(vllm_config, "additional_config", None) or {}).get("shared_engram"):
        return None
    shared = get_ascend_config().shared_engram
    if shared is None:
        return None
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    if "filter_unused_per_tensor" not in DefaultModelLoader.Source.__dataclass_fields__ and not getattr(
        DefaultModelLoader, "_ascend_external_weight_filter", False
    ):
        raise RuntimeError("Shared PLE requires vLLM per-tensor checkpoint filtering")
    if not getattr(vllm_config, "use_v2_model_runner", False) or getattr(vllm_config, "shutdown_timeout", 0) <= 0:
        raise ValueError("Shared PLE requires Model Runner V2 and --shutdown-timeout > 0")
    if getattr(vllm_config.load_config, "load_format", None) == "dummy" or getattr(
        vllm_config.model_config, "enable_sleep_mode", False
    ):
        raise ValueError("Shared PLE does not support dummy loading or sleep mode")
    engram = vllm_config.engram_config
    if engram is None or not engram.cpu_offload or getattr(engram, "disk_offload", False):
        raise ValueError("Shared PLE requires CPU offload without disk offload")
    if len(getattr(vllm_config.model_config.hf_text_config, "ple_layer_ids", ())) != 1:
        raise ValueError("Shared PLE requires exactly one PLE layer")
    if shared.backend in ("a2", "a5"):
        parallel = vllm_config.parallel_config
        if (
            getattr(vllm_config, "speculative_config", None) is not None
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or getattr(parallel, "enable_dbo", False)
            or getattr(parallel, "enable_elastic_ep", False)
        ):
            raise ValueError("Shared PLE HIXL requires PCP=DCP=1 without speculative decoding, DBO or elastic EP")
    return shared


class ExternalPLEWeightsMixin:
    @property
    def filter_unused_checkpoint_weights(self):
        names: set[str] = set()
        for module in cast(torch.nn.Module, self).modules():
            names.update(getattr(module, "ple_external_checkpoint_names", ()))
        self._ple_external_checkpoint_names = frozenset(names)
        return bool(names)

    def is_unused_checkpoint_weight(self, name):
        if name in getattr(self, "_ple_external_checkpoint_names", ()):
            return True
        parent = getattr(super(), "is_unused_checkpoint_weight", None)
        return bool(parent(name)) if parent is not None else False


@_reader_operation
def _fetch_ple_rows(reader, ids, output, vocab_start, vocab_end):
    selected = ids.to(device="cpu", dtype=torch.int64).contiguous().numpy().reshape(-1)
    selected = np.where((selected >= vocab_start) & (selected < vocab_end), selected, -1).astype(np.int64, copy=False)
    reader._transfer_rows(dict(op="lookup", layer="ple", count=len(selected)), selected, output.numel() * 2)
    if len(selected):
        output.copy_(reader.buffer[: output.numel() * 2].view(torch.bfloat16).reshape(output.shape))
    reader._last_use = torch.npu.Event()
    reader._last_use.record(torch.npu.current_stream())


@Qwen4ExpPLEPinnedHostEmbedding.register_oot
class AscendPLEHostEmbedding(Qwen4ExpPLEPinnedHostEmbedding):
    supports_prefetch = False

    def __init__(self, *args, **kwargs):
        self._host_table: HostUvaBuffer | SharedUvaBuffer | None = None
        method = kwargs.get("embedding_method")
        self._quantized = isinstance(method, AscendPLEQuantizedEmbeddingMethod)
        self._mxfp8 = isinstance(method, AscendPLEQuantizedEmbeddingMethod) and method.mxfp8
        self._scale_dtype = method.scale_dtype if isinstance(method, AscendPLEQuantizedEmbeddingMethod) else None
        self._weight_dtype = (
            method.weight_dtype if isinstance(method, AscendPLEQuantizedEmbeddingMethod) else torch.bfloat16
        )
        if self._quantized:
            if get_etp_group().world_size != get_tp_group().world_size:
                raise ValueError("Quantized PLE host lookup requires ETP=TP")
            if kwargs.get("params_dtype") != torch.bfloat16:
                raise ValueError("Quantized PLE host lookup requires BF16 activations")
        self._shared_group = None
        self._shared_client = None
        self._shared_ptrs = None
        self._shared_checkpoint = None
        self._preparation_bound = False
        self._closing = False
        vllm_config = get_current_vllm_config()
        self._shared = shared_ple_config(vllm_config)
        if self._shared is not None and self._quantized:
            raise ValueError("Shared PLE requires BF16 weights")
        self._tracked_host_owner = register_host_offload_owner(self)
        config = vllm_config.engram_config
        if self._shared is None and config is not None and config.dp_shared_memory:
            group = get_engram_dp_group()
            if group is not None and group.world_size > 1:
                if not all(in_the_same_node_as(group.cpu_group)):
                    logger.info(
                        "PLE dp_shared_memory requested but DP replicas span multiple nodes; "
                        "falling back to private host tables."
                    )
                else:
                    self._shared_group = group
            else:
                logger.info(
                    "PLE dp_shared_memory requested but no local DP peers; falling back to private host tables."
                )
        try:
            # Bypass the CUDA pinned-host initializer while retaining the public
            # embedding contract and checkpoint/ETP sharding implementation.
            Qwen4ExpPLEEmbedding.__init__(self, *args, **kwargs)
            if self._shared is not None:
                self._attach_shared(vllm_config, kwargs)
        except HostUvaBufferCleanupError as exc:
            # The buffer constructor could not finish assigning _host_table.
            # Preserve its owner without hiding the original cleanup failure.
            self._host_table = exc.host_buffer
            raise
        except Exception as initialization_error:
            if self._shared is not None:
                if not self._tracked_host_owner:
                    from vllm_ascend.models.deepseek_v41.engram.shared_pool import rollback_shared_engram

                    rollback_shared_engram(self.close_host_offload)
                raise
            try:
                self.close_host_offload()
            except Exception as cleanup_error:
                raise HostUvaBufferCleanupError(
                    self._host_table, initialization_error, cleanup_error
                ) from cleanup_error
            raise

    def _attach_shared(self, config, kwargs):
        from vllm_ascend.models.deepseek_v41.engram.shared_pool import SharedPoolClient
        from vllm_ascend.models.deepseek_v41.engram.shared_rdma import MAX_ROWS, RemoteReader

        checkpoint = PLECheckpoint(config.model_config.model)
        if checkpoint.width != self.embedding_dim:
            raise ValueError("Shared PLE checkpoint does not match embedding width")
        model_layer = re.search(
            r"(?:^|\.)layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding$", kwargs.get("prefix", "")
        )
        checkpoint_layer = re.search(r"(?:^|\.)layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding$", checkpoint.prefix)
        if model_layer is None or checkpoint_layer is None or model_layer.group(1) != checkpoint_layer.group(1):
            raise ValueError("Shared PLE checkpoint prefix does not match the model layer")
        self._shared_checkpoint = checkpoint
        device = torch.device("npu", torch.npu.current_device())
        if self._shared.backend == "a3":
            from vllm_ascend.models.deepseek_v41.engram.common import warmup_shared_engram_aicpu

            warmup_shared_engram_aicpu()
            self._shared_client = SharedPoolClient(self._shared.endpoint, checkpoint.identity, device=device)
            self._shared_client.view(
                checkpoint.prefix, 0, 1, rows=checkpoint.backing_rows, width=checkpoint.width, dtype=torch.bfloat16
            )
            buffer = self._shared_client.buffers[checkpoint.prefix]
            pointers = []
            for start in range(0, checkpoint.rows, checkpoint.rows_per_mapping):
                count = min(checkpoint.rows_per_mapping, checkpoint.rows - start)
                pointers.append(
                    buffer.device_pointer((start // checkpoint.rows_per_mapping) * GIB, count * checkpoint.width * 2)
                )
            self._shared_ptrs = torch.tensor(pointers, dtype=torch.int64, device=device)
        else:
            if self.etp_data_parallel_size != 1:
                raise ValueError("Shared PLE HIXL requires ETP=TP")
            tokens, heads = kwargs.get("max_total_tokens", 0), kwargs.get("num_ngram_heads", 0)
            if tokens <= 0 or heads <= 0 or tokens * heads > MAX_ROWS or tokens * heads > self._shared.capacity:
                raise ValueError("Shared PLE HIXL row capacity is insufficient")
            self._shared_client = RemoteReader(
                config.model_config.model,
                self._shared.endpoint,
                capacity=self._shared.capacity,
                library_path=self._shared.library_path,
                listen_port=self._shared.listen_port,
                local_comm_res=self._shared.local_comm_res,
                metadata=checkpoint.metadata(),
            )
            self.register_buffer(
                "prepared_rows",
                torch.zeros((tokens, heads, self.embedding_dim), dtype=torch.bfloat16, device=device),
                persistent=False,
            )
        weight = self._parameters.pop("weight")
        self.register_buffer("weight", weight.detach(), persistent=False)
        self.ple_external_checkpoint_names = checkpoint.weight_names

    def bind_preparation(self):
        self._preparation_bound = True

    def prepare_rows(self, ids):
        if not self._preparation_bound or self._closing or self._shared_client is None:
            raise RuntimeError("Remote PLE preparation is unavailable")
        if ids.ndim != 2 or ids.shape[1] != self.prepared_rows.shape[1] or ids.shape[0] > self.prepared_rows.shape[0]:
            raise ValueError("Remote PLE IDs exceed the prepared graph buffer")
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("Shared PLE HIXL fetch must run outside graph capture")
        checkpoint = self._shared_checkpoint
        assert checkpoint is not None
        torch.npu.synchronize(self.prepared_rows.device)
        _fetch_ple_rows(
            self._shared_client,
            ids,
            self.prepared_rows[: ids.shape[0]],
            self.shard_indices.org_vocab_start_index,
            min(self.shard_indices.org_vocab_end_index, checkpoint.rows),
        )

    def clear_prepared_rows(self):
        torch.npu.synchronize(self.prepared_rows.device)
        self.prepared_rows.zero_()

    def close_host_offload(self) -> None:
        """Release after graph teardown; failed unregister remains retryable."""
        self._closing = True
        client = getattr(self, "_shared_client", None)
        if client is not None:
            client.close()
            self._shared_client = None
            self._shared_ptrs = None
        if self._host_scale is not None:
            # Drop aliases before closing shared storage, retaining failed owners.
            if hasattr(self, "_parameters"):
                self.register_parameter("weight_scale", None)
            self._host_scale.close()
            self._host_scale = None
        if self._quantized and hasattr(self, "weight_scale"):
            self.register_parameter("weight_scale", None)
        table = self._host_table
        if table is None:
            return
        # Drop the parameter alias before closing shared-memory storage.
        # Retain the owner on failure so unregister/free can be retried.
        if hasattr(self, "_parameters"):
            self.register_parameter("weight", None)
        table.close()
        self._host_table = None
        logger.info("PLE registered host table released")

    def allocate_embedding_weight(self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype) -> torch.Tensor:
        allowed = (self._weight_dtype, self._scale_dtype) if self._quantized else (torch.bfloat16,)
        if dtype not in allowed:
            raise ValueError(f"Unsupported PLE host storage dtype: {dtype}")
        if self._shared is not None:
            return torch.empty((num_embeddings, embedding_dim), dtype=dtype, device="meta")
        device = torch.device("npu", torch.npu.current_device())
        if dtype == self._scale_dtype:
            # Each DP replica keeps its own ETP-local scale shard on its NPU.
            return torch.zeros((num_embeddings, embedding_dim), dtype=dtype, device=device)
        table: HostUvaBuffer | SharedUvaBuffer
        if self._shared_group is None:
            table = HostUvaBuffer((num_embeddings, embedding_dim), dtype, device, "_host_table", self)
        else:
            table = SharedUvaBuffer(
                (num_embeddings, embedding_dim), dtype, device, self._shared_group, "_host_table", self
            )
        self._host_table = table
        tensor = table.tensor
        assert tensor is not None
        if self._shared_group is None or self._shared_group.rank_in_group == 0:
            tensor.view(torch.uint8).zero_()
        if self._shared_group is not None:
            dist.barrier(group=self._shared_group.cpu_group)
        logger.info(
            "PLE host storage: %d rows x %d, dtype=%s, %.3f GiB per rank",
            num_embeddings,
            embedding_dim,
            dtype,
            tensor.numel() * tensor.element_size() / 1024**3,
        )
        return tensor

    def weight_loader(self, param, loaded_weight, checkpoint_start=None) -> None:
        group = self._shared_group
        if group is None:
            super().weight_loader(param, loaded_weight, checkpoint_start)
            return
        error = None
        # Only the host weight table is shared across DP replicas.
        if group.rank_in_group == 0 or param is getattr(self, "weight_scale", None):
            try:
                super().weight_loader(param, loaded_weight, checkpoint_start)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        errors = [None] * group.world_size
        dist.all_gather_object(errors, error, group=group.cpu_group)
        if any(error is not None for error in errors):
            raise RuntimeError(f"PLE shared weight loading failed: {errors}")

    def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
        raise NotImplementedError("PLE asynchronous prefetch is not enabled")

    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        raise NotImplementedError("PLE asynchronous prefetch is not enabled")

    def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
        return self.sync_lookup(ngram_ids)

    def _lookup(self, input_ids: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
        shared = getattr(self, "_shared", None)
        if shared is not None and shared.backend in ("a2", "a5"):
            if not self._preparation_bound or self._closing:
                raise RuntimeError("Remote PLE requires model-state preparation")
            if (
                input_ids.ndim != 2
                or input_ids.shape[0] > self.prepared_rows.shape[0]
                or input_ids.shape[1] != self.prepared_rows.shape[1]
            ):
                raise ValueError("Remote PLE graph input exceeds its prepared buffer")
            return self.prepared_rows[: input_ids.shape[0]]
        if shared is None and (self._host_table is None or self.weight is None):
            raise RuntimeError("PLE host table has been closed")
        if self._quantized and self.weight_scale is None:
            raise RuntimeError("Quantized PLE host storage has been closed")
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if output is None:
            output = torch.empty(expected_shape, dtype=torch.bfloat16, device=input_ids.device)
        elif (
            tuple(output.shape) != expected_shape
            or output.dtype != torch.bfloat16
            or output.device != input_ids.device
            or (self._quantized and not output.is_contiguous())
        ):
            raise ValueError("PLE prefetch output must match the input shape, lookup dtype, and input device")
        if self._quantized:
            assert self._host_table is not None and self._host_scale is not None
            ple_lookup_quantized.lookup_quantized(
                self._host_table.ptrs,
                self.weight_scale,
                input_ids,
                output,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                CHUNK_ROWS,
                is_mxfp8=self._mxfp8,
            )
        else:
            if shared is None:
                assert self._host_table is not None
                ptrs = self._host_table.ptrs
                end = self.shard_indices.org_vocab_end_index
                chunk = CHUNK_ROWS
            else:
                if self._closing or self._shared_ptrs is None or self._shared_checkpoint is None:
                    raise RuntimeError("Shared PLE reader is closed")
                ptrs = self._shared_ptrs
                end = min(self.shard_indices.org_vocab_end_index, self._shared_checkpoint.rows)
                chunk = self._shared_checkpoint.rows_per_mapping
            ple_lookup.lookup(
                ptrs,
                input_ids,
                output,
                self.shard_indices.org_vocab_start_index,
                end,
                chunk,
                global_rows=shared is not None,
            )
        return output
