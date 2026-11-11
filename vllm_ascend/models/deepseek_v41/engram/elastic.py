# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CANN ElasticBuffer offload for DeepSeek V4.1 Engram tables."""

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn
from vllm.logger import logger

from .diagnostics import log_engram_runtime

E8M0_ONE_BITS = 127
E8M0_NAN_BITS = 255
FP32_EXPONENT_SHIFT = 23
FP32_E8M0_MIN_BITS = 1 << 22  # 2**-127, a float32 subnormal.
FP32_NAN_BITS = 0x7FC00000
SCALE_COPY_CHUNK_ROWS = 8192
ENGRAM_STORAGE_ALIGNMENT = 2 * 1024 * 1024


def _validate_engram_op_schema(name: str, parameters: tuple[tuple[str, str], ...]) -> str:
    """Check the registered inference ABI before native resources are created."""
    try:
        schema = getattr(torch.ops.cann_ops_transformer, name).default._schema
        actual = tuple((argument.name, str(argument.type)) for argument in schema.arguments)
        returns = tuple(str(result.type) for result in schema.returns)
    except AttributeError as exc:
        raise RuntimeError(f"Engram MXFP8 requires the registered {name} operation") from exc
    if actual != parameters or returns != ("Tensor",):
        raise RuntimeError(
            f"Engram MXFP8 requires the inference tensor-scale ABI for {name}; "
            f"expected parameters={parameters} and one Tensor result; installed schema: {schema}"
        )
    return str(schema)


class EngramTableState(Enum):
    EMPTY = "empty"
    STAGED = "staged"
    READY = "ready"
    POISONED = "poisoned"
    DESTROYED = "destroyed"


@dataclass(slots=True)
class EngramFetchHandle:
    """One outstanding ElasticBuffer fetch and its completion callback."""

    owner: object
    shape: tuple[int, ...]
    flat_ids: torch.Tensor
    wait: Callable[[], tuple[torch.Tensor, torch.Tensor]]
    completed: bool = False


def create_engram_process_group(
    engram_tp_size: int,
    *,
    layer_id: int | None = None,
    expected_world_size: int | None = None,
):
    """Create a dedicated physical group for one Engram table."""
    if engram_tp_size <= 0:
        logger.error(
            "FOR-ENGRAM process group creation failed: layer=%s engram_tp_size=%d must be positive",
            layer_id,
            engram_tp_size,
        )
        raise ValueError("engram_tp_size must be positive")
    if not dist.is_available() or not dist.is_initialized():
        logger.error(
            "FOR-ENGRAM process group creation failed: layer=%s distributed is not initialized",
            layer_id,
        )
        raise RuntimeError("Engram ElasticBuffer requires an initialized distributed process group")

    world_size = dist.get_world_size()
    if expected_world_size is not None and world_size != expected_world_size:
        logger.error(
            "FOR-ENGRAM process group creation failed: layer=%s runtime_world_size=%d configured_world_size=%d",
            layer_id,
            world_size,
            expected_world_size,
        )
        raise RuntimeError(
            "Engram requires the PyTorch distributed world to span all configured "
            f"ranks; runtime={world_size}, configured={expected_world_size}"
        )
    if world_size % engram_tp_size:
        logger.error(
            "FOR-ENGRAM process group creation failed: layer=%s world_size=%d engram_tp_size=%d",
            layer_id,
            world_size,
            engram_tp_size,
        )
        raise ValueError(f"engram_tp_size={engram_tp_size} must divide distributed world_size={world_size}")
    rank = dist.get_rank()
    selected = None
    selected_ranks = None
    logger.debug(
        "FOR-ENGRAM process group creation started: layer=%s rank=%d world_size=%d engram_tp_size=%d",
        layer_id,
        rank,
        world_size,
        engram_tp_size,
    )
    # Every rank creates all groups in the same order. Separate calls for the
    # Engram layers deliberately produce separate physical resources.
    try:
        for first in range(0, world_size, engram_tp_size):
            ranks = list(range(first, first + engram_tp_size))
            group = dist.new_group(ranks=ranks, backend=dist.get_backend())
            if rank in ranks:
                selected = group
                selected_ranks = ranks
    except BaseException:
        logger.exception(
            "FOR-ENGRAM process group creation failed: layer=%s rank=%d",
            layer_id,
            rank,
        )
        raise
    if selected is None:
        logger.error(
            "FOR-ENGRAM process group creation failed: no local group for layer=%s rank=%d",
            layer_id,
            rank,
        )
        raise RuntimeError("Failed to create the local Engram process group")
    logger.debug(
        "FOR-ENGRAM process group creation completed: layer=%s rank=%d ranks=%s",
        layer_id,
        rank,
        selected_ranks,
    )
    return selected, True


class ElasticEngramEmbedding(nn.Module):
    """Row-sharded Engram table backed by CANN ElasticBuffer.

    FP8-only storage preserves E4M3 bytes in host memory and replicates
    E8M0 scales on each NPU; fetched
    rows are decoded to BF16 by graph-capturable device operations.
    """

    def __init__(
        self,
        rows: int,
        width: int,
        engram_tp_size: int,
        group=None,
        owns_group: bool = False,
        layer_id: int | None = None,
        storage_format: str = "mxfp8",
        minimum_rows: int | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        if rows <= 0 or width <= 0:
            raise ValueError("Engram table dimensions must be positive")
        if engram_tp_size <= 0:
            raise ValueError("engram_tp_size must be positive")
        if storage_format == "fp8":
            storage_format = "mxfp8"
        if storage_format != "mxfp8":
            raise ValueError("Engram offload only supports FP8 storage (fp8 or mxfp8)")
        minimum_rows = rows if minimum_rows is None else int(minimum_rows)
        if minimum_rows <= 0 or minimum_rows > rows:
            raise ValueError("Engram minimum_rows must be positive and no greater than rows")
        if group is None:
            raise ValueError("ElasticBuffer Engram requires a non-null process group")

        self.rows = int(rows)
        self.width = int(width)
        self.minimum_rows = minimum_rows
        self.engram_tp_size = int(engram_tp_size)
        self.storage_format = storage_format
        self.group = group
        self.owns_group = owns_group
        self.layer_id = layer_id
        self.device = None if device is None else torch.device(device)
        self.rank = dist.get_rank(group)
        self.shard_rows = (self.rows + self.engram_tp_size - 1) // self.engram_tp_size
        self.padded_rows = self.shard_rows * self.engram_tp_size
        if self.padded_rows > torch.iinfo(torch.int32).max:
            raise ValueError("Engram global row IDs do not fit ElasticBuffer's int32 ABI")
        self.start = self.rank * self.shard_rows
        self.end = min(self.start + self.shard_rows, self.rows)
        self._host_weight: torch.Tensor | None = None
        self._host_scale: torch.Tensor | None = None
        self._scale: torch.Tensor | None = None
        self._scale_block_size = 0
        self._probe_ids: torch.Tensor | None = None
        self._probe_codes: torch.Tensor | None = None
        self._probe_scales: torch.Tensor | None = None
        self._fetch_logged = False
        self._failure_diagnostics_logged = False
        self.engram_buffer = None
        self._fetch_in_flight = False
        self._state = EngramTableState.EMPTY
        logger.debug(
            "FOR-ENGRAM table manager initialized: layer=%s tp_rank=%d "
            "tp_size=%d global_rows=%d minimum_rows=%d shard_rows=%d "
            "width=%d storage=%s",
            self.layer_id,
            self.rank,
            self.engram_tp_size,
            self.rows,
            self.minimum_rows,
            self.shard_rows,
            self.width,
            self.storage_format,
        )

    @property
    def state(self) -> EngramTableState:
        return self._state

    @staticmethod
    def _weight_map(root: Path) -> dict[str, str]:
        index_path = root / "model.safetensors.index.json"
        if index_path.is_file():
            return json.loads(index_path.read_text())["weight_map"]
        single_file = root / "model.safetensors"
        if not single_file.is_file():
            raise FileNotFoundError(f"No model.safetensors.index.json or model.safetensors under {root}")
        with safe_open(single_file, framework="pt", device="cpu") as file:
            keys = file.keys()  # safe_open exposes keys(), but is not iterable.
            return {key: single_file.name for key in keys}

    @staticmethod
    def _scale_to_float(scale: torch.Tensor) -> torch.Tensor:
        if scale.dtype == torch.uint8:
            exponent = scale.to(torch.int32) - E8M0_ONE_BITS
            return torch.ldexp(
                torch.ones_like(exponent, dtype=torch.float32),
                exponent,
            )
        return scale.float()

    def _target_device(self) -> torch.device:
        if self.device is not None:
            return self.device
        npu = getattr(torch, "npu", None)
        if npu is None:
            raise RuntimeError("Engram distributed preflight requires an initialized NPU device")
        return torch.device("npu", npu.current_device())

    def _sync_phase_status(self, succeeded: bool, phase: str) -> bool:
        """Prevent entry into ElasticBuffer barriers after rank-local failure."""
        if self.engram_tp_size == 1:
            return succeeded
        status = torch.tensor(
            int(succeeded),
            dtype=torch.int32,
            device=self._target_device(),
        )
        dist.all_reduce(status, op=dist.ReduceOp.MIN, group=self.group)
        # This runs only during model startup, before any inference hot path.
        all_succeeded = bool(status.item())
        return all_succeeded

    def _release_staging(self) -> None:
        self._host_weight = None
        self._host_scale = None
        self._probe_ids = None
        self._probe_codes = None
        self._probe_scales = None

    def _resolve_keys(self, root: Path, key: str) -> tuple[dict[str, str], str, str]:
        weight_map = self._weight_map(root)
        if key not in weight_map:
            alternate = key.removeprefix("model.") if key.startswith("model.") else f"model.{key}"
            if alternate in weight_map:
                key = alternate
        base = key.removesuffix(".weight")
        scale_key = next(
            (candidate for candidate in (base + ".scale", base + ".weight_scale_inv") if candidate in weight_map),
            base + ".scale",
        )
        if key not in weight_map or scale_key not in weight_map:
            raise KeyError(f"Engram checkpoint requires paired tensors {key} and {scale_key}")
        return weight_map, key, scale_key

    def _validate_checkpoint_metadata(
        self,
        key: str,
        weight_shape: list[int],
        weight_dtype: str,
        scale_key: str,
        scale_shape: list[int],
        scale_dtype: str,
    ) -> tuple[int, int]:
        if len(weight_shape) != 2 or weight_shape[1] != self.width:
            raise ValueError(f"{key}: expected [rows, {self.width}], got {weight_shape}")
        checkpoint_rows = weight_shape[0]
        if checkpoint_rows < self.minimum_rows or checkpoint_rows > self.rows:
            raise ValueError(
                f"{key}: checkpoint rows must be in [{self.minimum_rows}, {self.rows}], got {checkpoint_rows}"
            )
        if weight_dtype not in ("F8_E4M3", "F8_E4M3FN"):
            raise TypeError(f"{key}: expected FP8 E4M3 checkpoint data, got {weight_dtype}")
        if (
            len(scale_shape) != 2
            or scale_shape[0] != checkpoint_rows
            or scale_shape[1] <= 0
            or self.width % scale_shape[1]
        ):
            raise ValueError(f"{scale_key}: incompatible scale shape {scale_shape} for weight {weight_shape}")
        if scale_dtype not in ("F8_E8M0", "F8_E8M0FNU", "U8"):
            raise TypeError(f"{scale_key}: expected E8M0 or uint8 checkpoint data, got {scale_dtype}")
        block_size = self.width // scale_shape[1]
        return checkpoint_rows, block_size

    def _load_checkpoint_local(
        self,
        root: Path,
        key: str,
        chunk_rows: int,
    ) -> tuple[str, int, int, int]:
        if chunk_rows <= 0:
            raise ValueError("Engram chunk_rows must be positive")
        weight_map, key, scale_key = self._resolve_keys(root, key)
        with safe_open(root / weight_map[key], framework="pt", device="cpu") as weight_file:
            weight = weight_file.get_slice(key)
            weight_shape = weight.get_shape()
            weight_dtype = weight.get_dtype()
            with safe_open(root / weight_map[scale_key], framework="pt", device="cpu") as scale_file:
                scale = scale_file.get_slice(scale_key)
                scale_shape = scale.get_shape()
                checkpoint_rows, block_size = self._validate_checkpoint_metadata(
                    key,
                    weight_shape,
                    weight_dtype,
                    scale_key,
                    scale_shape,
                    scale.get_dtype(),
                )
                self._scale_block_size = block_size
                logger.debug(
                    "FOR-ENGRAM checkpoint layout: layer=%s tp_rank=%d weight_shape=%s weight_dtype=%s "
                    "scale_shape=%s scale_dtype=%s block_size=%d storage=%s full_scale_bytes=%d",
                    self.layer_id,
                    self.rank,
                    weight_shape,
                    weight_dtype,
                    scale_shape,
                    scale.get_dtype(),
                    block_size,
                    self.storage_format,
                    self.padded_rows * scale_shape[1],
                )
                source_end = min(self.end, checkpoint_rows)
                copied_rows = max(source_end - self.start, 0)

                self._stage_fp8_checkpoint(weight, scale, checkpoint_rows, chunk_rows, scale_key)
        return key, checkpoint_rows, copied_rows, block_size

    def _stage_fp8_checkpoint(self, weight, scale, checkpoint_rows: int, chunk_rows: int, scale_key: str) -> None:
        """Retain raw codes, full scale bytes, and a bounded cross-rank probe."""
        scale_cols = self.width // self._scale_block_size
        codes = torch.zeros((self.shard_rows, self.width), dtype=torch.uint8, device="cpu")
        for start in range(self.start, min(self.end, checkpoint_rows), chunk_rows):
            stop = min(start + chunk_rows, self.end, checkpoint_rows)
            chunk = weight[start:stop]
            if chunk.dtype != torch.float8_e4m3fn:
                raise TypeError(f"Engram FP8 checkpoint slice must be E4M3FN, got {chunk.dtype}")
            codes[start - self.start : stop - self.start].copy_(chunk.view(torch.uint8))
        self._host_weight = codes.view(torch.float8_e4m3fn)
        self._host_scale = torch.full((self.padded_rows, scale_cols), E8M0_ONE_BITS, dtype=torch.uint8, device="cpu")
        for start in range(0, checkpoint_rows, chunk_rows):
            stop = min(start + chunk_rows, checkpoint_rows)
            chunk = scale[start:stop]
            if chunk.dtype not in (torch.uint8, getattr(torch, "float8_e8m0fnu", None)):
                raise TypeError(f"{scale_key}: expected E8M0 bytes, got {chunk.dtype}")
            bits = chunk.view(torch.uint8)
            if bool((bits == E8M0_NAN_BITS).any()):
                raise ValueError(f"{scale_key}: E8M0 scales must be finite and positive (invalid byte 255)")
            self._host_scale[start:stop].copy_(bits)

        # Two rows per shard exercise both local and remote reads, including
        # tail padding. This tiny extra checkpoint read is only for startup QA.
        ids = sorted(
            {
                row
                for rank in range(self.engram_tp_size)
                for row in (rank * self.shard_rows, (rank + 1) * self.shard_rows - 1)
            }
        )
        self._probe_ids = torch.tensor(ids, dtype=torch.int32, device="cpu")
        self._probe_codes = torch.zeros((len(ids), self.width), dtype=torch.uint8, device="cpu")
        for index, row in enumerate(ids):
            if row < checkpoint_rows:
                self._probe_codes[index].copy_(weight[row : row + 1].view(torch.uint8)[0])
        self._probe_scales = self._host_scale[self._probe_ids.long()].clone()

    def load_checkpoint(
        self,
        model_path: str | Path,
        key: str,
        chunk_rows: int = 8192,
    ) -> None:
        if self._state is not EngramTableState.EMPTY:
            raise RuntimeError(f"Engram table {key} cannot load from state {self._state.value}")
        root = Path(model_path)
        logger.debug(
            "FOR-ENGRAM checkpoint shard loading started: layer=%s key=%s tp_rank=%d row_range=[%d,%d) storage=%s",
            self.layer_id,
            key,
            self.rank,
            self.start,
            self.end,
            self.storage_format,
        )
        local_error: Exception | None = None
        result: tuple[str, int, int, int] | None = None
        try:
            # safetensors may create intermediate tensors using the default
            # device even with device="cpu"; never materialize table shards on NPU.
            with torch.device("cpu"):
                result = self._load_checkpoint_local(root, key, chunk_rows)
        except Exception as exc:
            local_error = exc
            logger.exception(
                "FOR-ENGRAM checkpoint shard local preflight failed: layer=%s key=%s tp_rank=%d",
                self.layer_id,
                key,
                self.rank,
            )

        try:
            all_succeeded = self._sync_phase_status(local_error is None, "checkpoint-load")
        except Exception:
            self._release_staging()
            self._state = EngramTableState.POISONED
            logger.exception(
                "FOR-ENGRAM checkpoint distributed preflight failed: layer=%s tp_rank=%d",
                self.layer_id,
                self.rank,
            )
            raise
        if local_error is not None:
            self._release_staging()
            self._state = EngramTableState.POISONED
            raise local_error
        if not all_succeeded:
            self._release_staging()
            self._state = EngramTableState.POISONED
            raise RuntimeError(
                "Engram checkpoint loading failed on another rank; "
                "all ranks aborted before ElasticBuffer initialization"
            )

        assert result is not None
        resolved_key, checkpoint_rows, copied_rows, block_size = result
        self._state = EngramTableState.STAGED
        logger.debug(
            "FOR-ENGRAM checkpoint shard loading completed: layer=%s key=%s "
            "tp_rank=%d checkpoint_rows=%d configured_rows=%d copied_rows=%d "
            "padded_shard_rows=%d checkpoint_scale_block_size=%d "
            "staged_dtype=%s state=%s",
            self.layer_id,
            resolved_key,
            self.rank,
            checkpoint_rows,
            self.rows,
            copied_rows,
            self.shard_rows,
            block_size,
            self._host_weight.dtype,
            self._state.value,
        )

    def _destroy_buffer_after_failure(self, buffer) -> None:
        if buffer is None:
            return
        self.engram_buffer = buffer
        try:
            buffer.destroy()
            self.engram_buffer = None
        except Exception:
            # Keep ownership: a later destroy() retries the failed release.
            self._state = EngramTableState.POISONED
            logger.exception("FOR-ENGRAM failed to destroy a partial ElasticBuffer")

    def _check_fp8_runtime(self, elastic_buffer_cls) -> None:
        """Reject unsupported wrappers before entering any ElasticBuffer barrier."""
        from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

        if self._target_device().type != "npu" or not get_current_hardware_profile().supports(
            HardwareCapability.ENGRAM_MXFP8
        ):
            raise RuntimeError("Engram MXFP8 ElasticBuffer requires an Ascend 950 NPU installation")
        if self.engram_tp_size < 2:
            raise ValueError("Engram MXFP8 ElasticBuffer requires engram_tp_size >= 2")
        if getattr(torch, "float8_e8m0fnu", None) is None:
            raise RuntimeError("Engram MXFP8 requires torch.float8_e8m0fnu support")
        try:
            constructor = inspect.signature(elastic_buffer_cls)
            constructor.bind(self.group, num_cpu_bytes=0, with_grad=False, explicitly_destroy=True, netlayer="auto")
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Engram MXFP8 requires the current ops-transformer ElasticBuffer constructor "
                "with with_grad=False, explicitly_destroy=True and netlayer='auto'"
            ) from exc
        try:
            signature = inspect.signature(elastic_buffer_cls.engram_write)
            signature.bind(None, None, sf=None, zero_copy=False)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Engram MXFP8 requires ElasticBuffer.engram_write(storage, sf=..., zero_copy=False)"
            ) from exc
        schema = _validate_engram_op_schema(
            "engram_fetch",
            (
                ("context", "Tensor"),
                ("indices", "Tensor"),
                ("hidden_size", "int"),
                ("num_entries", "int"),
                ("dtype", "int"),
                ("sf_table", "Tensor"),
                ("fetched_sf", "Tensor"),
            ),
        )
        wait_schema = _validate_engram_op_schema("engram_fetch_wait", (("context", "Tensor"), ("fetched", "Tensor")))
        logger.debug(
            "FOR-ENGRAM FP8 interface preflight: layer=%s tp_rank=%d constructor_signature=%s write_signature=%s "
            "fetch_schema=%s wait_schema=%s with_grad=false zero_copy=false netlayer=auto "
            "opp_validation=pending-runtime-probe storage_policy=fp8-only",
            self.layer_id,
            self.rank,
            constructor,
            signature,
            schema,
            wait_schema,
        )

    def _prepare_fp8_scale(self) -> None:
        if self._scale_block_size <= 0:
            raise RuntimeError("Engram FP8 scale block size was not initialized")
        expected = (self.padded_rows, self.width // self._scale_block_size)
        scale = self._host_scale
        if (
            scale is None
            or tuple(scale.shape) != expected
            or scale.dtype != torch.uint8
            or not scale.is_cpu
            or not scale.is_contiguous()
        ):
            raise RuntimeError(f"Engram FP8 requires a full contiguous CPU scale byte table {expected}")
        device = self._target_device()
        if device.type == "npu":
            free_bytes, total_bytes = torch.npu.mem_get_info(device)
            logger.debug(
                "FOR-ENGRAM FP8 scale capacity: layer=%s tp_rank=%d shape=%s scale_bytes=%d "
                "free_hbm_bytes=%d total_hbm_bytes=%d replication=full-table-per-rank",
                self.layer_id,
                self.rank,
                expected,
                scale.numel(),
                free_bytes,
                total_bytes,
            )
            if scale.numel() > free_bytes:
                raise RuntimeError(f"Engram E8M0 scale table needs {scale.numel()} HBM bytes; only {free_bytes} free")
        scale_device = torch.empty(expected, dtype=torch.uint8, device=device)
        for start in range(0, self.padded_rows, SCALE_COPY_CHUNK_ROWS):
            stop = min(start + SCALE_COPY_CHUNK_ROWS, self.padded_rows)
            scale_device[start:stop].copy_(scale[start:stop])
        self._scale = scale_device.view(torch.float8_e8m0fnu)
        logger.debug(
            "FOR-ENGRAM FP8 scale ready: layer=%s tp_rank=%d shape=%s dtype=%s device=%s bytes=%d",
            self.layer_id,
            self.rank,
            tuple(self._scale.shape),
            self._scale.dtype,
            self._scale.device,
            self._scale.numel(),
        )

    def _verify_fp8_runtime(self, buffer) -> None:
        """Check actual FP8 reads and dequantization once, before serving requests."""
        local_error = None
        ids = None
        output = None
        try:
            assert self._probe_ids is not None and self._probe_codes is not None and self._probe_scales is not None
            ids = self._probe_ids.to(device=self._target_device())
            output = self._fetch_fp8(buffer, ids)
            decoded = self._validate_fetch(output, ids.numel())
            fetched, fetched_sf = output
            if not torch.equal(fetched.view(torch.uint8).cpu(), self._probe_codes):
                raise RuntimeError("Engram FP8 startup probe: fetched E4M3 bytes differ from checkpoint")
            if not torch.equal(fetched_sf.view(torch.uint8).cpu(), self._probe_scales):
                raise RuntimeError("Engram FP8 startup probe: fetched E8M0 bytes differ from checkpoint")
            expected = (
                (
                    self._probe_codes.view(torch.float8_e4m3fn).float().reshape(ids.numel(), -1, self._scale_block_size)
                    * self._scale_to_float(self._probe_scales).unsqueeze(-1)
                )
                .reshape(ids.numel(), self.width)
                .to(torch.bfloat16)
            )
            # The CPU transfer synchronizes only this bounded startup probe.
            # Inference logging never copies tensors or waits on the device.
            if not torch.equal(decoded.cpu().view(torch.int16), expected.view(torch.int16)):
                raise RuntimeError("Engram FP8 startup probe: BF16 output differs from CPU checkpoint decode")
        except Exception as exc:
            local_error = exc
            self._log_fp8_failure(buffer, ids, stage="fp8-runtime-probe", output=output)
            logger.exception("FOR-ENGRAM FP8 runtime probe failed: layer=%s tp_rank=%d", self.layer_id, self.rank)
        finally:
            if local_error is None:
                log_engram_runtime(self.layer_id, self.rank, stage="post-probe", buffer=buffer)
        all_succeeded = self._sync_phase_status(local_error is None, "fp8-runtime-probe")
        if local_error is not None:
            raise local_error
        if not all_succeeded:
            raise RuntimeError("Engram FP8 runtime probe failed on another rank")
        logger.debug(
            "FOR-ENGRAM FP8 runtime verified: layer=%s tp_rank=%d sample_rows=%d sampled_ranks=%d "
            "write=E4M3 fetch=E4M3 wait=passed scale=E8M0 code_bytes=exact scale_bytes=exact "
            "dequant=FP32-to-BF16 bf16_bits=exact",
            self.layer_id,
            self.rank,
            self._probe_ids.numel(),
            self.engram_tp_size,
        )

    def offload_weights(self) -> None:
        """Write the staged E4M3 shard and E8M0 scale table into ElasticBuffer."""
        logger.debug(
            "FOR-ENGRAM ElasticBuffer write function entered: layer=%s tp_rank=%d state=%s",
            self.layer_id,
            self.rank,
            self._state.value,
        )
        if self._state is EngramTableState.READY:
            return
        if self._state is not EngramTableState.STAGED:
            raise RuntimeError(
                f"Engram checkpoint shard must be staged before offload; current state is {self._state.value}"
            )
        assert self._host_weight is not None
        storage = self._host_weight
        weight_bytes = storage.numel() * storage.element_size()
        logger.debug(
            "FOR-ENGRAM ElasticBuffer offload started: layer=%s tp_rank=%d "
            "shard_rows=%d width=%d storage=%s weight_bytes=%d",
            self.layer_id,
            self.rank,
            self.shard_rows,
            self.width,
            self.storage_format,
            weight_bytes,
        )

        local_error: Exception | None = None
        elastic_buffer_cls = None
        buffer = None
        num_cpu_bytes = 0
        try:
            from cann_ops_transformer.ops import ElasticBuffer as elastic_buffer_cls

            log_engram_runtime(self.layer_id, self.rank, stage="pre-write")
            expected_weight_shape = (self.shard_rows, self.width)
            if not storage.is_cpu or not storage.is_contiguous() or tuple(storage.shape) != expected_weight_shape:
                raise RuntimeError(
                    "Engram staged weight must be a contiguous CPU tensor with "
                    f"shape {expected_weight_shape}, got {storage.device} "
                    f"{tuple(storage.shape)}"
                )
            expected_dtype = torch.float8_e4m3fn
            if storage.dtype != expected_dtype:
                raise TypeError(f"Engram {self.storage_format} weight must be {expected_dtype}, got {storage.dtype}")
            self._check_fp8_runtime(elastic_buffer_cls)
            self._prepare_fp8_scale()
            num_cpu_bytes = elastic_buffer_cls.get_engram_storage_size_hint(
                self.shard_rows,
                self.width,
                storage.dtype,
            )
            if (
                not isinstance(num_cpu_bytes, int)
                or isinstance(num_cpu_bytes, bool)
                or num_cpu_bytes < weight_bytes
                or num_cpu_bytes % ENGRAM_STORAGE_ALIGNMENT
            ):
                raise RuntimeError(
                    "Engram storage size hint must be an integer capacity aligned to 2 MiB "
                    f"and cover the weight payload; hint={num_cpu_bytes}, weight_bytes={weight_bytes}"
                )
        except Exception as exc:
            local_error = exc
            logger.exception(
                "FOR-ENGRAM ElasticBuffer format preflight failed: layer=%s tp_rank=%d storage=%s",
                self.layer_id,
                self.rank,
                self.storage_format,
            )

        try:
            all_succeeded = self._sync_phase_status(local_error is None, "offload-format")
        except Exception:
            self._release_staging()
            self._scale = None
            self._state = EngramTableState.POISONED
            logger.exception(
                "FOR-ENGRAM ElasticBuffer format synchronization failed: layer=%s tp_rank=%d",
                self.layer_id,
                self.rank,
            )
            raise
        if local_error is not None or not all_succeeded:
            self._release_staging()
            self._scale = None
            self._state = EngramTableState.POISONED
            if local_error is not None:
                raise local_error
            raise RuntimeError(
                "Engram offload format preparation failed on another rank; "
                "all ranks aborted before ElasticBuffer initialization"
            )

        assert elastic_buffer_cls is not None
        local_error = None
        try:
            buffer = elastic_buffer_cls(
                self.group,
                num_cpu_bytes=num_cpu_bytes,
                with_grad=False,
                explicitly_destroy=True,
                netlayer="auto",
            )
        except Exception as exc:
            local_error = exc
            logger.exception(
                "FOR-ENGRAM ElasticBuffer initialization failed: layer=%s tp_rank=%d storage=%s",
                self.layer_id,
                self.rank,
                self.storage_format,
            )
        try:
            all_succeeded = self._sync_phase_status(local_error is None, "elastic-buffer-init")
        except Exception:
            self._destroy_buffer_after_failure(buffer)
            self._release_staging()
            self._scale = None
            self._state = EngramTableState.POISONED
            logger.exception(
                "FOR-ENGRAM ElasticBuffer initialization synchronization failed: layer=%s tp_rank=%d",
                self.layer_id,
                self.rank,
            )
            raise
        if local_error is not None or not all_succeeded:
            self._destroy_buffer_after_failure(buffer)
            self._release_staging()
            self._scale = None
            self._state = EngramTableState.POISONED
            if local_error is not None:
                raise local_error
            raise RuntimeError(
                "Engram ElasticBuffer initialization failed on another rank; all ranks aborted before engram_write"
            )

        assert buffer is not None
        local_error = None
        try:
            # Inference owns an internal host copy; staging can be released
            # after the startup probe without retaining a zero-copy mapping.
            buffer.engram_write(storage, sf=self._scale, zero_copy=False)
        except Exception as exc:
            local_error = exc
            logger.exception(
                "FOR-ENGRAM ElasticBuffer write failed: layer=%s tp_rank=%d num_cpu_bytes=%d storage=%s",
                self.layer_id,
                self.rank,
                num_cpu_bytes,
                self.storage_format,
            )
        try:
            all_succeeded = self._sync_phase_status(local_error is None, "elastic-buffer-write")
        except Exception:
            self._destroy_buffer_after_failure(buffer)
            self._release_staging()
            self._scale = None
            self._state = EngramTableState.POISONED
            logger.exception(
                "FOR-ENGRAM ElasticBuffer write synchronization failed: layer=%s tp_rank=%d",
                self.layer_id,
                self.rank,
            )
            raise
        if local_error is not None or not all_succeeded:
            self._destroy_buffer_after_failure(buffer)
            self._release_staging()
            self._scale = None
            self._state = EngramTableState.POISONED
            if local_error is not None:
                raise local_error
            raise RuntimeError("Engram ElasticBuffer write failed on another rank; all ranks aborted before probe")

        try:
            self._verify_fp8_runtime(buffer)
        except Exception:
            self._destroy_buffer_after_failure(buffer)
            self._release_staging()
            self._scale = None
            self._state = EngramTableState.POISONED
            logger.exception(
                "FOR-ENGRAM ElasticBuffer runtime validation failed: layer=%s tp_rank=%d num_cpu_bytes=%d storage=%s",
                self.layer_id,
                self.rank,
                num_cpu_bytes,
                self.storage_format,
            )
            raise

        self.engram_buffer = buffer
        self._release_staging()
        self._state = EngramTableState.READY
        logger.debug(
            "FOR-ENGRAM ElasticBuffer offload completed: layer=%s tp_rank=%d "
            "num_cpu_bytes=%d storage=%s staging_released=true state=%s",
            self.layer_id,
            self.rank,
            num_cpu_bytes,
            self.storage_format,
            self._state.value,
        )

    def _validate_fetch(
        self,
        output,
        expected_rows: int,
    ) -> torch.Tensor:
        expected_weight_shape = (expected_rows, self.width)
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("Engram MXFP8 fetch must return (E4M3 data, E8M0 scale)")
        fetched, scales = output
        if self._scale_block_size <= 0:
            raise RuntimeError("Engram FP8 scale block size was not initialized")
        expected_scale_shape = (expected_rows, self.width // self._scale_block_size)
        if (
            not isinstance(fetched, torch.Tensor)
            or not isinstance(scales, torch.Tensor)
            or fetched.dtype != torch.float8_e4m3fn
            or scales.dtype != torch.float8_e8m0fnu
            or tuple(fetched.shape) != expected_weight_shape
            or tuple(scales.shape) != expected_scale_shape
            or fetched.device != scales.device
            or not fetched.is_contiguous()
            or not scales.is_contiguous()
            or (self._scale is not None and fetched.device != self._scale.device)
        ):
            actual = [
                (tuple(t.shape), str(t.dtype), str(t.device), t.is_contiguous())
                if isinstance(t, torch.Tensor)
                else type(t).__name__
                for t in output
            ]
            raise RuntimeError(
                f"Engram MXFP8 fetch expects contiguous E4M3 {expected_weight_shape} and "
                f"E8M0 {expected_scale_shape} on the scale-table device; "
                f"actual (shape, dtype, device, contiguous)={actual}"
            )
        # Construct FP32 scales without a numerical uint8 -> E8M0 cast.
        # E8M0 byte 0 is 2**-127, not zero; byte 255 remains NaN.
        exponent = scales.view(torch.uint8).to(torch.int32)
        bits = torch.where(exponent == 0, FP32_E8M0_MIN_BITS, exponent << FP32_EXPONENT_SHIFT)
        bits = torch.where(exponent == E8M0_NAN_BITS, FP32_NAN_BITS, bits)
        scale_fp32 = bits.view(torch.float32)
        return (
            fetched.float()
            .reshape(expected_rows, expected_scale_shape[1], self._scale_block_size)
            .mul_(scale_fp32.unsqueeze(-1))
            .reshape(expected_weight_shape)
            .to(torch.bfloat16)
        )

    def _log_fp8_failure(self, buffer, ids, *, stage: str, output=None) -> None:
        """Report the first failure per table using descriptors, without device sync."""
        if self._failure_diagnostics_logged:
            return
        self._failure_diagnostics_logged = True
        try:
            tensors = {
                "indices": ids,
                "staged_weight": self._host_weight,
                "sf_table": self._scale,
                "context": getattr(buffer, "_engram_context_tensor", None),
                "buffer_sf_table": getattr(buffer, "_engram_sf", None),
            }
            if isinstance(output, tuple) and len(output) == 2:
                tensors.update(fetched=output[0], fetched_sf=output[1])
            rows = ids.numel() if isinstance(ids, torch.Tensor) else None
            log_engram_runtime(
                self.layer_id,
                self.rank,
                stage=stage,
                buffer=buffer,
                failure=True,
                tensors=tensors,
                parameters={
                    "storage": self.storage_format,
                    "table_state": self._state.value,
                    "group_size": self.engram_tp_size,
                    "global_rows": self.rows,
                    "padded_rows": self.padded_rows,
                    "num_entries_per_rank": self.shard_rows,
                    "hidden_size": self.width,
                    "buffer_num_entries_per_rank": getattr(buffer, "_engram_num_entries", None),
                    "buffer_hidden_size": getattr(buffer, "_engram_hidden_size", None),
                    "buffer_dtype": str(getattr(buffer, "_engram_dtype", "unavailable")),
                    "scale_block_size": self._scale_block_size,
                    "expected_fetch_shape": (rows, self.width),
                    "expected_fetch_dtype": str(torch.float8_e4m3fn),
                    "expected_fetched_sf_shape": (
                        (rows, self.width // self._scale_block_size) if self._scale_block_size > 0 else None
                    ),
                    "expected_fetched_sf_dtype": str(getattr(torch, "float8_e8m0fnu", "unavailable")),
                },
            )
        except Exception as exc:
            logger.warning(
                "FOR-ENGRAM failure diagnostics unavailable: layer=%s tp_rank=%d stage=%s reason=%s",
                self.layer_id,
                self.rank,
                stage,
                exc,
            )

    def _start_fetch_fp8(
        self,
        buffer,
        flat_ids: torch.Tensor,
    ) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
        try:
            wait = buffer.engram_fetch(flat_ids)
        except Exception as exc:
            self._log_fp8_failure(buffer, flat_ids, stage="fetch-submit")
            raise RuntimeError(
                "Engram MXFP8 fetch submission failed (stage=fetch-submit). "
                "See the chained native error and FOR-ENGRAM runtime diagnostics. "
                "The native GetWorkspaceSize/execute stage is not determined by this wrapper."
            ) from exc
        if not callable(wait):
            self._log_fp8_failure(buffer, flat_ids, stage="fetch-submit")
            raise RuntimeError("Engram MXFP8 fetch must return a completion callback")

        def wait_for_result() -> tuple[torch.Tensor, torch.Tensor]:
            try:
                return wait()
            except Exception as exc:
                self._log_fp8_failure(buffer, flat_ids, stage="fetch-wait")
                raise RuntimeError(
                    "Engram MXFP8 fetch completion failed (stage=fetch-wait). "
                    "See the chained native error and FOR-ENGRAM runtime diagnostics."
                ) from exc

        return wait_for_result

    def _fetch_fp8(self, buffer, flat_ids: torch.Tensor):
        return self._start_fetch_fp8(buffer, flat_ids)()

    @torch.inference_mode()
    def start_fetch(self, ids: torch.Tensor) -> EngramFetchHandle:
        """Launch one FP8 fetch without consuming its completion callback."""
        if self._state is not EngramTableState.READY or self.engram_buffer is None:
            raise RuntimeError(f"Engram ElasticBuffer is not ready; current state is {self._state.value}")
        if self._fetch_in_flight:
            raise RuntimeError("Engram does not permit overlapping fetches on one table")
        shape = tuple(ids.shape)
        flat_ids = ids.reshape(-1).to(dtype=torch.int32).contiguous()
        self._fetch_in_flight = True
        try:
            wait = self._start_fetch_fp8(self.engram_buffer, flat_ids)
        except Exception:
            self._state = EngramTableState.POISONED
            logger.exception(
                "FOR-ENGRAM ElasticBuffer fetch launch failed and table was poisoned: layer=%s tp_rank=%d",
                self.layer_id,
                self.rank,
            )
            raise
        return EngramFetchHandle(owner=self, shape=shape, flat_ids=flat_ids, wait=wait)

    @torch.inference_mode()
    def finish_fetch(self, handle: EngramFetchHandle) -> torch.Tensor:
        """Wait for, validate, and dequantize a fetch from :meth:`start_fetch`."""
        if handle.owner is not self:
            raise ValueError("Engram fetch handle belongs to another table")
        if handle.completed:
            raise RuntimeError("Engram fetch handle has already been completed")
        if not self._fetch_in_flight:
            raise RuntimeError("Engram table has no fetch in flight")
        handle.completed = True
        try:
            output = handle.wait()
        except Exception:
            self._state = EngramTableState.POISONED
            logger.exception(
                "FOR-ENGRAM ElasticBuffer fetch wait failed and table was poisoned: layer=%s tp_rank=%d",
                self.layer_id,
                self.rank,
            )
            raise
        self._fetch_in_flight = False
        try:
            result = self._validate_fetch(output, handle.flat_ids.numel())
        except Exception:
            self._log_fp8_failure(self.engram_buffer, handle.flat_ids, stage="fetch-validation", output=output)
            self._state = EngramTableState.POISONED
            logger.exception(
                "FOR-ENGRAM ElasticBuffer fetch ABI validation failed: layer=%s tp_rank=%d storage=%s",
                self.layer_id,
                self.rank,
                self.storage_format,
            )
            raise
        if handle.flat_ids.numel() and not self._fetch_logged:
            fetched, scales = output
            logger.debug(
                "FOR-ENGRAM inference fetch active: layer=%s tp_rank=%d storage=%s fetch_dtype=%s "
                "fetch_shape=%s output_dtype=%s output_shape=%s device=%s "
                "scale_block_size=%d scale_dtype=%s wait=callback-returned dequant=%s",
                self.layer_id,
                self.rank,
                self.storage_format,
                fetched.dtype,
                tuple(fetched.shape),
                result.dtype,
                tuple(result.shape),
                result.device,
                self._scale_block_size,
                scales.dtype,
                "FP32-to-BF16",
            )
            self._fetch_logged = True
        return result.view(*handle.shape, self.width)

    @torch.inference_mode()
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        logger.debug(
            "FOR-ENGRAM ElasticBuffer fetch function entered: layer=%s tp_rank=%d ids_shape=%s state=%s",
            self.layer_id,
            self.rank,
            tuple(ids.shape),
            self._state.value,
        )
        return self.finish_fetch(self.start_fetch(ids))

    def destroy(self) -> None:
        if self._state is EngramTableState.DESTROYED:
            return
        if self._fetch_in_flight and self._state is not EngramTableState.POISONED:
            raise RuntimeError("Cannot destroy Engram while a fetch is in flight")
        logger.debug(
            "FOR-ENGRAM table destroy started: layer=%s tp_rank=%d buffer_initialized=%s owns_group=%s state=%s",
            self.layer_id,
            self.rank,
            self.engram_buffer is not None,
            self.owns_group,
            self._state.value,
        )
        try:
            if self.engram_buffer is not None:
                self.engram_buffer.destroy()
                self.engram_buffer = None
            if self.owns_group and self.group is not None and dist.is_initialized():
                dist.destroy_process_group(self.group)
        except BaseException:
            self._state = EngramTableState.POISONED
            logger.exception(
                "FOR-ENGRAM table destroy failed: layer=%s tp_rank=%d",
                self.layer_id,
                self.rank,
            )
            raise
        self.group = None
        self._release_staging()
        self._scale = None
        self._fetch_in_flight = False
        self._state = EngramTableState.DESTROYED
        logger.debug(
            "FOR-ENGRAM table destroy completed: layer=%s tp_rank=%d state=%s",
            self.layer_id,
            self.rank,
            self._state.value,
        )
