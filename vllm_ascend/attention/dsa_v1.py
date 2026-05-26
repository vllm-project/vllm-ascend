import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, TypeVar

import torch
import torch.nn.functional as F
import torch_npu
import vllm.envs as envs_vllm
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backend import AttentionBackend, AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec

from vllm_ascend import envs
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.abstract import DSAAttentionImpl
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, split_decodes_and_prefills
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import (
    AscendDeviceType,
    attention_calculation_stream,
    get_ascend_device_type,
    npu_stream_switch,
    olora_tp_enable,
)
from vllm_ascend.worker.npu_input_batch import NPUInputBatch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_ascend.ops.triton.rms_norm import triton_q_rms

if HAS_TRITON:
    from vllm_ascend.ops.triton.rms_norm import triton_q_rms  # noqa: F811
else:
    triton_q_rms = None  # type: ignore

BUILD_METADATA_STEP_PREFILL = 0
BUILD_METADATA_STEP_DECODE = 1
_DSA_DEBUG_NONFINITE_LOGGED: set[tuple[str, str, int, str]] = set()
_DSA_DEBUG_SLOT_LOGGED: set[tuple[str, str, int, str]] = set()
_DSA_DEBUG_COMPRESSED_BAD_SLOT_LOGGED: set[tuple[str, str, int, str]] = set()
_DSA_DEBUG_BAD_SLOT_LOGGED: set[tuple[str, str, int, str]] = set()
_DSA_DEBUG_SAS_INPUT_LOGGED: set[tuple[str, str, int, str]] = set()
_DSA_DEBUG_ROW_MISMATCH_LOGGED: set[tuple[str, str, int, str]] = set()
_DSA_DEBUG_DECODE_SLOT_COLLISION_LOGGED: set[tuple[str, str, int, str]] = set()

# mypy: disable-error-code="has-type"


def _dsa_debug_metadata_summary(attn_metadata) -> str:
    metadata = attn_metadata[0] if isinstance(attn_metadata, list) else attn_metadata
    if metadata is None:
        return "metadata=None"
    return (
        f"num_decodes={getattr(metadata, 'num_decodes', None)} "
        f"num_prefills={getattr(metadata, 'num_prefills', None)} "
        f"num_decode_tokens={getattr(metadata, 'num_decode_tokens', None)} "
        f"num_actual_tokens={getattr(metadata, 'num_actual_tokens', None)} "
        f"num_input_tokens={getattr(metadata, 'num_input_tokens', None)}"
    )


def _dsa_debug_should_check(attn_metadata, phase: str) -> bool:
    metadata = attn_metadata[0] if isinstance(attn_metadata, list) else attn_metadata
    if metadata is None:
        return False
    if getattr(metadata, "num_prefills", 0) > 0:
        return True
    if phase != "decode":
        return False
    if envs.VLLM_ASCEND_DSA_DEBUG_DECODE_GRAPH:
        return True
    forward_context = get_forward_context()
    return getattr(forward_context, "cudagraph_runtime_mode", CUDAGraphMode.NONE) == CUDAGraphMode.NONE


def _dsa_debug_check_finite(
    tensor: torch.Tensor | None,
    layer_name: str,
    phase: str,
    stage: str,
    compress_ratio: int,
    attn_metadata=None,
) -> bool:
    if not _dsa_debug_should_check(attn_metadata, phase):
        return True
    if tensor is None or tensor.numel() == 0 or not tensor.is_floating_point():
        return True

    try:
        is_finite = torch.isfinite(tensor)
        if bool(is_finite.all().item()):
            return True
        nan_count = int(torch.isnan(tensor).sum().item())
        inf_count = int(torch.isinf(tensor).sum().item())
        finite_count = int(is_finite.sum().item())
        all_nan = nan_count == tensor.numel()
        all_nonfinite = finite_count == 0
        bad_rows_head = []
        if tensor.ndim > 0 and tensor.shape[0] > 0:
            row_is_finite = is_finite.reshape(tensor.shape[0], -1).all(dim=1)
            bad_rows = torch.nonzero(~row_is_finite, as_tuple=False).reshape(-1)
            bad_rows_head = bad_rows[: min(16, bad_rows.numel())].detach().cpu().tolist()
    except RuntimeError as exc:
        logger.warning(
            "[DSA_DEBUG_NAN] finite check failed: layer=%s phase=%s "
            "stage=%s compress_ratio=%s shape=%s dtype=%s error=%s",
            layer_name,
            phase,
            stage,
            compress_ratio,
            tuple(tensor.shape),
            tensor.dtype,
            exc,
        )
        return True

    key = (layer_name, phase, compress_ratio, stage)
    if key not in _DSA_DEBUG_NONFINITE_LOGGED:
        _DSA_DEBUG_NONFINITE_LOGGED.add(key)
        forward_context = get_forward_context()
        logger.warning(
            "[DSA_DEBUG_NAN] non-finite tensor: layer=%s phase=%s "
            "stage=%s compress_ratio=%s shape=%s dtype=%s nan_count=%s "
            "inf_count=%s finite_count=%s numel=%s all_nan=%s "
            "all_nonfinite=%s bad_rows=%s mode=%s num_tokens=%s %s",
            layer_name,
            phase,
            stage,
            compress_ratio,
            tuple(tensor.shape),
            tensor.dtype,
            nan_count,
            inf_count,
            finite_count,
            tensor.numel(),
            all_nan,
            all_nonfinite,
            bad_rows_head,
            getattr(forward_context, "cudagraph_runtime_mode", None),
            getattr(forward_context, "num_tokens", None),
            _dsa_debug_metadata_summary(attn_metadata),
        )
    return False


def _dsa_debug_log_slot_mapping(
    slot_mapping: torch.Tensor | None,
    layer_name: str,
    phase: str,
    stage: str,
    compress_ratio: int,
    attn_metadata=None,
) -> None:
    if not _dsa_debug_should_check(attn_metadata, phase):
        return
    if slot_mapping is None or slot_mapping.numel() == 0:
        return

    key = (layer_name, phase, compress_ratio, stage)
    if key in _DSA_DEBUG_SLOT_LOGGED:
        return
    _DSA_DEBUG_SLOT_LOGGED.add(key)

    try:
        valid_mask = slot_mapping >= 0
        valid_count = int(valid_mask.sum().item())
        slot_min = int(slot_mapping.min().item())
        slot_max = int(slot_mapping.max().item())
        slot_head = slot_mapping.reshape(-1)[: min(16, slot_mapping.numel())].detach().cpu().tolist()
    except RuntimeError as exc:
        logger.warning(
            "[DSA_DEBUG_SLOT] slot check failed: layer=%s phase=%s "
            "stage=%s compress_ratio=%s slot_shape=%s error=%s %s",
            layer_name,
            phase,
            stage,
            compress_ratio,
            tuple(slot_mapping.shape),
            exc,
            _dsa_debug_metadata_summary(attn_metadata),
        )
        return

    logger.warning(
        "[DSA_DEBUG_SLOT] layer=%s phase=%s stage=%s compress_ratio=%s "
        "slot_shape=%s slot_numel=%s valid_slot_count=%s slot_min=%s "
        "slot_max=%s slot_head=%s %s",
        layer_name,
        phase,
        stage,
        compress_ratio,
        tuple(slot_mapping.shape),
        slot_mapping.numel(),
        valid_count,
        slot_min,
        slot_max,
        slot_head,
        _dsa_debug_metadata_summary(attn_metadata),
    )


def _dsa_debug_log_bad_slot_rows(
    slot_mapping: torch.Tensor | None,
    layer_name: str,
    phase: str,
    stage: str,
    compress_ratio: int,
    row_count: int | None = None,
    attn_metadata=None,
) -> None:
    if not _dsa_debug_should_check(attn_metadata, phase):
        return
    if slot_mapping is None or slot_mapping.numel() == 0:
        return

    flat_slot_mapping = slot_mapping.reshape(-1)
    if row_count is None:
        row_count = flat_slot_mapping.shape[0]
    row_count = min(row_count, flat_slot_mapping.shape[0])
    if row_count == 0:
        return
    flat_slot_mapping = flat_slot_mapping[:row_count]
    valid_mask = flat_slot_mapping >= 0

    try:
        valid_count = int(valid_mask.sum().item())
    except RuntimeError as exc:
        logger.warning(
            "[DSA_DEBUG_BAD_SLOT] valid mask check failed: layer=%s "
            "phase=%s stage=%s compress_ratio=%s slot_shape=%s "
            "row_count=%s error=%s %s",
            layer_name,
            phase,
            stage,
            compress_ratio,
            tuple(slot_mapping.shape),
            row_count,
            exc,
            _dsa_debug_metadata_summary(attn_metadata),
        )
        return
    if valid_count == row_count:
        return

    key = (layer_name, phase, compress_ratio, stage)
    if key in _DSA_DEBUG_BAD_SLOT_LOGGED:
        return
    _DSA_DEBUG_BAD_SLOT_LOGGED.add(key)
    try:
        bad_rows = torch.nonzero(~valid_mask, as_tuple=False).reshape(-1)
        bad_rows_head = bad_rows[: min(16, bad_rows.numel())].detach().cpu().tolist()
        bad_slots_head = flat_slot_mapping[bad_rows[: min(16, bad_rows.numel())]].detach().cpu().tolist()
    except RuntimeError as exc:
        bad_rows_head = f"error={exc}"
        bad_slots_head = f"error={exc}"

    logger.warning(
        "[DSA_DEBUG_BAD_SLOT] layer=%s phase=%s stage=%s "
        "compress_ratio=%s slot_shape=%s row_count=%s valid_slot_count=%s "
        "bad_rows=%s bad_row_slots=%s %s",
        layer_name,
        phase,
        stage,
        compress_ratio,
        tuple(slot_mapping.shape),
        row_count,
        valid_count,
        bad_rows_head,
        bad_slots_head,
        _dsa_debug_metadata_summary(attn_metadata),
    )


def _dsa_debug_log_row_mismatch(
    kv_rows: int,
    slot_rows: int,
    layer_name: str,
    phase: str,
    stage: str,
    compress_ratio: int,
    attn_metadata=None,
) -> None:
    if not _dsa_debug_should_check(attn_metadata, phase):
        return
    if kv_rows == slot_rows:
        return
    key = (layer_name, phase, compress_ratio, stage)
    if key in _DSA_DEBUG_ROW_MISMATCH_LOGGED:
        return
    _DSA_DEBUG_ROW_MISMATCH_LOGGED.add(key)
    logger.warning(
        "[DSA_DEBUG_ROW_MISMATCH] layer=%s phase=%s stage=%s "
        "compress_ratio=%s kv_rows=%s slot_rows=%s %s",
        layer_name,
        phase,
        stage,
        compress_ratio,
        kv_rows,
        slot_rows,
        _dsa_debug_metadata_summary(attn_metadata),
    )


def _dsa_debug_tensor_head(tensor: torch.Tensor | None, limit: int = 16):
    if tensor is None or tensor.numel() == 0:
        return []
    return tensor.reshape(-1)[: min(limit, tensor.numel())].detach().cpu().tolist()


def _dsa_debug_log_sas_inputs(
    metadata,
    layer_name: str,
    phase: str,
    stage: str,
    compress_ratio: int,
    attn_metadata=None,
) -> None:
    if not _dsa_debug_should_check(attn_metadata, phase):
        return
    if metadata is None:
        return

    key = (layer_name, phase, compress_ratio, stage)
    if key in _DSA_DEBUG_SAS_INPUT_LOGGED:
        return
    _DSA_DEBUG_SAS_INPUT_LOGGED.add(key)

    query_start_loc = getattr(metadata, "query_start_loc", None)
    seq_lens = getattr(metadata, "seq_lens", None)
    start_pos = getattr(metadata, "start_pos", None)
    block_table = getattr(metadata, "block_table", None)
    slot_mapping = getattr(metadata, "slot_mapping", None)
    sas_metadata = getattr(metadata, "sas_metadata", None)

    try:
        valid_slot_count = None
        slot_min = None
        slot_max = None
        block_table_first_col_head = None
        if block_table is not None and block_table.ndim >= 2 and block_table.shape[0] > 0:
            block_table_first_col_head = _dsa_debug_tensor_head(block_table[:, 0])
        if slot_mapping is not None and slot_mapping.numel() > 0:
            valid_slot_count = int((slot_mapping.reshape(-1) >= 0).sum().item())
            slot_min = int(slot_mapping.min().item())
            slot_max = int(slot_mapping.max().item())
        logger.warning(
            "[DSA_DEBUG_SAS_INPUT] layer=%s phase=%s stage=%s "
            "compress_ratio=%s q_start_shape=%s q_start_head=%s "
            "seq_lens_shape=%s seq_lens_head=%s start_pos_shape=%s "
            "start_pos_head=%s block_table_shape=%s block_table_head=%s "
            "block_table_first_col_head=%s "
            "slot_shape=%s slot_valid_count=%s slot_min=%s slot_max=%s "
            "slot_head=%s sas_shape=%s sas_head=%s %s",
            layer_name,
            phase,
            stage,
            compress_ratio,
            tuple(query_start_loc.shape) if query_start_loc is not None else None,
            _dsa_debug_tensor_head(query_start_loc),
            tuple(seq_lens.shape) if seq_lens is not None else None,
            _dsa_debug_tensor_head(seq_lens),
            tuple(start_pos.shape) if start_pos is not None else None,
            _dsa_debug_tensor_head(start_pos),
            tuple(block_table.shape) if block_table is not None else None,
            _dsa_debug_tensor_head(block_table),
            block_table_first_col_head,
            tuple(slot_mapping.shape) if slot_mapping is not None else None,
            valid_slot_count,
            slot_min,
            slot_max,
            _dsa_debug_tensor_head(slot_mapping),
            tuple(sas_metadata.shape) if sas_metadata is not None else None,
            _dsa_debug_tensor_head(sas_metadata),
            _dsa_debug_metadata_summary(attn_metadata),
        )
    except RuntimeError as exc:
        logger.warning(
            "[DSA_DEBUG_SAS_INPUT] metadata summary failed: layer=%s "
            "phase=%s stage=%s compress_ratio=%s error=%s %s",
            layer_name,
            phase,
            stage,
            compress_ratio,
            exc,
            _dsa_debug_metadata_summary(attn_metadata),
        )


def _dsa_select_valid_compressed_prefill_rows(
    compressed_kv: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    layer_name: str,
    phase: str,
    stage: str,
    compress_ratio: int,
    attn_metadata=None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    if not _dsa_debug_should_check(attn_metadata, phase):
        return compressed_kv, slot_mapping
    if compressed_kv is None or compressed_kv.numel() == 0 or slot_mapping.numel() == 0:
        return compressed_kv, slot_mapping

    flat_compressed_kv = compressed_kv.reshape(-1, compressed_kv.shape[-1])
    flat_slot_mapping = slot_mapping.reshape(-1)
    kv_rows = flat_compressed_kv.shape[0]
    slot_rows = flat_slot_mapping.shape[0]
    _dsa_debug_log_row_mismatch(
        kv_rows,
        slot_rows,
        layer_name,
        phase,
        stage,
        compress_ratio,
        attn_metadata,
    )
    row_count = min(flat_compressed_kv.shape[0], flat_slot_mapping.shape[0])
    if row_count == 0:
        return compressed_kv, slot_mapping

    aligned_compressed_kv = flat_compressed_kv[:row_count]
    flat_slot_mapping = flat_slot_mapping[:row_count]
    valid_mask = flat_slot_mapping >= 0

    try:
        valid_count = int(valid_mask.sum().item())
    except RuntimeError as exc:
        logger.warning(
            "[DSA_DEBUG_COMPRESSED_BAD_SLOT] valid mask check failed: "
            "layer=%s phase=%s stage=%s compress_ratio=%s "
            "compressed_shape=%s slot_shape=%s error=%s %s",
            layer_name,
            phase,
            stage,
            compress_ratio,
            tuple(compressed_kv.shape),
            tuple(slot_mapping.shape),
            exc,
            _dsa_debug_metadata_summary(attn_metadata),
        )
        return compressed_kv, slot_mapping

    if valid_count == row_count:
        if kv_rows != slot_rows:
            return aligned_compressed_kv, flat_slot_mapping
        return compressed_kv, flat_slot_mapping

    key = (layer_name, phase, compress_ratio, stage)
    if key not in _DSA_DEBUG_COMPRESSED_BAD_SLOT_LOGGED:
        _DSA_DEBUG_COMPRESSED_BAD_SLOT_LOGGED.add(key)
        try:
            bad_rows = torch.nonzero(~valid_mask, as_tuple=False).reshape(-1)
            bad_rows_head = bad_rows[: min(16, bad_rows.numel())].detach().cpu().tolist()
            bad_slots_head = flat_slot_mapping[bad_rows[: min(16, bad_rows.numel())]].detach().cpu().tolist()
        except RuntimeError as exc:
            bad_rows_head = f"error={exc}"
            bad_slots_head = f"error={exc}"
        logger.warning(
            "[DSA_DEBUG_COMPRESSED_BAD_SLOT] layer=%s phase=%s stage=%s "
            "compress_ratio=%s compressed_shape=%s slot_shape=%s "
            "valid_slot_count=%s row_count=%s bad_rows=%s bad_row_slots=%s %s",
            layer_name,
            phase,
            stage,
            compress_ratio,
            tuple(compressed_kv.shape),
            tuple(slot_mapping.shape),
            valid_count,
            row_count,
            bad_rows_head,
            bad_slots_head,
            _dsa_debug_metadata_summary(attn_metadata),
        )

    if valid_count == 0:
        return None, flat_slot_mapping[:0]

    return aligned_compressed_kv[valid_mask], flat_slot_mapping[valid_mask]


def _dsa_select_valid_prefill_rows(
    kv: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    layer_name: str,
    phase: str,
    stage: str,
    compress_ratio: int,
    attn_metadata=None,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    if not _dsa_debug_should_check(attn_metadata, phase):
        return kv, slot_mapping
    if kv is None or kv.numel() == 0 or slot_mapping.numel() == 0:
        return kv, slot_mapping

    flat_kv = kv.reshape(-1, kv.shape[-1])
    flat_slot_mapping = slot_mapping.reshape(-1)
    kv_rows = flat_kv.shape[0]
    slot_rows = flat_slot_mapping.shape[0]
    _dsa_debug_log_row_mismatch(
        kv_rows,
        slot_rows,
        layer_name,
        phase,
        stage,
        compress_ratio,
        attn_metadata,
    )
    row_count = min(flat_kv.shape[0], flat_slot_mapping.shape[0])
    if row_count == 0:
        return kv, slot_mapping

    aligned_kv = flat_kv[:row_count]
    flat_slot_mapping = flat_slot_mapping[:row_count]
    valid_mask = flat_slot_mapping >= 0

    try:
        valid_count = int(valid_mask.sum().item())
    except RuntimeError as exc:
        logger.warning(
            "[DSA_DEBUG_BAD_SLOT] valid mask check failed: layer=%s "
            "phase=%s stage=%s compress_ratio=%s kv_shape=%s "
            "slot_shape=%s error=%s %s",
            layer_name,
            phase,
            stage,
            compress_ratio,
            tuple(kv.shape),
            tuple(slot_mapping.shape),
            exc,
            _dsa_debug_metadata_summary(attn_metadata),
        )
        return kv, slot_mapping

    if valid_count == row_count:
        if kv_rows != slot_rows:
            return aligned_kv, flat_slot_mapping
        return kv, flat_slot_mapping

    _dsa_debug_log_bad_slot_rows(
        slot_mapping,
        layer_name,
        phase,
        stage,
        compress_ratio,
        row_count,
        attn_metadata,
    )

    if valid_count == 0:
        return None, flat_slot_mapping[:0]

    return aligned_kv[valid_mask], flat_slot_mapping[valid_mask]


def _dsa_debug_log_decode_slot_collision(
    slot_mapping: torch.Tensor | None,
    layer_name: str,
    phase: str,
    stage: str,
    compress_ratio: int,
    attn_metadata=None,
) -> None:
    if not _dsa_debug_should_check(attn_metadata, phase):
        return
    if slot_mapping is None or slot_mapping.numel() <= 1:
        return

    key = (layer_name, phase, compress_ratio, stage)
    if key in _DSA_DEBUG_DECODE_SLOT_COLLISION_LOGGED:
        return

    try:
        flat_slot_mapping = slot_mapping.reshape(-1)
        valid_slot_mapping = flat_slot_mapping[flat_slot_mapping >= 0]
        if valid_slot_mapping.numel() <= 1:
            return
        unique_slots = torch.unique(valid_slot_mapping)
        if unique_slots.numel() == valid_slot_mapping.numel():
            return
        _DSA_DEBUG_DECODE_SLOT_COLLISION_LOGGED.add(key)
        logger.warning(
            "[DSA_DEBUG_DECODE_SLOT_COLLISION] layer=%s phase=%s stage=%s "
            "compress_ratio=%s valid_slot_count=%s unique_slot_count=%s "
            "slot_head=%s %s",
            layer_name,
            phase,
            stage,
            compress_ratio,
            int(valid_slot_mapping.numel()),
            int(unique_slots.numel()),
            flat_slot_mapping[: min(16, flat_slot_mapping.numel())].detach().cpu().tolist(),
            _dsa_debug_metadata_summary(attn_metadata),
        )
    except RuntimeError as exc:
        logger.warning(
            "[DSA_DEBUG_DECODE_SLOT_COLLISION] check failed: layer=%s "
            "phase=%s stage=%s compress_ratio=%s error=%s %s",
            layer_name,
            phase,
            stage,
            compress_ratio,
            exc,
            _dsa_debug_metadata_summary(attn_metadata),
        )


def hadamard_transform_ref(
    x: torch.Tensor,
    hadamard: torch.Tensor,
    scale: float = 1.0,
):
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, hadamard)
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


def rotate_activation(x: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    hidden_size = x.size(-1)
    return hadamard_transform_ref(x, hadamard=hadamard, scale=hidden_size**-0.5)


def pad_to_blocks(x: torch.Tensor, length_list: torch.Tensor, block_size: int = 128):
    """
    Pads a ragged/packed tensor into fixed-size blocks.

    Args:
        x: Input tensor of shape [t, n, d] where t = sum(length_list).
        length_list: Tensor of shape [bs] containing valid sequence lengths.
        block_size: The size of each block (default 128).

    Returns:
        padded_blocks: Tensor of shape [total_blocks, block_size, n, d].
    """
    # 1. Validation
    if x.shape[0] != length_list.sum():
        raise ValueError(f"Input dimension 0 ({x.shape[0]}) does not match sum of length_list ({length_list.sum()})")

    bs = length_list.shape[0]
    n, d = x.shape[1], x.shape[2]

    # 2. Calculate how many blocks are needed for each request
    # Formula: ceil(length / block_size) -> (length + block_size - 1) // block_size
    blocks_per_req = (length_list + block_size - 1) // block_size
    total_blocks = blocks_per_req.sum() + 1

    # 3. Allocate output tensor with zeros (this handles the padding automatically)
    # Shape: [total_blocks, block_size, n, d]
    out = torch.zeros((total_blocks, block_size, n, d), dtype=x.dtype, device=x.device)

    # 4. Fill data
    input_offset = 0
    block_offset = 1

    for i in range(bs):
        length = length_list[i]
        num_blocks = blocks_per_req[i]

        if length > 0:
            # Slice the valid data for this request from the packed input
            # Shape: [length, n, d]
            req_data = x[input_offset : input_offset + length]

            # Select the assigned blocks in the output
            # Shape: [num_blocks, block_size, n, d]
            target_blocks = out[block_offset : block_offset + num_blocks]

            # View as a flat sequence to easily copy the data
            # Shape: [num_blocks * block_size, n, d]
            target_flat = target_blocks.view(-1, n, d)

            # Copy valid data into the beginning of the allocated blocks
            # The rest remains zeros
            target_flat[:length] = req_data

        # Update pointers
        input_offset += length
        block_offset += num_blocks

    return out


class AscendDSABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.
        return "ASCEND_DSA" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_builder_cls():
        return AscendDSAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int) -> tuple[int, ...]:
        return num_blocks, block_size, num_kv_heads, head_size

    @staticmethod
    def get_scale_shape(num_blocks: int, block_size: int, scale_size: int) -> tuple[int, ...]:
        return num_blocks, block_size, scale_size

    @staticmethod
    def get_impl_cls() -> type["DSAAttentionImpl"]:
        return AscendDSAImpl

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        kernel_block_sizes = [8, 16, 128] if get_ascend_device_type() == AscendDeviceType.A5 else [8, 32, 128]
        return kernel_block_sizes


@dataclass
class AscendDSAPrefillMetadata:
    """Prefill Specific Metadata for Ascend"""

    attn_mask: torch.Tensor
    query_lens: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    input_positions: torch.Tensor
    query_start_loc: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    max_query_len: int
    max_seq_lens: int

    sin: torch.Tensor = None
    cos: torch.Tensor = None
    compress_sin: torch.Tensor = None
    compress_cos: torch.Tensor = None
    start_pos: torch.Tensor | None = None
    sas_metadata: torch.Tensor = None
    qli_metadata: torch.Tensor = None
    cu_c4_cmp_seqlen_list: torch.Tensor = None
    cu_c128_cmp_seqlen_list: torch.Tensor = None


@dataclass
class AscendDSADecodeMetadata:
    # Input positions for rotrary embeddings since for MLA the rotary
    # position embeddings are applied inside the attention backend
    input_positions: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    max_seqlen_kv: int
    max_seqlen_q: int
    seq_lens_list: list[int]
    max_seq_lens: int
    slot_mapping: torch.Tensor

    query_start_loc: torch.tensor = None
    query_start_loc_cpu: torch.tensor = None
    attn_mask: torch.Tensor | None = None
    sin: torch.Tensor = None
    cos: torch.Tensor = None
    compress_sin: torch.Tensor = None
    compress_cos: torch.Tensor = None
    cp_seq_len: torch.Tensor = None
    batch_seq_mask: torch.Tensor = None
    start_pos: torch.Tensor = None
    sas_metadata: torch.Tensor = None
    qli_metadata: torch.Tensor = None


@dataclass
class AscendDSAMetadata:
    """Metadata for MLACommon.
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    sin: torch.Tensor
    cos: torch.Tensor

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    query_lens: list[int] | None = None
    # The dimension of the attention heads
    head_dim: int | None = None
    attn_mask: torch.Tensor = None
    # chunked prefill by default if no attn_states passed
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    decode: AscendDSADecodeMetadata | None = None
    prefill: AscendDSAPrefillMetadata | None = None
    reshape_cache_event: torch.npu.Event = None

    # metadata for dsv4 indexer

    hadamard: torch.Tensor | None = None

    start_pos: torch.Tensor | None = None

    def __post_init__(self):
        pass


M = TypeVar("M", bound=AscendDSAMetadata)


class AscendDSAMetadataBuilder(AttentionMetadataBuilder[AscendDSAMetadata]):
    # Does this backend/builder support ACL Graphs for attention (default: no).
    aclgraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    hadamard = None
    start_pos_prefill: torch.Tensor | None = None
    start_pos_decode: torch.Tensor | None = None
    decode_sas_metadata: torch.Tensor | None = None
    decode_qli_metadata: torch.Tensor | None = None
    prefill_ratio_to_sas_metadata: dict | None = None
    decode_ratio_to_sas_metadata: dict | None = None
    block_size: int | None = 128
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        kv_cache_spec: MLAAttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[AscendDSAMetadata] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.metadata_cls = metadata_cls if metadata_cls is not None else AscendDSAMetadata
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        scheduler_config = vllm_config.scheduler_config
        # self.block_size = vllm_config.cache_config.block_size
        self.max_blocks = (vllm_config.model_config.max_model_len + self.block_size - 1) // self.block_size

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        self.spec_slot_mapping = None
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.spec_slot_mapping = [
                torch.zeros(
                    (vllm_config.scheduler_config.max_num_batched_tokens, 2), dtype=torch.int32, device=self.device
                )
                for _ in range(spec_token_num)
            ]
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )

        self.reorder_batch_threshold = self.decode_threshold
        self.rope_dim = self.model_config.hf_text_config.qk_rope_head_dim
        self.cos_cache = None
        self.sin_cache = None

        self.cu_seq_lens_cpu: torch.Tensor = None
        self.num_decodes = 0
        self.num_prefills = 0
        self.num_decode_tokens = 0
        self.num_prefill_tokens = 0
        self.context_lens_cpu: torch.Tensor = None
        self.num_actual_tokens: int | None = None
        self.block_table: torch.Tensor = None
        self.slot_mapping: torch.Tensor = None
        self.graph_pad_size = 0
        self.query_lens: torch.Tensor = None
        self.seq_lens: torch.Tensor = None
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

        self.compressor_ratio = getattr(kv_cache_spec, "compress_ratio", 0)
        hf_config = self.model_config.hf_config

        if AscendDSAMetadataBuilder.hadamard is None:
            if hf_config.model_type == "deepseek_v4":
                indexer_head_dim = hf_config.index_head_dim
                try:
                    from scipy.linalg import hadamard  # type: ignore[import-untyped]
                except ImportError as e:
                    raise ImportError("Please install scipy") from e
                log_dim = math.ceil(math.log2(indexer_head_dim))
                dim_padded = 2**log_dim
                AscendDSAMetadataBuilder.hadamard = torch.tensor(
                    hadamard(dim_padded, dtype=float), dtype=torch.float, device=self.device
                ).to(torch.bfloat16)
        self.start_pos_prefill = torch.zeros(scheduler_config.max_num_seqs, dtype=torch.int32, device=self.device)
        self.start_pos_decode = torch.zeros(scheduler_config.max_num_seqs, dtype=torch.int32, device=self.device)
        self.decode_sas_metadata = torch.zeros(1024, dtype=torch.int32, device=self.device)
        self.decode_qli_metadata = torch.zeros(1024, dtype=torch.int32, device=self.device)
        self.cu_seqlens_ori_kv = torch.tensor([], device=self.device)
        self.cu_seqlens_cmp_kv = torch.tensor([], device=self.device)
        self.seqused_q = torch.tensor([], device=self.device)
        self._zero_i32 = torch.tensor([0], device=self.device, dtype=torch.int32)
        # Note(qcs): we use two dimension slot_mapping for kvcache with shape
        # [block_nums, block_size, head_num, head_dim]
        self.slot_mapping = torch.zeros(
            (vllm_config.scheduler_config.max_num_batched_tokens, 2), dtype=torch.int32, device=self.device
        )
        self.a5_decode_slot_mapping = None
        if get_ascend_device_type() in {AscendDeviceType.A5}:
            self.a5_decode_slot_mapping = torch.full(
                (vllm_config.scheduler_config.max_num_batched_tokens,),
                -1,
                dtype=torch.int32,
                device=self.device,
            )

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendDSAMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.UNIFORM_BATCH

    def reorder_batch(self, input_batch: "NPUInputBatch", scheduler_output: "SchedulerOutput") -> bool:
        # We now want to reorder the batch so that the "decode" requests are at
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound, TODO(lucas): figure out a
        # better naming here)
        decodes = []
        prefills = []

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            if num_tokens <= self.decode_threshold:
                decodes.append(i)
            else:
                prefills.append(i)

        # We hope that this is fairly minimal since decodes
        # should be around for a number of iterations so hopefully they are
        # relatively stationary (and new request are generally appended to the
        # persistent batch so already should be at the back)
        # To achieve this we loop over the decodes in descending order and
        # the prefills in ascending order. We swap decodes from the  "back"
        # i.e. past where the last decode should be in the reodorered with
        # prefills from the front of the batch.
        # `decodes` and `prefills` are already in ascending order just based on
        # the above loop
        num_decodes = len(decodes)
        num_prefills = len(prefills)
        first_prefill = 0
        modified_batch = False

        for i in range(1, min(num_decodes, num_prefills) + 1):
            # If the decode is at the "back" of the batch, i, we can swap it
            # with the prefill closest to the front of the batch
            if decodes[num_decodes - i] >= num_decodes:
                input_batch.swap_states(prefills[first_prefill], decodes[num_decodes - i])
                first_prefill += 1
                modified_batch = True
            else:
                break

        # Save for next `build` call
        # TODO(lucas): this is a bit of a hack, we should probably have a
        # better way of doing this
        return modified_batch

    def set_num_actual_tokens(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ):
        self.num_actual_tokens = common_attn_metadata.num_actual_tokens

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendDSAMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        use_padded_decode_metadata = (
            kwargs.get("use_padded_decode_metadata", False)
            and common_attn_metadata.max_query_len <= self.decode_threshold
            and common_attn_metadata.num_input_tokens > num_actual_tokens
        )
        if use_padded_decode_metadata:
            # In FULL graph decode, hidden_states keeps the padded TND length.
            # The sparse-attention metadata must describe that padded q length,
            # while the public metadata still reports the real token count.
            common_attn_metadata = replace(
                common_attn_metadata,
                num_actual_tokens=common_attn_metadata.num_input_tokens,
            )
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        num_reqs_actual = kwargs.get("num_reqs_actual")
        self.prefill_ratio_to_sas_metadata = kwargs.get("prefill_ratio_to_sas_metadata")
        self.decode_ratio_to_sas_metadata = kwargs.get("decode_ratio_to_sas_metadata")
        assert self.prefill_ratio_to_sas_metadata is not None
        assert self.decode_ratio_to_sas_metadata is not None
        self.block_size = kwargs.get("block_size", 128)

        self.common_ratio_to_sas_metadata = kwargs.get("common_ratio_to_sas_metadata")
        assert self.common_ratio_to_sas_metadata is not None

        if self.common_ratio_to_sas_metadata.get("num_decodes", None) is None:
            self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = (
                split_decodes_and_prefills(common_attn_metadata, decode_threshold=self.decode_threshold)
            )
            self.common_ratio_to_sas_metadata["num_decodes"] = self.num_decodes
            self.common_ratio_to_sas_metadata["num_prefills"] = self.num_prefills
            self.common_ratio_to_sas_metadata["num_decode_tokens"] = self.num_decode_tokens
            self.common_ratio_to_sas_metadata["num_prefill_tokens"] = self.num_prefill_tokens
            self.num_actual_tokens = num_actual_tokens
            assert self.num_decodes + self.num_prefills == num_reqs
            assert self.num_decode_tokens + self.num_prefill_tokens == common_attn_metadata.num_actual_tokens
            num_input_tokens = common_attn_metadata.num_input_tokens
            input_positions = common_attn_metadata.positions[:num_input_tokens].long()
            if num_actual_tokens < num_input_tokens:
                input_positions = input_positions.clone()
                input_positions[num_actual_tokens:].fill_(0)
            self.common_ratio_to_sas_metadata["input_positions"] = input_positions
            if self.num_prefills:
                cos, sin = get_cos_and_sin_dsa(input_positions)
            else:
                cos, sin = get_cos_and_sin_dsa(input_positions, True)
            self.common_ratio_to_sas_metadata["cos"] = cos
            self.common_ratio_to_sas_metadata["sin"] = sin
            self.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
            if (
                use_padded_decode_metadata
                and num_reqs_actual is not None
                and num_reqs_actual < num_reqs
            ):
                self.seq_lens[num_reqs_actual:num_reqs].fill_(1)
            self.common_ratio_to_sas_metadata["seq_lens"] = self.seq_lens

            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
            query_seq_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            self.query_lens = query_seq_lens_cpu[:num_reqs]
            self.common_ratio_to_sas_metadata["query_lens"] = self.query_lens
        else:
            self.num_decodes, self.num_prefills, self.num_decode_tokens, self.num_prefill_tokens = (
                self.common_ratio_to_sas_metadata["num_decodes"],
                self.common_ratio_to_sas_metadata["num_prefills"],
                self.common_ratio_to_sas_metadata["num_decode_tokens"],
                self.common_ratio_to_sas_metadata["num_prefill_tokens"],
            )
            self.num_actual_tokens = num_actual_tokens
            num_input_tokens = common_attn_metadata.num_input_tokens
            input_positions = self.common_ratio_to_sas_metadata["input_positions"]
            cos, sin = self.common_ratio_to_sas_metadata["cos"], self.common_ratio_to_sas_metadata["sin"]
            self.seq_lens = self.common_ratio_to_sas_metadata["seq_lens"]
            if (
                use_padded_decode_metadata
                and num_reqs_actual is not None
                and num_reqs_actual < num_reqs
            ):
                self.seq_lens[num_reqs_actual:num_reqs].fill_(1)
            self.query_lens = self.common_ratio_to_sas_metadata["query_lens"]

        # NOTE: Currently, MTP-fullgraph is incompatibility pcp
        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        if get_ascend_device_type() in {AscendDeviceType.A5}:
            self.slot_mapping = slot_mapping
        else:
            self.slot_mapping[:num_input_tokens] = torch.stack(
                [slot_mapping // self.block_size, slot_mapping % self.block_size], axis=-1
            )

        self.graph_pad_size = common_attn_metadata.graph_pad_size
        block_table_size = self.get_block_table_size(common_attn_metadata, BUILD_METADATA_STEP_PREFILL)
        self.block_table = common_attn_metadata.block_table_tensor[:block_table_size]

        prefill_metadata = None
        if self.num_prefills > 0:
            prefill_metadata = self.build_prefill_metadata(common_prefix_len, common_attn_metadata)

        decode_metadata = None

        if self.num_decodes > 0:
            decode_metadata = self.build_decode_metadata(
                common_prefix_len,
                common_attn_metadata,
                num_reqs_actual,
                use_padded_decode_metadata,
            )

        return self.metadata_cls(  # type: ignore
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=self.num_actual_tokens,
            query_lens=self.query_lens,
            slot_mapping=None,
            head_dim=self.model_config.get_head_size(),
            num_decodes=self.num_decodes,
            num_decode_tokens=self.num_decode_tokens,
            num_prefills=self.num_prefills,
            attn_mask=None,
            attn_state=common_attn_metadata.attn_state,
            prefill=prefill_metadata,
            decode=decode_metadata,
            query_start_loc=query_start_loc,
            block_tables=None,
            seq_lens=self.seq_lens,
            cos=cos,
            sin=sin,
            hadamard=AscendDSAMetadataBuilder.hadamard,
        )

    def build_prefill_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> AscendDSAPrefillMetadata:
        assert self.prefill_ratio_to_sas_metadata is not None
        assert self.decode_ratio_to_sas_metadata is not None
        query_start_loc = common_attn_metadata.query_start_loc

        # reqs_start: the start request position of prefill request
        reqs_start = self.num_decodes
        # reqs_start: the start token position of prefill request
        tokens_start = self.num_decode_tokens

        if self.prefill_ratio_to_sas_metadata.get("prefill_input_positions", None) is None:
            input_positions = common_attn_metadata.positions[: self.num_actual_tokens].long()
            max_query_len = self.query_lens[reqs_start:].max().item()
            # Prefer _seq_lens_cpu (always available, updated during draft
            # iterations) over seq_lens_cpu (None in async spec decode mode).
            if common_attn_metadata._seq_lens_cpu is not None:
                _seq_lens_cpu = common_attn_metadata._seq_lens_cpu
            elif common_attn_metadata.seq_lens_cpu is not None:
                _seq_lens_cpu = common_attn_metadata.seq_lens_cpu
            else:
                _seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
            max_seq_lens = _seq_lens_cpu[reqs_start:].max().item()
            self.prefill_ratio_to_sas_metadata["input_positions"] = input_positions
            self.prefill_ratio_to_sas_metadata["max_query_len"] = max_query_len
            self.prefill_ratio_to_sas_metadata["max_seq_lens"] = max_seq_lens

            prefill_query_start_loc = query_start_loc[reqs_start:] - query_start_loc[reqs_start]
            prefill_input_positions = input_positions[tokens_start:]
            self.prefill_ratio_to_sas_metadata["prefill_input_positions"] = prefill_input_positions
            self.prefill_ratio_to_sas_metadata["prefill_query_start_loc"] = prefill_query_start_loc

            cos, sin = get_cos_and_sin_dsa(prefill_input_positions)
            self.prefill_ratio_to_sas_metadata["cos"] = cos
            self.prefill_ratio_to_sas_metadata["sin"] = sin

            prefill_seq_lens = self.seq_lens[reqs_start:]
            num_prefill = prefill_seq_lens.shape[0]
            self.prefill_ratio_to_sas_metadata["prefill_seq_lens"] = prefill_seq_lens
            self.prefill_ratio_to_sas_metadata["num_prefill"] = num_prefill
        else:
            input_positions = self.prefill_ratio_to_sas_metadata["input_positions"]
            max_query_len = self.prefill_ratio_to_sas_metadata["max_query_len"]
            max_seq_lens = self.prefill_ratio_to_sas_metadata["max_seq_lens"]
            prefill_input_positions = self.prefill_ratio_to_sas_metadata["prefill_input_positions"]
            prefill_query_start_loc = self.prefill_ratio_to_sas_metadata["prefill_query_start_loc"]
            cos = self.prefill_ratio_to_sas_metadata["cos"]
            sin = self.prefill_ratio_to_sas_metadata["sin"]
            prefill_seq_lens = self.prefill_ratio_to_sas_metadata["prefill_seq_lens"]
            num_prefill = self.prefill_ratio_to_sas_metadata["num_prefill"]

        def _get_padded_compressed_position(prefill_input_positions, compress_ratio):
            if compress_ratio <= 1:
                return prefill_input_positions
            mask = ((prefill_input_positions + 1) % compress_ratio) == 0
            input_positions = prefill_input_positions[mask]
            input_positions = (input_positions + 1) - compress_ratio
            target_shape = (
                min(self.num_prefill_tokens, self.num_prefill_tokens // compress_ratio + self.num_prefills),
            )
            pad_right = target_shape[0] - input_positions.shape[0]
            pad_positions = F.pad(input_positions, (0, pad_right), value=0.0)
            return pad_positions

        def _get_cmp_seq_lens(prefill_seq_lens, compress_ratio):
            # Note(qcs): some models use compress_ratio=0 as non-compression tag.
            _cmp_seq_lens = prefill_seq_lens // compress_ratio if compress_ratio >= 1 else prefill_seq_lens
            return torch.concat(
                (torch.tensor([0], device=_cmp_seq_lens.device), torch.cumsum(_cmp_seq_lens, -1)), dim=-1
            )

        def _get_compressed_decode_token_start_and_end(decode_input_positions, compress_ratio):
            # Note(qcs): some models use compress_ratio=0 as non-compression tag.
            if compress_ratio == 0:
                compress_ratio = 1
            # TODO(yilin): decode_input_positions is a device tensor,
            # this will introduce sync operation. Refactor me to torch.where instead
            mask = ((decode_input_positions + 1) % compress_ratio) == 0
            compressed_decode_num = mask.sum()

            end = min(self.num_prefill_tokens, self.num_prefill_tokens // compress_ratio + self.num_prefills)
            return compressed_decode_num, end

        if self.prefill_ratio_to_sas_metadata.get(f"c{self.compressor_ratio}_cos", None) is None:
            compress_cos, compress_sin = get_cos_and_sin_dsa(
                _get_padded_compressed_position(prefill_input_positions, self.compressor_ratio)
            )
            self.prefill_ratio_to_sas_metadata[f"c{self.compressor_ratio}_cos"] = compress_cos
            self.prefill_ratio_to_sas_metadata[f"c{self.compressor_ratio}_sin"] = compress_sin
        else:
            compress_cos = self.prefill_ratio_to_sas_metadata[f"c{self.compressor_ratio}_cos"]
            compress_sin = self.prefill_ratio_to_sas_metadata[f"c{self.compressor_ratio}_sin"]

        if self.prefill_ratio_to_sas_metadata.get(f"compressed_c{self.compressor_ratio}_tokens_start", None) is None:
            decode_input_positions = input_positions[:tokens_start]
            compressed_tokens_start, compressed_tokens_end = _get_compressed_decode_token_start_and_end(
                decode_input_positions, self.compressor_ratio
            )
            self.prefill_ratio_to_sas_metadata[f"compressed_c{self.compressor_ratio}_tokens_start"] = (
                compressed_tokens_start
            )
            self.prefill_ratio_to_sas_metadata[f"compressed_c{self.compressor_ratio}_tokens_ebd"] = (
                compressed_tokens_end
            )
        else:
            compressed_tokens_start = self.prefill_ratio_to_sas_metadata[
                f"compressed_c{self.compressor_ratio}_tokens_start"
            ]
            compressed_tokens_end = self.prefill_ratio_to_sas_metadata[
                f"compressed_c{self.compressor_ratio}_tokens_ebd"
            ]

        prefill_slot_mapping = self.slot_mapping[
            compressed_tokens_start : compressed_tokens_end + compressed_tokens_start
        ]

        assert self.start_pos_prefill is not None
        self.start_pos_prefill.fill_(0)
        seq_lens_q = prefill_query_start_loc[1:] - prefill_query_start_loc[:-1]
        self.start_pos_prefill[:num_prefill] = self.seq_lens[reqs_start:] - seq_lens_q

        tp_size = get_tensor_model_parallel_world_size()
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size
        index_topk = self.model_config.hf_config.index_topk

        cu_c4_cmp_seqlen_list = None
        cu_c128_cmp_seqlen_list = None

        layer_name = f"c{self.compressor_ratio}"
        is_a5 = get_ascend_device_type() in {AscendDeviceType.A5}
        metadata_op = (
            torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv_metadata
            if is_a5
            else torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata
        )
        metadata_kwargs = {"kv_quant_mode": 1} if is_a5 else {"device": str(self.seqused_q.device)}
        if self.compressor_ratio <= 1:
            if self.prefill_ratio_to_sas_metadata.get(layer_name) is None:
                self.prefill_ratio_to_sas_metadata[layer_name] = metadata_op(
                    **metadata_kwargs,
                    num_heads_q=n_local_heads,
                    num_heads_kv=1,
                    head_dim=self.model_config.get_head_size(),
                    cu_seqlens_q=prefill_query_start_loc,
                    cu_seqlens_ori_kv=prefill_query_start_loc,
                    cu_seqlens_cmp_kv=None,
                    seqused_q=self.seqused_q,
                    seqused_kv=self.seq_lens[reqs_start:],
                    max_seqlen_q=seq_lens_q.max(),
                    max_seqlen_kv=self.seq_lens[reqs_start:].max(),
                    batch_size=len(self.seq_lens[reqs_start:]),
                    cmp_ratio=1,
                    ori_mask_mode=4,  # 4:sliding window
                    ori_win_left=self.model_config.hf_config.sliding_window - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=False,
                )
            sas_metadata = self.prefill_ratio_to_sas_metadata[layer_name]
        elif self.compressor_ratio == 4:
            if self.prefill_ratio_to_sas_metadata.get(layer_name) is None:
                self.prefill_ratio_to_sas_metadata[layer_name] = metadata_op(
                    **metadata_kwargs,
                    num_heads_q=n_local_heads,
                    num_heads_kv=1,
                    head_dim=self.model_config.get_head_size(),
                    cu_seqlens_q=prefill_query_start_loc,
                    cu_seqlens_ori_kv=prefill_query_start_loc,
                    cu_seqlens_cmp_kv=cu_c4_cmp_seqlen_list,
                    seqused_q=self.seqused_q,
                    seqused_kv=self.seq_lens[reqs_start:],
                    max_seqlen_q=seq_lens_q.max(),
                    max_seqlen_kv=self.seq_lens[reqs_start:].max(),
                    batch_size=len(self.seq_lens[reqs_start:]),
                    cmp_topk=index_topk,
                    # topk=index_topk,
                    cmp_ratio=4,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self.model_config.hf_config.sliding_window - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=True,
                )
            sas_metadata = self.prefill_ratio_to_sas_metadata[layer_name]
        else:
            if self.prefill_ratio_to_sas_metadata.get(layer_name) is None:
                self.prefill_ratio_to_sas_metadata[layer_name] = metadata_op(
                    **metadata_kwargs,
                    num_heads_q=n_local_heads,
                    num_heads_kv=1,
                    head_dim=self.model_config.get_head_size(),
                    cu_seqlens_q=prefill_query_start_loc,
                    cu_seqlens_ori_kv=prefill_query_start_loc,
                    cu_seqlens_cmp_kv=cu_c128_cmp_seqlen_list,
                    seqused_q=self.seqused_q,
                    seqused_kv=self.seq_lens[reqs_start:],
                    max_seqlen_q=seq_lens_q.max(),
                    max_seqlen_kv=self.seq_lens[reqs_start:].max(),
                    batch_size=len(self.seq_lens[reqs_start:]),
                    cmp_ratio=128,  #
                    ori_mask_mode=4,  # 4:sliding window
                    cmp_mask_mode=3,  # 3:causal
                    ori_win_left=self.model_config.hf_config.sliding_window - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=True,
                )
            sas_metadata = self.prefill_ratio_to_sas_metadata[layer_name]
        if self.prefill_ratio_to_sas_metadata.get("qli") is None:
            self.prefill_ratio_to_sas_metadata["qli"] = torch.ops._C_ascend.npu_quant_lightning_indexer_metadata(
                actual_seq_lengths_query=prefill_query_start_loc[1:].clone(),
                actual_seq_lengths_key=self.seq_lens[reqs_start:].clone(),
                num_heads_q=self.model_config.hf_config.index_n_heads,  # 64
                num_heads_k=1,
                head_dim=self.model_config.hf_config.index_head_dim,  # 128
                query_quant_mode=0,
                key_quant_mode=0,
                batch_size=len(self.seq_lens[reqs_start:]),
                max_seqlen_q=seq_lens_q.max().item(),
                max_seqlen_k=self.seq_lens[reqs_start:].max().item(),
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.model_config.hf_config.index_topk,  # 512
                sparse_mode=3,
                pre_tokens=(1 << 63) - 1,
                next_tokens=(1 << 63) - 1,
                cmp_ratio=4,
                device=str(self.seqused_q.device),
            )
        qli_metadata = self.prefill_ratio_to_sas_metadata.get("qli")

        return AscendDSAPrefillMetadata(
            attn_mask=None,
            query_lens=self.query_lens[reqs_start:].to(torch.int32),
            seq_lens=self.seq_lens[reqs_start:],
            context_lens=self.seq_lens[reqs_start:],
            input_positions=prefill_input_positions,
            block_table=self.block_table[reqs_start:, ...],
            slot_mapping=prefill_slot_mapping,
            max_query_len=max_query_len,
            max_seq_lens=max_seq_lens,
            query_start_loc=prefill_query_start_loc,
            sin=sin,
            cos=cos,
            compress_sin=compress_sin,
            compress_cos=compress_cos,
            start_pos=self.start_pos_prefill[:num_prefill],
            sas_metadata=sas_metadata,
            qli_metadata=qli_metadata,
            cu_c4_cmp_seqlen_list=cu_c4_cmp_seqlen_list,
            cu_c128_cmp_seqlen_list=cu_c128_cmp_seqlen_list,
        )

    def build_decode_metadata(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        num_reqs_actual: int | None,
        use_padded_decode_metadata: bool = False,
    ) -> AscendDSADecodeMetadata:
        assert self.decode_ratio_to_sas_metadata is not None
        if self.decode_ratio_to_sas_metadata.get("query_start_loc", None) is None:
            query_start_loc = common_attn_metadata.query_start_loc[: self.num_decodes + 1]
            self.decode_ratio_to_sas_metadata["query_start_loc"] = query_start_loc
            input_positions = common_attn_metadata.positions[: self.num_decode_tokens].long()
            if self.num_actual_tokens is not None and self.num_actual_tokens < self.num_decode_tokens:
                input_positions = input_positions.clone()
                input_positions[self.num_actual_tokens : self.num_decode_tokens].fill_(0)
            self.decode_ratio_to_sas_metadata["input_positions"] = input_positions
            cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=True)
            self.decode_ratio_to_sas_metadata["cos"] = cos
            self.decode_ratio_to_sas_metadata["sin"] = sin

            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: self.num_decodes + 1]
            input_positions_cpu = common_attn_metadata.positions_cpu[: self.num_decode_tokens].long()
            if self.num_actual_tokens is not None and self.num_actual_tokens < self.num_decode_tokens:
                input_positions_cpu = input_positions_cpu.clone()
                input_positions_cpu[self.num_actual_tokens : self.num_decode_tokens].fill_(0)

            # Prefer _seq_lens_cpu (always available, updated during draft
            # iterations) over seq_lens_cpu (None in async spec decode mode).
            if common_attn_metadata._seq_lens_cpu is not None:
                _seq_lens_cpu = common_attn_metadata._seq_lens_cpu
            elif common_attn_metadata.seq_lens_cpu is not None:
                _seq_lens_cpu = common_attn_metadata.seq_lens_cpu
            else:
                _seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
            max_seq_lens = _seq_lens_cpu[: self.num_decodes].max().item()
            decode_input_positions = input_positions_cpu
            seq_lens_list = _seq_lens_cpu[: self.num_decodes].tolist()
            self.decode_ratio_to_sas_metadata["query_start_loc_cpu"] = query_start_loc_cpu
            self.decode_ratio_to_sas_metadata["decode_input_positions"] = decode_input_positions
            self.decode_ratio_to_sas_metadata["max_seq_lens"] = max_seq_lens
            self.decode_ratio_to_sas_metadata["seq_lens_list"] = seq_lens_list

            max_seqlen_kv = torch.max(_seq_lens_cpu[: self.num_decodes]).item()
            max_seqlen_q = torch.max(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).item()
            self.decode_ratio_to_sas_metadata["max_seqlen_kv"] = max_seqlen_kv
            self.decode_ratio_to_sas_metadata["max_seqlen_q"] = max_seqlen_q

            seq_lens_q = query_start_loc[1:] - query_start_loc[:-1]
            start_pos_decode = self.seq_lens[: self.num_decodes] - seq_lens_q
            self.decode_ratio_to_sas_metadata["start_pos_decode"] = start_pos_decode
        else:
            query_start_loc = self.decode_ratio_to_sas_metadata["query_start_loc"]
            input_positions = self.decode_ratio_to_sas_metadata["input_positions"]
            cos = self.decode_ratio_to_sas_metadata["cos"]
            sin = self.decode_ratio_to_sas_metadata["sin"]
            query_start_loc_cpu = self.decode_ratio_to_sas_metadata["query_start_loc_cpu"]
            decode_input_positions = self.decode_ratio_to_sas_metadata["decode_input_positions"]
            max_seq_lens = self.decode_ratio_to_sas_metadata["max_seq_lens"]
            seq_lens_list = self.decode_ratio_to_sas_metadata["seq_lens_list"]
            max_seqlen_kv = self.decode_ratio_to_sas_metadata["max_seqlen_kv"]
            max_seqlen_q = self.decode_ratio_to_sas_metadata["max_seqlen_q"]
            start_pos_decode = self.decode_ratio_to_sas_metadata["start_pos_decode"]

        block_table_size = self.get_block_table_size(common_attn_metadata, BUILD_METADATA_STEP_DECODE)

        cp_seq_len, batch_seq_mask = None, None

        def _get_padded_compressed_position(decode_input_positions, compress_ratio, device):
            if compress_ratio <= 1:
                return decode_input_positions
            mask = ((decode_input_positions + 1) % compress_ratio) == 0
            input_positions = decode_input_positions[mask]
            input_positions = (input_positions + 1) - compress_ratio
            target_shape = (min(self.num_decode_tokens, self.num_decode_tokens // compress_ratio + self.num_decodes),)
            pad_right = target_shape[0] - input_positions.shape[0]
            pad_positions = F.pad(input_positions, (0, pad_right), value=0.0)
            gpu_pad_positions = pad_positions.pin_memory().to(device, non_blocking=True)
            return gpu_pad_positions

        layer_name = f"c{self.compressor_ratio}"
        if self.decode_ratio_to_sas_metadata.get(layer_name + "_cos", None) is None:
            compress_cos, compress_sin = get_cos_and_sin_dsa(
                {
                    layer_name: _get_padded_compressed_position(
                        decode_input_positions, self.compressor_ratio, input_positions.device
                    )
                },
                use_cache=True,
            )
            self.decode_ratio_to_sas_metadata[layer_name + "_cos"] = compress_cos
            self.decode_ratio_to_sas_metadata[layer_name + "_sin"] = compress_sin
        else:
            compress_cos = self.decode_ratio_to_sas_metadata[layer_name + "_cos"]
            compress_sin = self.decode_ratio_to_sas_metadata[layer_name + "_sin"]

        def _get_compressed_decode_token_start(decode_input_positions, compress_ratio):
            # Note(qcs): some models use compress_ratio=0 as non-compression tag.
            if compress_ratio == 0:
                compress_ratio = 1
            mask = ((decode_input_positions + 1) % compress_ratio) == 0
            compressed_decode_num = mask.sum().item()
            return compressed_decode_num

        if self.decode_ratio_to_sas_metadata.get("compressed_tokens_start_" + str(self.compressor_ratio), None) is None:
            compressed_tokens_start = _get_compressed_decode_token_start(decode_input_positions, self.compressor_ratio)
            self.decode_ratio_to_sas_metadata["compressed_tokens_start_" + str(self.compressor_ratio)] = (
                compressed_tokens_start
            )
        else:
            compressed_tokens_start = self.decode_ratio_to_sas_metadata[
                "compressed_tokens_start_" + str(self.compressor_ratio)
            ]

        tmp_compressor_ration = self.compressor_ratio if self.compressor_ratio != 0 else 1
        target_shape = min(
            self.num_decode_tokens,
            self.num_decode_tokens // tmp_compressor_ration + self.num_decodes)
        if get_ascend_device_type() in {AscendDeviceType.A5}:
            assert self.a5_decode_slot_mapping is not None
            slot_mapping = self.a5_decode_slot_mapping[:target_shape]
            slot_mapping.fill_(-1)
            valid_slot_mapping_len = min(compressed_tokens_start, target_shape)
            if valid_slot_mapping_len > 0:
                slot_mapping[:valid_slot_mapping_len].copy_(self.slot_mapping[:valid_slot_mapping_len])
        else:
            slot_mapping = self.slot_mapping[:compressed_tokens_start]
            pad_size = target_shape - slot_mapping.shape[0]
            if pad_size > 0:
                if slot_mapping.ndim == 1:
                    slot_mapping = F.pad(slot_mapping, (0, pad_size), value=-1)
                else:
                    slot_mapping = F.pad(slot_mapping, (0, 0, 0, pad_size), value=-1)
            else:
                slot_mapping = slot_mapping[:target_shape]

        assert self.start_pos_decode is not None
        self.start_pos_decode.fill_(0)
        self.start_pos_decode[: self.num_decodes] = start_pos_decode

        if (
            use_padded_decode_metadata
            and num_reqs_actual is not None
            and num_reqs_actual < self.num_decodes
        ):
            self.start_pos_decode[num_reqs_actual:].fill_(0)
            self.block_table[num_reqs_actual : self.num_decodes, ...].fill_(0)

        tp_size = get_tensor_model_parallel_world_size()
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size
        index_topk = self.model_config.hf_config.index_topk

        assert self.decode_sas_metadata is not None
        is_a5 = get_ascend_device_type() in {AscendDeviceType.A5}
        if is_a5:
            if self.decode_ratio_to_sas_metadata.get("cu_seqlens_ori_kv", None) is None:
                cu_seqlens_ori_kv = torch.cat([
                    self._zero_i32,
                    torch.cumsum(self.seq_lens[: self.num_decodes], dim=0).to(torch.int32),
                ])
                self.decode_ratio_to_sas_metadata["cu_seqlens_ori_kv"] = cu_seqlens_ori_kv
            else:
                cu_seqlens_ori_kv = self.decode_ratio_to_sas_metadata["cu_seqlens_ori_kv"]
        else:
            cu_seqlens_ori_kv = self.cu_seqlens_ori_kv
        metadata_op = (
            torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv_metadata
            if is_a5
            else torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata
        )
        metadata_kwargs = {"kv_quant_mode": 1} if is_a5 else {"device": str(self.seqused_q.device)}
        cu_seqlens_cmp_kv = None if is_a5 else self.cu_seqlens_cmp_kv
        if self.compressor_ratio <= 1:
            if self.decode_ratio_to_sas_metadata.get(layer_name) is None:
                self.decode_ratio_to_sas_metadata[layer_name] = metadata_op(
                    **metadata_kwargs,
                    num_heads_q=n_local_heads,
                    num_heads_kv=1,
                    head_dim=self.model_config.get_head_size(),
                    cu_seqlens_q=query_start_loc,  # cached
                    cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                    cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                    seqused_q=self.seqused_q,
                    seqused_kv=self.seq_lens[: self.num_decodes],  # cached
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    batch_size=len(self.seq_lens[: self.num_decodes]),  # cached
                    cmp_ratio=1,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self.model_config.hf_config.sliding_window - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=False,
                )
            self.decode_sas_metadata[:1024] = self.decode_ratio_to_sas_metadata[layer_name]
        elif self.compressor_ratio == 4:
            if self.decode_ratio_to_sas_metadata.get(layer_name) is None:
                self.decode_ratio_to_sas_metadata[layer_name] = metadata_op(
                    **metadata_kwargs,
                    num_heads_q=n_local_heads,
                    num_heads_kv=1,
                    head_dim=self.model_config.get_head_size(),
                    cu_seqlens_q=query_start_loc,  # cached
                    cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                    cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                    seqused_q=self.seqused_q,
                    seqused_kv=self.seq_lens[: self.num_decodes],  # cached
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    batch_size=len(self.seq_lens[: self.num_decodes]),  # cached
                    cmp_topk=index_topk,
                    # topk=index_topk,
                    cmp_ratio=4,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self.model_config.hf_config.sliding_window - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=True,
                )
            self.decode_sas_metadata[:1024] = self.decode_ratio_to_sas_metadata[layer_name]
        else:
            if self.decode_ratio_to_sas_metadata.get(layer_name) is None:
                self.decode_ratio_to_sas_metadata[layer_name] = metadata_op(
                    **metadata_kwargs,
                    num_heads_q=n_local_heads,
                    num_heads_kv=1,
                    head_dim=self.model_config.get_head_size(),
                    cu_seqlens_q=query_start_loc,
                    cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                    cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                    seqused_q=self.seqused_q,
                    seqused_kv=self.seq_lens[: self.num_decodes],
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=max_seqlen_kv,
                    batch_size=len(self.seq_lens[: self.num_decodes]),
                    cmp_ratio=128,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self.model_config.hf_config.sliding_window - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    has_ori_kv=True,
                    has_cmp_kv=True,
                )
            self.decode_sas_metadata[:1024] = self.decode_ratio_to_sas_metadata[layer_name]
        assert self.decode_qli_metadata is not None
        if self.decode_ratio_to_sas_metadata.get("qli") is None:
            self.decode_ratio_to_sas_metadata["qli"] = torch.ops._C_ascend.npu_quant_lightning_indexer_metadata(
                actual_seq_lengths_query=query_start_loc[1:].clone(),
                actual_seq_lengths_key=self.seq_lens[: self.num_decodes].clone(),
                num_heads_q=self.model_config.hf_config.index_n_heads,  # 64
                num_heads_k=1,
                head_dim=self.model_config.hf_config.index_head_dim,  # 128
                query_quant_mode=0,
                key_quant_mode=0,
                batch_size=len(self.seq_lens[: self.num_decodes]),
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_kv,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.model_config.hf_config.index_topk,  # 512
                sparse_mode=3,
                pre_tokens=(1 << 63) - 1,
                next_tokens=(1 << 63) - 1,
                cmp_ratio=4,
                device=str(self.seqused_q.device),
            )
        self.decode_qli_metadata[:1024] = self.decode_ratio_to_sas_metadata.get("qli")
        decode_metadata = AscendDSADecodeMetadata(
            input_positions=input_positions,
            block_table=self.block_table[:block_table_size, ...],
            slot_mapping=slot_mapping,
            seq_lens=self.seq_lens[: self.num_decodes],  # cached
            seq_lens_list=seq_lens_list,
            max_seq_lens=max_seq_lens,
            max_seqlen_kv=max_seqlen_kv,
            max_seqlen_q=max_seqlen_q,
            attn_mask=None,
            query_start_loc=query_start_loc,  # cached
            query_start_loc_cpu=query_start_loc_cpu,
            sin=sin[: self.num_decode_tokens, ...],
            cos=cos[: self.num_decode_tokens, ...],
            compress_sin=compress_sin,
            compress_cos=compress_cos,
            cp_seq_len=cp_seq_len,
            batch_seq_mask=batch_seq_mask,
            start_pos=self.start_pos_decode[: self.num_decodes],  # cached
            sas_metadata=self.decode_sas_metadata,
            qli_metadata=self.decode_qli_metadata,
        )
        return decode_metadata

    def build_for_drafting(
        self,
        draft_step: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> AscendDSADecodeMetadata:
        assert self.compressor_ratio <= 1, "vLLM-Ascend only support SWA-layer for Deepseek-V4 now."
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.decode_threshold
        )
        num_input_tokens = common_attn_metadata.num_input_tokens
        input_positions = common_attn_metadata.positions[:num_input_tokens].long()
        if num_prefills:
            cos, sin = get_cos_and_sin_dsa(input_positions)
        else:
            cos, sin = get_cos_and_sin_dsa(input_positions, True)

        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        if get_ascend_device_type() in {AscendDeviceType.A5}:
            self.spec_slot_mapping[draft_step - 1] = slot_mapping  # type: ignore[index]
        else:
            self.spec_slot_mapping[draft_step - 1][:num_input_tokens] = torch.stack(  # type: ignore[index]
                [slot_mapping // self.block_size, slot_mapping % self.block_size], axis=-1
            )
        # logger.info(f'{draft_step=} {slot_mapping=} {self.spec_slot_mapping[draft_step - 1]=}')

        prefill_metadata = None
        if num_prefills > 0:
            prefill_metadata = self.build_prefill_metadata_for_drafting(
                draft_step=draft_step,
                common_attn_metadata=common_attn_metadata,
                reqs_start=num_decodes,
                tokens_start=num_decode_tokens,
                num_prefill_tokens=num_prefill_tokens,
            )

        decode_metadata = None
        if num_decodes > 0:
            decode_metadata = self.build_decode_metadata_for_drafting(
                draft_step=draft_step,
                common_attn_metadata=common_attn_metadata,
                num_decodes=num_decodes,
                num_decode_tokens=num_decode_tokens,
            )

        return self.metadata_cls(  # type: ignore
            num_input_tokens=common_attn_metadata.num_input_tokens,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            query_lens=None,
            slot_mapping=None,
            head_dim=self.model_config.get_head_size(),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            attn_mask=None,
            attn_state=common_attn_metadata.attn_state,
            prefill=prefill_metadata,
            decode=decode_metadata,
            query_start_loc=None,
            block_tables=None,
            seq_lens=None,
            cos=cos,
            sin=sin,
            hadamard=None,
        )

    def build_prefill_metadata_for_drafting(
        self,
        draft_step: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        **kwargs,
    ) -> AscendDSAPrefillMetadata:
        tp_size = get_tensor_model_parallel_world_size()
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size

        reqs_start = kwargs.get("reqs_start")
        tokens_start = kwargs.get("tokens_start")
        num_prefill_tokens = kwargs.get("num_prefill_tokens")
        query_start_loc = common_attn_metadata.query_start_loc
        prefill_query_start_loc = query_start_loc[reqs_start:] - query_start_loc[reqs_start]
        seq_lens_q = prefill_query_start_loc[1:] - prefill_query_start_loc[:-1]
        seq_lens = common_attn_metadata.seq_lens[reqs_start:]

        num_actual_tokens = common_attn_metadata.num_actual_tokens
        input_positions = common_attn_metadata.positions[:num_actual_tokens].long()
        prefill_input_positions = input_positions[tokens_start:]
        cos, sin = get_cos_and_sin_dsa(prefill_input_positions)

        prefill_slot_mapping = self.spec_slot_mapping[draft_step - 1][tokens_start:num_prefill_tokens]  # type: ignore[index]
        block_table = common_attn_metadata.block_table_tensor[: common_attn_metadata.num_reqs]

        is_a5 = get_ascend_device_type() in {AscendDeviceType.A5}
        metadata_op = (
            torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv_metadata
            if is_a5
            else torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata
        )
        metadata_kwargs = {"kv_quant_mode": 1} if is_a5 else {"device": str(self.seqused_q.device)}
        sas_metadata = metadata_op(
            **metadata_kwargs,
            num_heads_q=n_local_heads,
            num_heads_kv=1,
            head_dim=self.model_config.get_head_size(),
            cu_seqlens_q=prefill_query_start_loc,
            cu_seqlens_ori_kv=prefill_query_start_loc,
            cu_seqlens_cmp_kv=None,
            seqused_q=self.seqused_q,
            seqused_kv=seq_lens,
            max_seqlen_q=seq_lens_q.max(),
            max_seqlen_kv=seq_lens.max(),
            batch_size=len(seq_lens),
            cmp_ratio=1,
            ori_mask_mode=4,  # 4:sliding window
            ori_win_left=self.model_config.hf_config.sliding_window - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=False,
        )

        return AscendDSAPrefillMetadata(
            attn_mask=None,
            query_lens=None,
            seq_lens=seq_lens,
            context_lens=None,
            input_positions=None,  # type: ignore[arg-type]
            block_table=block_table[reqs_start:, ...],
            slot_mapping=prefill_slot_mapping,
            max_query_len=None,  # type: ignore[arg-type]
            max_seq_lens=None,  # type: ignore[arg-type]
            query_start_loc=prefill_query_start_loc,
            sin=sin,
            cos=cos,
            compress_sin=None,
            compress_cos=None,
            start_pos=None,
            sas_metadata=sas_metadata,
            qli_metadata=None,
            cu_c4_cmp_seqlen_list=None,
            cu_c128_cmp_seqlen_list=None,
        )

    def build_decode_metadata_for_drafting(
        self,
        draft_step: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        **kwargs,
    ) -> AscendDSADecodeMetadata:
        tp_size = get_tensor_model_parallel_world_size()
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size

        num_decodes = kwargs.get("num_decodes")
        num_decode_tokens = kwargs.get("num_decode_tokens")
        num_decodes_typed = num_decodes or 0
        num_decode_tokens_typed = num_decode_tokens or 0
        query_start_loc = common_attn_metadata.query_start_loc[: num_decodes_typed + 1]
        seq_lens = common_attn_metadata.seq_lens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_decodes_typed + 1]
        max_seqlen_q = torch.max(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]).item()

        if common_attn_metadata._seq_lens_cpu is not None:
            _seq_lens_cpu = common_attn_metadata._seq_lens_cpu
        elif common_attn_metadata.seq_lens_cpu is not None:
            _seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        else:
            _seq_lens_cpu = common_attn_metadata.seq_lens.cpu()
        max_seqlen_kv = torch.max(_seq_lens_cpu[:num_decodes]).item()

        input_positions = common_attn_metadata.positions[:num_decode_tokens_typed].long()
        cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=True)

        slot_mapping = self.spec_slot_mapping[draft_step - 1][:num_decode_tokens_typed]  # type: ignore[index]
        block_table = common_attn_metadata.block_table_tensor

        is_a5 = get_ascend_device_type() in {AscendDeviceType.A5}
        metadata_op = (
            torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv_metadata
            if is_a5
            else torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata
        )
        metadata_kwargs = {"kv_quant_mode": 1} if is_a5 else {"device": str(self.seqused_q.device)}
        cu_seqlens_ori_kv = (
            torch.cat([
                self._zero_i32,
                torch.cumsum(seq_lens[:num_decodes_typed], dim=0).to(torch.int32),
            ])
            if is_a5
            else torch.tensor([], device=self.device, dtype=torch.int32)
        )
        decode_sas_metadata = metadata_op(
            **metadata_kwargs,
            num_heads_q=n_local_heads,
            num_heads_kv=1,
            head_dim=self.model_config.get_head_size(),
            cu_seqlens_q=query_start_loc,  # cached
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=None,
            seqused_q=self.seqused_q,
            seqused_kv=seq_lens[:num_decodes],  # cached
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=len(seq_lens[:num_decodes]),  # cached
            cmp_ratio=1,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=self.model_config.hf_config.sliding_window - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            has_ori_kv=True,
            has_cmp_kv=False,
        )

        decode_metadata = AscendDSADecodeMetadata(
            input_positions=None,
            block_table=block_table[:num_decodes, ...],
            slot_mapping=slot_mapping,
            seq_lens=seq_lens[:num_decodes],  # cached
            seq_lens_list=None,  # type: ignore[arg-type]
            max_seq_lens=None,  # type: ignore[arg-type]
            max_seqlen_kv=None,  # type: ignore[arg-type]
            max_seqlen_q=None,  # type: ignore[arg-type]
            attn_mask=None,
            query_start_loc=query_start_loc,  # cached
            query_start_loc_cpu=None,
            sin=sin[:num_decode_tokens, ...],
            cos=cos[:num_decode_tokens, ...],
            compress_sin=None,
            compress_cos=None,
            cp_seq_len=None,
            batch_seq_mask=None,
            start_pos=None,  # cached
            sas_metadata=decode_sas_metadata,
            qli_metadata=None,
        )
        return decode_metadata

    def get_block_table_size(self, common_attn_metadata: AscendCommonAttentionMetadata, build_metadata_step: int):
        if build_metadata_step == BUILD_METADATA_STEP_PREFILL:
            # If graph_pad_size > -1, mean is running in fullgraph mode.
            # NOTE: Maybe this block_table change can be removed when graph_pad_size > 1.
            # if self.graph_pad_size > common_attn_metadata.num_reqs and \
            #         self.speculative_config.disable_padded_drafter_batch:
            #     return self.graph_pad_size
            return common_attn_metadata.num_reqs
        return self.num_decodes

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
        **kwargs,
    ):
        if attn_state in {AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding}:
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly and SpecDecoding state"
            )

        assert attn_metadata is not None
        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendDSAImpl(DSAAttentionImpl):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        n_heads: int,
        scale: float,
        n_local_heads: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int | None,
        nope_head_dim: int,
        n_groups: int,
        n_local_groups: int,
        window_size: int,
        compress_ratio: int,
        **kwargs,
    ):
        self.num_heads = n_heads
        self.n_local_heads = n_local_heads
        self.scale = scale
        self.o_lora_rank = o_lora_rank
        self.nope_head_dim = nope_head_dim
        self.rope_head_dim = rope_head_dim
        self.head_dim = head_dim
        self.n_group = n_groups
        self.n_local_groups = n_local_groups
        self.window_size = window_size
        self.q_lora_rank = q_lora_rank
        self.compress_ratio = compress_ratio
        self.softmax_scale = self.head_dim**-0.5

        # MLA Args
        self.wq_a = kwargs["wq_a"]
        self.wq_b = kwargs["wq_b"]
        self.wkv = kwargs["wkv"]
        self.q_norm = kwargs["q_norm"]
        self.q_norm_without_weight = kwargs["q_norm_without_weight"]
        self.kv_norm = kwargs["kv_norm"]

        self.indexer = kwargs.get("indexer")
        self.compressor = kwargs.get("compressor")

        self.wo_a = kwargs["wo_a"]
        self.wo_b = kwargs["wo_b"]

        self.eps = kwargs["eps"]

        self.attn_sink = kwargs["attn_sink"]

        ascend_config = get_ascend_config()
        self.multistream_dsa_preprocess = ascend_config.multistream_dsa_preprocess

        self.vllm_config = get_current_vllm_config()

        # indexer param
        if self.indexer is not None:
            self.indexer_heads: int = self.indexer.n_heads
            self.inderxer_dim: int = self.indexer.head_dim
            self.inderxer_wq_b = self.indexer.wq_b
            self.weights_proj = self.indexer.weights_proj
            self.indexer_softmax_scale = self.inderxer_dim**-0.5

            self.indexer_compress = self.indexer.compressor

            # indexer_compressor
            self.indexcom_ape = self.indexer.compressor.ape
            self.indexcom_wkv = self.indexer.compressor.wkv
            self.indexcom_wgate = self.indexer.compressor.wgate
            self.indexcom_norm = self.indexer.compressor.norm

            self.indexcom_head_dim = self.indexer.compressor.head_dim
            self.indexcom_rotate = self.indexer.compressor.rotate
            self.index_topk = self.indexer.index_topk

        # compress param
        if self.compressor is not None:
            self.compressor_head_dim = self.compressor.head_dim
            self.compressor_overlap = self.compressor.overlap
            self.compressor_rotate = self.compressor.rotate

            self.compressor_ape = self.compressor.ape
            self.compressor_wkv = self.compressor.wkv
            self.compressor_wgate = self.compressor.wgate
            self.compressor_norm = self.compressor.norm
            self.compressor_norm_eps = self.compressor.norm_eps

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        pass

    # TODO: cast to bfloat16 to speed up
    def rope_single(self, x, cos, sin, inverse=False):
        if inverse:
            sin = -sin
        tnd_layout = 1
        if len(x.shape) == 3:
            num_tokens, num_heads, rotary_dim = x.shape
        else:
            tnd_layout = 0
            _, num_tokens, num_heads, rotary_dim = x.shape
        x_rot = torch_npu.npu_rotary_mul(
            x.reshape(num_tokens, num_heads, 1, rotary_dim), cos, sin, rotary_mode="interleave"
        )
        if tnd_layout:
            x = x_rot.reshape(num_tokens, -1, rotary_dim)
        else:
            x = x_rot.reshape(1, num_tokens, -1, rotary_dim)
        return x

    def forward(  # type: ignore[override]
        self,
        layer_name,
        hidden_states: torch.Tensor,  # query in unified attn
        kv_cache: tuple[torch.Tensor],
        attn_metadata: list[M],
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)
        if not isinstance(attn_metadata, list):
            attn_metadata = [attn_metadata]
        output_padded = output
        # Process for Flash Comm V1
        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states, need_gather_q_kv)
        has_prefill = attn_metadata[0].num_prefills > 0  # type: ignore[index]
        has_decode = attn_metadata[0].num_decodes > 0  # type: ignore[index]
        decode_tokens = attn_metadata[0].num_decode_tokens  # type: ignore[index]
        actual_tokens = attn_metadata[0].num_actual_tokens  # type: ignore[index]
        prefill_hidden_states = hidden_states[decode_tokens:actual_tokens]
        decode_hidden_states = hidden_states[:decode_tokens]
        phase = "mixed" if has_decode and has_prefill else "prefill" if has_prefill else "decode"
        if has_prefill:
            _dsa_debug_check_finite(
                prefill_hidden_states,
                layer_name,
                phase,
                "prefill_hidden_states",
                self.compress_ratio,
                attn_metadata,
            )
        if has_decode and has_prefill:
            _dsa_debug_check_finite(
                hidden_states[:actual_tokens],
                layer_name,
                "mixed",
                "forward_input_hidden_states",
                self.compress_ratio,
                attn_metadata,
            )

        forward_context = get_forward_context()
        o_proj_input_shape = (forward_context.num_tokens, self.n_local_heads, self.head_dim)
        o_proj_input = torch.empty(o_proj_input_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        if has_prefill:
            assert attn_metadata[0].prefill is not None
            output_prefill = self._forward_prefill(layer_name, prefill_hidden_states, kv_cache, attn_metadata)  # type: ignore[arg-type]
            _dsa_debug_check_finite(
                output_prefill,
                layer_name,
                phase,
                "prefill_attn_output",
                self.compress_ratio,
                attn_metadata,
            )
            o_proj_input[decode_tokens:actual_tokens] = output_prefill
            cos = attn_metadata[0].prefill.cos[layer_name]  # type: ignore[index]
            sin = attn_metadata[0].prefill.sin[layer_name]  # type: ignore[index]

        if has_decode:
            assert attn_metadata[0].decode is not None  # type: ignore[index]
            output_decode = self._forward_decode(layer_name, decode_hidden_states, kv_cache, attn_metadata)  # type: ignore[arg-type]
            _dsa_debug_check_finite(
                output_decode,
                layer_name,
                "mixed_decode" if has_prefill else "decode",
                "decode_attn_output",
                self.compress_ratio,
                attn_metadata,
            )
            o_proj_input[:decode_tokens] = output_decode
            cos = attn_metadata[0].decode.cos[layer_name]  # type: ignore[index]
            sin = attn_metadata[0].decode.sin[layer_name]  # type: ignore[index]

        cos = attn_metadata[0].cos[layer_name]  # type: ignore[index]
        sin = attn_metadata[0].sin[layer_name]  # type: ignore[index]
        num_tokens = o_proj_input.shape[0]
        if actual_tokens < num_tokens:
            o_proj_input[actual_tokens:].zero_()
        if has_prefill:
            _dsa_debug_check_finite(
                o_proj_input[decode_tokens:actual_tokens],
                layer_name,
                phase,
                "prefill_o_proj_input_before_rotary",
                self.compress_ratio,
                attn_metadata,
            )
        if has_decode:
            _dsa_debug_check_finite(
                o_proj_input[:decode_tokens],
                layer_name,
                "mixed_decode" if has_prefill else "decode",
                "decode_o_proj_input_before_rotary",
                self.compress_ratio,
                attn_metadata,
            )
        if has_decode and has_prefill:
            _dsa_debug_check_finite(
                o_proj_input[:actual_tokens],
                layer_name,
                "mixed",
                "o_proj_input_before_rotary",
                self.compress_ratio,
                attn_metadata,
            )

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            o_proj_input.unsqueeze(1),
            cos,
            -sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )
        if has_prefill:
            _dsa_debug_check_finite(
                o_proj_input[decode_tokens:actual_tokens],
                layer_name,
                phase,
                "prefill_o_proj_input_after_rotary",
                self.compress_ratio,
                attn_metadata,
            )
        if has_decode:
            _dsa_debug_check_finite(
                o_proj_input[:decode_tokens],
                layer_name,
                "mixed_decode" if has_prefill else "decode",
                "decode_o_proj_input_after_rotary",
                self.compress_ratio,
                attn_metadata,
            )
        if has_decode and has_prefill:
            _dsa_debug_check_finite(
                o_proj_input[:actual_tokens],
                layer_name,
                "mixed",
                "o_proj_input_after_rotary",
                self.compress_ratio,
                attn_metadata,
            )

        # o
        if get_ascend_device_type() in {AscendDeviceType.A5}:

            o = o_proj_input.view(num_tokens, self.n_local_groups, -1)
            o, swiglu_out_scale = torch_npu.npu_dynamic_mx_quant(o, dst_type=torch.float8_e4m3fn)
            o = torch_npu.npu_transpose_quant_batchmatmul(o, self.wo_a.weight, dtype=torch.bfloat16, bias=None, group_sizes=(0, 0, 32),
                                                        x1_scale=swiglu_out_scale.view(torch.float8_e8m0fnu), x2_scale=self.wo_a.weight_scale.view(torch.float8_e8m0fnu),
                                                        perm_x1=(1,0,2), perm_x2=(0,1,2), perm_y=(1,0,2))
            if has_prefill:
                _dsa_debug_check_finite(
                    o[decode_tokens:actual_tokens],
                    layer_name,
                    phase,
                    "prefill_wo_a_output",
                    self.compress_ratio,
                    attn_metadata,
                )
            if has_decode:
                _dsa_debug_check_finite(
                    o[:decode_tokens],
                    layer_name,
                    "mixed_decode" if has_prefill else "decode",
                    "decode_wo_a_output",
                    self.compress_ratio,
                    attn_metadata,
                )
            if has_decode and has_prefill:
                _dsa_debug_check_finite(
                    o[:actual_tokens],
                    layer_name,
                    "mixed",
                    "wo_a_output",
                    self.compress_ratio,
                    attn_metadata,
                )
            o = o.reshape(num_tokens, -1)
            output[...] = self.wo_b(o)
        else:
            o_proj_input = o_proj_input.view(num_tokens, self.n_local_groups, -1)
            if olora_tp_enable():
                o_proj_input = self.wo_a(o_proj_input)
            else:
            # wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
            # o = torch.einsum("tgd,grd->tgr", o, wo_a)
                o_proj_input = torch_npu.npu_transpose_batchmatmul(
                    o_proj_input,
                    self.wo_a.weight,
                    bias=None,
                    scale=None,
                    perm_x1=(1, 0, 2),
                    perm_x2=(0, 1, 2),
                    perm_y=(1, 0, 2),
                    batch_split_factor=1,
                )
            o_proj_input = o_proj_input.reshape(num_tokens, -1)
            output[...] = self.wo_b(o_proj_input)

        if actual_tokens < output.shape[0]:
            output[actual_tokens:].zero_()
        if has_prefill:
            _dsa_debug_check_finite(
                output[decode_tokens:actual_tokens],
                layer_name,
                phase,
                "prefill_forward_output",
                self.compress_ratio,
                attn_metadata,
            )
        if has_decode:
            _dsa_debug_check_finite(
                output[:decode_tokens],
                layer_name,
                "mixed_decode" if has_prefill else "decode",
                "decode_forward_output",
                self.compress_ratio,
                attn_metadata,
            )
        if has_decode and has_prefill:
            _dsa_debug_check_finite(
                output[:actual_tokens],
                layer_name,
                "mixed",
                "forward_output",
                self.compress_ratio,
                attn_metadata,
            )
        return output_padded

    def _forward_prefill(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendDSAMetadata,
    ):
        compress_common_attn_metadata = None
        is_a5 = get_ascend_device_type() in {AscendDeviceType.A5}

        if self.compress_ratio == 4:
            if is_a5:
                (compress_kv_cache, swa_kv_cache, state_cache, _, _, _, _) = kv_cache
            else:
                (compress_kv_cache, swa_kv_cache, state_cache, _, _, _) = kv_cache  # type: ignore[misc]
            # sorted keys: [attn, compressor.state_cache, indexer.compressor.state_cache, indexer.k_cache, swa_cache]
            (compressor_attn_metadata, compressor_kv_state_metadata, _, _, swa_metadata) = attn_metadata  # type: ignore[misc]
            compress_common_attn_metadata = compressor_attn_metadata
        elif self.compress_ratio == 128:
            if is_a5:
                (compress_kv_cache, swa_kv_cache, state_cache, _, _, _, _) = kv_cache
            else:
                (compress_kv_cache, swa_kv_cache, state_cache, _, _, _) = kv_cache  # type: ignore[misc]
            # sorted keys: [attn, compressor.state_cache, swa_cache]
            (compressor_attn_metadata, compressor_kv_state_metadata, swa_metadata) = attn_metadata  # type: ignore[misc]
            compress_common_attn_metadata = compressor_attn_metadata
        else:
            if is_a5:
                (_, swa_kv_cache, _, _, _, _, _) = kv_cache
            else:
                (
                    _,
                    swa_kv_cache,
                    _,
                    _,
                    _,
                    _,
                ) = kv_cache  # type: ignore[misc]
            # sorted keys: [swa_cache]
            (swa_metadata,) = attn_metadata  # type: ignore[misc]
            compress_common_attn_metadata = swa_metadata

        assert compress_common_attn_metadata.prefill is not None
        cos = compress_common_attn_metadata.prefill.cos[layer_name]
        sin = compress_common_attn_metadata.prefill.sin[layer_name]
        actual_seq_lengths_query = compress_common_attn_metadata.prefill.query_start_loc
        actual_seq_lengths_key = compress_common_attn_metadata.prefill.seq_lens
        phase = (
            "mixed_prefill"
            if attn_metadata[0].num_decodes > 0 and attn_metadata[0].num_prefills > 0
            else "prefill"
        )
        _dsa_debug_check_finite(
            hidden_states,
            layer_name,
            phase,
            "prefill_inner_hidden_states",
            self.compress_ratio,
            attn_metadata,
        )

        # mlaprolog
        # q
        qr = self.q_norm(self.wq_a(hidden_states))
        _dsa_debug_check_finite(
            qr,
            layer_name,
            phase,
            "prefill_qr_after_q_norm",
            self.compress_ratio,
            attn_metadata,
        )
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        _dsa_debug_check_finite(
            q,
            layer_name,
            phase,
            "prefill_q_after_wq_b",
            self.compress_ratio,
            attn_metadata,
        )
        if is_a5:
            q = self.q_norm_without_weight(q)
        else:
            q = triton_q_rms(q, self.eps)
        _dsa_debug_check_finite(
            q,
            layer_name,
            phase,
            "prefill_q_after_q_norm_without_weight",
            self.compress_ratio,
            attn_metadata,
        )

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )
        _dsa_debug_check_finite(
            q,
            layer_name,
            phase,
            "prefill_q_after_rotary",
            self.compress_ratio,
            attn_metadata,
        )
        # win kv & tok_dis
        kv = self.wkv(hidden_states)
        _dsa_debug_check_finite(
            kv,
            layer_name,
            phase,
            "prefill_kv_after_wkv",
            self.compress_ratio,
            attn_metadata,
        )
        kv = self.kv_norm(kv)
        _dsa_debug_check_finite(
            kv,
            layer_name,
            phase,
            "prefill_kv_after_kv_norm",
            self.compress_ratio,
            attn_metadata,
        )
        assert self.rope_head_dim is not None
        kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            kv.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )
        _dsa_debug_check_finite(
            kv,
            layer_name,
            phase,
            "prefill_kv_after_rotary",
            self.compress_ratio,
            attn_metadata,
        )

        # swa exec kv
        if is_a5:
            _dsa_debug_log_slot_mapping(
                swa_metadata.prefill.slot_mapping,
                layer_name,
                phase,
                "prefill_swa_slot_mapping_before_epilog",
                self.compress_ratio,
                attn_metadata,
            )
            _dsa_debug_log_bad_slot_rows(
                swa_metadata.prefill.slot_mapping,
                layer_name,
                phase,
                "prefill_swa_before_epilog",
                self.compress_ratio,
                kv.view(-1, kv.shape[-1]).shape[0],
                attn_metadata,
            )
            swa_kv, swa_slot_mapping = _dsa_select_valid_prefill_rows(
                kv,
                swa_metadata.prefill.slot_mapping,
                layer_name,
                phase,
                "prefill_swa_before_epilog",
                self.compress_ratio,
                attn_metadata,
            )
            _dsa_debug_check_finite(
                swa_kv,
                layer_name,
                phase,
                "prefill_swa_kv_valid_rows_before_epilog",
                self.compress_ratio,
                attn_metadata,
            )
            if swa_kv is not None and swa_slot_mapping.numel() > 0:
                torch.ops._C_ascend.kv_compress_epilog(
                    kv_compress_cache=swa_kv_cache.view(-1, 1, swa_kv_cache.shape[-1]),
                    x=swa_kv.reshape(-1, swa_kv.shape[-1]),
                    slot_mapping=swa_slot_mapping,
                    quant_group_size=64,
                    quant_mode=2,
                    round_scale_flag=True,
                    layout=1,
                )
        else:
            torch.ops._C_ascend.npu_scatter_nd_update_v2(swa_kv_cache, swa_metadata.prefill.slot_mapping, kv)

        compress_cos = compress_common_attn_metadata.prefill.compress_cos[layer_name]
        compress_sin = compress_common_attn_metadata.prefill.compress_sin[layer_name]
        if self.compress_ratio > 1:
            compress_topk_idxs = None
            if self.compress_ratio == 4:
                compress_topk_idxs = self.indexer_select_qli(
                    x=hidden_states,
                    qr=qr,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,  # type: ignore[arg-type]
                    cos=cos,
                    sin=sin,
                    compressed_cos=compress_cos,
                    compressed_sin=compress_sin,
                    actual_seq_lengths_query=actual_seq_lengths_query,
                    actual_seq_lengths_key=actual_seq_lengths_key,
                    with_prefill=True,
                )

            coff = 2 if self.compressor_overlap else 1

            # compressor
            compressed_kv = torch.ops._C_ascend.compressor(
                hidden_states,
                self.compressor_wkv.weight,
                self.compressor_wgate.weight,
                # TODO(yilin): adapt to the latest operator
                state_cache.squeeze(-2),
                self.compressor_ape,
                self.compressor_norm.weight,
                compress_sin.view(-1, compress_sin.shape[-1]),
                compress_cos.view(-1, compress_cos.shape[-1]),
                # TODO(lxs): adapt the block table
                state_block_table=compressor_kv_state_metadata.prefill.block_table,
                cu_seqlens=actual_seq_lengths_query,
                seqused=None,
                start_pos=compress_common_attn_metadata.prefill.start_pos,
                rope_head_dim=self.rope_head_dim,
                cmp_ratio=self.compress_ratio,
                coff=coff,
                norm_eps=self.compressor_norm_eps,
                rotary_mode=2,
                cache_mode=1,
            )
            _dsa_debug_check_finite(
                compressed_kv,
                layer_name,
                phase,
                "prefill_raw_compressed_kv_after_compressor",
                self.compress_ratio,
                attn_metadata,
            )

            if compressed_kv.numel() == 0:
                compressed_kv = None

            # kv_compress_epilog
            if is_a5:
                _dsa_debug_log_slot_mapping(
                    compressor_attn_metadata.prefill.slot_mapping,
                    layer_name,
                    phase,
                    "prefill_compressed_slot_mapping_before_epilog",
                    self.compress_ratio,
                    attn_metadata,
                )
                compressed_kv, compressed_slot_mapping = _dsa_select_valid_compressed_prefill_rows(
                    compressed_kv,
                    compressor_attn_metadata.prefill.slot_mapping,
                    layer_name,
                    phase,
                    "prefill_compressed_before_epilog",
                    self.compress_ratio,
                    attn_metadata,
                )
                _dsa_debug_check_finite(
                    compressed_kv,
                    layer_name,
                    phase,
                    "prefill_compressed_kv_valid_rows_before_epilog",
                    self.compress_ratio,
                    attn_metadata,
                )
                if compressed_kv is not None and compressed_slot_mapping.numel() > 0:
                    torch.ops._C_ascend.kv_compress_epilog(
                        kv_compress_cache=compress_kv_cache.view(-1, 1, compress_kv_cache.shape[-1]),
                        x=compressed_kv.reshape(-1, compressed_kv.shape[-1]),
                        slot_mapping=compressed_slot_mapping,
                        quant_group_size=64,
                        quant_mode=2,
                        round_scale_flag=True,
                        layout=1,
                    )
            else:
                torch.ops._C_ascend.npu_scatter_nd_update_v2(
                    compress_kv_cache, compressor_attn_metadata.prefill.slot_mapping, compressed_kv
                )
        if is_a5:
            _dsa_debug_log_sas_inputs(
                swa_metadata.prefill,
                layer_name,
                phase,
                "prefill_swa_sas_before_attn",
                self.compress_ratio,
                attn_metadata,
            )
            if self.compress_ratio > 1:
                _dsa_debug_log_sas_inputs(
                    compressor_attn_metadata.prefill,
                    layer_name,
                    phase,
                    "prefill_compressed_sas_before_attn",
                    self.compress_ratio,
                    attn_metadata,
                )

        if is_a5:
            if self.compress_ratio <= 1:
                attn_output = torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv(
                    q,
                    ori_kv=swa_kv_cache,
                    ori_block_table=swa_metadata.prefill.block_table,
                    cu_seqlens_q=actual_seq_lengths_query,
                    seqused_kv=actual_seq_lengths_key,
                    sinks=self.attn_sink,
                    metadata=compress_common_attn_metadata.prefill.sas_metadata,
                    kv_quant_mode=1,
                    tile_size=64,
                    rope_head_dim=64,
                    softmax_scale=self.softmax_scale,
                    cmp_ratio=1,
                    ori_mask_mode=4,
                    ori_win_left=self.window_size - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                )[0]
            elif self.compress_ratio == 4:
                attn_output = torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv(
                    q,
                    ori_kv=swa_kv_cache,
                    cmp_kv=compress_kv_cache,
                    cmp_sparse_indices=compress_topk_idxs,
                    ori_block_table=swa_metadata.prefill.block_table,
                    cmp_block_table=compressor_attn_metadata.prefill.block_table,
                    cu_seqlens_q=actual_seq_lengths_query,
                    seqused_kv=actual_seq_lengths_key,
                    sinks=self.attn_sink,
                    metadata=compress_common_attn_metadata.prefill.sas_metadata,
                    kv_quant_mode=1,
                    tile_size=64,
                    rope_head_dim=64,
                    softmax_scale=self.softmax_scale,
                    cmp_ratio=self.compress_ratio,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self.window_size - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                )[0]
            else:
                attn_output = torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv(
                    q,
                    ori_kv=swa_kv_cache,
                    cmp_kv=compress_kv_cache,
                    ori_block_table=swa_metadata.prefill.block_table,
                    cmp_block_table=compressor_attn_metadata.prefill.block_table,
                    cu_seqlens_q=actual_seq_lengths_query,
                    seqused_kv=actual_seq_lengths_key,
                    sinks=self.attn_sink,
                    metadata=compress_common_attn_metadata.prefill.sas_metadata,
                    kv_quant_mode=1,
                    tile_size=64,
                    rope_head_dim=64,
                    softmax_scale=self.softmax_scale,
                    cmp_ratio=self.compress_ratio,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self.window_size - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                )[0]
        elif self.compress_ratio <= 1:
            attn_output = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
                q,
                ori_kv=swa_kv_cache,
                ori_block_table=swa_metadata.prefill.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                cu_seqlens_ori_kv=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=compress_common_attn_metadata.prefill.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio,
                ori_mask_mode=4,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
            )[0]
        elif self.compress_ratio == 4:
            attn_output = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                cmp_sparse_indices=compress_topk_idxs,
                ori_block_table=swa_metadata.prefill.block_table,
                cmp_block_table=compressor_attn_metadata.prefill.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                cu_seqlens_ori_kv=actual_seq_lengths_query,
                cu_seqlens_cmp_kv=compress_common_attn_metadata.prefill.cu_c4_cmp_seqlen_list,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=compress_common_attn_metadata.prefill.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
            )[0]
        else:
            attn_output = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                ori_block_table=swa_metadata.prefill.block_table,
                cmp_block_table=compressor_attn_metadata.prefill.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                cu_seqlens_ori_kv=actual_seq_lengths_query,
                cu_seqlens_cmp_kv=compress_common_attn_metadata.prefill.cu_c128_cmp_seqlen_list,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=compressor_attn_metadata.prefill.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
            )[0]
        _dsa_debug_check_finite(
            attn_output,
            layer_name,
            phase,
            "prefill_sparse_attn_output",
            self.compress_ratio,
            attn_metadata,
        )
        return attn_output

    def _forward_decode(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple,
        attn_metadata: AscendDSAMetadata,
    ):
        assert attn_metadata[0].decode is not None  # type: ignore[index]
        compress_common_attn_metadata = None
        is_a5 = get_ascend_device_type() in {AscendDeviceType.A5}

        if self.compress_ratio == 4:
            if is_a5:
                (compress_kv_cache, swa_kv_cache, state_cache, _, _, _, _) = kv_cache
            else:
                (compress_kv_cache, swa_kv_cache, state_cache, _, _, _) = kv_cache  # type: ignore[misc]
            # sorted keys: [attn, compressor.state_cache, indexer.compressor.state_cache, indexer.k_cache, swa_cache]
            (compressor_attn_metadata, compressor_kv_state_metadata, _, _, swa_metadata) = attn_metadata  # type: ignore[misc]
            compress_common_attn_metadata = compressor_attn_metadata
        elif self.compress_ratio == 128:
            if is_a5:
                (compress_kv_cache, swa_kv_cache, state_cache, _, _, _, _) = kv_cache
            else:
                (compress_kv_cache, swa_kv_cache, state_cache, _, _, _) = kv_cache  # type: ignore[misc]
            # sorted keys: [attn, compressor.state_cache, swa_cache]
            (compressor_attn_metadata, compressor_kv_state_metadata, swa_metadata) = attn_metadata  # type: ignore[misc]
            compress_common_attn_metadata = compressor_attn_metadata
        else:
            if is_a5:
                (_, swa_kv_cache, _, _, _, _, _) = kv_cache
            else:
                (_, swa_kv_cache, _, _, _, _) = kv_cache
            # sorted keys: [swa_cache]
            (swa_metadata,) = attn_metadata  # type: ignore[misc]
            compress_common_attn_metadata = swa_metadata
        cos = compress_common_attn_metadata.decode.cos[layer_name]
        sin = compress_common_attn_metadata.decode.sin[layer_name]
        actual_seq_lengths_query = compress_common_attn_metadata.decode.query_start_loc
        actual_seq_lengths_key = compress_common_attn_metadata.decode.seq_lens
        phase = "mixed_decode" if attn_metadata[0].num_prefills > 0 else "decode"  # type: ignore[index]
        _dsa_debug_check_finite(
            hidden_states,
            layer_name,
            phase,
            "decode_inner_hidden_states",
            self.compress_ratio,
            attn_metadata,
        )
        wait_hidden_state_cal_event = (
            torch.npu.current_stream().record_event() if self.multistream_dsa_preprocess else None
        )

        # q
        if (not isinstance(self.wq_b.quant_method, AscendUnquantizedLinearMethod)) and isinstance(
            self.wq_b.quant_method.quant_method, AscendW8A8DynamicLinearMethod
        ):
            q_a = self.wq_a(hidden_states)
            qr, qr_pertoken_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
                q_a, self.q_norm.weight, epsilon=self.eps
            )
            q = torch_npu.npu_quant_matmul(
                qr,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=qr_pertoken_scale,
                bias=self.wq_b.bias,
                output_dtype=hidden_states.dtype,
            ).unflatten(-1, (self.n_local_heads, self.head_dim))
        else:
            qr = q = self.q_norm(self.wq_a(hidden_states))
            q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
            qr_pertoken_scale = None
        _dsa_debug_check_finite(
            q,
            layer_name,
            phase,
            "decode_q_after_wq_b",
            self.compress_ratio,
            attn_metadata,
        )

        if is_a5:
            q = self.q_norm_without_weight(q)
        else:
            q = triton_q_rms(q, self.eps)
        _dsa_debug_check_finite(
            q,
            layer_name,
            phase,
            "decode_q_after_q_norm_without_weight",
            self.compress_ratio,
            attn_metadata,
        )

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )
        _dsa_debug_check_finite(
            q,
            layer_name,
            phase,
            "decode_q_after_rotary",
            self.compress_ratio,
            attn_metadata,
        )

        with npu_stream_switch(attention_calculation_stream(), enabled=self.multistream_dsa_preprocess):
            if wait_hidden_state_cal_event:
                torch.npu.current_stream().wait_event(wait_hidden_state_cal_event)

            # win kv & tok_dis
            kv = self.wkv(hidden_states)
            _dsa_debug_check_finite(
                kv,
                layer_name,
                phase,
                "decode_kv_after_wkv",
                self.compress_ratio,
                attn_metadata,
            )
            kv = self.kv_norm(kv)
            _dsa_debug_check_finite(
                kv,
                layer_name,
                phase,
                "decode_kv_after_kv_norm",
                self.compress_ratio,
                attn_metadata,
            )
            assert self.rope_head_dim is not None
            kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)

            torch.ops._C_ascend.inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[self.nope_head_dim, self.head_dim],
            )
            _dsa_debug_check_finite(
                kv,
                layer_name,
                phase,
                "decode_kv_after_rotary",
                self.compress_ratio,
                attn_metadata,
            )

            # swa exec kv
            if is_a5:
                _dsa_debug_log_slot_mapping(
                    swa_metadata.decode.slot_mapping,
                    layer_name,
                    phase,
                    "decode_swa_slot_mapping_before_epilog",
                    self.compress_ratio,
                    attn_metadata,
                )
                _dsa_debug_log_bad_slot_rows(
                    swa_metadata.decode.slot_mapping,
                    layer_name,
                    phase,
                    "decode_swa_before_epilog",
                    self.compress_ratio,
                    kv.view(-1, kv.shape[-1]).shape[0],
                    attn_metadata,
                )
                _dsa_debug_log_decode_slot_collision(
                    swa_metadata.decode.slot_mapping,
                    layer_name,
                    phase,
                    "decode_swa_before_epilog",
                    self.compress_ratio,
                    attn_metadata,
                )
                torch.ops._C_ascend.kv_compress_epilog(
                    kv_compress_cache=swa_kv_cache.view(-1, 1, swa_kv_cache.shape[-1]),
                    x=kv.view(-1, kv.shape[-1]),
                    slot_mapping=swa_metadata.decode.slot_mapping,
                    quant_group_size=64,
                    quant_mode=2,
                    round_scale_flag=True,
                    layout=1,
                )
            else:
                torch.ops._C_ascend.npu_scatter_nd_update_v2(swa_kv_cache, swa_metadata.decode.slot_mapping, kv)

            wait_attention_cal_event = (
                torch.npu.current_stream().record_event() if self.multistream_dsa_preprocess else None
            )

        if wait_attention_cal_event:
            torch.npu.current_stream().wait_event(wait_attention_cal_event)

        if self.compress_ratio > 1:
            compress_cos = compress_common_attn_metadata.decode.compress_cos[layer_name]
            compress_sin = compress_common_attn_metadata.decode.compress_sin[layer_name]
            compress_topk_idxs = None
            if self.compress_ratio == 4:
                compress_topk_idxs = self.indexer_select_qli(
                    x=hidden_states,
                    qr=qr,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,  # type: ignore[arg-type]
                    cos=cos,
                    sin=sin,
                    compressed_cos=compress_cos,
                    compressed_sin=compress_sin,
                    actual_seq_lengths_query=actual_seq_lengths_query,
                    actual_seq_lengths_key=actual_seq_lengths_key,
                    with_prefill=False,
                    qr_pertoken_scale=qr_pertoken_scale,
                )

            coff = 2 if self.compressor_overlap else 1

            # compressor
            if is_a5:
                _dsa_debug_log_sas_inputs(
                    compressor_kv_state_metadata.decode,
                    layer_name,
                    phase,
                    "decode_compressor_state_before_compressor",
                    self.compress_ratio,
                    attn_metadata,
                )
            compressed_kv = torch.ops._C_ascend.compressor(
                hidden_states,
                self.compressor_wkv.weight,
                self.compressor_wgate.weight,
                state_cache.squeeze(-2),
                self.compressor_ape,
                self.compressor_norm.weight,
                compress_sin.view(-1, compress_sin.shape[-1]),
                compress_cos.view(-1, compress_cos.shape[-1]),
                state_block_table=compressor_kv_state_metadata.decode.block_table,
                cu_seqlens=actual_seq_lengths_query,
                seqused=None,
                start_pos=compress_common_attn_metadata.decode.start_pos,
                rope_head_dim=self.rope_head_dim,
                cmp_ratio=self.compress_ratio,
                coff=coff,
                norm_eps=self.compressor_norm_eps,
                rotary_mode=2,
                cache_mode=1,
            )
            _dsa_debug_check_finite(
                compressed_kv,
                layer_name,
                phase,
                "decode_raw_compressed_kv_after_compressor",
                self.compress_ratio,
                attn_metadata,
            )
            # kv_compress_epilog
            if is_a5:
                if len(compressor_attn_metadata.decode.slot_mapping):
                    _dsa_debug_log_slot_mapping(
                        compressor_attn_metadata.decode.slot_mapping,
                        layer_name,
                        phase,
                        "decode_compressed_slot_mapping_before_epilog",
                        self.compress_ratio,
                        attn_metadata,
                    )
                    _dsa_debug_log_bad_slot_rows(
                        compressor_attn_metadata.decode.slot_mapping,
                        layer_name,
                        phase,
                        "decode_compressed_before_epilog",
                        self.compress_ratio,
                        compressed_kv.reshape(-1, compressed_kv.shape[-1]).shape[0],
                        attn_metadata,
                    )
                    _dsa_debug_log_decode_slot_collision(
                        compressor_attn_metadata.decode.slot_mapping,
                        layer_name,
                        phase,
                        "decode_compressed_before_epilog",
                        self.compress_ratio,
                        attn_metadata,
                    )
                    torch.ops._C_ascend.kv_compress_epilog(
                        kv_compress_cache=compress_kv_cache.view(-1, 1, compress_kv_cache.shape[-1]),
                        x=compressed_kv.reshape(-1, compressed_kv.shape[-1]),
                        slot_mapping=compressor_attn_metadata.decode.slot_mapping,
                        quant_group_size=64,
                        quant_mode=2,
                        round_scale_flag=True,
                        layout=1,
                    )
            else:
                torch.ops._C_ascend.npu_scatter_nd_update_v2(
                    compress_kv_cache, compressor_attn_metadata.decode.slot_mapping, compressed_kv
                )
        if is_a5:
            _dsa_debug_log_sas_inputs(
                swa_metadata.decode,
                layer_name,
                phase,
                "decode_swa_sas_before_attn",
                self.compress_ratio,
                attn_metadata,
            )
            if self.compress_ratio > 1:
                _dsa_debug_log_sas_inputs(
                    compressor_attn_metadata.decode,
                    layer_name,
                    phase,
                    "decode_compressed_sas_before_attn",
                    self.compress_ratio,
                    attn_metadata,
                )
        if is_a5:
            if self.compress_ratio <= 1:
                attn_output = torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv(
                    q,
                    ori_kv=swa_kv_cache,
                    ori_block_table=swa_metadata.decode.block_table,
                    cu_seqlens_q=actual_seq_lengths_query,
                    seqused_kv=actual_seq_lengths_key,
                    sinks=self.attn_sink,
                    metadata=swa_metadata.decode.sas_metadata,
                    kv_quant_mode=1,
                    cmp_ratio=1,
                    tile_size=64,
                    rope_head_dim=64,
                    softmax_scale=self.softmax_scale,
                    ori_mask_mode=4,
                    ori_win_left=self.window_size - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                )[0]
            elif self.compress_ratio == 4:
                attn_output = torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv(
                    q,
                    ori_kv=swa_kv_cache,
                    cmp_kv=compress_kv_cache,
                    cmp_sparse_indices=compress_topk_idxs,
                    ori_block_table=swa_metadata.decode.block_table,
                    cmp_block_table=compressor_attn_metadata.decode.block_table,
                    cu_seqlens_q=actual_seq_lengths_query,
                    seqused_kv=actual_seq_lengths_key,
                    sinks=self.attn_sink,
                    metadata=compressor_attn_metadata.decode.sas_metadata,
                    kv_quant_mode=1,
                    tile_size=64,
                    rope_head_dim=64,
                    softmax_scale=self.softmax_scale,
                    cmp_ratio=self.compress_ratio,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self.window_size - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                )[0]
            else:
                attn_output = torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv(
                    q,
                    ori_kv=swa_kv_cache,
                    cmp_kv=compress_kv_cache,
                    ori_block_table=swa_metadata.decode.block_table,
                    cmp_block_table=compressor_attn_metadata.decode.block_table,
                    cu_seqlens_q=actual_seq_lengths_query,
                    seqused_kv=actual_seq_lengths_key,
                    sinks=self.attn_sink,
                    metadata=compressor_attn_metadata.decode.sas_metadata,
                    kv_quant_mode=1,
                    tile_size=64,
                    rope_head_dim=64,
                    softmax_scale=self.softmax_scale,
                    cmp_ratio=self.compress_ratio,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self.window_size - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                )[0]
        elif self.compress_ratio <= 1:
            attn_output = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
                q,
                ori_kv=swa_kv_cache,
                ori_block_table=swa_metadata.decode.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=swa_metadata.decode.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio,
                ori_mask_mode=4,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
            )[0]
        elif self.compress_ratio == 4:
            attn_output = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                cmp_sparse_indices=compress_topk_idxs,
                ori_block_table=swa_metadata.decode.block_table,
                cmp_block_table=compressor_attn_metadata.decode.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=compressor_attn_metadata.decode.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
            )[0]
        else:
            attn_output = torch.ops._C_ascend.npu_sparse_attn_sharedkv(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                ori_block_table=swa_metadata.decode.block_table,
                cmp_block_table=compressor_attn_metadata.decode.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=compressor_attn_metadata.decode.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
            )[0]
        _dsa_debug_check_finite(
            attn_output,
            layer_name,
            phase,
            "decode_sparse_attn_output",
            self.compress_ratio,
            attn_metadata,
        )
        return attn_output

    def indexer_select_qli(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: list[M],
        cos: torch.Tensor,
        sin: torch.Tensor,
        compressed_cos: torch.Tensor,
        compressed_sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        with_prefill: bool = False,
        qr_pertoken_scale: torch.Tensor = None,
    ):
        if get_ascend_device_type() in {AscendDeviceType.A5}:
            (_, _, _, indexer_state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache) = kv_cache
        else:
            (_, _, _, indexer_state_cache, indexer_k_cache, indexer_scale_cache) = kv_cache
        # sorted keys: [attn, compressor.state_cache, indexer.compressor.state_cache, indexer.k_cache, swa_cache]
        (_, _, indexer_kv_state_metadata, indexer_kv_scale_metadata, _) = attn_metadata  # type: ignore[misc]

        if (
            (not isinstance(self.inderxer_wq_b.quant_method, AscendUnquantizedLinearMethod))
            and isinstance(self.inderxer_wq_b.quant_method.quant_method, AscendW8A8DynamicLinearMethod)
            and qr_pertoken_scale is not None
            and get_ascend_device_type() not in {AscendDeviceType.A5}
        ):
            q = torch_npu.npu_quant_matmul(
                qr,
                self.inderxer_wq_b.weight,
                self.inderxer_wq_b.weight_scale,
                pertoken_scale=qr_pertoken_scale,
                bias=self.inderxer_wq_b.bias,
                output_dtype=x.dtype,
            )
        else:
            q = self.inderxer_wq_b(qr)
        q = q.view(-1, self.indexer_heads, self.indexcom_head_dim)  # [T, N, D]

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.indexcom_head_dim - self.rope_head_dim, self.indexcom_head_dim],
        )

        q = rotate_activation(q, indexer_kv_scale_metadata.hadamard)
        coff = 2 if self.compressor_overlap else 1

        if with_prefill:
            assert indexer_kv_scale_metadata.prefill is not None
            kv_block_table = indexer_kv_state_metadata.prefill.block_table  # type: ignore[union-attr]
            start_pos = indexer_kv_scale_metadata.prefill.start_pos
        else:
            assert indexer_kv_scale_metadata.decode is not None
            kv_block_table = indexer_kv_state_metadata.decode.block_table  # type: ignore[union-attr]
            start_pos = indexer_kv_scale_metadata.decode.start_pos

        kv = torch.ops._C_ascend.compressor(
            x,
            self.indexcom_wkv.weight,
            self.indexcom_wgate.weight,
            indexer_state_cache.squeeze(-2),
            self.indexcom_ape,
            self.indexcom_norm.weight,
            compressed_sin.view(-1, compressed_sin.shape[-1]),
            compressed_cos.view(-1, compressed_cos.shape[-1]),
            state_block_table=kv_block_table,
            cu_seqlens=actual_seq_lengths_query,
            seqused=None,
            start_pos=start_pos,
            rope_head_dim=self.rope_head_dim,
            cmp_ratio=self.compress_ratio,
            coff=coff,
            norm_eps=self.compressor_norm_eps,
            rotary_mode=2,
            cache_mode=1,
        )

        if kv.numel() == 0:
            kv = None
        elif self.indexer.compressor.rotate:  # type: ignore[union-attr]
            kv = rotate_activation(kv, indexer_kv_scale_metadata.hadamard)

        weights = self.weights_proj(x) * (self.indexer_softmax_scale * self.indexer_heads**-0.5)

        soc_version = get_ascend_device_type()
        dst_type = torch.float8_e4m3fn if soc_version in {AscendDeviceType.A5} else torch.int8

        q, q_scale = torch_npu.npu_dynamic_quant(q, dst_type=dst_type)
        if kv is not None and get_ascend_device_type() not in {AscendDeviceType.A5}:
            kv, kv_scale = torch_npu.npu_dynamic_quant(kv, dst_type=dst_type)
            kv_scale = kv_scale.unsqueeze(-1)

        if soc_version not in {AscendDeviceType.A5}:
            q_scale = q_scale.to(torch.float16)
            if kv is not None:
                kv_scale = kv_scale.to(torch.float16)
                kv_scale = kv_scale.unsqueeze(-1)

        if with_prefill:
            assert indexer_kv_scale_metadata.prefill is not None
            if kv is not None:
                if soc_version in {AscendDeviceType.A5}:
                    torch.ops._C_ascend.indexer_compress_epilog_v2(
                        indexer_compress_cache=indexer_full_cache.view(torch.uint8),
                        x=kv,
                        slot_mapping=indexer_kv_scale_metadata.prefill.slot_mapping,
                        layout=2,
                    )
                else:
                    torch.ops._C_ascend.npu_scatter_nd_update_v2(
                        indexer_k_cache, indexer_kv_scale_metadata.prefill.slot_mapping, kv
                    )
                    torch.ops._C_ascend.npu_scatter_nd_update_v2(
                        indexer_scale_cache, indexer_kv_scale_metadata.prefill.slot_mapping, kv_scale
                    )
        else:
            assert indexer_kv_scale_metadata.decode is not None
            if kv is not None:
                if soc_version in {AscendDeviceType.A5}:
                    torch.ops._C_ascend.indexer_compress_epilog_v2(
                        indexer_compress_cache=indexer_full_cache.view(torch.uint8),
                        x=kv,
                        slot_mapping=indexer_kv_scale_metadata.decode.slot_mapping,
                        layout=2,
                    )
                else:
                    torch.ops._C_ascend.npu_scatter_nd_update_v2(
                        indexer_k_cache, indexer_kv_scale_metadata.decode.slot_mapping, kv
                    )
                    torch.ops._C_ascend.npu_scatter_nd_update_v2(
                        indexer_scale_cache, indexer_kv_scale_metadata.decode.slot_mapping, kv_scale
                    )

        if with_prefill:
            assert indexer_kv_scale_metadata.prefill is not None
            qlens = indexer_kv_scale_metadata.prefill.query_start_loc[1:]
            kvlens = indexer_kv_scale_metadata.prefill.seq_lens
            block_table = indexer_kv_scale_metadata.prefill.block_table
            qli_metadata = indexer_kv_scale_metadata.prefill.qli_metadata
        else:
            assert indexer_kv_scale_metadata.decode is not None
            qlens = indexer_kv_scale_metadata.decode.query_start_loc[1:]
            kvlens = indexer_kv_scale_metadata.decode.seq_lens
            block_table = indexer_kv_scale_metadata.decode.block_table
            qli_metadata = indexer_kv_scale_metadata.decode.qli_metadata

        topk_idxs, _ = torch.ops._C_ascend.npu_quant_lightning_indexer(
            query=q,
            key=indexer_k_cache,
            weights=weights.to(torch.float16) \
                if soc_version not in {AscendDeviceType.A5} else weights.float(),
            query_dequant_scale=q_scale \
                if soc_version not in {AscendDeviceType.A5} else q_scale.float(),
            key_dequant_scale=indexer_scale_cache.squeeze(-2).to(torch.float16) \
                if soc_version not in {AscendDeviceType.A5} else indexer_scale_cache.squeeze(-2).float(),
            actual_seq_lengths_query=qlens,
            actual_seq_lengths_key=kvlens,
            block_table=block_table,
            metadata=qli_metadata,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
            pre_tokens=(1 << 63) - 1,
            next_tokens=(1 << 63) - 1,
            cmp_ratio=4,
            return_value=False,
        )
        return topk_idxs
