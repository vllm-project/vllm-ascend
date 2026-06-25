import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeAlias

import torch
import torch.nn.functional as F
import torch_npu
import vllm.envs as envs_vllm
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backend import AttentionBackend, AttentionCGSupport, AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.attention.abstract import DSAAttentionImpl
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, split_decodes_and_prefills
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.cv_linear import CVLinearWrapper
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.ops.rope_dsv4 import get_cos_and_sin_dsa
from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicLinearMethod
from vllm_ascend.utils import (
    AscendDeviceType,
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

_DSV4_DSA_OVERLAP_STREAM = None


def dsv4_dsa_overlap_stream() -> torch.npu.Stream:
    global _DSV4_DSA_OVERLAP_STREAM
    if _DSV4_DSA_OVERLAP_STREAM is None:
        _DSV4_DSA_OVERLAP_STREAM = torch_npu.npu.Stream()
    return _DSV4_DSA_OVERLAP_STREAM


# mypy: disable-error-code="has-type"


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


def hadamard_linear(x: torch.Tensor, hadamard: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], int]:
    """
    Part 1 of rotate_activation: Execute F.linear (matrix multiplication).
    This runs in main stream, parallel with aux_stream kv_scatter.

    Returns:
        Tuple of (linear_output, original_shape, original_dim)
    """
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, hadamard)
    return out, x_shape, dim


def hadamard_scale(out: torch.Tensor, x_shape: tuple[int, ...], dim: int, scale: float = 1.0) -> torch.Tensor:
    """
    Part 2 of rotate_activation: Execute scale multiplication and reshape.
    This runs in main stream after aux_stream completes.
    """
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


def _is_w8a8_dynamic(linear) -> bool:
    """True iff ``linear`` is wired up with ``AscendW8A8DynamicLinearMethod``."""
    qm = getattr(linear, "quant_method", None)
    if qm is None or isinstance(qm, AscendUnquantizedLinearMethod):
        return False
    inner = getattr(qm, "quant_method", None)
    return isinstance(inner, AscendW8A8DynamicLinearMethod)


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
        from vllm_ascend.utils import enable_dsa_cp

        if enable_dsa_cp():
            from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPMetadataBuilder

            return AscendDSACPMetadataBuilder
        return AscendDSAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int, head_size: int) -> tuple[int, ...]:
        return num_blocks, block_size, num_kv_heads, head_size

    @staticmethod
    def get_scale_shape(num_blocks: int, block_size: int, scale_size: int) -> tuple[int, ...]:
        return num_blocks, block_size, scale_size

    @staticmethod
    def get_impl_cls() -> type["DSAAttentionImpl"]:
        from vllm_ascend.utils import enable_dsa_cp

        if enable_dsa_cp():
            from vllm_ascend.attention.context_parallel.dsa_cp import AscendDSACPImpl

            return AscendDSACPImpl
        return AscendDSAImpl

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [2, 4, 8, 16, 32, 64, 128]


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


DSAMetadataList: TypeAlias = list[AscendDSAMetadata]
DSAPrepareResult: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    AscendDSAMetadata,
    bool,
]


def _require_prefill_metadata(metadata: AscendDSAMetadata) -> AscendDSAPrefillMetadata:
    assert metadata.prefill is not None
    return metadata.prefill


def _require_decode_metadata(metadata: AscendDSAMetadata) -> AscendDSADecodeMetadata:
    assert metadata.decode is not None
    return metadata.decode


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
        kv_cache_spec: AscendMLAAttentionSpec,
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
        if get_ascend_device_type() in {AscendDeviceType.A5}:
            self.slot_mapping_shape = (vllm_config.scheduler_config.max_num_batched_tokens,)  # type: ignore
        else:
            self.slot_mapping_shape = (vllm_config.scheduler_config.max_num_batched_tokens, 2)  # type: ignore
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.spec_slot_mapping = [
                torch.zeros(self.slot_mapping_shape, dtype=torch.int32, device=self.device)
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
                if self.vllm_config.model_config.enable_sleep_mode:
                    # Sleep mode allocates KV inside CaMemAllocator; tag Hadamard so
                    # sleep/wake does not treat it as KV cache.
                    from vllm_ascend.device_allocator.camem import CaMemAllocator

                    allocator = CaMemAllocator.get_instance()
                    with allocator.use_allocation_tag(CaMemAllocator.sleep_persistent_tag):
                        AscendDSAMetadataBuilder.hadamard = torch.tensor(
                            hadamard(dim_padded, dtype=float), dtype=torch.float, device=self.device
                        ).to(torch.bfloat16)
                else:
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
        self.slot_mapping = torch.zeros(self.slot_mapping_shape, dtype=torch.int32, device=self.device)

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
            self.set_num_actual_tokens(common_attn_metadata)
            assert self.num_decodes + self.num_prefills == num_reqs
            assert self.num_decode_tokens + self.num_prefill_tokens == common_attn_metadata.num_actual_tokens
            num_input_tokens = common_attn_metadata.num_input_tokens
            input_positions = common_attn_metadata.positions[:num_input_tokens].long()
            self.common_ratio_to_sas_metadata["input_positions"] = input_positions
            if self.num_prefills:
                cos, sin = get_cos_and_sin_dsa(input_positions)
            else:
                cos, sin = get_cos_and_sin_dsa(input_positions, True)
            self.common_ratio_to_sas_metadata["cos"] = cos
            self.common_ratio_to_sas_metadata["sin"] = sin
            self.seq_lens = common_attn_metadata.seq_lens[:num_reqs]
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
            self.set_num_actual_tokens(common_attn_metadata)
            num_input_tokens = common_attn_metadata.num_input_tokens
            input_positions = self.common_ratio_to_sas_metadata["input_positions"]
            cos, sin = self.common_ratio_to_sas_metadata["cos"], self.common_ratio_to_sas_metadata["sin"]
            self.seq_lens = self.common_ratio_to_sas_metadata["seq_lens"]
            self.query_lens = self.common_ratio_to_sas_metadata["query_lens"]

        # NOTE: Currently, MTP-fullgraph is incompatibility pcp
        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        self.slot_mapping[:num_input_tokens] = DeviceOperator.format_dsa_slot_mapping(slot_mapping, self.block_size)

        self.graph_pad_size = common_attn_metadata.graph_pad_size
        block_table_size = self.get_block_table_size(common_attn_metadata, BUILD_METADATA_STEP_PREFILL)
        self.block_table = common_attn_metadata.block_table_tensor[:block_table_size]

        prefill_metadata = None
        if self.num_prefills > 0:
            prefill_metadata = self.build_prefill_metadata(common_prefix_len, common_attn_metadata)

        decode_metadata = None

        if self.num_decodes > 0:
            decode_metadata = self.build_decode_metadata(common_prefix_len, common_attn_metadata, num_reqs_actual)

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
        metadata_op = DeviceOperator.get_dsa_sparse_attn_metadata_op()
        metadata_kwargs = DeviceOperator.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
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
    ) -> AscendDSADecodeMetadata:
        assert self.decode_ratio_to_sas_metadata is not None
        if self.decode_ratio_to_sas_metadata.get("query_start_loc", None) is None:
            query_start_loc = common_attn_metadata.query_start_loc[: self.num_decodes + 1]
            self.decode_ratio_to_sas_metadata["query_start_loc"] = query_start_loc
            input_positions = common_attn_metadata.positions[: self.num_decode_tokens].long()
            self.decode_ratio_to_sas_metadata["input_positions"] = input_positions
            cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=True)
            self.decode_ratio_to_sas_metadata["cos"] = cos
            self.decode_ratio_to_sas_metadata["sin"] = sin

            query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: self.num_decodes + 1]
            input_positions_cpu = common_attn_metadata.positions_cpu[: self.num_decode_tokens].long()

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

        slot_mapping = DeviceOperator.pad_dsa_decode_slot_mapping(
            self.slot_mapping[:compressed_tokens_start], self.num_decode_tokens, self.compressor_ratio, self.num_decodes
        )

        assert self.start_pos_decode is not None
        self.start_pos_decode.fill_(0)
        self.start_pos_decode[: self.num_decodes] = start_pos_decode

        if num_reqs_actual is not None and num_reqs_actual < self.num_decodes:
            self.start_pos_decode[num_reqs_actual:].fill_(0)
            self.block_table[num_reqs_actual : self.num_decodes, ...].fill_(0)

        tp_size = get_tensor_model_parallel_world_size()
        n_local_heads = self.model_config.hf_config.num_attention_heads // tp_size
        index_topk = self.model_config.hf_config.index_topk

        assert self.decode_sas_metadata is not None

        cu_seqlens_ori_kv = DeviceOperator.get_dsa_decode_cu_seqlens_ori_kv(
            self.decode_ratio_to_sas_metadata,
            "cu_seqlens_ori_kv",
            self.seq_lens,
            self.num_decodes,
            self._zero_i32,
            self.cu_seqlens_ori_kv,
        )
        metadata_op = DeviceOperator.get_dsa_sparse_attn_metadata_op()
        metadata_kwargs = DeviceOperator.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
        cu_seqlens_cmp_kv = DeviceOperator.get_dsa_decode_cu_seqlens_cmp_kv(self.cu_seqlens_cmp_kv)
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
        common_attn_metadata: AscendCommonAttentionMetadata,
        draft_index: int,
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
            # disable use_cache, otherwise, draft_index>0 will override draft_index=0
            # take care of this, if full graph is needed then rope cache is inevitable
            cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=False)

        slot_mapping = common_attn_metadata.slot_mapping[:num_input_tokens]
        self.spec_slot_mapping[draft_index - 1][:num_input_tokens] = DeviceOperator.format_dsa_slot_mapping(  # type: ignore[index]
            slot_mapping, self.block_size
        )

        prefill_metadata = None
        if num_prefills > 0:
            prefill_metadata = self.build_prefill_metadata_for_drafting(
                draft_index=draft_index,
                common_attn_metadata=common_attn_metadata,
                reqs_start=num_decodes,
                tokens_start=num_decode_tokens,
                num_prefill_tokens=num_prefill_tokens,
            )

        decode_metadata = None
        if num_decodes > 0:
            decode_metadata = self.build_decode_metadata_for_drafting(
                draft_index=draft_index,
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
        draft_index: int,
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

        prefill_slot_mapping = self.spec_slot_mapping[draft_index - 1][tokens_start:num_prefill_tokens]  # type: ignore[index]
        block_table = common_attn_metadata.block_table_tensor[: common_attn_metadata.num_reqs]

        metadata_op = DeviceOperator.get_dsa_sparse_attn_metadata_op()
        metadata_kwargs = DeviceOperator.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)
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
            ori_mask_mode=4,
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
        draft_index: int,
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
        # disable use_cache, otherwise, draft_index>0 will override draft_index=0
        # take care of this, if full graph is needed then rope cache is inevitable
        cos, sin = get_cos_and_sin_dsa(input_positions, use_cache=False)

        slot_mapping = self.spec_slot_mapping[draft_index - 1][:num_decode_tokens_typed]  # type: ignore[index]
        block_table = common_attn_metadata.block_table_tensor

        metadata_op = DeviceOperator.get_dsa_sparse_attn_metadata_op()
        metadata_kwargs = DeviceOperator.get_dsa_sparse_attn_metadata_kwargs(self.seqused_q.device)

        decode_sas_metadata = metadata_op(
            **metadata_kwargs,
            num_heads_q=n_local_heads,
            num_heads_kv=1,
            head_dim=self.model_config.get_head_size(),
            cu_seqlens_q=query_start_loc,
            cu_seqlens_ori_kv=self.cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=self.cu_seqlens_cmp_kv,
            seqused_q=self.seqused_q,
            seqused_kv=seq_lens[:num_decodes],
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            batch_size=len(seq_lens[:num_decodes]),
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
            seq_lens=seq_lens[:num_decodes],
            seq_lens_list=None,  # type: ignore[arg-type]
            max_seq_lens=None,  # type: ignore[arg-type]
            max_seqlen_kv=None,  # type: ignore[arg-type]
            max_seqlen_q=None,  # type: ignore[arg-type]
            attn_mask=None,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=None,
            sin=sin[:num_decode_tokens, ...],
            cos=cos[:num_decode_tokens, ...],
            compress_sin=None,
            compress_cos=None,
            cp_seq_len=None,
            batch_seq_mask=None,
            start_pos=None,
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

        # CV wrapper: split wq_a/wkv/wq_b into quantize(Vector) + matmul(Cube)
        self.cv_wq_a = CVLinearWrapper(self.wq_a)
        self.cv_wkv = CVLinearWrapper(self.wkv)
        self.cv_wq_b = CVLinearWrapper(self.wq_b)

        self.indexer = kwargs.get("indexer")
        self.compressor = kwargs.get("compressor")

        self.wo_a = kwargs["wo_a"]
        self.wo_b = kwargs["wo_b"]

        self.eps = kwargs["eps"]

        self.attn_sink = kwargs["attn_sink"]

        ascend_config = get_ascend_config()
        self.multistream_dsv4_dsa_overlap = ascend_config.multistream_dsv4_dsa_overlap
        self.vllm_config = get_current_vllm_config()

        # indexer param
        if self.indexer is not None:
            self.indexer_heads: int = self.indexer.n_heads
            self.inderxer_dim: int = self.indexer.head_dim
            self.inderxer_wq_b = self.indexer.wq_b
            self.cv_inderxer_wq_b = CVLinearWrapper(self.inderxer_wq_b)
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

        # IndexCache: skip_topk indicates this layer reuses topk from a previous
        # indexer-bearing layer; use_index_cache marks whether the buffer must
        # be kept fresh on non-skip layers so downstream skip layers can read.
        self.skip_topk = kwargs.get("skip_topk", False)
        self.topk_indices_buffer = kwargs.get("topk_indices_buffer")
        self.use_index_cache = self.skip_topk or getattr(
            self.vllm_config.model_config.hf_config,
            "use_index_cache",
            False,
        )

    def _get_indexcache_topk_indices(self, num_tokens: int, offset: int = 0) -> torch.Tensor:
        if self.topk_indices_buffer is None:
            raise RuntimeError("IndexCache requires topk_indices_buffer when skip_topk is enabled.")
        topk_indices = self.topk_indices_buffer[offset : offset + num_tokens]
        if topk_indices.dim() == 2:
            topk_indices = topk_indices.unsqueeze(1)
        return topk_indices

    def _update_indexcache_topk_indices(self, topk_indices: torch.Tensor, offset: int = 0) -> None:
        if self.topk_indices_buffer is None:
            return
        num_tokens = topk_indices.shape[0]
        topk_tokens = topk_indices.shape[-1]
        topk_indices_to_cache = topk_indices
        topk_indices_buffer = self.topk_indices_buffer[offset : offset + num_tokens, :topk_tokens]
        if topk_indices_to_cache.dim() == 3 and topk_indices_buffer.dim() == 2:
            assert topk_indices_to_cache.shape[1] == 1
            topk_indices_to_cache = topk_indices_to_cache.squeeze(1)
        topk_indices_buffer.copy_(topk_indices_to_cache)

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
        kv_cache: tuple[torch.Tensor, ...] | None,
        attn_metadata: DSAMetadataList,
        need_gather_q_kv: bool = False,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:
            return output.fill_(0)
        if not isinstance(attn_metadata, list):
            attn_metadata = [attn_metadata]
        output_padded = output
        # Process for Flash Comm V1
        has_prefill = attn_metadata[0].num_prefills > 0
        has_decode = attn_metadata[0].num_decodes > 0
        decode_tokens = attn_metadata[0].num_decode_tokens
        actual_tokens = attn_metadata[0].num_actual_tokens

        # Process for Flash Comm V1
        hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(hidden_states, need_gather_q_kv)
        prefill_hidden_states = hidden_states[decode_tokens:actual_tokens]
        decode_hidden_states = hidden_states[:decode_tokens]

        forward_context = get_forward_context()
        o_proj_input_shape = (forward_context.num_tokens, self.n_local_heads, self.head_dim)
        o_proj_input = torch.empty(o_proj_input_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        assert kv_cache is not None, "kv_cache tensor tuple must be provided."
        if has_prefill:
            assert attn_metadata[0].prefill is not None
            output_prefill = self._forward_prefill(
                layer_name,
                prefill_hidden_states,
                kv_cache,
                attn_metadata,
            )  # type: ignore[arg-type]
            o_proj_input[decode_tokens:actual_tokens] = output_prefill
            cos = attn_metadata[0].prefill.cos[layer_name]
            sin = attn_metadata[0].prefill.sin[layer_name]

        if has_decode:
            assert attn_metadata[0].decode is not None
            output_decode = self._forward_decode(layer_name, decode_hidden_states, kv_cache, attn_metadata)
            o_proj_input[:decode_tokens] = output_decode
            cos = attn_metadata[0].decode.cos[layer_name]
            sin = attn_metadata[0].decode.sin[layer_name]

        cos = attn_metadata[0].cos[layer_name]
        sin = attn_metadata[0].sin[layer_name]
        num_tokens = o_proj_input.shape[0]

        torch.ops._C_ascend.inplace_partial_rotary_mul(
            o_proj_input.unsqueeze(1),
            cos,
            -sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        # o
        if get_ascend_device_type() in {AscendDeviceType.A5}:
            o = o_proj_input.view(num_tokens, self.n_local_groups, -1)
            o, swiglu_out_scale = torch_npu.npu_dynamic_mx_quant(o, dst_type=torch.float8_e4m3fn)
            o = torch_npu.npu_transpose_quant_batchmatmul(
                o,
                self.wo_a.weight,
                dtype=torch.bfloat16,
                bias=None,
                group_sizes=(0, 0, 32),
                x1_scale=swiglu_out_scale.view(torch.float8_e8m0fnu),
                x2_scale=self.wo_a.weight_scale.view(torch.float8_e8m0fnu),
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
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

        return output_padded

    def _mla_prolog_multistream(self, hidden_states, cos, sin, swa_kv_cache, slot_mapping, is_prefill=False):
        """3-block multi-stream: 3-stage CV parallel + serial tail

        Block partition (V: Vector, C: Cube, AIV: AI Vector):
          Part1: q_quant[V] -> q_a_down[C]  ||  kv_quant[V]
          Part2: q_norm[V] + q_b_quant[V]  ||  kv_matmul[C]
          Part3: q_b_matmul[C]             ||  kv_norm[V] + rope[V] + scatter[AIV]
          Tail:  q_rms[V] + rope[V] (wait for auxiliary stream to complete)

        Each stream's data is self-contained; no cross-stream sync is needed between blocks.
        Only the tail wait_stream ensures scatter is complete.
        """
        main_stream = torch.npu.current_stream()
        aux_stream = dsv4_dsa_overlap_stream()

        is_w8a8 = _is_w8a8_dynamic(self.wq_b)

        # Part1: q_quant[V] -> q_a_down[C]  ||  kv_quant[V]
        q_quant, q_pertoken_scale = self.cv_wq_a.quantize(hidden_states)

        e_q_quant_done = main_stream.record_event()

        with npu_stream_switch(aux_stream, enabled=True):
            torch.npu.current_stream().wait_event(e_q_quant_done)
            kv_quant, kv_pertoken_scale = self.cv_wkv.quantize(hidden_states)

        wq_a_result = self.cv_wq_a.matmul(q_quant, q_pertoken_scale)
        main_stream.wait_stream(aux_stream)

        # Part2: q_norm[V] + q_b_quant[V]  ||  kv_matmul[C]
        e_part2_start = main_stream.record_event()

        with npu_stream_switch(aux_stream, enabled=True):
            torch.npu.current_stream().wait_event(e_part2_start)
            kv = self.cv_wkv.matmul(kv_quant, kv_pertoken_scale)

        if is_prefill:
            qr = self.q_norm(wq_a_result)
            q_b_quant, q_b_scale = self.cv_wq_b.quantize(qr)
            qr_pertoken_scale = None
        elif is_w8a8:
            qr, qr_pertoken_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(
                wq_a_result, self.q_norm.weight, epsilon=self.eps
            )
            q_b_quant, q_b_scale = qr, qr_pertoken_scale
        else:
            qr = self.q_norm(wq_a_result)
            q_b_quant, q_b_scale = qr, None
            qr_pertoken_scale = None

        main_stream.wait_stream(aux_stream)

        # Part3: q_b_matmul[C]  ||  kv_norm[V] + rope[V] + scatter[AIV]
        e_part3_start = main_stream.record_event()

        with npu_stream_switch(aux_stream, enabled=True):
            torch.npu.current_stream().wait_event(e_part3_start)
            kv = self.kv_norm(kv)
            assert self.rope_head_dim is not None
            kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)
            torch.ops._C_ascend.inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[self.nope_head_dim, self.head_dim],
            )
            DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, kv, slot_mapping)

        if is_prefill:
            q = self.cv_wq_b.matmul(q_b_quant, q_b_scale).unflatten(-1, (self.n_local_heads, self.head_dim))
        elif is_w8a8:
            q = torch_npu.npu_quant_matmul(
                q_b_quant,
                self.wq_b.weight,
                self.wq_b.weight_scale,
                pertoken_scale=q_b_scale,
                bias=self.wq_b.bias,
                output_dtype=hidden_states.dtype,
            ).unflatten(-1, (self.n_local_heads, self.head_dim))
        else:
            q = self.cv_wq_b.matmul(q_b_quant, q_b_scale).unflatten(-1, (self.n_local_heads, self.head_dim))

        # Serial tail: wait for auxiliary stream then execute q_rms[V] + rope[V]
        main_stream.wait_stream(aux_stream)

        q = DeviceOperator.apply_dsa_q_rms(q, self.eps, self.q_norm_without_weight)
        torch.ops._C_ascend.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.nope_head_dim, self.head_dim],
        )

        return q, qr, qr_pertoken_scale

    def _forward_prefill(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: DSAMetadataList,
    ):
        compress_common_attn_metadata = None
        (compress_kv_cache, swa_kv_cache, state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache) = (
            DeviceOperator.unpack_dsa_forward_kv_cache(kv_cache, self.compress_ratio)
        )

        if self.compress_ratio == 4:
            # sorted keys: [attn, compressor.state_cache, indexer.compressor.state_cache, indexer.k_cache, swa_cache]
            (compressor_attn_metadata, compressor_kv_state_metadata, _, indexer_kv_scale_metadata, swa_metadata) = (
                attn_metadata
            )
            compress_common_attn_metadata = compressor_attn_metadata
        elif self.compress_ratio == 128:
            # sorted keys: [attn, compressor.state_cache, swa_cache]
            (compressor_attn_metadata, compressor_kv_state_metadata, swa_metadata) = attn_metadata
            compress_common_attn_metadata = compressor_attn_metadata
        else:
            # sorted keys: [swa_cache]
            (swa_metadata,) = attn_metadata
            compress_common_attn_metadata = swa_metadata

        common_prefill_metadata = _require_prefill_metadata(compress_common_attn_metadata)
        swa_prefill_metadata = _require_prefill_metadata(swa_metadata)
        cos = common_prefill_metadata.cos[layer_name]
        sin = common_prefill_metadata.sin[layer_name]
        actual_seq_lengths_query = common_prefill_metadata.query_start_loc
        actual_seq_lengths_key = common_prefill_metadata.seq_lens

        if self.multistream_dsv4_dsa_overlap:
            # mla prolog: q + kv dual-stream parallel
            q, qr, _ = self._mla_prolog_multistream(
                hidden_states, cos, sin, swa_kv_cache, swa_prefill_metadata.slot_mapping, is_prefill=True
            )
        else:
            # mlaprolog
            share_hs_quant = _is_w8a8_dynamic(self.wq_a) and _is_w8a8_dynamic(self.wkv)
            if share_hs_quant:
                hs_int8, hs_pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
                q_a = torch_npu.npu_quant_matmul(
                    hs_int8,
                    self.wq_a.weight,
                    self.wq_a.weight_scale,
                    pertoken_scale=hs_pertoken_scale,
                    bias=self.wq_a.bias,
                    output_dtype=hidden_states.dtype,
                )
            else:
                q_a = self.wq_a(hidden_states)

            # q
            if _is_w8a8_dynamic(self.wq_b):
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
                qr = self.q_norm(q_a)
                q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
                qr_pertoken_scale = None
            q = DeviceOperator.apply_dsa_q_rms(q, self.eps, self.q_norm_without_weight)

            torch.ops._C_ascend.inplace_partial_rotary_mul(
                q.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[self.nope_head_dim, self.head_dim],
            )
            # win kv & tok_dis
            if share_hs_quant:
                kv = torch_npu.npu_quant_matmul(
                    hs_int8,
                    self.wkv.weight,
                    self.wkv.weight_scale,
                    pertoken_scale=hs_pertoken_scale,
                    bias=self.wkv.bias,
                    output_dtype=hidden_states.dtype,
                )
            else:
                kv = self.wkv(hidden_states)
            kv = self.kv_norm(kv)
            assert self.rope_head_dim is not None
            kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)

            torch.ops._C_ascend.inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[self.nope_head_dim, self.head_dim],
            )

            # swa exec kv
            DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, kv, swa_prefill_metadata.slot_mapping)

        compress_cos = common_prefill_metadata.compress_cos[layer_name]
        compress_sin = common_prefill_metadata.compress_sin[layer_name]

        attn_op = DeviceOperator.get_dsa_sparse_attn_op()
        extra_attn_kwargs: dict = DeviceOperator.get_dsa_sparse_attn_base_kwargs()
        DeviceOperator.add_dsa_sparse_attn_extra_kwargs(extra_attn_kwargs, cu_seqlens_ori_kv=actual_seq_lengths_query)

        if self.compress_ratio <= 1:
            return attn_op(
                q,
                ori_kv=swa_kv_cache,
                ori_block_table=swa_prefill_metadata.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=common_prefill_metadata.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=max(self.compress_ratio, 1),
                ori_mask_mode=4,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                **extra_attn_kwargs,
            )[0]

        if self.compress_ratio > 1:
            compressor_prefill_metadata = _require_prefill_metadata(compressor_attn_metadata)
            compressor_state_prefill_metadata = _require_prefill_metadata(compressor_kv_state_metadata)
            compress_topk_idxs = None
            # Only call indexer_select_qli when compress_ratio == 4 (requires 5 elements in attn_metadata)
            if self.compress_ratio == 4:
                # IndexCache: prefill segment lives at buffer[num_decode_tokens:]
                # because dsa_v1 forward splits hidden_states as
                # [decode | prefill]. See AscendDSAImpl.forward.
                prefill_offset = attn_metadata[0].num_decode_tokens
                prefill_num_tokens = hidden_states.shape[0]
                if self.skip_topk:
                    compress_topk_idxs = self._get_indexcache_topk_indices(prefill_num_tokens, offset=prefill_offset)
                else:
                    if self.multistream_dsv4_dsa_overlap:
                        indexer_q = self.cv_indexer_select_qli(  # multistream version
                            x=hidden_states,
                            qr=qr,
                            kv_cache=kv_cache,
                            attn_metadata=attn_metadata,
                            cos=cos,
                            sin=sin,
                            compressed_cos=compress_cos,
                            compressed_sin=compress_sin,
                            actual_seq_lengths_query=actual_seq_lengths_query,
                            with_prefill=True,
                        )
                    else:
                        compress_topk_idxs = self.indexer_select_qli(  # original version
                            x=hidden_states,
                            qr=qr,
                            kv_cache=kv_cache,
                            attn_metadata=attn_metadata,
                            cos=cos,
                            sin=sin,
                            compressed_cos=compress_cos,
                            compressed_sin=compress_sin,
                            actual_seq_lengths_query=actual_seq_lengths_query,
                            actual_seq_lengths_key=actual_seq_lengths_key,
                            with_prefill=True,
                            qr_pertoken_scale=qr_pertoken_scale,
                        )

            coff = 2 if self.compressor_overlap else 1

            # Inline compressor + scatter (c128, c4 non-dual)
            compressed_kv = torch.ops._C_ascend.compressor(
                hidden_states,
                self.compressor_wkv.weight,
                self.compressor_wgate.weight,
                state_cache.squeeze(-2),
                self.compressor_ape,
                self.compressor_norm.weight,
                compress_sin.view(-1, compress_sin.shape[-1]),
                compress_cos.view(-1, compress_cos.shape[-1]),
                state_block_table=compressor_state_prefill_metadata.block_table,
                cu_seqlens=actual_seq_lengths_query,
                seqused=None,
                start_pos=common_prefill_metadata.start_pos,
                rope_head_dim=self.rope_head_dim,
                cmp_ratio=self.compress_ratio,
                coff=coff,
                norm_eps=self.compressor_norm_eps,
                rotary_mode=2,
                cache_mode=1,
            )

            if compressed_kv.numel() == 0:
                compressed_kv = None

            # For multistream_dsv4_dsa_overlap with compress_ratio=4:
            # aux_stream: indexer_weights_proj (parallel with main q_quant + kv_scatter)
            # main stream: compressed_kv -> q_quant -> kv_scatter -> wait aux_stream -> lightning_indexer
            if self.multistream_dsv4_dsa_overlap and self.compress_ratio == 4 and not self.skip_topk:
                main_stream = torch.npu.current_stream()
                aux_stream = dsv4_dsa_overlap_stream()
                e_compressed_kv_done = main_stream.record_event()
                with npu_stream_switch(aux_stream, enabled=True):
                    torch.npu.current_stream().wait_event(e_compressed_kv_done)
                    weights_proj_output = self.weights_proj(hidden_states)
                # Main stream: q_quant (between compressed_kv and kv_scatter)
                q_quant, q_scale = DeviceOperator.indexer_quantize_query(indexer_q)

            DeviceOperator.dsa_kv_compress_scatter(
                compress_kv_cache, compressed_kv, compressor_prefill_metadata.slot_mapping
            )

            if self.multistream_dsv4_dsa_overlap and self.compress_ratio == 4 and not self.skip_topk:
                # Wait aux_stream weights_proj done, then compute dot
                main_stream.wait_stream(aux_stream)
                weights = weights_proj_output * (self.indexer_softmax_scale * self.indexer_heads**-0.5)
                # lightning_indexer
                indexer_scale_prefill_metadata = _require_prefill_metadata(indexer_kv_scale_metadata)
                qlens = indexer_scale_prefill_metadata.query_start_loc[1:]
                kvlens = indexer_scale_prefill_metadata.seq_lens
                block_table = indexer_scale_prefill_metadata.block_table
                qli_metadata = indexer_scale_prefill_metadata.qli_metadata
                compress_topk_idxs, _ = torch.ops._C_ascend.npu_quant_lightning_indexer(
                    query=q_quant,
                    key=indexer_k_cache,
                    weights=DeviceOperator.prepare_dsa_indexer_weights(weights),
                    query_dequant_scale=DeviceOperator.prepare_dsa_indexer_query_scale(q_scale),
                    key_dequant_scale=DeviceOperator.prepare_dsa_indexer_key_scale(indexer_scale_cache),
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

            if self.compress_ratio == 4 and self.use_index_cache:
                self._update_indexcache_topk_indices(compress_topk_idxs, offset=prefill_offset)

            if self.compress_ratio == 4:
                DeviceOperator.add_dsa_sparse_attn_extra_kwargs(
                    extra_attn_kwargs, cu_seqlens_cmp_kv=common_prefill_metadata.cu_c4_cmp_seqlen_list
                )
                attn_output = attn_op(
                    q,
                    ori_kv=swa_kv_cache,
                    cmp_kv=compress_kv_cache,
                    cmp_sparse_indices=compress_topk_idxs,
                    ori_block_table=swa_prefill_metadata.block_table,
                    cmp_block_table=compressor_prefill_metadata.block_table,
                    cu_seqlens_q=actual_seq_lengths_query,
                    seqused_kv=actual_seq_lengths_key,
                    sinks=self.attn_sink,
                    metadata=common_prefill_metadata.sas_metadata,
                    softmax_scale=self.softmax_scale,
                    cmp_ratio=self.compress_ratio,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self.window_size - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    **extra_attn_kwargs,
                )[0]
            else:
                DeviceOperator.add_dsa_sparse_attn_extra_kwargs(
                    extra_attn_kwargs, cu_seqlens_cmp_kv=common_prefill_metadata.cu_c128_cmp_seqlen_list
                )
                attn_output = attn_op(
                    q,
                    ori_kv=swa_kv_cache,
                    cmp_kv=compress_kv_cache,
                    ori_block_table=swa_prefill_metadata.block_table,
                    cmp_block_table=compressor_prefill_metadata.block_table,
                    cu_seqlens_q=actual_seq_lengths_query,
                    seqused_kv=actual_seq_lengths_key,
                    sinks=self.attn_sink,
                    metadata=common_prefill_metadata.sas_metadata,
                    softmax_scale=self.softmax_scale,
                    cmp_ratio=self.compress_ratio,
                    ori_mask_mode=4,
                    cmp_mask_mode=3,
                    ori_win_left=self.window_size - 1,
                    ori_win_right=0,
                    layout_q="TND",
                    layout_kv="PA_ND",
                    **extra_attn_kwargs,
                )[0]
        return attn_output

    def _forward_decode(
        self,
        layer_name,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: DSAMetadataList,
    ):
        assert attn_metadata[0].decode is not None
        compress_common_attn_metadata = None

        (compress_kv_cache, swa_kv_cache, state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache) = (
            DeviceOperator.unpack_dsa_forward_kv_cache(kv_cache, self.compress_ratio)
        )

        if self.compress_ratio == 4:
            # sorted keys: [attn, compressor.state_cache, indexer.compressor.state_cache, indexer.k_cache, swa_cache]
            (compressor_attn_metadata, compressor_kv_state_metadata, _, indexer_kv_scale_metadata, swa_metadata) = (
                attn_metadata
            )
            compress_common_attn_metadata = compressor_attn_metadata
        elif self.compress_ratio == 128:
            # sorted keys: [attn, compressor.state_cache, swa_cache]
            (compressor_attn_metadata, compressor_kv_state_metadata, swa_metadata) = attn_metadata
            compress_common_attn_metadata = compressor_attn_metadata
        else:
            # sorted keys: [swa_cache]
            (swa_metadata,) = attn_metadata
            compress_common_attn_metadata = swa_metadata
        common_decode_metadata = _require_decode_metadata(compress_common_attn_metadata)
        swa_decode_metadata = _require_decode_metadata(swa_metadata)
        cos = common_decode_metadata.cos[layer_name]
        sin = common_decode_metadata.sin[layer_name]
        actual_seq_lengths_query = common_decode_metadata.query_start_loc
        actual_seq_lengths_key = common_decode_metadata.seq_lens

        if self.multistream_dsv4_dsa_overlap:
            # mla prolog: q + kv dual-stream parallel
            q, qr, qr_pertoken_scale = self._mla_prolog_multistream(
                hidden_states, cos, sin, swa_kv_cache, swa_decode_metadata.slot_mapping, is_prefill=False
            )
        else:
            # Share one dynamic-quant of hidden_states between wq_a (main stream)
            # and wkv (attention stream) when both sides are W8A8 dynamic.
            share_hs_quant = _is_w8a8_dynamic(self.wq_a) and _is_w8a8_dynamic(self.wkv)
            if share_hs_quant:
                hs_int8, hs_pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)

            # q
            if _is_w8a8_dynamic(self.wq_b):
                if share_hs_quant:
                    q_a = torch_npu.npu_quant_matmul(
                        hs_int8,
                        self.wq_a.weight,
                        self.wq_a.weight_scale,
                        pertoken_scale=hs_pertoken_scale,
                        bias=self.wq_a.bias,
                        output_dtype=hidden_states.dtype,
                    )
                else:
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
                if share_hs_quant:
                    q_a = torch_npu.npu_quant_matmul(
                        hs_int8,
                        self.wq_a.weight,
                        self.wq_a.weight_scale,
                        pertoken_scale=hs_pertoken_scale,
                        bias=self.wq_a.bias,
                        output_dtype=hidden_states.dtype,
                    )
                    qr = q = self.q_norm(q_a)
                else:
                    qr = q = self.q_norm(self.wq_a(hidden_states))
                q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
                qr_pertoken_scale = None

            q = DeviceOperator.apply_dsa_q_rms(q, self.eps, self.q_norm_without_weight)

            torch.ops._C_ascend.inplace_partial_rotary_mul(
                q.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[self.nope_head_dim, self.head_dim],
            )

            # win kv & tok_dis
            if share_hs_quant:
                kv = torch_npu.npu_quant_matmul(
                    hs_int8,
                    self.wkv.weight,
                    self.wkv.weight_scale,
                    pertoken_scale=hs_pertoken_scale,
                    bias=self.wkv.bias,
                    output_dtype=hidden_states.dtype,
                )
            else:
                kv = self.wkv(hidden_states)
            kv = self.kv_norm(kv)
            assert self.rope_head_dim is not None
            kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)

            torch.ops._C_ascend.inplace_partial_rotary_mul(
                kv.unsqueeze(1),
                cos,
                sin,
                rotary_mode="interleave",
                partial_slice=[self.nope_head_dim, self.head_dim],
            )

            # swa exec kv
            DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, kv, swa_decode_metadata.slot_mapping)

        if self.compress_ratio > 1:
            compressor_decode_metadata = _require_decode_metadata(compressor_attn_metadata)
            compressor_state_decode_metadata = _require_decode_metadata(compressor_kv_state_metadata)
            compress_cos = common_decode_metadata.compress_cos[layer_name]
            compress_sin = common_decode_metadata.compress_sin[layer_name]
            compress_topk_idxs = None
            if self.compress_ratio == 4:
                # IndexCache: decode segment occupies buffer[:num_decode_tokens]
                decode_num_tokens = hidden_states.shape[0]
                if self.skip_topk:
                    compress_topk_idxs = self._get_indexcache_topk_indices(decode_num_tokens, offset=0)
                else:
                    if self.multistream_dsv4_dsa_overlap:
                        indexer_q = self.cv_indexer_select_qli(  # multistream version
                            x=hidden_states,
                            qr=qr,
                            kv_cache=kv_cache,
                            attn_metadata=attn_metadata,
                            cos=cos,
                            sin=sin,
                            compressed_cos=compress_cos,
                            compressed_sin=compress_sin,
                            actual_seq_lengths_query=actual_seq_lengths_query,
                            with_prefill=False,
                            qr_pertoken_scale=qr_pertoken_scale,
                        )
                    else:
                        compress_topk_idxs = self.indexer_select_qli(  # original version
                            x=hidden_states,
                            qr=qr,
                            kv_cache=kv_cache,
                            attn_metadata=attn_metadata,
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

            # Inline compressor + scatter (c128, c4 non-dual)
            compressed_kv = torch.ops._C_ascend.compressor(
                hidden_states,
                self.compressor_wkv.weight,
                self.compressor_wgate.weight,
                state_cache.squeeze(-2),
                self.compressor_ape,
                self.compressor_norm.weight,
                compress_sin.view(-1, compress_sin.shape[-1]),
                compress_cos.view(-1, compress_cos.shape[-1]),
                state_block_table=compressor_state_decode_metadata.block_table,
                cu_seqlens=actual_seq_lengths_query,
                seqused=None,
                start_pos=common_decode_metadata.start_pos,
                rope_head_dim=self.rope_head_dim,
                cmp_ratio=self.compress_ratio,
                coff=coff,
                norm_eps=self.compressor_norm_eps,
                rotary_mode=2,
                cache_mode=1,
            )

            # For multistream_dsv4_dsa_overlap with compress_ratio=4:
            # aux_stream: indexer_weights_proj (parallel with main q_quant + kv_scatter)
            # main stream: compressed_kv -> q_quant -> kv_scatter -> wait aux_stream -> lightning_indexer
            if self.multistream_dsv4_dsa_overlap and self.compress_ratio == 4 and not self.skip_topk:
                main_stream = torch.npu.current_stream()
                aux_stream = dsv4_dsa_overlap_stream()
                e_compressed_kv_done = main_stream.record_event()
                with npu_stream_switch(aux_stream, enabled=True):
                    torch.npu.current_stream().wait_event(e_compressed_kv_done)
                    weights_proj_output = self.weights_proj(hidden_states)
                # Main stream: q_quant (between compressed_kv and kv_scatter)
                q_quant, q_scale = DeviceOperator.indexer_quantize_query(indexer_q)

            DeviceOperator.dsa_kv_compress_scatter(
                compress_kv_cache, compressed_kv, compressor_decode_metadata.slot_mapping
            )

            if self.multistream_dsv4_dsa_overlap and self.compress_ratio == 4 and not self.skip_topk:
                # Wait aux_stream weights_proj done
                main_stream.wait_stream(aux_stream)
                weights = weights_proj_output * (self.indexer_softmax_scale * self.indexer_heads**-0.5)
                # lightning_indexer
                indexer_scale_decode_metadata = _require_decode_metadata(indexer_kv_scale_metadata)
                qlens = indexer_scale_decode_metadata.query_start_loc[1:]
                kvlens = indexer_scale_decode_metadata.seq_lens
                block_table = indexer_scale_decode_metadata.block_table
                qli_metadata = indexer_scale_decode_metadata.qli_metadata
                compress_topk_idxs, _ = torch.ops._C_ascend.npu_quant_lightning_indexer(
                    query=q_quant,
                    key=indexer_k_cache,
                    weights=DeviceOperator.prepare_dsa_indexer_weights(weights),
                    query_dequant_scale=DeviceOperator.prepare_dsa_indexer_query_scale(q_scale),
                    key_dequant_scale=DeviceOperator.prepare_dsa_indexer_key_scale(indexer_scale_cache),
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

            if self.compress_ratio == 4 and self.use_index_cache:
                self._update_indexcache_topk_indices(compress_topk_idxs, offset=0)

        attn_op = DeviceOperator.get_dsa_sparse_attn_op()
        extra_attn_kwargs: dict = DeviceOperator.get_dsa_sparse_attn_base_kwargs()

        if self.compress_ratio <= 1:
            attn_output = attn_op(
                q,
                ori_kv=swa_kv_cache,
                ori_block_table=swa_decode_metadata.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=swa_decode_metadata.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=max(self.compress_ratio, 1),
                ori_mask_mode=4,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                **extra_attn_kwargs,
            )[0]
        elif self.compress_ratio == 4:
            attn_output = attn_op(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                cmp_sparse_indices=compress_topk_idxs,
                ori_block_table=swa_decode_metadata.block_table,
                cmp_block_table=compressor_decode_metadata.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=compressor_decode_metadata.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                **extra_attn_kwargs,
            )[0]
        else:
            attn_output = attn_op(
                q,
                ori_kv=swa_kv_cache,
                cmp_kv=compress_kv_cache,
                ori_block_table=swa_decode_metadata.block_table,
                cmp_block_table=compressor_decode_metadata.block_table,
                cu_seqlens_q=actual_seq_lengths_query,
                seqused_kv=actual_seq_lengths_key,
                sinks=self.attn_sink,
                metadata=compressor_decode_metadata.sas_metadata,
                softmax_scale=self.softmax_scale,
                cmp_ratio=self.compress_ratio,
                ori_mask_mode=4,
                cmp_mask_mode=3,
                ori_win_left=self.window_size - 1,
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                **extra_attn_kwargs,
            )[0]
        return attn_output

    def _indexer_qkv_prepare(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: DSAMetadataList,
        cos: torch.Tensor,
        sin: torch.Tensor,
        compressed_cos: torch.Tensor,
        compressed_sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        with_prefill: bool = False,
        qr_pertoken_scale: torch.Tensor = None,
    ):
        (indexer_state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache) = (
            DeviceOperator.unpack_dsa_indexer_kv_cache(kv_cache)
        )
        (
            _,
            _,
            indexer_kv_state_metadata,
            indexer_kv_scale_metadata,
            _,
        ) = attn_metadata

        if (
            _is_w8a8_dynamic(self.inderxer_wq_b)
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
            indexer_state_prefill_metadata = _require_prefill_metadata(indexer_kv_state_metadata)
            indexer_scale_prefill_metadata = _require_prefill_metadata(indexer_kv_scale_metadata)
            kv_block_table = indexer_state_prefill_metadata.block_table
            start_pos = indexer_scale_prefill_metadata.start_pos
        else:
            indexer_state_decode_metadata = _require_decode_metadata(indexer_kv_state_metadata)
            indexer_scale_decode_metadata = _require_decode_metadata(indexer_kv_scale_metadata)
            kv_block_table = indexer_state_decode_metadata.block_table
            start_pos = indexer_scale_decode_metadata.start_pos

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
        elif self.indexcom_rotate:
            kv = rotate_activation(kv, indexer_kv_scale_metadata.hadamard)

        return (
            q,
            kv,
            indexer_k_cache,
            indexer_scale_cache,
            indexer_full_cache,
            indexer_kv_state_metadata,
            indexer_kv_scale_metadata,
            with_prefill,
        )

    def _indexer_qli_finish(
        self,
        q: torch.Tensor,
        kv: torch.Tensor | None,
        weights: torch.Tensor,
        indexer_k_cache: torch.Tensor,
        indexer_scale_cache: torch.Tensor,
        indexer_full_cache: torch.Tensor | None,
        indexer_kv_state_metadata,
        indexer_kv_scale_metadata,
        with_prefill: bool,
    ):
        q, q_scale, kv, kv_scale = self._indexer_quant_scatter(
            q,
            kv,
            indexer_k_cache,
            indexer_scale_cache,
            indexer_full_cache,
            indexer_kv_scale_metadata,
            with_prefill,
        )
        return self._indexer_qli(
            q,
            weights,
            q_scale,
            indexer_k_cache,
            indexer_scale_cache,
            indexer_kv_scale_metadata,
            with_prefill,
        )

    def _indexer_quant_scatter(
        self,
        q: torch.Tensor,
        kv: torch.Tensor | None,
        indexer_k_cache: torch.Tensor,
        indexer_scale_cache: torch.Tensor,
        indexer_full_cache: torch.Tensor | None,
        indexer_kv_scale_metadata,
        with_prefill: bool,
    ):
        slot_mapping = (
            indexer_kv_scale_metadata.prefill.slot_mapping
            if with_prefill
            else indexer_kv_scale_metadata.decode.slot_mapping
        )
        return DeviceOperator.indexer_quant_scatter(
            q, kv, indexer_k_cache, indexer_scale_cache, indexer_full_cache, slot_mapping
        )

    def _indexer_qli(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        q_scale: torch.Tensor,
        indexer_k_cache: torch.Tensor,
        indexer_scale_cache: torch.Tensor,
        indexer_kv_scale_metadata,
        with_prefill: bool,
    ):
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
            weights=DeviceOperator.prepare_dsa_indexer_weights(weights),
            query_dequant_scale=DeviceOperator.prepare_dsa_indexer_query_scale(q_scale),
            key_dequant_scale=DeviceOperator.prepare_dsa_indexer_key_scale(indexer_scale_cache),
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

    def indexer_select_qli(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: DSAMetadataList,
        cos: torch.Tensor,
        sin: torch.Tensor,
        compressed_cos: torch.Tensor,
        compressed_sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor | None = None,
        with_prefill: bool = False,
        qr_pertoken_scale: torch.Tensor = None,
    ):
        q, kv, ik, isc, ifc, indexer_kv_state_meta, isc_meta, wp = self._indexer_qkv_prepare(
            x,
            qr,
            kv_cache,
            attn_metadata,
            cos,
            sin,
            compressed_cos,
            compressed_sin,
            actual_seq_lengths_query,
            with_prefill,
            qr_pertoken_scale,
        )

        weights = self.weights_proj(x) * (self.indexer_softmax_scale * self.indexer_heads**-0.5)

        return self._indexer_qli_finish(q, kv, weights, ik, isc, ifc, indexer_kv_state_meta, isc_meta, wp)

    def cv_indexer_select_qli(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: DSAMetadataList,
        cos: torch.Tensor,
        sin: torch.Tensor,
        compressed_cos: torch.Tensor,
        compressed_sin: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        with_prefill: bool = False,
        qr_pertoken_scale: torch.Tensor = None,
    ):
        """
        Multistream version: 4-block segmentation, main stream and aux stream
        alternate submission to achieve V/C engine parallel

        Core strategy:
        - Part0: Main pre-compute qr_quant[V] + compressor[C/mixed] + kv_hadamard[V]
        - Part1: Main matmul[C] ∥ Aux kv_quant[V] + scatter_k_cache[AIV]
        - Part2: Main rope[V] (serial)
        - Part3: Main q_hadamard[C] ∥ Aux scatter_scale_cache[AIV]
        - Part4: Caller runs weights_proj + q_quant + indexer
        """
        (indexer_state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache) = (
            DeviceOperator.unpack_dsa_indexer_kv_cache(kv_cache)
        )
        # sorted keys: [attn, compressor.state_cache, indexer.compressor.state_cache, indexer.k_cache, swa_cache]
        (_, _, indexer_kv_state_metadata, indexer_kv_scale_metadata, _) = attn_metadata

        main_stream = torch.npu.current_stream()
        aux_stream = dsv4_dsa_overlap_stream()

        # ===== Part0: Pre-compute on main =====
        if _is_w8a8_dynamic(self.inderxer_wq_b) and qr_pertoken_scale is not None:
            qr_quant_ready = qr
            qr_scale_ready = qr_pertoken_scale
        else:
            qr_quant_ready, qr_scale_ready = self.cv_inderxer_wq_b.quantize(qr)

        coff = 2 if self.compressor_overlap else 1

        if with_prefill:
            indexer_state_prefill_metadata = _require_prefill_metadata(indexer_kv_state_metadata)
            indexer_scale_prefill_metadata = _require_prefill_metadata(indexer_kv_scale_metadata)
            kv_block_table = indexer_state_prefill_metadata.block_table
            start_pos = indexer_scale_prefill_metadata.start_pos
        else:
            indexer_state_decode_metadata = _require_decode_metadata(indexer_kv_state_metadata)
            indexer_scale_decode_metadata = _require_decode_metadata(indexer_kv_scale_metadata)
            kv_block_table = indexer_state_decode_metadata.block_table
            start_pos = indexer_scale_decode_metadata.start_pos

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
        elif self.indexcom_rotate:
            kv = rotate_activation(kv, indexer_kv_scale_metadata.hadamard)

        # ===== Part1: matmul[C] ∥ kv_quant[V] + scatter_k_cache[AIV] =====
        # Record event before main stream operations for aux_stream to wait
        e_kv_ready = main_stream.record_event()

        # Aux: kv_quant + scatter_k_cache (parallel with main matmul + rope)
        if kv is not None:
            slot_mapping_indexer = (
                indexer_scale_prefill_metadata.slot_mapping
                if with_prefill
                else indexer_scale_decode_metadata.slot_mapping
            )
            with npu_stream_switch(aux_stream, enabled=True):
                torch.npu.current_stream().wait_event(e_kv_ready)
                kv, kv_scale = DeviceOperator.indexer_quant_scatter_part1(
                    kv, indexer_k_cache, indexer_full_cache, slot_mapping_indexer
                )

        # Main: matmul q from qr (directly submit, V/C different engines dispatch naturally)
        if _is_w8a8_dynamic(self.inderxer_wq_b) and qr_pertoken_scale is not None:
            q = torch_npu.npu_quant_matmul(
                qr_quant_ready,
                self.inderxer_wq_b.weight,
                self.inderxer_wq_b.weight_scale,
                pertoken_scale=qr_scale_ready,
                bias=self.inderxer_wq_b.bias,
                output_dtype=x.dtype,
            )
        else:
            q = self.cv_inderxer_wq_b.matmul(qr_quant_ready, qr_scale_ready)  # qr_matmul

        if kv is not None:
            main_stream.wait_stream(aux_stream)

        q = q.view(-1, self.indexer_heads, self.indexcom_head_dim)

        # ===== Part2: rope[V] (main only) =====
        torch.ops._C_ascend.inplace_partial_rotary_mul(  # rope
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[self.indexcom_head_dim - self.rope_head_dim, self.indexcom_head_dim],
        )

        # Wait for aux_stream kv_scatter to complete before proceeding
        if kv is not None:
            main_stream.wait_stream(aux_stream)

        e_rope_done = main_stream.record_event()

        # ===== Part3: q_hadamard[C] ∥ scatter_scale_cache[AIV] =====
        # Note: On A5, indexer_compress_epilog_v2 in Part1 handles both k_cache
        # and scale_cache in one fused operation, so Part3 is skipped
        # (kv_scale is None on A5 from indexer_quant_scatter_part1).
        if kv is not None and kv_scale is not None:
            slot_mapping_indexer_part3 = (
                indexer_scale_prefill_metadata.slot_mapping
                if with_prefill
                else indexer_scale_decode_metadata.slot_mapping
            )
            with npu_stream_switch(aux_stream, enabled=True):
                torch.npu.current_stream().wait_event(e_rope_done)
                DeviceOperator.dsa_indexer_scatter_scale_part3(
                    kv_scale, indexer_scale_cache, slot_mapping_indexer_part3
                )

        # Main: q_hadamard[Part1 - linear] (directly submit, C/AIV different engines dispatch naturally)
        # Part1: F.linear - parallel with aux_stream kv_scatter
        hidden_size = q.size(-1)
        q_linear, q_shape, q_dim = hadamard_linear(q, indexer_kv_scale_metadata.hadamard)

        if kv is not None:
            main_stream.wait_stream(aux_stream)

        # Main: q_hadamard[Part2 - scale] (after aux_stream completes)
        # Part2: scale * reshape - dot multiplication
        q = hadamard_scale(q_linear, q_shape, q_dim, scale=hidden_size**-0.5)

        return q
