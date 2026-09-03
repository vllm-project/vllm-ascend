#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch_npu
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (  # type: ignore
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.attention.backends.registry import (  # type: ignore
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, CrossAttentionSpec

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    PagedAttentionGraphParam,
    cache_graph_workspace,
    enable_dcp,
    enable_pcp,
    needs_layer_aware_fia_graph_replay,
    notify_kv_cache_written,
    split_decodes_and_prefills,
    update_paged_attention_graph_param,
    using_paged_attention,
)
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_draft_graph_prefill_params,
    get_graph_params,
    update_draft_graph_params_workspaces,
    update_graph_params_workspaces,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.device.mxfp_kv_cache import scatter_mxfp_k_scale_cache
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import record_attention_compute_start
from vllm_ascend.utils import vllm_version_is, weak_ref_tensors

if vllm_version_is("0.27.1"):
    from vllm.model_executor.layers.attention.pcp import _gather_prefill_cache_inputs  # type: ignore[import-not-found]
else:
    from vllm.v1.attention.ops.pcp import _gather_prefill_cache_inputs  # type: ignore[import-not-found]

# default max value of sliding window size
SWA_INT_MAX = 2147483647
_ATTN_KEYS_BUFFER = None


@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["AscendAttentionBackendImpl"]:
        pcp_enabled = enable_pcp()
        dcp_enabled = enable_dcp()
        if pcp_enabled and dcp_enabled:
            raise NotImplementedError("Ascend MRV2 GQA does not support PCP and DCP simultaneously yet.")
        if pcp_enabled:
            return AscendAttentionPCPImpl
        if dcp_enabled:
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionDCPImpl

            return AscendAttentionDCPImpl
        return AscendAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        pcp_enabled = enable_pcp()
        dcp_enabled = enable_dcp()
        if pcp_enabled and dcp_enabled:
            raise NotImplementedError("Ascend MRV2 GQA does not support PCP and DCP simultaneously yet.")
        if pcp_enabled:
            return AscendAttentionPCPMetadataBuilder
        if dcp_enabled:
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionDCPMetadataBuilder

            return AscendAttentionDCPMetadataBuilder
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: list[torch.Tensor],
        dst_kv_cache: list[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        # C8-MXFP layers carry (k, v, k_scale, v_scale) tuples; generalize to
        # any cache tuple length so scale caches are swapped alongside K/V.
        for src_cache, dst_cache in zip(src_kv_cache, dst_kv_cache):
            dst_cache[dst_indices] = src_cache[src_indices].to(dst_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            for cache in kv_cache:
                cache[dst_indices] = cache[src_indices]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


class AscendC8MXFPAttentionBackend(AscendAttentionBackend):
    """Backend for C8-MXFP KV cache layers (QFA dual-operator interface).

    C8-MXFP QFA requires 512-token pages (the D=256 requirement doc allows
    512/1024). This must not be advertised by the generic backend because
    hybrid BF16 models (e.g. Qwen3.5/3.6 linear-attention mixes) use its
    128-token logical block layout when reshaping their KV cache.
    """

    @staticmethod
    def get_impl_cls() -> type["AscendC8MXFPAttentionBackendImpl"]:
        if enable_pcp() or enable_dcp():
            raise NotImplementedError("C8_MXFP attention does not support PCP/DCP yet.")
        return AscendC8MXFPAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["AscendC8MXFPMetadataBuilder"]:
        if enable_pcp() or enable_dcp():
            raise NotImplementedError("C8_MXFP attention does not support PCP/DCP yet.")
        return AscendC8MXFPMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [512]


class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@dataclass
class AscendMetadata:
    """
    Per-layer attention metadata for Ascend FlashAttention backend.

    Contains attention masks, token counts, sequence lengths and KV cache
    related properties for attention computation.
    """

    # **************************** Basic Properties ************************** #
    attn_mask: torch.Tensor | None = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # Number of tokens excluding padding.
    num_actual_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0

    # The sequence length per sequence. Sequence length means the computed
    # tokens + new tokens (is None if it is a decoding).
    # (batch_size,)
    # TODO(Angazenn): The following parameters are quite redundant and
    # contains similar information (such as seq_lens seq_lens_list). We
    # should simplified these parameters once attention schema in vLLM-Ascend
    # is unified.
    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    seq_lens_list: list[int] = None  # type: ignore
    actual_seq_lengths_q: list[int] = None  # type: ignore

    query_start_loc: torch.Tensor = None
    # Maximum query length in the batch (None for decoding).
    max_query_len: int | None = None

    # Persistent GPU-side length sources, refreshed in place by the model
    # runner every step: query_start_loc_gpu is the 0-prefixed cumulative
    # query boundaries (int32) and seq_lens_gpu the per-request KV lengths
    # (int32). Graph-capturing backends must source cu_seqlens/seqused from
    # these instead of the CPU lists above -- ACL-graph replay re-executes
    # only device work, so Python-side refreshes of hand-managed buffers
    # never run again after capture.
    query_start_loc_gpu: torch.Tensor = None
    seq_lens_gpu: torch.Tensor = None

    # ********************** KV Cache Related Properties ********************* #
    # Block addresses per sequence (Seq id -> list of physical block).
    # (batch_size, max_blocks_per_seq)
    block_tables: torch.Tensor = None

    # The indices of the token slots that input tokens will be stored into.
    # E.g., if `slot_mapping` is [35, 2, 17] and the block size is 16, the
    # three tokens are stored in the 3rd slot in block 2, 2nd slot in block 0,
    # and 1st slot in block 1, respectively.
    # (num_tokens,)
    slot_mapping: torch.Tensor = None
    causal: bool = True
    # runner_type in model_config.
    model_runner_type: str = ""
    # prefill reshape_and_cache event
    reshape_cache_event: torch.npu.Event = None

    # Per-step scratch for C8-MXFP (QFA) layers. The builder creates one
    # AscendMetadata per step shared by all attention layers, so the QFA
    # metadata operator output and the int8 causal mask are computed once per
    # step (per decode/prefill subset) instead of once per layer. Keys:
    # "decode" / "prefill" -> QFA metadata tensor, "attn_mask_int8" -> Tensor.
    qfa_metadata_cache: dict = field(default_factory=dict)


@dataclass
class AscendAttentionPCPMetadata(AscendMetadata):
    """GQA metadata needed to write the complete PCP KV cache."""

    pcp_local_num_input_tokens: int = 0


class AscendAttentionMetadataBuilder(AttentionMetadataBuilder[AscendMetadata]):
    """
    Builder for constructing AscendMetadata from CommonAttentionMetadata.

    Handles attention mask generation and metadata preparation for
    Ascend FlashAttention backend.
    """

    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: int = 1
    metadata_cls: type[AscendMetadata] = AscendMetadata

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.compilation_config = vllm_config.compilation_config
        self.device = device
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len, AscendAttentionBackend.get_supported_kernel_block_sizes()[0]
        )

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )

        self.reorder_batch_threshold = self.decode_threshold

        scheduler_config = vllm_config.scheduler_config
        self.chunked_prefill_enabled = scheduler_config.enable_chunked_prefill
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendAttentionMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.ALWAYS

    def reorder_batch(self, input_batch, scheduler_output: "SchedulerOutput") -> bool:
        return False

    def _split_decodes_and_prefills(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> tuple[int, int, int, int]:
        return split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.decode_threshold,
        )

    def _build_backend_metadata(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        *,
        block_table: torch.Tensor,
        query_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        num_decodes: int,
        num_prefills: int,
    ) -> dict[str, Any]:
        """Extension point for layouts such as DCP.

        The base builder owns common token classification, padding, masks and
        cache metadata. Specialized backends only add their phase-specific
        metadata here.
        """
        return {}

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = self._split_decodes_and_prefills(
            common_attn_metadata
        )

        block_table = common_attn_metadata.block_table_tensor
        # Prefer _seq_lens_cpu (always available, updated during draft
        # iterations) over seq_lens_cpu (None in async spec decode mode).
        if common_attn_metadata._seq_lens_cpu is not None:
            seq_lens = common_attn_metadata._seq_lens_cpu[:num_reqs]
        elif common_attn_metadata.seq_lens_cpu is not None:
            seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
        else:
            seq_lens = common_attn_metadata.seq_lens[:num_reqs].to("cpu")

        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]
        # this slot_mapping override doesn't work since vllm will override it again. We should fix it vllm.
        # see: https://github.com/vllm-project/vllm/blob/ce88756b967c2c5006746a424c15dd59a284ed8c/vllm/model_executor/layers/attention/cross_attention.py#L117
        if isinstance(self.kv_cache_spec, CrossAttentionSpec):
            seq_lens = common_attn_metadata.seq_lens
            slot_mapping = common_attn_metadata.slot_mapping.to(torch.int32)
        elif self.speculative_config and self.speculative_config.parallel_drafting:
            seq_lens = common_attn_metadata.seq_lens

        attn_state = common_attn_metadata.attn_state

        # Get attn_mask from singleton AttentionMaskBuilder
        attn_mask = self.attn_mask_builder.get_attention_mask(common_attn_metadata.causal, self.model_config)

        # TODO: Yet another unnecessary H2D while we already have a query_start_loc on device
        query_start_loc = query_start_loc_cpu.pin_memory().to(self.device, non_blocking=True)

        actual_seq_lengths_q = query_start_loc_cpu[1:].tolist()
        seq_lens_list = seq_lens.tolist()
        # Sequence-parallel (or cudagraph) padding makes the model runner insert a
        # dummy padding request into query_start_loc to satisfy the FIA TND-layout
        # constraint (sum of q lengths == hidden_states.shape[0]), bumping the
        # q-derived batchSize by one. The query_start_loc buffer is sized
        # `max_num_reqs + 2` to hold it, but the seq_lens and block_table buffers
        # are only `max_num_reqs`, so when the batch is full the padded request
        # overflows and `[:num_reqs_padded]` silently truncates them. FIA then
        # fails (error 561002) checking, in order, the `actualSeqLengthsKv` length
        # and then the block_table row count against batchSize. Pad them to match:
        # the dummy request points at block 0, and its output is harmless because:
        #   (1) read side: the attention output for padding tokens is trimmed by
        #       `hidden_states = hidden_states[:-pad_size, :]` downstream;
        #   (2) write side: reshape_and_cache slices key/value/slot_mapping to
        #       `[:num_actual_tokens]` (unpadded count), so the dummy request
        #       never writes to KV cache.
        # So any valid positive KV length / zero block row is fine. Pad both
        # seq_lens_list and the seq_lens tensor: full_graph_fia_v2 passes the
        # seq_lens tensor (not seq_lens_list) as actual_seq_kvlen during graph
        # capture, and _get_fia_params derives the PrefillCacheHit batch size from
        # seq_lens.shape[0], so the tensor has to carry the dummy request too.
        num_reqs_fia = len(actual_seq_lengths_q)
        if len(seq_lens_list) < num_reqs_fia:
            padding_len = num_reqs_fia - len(seq_lens_list)
            seq_lens_list = seq_lens_list + [1] * padding_len
            seq_lens = torch.cat([seq_lens, seq_lens.new_ones(padding_len)])
        if block_table is not None and block_table.shape[0] < num_reqs_fia:
            block_table = torch.cat(
                [
                    block_table,
                    block_table.new_zeros((num_reqs_fia - block_table.shape[0], block_table.shape[1])),
                ],
                dim=0,
            )

        backend_metadata = self._build_backend_metadata(
            common_attn_metadata,
            block_table=block_table,
            query_lens=query_start_loc_cpu[1:] - query_start_loc_cpu[:-1],
            seq_lens=seq_lens,
            num_decodes=num_decodes,
            num_prefills=num_prefills,
        )
        attn_metadata = self.metadata_cls(
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            query_start_loc_gpu=common_attn_metadata.query_start_loc,
            seq_lens_gpu=common_attn_metadata.seq_lens,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            seq_lens_list=seq_lens_list,
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=actual_seq_lengths_q,
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            causal=common_attn_metadata.causal,
            model_runner_type=self.model_config.runner_type,
            **backend_metadata,
        )
        return attn_metadata

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):
        if attn_state in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.ChunkedPrefill,
            AscendAttentionState.SpecDecoding,
        ):
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly and ChunkedPrefill state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendAttentionPCPMetadataBuilder(AscendAttentionMetadataBuilder):
    """Build GQA metadata while retaining expanded cache slots."""

    metadata_cls = AscendAttentionPCPMetadata

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size

    def _split_decodes_and_prefills(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
    ) -> tuple[int, int, int, int]:
        return split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.decode_threshold,
            treat_short_extends_as_decodes=False,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendAttentionPCPMetadata:
        expanded_slot_mapping = common_attn_metadata.slot_mapping
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build,
        )
        assert isinstance(metadata, AscendAttentionPCPMetadata)
        if expanded_slot_mapping.numel() % self.pcp_size != 0:
            raise RuntimeError(
                "PCP slot mapping size must be divisible by the PCP world size: "
                f"{expanded_slot_mapping.numel()} % {self.pcp_size} != 0."
            )

        local_num_input_tokens = expanded_slot_mapping.numel() // self.pcp_size
        if metadata.num_actual_tokens > local_num_input_tokens:
            raise RuntimeError(
                "PCP actual token count exceeds the rank-local padded token count: "
                f"{metadata.num_actual_tokens} > {local_num_input_tokens}."
            )

        metadata.slot_mapping = expanded_slot_mapping
        metadata.pcp_local_num_input_tokens = local_num_input_tokens
        if metadata.num_prefills > 0:
            metadata.attn_state = AscendAttentionState.ChunkedPrefill
        return metadata


class AscendC8MXFPMetadataBuilder(AscendAttentionMetadataBuilder):
    """Metadata builder for the C8-MXFP (QFA) backend.

    The generic Ascend builder sizes its block-table width with the
    128-token generic block; this backend uses 512-token kernel blocks.

    Cudagraph support (final form, no .out wrapper variant needed):
    - PIECEWISE: QFA executes outside the compiled region as a plain call.
    - FULL / FULL_DECODE_ONLY: npugraph_ex captures the allocating QFA
      wrapper plus the in-graph metadata op natively (golden-test
      GRAPH_PATH=7 methodology). Replay correctness relies on the model
      runner's persistent length buffers (query_start_loc_gpu /
      seq_lens_gpu), which the impl derives QFA's cu_seqlens/seqused from
      through captured device-side ops, so every replay re-reads the
      current step's lengths.
    """

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec,
    ) -> AttentionCGSupport:
        mode = vllm_config.compilation_config.cudagraph_mode
        if mode.has_piecewise_cudagraphs() and not mode.has_full_cudagraphs():
            # PIECEWISE: QFA runs outside the compiled region as a plain
            # call (validated on-device). UNIFORM_BATCH is the support
            # level the other piecewise-capable Ascend backends report
            # (e.g. mla_v1); AttentionCGSupport has no PARTIAL member.
            return AttentionCGSupport.UNIFORM_BATCH
        # FULL (incl. FULL_DECODE_ONLY): npugraph_ex captures the
        # allocating QFA wrapper natively (golden-test GRAPH_PATH=7).
        return AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len, AscendC8MXFPAttentionBackend.get_supported_kernel_block_sizes()[0]
        )


class AscendAttentionBackendImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        sinks: torch.Tensor = None,
        **kwargs,
    ) -> None:
        self.vllm_config = get_current_vllm_config()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32, device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None
        self.is_kv_producer = (
            self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.is_kv_producer
        )
        self.enable_c8_quant = self.vllm_config.quant_config is not None and getattr(
            self.vllm_config.quant_config, "enable_c8_quant", False
        )
        self._use_layer_aware_fia_graph_replay = needs_layer_aware_fia_graph_replay()
        self._use_max_workspace_for_fia_graph = self._use_layer_aware_fia_graph_replay
        self.sinks = sinks
        self.layerIndex = 0
        # Some mixed-attention models cannot rely on the iteration order of
        # attn_metadata during graph replay. Record the captured layer name only
        # for that path.
        self._layer_name: str | None = None

    def _graph_metadata_layer_name(self, layer: AttentionLayer | None = None) -> str | None:
        layer_name = layer.layer_name if layer is not None else self._layer_name
        # KV-sharing layers replay with the target layer's metadata instead of
        # their own module name, matching vLLM's shared KV-cache ownership.
        return self.kv_sharing_target_layer_name or layer_name

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        draft_attn_metadatas=None,
    ):
        use_layer_aware_replay = needs_layer_aware_fia_graph_replay()
        if using_paged_attention(num_tokens, vllm_config):
            # Paged Attention update logic
            if _EXTRA_CTX.is_draft_model:
                if _EXTRA_CTX.is_draft_model_prefill:
                    graph_params = get_draft_graph_prefill_params()
                else:
                    graph_params = get_draft_graph_params()
            else:
                graph_params = get_graph_params()
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    forward_context.attn_metadata,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
                    (
                        query,
                        key_cache,
                        value_cache,
                        num_kv_heads,
                        num_heads,
                        scale,
                        block_table,
                        seq_lens,
                        output,
                    ) = param
                    seq_lens = forward_context.attn_metadata[key].seq_lens

                    workspace = torch_npu._npu_paged_attention_get_workspace(
                        query=query,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        num_kv_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale_value=scale,
                        block_table=block_table,
                        context_lens=seq_lens,
                        out=output,
                    )
                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu._npu_paged_attention(
                        query=query,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        num_kv_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale_value=scale,
                        block_table=block_table,
                        context_lens=seq_lens,
                        out=output,
                        workspace=workspace,
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)
        elif _EXTRA_CTX.sinks:
            # FIA update logic
            if _EXTRA_CTX.is_draft_model:
                graph_params = get_draft_graph_params()
                attn_metadata = draft_attn_metadatas
                draft_attn_key_steps = [
                    (draft_step, key)
                    for draft_step, per_step_metadata in enumerate(attn_metadata)
                    for key in per_step_metadata
                ]
                attn_keys = [key for _, key in draft_attn_key_steps]
            else:
                graph_params = get_graph_params()
                attn_metadata = forward_context.attn_metadata
                attn_keys = list(attn_metadata.keys())
            # For Qwen3-next, since the kv_cache_config has already categorized
            # linear_attn and self_attn, the attn_metadata is first arranged with
            # self_attn followed by linear_attn. Therefore, using zip directly
            # filters out the update operations for linear_attn.
            # TODO: We use a new variable `attn_keys` to ensure the loop count is
            # correct after get by `zip` because of the new structure of the attn_metadata
            # when running with the merged full eagle-graph. Should check it with Qwen3-next.
            num_layers = len(attn_keys)
            if num_layers == 0:
                return
            captured_attn_params = graph_params.attn_params[num_tokens]
            handles = graph_params.handles[num_tokens]
            events = graph_params.events[num_tokens]
            graph_param_count = len(captured_attn_params)
            workspace = graph_params.workspaces.get(num_tokens)
            if _EXTRA_CTX.is_draft_model:
                if graph_param_count > len(draft_attn_key_steps):
                    repeat_count = cdiv(graph_param_count, len(draft_attn_key_steps))
                    draft_attn_key_steps = (draft_attn_key_steps * repeat_count)[:graph_param_count]
                else:
                    draft_attn_key_steps = draft_attn_key_steps[:graph_param_count]
                attn_keys = [key for _, key in draft_attn_key_steps]
            elif use_layer_aware_replay:
                # One graph size can contain captured FIA ops from all layers.
                # Repeat attn keys to match the captured op count, then use the
                # stored layer name in each op param to resolve the exact
                # metadata entry during replay.
                attn_keys = [attn_keys[index % num_layers] for index in range(graph_param_count)]
            attn_count = 0
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    attn_keys,
                    captured_attn_params,
                    handles,
                    events,
                ):
                    (
                        query,
                        key_cache,
                        value,
                        block_tables,
                        attn_mask,
                        block_size,
                        seq_lens,
                        num_kv_heads,
                        num_heads,
                        scale,
                        sliding_window,
                        sinks,
                        attn_output,
                        softmax_lse,
                        layer_name,
                    ) = param

                    if _EXTRA_CTX.is_draft_model:
                        draft_step, key = draft_attn_key_steps[attn_count]
                        seq_lens = attn_metadata[draft_step][key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[draft_step][key].actual_seq_lengths_q
                        attn_count = attn_count + 1
                    else:
                        metadata_key = layer_name if layer_name is not None and layer_name in attn_metadata else key
                        seq_lens = attn_metadata[metadata_key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[metadata_key].actual_seq_lengths_q

                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu.npu_fused_infer_attention_score_v2.out(
                        query=query,
                        key=key_cache,
                        value=value,
                        block_table=block_tables,
                        atten_mask=attn_mask,
                        input_layout="TND",
                        block_size=block_size,
                        actual_seq_qlen=actual_seq_lengths_q,
                        actual_seq_kvlen=seq_lens,
                        num_key_value_heads=num_kv_heads,
                        num_query_heads=num_heads,
                        sparse_mode=4 if sliding_window is not None else 3,
                        pre_tokens=sliding_window if sliding_window is not None else SWA_INT_MAX,
                        next_tokens=0,
                        softmax_scale=scale,
                        learnable_sink=sinks,
                        workspace=workspace,
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)
        else:
            # FIA update logic
            if _EXTRA_CTX.is_draft_model:
                if _EXTRA_CTX.is_draft_model_prefill:
                    graph_params = get_draft_graph_prefill_params()
                else:
                    graph_params = get_draft_graph_params()
                attn_metadata = draft_attn_metadatas
                draft_attn_key_steps = [
                    (draft_step, key)
                    for draft_step, per_step_metadata in enumerate(attn_metadata)
                    for key in per_step_metadata
                ]
                attn_keys = [key for _, key in draft_attn_key_steps]
            else:
                graph_params = get_graph_params()
                attn_metadata = forward_context.attn_metadata
                # Only standard (FIA) attention layers have captured graph
                # params here; linear/GDN layers (GDNAttentionMetadata) are
                # updated separately by update_conv1d_graph_params. So we filter by `seq_lens_list`
                attn_keys = [k for k in attn_metadata if hasattr(attn_metadata[k], "seq_lens_list")]
                if not use_layer_aware_replay:
                    # In some speculative methods (such as DFlash), the order of
                    # attn_keys in the Target model will be disrupted instead of
                    # increasing by layer index, so need regular expressions to
                    # reorder the attn_keys and store the results in
                    # _ATTN_KEYS_BUFFER.
                    attn_keys_length = len(graph_params.attn_params[num_tokens])
                    global _ATTN_KEYS_BUFFER
                    if attn_keys_length == 0:
                        return
                    if not _ATTN_KEYS_BUFFER or len(_ATTN_KEYS_BUFFER) != attn_keys_length:
                        import regex as re

                        def extract_layer_index(key: str) -> int:
                            match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", key)
                            return int(match.group(1)) if match else 0

                        def is_direct_target_attn_key(key: str) -> bool:
                            return (
                                re.search(
                                    r"(?:^|\.)layers\.(\d+)\.self_attn\.attn$",
                                    key,
                                )
                                is not None
                            )

                        attn_keys_to_order = attn_keys[:attn_keys_length]
                        if getattr(speculative_config, "method", None) == "mtp":
                            # Step3.5 MTP can expose draft KV-cache groups in the
                            # target runtime metadata.  The target FULL graph only
                            # captures direct base-model self-attention handles, so
                            # select that target key domain instead of depending on
                            # the current draft module name.
                            direct_target_attn_keys = [key for key in attn_keys if is_direct_target_attn_key(key)]
                            if len(direct_target_attn_keys) >= attn_keys_length:
                                attn_keys_to_order = direct_target_attn_keys

                        attn_keys_tmp = attn_keys_to_order
                        attn_keys_tmp.sort(key=extract_layer_index)
                        _ATTN_KEYS_BUFFER = attn_keys_tmp[:attn_keys_length]
                    attn_keys[:attn_keys_length] = _ATTN_KEYS_BUFFER
            # For Qwen3-next, since the kv_cache_config has already categorized
            # linear_attn and self_attn, the attn_metadata is first arranged with
            # self_attn followed by linear_attn. Therefore, using zip directly
            # filters out the update operations for linear_attn.
            # TODO: We use a new variable `attn_keys` to ensure the loop count is
            # correct after get by `zip` because of the new structure of the attn_metadata
            # when running with the merged full eagle-graph. Should check it with Qwen3-next.
            num_layers = len(attn_keys)
            if num_layers == 0:
                return
            captured_attn_params = graph_params.attn_params[num_tokens]
            handles = graph_params.handles[num_tokens]
            events = graph_params.events[num_tokens]
            graph_param_count = len(captured_attn_params)
            workspace = graph_params.workspaces.get(num_tokens)
            if _EXTRA_CTX.is_draft_model:
                if graph_param_count > len(draft_attn_key_steps):
                    repeat_count = cdiv(graph_param_count, len(draft_attn_key_steps))
                    draft_attn_key_steps = (draft_attn_key_steps * repeat_count)[:graph_param_count]
                else:
                    draft_attn_key_steps = draft_attn_key_steps[:graph_param_count]
                attn_keys = [key for _, key in draft_attn_key_steps]
            elif use_layer_aware_replay:
                # Keep the replay loop length aligned with captured FIA ops;
                # layer-specific metadata lookup below prevents global/sliding
                # window layers from accidentally sharing the same metadata.
                attn_keys = [attn_keys[index % num_layers] for index in range(graph_param_count)]
            attn_count = 0
            layer_count = 0
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    attn_keys,
                    captured_attn_params,
                    handles,
                    events,
                ):
                    if isinstance(param, PagedAttentionGraphParam):
                        if _EXTRA_CTX.is_draft_model:
                            draft_step, key = draft_attn_key_steps[attn_count]
                            block_table = attn_metadata[draft_step][key].block_tables
                            seq_lens = attn_metadata[draft_step][key].seq_lens
                            attn_count = attn_count + 1
                        else:
                            layer_name = param.layer_name
                            metadata_key = layer_name if layer_name is not None and layer_name in attn_metadata else key
                            block_table = attn_metadata[metadata_key].block_tables
                            seq_lens = attn_metadata[metadata_key].seq_lens
                        update_paged_attention_graph_param(
                            update_stream,
                            handle,
                            event,
                            param,
                            block_table,
                            seq_lens,
                        )
                        continue
                    (
                        query,
                        key_cache,
                        value,
                        block_tables,
                        attn_mask,
                        block_size,
                        seq_lens,
                        query_start_loc,
                        num_kv_heads,
                        num_heads,
                        scale,
                        attn_output,
                        softmax_lse,
                        sparse_mode,
                        pre_tokens,
                        next_tokens,
                        sliding_window,
                        c8_k_aq_scale,
                        c8_k_aq_offset,
                        c8_v_aq_scale,
                        c8_v_aq_offset,
                        layer_name,
                    ) = param

                    if _EXTRA_CTX.is_draft_model:
                        draft_step, key = draft_attn_key_steps[attn_count]
                        metadata = attn_metadata[draft_step][key]
                        seq_lens = metadata.seq_lens_list
                        actual_seq_lengths_q = metadata.actual_seq_lengths_q
                        block_tables = metadata.block_tables
                        attn_count = attn_count + 1
                        if not metadata.causal:
                            sparse_mode = 0
                    else:
                        metadata_key = layer_name if layer_name is not None and layer_name in attn_metadata else key
                        seq_lens = attn_metadata[metadata_key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[metadata_key].actual_seq_lengths_q
                        # NOTE:
                        # For models with sliding-window attention on the FIA full-graph replay path,
                        # rebinding `block_tables` to the latest metadata tensor causes corrupted /
                        # repeated outputs in our repro on Ascend NPU.
                        #
                        # Keep the captured block_tables tensor on this affected path.
                        # Non-SWA models preserve the original behavior and continue to refresh
                        # block_tables from attn_metadata.
                        if not sliding_window:
                            block_tables = attn_metadata[metadata_key].block_tables
                    layer_count += 1

                    torch.npu.graph_task_update_begin(update_stream, handle)
                    input_layout = "TND"
                    extra_args = {}
                    if c8_k_aq_scale is not None:
                        extra_args = {
                            "key_antiquant_scale": c8_k_aq_scale,
                            "value_antiquant_scale": c8_v_aq_scale,
                            "key_antiquant_mode": 0,
                            "value_antiquant_mode": 0,
                            "inner_precise": 1,
                        }
                        input_layout = "BNSD"
                        sparse_mode = 0
                    torch_npu.npu_fused_infer_attention_score.out(
                        query=query,
                        key=key_cache,
                        value=value,
                        block_table=block_tables,
                        atten_mask=attn_mask,
                        input_layout=input_layout,
                        block_size=block_size,
                        actual_seq_lengths=actual_seq_lengths_q,
                        actual_seq_lengths_kv=seq_lens,
                        num_key_value_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale=scale,
                        sparse_mode=sparse_mode,
                        pre_tokens=pre_tokens,
                        next_tokens=next_tokens,
                        **extra_args,
                        workspace=workspace,
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)

                    event.record(update_stream)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)

    def full_graph_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer=None,
    ) -> torch.Tensor:
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)

        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        if _EXTRA_CTX.is_draft_model:
            if _EXTRA_CTX.is_draft_model_prefill:
                graph_params = get_draft_graph_prefill_params()
            else:
                graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        input_layout = "TND"
        attn_mask = attn_metadata.attn_mask
        sparse_mode = 4 if self.sliding_window else 3 if attn_metadata.causal else 0
        pre_tokens = self.sliding_window or SWA_INT_MAX
        next_tokens = 0 if self.sliding_window else SWA_INT_MAX

        extra_args = {}
        if self.enable_c8_quant and layer is not None:
            extra_args = {
                "key_antiquant_scale": layer._c8_k_aq_scale_nz_bnsd,
                "value_antiquant_scale": layer._c8_v_aq_scale_nz_bnsd,
                "key_antiquant_mode": 0,
                "value_antiquant_mode": 0,
                "inner_precise": 1,
            }

            # change key/value shape
            _, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self._nz_5d_view(self.key_cache, block_size)
            value = self._nz_5d_view(self.value_cache, block_size)

            # TODO: change layerout from BNSD to TND.
            input_layout = "BNSD"
            query = query.unsqueeze(2)
            output = output.unsqueeze(2)
            attn_mask = None
            sparse_mode = 0
        use_max_workspace = self._use_max_workspace_for_fia_graph
        workspace = graph_params.workspaces.get(num_tokens)
        should_update_workspace_cache = False
        if use_max_workspace:
            # Some models mix attention layer shapes under the same graph size.
            # During capture, keep the largest required workspace for that size.
            candidate_workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_mask,
                block_table=block_table,
                input_layout=input_layout,
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=sparse_mode,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
                scale=self.scale,
                **extra_args,
            )
            workspace = cache_graph_workspace(
                graph_params,
                num_tokens,
                candidate_workspace,
                use_max_workspace=use_max_workspace,
            )
            should_update_workspace_cache = True
        elif workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_mask,
                block_table=block_table,
                input_layout=input_layout,
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=sparse_mode,
                pre_tokens=pre_tokens,
                next_tokens=next_tokens,
                scale=self.scale,
                **extra_args,
            )
            should_update_workspace_cache = True
        if should_update_workspace_cache:
            if _EXTRA_CTX.is_draft_model:
                update_draft_graph_params_workspaces(num_tokens, workspace)
            else:
                update_graph_params_workspaces(num_tokens, workspace)

        # Handle graph capturing mode
        stream = torch_npu.npu.current_stream()

        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        attn_params = (
            weak_ref_tensors(query),
            weak_ref_tensors(key),
            weak_ref_tensors(value),
            weak_ref_tensors(block_table),
            weak_ref_tensors(attn_mask) if attn_mask is not None else None,
            block_size,
            actual_seq_lengths_kv,
            actual_seq_lengths_q,
            self.num_kv_heads,
            self.num_heads,
            self.scale,
            weak_ref_tensors(output),
            weak_ref_tensors(softmax_lse),
            sparse_mode,
            pre_tokens,
            next_tokens,
            self.sliding_window,
        )
        if self.enable_c8_quant and layer is not None:
            attn_params = attn_params + (
                weak_ref_tensors(layer._c8_k_aq_scale_nz_bnsd),
                None,
                weak_ref_tensors(layer._c8_v_aq_scale_nz_bnsd),
                None,
            )  # type: ignore
        else:
            attn_params = attn_params + (None, None, None, None)  # type: ignore
        layer_name = self._graph_metadata_layer_name(layer) if self._use_layer_aware_fia_graph_replay else None
        attn_params = attn_params + (layer_name,)  # type: ignore
        graph_params.attn_params[num_tokens].append(attn_params)

        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_mask,
            block_table=block_table,
            input_layout=input_layout,
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            workspace=workspace,
            out=[output, softmax_lse],
            **extra_args,
        )

        output = output.view(num_tokens, self.num_heads, self.head_size)

        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
        return output, num_tokens

    def full_graph_fia_v2(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)
        actual_seq_lengths_kv = attn_metadata.seq_lens
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        if _EXTRA_CTX.is_draft_model:
            graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()

        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        use_max_workspace = self._use_max_workspace_for_fia_graph
        workspace = graph_params.workspaces.get(num_tokens)
        should_update_workspace_cache = False
        if use_max_workspace:
            # See full_graph_fia: this path needs the max workspace across layer
            # variants sharing the same graph size.
            candidate_workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_qlen=actual_seq_lengths_q,
                actual_seq_kvlen=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                softmax_scale=self.scale,
                num_query_heads=self.num_heads,
                sparse_mode=4 if self.sliding_window is not None else 3,
                pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
                next_tokens=0,
                learnable_sink=self.sinks,
            )
            workspace = cache_graph_workspace(
                graph_params,
                num_tokens,
                candidate_workspace,
                use_max_workspace=use_max_workspace,
            )
            should_update_workspace_cache = True
        elif workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_qlen=actual_seq_lengths_q,
                actual_seq_kvlen=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                softmax_scale=self.scale,
                num_query_heads=self.num_heads,
                sparse_mode=4 if self.sliding_window is not None else 3,
                pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
                next_tokens=0,
                learnable_sink=self.sinks,
            )
            should_update_workspace_cache = True
        if should_update_workspace_cache:
            if _EXTRA_CTX.is_draft_model:
                update_draft_graph_params_workspaces(num_tokens, workspace)
            else:
                update_graph_params_workspaces(num_tokens, workspace)

        # Handle graph capturing mode
        stream = torch_npu.npu.current_stream()

        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        graph_params.attn_params[num_tokens].append(
            (
                weak_ref_tensors(query),
                weak_ref_tensors(key),
                weak_ref_tensors(value),
                weak_ref_tensors(block_table),
                weak_ref_tensors(attn_metadata.attn_mask),
                block_size,
                actual_seq_lengths_kv,
                self.num_kv_heads,
                self.num_heads,
                self.scale,
                self.sliding_window,
                self.sinks,
                weak_ref_tensors(output),
                weak_ref_tensors(softmax_lse),
                self._graph_metadata_layer_name() if self._use_layer_aware_fia_graph_replay else None,
            )
        )
        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score_v2.out(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_qlen=actual_seq_lengths_q,
            actual_seq_kvlen=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_query_heads=self.num_heads,
            sparse_mode=4 if self.sliding_window is not None else 3,
            pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
            next_tokens=0,
            softmax_scale=self.scale,
            learnable_sink=self.sinks,
            workspace=workspace,
            out=[output, softmax_lse],
        )
        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
        return output, num_tokens

    def full_graph_pa(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ):
        graph_params = get_graph_params()
        num_tokens = query.shape[0]
        if _EXTRA_CTX.capturing:
            # Get workspace from cache or calculate it if not present.
            workspace = graph_params.workspaces.get(num_tokens)
            if workspace is None:
                workspace = torch_npu._npu_paged_attention_get_workspace(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=attn_metadata.block_tables,
                    context_lens=attn_metadata.seq_lens,
                    out=output,
                )
                update_graph_params_workspaces(num_tokens, workspace)

            # Handle graph capturing mode
            stream = torch_npu.npu.current_stream()

            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[num_tokens].append(event)
            graph_params.attn_params[num_tokens].append(
                PagedAttentionGraphParam(
                    (
                        weak_ref_tensors(query),
                        weak_ref_tensors(self.key_cache),
                        weak_ref_tensors(self.value_cache),
                        self.num_kv_heads,
                        self.num_heads,
                        self.scale,
                        attn_metadata.block_tables,
                        attn_metadata.seq_lens,
                        weak_ref_tensors(output),
                    ),
                    self._graph_metadata_layer_name() if self._use_layer_aware_fia_graph_replay else None,
                )
            )

            torch.npu.graph_task_group_begin(stream)
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                block_table=attn_metadata.block_tables,
                context_lens=attn_metadata.seq_lens,
                out=output,
                workspace=workspace,
            )
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
            return output

    def _get_fia_params(self, key: torch.Tensor, value: torch.Tensor, attn_metadata: AscendMetadata, kv_cache=None):
        # PrefillNoCache doesn't need key_cache, but other modes do
        # Only initialize/require cache for modes that actually use it
        if attn_metadata.attn_state != AscendAttentionState.PrefillNoCache:
            # Initialize cache from kv_cache if not already set (for DecodeOnly mode)
            if self.key_cache is None and kv_cache is not None:
                if (
                    isinstance(kv_cache, torch.Tensor)
                    and kv_cache.dim() > 0
                    and kv_cache.shape[0] == 2
                    or isinstance(kv_cache, (list, tuple))
                    and len(kv_cache) >= 2
                ):
                    self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

            if self.key_cache is None:
                raise RuntimeError(
                    f"key_cache is None in _get_fia_params for mode {attn_metadata.attn_state}. kv_cache={kv_cache}"
                )

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            block_size = 128
            block_table = None
            actual_seq_lengths_kv = attn_metadata.actual_seq_lengths_q
            if self.attn_type == AttentionType.ENCODER_DECODER:
                actual_seq_lengths_kv = torch.cumsum(attn_metadata.seq_lens, dim=0).tolist()
        elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            batch_size = attn_metadata.seq_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        # chunked prefill.
        else:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        return key, value, block_size, block_table, actual_seq_lengths_kv

    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        kv_cache=None,
    ):
        # we inherit ForwardContext in model runner v2, when enable model
        # runner v2, there is not capturing attribute in forward_context,
        # just use getattr to avoid attribute error.
        if _EXTRA_CTX.capturing:
            if self.sinks is not None:
                attn_output, num_tokens = self.full_graph_fia_v2(query, key, value, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            else:
                attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
        passed_value = value
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(
            key, value, attn_metadata, kv_cache
        )
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]
        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]
        # Get workspace from cache or calculate it if not present.
        if self.sinks is not None:
            actual_seq_qlen = attn_metadata.actual_seq_lengths_q
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                actual_seq_qlen = torch.tensor([1] * len(attn_metadata.seq_lens_list), dtype=torch.int32).cumsum(dim=0)
            if self.sliding_window is not None:
                sparse_mode = 4
            else:
                sparse_mode = 3
            attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                query,
                key.contiguous(),
                value.contiguous(),
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
                next_tokens=0,
                atten_mask=attn_metadata.attn_mask,
                sparse_mode=sparse_mode,
                softmax_scale=self.scale,
                block_table=block_table,
                block_size=block_size,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_lengths_kv,
                learnable_sink=self.sinks,
            )
        else:
            if not attn_metadata.causal:
                attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    sparse_mode=0,
                )
            elif self.sliding_window is not None:
                attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    atten_mask=attn_metadata.attn_mask,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    pre_tokens=self.sliding_window,
                    next_tokens=0,
                    sparse_mode=4,
                )
            else:
                # ChunkedPrefill mixing prefill+decode: split into a per-phase
                # FIA call each (A5 only).
                if (
                    get_current_hardware_profile().supports(HardwareCapability.CHUNKED_PREFILL_PHASE_SPLIT)
                    and attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill
                    and attn_metadata.num_decodes > 0
                    and attn_metadata.num_prefills > 0
                ):
                    return self._forward_fia_chunked_prefill_split(
                        query, key, value, key, passed_value, block_size, block_table, attn_metadata, output
                    )
                attn_output, _ = DeviceOperator.npu_fused_infer_attention_score(
                    query=query,
                    key=key,
                    value=value,
                    atten_mask=attn_metadata.attn_mask,
                    block_table=block_table,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    head_size=self.head_size,
                    scale=self.scale,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    current_key=key,
                    current_value=passed_value,
                    attn_metadata=attn_metadata,
                    is_prefill_no_cache=attn_metadata.attn_state == AscendAttentionState.PrefillNoCache,
                    sparse_mode=3,
                )

            attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def _forward_fia_chunked_prefill_split(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        block_size: int,
        block_table: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """ChunkedPrefill with mixed prefill/decode: run decode and prefill in
        separate FIA calls. split_decodes_and_prefills has reordered the batch
        so decodes occupy [0, num_decode_tokens) and prefills the rest
        """
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        seq_lens_list = attn_metadata.seq_lens_list
        num_tokens = int(actual_seq_qlen[-1])

        # decode part
        if num_decode_tokens > 0:
            decode_out, _ = DeviceOperator.npu_fused_infer_attention_score(
                query=query[:num_decode_tokens],
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table[:num_decodes],
                input_layout="TND",
                block_size=block_size,
                # cumulative offset from 0; leading num_decodes entries used as-is
                actual_seq_lengths=actual_seq_qlen[:num_decodes],
                actual_seq_lengths_kv=seq_lens_list[:num_decodes],
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                head_size=self.head_size,
                scale=self.scale,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                current_key=key,
                current_value=value,
                attn_metadata=attn_metadata,
                is_prefill_no_cache=False,
                sparse_mode=3,
            )
            output[:num_decode_tokens] = decode_out.view(num_decode_tokens, self.num_heads, self.head_size)

        # prefill part
        if attn_metadata.num_prefills > 0:
            # rebase cumulative q offsets to start at 0 for the prefill slice
            prefill_seq_qlen = [
                actual_seq_qlen[i] - num_decode_tokens for i in range(num_decodes, len(actual_seq_qlen))
            ]
            prefill_out, _ = DeviceOperator.npu_fused_infer_attention_score(
                query=query[num_decode_tokens:num_tokens],
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table[num_decodes:],
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=prefill_seq_qlen,
                actual_seq_lengths_kv=seq_lens_list[num_decodes:],
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                head_size=self.head_size,
                scale=self.scale,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                current_key=key,
                current_value=value,
                attn_metadata=attn_metadata,
                is_prefill_no_cache=False,
                sparse_mode=3,
            )
            n_prefill = num_tokens - num_decode_tokens
            output[num_decode_tokens:num_tokens] = prefill_out.view(n_prefill, self.num_heads, self.head_size)
        return output

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if _EXTRA_CTX.capturing:
            return self.full_graph_pa(query, attn_metadata, output)
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=attn_metadata.block_tables,
            context_lens=attn_metadata.seq_lens,
            out=output,
        )
        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        _: torch.Tensor,
    ) -> torch.Tensor:
        # use default sparse_mode 0 in normal scenario, which means no mask works on it
        # Pad actual_seq_len with 0 when num_tokens > actual_seq_len in TND layout
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        if query.shape[0] > actual_seq_qlen[-1]:
            actual_seq_qlen = actual_seq_qlen + [0]
        return torch_npu.npu_fusion_attention(
            query=query,
            key=key,
            value=value,
            head_num=self.num_heads,
            input_layout="TND",
            scale=self.scale,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_qlen,
        )[0]

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: list[torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY):
            return

        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        DeviceOperator.reshape_and_cache(
            key=key,
            value=value,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            slot_mapping=slot_mapping,
        )

    def reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if len(kv_cache) > 1:
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            if self.kv_sharing_target_layer_name is not None:
                # KV-sharing target layers consume another layer's cache.
                # Writing their dummy/current K/V would overwrite shared slots.
                if self.is_kv_producer:
                    attn_metadata.reshape_cache_event.record()
                return query, key, value, output
            slots = attn_metadata.slot_mapping
            encoder_decoder = self.attn_type == AttentionType.ENCODER_DECODER
            DeviceOperator.reshape_and_cache(
                key=key[: attn_metadata.num_actual_tokens] if not encoder_decoder else key,
                value=value[: attn_metadata.num_actual_tokens] if not encoder_decoder else value,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                # quick fix to make sure slots is int32 for cross attention case.
                # see: https://github.com/vllm-project/vllm/blob/ce88756b967c2c5006746a424c15dd59a284ed8c/vllm/model_executor/layers/attention/cross_attention.py#L117
                slot_mapping=slots[: attn_metadata.num_actual_tokens] if not encoder_decoder else slots.to(torch.int32),
            )
            notify_kv_cache_written()
        return query, key, value, output

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        record_attention_compute_start()
        num_tokens = query.shape[0]

        if (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and self.sliding_window is None
            and using_paged_attention(num_tokens, self.vllm_config, self.head_size)
        ):
            output = self.forward_paged_attention(query, attn_metadata, output)
        else:
            output = self.forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)

        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        if self._use_layer_aware_fia_graph_replay:
            self._layer_name = layer.layer_name

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendAttentionBackendImpl")

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        # Initialize key_cache and value_cache from kv_cache if not already set.
        # This is needed for DecodeOnly mode where key/value are None but we still
        # need access to the cache for attention computation.
        if self.key_cache is None and kv_cache is not None:
            if (
                isinstance(kv_cache, torch.Tensor)
                and kv_cache.dim() > 0
                and kv_cache.shape[0] == 2
                or isinstance(kv_cache, (list, tuple))
                and len(kv_cache) >= 2
            ):
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        output_padded = None
        if key is not None and value is not None:
            output_padded = output
            query, key, value, output_padded = self.reshape_and_cache(
                query, key, value, kv_cache, attn_metadata, output
            )
        # pooling model branch
        if attn_metadata.model_runner_type == "pooling" and not attn_metadata.causal:
            attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output
        if output_padded is not None:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output_padded)
        else:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output)
        output[:num_tokens] = attn_output[:num_tokens]
        return output


class AscendAttentionPCPImpl(AscendAttentionBackendImpl):
    """MRV2 GQA implementation for prefill context parallelism."""

    supports_pcp = True

    def reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if len(kv_cache) <= 1:
            return query, key, value, output
        expanded_slot_mapping = attn_metadata.slot_mapping
        local_num_input_tokens = attn_metadata.pcp_local_num_input_tokens
        if key.shape[0] < local_num_input_tokens:
            raise RuntimeError(
                f"PCP GQA input is shorter than the rank-local padded batch: {key.shape[0]} < {local_num_input_tokens}."
            )

        (cache_key, cache_value), cache_slot_mapping = _gather_prefill_cache_inputs(
            (
                key[:local_num_input_tokens],
                value[:local_num_input_tokens],
            ),
            expanded_slot_mapping,
            attn_metadata.num_decode_tokens,
        )
        local_num_actual_tokens = attn_metadata.num_actual_tokens
        try:
            attn_metadata.slot_mapping = cache_slot_mapping
            attn_metadata.num_actual_tokens = cache_key.shape[0]
            super().reshape_and_cache(query, cache_key, cache_value, kv_cache, attn_metadata, output)
        finally:
            attn_metadata.slot_mapping = expanded_slot_mapping
            attn_metadata.num_actual_tokens = local_num_actual_tokens

        return query, key, value, output


class AscendC8AttentionBackendImpl(AscendAttentionBackendImpl):
    """Attention backend implementation for INT8 KV cache (C8/QuaRot) models.

    This subclass handles static per-channel INT8 KV cache quantization.
    It is activated via class surgery in AscendC8KVCacheAttentionMethod.create_weights
    (vllm_ascend/quantization/methods/kv_c8.py)
    so that C8 attention layers automatically use this forward path.
    """

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if self._use_layer_aware_fia_graph_replay:
            self._layer_name = layer.layer_name

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendC8AttentionBackendImpl")

        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        self._prepare_c8_scales(layer, query.device)
        float_key, float_value = None, None
        if self.vllm_config.kv_transfer_config is None:
            if key is not None and value is not None:
                if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
                    float_key, float_value = key, value
                key, value = self._quantize_kv_to_int8(key, value, layer, attn_metadata.num_actual_tokens)
                query, key, value, _ = self._reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)
            # pooling model branch
            if attn_metadata.model_runner_type == "pooling":
                attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output

            # When `modelrunnerv2` compiles the graph, the value of `attn_metadata.attn_state` is `None`;
            # therefore, the graph-mode condition needs to be evaluated earlier.
            if _EXTRA_CTX.capturing:
                attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output, layer)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                return self._forward_c8_decode(query, attn_metadata, output, layer)
            elif attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
                return self._forward_c8_chunked_prefill(query, float_key, float_value, attn_metadata, output, layer)
            else:
                return self._forward_c8_fused_infer_attention(
                    query,
                    float_key if float_key is not None else key,
                    float_value if float_value is not None else value,
                    attn_metadata,
                    output,
                    layer,
                )
        else:
            if attn_metadata.attn_state != AscendAttentionState.DecodeOnly and self.is_kv_producer:
                output_padded = None
                if key is not None and value is not None:
                    output_padded = output
                    query, key, value, output_padded = self.reshape_and_cache(
                        query, key, value, kv_cache, attn_metadata, output
                    )
                # pooling model branch
                if attn_metadata.model_runner_type == "pooling":
                    attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                if output_padded is not None:
                    attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output_padded)
                else:
                    attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            elif not self.is_kv_producer:
                if key is not None and value is not None:
                    key, value = self._quantize_kv_to_int8(key, value, layer, attn_metadata.num_actual_tokens)
                    query, key, value, _ = self._reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)
                # pooling model branch
                if attn_metadata.model_runner_type == "pooling":
                    attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                if _EXTRA_CTX.capturing:
                    attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output, layer)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                    return self._forward_c8_decode(query, attn_metadata, output, layer)

    def _nz_5d_view(self, cache: torch.Tensor, block_size: int) -> torch.Tensor:
        """View a KV cache tensor in NZ 5D layout: (num_blocks, num_kv_heads, head_size//nz, block_size, nz)."""
        NZ_FMT_LAST_DIM = 32
        return cache.view(-1, self.num_kv_heads, self.head_size // NZ_FMT_LAST_DIM, block_size, NZ_FMT_LAST_DIM)

    def _prepare_c8_scales(self, layer: AttentionLayer, device: torch.device) -> None:
        """Shard per-channel C8 scales/offsets to this TP rank and pre-compute
        BF16 BNSD antiquant tensors for FIA V1 decode fast path.
        """
        if hasattr(layer, "_c8_scales_prepared"):
            return

        def _shard_and_reshape(raw: torch.Tensor) -> torch.Tensor:
            if raw.numel() == 1:
                return raw.to(device=device)
            expected = self.num_kv_heads * self.head_size
            if raw.numel() != expected:
                total_kv_heads = raw.numel() // self.head_size
                tp_rank = get_tensor_model_parallel_rank()
                tp_size = get_tensor_model_parallel_world_size()
                kv_head_start = tp_rank * total_kv_heads // tp_size
                raw = raw.view(total_kv_heads, self.head_size)[
                    kv_head_start : kv_head_start + self.num_kv_heads
                ].contiguous()
            return raw.view(1, self.num_kv_heads, self.head_size).to(device=device)

        layer._c8_k_scale = _shard_and_reshape(layer.k_cache_scale.data)
        layer._c8_k_offset = _shard_and_reshape(layer.k_cache_offset.data)
        layer._c8_v_scale = _shard_and_reshape(layer.v_cache_scale.data)
        layer._c8_v_offset = _shard_and_reshape(layer.v_cache_offset.data)

        layer._c8_k_inv_scale = 1.0 / layer._c8_k_scale
        layer._c8_v_inv_scale = 1.0 / layer._c8_v_scale

        nz_bnsd = (self.num_kv_heads, 1, self.head_size)
        layer._c8_k_aq_scale_nz_bnsd = layer._c8_k_scale.view(nz_bnsd).contiguous()
        layer._c8_v_aq_scale_nz_bnsd = layer._c8_v_scale.view(nz_bnsd).contiguous()

        layer._c8_scales_prepared = True

    def _dequant_paged_kv_to_dense(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: list,
        target_dtype: torch.dtype,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather paged INT8 KV blocks and dequantize."""
        batch_size = block_table.shape[0]
        max_blocks_per_seq = block_table.shape[1]

        # NZ 5D view: (num_blocks, num_kv_heads, head_size//nz, block_size, nz)
        block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
        max_tokens_padded = max_blocks_per_seq * block_size

        flat_ids = block_table.reshape(-1)
        key_nz = self._nz_5d_view(key, block_size)
        value_nz = self._nz_5d_view(value, block_size)

        # Gather: (batch*max_blocks, H, D//nz, S, nz)
        gathered_k = key_nz[flat_ids]
        gathered_v = value_nz[flat_ids]
        # NZ→ND conversion: permute (S, H, D//nz, nz) → reshape (S, H, D)
        gathered_k = (
            gathered_k.permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(batch_size, max_tokens_padded, self.num_kv_heads, self.head_size)
        )
        gathered_v = (
            gathered_v.permute(0, 3, 1, 2, 4)
            .contiguous()
            .view(batch_size, max_tokens_padded, self.num_kv_heads, self.head_size)
        )

        seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=key.device)
        positions = torch.arange(max_tokens_padded, dtype=torch.long, device=key.device)
        valid_mask = (positions.unsqueeze(0) < seq_lens_t.unsqueeze(1)).view(-1)

        dense_k = gathered_k.view(-1, self.num_kv_heads, self.head_size)[valid_mask]
        dense_v = gathered_v.view(-1, self.num_kv_heads, self.head_size)[valid_mask]

        # Scale-only dequant for NZ (symmetric)
        dense_k = dense_k.to(target_dtype) * layer._c8_k_scale
        dense_v = dense_v.to(target_dtype) * layer._c8_v_scale
        return dense_k, dense_v

    def _quantize_kv_to_int8(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: AttentionLayer,
        num_actual_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize K/V from float to INT8 using static per-channel C8 scales."""
        actual_key = key[:num_actual_tokens]
        actual_value = value[:num_actual_tokens]

        k_int8 = torch.clamp(
            torch.round(actual_key * layer._c8_k_inv_scale + layer._c8_k_offset),
            -128,
            127,
        ).to(torch.int8)
        v_int8 = torch.clamp(
            torch.round(actual_value * layer._c8_v_inv_scale + layer._c8_v_offset),
            -128,
            127,
        ).to(torch.int8)
        return k_int8, v_int8

    def _forward_c8_decode(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """C8 decode via FIA V1 BNSD with native paged INT8 KV + perchannel antiquant."""
        num_block, block_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
        assert block_size % 32 == 0, f"C8 INT8 KV cache requires block_size to be a multiple of 32, got {block_size}"
        batch_size = len(attn_metadata.seq_lens_list)

        key = self._nz_5d_view(self.key_cache, block_size)
        value = self._nz_5d_view(self.value_cache, block_size)

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query[:batch_size].unsqueeze(2),
            key,
            value,
            key_antiquant_scale=layer._c8_k_aq_scale_nz_bnsd,
            value_antiquant_scale=layer._c8_v_aq_scale_nz_bnsd,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BNSD",
            scale=self.scale,
            block_size=block_size,
            antiquant_mode=0,
            key_antiquant_mode=0,
            value_antiquant_mode=0,
            inner_precise=1,
            sparse_mode=0,
        )
        attn_output = attn_output.squeeze(2)
        output[:batch_size] = attn_output
        return output

    def _forward_c8_chunked_prefill(
        self,
        query: torch.Tensor,
        float_key: torch.Tensor | None,
        float_value: torch.Tensor | None,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """C8 ChunkedPrefill: decode via FIA V1 BNSD paged INT8 (zero gather),
        prefill via FIA V1 TND with float KV (new) or gather+dequant (continuing).
        """
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_decodes = attn_metadata.num_decodes
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]

        if num_decode_tokens > 0:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
            assert block_size % 32 == 0, (
                f"C8 INT8 KV cache requires block_size to be a multiple of 32, got {block_size}"
            )
            kv_k = self._nz_5d_view(self.key_cache, block_size)
            kv_v = self._nz_5d_view(self.value_cache, block_size)

            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query[:num_decode_tokens].unsqueeze(2),
                kv_k,
                kv_v,
                key_antiquant_scale=layer._c8_k_aq_scale_nz_bnsd,
                value_antiquant_scale=layer._c8_v_aq_scale_nz_bnsd,
                block_table=attn_metadata.block_tables[:num_decodes],
                actual_seq_lengths_kv=attn_metadata.seq_lens_list[:num_decodes],
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BNSD",
                scale=self.scale,
                block_size=block_size,
                antiquant_mode=0,
                key_antiquant_mode=0,
                value_antiquant_mode=0,
                inner_precise=1,
                sparse_mode=0,
            )
            output[:num_decode_tokens] = attn_out.squeeze(2)

        if attn_metadata.num_prefills > 0:
            prefill_q = query[num_decode_tokens:num_tokens]

            prefill_seq_qlen = [
                actual_seq_qlen[i] - num_decode_tokens for i in range(num_decodes, len(actual_seq_qlen))
            ]

            all_new_prefill = True
            for i in range(num_decodes, len(attn_metadata.seq_lens_list)):
                q_start = actual_seq_qlen[i - 1] if i > 0 else 0
                qlen_i = actual_seq_qlen[i] - q_start
                if attn_metadata.seq_lens_list[i] > qlen_i:
                    all_new_prefill = False
                    break

            if all_new_prefill and float_key is not None and float_value is not None:
                prefill_k = float_key[num_decode_tokens:num_tokens]
                prefill_v = float_value[num_decode_tokens:num_tokens]
                prefill_seq_kvlen = prefill_seq_qlen
            else:
                num_block, blk_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
                paged_k = self._nz_5d_view(self.key_cache, blk_size)
                paged_v = self._nz_5d_view(self.value_cache, blk_size)

                prefill_bt = attn_metadata.block_tables[num_decodes:]
                prefill_sl = attn_metadata.seq_lens_list[num_decodes:]
                prefill_k, prefill_v = self._dequant_paged_kv_to_dense(
                    paged_k, paged_v, prefill_bt, prefill_sl, query.dtype, layer
                )
                prefill_seq_kvlen = torch.tensor(prefill_sl, dtype=torch.int32).cumsum(dim=0)

            # block_table is None for prefill; FIA ignores block_size in this case.
            # Use cache block_size for consistency rather than a magic number.
            cache_block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query=prefill_q,
                key=prefill_k,
                value=prefill_v,
                atten_mask=attn_metadata.attn_mask,
                block_table=None,
                input_layout="TND",
                block_size=cache_block_size,
                actual_seq_lengths=prefill_seq_qlen,
                actual_seq_lengths_kv=prefill_seq_kvlen,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )
            n_prefill = num_tokens - num_decode_tokens
            attn_out = attn_out.view(n_prefill, self.num_heads, self.head_size)
            output[num_decode_tokens:num_tokens] = attn_out[:n_prefill]

        return output

    def _forward_c8_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ):
        """C8 FIA V1 TND for prefill states (PrefillNoCache uses float KV directly,
        PrefillCacheHit gathers + dequants paged INT8 KV).
        """
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)

        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]
        query = query[:num_tokens]

        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]

        if key.dtype == torch.int8:
            if block_table is not None:
                seq_lens = (
                    actual_seq_lengths_kv if isinstance(actual_seq_lengths_kv, list) else actual_seq_lengths_kv.tolist()
                )
                key, value = self._dequant_paged_kv_to_dense(key, value, block_table, seq_lens, query.dtype, layer)
                block_table = None
                # block_table is None after dequant; FIA ignores block_size.
                # Use cache block_size for consistency rather than a magic number.
                block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
                actual_seq_lengths_kv = torch.tensor(seq_lens, dtype=torch.int32).cumsum(dim=0)
            else:
                key = (key.to(query.dtype) - layer._c8_k_offset) * layer._c8_k_scale
                value = (value.to(query.dtype) - layer._c8_v_offset) * layer._c8_v_scale

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output
        return output

    def _reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if len(kv_cache) > 1:
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            if self.kv_sharing_target_layer_name is not None:
                # C8/NZ cache writes follow the same KV-sharing rule.
                if self.is_kv_producer:
                    attn_metadata.reshape_cache_event.record()
                return query, key, value, output
            slots = attn_metadata.slot_mapping

            encoder_decoder = self.attn_type == AttentionType.ENCODER_DECODER

            # NZ write path: 5D view + npu_scatter_pa_kv_cache
            block_size = self.vllm_config.cache_config.block_size
            k_cache_layer = self._nz_5d_view(self.key_cache, block_size)
            v_cache_layer = self._nz_5d_view(self.value_cache, block_size)

            torch_npu.npu_scatter_pa_kv_cache(
                key=key[: attn_metadata.num_actual_tokens] if not encoder_decoder else key,
                value=value[: attn_metadata.num_actual_tokens] if not encoder_decoder else value,
                key_cache=k_cache_layer,
                value_cache=v_cache_layer,
                slot_mapping=slots[: attn_metadata.num_actual_tokens] if not encoder_decoder else slots,
            )

            notify_kv_cache_written()
        return query, key, value, output


QFA_QUANT_MODE_MXFP8 = 1
QFA_MASK_MODE_CAUSAL = 3
QFA_LAYOUT_TND = "TND"
QFA_LAYOUT_PA_BBND = "PA_BBND"


def _build_qfa_cu_seqlens(cumulative_seq_lengths: list[int], device: torch.device) -> torch.Tensor:
    """Build QFA ``cu_seqlens``: int32 (B+1,) with a leading 0.

    FIA-style ``actual_seq_lengths`` are B cumulative entries; QFA expects the
    cumulative sums prefixed with 0 so batch i spans
    ``[cu_seqlens[i], cu_seqlens[i + 1])`` (QFA requirement doc, sequence
    length conversion rules).
    """
    return torch.tensor([0, *cumulative_seq_lengths], dtype=torch.int32, device=device)


# Resolved lazily and cached: (main_op, metadata_op). The QFA dual operators
# are delivered through the cann_ops_transformer package shipped with the
# CANN toolkit (confirmed final delivery form). That wrapper's call shape
# (verified on-device): positional q/k/v/q_descale/k_descale/v_descale/
# quant_mode, p_scale instead of quant_scale_p, an extra layout_q_descale,
# no pa_block_size, and a required non-null v_descale placeholder on the
# metadata call for quant_mode=1 (batch_size must not be passed with a TND
# layout_q; the op infers it from cu_seqlens_q).
# Graph capture: torch_npu's npugraph_ex backend (the mechanism this vLLM
# build uses for FULL graphs, confirmed in the on-device capture stack)
# captures the ALLOCATING wrapper directly -- internal at::empty allocations
# land in the graph's private pool and replay safely. The ops-transformer
# golden tests exercise exactly this path (GRAPH_PATH=7: torch.compile with
# backend="npugraph_ex", metadata op called INSIDE forward, no .out variant).
# No task_group/update machinery is needed on our side.
_QFA_OPS: tuple[Any, Any] | None = None


def _get_qfa_ops() -> tuple[Any, Any]:
    global _QFA_OPS
    if _QFA_OPS is None:
        try:
            from cann_ops_transformer.ops import quant_flash_attn as main_op
            from cann_ops_transformer.ops import quant_flash_attn_metadata as metadata_op
        except ImportError:
            raise RuntimeError(
                "C8_MXFP requires the QFA dual operators delivered in the "
                "cann_ops_transformer package (shipped with the CANN toolkit): "
                "cann_ops_transformer.ops.quant_flash_attn(_metadata) could not "
                "be imported in this environment."
            ) from None
        _QFA_OPS = (main_op, metadata_op)
    return _QFA_OPS


class AscendC8MXFPAttentionBackendImpl(AscendAttentionBackendImpl):
    """MXFP8 KV cache backend computed by the QFA dual-operator interface.

    forward() quantizes Q/K dynamically (``npu_dynamic_mx_quant``, FP8 E4M3 +
    per-token-group E8M0 scales) and V statically (the checkpoint's
    per-channel E8M0 scale), scatters quantized K/V plus their scale caches
    into the paged cache, and calls
    ``cann_ops_transformer.ops.quant_flash_attn_metadata`` +
    ``cann_ops_transformer.ops.quant_flash_attn`` directly on the paged cache.

    Layout: PA_BBND. K/V and both E8M0 scale caches are stored in the natural
    ``[num_blocks, block_size, num_kv_heads, head_dim]`` order that
    reshape_and_cache writes, which is exactly what QFA's PA_BBND reads -- so
    no layer ever transposes or copies the cache (the per-step
    ``transpose(1,2).contiguous()`` full-cache copy of the FIA-based design is
    gone; validated on-device by the vendored-QFA bring-up on the same
    operator).

    One QFA call per step: decode and prefill requests share a single
    CAUSAL-masked invocation (cu_seqlens_q over the whole batch), instead of
    per-subset calls -- the causal mask already covers decode rows and the
    batch is smaller to feed. PrefillNoCache also reads from pages:
    reshape_and_cache has written this step's K/V before attention runs.

    Graph capture: handled natively by torch_npu's npugraph_ex backend (the
    FULL-graph mechanism of this vLLM build), following the ops-transformer
    golden-test methodology (GRAPH_PATH=7). The allocating wrapper is
    captured directly (at::empty outputs land in the graph pool) and the
    AICPU metadata op runs inline inside the graph. Replay safety relies on
    every per-step input being a stable-address tensor whose content is
    refreshed outside Python: block_table / slot_mapping come from the
    model runner's persistent CpuGpuBuffer storages, and cu_seqlens_q /
    seqused_kv are derived IN-GRAPH from the runner's persistent
    query_start_loc / seq_lens buffers (captured clamp/cummax ops re-execute
    each replay). The K-scale scatter uses a device-side mask pattern (no
    host sync). No Python-side buffer refresh exists in the captured
    region -- ACL-graph replay never re-runs Python, so such refreshes
    would freeze at capture values. Speculative decoding remains out of
    scope for v1.

    NOTE: the QFA dual operators are called through _get_qfa_ops(), which
    resolves cann_ops_transformer.ops.quant_flash_attn(_metadata) -- the
    confirmed final delivery form, shipped with the CANN toolkit. That
    wrapper's signature (verified on-device via inspect + the vendored-QFA
    bring-up) differs from the requirement doc's torch_npu example: positional
    q_descale/k_descale/v_descale/quant_mode, p_scale instead of
    quant_scale_p, an extra layout_q_descale, no pa_block_size, and a
    required v_descale placeholder on the metadata call for quant_mode=1.
    """

    # Installed via ``layer.impl.__class__`` assignment, which does not call
    # this subclass's constructor. Class-level defaults are therefore
    # required for objects that predate the class swap.
    enable_hamming_sparse: bool = False
    _v_scale_filled_caches: set[torch.Tensor] | None = None

    def _qfa_step_cache(self, attn_metadata: AscendMetadata) -> dict:
        cache = getattr(attn_metadata, "qfa_metadata_cache", None)
        if cache is None:
            cache = {}
            attn_metadata.qfa_metadata_cache = cache
        return cache

    def _get_qfa_metadata(
        self,
        attn_metadata: AscendMetadata,
        *,
        cu_seqlens_q: torch.Tensor,
        seqused_kv: torch.Tensor,
        max_seqlen_q: int,
    ):
        """Return the QFA metadata plan (AICPU op output).

        Eager: computed once per step and cached on the AscendMetadata (the
        plan depends on heads/batch, identical across full-attention layers).
        npugraph_ex capture: the Python-side cache is bypassed so the metadata
        op executes INSIDE the captured region on every layer visit -- the
        graph then recomputes the plan from the replayed length buffers each
        step (golden-test GRAPH_PATH=7 methodology: Network.forward calls the
        metadata op inline before the main op). The per-layer redundant calls
        during capture are free (AICPU, ~us) and each layer's captured call
        consumes the plan it just produced.
        """
        # Bypass the cache during ANY graph capture (FULL and PIECEWISE
        # warmups alike): the metadata op must execute inside the captured
        # region so each replay recomputes the plan from the replayed
        # length buffers. During PIECEWISE capture the attention runs
        # eagerly between the compiled pieces, so the uncached per-layer
        # calls are just normal eager executions (~us, AICPU).
        use_step_cache = not _EXTRA_CTX.capturing
        cache = self._qfa_step_cache(attn_metadata) if use_step_cache else {}
        metadata = cache.get("step")
        if metadata is None:
            # TND + PA: pass cu_seqlens_q only; the KV side is addressed via
            # block_table + seqused_kv (QFA requirement doc, 3.2.3).
            # quant_mode=1 refuses a null v_descale at the aclnn entry
            # (quant_flash_attn_metadata_check.h), but under PA_BBND nothing
            # reads it -- a minimal 5D E8M0 placeholder suffices. batch_size
            # must NOT be passed with a TND layout_q (the checker rejects
            # it); the op infers it from cu_seqlens_q.
            _, metadata_op = _get_qfa_ops()
            # NOTE: torch_npu.float8_e8m0fnu is the integer dtype ID (293) on
            # this torch_npu build, not a torch.dtype; tensor.view() would
            # parse it as a target shape. Bitcast with the stock torch dtype
            # instead (itemsize 1 -> 1, shape preserved).
            v_descale_stub = torch.zeros(
                1, 1, 1, 1, 2, dtype=torch.uint8, device=cu_seqlens_q.device
            ).view(torch.float8_e8m0fnu)
            metadata = metadata_op(
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                QFA_QUANT_MODE_MXFP8,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=None,
                seqused_q=None,
                seqused_kv=seqused_kv,
                v_descale=v_descale_stub,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=-1,
                mask_mode=QFA_MASK_MODE_CAUSAL,
                win_left=-1,
                win_right=-1,
                layout_q=QFA_LAYOUT_TND,
                layout_q_descale=QFA_LAYOUT_TND,
                layout_kv=QFA_LAYOUT_PA_BBND,
                layout_out=QFA_LAYOUT_TND,
            )
            if use_step_cache:
                cache["step"] = metadata
        return metadata

    def _qfa_int8_mask(self, attn_metadata: AscendMetadata) -> torch.Tensor | None:
        """QFA's attn_mask is INT8/UINT8/bool; the shared builder already
        hands out an int8 2048x2048 causal mask, so only convert when some
        other mask source slips in."""
        if attn_metadata.attn_mask is None:
            return None
        if attn_metadata.attn_mask.dtype == torch.int8:
            return attn_metadata.attn_mask
        return attn_metadata.attn_mask.to(torch.int8)

    def _run_qfa(
        self,
        quant_query: torch.Tensor,
        query_scale: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        *,
        cu_seqlens_q: torch.Tensor,
        seqused_kv: torch.Tensor,
        qfa_metadata,
        max_seqlen_q: int,
        num_tokens: int,
        output: torch.Tensor,
    ) -> torch.Tensor:
        key, value, key_scale, value_scale = kv_cache
        # The scale caches are stored as raw uint8 (index_put_ on float8
        # either errors or falls back to AICPU); QFA's checker wants E8M0, so
        # bitcast at the call boundary (torch.float8_e8m0fnu -- the torch
        # dtype; torch_npu.float8_e8m0fnu is the integer ID 293 on this
        # build and would be parsed as a view *shape*). Same for the q scale
        # when the quant helper returns it as uint8 bytes.
        if key_scale.dtype != torch.float8_e8m0fnu:
            key_scale = key_scale.view(torch.float8_e8m0fnu)
        if value_scale.dtype != torch.float8_e8m0fnu:
            value_scale = value_scale.view(torch.float8_e8m0fnu)
        if query_scale.dtype != torch.float8_e8m0fnu:
            query_scale = query_scale.view(torch.float8_e8m0fnu)
        main_op, _ = _get_qfa_ops()
        # cann_ops_transformer delivery signature (verified on-device):
        # q/k/v/q_descale/k_descale/v_descale/quant_mode positional, p_scale
        # instead of quant_scale_p, layout_q_descale, and no pa_block_size
        # (the op infers the block size from the k/v cache shapes).
        # The allocating wrapper is capture-safe under npugraph_ex (internal
        # at::empty allocations land in the graph pool); the ops-transformer
        # golden tests capture exactly this call (GRAPH_PATH=7).
        result = main_op(
            quant_query,
            key,
            value,
            query_scale,
            key_scale,
            value_scale,
            QFA_QUANT_MODE_MXFP8,
            block_table=attn_metadata.block_tables,
            p_scale=None,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=None,
            seqused_q=None,
            seqused_kv=seqused_kv,
            sinks=None,
            attn_mask=self._qfa_int8_mask(attn_metadata),
            metadata=qfa_metadata,
            softmax_scale=self.scale,
            mask_mode=QFA_MASK_MODE_CAUSAL,
            win_left=-1,
            win_right=-1,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=-1,
            layout_q=QFA_LAYOUT_TND,
            layout_q_descale=QFA_LAYOUT_TND,
            layout_kv=QFA_LAYOUT_PA_BBND,
            layout_out=QFA_LAYOUT_TND,
            return_softmax_lse=False,
        )
        # return_softmax_lse=False yields an empty LSE tensor in the
        # cann_ops flavor; tolerate both tuple and single-tensor returns.
        attn_output = result[0] if isinstance(result, tuple) else result
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output
        return output

    def _forward_mxfp8_attention(
        self,
        quant_query: torch.Tensor,
        query_scale: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """One QFA call per step for the whole batch (decode + prefill alike).

        The causal mask covers decode rows as well, so the per-subset split of
        the FIA-based design is unnecessary. PrefillNoCache is included:
        reshape_and_cache has already written this step's K/V into the paged
        cache before attention runs (single-call design validated on-device
        by the vendored-QFA bring-up).

        npugraph_ex capture compatibility (golden-test methodology,
        GRAPH_PATH=7): cu_seqlens_q / seqused_kv are derived ON DEVICE from
        the model runner's persistent int32 buffers (query_start_loc.gpu /
        seq_lens), which _prepare_inputs refreshes in place every step.
        The deriving ops are captured together with the QFA calls, so every
        replay re-executes them and the operators always see the current
        step's lengths. This replaces the former Python-side staging
        writes: ACL-graph replay never re-runs Python, so those refreshes
        only executed during capture and froze the buffers at capture
        values (and the pinned staging buffer itself was racy under the
        async scheduler, where the host could overwrite it while the
        previous step's async H2D copy was still in flight).
        """
        if not attn_metadata.causal:
            raise NotImplementedError("C8_MXFP attention does not support non-causal attention yet.")
        if self.sliding_window is not None:
            raise NotImplementedError("C8_MXFP attention does not support sliding window attention yet.")

        # T as QFA sees it: the query rows actually fed to the operator
        # (forward slices query[:num_actual_tokens] before quantizing), so
        # the TND constraint cu_seqlens_q[-1] == T, the view/output row
        # counts and the sanitize clamp bound below all share one source.
        # At graph capture this is the bucket size; eagerly it is the real
        # token count -- both stay clean where actual_seq_lengths_q[-1]
        # may carry stale tail entries (padded batches).
        num_tokens = quant_query.shape[0]
        if num_tokens <= 0:
            return output

        qsl_gpu = attn_metadata.query_start_loc_gpu
        seq_lens_gpu = attn_metadata.seq_lens_gpu
        if qsl_gpu is None or seq_lens_gpu is None:
            raise RuntimeError(
                "C8_MXFP attention requires the GPU-side length sources "
                "(query_start_loc_gpu / seq_lens_gpu) on AscendMetadata."
            )
        # Sanitize the tail beyond the current requests: unused
        # query_start_loc slots carry -1 (the FIA padding convention) and
        # may also hold stale entries from larger earlier steps (the FULL
        # dummy-request padding re-copies the whole CPU buffer to GPU).
        # clamp to [0, num_tokens] bounds both; cummax restores
        # monotonicity, turning the tail into zero-length requests whose
        # cu_seqlens_q[-1] still equals the token total. Unused seq_lens
        # slots are zero-filled by the runner every step; clamp(min=1)
        # matches the dummy-request convention (block 0, one token). On
        # clean eager data both ops are identity transforms.
        cu_seqlens_q = qsl_gpu.clamp(min=0, max=num_tokens).cummax(dim=0).values
        seqused_kv = seq_lens_gpu.clamp(min=1)
        # Upper bound on any single query length (the step's total token
        # count); the vendored-QFA bring-up measured prefill against a constant
        # max_model_len bound as safe too, so a loose bound only affects
        # tiling, not correctness.
        max_seqlen_q = num_tokens

        qfa_metadata = self._get_qfa_metadata(
            attn_metadata,
            cu_seqlens_q=cu_seqlens_q,
            seqused_kv=seqused_kv,
            max_seqlen_q=max_seqlen_q,
        )
        return self._run_qfa(
            quant_query,
            query_scale,
            kv_cache,
            attn_metadata,
            cu_seqlens_q=cu_seqlens_q,
            seqused_kv=seqused_kv,
            qfa_metadata=qfa_metadata,
            max_seqlen_q=max_seqlen_q,
            num_tokens=num_tokens,
            output=output,
        )

    # KV cache writes for C8_MXFP happen in reshape_and_cache(), invoked from forward()
    # when key/value are present. This hook is only reached when attention is split from
    # cache update, e.g. Attention.forward with forward_includes_kv_cache_update=False
    # (unified_kv_cache_update -> do_kv_cache_update). AscendAttentionBackend keeps
    # forward_includes_kv_cache_update=True, so normal inference never calls this.
    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: list[torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> None:
        raise NotImplementedError(
            "C8_MXFP KV cache update is only supported via reshape_and_cache in forward()."
        )

    def reshape_and_cache(
        self,
        quant_key: torch.Tensor,
        quant_value: torch.Tensor,
        key_scale: torch.Tensor,
        value_scale: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
    ) -> None:
        num_actual_tokens = quant_key.shape[0]
        slot_mapping = attn_metadata.slot_mapping[:num_actual_tokens]
        key_cache, value_cache = kv_cache[0], kv_cache[1]
        DeviceOperator.reshape_and_cache(
            key=quant_key,
            value=quant_value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_mapping=slot_mapping,
        )

        # Scatter the dynamic per-token K scales; V's static per-channel scale
        # is broadcast into its group-layout cache once per cache instance
        # (tracked by identity so memory-profiling dummy caches and the real
        # cache are both initialized without a reset hook).
        key_scale_cache, value_scale_cache = kv_cache[2], kv_cache[3]
        scatter_mxfp_k_scale_cache(
            # Byte view: index_put_ on float8 either errors or falls back to
            # AICPU (the cache side is already uint8 raw storage).
            key_scale.view(torch.uint8) if key_scale.dtype != torch.uint8 else key_scale,
            key_scale_cache,
            slot_mapping,
            key_cache.shape[1],
        )
        filled_caches = self._v_scale_filled_caches
        if filled_caches is None:
            filled_caches = set()
            self._v_scale_filled_caches = filled_caches
        if value_scale_cache not in filled_caches:
            # (hidden_size) -> (num_kv_heads, head_size) -> broadcast ->
            # (num_blocks, block_size // 64, num_kv_heads, head_size, 2)
            # (PA_BBND). Derive num_kv_heads / v head_dim from the cache
            # layout instead of self.num_kv_heads / self.head_size so models
            # whose V head dim differs from the Q/K head dim stay correct.
            num_kv_heads = value_scale_cache.shape[2]
            v_head_dim = value_scale_cache.shape[3]
            value_scale_cache.copy_(value_scale.view(1, 1, num_kv_heads, v_head_dim, 1))
            filled_caches.add(value_scale_cache)

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for AscendC8MXFPAttentionBackendImpl"
            )
        if attn_metadata is None:
            return output.fill_(0)
        if getattr(self, "enable_hamming_sparse", False):
            raise NotImplementedError("C8_MXFP attention does not support hamming sparse KV compression yet.")
        if self.vllm_config.speculative_config is not None:
            raise NotImplementedError("C8_MXFP v1 does not support speculative decoding yet.")
        if self.vllm_config.kv_transfer_config is not None:
            raise NotImplementedError("C8_MXFP v1 does not support PD disaggregation (kv_transfer) yet.")
        if _EXTRA_CTX.capturing:
            # Graph capture is handled natively by npugraph_ex (the backend
            # this vLLM build uses for FULL graphs -- confirmed in the
            # on-device capture stack): it captures the ALLOCATING QFA
            # wrapper directly (internal at::empty lands in the graph pool),
            # and the ops-transformer golden tests exercise exactly this
            # (GRAPH_PATH=7: torch.compile(backend="npugraph_ex") with the
            # metadata op called inline in forward). The requirements on our
            # side are only graph-safety of the surrounding Python: no host
            # syncs (scatter uses the device-side mask pattern) and no
            # Python-side refresh of tensors the captured region consumes
            # (replay never re-runs Python; the per-step lengths are derived
            # in-graph from the runner's persistent buffers instead).
            pass
        if kv_cache is None or len(kv_cache) < 4:
            raise RuntimeError(
                "C8_MXFP attention requires a (k, v, k_scale, v_scale) KV cache "
                f"tuple, got: {type(kv_cache)} with length "
                f"{len(kv_cache) if kv_cache is not None else 0}."
            )

        record_attention_compute_start()

        query_mxfp8, query_scale = torch_npu.npu_dynamic_mx_quant(
            query[: attn_metadata.num_actual_tokens],
            dst_type=torch.float8_e4m3fn,
        )

        # KV-sharing consumer layers reuse another layer's cache; writing
        # their (dummy) K/V would corrupt the shared slots, so only the
        # owner layer quantizes and scatters K/V. key/value may also be None
        # on pure decode paths.
        if key is not None and value is not None and self.kv_sharing_target_layer_name is None:
            key_mxfp8, key_scale = torch_npu.npu_dynamic_mx_quant(
                key[: attn_metadata.num_actual_tokens],
                dst_type=torch.float8_e4m3fn,
            )

            original_value_shape = value.shape
            value = value.view(original_value_shape[0], -1)
            value_mxfp8 = torch_npu.npu_quantize(
                value[: attn_metadata.num_actual_tokens],
                layer.v_cache_scale_float_reciprocal,
                None,
                torch.float8_e4m3fn,
                -1,
                False,
            )
            value_mxfp8 = value_mxfp8.view((attn_metadata.num_actual_tokens, *original_value_shape[1:]))

            self.reshape_and_cache(
                key_mxfp8, value_mxfp8, key_scale, layer.v_cache_scale, kv_cache, attn_metadata
            )

        # PA_BBND: QFA reads the paged cache in the order it is stored, so the
        # cache tuple is passed through as-is -- no transpose, no copy.
        return self._forward_mxfp8_attention(query_mxfp8, query_scale, kv_cache, attn_metadata, output)
