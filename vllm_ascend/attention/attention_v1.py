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

from dataclasses import dataclass
from enum import Enum
from typing import Literal, NamedTuple

import torch
import torch_npu
import vllm.envs as envs_vllm
from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
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
from vllm_ascend.attention.context_parallel.common_cp import AscendMetadataForDecode, AscendMetadataForPrefill
from vllm_ascend.attention.kvcomp_attn.attention_utils import (
    get_kvcomp_decode_params,
    is_enable_hamming_sparse,
    reshape_and_cache_kvcomp,
)
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    enable_cp,
    notify_kv_cache_written,
    split_decodes_and_prefills,
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
from vllm_ascend.ops.flashcomm2_oshard_manager import flashcomm2_oshard_manager
from vllm_ascend.utils import weak_ref_tensors

from vllm_ascend.worker.kvcomp_utils import KVCompMetaData

# default max value of sliding window size
SWA_INT_MAX = 2147483647
_ATTN_KEYS_BUFFER = None

# Ascend FIA TND currently supports these head dimensions on the vLLM-Ascend
# path. Larger heterogeneous-head models need a prefill fallback to avoid
# unsupported-kernel behavior.
# A5 (910B) supports head_dim=512 natively via FIA v2.
# A2/A3 (TND) only supports up to head_dim=256; 512 triggers the
# large-head fallback path.
FIA_TND_SUPPORTED_HEAD_SIZES = {64, 128, 192, 256, 512}

# Global counter for assigning unique per-layer indices to
# AscendAttentionBackendImpl instances during graph construction.
_next_impl_index = 0

GraphParamKind = Literal["paged_attention", "fia"]


class AttentionGraphParam(NamedTuple):
    """Captured attention graph metadata.

    `kind` records which attention op was captured, and `layer_name` binds the
    captured params back to the real attention layer during graph replay. This
    avoids inferring op type from tuple length or relying on metadata dict order.
    """

    kind: GraphParamKind
    params: tuple
    layer_name: str | None


def _normalize_graph_param(param: AttentionGraphParam, fallback_layer_name: str) -> tuple[GraphParamKind, tuple, str]:
    if not isinstance(param, AttentionGraphParam):
        raise TypeError(f"Expected AttentionGraphParam, got {type(param).__name__}")
    return param.kind, param.params, param.layer_name or fallback_layer_name


def _get_graph_param_kind(param: AttentionGraphParam) -> GraphParamKind:
    kind, _, _ = _normalize_graph_param(param, "")
    return kind


def _uses_sliding_window_attention(vllm_config: VllmConfig) -> bool:
    return getattr(vllm_config.model_config.hf_text_config, "sliding_window", None) is not None


@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.
        return "CUSTOM" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AscendAttentionBackendImpl"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPImpl

            return AscendAttentionCPImpl
        return AscendAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPMetadataBuilder

            return AscendAttentionCPMetadataBuilder
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: list[torch.Tensor],
        dst_kv_cache: list[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            key_caches = kv_cache[0]
            value_caches = kv_cache[1]
            key_caches[dst_indices] = key_caches[src_indices]
            value_caches[dst_indices] = value_caches[src_indices]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


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
    num_actual_tokens_pcp_padded: int = 0
    num_actual_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0
    num_decodes_flatten: int = 0

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
    # pcp
    prefill: AscendMetadataForPrefill | None = None
    # dcp
    decode_meta: AscendMetadataForDecode | None = None

    causal: bool = True
    # runner_type in model_config.
    model_runner_type: str = ""
    # prefill reshape_and_cache event
    reshape_cache_event: torch.npu.Event = None

    kvcomp_metadata: KVCompMetaData | None = None


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

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.decode_threshold
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
        attn_mask = self.attn_mask_builder.get_attention_mask(self.model_config)

        # TODO: Yet another unnecessary H2D while we already have a query_start_loc on device
        query_start_loc = query_start_loc_cpu.pin_memory().to(self.device, non_blocking=True)

        attn_metadata = AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            seq_lens_list=seq_lens.tolist(),
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            attn_state=attn_state,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            causal=common_attn_metadata.causal,
            model_runner_type=self.model_config.runner_type,
            kvcomp_metadata=common_attn_metadata.kvcomp_metadata,
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
        self.sinks = sinks
        self.layerIndex = 0
        self.enable_hamming_sparse = is_enable_hamming_sparse()
        self._layer_name: str | None = None
        global _next_impl_index
        self._impl_idx = _next_impl_index
        _next_impl_index += 1

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        num_dcp_pcp_tokens=None,
        draft_attn_metadatas=None,
    ):
        if _EXTRA_CTX.is_draft_model:
            if _EXTRA_CTX.is_draft_model_prefill:
                graph_params = get_draft_graph_prefill_params()
            else:
                graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()

        # Use key-based lookup (attn_params_by_key) when available.
        # Falls back to zip-based pairing for backwards compatibility.
        params_by_key = graph_params.attn_params_by_key.get(num_tokens, {})

        if _EXTRA_CTX.is_draft_model:
            attn_metadata = draft_attn_metadatas
            attn_keys = list(attn_metadata[0].keys())
        else:
            attn_metadata = forward_context.attn_metadata
            attn_keys = list(attn_metadata.keys())
            # Sort attn_keys by layer index for deterministic order
            attn_keys_length = len(graph_params.attn_params.get(num_tokens, []))
            if attn_keys_length > 0:
                global _ATTN_KEYS_BUFFER
                if _ATTN_KEYS_BUFFER is None:
                    import regex as re

                    def extract_layer_index(key: str) -> int:
                        match = re.search(r"(\d+)", key)
                        return int(match.group(1)) if match else 0

                    attn_keys_tmp = attn_keys[:attn_keys_length]
                    attn_keys_tmp.sort(key=extract_layer_index)
                    _ATTN_KEYS_BUFFER = attn_keys_tmp
                attn_keys[:attn_keys_length] = _ATTN_KEYS_BUFFER

        num_layers = len(attn_keys)
        if num_layers == 0:
            return

        # Key-based lookup is only valid for the target model.
        # For the draft model, each layer appears multiple times in the
        # captured graph (once per spec step), so params_by_key (which
        # stores only the last write per layer_name) would miss all but
        # the final step's task groups, leaving the rest with stale
        # capture-time parameters.  Force zip-based pairing for draft.
        use_key_lookup = (len(params_by_key) > 0
                          and not _EXTRA_CTX.is_draft_model)

        if _EXTRA_CTX.is_draft_model:
            # Zip-based: multiply keys to match captured count (one
            # task group per layer per spec step).
            captured_param_count = len(graph_params.attn_params.get(num_tokens, []))
            if captured_param_count > 0:
                attn_keys = attn_keys * (captured_param_count // num_layers)
            _iter_items = [(None, k) for k in attn_keys]
        else:
            # Target model: filter to keys present in params_by_key
            if use_key_lookup:
                attn_keys = [k for k in attn_keys if k in params_by_key]
            _iter_items = [(None, k) for k in attn_keys]

        attn_count = 0
        with torch.npu.stream(update_stream):
            for _step_idx, key in _iter_items:
                if use_key_lookup:
                    # ---- key-based lookup (order-independent) ----
                    param_info = params_by_key[key]
                    # Skip task-group-free large_head layers: these capture
                    # padded metadata tensor addresses directly and read
                    # up-to-date content on each replay.
                    if param_info.get("large_head"):
                        continue
                    param_tuple = param_info["params"]
                    handle = param_info["handle"]
                    event = param_info["event"]

                    if isinstance(param_tuple, AttentionGraphParam):
                        param_kind = param_tuple.kind
                        _, params, layer_name = _normalize_graph_param(param_tuple, key)
                    else:
                        param_kind = "paged_attention" if len(param_tuple) == 9 else "fia"
                        layer_name = key
                        params = param_tuple
                else:
                    # ---- fallback: zip-based (original behaviour) ----
                    param_idx = attn_count % max(len(graph_params.attn_params.get(num_tokens, [1])), 1)
                    param_tuple = graph_params.attn_params[num_tokens][param_idx]
                    handle = graph_params.handles[num_tokens][param_idx]
                    event = graph_params.events[num_tokens][param_idx]

                    if isinstance(param_tuple, AttentionGraphParam):
                        param_kind, params, layer_name = _normalize_graph_param(param_tuple, key)
                    else:
                        param_kind = "paged_attention" if len(param_tuple) == 9 else "fia"
                        layer_name = key
                        params = param_tuple

                # ---- dispatch based on param kind ----
                if param_kind == "paged_attention":
                    if handle is None:
                        event.record(update_stream)
                        continue
                    (
                        query, key_cache, value_cache,
                        num_kv_heads, num_heads, scale,
                        block_table, seq_lens, output,
                    ) = params
                    draft_step = _step_idx if _step_idx is not None else (attn_count // num_layers)
                    if _EXTRA_CTX.is_draft_model:
                        block_table = attn_metadata[draft_step][key].block_tables
                        seq_lens = attn_metadata[draft_step][key].seq_lens
                    else:
                        metadata_key = layer_name if layer_name in attn_metadata else key
                        current_attn_metadata = attn_metadata[metadata_key]
                        block_table = current_attn_metadata.block_tables
                        seq_lens = current_attn_metadata.seq_lens
                    workspace = torch_npu._npu_paged_attention_get_workspace(
                        query=query, key_cache=key_cache, value_cache=value_cache,
                        num_kv_heads=num_kv_heads, num_heads=num_heads,
                        scale_value=scale, block_table=block_table,
                        context_lens=seq_lens, out=output,
                    )
                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu._npu_paged_attention(
                        query=query, key_cache=key_cache, value_cache=value_cache,
                        num_kv_heads=num_kv_heads, num_heads=num_heads,
                        scale_value=scale, block_table=block_table,
                        context_lens=seq_lens, out=output, workspace=workspace,
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)

                elif _EXTRA_CTX.sinks:
                    # FIA-V2 replay (sinks)
                    (
                        query, key_cache, value, block_tables, attn_mask,
                        block_size, seq_lens, num_kv_heads, num_heads, scale,
                        sliding_window, sinks, attn_output, softmax_lse,
                    ) = params[:14]
                    draft_step = _step_idx if _step_idx is not None else (attn_count // num_layers)
                    if _EXTRA_CTX.is_draft_model:
                        seq_lens = attn_metadata[draft_step][key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[draft_step][key].actual_seq_lengths_q
                    else:
                        seq_lens = attn_metadata[key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[key].actual_seq_lengths_q
                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu.npu_fused_infer_attention_score_v2.out(
                        query=query, key=key_cache, value=value,
                        block_table=block_tables, atten_mask=attn_mask,
                        input_layout="TND", block_size=block_size,
                        actual_seq_qlen=actual_seq_lengths_q,
                        actual_seq_kvlen=seq_lens,
                        num_key_value_heads=num_kv_heads,
                        num_query_heads=num_heads,
                        sparse_mode=4 if sliding_window is not None else 3,
                        pre_tokens=sliding_window if sliding_window is not None else SWA_INT_MAX,
                        next_tokens=0, softmax_scale=scale,
                        learnable_sink=sinks,
                        workspace=graph_params.workspaces.get(num_tokens),
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)

                else:
                    # FIA replay (no sinks)
                    (
                        query, key_cache, value, block_tables, attn_mask,
                        block_size, seq_lens, actual_seq_lengths_q,
                        num_kv_heads, num_heads, scale, attn_output, softmax_lse,
                        sparse_mode, pre_tokens, next_tokens,
                        c8_k_aq_scale, c8_k_aq_offset,
                        c8_v_aq_scale, c8_v_aq_offset,
                    ) = params[:21]
                    draft_step = _step_idx if _step_idx is not None else (attn_count // num_layers)
                    if _EXTRA_CTX.is_draft_model:
                        seq_lens = attn_metadata[draft_step][key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[draft_step][key].actual_seq_lengths_q
                    else:
                        seq_lens = attn_metadata[key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[key].actual_seq_lengths_q
                    import sys
                    _head_dim = query.shape[2] if query.dim() >= 3 else 0
                    _ws = graph_params.workspaces.get(num_tokens)
                    # FIA (both V1 and V2) crashes on Ascend 910B4 when
                    # num_tokens > 1 AND sparse_mode=4 (sliding window)
                    # AND head_dim > 128.  These layers use
                    # forward_paged_attention during capture.  Use
                    # _npu_paged_attention for the update to match.
                    # NOTE: In FULL_DECODE_ONLY mode, FIA IS used
                    # for sliding window layers (head_dim=256) during
                    # capture (PA fails during FDO capture).  The
                    # update must match — skip the PA fallback during
                    # FDO replay.
                    use_pa = (num_tokens > 1 and sparse_mode == 4
                              and _head_dim > 128
                              and get_forward_context().cudagraph_runtime_mode
                              != CUDAGraphMode.FULL)
                    if use_pa:
                        # _npu_paged_attention expects tensors for
                        # context_lens.  seq_lens from attn_metadata
                        # may be a list.
                        _pa_seq_lens = torch.tensor(
                            seq_lens, dtype=torch.int32, device=query.device
                        ) if not isinstance(seq_lens, torch.Tensor) else seq_lens
                        _pa_workspace = torch_npu._npu_paged_attention_get_workspace(
                            query=query,
                            key_cache=key_cache,
                            value_cache=value,
                            num_kv_heads=num_kv_heads,
                            num_heads=num_heads,
                            scale_value=scale,
                            block_table=block_tables,
                            context_lens=_pa_seq_lens,
                            out=attn_output,
                        )
                        torch.npu.graph_task_update_begin(update_stream, handle)
                        torch_npu._npu_paged_attention(
                            query=query,
                            key_cache=key_cache,
                            value_cache=value,
                            num_kv_heads=num_kv_heads,
                            num_heads=num_heads,
                            scale_value=scale,
                            block_table=block_tables,
                            context_lens=_pa_seq_lens,
                            out=attn_output,
                            workspace=_pa_workspace,
                        )
                        torch.npu.graph_task_update_end(update_stream)
                        event.record(update_stream)
                        attn_count += 1
                        continue

                    input_layout = "TND"
                    extra_args = {}
                    # Detect c8_quant from captured params (static method, no self)
                    if c8_k_aq_scale is not None:
                        extra_args = {
                            "key_antiquant_scale": c8_k_aq_scale,
                            "key_antiquant_offset": c8_k_aq_offset,
                            "value_antiquant_scale": c8_v_aq_scale,
                            "value_antiquant_offset": c8_v_aq_offset,
                            "key_antiquant_mode": 0,
                            "value_antiquant_mode": 0,
                        }
                        input_layout = "BNSD"
                        sparse_mode = 0
                    _ws = graph_params.workspaces.get(num_tokens)
                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu.npu_fused_infer_attention_score.out(
                        query=query, key=key_cache, value=value,
                        block_table=block_tables, atten_mask=attn_mask,
                        input_layout=input_layout, block_size=block_size,
                        actual_seq_lengths=actual_seq_lengths_q,
                        actual_seq_lengths_kv=seq_lens,
                        num_key_value_heads=num_kv_heads, num_heads=num_heads,
                        scale=scale, sparse_mode=sparse_mode,
                        pre_tokens=pre_tokens, next_tokens=next_tokens,
                        **extra_args,
                        workspace=_ws,
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)
                attn_count += 1


    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        if flashcomm2_oshard_manager.flashcomm2_oshard_enable():
            flashcomm2_oshard_manager.post_process_after_loading()

    def full_graph_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer=None,
    ) -> torch.Tensor:
        passed_key = key
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)
        if self.enable_hamming_sparse and attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
            reshape_and_cache_kvcomp(attn_metadata.kvcomp_metadata, self.layerIndex, passed_key)
        elif self.enable_hamming_sparse:
            block_table, actual_seq_lengths_kv = get_kvcomp_decode_params(
                self.layerIndex, attn_metadata.kvcomp_metadata, query, passed_key, block_table, actual_seq_lengths_kv
            )

        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        if _EXTRA_CTX.is_draft_model:
            if _EXTRA_CTX.is_draft_model_prefill:
                graph_params = get_draft_graph_prefill_params()
            else:
                graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        # Prepare tensors for attention output
        # TODO: Refactor this to step-level instead of layer-level

        # Get workspace from cache or calculate it if not present.
        workspace = graph_params.workspaces.get(num_tokens)
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        input_layout = "TND"
        attn_mask = attn_metadata.attn_mask
        sparse_mode = 4 if self.sliding_window else 3 if attn_metadata.causal else 0
        pre_tokens = self.sliding_window or SWA_INT_MAX
        next_tokens = 0 if self.sliding_window else SWA_INT_MAX

        extra_args = {}
        if self.enable_c8_quant:
            extra_args = {
                "key_antiquant_scale": layer._c8_k_aq_scale,
                "key_antiquant_offset": layer._c8_k_aq_offset,
                "value_antiquant_scale": layer._c8_v_aq_scale,
                "value_antiquant_offset": layer._c8_v_aq_offset,
                "key_antiquant_mode": 0,
                "value_antiquant_mode": 0,
            }
            # TODO: Convert kvcache to NZ, and change layerout from BNSD to TND.
            input_layout = "BNSD"
            query = query.unsqueeze(2)
            output = output.unsqueeze(2)
            attn_mask = None
            sparse_mode = 0
        if workspace is None:
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
        # Record the owning layer so graph replay can refresh metadata by the
        # real attention layer instead of relying on dict iteration order.
        layer_name = layer.layer_name if layer is not None else self._layer_name
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
        )
        if self.enable_c8_quant:
            attn_params = attn_params + (
                weak_ref_tensors(layer._c8_k_aq_scale),
                weak_ref_tensors(layer._c8_k_aq_offset),
                weak_ref_tensors(layer._c8_v_aq_scale),
                weak_ref_tensors(layer._c8_v_aq_offset),
            )  # type: ignore
        else:
            attn_params = attn_params + (None, None, None, None)  # type: ignore
        graph_params.attn_params[num_tokens].append(AttentionGraphParam("fia", attn_params, layer_name))

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

        # Store by key for order-independent replay lookup.
        # Skip for draft model: each layer appears multiple times
        # (once per spec step), so by-name storage would lose all
        # but the last step's task group.
        if (not _EXTRA_CTX.is_draft_model
                and layer_name
                and num_tokens in graph_params.attn_params_by_key):
            graph_params.attn_params_by_key[num_tokens][layer_name] = {
                "params": graph_params.attn_params[num_tokens][-1],
                "handle": handle,
                "event": event,
            }

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
        workspace = graph_params.workspaces.get(num_tokens)
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        if workspace is None:
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
            )
        )
        global _FIA_CAPTURE_COUNT
        _FIA_CAPTURE_COUNT += 1
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
        if _EXTRA_CTX.is_draft_model:
            graph_params = get_draft_graph_params()
        else:
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
                AttentionGraphParam(
                    "paged_attention",
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
                    self._layer_name,
                )
            )

            # PIECEWISE: PA with task groups works correctly.
            # FULL_DECODE_ONLY: PA fails both with and without task groups
            # (Inner error during capture, setup failed during replay).
            # Use PyTorch SDPA which decomposes into capturable ops.
            _is_fdo = (
                get_forward_context().cudagraph_runtime_mode
                == CUDAGraphMode.FULL
            )
            if _is_fdo:
                import torch.nn.functional as F
                # Gather KV from paged cache → dense [num_tokens, heads, dim]
                _block_size = self.key_cache.shape[1]
                _num_kv_heads = self.num_kv_heads
                _num_heads = self.num_heads
                _head_dim = self.head_size
                # Use block_table to gather: [num_reqs, max_blocks] → flat
                _bt = attn_metadata.block_tables.long()
                _num_reqs = _bt.shape[0]
                _max_blocks = _bt.shape[1]
                _flat_ids = _bt.reshape(-1)
                _k = self.key_cache.index_select(0, _flat_ids)
                _k = _k.view(_num_reqs, _max_blocks, _block_size, _num_kv_heads, _head_dim)
                _k = _k.permute(0, 3, 1, 2, 4).reshape(_num_reqs, _num_kv_heads, _max_blocks * _block_size, _head_dim)
                # Slice to actual seq lens
                _max_len = int(_bt.shape[1] * _block_size)
                _k = _k[:, :, :_max_len, :].reshape(_num_reqs * _max_len, _num_kv_heads, _head_dim)[:num_tokens]
                _v = self.value_cache.index_select(0, _flat_ids)
                _v = _v.view(_num_reqs, _max_blocks, _block_size, _num_kv_heads, _head_dim)
                _v = _v.permute(0, 3, 1, 2, 4).reshape(_num_reqs, _num_kv_heads, _max_blocks * _block_size, _head_dim)
                _v = _v[:, :, :_max_len, :].reshape(_num_reqs * _max_len, _num_kv_heads, _head_dim)[:num_tokens]
                # GQA: repeat KV heads
                if _num_heads != _num_kv_heads:
                    _rep = _num_heads // _num_kv_heads
                    _k = _k.repeat_interleave(_rep, dim=1)
                    _v = _v.repeat_interleave(_rep, dim=1)
                _q = query[:num_tokens]
                _attn_out = F.scaled_dot_product_attention(
                    _q.unsqueeze(0).transpose(1, 2),
                    _k.unsqueeze(0).transpose(1, 2),
                    _v.unsqueeze(0).transpose(1, 2),
                    is_causal=True,
                    scale=self.scale,
                )
                output[:num_tokens] = _attn_out.squeeze(0).transpose(0, 1)[:num_tokens]
                # Keep handles aligned with attn_params for downstream indexing
                graph_params.handles[num_tokens].append(None)
            else:
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

            # Store by key for order-independent replay lookup.
            # Skip for draft model: each layer appears multiple
            # times (once per spec step), so by-name storage would
            # lose all but the last step's task group.
            layer_name = self._layer_name
            if (not _EXTRA_CTX.is_draft_model
                    and layer_name
                    and num_tokens in graph_params.attn_params_by_key):
                graph_params.attn_params_by_key[num_tokens][layer_name] = {
                    "params": graph_params.attn_params[num_tokens][-1],
                    "handle": graph_params.handles[num_tokens][-1] if graph_params.handles.get(num_tokens) else None,
                    "event": event,
                }

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
        passed_key = key
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(
            key, value, attn_metadata, kv_cache
        )
        if self.enable_hamming_sparse and attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
            reshape_and_cache_kvcomp(attn_metadata.kvcomp_metadata, self.layerIndex, passed_key)
        elif self.enable_hamming_sparse:
            block_table, actual_seq_lengths_kv = get_kvcomp_decode_params(
                self.layerIndex, attn_metadata.kvcomp_metadata, query, passed_key, block_table, actual_seq_lengths_kv
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
                key,
                value,
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
                    sparse_mode=3,
                )

            attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if _EXTRA_CTX.capturing:
            return self.full_graph_pa(query, attn_metadata, output)
        # KV sharing: swap to target's key_cache
        _target_impl = getattr(self, '_kv_share_target_impl', None)
        _has_tgt_cache = (
            _target_impl is not None
            and getattr(_target_impl, 'key_cache', None) is not None
        )
        if _has_tgt_cache:
            _kc = _target_impl.key_cache
            _vc = _target_impl.value_cache
        else:
            _kc = self.key_cache
            _vc = self.value_cache

        torch_npu._npu_paged_attention(
            query=query,
            key_cache=_kc,
            value_cache=_vc,
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
        return torch_npu.npu_fusion_attention(
            query=query,
            key=key,
            value=value,
            head_num=self.num_heads,
            input_layout="TND",
            scale=self.scale,
            actual_seq_qlen=attn_metadata.actual_seq_lengths_q,
            actual_seq_kvlen=attn_metadata.actual_seq_lengths_q,
        )[0]

    def _forward_shared_kv_prefill_attention(
        self,
        query: torch.Tensor,
        shared_key: torch.Tensor,
        shared_value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Manual PyTorch attention with already-dense shared KV from block_table.

        Ascend FIA (npu_fusion_attention) cannot handle cross-attention where
        actual_seq_qlen differs from actual_seq_kvlen — it either crashes with
        mask shape errors or produces zero output.  Use PyTorch's
        scaled_dot_product_attention instead, which correctly supports
        cross-attention with arbitrary Q/KV lengths and GQA (grouped-query
        attention).
        """
        import torch.nn.functional as F
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        # Slice to actual query tokens.
        # vLLM uses [seq_len, num_heads, head_dim] but PyTorch sdpa in 3D
        # format requires equal Q/KV sequence lengths.  For cross-attention
        # (where num_tokens != KV length) we must use 4D format:
        #   [batch, num_heads, seq_len, head_dim]
        q = query[:num_tokens]  # [T, H, D]
        k = shared_key           # [S, Hkv, D]
        v = shared_value         # [S, Hkv, D]

        # Create causal cross-attention mask.  Query position i can attend
        # to KV positions [0, kv_len - q_len + i].  For sliding-window
        # layers, additionally restrict to the last `sliding_window` tokens.
        S = k.shape[0]
        offset = S - num_tokens
        mask = torch.ones(num_tokens, S, dtype=q.dtype, device=q.device) * float('-inf')
        for i in range(num_tokens):
            window_start = max(0, i + offset - self.sliding_window + 1) \
                if self.sliding_window is not None and S > self.sliding_window \
                else 0
            causal_end = i + offset + 1
            mask[i, window_start:causal_end] = 0
        attn_mask = mask

        # Handle GQA: expand KV heads to match Q heads.
        # Ascend NPU's scaled_dot_product_attention does not broadcast
        # head dimension, so we must explicitly repeat KV heads.
        # Use repeat_interleave (not expand+reshape) to preserve the
        # correct head mapping: Q head i -> KV head i//n_rep.
        if q.shape[1] != k.shape[1]:
            n_rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(n_rep, dim=1)  # [S, Hkv, D] -> [S, Hq, D]
            v = v.repeat_interleave(n_rep, dim=1)  # [S, Hkv, D] -> [S, Hq, D]

        # Always use 4D format [B, H, L, D] for Ascend NPU.
        # 3D format [T, H, D] causes inplace_add shape mismatch in
        # Ascend's SDPA kernel when num_tokens is large (e.g. >2000).
        q_4d = q.unsqueeze(0).transpose(1, 2)   # [T, H, D] -> [1, H, T, D]
        k_4d = k.unsqueeze(0).transpose(1, 2)   # [S, H, D] -> [1, H, S, D]
        v_4d = v.unsqueeze(0).transpose(1, 2)   # [S, H, D] -> [1, H, S, D]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [T, S] -> [1, 1, T, S]
        attn_output = F.scaled_dot_product_attention(
            q_4d, k_4d, v_4d,
            attn_mask=attn_mask,
            scale=self.scale,
        )  # [1, H, T, D]
        attn_output = attn_output.squeeze(0).transpose(0, 1)  # [T, H, D]

        output[:num_tokens] = attn_output
        return output

    def _gather_paged_kv_to_dense(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_size = key_cache.shape[1]
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.long, device=key_cache.device)
        max_seq_len = int(seq_lens_tensor.max().item())
        num_blocks = cdiv(max_seq_len, block_size)
        block_table_sliced = block_table[: len(seq_lens), :num_blocks].long()

        flat_block_ids = block_table_sliced.reshape(-1)
        max_tokens_padded = num_blocks * block_size
        dense_shape = (
            len(seq_lens),
            max_tokens_padded,
            self.num_kv_heads,
            self.head_size,
        )
        gathered_key = key_cache.index_select(0, flat_block_ids).reshape(dense_shape)
        gathered_value = value_cache.index_select(0, flat_block_ids).reshape(dense_shape)

        positions = torch.arange(max_tokens_padded, dtype=torch.long, device=key_cache.device)
        valid_mask = positions.unsqueeze(0) < seq_lens_tensor.unsqueeze(1)
        return gathered_key[valid_mask].contiguous(), gathered_value[valid_mask].contiguous()

    def _get_current_token_shared_kv(
        self,
        attn_metadata: AscendMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Gather current-token KV from the producer layer's shared cache."""

        if self.key_cache is None or self.value_cache is None:
            return None, None
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        if attn_metadata.slot_mapping is None or attn_metadata.slot_mapping.numel() < num_tokens:
            return None, None
        slots = attn_metadata.slot_mapping[:num_tokens].long()
        key = self.key_cache.reshape(-1, self.num_kv_heads, self.head_size).index_select(0, slots)
        value = self.value_cache.reshape(-1, self.num_kv_heads, self.head_size).index_select(0, slots)
        return key, value

    def _get_shared_kv_from_block_table(
        self,
        attn_metadata: AscendMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        """Gather K/V from the shared target cache using block tables.

        Used when slot_mapping is not available (e.g., during speculative
        decoding where the draft model inherits attn_metadata from the
        target but slot_mapping may not be populated for draft layers).

        IMPORTANT: For KV-sharing draft layers, self.key_cache points to the
        draft model's own (empty) cache.  We must swap to the target layer's
        cache via _kv_share_target_impl, mirroring the PA path fix.
        """
        # Swap to target cache when KV-sharing (mirrors PA path fix in
        # forward_paged_attention).
        _tgt_impl = getattr(self, '_kv_share_target_impl', None)
        _swapped = False
        if _tgt_impl is not None and _tgt_impl.key_cache is not None:
            read_kc = _tgt_impl.key_cache
            read_vc = _tgt_impl.value_cache
            _swapped = True
        else:
            read_kc = self.key_cache
            read_vc = self.value_cache

        if read_kc is None or read_vc is None:
            with open('/tmp/attn_path_debug.log', 'a') as _f:
                _f.write(f"[KV_SHARE_BLOCK] layer={self._layer_name} "
                         f"ERROR=null_cache swapped={_swapped}\n")
            return None, None
        # ── BLOCK TABLE ROUTING ──
        # Draft layers share KV with target layers that may be in DIFFERENT
        # KV cache groups.  attn_metadata.block_tables is the common (gid=0)
        # table; using it for layers whose target is in gid≠0 reads from the
        # wrong pool.  Route each layer to its per-group block_table via
        # _kv_share_gid (set by _store_gids_on_impls) + _per_group_bt_ref
        # (the {gid: block_table} dict set by set_per_group_block_table).
        _my_gid = getattr(self, '_kv_share_gid', None)
        _per_group_bt = getattr(self, '_per_group_bt_ref', None)
        _routed_bt = None
        if _my_gid is not None and _per_group_bt is not None and _my_gid in _per_group_bt:
            _routed_bt = _per_group_bt[_my_gid]
        block_table = _routed_bt if _routed_bt is not None else attn_metadata.block_tables
        seq_lens = attn_metadata.seq_lens_list
        if block_table is None or not seq_lens:
            with open('/tmp/attn_path_debug.log', 'a') as _f:
                _f.write(f"[KV_SHARE_BLOCK] layer={self._layer_name} "
                         f"ERROR=no_block_table bt_is_none={block_table is None} "
                         f"seq_lens={seq_lens}\n")
            return None, None

        try:
            dense_key, dense_value = self._gather_paged_kv_to_dense(
                read_kc, read_vc, block_table, seq_lens,
            )
            _k_mean = dense_key.float().mean().item()
            _k_std = dense_key.float().std().item()
            _v_mean = dense_value.float().mean().item()
            _per_group_bt = getattr(self, '_per_group_bt_ref', None)
            _fallback_gid = -1
            if abs(_k_mean) < 1e-6 and _per_group_bt is not None and len(_per_group_bt) > 0:
                # REVERSE iteration: try highest gid first (global attn = 5)
                for _gid, _bt in reversed(list(_per_group_bt.items())):
                    try:
                        _dk, _dv = self._gather_paged_kv_to_dense(
                            read_kc, read_vc, _bt, seq_lens,
                        )
                        _km = _dk.float().mean().item()
                        if abs(_km) > 1e-6:
                            dense_key, dense_value = _dk, _dv
                            block_table = _bt
                            _k_mean = _km
                            _k_std = _dk.float().std().item()
                            _v_mean = _dv.float().mean().item()
                            _fallback_gid = _gid
                            break
                    except Exception:
                        continue
            return dense_key, dense_value
        except Exception as e:
            with open('/tmp/attn_path_debug.log', 'a') as _f:
                _f.write(f"[KV_SHARE_BLOCK] layer={self._layer_name} "
                         f"ERROR={e}\n")
            return None, None

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
                # KV-sharing target layers, used by Gemma4 local/global layer
                # pairs, consume the producer layer's cache. Re-caching here
                # would overwrite the shared KV slots before attention reads it.
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
        num_tokens = query.shape[0]
        _kv_share = getattr(self, 'kv_sharing_target_layer_name', None)

        # KV-sharing layers (e.g., Gemma4 MTP draft) read K/V from the
        # target layer's cache.  Ensure self.key_cache / self.value_cache
        # are initialised from the kv_cache tuple BEFORE calling
        # _get_current_token_shared_kv, otherwise they will still be
        # None (the draft layers do not own a private cache) and the
        # shared-KV prefill path is skipped.
        if (
            self.kv_sharing_target_layer_name is not None
            and self.key_cache is None
            and kv_cache is not None
            and len(kv_cache) >= 2
        ):
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        # On A5, FIA v2 supports head_dim=512 natively — no large-head fallback needed.

        _kv_prefill_eligible = (
            self.kv_sharing_target_layer_name is not None
            and key is not None
            and value is not None
            and query.shape[0] == key.shape[0]
            and attn_metadata.attn_state in (AscendAttentionState.PrefillNoCache, AscendAttentionState.ChunkedPrefill, AscendAttentionState.SpecDecoding)
        )

        if _kv_prefill_eligible:
            # Try slot_mapping-based lookup first (needed when the
            # same request's target K/V are at known cache slots).
            # BUT: for SpecDecoding, slot_mapping may only cover the
            # most recent target-model token slots, not the full KV
            # cache.  _get_current_token_shared_kv would then return
            # a tiny K/V slice instead of None, preventing the
            # block-table fallback below.  Skip it for SpecDecoding.
            # Draft model layers do not write KV to the cache (they read
            # from the target's shared cache).  Their slot_mapping points to
            # empty/wrong positions, always fall through to the block_table
            # gather instead.
            if (attn_metadata.attn_state != AscendAttentionState.SpecDecoding
                    and not getattr(_EXTRA_CTX, 'is_draft_model', False)):
                shared_key, shared_value = self._get_current_token_shared_kv(attn_metadata)
            else:
                shared_key, shared_value = None, None

            # Fall back to block-table gathering.  This is the normal
            # path for speculative decoding where the draft model
            # inherits the target's paged KV cache but slot_mapping
            # may not be populated for the draft's attn_metadata.
            if shared_key is None or shared_value is None:
                shared_key, shared_value = self._get_shared_kv_from_block_table(
                    attn_metadata
                )

            if shared_key is not None and shared_value is not None:
                return self._forward_shared_kv_prefill_attention(
                    query,
                    shared_key,
                    shared_value,
                    attn_metadata,
                    output,
                )

        _pa_usable = using_paged_attention(num_tokens, self.vllm_config)

        _cond_a = (
            attn_metadata.attn_state in (AscendAttentionState.DecodeOnly, AscendAttentionState.SpecDecoding)
            and (self.sliding_window is None or self.kv_sharing_target_layer_name is not None)
            and (_pa_usable or self.kv_sharing_target_layer_name is not None)
        )

        if _cond_a:
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
        if self.enable_hamming_sparse:
            self.layerIndex = int(layer.layer_name.split(".")[2])
        self._layer_name = layer.layer_name

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendAttentionBackendImpl")

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        num_tokens = query.shape[0]

        _kv_share = getattr(self, 'kv_sharing_target_layer_name', None)
        _is_draft = getattr(_EXTRA_CTX, 'is_draft_model', False)



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
            # Gemma4 MTP draft layers are Q-only: K/V come from the shared
            # target KV cache.  Gemma4MTPAttention.forward() creates a dummy
            # K/V via torch.empty() and passes it as key/value to self.attn().
            # Writing this uninitialized memory back via reshape_and_cache
            # would corrupt the shared target KV cache, causing progressive
            # degradation across sequential loop steps:
            #   draft[0] fine (merged fwd), draft[1] occasionally OK (7-13%),
            #   draft[2-4] always garbage.
            # Skip the write for draft KV-shared layers — they only READ.
            _is_draft_kv_share = (
                getattr(self, 'kv_sharing_target_layer_name', None) is not None
                and getattr(_EXTRA_CTX, 'is_draft_model', False)
            )

            if not _is_draft_kv_share:
                output_padded = output
                query, key, value, output_padded = self.reshape_and_cache(
                    query, key, value, kv_cache, attn_metadata, output
                )

        _is_pooling = (attn_metadata.model_runner_type == "pooling" and not attn_metadata.causal)

        # pooling model branch
        if _is_pooling:

            attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output



        if output_padded is not None:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output_padded)
        else:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output)
        output[:num_tokens] = attn_output[:num_tokens]
        return output


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
                query, key, value, _ = self.reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)
            # pooling model branch
            if attn_metadata.model_runner_type == "pooling":
                attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                if _EXTRA_CTX.capturing:
                    attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output, layer)
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
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
                    query, key, value, _ = self.reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)
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

        bnsd = (1, self.num_kv_heads, 1, self.head_size)
        layer._c8_k_aq_scale = layer._c8_k_scale.view(bnsd).contiguous()
        layer._c8_k_aq_offset = layer._c8_k_offset.view(bnsd).contiguous()
        layer._c8_v_aq_scale = layer._c8_v_scale.view(bnsd).contiguous()
        layer._c8_v_aq_offset = layer._c8_v_offset.view(bnsd).contiguous()

        layer._c8_k_inv_scale = 1.0 / layer._c8_k_scale
        layer._c8_v_inv_scale = 1.0 / layer._c8_v_scale

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
        block_size = key.shape[1]
        H = key.shape[2]
        max_blocks_per_seq = block_table.shape[1]
        max_tokens_padded = max_blocks_per_seq * block_size

        flat_ids = block_table.reshape(-1)
        gathered_k = key[flat_ids].view(batch_size, max_tokens_padded, H)
        gathered_v = value[flat_ids].view(batch_size, max_tokens_padded, H)

        seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=key.device)
        positions = torch.arange(max_tokens_padded, dtype=torch.long, device=key.device)
        valid_mask = (positions.unsqueeze(0) < seq_lens_t.unsqueeze(1)).view(-1)

        dense_k = gathered_k.view(-1, H)[valid_mask]
        dense_v = gathered_v.view(-1, H)[valid_mask]

        dense_k = dense_k.view(-1, self.num_kv_heads, self.head_size)
        dense_v = dense_v.view(-1, self.num_kv_heads, self.head_size)
        dense_k = (dense_k.to(target_dtype) - layer._c8_k_offset) * layer._c8_k_scale
        dense_v = (dense_v.to(target_dtype) - layer._c8_v_offset) * layer._c8_v_scale
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
        key = self.key_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]
        value = self.value_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]
        batch_size = len(attn_metadata.seq_lens_list)

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query[:batch_size].unsqueeze(2),
            key,
            value,
            key_antiquant_scale=layer._c8_k_aq_scale,
            key_antiquant_offset=layer._c8_k_aq_offset,
            value_antiquant_scale=layer._c8_v_aq_scale,
            value_antiquant_offset=layer._c8_v_aq_offset,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BNSD",
            scale=self.scale,
            block_size=block_size,
            key_antiquant_mode=0,
            value_antiquant_mode=0,
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
            kv_k = self.key_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]
            kv_v = self.value_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]

            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query[:num_decode_tokens].unsqueeze(2),
                kv_k,
                kv_v,
                key_antiquant_scale=layer._c8_k_aq_scale,
                key_antiquant_offset=layer._c8_k_aq_offset,
                value_antiquant_scale=layer._c8_v_aq_scale,
                value_antiquant_offset=layer._c8_v_aq_offset,
                block_table=attn_metadata.block_tables[:num_decodes],
                actual_seq_lengths_kv=attn_metadata.seq_lens_list[:num_decodes],
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BNSD",
                scale=self.scale,
                block_size=block_size,
                key_antiquant_mode=0,
                value_antiquant_mode=0,
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
                paged_k = self.key_cache.view(num_block, blk_size, -1)  # type: ignore[attr-defined]
                paged_v = self.value_cache.view(num_block, blk_size, -1)  # type: ignore[attr-defined]
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
