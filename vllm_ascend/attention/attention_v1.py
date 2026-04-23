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

import torch
import torch_npu
import vllm.envs as envs_vllm
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
from vllm_ascend.attention.context_parallel.common_cp import AscendMetadataForDecode, AscendMetadataForPrefill
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    enable_cp,
    split_decodes_and_prefills,
    using_paged_attention,
)
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_graph_params,
    update_draft_graph_params_workspaces,
    update_graph_params_workspaces,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.flashcomm2_oshard_manager import flashcomm2_oshard_manager
from vllm_ascend.utils import weak_ref_tensors

# default max value of sliding window size
SWA_INT_MAX = 2147483647


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

    # sliding window attention mask
    swa_mask: torch.Tensor | None = None


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

        # Get attn_mask and swa_mask from singleton AttentionMaskBuilder
        attn_mask = self.attn_mask_builder.get_attention_mask(self.model_config)

        swa_mask = None
        is_swa = hasattr(self.model_config.hf_text_config, "sliding_window")
        if self.model_config is not None and is_swa:
            swa_mask = self.attn_mask_builder.get_swa_mask(
                self.model_config.dtype, self.model_config.hf_text_config.sliding_window
            )

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
            swa_mask=swa_mask,
            attn_state=attn_state,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            causal=common_attn_metadata.causal,
            model_runner_type=self.model_config.runner_type,
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

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None
        self.is_kv_producer = (
            self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.is_kv_producer
        )
        self.sinks = sinks

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
        if using_paged_attention(num_tokens, vllm_config):
            # Paged Attention update logic
            if _EXTRA_CTX.is_draft_model:
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
        else:
            # FIA update logic
            if _EXTRA_CTX.is_draft_model:
                graph_params = get_draft_graph_params()
                attn_metadata = draft_attn_metadatas
                attn_keys = list(attn_metadata[0].keys())
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
            if _EXTRA_CTX.is_draft_model:
                attn_keys = attn_keys * (len(graph_params.attn_params[num_tokens]) // num_layers)
            attn_count = 0
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    attn_keys,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
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
                    ) = param

                    sparse_mode = 3
                    if _EXTRA_CTX.is_draft_model:
                        draft_step = attn_count // num_layers
                        seq_lens = attn_metadata[draft_step][key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[draft_step][key].actual_seq_lengths_q
                        block_tables = attn_metadata[draft_step][key].block_tables
                        attn_count = attn_count + 1
                        if not attn_metadata[draft_step][key].causal:
                            sparse_mode = 0
                    else:
                        seq_lens = attn_metadata[key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[key].actual_seq_lengths_q
                        block_tables = attn_metadata[key].block_tables

                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu.npu_fused_infer_attention_score.out(
                        query=query,
                        key=key_cache,
                        value=value,
                        block_table=block_tables,
                        atten_mask=attn_mask,
                        input_layout="TND",
                        block_size=block_size,
                        actual_seq_lengths=actual_seq_lengths_q,
                        actual_seq_lengths_kv=seq_lens,
                        num_key_value_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale=scale,
                        sparse_mode=sparse_mode,
                        workspace=graph_params.workspaces.get(num_tokens),
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)

                    event.record(update_stream)

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
    ) -> torch.Tensor:
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)

        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        if _EXTRA_CTX.is_draft_model:
            graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        # Prepare tensors for attention output
        # TODO: Refactor this to step-level instead of layer-level

        # Get workspace from cache or calculate it if not present.
        workspace = graph_params.workspaces.get(num_tokens)
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=3 if attn_metadata.causal else 0,
                scale=self.scale,
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
                weak_ref_tensors(attn_metadata.attn_mask) if attn_metadata.attn_mask is not None else None,
                block_size,
                actual_seq_lengths_kv,
                actual_seq_lengths_q,
                self.num_kv_heads,
                self.num_heads,
                self.scale,
                weak_ref_tensors(output),
                weak_ref_tensors(softmax_lse),
            )
        )

        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3 if attn_metadata.causal else 0,
            workspace=workspace,
            out=[output, softmax_lse],
        )

        output = output.view(num_tokens, self.num_heads, self.head_size)

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

    def _forward_fia_slidingwindow(self, query: torch.Tensor, attn_metadata: AscendMetadata, output: torch.Tensor):
        batch_size = attn_metadata.seq_lens.shape[0]
        block_size = 128
        query = query.view(batch_size, 1, self.num_heads * self.head_size)
        key = self.key_cache
        value = self.value_cache
        if self.key_cache is not None and self.value_cache is not None:
            block_size = self.key_cache.shape[1]
            key = self.key_cache.flatten(2, 3).contiguous()
            value = self.value_cache.flatten(2, 3).contiguous()

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BSH",
            block_size=block_size,
            pre_tokens=self.sliding_window,
            scale=self.scale,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths=[1] * len(attn_metadata.seq_lens),
            actual_seq_lengths_kv=attn_metadata.seq_lens,
        )

        attn_output = attn_output.view(batch_size, self.num_heads, self.head_size)
        output[:batch_size] = attn_output[:batch_size]
        return output

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
            attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output
        if (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and self.sliding_window is not None
            and attn_metadata.seq_lens.shape[0] == query.size(0)
            and self.sinks is None
        ):
            return self._forward_fia_slidingwindow(query, attn_metadata, output)
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
                atten_mask = attn_metadata.swa_mask
                sparse_mode = 4
            else:
                atten_mask = attn_metadata.attn_mask
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
                atten_mask=atten_mask,
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
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event = torch.npu.Event()
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
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
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()
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
        if (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and using_paged_attention(num_tokens, self.vllm_config)
            and self.sliding_window is None
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


class AscendC8AttentionBackendImpl(AscendAttentionBackendImpl):
    """C8 INT8 KV Cache 量化的 Attention Backend 实现。

    背景：
        C8 是 ModelSlim 工具链产出的一种 KV cache 静态 per-channel INT8 量化格式。
        量化公式为：q = round(x * inv_scale + offset)，其中 scale/offset 按每个 head 的每个
        通道（channel）独立标定，存储在模型权重文件中（k_cache_scale、k_cache_offset 等）。
        相比 BF16 KV cache，INT8 可将 KV 内存占用减少约 50%，支持更大并发和更长上下文。

    激活方式（class surgery）：
        本类不通过正常继承链激活，而是在
        vllm_ascend/quantization/methods/kv_c8.py 的
        AscendC8KVCacheAttentionMethod.create_weights() 中，通过
        `layer.impl.__class__ = AscendC8AttentionBackendImpl`
        直接替换 layer.impl 的类型，从而让所有 C8 attention 层自动走本类的 forward 路径。

    CANN 9.0.0 NZ 格式适配说明：
        本实现面向 CANN 9.0.0 引入的 NZ 格式 KV cache：
        - KV cache 物理布局：[num_block, KV_N, D//32, block_size, 32]
        - FIA 算子支持 NZ 格式直接 perchannel 反量化，无需在 Python 层做手动 dequant
        - NZ 格式约束：不支持传入 antiquant_offset，因此本类仅适用于对称量化（offset == 0）的 checkpoint
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
        """C8 Attention 主入口，负责写 cache 并按推理状态分发到对应的计算路径。

        整体流程：
            1. 将当前 batch 的 BF16 K/V 量化为 INT8，写入 paged KV cache
            2. 根据 attn_state（DecodeOnly / ChunkedPrefill / PrefillNoCache / PrefillCacheHit）
               分发到对应的 FIA 计算函数

        参数：
            layer         : 当前 attention 层对象，挂载了 k_cache_scale / k_cache_offset 等量化参数
            query         : 当前 batch 的 Query 张量，shape [total_tokens, num_heads, head_size]，BF16
            key           : 当前 batch 的 Key 张量，shape [total_tokens, num_kv_heads, head_size]，BF16
            value         : 当前 batch 的 Value 张量，同 key
            kv_cache      : paged KV cache 的两块内存 (key_cache, value_cache)，INT8，由 vLLM 框架管理
            attn_metadata : 本次推理的元数据，包含 attn_state、seq_lens、block_tables 等调度信息
            output        : 预分配的输出张量，shape 同 query，结果原地写入
            output_scale  : 输出量化 scale（暂不支持，传入非 None 会报错）
            output_block_scale : 输出 block 量化 scale（暂不支持）

        返回：
            output        : 填充完毕的 attention 输出张量
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendC8AttentionBackendImpl")

        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        float_key, float_value = None, None
        if key is not None and value is not None:
            # 非纯 decode 场景需要把本 batch 的 BF16 K/V 写入 KV cache；
            # 同时保留一份 float 副本供 prefill 路径直接使用（避免从 cache 读回再 dequant）
            if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
                float_key, float_value = key, value
            # 将 BF16 K/V 量化为 INT8，然后写入 paged KV cache 对应的 block 位置
            key, value = self._quantize_kv_to_int8(key, value, layer, attn_metadata.num_actual_tokens)
            query, key, value, _ = self.reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)

        if attn_metadata.model_runner_type == "pooling":
            # embedding/pooling 场景走标准的 encoder attention，不走 C8 paged 路径
            attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output

        # 首次调用时初始化 scale 张量（幂等，后续调用直接返回）
        self._prepare_c8_scales(layer, query.device)

        # 根据推理阶段分发：
        #   DecodeOnly      → 纯增量解码，Q_S=1，走 BNSD + NZ 路径
        #   ChunkedPrefill  → 同一 batch 混有 decode 和 prefill token，需分别处理
        #   其余（PrefillNoCache / PrefillCacheHit）→ 纯 prefill，走 TND 路径
        if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            return self._forward_c8_decode(query, attn_metadata, output, layer)
        elif attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
            return self._forward_c8_chunked_prefill(query, float_key, float_value, attn_metadata, output, layer)
        else:
            return self._forward_c8_fused_infer_attention(
                query,
                # PrefillNoCache 时 float_key 有值（新 token），PrefillCacheHit 时也有值
                # 但 _forward_c8_fused_infer_attention 内部会根据 attn_state 决定是直接用
                # float KV 还是从 paged cache 读取 NZ INT8 KV
                float_key if float_key is not None else key,
                float_value if float_value is not None else value,
                attn_metadata,
                output,
                layer,
            )

    def _prepare_c8_scales(self, layer: AttentionLayer, device: torch.device) -> None:
        """初始化并缓存 C8 量化所需的 scale/offset 张量（幂等，仅在首次 forward 时执行）。

        本函数做三件事：
            1. 将模型权重中加载的全局 per-channel scale/offset 按当前 TP rank 切片，
               得到本卡负责的 KV head 对应的 [KV_N, D] 形状 scale/offset
            2. 对 offset 做合法性校验：NZ 格式下 FIA 不接受 antiquant_offset 参数，
               因此只有对称量化（offset == 0）的 checkpoint 才能走 NZ 路径
            3. 预计算两组衍生张量：
               - FIA 调用用的 antiquant_scale（[KV_N, 1, D]，BF16）
               - 写 cache 量化用的 inv_scale、offset（BF16）

        参数：
            layer  : attention 层对象，需已通过 weight_loader 加载以下参数：
                       layer.k_cache_scale  : K cache 每通道 scale，shape [total_kv_heads * D]
                       layer.k_cache_offset : K cache 每通道 zero-point，shape 同上
                       layer.v_cache_scale  : V cache 每通道 scale
                       layer.v_cache_offset : V cache 每通道 zero-point
            device : 目标计算设备，用于将 scale 张量搬运到对应的 NPU 卡上

        执行后在 layer 上挂载的属性：
            _c8_k_scale / _c8_v_scale     : [KV_N, D]，float32，本 TP rank 的 scale
            _c8_k_offset / _c8_v_offset   : [KV_N, D]，float32，本 TP rank 的 offset
            _c8_k_aq_scale / _c8_v_aq_scale : [KV_N, 1, D]，bfloat16，传给 FIA 的 antiquant_scale
            _c8_k_inv_scale_bf16          : [KV_N, D]，bfloat16，1/scale，写 cache 时用
            _c8_k_offset_bf16             : [KV_N, D]，bfloat16，offset，写 cache 时用
            _c8_scales_prepared           : True，标记已初始化，防止重复执行

        为什么 FIA NZ 不能传 offset：
            CANN 算子文档明确：NZ 格式下"不支持配置 key_antiquant_offset 和 value_antiquant_offset"。
            如果 checkpoint 的 offset 不为 0 却直接使用 NZ 路径，FIA 读回时会对每个通道产生
            固定偏差（约 offset × scale），导致 attention 计算结果错误且没有任何报错提示。
            因此在这里做硬检查，一旦发现非零 offset 立即抛出异常，避免静默数值污染。
        """
        if hasattr(layer, "_c8_scales_prepared"):
            return

        def _shard_and_reshape(raw: torch.Tensor) -> torch.Tensor:
            """将全局 scale/offset 按 TP rank 切片，返回 [KV_N, D] 形状的本地张量。

            ModelSlim 导出的 scale 是全局所有 KV head 拼在一起的一维向量
            （长度 = total_kv_heads * head_size）。多卡 TP 时每张卡只负责其中一段，
            需要按 [total_kv_heads, head_size] 展开后取对应行。
            """
            if raw.numel() == 1:
                # pertensor 场景：全局只有一个标量，所有 head 共享，直接搬运
                return raw.to(device=device)
            expected = self.num_kv_heads * self.head_size
            if raw.numel() != expected:
                # 当前 raw 包含所有 TP rank 的 KV head，需要按 rank 切片
                total_kv_heads = raw.numel() // self.head_size
                tp_rank = get_tensor_model_parallel_rank()
                tp_size = get_tensor_model_parallel_world_size()
                kv_head_start = tp_rank * total_kv_heads // tp_size
                raw = raw.view(total_kv_heads, self.head_size)[
                    kv_head_start : kv_head_start + self.num_kv_heads
                ].contiguous()
            return raw.view(self.num_kv_heads, self.head_size).to(device=device)

        layer._c8_k_scale = _shard_and_reshape(layer.k_cache_scale.data)
        layer._c8_k_offset = _shard_and_reshape(layer.k_cache_offset.data)
        layer._c8_v_scale = _shard_and_reshape(layer.v_cache_scale.data)
        layer._c8_v_offset = _shard_and_reshape(layer.v_cache_offset.data)

        # NZ 格式约束：FIA 不支持传入 antiquant_offset。
        # 若 offset 不为 0，FIA 读回时每通道会有 offset * scale 的系统偏差，且不报错。
        # 这里做硬检查，防止非对称量化的 checkpoint 静默跑出错误结果。
        _k_offset_max = layer._c8_k_offset.abs().max().item()
        _v_offset_max = layer._c8_v_offset.abs().max().item()
        if _k_offset_max > 1e-6 or _v_offset_max > 1e-6:
            raise ValueError(
                f"C8 NZ 格式要求对称量化（offset == 0），"
                f"但当前层存在非零 KV cache offset "
                f"（max |k_offset|={_k_offset_max:.6f}, max |v_offset|={_v_offset_max:.6f}）。"
                "NZ 模式无法正确反量化非对称 C8 权重，"
                "请使用对称量化的 C8 checkpoint，或禁用 NZ 模式。"
            )

        # 为 FIA NZ perchannel 模式预计算 antiquant_scale。
        # FIA 算子要求：shape [KV_N, 1, D]，dtype bfloat16。
        # 中间的维度 "1" 对应 sequence 维，广播到所有 token 位置。
        nz_scale_shape = (self.num_kv_heads, 1, self.head_size)
        layer._c8_k_aq_scale = layer._c8_k_scale.to(torch.bfloat16).view(nz_scale_shape).contiguous()
        layer._c8_v_aq_scale = layer._c8_v_scale.to(torch.bfloat16).view(nz_scale_shape).contiguous()

        # 写 cache 时的量化辅助张量（_quantize_kv_to_int8 使用）。
        # 量化公式：q = clamp(round(x * inv_scale + offset), -128, 127)
        # 对称量化下 offset == 0，公式退化为 q = round(x / scale)。
        layer._c8_k_inv_scale_bf16 = (1.0 / layer._c8_k_scale).to(torch.bfloat16)
        layer._c8_k_offset_bf16 = layer._c8_k_offset.to(torch.bfloat16)
        layer._c8_v_inv_scale_bf16 = (1.0 / layer._c8_v_scale).to(torch.bfloat16)
        layer._c8_v_offset_bf16 = layer._c8_v_offset.to(torch.bfloat16)

        layer._c8_scales_prepared = True

    def _get_kv_cache_nz(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        """将 paged INT8 KV cache 从 vLLM 原生 ND 格式转换为昇腾 NZ 格式，供 FIA 算子使用。

        背景：
            vLLM 框架内部的 paged KV cache 采用 ND 格式：
                [num_block, block_size, KV_N, D]
            其中 num_block 是总块数，block_size 是每块可存放的 token 数，KV_N 是 KV head 数，D 是 head_size。

            CANN 9.0.0 的 FIA 算子在 NZ 格式 + 伪量化场景下，要求 KV cache 布局为：
                [num_block, KV_N, D//32, block_size, 32]
            其中最后两维 (block_size, 32) 是 NZ 的"碎块"形式，D 维被拆成 D//32 组，每组 32 个元素。
            这种布局对昇腾硬件的 cube 单元更友好，可以减少访存跳转并提升吞吐。

        转换步骤（不拷贝数据，通过 view + permute 实现）：
            原始：[num_block, block_size, KV_N, D]
            step1 view    → [num_block, block_size, KV_N, D//32, 32]   （把 D 拆成两段）
            step2 permute → [num_block, KV_N, D//32, block_size, 32]   （把 KV_N 和 block_size 换位）
            step3 contiguous()  （permute 后内存不连续，需调用 contiguous 确保 FIA 可以直接访问）

        约束（CANN 9.0.0 文档要求）：
            - block_size 必须是 128 或 512
            - head_size（D）必须是 32 的整数倍

        返回：
            key_nz   : NZ 格式的 K cache，shape [num_block, KV_N, D//32, block_size, 32]，INT8
            value_nz : NZ 格式的 V cache，shape 同上，INT8
            block_size : 当前 KV cache 的 block_size，传给 FIA 的 block_size 参数使用
        """
        num_block, block_size, kv_n, head_size = self.key_cache.shape  # type: ignore[union-attr]
        assert block_size in (128, 512), (
            f"C8 NZ 格式要求 block_size 为 128 或 512，当前为 {block_size}"
        )
        assert head_size % 32 == 0, (
            f"C8 NZ 格式要求 head_size 是 32 的整数倍，当前为 {head_size}"
        )

        def _to_nz(cache: torch.Tensor) -> torch.Tensor:
            # view 不改变元素顺序，只重新解释形状；permute 改变维度顺序后内存变得不连续，
            # contiguous() 触发一次数据重排，使内存连续，FIA 才能正确读取
            return (
                cache.view(num_block, block_size, kv_n, head_size // 32, 32)
                .permute(0, 2, 3, 1, 4)
                .contiguous()
            )

        return _to_nz(self.key_cache), _to_nz(self.value_cache), block_size  # type: ignore[arg-type]

    def _quantize_kv_to_int8(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: AttentionLayer,
        num_actual_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """将当前 batch 的 BF16 K/V 张量量化为 INT8，用于写入 paged KV cache。

        量化公式（per-channel，每个通道独立 scale/offset）：
            q = clamp(round(x * inv_scale + offset), -128, 127)
        等价于：
            q = clamp(round(x / scale + offset), -128, 127)
        对称量化（offset == 0）时退化为：
            q = clamp(round(x / scale), -128, 127)

        参数：
            key              : BF16 Key 张量，shape [padded_tokens, num_kv_heads, head_size]
                               注意 vLLM 会把 tensor 预分配到最大长度，有效 token 只在前 num_actual_tokens 行
            value            : BF16 Value 张量，shape 同 key
            layer            : attention 层对象，需已调用 _prepare_c8_scales 初始化量化参数
            num_actual_tokens: 当前 batch 实际有效的 token 数量（排除 padding）

        返回：
            k_int8 : INT8 量化后的 Key，shape [num_actual_tokens, num_kv_heads, head_size]
            v_int8 : INT8 量化后的 Value，shape 同上

        注意：
            只对前 num_actual_tokens 行做量化，其余 padding 位置不处理，
            后续 reshape_and_cache 也只会写入这些有效行。
        """
        self._prepare_c8_scales(layer, key.device)

        # 截取有效 token，避免对 padding 位置做无意义计算
        actual_key = key[:num_actual_tokens]
        actual_value = value[:num_actual_tokens]

        k_int8 = torch.clamp(
            torch.round(actual_key * layer._c8_k_inv_scale_bf16 + layer._c8_k_offset_bf16),
            -128,
            127,
        ).to(torch.int8)
        v_int8 = torch.clamp(
            torch.round(actual_value * layer._c8_v_inv_scale_bf16 + layer._c8_v_offset_bf16),
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
        """纯解码（DecodeOnly）阶段的 C8 attention 计算，使用 FIA BNSD 布局 + NZ 格式 INT8 KV cache。

        场景说明：
            Decode 阶段每个请求只有 1 个新 query token（自回归生成），
            需要 attend 到 KV cache 中已存储的全部历史 token。
            FIA 使用 BNSD 布局，其中 S（sequence 维）= 1，对应 unsqueeze(2) 操作。

        CANN 9.0.0 NZ 格式约束：
            - KV 输入必须是 NZ 布局：[num_block, KV_N, D//32, block_size, 32]
            - antiquant_scale shape：[KV_N, 1, D]，dtype bfloat16
            - 不支持 antiquant_offset（已在 _prepare_c8_scales 处校验为 0）
            - block_size 必须是 128 或 512
            - inner_precise=1 启用高性能模式（NZ 格式必须设置此值）

        参数：
            query        : Query 张量，shape [batch_size, num_heads, head_size]（Decode 时 S=1 已隐含）
            attn_metadata: 推理元数据，其中：
                           seq_lens_list  → 每个请求在 KV cache 中的历史 token 数（原始长度，非累积和）
                           block_tables   → 每个请求使用的 KV cache block 索引表
            output       : 预分配的输出张量，结果原地写入
            layer        : attention 层，提供 _c8_k_aq_scale 等量化参数

        返回：
            output       : 写入 attention 结果后的输出张量
        """
        key_nz, value_nz, block_size = self._get_kv_cache_nz()
        batch_size = len(attn_metadata.seq_lens_list)

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            # Decode 时每个请求只有 1 个 query token；
            # BNSD 布局要求 shape [B, N, S, D]，S=1 通过 unsqueeze(2) 插入
            query[:batch_size].unsqueeze(2),
            key_nz,    # NZ 格式，shape [num_block, KV_N, D//32, block_size, 32]
            value_nz,  # NZ 格式，shape 同上
            key_antiquant_scale=layer._c8_k_aq_scale,    # [KV_N, 1, D]，FIA 内核 perchannel 反量化
            value_antiquant_scale=layer._c8_v_aq_scale,
            block_table=attn_metadata.block_tables,       # [batch_size, max_blocks_per_seq]，paged 地址映射
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,  # 每个请求已填充的 KV token 数
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BNSD",
            scale=self.scale,          # attention softmax 缩放因子，通常为 1/sqrt(D)
            block_size=block_size,     # paged KV cache 每块的 token 容量
            key_antiquant_mode=0,      # 0 = perchannel 模式（与 antiquant_scale shape 对应）
            value_antiquant_mode=0,
            inner_precise=1,           # 高性能模式，NZ 格式必须设置，否则结果不正确
            sparse_mode=0,             # 0 = 无稀疏掩码（Decode 不需要 causal mask）
        )
        # squeeze 去掉 unsqueeze 插入的 S 维，恢复 [batch_size, num_heads, head_size]
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
        """ChunkedPrefill 阶段的 C8 attention 计算（同一 batch 混有 decode 和 prefill 请求）。

        ChunkedPrefill 场景说明：
            vLLM 的 ChunkedPrefill 调度器会把正在解码的请求（decode）和正在预填充的请求（prefill）
            打包到同一个 batch 中以提高 GPU/NPU 利用率。两类请求的 attention 计算方式不同，
            必须分别处理后将结果写回 output 的对应位置。

        三条子路径：
            1. Decode（BNSD + NZ）：
               每个 decode 请求只有 1 个新 query token，attend 到 NZ 格式 INT8 KV cache。
               与 _forward_c8_decode 逻辑相同，但只取前 num_decodes 条请求。

            2. All-new Prefill（TND + float KV）：
               prefill 请求没有任何历史 KV cache（seq_lens == qlen），
               直接用本次计算得到的 BF16 float KV 做注意力，不涉及 paged cache。
               不需要 antiquant 参数。

            3. Cache-hit Prefill（TND + NZ INT8 KV）：
               prefill 请求有部分历史 token 已在 KV cache 中（seq_lens > qlen），
               需要从 paged NZ INT8 KV cache 读历史 token，由 FIA 内核完成 perchannel 反量化。
               CANN 9.0.0 之前需要 Python 层手动 gather+dequant，现在 FIA 原生支持。

        参数：
            query        : 当前 batch 全部 token 的 Query，shape [total_tokens, num_heads, head_size]
                           前 num_decode_tokens 行是 decode 请求，后面是 prefill 请求
            float_key    : BF16 格式的 Key（量化前），shape 同 query；all-new prefill 时使用
                           None 表示纯 decode batch（理论上不会出现在 ChunkedPrefill 场景）
            float_value  : BF16 格式的 Value，同 float_key
            attn_metadata: 推理元数据，其中：
                           num_decode_tokens  → decode 部分占用的 token 行数
                           num_decodes        → decode 请求数量
                           actual_seq_lengths_q → Q 的累积长度列表（用于定位每个请求在 query 中的位置）
                           seq_lens_list      → 每个请求在 KV cache 中的历史 token 总数（原始长度）
                           block_tables       → paged KV cache 块索引表
                           attn_mask          → causal attention mask（prefill 需要）
            output       : 预分配的输出张量，decode 和 prefill 结果分别原地写入对应位置
            layer        : attention 层，提供量化 scale 参数

        返回：
            output       : 填充完毕的输出张量
        """
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_decodes = attn_metadata.num_decodes
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q  # Q 长度累积和，如 [4, 9, 15]
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]  # 本 batch 实际总 token 数

        # NZ cache 只获取一次，decode 和 cache-hit prefill 共用同一份，避免重复转换
        key_nz, value_nz, block_size = self._get_kv_cache_nz()

        # ── 子路径 1：Decode（BNSD + NZ INT8 KV）──────────────────────────────────
        if num_decode_tokens > 0:
            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                # Decode: 每请求 1 个 token，BNSD 要求 shape [B, N, S, D]，S=1 → unsqueeze(2)
                query[:num_decode_tokens].unsqueeze(2),
                key_nz,
                value_nz,
                key_antiquant_scale=layer._c8_k_aq_scale,
                value_antiquant_scale=layer._c8_v_aq_scale,
                block_table=attn_metadata.block_tables[:num_decodes],           # 只取 decode 请求的 block 表
                actual_seq_lengths_kv=attn_metadata.seq_lens_list[:num_decodes],# 每个 decode 请求的历史 KV 长度
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BNSD",
                scale=self.scale,
                block_size=block_size,
                key_antiquant_mode=0,
                value_antiquant_mode=0,
                inner_precise=1,
                sparse_mode=0,  # decode 无需 causal mask
            )
            output[:num_decode_tokens] = attn_out.squeeze(2)

        # ── 子路径 2/3：Prefill（TND）─────────────────────────────────────────────
        if attn_metadata.num_prefills > 0:
            # 从 query 中截取属于 prefill 请求的部分
            prefill_q = query[num_decode_tokens:num_tokens]
            n_prefill = num_tokens - num_decode_tokens

            # prefill_seq_qlen：prefill 部分各请求的 Q 累积长度，需减去 decode 占用的 offset
            # 例如 actual_seq_qlen = [4, 9, 15]，num_decode_tokens=4，num_decodes=1，
            # 则 prefill_seq_qlen = [9-4, 15-4] = [5, 11]
            prefill_seq_qlen = [
                actual_seq_qlen[i] - num_decode_tokens for i in range(num_decodes, len(actual_seq_qlen))
            ]

            # 判断所有 prefill 请求是否都是全新请求（无历史 KV cache）：
            # 对于全新请求：seq_lens（KV 总长） == qlen（本次 Q 长度），即历史上没有 token
            # 对于 cache-hit 请求：seq_lens > qlen，说明有 token 已在 cache 中
            all_new_prefill = True
            for i in range(num_decodes, len(attn_metadata.seq_lens_list)):
                # q_start：该请求在 actual_seq_qlen 数组中的起始累积值（前一个请求的末尾）
                q_start = actual_seq_qlen[i - 1] if i > 0 else 0
                qlen_i = actual_seq_qlen[i] - q_start  # 该请求的 Q token 数
                if attn_metadata.seq_lens_list[i] > qlen_i:
                    # KV 总长 > Q 长度，说明该请求有历史 cache，不是全新请求
                    all_new_prefill = False
                    break

            if all_new_prefill and float_key is not None and float_value is not None:
                # ── 子路径 2：All-new Prefill，直接用 float KV，不查 cache ────────
                prefill_k = float_key[num_decode_tokens:num_tokens]
                prefill_v = float_value[num_decode_tokens:num_tokens]
                attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                    query=prefill_q,
                    key=prefill_k,    # BF16 float，TND 布局，shape [n_prefill_tokens, num_kv_heads, D]
                    value=prefill_v,
                    atten_mask=attn_metadata.attn_mask,  # causal mask，prefill 需要下三角掩码
                    block_table=None,                    # 无 paged cache，不需要 block 表
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=prefill_seq_qlen,    # Q 每请求的长度（TND 需要用于 causal mask 边界）
                    actual_seq_lengths_kv=prefill_seq_qlen, # KV 长度 == Q 长度（全新 prefill，历史为 0）
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    sparse_mode=3,  # 3 = 下三角 causal mask 模式
                )
            else:
                # ── 子路径 3：Cache-hit Prefill，NZ INT8 KV + FIA perchannel 反量化 ──
                # CANN 9.0.0 支持 TND + NZ 格式 + block_table + antiquant_scale，
                # FIA 内核自动完成从 paged INT8 cache 读取、反量化、attention 计算全流程，
                # 不再需要 Python 层手动 gather + dequant（旧方案的性能瓶颈所在）。
                prefill_bt = attn_metadata.block_tables[num_decodes:]  # prefill 请求的 block 索引表
                # actual_seq_lengths_kv 传原始 seq_lens（非累积和），与文件中其他 TND + block_table
                # 路径（见 _get_fia_params）的语义保持一致
                prefill_sl = attn_metadata.seq_lens_list[num_decodes:]

                attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                    query=prefill_q,
                    key=key_nz,    # NZ 格式 INT8 KV，FIA 内核读 paged cache 并反量化
                    value=value_nz,
                    key_antiquant_scale=layer._c8_k_aq_scale,   # [KV_N, 1, D]，perchannel 反量化系数
                    value_antiquant_scale=layer._c8_v_aq_scale,
                    atten_mask=attn_metadata.attn_mask,
                    block_table=prefill_bt,
                    input_layout="TND",
                    block_size=block_size,
                    actual_seq_lengths=prefill_seq_qlen,  # 每个 prefill 请求的 Q 长度
                    actual_seq_lengths_kv=prefill_sl,     # 每个 prefill 请求的 KV 总长（含历史）
                    num_key_value_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale=self.scale,
                    key_antiquant_mode=0,   # perchannel 模式
                    value_antiquant_mode=0,
                    inner_precise=1,        # NZ 格式必须开启高性能模式
                    sparse_mode=3,
                )

            # TND 输出 shape 是 [n_prefill_tokens, num_heads * head_size]，需要 view 成 [T, N, D]
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
        """纯 Prefill 阶段的 C8 attention 计算，处理 PrefillNoCache 和 PrefillCacheHit 两种状态。

        与 _forward_c8_chunked_prefill 的区别：
            本函数处理的 batch 中只有 prefill 请求（无 decode 请求混入），
            即 attn_state 是 PrefillNoCache 或 PrefillCacheHit，而非 ChunkedPrefill。
            ChunkedPrefill 由专门的 _forward_c8_chunked_prefill 处理。

        两条子路径：

        PrefillNoCache（全新预填充，没有任何历史 KV cache）：
            - 直接使用本次计算的 BF16 float KV，不访问 paged cache
            - FIA 调用无需 antiquant 参数，KV 就是普通 BF16 张量
            - actual_seq_lengths_kv == actual_seq_lengths_q（KV 和 Q 等长，无历史）

        PrefillCacheHit（前缀命中，部分 token 已在 KV cache 中）：
            - 历史 token 存储在 NZ 格式 INT8 paged KV cache 中
            - CANN 9.0.0 FIA 支持 TND + NZ + block_table + antiquant_scale 组合，
              内核自动从 paged cache 读取、反量化、合并注意力，无需 Python 层额外处理
            - actual_seq_lengths_kv 使用原始 seq_lens_list（非累积和），
              与文件中 _get_fia_params 及其他 TND + block_table 路径语义一致

        参数：
            query        : 当前 batch 的 Query，shape [padded_tokens, num_heads, head_size]，BF16
            key          : BF16 float Key（仅 PrefillNoCache 真正使用；PrefillCacheHit 时此参数不传给 FIA）
            value        : BF16 float Value，同 key
            attn_metadata: 推理元数据，其中：
                           attn_state         → PrefillNoCache 或 PrefillCacheHit
                           actual_seq_lengths_q → Q 的累积长度列表
                           seq_lens           → 每个请求的 KV 总长张量（用于计算 batch_size）
                           seq_lens_list      → 每个请求的 KV 总长列表（原始长度，传给 FIA）
                           block_tables       → paged KV cache 块索引表（PrefillCacheHit 使用）
                           attn_mask          → causal attention mask
            output       : 预分配的输出张量，结果原地写入
            layer        : attention 层，提供 _c8_k_aq_scale 等量化参数

        返回：
            output       : 填充完毕的输出张量
        """
        self._prepare_c8_scales(layer, query.device)

        actual_seq_qlen = attn_metadata.actual_seq_lengths_q  # Q 长度累积和
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]  # 本 batch 实际 token 总数
        # 截取有效 token 行，去掉 vLLM 框架预分配的 padding 部分
        query = query[:num_tokens]

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            # ── PrefillNoCache：直接使用 BF16 float KV，不需要 paged cache ─────────
            if self.attn_type != AttentionType.ENCODER_DECODER:
                # 自回归（decoder-only）模型：KV 和 Q 来自同一段 token，等长截取
                # encoder-decoder 模型的 K/V 来自 encoder 侧，长度可能不同，不截取
                key = key[:num_tokens]
                value = value[:num_tokens]
            # PrefillNoCache 时 KV 总长 == Q 长（无历史），两者相同
            actual_seq_lengths_kv = actual_seq_qlen
            attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=None,       # 无 paged cache
                input_layout="TND",
                block_size=128,         # PrefillNoCache 不使用 paged KV，block_size 填默认值即可
                actual_seq_lengths=actual_seq_qlen,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,          # 下三角 causal mask
            )
        else:
            # ── PrefillCacheHit：NZ INT8 paged KV + FIA 内核 perchannel 反量化 ─────
            # CANN 9.0.0 的 FIA 支持 TND + NZ + page_attention + antiquant_scale 的组合，
            # 无需在 Python 层 gather token、做手动 dequant，所有操作在内核中一次完成。
            key_nz, value_nz, block_size = self._get_kv_cache_nz()

            # seq_lens 是 tensor，其行数即为本 batch 的请求数
            batch_size = attn_metadata.seq_lens.shape[0]
            # block_tables 可能预分配了更多行，只取与请求数对应的行
            block_table = attn_metadata.block_tables[:batch_size, :]

            # seq_lens_list 是原始每请求 KV 长度（非累积和），与文件其他 TND+block_table 路径一致
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
            attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key_nz,    # NZ 格式 INT8 KV，FIA 读 paged cache 并反量化
                value=value_nz,
                key_antiquant_scale=layer._c8_k_aq_scale,   # [KV_N, 1, D]，perchannel 反量化系数
                value_antiquant_scale=layer._c8_v_aq_scale,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=actual_seq_qlen,           # Q 每请求累积长度
                actual_seq_lengths_kv=actual_seq_lengths_kv,  # KV 每请求原始长度（含历史）
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                key_antiquant_mode=0,   # perchannel 模式
                value_antiquant_mode=0,
                inner_precise=1,        # NZ 格式必须开启高性能模式
                sparse_mode=3,
            )

        # TND 输出 shape 是 [num_tokens, num_heads * head_size]，reshape 为 [T, N, D]
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output
        return output
