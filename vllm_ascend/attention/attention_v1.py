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
from typing import ClassVar

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
import vllm.envs as envs_vllm
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, CrossAttentionSpec

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
from vllm_ascend.utils import vllm_version_is, weak_ref_tensors, prefill_context_parallel_enable

# isort: off
if prefill_context_parallel_enable():
    from vllm.distributed import get_pcp_group

if vllm_version_is("0.13.0"):
    from vllm.v1.attention.backends.utils import AttentionCGSupport, AttentionMetadataBuilder
    from vllm.attention.backends.abstract import (  # type: ignore
        AttentionBackend,
        AttentionImpl,
        AttentionLayer,
        AttentionType,
    )
    from vllm.attention.backends.registry import (  # type: ignore
        AttentionBackendEnum,
        register_backend,
    )
else:
    from vllm.v1.attention.backend import (  # type: ignore
        AttentionBackend,
        AttentionCGSupport,
        AttentionImpl,
        AttentionLayer,
        AttentionType,
        AttentionMetadataBuilder,
    )
    from vllm.v1.attention.backends.registry import (  # type: ignore
        AttentionBackendEnum,
        register_backend,
    )
# isort: on

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
    def get_supported_block_size() -> list[int]:
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

    # The sequence length per sequence. Sequence length means the computed
    # tokens + new tokens (is None if it is a decoding).
    # (batch_size,)
    # TODO(Angazenn): The following parameters are quite redundant and
    # contains similar information (such as seq_lens seq_lens_list). We
    # should simplified these parameters once attention schema in vLLM-Ascend
    # is unified.
    seq_lens: torch.Tensor = None
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

    use_hybrid_attn: bool = False

    pcp_unpad_mask: torch.Tensor = None


class AscendAttentionMetadataBuilder(AttentionMetadataBuilder[AscendMetadata]):
    """
    Builder for constructing AscendMetadata from CommonAttentionMetadata.

    Handles attention mask generation and metadata preparation for
    Ascend FlashAttention backend.
    """

    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: ClassVar[int] = 1

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
            self.model_config.max_model_len, AscendAttentionBackend.get_supported_block_size()[0]
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

        AscendAttentionMetadataBuilder.reorder_batch_threshold = self.decode_threshold

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
        seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]

        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]
        if isinstance(self.kv_cache_spec, CrossAttentionSpec):
            seq_lens = common_attn_metadata.seq_lens
            slot_mapping = common_attn_metadata.slot_mapping.to(torch.int32)
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
            seq_lens_list=seq_lens.tolist(),
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            swa_mask=swa_mask,
            attn_state=attn_state,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            prefill=prefill_metadata,
            decode_meta=decode_metadata,
            use_hybrid_attn=use_hybrid_attn,
            pcp_unpad_mask=pcp_unpad_mask,
            causal=common_attn_metadata.causal,
            model_runner_type=self.model_config.runner_type,
        )
        return attn_metadata

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):
        if attn_state in (AscendAttentionState.DecodeOnly, AscendAttentionState.ChunkedPrefill):
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
        forward_context = get_forward_context()
        if forward_context.is_draft_model:
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
                sparse_mode=3,
                scale=self.scale,
            )
            if forward_context.is_draft_model:
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
            sparse_mode=3,
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
        forward_context: ForwardContext = get_forward_context()
        num_tokens = query.shape[0]
        if forward_context.capturing:
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

    def _get_fia_params(self, key: torch.Tensor, value: torch.Tensor, attn_metadata: AscendMetadata):
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

        output, _ = torch_npu.npu_fused_infer_attention_score(
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

        output = output.view(batch_size, self.num_heads, self.head_size)
        return output

    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        forward_context: ForwardContext = get_forward_context()
        # we inherit ForwardContext in model runner v2, when enable model
        # runner v2, there is not capturing attribute in forward_context,
        # just use getattr to avoid attribute error.
        if getattr(forward_context, "capturing", False):
            attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output
        if (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and self.sliding_window is not None
            and attn_metadata.seq_lens.shape[0] == query.size(0)
        ):
            return self._forward_fia_slidingwindow(query, attn_metadata, output)
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]
        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]
        # Get workspace from cache or calculate it if not present.
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
        if self.sliding_window is not None and attn_metadata.seq_lens.shape[
                0] == query.size(0):
            batch_size = attn_metadata.seq_lens.shape[0]
            block_size = 128
            query = query.view(batch_size, 1, self.num_heads * self.head_size)
            key = self.key_cache
            value = self.value_cache
            if self.key_cache is not None and self.value_cache is not None:
                block_size = self.key_cache.shape[1]
                key = self.key_cache.flatten(2, 3).contiguous()
                value = self.value_cache.flatten(2, 3).contiguous()

            output, _ = torch_npu.npu_fused_infer_attention_score(
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
                actual_seq_lengths_kv=attn_metadata.seq_lens)

            output = output.view(batch_size, self.num_heads, self.head_size)
        else:
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                block_table=attn_metadata.block_tables,
                context_lens=attn_metadata.seq_lens,
                out=output)
        return output

    def _attention_with_nomask_and_mask(self, q: torch.Tensor,
                                        q_seqlens: list[int],
                                        k_nomask: torch.Tensor,
                                        v_nomask: torch.Tensor,
                                        kv_seqlens_nomask: list[int],
                                        k_mask: torch.Tensor,
                                        v_mask: torch.Tensor,
                                        kv_seqlens_mask: list[int],
                                        mask: torch.Tensor,
                                        attn_metadata) -> torch.Tensor:
        # nomask Attention
        if k_nomask is not None:
            attn_out_nomask, attn_lse_nomask = torch.ops.npu.npu_fused_infer_attention_score(
                q,
                k_nomask,
                v_nomask,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                atten_mask=None,
                scale=self.scale,
                sparse_mode=0,
                antiquant_mode=0,
                antiquant_scale=None,
                softmax_lse_flag=True,
                actual_seq_lengths_kv=kv_seqlens_nomask,
                actual_seq_lengths=q_seqlens)

        # mask Attention
        attn_out_mask, attn_lse_mask = torch.ops.npu.npu_fused_infer_attention_score(
            q,
            k_mask,
            v_mask,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="TND",
            atten_mask=mask,
            scale=self.scale,
            sparse_mode=3,
            antiquant_mode=0,
            antiquant_scale=None,
            softmax_lse_flag=True,
            actual_seq_lengths_kv=kv_seqlens_mask,
            actual_seq_lengths=q_seqlens)
        # update
        output = attn_out_mask
        attn_lse = attn_lse_mask
        if k_nomask is not None:
            if attn_metadata.prefill is not None and attn_metadata.prefill.chunked_context is None:
                output = self._npu_attn_out_lse_update(attn_lse_mask,
                                                       attn_lse_nomask,
                                                       attn_out_mask,
                                                       attn_out_nomask)
                attn_lse = None
            else:
                output, attn_lse = self._update_out_and_lse(
                    torch.stack([attn_out_nomask, attn_out_mask], dim=0),
                    torch.stack([attn_lse_nomask, attn_lse_mask], dim=0))

        return output, attn_lse

    def _npu_attn_out_lse_update(self, attn_lse_mask, attn_lse_nomask,
                                 attn_out_mask, attn_out_nomask):
        T = attn_out_mask.shape[0]
        N = attn_out_mask.shape[1]
        D = attn_out_mask.shape[2]
        attn_out_mask, attn_lse_mask = self._out_lse_reshape(
            attn_out_mask, attn_lse_mask)
        attn_out_nomask, attn_lse_nomask = self._out_lse_reshape(
            attn_out_nomask, attn_lse_nomask)
        attn_out_mask = attn_out_mask.to(torch.float32)
        attn_out_nomask = attn_out_nomask.to(torch.float32)
        attn_lse_mask = attn_lse_mask.to(torch.float32)
        attn_lse_nomask = attn_lse_nomask.to(torch.float32)
        attn_output = [attn_out_nomask, attn_out_mask]
        attn_lse = [attn_lse_nomask, attn_lse_mask]
        update_type = 0
        output, _ = torch_npu.npu_attention_update(attn_lse, attn_output,
                                                   update_type)
        output = output.view(T, N, D)
        return output

    def _forward_prefill_cp(self, query: torch.Tensor, key: torch.Tensor,
                            value: torch.Tensor,
                            attn_metadata: AscendMetadata, layer_idx: int) -> torch.Tensor:
        assert attn_metadata is not None
        assert attn_metadata.prefill is not None
        assert attn_metadata.prefill.pcp_metadata is not None
        # Use precomputed indices from the metadata (already converted to tensors and on device)
        q_head_idx = attn_metadata.prefill.pcp_metadata.q_head_idx
        q_tail_idx = attn_metadata.prefill.pcp_metadata.q_tail_idx
        kv_with_q_head_nomask_idx = attn_metadata.prefill.pcp_metadata.kv_with_q_head_nomask_idx
        kv_with_q_head_mask_idx = attn_metadata.prefill.pcp_metadata.kv_with_q_head_mask_idx
        kv_with_q_tail_nomask_idx = attn_metadata.prefill.pcp_metadata.kv_with_q_tail_nomask_idx
        kv_with_q_tail_mask_idx = attn_metadata.prefill.pcp_metadata.kv_with_q_tail_mask_idx
        attn_mask_seqlens = attn_metadata.prefill.pcp_metadata.attn_mask_seqlens
        head_attn_nomask_seqlens = attn_metadata.prefill.pcp_metadata.head_attn_nomask_seqlens
        tail_attn_nomask_seqlens = attn_metadata.prefill.pcp_metadata.tail_attn_nomask_seqlens
        mask = attn_metadata.prefill.pcp_metadata.pcp_prefill_mask
        use_hybrid_attn = attn_metadata.use_hybrid_attn

        if use_hybrid_attn:
            fa_query_idx = attn_metadata.prefill.pcp_metadata.pcp_fa_query_idx
            query = torch.index_select(query, 0, fa_query_idx)

        # 1. Attention calculation in the first half of Q in load balancing
        output_heads, lse_heads = self._attention_with_nomask_and_mask(
            q=torch.index_select(query, 0, q_head_idx),
            q_seqlens=attn_mask_seqlens,
            k_nomask=torch.index_select(key, 0, kv_with_q_head_nomask_idx)
            if self.pcp_rank > 0 else None,
            v_nomask=torch.index_select(value, 0, kv_with_q_head_nomask_idx)
            if self.pcp_rank > 0 else None,
            kv_seqlens_nomask=head_attn_nomask_seqlens,
            k_mask=torch.index_select(key, 0, kv_with_q_head_mask_idx),
            v_mask=torch.index_select(value, 0, kv_with_q_head_mask_idx),
            kv_seqlens_mask=attn_mask_seqlens,
            mask=mask,
            attn_metadata=attn_metadata)

        # 2. the Attention calculation in the latter half of Q in load balancing
        # pcp_rank0: Q3*KV0~KV2 + Q3*KV3
        # pcp_rank1: Q2*KV0~KV1 + Q2*KV2
        output_tails, lse_tails = self._attention_with_nomask_and_mask(
            q=torch.index_select(query, 0, q_tail_idx),
            q_seqlens=attn_mask_seqlens,
            k_nomask=torch.index_select(key, 0, kv_with_q_tail_nomask_idx),
            v_nomask=torch.index_select(value, 0, kv_with_q_tail_nomask_idx),
            kv_seqlens_nomask=tail_attn_nomask_seqlens,
            k_mask=torch.index_select(key, 0, kv_with_q_tail_mask_idx),
            v_mask=torch.index_select(value, 0, kv_with_q_tail_mask_idx),
            kv_seqlens_mask=attn_mask_seqlens,
            mask=mask,
            attn_metadata=attn_metadata)

        q_full_idx = attn_metadata.prefill.pcp_metadata.q_full_idx
        output = torch.index_select(
            torch.cat([output_heads, output_tails], dim=0), 0, q_full_idx)
        attn_lse = None
        if attn_metadata.prefill is not None and attn_metadata.prefill.chunked_context is not None:
            attn_lse = torch.index_select(
                torch.cat([lse_heads, lse_tails], dim=0), 0, q_full_idx)
        return output, attn_lse

    def _out_lse_reshape(self, attn_out: torch.Tensor,
                         attn_lse: torch.Tensor) -> torch.Tensor:
        attn_out = attn_out.contiguous().view(
            attn_out.shape[0] * attn_out.shape[1], attn_out.shape[2])
        attn_lse = attn_lse.contiguous().view(
            attn_lse.shape[0] * attn_lse.shape[1] * attn_lse.shape[2])
        return attn_out, attn_lse

    def _npu_attention_update(
            self, attn_out_lse_list: List[torch.Tensor]) -> torch.Tensor:
        update_type = 0

        batch = attn_out_lse_list[0].shape[0]
        num_heads = attn_out_lse_list[0].shape[1]
        head_dim = attn_out_lse_list[0].shape[2] - 1

        attn_out_split_cp = []
        attn_lse_split_cp = []

        for i in attn_out_lse_list:
            attn_out_allgather, attn_lse_allgather = self._out_lse_reshape(
                *torch.split(i, [self.head_size, 1], dim=-1))
            attn_out_split_cp.append(attn_out_allgather)
            attn_lse_split_cp.append(attn_lse_allgather)

        attn_out, attn_lse = torch_npu.npu_attention_update(
            attn_lse_split_cp, attn_out_split_cp, update_type)
        attn_out = attn_out.view(batch, num_heads, head_dim)

        return attn_out

    def _forward_decode_pcp_dcp(self, query: torch.Tensor,
                                attn_metadata: AscendMetadata) -> torch.Tensor:
        assert self.key_cache is not None
        assert self.value_cache is not None

        if self.dcp_size > 1:
            query = get_dcp_group().all_gather(query, 1)
            num_heads = self.num_heads * self.dcp_size
        else:
            num_heads = self.num_heads

        k_nope = self.key_cache.view(self.key_cache.shape[0],
                                     self.key_cache.shape[1], -1)
        value = self.value_cache.view(self.key_cache.shape[0],
                                      self.key_cache.shape[1], -1)
        common_kwargs = {
            'num_heads':
            num_heads,
            'num_key_value_heads':
            self.num_kv_heads,
            'input_layout':
            'TND',
            'atten_mask':
            None,
            'scale':
            self.scale,
            'antiquant_mode':
            0,
            'antiquant_scale':
            None,
            'softmax_lse_flag':
            True,
            'block_table':
            attn_metadata.decode_meta.block_tables,
            'block_size':
            self.key_cache.shape[1],
            'actual_seq_lengths_kv':
            attn_metadata.decode_meta.
            num_computed_tokens_of_pcp_dcp[:, self.pcp_rank, self.dcp_rank],
            'actual_seq_lengths':
            attn_metadata.actual_seq_lengths_q[:attn_metadata.num_decodes],
        }
        graph_params = get_graph_params()
        forward_context: ForwardContext = get_forward_context()
        if forward_context.capturing:
            stream = torch_npu.npu.current_stream()

            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[num_tokens].append(event)

            workspace = graph_params.workspaces.get(num_tokens)
            if workspace is None:
                workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                    query, k_nope, value, **common_kwargs)
                update_graph_params_workspaces(num_tokens,
                                               weak_ref_tensors(workspace))
            attn_out = torch.empty_like(query)
            attn_lse = torch.empty((num_tokens, num_heads, 1),
                                   dtype=torch.float,
                                   device=query.device)

            graph_params.attn_params[num_tokens].append((
                weak_ref_tensors(query), weak_ref_tensors(k_nope),
                weak_ref_tensors(value), self.num_heads, self.num_kv_heads,
                self.scale, attn_metadata.block_tables,
                self.key_cache.shape[1], attn_metadata.decode_meta.
                num_computed_tokens_of_pcp_dcp[:, self.pcp_rank,
                                               self.dcp_rank],
                attn_metadata.actual_seq_lengths_q[:attn_metadata.num_decodes],
                weak_ref_tensors(attn_out), weak_ref_tensors(attn_lse),
                self.dcp_size, self.pcp_rank, self.dcp_rank))
            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_fused_infer_attention_score.out(
                query,
                k_nope,
                value,
                **common_kwargs,
                workspace=workspace,
                out=[attn_out, attn_lse])
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
        else:
            attn_out, attn_lse = torch_npu.npu_fused_infer_attention_score(
                query, k_nope, value, **common_kwargs)

        out_mask = attn_metadata.decode_meta.batch_seq_mask[:, None,
                                                            None].expand_as(
                                                                attn_out)
        attn_out = torch.where(out_mask, 0, attn_out)

        lse_mask = attn_metadata.decode_meta.batch_seq_mask[:, None,
                                                            None].expand_as(
                                                                attn_lse)
        attn_lse = torch.where(lse_mask, -torch.inf, attn_lse)

        attn_out_lse_list = []
        # Concat out&lse: [bs,num_heads,v_head_dim] + [bs,num_heads,1] -> [bs,num_heads,v_head_dim+1]
        attn_out_lse = torch.cat([attn_out, attn_lse], dim=-1)
        if self.dcp_size > 1:
            # permute: [bs, num_heads, v_head_dim+1] -> [num_heads, v_head_dim+1, bs]
            attn_out_lse = attn_out_lse.permute([1, 2, 0]).contiguous()
            attn_out_lse_all2all = torch.empty_like(attn_out_lse)
            dist.all_to_all_single(attn_out_lse_all2all,
                                   attn_out_lse,
                                   group=self.dcp_group)
            # permute: [num_heads, v_head_dim+1, bs] -> [bs, num_heads, v_head_dim+1]
            attn_out_lse_all2all = attn_out_lse_all2all.permute([2, 0, 1])
            if self.pcp_size > 1:
                attn_out_lse = attn_out_lse_all2all.contiguous()
            attn_out_lse_list = list(
                torch.chunk(attn_out_lse_all2all, self.dcp_size, dim=1))

        if self.pcp_size > 1:
            # AllGather out&lse within CP group
            attn_out_lse_list = [
                torch.empty_like(attn_out_lse) for _ in range(self.pcp_size)
            ]
            dist.all_gather(attn_out_lse_list,
                            attn_out_lse,
                            group=self.pcp_group)
        if self.dcp_size > 1 and self.pcp_size > 1:
            attn_out_lse_list_pcp_dcp = []
            for s in attn_out_lse_list:
                attn_out_lse_list_split = list(
                    torch.chunk(s, self.dcp_size, dim=1))
                attn_out_lse_list_pcp_dcp += attn_out_lse_list_split
            attn_out_lse_list = attn_out_lse_list_pcp_dcp
        # Update out&lse
        attn_out = self._npu_attention_update(attn_out_lse_list)
        return attn_out

    def _update_out_and_lse(self, out_list: torch.Tensor,
                            lse_list: torch.Tensor) -> torch.Tensor:
        """LSE_final = log(sum(exp(LSE_i))), O_final = sum(exp(LSE_i - LSE_final) * O_i)
        Args:
            out_list: shape = [N, batch_size, num_heads, head_size]
            lse_list: shape = [N, batch_size, num_heads, 1]
        Returns:
            out_final: shape = [batch_size, num_heads, head_size]
            lse_final: shape = [batch_size, num_heads, 1]
        """
        lse_final = torch.logsumexp(lse_list, dim=0, keepdim=False)
        out_final = torch.sum(torch.exp(lse_list - lse_final) * out_list,
                              dim=0)
        return out_final, lse_final

    def _forward_pcp_dcp(self, query: torch.Tensor, key: torch.Tensor,
                         value: torch.Tensor, kv_cache: tuple[torch.Tensor],
                         attn_metadata: AscendMetadata,
                         output: torch.Tensor, layer_idx: int) -> torch.Tensor:
        assert attn_metadata is not None
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens

        if has_decode:
            if not attn_metadata.use_hybrid_attn or not has_prefill:
                decode_query = query[:num_decode_tokens]
            else:
                decode_query = query[:num_decode_tokens * self.pcp_size: self.pcp_size]
            output_decode = self._forward_decode_pcp_dcp(
                decode_query, attn_metadata)
            output[:num_decode_tokens] = output_decode
        if has_prefill:
            assert attn_metadata.prefill is not None

            num_actual_tokens_pcp_padded = attn_metadata.num_actual_tokens_pcp_padded // self.pcp_size
            key = key[self.pcp_size * num_decode_tokens:]
            value = value[self.pcp_size * num_decode_tokens:]
            if attn_metadata.use_hybrid_attn:
                prefill_query = query[self.pcp_size * num_decode_tokens:]
            else:
                prefill_query = query[
                    num_decode_tokens : num_actual_tokens_pcp_padded]
            
            if self.pcp_size > 1:
                # Scenario of Enabling PCP or PCP&DCP
                attn_output_prefill, attn_lse_prefill = self._forward_prefill_cp(
                    prefill_query, key, value, attn_metadata, layer_idx)
            else:
                # Scenario of Enabling DCP Individually
                attn_output_prefill, attn_lse_prefill = torch.ops.npu.npu_fused_infer_attention_score(
                    prefill_query,
                    key,
                    value,
                    num_heads=self.num_heads,
                    num_key_value_heads=self.num_kv_heads,
                    input_layout="TND",
                    atten_mask=attn_metadata.attn_mask,
                    scale=self.scale,
                    sparse_mode=3,
                    antiquant_mode=0,
                    antiquant_scale=None,
                    softmax_lse_flag=True,
                    actual_seq_lengths_kv=attn_metadata.prefill.
                    actual_seq_lengths_q,
                    actual_seq_lengths=attn_metadata.prefill.
                    actual_seq_lengths_q)

            self._process_chunk_prefill(attn_output_prefill, attn_lse_prefill,
                                        kv_cache, prefill_query, attn_metadata)
            if attn_metadata.use_hybrid_attn:
                # layer_idx != num_layers - 1
                pcp_allgather_restore_idx = attn_metadata.prefill.pcp_allgather_restore_idx
                attn_output_prefill = get_pcp_group().all_gather(attn_output_prefill.contiguous(), dim=0)
                attn_output_prefill = torch.index_select(attn_output_prefill, 0, pcp_allgather_restore_idx)
                fla_padding = attn_output_prefill.shape[0] + num_decode_tokens - output.shape[0]
                output = F.pad(output, pad=(0, 0, 0, 0, 0, fla_padding), mode='constant', value=0)
            output[num_decode_tokens: attn_output_prefill.shape[0] +
                   num_decode_tokens] = attn_output_prefill
        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        _: torch.Tensor,
    ) -> torch.Tensor:
        assert attn_metadata is not None

        if attn_metadata.causal:
            # use sparse_mode 3 in causal scenario
            return torch_npu.npu_fusion_attention(
                query=query,
                key=key,
                value=value,
                head_num=self.num_heads,
                input_layout="TND",
                scale=self.scale,
                sparse_mode=3,
                atten_mask=attn_metadata.attn_mask,
                actual_seq_qlen=attn_metadata.actual_seq_lengths_q,
                actual_seq_kvlen=attn_metadata.actual_seq_lengths_q,
            )[0]
        else:
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

    def reshape_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
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
                slot_mapping=slots[: attn_metadata.num_actual_tokens] if not encoder_decoder else slots,
            )
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()
        return key, value

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
            output = self.forward_fused_infer_attention(query, key, value, attn_metadata, output)

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
        if self.attn_type != AttentionType.DECODER and self.attn_type != AttentionType.ENCODER_ONLY:
            raise NotImplementedError("Encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "PallasAttentionBackendImpl")
        layer_idx = extract_layer_index(layer.layer_name)

        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        num_decode_tokens = attn_metadata.num_decode_tokens
        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        output_padded = output

        if len(kv_cache) > 1:
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

            if has_decode:
                if self.pcp_size * self.dcp_size > 1 and not attn_metadata.use_hybrid_attn:
                    slot_mapping = attn_metadata.slot_mapping[:num_decode_tokens * self.pcp_size: self.pcp_size]
                else:
                    slot_mapping = attn_metadata.slot_mapping[:num_decode_tokens]
                torch_npu._npu_reshape_and_cache(
                    key=key[:num_decode_tokens],
                    value=value[:num_decode_tokens],
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    slot_indices=slot_mapping)

            if has_prefill:
                if self.pcp_size > 1:
                    if not attn_metadata.use_hybrid_attn:
                        num_actual_tokens_pcp_padded = attn_metadata.num_actual_tokens_pcp_padded // self.pcp_size
                        kv = torch.cat([key, value], dim=-1)
                        all_kv = get_pcp_group().all_gather(
                            kv[:num_actual_tokens_pcp_padded].contiguous(), dim=0)
                        pcp_allgather_restore_idx = attn_metadata.prefill.pcp_allgather_restore_idx if attn_metadata.prefill else None
                        all_kv = torch.index_select(all_kv, 0,
                                                    pcp_allgather_restore_idx)
                        key, value = all_kv.split([self.head_size, self.head_size],
                                                dim=-1)
                    else:
                        num_actual_tokens_pcp_padded = attn_metadata.num_actual_tokens_pcp_padded
                        pcp_padded_tokens_fla = attn_metadata.prefill.pcp_metadata.pcp_padded_tokens_fla
                        num_tokens_pcp_padded_fla = num_tokens + pcp_padded_tokens_fla

                        qkv_fla = torch.cat([query.reshape(num_tokens, -1), key.reshape(num_tokens, -1), value.reshape(num_tokens, -1)], dim=-1)
                        if pcp_padded_tokens_fla > 0:
                            qkv_fla = F.pad(qkv_fla, pad=(0, 0, 0, pcp_padded_tokens_fla), mode='constant', value=0)
                        all_qkv = get_pcp_group().all_gather(qkv_fla[:num_tokens_pcp_padded_fla].contiguous(), dim=0)

                        pcp_enter_fa_restore_idx = attn_metadata.prefill.pcp_metadata.pcp_enter_fa_restore_idx if attn_metadata.prefill.pcp_metadata else None
                        # 恢复完整序列多batch的顺序
                        actual_qkv = torch.index_select(all_qkv, 0,
                                                    pcp_enter_fa_restore_idx)
                        qkv_fa_padding_workspace = query.new_empty((num_actual_tokens_pcp_padded, (self.num_heads + 2 * self.num_kv_heads) * self.head_size))
                        
                        qkv_fa_padding_workspace[:attn_metadata.num_decode_tokens * self.pcp_size] = actual_qkv[:attn_metadata.num_decode_tokens * self.pcp_size]
                        pcp_unpad_mask = attn_metadata.pcp_unpad_mask[attn_metadata.num_decodes * self.pcp_size:]
                        qkv_fa_padding_workspace[attn_metadata.num_decode_tokens * self.pcp_size:][pcp_unpad_mask] = actual_qkv[attn_metadata.num_decode_tokens * self.pcp_size:]
                        
                        query, key, value = qkv_fa_padding_workspace.split([self.num_heads * self.head_size, self.num_kv_heads * self.head_size, self.num_kv_heads * self.head_size], dim=-1)
                        query = query.reshape(-1, self.num_heads, self.head_size)
                        key = key.reshape(-1, self.num_kv_heads, self.head_size)
                        value = value.reshape(-1, self.num_kv_heads, self.head_size)
                        
                        output_local_padded_tokens_fa = num_actual_tokens_pcp_padded // self.pcp_size - num_tokens
                        if output_local_padded_tokens_fa > 0:
                            output_padded = F.pad(output, pad=(0, 0, 0, 0, 0, output_local_padded_tokens_fa), mode='constant', value=0)

                prefill_cached_start = self.pcp_size * num_decode_tokens
                torch_npu._npu_reshape_and_cache(
                    key=key[prefill_cached_start:attn_metadata.
                            num_actual_tokens_pcp_padded],
                    value=value[prefill_cached_start:attn_metadata.
                                num_actual_tokens_pcp_padded],
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    slot_indices=attn_metadata.
                    slot_mapping[prefill_cached_start:attn_metadata.
                                 num_actual_tokens_pcp_padded])

        forward_context: ForwardContext = get_forward_context()
        if not forward_context.capturing:
            if self.pcp_size * self.dcp_size > 1:
                attn_output = self._forward_pcp_dcp(query, key, value,
                                                kv_cache, attn_metadata,
                                                output_padded, layer_idx)
                output[:num_tokens] = attn_output[:num_tokens]
                # if get_pcp_group().rank_in_group == 1 and get_tp_group().rank_in_group == 0:
                #     print(num_tokens, output.shape, output.mean(1)[:5])
                # if get_pcp_group().rank_in_group == 1 and get_tp_group().rank_in_group == 0 and query.shape[0] < 70 and query.shape[0] != 2:
                #     # print(hidden_states[1:56, :], hidden_states.shape)
                #     # [0,1,...,32,33,...,64,65,66,67]
                #     print(output[:num_tokens], output[:num_tokens].shape)
                return output[:num_tokens]
            if self.attn_type == AttentionType.ENCODER_ONLY:
                attn_output = self._forward_encode(query, key, value,
                                                   attn_metadata, output)
                output[:num_tokens] = attn_output[:num_tokens]
                return output
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                output = self._forward_decode_only(query, attn_metadata,
                                                   output)
            else:
                output = self._forward_prefill(query, key, value,
                                               attn_metadata, output)
                # if get_tp_group().rank_in_group == 0:
                #     print(output.shape, output.mean(1)[:5])
        else:
            attn_output, num_tokens = self.full_graph_attention(
                query, key, value, kv_cache, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
        # if get_tp_group().rank_in_group == 0 and query.shape[0] < 70 and query.shape[0] != 2:
        # #     # print(hidden_states[57:112, :], hidden_states.shape)
        # #     # [0, 1, ..., 32, 33, ..., 64]
        #     print(output[33:], output[33:].shape)

        return output
