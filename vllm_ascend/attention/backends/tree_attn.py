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
"""Ascend NPU Tree Attention Backend for speculative_token_tree."""

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

import torch
import torch_npu

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# NPU kernel requires mask padding to 2048x2048
PAD_SIZE = 2048


# Local AscendAttentionState to avoid circular imports
class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@dataclass
class AscendMetadataForTree:
    """Simplified AscendMetadata for tree attention.

    Avoids circular dependency with attention_v1.py.
    """
    # Basic Properties
    attn_mask: torch.Tensor | None = None
    attn_state: "AscendAttentionState" = AscendAttentionState.ChunkedPrefill

    # Number of tokens excluding padding.
    num_actual_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0
    num_prefill_tokens: int = 0

    # Sequence lengths
    seq_lens: torch.Tensor = None
    query_start_loc: torch.Tensor = None
    max_query_len: int | None = None
    max_seq_len: int = 0

    # KV Cache Related Properties
    block_tables: torch.Tensor = None
    slot_mapping: torch.Tensor = None


class AscendTreeAttentionBackend(AttentionBackend):
    """Tree Attention Backend for Ascend NPU."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "TREE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AscendTreeAttentionImpl"]:
        return AscendTreeAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_builder_cls() -> type["AscendTreeAttentionMetadataBuilder"]:
        return AscendTreeAttentionMetadataBuilder

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class AscendTreeAttentionMetadata(AscendMetadataForTree):
    """Extends AscendMetadataForTree with tree attention fields.

    tree_attn_bias: GPU-format float32 mask (0=attend, -inf=block), for reference
    tree_attn_mask: NPU-format int8 mask (0=attend, 1=block, pad 2048x2048)
    """
    tree_attn_bias: torch.Tensor | None = None
    tree_attn_mask: torch.Tensor | None = None

    # Cached Prefill/decode metadata.
    _cached_prefill_metadata: "AscendTreeAttentionMetadata | None" = field(
        default=None, repr=False
    )
    _cached_decode_metadata: "AscendTreeAttentionMetadata | None" = field(
        default=None, repr=False
    )

    @property
    def prefill_metadata(self) -> "AscendTreeAttentionMetadata | None":
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        q_start_loc = self.query_start_loc[self.num_decodes:]
        q_seqlens = torch.diff(q_start_loc)
        kv_seqlens = self.seq_lens[self.num_decodes:]

        self._cached_prefill_metadata = AscendTreeAttentionMetadata(
            num_actual_tokens=self.num_prefill_tokens,
            max_query_len=int(q_seqlens.max().item()),
            query_start_loc=q_start_loc - q_start_loc[0],
            max_seq_len=int(kv_seqlens.max().item()),
            seq_lens=kv_seqlens,
            block_tables=self.block_tables[self.num_decodes:],
            slot_mapping=self.slot_mapping[self.num_decode_tokens:],
            tree_attn_mask=None,  # prefill does not need tree mask
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> "AscendTreeAttentionMetadata | None":
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata

        q_start_loc = self.query_start_loc[:self.num_decodes + 1]
        q_seqlens = torch.diff(q_start_loc)
        kv_seqlens = self.seq_lens[:self.num_decodes]

        self._cached_decode_metadata = AscendTreeAttentionMetadata(
            num_actual_tokens=self.num_decode_tokens,
            max_query_len=int(q_seqlens.max().item()),
            query_start_loc=q_start_loc,
            max_seq_len=int(kv_seqlens.max().item()),
            seq_lens=kv_seqlens,
            block_tables=self.block_tables[:self.num_decodes],
            slot_mapping=self.slot_mapping[:self.num_decode_tokens],
            tree_attn_bias=self.tree_attn_bias,
            tree_attn_mask=self.tree_attn_mask,
        )
        return self._cached_decode_metadata


def _is_ancestor(potential_ancestor: tuple, potential_descendant: tuple) -> bool:
    """Check if potential_ancestor is an ancestor prefix of potential_descendant."""
    if len(potential_ancestor) >= len(potential_descendant):
        return False
    return potential_descendant[:len(potential_ancestor)] == potential_ancestor


def _get_depth_counts(sorted_tree_choices: list[tuple[int, ...]]) -> list[int]:
    """Count the number of nodes at each depth level of the tree."""
    depth_counts = []
    prev_depth = 0
    for path in sorted_tree_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    return depth_counts


def _prepare_tree_attn_bias_gpu(
    sorted_tree_choices: list[tuple[int, ...]],
    depth_counts: list[int],
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build GPU-format tree attention mask (float32, 0=attend, -inf=block).

    Logic is identical to _prepare_tree_attn_bias in upstream vLLM tree_attn.py.
    """
    tree_len = len(sorted_tree_choices) + 1  # +1 for root
    tree_attn_mask = torch.full(
        (tree_len, tree_len), -torch.inf, device=device, dtype=dtype
    )

    # Diagonal: each token attends to itself
    for i in range(tree_len):
        tree_attn_mask[i, i] = 0

    # All tokens attend to root (column 0)
    tree_attn_mask[:, 0] = 0

    # Ancestry: draft tokens attend to their ancestor nodes
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            # Retrieve ancestor position.
            if len(cur_tree_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_tree_choice) - 1):
                ancestor_idx.append(
                    sorted_tree_choices.index(cur_tree_choice[: c + 1]) + 1
                )
            tree_attn_mask[j + start + 1, ancestor_idx] = 0
        start += depth_counts[i]

    return tree_attn_mask


def _convert_tree_mask_for_npu(
    gpu_mask: torch.Tensor,
    pad_size: int = PAD_SIZE,
) -> torch.Tensor:
    """Convert GPU float32 mask to NPU int8 mask.

    GPU: -inf = block, 0 = attend
    NPU: 0 = attend, 1 = block (int8)

    Args:
        gpu_mask: GPU-format mask (tree_len x tree_len), float32
        pad_size: NPU kernel required pad size, default 2048

    Returns:
        NPU-format mask (pad_size x pad_size), int8
    """
    tree_len = gpu_mask.shape[0]
    npu_mask = torch.ones((pad_size, pad_size), dtype=torch.int8)
    # GPU mask where 0 → NPU mask 0 (attend)
    # GPU mask where -inf → NPU mask 1 (block)
    npu_mask[:tree_len, :tree_len] = (gpu_mask == float('-inf')).to(torch.int8)
    return npu_mask


class AscendTreeAttentionMetadataBuilder(
    AttentionMetadataBuilder[AscendTreeAttentionMetadata]
):
    """NPU Tree Attention Metadata Builder.

    Parses speculative_token_tree config and builds NPU-adapted tree mask.
    """

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size
        self.vllm_config = vllm_config
        self.device = device

        # Parse speculative_token_tree config
        spec_config = vllm_config.speculative_config
        spec_token_tree: str | None = None
        if spec_config:
            spec_token_tree = spec_config.speculative_token_tree

        tree_choices: list[tuple[int, ...]] = (
            ast.literal_eval(spec_token_tree)
            if spec_token_tree is not None
            else [(0,)]
        )

        # Build GPU-format tree attention bias
        depth_counts = _get_depth_counts(tree_choices)
        self.tree_attn_bias = _prepare_tree_attn_bias_gpu(
            tree_choices,
            depth_counts,
            dtype=torch.float32,
            device=device,
        )

        # Convert to NPU-format mask
        self.tree_attn_mask = _convert_tree_mask_for_npu(
            self.tree_attn_bias, pad_size=PAD_SIZE
        )

        self.reorder_batch_threshold = self.tree_attn_bias.shape[0]

        # Mask caching: precompute all possible slice masks
        self._mask_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._bias_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._precompute_slice_masks()

    def _precompute_slice_masks(self):
        """Precompute all possible slice masks to avoid redundant computation."""
        tree_len = self.tree_attn_bias.shape[0]
        # Precompute all possible [start:end, start:end] slices
        for start in range(1, tree_len):
            for end in range(start + 1, tree_len + 1):
                key = (start, end)
                sliced_bias = self.tree_attn_bias[start:end, start:end].contiguous()
                self._bias_cache[key] = sliced_bias
                self._mask_cache[key] = _convert_tree_mask_for_npu(
                    sliced_bias, pad_size=PAD_SIZE
                )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendTreeAttentionMetadata:
        decode_threshold = self.tree_attn_bias.shape[0]
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=decode_threshold
            )
        )

        num_actual_tokens = common_attn_metadata.num_actual_tokens
        q_start_loc = common_attn_metadata.query_start_loc
        max_query_len = common_attn_metadata.max_query_len
        kv_seqlens = common_attn_metadata.seq_lens
        max_seq_len = common_attn_metadata.max_seq_len
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        return AscendTreeAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            max_query_len=max_query_len,
            query_start_loc=q_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=kv_seqlens,
            block_tables=block_table,
            slot_mapping=slot_mapping,
            tree_attn_bias=self.tree_attn_bias,
            tree_attn_mask=self.tree_attn_mask,
        )

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> AscendTreeAttentionMetadata:
        """Build attention metadata for a tree level.

        draft_index=0: prefill (root level), use empty bias
        draft_index>0: draft level, slice tree mask (uses cache)
        """
        # Cache the original tree attention bias.
        orig_tree_attn_bias = self.tree_attn_bias
        orig_tree_attn_mask = self.tree_attn_mask

        if draft_index == 0:
            # Use prefill for drafting at the root level.
            self.tree_attn_bias = torch.empty(0)
            self.tree_attn_mask = torch.empty(0, dtype=torch.int8)
        else:
            # Use cached slice mask to avoid redundant computation
            start, end = 1, 1 + common_attn_metadata.max_query_len
            cache_key = (start, end)
            if cache_key in self._bias_cache:
                self.tree_attn_bias = self._bias_cache[cache_key]
                self.tree_attn_mask = self._mask_cache[cache_key]
            else:
                # Cache miss, fall back to on-the-fly computation
                self.tree_attn_bias = self.tree_attn_bias[start:end, start:end].contiguous()
                self.tree_attn_mask = _convert_tree_mask_for_npu(
                    self.tree_attn_bias, pad_size=PAD_SIZE
                )

        # Build attention metadata.
        attn_metadata = self.build(0, common_attn_metadata, fast_build=True)

        # Reset the tree attention bias to the original value.
        self.tree_attn_bias = orig_tree_attn_bias
        self.tree_attn_mask = orig_tree_attn_mask
        return attn_metadata


class AscendTreeAttentionImpl(AttentionImpl):
    """NPU Tree Attention Implementation.

    Uses npu_fused_infer_attention_score for attention computation.
    - Prefill: BSH layout + sparse_mode=0
    - Decode: TND layout + sparse_mode=3 + tree_attn_mask
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if logits_soft_cap is None:
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and encoder/decoder cross-attention "
                "are not implemented for AscendTreeAttentionImpl."
            )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Update KV cache."""
        from vllm import _custom_ops as ops

        key_cache, value_cache = kv_cache.unbind(0)
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AscendTreeAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with TreeAttention on NPU.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
            output: shape = [num_tokens, num_heads * head_size]

        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for "
                "AscendTreeAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        key_cache, value_cache = kv_cache.unbind(0)

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens

        # Prefill: BSH layout + sparse_mode=0
        if prefill_meta := attn_metadata.prefill_metadata:
            self._forward_prefill(
                query=query[num_decode_tokens:num_actual_tokens],
                key=key[num_decode_tokens:num_actual_tokens],
                value=value[num_decode_tokens:num_actual_tokens],
                output=output[num_decode_tokens:num_actual_tokens],
                prefill_meta=prefill_meta,
                layer=layer,
            )

        # Decode: TND layout + sparse_mode=3 + tree_attn_mask
        if decode_meta := attn_metadata.decode_metadata:
            self._forward_decode(
                query=query[:num_decode_tokens],
                key_cache=key_cache,
                value_cache=value_cache,
                output=output[:num_decode_tokens],
                decode_meta=decode_meta,
                layer=layer,
            )

        return output

    def _forward_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        prefill_meta: AscendTreeAttentionMetadata,
        layer: torch.nn.Module,
    ):
        """Prefill phase: BSH layout + sparse_mode=0.

        Uses full custom mask if available.
        """
        # BSH layout: [batch, seq, hidden]
        # Reshape [tokens, heads, head_size] to [1, tokens, heads * head_size]
        batch_size = prefill_meta.query_start_loc.shape[0] - 1
        q_bsh = query.view(batch_size, -1, self.num_heads * self.head_size)
        k_bsh = key.view(batch_size, -1, self.num_kv_heads * self.head_size)
        v_bsh = value.view(batch_size, -1, self.num_kv_heads * self.head_size)

        # Standard prefill mask (causal mask)
        attn_mask = prefill_meta.attn_mask

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=q_bsh,
            key=k_bsh,
            value=v_bsh,
            atten_mask=attn_mask,
            input_layout="BSH",
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            scale=self.scale,
            sparse_mode=0,
        )

        # Flatten output back to [tokens, heads * head_size]
        output.copy_(attn_output.view(-1, self.num_heads * self.head_size))

    def _forward_decode(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        decode_meta: AscendTreeAttentionMetadata,
        layer: torch.nn.Module,
    ):
        """Decode phase: TND layout + sparse_mode=3 + tree_attn_mask.

        Uses tree attention mask to restrict attend scope.
        Supports ACL Graph: when is_draft_model=True, precompute workspace and use
        .out() variant to avoid extra memory allocation.
        """
        num_tokens = decode_meta.num_actual_tokens
        query = query[:num_tokens]

        # Prepare actual_seq_lengths (use tensor to avoid CPU-GPU sync)
        actual_seq_lengths = decode_meta.query_start_loc[1:].to(torch.int64)
        actual_seq_lengths_kv = decode_meta.seq_lens.to(torch.int64)

        # Use tree attention mask
        tree_mask = decode_meta.tree_attn_mask
        if tree_mask is None:
            # Fall back to standard causal mask if tree mask is unavailable
            tree_mask = decode_meta.attn_mask

        block_size = key_cache.shape[1]

        # ACL Graph: when in draft model context, precompute workspace and use .out()
        from vllm_ascend.ascend_forward_context import _EXTRA_CTX

        if _EXTRA_CTX.is_draft_model:
            from vllm_ascend.compilation.acl_graph import (
                get_draft_graph_params,
                update_draft_graph_params_workspaces,
            )

            graph_params = get_draft_graph_params()
            num_input_tokens = num_tokens
            workspace = graph_params.workspaces.get(num_input_tokens)
            if workspace is None:
                workspace = (
                    torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                        query=query,
                        key=key_cache,
                        value=value_cache,
                        atten_mask=tree_mask,
                        block_table=decode_meta.block_tables,
                        input_layout="TND",
                        block_size=block_size,
                        actual_seq_lengths=actual_seq_lengths,
                        actual_seq_lengths_kv=actual_seq_lengths_kv,
                        num_key_value_heads=self.num_kv_heads,
                        num_heads=self.num_heads,
                        sparse_mode=3,
                        scale=self.scale,
                    )
                )
                update_draft_graph_params_workspaces(
                    num_input_tokens, workspace
                )

            attn_output = torch.empty_like(query)
            softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
            torch_npu.npu_fused_infer_attention_score.out(
                query=query,
                key=key_cache,
                value=value_cache,
                atten_mask=tree_mask,
                block_table=decode_meta.block_tables,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=3,
                scale=self.scale,
                out=(attn_output, softmax_lse),
            )
        else:
            attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key_cache,
                value=value_cache,
                atten_mask=tree_mask,
                block_table=decode_meta.block_tables,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )

        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
