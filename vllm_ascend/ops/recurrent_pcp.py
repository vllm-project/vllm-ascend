# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-stable PCP token permutations for recurrent layers."""

from copy import copy

import torch
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.gdn_attn_builder import (
    AscendGDNAttentionBackend,
    AscendGDNAttentionMetadataBuilder,
    AscendGDNHostMetadataBackend,
    AscendGDNHostMetadataBuilder,
)


class PCPRecurrentMetadataBuilderMixin:
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        if pcp_size > 1:
            # Runner capture sizes count local tokens. Recurrence uses the
            # global batch, including requests owned by other decode shards.
            vllm_config = copy(vllm_config)
            vllm_config.compilation_config = copy(vllm_config.compilation_config)
            capture_size = vllm_config.compilation_config.max_cudagraph_capture_size
            if capture_size is not None:
                vllm_config.compilation_config.max_cudagraph_capture_size = capture_size * pcp_size
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        capacity = max(vllm_config.scheduler_config.max_num_batched_tokens, self.decode_cudagraph_max_bs)
        self._pcp_restore_idx = torch.zeros(capacity * self.pcp_size, dtype=torch.long, device=device)
        self._pcp_restore_mask = torch.zeros(capacity * self.pcp_size, dtype=torch.bool, device=device)
        self._pcp_local_idx = torch.zeros(capacity, dtype=torch.long, device=device)
        self._pcp_local_mask = torch.zeros(capacity, dtype=torch.bool, device=device)

    def _attach_pcp(self, metadata, context):
        if context is None:
            return metadata
        local_tokens = context.local_num_tokens
        if local_tokens is None:
            raise ValueError("Recurrent PCP metadata requires the padded local token count")
        global_tokens = local_tokens * self.pcp_size
        restore = context.hidden_restore_idx[: context.global_batch.num_tokens]
        if global_tokens > self._pcp_restore_idx.numel() or restore.numel() > global_tokens:
            raise ValueError("Recurrent PCP token permutation exceeds the input capacity")
        restore_idx = self._pcp_restore_idx[:global_tokens]
        restore_mask = self._pcp_restore_mask[:global_tokens]
        restore_idx.zero_()
        restore_idx[: restore.numel()].copy_(restore)
        restore_mask.zero_()
        restore_mask[: restore.numel()].fill_(True)
        local_idx = self._pcp_local_idx[:local_tokens]
        local_mask = self._pcp_local_mask[:local_tokens]
        local_idx.zero_()
        local_mask.zero_()
        indices = context.local_token_indices
        if indices is None:
            # Capture/idle batches bypass partition_batch. Decode-sharded
            # dummies own disjoint requests; replicated dummies share them.
            start = self.pcp_rank * local_tokens if context.global_batch.num_tokens > local_tokens else 0
            indices = torch.arange(start, start + local_tokens, device=local_idx.device)
        local_idx[: indices.numel()].copy_(indices)
        local_mask[: indices.numel()].fill_(True)
        metadata.pcp_restore_idx = restore_idx
        metadata.pcp_restore_mask = restore_mask
        metadata.pcp_local_idx = local_idx
        metadata.pcp_local_mask = local_mask
        return metadata

    def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs):
        metadata = super().build(common_prefix_len, common_attn_metadata, *args, **kwargs)
        return self._attach_pcp(metadata, getattr(common_attn_metadata, "pcp_context", None))


class KimiKDAHostMetadataBuilder(PCPRecurrentMetadataBuilderMixin, AscendGDNHostMetadataBuilder):
    pass


class AscendGDNPCPMetadataBuilder(PCPRecurrentMetadataBuilderMixin, AscendGDNAttentionMetadataBuilder):
    pass


class AscendGDNPCPBackend(AscendGDNAttentionBackend):
    @classmethod
    def supports_pcp(cls) -> bool:
        return True

    @staticmethod
    def get_builder_cls() -> type[AscendGDNPCPMetadataBuilder]:
        return AscendGDNPCPMetadataBuilder


class KimiKDAHostMetadataBackend(AscendGDNHostMetadataBackend):
    @staticmethod
    def get_builder_cls():
        return KimiKDAHostMetadataBuilder

    @classmethod
    def supports_pcp(cls) -> bool:
        return True


def pcp_gather_tokens(hidden_states: torch.Tensor, pcp_size: int, layer_name: str) -> torch.Tensor:
    """All-gather token rows and restore request order at the padded extent."""
    context = get_forward_context()
    gathered = get_pcp_group().all_gather(hidden_states.contiguous(), dim=0)
    metadata = (context.attn_metadata or {}).get(layer_name)
    if metadata is None:
        # Profiling may omit the recurrent metadata. Keep the gathered extent.
        return gathered
    output = gathered.index_select(0, metadata.pcp_restore_idx)
    return output.masked_fill(~metadata.pcp_restore_mask[:, None], 0)


def pcp_gather_tokens_fake(hidden_states: torch.Tensor, pcp_size: int, layer_name: str) -> torch.Tensor:
    return hidden_states.new_empty((hidden_states.shape[0] * pcp_size, hidden_states.shape[1]))


def pcp_scatter_tokens(output: torch.Tensor, pcp_size: int, layer_name: str) -> torch.Tensor:
    """Select and pad the current rank's rows from a restored tensor."""
    metadata = (get_forward_context().attn_metadata or {}).get(layer_name)
    if metadata is None:
        return output[: output.shape[0] // pcp_size].clone()
    local_output = output.index_select(0, metadata.pcp_local_idx)
    return local_output.masked_fill(~metadata.pcp_local_mask[:, None], 0)


def pcp_scatter_tokens_fake(output: torch.Tensor, pcp_size: int, layer_name: str) -> torch.Tensor:
    return output.new_empty((output.shape[0] // pcp_size, output.shape[1]))


# Fake shapes depend only on input shapes and configured parallel size. Python
# request metadata stays outside Dynamo; tensor permutations and collectives
# remain capturable in PIECEWISE/FULL graphs. Builders refresh stable buffers.
direct_register_custom_op(
    op_name="pcp_gather_tokens",
    op_func=pcp_gather_tokens,
    mutates_args=[],
    fake_impl=pcp_gather_tokens_fake,
)
direct_register_custom_op(
    op_name="pcp_scatter_tokens",
    op_func=pcp_scatter_tokens,
    mutates_args=[],
    fake_impl=pcp_scatter_tokens_fake,
)
