# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-stable token permutations for head-sharded Kimi KDA."""

from copy import copy

import torch
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ops.gdn_attn_builder import AscendGDNHostMetadataBackend, AscendGDNHostMetadataBuilder


class KimiKDAHostMetadataBuilder(AscendGDNHostMetadataBuilder):
    consumes_pcp_context = True

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        if pcp_size > 1:
            # Runner capture sizes count local tokens. KDA runs on the restored
            # global batch, including requests owned by other decode shards.
            vllm_config = copy(vllm_config)
            vllm_config.compilation_config = copy(vllm_config.compilation_config)
            capture_size = vllm_config.compilation_config.max_cudagraph_capture_size
            if capture_size is not None:
                vllm_config.compilation_config.max_cudagraph_capture_size = capture_size * pcp_size
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        sequence_capacity = max(vllm_config.scheduler_config.max_num_seqs, self.decode_cudagraph_max_bs)
        self._spec_conv_indices = torch.empty(sequence_capacity, dtype=torch.int32, device=device)
        self._non_spec_conv_indices = torch.empty_like(self._spec_conv_indices)
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
            raise ValueError("KDA PCP metadata requires the padded local token count")
        global_tokens = local_tokens * self.pcp_size
        restore = context.hidden_restore_idx
        if global_tokens > self._pcp_restore_idx.numel() or restore.numel() > global_tokens:
            raise ValueError("KDA PCP token permutation exceeds the input capacity")
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

    @staticmethod
    def _prepare_conv_indices(indices, buffer):
        # Convolution needs the first state slot; recurrent KDA still needs
        # all speculative slots. Materialize once per group, not per layer.
        indices = indices[:, 0] if indices.ndim == 2 else indices
        if indices.is_contiguous():
            return indices
        prepared = buffer[: indices.shape[0]]
        prepared.copy_(indices, non_blocking=True)
        return prepared

    def _reset_spec_decode_graph_inputs(self, graph_batch_size):
        super()._reset_spec_decode_graph_inputs(graph_batch_size)
        # A retained speculative graph also holds the prepared conv pointer.
        self._spec_conv_indices[:graph_batch_size].fill_(PAD_SLOT_ID)

    def build(self, *args, pcp_context=None, pcp_cache_group_idx=None, **kwargs):
        metadata = super().build(*args, **kwargs)
        spec = metadata.spec_decode_metadata
        if spec is not None:
            conv = spec.spec_causal_conv1d
            conv.cache_indices = self._prepare_conv_indices(conv.cache_indices, self._spec_conv_indices)
        non_spec = metadata.non_spec_prefill_metadata or metadata.non_spec_decode_metadata
        if non_spec is not None:
            conv = non_spec.causal_conv1d
            conv.cache_indices = self._prepare_conv_indices(conv.cache_indices, self._non_spec_conv_indices)
        return self._attach_pcp(metadata, pcp_context)

    def build_for_cudagraph_capture(self, common_attn_metadata, *, pcp_context=None, pcp_cache_group_idx=None):
        metadata = super().build_for_cudagraph_capture(common_attn_metadata)
        return self._attach_pcp(metadata, pcp_context)


class KimiKDAHostMetadataBackend(AscendGDNHostMetadataBackend):
    @staticmethod
    def get_builder_cls():
        return KimiKDAHostMetadataBuilder

    @classmethod
    def supports_pcp(cls) -> bool:
        return True


def kda_pcp_gather(hidden_states: torch.Tensor, pcp_size: int, layer_name: str) -> torch.Tensor:
    context = get_forward_context()
    gathered = context.no_compile_layers[layer_name].pcp_group.all_gather(hidden_states.contiguous(), dim=0)
    metadata_dict = context.attn_metadata
    if metadata_dict is None:
        return gathered
    metadata = metadata_dict[layer_name]
    output = gathered.index_select(0, metadata.pcp_restore_idx)
    return output.masked_fill(~metadata.pcp_restore_mask[:, None], 0)


def kda_pcp_gather_fake(hidden_states: torch.Tensor, pcp_size: int, layer_name: str) -> torch.Tensor:
    return hidden_states.new_empty((hidden_states.shape[0] * pcp_size, hidden_states.shape[1]))


def kda_pcp_scatter(output: torch.Tensor, pcp_size: int, layer_name: str) -> torch.Tensor:
    metadata_dict = get_forward_context().attn_metadata
    local_tokens = output.shape[0] // pcp_size
    if metadata_dict is None:
        return output[:local_tokens].clone()
    metadata = metadata_dict[layer_name]
    local_output = output.index_select(0, metadata.pcp_local_idx)
    return local_output.masked_fill(~metadata.pcp_local_mask[:, None], 0)


def kda_pcp_scatter_fake(output: torch.Tensor, pcp_size: int, layer_name: str) -> torch.Tensor:
    return output.new_empty((output.shape[0] // pcp_size, output.shape[1]))


# Fake shapes depend only on input shapes and configured parallel size. Python
# request metadata stays outside Dynamo; tensor permutations and collectives
# remain capturable in PIECEWISE/FULL graphs. Builders refresh stable buffers.
direct_register_custom_op(
    op_name="kda_pcp_gather",
    op_func=kda_pcp_gather,
    mutates_args=[],
    fake_impl=kda_pcp_gather_fake,
)
direct_register_custom_op(
    op_name="kda_pcp_scatter",
    op_func=kda_pcp_scatter,
    mutates_args=[],
    fake_impl=kda_pcp_scatter_fake,
)
