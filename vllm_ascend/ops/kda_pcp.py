# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP metadata and token exchange for head-sharded KDA."""

from copy import copy
from dataclasses import replace

import torch
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.gdn_attn_builder import AscendGDNHostMetadataBackend, AscendGDNHostMetadataBuilder


def prepare_kda_pcp_config(vllm_config):
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    if pcp_size > 1:
        # Runner capture sizes count local tokens. KDA runs on the restored
        # global batch, including requests owned by other decode shards.
        vllm_config = copy(vllm_config)
        vllm_config.compilation_config = copy(vllm_config.compilation_config)
        capture_size = vllm_config.compilation_config.max_cudagraph_capture_size
        if capture_size is not None:
            vllm_config.compilation_config.max_cudagraph_capture_size = capture_size * pcp_size
    return vllm_config


def prepare_glm_pcp_config(vllm_config):
    if (
        vllm_config.use_v2_model_runner
        and vllm_config.parallel_config.pcp_shard_decode_requests
        and vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
    ):
        return prepare_kda_pcp_config(vllm_config)
    return vllm_config


class Glm5NextPCPMetadataMixin:
    def _init_pcp_permutation(self, vllm_config, device, decode_cudagraph_max_bs: int) -> None:
        self._pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self._is_pcp_decode_sharded = vllm_config.parallel_config.pcp_shard_decode_requests
        self._pcp_gather_indices: torch.Tensor | None = None
        if self._pcp_size > 1:
            capacity = max(vllm_config.scheduler_config.max_num_batched_tokens, decode_cudagraph_max_bs)
            self._pcp_gather_indices = torch.empty(capacity * self._pcp_size, dtype=torch.int64, device=device)

    def _attach_pcp_permutation(self, metadata):
        context = getattr(metadata, "pcp_context", None)
        if context is None or self._pcp_size == 1:
            return metadata
        if not self._is_pcp_decode_sharded and not context.global_batch.has_prefill:
            # Replicated decode never gathers KDA tokens or consumes this permutation.
            return metadata
        local_tokens = context.local_num_tokens
        if local_tokens is None or local_tokens <= 0:
            raise ValueError("GLM KDA PCP metadata requires a positive padded local token count")
        count = self._pcp_size * local_tokens
        if self._pcp_gather_indices is None or count > self._pcp_gather_indices.numel():
            raise ValueError("GLM KDA PCP permutation exceeds the metadata buffer capacity")
        indices = self._pcp_gather_indices[:count]
        source = context.padded_gather_idx
        if source is not None:
            if source.numel() != count:
                raise ValueError("GLM KDA PCP permutation does not match the padded local token count")
            indices.copy_(source)
        elif context.global_batch.is_dummy:
            global_tokens = context.global_batch.num_tokens_after_padding
            if global_tokens <= 0:
                raise ValueError("GLM KDA PCP dummy metadata requires positive global tokens")
            torch.arange(count, device=indices.device, out=indices)
            if global_tokens == local_tokens:
                # Nonuniform profiling and replicated decode share one global view.
                indices.remainder_(global_tokens)
            else:
                # Sharded capture has disjoint rank segments, with capped global padding.
                indices.masked_fill_(indices >= global_tokens, 0)
        else:
            raise ValueError("GLM KDA PCP runtime metadata requires a partitioned token layout")
        # Only GLM KDA retains this address in a graph. Keep the common PCP
        # context and upstream partition indices unchanged for other consumers.
        metadata.pcp_context = replace(context, padded_gather_idx=indices)
        return metadata


class Glm5NextGDNHostMetadataBuilder(Glm5NextPCPMetadataMixin, AscendGDNHostMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        vllm_config = prepare_glm_pcp_config(vllm_config)
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_pcp_permutation(vllm_config, device, self.decode_cudagraph_max_bs)

    def build(self, *args, **kwargs):
        return self._attach_pcp_permutation(super().build(*args, **kwargs))


def init_kda_pcp_buffers(builder, vllm_config, device):
    builder.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    builder.pcp_rank = get_pcp_group().rank_in_group if builder.pcp_size > 1 else 0
    capacity = max(vllm_config.scheduler_config.max_num_batched_tokens, builder.decode_cudagraph_max_bs)
    builder._pcp_restore_idx = torch.zeros(capacity * builder.pcp_size, dtype=torch.long, device=device)
    builder._pcp_restore_mask = torch.zeros(capacity * builder.pcp_size, dtype=torch.bool, device=device)
    builder._pcp_local_idx = torch.zeros(capacity, dtype=torch.long, device=device)
    builder._pcp_local_mask = torch.zeros(capacity, dtype=torch.bool, device=device)


class KimiKDAHostMetadataBuilder(AscendGDNHostMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        vllm_config = prepare_kda_pcp_config(vllm_config)
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        init_kda_pcp_buffers(self, vllm_config, device)

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

    def build(self, *args, **kwargs):
        metadata = super().build(*args, **kwargs)
        return self._attach_pcp(metadata, metadata.pcp_context)


class KimiKDAHostMetadataBackend(AscendGDNHostMetadataBackend):
    @staticmethod
    def get_builder_cls():
        return KimiKDAHostMetadataBuilder

    @classmethod
    def supports_pcp(cls) -> bool:
        return True


class KimiKDARecoverSSMMetadataBackend(KimiKDAHostMetadataBackend):
    @staticmethod
    def get_builder_cls():
        from vllm_ascend.models.kimi_k3.recoverssm_metadata import AscendKimiKDARecoverSSMMetadataBuilder

        return AscendKimiKDARecoverSSMMetadataBuilder


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
