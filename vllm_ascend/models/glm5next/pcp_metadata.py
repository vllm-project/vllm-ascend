# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM KDA metadata owns its global graph capacity and token permutation."""

from dataclasses import replace

import torch

from vllm_ascend.attention.context_parallel.common_cp import is_pcp_decode_sharding_enabled
from vllm_ascend.ops.gdn_attn_builder import AscendGDNHostMetadataBuilder
from vllm_ascend.ops.kda_pcp import prepare_kda_pcp_config


def prepare_glm_pcp_config(vllm_config):
    if (
        vllm_config.use_v2_model_runner
        and is_pcp_decode_sharding_enabled(vllm_config)
        and vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs()
    ):
        return prepare_kda_pcp_config(vllm_config)
    return vllm_config


class Glm5NextPCPMetadataMixin:
    def _init_pcp_permutation(self, vllm_config, device, decode_cudagraph_max_bs: int) -> None:
        self._pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self._is_pcp_decode_sharded = is_pcp_decode_sharding_enabled(vllm_config)
        self._pcp_gather_indices: torch.Tensor | None = None
        if self._pcp_size > 1:
            capacity = max(vllm_config.scheduler_config.max_num_batched_tokens, decode_cudagraph_max_bs)
            self._pcp_gather_indices = torch.empty(capacity * self._pcp_size, dtype=torch.int64, device=device)

    def _attach_pcp_permutation(self, metadata):
        context = getattr(metadata, "pcp_context", None)
        if context is None or self._pcp_size == 1:
            return metadata
        if not self._is_pcp_decode_sharded and not bool(context.global_batch.is_prefilling_np.any()):
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
