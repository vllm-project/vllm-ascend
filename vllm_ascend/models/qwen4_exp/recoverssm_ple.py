# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep PLE history consistent with RecoverSSM's neutral align copy bias."""

from copy import copy
from dataclasses import dataclass, fields
from typing import Any

import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.attention.backends.recoverssm_metadata import RecoverSSMMetadata, RecoverSSMPostprocessMetadata
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadata,
    PleShortConvAttentionMetadataBuilder,
)

from vllm_ascend.models.qwen_gdn_recoverssm import _uses_gdn_recoverssm
from vllm_ascend.ops.triton.mamba.compact_conv_state import compact_conv_state
from vllm_ascend.ops.triton.mamba.prepare_commit_plan import prepare_commit_plan
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


@dataclass
class PLERecoverSSMCommitContext:
    layers: tuple[Any, ...]
    bindings: tuple[tuple[torch.Tensor, ...], ...]
    convs: tuple[torch.Tensor, ...]
    tables: tuple[torch.Tensor, ...]
    plan: tuple[torch.Tensor, ...]
    spec_query_len: int
    history_len: int

    @classmethod
    def create(cls, layers, *, spec_query_len: int, max_num_reqs: int):
        if not layers or spec_query_len < 1 or max_num_reqs < 1:
            raise ValueError("PLE RecoverSSM needs layers and positive capacities")
        init_device_properties_triton()
        history = layers[0].conv_state_len
        capacity = history + spec_query_len - 1
        bindings = tuple(tuple(layer.kv_cache) for layer in layers)
        convs = tuple(
            (cache[0] if is_conv_state_dim_first() else cache[0].transpose(-1, -2))[:, :, -capacity:]
            for cache in bindings
        )
        if any(layer.conv_state_len != history for layer in layers) or any(
            conv.shape != convs[0].shape or conv.shape[2] < capacity or conv.dtype != convs[0].dtype for conv in convs
        ):
            raise ValueError("PLE RecoverSSM needs matching history shapes and dtypes")
        device = convs[0].device
        tables = tuple(
            torch.tensor(values, dtype=torch.int64, device=device)
            for values in (
                [conv.data_ptr() for conv in convs],
                [conv.stride(0) for conv in convs],
                [conv.stride(1) for conv in convs],
                [conv.stride(2) for conv in convs],
            )
        )
        plan = tuple(torch.empty(max_num_reqs, dtype=torch.int32, device=device) for _ in range(6))
        return cls(tuple(layers), bindings, convs, tables, plan, spec_query_len, history)

    def commit(
        self,
        accepted,
        state_indices,
        query_start_loc,
        *,
        request_indices,
        block_table,
        num_computed_tokens,
        mamba_block_size,
    ):
        if any(
            any(current is not bound for current, bound in zip(layer.kv_cache, binding))
            for layer, binding in zip(self.layers, self.bindings)
        ):
            raise RuntimeError("PLE RecoverSSM caches were rebound; rebuild the commit context")
        count = state_indices.numel()
        if count > self.plan[0].numel() or mamba_block_size < self.spec_query_len:
            raise ValueError("PLE RecoverSSM request capacity or align block size is incompatible")
        if not count:
            return
        lens, state_lens, boundary_lens, final, boundary, recovery = (tensor[:count] for tensor in self.plan)
        prepare_commit_plan(
            accepted,
            state_indices,
            query_start_loc,
            lens,
            state_lens,
            boundary_lens,
            final,
            boundary,
            recovery,
            spec_query_len=self.spec_query_len,
            null_block_id=0,
            request_indices=request_indices,
            block_table=block_table,
            num_computed_tokens=num_computed_tokens,
            mamba_block_size=mamba_block_size,
        )
        compact_conv_state(
            self.convs[0],
            self.tables[0],
            self.tables[1],
            self.tables[2],
            self.tables[3],
            state_indices,
            lens,
            final,
            boundary,
            recovery,
            conv_history_len=self.history_len,
            align_mode=True,
            dim_stride=None,
            null_block_id=0,
        )


class QwenPLERecoverSSMMetadata(PleShortConvAttentionMetadata, RecoverSSMMetadata):
    def commit_recoverssm_state(self, num_accepted_tokens):
        align = self.ple_recoverssm_align
        self.ple_recoverssm_context.commit(
            num_accepted_tokens,
            self.spec_state_indices_tensor[: align.num_spec_decodes],
            self.spec_query_start_loc[: align.num_spec_decodes + 1],
            request_indices=align.request_indices,
            block_table=align.block_table,
            num_computed_tokens=align.num_computed_tokens,
            mamba_block_size=align.block_size,
        )
        return align


class _QwenPLEMetadataBuilder(PleShortConvAttentionMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        # The parent Mamba builder reads the global flag to allocate scratch
        # for a five-tensor Mamba2 cache. PLE has a single convolution tensor.
        ple_config = copy(vllm_config)
        ple_config.cache_config = copy(vllm_config.cache_config)
        ple_config.cache_config.use_replayssm = False
        super().__init__(kv_cache_spec, layer_names, ple_config, device)


class QwenPLERecoverSSMMetadataBuilder(_QwenPLEMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._commit_context = None
        self._neutral_accepted = torch.ones(vllm_config.scheduler_config.max_num_seqs, dtype=torch.int32, device=device)

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        fast_build=False,
        *,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        num_actual_reqs=None,
        **kwargs,
    ):
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
            **kwargs,
        )
        if not metadata.num_spec_decodes:
            return metadata
        # Graph padding may carry full speculative query spans. The sampler
        # and its request-state mapping contain only real requests.
        m = common_attn_metadata
        count = m.num_reqs if num_actual_reqs is None else num_actual_reqs
        mask = (num_decode_draft_tokens_cpu[:count] >= 0) & (m.query_start_loc_cpu[: count + 1].diff() > 0)
        rows = mask.nonzero(as_tuple=True)[0]
        if not rows.numel():
            return metadata
        if self._commit_context is None:
            layers = self.compilation_config.static_forward_context
            self._commit_context = PLERecoverSSMCommitContext.create(
                [layers[name] for name in self.layer_names],
                spec_query_len=self.num_spec + 1,
                max_num_reqs=self._neutral_accepted.numel(),
            )
        result = QwenPLERecoverSSMMetadata(
            **{field.name: getattr(metadata, field.name) for field in fields(PleShortConvAttentionMetadata)}
        )
        result.__dict__.update(metadata.__dict__)
        request_indices = rows.to(m.query_start_loc.device, non_blocking=True)
        result.ple_recoverssm_context = self._commit_context
        result.ple_recoverssm_align = RecoverSSMPostprocessMetadata(
            num_spec_decodes=rows.numel(),
            request_indices=request_indices,
            block_table=m.block_table_tensor,
            num_computed_tokens=m.compute_num_computed_tokens(),
            block_size=self.kv_cache_spec.block_size,
        )
        # PLE now commits accepted history in postprocessing, like GDN. Both
        # groups can therefore share the upstream neutral next-step copy bias.
        result.num_accepted_tokens = self._neutral_accepted[: metadata.spec_query_start_loc.numel() - 1]
        return result


class QwenPLERecoverSSMBackend(PleShortConvAttentionBackend):
    @staticmethod
    def get_builder_cls():
        config = get_current_vllm_config()
        if config.cache_config.use_replayssm:
            if _uses_gdn_recoverssm(config) and config.cache_config.mamba_cache_mode == "align":
                return QwenPLERecoverSSMMetadataBuilder
            return _QwenPLEMetadataBuilder
        return PleShortConvAttentionMetadataBuilder
