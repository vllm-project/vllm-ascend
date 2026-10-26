# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse upstream RecoverSSM request/commit metadata with AscendC descriptors."""

from contextlib import contextmanager
from dataclasses import replace

import torch
from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadataBuilder
from vllm.v1.attention.backend import AttentionCGSupport

from vllm_ascend.ops.gdn_attn_builder import (
    GDNCausalConv1dMetadata,
    GDNDecodeMetadata,
    GDNPrefillMetadata,
    GDNSpecCausalConv1dMetadata,
    GDNSpecDecodeMetadata,
    _build_actual_seq_lengths,
    _build_non_spec_chunked_prefill_metadata,
)
from vllm_ascend.ops.triton.kda.recoverssm import KDARecoverSSMCommitContext


@contextmanager
def _disable_upstream_graph_staging(builder):
    # Only the parent build uses CUDA-specific staging. Restore the real
    # graph mode before preparing persistent NPU buffers, even on failure.
    builder._ascend_full_graph = builder.use_full_cuda_graph
    builder.use_full_cuda_graph = False
    try:
        yield
    finally:
        builder.use_full_cuda_graph = builder._ascend_full_graph
        del builder._ascend_full_graph


class AscendKDARecoverSSMMetadataBuilder(KimiK3KDAMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def _get_recoverssm_context(self):
        if self.recoverssm_context is None:
            forward_context = self.vllm_config.compilation_config.static_forward_context
            self.recoverssm_context = KDARecoverSSMCommitContext.create(
                [forward_context[name] for name in self.layer_names],
                spec_query_len=self.num_spec + 1,
                max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs,
            )
        return self.recoverssm_context

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        fast_build=False,
        num_actual_reqs=None,
    ):
        m = common_attn_metadata
        if num_actual_reqs is not None and num_actual_reqs < m.num_reqs:
            # FIA pads graph requests with full query spans. They must not
            # become sampler rows: commit receives only actual request counts.
            starts_cpu = m.query_start_loc_cpu.clone()
            starts_cpu[num_actual_reqs + 1 :] = starts_cpu[num_actual_reqs]
            # Clamp on the current device stream. A blocking host-to-device
            # copy here waits for the previous MTP graph before target launch.
            starts = torch.minimum(m.query_start_loc, m.query_start_loc[num_actual_reqs])
            m = m.replace(
                query_start_loc_cpu=starts_cpu,
                query_start_loc=starts,
                num_actual_tokens=int(starts_cpu[num_actual_reqs]),
            )
            if num_decode_draft_tokens_cpu is not None:
                num_decode_draft_tokens_cpu = num_decode_draft_tokens_cpu.clone()
                num_decode_draft_tokens_cpu[num_actual_reqs:] = -1
        with _disable_upstream_graph_staging(self):
            metadata = super().build(common_prefix_len, m, num_accepted_tokens, num_decode_draft_tokens_cpu, fast_build)
        metadata.non_spec_prefill_metadata = None
        metadata.non_spec_decode_metadata = None
        metadata.spec_decode_metadata = None
        if metadata.num_spec_decodes:
            self._stage_spec(metadata, m.num_reqs)
            state_indices = metadata.spec_state_indices_tensor[:, 0]
            starts = metadata.spec_query_start_loc
            metadata.spec_decode_metadata = GDNSpecDecodeMetadata(
                spec_causal_conv1d=GDNSpecCausalConv1dMetadata(
                    query_start_loc=starts,
                    cache_indices=state_indices,
                    num_accepted_tokens=metadata.num_accepted_tokens,
                ),
                actual_seq_lengths=None,
            )
        if metadata.num_prefills:
            cpu_starts = self._non_spec_cpu_starts(m, metadata, num_decode_draft_tokens_cpu)
            decode_tokens = metadata.num_decode_tokens
            prefill_starts = metadata.non_spec_query_start_loc[metadata.num_decodes :] - decode_tokens
            prefill_cpu_starts = cpu_starts[metadata.num_decodes :] - decode_tokens
            metadata.prefill_query_start_loc = prefill_starts
            metadata.prefill_state_indices = metadata.non_spec_state_indices_tensor[metadata.num_decodes :]
            metadata.prefill_has_initial_state = metadata.has_initial_state[metadata.num_decodes :]
            chunk = _build_non_spec_chunked_prefill_metadata(
                self, prefill_cpu_starts, m.query_start_loc.device, build_device_metadata=False
            )
            metadata.non_spec_prefill_metadata = GDNPrefillMetadata(
                causal_conv1d=GDNCausalConv1dMetadata(
                    query_start_loc=metadata.non_spec_query_start_loc,
                    cache_indices=metadata.non_spec_state_indices_tensor,
                    initial_state_mode=metadata.has_initial_state,
                ),
                chunk=chunk,
            )
        elif metadata.num_decodes:
            starts = metadata.non_spec_query_start_loc
            if starts is None:
                starts = torch.arange(metadata.num_decodes + 1, dtype=torch.int32, device=m.query_start_loc.device)
                metadata.non_spec_query_start_loc = starts
            metadata.non_spec_decode_metadata = GDNDecodeMetadata(
                causal_conv1d=GDNCausalConv1dMetadata(
                    query_start_loc=starts,
                    cache_indices=metadata.non_spec_state_indices_tensor,
                    initial_state_mode=None,
                ),
                actual_seq_lengths=_build_actual_seq_lengths(starts, metadata.num_decodes),
            )
        return metadata

    def build_for_cudagraph_capture(self, common_attn_metadata, **kwargs):
        m = common_attn_metadata
        return self.build(0, m, torch.diff(m.query_start_loc), torch.diff(m.query_start_loc_cpu) - 1)

    def _non_spec_cpu_starts(self, m, metadata, draft_counts):
        if metadata.num_spec_decodes == 0:
            return m.query_start_loc_cpu
        lengths = m.query_start_loc_cpu.diff()
        spec = ((draft_counts >= 0) | ((~m.is_prefilling) & (lengths > 0))) & (lengths <= self.num_spec + 1)
        non_spec_lengths = lengths[(~spec) & (lengths > 0)]
        return torch.cat((torch.zeros(1, dtype=lengths.dtype), non_spec_lengths.cumsum(0)))

    def _stage_spec(self, metadata, batch_size):
        if not self.use_full_cuda_graph or metadata.num_prefills or metadata.num_decodes:
            return
        count = metadata.num_spec_decodes
        states = self.spec_state_indices_tensor[:batch_size]
        starts = self.spec_query_start_loc[: batch_size + 1]
        accepted = self.num_accepted_tokens[:batch_size]
        states[:count].copy_(metadata.spec_state_indices_tensor)
        states[count:].zero_()
        starts[: count + 1].copy_(metadata.spec_query_start_loc)
        starts[count + 1 :].copy_(metadata.spec_query_start_loc[-1:].expand(batch_size - count))
        accepted.fill_(1)
        metadata.spec_state_indices_tensor = states
        metadata.spec_query_start_loc = starts
        metadata.num_accepted_tokens = accepted
        metadata.recoverssm_commit = replace(metadata.recoverssm_commit, state_indices=states, query_start_loc=starts)
