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

    def build(
        self,
        common_prefix_len,
        common_attn_metadata,
        num_accepted_tokens=None,
        num_decode_draft_tokens_cpu=None,
        fast_build=False,
        num_actual_reqs=None,
    ):
        m, num_decode_draft_tokens_cpu = self._normalize_graph_padding(
            common_attn_metadata, num_actual_reqs, num_decode_draft_tokens_cpu
        )
        with _disable_upstream_graph_staging(self):
            metadata = super().build(common_prefix_len, m, num_accepted_tokens, num_decode_draft_tokens_cpu, fast_build)
        self._build_conv_metadata(m, metadata, num_decode_draft_tokens_cpu)
        return metadata

    def _normalize_graph_padding(self, m, num_actual_reqs, num_decode_draft_tokens_cpu):
        # Exclude graph-padded requests from sampler rows and accepted-state commit metadata.
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
        return m, num_decode_draft_tokens_cpu

    def _build_conv_metadata(self, m, metadata, num_decode_draft_tokens_cpu):
        # Build Ascend convolution descriptors for speculative decode, prefill and ordinary decode.
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
            chunk = self._build_prefill_chunk_metadata(m, metadata, num_decode_draft_tokens_cpu)
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

    def _build_prefill_chunk_metadata(self, m, metadata, num_decode_draft_tokens_cpu):
        # Remove decode rows and rebase prefill offsets before building chunk metadata.
        cpu_starts = self._non_spec_cpu_starts(m, metadata, num_decode_draft_tokens_cpu)
        decode_tokens = metadata.num_decode_tokens
        metadata.prefill_query_start_loc = metadata.non_spec_query_start_loc[metadata.num_decodes :] - decode_tokens
        prefill_cpu_starts = cpu_starts[metadata.num_decodes :] - decode_tokens
        metadata.prefill_state_indices = metadata.non_spec_state_indices_tensor[metadata.num_decodes :]
        metadata.prefill_has_initial_state = metadata.has_initial_state[metadata.num_decodes :]
        return _build_non_spec_chunked_prefill_metadata(
            self, prefill_cpu_starts, m.query_start_loc.device, build_device_metadata=False
        )

    def _non_spec_cpu_starts(self, m, metadata, draft_counts):
        if metadata.num_spec_decodes == 0:
            return m.query_start_loc_cpu
        lengths = m.query_start_loc_cpu.diff()
        spec = ((draft_counts >= 0) | ((~m.is_prefilling) & (lengths > 0))) & (lengths <= self.num_spec + 1)
        non_spec_lengths = lengths[(~spec) & (lengths > 0)]
        return torch.cat((torch.zeros(1, dtype=lengths.dtype), non_spec_lengths.cumsum(0)))

    def _stage_spec(self, metadata, batch_size):
        if (
            not self.use_full_cuda_graph
            or metadata.num_prefills
            or metadata.num_decodes
            # Eager fallback may exceed the captured token capacity.
            or metadata.num_spec_decode_tokens > self.decode_cudagraph_max_bs
            # State rows use max_num_seqs; offset/acceptance rows use the capture limit.
            or batch_size > min(self.spec_state_indices_tensor.shape[0], self.decode_cudagraph_max_bs)
        ):
            return
        # TODO: Fuse staging with an Ascend Triton kernel; upstream currently requires CUDA tensors.
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
