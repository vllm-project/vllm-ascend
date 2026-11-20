# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.config import CUDAGraphMode
from vllm.model_executor.models.diffusion_gemma import DiffusionGemmaModelState

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata


class AscendDiffusionGemmaModelState(DiffusionGemmaModelState):
    def __init__(self, vllm_config, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)
        self._token_positions = torch.arange(self.max_num_tokens, device=self.device, dtype=torch.int32)
        self._token_causal = torch.empty(self.max_num_tokens, device=self.device, dtype=torch.bool)

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
        ubatch_idx=0,
    ):
        assert ubatch_idx == 0, "DiffusionGemma does not support microbatching."
        actual_reqs = input_batch.num_reqs
        num_reqs = actual_reqs
        num_tokens = input_batch.num_tokens
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding

        # A persistent token mask lets replay use the current request phases
        # without recapturing attention when a canvas moves to its commit step.
        starts = input_batch.query_start_loc
        mask = self._token_causal[:num_tokens]
        if actual_reqs:
            slots = input_batch.idx_mapping[:actual_reqs]
            positions = self._token_positions[:num_tokens]
            ends = starts[1 : actual_reqs + 1]
            req = torch.searchsorted(ends.contiguous(), positions, right=True)
            phases = self.diffusion_states.is_encoder_phase[slots]
            mask.copy_(phases[req.clamp_max(actual_reqs - 1)])
        else:
            mask.fill_(True)

        # FIA requires exact host-visible lengths, including after a rewind.
        seq_lens = input_batch.seq_lens[:num_reqs]
        metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_actual_reqs=actual_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=starts,
            query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
            max_query_len=int(input_batch.num_scheduled_tokens.max(initial=0)),
            seq_lens=seq_lens,
            seq_lens_np=seq_lens.cpu().numpy(),
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            attn_state=AscendAttentionState.ChunkedPrefill,
            for_cudagraph_capture=for_capture,
            full_graph_mode=cudagraph_mode == CUDAGraphMode.FULL,
        )
        for layer_metadata in metadata.values():
            layer_metadata.diffusion_causal = mask
        self.attn_metadata = metadata
        return metadata
