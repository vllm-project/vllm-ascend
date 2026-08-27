# SPDX-License-Identifier: Apache-2.0
"""NPU adaptive cost-table profiling run."""
from __future__ import annotations

import os

import numpy as np
import torch

from vllm.config import CUDAGraphMode
from vllm.distributed import get_pp_group, get_tp_group

from .globals import logger, ENV_PROFILE_FORCE_EAGER
from .utils import _npu_event

@torch.inference_mode()
def _adaptive_profile_run(
    self,
    scheduled_tokens: "list[int]",
    seq_lens: int = 1024,
    n_warmup: int = 3,
    n_measure: int = 5,
):
    """Profile verifier forward latency for one (batch_size, query_len) shape.

    **NPU port** of PR #44885's GPUModelRunner._adaptive_profile_run.  Modelled
    on vllm-ascend's ``NPUModelRunner._dummy_run``: it builds attention metadata
    with the NPU signature of ``_build_attention_metadata`` (no ``slot_mappings``
    kwarg, no ``_get_slot_mappings`` pre-step), then times ``self._model_forward``
    inside ``set_ascend_forward_context`` using ``torch.npu.Event``.  Profiling
    is forced eager by default for safety; set
    ``VLLM_DCUT_PROFILE_FORCE_EAGER=0`` to let the dispatcher pick the same
    graph/eager runtime mode as serving.

    Returns (runtime_mode, avg_forward_ms, num_tokens_padded).
    """
    # Import here so a stand-alone `import dcut` never hard-requires vllm-ascend.
    from vllm_ascend.ascend_forward_context import set_ascend_forward_context
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    num_scheduled_tokens = np.array(scheduled_tokens, dtype=np.int32)
    self.query_lens = torch.from_numpy(num_scheduled_tokens)
    self.attn_state = AscendAttentionState.ChunkedPrefill
    num_reqs = len(scheduled_tokens)
    num_tokens_unpadded = int(num_scheduled_tokens.sum())
    max_query_len = int(num_scheduled_tokens.max())

    assert num_tokens_unpadded <= self.max_num_tokens, (
        f"adaptive profile: num_tokens={num_tokens_unpadded} > "
        f"max_num_tokens={self.max_num_tokens}"
    )

    profile_force_eager = (
        os.environ.get(ENV_PROFILE_FORCE_EAGER, "1").lower()
        not in ("0", "false", "no")
    )
    profile_uniform_decode = (
        not profile_force_eager
        and max_query_len == self.uniform_decode_query_len
        and num_tokens_unpadded == max_query_len * num_reqs
    )
    # Same dispatcher entrypoint as _dummy_run. By default the profile run is
    # eager because manual graph replay on NPU is more fragile; the env override
    # is useful when comparing eager vs graph cost curves. Only the untrimmed
    # rectangle is uniform; shorter D-Cut candidates must profile ragged FULL.
    _cudagraph_mode, batch_desc, _should_ubatch, num_tokens_across_dp, _ = (
        self._determine_batch_execution_and_padding(
            num_tokens=num_tokens_unpadded,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens,
            max_num_scheduled_tokens=max_query_len,
            use_cascade_attn=False,
            allow_microbatching=False,
            force_eager=profile_force_eager,
            force_uniform_decode=profile_uniform_decode,
        )
    )

    num_tokens_padded = batch_desc.num_tokens
    num_reqs_padded = (
        batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
    )
    # allow_microbatching=False -> no ubatching for the profile run.
    ubatch_slices = None

    with self.synchronize_input_prep():
        # seq_lens / query_start_loc are needed by attention backends in all
        # modes.  Use the configured warmup_seq_lens so attention cost reflects
        # realistic long-context inference rather than trivial seq_len=1.
        self.optimistic_seq_lens_cpu[:num_reqs] = seq_lens
        self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
        self.seq_lens.copy_(self.optimistic_seq_lens_cpu, non_blocking=True)

        cum_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens, self.query_pos.np
        )
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
        self.query_start_loc.np[num_reqs + 1 : num_reqs_padded + 1].fill(
            cum_num_tokens[-1]
        )
        if _cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs_padded = self._pad_query_start_loc_for_fia(
                self.query_start_loc,
                num_tokens_padded,
                num_reqs_padded,
                num_reqs,
                _cudagraph_mode,
                batch_desc.num_reqs,
            )
        else:
            # PIECEWISE capture expects its final cuSeqlen to equal padded Q.
            self.query_start_loc.np[num_reqs_padded] = num_tokens_padded
            self.query_start_loc.copy_to_gpu()

        # GDN must retain the real ragged boundaries. FIA may append one dummy
        # request to the main qsl, but passing that row into GDN changes its
        # spec/decode classification and can update the wrong recurrent state.
        if self._has_gdn:
            self.gdn_query_start_loc.np[0] = 0
            self.gdn_query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
            self.gdn_query_start_loc.np[num_reqs + 1 :].fill(
                cum_num_tokens[-1]
            )
            self.gdn_query_start_loc.copy_to_gpu()
        self.input_batch.block_table.commit_block_table(num_reqs_padded)

        # Mark requests as decode (num_computed_tokens > 0) so the model
        # treats them as decode-phase, not prefill.  Required for the
        # dispatcher to select PIECEWISE cudagraph during profiling.
        for i in range(num_reqs):
            self.input_batch.num_computed_tokens_cpu[i] = seq_lens
        self.input_batch.num_computed_tokens_cpu[num_reqs:].fill(0)

        # Mark every sequence as a spec-decode so hybrid GDN/Mamba backends
        # take the cheap incremental spec-decode path instead of the expensive
        # prefill chunk-scan.  Per-request draft_len must respect the
        # spec_state_indices_tensor width (num_spec + 1) to avoid OOB in
        # npu_causal_conv1d_custom / npu_recurrent_gated_delta_rule.
        if self.speculative_config is not None:
            num_spec = self.num_spec_tokens
            if hasattr(self, "num_decode_draft_tokens"):
                for i in range(num_reqs):
                    self.num_decode_draft_tokens.np[i] = min(
                        int(num_scheduled_tokens[i]) - 1, num_spec
                    )
                self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
                self.num_decode_draft_tokens.copy_to_gpu()
            if hasattr(self, "num_accepted_tokens"):
                # num_accepted_tokens must be <= num_spec + 1 (the width of
                # spec_state_indices_tensor) to prevent OOB state access in
                # the Mamba/GDN conv1d sliding-window update.
                for i in range(num_reqs):
                    self.num_accepted_tokens.np[i] = min(
                        int(num_scheduled_tokens[i]), num_spec + 1
                    )
                self.num_accepted_tokens.np[num_reqs:].fill(1)
                self.num_accepted_tokens.copy_to_gpu()

        # NPU _build_attention_metadata: no `slot_mappings` kwarg; takes
        # `num_scheduled_tokens_np` and `num_reqs_padded` instead.
        attn_metadata, _ = self._build_attention_metadata(
            num_tokens=num_tokens_unpadded,
            num_reqs=num_reqs,
            max_query_len=max_query_len,
            num_tokens_padded=num_tokens_padded,
            num_reqs_padded=num_reqs_padded,
            ubatch_slices=ubatch_slices,
            for_cudagraph_capture=False,
            use_spec_decode=self.speculative_config is not None,
            num_scheduled_tokens_np=num_scheduled_tokens,
        )

    # Inputs — identical construction to _dummy_run so model kwargs (e.g. aux
    # hidden states for DFlash/Eagle3) are always provided.  Real verifier decode
    # steps are text-only, so we skip the vision encoder: mm-wrapped models route
    # through inputs_embeds, so we still supply a dummy embeds buffer for them —
    # just without the mm kwargs that would trigger the encoder.
    use_embeds = self.enable_prompt_embeds or (
        self.supports_mm_inputs and not self.model_config.is_encoder_decoder
    )
    if use_embeds:
        input_ids = None
        inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
    else:
        input_ids = self.input_ids.gpu[:num_tokens_padded]
        inputs_embeds = None

    if self.uses_mrope:
        positions = self.mrope_positions.gpu[:, :num_tokens_padded]
    elif self.uses_xdrope_dim > 0:
        positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
    else:
        positions = self.positions[:num_tokens_padded]

    intermediate_tensors = None
    if not get_pp_group().is_first_rank:
        from vllm.v1.outputs import IntermediateTensors  # lazy: PP>1 only
        if self.intermediate_tensors is None:
            self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=self.max_num_tokens,
                dtype=self.model_config.dtype,
                device=self.device,
            )
        intermediate_tensors = IntermediateTensors(
            {k: v[:num_tokens_padded] for k, v in self.intermediate_tensors.items()}
        )

    _mode_names = {
        CUDAGraphMode.FULL: "FCG",
        CUDAGraphMode.PIECEWISE: "PCG",
        CUDAGraphMode.NONE: "eager",
    }
    avg_ms = 0.0
    with set_ascend_forward_context(
        attn_metadata,
        self.vllm_config,
        num_tokens=num_tokens_padded,
        num_tokens_across_dp=num_tokens_across_dp,
        in_profile_run=True,
        num_actual_tokens=num_tokens_padded,
        aclgraph_runtime_mode=_cudagraph_mode,
        batch_descriptor=batch_desc,
        model_instance=self.model,
        has_sinks=self._has_sinks,
        input_ids=input_ids,
    ):

        def _forward() -> None:
            self._model_forward(
                num_tokens_padded,
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )

        for _ in range(max(n_warmup, 0)):
            _forward()
        torch.npu.synchronize()

        if n_measure > 0:
            start_ev = _npu_event(enable_timing=True)
            end_ev = _npu_event(enable_timing=True)
            start_ev.record()
            for _ in range(n_measure):
                _forward()
            end_ev.record()
            torch.npu.synchronize()
            avg_ms = start_ev.elapsed_time(end_ev) / n_measure

    mode_str = _mode_names.get(_cudagraph_mode, str(_cudagraph_mode))
    return mode_str, avg_ms, int(num_tokens_padded)
