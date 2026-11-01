# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpine proposer for the vLLM V1 spec-decode path on Ascend.

DSpine is dispatched as ``method="dflash"`` (the checkpoint advertises
``DFlashDraftModel`` plus ``dflash_config.selected_prefix_code_bridge``); this
proposer reuses the whole DFlash single-pass pipeline and only changes two
things:

1. Block-internal attention becomes *causal* (``draft_block_causal``): the
   first-pass metadata sets ``cad.causal = True`` so FIA runs TND +
   ``sparse_mode=3`` with the compressed causal mask instead of the
   non-causal DFlash layout.
2. ``compute_draft_token_ids`` runs the serial last-write refinement instead of
   a per-row argmax: top-K candidates per proposal row, an exact recompute of
   the final bridge write for every (sender, candidate) pair, then a greedy
   walk from the anchor. The merged draft path reaches this method under the
   same ``use_dflash2_selector`` gate DFlash2 uses; greedy is the only mode
   (probabilistic sampling is rejected like in ``AscendDflash2Proposer``).
"""

from dataclasses import replace
from typing import Any

import torch
from vllm import ModelRegistry
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.models.qwen3_dspine import (
    DSpineQwen3ForCausalLM,
    dspine_serial_chain,
    dspine_serial_table,
)
from vllm_ascend.worker.v1.spec_decode.dflash_proposer import AscendDflashProposer

DSPINE_ARCHITECTURE = "DFlashDSpineDraftModel"
DSPINE_MODEL_PATH = "vllm_ascend.models.qwen3_dspine:DSpineQwen3ForCausalLM"


def is_dspine_draft(speculative_config: Any) -> bool:
    """True when method is dflash and the draft carries a DSpine code bridge."""
    if getattr(speculative_config, "method", None) != "dflash":
        return False
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    hf_config = getattr(draft_model_config, "hf_config", None)
    dflash_cfg = getattr(hf_config, "dflash_config", None)
    bridge = (
        dflash_cfg.get("selected_prefix_code_bridge")
        if isinstance(dflash_cfg, dict)
        else getattr(dflash_cfg, "selected_prefix_code_bridge", None)
    )
    return bool(bridge)


class AscendDSpineProposer(AscendDflashProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner=runner)
        assert vllm_config.speculative_config is not None
        if vllm_config.speculative_config.draft_sample_method == "probabilistic":
            raise ValueError(
                "DSpine probabilistic draft sampling is not supported on the v1 "
                "model runner; use greedy (the default) instead."
            )
        # The checkpoint declares DFlashDraftModel so SpeculativeConfig's early
        # architecture validation passes; the code bridge is only detectable in
        # dflash_config. Retarget the draft architecture to the DSpine class
        # now — the worker process reaches this __init__ and the plugin's
        # models registration has run by then. Register defensively as well so
        # paths that never loaded the general plugin still resolve it.
        ModelRegistry.register_model(DSPINE_ARCHITECTURE, DSPINE_MODEL_PATH)
        draft_model_config = self.speculative_config.draft_model_config
        draft_model_config.hf_config.architectures = [DSPINE_ARCHITECTURE]
        draft_model_config.model_arch_config = replace(
            draft_model_config.model_arch_config,
            architectures=[DSPINE_ARCHITECTURE],
        )
        # Route the merged draft through compute_draft_token_ids (same gate as
        # AscendDflash2Proposer): the serial refinement replaces the argmax.
        self.use_dflash2_selector = True

    def load_model(self, model: torch.nn.Module) -> None:
        super().load_model(model)
        if not isinstance(self.model, DSpineQwen3ForCausalLM):
            raise ValueError(
                f"DSpine draft loaded class {type(self.model).__name__}; " "expected DSpineQwen3ForCausalLM"
            )
        block_size = self.model.model.selected_prefix_bridge.block_size
        if self.num_speculative_tokens + 1 != block_size:
            raise ValueError(
                f"DSpine block_size={block_size} requires "
                f"num_speculative_tokens={block_size - 1}, got "
                f"{self.num_speculative_tokens}"
            )
        # lm_head is shared with the target inside super().load_model(); the
        # serial tables must be built from the shared weight.
        self.model.build_dspine_serial_tables()

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        out = super().set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=cad,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            req_scheduled_tokens=req_scheduled_tokens,
            long_seq_metadata=long_seq_metadata,
            num_prefill_reqs=num_prefill_reqs,
            num_decode_reqs=num_decode_reqs,
        )
        # DSpine block rows attend causally inside the block (context + earlier
        # rows), unlike DFlash's non-causal all-to-all block attention. Setting
        # causal lets the metadata builder attach the compressed causal mask
        # and FIA run sparse_mode=3.
        out[2].causal = True
        return out

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
        **kwargs,
    ) -> None:
        # Identical to AscendDflashProposer.dummy_run except that the FULL-graph
        # capture metadata is built with causal=True and keeps its mask.
        num_query_tokens = min(num_tokens, self.max_query_tokens)

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_query_tokens, is_draft_model=True)

        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = num_reqs * num_query_per_req

        context_positions = self._context_positions_buffer[:num_input_tokens]
        context_states = self.hidden_states[:num_input_tokens]

        multi_steps_attn_metadata = []
        if aclgraph_runtime_mode == CUDAGraphMode.FULL and len(self.runner.attn_groups) > 0:
            builder = self.draft_attn_groups[0].get_metadata_builder()
            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.arange_dflash[: num_reqs + 1] * num_query_per_req,
                query_start_loc_cpu=torch.from_numpy(self.token_arange_np[: num_reqs + 1]).clone() * num_query_per_req,
                seq_lens_cpu=self.runner.optimistic_seq_lens_cpu,
                seq_lens_cpu_upper_bound=self.runner.optimistic_seq_lens_cpu,
                seq_lens=self.runner.seq_lens[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=num_query_tokens,
                max_query_len=num_query_per_req,
                max_seq_len=0,
                slot_mapping=self._slot_mapping_buffer[:num_query_total],
                attn_state=AscendAttentionState.ChunkedPrefill,
                causal=True,
                is_prefilling=torch.zeros(num_reqs, dtype=torch.bool),
                block_table_tensor=self.runner.input_batch.block_table[self.kv_cache_gid].get_device_tensor()[
                    :num_reqs
                ],
            )

            attn_metadata_dflash = builder.build_for_graph_capture(
                common_attn_metadata,
                AscendAttentionState.ChunkedPrefill,
            )

            attn_metadata_dflash.attn_state = AscendAttentionState.ChunkedPrefill

            per_layer_attn_metadata = dict()
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata_dflash
            multi_steps_attn_metadata.append(per_layer_attn_metadata)

        self.token_indices_to_sample.fill_(0)
        self._clear_dummy_slot_mappings()

        if aclgraph_runtime_mode == CUDAGraphMode.FULL:
            self._maybe_update_metadata(
                self.draft_attn_groups[0].backend,
                multi_steps_attn_metadata,
            )

        with set_ascend_forward_context(
            multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_input_tokens,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata,
        ):
            if is_profile:
                self.model.precompute_and_store_context_kv(context_states, context_positions)
                self.model(
                    input_ids=self.input_ids[:num_input_tokens],
                    positions=self._get_positions(num_input_tokens),
                    inputs_embeds=None,
                )
            else:
                self._dflash_num_context = num_input_tokens
                self._runnable(
                    num_input_tokens=num_input_tokens,
                    batch_size=num_reqs,
                    token_indices_to_sample=self.token_indices_to_sample[: num_reqs * self.num_speculative_tokens],
                    target_positions=self._get_positions(num_input_tokens),
                    inputs_embeds=None,
                    multi_steps_attn_metadata=multi_steps_attn_metadata,
                    num_tokens=num_input_tokens,
                )

            forward_context = get_forward_context()
            if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and not _EXTRA_CTX.capturing:
                self._update_full_graph_params(forward_context, num_tokens, multi_steps_attn_metadata)

    def compute_draft_token_ids(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Serial last-write refinement over the pass-1 top-K candidates.

        ``hidden_states`` are the final-normed proposal rows (request-major,
        rows 1..T-1 of each block). Returns (int64 [bs*(T-1)], None): DSpine
        always drafts greedily so draft probs are never produced.
        """
        model = self.model
        dm = model.model
        bridge = dm.selected_prefix_bridge
        if model._dspine_tg is None or model._dspine_tsv is None:
            raise RuntimeError("DSpine serial tables were not built")
        t = bridge.block_size
        n = hidden_states.shape[0]
        if n % (t - 1):
            raise ValueError(f"DSpine proposal rows {n} are not a multiple of block_size-1={t - 1}")
        bs = n // (t - 1)
        m = bs * t

        logits = model.logits_processor(model.lm_head, hidden_states)
        cand = logits.topk(dm.head_topk, dim=-1).indices.to(torch.int64)
        del logits

        idx = dspine_serial_table(
            dm.stash_raw[:m],
            dm.norm.weight,
            cand,
            self.input_ids[:m],
            model.lm_head.weight,
            model._dspine_tg,
            model._dspine_tsv,
            dm.stash_packed[:m],
            dm.stash_rms[:m],
            bridge.gate.bias,
            t,
        )
        return dspine_serial_chain(idx, cand, t), None
