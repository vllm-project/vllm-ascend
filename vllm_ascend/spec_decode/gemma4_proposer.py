# SPDX-License-Identifier: Apache-2.0
"""Gemma4 MTP proposer with Ascend NPU support.

Uses multiple inheritance to combine the upstream vLLM Gemma4Proposer
(GPU logic) with vllm-ascend's AscendSpecDecodeBaseProposer
(NPU initialization, ACL graph, attention metadata).
"""

from dataclasses import replace

import torch
from vllm.config import get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.spec_decode.gemma4 import (
    Gemma4Proposer as _VllmGemma4Proposer,
)

from vllm_ascend.spec_decode.llm_base_proposer import (
    AscendSpecDecodeBaseProposer,
)


class AscendGemma4Proposer(_VllmGemma4Proposer, AscendSpecDecodeBaseProposer):
    """Gemma4 MTP proposer adapted for Ascend NPUs.

    MRO: AscendGemma4Proposer -> Gemma4Proposer ->
         AscendSpecDecodeBaseProposer -> SpecDecodeBaseProposer

    Gemma4Proposer provides:
      - _setup_gemma4_kv_sharing()
      - build_per_group_and_layer_attn_metadata()
      - initialize_attn_backend() (multi-group KV)
      - set_per_group_block_table()
      - _create_draft_vllm_config()

    AscendSpecDecodeBaseProposer provides:
      - NPU initialization (pcp_size, ACL graph, attention state)
      - _run_merged_draft() with Ascend-specific metadata
      - _propose() with slot mapping management
      - dummy_run() for ACL graph capture
      - load_model() / _get_model()
    """

    def __init__(
        self,
        vllm_config,
        device,
        runner=None,
    ):
        # 1. Ascend init: sets up pcp_size, use_cuda_graph,
        #    attn_mask_builder, slot_mapping buffers, ACL graph wrapper,
        #    self.runner reference, self._runnable, etc.
        AscendSpecDecodeBaseProposer.__init__(
            self,
            vllm_config,
            device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )

        # 2. Gemma4-specific attributes (same as upstream Gemma4Proposer.__init__
        #    after super()).  We set these directly rather than calling
        #    Gemma4Proposer.__init__ because in our multiple-inheritance MRO
        #    its super().__init__() would resolve to AscendSpecDecodeBaseProposer,
        #    causing a double-init of the Ascend base.
        self.constant_draft_positions = True
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        # Kept empty: upstream _greedy_sample reads this; on NPU we never
        # populate it (CUDA-graph centroids are GPU-only), so the upstream
        # fast-path is skipped and our override below handles masked vocab.
        self._centroids_sizes: list[int] = []

    # ---- _create_draft_vllm_config -------------------------------------------
    # Override to also replace model_config with the draft model's config.
    # Without this, vllm_config.model_config remains the target's config,
    # and the draft model's _patch_config() crashes on None sub_configs
    # (e.g. target's audio_config=None in Gemma4Config).

    def _create_draft_vllm_config(self):
        base = super()._create_draft_vllm_config()
        base = replace(
            base,
            model_config=self.speculative_config.draft_model_config,
        )
        return base

    # ---- _greedy_sample ----------------------------------------------------
    # Override to enable centroids masking in eager mode on Ascend NPU.
    # Upstream uses CUDA graphs with pre-captured centroids sizes, which
    # is not available on NPU. We bypass CUDA graphs and call
    # get_top_tokens() directly — this uses the same sparse centroid
    # vocabulary restriction but runs in eager mode.

    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        model = self.get_model()
        if getattr(model, "masked_embedding", None) is not None:
            return model.get_top_tokens(hidden_states)
        return super()._greedy_sample(hidden_states)

    # ---- _maybe_share_lm_head ----------------------------------------------
    # Gemma4 MTP's lm_head operates in draft hidden_size (e.g. 1024),
    # not the target's backbone hidden_size (e.g. 5376).  Sharing
    # would break compute_logits.  Both upstream Gemma4Proposer and
    # AscendSpecDecodeBaseProposer override this; we need the upstream
    # behaviour BUT also the Ascend ACLGraphWrapper setup that
    # Ascend._maybe_share_lm_head does for full-graph mode.
    #
    # Solution: skip lm_head sharing, but call Ascend's ACL setup.

    def _maybe_share_lm_head(self, target_language_model):
        """Keep draft lm_head; delegate ACL graph setup to Ascend parent."""
        from vllm.logger import logger

        logger.info("Gemma4 MTP: keeping draft model's own lm_head (draft_dim != backbone_dim).")
        # The Ascend parent's _maybe_share_lm_head only shares for
        # eagle/dflash or deepseek_mla — neither applies here.
        # But it also wraps self._runnable in ACLGraphWrapper for
        # full-graph mode.  Call it for that side-effect.
        AscendSpecDecodeBaseProposer._maybe_share_lm_head(self, target_language_model)

    # ---- _sync_kv_sharing_target_to_impl ------------------------------------
    # _setup_gemma4_kv_sharing (upstream) wires each draft attention layer to
    # share KV with a target layer by setting `kv_sharing_target_layer_name`
    # on the vLLM Attention *wrapper*.  At runtime, though, the Ascend path
    # reads that name off the backend *impl* (AscendAttention), which was
    # constructed with None.  Propagate the name wrapper -> impl here so KV
    # sharing actually resolves the target cache; without it the draft reads
    # its own (wrong) cache and acceptance collapses (~5%).
    #
    # (An earlier version of this method also rewrote num_kv_heads per layer
    # to match the heterogeneous GQA groups.  That is unnecessary now: in
    # 0.25.1 the Attention wrapper is constructed with the correct per-layer
    # count already -- full_attention layers carry num_global_key_value_heads
    # (4), not num_key_value_heads (16) -- so there is no head-count mismatch
    # to fix.  KV-head alignment is left to the backend.)

    def _sync_kv_sharing_target_to_impl(self) -> None:
        from vllm.logger import logger

        draft_model = self.get_model()
        num_layers = 0
        synced = 0
        for idx, layer in enumerate(draft_model.model.layers):
            num_layers += 1
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None:
                continue
            attn = getattr(self_attn, "attn", None)
            if attn is None:
                continue
            tgt_name = getattr(attn, "kv_sharing_target_layer_name", None)
            impl = attn.impl
            if impl is None or tgt_name is None:
                logger.info(
                    "MTP KV-sharing sync: draft layer %d skipped (impl=%s, target=%s).",
                    idx,
                    "present" if impl is not None else "None",
                    tgt_name,
                )
                continue
            # AscendAttention.kv_sharing_target_layer_name is a @property
            # descriptor, so a plain assignment is a silent no-op; bypass it.
            object.__setattr__(impl, "kv_sharing_target_layer_name", tgt_name)
            synced += 1
            logger.info(
                "MTP KV-sharing sync: draft layer %d -> impl.target=%s.",
                idx,
                tgt_name,
            )
        logger.info(
            "MTP KV-sharing sync: propagated to %d/%d draft layers.",
            synced,
            num_layers,
        )

    # ---- load_model --------------------------------------------------------
    # We need BOTH:
    #   a) Ascend's load_model (loads draft model, identifies draft layers,
    #      shares embeddings, handles multimodality, etc.)
    #   b) Gemma4's _setup_gemma4_kv_sharing (wires kv_sharing_target_layer_name
    #      on each draft attention layer)

    def load_model(self, target_model):
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,
            ).keys()
        )

        # Ascend load: loads the draft model, finds draft attn layers,
        # shares embed_tokens with target, handles multimodal, etc.
        AscendSpecDecodeBaseProposer.load_model(self, target_model)

        # The Ascend base load_model doesn't run the upstream
        # supports_mm_inputs detection (which probes embed_input_ids).
        # Gemma4 MTP draft only consumes backbone hidden states;
        # it doesn't handle multimodal embeddings.
        self.supports_mm_inputs = False

        # Wire cross-model KV sharing: each draft attention layer
        # reads K/V from the corresponding target layer's cache.
        _VllmGemma4Proposer._setup_gemma4_kv_sharing(self, target_attn_layer_names)

        # Propagate each draft layer's kv_sharing_target_layer_name from the
        # Attention wrapper onto the AscendAttention impl, which is what the
        # runtime KV-sharing path reads.  See the method doc above.
        self._sync_kv_sharing_target_to_impl()

        # Centroids CUDA graphs are CUDA-only; skip on Ascend.
        # The upstream check calls _setup_centroids_cuda_graphs()
        # when masked_embedding is present, but that uses
        # torch.cuda.CUDAGraph which is not available on NPU.
        # If centroids are needed, they run in eager mode.
