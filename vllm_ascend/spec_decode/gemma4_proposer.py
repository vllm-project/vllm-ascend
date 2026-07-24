# SPDX-License-Identifier: Apache-2.0
"""Gemma4 MTP proposer with Ascend NPU support.

Uses multiple inheritance to combine the upstream vLLM Gemma4Proposer
(GPU logic) with vllm-ascend's AscendSpecDecodeBaseProposer
(NPU initialization, ACL graph, attention metadata).
"""

from dataclasses import replace

import torch
from vllm.config import get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.spec_decode.gemma4 import (
    Gemma4Proposer as _VllmGemma4Proposer,
)

from vllm_ascend.spec_decode.llm_base_proposer import (
    AscendSpecDecodeBaseProposer,
)

logger = init_logger(__name__)


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
        logger.info("Gemma4 MTP: keeping draft model's own lm_head (draft_dim != backbone_dim).")
        # The Ascend parent's _maybe_share_lm_head only shares for
        # eagle/dflash or deepseek_mla — neither applies here.
        # But it also wraps self._runnable in ACLGraphWrapper for
        # full-graph mode.  Call it for that side-effect.
        AscendSpecDecodeBaseProposer._maybe_share_lm_head(self, target_language_model)

    # ---- _fix_draft_kv_head_counts -------------------------------------------
    # When the draft model's attention layer is configured with a different
    # number of KV heads than the target layer it shares with (e.g. draft
    # reports 16 KV heads from HuggingFace config but the target's
    # full_attention uses num_global_kv_heads=4), reading that cache under the
    # draft's head counts would misinterpret the layout.  Align only
    # num_kv_heads; query heads (num_heads) are tied to the draft's Q
    # projection weight and must not change.

    def _fix_draft_kv_head_counts(self) -> None:
        target_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,
        )

        draft_model = self.get_model()
        for draft_idx, layer in enumerate(draft_model.model.layers):
            attn = layer.self_attn.attn
            tgt_name = attn.kv_sharing_target_layer_name
            if tgt_name is None:
                continue
            target_module = target_attn_layers.get(tgt_name)
            if target_module is None:
                logger.warning(
                    "Draft layer %d shares KV with '%s' but target module not found — skipping head-count fix.",
                    draft_idx,
                    tgt_name,
                )
                continue

            draft_nkv = attn.num_kv_heads
            tgt_nkv = target_module.num_kv_heads
            draft_nh = attn.num_heads
            tgt_nh = target_module.num_heads

            # Query heads are tied to the draft's Q projection weight
            # ([num_heads * head_dim, hidden_size]) and must match the
            # target's.  If they differ, KV sharing cannot be used safely.
            assert draft_nh == tgt_nh, (
                f"Draft layer {draft_idx} has {draft_nh} query heads but "
                f"target '{tgt_name}' has {tgt_nh}; query-head mismatch "
                f"is not supported with KV sharing."
            )

            # Sync kv_sharing_target_layer_name to the backend impl.
            impl = attn.impl
            if impl is not None:
                object.__setattr__(impl, "kv_sharing_target_layer_name", tgt_name)

            if draft_nkv != tgt_nkv:
                logger.info(
                    "MTP KV-sharing head fix: draft layer %d (kv_heads=%d) -> target '%s' (kv_heads=%d)",
                    draft_idx,
                    draft_nkv,
                    tgt_name,
                    tgt_nkv,
                )
                object.__setattr__(attn, "num_kv_heads", tgt_nkv)
                if impl is not None:
                    object.__setattr__(impl, "num_kv_heads", tgt_nkv)
                # Also fix Gemma4MTPAttention's own num_kv_heads, used to
                # create the dummy KV tensor in forward().  If this doesn't
                # match Attention.num_kv_heads, the reshape inside
                # Attention.forward produces a different batch dimension for
                # key vs query, causing the KV-sharing prefill condition
                #   query.shape[0] == key.shape[0]
                # to fail.
                mtp_attn = layer.self_attn
                if mtp_attn.num_kv_heads != tgt_nkv:
                    object.__setattr__(mtp_attn, "num_kv_heads", tgt_nkv)

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

        # Fix num_kv_heads mismatch: the draft model's attention layers
        # may have a different GQA configuration than the target layers
        # they share KV caches with (e.g., the draft reads 16 KV heads
        # from its config but the target's full_attention layer only has
        # 4 global KV heads).  If they don't match, reading the shared cache
        # under the draft's head counts misinterprets the layout, leading to
        # garbage attention outputs and downstream crashes.
        self._fix_draft_kv_head_counts()

        # Centroids CUDA graphs are CUDA-only; skip on Ascend.
        # The upstream check calls _setup_centroids_cuda_graphs()
        # when masked_embedding is present, but that uses
        # torch.cuda.CUDAGraph which is not available on NPU.
        # If centroids are needed, they run in eager mode.
