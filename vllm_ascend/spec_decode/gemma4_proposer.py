# SPDX-License-Identifier: Apache-2.0
"""Gemma4 MTP proposer with Ascend NPU support.

Combines upstream Gemma4Proposer (GPU logic) with vllm-ascend's
AscendSpecDecodeBaseProposer (NPU initialization, ACL graph, attention metadata).
"""

from dataclasses import replace

import torch
from vllm.config import get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.spec_decode.gemma4 import Gemma4Proposer as _VllmGemma4Proposer

from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


class AscendGemma4Proposer(_VllmGemma4Proposer, AscendSpecDecodeBaseProposer):
    """Gemma4 MTP proposer adapted for Ascend NPUs.

    MRO: AscendGemma4Proposer -> Gemma4Proposer ->
         AscendSpecDecodeBaseProposer -> SpecDecodeBaseProposer
    """

    def __init__(self, vllm_config, device, runner=None):
        AscendSpecDecodeBaseProposer.__init__(
            self, vllm_config, device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )

        # Set Gemma4-specific attributes directly rather than calling
        # Gemma4Proposer.__init__ — in our MRO its super().__init__() would
        # resolve to AscendSpecDecodeBaseProposer, double-initing the Ascend base.
        self.constant_draft_positions = True
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        self._centroids_sizes: list[int] = []

    def _create_draft_vllm_config(self):
        """Replace model_config with the draft model's config.

        Without this, vllm_config.model_config stays the target's config and
        the draft model's _patch_config() crashes on None sub_configs.
        """
        base = super()._create_draft_vllm_config()
        return replace(base, model_config=self.speculative_config.draft_model_config)

    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Eager-mode centroids masking on Ascend (no CUDA graphs available)."""
        model = self.get_model()
        if getattr(model, "masked_embedding", None) is not None:
            return model.get_top_tokens(hidden_states)
        return super()._greedy_sample(hidden_states)

    def _maybe_share_lm_head(self, target_language_model):
        """Keep draft lm_head; delegate ACL graph setup to Ascend parent.

        Gemma4 MTP's lm_head operates in draft hidden_size, not the target's
        backbone hidden_size, so sharing would break compute_logits.
        """
        from vllm.logger import logger

        logger.info("Gemma4 MTP: keeping draft model's own lm_head (draft_dim != backbone_dim).")
        AscendSpecDecodeBaseProposer._maybe_share_lm_head(self, target_language_model)

    def _sync_kv_sharing_target_to_impl(self) -> None:
        """Propagate kv_sharing_target_layer_name from Attention wrapper to impl.

        _setup_gemma4_kv_sharing (upstream) sets this on the vLLM Attention
        wrapper. The Ascend runtime reads it from the backend impl
        (AscendAttention), which was constructed with None. Without this sync
        the draft reads its own cache and acceptance collapses.
        """
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
            # descriptor; plain assignment is a silent no-op — bypass it.
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

    def load_model(self, target_model):
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config, AttentionLayerBase,
            ).keys()
        )

        AscendSpecDecodeBaseProposer.load_model(self, target_model)

        # Gemma4 MTP draft only consumes backbone hidden states,
        # not multimodal embeddings.
        self.supports_mm_inputs = False

        # Wire cross-model KV sharing: each draft attention layer reads K/V
        # from the corresponding target layer's cache.
        _VllmGemma4Proposer._setup_gemma4_kv_sharing(self, target_attn_layer_names)

        # Sync wrapper→impl so the Ascend runtime path can resolve target cache.
        self._sync_kv_sharing_target_to_impl()
