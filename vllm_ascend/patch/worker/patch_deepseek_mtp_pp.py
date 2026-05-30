#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Patch: DeepSeekMTP pipeline parallelism support.
# On non-last PP ranks the MTP draft model is replaced with a minimal stub;
# full weights are only loaded on the last PP stage.
# Also fixes embedding / LM head sharing for MTP when PP > 1.
#

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Save originals
# ---------------------------------------------------------------------------
_original_mtp_init = DeepSeekMTP.__init__
_original_mtp_forward = DeepSeekMTP.forward
_original_mtp_compute_logits = DeepSeekMTP.compute_logits
_original_mtp_load_weights = DeepSeekMTP.load_weights
_original_mtp_embed_input_ids = DeepSeekMTP.embed_input_ids


# ---------------------------------------------------------------------------
# Patched DeepSeekMTP.__init__
# ---------------------------------------------------------------------------
def _patched_mtp_init(self, *, vllm_config, prefix: str = ""):
    pp_group = get_pp_group()

    if pp_group.world_size > 1 and not pp_group.is_last_rank:
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = PPMissingLayer()
        # MoE interface attributes (needed for compatibility checks)
        self.expert_weights: list = []
        self.num_moe_layers = 0
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers: list = []
        self.moe_mlp_layers: list = []
        return

    # Last PP rank or PP=1: full initialization.
    _original_mtp_init(self, vllm_config=vllm_config, prefix=prefix)


# ---------------------------------------------------------------------------
# Patched DeepSeekMTP.forward
# ---------------------------------------------------------------------------
def _patched_mtp_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    spec_step_idx: int = 0,
) -> torch.Tensor:
    if get_pp_group().world_size > 1 and not get_pp_group().is_last_rank:
        return hidden_states
    return _original_mtp_forward(
        self,
        input_ids,
        positions,
        hidden_states,
        intermediate_tensors,
        inputs_embeds,
        spec_step_idx,
    )


# ---------------------------------------------------------------------------
# Patched DeepSeekMTP.compute_logits
# ---------------------------------------------------------------------------
def _patched_mtp_compute_logits(
    self,
    hidden_states: torch.Tensor,
    spec_step_idx: int = 0,
) -> torch.Tensor | None:
    if get_pp_group().world_size > 1 and not get_pp_group().is_last_rank:
        return None
    return _original_mtp_compute_logits(self, hidden_states, spec_step_idx)


# ---------------------------------------------------------------------------
# Patched DeepSeekMTP.load_weights
# ---------------------------------------------------------------------------
def _patched_mtp_load_weights(
    self, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    if get_pp_group().world_size > 1 and not get_pp_group().is_last_rank:
        return set()
    return _original_mtp_load_weights(self, weights)


# ---------------------------------------------------------------------------
# Patched DeepSeekMTP.embed_input_ids
# ---------------------------------------------------------------------------
def _patched_mtp_embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
    if get_pp_group().world_size > 1 and not get_pp_group().is_last_rank:
        return input_ids
    return _original_mtp_embed_input_ids(self, input_ids)


# ---------------------------------------------------------------------------
# Patch AscendEagleProposer for MTP + PP
# ---------------------------------------------------------------------------
def _patch_proposer_for_mtp_pp():
    try:
        from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
    except ImportError:
        return

    _original_load_model = AscendEagleProposer.load_model
    _original_maybe_share_embeddings = AscendEagleProposer._maybe_share_embeddings
    _original_maybe_share_lm_head = AscendEagleProposer._maybe_share_lm_head

    def _patched_load_model(self, model: nn.Module) -> None:
        if (
            self.method == "mtp"
            and get_pp_group().world_size > 1
            and not get_pp_group().is_last_rank
        ):
            with self.maybe_eager_context:
                self.model = self._get_model()
            self._draft_attn_layer_names = set()
            self.attn_layer_names = []
            self.piece_all_attn_layer_name = []
            return

        # MTP on last PP rank, non-MTP methods, or PP=1: full load_model.
        _original_load_model(self, model)

    def _patched_maybe_share_embeddings(
        self, target_language_model: nn.Module
    ) -> None:
        if get_pp_group().world_size > 1:
            if get_pp_group().is_last_rank:
                # On the last PP rank the target model holds real embed_tokens;
                # share them with the MTP draft model to avoid loading a
                # duplicate copy that may be on a different device.
                if hasattr(target_language_model.model, "embed_tokens"):
                    target_embed_tokens = target_language_model.model.embed_tokens
                elif hasattr(target_language_model.model, "embedding"):
                    target_embed_tokens = target_language_model.model.embedding
                else:
                    raise AttributeError(
                        "Target model does not have 'embed_tokens' or 'embedding' attribute"
                    )

                if self.method == "mtp":
                    logger.info(
                        "PP>1 MTP: Sharing target model embedding weights with "
                        "the draft model on last PP rank."
                    )
                    if hasattr(self.model.model, "embed_tokens"):
                        del self.model.model.embed_tokens
                    self.model.model.embed_tokens = target_embed_tokens
                else:
                    # For EAGLE / EAGLE3 with PP>1, also share on last rank
                    logger.info(
                        "PP>1 %s: Sharing target model embedding weights on last "
                        "PP rank.",
                        self.method,
                    )
                    if hasattr(self.model.model, "embed_tokens"):
                        del self.model.model.embed_tokens
                    self.model.model.embed_tokens = target_embed_tokens
            else:
                logger.info(
                    "Non-last PP rank, skipping embedding sharing for draft model."
                )
            return

        # PP=1: use original logic.
        _original_maybe_share_embeddings(self, target_language_model)

    def _patched_maybe_share_lm_head(self, model: nn.Module) -> None:
        if get_pp_group().world_size > 1 and not get_pp_group().is_last_rank:
            logger.info(
                "Non-last PP rank, skipping LM head sharing for draft model."
            )
            return

        # Last PP rank or PP=1: use original logic.
        _original_maybe_share_lm_head(self, model)

    AscendEagleProposer.load_model = _patched_load_model
    AscendEagleProposer._maybe_share_embeddings = _patched_maybe_share_embeddings
    AscendEagleProposer._maybe_share_lm_head = _patched_maybe_share_lm_head


# ---------------------------------------------------------------------------
# Apply patches
# ---------------------------------------------------------------------------
DeepSeekMTP.__init__ = _patched_mtp_init
DeepSeekMTP.forward = _patched_mtp_forward
DeepSeekMTP.compute_logits = _patched_mtp_compute_logits
DeepSeekMTP.load_weights = _patched_mtp_load_weights
DeepSeekMTP.embed_input_ids = _patched_mtp_embed_input_ids
_patch_proposer_for_mtp_pp()
