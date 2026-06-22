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

    # Materialize — the original pass consumes the iterator.
    weights_list = list(weights)

    # Original load_weights only loads MTP-specific layers (index >=
    # num_hidden_layers).  Embedding and lm_head are meant to be shared
    # from the target model via _maybe_share_embeddings / _maybe_share_lm_head,
    # but when PP>1 the target model's embedding/lm_head live on different
    # PP stages so sharing is not possible.  Load them directly from the
    # checkpoint instead.
    loaded = _original_mtp_load_weights(self, weights_list)

    from vllm.model_executor.model_loader.weight_utils import (
        default_weight_loader,
    )

    params_dict = dict(self.named_parameters())
    # Re-iterate to catch embed_tokens and lm_head that the original skipped.
    for name, loaded_weight in weights_list:
        if name in loaded:
            continue
        if name in params_dict:
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded.add(name)

    return loaded


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
            # With PP>1, the target model's embed_tokens only exists on the
            # first PP rank.  On the last PP rank (where the draft model runs)
            # it is PPMissingLayer, so the draft model must load its own
            # embedding.  Simply skip sharing on all PP>1 ranks — the draft
            # model will load its own embedding weights.
            logger.info(
                "PP>1: skipping embedding sharing for draft model, "
                "draft loads its own embedding weights."
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
