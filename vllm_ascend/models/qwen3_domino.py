# SPDX-License-Identifier: Apache-2.0
#
# This file is a part of the vllm-ascend project.
#
# Domino speculative-decoding draft model.
#
# Domino extends DFlash (parallel drafting) with a recurrent GRU correction head
# that refines the parallel drafts position-by-position using the realized tokens.
# See `vllm_ascend/spec_decode/domino_proposer.py` for the correction loop.
#
# The Domino checkpoint (`Huang2020/Qwen3-8B-Domino-b16`) ships the DFlash weights
# plus four correction-head weights (`prefix_gru.weight_ih_l0`,
# `prefix_gru.weight_hh_l0`, `embed_proj.0.weight`, `embed_proj.2.weight`) and no
# `embed_tokens`/`lm_head` of its own. The base proposer shares the target
# model's `embed_tokens`/`lm_head` onto this draft (same mechanism DFlash uses).

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM


def _is_domino_projector(projector_type: str | None) -> bool:
    return projector_type in ("domino", "causal_v5")


class DominoQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    """DFlash draft model + Domino recurrent correction head.

    Adds two modules on top of the upstream ``DFlashQwen3ForCausalLM``:
      * ``prefix_gru``  - a 1-layer GRU (``bias=False``) mapping
        ``hidden_size -> gru_hidden_dim`` that recurrently carries the realized
        token information across draft positions.
      * ``embed_proj``  - a ``Linear -> SiLU -> Linear`` MLP (both ``bias=False``)
        producing a per-position vocabulary bias that is added to the parallel
        draft ``base_logits`` before sampling.

    Both ``embed_tokens`` and ``lm_head`` are absent from the checkpoint and are
    shared from the target model by the base proposer at load time, so
    ``compute_logits`` / ``embed_tokens`` resolve to the target's (identical to
    the HF reference which uses ``target.lm_head`` / ``target.embed_tokens``).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = self.config
        drafter_config: dict = getattr(config, "eagle_config", {}) or {}
        drafter_config.update(getattr(config, "dflash_config", {}) or {})

        projector_type = drafter_config.get("projector_type")
        if not _is_domino_projector(projector_type):
            raise ValueError(
                "DominoQwen3ForCausalLM only supports Domino checkpoints "
                f"(projector_type == 'domino'); got projector_type={projector_type!r}."
            )

        self.shift_label = drafter_config.get("shift_label")
        if self.shift_label is not True:
            raise ValueError(
                "Domino checkpoints require dflash_config.shift_label=true; "
                f"got {self.shift_label!r}."
            )

        hidden_size = config.hidden_size
        vocab_size = getattr(config, "draft_vocab_size", None) or config.vocab_size
        emb_dim = drafter_config.get("emb_dim")
        if emb_dim is None:
            raise ValueError("Domino config must define emb_dim (in dflash_config).")
        gru_hidden_dim = drafter_config.get("gru_hidden_dim", emb_dim)

        self.pure_draft_prefix_len: int = int(drafter_config.get("pure_draft_prefix_len", 0))
        self.mask_token_id = drafter_config.get("mask_token_id")

        # Correction head. nn.GRU is bias=False to match the checkpoint weights
        # (only weight_ih_l0 / weight_hh_l0 are shipped). The GRU stays in the
        # model dtype (bf16); the fused path computes in fp32 (bf16-safe), so no
        # fp16 cast is needed (the eager nn.GRU path would hit Ascend's
        # fp16-only DynamicGRUV2 - pitfall #1 - but is only used on CPU/UT).
        self.prefix_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bias=False,
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_size + gru_hidden_dim, emb_dim, bias=False),
            nn.SiLU(),
            nn.Linear(emb_dim, vocab_size, bias=False),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        # Split the correction-head weights (live on the ForCausalLM, must NOT
        # receive the ``model.`` prefix that DFlashQwen3ForCausalLM.load_weights
        # applies to layer weights) from the DFlash weights.
        domino_weights: dict[str, torch.Tensor] = {}
        dflash_weights: list[tuple[str, torch.Tensor]] = []
        for name, loaded_weight in weights:
            if name.startswith("prefix_gru.") or name.startswith("embed_proj."):
                domino_weights[name] = loaded_weight
            else:
                dflash_weights.append((name, loaded_weight))

        # Load the DFlash weights (layers/fc/norm/hidden_norm) via the upstream
        # loader, which handles the q/k/v -> qkv stacking and builds the fused KV
        # buffers.
        super().load_weights(dflash_weights)

        # Load the correction-head weights directly onto ``self``.
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in domino_weights.items():
            if name not in params_dict:
                raise ValueError(f"Unexpected Domino correction-head weight: {name}")
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        # The GRU stays in the model dtype (bf16). The fused correction path
        # (domino_correct_fused -> fused_gru_cell triton kernel) computes in
        # fp32 and is bf16-safe, bypassing Ascend's fp16-only DynamicGRUV2. The
        # eager ``_run_prefix_gru`` (nn.GRU) is only used on CPU (UT) / as an
        # ImportError fallback, never on NPU in normal operation.

    @torch.inference_mode()
    def _run_prefix_gru(
        self,
        embeds: torch.Tensor,
        gru_hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the prefix/correction GRU via nn.GRU (eager path only).

        Used only by the eager ``domino_correct`` (CPU UT / ImportError
        fallback). On NPU the fused path (``domino_correct_fused`` ->
        ``fused_gru_cell`` triton kernel) is used instead, which is bf16-safe and
        bypasses Ascend's fp16-only ``DynamicGRUV2``.

        Args:
            embeds: ``[B, T, hidden_size]`` input embeddings.
            gru_hidden: optional ``[1, B, gru_hidden_dim]`` previous state.

        Returns:
            ``(output, h_n)`` where ``h_n`` is ``[1, B, gru_hidden_dim]``.
        """
        if gru_hidden is not None:
            return self.prefix_gru(embeds, gru_hidden)
        return self.prefix_gru(embeds)
