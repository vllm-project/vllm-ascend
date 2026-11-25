# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import GemmaRMSNorm

from vllm_ascend.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM, DFlash2Qwen3Model
from vllm_ascend.worker.v2.spec_decode.dmtp.config import AscendDMTPSpeculatorConfig


class MTPT1Residual(nn.Module):
    """The checkpoint's replicated adapter for the native MTP output that predicted T1."""

    def __init__(self, hidden_size: int, rank: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.eps = eps
        self.down = nn.Linear(hidden_size, rank, bias=False, dtype=dtype)
        self.up = nn.Linear(rank, hidden_size, bias=False, dtype=dtype)

    def forward(self, embeddings: torch.Tensor, mtp_hidden: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3 or embeddings.shape[1] != 8 or mtp_hidden.shape != embeddings[:, 1].shape:
            raise ValueError("DMTP needs [batch, 8, hidden] embeddings and [batch, hidden] MTP predictor states.")
        hidden = mtp_hidden.float()
        normalized = (hidden * torch.rsqrt(hidden.square().mean(dim=-1, keepdim=True) + self.eps)).to(
            self.down.weight.dtype
        )
        residual = self.up(F.silu(self.down(normalized)))
        return torch.cat((embeddings[:, :1], embeddings[:, 1:2] + residual[:, None], embeddings[:, 2:]), dim=1)


class Qwen3DMTPModel(DFlash2Qwen3Model):
    def __init__(self, *, vllm_config: VllmConfig, start_layer_id: int = 0, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix)
        config = self.config
        # Only the context projection has zero-centered gamma in this export.
        # The transformer, Q/K and final norms use ordinary multiplicative weights.
        self.hidden_norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mtp_prefix_conditioner = MTPT1Residual(
            config.hidden_size,
            config.dflash_config["mtp_prefix_rank"],
            config.dflash_config["mtp_prefix_eps"],
            vllm_config.model_config.dtype,
        )

    def _normalize_context_states(self, context_states: torch.Tensor) -> torch.Tensor:
        return self.hidden_norm(context_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        source_names: set[str] = set()

        def record_sources():
            for name, tensor in weights:
                source_names.add(name)
                yield name, tensor

        loaded = super().load_weights(record_sources())
        # vLLM shares the embedding with the target after draft loading. The
        # exported MASK uses that embedding too. Everything else is trained.
        required = set(dict(self.named_parameters())) - {"embed_tokens.weight", "mask_embedding"}
        missing = required - loaded
        # The loader reports a packed QKV or gate/up parameter as loaded after
        # seeing any one component. Check the source names for each split pack.
        for source, (packed, _) in self.hf_to_vllm_mapper.orig_to_new_stacked.items():
            suffix = packed + ".weight"
            for name in loaded - source_names:
                if name.endswith(suffix):
                    component = name.removesuffix(suffix) + source + ".weight"
                    if component not in source_names:
                        missing.add(component)
        if missing:
            raise ValueError(f"Incomplete DMTP draft checkpoint. Missing weights: {sorted(missing)}")
        return loaded


class Qwen3DMTPForCausalLM(DFlash2Qwen3ForCausalLM):
    model_cls = Qwen3DMTPModel
    speculator_config_class = AscendDMTPSpeculatorConfig

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        spec = vllm_config.speculative_config
        if spec is None or spec.custom_method != "dmtp":
            raise ValueError("DFlash2T1ResidualDraftModel requires speculative method 'dmtp'.")
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def embed_dmtp_inputs(self, input_ids: torch.Tensor, mtp_hidden: torch.Tensor) -> torch.Tensor:
        """Embed [A, T1, MASK × 6], applying the learned residual only at T1."""
        if input_ids.ndim != 2 or input_ids.shape[1] != 8:
            raise ValueError("DMTP expects eight input slots per request: A, T1 and six masks.")
        embeddings = self.embed_input_ids(input_ids.flatten()).view(*input_ids.shape, -1)
        return self.model.mtp_prefix_conditioner(embeddings, mtp_hidden)
