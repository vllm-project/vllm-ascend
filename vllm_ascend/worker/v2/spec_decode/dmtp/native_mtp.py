# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Eager Qwen3.6 native predictor with private, token-aligned causal KV."""

import copy
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)

if TYPE_CHECKING:
    from transformers import Qwen3_5TextConfig
    from vllm.config import VllmConfig

# Bound the temporary eager scores (num_heads × query rows × context length).
PREFILL_CHUNK_SIZE = 128


@dataclass(frozen=True)
class NativeMTPCache:
    keys: torch.Tensor  # [1, kv_heads, max_model_len, head_dim]
    values: torch.Tensor
    length: int = 0

    def update(self, keys: torch.Tensor, values: torch.Tensor, layer_idx: int, *args, **kwargs):
        """HF decoder attention calls this once. Only committed rows are appended."""
        end = self.length + keys.shape[2]
        self.keys[:, :, self.length : end].copy_(keys)
        self.values[:, :, self.length : end].copy_(values)
        return self.keys[:, :, :end], self.values[:, :, :end]


class _NativeMTPWeights(nn.Module):
    """Matches the target checkpoint's mtp.* names and upstream Qwen merge order."""

    def __init__(self, config: "Qwen3_5TextConfig"):
        super().__init__()
        size = config.hidden_size
        self.fc = nn.Linear(2 * size, size, bias=False)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(size, config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(size, config.rms_norm_eps)
        self.norm = Qwen3_5RMSNorm(size, config.rms_norm_eps)
        # Target layer 0 is GDN. The appended native predictor is full attention.
        layer_config = copy.copy(config)
        layer_config.layer_types = ["full_attention"]
        layer_config._attn_implementation = "eager"
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(layer_config, layer_idx=0)])

    def forward(self, embeds, previous_hidden, mask, position_embeddings, cache):
        hidden = self.fc(torch.cat((self.pre_fc_norm_embedding(embeds), self.pre_fc_norm_hidden(previous_hidden)), -1))
        hidden = self.layers[0](
            hidden.unsqueeze(0),
            position_embeddings=position_embeddings,
            attention_mask=mask,
            past_key_values=cache,
        )
        return self.norm(hidden.squeeze(0))


def _load_weights(model_path, revision, download_dir, weights: _NativeMTPWeights, device: torch.device):
    """Read only mtp tensors from the target's indexed safetensors shards."""
    local_dir = Path(model_path)

    def resolve(filename):
        if local_dir.is_dir():
            return local_dir / filename
        return Path(hf_hub_download(model_path, filename, cache_dir=download_dir, revision=revision))

    weight_map = json.loads(resolve("model.safetensors.index.json").read_text())["weight_map"]
    names_by_shard = defaultdict(list)
    expected = weights.state_dict()
    native_keys = {k.removeprefix("mtp.") for k in weight_map if k.startswith("mtp.")}
    if native_keys != expected.keys():
        raise ValueError(
            f"Unsupported target MTP checkpoint; missing={sorted(expected.keys() - native_keys)}, "
            f"unexpected={sorted(native_keys - expected.keys())}"
        )
    for name in sorted(native_keys):
        names_by_shard[weight_map[f"mtp.{name}"]].append(name)
    loaded = {}
    for shard, names in names_by_shard.items():
        with safe_open(resolve(shard), framework="pt", device="cpu") as reader:
            for name in names:
                tensor = reader.get_tensor(f"mtp.{name}")
                if tensor.dtype != torch.bfloat16:
                    raise ValueError(f"Expected BF16 target MTP weight mtp.{name}, got {tensor.dtype}")
                loaded[name] = tensor.to(device=device)
    weights.load_state_dict(loaded, strict=True, assign=True)


class NativeMTP:
    """Replicated single-layer MTP. Target embedding and LM head retain their TP layout.

    A native row consumes token at position i and final normalized target hidden
    at i - 1. Its normalized output scores token i + 1 and conditions DFlash2.
    KV row zero therefore corresponds to absolute text position one.
    """

    def __init__(
        self,
        weights: _NativeMTPWeights,
        config: "Qwen3_5TextConfig",
        embed: Callable[[torch.Tensor], torch.Tensor],
        max_reqs: int,
        max_tokens: int,
    ):
        self.weights = weights.eval()
        self.embed = embed
        device = weights.fc.weight.device
        # RoPE's nonpersistent inverse frequencies must remain FP32.
        self.rope = Qwen3_5TextRotaryEmbedding(config).to(device=device)
        shape = (2, max_reqs, config.num_key_value_heads, max_tokens, config.head_dim)
        self.kv_pool = torch.empty(shape, dtype=weights.fc.weight.dtype, device=device)

    @property
    def cache_bytes_per_token(self) -> int:
        return 2 * self.kv_pool.shape[2] * self.kv_pool.shape[4] * self.kv_pool.element_size()

    @classmethod
    def from_target(cls, vllm_config: "VllmConfig", target_model: nn.Module) -> "NativeMTP":
        model_config = vllm_config.model_config
        config = model_config.hf_text_config
        if (
            config.model_type != "qwen3_5_text"
            or getattr(config, "mtp_num_hidden_layers", 0) != 1
            or getattr(config, "mtp_use_dedicated_embeddings", False)
            or model_config.quantization is not None
            or model_config.dtype != torch.bfloat16
        ):
            raise ValueError("DMTP requires BF16 dense Qwen3.6 with one native MTP layer and shared target embeddings")
        language_model = getattr(target_model, "language_model", target_model)
        device = language_model.model.embed_tokens.weight.device
        with torch.device("meta"):
            weights = _NativeMTPWeights(config)
        if vllm_config.load_config.load_format == "dummy":
            from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight

            weights = weights.to(dtype=model_config.dtype).to_empty(device=device)
            for tensor in weights.state_dict().values():
                initialize_single_dummy_weight(tensor)
        else:
            _load_weights(
                model_config.model, model_config.revision, vllm_config.load_config.download_dir, weights, device
            )
        return cls(
            weights,
            config,
            language_model.embed_input_ids,
            vllm_config.scheduler_config.max_num_seqs,
            model_config.max_model_len,
        )

    def empty_cache(self, slot: int) -> NativeMTPCache:
        if not 0 <= slot < self.kv_pool.size(1):
            raise ValueError(f"Invalid DMTP KV slot {slot}")
        return NativeMTPCache(self.kv_pool[0, slot : slot + 1], self.kv_pool[1, slot : slot + 1])

    @torch.inference_mode()
    def append(
        self,
        cache: NativeMTPCache,
        input_ids: torch.Tensor,
        previous_target_hidden: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[NativeMTPCache, torch.Tensor]:
        n = input_ids.numel()
        if input_ids.ndim != 1 or not n or previous_target_hidden.shape != (n, self.weights.fc.out_features):
            raise ValueError("Expected one committed token ID and one previous target hidden row per native position")
        end = cache.length + n
        if end > cache.keys.size(2):
            raise ValueError("Native DMTP KV capacity exceeded")
        if positions is None:
            positions = torch.arange(cache.length + 1, end + 1, device=input_ids.device)
        elif positions.shape != (n,):
            raise ValueError("Expected one absolute text position per native token")

        outputs = []
        for start in range(0, n, PREFILL_CHUNK_SIZE):
            stop = min(start + PREFILL_CHUNK_SIZE, n)
            embeds = self.embed(input_ids[start:stop])
            pos = positions[start:stop].reshape(1, 1, -1)
            position_embeddings = self.rope(embeds, pos)
            # HF eager attention consumes an additive mask. A rectangular causal
            # mask must align its diagonal to the end of the *cached* prefix.
            chunk_len = stop - start
            mask = None
            if chunk_len > 1:
                mask = embeds.new_full((chunk_len, cache.length + chunk_len), float("-inf"))
                mask = mask.triu(diagonal=cache.length + 1)
            hidden = self.weights(embeds, previous_target_hidden[start:stop], mask, position_embeddings, cache)
            outputs.append(hidden)
            cache = NativeMTPCache(cache.keys, cache.values, cache.length + chunk_len)
        return cache, outputs[0] if len(outputs) == 1 else torch.cat(outputs)
