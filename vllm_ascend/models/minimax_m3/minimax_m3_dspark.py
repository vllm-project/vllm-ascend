# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Experimental M3.1 Flash DSpark adapter using the existing Ascend protocol.

Only draft model differences are implemented here: Flash norms, MLP, the
standalone-checkpoint configuration contract and native BF16/MXFP8 weight
mapping. Target verification, Markov drafting, confidence, batching and
cache metadata use the supplied vLLM/vLLM-Ascend implementations unchanged.

``patch/platform/patch_speculative_config.py`` wires ``draft_hf_override``
into ``SpeculativeConfig``. The draft geometry comes from the checkpoint's
own ``dspark/config.json``; ``draft_config_from_file`` only adds the
aliases the DSpark stack reads and corrects the export's target-model
leftovers.
"""

import json
from pathlib import Path

import regex as re
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3Attention,
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    _resolve_layer_attention,
)
from vllm.model_executor.models.qwen3_dspark import Qwen3DSparkModel

from vllm_ascend.models.minimax_m3.minimax_m3 import MiniMaxM3MLP
from vllm_ascend.models.qwen3_dspark import AscendQwen3DSparkForCausalLM

PREFIX = "language_model.model.dspark."
ADAPTER_ARCHITECTURE = "DSparkMiniMaxDraftModel"
MAX_SAFETENSORS_HEADER_BYTES = 10 * 1024 * 1024
MXFP8_GROUP_SIZE = 32

_MXFP8_PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.qkv_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.gate_up_proj",
    "mlp.down_proj",
)


def map_weight_name(name):
    if not name.startswith(PREFIX):
        raise ValueError(f"Unexpected tensor outside standalone DSpark: {name}")
    name = name[len(PREFIX) :]
    name = re.sub(r"^(layers\.\d+)\.decoder_layer\.", r"\1.", name)
    if name.startswith("final_norm."):
        name = "norm." + name[len("final_norm.") :]
    return name


def expected_shapes(config):
    h = config["hidden_size"]
    hd = config["head_dim"]
    q = config["num_attention_heads"] * hd
    kv = config["num_key_value_heads"] * hd
    ffn = config["dspark_ffn_hidden_size"]
    n = config["num_hidden_layers"]
    vocab = config["vocab_size"]
    rank = config["dspark_markov_rank"]
    shapes = {
        "fc.weight": [h, h * len(config["dspark_target_layer_ids"])],
        "hidden_norm.weight": [h],
        "norm.weight": [h],
        "markov_head.markov_w1.weight": [vocab, rank],
        "markov_head.markov_w2.weight": [vocab, rank],
        "confidence_head.proj.weight": [1, h + rank],
        "confidence_head.proj.bias": [1],
    }
    for i in range(n):
        p = f"layers.{i}."
        for norm in ("input_layernorm", "post_attention_layernorm"):
            shapes[p + norm + ".weight"] = [h]
        for norm in ("q_norm", "k_norm"):
            shapes[p + "self_attn." + norm + ".weight"] = [hd]
        matrices = {
            "self_attn.q_proj": [q, h],
            "self_attn.k_proj": [kv, h],
            "self_attn.v_proj": [kv, h],
            "self_attn.o_proj": [h, q],
            "mlp.gate_proj": [ffn, h],
            "mlp.up_proj": [ffn, h],
            "mlp.down_proj": [h, ffn],
        }
        for name, shape in matrices.items():
            shapes[p + name + ".weight"] = shape
    return shapes


def validate_header(header, config):
    expected = expected_shapes(config)
    mapped = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        mapped_name = map_weight_name(name)
        if mapped_name in mapped:
            raise ValueError(f"Duplicate mapped tensor: {mapped_name}")
        mapped[mapped_name] = info
    uses_mxfp8 = any(info["dtype"] == "F8_E4M3" for info in mapped.values())
    quantized = {name: shape for name, shape in expected.items() if name.startswith("layers.") and len(shape) == 2}
    scales = (
        {
            name.removesuffix("weight") + "weight_scale_inv": [
                shape[0],
                (shape[1] + MXFP8_GROUP_SIZE - 1) // MXFP8_GROUP_SIZE,
            ]
            for name, shape in quantized.items()
        }
        if uses_mxfp8
        else {}
    )
    schema = set(expected) | set(scales)
    if set(mapped) != schema:
        raise ValueError(f"Tensor schema mismatch: missing={schema - set(mapped)}, extra={set(mapped) - schema}")
    for name, shape in {**expected, **scales}.items():
        if list(mapped[name]["shape"]) != shape:
            raise ValueError(f"Tensor shape mismatch: {name}, {mapped[name]['shape']} != {shape}")
        dtype = mapped[name]["dtype"]
        expected_dtype = "U8" if name in scales else "F8_E4M3" if uses_mxfp8 and name in quantized else "BF16"
        if dtype != expected_dtype:
            raise ValueError(f"Expected native {expected_dtype} DSpark tensor, got {name}: {dtype}")
    return mapped


def iter_draft_weights(weights, config):
    # Keep header-only configuration preparation independent of Torch imports.
    import torch

    tensors = {}
    for name, value in weights:
        if name in tensors:
            raise ValueError(f"Duplicate input tensor: {name}")
        tensors[name] = value
    header = {}
    for name, value in tensors.items():
        header[name] = {
            "shape": list(value.shape),
            "dtype": {torch.bfloat16: "BF16", torch.float8_e4m3fn: "F8_E4M3", torch.uint8: "U8"}.get(
                value.dtype, str(value.dtype)
            ),
        }
    validate_header(header, config)
    for name, value in tensors.items():
        yield map_weight_name(name).replace("weight_scale_inv", "weight_scale"), value


def read_draft_header(model_root):
    """Read draft metadata without loading or converting weight tensors."""
    draft = Path(model_root) / "dspark"
    shards = sorted(draft.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"MiniMax DSpark weights are missing from {draft}")
    header = {}
    for shard in shards:
        with shard.open("rb") as stream:
            size = int.from_bytes(stream.read(8), "little")
            if not 0 < size < MAX_SAFETENSORS_HEADER_BYTES:
                raise ValueError(f"Invalid DSpark safetensors header size: {shard}")
            metadata = json.loads(stream.read(size))
        for name, info in metadata.items():
            if name == "__metadata__":
                continue
            if name in header:
                raise ValueError(f"Duplicate DSpark tensor across shards: {name}")
            header[name] = info
    return header


def draft_config_from_file(source, header):
    """Normalize the checkpoint's ``dspark/config.json`` for the Ascend stack.

    The draft geometry comes from the file itself. What remains here is the
    contract the DSpark speculator and this adapter read: alias names, norm
    defaults and corrections for the target-model leftovers every known export
    carries in its dense draft (MoE routing and FFN width). The export's own
    ``quantization_config`` describes the ModelSlim calibration, so MXFP8
    ascend sections are derived from the tensor header instead.
    """
    draft = source.get("text_config", source)
    architectures = source.get("architectures") or draft.get("architectures") or []
    if ADAPTER_ARCHITECTURE not in architectures:
        raise ValueError("Expected a MiniMax DSpark draft config, not a target model config")
    block_size = draft["dspark_block_size"]
    noise_token = draft["dspark_noise_token_id"]
    config = dict(draft)
    config.update(
        {
            "architectures": [ADAPTER_ARCHITECTURE],
            "model_type": "minimax_m3_text",
            "ascend_minimax_dspark": 1,
            # Aliases the upstream DSpark speculator reads.
            "n_predict": block_size,
            "target_layer_ids": list(draft["dspark_target_layer_ids"]),
            "markov_rank": draft.get("markov_rank", draft["dspark_markov_rank"]),
            "markov_head_type": draft.get("markov_head_type", draft.get("dspark_markov_head_type", "vanilla")),
            "dspark_markov_head_type": draft.get("dspark_markov_head_type", "vanilla"),
            "mask_token_id": noise_token,
            "ptd_token_id": noise_token,
            "draft_vocab_size": draft["vocab_size"],
            # Norm types and confidence flags the adapter requires.
            "dspark_hidden_norm_type": draft.get("dspark_hidden_norm_type", "gemma"),
            "dspark_final_norm_type": draft.get("dspark_final_norm_type", "gemma"),
            "enable_confidence_head": draft.get("enable_confidence_head", True),
            "confidence_head_with_markov": draft.get("confidence_head_with_markov", True),
            "use_aux_hidden_state": True,
            "sample_from_anchor": True,
            # Every known export copies the target's MoE routing and FFN width
            # into the dense draft; keep the draft dense.
            "intermediate_size": draft["dspark_ffn_hidden_size"],
            "num_local_experts": 0,
            "num_experts_per_tok": 0,
            "n_shared_experts": 0,
            "sparse_attention_config": None,
            "rope_parameters": draft.get("rope_parameters")
            or {
                "rope_type": "default",
                "rope_theta": draft["rope_theta"],
                "partial_rotary_factor": draft.get("partial_rotary_factor", draft["rotary_dim"] / draft["head_dim"]),
            },
        }
    )
    config.pop("quantization_config", None)
    mapped = validate_header(header, config)
    if any(info["dtype"] == "F8_E4M3" for info in mapped.values()):
        target_layers = (draft.get("dflash_config") or {}).get("num_target_layers")
        if not isinstance(target_layers, int):
            raise ValueError("MXFP8 DSpark exports must declare dflash_config.num_target_layers")
        quant = {"quant_method": "ascend", "group_size": MXFP8_GROUP_SIZE}
        # DFlash offsets runtime attention names past the target's layers.
        for layer in range(draft["num_hidden_layers"]):
            for projection in _MXFP8_PROJECTIONS:
                quant[f"model.layers.{target_layers + layer}.{projection}.weight"] = "W8A8_MXFP8"
        config["quantization_config"] = quant
    return config


def draft_hf_override(hf_config, *, model_root):
    # Called only for a MiniMax DSpark draft, through the target's copied
    # hf_overrides. A module-level callable remains picklable for workers.
    # HFConfigParser probes callables with an empty sentinel config before
    # loading the real checkpoint. Preserve its source config class selection.
    if hf_config.model_type.startswith("dummy_"):
        return hf_config
    from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig

    config_path = Path(model_root) / "dspark" / "config.json"
    source = json.loads(config_path.read_text())
    config = draft_config_from_file(source, read_draft_header(model_root))
    return MiniMaxM3TextConfig(**config)


class MiniMaxFlashDraftLayer(DFlashQwen3DecoderLayer):
    def __init__(self, vllm_config, *, config, layer_idx, cache_config=None, quant_config=None, prefix=""):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        sliding_window, causal = _resolve_layer_attention(config, layer_idx)
        self.self_attn = DFlashQwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=dict(config.rope_parameters),
            max_position=config.max_position_embeddings,
            head_dim=config.head_dim,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=False,
            sliding_window=sliding_window,
            causal=causal,
            is_neox_style=True,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn.q_norm = GemmaRMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.self_attn.k_norm = GemmaRMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.mlp = MiniMaxM3MLP(
            config,
            intermediate_size=config.dspark_ffn_hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class MiniMaxFlashDSparkModel(Qwen3DSparkModel):
    decoder_layer_cls = MiniMaxFlashDraftLayer

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        config = self.config
        for name, choice in (("hidden_norm", config.dspark_hidden_norm_type), ("norm", config.dspark_final_norm_type)):
            if choice not in ("rms", "gemma"):
                raise ValueError(f"Explicit draft norm type required for {name}")
            cls = GemmaRMSNorm if choice == "gemma" else RMSNorm
            setattr(self, name, cls(config.hidden_size, eps=config.rms_norm_eps))

    def _build_fused_kv_buffers(self):
        # Native MXFP8 projections retain their packed weights and scales.
        if self.quant_config is None:
            super()._build_fused_kv_buffers()

    def precompute_and_store_context_kv(self, context_states, context_positions, context_slot_mapping=None):
        # Use the actual norm modules (including Gemma's +1) and assign RoPE's
        # returned output. This remains correct for out-of-place NPU RoPE.
        length = context_states.shape[0]
        normalized = self.hidden_norm(context_states)
        if self.quant_config is None:
            if not hasattr(self, "_num_attn_layers"):
                self._build_fused_kv_buffers()
            layers, nkv, hd = self._num_attn_layers, self._num_kv_heads, self._head_dim
            kv = F.linear(normalized, self._fused_kv_weight, self._fused_kv_bias)
            kv = kv.view(length, layers, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            nkv, hd = attn.num_kv_heads, attn.head_dim
            if self.quant_config is not None:
                qkv, _ = attn.qkv_proj(normalized)
                _, keys, values = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
                keys, values = keys.reshape(length, nkv, hd), values.reshape(length, nkv, hd)
            else:
                keys, values = kv[0, i], kv[1, i]
            keys = attn.k_norm(keys).reshape(length, nkv * hd)
            keys, _ = attn.rotary_emb(context_positions, keys, None)
            if context_slot_mapping is None:
                continue
            slots = context_slot_mapping[i] if isinstance(context_slot_mapping, (list, tuple)) else context_slot_mapping
            if slots is None:
                continue
            inner = attn.attn
            inner.impl.do_kv_cache_update(inner, keys.view(length, nkv, hd), values, inner.kv_cache, slots)


class MiniMaxFlashDSparkForCausalLM(AscendQwen3DSparkForCausalLM):
    model_cls = MiniMaxFlashDSparkModel

    def __init__(self, *, vllm_config, prefix=""):
        config = vllm_config.speculative_config.draft_model_config.hf_config
        if getattr(config, "ascend_minimax_dspark", None) != 1:
            raise ValueError("Use the MiniMax M3.1 DSpark configuration")
        if not config.use_gemma_norm or config.hidden_act != "swigluoai":
            raise ValueError("Unexpected MiniMax draft decoder semantics")
        if getattr(config, "dspark_hybrid_context_fusion", False):
            raise ValueError("Hybrid context fusion is not implemented by this adapter")
        # DFlash's constructor honors model_cls; the DSpark constructor
        # currently hardcodes Qwen3DSparkModel. All DSpark methods are inherited.
        DFlashQwen3ForCausalLM.__init__(self, vllm_config=vllm_config, prefix=prefix)
        self.target_vocab_size = vllm_config.model_config.get_vocab_size()
        if config.vocab_size != self.target_vocab_size:
            raise ValueError("The unembedded Flash draft must match target vocabulary")
        self.enable_confidence_head = config.enable_confidence_head
        self.has_own_embed_tokens = False
        self.has_own_lm_head = False

    def load_weights(self, weights):
        normalized = iter_draft_weights(weights, self.config.to_dict())
        return super().load_weights(normalized)
