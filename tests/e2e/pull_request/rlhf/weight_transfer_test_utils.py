# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared model matrix and assertions for one/two-card weight-update E2Es."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import regex as re
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource


@dataclass(frozen=True)
class WeightUpdateModelCase:
    """A layer-reduced model whose complete parameter set is reloaded."""

    id: str
    model: str
    hf_overrides: dict[str, Any]
    meta_config_attribute: str | None = None
    checkpoint_model_prefix: str | None = None
    checkpoint_name_map: Callable[[str], str] | None = None
    expert_intermediate_size: int | None = None
    extra_server_args: tuple[str, ...] = ()

    def server_args(self) -> list[str]:
        # Run the worker out-of-process. With a single-process executor the
        # worker shares the engine core's ``VllmConfig``, and ``EngineCoreProc``
        # rewrites ``cache_config.block_size`` to the minimum block size across
        # the KV cache groups before the worker recomputes the KV cache specs.
        # DeepSeek-V4's block-size tables are keyed by the user-facing 32/64/128,
        # so the rewritten value (8/4/2) raises ``KeyError`` on every lookup. An
        # out-of-process worker gets its own config copy, taken before that
        # rewrite, and the single-worker topology of these cases is unchanged.
        return [
            "--hf-overrides",
            json.dumps(self.hf_overrides),
            "--distributed-executor-backend",
            "mp",
            *self.extra_server_args,
        ]


# Qwen and GLM reduce only layer count. DeepSeek keeps the first four
# compress-ratio entries and reduces experts to fit the single-NPU IPC path;
# its original top-6 routing remains unchanged and valid with eight experts.
_DSV4_SELF_ATTN_RENAMES = (
    # Descriptive HF names -> the compact names the checkpoint/loader expect.
    ("self_attn.compressor.position_bias", "self_attn.compressor.ape"),
    ("self_attn.compressor.kv_proj.weight", "self_attn.compressor.wkv.weight"),
    ("self_attn.compressor.gate_proj.weight", "self_attn.compressor.wgate.weight"),
    ("self_attn.compressor.kv_norm.weight", "self_attn.compressor.norm.weight"),
    ("self_attn.q_a_proj.weight", "self_attn.wq_a.weight"),
    ("self_attn.q_a_norm.weight", "self_attn.q_norm.weight"),
    ("self_attn.q_b_proj.weight", "self_attn.wq_b.weight"),
    ("self_attn.kv_proj.weight", "self_attn.wkv.weight"),
    ("self_attn.o_a_proj.weight", "self_attn.wo_a.weight"),
    ("self_attn.o_b_proj.weight", "self_attn.wo_b.weight"),
)

_DSV4_EXPERT_SHARDS = {"gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"}


def deepseek_v4_checkpoint_name(name: str) -> str:
    """Translate an HF DeepSeek-V4 parameter name into its served-model form.

    ``DeepSeek-V4-Flash`` differs from the HF ``from_config`` model in ways that
    ``load_weights`` cannot bridge on its own:

    * routed experts ship as per-expert ``w1/w3/w2`` (HF: fused ``gate_up_proj``
      / ``down_proj``); the loader fuses them into ``w13``/``w2``;
    * the compressor lives *under* the indexer (HF: the indexer sits under the
      compressor) and its norm is called ``norm`` (HF/checkpoint: ``kv_norm``);
    * ``wq_b``/``weights_proj`` belong to the indexer itself, and the attention
      projections use ``wq_a``/``wkv``/``wo_a``/``wo_b``.

    Without this mapping ``load_weights`` raises ``KeyError`` on the first
    unmatched tensor.
    """
    if name == "model.embed_tokens.weight":
        return "embed.weight"
    if name == "model.norm.weight":
        return "norm.weight"
    if name == "lm_head.weight":
        return "head.weight"

    m = re.match(r"^model\.hc_head\.hc_(base|fn|scale)$", name)
    if m:
        return f"hc_head_{m.group(1)}"
    m = re.match(r"^model\.layers\.(\d+)\.(attn|ffn)_hc\.(base|fn|scale)$", name)
    if m:
        return f"layers.{m.group(1)}.hc_{m.group(2)}_{m.group(3)}"
    m = re.match(r"^model\.layers\.(\d+)\.self_attn\.sinks$", name)
    if m:
        return f"layers.{m.group(1)}.attn.attn_sink"

    m = re.match(r"^(.*\.mlp\.experts\.\d+)\.(gate_proj|down_proj|up_proj)\.weight$", name)
    if m:
        return f"{m.group(1)}.{_DSV4_EXPERT_SHARDS[m.group(2)]}.weight"

    m = re.match(r"^(.*\.self_attn)\.compressor\.indexer\.(.+)$", name)
    if m:
        base = m.group(1)
        tail = {
            "position_bias": "ape",
            "kv_proj.weight": "wkv.weight",
            "gate_proj.weight": "wgate.weight",
            "kv_norm.weight": "norm.weight",
            "q_b_proj.weight": "wq_b.weight",
            "scorer.weights_proj.weight": "weights_proj.weight",
        }.get(m.group(2), m.group(2))
        if tail.startswith(("wq_b.", "weights_proj.")):
            return f"{base}.indexer.{tail}"
        return f"{base}.indexer.compressor.{tail}"

    for old, new in _DSV4_SELF_ATTN_RENAMES:
        if name.endswith(old):
            return name[: -len(old)] + new
    return name


MODEL_CASES = (
    WeightUpdateModelCase(
        id="qwen3.5-35b-a3b-moe-layout",
        model="Qwen/Qwen3.5-35B-A3B",
        hf_overrides={
            "architectures": ["Qwen3_5MoeForCausalLM"],
            "text_config": {
                "num_hidden_layers": 4,
                "layer_types": [
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                ],
            },
        },
        # The public checkpoint is multimodal, while this test serves its
        # language model architecture. Enumerate every parameter from the text
        # config so visual parameters are neither skipped nor sent by mistake.
        meta_config_attribute="text_config",
        checkpoint_model_prefix="model.language_model.",
    ),
    WeightUpdateModelCase(
        id="deepseek-v4-flash-bf16-derived-gate",
        model="kylesayrs/DeepSeek-V4-Flash-bf16",
        hf_overrides={
            "num_hidden_layers": 4,
            "n_routed_experts": 8,
        },
        checkpoint_name_map=deepseek_v4_checkpoint_name,
        # The HF ``from_config`` model sizes every expert MLP by
        # ``intermediate_size`` (18432), while the checkpoint - and therefore the
        # served model - use ``moe_intermediate_size`` (2048); verified against
        # the real safetensors header of ``experts.0.w1.weight`` == [2048, 4096].
        expert_intermediate_size=2048,
        extra_server_args=("--tokenizer-mode", "deepseek_v4"),
    ),
    WeightUpdateModelCase(
        id="glm-5.1-sfa-derived-kv",
        model="zai-org/GLM-5.1",
        hf_overrides={
            "num_hidden_layers": 4,
            # NPU IPC co-locates the trainer payload with the server on one
            # chip, so 256 experts x 24 GiB of weights cannot fit alongside the
            # reload headroom on a 64 GiB device. Mirror the DeepSeek case's
            # single-chip IPC budget by keeping only the first 8 experts.
            "n_routed_experts": 8,
        },
    ),
)


def expand_fused_expert_params(name: str, shape: tuple[int, ...]) -> list[tuple[str, tuple[int, ...]]]:
    """Map HF fused routed-expert tensors onto checkpoint-facing per-expert names.

    ``GlmMoeDsaForCausalLM`` keeps routed experts fused as
    ``mlp.experts.gate_up_proj`` ``[E, 2I, H]`` and ``mlp.experts.down_proj``
    ``[E, H, I]``, while the published checkpoint - and therefore vLLM's
    ``expert_params_mapping`` - stores one tensor per expert
    (``mlp.experts.{e}.gate_proj.weight`` etc.).  Without this expansion the
    fused names reach ``load_weights`` unmatched and raise
    ``KeyError: 'layers.N.mlp.experts.gate_up_proj'``.
    """
    for suffix, kind in ((".mlp.experts.gate_up_proj", "gate_up"), (".mlp.experts.down_proj", "down")):
        if not name.endswith(suffix):
            continue
        prefix = name[: -len(suffix.rsplit(".", 1)[-1])]  # keeps the trailing dot
        if kind == "gate_up":
            num_experts, fused_inter, hidden = shape
            inter = fused_inter // 2
            out: list[tuple[str, tuple[int, ...]]] = []
            for expert in range(num_experts):
                out.append((f"{prefix}{expert}.gate_proj.weight", (inter, hidden)))
                out.append((f"{prefix}{expert}.up_proj.weight", (inter, hidden)))
            return out
        num_experts, hidden, inter = shape
        return [(f"{prefix}{expert}.down_proj.weight", (hidden, inter)) for expert in range(num_experts)]
    return [(name, shape)]


def resize_expert_shape(name: str, shape: tuple[int, ...], intermediate_size: int | None) -> tuple[int, ...]:
    """Force expert MLP tensors to the checkpoint's intermediate dimension.

    ``DeepSeek-V4`` builds routed and shared experts from ``intermediate_size``
    in HF's ``from_config`` path but stores ``moe_intermediate_size`` in the
    checkpoint, so the payload shape has to be corrected or vLLM's loader trips
    ``assert args[0].numel() == args[1].numel()`` while copying an expert shard.
    """
    if intermediate_size is None:
        return shape
    if name.endswith((".mlp.shared_experts.gate_proj.weight", ".mlp.shared_experts.up_proj.weight")):
        return (intermediate_size, shape[-1])
    if name.endswith(".mlp.shared_experts.down_proj.weight"):
        return (shape[0], intermediate_size)
    if ".mlp.experts." in name and name.endswith((".gate_proj.weight", ".up_proj.weight")):
        return (intermediate_size, shape[-1])
    if ".mlp.experts." in name and name.endswith(".down_proj.weight"):
        return (shape[0], intermediate_size)
    return shape


def pytest_model_cases() -> list[Any]:
    """Return model params with per-checkpoint CI discovery markers."""
    import pytest

    return [pytest.param(case, id=case.id, marks=pytest.mark.e2e_model(case.model)) for case in MODEL_CASES]


def _apply_hf_overrides(config, overrides: dict[str, Any]) -> None:
    for name, value in overrides.items():
        current = getattr(config, name, None)
        if isinstance(value, dict) and current is not None:
            current.update(value)
        else:
            setattr(config, name, value)


FIXED_WEIGHT_SEED = 20260915


class FixedRandomWeightSource(WeightSource):
    """Generate every reduced-model parameter deterministically on demand.

    A meta-device Transformers model provides the complete checkpoint-facing
    name and shape set without allocating model storage. Each iteration derives
    a stable per-parameter seed and regenerates the same BF16 values, avoiding
    both checkpoint weight downloads and a persistent second model copy.
    """

    def __init__(self, case: WeightUpdateModelCase, device: torch.device) -> None:
        config = AutoConfig.from_pretrained(case.model, trust_remote_code=True)
        _apply_hf_overrides(config, case.hf_overrides)
        meta_config = getattr(config, case.meta_config_attribute) if case.meta_config_attribute else config
        with torch.device("meta"):
            meta_model = AutoModelForCausalLM.from_config(meta_config, trust_remote_code=True)

        # Two distinct steps: ``_checkpoint_name`` applies the namespace prefix
        # to the raw meta-model name (once), then the checkpoint rename map runs
        # on the *expanded* names, because the meta model exposes routed experts
        # fused and the per-expert names (gate/up/down -> w1/w3/w2) only exist
        # after ``expand_fused_expert_params`` has split them.
        parameters = [
            (
                self._apply_name_map(expanded_name, case),
                resize_expert_shape(expanded_name, expanded_shape, case.expert_intermediate_size),
            )
            for name, parameter in meta_model.named_parameters()
            for expanded_name, expanded_shape in expand_fused_expert_params(
                self._checkpoint_name(name, case), tuple(parameter.shape)
            )
        ]
        del meta_model

        assert parameters, f"{case.id}: reduced meta model contains no parameters"
        names = [name for name, _ in parameters]
        assert len(set(names)) == len(names), f"{case.id}: generated checkpoint parameter names are not unique"
        self._parameters = parameters
        self._device = device

    @staticmethod
    def _checkpoint_name(name: str, case: WeightUpdateModelCase) -> str:
        """Apply the case's checkpoint namespace prefix exactly once."""
        if case.checkpoint_model_prefix is not None and name.startswith("model."):
            return case.checkpoint_model_prefix + name.removeprefix("model.")
        return name

    @staticmethod
    def _apply_name_map(name: str, case: WeightUpdateModelCase) -> str:
        """Apply the case's checkpoint rename map (idempotent per rule)."""
        if case.checkpoint_name_map is not None:
            return case.checkpoint_name_map(name)
        return name

    @staticmethod
    def _parameter_seed(name: str) -> int:
        digest = hashlib.sha256(f"{FIXED_WEIGHT_SEED}:{name}".encode()).digest()
        return int.from_bytes(digest[:8], "little") % (2**63 - 1)

    def _make_tensor(self, name: str, shape: tuple[int, ...]) -> torch.Tensor:
        seed = self._parameter_seed(name)
        torch.manual_seed(seed)
        torch.npu.manual_seed(seed)
        tensor = torch.empty(shape, dtype=torch.bfloat16, device=self._device)
        if name.endswith("norm.weight"):
            return tensor.uniform_(0.9, 1.1)
        return tensor.uniform_(-0.02, 0.02)

    def metadata(self) -> list[ParamMeta]:
        return [ParamMeta(name, torch.bfloat16, shape) for name, shape in self._parameters]

    def __iter__(self):
        with torch.no_grad():
            for name, shape in self._parameters:
                yield name, self._make_tensor(name, shape)


PROMPTS = [
    "The capital of France is",
    "Explain why the sky is blue in one sentence:",
]


def generation_signature(client, model: str) -> list[tuple[str, tuple[float, ...]]]:
    """Capture deterministic text and logprobs for exact reload comparison."""
    signature = []
    for prompt in PROMPTS:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=8,
            temperature=0,
            logprobs=1,
            seed=0,
        )
        choice = response.choices[0]
        token_logprobs = tuple(choice.logprobs.token_logprobs or ())
        assert token_logprobs, f"{model}: generation returned no token logprobs"
        signature.append((choice.text, token_logprobs))
    return signature


def assert_dummy_then_fixed_reload(
    dummy_signature,
    fixed_baseline,
    reloaded_signature,
    case: WeightUpdateModelCase,
) -> None:
    assert fixed_baseline != dummy_signature, f"{case.id}: loading the fixed complete payload had no observable effect"
    assert reloaded_signature == fixed_baseline, (
        f"{case.id}: reloading the same fixed complete payload changed FULL_DECODE_ONLY output"
    )
