# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared model matrix and assertions for one/two-card weight-update E2Es."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

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
    extra_server_args: tuple[str, ...] = ()

    def server_args(self) -> list[str]:
        return ["--hf-overrides", json.dumps(self.hf_overrides), *self.extra_server_args]


# Qwen and GLM reduce only layer count. DeepSeek keeps the first four
# compress-ratio entries and reduces experts to fit the single-NPU IPC path;
# its original top-6 routing remains unchanged and valid with eight experts.
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
        extra_server_args=("--tokenizer-mode", "deepseek_v4"),
    ),
    WeightUpdateModelCase(
        id="glm-5.1-sfa-derived-kv",
        model="zai-org/GLM-5.1",
        hf_overrides={
            "num_hidden_layers": 4,
        },
    ),
)


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

        parameters = [
            (self._checkpoint_name(name, case), tuple(parameter.shape))
            for name, parameter in meta_model.named_parameters()
        ]
        del meta_model

        assert parameters, f"{case.id}: reduced meta model contains no parameters"
        names = [name for name, _ in parameters]
        assert len(set(names)) == len(names), f"{case.id}: generated checkpoint parameter names are not unique"
        self._parameters = parameters
        self._device = device

    @staticmethod
    def _checkpoint_name(name: str, case: WeightUpdateModelCase) -> str:
        if case.checkpoint_model_prefix is not None and name.startswith("model."):
            return case.checkpoint_model_prefix + name.removeprefix("model.")
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
