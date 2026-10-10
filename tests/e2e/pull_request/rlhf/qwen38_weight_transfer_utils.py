# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-PLE device mapping, worker checks and checkpoint-backed E2E setup."""

import os
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer

from tests.e2e.pull_request.rlhf.prepare_qwen38_checkpoint import PROFILE
from tests.e2e.pull_request.rlhf.weight_transfer_test_utils import PROMPTS, WeightUpdateModelCase

LONG_PROMPT_TOKENS = 3072
MEMORY_UTILIZATION = 0.45
CONTROL_TIMEOUT = 120
FAMILIES = ("QwenGatedDeltaNetAttention", "Qwen4ExpQSAAttention", "Qwen4ExpSparseMoeBlock", "GatedResidual")


def checkpoint_case(request) -> tuple[WeightUpdateModelCase, Path, str, list[str | list[int]]]:
    checkpoint = request.config.getoption("--qwen38-no-ple-checkpoint")
    digest = request.config.getoption("--qwen38-no-ple-manifest-sha256")
    if not checkpoint or not digest:
        raise ValueError("Qwen3.8 no-PLE tests require --qwen38-no-ple-checkpoint and --qwen38-no-ple-manifest-sha256")
    directory = Path(checkpoint).resolve()
    case = WeightUpdateModelCase(
        id=PROFILE,
        model=str(directory),
        hf_overrides={},
        extra_server_args=("--language-model-only", "--worker-extension-cls", f"{__name__}.NoPLEWorkerProbe"),
        max_model_len=4096,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
    )
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    tokens = tokenizer.encode("Explain how communication transfers model weights. ", add_special_tokens=False)
    if not tokens:
        raise ValueError("Tokenizer returned an empty prompt")
    long_prompt = (tokens * (LONG_PROMPT_TOKENS // len(tokens) + 1))[:LONG_PROMPT_TOKENS]
    return case, directory, digest, [*PROMPTS, long_prompt]


def physical_device(logical_index: int) -> str:
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    return visible.split(",")[logical_index].strip() if visible else str(logical_index)


def post(server, route: str, **kwargs):
    response = requests.post(server.url_for(route), timeout=CONTROL_TIMEOUT, **kwargs)
    response.raise_for_status()
    return response


def worker_rpc(server, method: str):
    return post(server, "collective_rpc", json={"method": method}).json()


class NoPLEWorkerProbe:
    """Read-only module checks and eager prefill hooks; no QSA top-k claim."""

    model_runner: Any
    _family_calls: dict[str, int]
    _probe_hooks: list[Any]
    _probe_methods: list[tuple[Any, str, Any]]

    def install_no_ple_probe(self) -> str:
        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import npu_generate_uuid

        model = self.model_runner.get_model()
        text = self.model_runner.model_config.hf_config.text_config
        assert text.num_hidden_layers == 4 and text.ple_layer_ids == []
        assert text.num_experts == 512 and text.num_experts_per_tok == 10
        assert not any("ple" in name.split(".") for name, _ in model.named_parameters())
        assert not any("ple" in name.split(".") for name, _ in model.named_modules())
        layers = [m for m in model.modules() if type(m).__name__ == "Qwen4ExpDecoderLayer"]
        assert len(layers) == 4
        assert all(getattr(layer, "ple", None) is None for layer in layers)
        self._family_calls = dict.fromkeys(FAMILIES, 0)
        self._probe_hooks = []
        self._probe_methods = []
        for family in FAMILIES:
            modules = [m for m in model.modules() if type(m).__name__ == family]
            assert modules, f"Missing model family: {family}"
            for module in modules:
                if family == "GatedResidual":
                    # Hyperconnections run mix/combine_and_mix directly rather
                    # than Module.forward, so forward hooks cannot observe them.
                    for method_name in ("mix", "combine_and_mix"):
                        original = getattr(module, method_name)

                        def record_method(*args, original=original, **kwargs):
                            self._family_calls["GatedResidual"] += 1
                            return original(*args, **kwargs)

                        self._probe_methods.append((module, method_name, original))
                        setattr(module, method_name, record_method)
                    continue

                def record_call(_module, _args, family=family):
                    self._family_calls[family] += 1

                self._probe_hooks.append(module.register_forward_pre_hook(record_call))
        return npu_generate_uuid()

    def check_no_ple_execution(self) -> dict[str, int]:
        assert all(self._family_calls.values()), self._family_calls
        for hook in self._probe_hooks:
            hook.remove()
        self._probe_hooks.clear()
        for module, method_name, original in self._probe_methods:
            setattr(module, method_name, original)
        self._probe_methods.clear()
        return self._family_calls
