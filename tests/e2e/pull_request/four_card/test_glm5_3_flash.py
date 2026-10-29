# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Four-card MRV2 functional regression with synthetic Flash weights.

Keep the first nine Flash layers and production widths/experts. No checkpoint,
tokenizer download, C8, multimodal input or runtime patch is installed here.
Use the production multimodal model wrapper even for text-only requests: its
vision tower initializes before the language model, as in full-model serving.
Parallel coverage is TP4+EP with PCP=1 and DCP=1.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from vllm import SamplingParams
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

from tests.e2e.conftest import VllmRunner

NUM_LAYERS = 9
NUM_MTP_LAYERS = 1
NUM_SPECULATIVE_TOKENS = 5
OUTPUT_TOKENS = 16
TP_SIZE = 4
LOADER = "glm53flash_functional_dummy"


def _write_model(destination):
    assets = Path(__file__).with_name("glm53flash_assets")
    text_config = json.loads((assets / "config.json").read_text())
    text_config["num_nextn_predict_layers"] = NUM_MTP_LAYERS
    wrapper = json.loads((assets / "multimodal.json").read_text())
    config = {**wrapper, "text_config": {key: value for key, value in text_config.items() if key != "architectures"}}
    template = json.loads((assets / "quant.json").read_text())
    quant = {
        "model.language_model." + key[len("model.") :] if key.startswith("model.") else key: value
        for key, value in template.items()
        if not key.startswith("model.layers.")
    }
    for layer in range(NUM_LAYERS + NUM_MTP_LAYERS):
        source = 0 if layer < text_config["first_k_dense_replace"] else 3 if layer in (3, 7, NUM_LAYERS) else 4
        prefix = f"model.layers.{source}."
        for key, value in template.items():
            if not key.startswith(prefix):
                continue
            target = f"model.language_model.layers.{layer}." + key[len(prefix) :]
            if ".experts.0." in target:
                for expert in range(text_config["n_routed_experts"]):
                    quant[target.replace(".experts.0.", f".experts.{expert}.")] = value
            else:
                quant[target] = value
    for name in (
        "enorm.weight",
        "hnorm.weight",
        "eh_proj.weight",
        "shared_head.norm.weight",
        "shared_head.head.weight",
    ):
        quant[f"model.language_model.layers.{NUM_LAYERS}.{name}"] = "FLOAT"
    destination.mkdir()
    (destination / "config.json").write_text(json.dumps(config))
    (destination / "quant_model_description.json").write_text(json.dumps(quant))
    for name in ("processor_config.json", "tokenizer_config.json", "tokenizer.json"):
        (destination / name).write_bytes((assets / name).read_bytes())
    return config


@register_model_loader(LOADER)
class FlashDummyLoader(DummyModelLoader):
    """Test-only bounded values for recurrent state, mHC and quant scales.

    These are synthetic parameters, not a checkpoint or an accuracy baseline.
    Keep initialization local to this test rather than changing the model loader.
    """

    def load_weights(self, model, model_config):
        with torch.no_grad():
            for name, value in model.named_parameters():
                seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
                generator = torch.Generator(device=value.device).manual_seed(seed)
                if name.endswith("weight") and value.dtype == torch.int8:
                    value.random_(-8, 9, generator=generator)
                elif not value.is_floating_point() or "A_log" in name:
                    value.zero_()
                elif "dt_bias" in name:
                    value.fill_(-2.0)
                elif "hc_" in name:
                    if name.endswith("scale"):
                        value.fill_(1.0)
                    elif name.endswith("base"):
                        value.zero_()
                    else:
                        value.uniform_(-0.001, 0.001, generator=generator)
                elif "scale" in name:
                    value.fill_(0.01)
                elif "offset" in name or name.endswith("bias"):
                    value.zero_()
                elif "norm" in name and name.endswith("weight"):
                    value.fill_(1.0)
                else:
                    value.uniform_(-0.01, 0.01, generator=generator)


class FlashWorker:
    """Read-only model-path checks; no hooks into logits or graph replay."""

    model_runner: Any  # Supplied by the worker extension host at runtime.

    def check_flash_paths(self):
        modules = list(self.model_runner.model.modules())
        names = [type(module).__name__ for module in modules]
        assert names.count("Glm5NextDecoderLayer") == NUM_LAYERS
        assert names.count("Glm5NextLinearAttention") == 7
        assert names.count("Glm5NextMLAAttention") == 2
        assert sum(hasattr(module, "hc_attn_fn") for module in modules) == NUM_LAYERS
        methods = [getattr(module, "quant_method", None) for module in modules]
        schemes = {type(getattr(method, "quant_method", method)).__name__ for method in methods}
        assert "AscendW8A8DynamicFusedMoEMethod" in schemes
        return True


@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "graph"])
@pytest.mark.parametrize("use_replayssm", [False, True], ids=["no_recover", "recover"])
def test_glm53flash_tp4(tmp_path, monkeypatch, enforce_eager, use_replayssm):
    for name, value in {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_BUFFSIZE": "400",
    }.items():
        monkeypatch.setenv(name, value)
    model = tmp_path / "model"
    config = _write_model(model)
    assert config["architectures"] == ["Glm5NextForConditionalGeneration"]
    settings = dict(
        load_format=LOADER,
        worker_extension_cls=f"{__name__}.FlashWorker",
        dtype="bfloat16",
        tensor_parallel_size=TP_SIZE,
        # TODO: Add PCP coverage once GLM-Flash supports PCP, including MTP
        # and RecoverSSM in eager and graph modes within the four-card budget.
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=True,
        quantization="ascend",
        distributed_executor_backend="mp",
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=512,
        # The wrapper still initializes its vision tower; this smoke sends
        # token IDs only and does not validate multimodal prompt expansion.
        limit_mm_per_prompt={"image": 0, "video": 0},
        block_size=128,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        gpu_memory_utilization=0.8,
        seed=1024,
        enforce_eager=enforce_eager,
        # TODO: Current dummy weights cannot validate meaningful acceptance rates.
        # Add a real-weight e2e or suitable new weights before checking acceptance.
        speculative_config={
            "method": "deepseek_mtp",
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
            "enforce_eager": enforce_eager,
        },
        use_replayssm=use_replayssm,
        additional_config={"enable_cpu_binding": False, "enable_fused_mc2": 0},
    )
    if not enforce_eager:
        settings["compilation_config"] = {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [6, 12, 24],
        }
    with VllmRunner(str(model), **settings) as runner:
        assert runner.model.collective_rpc("check_flash_paths") == [True] * TP_SIZE
        params = SamplingParams(temperature=0, max_tokens=OUTPUT_TOKENS, ignore_eos=True, detokenize=False)
        # A second mixed batch reuses slots after the first request finishes.
        # 127/128/129 and 511/512/513 straddle input/chunk boundaries.
        for lengths in ((128,), (127, 128, 129), (511, 512, 513)):
            outputs = runner.model.generate(
                [{"prompt_token_ids": [10 + i % 1000 for i in range(length)]} for length in lengths],
                params,
                use_tqdm=False,
            )
            assert len(outputs) == len(lengths)
            for output in outputs:
                assert output.finished and len(output.outputs) == 1
                tokens = list(output.outputs[0].token_ids)
                assert len(tokens) == OUTPUT_TOKENS
                assert all(0 <= token < config["text_config"]["vocab_size"] for token in tokens)
