# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3-style local dummy-model functional smoke, not accuracy or performance.

Keep the first nine Flash layers and production widths/experts. No checkpoint,
tokenizer download, MTP, C8, multimodal input or runtime patch is installed here.
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from vllm import SamplingParams
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

from tests.e2e.conftest import VllmRunner

NUM_LAYERS = 9
OUTPUT_TOKENS = 8
LOADER = "glm53flash_functional_dummy"


def _write_model(destination):
    assets = Path(__file__).with_name("glm53flash_assets")
    config = json.loads((assets / "config.json").read_text())
    template = json.loads((assets / "quant.json").read_text())
    quant = {key: value for key, value in template.items() if not key.startswith("model.layers.")}
    for layer in range(config["num_hidden_layers"]):
        source = 0 if layer < config["first_k_dense_replace"] else 3 if layer in (3, 7) else 4
        prefix = f"model.layers.{source}."
        for key, value in template.items():
            if not key.startswith(prefix):
                continue
            target = f"model.layers.{layer}." + key[len(prefix) :]
            if ".experts.0." in target:
                for expert in range(config["n_routed_experts"]):
                    quant[target.replace(".experts.0.", f".experts.{expert}.")] = value
            else:
                quant[target] = value
    destination.mkdir()
    (destination / "config.json").write_text(json.dumps(config))
    (destination / "quant_model_description.json").write_text(json.dumps(quant))
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
def test_glm53flash_tp4(tmp_path, monkeypatch, enforce_eager):
    for name, value in {
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_BUFFSIZE": "400",
    }.items():
        monkeypatch.setenv(name, value)
    model = tmp_path / "model"
    config = _write_model(model)
    settings = dict(
        load_format=LOADER,
        worker_extension_cls=f"{__name__}.FlashWorker",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        quantization="ascend",
        distributed_executor_backend="mp",
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=512,
        block_size=128,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.8,
        seed=1024,
        enforce_eager=enforce_eager,
        additional_config={"enable_cpu_binding": False, "enable_fused_mc2": 0},
    )
    if not enforce_eager:
        settings["compilation_config"] = {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4],
        }
    with VllmRunner(str(model), **settings) as runner:
        assert runner.model.collective_rpc("check_flash_paths") == [True] * 4
        params = SamplingParams(temperature=0, max_tokens=OUTPUT_TOKENS, ignore_eos=True, detokenize=False)
        # A single request followed by a mixed batch exercises both API shapes.
        # 127/128/129 are input sizes, not a claim about the effective cache page.
        for lengths in ((128,), (127, 128, 129)):
            outputs = runner.model.generate(
                [{"prompt_token_ids": [10 + i % 1000 for i in range(length)]} for length in lengths],
                params,
                use_tqdm=False,
            )
            assert len(outputs) == len(lengths)
            for output in outputs:
                assert output.finished and len(output.outputs) == 1
                tokens = output.outputs[0].token_ids
                assert len(tokens) == OUTPUT_TOKENS
                assert all(0 <= token < config["vocab_size"] for token in tokens)
