# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 DP4/TP1/EP smoke using the same nine-layer local model as A3.

Synthetic MXFP8 weights exercise prefill/decode in eager and graph modes.
This does not validate checkpoint accuracy, performance, MTP or multimodal input.
"""

import hashlib
import json
from typing import Any

import pytest
import torch
from vllm import SamplingParams
from vllm.distributed.parallel_state import get_dp_group, get_ep_group, get_tp_group
from vllm.model_executor.model_loader import register_model_loader

from tests.e2e.conftest import DPVllmRunner
from tests.e2e.pull_request.four_card.test_glm5_3_flash import NUM_LAYERS, OUTPUT_TOKENS, FlashDummyLoader, _write_model

DP_SIZE = 4
LOADER = "glm53flash_a5_functional_dummy"
MXFP8_SCHEME = "W8A8_MXFP8"
MXFP8_SCALE_EXPONENT = 120  # E8M0: 2 ** (120 - 127) = 1/128.
INIT_CHUNK_SIZE = 1024 * 1024


def _write_a5_model(destination):
    config = _write_model(destination)
    quant_path = destination / "quant_model_description.json"
    template = json.loads(quant_path.read_text())
    # Match the public GLM-5.3-Flash-w8a8-mxfp8 quantization recipe:
    # dense/routed/shared MLPs plus these four sparse-MLA projections.
    # Linear attention, the indexer and kv_b_proj remain unquantized.
    quant = {
        key: MXFP8_SCHEME if value == "W8A8_DYNAMIC" else value
        for key, value in template.items()
        if not key.endswith(".weight_offset") and key != "is_rot_used"
    }
    quant["group_size"] = 32
    for layer, layer_type in enumerate(config["text_config"]["layer_types"]):
        if layer_type == "deepseek_sparse_attention":
            for projection in ("kv_a_proj_with_mqa", "q_a_proj", "q_b_proj", "o_proj"):
                prefix = f"model.language_model.layers.{layer}.self_attn.{projection}"
                quant[f"{prefix}.weight"] = MXFP8_SCHEME
                quant[f"{prefix}.weight_scale"] = MXFP8_SCHEME
    quant_path.write_text(json.dumps(quant))
    return config


@register_model_loader(LOADER)
class FlashA5DummyLoader(FlashDummyLoader):
    @staticmethod
    def initialize_parameter(name, value):
        if value.dtype == torch.float8_e4m3fn:
            seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
            generator = torch.Generator(device=value.device).manual_seed(seed)
            # FP8 uniform_ is not implemented on all backends. Generate bounded
            # FP32 chunks then cast, without a full-size FP32 MoE allocation.
            for chunk in value.view(-1).split(INIT_CHUNK_SIZE):
                source = torch.empty(chunk.numel(), dtype=torch.float32, device=value.device)
                source.uniform_(-1.0, 1.0, generator=generator)
                chunk.copy_(source.to(value.dtype))
        elif value.dtype == torch.uint8 and name.endswith("weight_scale"):
            value.fill_(MXFP8_SCALE_EXPONENT)
        else:
            FlashDummyLoader.initialize_parameter(name, value)


class FlashA5Worker:
    """Importing this extension also registers the loader in spawned workers."""

    model_runner: Any

    def check_flash_a5_paths(self):
        assert get_tp_group().world_size == 1
        assert get_dp_group().world_size == DP_SIZE
        assert get_ep_group().world_size == DP_SIZE
        modules = list(self.model_runner.model.modules())
        names = [type(module).__name__ for module in modules]
        assert names.count("Glm5NextDecoderLayer") == NUM_LAYERS
        assert names.count("Glm5NextLinearAttention") == 7
        assert names.count("Glm5NextMLAAttention") == 2
        assert sum(hasattr(module, "hc_attn_fn") for module in modules) == NUM_LAYERS
        methods = [getattr(module, "quant_method", None) for module in modules]
        schemes = {type(getattr(method, "quant_method", method)).__name__ for method in methods}
        assert "AscendW8A8MXFP8DynamicFusedMoEMethod" in schemes
        assert "AscendW8A8MXFP8DynamicLinearMethod" in schemes
        assert "AscendW8A8DynamicFusedMoEMethod" not in schemes
        if not self.model_runner.model_config.enforce_eager:
            entries = getattr(self.model_runner.model, "concrete_aclgraph_entries", {})
            assert any(entry.aclgraph is not None for entry in entries.values()), "No ACL graphs captured"
        return True


@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "graph"])
def test_glm53flash_a5_dp4(tmp_path, monkeypatch, enforce_eager):
    for name, value in {
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_BUFFSIZE": "1024",
        "OMP_NUM_THREADS": "1",
        "TASK_QUEUE_ENABLE": "1",
    }.items():
        monkeypatch.setenv(name, value)
    model = tmp_path / "model"
    config = _write_a5_model(model)
    assert config["architectures"] == ["Glm5NextForConditionalGeneration"]
    settings = dict(
        load_format=LOADER,
        worker_extension_cls=f"{__name__}.FlashA5Worker",
        dtype="bfloat16",
        data_parallel_size=DP_SIZE,
        tensor_parallel_size=1,
        enable_expert_parallel=True,
        quantization="ascend",
        distributed_executor_backend="mp",
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=512,
        limit_mm_per_prompt={"image": 0, "video": 0},
        block_size=128,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        # A smoke test needs only a small cache. Do not scale recurrent-state
        # backing storage (and gather workspaces) with the host's free HBM.
        kv_cache_memory_bytes=1024 * 1024 * 1024,
        seed=1024,
        enforce_eager=enforce_eager,
        additional_config={"enable_cpu_binding": False, "enable_fused_mc2": 0},
    )
    if not enforce_eager:
        settings["compilation_config"] = {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4],
        }
    with DPVllmRunner(str(model), **settings) as runner:
        assert runner.collective_rpc("check_flash_a5_paths") == [[True]] * DP_SIZE
        params = SamplingParams(temperature=0, max_tokens=OUTPUT_TOKENS, ignore_eos=True, detokenize=False)
        # One request per rank, then three per rank. Every DP rank exercises
        # both API batch shapes and the 127/128/129 input-length boundaries.
        for lengths in ((128,) * DP_SIZE, (127, 128, 129) * DP_SIZE):
            prompts = [[10 + i % 1000 for i in range(length)] for length in lengths]
            outputs = runner.generate(prompts, params, use_tqdm=False)
            assert len(outputs) == len(prompts)
            for prompt, (sequences, _) in zip(prompts, outputs):
                assert len(sequences) == 1
                assert sequences[0][: len(prompt)] == prompt
                tokens = sequences[0][len(prompt) :]
                assert len(tokens) == OUTPUT_TOKENS
                assert all(0 <= token < config["text_config"]["vocab_size"] for token in tokens)
