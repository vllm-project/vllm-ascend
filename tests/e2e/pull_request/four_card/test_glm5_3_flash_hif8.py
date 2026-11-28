# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Four-card HIF8 C8 KV cache regression with synthetic Flash weights.

Build on the functional Flash smoke test: the same synthetic
nine-layer-plus-MTP checkpoint recipe and dummy weight policy, plus
QK_HIF8_PER_TENSOR layer recipes on the sparse-attention layers and the
MTP draft layer, served with ``--kv-cache-dtype hif8``. Weights stay
synthetic (no checkpoint download): the recipe drives scheme selection,
per-tensor scale handling, the packed HIF8 KV cache spec and the
aclnnQuantSparseFlashMla path end to end in full-decode-only graph
mode, so this is a functional regression, not an accuracy baseline.
Parallel coverage is TP4+EP with PCP=1 and DCP=1 (HIF8 rejects context
parallel).
"""

import json
from typing import Any

import pytest
import regex as re
import torch
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.four_card.test_glm5_3_flash import (
    LOADER,
    NUM_LAYERS,
    NUM_MTP_LAYERS,
    NUM_SPECULATIVE_TOKENS,
    OUTPUT_TOKENS,
    TP_SIZE,
    FlashWorker,
    _write_model,
)

# Sparse-attention layer slots carrying the HIF8 C8 recipe: the two main
# sparse layers plus the MTP draft layer (the layer-9 slot).
MAIN_SPARSE_LAYERS = (3, 7)
HIF8_LAYERS = MAIN_SPARSE_LAYERS + (NUM_LAYERS,)
SCALE_FILL_VALUE = 0.01


def _write_hif8_model(destination):
    """Write the synthetic Flash model, then inject the HIF8 C8 recipe.

    Mirrors the ModelSlim export of the W4A4C8 GLM-5.3-Flash checkpoint:
    per-tensor fa_q/fa_k scales on every sparse layer plus a leftover
    FAKQuant declaration that must be stood down in favor of the HIF8
    layer recipes.
    """
    config = _write_model(destination)
    quant_path = destination / "quant_model_description.json"
    quant = json.loads(quant_path.read_text())
    for layer in HIF8_LAYERS:
        prefix = f"model.language_model.layers.{layer}.self_attn"
        quant[f"{prefix}.quant_type"] = "QK_HIF8_PER_TENSOR"
        quant[f"{prefix}.fa_q.scale"] = "FAQuant"
        quant[f"{prefix}.fa_k.scale"] = "FAQuant"
    quant["fa_quant_type"] = "FAKQuant"
    quant_path.write_text(json.dumps(quant))
    return config


def _layer_ids_of(name):
    return {int(match) for match in re.findall(r"layers\.(\d+)", name)}


class FlashHif8Worker(FlashWorker):
    """HIF8 checks layered on the Flash model-path checks."""

    model_runner: Any  # Supplied by the worker extension host at runtime.

    def _assert_hif8_layer(self, name, module, expect_draft):
        assert getattr(module, "hif8_quant_layer", False), f"{name} is not on the HIF8 path"
        for param in ("hif8_q_descale", "hif8_kv_descale", "hif8_q_scale_reciprocal", "hif8_kv_scale_reciprocal"):
            tensor = getattr(module, param)
            assert torch.isfinite(tensor).all(), f"{name}.{param} is not finite"
        if not expect_draft:
            # Target-model scales come from the dummy fill policy in the
            # Flash loader ("scale" parameters are filled with 0.01).
            assert module.hif8_q_descale.item() == pytest.approx(SCALE_FILL_VALUE)
            assert module.hif8_kv_descale.item() == pytest.approx(SCALE_FILL_VALUE)
            assert module.hif8_q_scale_reciprocal.item() == pytest.approx(1.0 / SCALE_FILL_VALUE)

    def check_hif8_paths(self):
        assert self.check_flash_paths()
        methods = [getattr(module, "quant_method", None) for module in self.model_runner.model.modules()]
        schemes = {type(getattr(method, "quant_method", method)).__name__ for method in methods}
        assert "AscendHIF8KVCacheAttentionMethod" in schemes
        # Every sparse-attention layer runs the HIF8 scheme, and only those.
        target_hif8 = [
            (name, module)
            for name, module in self.model_runner.model.named_modules()
            if getattr(module, "hif8_quant_layer", False)
        ]
        assert len(target_hif8) == len(MAIN_SPARSE_LAYERS)
        assert {layer_id for name, _ in target_hif8 for layer_id in _layer_ids_of(name)} == set(MAIN_SPARSE_LAYERS)
        for name, module in target_hif8:
            self._assert_hif8_layer(name, module, expect_draft=False)
        # The MTP draft layer is HIF8-quantized too, like the real export.
        draft_hif8 = [
            (name, module)
            for name, module in self.model_runner.speculator.model.named_modules()
            if getattr(module, "hif8_quant_layer", False)
        ]
        assert len(draft_hif8) == NUM_MTP_LAYERS
        assert all(_layer_ids_of(name) == {NUM_LAYERS} for name, _ in draft_hif8)
        for name, module in draft_hif8:
            self._assert_hif8_layer(name, module, expect_draft=True)
        return True


@pytest.mark.parametrize("prefix_caching", [False, True], ids=["no_prefix", "prefix"])
def test_glm53flash_hif8_tp4(tmp_path, monkeypatch, prefix_caching):
    for name, value in {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "HCCL_BUFFSIZE": "400",
    }.items():
        monkeypatch.setenv(name, value)
    model = tmp_path / "model"
    config = _write_hif8_model(model)
    assert config["architectures"] == ["Glm5NextForConditionalGeneration"]
    settings = dict(
        load_format=LOADER,
        worker_extension_cls=f"{__name__}.FlashHif8Worker",
        dtype="bfloat16",
        tensor_parallel_size=TP_SIZE,
        # HIF8 rejects context parallel; revisit once the GLM-Flash PCP
        # support lands and the payload can move through the gather flow.
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=True,
        quantization="ascend",
        distributed_executor_backend="mp",
        # The HIF8 page carries half the bf16 bytes per token, so the
        # mamba-page coverage rule raises the effective attention block
        # size well past the requested 128 (2304 for this config). Keep
        # max_model_len ahead of two effective blocks plus the output so
        # the prefix-caching reuse check below stays meaningful.
        max_model_len=8192,
        max_num_seqs=4,
        max_num_batched_tokens=512,
        # The wrapper still initializes its vision tower; this smoke sends
        # token IDs only and does not validate multimodal prompt expansion.
        limit_mm_per_prompt={"image": 0, "video": 0},
        block_size=128,
        enable_chunked_prefill=True,
        enable_prefix_caching=prefix_caching,
        mamba_cache_mode="align" if prefix_caching else "none",
        kv_cache_dtype="hif8",
        gpu_memory_utilization=0.8,
        seed=1024,
        enforce_eager=False,
        # The dummy weights carry no acceptance signal; this exercises the
        # HIF8 path under speculative decoding, not draft acceptance.
        speculative_config={
            "method": "deepseek_mtp",
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
            "enforce_eager": False,
        },
        additional_config={"enable_cpu_binding": False, "enable_fused_mc2": 0, "enable_flashcomm1": True},
    )
    settings["compilation_config"] = {
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [6, 12, 24],
    }
    with VllmRunner(str(model), **settings) as runner:
        assert runner.model.collective_rpc("check_hif8_paths") == [True] * TP_SIZE
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
        if prefix_caching:
            block_size = runner.model.llm_engine.vllm_config.cache_config.block_size
            prefix_length = 2 * block_size + 17
            assert prefix_length + OUTPUT_TOKENS < settings["max_model_len"]
            prompt = [{"prompt_token_ids": [10 + i % 1000 for i in range(prefix_length)]}]
            cold = runner.model.generate(prompt, params, use_tqdm=False)[0]
            warm = runner.model.generate(prompt, params, use_tqdm=False)[0]
            assert warm.num_cached_tokens >= block_size
            assert warm.outputs[0].token_ids == cold.outputs[0].token_ids
