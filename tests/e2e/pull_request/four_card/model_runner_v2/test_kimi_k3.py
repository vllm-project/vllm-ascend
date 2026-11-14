# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 Model Runner V2 PR smoke coverage.

The MRV1 Kimi-K3 tests already build the smallest useful local model and cover
the model-specific execution paths.  Reuse the model builder and scenario helpers here
so MRV1 and MRV2 keep the same functional coverage.  These are deliberately
functional smoke tests: the local models use dummy weights and do not replace
the full-checkpoint nightly GPQA or AISBench jobs.
"""

import json
import math
import shutil
from pathlib import Path
from typing import Any

import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner
from tests.e2e.pull_request.four_card.test_kimi_k3 import (
    _engine_args,
    _generate,
    _prompt,
    build_k3_models,
    run_k3_gqa_w4a8_dp2_tp2,
    run_k3_mla_block5_tp4,
    run_k3_mla_pd_tp2,
    run_k3_mtp_image_tp4,
)

pytestmark = pytest.mark.e2e_model("Kimi-K3")


@pytest.fixture(scope="module")
def k3_models(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    return build_k3_models(tmp_path_factory.mktemp("k3-dummy"))


@pytest.fixture(scope="module")
def k31_models(tmp_path_factory: pytest.TempPathFactory, k3_models: dict[str, str]) -> dict[str, str]:
    """Reuse the local K3 tokenizer and add one K3.1 sparse-attention layer."""
    root = tmp_path_factory.mktemp("k31-dummy")
    models = dict(k3_models)
    target = root / "target"
    shutil.copytree(k3_models["target"], target)
    path = target / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["architectures"] = ["KimiK31ForConditionalGeneration"]
    config["model_type"] = "kimi_k31"
    config["text_config"]["dsa_config"] = {
        "dsa_layers": [2],
        "index_topk": 2048,
        "index_num_heads": 32,
        "index_head_dim": 128,
        "index_rope_dim": 0,
        "index_k_norm_type": "layernorm",
    }
    path.write_text(json.dumps(config), encoding="utf-8")
    models["target"] = str(target)
    return models


class KimiRecoverSSMWorker:
    """Check the selected attention layer and builder on each worker."""

    model_runner: Any

    def kimi_cache_block_size(self):
        groups = self.model_runner.kv_cache_config.kv_cache_groups
        return math.lcm(*(group.kv_cache_spec.block_size for group in groups))

    def _kimi_recoverssm_builders(self):
        from vllm_ascend.models.kimi_k3.recoverssm_metadata import AscendKimiKDARecoverSSMMetadataBuilder

        return [
            group.get_metadata_builder(0)
            for groups in self.model_runner.attn_groups
            for group in groups
            if isinstance(group.get_metadata_builder(0), AscendKimiKDARecoverSSMMetadataBuilder)
        ]

    def check_kimi_recoverssm_path(self, require_context: bool, expect_dsa: bool):
        from vllm_ascend.models.kimi_k3.kimi_k3 import AscendKimiDSAAttention
        from vllm_ascend.ops.kimi_kda import AscendKimiK3DeltaAttention

        modules = list(self.model_runner.model.modules())
        layers = [layer for layer in modules if isinstance(layer, AscendKimiK3DeltaAttention)]
        builders = self._kimi_recoverssm_builders()
        assert layers and all(layer.use_recoverssm and layer.recoverssm_records is not None for layer in layers)
        assert any(isinstance(layer, AscendKimiDSAAttention) for layer in modules) is expect_dsa
        assert builders
        if require_context:
            assert all(builder.recoverssm_context is not None for builder in builders)
        return True

    def install_kimi_commit_probe(self):
        self.kimi_commit_calls = 0
        builders = self._kimi_recoverssm_builders()
        assert builders
        for builder in builders:
            context = builder._get_recoverssm_context()
            original_commit = context.commit

            def count_commit(*args, _original=original_commit, **kwargs):
                self.kimi_commit_calls += 1
                return _original(*args, **kwargs)

            context.commit = count_commit
        return True

    def read_kimi_commit_probe(self):
        return self.kimi_commit_calls


@pytest.fixture(autouse=True)
def k3_runtime_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the reused Kimi smoke scenarios with Model Runner V2."""

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("HCCL_OP_EXPANSION_MODE", "AIV")
    monkeypatch.setenv("HCCL_BUFFSIZE", "512")


def test_k3_mla_block5_tp4(k3_models: dict[str, str]) -> None:
    """Reuse MLA Block5, prefix-cache, and D-Spark coverage with MRV2."""

    run_k3_mla_block5_tp4(k3_models)


def test_k3_gqa_w4a8_dp2_tp2(k3_models: dict[str, str]) -> None:
    """Reuse quantized GQA and local DP coverage with MRV2."""

    run_k3_gqa_w4a8_dp2_tp2(k3_models)


def test_k3_mtp_image_tp4(k3_models: dict[str, str]) -> None:
    """Reuse MTP and multimodal image coverage with MRV2."""

    run_k3_mtp_image_tp4(k3_models)


def test_k3_mla_pd_tp2(k3_models: dict[str, str]) -> None:
    """Reuse same-host P/D and Mooncake KV-transfer coverage with MRV2."""

    run_k3_mla_pd_tp2(k3_models)


def test_k3_mrv2_basic_generate_tp4(k3_models: dict[str, str]) -> None:
    """Exercise the V2 runner without speculative decoding.

    The reused tests intentionally exercise speculative paths.  This smoke
    keeps one plain generation path so a runner or scheduler regression cannot
    be hidden by the speculative-draft setup.
    """

    args = _engine_args(k3_models, "mla")
    args.pop("speculative_config")
    args["max_model_len"] = 1024
    args["enforce_eager"] = True

    with VllmRunner(k3_models["target"], **args) as runner:
        _generate(
            runner.model,
            [_prompt(1), _prompt(129, salt=137)],
        )


@pytest.mark.parametrize("family", ["k31"])
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("prefix_caching", [False, True], ids=["no_prefix", "prefix"])
def test_kimi_recoverssm_tp4(k3_models, k31_models, family, graph, prefix_caching):
    """Run several verify/commit steps and check the actual KDA builder."""
    models = k3_models if family == "k3" else k31_models
    args = _engine_args(models, "mla")
    args.update(
        worker_extension_cls=f"{__name__}.KimiRecoverSSMWorker",
        use_replayssm=True,
        enable_prefix_caching=prefix_caching,
        mamba_cache_mode="align" if prefix_caching else "none",
        enforce_eager=not graph,
    )
    if prefix_caching:
        args["max_model_len"] = 4096
    with VllmRunner(str(Path(models["target"])), **args) as runner:
        assert runner.model.collective_rpc("check_kimi_recoverssm_path", args=(False, family == "k31")) == [True] * 4
        assert runner.model.collective_rpc("install_kimi_commit_probe") == [True] * 4
        sampling = SamplingParams(temperature=0, max_tokens=12, ignore_eos=True, detokenize=False)
        prompt_batches = ([_prompt(1), _prompt(129, salt=137)], [_prompt(64, salt=71)])
        for prompts in prompt_batches:
            outputs = runner.model.generate(prompts, sampling, use_tqdm=False)
            assert len(outputs) == len(prompts)
            assert all(output.finished and len(output.outputs[0].token_ids) == 12 for output in outputs)
        if prefix_caching:
            assert runner.model.reset_prefix_cache()
            block_sizes = runner.model.collective_rpc("kimi_cache_block_size")
            assert len(set(block_sizes)) == 1
            block_size = block_sizes[0]
            assert block_size + 1 + sampling.max_tokens <= args["max_model_len"]
            prefix_prompt = [_prompt(block_size + 1, salt=421)]
            cold = runner.model.generate(prefix_prompt, sampling, use_tqdm=False)[0]
            warm = runner.model.generate(prefix_prompt, sampling, use_tqdm=False)[0]
            assert warm.num_cached_tokens >= block_size
            assert warm.outputs[0].token_ids == cold.outputs[0].token_ids
        assert runner.model.collective_rpc("check_kimi_recoverssm_path", args=(True, family == "k31")) == [True] * 4
        assert all(count > 0 for count in runner.model.collective_rpc("read_kimi_commit_probe"))
