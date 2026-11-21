# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from vllm.config import ParallelConfig, replace
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

from vllm_ascend.patch.worker import patch_dspark
from vllm_ascend.patch.worker.patch_dspark import dspark_context_parallel_config, replace_dspark_config
from vllm_ascend.worker.v2 import model_runner
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator


def _config(*, pp=1, pcp=8, dcp=8, use_mla=True, model_type="k3_dspark"):
    parallel = ParallelConfig(
        tensor_parallel_size=1,
        pipeline_parallel_size=pp,
        prefill_context_parallel_size=pcp,
        decode_context_parallel_size=dcp,
        distributed_executor_backend="mp",
        cp_kv_cache_interleave_size=128,
    )
    draft = SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type), use_mla=use_mla)
    return SimpleNamespace(
        parallel_config=parallel,
        engram_config=None,
        speculative_config=SimpleNamespace(
            method="dspark",
            draft_model_config=draft,
            target_parallel_config=parallel,
            draft_decode_context_parallel_size=dcp if use_mla else 1,
        ),
    )


@pytest.mark.parametrize("pcp,dcp", [(2, 2), (8, 8)])
def test_real_dflash_constructor_reaches_parent_with_physical_dcp(monkeypatch, pcp, dcp):
    assert patch_dspark.dflash_speculator_module.replace is replace_dspark_config
    config = _config(pcp=pcp, dcp=dcp)
    observed = []

    class Constructed(Exception):
        pass

    def parent_init(self, draft_config, device):
        observed.append(draft_config.parallel_config)
        raise Constructed

    monkeypatch.setattr(DraftModelSpeculator, "__init__", parent_init)
    # Exercise the actual upstream PCP=1 replacement through the Ascend entry.
    with pytest.raises(Constructed):
        AscendDSparkSpeculator(config, torch.device("cpu"))
    parallel = observed[0]
    assert parallel.prefill_context_parallel_size == 1
    assert parallel.decode_context_parallel_size == dcp
    assert parallel.cp_kv_cache_interleave_size == 128
    assert config.parallel_config.prefill_context_parallel_size == pcp
    # The construction exception must not weaken general topology validation.
    with pytest.raises(ValueError, match="must be divisible"):
        dflash_speculator.replace(config.parallel_config, prefill_context_parallel_size=1)


def test_runner_parent_can_construct_upstream_dspark_with_physical_dcp(monkeypatch):
    config = _config(pp=2)
    observed = []

    class DraftBaseReached(Exception):
        pass

    def construct_upstream_draft(self, runner_config, device):
        # On the last PP stage the parent constructs DSpark before Ascend's
        # own speculator factory is called.
        DSparkSpeculator(runner_config, device)

    def stop_at_draft_base(self, draft_config, device):
        observed.append(draft_config.parallel_config)
        raise DraftBaseReached

    monkeypatch.setattr(model_runner, "get_ascend_config", lambda: SimpleNamespace())
    monkeypatch.setattr(model_runner, "set_potential_max_tokens", lambda config: None)
    monkeypatch.setattr(model_runner, "resolve_spec_pp_support", lambda config: None)
    monkeypatch.setattr(model_runner, "torch_cuda_wrapper", nullcontext)
    monkeypatch.setattr(model_runner.GPUModelRunner, "__init__", construct_upstream_draft)
    monkeypatch.setattr(DraftModelSpeculator, "__init__", stop_at_draft_base)

    with pytest.raises(DraftBaseReached):
        model_runner.NPUModelRunner(config, torch.device("cpu"))

    (query_parallel,) = observed
    assert query_parallel.pipeline_parallel_size == 2
    assert query_parallel.prefill_context_parallel_size == 1
    assert query_parallel.decode_context_parallel_size == 8
    assert config.parallel_config.prefill_context_parallel_size == 8
    with pytest.raises(ValueError, match="must be divisible"):
        dflash_speculator.replace(config.parallel_config, prefill_context_parallel_size=1)


def test_query_adapter_keeps_other_parallel_validation():
    config = _config()
    with dspark_context_parallel_config(config):
        with pytest.raises(ValueError):
            replace_dspark_config(config.parallel_config, prefill_context_parallel_size=1, tensor_parallel_size=0)
        with pytest.raises(ValueError):
            replace_dspark_config(
                config.parallel_config, prefill_context_parallel_size=1, decode_context_parallel_size=3
            )


@pytest.mark.parametrize("use_mla,model_type", [(False, "k3_dspark"), (True, "deepseek_v4")])
def test_other_drafts_retain_upstream_validation(use_mla, model_type):
    config = _config(use_mla=use_mla, model_type=model_type)
    with dspark_context_parallel_config(config):
        with pytest.raises(ValueError, match="must be divisible"):
            replace_dspark_config(config.parallel_config, prefill_context_parallel_size=1)
        if not use_mla:
            query = replace_dspark_config(
                config.parallel_config, prefill_context_parallel_size=1, decode_context_parallel_size=1
            )
            assert query.decode_context_parallel_size == 1


def test_query_scope_isolated_between_threads_and_nested_drafts():
    config = _config()

    def unrelated_thread():
        with pytest.raises(ValueError, match="must be divisible"):
            replace_dspark_config(config.parallel_config, prefill_context_parallel_size=1)

    with dspark_context_parallel_config(config):
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(unrelated_thread).result()
        with dspark_context_parallel_config(_config(use_mla=False)):
            unrelated_thread()
        query = replace_dspark_config(config.parallel_config, prefill_context_parallel_size=1)
        assert query.decode_context_parallel_size == 8
    unrelated_thread()


def test_dcp1_uses_unmodified_upstream_replacement():
    config = _config(dcp=1)
    with dspark_context_parallel_config(config):
        actual = replace_dspark_config(config.parallel_config, prefill_context_parallel_size=1)
    expected = replace(config.parallel_config, prefill_context_parallel_size=1)
    assert actual == expected
